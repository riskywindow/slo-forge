"""Topology-aware collective ring ordering experiment.

This module deliberately optimizes a narrow physical operation instead of
claiming to improve a collective implementation.  It selects a rank order for
ring collectives from calibrated link curves, retains the exact routes used by
the model, and recommends an on-device experiment only after modeled paired
evidence has a confidence interval above a practical threshold. Modeled trials
can never enable a production default, even when their input curves are
hardware-calibrated.
"""

from __future__ import annotations

import hashlib
import heapq
import json
import math
import os
import random
import statistics
import tempfile
from collections import defaultdict
from contextlib import suppress
from enum import StrEnum
from itertools import pairwise, permutations
from pathlib import Path
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from sloforge.fabric.ir import (
    CollectiveOperation,
    CurvePoint,
    DiscoverySource,
    HealthState,
    PhysicalExecutionPlan,
    RankPlacement,
    TopologyEdge,
    TopologyGraph,
    canonical_hash,
)

PositiveFinite = Annotated[float, Field(gt=0.0, allow_inf_nan=False)]
NonNegativeFinite = Annotated[float, Field(ge=0.0, allow_inf_nan=False)]
Probability = Annotated[float, Field(ge=0.0, le=1.0, allow_inf_nan=False)]
Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
MAX_TRACE_EVIDENCE_BYTES = 16 * 1024 * 1024


class PerformanceModel(BaseModel):
    """Strict immutable value used by the experiment artifact."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, validate_default=True)


class CalibrationError(ValueError):
    """Raised when a topology lacks measured curves required by the model."""


class CalibrationMode(StrEnum):
    MEASURED_HARDWARE = "measured_hardware"
    SYNTHETIC_CALIBRATED = "synthetic_calibrated"


class OptimizationMethod(StrEnum):
    REFERENCE = "reference"
    EXACT = "exact"
    GREEDY_TWO_OPT = "greedy_two_opt"


class IntegrationStatus(StrEnum):
    ENABLE = "enable"
    KEEP_REFERENCE = "keep_reference"
    MEASURE_ON_HARDWARE = "measure_on_hardware"
    INCONCLUSIVE = "inconclusive"


class SamplePhase(StrEnum):
    WARMUP = "warmup"
    MEASUREMENT = "measurement"


class RankWaitSample(PerformanceModel):
    rank_id: Annotated[int, Field(ge=0)]
    wait_microseconds: NonNegativeFinite


class CollectiveTraceEvidence(PerformanceModel):
    """Trace evidence that makes the optimization target auditable."""

    schema_version: Literal["sloforge.fabric.collective-evidence/v1"] = (
        "sloforge.fabric.collective-evidence/v1"
    )
    artifact_uri: Annotated[str, Field(min_length=1)]
    artifact_sha256: Sha256
    plan_id: Annotated[str, Field(min_length=1)]
    operation_id: Annotated[str, Field(min_length=1)]
    calibration_mode: CalibrationMode
    observed_duration_microseconds: tuple[PositiveFinite, ...]
    collective_critical_path_fraction: Probability
    rank_wait: tuple[RankWaitSample, ...]
    fault_free: bool

    @model_validator(mode="after")
    def enough_samples(self) -> Self:
        if len(self.observed_duration_microseconds) < 3:
            raise ValueError("collective trace evidence requires at least three observations")
        ranks = [sample.rank_id for sample in self.rank_wait]
        if len(ranks) != len(set(ranks)):
            raise ValueError("rank wait evidence must contain each rank at most once")
        return self


class RankOrderingExperimentConfig(PerformanceModel):
    seed: Annotated[int, Field(ge=0)]
    message_sizes_bytes: tuple[Annotated[int, Field(gt=0)], ...]
    optimization_message_bytes: Annotated[int, Field(gt=0)]
    warmup_trials: Annotated[int, Field(ge=1)] = 5
    measured_trials: Annotated[int, Field(ge=5)] = 40
    bootstrap_rounds: Annotated[int, Field(ge=200)] = 2_000
    confidence_level: Annotated[float, Field(gt=0.5, lt=1.0)] = 0.95
    exact_rank_limit: Annotated[int, Field(ge=2, le=9)] = 8
    heuristic_pass_limit: Annotated[int, Field(ge=1, le=100)] = 20
    link_variation_fraction: Annotated[float, Field(ge=0.0, le=0.5)] = 0.03
    minimum_improvement_percent: NonNegativeFinite = 2.0
    minimum_collective_critical_path_fraction: Probability = 0.10

    @model_validator(mode="after")
    def valid_regimes(self) -> Self:
        if not self.message_sizes_bytes:
            raise ValueError("at least one message-size regime is required")
        if len(set(self.message_sizes_bytes)) != len(self.message_sizes_bytes):
            raise ValueError("message-size regimes must be unique")
        return self


class RankOrderingExperimentInput(PerformanceModel):
    """Complete reproducible input bundle for a rank-ordering experiment."""

    schema_version: Literal["sloforge.fabric.rank-ordering-input/v1"] = (
        "sloforge.fabric.rank-ordering-input/v1"
    )
    topology: TopologyGraph
    physical_plan: PhysicalExecutionPlan
    trace_evidence: CollectiveTraceEvidence
    config: RankOrderingExperimentConfig

    @model_validator(mode="after")
    def references_match(self) -> Self:
        if self.physical_plan.topology_fingerprint.value != canonical_hash(self.topology):
            raise ValueError("input physical plan does not reference the bundled topology")
        if self.trace_evidence.plan_id != self.physical_plan.plan_id:
            raise ValueError("input evidence does not reference the bundled physical plan")
        return self


class EdgeTraversal(PerformanceModel):
    edge_id: Annotated[str, Field(min_length=1)]
    contention_domain: Annotated[str, Field(min_length=1)]
    latency_microseconds: NonNegativeFinite
    serialization_microseconds: NonNegativeFinite
    calibration_confidence: Probability


class PairRoute(PerformanceModel):
    source_rank: Annotated[int, Field(ge=0)]
    target_rank: Annotated[int, Field(ge=0)]
    source_gpu: Annotated[str, Field(min_length=1)]
    target_gpu: Annotated[str, Field(min_length=1)]
    traversals: tuple[EdgeTraversal, ...]
    propagation_latency_microseconds: NonNegativeFinite
    bottleneck_bandwidth_gbps: PositiveFinite

    @model_validator(mode="after")
    def is_meaningful_route(self) -> Self:
        if self.source_rank == self.target_rank:
            raise ValueError("collective route cannot connect a rank to itself")
        if not self.traversals:
            raise ValueError("collective route cannot be empty")
        return self


class OrderingEvaluation(PerformanceModel):
    rank_order: tuple[Annotated[int, Field(ge=0)], ...]
    message_bytes: Annotated[int, Field(gt=0)]
    routes: tuple[PairRoute, ...]
    predicted_duration_microseconds: PositiveFinite
    minimum_calibration_confidence: Probability

    @model_validator(mode="after")
    def forms_ring(self) -> Self:
        if len(self.rank_order) < 2 or len(set(self.rank_order)) != len(self.rank_order):
            raise ValueError("rank order must contain at least two unique ranks")
        expected = {
            (self.rank_order[index], self.rank_order[(index + 1) % len(self.rank_order)])
            for index in range(len(self.rank_order))
        }
        actual = {(route.source_rank, route.target_rank) for route in self.routes}
        if expected != actual:
            raise ValueError("routes do not form the declared rank-order ring")
        return self


class RankOrderingOptimization(PerformanceModel):
    operation_id: Annotated[str, Field(min_length=1)]
    method: OptimizationMethod
    reference: OrderingEvaluation
    optimized: OrderingEvaluation
    candidates_evaluated: Annotated[int, Field(ge=1)]
    predicted_improvement_percent: float = Field(allow_inf_nan=False)

    @model_validator(mode="after")
    def same_participants(self) -> Self:
        if set(self.reference.rank_order) != set(self.optimized.rank_order):
            raise ValueError("optimized ordering changed the collective participants")
        expected = (
            100.0
            * (
                self.reference.predicted_duration_microseconds
                - self.optimized.predicted_duration_microseconds
            )
            / self.reference.predicted_duration_microseconds
        )
        if not math.isclose(expected, self.predicted_improvement_percent, abs_tol=1e-9):
            raise ValueError("predicted improvement does not match ordering evaluations")
        return self


class RankOrderingTrial(PerformanceModel):
    message_bytes: Annotated[int, Field(gt=0)]
    trial_index: Annotated[int, Field(ge=0)]
    phase: SamplePhase
    reference_duration_microseconds: PositiveFinite
    optimized_duration_microseconds: PositiveFinite
    improvement_percent: float = Field(allow_inf_nan=False)
    link_factor_digest: Sha256


class PairedRobustSummary(PerformanceModel):
    message_bytes: Annotated[int, Field(ge=0)]
    sample_count: Annotated[int, Field(ge=5)]
    reference_median_microseconds: PositiveFinite
    optimized_median_microseconds: PositiveFinite
    median_improvement_percent: float = Field(allow_inf_nan=False)
    median_absolute_deviation_percent: NonNegativeFinite
    improvement_ci_low_percent: float = Field(allow_inf_nan=False)
    improvement_ci_high_percent: float = Field(allow_inf_nan=False)
    confidence_level: Annotated[float, Field(gt=0.5, lt=1.0)]

    @model_validator(mode="after")
    def valid_interval(self) -> Self:
        if self.improvement_ci_low_percent > self.improvement_ci_high_percent:
            raise ValueError("improvement confidence interval is inverted")
        return self


class IntegrationDecision(PerformanceModel):
    status: IntegrationStatus
    enabled_by_default: bool
    rationale: Annotated[str, Field(min_length=1)]
    limiting_regimes_bytes: tuple[Annotated[int, Field(gt=0)], ...]
    trace_justified: bool

    @model_validator(mode="after")
    def guarded_enablement(self) -> Self:
        if self.enabled_by_default != (self.status is IntegrationStatus.ENABLE):
            raise ValueError("only an enable decision may set enabled_by_default")
        return self


class RankOrderingExperiment(PerformanceModel):
    schema_version: Literal["sloforge.fabric.rank-ordering-experiment/v1"] = (
        "sloforge.fabric.rank-ordering-experiment/v1"
    )
    topology_fingerprint: Sha256
    physical_plan_hash: Sha256
    trace_evidence: CollectiveTraceEvidence
    config: RankOrderingExperimentConfig
    optimization: RankOrderingOptimization
    trials: tuple[RankOrderingTrial, ...]
    summaries: tuple[PairedRobustSummary, ...]
    aggregate_summary: PairedRobustSummary
    decision: IntegrationDecision
    artifact_hash: Sha256

    @model_validator(mode="after")
    def validate_artifact(self) -> Self:
        measured = [trial for trial in self.trials if trial.phase is SamplePhase.MEASUREMENT]
        expected = self.config.measured_trials * len(self.config.message_sizes_bytes)
        if len(measured) != expected:
            raise ValueError("experiment does not contain the requested measured trials")
        if _experiment_hash(self) != self.artifact_hash:
            raise ValueError("rank-ordering experiment artifact hash mismatch")
        return self


class ExperimentArtifactPaths(PerformanceModel):
    result_json: str
    raw_samples_jsonl: str
    report_markdown: str
    result_sha256: Sha256
    raw_samples_sha256: Sha256
    report_sha256: Sha256


def _interpolate(points: tuple[CurvePoint, ...], message_bytes: int) -> float:
    """Interpolate curve medians in log-message-size space, clamped at endpoints."""

    ordered = sorted(points, key=lambda point: point.message_bytes)
    if not ordered:
        raise CalibrationError("a required measured curve is empty")
    if message_bytes <= ordered[0].message_bytes:
        return ordered[0].median
    if message_bytes >= ordered[-1].message_bytes:
        return ordered[-1].median
    for left, right in pairwise(ordered):
        if left.message_bytes <= message_bytes <= right.message_bytes:
            log_left = math.log2(left.message_bytes)
            fraction = (math.log2(message_bytes) - log_left) / (
                math.log2(right.message_bytes) - log_left
            )
            return left.median + fraction * (right.median - left.median)
    raise AssertionError("message size did not fall within a bounded interpolation interval")


def _edge_values(edge: TopologyEdge, message_bytes: int) -> tuple[float, float, float]:
    if edge.health is HealthState.FAILED:
        raise CalibrationError(f"edge {edge.edge_id} is failed")
    if not edge.bandwidth_curve_gbps or not edge.latency_curve_us:
        raise CalibrationError(
            f"edge {edge.edge_id} lacks measured bandwidth and latency curves; "
            "theoretical fallback is intentionally disabled"
        )
    bandwidth = _interpolate(edge.bandwidth_curve_gbps, message_bytes)
    latency = _interpolate(edge.latency_curve_us, message_bytes)
    confidence = edge.measurement_confidence
    if confidence is None:
        raise CalibrationError(f"edge {edge.edge_id} lacks measurement confidence")
    return bandwidth, latency, confidence


def _adjacency(topology: TopologyGraph) -> dict[str, tuple[tuple[str, TopologyEdge], ...]]:
    mutable: dict[str, list[tuple[str, TopologyEdge]]] = defaultdict(list)
    for edge in topology.edges:
        if edge.health is HealthState.FAILED:
            continue
        mutable[edge.source_node_id].append((edge.target_node_id, edge))
        if edge.directionality == "bidirectional":
            mutable[edge.target_node_id].append((edge.source_node_id, edge))
    return {
        source: tuple(sorted(neighbors, key=lambda item: (item[0], item[1].edge_id)))
        for source, neighbors in mutable.items()
    }


def _route(
    topology: TopologyGraph,
    source_gpu: str,
    target_gpu: str,
    message_bytes: int,
    source_rank: int,
    target_rank: int,
) -> PairRoute:
    graph = _adjacency(topology)
    # (cost, node, edge identifiers, traversals) makes equal-cost selection stable.
    pending: list[tuple[float, str, tuple[str, ...], tuple[EdgeTraversal, ...]]] = [
        (0.0, source_gpu, (), ())
    ]
    best: dict[str, float] = {}
    while pending:
        cost, node_id, edge_ids, traversals = heapq.heappop(pending)
        if cost >= best.get(node_id, math.inf):
            continue
        best[node_id] = cost
        if node_id == target_gpu:
            return PairRoute(
                source_rank=source_rank,
                target_rank=target_rank,
                source_gpu=source_gpu,
                target_gpu=target_gpu,
                traversals=traversals,
                propagation_latency_microseconds=sum(
                    item.latency_microseconds for item in traversals
                ),
                bottleneck_bandwidth_gbps=min(
                    message_bytes * 8.0 / (item.serialization_microseconds * 1_000.0)
                    for item in traversals
                ),
            )
        for neighbor, edge in graph.get(node_id, ()):
            try:
                bandwidth, latency, confidence = _edge_values(edge, message_bytes)
            except CalibrationError:
                # An uncalibrated branch is not a usable route. It is excluded,
                # never replaced with theoretical bandwidth. If that disconnects
                # the endpoints, the explicit error below reports the failure.
                continue
            serialization = message_bytes * 8.0 / (bandwidth * 1_000.0)
            contention_domain = edge.contention_domain or edge.sharing_group or edge.edge_id
            traversal = EdgeTraversal(
                edge_id=edge.edge_id,
                contention_domain=contention_domain,
                latency_microseconds=latency,
                serialization_microseconds=serialization,
                calibration_confidence=confidence,
            )
            heapq.heappush(
                pending,
                (
                    cost + latency + serialization,
                    neighbor,
                    (*edge_ids, edge.edge_id),
                    (*traversals, traversal),
                ),
            )
    raise CalibrationError(
        f"no fully measured route connects rank {source_rank} ({source_gpu}) "
        f"to rank {target_rank} ({target_gpu})"
    )


def _ring_iterations(operation: CollectiveOperation, rank_count: int) -> int:
    if operation.operation == "all_reduce":
        return 2 * (rank_count - 1)
    if operation.operation in {"all_gather", "reduce_scatter", "all_to_all", "broadcast"}:
        return rank_count - 1
    if operation.operation == "send_recv":
        return 1
    raise ValueError(f"unsupported collective operation {operation.operation}")


def _rank_to_gpu(placement: RankPlacement) -> dict[int, str]:
    return {binding.rank_id: binding.gpu_id for binding in placement.bindings}


def _routes_for_order(
    topology: TopologyGraph,
    placement: RankPlacement,
    operation: CollectiveOperation,
    order: tuple[int, ...],
    message_bytes: int,
) -> tuple[PairRoute, ...]:
    if set(order) != set(operation.participating_ranks):
        raise ValueError("rank order must be a permutation of collective participants")
    rank_to_gpu = _rank_to_gpu(placement)
    missing = set(order) - set(rank_to_gpu)
    if missing:
        raise ValueError(f"rank placement does not bind ranks {sorted(missing)}")
    chunk_bytes = max(1, math.ceil(message_bytes / len(order)))
    return tuple(
        _route(
            topology,
            rank_to_gpu[source],
            rank_to_gpu[target],
            chunk_bytes,
            source,
            target,
        )
        for source, target in (
            (order[index], order[(index + 1) % len(order)]) for index in range(len(order))
        )
    )


def _duration_for_routes(
    routes: tuple[PairRoute, ...],
    operation: CollectiveOperation,
    edge_factors: dict[str, float] | None = None,
) -> float:
    factors = edge_factors or {}
    domain_service: dict[str, float] = defaultdict(float)
    maximum_propagation = 0.0
    for route in routes:
        propagation = 0.0
        for traversal in route.traversals:
            factor = factors.get(traversal.edge_id, 1.0)
            if factor <= 0.0 or not math.isfinite(factor):
                raise ValueError("edge variation factors must be positive and finite")
            propagation += traversal.latency_microseconds * factor
            domain_service[traversal.contention_domain] += (
                traversal.serialization_microseconds * factor
            )
        maximum_propagation = max(maximum_propagation, propagation)
    step_duration = maximum_propagation + max(domain_service.values())
    return step_duration * _ring_iterations(operation, len(routes))


def _evaluate_order(
    topology: TopologyGraph,
    placement: RankPlacement,
    operation: CollectiveOperation,
    order: tuple[int, ...],
    message_bytes: int,
) -> OrderingEvaluation:
    routes = _routes_for_order(topology, placement, operation, order, message_bytes)
    duration = _duration_for_routes(routes, operation)
    confidence = min(
        traversal.calibration_confidence for route in routes for traversal in route.traversals
    )
    return OrderingEvaluation(
        rank_order=order,
        message_bytes=message_bytes,
        routes=routes,
        predicted_duration_microseconds=duration,
        minimum_calibration_confidence=confidence,
    )


def _heuristic_orders(
    ranks: tuple[int, ...],
    evaluate: object,
    pass_limit: int,
) -> tuple[tuple[tuple[int, ...], ...], int]:
    evaluator = evaluate
    assert callable(evaluator)
    candidates: set[tuple[int, ...]] = {ranks}
    # Start nearest-neighbor construction from every rank. The bounded start
    # set avoids a first-rank artifact and stays quadratic for large groups.
    for start in ranks:
        selected = [start]
        remaining = set(ranks) - {start}
        while remaining:
            best_next = min(
                remaining,
                key=lambda rank: (
                    evaluator(
                        (*selected, rank, *sorted(remaining - {rank}))
                    ).predicted_duration_microseconds,
                    rank,
                ),
            )
            selected.append(best_next)
            remaining.remove(best_next)
        candidates.add(tuple(selected))

    best = min(
        candidates, key=lambda order: (evaluator(order).predicted_duration_microseconds, order)
    )
    passes = 0
    improved = True
    while improved and passes < pass_limit:
        improved = False
        passes += 1
        current = evaluator(best).predicted_duration_microseconds
        for left in range(1, len(best) - 1):
            for right in range(left + 1, len(best)):
                proposal = (*best[:left], *reversed(best[left : right + 1]), *best[right + 1 :])
                candidates.add(proposal)
                proposal_duration = evaluator(proposal).predicted_duration_microseconds
                if (proposal_duration, proposal) < (current, best):
                    best = proposal
                    current = proposal_duration
                    improved = True
    return tuple(sorted(candidates)), passes


def optimize_rank_order(
    topology: TopologyGraph,
    placement: RankPlacement,
    operation: CollectiveOperation,
    *,
    message_bytes: int,
    exact_rank_limit: int = 8,
    heuristic_pass_limit: int = 20,
) -> RankOrderingOptimization:
    """Optimize a ring order using exhaustive or bounded deterministic search."""

    reference_order = operation.rank_order
    if len(reference_order) < 2:
        raise ValueError("rank-order optimization requires at least two participants")
    cache: dict[tuple[int, ...], OrderingEvaluation] = {}

    def evaluate(order: tuple[int, ...]) -> OrderingEvaluation:
        if order not in cache:
            cache[order] = _evaluate_order(topology, placement, operation, order, message_bytes)
        return cache[order]

    reference = evaluate(reference_order)
    ranks = tuple(sorted(operation.participating_ranks))
    if len(ranks) <= exact_rank_limit:
        # Fix one rank to remove cycle rotations. Directed links mean reflection
        # symmetry cannot be assumed.
        anchor = ranks[0]
        orders = (
            (anchor, *tail) for tail in permutations(rank for rank in ranks if rank != anchor)
        )
        best = min(
            orders, key=lambda order: (evaluate(order).predicted_duration_microseconds, order)
        )
        method = OptimizationMethod.EXACT
    else:
        candidates, _passes = _heuristic_orders(ranks, evaluate, heuristic_pass_limit)
        best = min(
            candidates,
            key=lambda order: (evaluate(order).predicted_duration_microseconds, order),
        )
        method = OptimizationMethod.GREEDY_TWO_OPT
    optimized = evaluate(best)
    improvement = (
        100.0
        * (reference.predicted_duration_microseconds - optimized.predicted_duration_microseconds)
        / reference.predicted_duration_microseconds
    )
    return RankOrderingOptimization(
        operation_id=operation.operation_id,
        method=method,
        reference=reference,
        optimized=optimized,
        candidates_evaluated=len(cache),
        predicted_improvement_percent=improvement,
    )


def _percentile(values: list[float], quantile: float) -> float:
    if not values:
        raise ValueError("percentile requires samples")
    ordered = sorted(values)
    position = quantile * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _summarize(
    trials: tuple[RankOrderingTrial, ...],
    *,
    message_bytes: int,
    config: RankOrderingExperimentConfig,
    seed: int,
) -> PairedRobustSummary:
    if len(trials) < 5:
        raise ValueError("paired summary requires at least five measured trials")
    improvements = tuple(trial.improvement_percent for trial in trials)
    median_improvement = statistics.median(improvements)
    rng = random.Random(seed)
    bootstrapped = [
        statistics.median(rng.choices(improvements, k=len(improvements)))
        for _ in range(config.bootstrap_rounds)
    ]
    alpha = 1.0 - config.confidence_level
    return PairedRobustSummary(
        message_bytes=message_bytes,
        sample_count=len(trials),
        reference_median_microseconds=statistics.median(
            trial.reference_duration_microseconds for trial in trials
        ),
        optimized_median_microseconds=statistics.median(
            trial.optimized_duration_microseconds for trial in trials
        ),
        median_improvement_percent=median_improvement,
        median_absolute_deviation_percent=statistics.median(
            abs(value - median_improvement) for value in improvements
        ),
        improvement_ci_low_percent=_percentile(bootstrapped, alpha / 2.0),
        improvement_ci_high_percent=_percentile(bootstrapped, 1.0 - alpha / 2.0),
        confidence_level=config.confidence_level,
    )


def _link_factors(
    routes: tuple[PairRoute, ...], rng: random.Random, variation: float
) -> dict[str, float]:
    edge_ids = sorted({item.edge_id for route in routes for item in route.traversals})
    if variation == 0.0:
        return {edge_id: 1.0 for edge_id in edge_ids}
    # A log-normal multiplier remains positive and represents correlated latency
    # inflation / bandwidth deflation for each physical link in a trial.
    return {edge_id: math.exp(rng.gauss(-(variation**2) / 2.0, variation)) for edge_id in edge_ids}


def _trial(
    *,
    trial_index: int,
    phase: SamplePhase,
    message_bytes: int,
    reference: OrderingEvaluation,
    optimized: OrderingEvaluation,
    operation: CollectiveOperation,
    rng: random.Random,
    variation: float,
) -> RankOrderingTrial:
    all_routes = (*reference.routes, *optimized.routes)
    factors = _link_factors(all_routes, rng, variation)
    reference_duration = _duration_for_routes(reference.routes, operation, factors)
    optimized_duration = _duration_for_routes(optimized.routes, operation, factors)
    improvement = 100.0 * (reference_duration - optimized_duration) / reference_duration
    factor_digest = hashlib.sha256(
        json.dumps(factors, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return RankOrderingTrial(
        message_bytes=message_bytes,
        trial_index=trial_index,
        phase=phase,
        reference_duration_microseconds=reference_duration,
        optimized_duration_microseconds=optimized_duration,
        improvement_percent=improvement,
        link_factor_digest=factor_digest,
    )


def _decision(
    evidence: CollectiveTraceEvidence,
    summaries: tuple[PairedRobustSummary, ...],
    config: RankOrderingExperimentConfig,
) -> IntegrationDecision:
    trace_justified = (
        evidence.collective_critical_path_fraction
        >= config.minimum_collective_critical_path_fraction
        and evidence.fault_free
    )
    limiting = tuple(
        summary.message_bytes
        for summary in summaries
        if summary.improvement_ci_low_percent < config.minimum_improvement_percent
    )
    if not trace_justified:
        return IntegrationDecision(
            status=IntegrationStatus.KEEP_REFERENCE,
            enabled_by_default=False,
            rationale=(
                "collective trace evidence is not fault-free or does not meet the configured "
                "critical-path share; rank ordering is not justified"
            ),
            limiting_regimes_bytes=tuple(summary.message_bytes for summary in summaries),
            trace_justified=False,
        )
    if limiting:
        crosses_zero = any(
            summary.improvement_ci_low_percent <= 0.0 <= summary.improvement_ci_high_percent
            for summary in summaries
            if summary.message_bytes in limiting
        )
        return IntegrationDecision(
            status=(
                IntegrationStatus.INCONCLUSIVE if crosses_zero else IntegrationStatus.KEEP_REFERENCE
            ),
            enabled_by_default=False,
            rationale=(
                "one or more message-size regimes lack confidence-supported practical benefit; "
                "the reference ordering remains enabled"
            ),
            limiting_regimes_bytes=limiting,
            trace_justified=True,
        )
    calibration = evidence.calibration_mode.value.replace("_", " ")
    return IntegrationDecision(
        status=IntegrationStatus.MEASURE_ON_HARDWARE,
        enabled_by_default=False,
        rationale=(
            f"{calibration} inputs produce confidence-supported modeled benefit in every "
            "regime, but the paired A/B trials are digital-twin executions; production "
            "enablement requires matched on-device measurements of both orderings"
        ),
        limiting_regimes_bytes=(),
        trace_justified=True,
    )


def _find_operation(plan: PhysicalExecutionPlan, operation_id: str) -> CollectiveOperation:
    for operation in plan.collectives.operations:
        if operation.operation_id == operation_id:
            return operation
    raise ValueError(f"physical plan has no collective operation {operation_id!r}")


def _validate_evidence_sources(
    topology: TopologyGraph,
    evidence: CollectiveTraceEvidence,
    optimization: RankOrderingOptimization,
) -> None:
    used_ids = {
        traversal.edge_id
        for evaluation in (optimization.reference, optimization.optimized)
        for route in evaluation.routes
        for traversal in route.traversals
    }
    used = [edge for edge in topology.edges if edge.edge_id in used_ids]
    if len(used) != len(used_ids):
        raise ValueError("optimized route evidence references an edge outside the topology")
    synthetic = any(
        provenance.source is DiscoverySource.SYNTHETIC
        for edge in used
        for provenance in edge.discovery_provenance
    )
    if synthetic and evidence.calibration_mode is CalibrationMode.MEASURED_HARDWARE:
        raise ValueError("synthetic topology provenance cannot be labeled as measured hardware")
    if evidence.calibration_mode is CalibrationMode.MEASURED_HARDWARE and any(
        not edge.discovery_provenance for edge in used
    ):
        raise ValueError("measured-hardware topology edges require discovery provenance")


def _validate_trace_artifact(
    evidence: CollectiveTraceEvidence,
    *,
    evidence_root: Path | None,
) -> None:
    """Verify local trace bytes and every evidence field consumed by the gate.

    ``evidence_root`` resolves relative URIs; it is intentionally not a
    containment sandbox because evidence bundles may live outside the source
    tree. Absolute and parent-relative paths remain hash- and size-checked.
    """

    if "://" in evidence.artifact_uri:
        raise ValueError("trace evidence must be materialized as a local artifact")
    artifact = Path(evidence.artifact_uri)
    if not artifact.is_absolute():
        artifact = (evidence_root or Path.cwd()) / artifact
    if not artifact.is_file():
        raise FileNotFoundError(f"trace evidence artifact does not exist: {artifact}")
    with artifact.open("rb") as handle:
        content = handle.read(MAX_TRACE_EVIDENCE_BYTES + 1)
    if len(content) > MAX_TRACE_EVIDENCE_BYTES:
        raise ValueError(f"trace evidence exceeds the {MAX_TRACE_EVIDENCE_BYTES}-byte safety limit")
    observed_digest = hashlib.sha256(content).hexdigest()
    if observed_digest != evidence.artifact_sha256:
        raise ValueError(
            "trace evidence artifact hash mismatch: "
            f"expected {evidence.artifact_sha256}, observed {observed_digest}"
        )
    try:
        payload = json.loads(content)
    except json.JSONDecodeError as error:
        raise ValueError(f"trace evidence artifact is not valid JSON: {artifact}") from error
    expected: dict[str, object] = {
        "plan_id": evidence.plan_id,
        "operation_id": evidence.operation_id,
        "calibration_mode": evidence.calibration_mode.value,
        "observed_duration_microseconds": list(evidence.observed_duration_microseconds),
        "collective_critical_path_fraction": evidence.collective_critical_path_fraction,
        "rank_wait": [sample.model_dump(mode="json") for sample in evidence.rank_wait],
        "fault_free": evidence.fault_free,
    }
    mismatched = tuple(key for key, value in expected.items() if payload.get(key) != value)
    if mismatched:
        raise ValueError(
            "trace evidence fields do not match the referenced artifact: " + ", ".join(mismatched)
        )
    if evidence.calibration_mode is CalibrationMode.MEASURED_HARDWARE:
        provenance = payload.get("provenance")
        if not isinstance(provenance, dict) or provenance.get("hardware_exercised") is not True:
            raise ValueError("measured-hardware trace evidence must attest hardware_exercised=true")


def execute_rank_ordering_experiment(
    plan: PhysicalExecutionPlan,
    topology: TopologyGraph,
    evidence: CollectiveTraceEvidence,
    config: RankOrderingExperimentConfig,
    *,
    evidence_root: Path | None = None,
) -> RankOrderingExperiment:
    """Run deterministic paired trials and produce a content-addressed artifact."""

    if evidence.plan_id != plan.plan_id:
        raise ValueError("trace evidence references a different physical plan")
    topology_fingerprint = canonical_hash(topology)
    if plan.topology_fingerprint.value != topology_fingerprint:
        raise ValueError("physical plan topology fingerprint does not match experiment topology")
    operation = _find_operation(plan, evidence.operation_id)
    optimization = optimize_rank_order(
        topology,
        plan.rank_placement,
        operation,
        message_bytes=config.optimization_message_bytes,
        exact_rank_limit=config.exact_rank_limit,
        heuristic_pass_limit=config.heuristic_pass_limit,
    )
    _validate_evidence_sources(topology, evidence, optimization)
    _validate_trace_artifact(evidence, evidence_root=evidence_root)

    rng = random.Random(config.seed)
    trials: list[RankOrderingTrial] = []
    summaries: list[PairedRobustSummary] = []
    for regime_index, message_bytes in enumerate(config.message_sizes_bytes):
        reference = _evaluate_order(
            topology,
            plan.rank_placement,
            operation,
            optimization.reference.rank_order,
            message_bytes,
        )
        optimized = _evaluate_order(
            topology,
            plan.rank_placement,
            operation,
            optimization.optimized.rank_order,
            message_bytes,
        )
        for index in range(config.warmup_trials):
            trials.append(
                _trial(
                    trial_index=index,
                    phase=SamplePhase.WARMUP,
                    message_bytes=message_bytes,
                    reference=reference,
                    optimized=optimized,
                    operation=operation,
                    rng=rng,
                    variation=config.link_variation_fraction,
                )
            )
        regime_trials: list[RankOrderingTrial] = []
        for index in range(config.measured_trials):
            trial = _trial(
                trial_index=index,
                phase=SamplePhase.MEASUREMENT,
                message_bytes=message_bytes,
                reference=reference,
                optimized=optimized,
                operation=operation,
                rng=rng,
                variation=config.link_variation_fraction,
            )
            trials.append(trial)
            regime_trials.append(trial)
        summaries.append(
            _summarize(
                tuple(regime_trials),
                message_bytes=message_bytes,
                config=config,
                seed=config.seed + 10_000 + regime_index,
            )
        )
    measured = tuple(trial for trial in trials if trial.phase is SamplePhase.MEASUREMENT)
    aggregate = _summarize(
        measured,
        message_bytes=0,
        config=config,
        seed=config.seed + 20_000,
    )
    provisional = RankOrderingExperiment.model_construct(
        topology_fingerprint=topology_fingerprint,
        physical_plan_hash=canonical_hash(plan),
        trace_evidence=evidence,
        config=config,
        optimization=optimization,
        trials=tuple(trials),
        summaries=tuple(summaries),
        aggregate_summary=aggregate,
        decision=_decision(evidence, tuple(summaries), config),
        artifact_hash="",
    )
    payload = provisional.model_dump(mode="json")
    payload["artifact_hash"] = _experiment_hash(provisional)
    return RankOrderingExperiment.model_validate_json(json.dumps(payload))


def _experiment_hash(experiment: RankOrderingExperiment) -> str:
    payload = experiment.model_dump(mode="json", exclude={"artifact_hash"})
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        with suppress(FileNotFoundError):
            os.unlink(temporary_name)
        raise


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _report(experiment: RankOrderingExperiment) -> str:
    calibration = experiment.trace_evidence.calibration_mode.value.replace("_", " ")
    lines = [
        "# Collective rank-ordering experiment",
        "",
        f"- Artifact hash: `{experiment.artifact_hash}`",
        f"- Physical plan: `{experiment.trace_evidence.plan_id}`",
        f"- Collective: `{experiment.trace_evidence.operation_id}`",
        f"- Calibration: **{calibration}**",
        f"- Search: `{experiment.optimization.method.value}` "
        f"({experiment.optimization.candidates_evaluated} candidates)",
        f"- Reference order: `{list(experiment.optimization.reference.rank_order)}`",
        f"- Selected order: `{list(experiment.optimization.optimized.rank_order)}`",
        f"- Integration decision: **{experiment.decision.status.value}**",
        f"- Enabled by default: **{str(experiment.decision.enabled_by_default).lower()}**",
        "",
        "## Results",
        "",
        "| Message bytes | Reference median (us) | Selected median (us) | Median benefit | CI |",
        "|---:|---:|---:|---:|---:|",
    ]
    for summary in experiment.summaries:
        lines.append(
            f"| {summary.message_bytes} | {summary.reference_median_microseconds:.3f} | "
            f"{summary.optimized_median_microseconds:.3f} | "
            f"{summary.median_improvement_percent:.3f}% | "
            f"[{summary.improvement_ci_low_percent:.3f}%, "
            f"{summary.improvement_ci_high_percent:.3f}%] |"
        )
    lines.extend(
        (
            "",
            "## Integration decision",
            "",
            experiment.decision.rationale,
            "",
            "Warmup trials were retained in the raw artifact but excluded from all summaries. "
            "Trials use matched per-link perturbations, so the reported interval is paired.",
            "",
        )
    )
    if experiment.trace_evidence.calibration_mode is CalibrationMode.SYNTHETIC_CALIBRATED:
        caveat = (
            "This is a deterministic synthetic-calibration result, not a GPU measurement. "
            "It cannot enable a production default."
        )
    else:
        caveat = (
            "The input curves and trace are hardware-backed, but the paired ordering trials "
            "in this artifact are digital-twin executions. They cannot enable a production "
            "default without a matched on-device A/B run."
        )
    lines.extend((f"> {caveat}", ""))
    return "\n".join(lines)


def write_experiment_artifacts(
    output_directory: Path,
    experiment: RankOrderingExperiment,
    *,
    report_path: Path | None = None,
) -> ExperimentArtifactPaths:
    """Write canonical result, complete raw trials, and an artifact-derived report."""

    result_path = output_directory / "rank-ordering-experiment.json"
    raw_path = output_directory / "rank-ordering-raw-samples.jsonl"
    selected_report_path = report_path or output_directory / "rank-ordering-experiment.md"
    result = (
        json.dumps(
            experiment.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode()
    raw = "".join(
        json.dumps(trial.model_dump(mode="json"), sort_keys=True, separators=(",", ":")) + "\n"
        for trial in experiment.trials
    ).encode()
    report = _report(experiment).encode()
    _atomic_write(result_path, result)
    _atomic_write(raw_path, raw)
    _atomic_write(selected_report_path, report)
    return ExperimentArtifactPaths(
        result_json=str(result_path),
        raw_samples_jsonl=str(raw_path),
        report_markdown=str(selected_report_path),
        result_sha256=_sha256(result),
        raw_samples_sha256=_sha256(raw),
        report_sha256=_sha256(report),
    )
