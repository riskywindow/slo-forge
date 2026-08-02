"""Deterministic flow-level cold-start simulator for artifact DAGs."""

from __future__ import annotations

import heapq
import math
import random
from dataclasses import dataclass
from typing import Literal

from sloforge.warmpath.models import (
    ArtifactGraph,
    ArtifactPlacement,
    ColdStartSimulation,
    ColdStartTrial,
    MaterializationMode,
    StageMeasurement,
    StartupProfile,
    StartupStage,
    StartupStagePrediction,
    StorageTierSpec,
)
from sloforge.warmpath.statistics import percentile


@dataclass(frozen=True)
class _DurationEstimate:
    duration_ms: float
    relative_dispersion: float
    source: Literal["measured", "theoretical", "warm"]


def _measurements(profile: StartupProfile) -> dict[tuple[str, str], tuple[StageMeasurement, ...]]:
    grouped: dict[tuple[str, str], list[StageMeasurement]] = {}
    for measurement in profile.measurements:
        grouped.setdefault((measurement.artifact_id, measurement.tier_id), []).append(measurement)
    return {key: tuple(value) for key, value in grouped.items()}


def _duration(
    artifact_size: int,
    rebuild_time_ms: float,
    placement: ArtifactPlacement,
    tier: StorageTierSpec,
    measurements: tuple[StageMeasurement, ...],
) -> _DurationEstimate:
    if placement.mode == MaterializationMode.KEEP_WARM:
        return _DurationEstimate(0.0, 0.0, "warm")
    if placement.mode == MaterializationMode.REBUILD:
        return _DurationEstimate(rebuild_time_ms, 0.10, "theoretical")
    measured = sum(item.p95_ms for item in measurements)
    if measured > 0.0:
        dispersion = sum(item.median_absolute_deviation_ms for item in measurements) / measured
        return _DurationEstimate(measured, max(0.01, min(dispersion, 0.75)), "measured")
    transfer_ms = tier.base_read_latency_ms + (
        1_000.0 * artifact_size / tier.read_bandwidth_bytes_per_second
    )
    return _DurationEstimate(transfer_ms, 0.20, "theoretical")


def _schedule(
    *,
    graph: ArtifactGraph,
    placements: tuple[ArtifactPlacement, ...],
    profile: StartupProfile,
    rng: random.Random | None,
) -> tuple[
    float | None,
    bool,
    bool,
    tuple[StartupStagePrediction, ...],
    set[Literal["measured", "theoretical", "warm"]],
]:
    by_placement = {item.artifact_id: item for item in placements}
    tiers = {tier.tier_id: tier for tier in profile.tiers}
    measured = _measurements(profile)
    lanes = {
        tier.tier_id: [0.0 for _ in range(tier.maximum_parallel_reads)] for tier in profile.tiers
    }
    for values in lanes.values():
        heapq.heapify(values)
    finishes: dict[str, float] = {}
    predictions: list[StartupStagePrediction] = []
    used_sources: set[Literal["measured", "theoretical", "warm"]] = set()
    restore_failed = False
    used_rebuild = False
    deferred: list[tuple[str, str, float, Literal["measured", "theoretical", "warm"]]] = []

    for artifact in graph.topological_order():
        placement = by_placement.get(artifact.artifact_id)
        if placement is None:
            raise ValueError(f"missing placement for artifact {artifact.artifact_id}")
        tier = tiers.get(placement.tier_id)
        if tier is None:
            raise ValueError(f"placement references unknown tier {placement.tier_id}")
        estimate = _duration(
            artifact.size_bytes,
            artifact.rebuild_time_ms,
            placement,
            tier,
            measured.get((artifact.artifact_id, tier.tier_id), ()),
        )
        used_sources.add(estimate.source)
        duration = estimate.duration_ms
        dependency_finish = max((finishes[edge] for edge in artifact.dependencies), default=0.0)
        if placement.mode == MaterializationMode.LAZY_RESTORE:
            finishes[artifact.artifact_id] = dependency_finish
            deferred.append((artifact.artifact_id, tier.tier_id, duration, estimate.source))
            continue
        if rng is not None and duration > 0.0:
            sigma = math.sqrt(math.log(estimate.relative_dispersion**2 + 1.0))
            duration *= rng.lognormvariate(-(sigma**2) / 2.0, sigma)

        failed = (
            rng is not None
            and placement.mode == MaterializationMode.EAGER_RESTORE
            and rng.random() < tier.restore_failure_probability
        )
        if failed:
            restore_failed = True
            if artifact.rebuild_time_ms <= 0.0:
                return None, True, used_rebuild, tuple(predictions), used_sources
            duration += artifact.rebuild_time_ms
            used_rebuild = True

        lane_available = heapq.heappop(lanes[tier.tier_id])
        start = max(dependency_finish, lane_available)
        finish = start + duration
        heapq.heappush(lanes[tier.tier_id], finish)
        finishes[artifact.artifact_id] = finish
        stage = (
            StartupStage.RESTORE
            if placement.mode
            in {MaterializationMode.EAGER_RESTORE, MaterializationMode.LAZY_RESTORE}
            else StartupStage.RUNTIME_INITIALIZATION
        )
        predictions.append(
            StartupStagePrediction(
                artifact_id=artifact.artifact_id,
                stage=stage,
                start_ms=start,
                finish_ms=finish,
                resource=tier.tier_id,
                estimate_source=estimate.source,
            )
        )

    readiness_finishes = [
        finishes[item.artifact_id]
        for item in graph.artifacts
        if item.required_for_readiness
        or by_placement[item.artifact_id].mode != MaterializationMode.LAZY_RESTORE
    ]
    readiness = max(readiness_finishes, default=0.0)
    for artifact_id, tier_id, duration, source in deferred:
        lane_available = heapq.heappop(lanes[tier_id])
        start = max(readiness, lane_available)
        finish = start + duration
        heapq.heappush(lanes[tier_id], finish)
        predictions.append(
            StartupStagePrediction(
                artifact_id=artifact_id,
                stage=StartupStage.RESTORE,
                start_ms=start,
                finish_ms=finish,
                resource=tier_id,
                estimate_source=source,
            )
        )
    return (
        readiness,
        restore_failed,
        used_rebuild,
        tuple(predictions),
        used_sources,
    )


def simulate_cold_start(
    *,
    graph: ArtifactGraph,
    placements: tuple[ArtifactPlacement, ...],
    profile: StartupProfile,
    seed: int,
    trial_count: int = 101,
) -> ColdStartSimulation:
    """Simulate concurrent tier reads, dependencies, restore failures, and fallback rebuilds."""

    if trial_count < 3:
        raise ValueError("cold-start simulation requires at least three trials")
    if profile.graph_id != graph.graph_id:
        raise ValueError("profile and artifact graph identifiers differ")
    deterministic = _schedule(
        graph=graph,
        placements=placements,
        profile=profile,
        rng=None,
    )
    stage_predictions = deterministic[3]
    sources = deterministic[4]
    rng = random.Random(seed)
    trials: list[ColdStartTrial] = []
    successes: list[float] = []
    failures = 0
    for index in range(trial_count):
        ready, failed, rebuilt, _, _ = _schedule(
            graph=graph,
            placements=placements,
            profile=profile,
            rng=rng,
        )
        if failed:
            failures += 1
        if ready is not None:
            successes.append(ready)
        trials.append(
            ColdStartTrial(
                trial_index=index,
                ready_time_ms=ready,
                restore_failed=failed,
                used_rebuild_fallback=rebuilt,
            )
        )
    if not successes:
        raise RuntimeError("every simulated cold start failed without a rebuild fallback")
    p50 = percentile(successes, 0.5)
    p95 = percentile(successes, 0.95)
    low = min(successes)
    high = max(max(successes), p95)
    return ColdStartSimulation(
        seed=seed,
        trials=tuple(trials),
        p50_ready_time_ms=p50,
        p95_ready_time_ms=p95,
        interval_low_ms=min(low, p50),
        interval_high_ms=high,
        restore_failure_probability=failures / trial_count,
        stage_predictions=stage_predictions,
        estimate_sources=tuple(sorted(sources)),
    )
