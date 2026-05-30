"""Deterministic constrained planner for startup artifact placement."""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Literal

from sloforge.util import canonical_json, sha256_bytes
from sloforge.warmpath.models import (
    ArtifactGraph,
    ArtifactNode,
    ArtifactPlacement,
    ColdStartSimulation,
    MaterializationMode,
    RejectedWarmPathCandidate,
    StartupProfile,
    StorageTierSpec,
    WarmPathObjective,
    WarmPathPlan,
    compatibility_violations,
    safe_identifier,
    security_allows,
)
from sloforge.warmpath.simulator import simulate_cold_start


@dataclass(frozen=True)
class _Candidate:
    placements: tuple[ArtifactPlacement, ...]
    warm_replicas: int
    quick_score: float


def _hash_model(value: object) -> str:
    from pydantic import BaseModel

    if not isinstance(value, BaseModel):
        raise TypeError("canonical model hash requires a Pydantic model")
    return sha256_bytes(canonical_json(value.model_dump(mode="json")).encode())


def _tier_duration(artifact: ArtifactNode, tier: StorageTierSpec) -> float:
    return tier.base_read_latency_ms + (
        1_000.0 * artifact.size_bytes / tier.read_bandwidth_bytes_per_second
    )


def _readiness_dependencies(graph: ArtifactGraph) -> set[str]:
    by_id = {artifact.artifact_id: artifact for artifact in graph.artifacts}
    required = {
        artifact.artifact_id for artifact in graph.artifacts if artifact.required_for_readiness
    }
    pending = list(required)
    while pending:
        current = pending.pop()
        for dependency in by_id[current].dependencies:
            if dependency not in required:
                required.add(dependency)
                pending.append(dependency)
    return required


def _artifact_choices(
    artifact: ArtifactNode,
    *,
    tiers: tuple[StorageTierSpec, ...],
    compatible: bool,
    readiness_required: bool,
) -> tuple[ArtifactPlacement, ...]:
    choices: list[ArtifactPlacement] = []
    for tier in tiers:
        if not security_allows(tier, artifact):
            continue
        duration = _tier_duration(artifact, tier)
        if compatible:
            choices.append(
                ArtifactPlacement(
                    artifact_id=artifact.artifact_id,
                    tier_id=tier.tier_id,
                    mode=MaterializationMode.EAGER_RESTORE,
                    prefetch_order=0,
                    expected_duration_ms=duration,
                    estimate_source="theoretical",
                )
            )
            if artifact.lazy_restore_allowed and not readiness_required:
                choices.append(
                    ArtifactPlacement(
                        artifact_id=artifact.artifact_id,
                        tier_id=tier.tier_id,
                        mode=MaterializationMode.LAZY_RESTORE,
                        prefetch_order=0,
                        expected_duration_ms=duration,
                        estimate_source="theoretical",
                    )
                )
        if artifact.rebuild_time_ms > 0.0:
            choices.append(
                ArtifactPlacement(
                    artifact_id=artifact.artifact_id,
                    tier_id=tier.tier_id,
                    mode=MaterializationMode.REBUILD,
                    prefetch_order=0,
                    expected_duration_ms=artifact.rebuild_time_ms,
                    estimate_source="theoretical",
                )
            )
    return tuple(choices)


def _capacity_ok(
    placements: tuple[ArtifactPlacement, ...],
    graph: ArtifactGraph,
    tiers: dict[str, StorageTierSpec],
) -> bool:
    artifacts = {artifact.artifact_id: artifact for artifact in graph.artifacts}
    used: dict[str, int] = {tier_id: 0 for tier_id in tiers}
    for placement in placements:
        if placement.mode != MaterializationMode.KEEP_WARM:
            used[placement.tier_id] += artifacts[placement.artifact_id].size_bytes
    return all(used[tier_id] <= tiers[tier_id].capacity_bytes for tier_id in tiers)


def _hourly_cost(
    placements: tuple[ArtifactPlacement, ...],
    graph: ArtifactGraph,
    tiers: dict[str, StorageTierSpec],
    warm_replicas: int,
    objective: WarmPathObjective,
) -> float:
    artifacts = {artifact.artifact_id: artifact for artifact in graph.artifacts}
    storage = sum(
        artifacts[item.artifact_id].size_bytes
        / (1024.0**3)
        * tiers[item.tier_id].hourly_cost_per_gib
        for item in placements
        if item.mode != MaterializationMode.KEEP_WARM
    )
    return storage + warm_replicas * objective.warm_replica_hourly_cost


def _enumerate_candidates(
    choices: tuple[tuple[ArtifactPlacement, ...], ...],
    *,
    objective: WarmPathObjective,
    graph: ArtifactGraph,
    exhaustive_limit: int,
    beam_width: int,
) -> tuple[tuple[_Candidate, ...], Literal["exhaustive", "beam"]]:
    combination_count = 1
    for options in choices:
        combination_count *= len(options)
    warm_options = objective.maximum_warm_replicas + 1
    if combination_count * warm_options <= exhaustive_limit:
        candidates: list[_Candidate] = []
        for placements in itertools.product(*choices):
            score = sum(item.expected_duration_ms for item in placements)
            candidates.append(_Candidate(tuple(placements), 0, score))
        if objective.maximum_warm_replicas > 0:
            base = tuple(options[0] for options in choices)
            for count in range(1, objective.maximum_warm_replicas + 1):
                warm = tuple(
                    item.model_copy(
                        update={
                            "mode": MaterializationMode.KEEP_WARM,
                            "expected_duration_ms": 0.0,
                            "estimate_source": "warm",
                        }
                    )
                    for item in base
                )
                candidates.append(_Candidate(warm, count, 0.0))
        return tuple(candidates), "exhaustive"

    beam: list[tuple[ArtifactPlacement, ...]] = [()]
    for options in choices:
        expanded = [(*prefix, option) for prefix in beam for option in options]
        expanded.sort(
            key=lambda placements: (
                sum(item.expected_duration_ms for item in placements),
                tuple((item.tier_id, item.mode.value) for item in placements),
            )
        )
        beam = expanded[:beam_width]
    candidates = [
        _Candidate(items, 0, sum(item.expected_duration_ms for item in items)) for items in beam
    ]
    if objective.maximum_warm_replicas:
        warm = tuple(
            item.model_copy(
                update={
                    "mode": MaterializationMode.KEEP_WARM,
                    "expected_duration_ms": 0.0,
                    "estimate_source": "warm",
                }
            )
            for item in beam[0]
        )
        candidates.extend(
            _Candidate(warm, count, 0.0) for count in range(1, objective.maximum_warm_replicas + 1)
        )
    return tuple(candidates), "beam"


def compile_warmpath(
    *,
    graph: ArtifactGraph,
    profile: StartupProfile,
    objective: WarmPathObjective,
    seed: int = 17,
    simulation_trials: int = 101,
    exhaustive_limit: int = 20_000,
    beam_width: int = 128,
) -> WarmPathPlan:
    """Compile a measured cold-start profile into the lowest-objective feasible plan."""

    if profile.graph_id != graph.graph_id:
        raise ValueError("profile does not describe the supplied artifact graph")
    if exhaustive_limit <= 0 or beam_width <= 0:
        raise ValueError("optimizer limits must be positive")
    artifact_ids = {artifact.artifact_id for artifact in graph.artifacts}
    unknown_measurements = {item.artifact_id for item in profile.measurements} - artifact_ids
    if unknown_measurements:
        raise ValueError(
            f"profile contains measurements for unknown artifacts: {unknown_measurements}"
        )

    readiness = _readiness_dependencies(graph)
    choices: list[tuple[ArtifactPlacement, ...]] = []
    rejected: list[RejectedWarmPathCandidate] = []
    for artifact in graph.topological_order():
        violations = compatibility_violations(artifact.compatibility, profile.host)
        options = _artifact_choices(
            artifact,
            tiers=profile.tiers,
            compatible=not violations,
            readiness_required=artifact.artifact_id in readiness,
        )
        if not options:
            security_blocked = all(not security_allows(tier, artifact) for tier in profile.tiers)
            detail = (
                f"no storage tier admits security class {artifact.security_class.value}"
                if security_blocked
                else "; ".join(violations)
            )
            raise ValueError(f"artifact {artifact.artifact_id} is infeasible: {detail}")
        if violations:
            rejected.append(
                RejectedWarmPathCandidate(
                    candidate_id=safe_identifier(f"restore-{artifact.artifact_id}"),
                    reason_code="compatibility",
                    explanation=f"snapshot restore rejected: {'; '.join(violations)}; rebuild retained",
                )
            )
        choices.append(options)

    candidates, strategy = _enumerate_candidates(
        tuple(choices),
        objective=objective,
        graph=graph,
        exhaustive_limit=exhaustive_limit,
        beam_width=beam_width,
    )
    tiers = {tier.tier_id: tier for tier in profile.tiers}
    best: tuple[float, _Candidate, ColdStartSimulation, float] | None = None
    best_key: tuple[object, ...] | None = None
    evaluated = 0
    for index, candidate in enumerate(candidates):
        if not _capacity_ok(candidate.placements, graph, tiers):
            rejected.append(
                RejectedWarmPathCandidate(
                    candidate_id=f"candidate-{index}",
                    reason_code="capacity",
                    explanation="artifact bytes exceed at least one selected storage tier capacity",
                )
            )
            continue
        cost = _hourly_cost(candidate.placements, graph, tiers, candidate.warm_replicas, objective)
        if objective.maximum_hourly_cost is not None and cost > objective.maximum_hourly_cost:
            rejected.append(
                RejectedWarmPathCandidate(
                    candidate_id=f"candidate-{index}",
                    reason_code="cost_budget",
                    explanation=f"hourly cost {cost:.6f} exceeds {objective.maximum_hourly_cost:.6f}",
                )
            )
            continue
        try:
            simulation = simulate_cold_start(
                graph=graph,
                placements=candidate.placements,
                profile=profile,
                seed=seed + index,
                trial_count=simulation_trials,
            )
        except RuntimeError as error:
            rejected.append(
                RejectedWarmPathCandidate(
                    candidate_id=f"candidate-{index}",
                    reason_code="failure",
                    explanation=str(error),
                )
            )
            continue
        evaluated += 1
        if (
            objective.maximum_p95_ready_time_ms is not None
            and simulation.p95_ready_time_ms > objective.maximum_p95_ready_time_ms
        ):
            rejected.append(
                RejectedWarmPathCandidate(
                    candidate_id=f"candidate-{index}",
                    reason_code="startup_slo",
                    explanation=(
                        f"predicted p95 {simulation.p95_ready_time_ms:.3f} ms exceeds "
                        f"{objective.maximum_p95_ready_time_ms:.3f} ms"
                    ),
                )
            )
            continue
        score = (
            objective.ready_time_weight * simulation.p95_ready_time_ms
            + objective.hourly_cost_weight * cost
            + objective.failure_risk_weight * simulation.restore_failure_probability
        )
        key = (
            score,
            simulation.p95_ready_time_ms,
            cost,
            tuple((item.tier_id, item.mode.value) for item in candidate.placements),
        )
        if best_key is None or key < best_key:
            best = (score, candidate, simulation, cost)
            best_key = key
    if best is None:
        raise RuntimeError(
            "WarmPath found no candidate satisfying capacity, cost, and startup constraints"
        )

    score, candidate, simulation, cost = best
    stages = {item.artifact_id: item for item in simulation.stage_predictions}
    placements = tuple(
        placement.model_copy(
            update={
                "prefetch_order": index,
                "expected_duration_ms": (
                    stages[placement.artifact_id].finish_ms - stages[placement.artifact_id].start_ms
                ),
                "estimate_source": stages[placement.artifact_id].estimate_source,
            }
        )
        for index, placement in enumerate(candidate.placements)
    )
    graph_hash = _hash_model(graph)
    profile_hash = _hash_model(profile)
    plan_fingerprint = sha256_bytes(
        canonical_json(
            {
                "graph_hash": graph_hash,
                "profile_hash": profile_hash,
                "placements": [item.model_dump(mode="json") for item in placements],
                "seed": seed,
            }
        ).encode()
    )
    return WarmPathPlan(
        plan_id=f"warmpath-{plan_fingerprint[:16]}",
        profile_id=profile.profile_id,
        graph_id=graph.graph_id,
        host_fingerprint=profile.host.host_fingerprint,
        placements=placements,
        warm_replica_count=candidate.warm_replicas,
        predicted_p50_ready_time_ms=simulation.p50_ready_time_ms,
        predicted_p95_ready_time_ms=simulation.p95_ready_time_ms,
        prediction_interval_low_ms=simulation.interval_low_ms,
        prediction_interval_high_ms=simulation.interval_high_ms,
        predicted_hourly_cost=cost,
        predicted_restore_failure_probability=simulation.restore_failure_probability,
        objective_value=score,
        stage_predictions=simulation.stage_predictions,
        rejected_candidates=tuple(rejected[:1_000]),
        optimizer=strategy,
        optimizer_seed=seed,
        evaluated_candidate_count=evaluated,
        evidence_references=(
            profile.raw_artifact_directory,
            profile.environment_manifest_path,
        ),
        compiler_version="sloforge-warmpath/0.1.0",
        graph_hash=graph_hash,
        profile_hash=profile_hash,
    )
