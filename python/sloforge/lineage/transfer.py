"""Deterministic related-task retrieval and lineage-seeded search initialization."""

from __future__ import annotations

import hashlib
import math
from datetime import datetime

from .models import (
    ConstraintPredicate,
    DependencySelector,
    EvidenceFreshness,
    EvidenceRecord,
    EvidenceResult,
    RelatedTransformation,
    SearchInitialization,
    TaskFeatures,
    TransferOutcome,
    TransformationOutcome,
    TransformationRecord,
    UnseededProposal,
)
from .store import LineageStore
from .versioning import version_matches


def _ensure_aware(value: datetime) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("lineage evaluation time must include a timezone")


def _jaccard(left: tuple[str, ...], right: tuple[str, ...]) -> float:
    left_set = set(left)
    right_set = set(right)
    union = left_set | right_set
    return len(left_set & right_set) / len(union) if union else 1.0


def _task_similarity(left: TaskFeatures, right: TaskFeatures) -> float:
    return (
        0.25 * float(left.model_family == right.model_family)
        + 0.30 * _jaccard(left.operator_families, right.operator_families)
        + 0.20 * float(left.hardware_architecture == right.hardware_architecture)
        + 0.15 * _jaccard(left.workload_regimes, right.workload_regimes)
        + 0.10 * _jaccard(left.topology_features, right.topology_features)
    )


def _dependency_matches(task: TaskFeatures, selector: DependencySelector) -> bool:
    for dependency in task.dependencies:
        if dependency.kind is selector.kind and dependency.name == selector.name:
            return version_matches(dependency.version, selector.version_range)
    return False


def transformation_is_applicable(transformation: TransformationRecord, task: TaskFeatures) -> bool:
    return (
        task.model_family in transformation.applicable_model_families
        and task.hardware_architecture in transformation.applicable_hardware
        and bool(set(task.operator_families) & set(transformation.applicable_operations))
        and bool(set(task.workload_regimes) & set(transformation.applicable_workloads))
        and all(
            _dependency_matches(task, selector)
            for selector in transformation.dependency_preconditions
        )
    )


def constraint_matches(predicate: ConstraintPredicate, task: TaskFeatures) -> bool:
    checks: list[bool] = []
    if predicate.model_families:
        checks.append(task.model_family in predicate.model_families)
    if predicate.hardware_architectures:
        checks.append(task.hardware_architecture in predicate.hardware_architectures)
    if predicate.workload_regimes:
        checks.append(bool(set(task.workload_regimes) & set(predicate.workload_regimes)))
    checks.extend(_dependency_matches(task, item) for item in predicate.dependency_selectors)
    return bool(checks) and all(checks)


def effective_confidence(
    evidence: EvidenceRecord, *, as_of: datetime, half_life_days: float = 90.0
) -> float:
    _ensure_aware(as_of)
    if half_life_days <= 0:
        raise ValueError("half_life_days must be positive")
    if evidence.freshness is EvidenceFreshness.STALE or evidence.result is not EvidenceResult.PASS:
        return 0.0
    if evidence.observed_at > as_of or evidence.valid_until <= as_of:
        return 0.0
    age_days = (as_of - evidence.observed_at).total_seconds() / 86_400.0
    return evidence.base_confidence * math.pow(0.5, age_days / half_life_days)


def _evidence_relevance(evidence: EvidenceRecord, task: TaskFeatures) -> float:
    model = 1.0 if evidence.model_family == task.model_family else 0.35
    hardware = 1.0 if evidence.hardware_architecture == task.hardware_architecture else 0.2
    workload = _jaccard(evidence.workload_regimes, task.workload_regimes)
    target_dependencies = {(item.kind, item.name): item.version for item in task.dependencies}
    if evidence.dependencies:
        dependency = sum(
            target_dependencies.get((item.kind, item.name)) == item.version
            for item in evidence.dependencies
        ) / len(evidence.dependencies)
    else:
        dependency = 1.0
    return 0.3 * model + 0.3 * hardware + 0.2 * workload + 0.2 * dependency


def _tie_break(seed: int, task_id: str, transformation_id: str) -> str:
    value = f"{seed}:{task_id}:{transformation_id}".encode()
    return hashlib.sha256(value).hexdigest()


def retrieve_related_transformations(
    store: LineageStore,
    task: TaskFeatures,
    *,
    seed: int,
    as_of: datetime,
    limit: int = 10,
    scan_limit: int = 1000,
    half_life_days: float = 90.0,
) -> tuple[RelatedTransformation, ...]:
    if not 1 <= limit <= 1000:
        raise ValueError("retrieval limit must be between 1 and 1000")
    if not limit <= scan_limit <= 100_000:
        raise ValueError("scan_limit must be bounded and no smaller than limit")
    _ensure_aware(as_of)
    constraints = store.list_constraints(limit=scan_limit)
    prior_transfers = store.list_transfers(limit=scan_limit)
    ranked: list[tuple[RelatedTransformation, str]] = []
    for transformation in store.list_transformations(limit=scan_limit):
        if transformation.outcome is not TransformationOutcome.ACCEPTED:
            continue
        if not transformation_is_applicable(transformation, task):
            continue
        if any(
            (
                constraint.transformation_id == transformation.transformation_id
                or (
                    constraint.transformation_id is None
                    and constraint.transformation_family == transformation.family
                )
            )
            and constraint_matches(constraint.predicate, task)
            for constraint in constraints
        ):
            continue
        source_task = store.task_for_candidate(transformation.source_candidate_id)
        evidence = store.evidence_for_transformation(
            transformation.transformation_id, limit=scan_limit
        )
        confidence_pairs = tuple(
            (
                item,
                effective_confidence(item, as_of=as_of, half_life_days=half_life_days)
                * _evidence_relevance(item, task),
            )
            for item in evidence
        )
        useful = tuple(
            (item, confidence) for item, confidence in confidence_pairs if confidence > 0
        )
        if not useful:
            continue
        confidence = sum(item[1] for item in useful) / len(useful)
        feature_score = _task_similarity(source_task, task)
        history = [
            item
            for item in prior_transfers
            if item.transformation_id == transformation.transformation_id
        ]
        history_adjustment = sum(
            _task_similarity(store.get_task(item.target_task_id), task)
            * (
                0.08
                if item.outcome is TransferOutcome.IMPROVED
                else -0.20
                if item.outcome is TransferOutcome.NEGATIVE_TRANSFER
                else -0.08
                if item.outcome is TransferOutcome.REJECTED
                else 0.0
            )
            for item in history
        )
        benefit_bonus = 0.05 * math.tanh(max(0.0, transformation.expected_benefit))
        score = max(
            0.0,
            feature_score * (0.5 + 0.5 * confidence) + benefit_bonus + history_adjustment,
        )
        related = RelatedTransformation(
            transformation_id=transformation.transformation_id,
            source_task_id=source_task.task_id,
            score=score,
            effective_confidence=min(1.0, confidence),
            evidence_ids=tuple(item.evidence_id for item, _ in useful),
            rationale=(
                f"feature_similarity={feature_score:.6f}",
                f"evidence_confidence={confidence:.6f}",
                f"history_adjustment={history_adjustment:.6f}",
                "reverification_required",
            ),
        )
        ranked.append((related, _tie_break(seed, task.task_id, transformation.transformation_id)))
    ranked.sort(key=lambda item: (-item[0].score, item[1], item[0].transformation_id))
    return tuple(item[0] for item in ranked[:limit])


def initialize_search_from_lineage(
    store: LineageStore,
    task: TaskFeatures,
    *,
    seed: int,
    as_of: datetime,
    population_size: int,
    lineage_fraction: float = 0.6,
    scan_limit: int = 1000,
) -> SearchInitialization:
    if not 1 <= population_size <= 10_000:
        raise ValueError("population_size must be between 1 and 10000")
    if not 0.0 <= lineage_fraction <= 0.8:
        raise ValueError("lineage_fraction must preserve at least 20 percent diversity")
    lineage_slots = min(population_size, math.floor(population_size * lineage_fraction))
    seeds = (
        retrieve_related_transformations(
            store,
            task,
            seed=seed,
            as_of=as_of,
            limit=max(1, lineage_slots),
            scan_limit=scan_limit,
        )[:lineage_slots]
        if lineage_slots
        else ()
    )
    unseeded = tuple(
        UnseededProposal(
            proposal_id="unseeded-"
            + hashlib.sha256(f"{seed}:{task.task_id}:{index}".encode()).hexdigest()[:24],
            seed=seed + index + 1,
        )
        for index in range(population_size - len(seeds))
    )
    return SearchInitialization(
        task_id=task.task_id,
        seed=seed,
        lineage_seeds=seeds,
        unseeded_proposals=unseeded,
    )
