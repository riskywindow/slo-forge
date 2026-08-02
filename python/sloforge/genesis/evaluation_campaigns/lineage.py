"""Artifact-backed deterministic H5 lineage-transfer evaluation campaign.

The evaluator deliberately reports abstract evaluation-time and candidate units.  It
does not execute a hardware benchmark and cannot support a hardware speedup claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import statistics
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, model_validator

from sloforge.genesis.ir import canonical_json
from sloforge.lineage import (
    CandidateDisposition,
    CandidateRecord,
    DependencyKind,
    DependencySelector,
    DependencyVersion,
    EvidenceFreshness,
    EvidenceRecord,
    EvidenceResult,
    EvidenceTargetKind,
    InvalidationEvent,
    LineageStore,
    MetricDirection,
    ObjectiveMeasurement,
    RelatedTransformation,
    SearchInitialization,
    SemanticCategory,
    TaskFeatures,
    TransferOutcome,
    TransferRecord,
    TransformationOutcome,
    TransformationRecord,
    initialize_search_from_lineage,
    transformation_is_applicable,
)

SCHEMA_VERSION: Literal["1.0.0"] = "1.0.0"
EVALUATOR_ID: Literal["genesis-lineage-synthetic-v1"] = "genesis-lineage-synthetic-v1"
OBJECTIVE_UNIT: Literal["synthetic_objective_points"] = "synthetic_objective_points"
BASELINE_OBJECTIVE = 100.0
IMPROVEMENT_THRESHOLD = 2.0
POPULATION_SIZE = 6
LINEAGE_FRACTION = 0.5
_NOW = datetime(2026, 8, 2, tzinfo=UTC)
_AS_OF = _NOW + timedelta(days=1)


class H5CampaignValidationError(ValueError):
    """Raised when an H5 campaign artifact cannot be independently reproduced."""


class LineageScenario(StrEnum):
    EMPTY = "empty_lineage"
    UNRELATED = "unrelated_lineage"
    RELATED = "related_lineage"
    STALE_BEFORE_INVALIDATION = "stale_dependency_before_invalidation"
    STALE_AFTER_INVALIDATION = "stale_dependency_after_invalidation"


SCENARIO_ORDER: tuple[LineageScenario, ...] = (
    LineageScenario.EMPTY,
    LineageScenario.UNRELATED,
    LineageScenario.RELATED,
    LineageScenario.STALE_BEFORE_INVALIDATION,
    LineageScenario.STALE_AFTER_INVALIDATION,
)


class _StrictModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        validate_default=True,
        allow_inf_nan=False,
    )


class RawCandidateRecord(_StrictModel):
    schema_version: Literal["1.0.0"] = SCHEMA_VERSION
    evaluator_id: Literal["genesis-lineage-synthetic-v1"] = EVALUATOR_ID
    scenario: LineageScenario
    seed: int
    sequence_index: int
    candidate_id: str
    proposal_kind: Literal["lineage", "unseeded"]
    proposal_id: str
    transformation_id: str | None
    source_evidence_ids: tuple[str, ...]
    retrieval_score: float | None
    preconditions_checked: bool
    preconditions_satisfied: bool | None
    reverification_required: bool
    reverification_passed: bool | None
    correct: bool
    improved: bool
    objective_value: float | None
    objective_unit: Literal["synthetic_objective_points"] = OBJECTIVE_UNIT
    evaluation_time_units: int
    cumulative_time_units: int
    outcome: Literal["valid_improved", "valid_not_improved", "invalid"]
    invalid_reason: str | None
    evidence_id: str | None
    evaluation_evidence_sha256: str | None
    transfer_id: str | None
    transfer_outcome: TransferOutcome | None
    hardware_backed: Literal[False] = False

    @model_validator(mode="after")
    def validate_record(self) -> RawCandidateRecord:
        if self.sequence_index < 1:
            raise ValueError("candidate sequence index must be positive")
        if self.evaluation_time_units < 1 or self.cumulative_time_units < 1:
            raise ValueError("candidate evaluation units must be positive")
        if self.proposal_kind == "lineage":
            if (
                self.transformation_id is None
                or not self.source_evidence_ids
                or self.retrieval_score is None
                or not self.preconditions_checked
                or not self.reverification_required
                or self.reverification_passed is None
                or self.transfer_id is None
                or self.transfer_outcome is None
            ):
                raise ValueError("lineage proposal is missing transfer or reverification data")
        elif (
            any(
                value is not None
                for value in (
                    self.transformation_id,
                    self.retrieval_score,
                    self.preconditions_satisfied,
                    self.reverification_passed,
                    self.transfer_id,
                    self.transfer_outcome,
                )
            )
            or self.source_evidence_ids
        ):
            raise ValueError("unseeded proposal contains lineage-only data")
        if self.correct != (self.objective_value is not None):
            raise ValueError("correct candidate must have exactly one objective value")
        if self.correct != (self.evidence_id is not None):
            raise ValueError("correct candidate must have objective evidence")
        if self.correct != (self.evaluation_evidence_sha256 is not None):
            raise ValueError("correct candidate must bind its evaluation evidence")
        if self.improved and not self.correct:
            raise ValueError("invalid candidate cannot be improved")
        if (self.outcome == "invalid") != (not self.correct):
            raise ValueError("candidate outcome disagrees with correctness")
        if self.correct != (self.invalid_reason is None):
            raise ValueError("only invalid candidates require an invalid reason")
        return self


class H5CaseMetrics(_StrictModel):
    evaluated_candidates: int
    valid_candidates: int
    invalid_candidates: int
    lineage_candidates: int
    negative_transfers: int
    candidate_units_to_first_correct: int
    time_units_to_first_correct: int
    candidate_units_to_first_improved: int
    time_units_to_first_improved: int
    final_objective: float


class H5CaseResult(_StrictModel):
    schema_version: Literal["1.0.0"] = SCHEMA_VERSION
    scenario: LineageScenario
    seed: int
    target_task: TaskFeatures
    initialization: SearchInitialization
    invalidated_evidence_count: int
    initial_lineage_store_path: str
    initial_lineage_store_sha256: str
    evaluated_lineage_store_path: str
    evaluated_lineage_store_sha256: str
    raw_candidates_path: str
    raw_candidates_sha256: str
    metrics: H5CaseMetrics
    hardware_backed: Literal[False] = False


class H5ScenarioAggregate(_StrictModel):
    scenario: LineageScenario
    run_count: int
    median_candidate_units_to_first_correct: float
    median_time_units_to_first_correct: float
    median_candidate_units_to_first_improved: float
    median_time_units_to_first_improved: float
    total_invalid_candidates: int
    total_negative_transfers: int
    mean_final_objective: float


class H5PairedSeedEffect(_StrictModel):
    seed: int
    related_candidate_units_saved_to_first_correct: int
    related_time_units_saved_to_first_correct: int
    related_candidate_units_saved_to_first_improved: int
    related_time_units_saved_to_first_improved: int
    related_final_objective_reduction: float
    invalidation_candidate_units_saved_to_first_improved: int
    invalidation_time_units_saved_to_first_improved: int
    invalidation_negative_transfers_avoided: int


class H5EffectSummary(_StrictModel):
    related_faster_to_first_correct_every_seed: bool
    related_faster_to_first_improved_every_seed: bool
    related_better_final_objective_every_seed: bool
    stale_seed_suppressed_after_invalidation_every_seed: bool
    median_related_candidate_units_saved_to_first_improved: float
    median_related_time_units_saved_to_first_improved: float
    mean_related_final_objective_reduction: float
    total_invalidation_negative_transfers_avoided: int


class H5LineageCampaignReport(_StrictModel):
    schema_version: Literal["1.0.0"] = SCHEMA_VERSION
    evaluator_id: Literal["genesis-lineage-synthetic-v1"] = EVALUATOR_ID
    base_seed: int
    seeds: tuple[int, ...]
    scenario_order: tuple[LineageScenario, ...]
    baseline_objective: float
    improvement_threshold: float
    population_size: int
    lineage_fraction: float
    metric_scope: str
    hardware_backed_runs: Literal[0] = 0
    cases: tuple[H5CaseResult, ...]
    aggregates: tuple[H5ScenarioAggregate, ...]
    paired_seed_effects: tuple[H5PairedSeedEffect, ...]
    effect_summary: H5EffectSummary
    conclusion: Literal["supported_in_deterministic_synthetic_scope"]
    limitations: tuple[str, ...]


def _digest(value: str | bytes) -> str:
    encoded = value.encode() if isinstance(value, str) else value
    return hashlib.sha256(encoded).hexdigest()


def _dependency(
    name: str,
    version: str,
    *,
    kind: DependencyKind = DependencyKind.COMPILER,
) -> DependencyVersion:
    return DependencyVersion(
        kind=kind,
        name=name,
        version=version,
        content_hash=_digest(f"{kind.value}:{name}:{version}"),
    )


def _target_task(scenario: LineageScenario, seed: int) -> TaskFeatures:
    task_id = f"target-{scenario.value}-{seed}"
    return TaskFeatures(
        task_id=task_id,
        model_family="sparse-moe",
        operator_families=("expert-dispatch",),
        workload_regimes=("bimodal-prompts",),
        hardware_architecture="synthetic-sm90-topology",
        topology_features=("synthetic-nvlink",),
        dependencies=(_dependency("triton", "3.1.0"),),
        model_contract_hash=_digest(f"model:{task_id}"),
        workload_contract_hash=_digest(f"workload:{task_id}"),
    )


def _discovery_records(
    scenario: LineageScenario,
    *,
    seed: int,
    target: TaskFeatures,
) -> tuple[TaskFeatures, CandidateRecord, TransformationRecord, EvidenceRecord] | None:
    if scenario is LineageScenario.EMPTY:
        return None
    unrelated = scenario is LineageScenario.UNRELATED
    stale = scenario in {
        LineageScenario.STALE_BEFORE_INVALIDATION,
        LineageScenario.STALE_AFTER_INVALIDATION,
    }
    source_id = f"source-{scenario.value}-{seed}"
    source_dependency = (
        _dependency("torch", "2.8.0", kind=DependencyKind.RUNTIME)
        if unrelated
        else _dependency("triton", "2.2.0" if stale else "3.1.0")
    )
    source = TaskFeatures(
        task_id=source_id,
        model_family="dense-decoder" if unrelated else target.model_family,
        operator_families=("dense-attention",) if unrelated else target.operator_families,
        workload_regimes=("uniform-prompts",) if unrelated else target.workload_regimes,
        hardware_architecture=(
            "synthetic-cpu-topology" if unrelated else target.hardware_architecture
        ),
        topology_features=("synthetic-pcie",) if unrelated else target.topology_features,
        dependencies=(source_dependency,),
        model_contract_hash=_digest(f"model:{source_id}"),
        workload_contract_hash=_digest(f"workload:{source_id}"),
    )
    source_candidate_id = f"source-candidate-{scenario.value}-{seed}"
    transformation_id = f"transferable-{scenario.value}-{seed}"
    candidate = CandidateRecord(
        candidate_id=source_candidate_id,
        task_id=source.task_id,
        genome_hash=_digest(f"genome:{source_candidate_id}"),
        disposition=CandidateDisposition.ACCEPTED,
        created_at=_NOW,
    )
    transformation = TransformationRecord(
        transformation_id=transformation_id,
        family="expert-dispatch-scheduler",
        semantic_category=SemanticCategory.POLICY,
        source_candidate_id=source_candidate_id,
        affected_regions=("serving.scheduler",),
        preconditions=(
            "bounded queue",
            "independent request state",
            "target dependency satisfies declared range",
        ),
        # Applicability is deliberately declared for the target.  The
        # independent reverifier must still reject unrelated or stale
        # source evidence rather than trusting this declaration.
        applicable_model_families=(target.model_family,),
        applicable_operations=target.operator_families,
        applicable_hardware=(target.hardware_architecture,),
        applicable_workloads=target.workload_regimes,
        dependency_preconditions=(
            DependencySelector(
                kind=DependencyKind.COMPILER,
                name="triton",
                version_range=">=2.0,<4.0",
            ),
        ),
        expected_benefit=0.20,
        outcome=TransformationOutcome.ACCEPTED,
        proposal_source="deterministic-lineage-campaign-fixture",
        created_at=_NOW,
    )
    evidence = EvidenceRecord(
        evidence_id=f"source-evidence-{scenario.value}-{seed}",
        target_kind=EvidenceTargetKind.TRANSFORMATION,
        target_id=transformation_id,
        evidence_type="deterministic-synthetic-source-evaluation",
        result=EvidenceResult.PASS,
        content_hash=_digest(f"source-evidence:{scenario.value}:{seed}"),
        model_family=source.model_family,
        workload_regimes=source.workload_regimes,
        hardware_architecture=source.hardware_architecture,
        dependencies=source.dependencies,
        base_confidence=0.95,
        observed_at=_NOW,
        valid_until=_NOW + timedelta(days=365),
    )
    return source, candidate, transformation, evidence


def _record_discovery(
    store: LineageStore,
    scenario: LineageScenario,
    *,
    seed: int,
    target: TaskFeatures,
) -> None:
    records = _discovery_records(scenario, seed=seed, target=target)
    if records is None:
        return
    source, candidate, transformation, evidence = records
    store.record_task(source)
    store.record_candidate(candidate)
    store.record_transformation(transformation)
    store.record_evidence(evidence)


def _invalidation(seed: int) -> InvalidationEvent:
    return InvalidationEvent(
        invalidation_id=f"invalidate-triton-v2-{seed}",
        selector=DependencySelector(
            kind=DependencyKind.COMPILER,
            name="triton",
            version_range="2.x",
        ),
        reason="Triton 3.x target requires revalidation of Triton 2.x evidence",
        occurred_at=_AS_OF,
    )


def _populate_initial_store(
    path: Path, scenario: LineageScenario, *, seed: int
) -> tuple[TaskFeatures, SearchInitialization, int]:
    target = _target_task(scenario, seed)
    with LineageStore(path) as store:
        store.record_task(target)
        _record_discovery(store, scenario, seed=seed, target=target)
        invalidated = 0
        if scenario is LineageScenario.STALE_AFTER_INVALIDATION:
            invalidated = store.invalidate_dependency(_invalidation(seed))
        initialization = initialize_search_from_lineage(
            store,
            target,
            seed=seed,
            as_of=_AS_OF,
            population_size=POPULATION_SIZE,
            lineage_fraction=LINEAGE_FRACTION,
        )
    return target, initialization, invalidated


def _evidence_matches_target(evidence: EvidenceRecord, target: TaskFeatures) -> bool:
    target_dependencies = {(item.kind, item.name): item.version for item in target.dependencies}
    return (
        evidence.freshness is EvidenceFreshness.FRESH
        and evidence.result is EvidenceResult.PASS
        and evidence.model_family == target.model_family
        and evidence.hardware_architecture == target.hardware_architecture
        and set(target.workload_regimes).issubset(evidence.workload_regimes)
        and all(
            target_dependencies.get((item.kind, item.name)) == item.version
            for item in evidence.dependencies
        )
    )


def _bounded_jitter(*parts: object, modulus: int) -> int:
    payload = ":".join(str(part) for part in parts).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % modulus


def _evaluation_evidence_digest(
    *,
    scenario: LineageScenario,
    seed: int,
    candidate_id: str,
    proposal_id: str,
    correct: bool,
    objective_value: float | None,
    evaluation_time_units: int,
    reverification_passed: bool | None,
) -> str:
    return _digest(
        canonical_json(
            {
                "evaluator_id": EVALUATOR_ID,
                "scenario": scenario.value,
                "seed": seed,
                "candidate_id": candidate_id,
                "proposal_id": proposal_id,
                "correct": correct,
                "objective_value": objective_value,
                "objective_unit": OBJECTIVE_UNIT,
                "evaluation_time_units": evaluation_time_units,
                "reverification_passed": reverification_passed,
                "hardware_backed": False,
            }
        )
    )


def _expected_raw_candidates(
    store: LineageStore,
    scenario: LineageScenario,
    *,
    seed: int,
    target: TaskFeatures,
    initialization: SearchInitialization,
) -> tuple[RawCandidateRecord, ...]:
    transformations = {item.transformation_id: item for item in store.list_transformations()}
    proposals: list[tuple[str, RelatedTransformation | None]] = [
        (item.transformation_id, item) for item in initialization.lineage_seeds
    ]
    proposals.extend((item.proposal_id, None) for item in initialization.unseeded_proposals)
    records: list[RawCandidateRecord] = []
    cumulative_time_units = 0
    lineage_rank = 0
    for index, (proposal_id, lineage) in enumerate(proposals, start=1):
        candidate_id = f"eval-{scenario.value}-{seed}-{index:02d}"
        if lineage is not None:
            lineage_rank += 1
            transformation = transformations[lineage.transformation_id]
            evidence = store.evidence_for_transformation(transformation.transformation_id)
            preconditions_satisfied = transformation_is_applicable(transformation, target)
            reverification_passed = (
                preconditions_satisfied
                and lineage.requires_reverification
                and bool(evidence)
                and tuple(item.evidence_id for item in evidence) == lineage.evidence_ids
                and all(_evidence_matches_target(item, target) for item in evidence)
            )
            correct = reverification_passed
            objective_value = (
                78.0 + _bounded_jitter(seed, "related-seed", modulus=200) / 100.0
                if correct
                else None
            )
            evaluation_time_units = 2 + _bounded_jitter(seed, "lineage-reverify", modulus=2)
            invalid_reason = (
                None
                if correct
                else "independent reverification rejected unrelated or stale source evidence"
            )
            transfer_outcome = (
                TransferOutcome.IMPROVED if correct else TransferOutcome.NEGATIVE_TRANSFER
            )
            transfer_id = f"transfer-{scenario.value}-{seed}-{lineage_rank:02d}"
            source_evidence_ids = lineage.evidence_ids
            retrieval_score = lineage.score
        else:
            # Initialization assigns deterministic proposal seeds in slot order;
            # recover the slot from the tuple rather than relying on opaque IDs.
            slot = index - len(initialization.lineage_seeds)
            correct = slot not in {1, 5}
            if not correct:
                objective_value = None
            elif slot == 2:
                objective_value = BASELINE_OBJECTIVE
            elif slot == 3:
                objective_value = 91.0 + _bounded_jitter(seed, slot, modulus=300) / 100.0
            else:
                objective_value = 96.0 + _bounded_jitter(seed, slot, modulus=150) / 100.0
            evaluation_time_units = 2
            preconditions_satisfied = None
            reverification_passed = None
            invalid_reason = None if correct else "deterministic local proposal failed contracts"
            transfer_outcome = None
            transfer_id = None
            source_evidence_ids = ()
            retrieval_score = None
        cumulative_time_units += evaluation_time_units
        improved = bool(
            objective_value is not None
            and objective_value <= BASELINE_OBJECTIVE - IMPROVEMENT_THRESHOLD
        )
        evidence_id = f"evaluation-evidence-{candidate_id}" if correct else None
        evidence_digest = (
            _evaluation_evidence_digest(
                scenario=scenario,
                seed=seed,
                candidate_id=candidate_id,
                proposal_id=proposal_id,
                correct=correct,
                objective_value=objective_value,
                evaluation_time_units=evaluation_time_units,
                reverification_passed=reverification_passed,
            )
            if correct
            else None
        )
        records.append(
            RawCandidateRecord(
                scenario=scenario,
                seed=seed,
                sequence_index=index,
                candidate_id=candidate_id,
                proposal_kind="lineage" if lineage is not None else "unseeded",
                proposal_id=proposal_id,
                transformation_id=(lineage.transformation_id if lineage is not None else None),
                source_evidence_ids=source_evidence_ids,
                retrieval_score=retrieval_score,
                preconditions_checked=lineage is not None,
                preconditions_satisfied=preconditions_satisfied,
                reverification_required=lineage is not None,
                reverification_passed=reverification_passed,
                correct=correct,
                improved=improved,
                objective_value=objective_value,
                evaluation_time_units=evaluation_time_units,
                cumulative_time_units=cumulative_time_units,
                outcome=(
                    "invalid"
                    if not correct
                    else "valid_improved"
                    if improved
                    else "valid_not_improved"
                ),
                invalid_reason=invalid_reason,
                evidence_id=evidence_id,
                evaluation_evidence_sha256=evidence_digest,
                transfer_id=transfer_id,
                transfer_outcome=transfer_outcome,
            )
        )
    return tuple(records)


def _evaluated_candidate_record(
    record: RawCandidateRecord, target: TaskFeatures
) -> CandidateRecord:
    objective = (
        ObjectiveMeasurement(
            name="synthetic_objective",
            value=record.objective_value,
            unit=OBJECTIVE_UNIT,
            direction=MetricDirection.MINIMIZE,
            evidence_id=record.evidence_id,
        )
        if record.objective_value is not None and record.evidence_id is not None
        else None
    )
    return CandidateRecord(
        candidate_id=record.candidate_id,
        task_id=target.task_id,
        genome_hash=_digest(f"genome:{record.candidate_id}"),
        disposition=(
            CandidateDisposition.REJECTED
            if not record.correct
            else CandidateDisposition.ACCEPTED
            if record.improved
            else CandidateDisposition.SUPERSEDED
        ),
        transformation_ids=(
            (record.transformation_id,) if record.transformation_id is not None else ()
        ),
        objectives=(objective,) if objective is not None else (),
        created_at=_AS_OF,
    )


def _evaluated_evidence_record(
    record: RawCandidateRecord, target: TaskFeatures
) -> EvidenceRecord | None:
    if record.evidence_id is None or record.evaluation_evidence_sha256 is None:
        return None
    return EvidenceRecord(
        evidence_id=record.evidence_id,
        target_kind=EvidenceTargetKind.CANDIDATE,
        target_id=record.candidate_id,
        evidence_type="deterministic-synthetic-candidate-evaluation",
        result=EvidenceResult.PASS,
        content_hash=record.evaluation_evidence_sha256,
        model_family=target.model_family,
        workload_regimes=target.workload_regimes,
        hardware_architecture=target.hardware_architecture,
        dependencies=target.dependencies,
        base_confidence=1.0,
        observed_at=_AS_OF,
        valid_until=_AS_OF + timedelta(days=365),
    )


def _evaluated_transfer_record(
    record: RawCandidateRecord, target: TaskFeatures, *, rank: int
) -> TransferRecord | None:
    if record.transfer_id is None:
        return None
    assert record.transformation_id is not None
    assert record.retrieval_score is not None
    assert record.transfer_outcome is not None
    return TransferRecord(
        transfer_id=record.transfer_id,
        target_task_id=target.task_id,
        transformation_id=record.transformation_id,
        source_evidence_ids=record.source_evidence_ids,
        retrieval_score=record.retrieval_score,
        rank=rank,
        seed=record.seed,
        outcome=record.transfer_outcome,
        rationale=(
            "independent target reverification passed and improved the objective"
            if record.transfer_outcome is TransferOutcome.IMPROVED
            else "independent target reverification rejected the lineage seed"
        ),
        created_at=_AS_OF,
    )


def _record_evaluated_candidates(
    store: LineageStore,
    records: tuple[RawCandidateRecord, ...],
    *,
    target: TaskFeatures,
) -> None:
    lineage_rank = 0
    for record in records:
        candidate = _evaluated_candidate_record(record, target)
        evidence = _evaluated_evidence_record(record, target)
        if record.proposal_kind == "lineage":
            lineage_rank += 1
        transfer = _evaluated_transfer_record(record, target, rank=lineage_rank)
        if evidence is None:
            store.record_candidate(candidate)
        else:
            store.record_candidate_bundle(candidate, (), (evidence,))
        if transfer is not None:
            store.record_transfer(transfer)


def _raw_candidate_bytes(records: tuple[RawCandidateRecord, ...]) -> bytes:
    return b"".join(canonical_json(item) + b"\n" for item in records)


def _derive_case_metrics(records: tuple[RawCandidateRecord, ...]) -> H5CaseMetrics:
    if not records:
        raise H5CampaignValidationError("case contains no candidate records")
    first_correct = next((item for item in records if item.correct), None)
    first_improved = next((item for item in records if item.improved), None)
    objectives = tuple(item.objective_value for item in records if item.objective_value is not None)
    if first_correct is None or first_improved is None or not objectives:
        raise H5CampaignValidationError("case did not produce correct and improved candidates")
    return H5CaseMetrics(
        evaluated_candidates=len(records),
        valid_candidates=sum(item.correct for item in records),
        invalid_candidates=sum(not item.correct for item in records),
        lineage_candidates=sum(item.proposal_kind == "lineage" for item in records),
        negative_transfers=sum(
            item.transfer_outcome is TransferOutcome.NEGATIVE_TRANSFER for item in records
        ),
        candidate_units_to_first_correct=first_correct.sequence_index,
        time_units_to_first_correct=first_correct.cumulative_time_units,
        candidate_units_to_first_improved=first_improved.sequence_index,
        time_units_to_first_improved=first_improved.cumulative_time_units,
        final_objective=min(objectives),
    )


def _scenario_aggregates(
    cases: tuple[H5CaseResult, ...], seeds: tuple[int, ...]
) -> tuple[H5ScenarioAggregate, ...]:
    aggregates: list[H5ScenarioAggregate] = []
    for scenario in SCENARIO_ORDER:
        selected = tuple(item for item in cases if item.scenario is scenario)
        if len(selected) != len(seeds):
            raise H5CampaignValidationError(f"{scenario.value} does not cover every seed")
        aggregates.append(
            H5ScenarioAggregate(
                scenario=scenario,
                run_count=len(selected),
                median_candidate_units_to_first_correct=statistics.median(
                    item.metrics.candidate_units_to_first_correct for item in selected
                ),
                median_time_units_to_first_correct=statistics.median(
                    item.metrics.time_units_to_first_correct for item in selected
                ),
                median_candidate_units_to_first_improved=statistics.median(
                    item.metrics.candidate_units_to_first_improved for item in selected
                ),
                median_time_units_to_first_improved=statistics.median(
                    item.metrics.time_units_to_first_improved for item in selected
                ),
                total_invalid_candidates=sum(item.metrics.invalid_candidates for item in selected),
                total_negative_transfers=sum(item.metrics.negative_transfers for item in selected),
                mean_final_objective=statistics.fmean(
                    item.metrics.final_objective for item in selected
                ),
            )
        )
    return tuple(aggregates)


def _paired_effects(
    cases: tuple[H5CaseResult, ...], seeds: tuple[int, ...]
) -> tuple[H5PairedSeedEffect, ...]:
    by_key = {(item.seed, item.scenario): item for item in cases}
    effects: list[H5PairedSeedEffect] = []
    for seed in seeds:
        empty = by_key[(seed, LineageScenario.EMPTY)].metrics
        related = by_key[(seed, LineageScenario.RELATED)].metrics
        stale_before = by_key[(seed, LineageScenario.STALE_BEFORE_INVALIDATION)].metrics
        stale_after = by_key[(seed, LineageScenario.STALE_AFTER_INVALIDATION)].metrics
        effects.append(
            H5PairedSeedEffect(
                seed=seed,
                related_candidate_units_saved_to_first_correct=(
                    empty.candidate_units_to_first_correct
                    - related.candidate_units_to_first_correct
                ),
                related_time_units_saved_to_first_correct=(
                    empty.time_units_to_first_correct - related.time_units_to_first_correct
                ),
                related_candidate_units_saved_to_first_improved=(
                    empty.candidate_units_to_first_improved
                    - related.candidate_units_to_first_improved
                ),
                related_time_units_saved_to_first_improved=(
                    empty.time_units_to_first_improved - related.time_units_to_first_improved
                ),
                related_final_objective_reduction=(empty.final_objective - related.final_objective),
                invalidation_candidate_units_saved_to_first_improved=(
                    stale_before.candidate_units_to_first_improved
                    - stale_after.candidate_units_to_first_improved
                ),
                invalidation_time_units_saved_to_first_improved=(
                    stale_before.time_units_to_first_improved
                    - stale_after.time_units_to_first_improved
                ),
                invalidation_negative_transfers_avoided=(
                    stale_before.negative_transfers - stale_after.negative_transfers
                ),
            )
        )
    return tuple(effects)


def _effect_summary(
    cases: tuple[H5CaseResult, ...], effects: tuple[H5PairedSeedEffect, ...]
) -> H5EffectSummary:
    stale_after = tuple(
        item for item in cases if item.scenario is LineageScenario.STALE_AFTER_INVALIDATION
    )
    return H5EffectSummary(
        related_faster_to_first_correct_every_seed=all(
            item.related_candidate_units_saved_to_first_correct > 0
            and item.related_time_units_saved_to_first_correct > 0
            for item in effects
        ),
        related_faster_to_first_improved_every_seed=all(
            item.related_candidate_units_saved_to_first_improved > 0
            and item.related_time_units_saved_to_first_improved > 0
            for item in effects
        ),
        related_better_final_objective_every_seed=all(
            item.related_final_objective_reduction > 0.0 for item in effects
        ),
        stale_seed_suppressed_after_invalidation_every_seed=all(
            not item.initialization.lineage_seeds and item.invalidated_evidence_count == 1
            for item in stale_after
        ),
        median_related_candidate_units_saved_to_first_improved=statistics.median(
            item.related_candidate_units_saved_to_first_improved for item in effects
        ),
        median_related_time_units_saved_to_first_improved=statistics.median(
            item.related_time_units_saved_to_first_improved for item in effects
        ),
        mean_related_final_objective_reduction=statistics.fmean(
            item.related_final_objective_reduction for item in effects
        ),
        total_invalidation_negative_transfers_avoided=sum(
            item.invalidation_negative_transfers_avoided for item in effects
        ),
    )


def _sha256_file(path: Path) -> str:
    return _digest(path.read_bytes())


def _relative(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _run_case(
    output: Path,
    root: Path,
    scenario: LineageScenario,
    *,
    seed: int,
) -> H5CaseResult:
    output.mkdir(parents=True, exist_ok=False)
    initial_store_path = output / "lineage-initial.sqlite3"
    target, initialization, invalidated = _populate_initial_store(
        initial_store_path, scenario, seed=seed
    )
    evaluated_store_path = output / "lineage-evaluated.sqlite3"
    shutil.copyfile(initial_store_path, evaluated_store_path)
    with LineageStore(evaluated_store_path) as store:
        records = _expected_raw_candidates(
            store,
            scenario,
            seed=seed,
            target=target,
            initialization=initialization,
        )
        _record_evaluated_candidates(store, records, target=target)
    raw_path = output / "raw-candidates.jsonl"
    raw_path.write_bytes(_raw_candidate_bytes(records))
    return H5CaseResult(
        scenario=scenario,
        seed=seed,
        target_task=target,
        initialization=initialization,
        invalidated_evidence_count=invalidated,
        initial_lineage_store_path=_relative(initial_store_path, root),
        initial_lineage_store_sha256=_sha256_file(initial_store_path),
        evaluated_lineage_store_path=_relative(evaluated_store_path, root),
        evaluated_lineage_store_sha256=_sha256_file(evaluated_store_path),
        raw_candidates_path=_relative(raw_path, root),
        raw_candidates_sha256=_sha256_file(raw_path),
        metrics=_derive_case_metrics(records),
    )


def _resolve_artifact(root: Path, relative_path: str) -> Path:
    path = Path(relative_path)
    if path.is_absolute() or ".." in path.parts:
        raise H5CampaignValidationError(f"campaign artifact path is unsafe: {relative_path}")
    unresolved = root / path
    if unresolved.is_symlink():
        raise H5CampaignValidationError(f"campaign artifact must not be a symlink: {relative_path}")
    resolved = unresolved.resolve(strict=True)
    if not resolved.is_relative_to(root.resolve(strict=True)):
        raise H5CampaignValidationError(
            f"campaign artifact escapes its output directory: {relative_path}"
        )
    if not resolved.is_file():
        raise H5CampaignValidationError(f"campaign artifact is not a regular file: {relative_path}")
    return resolved


def _load_raw_candidates(path: Path) -> tuple[RawCandidateRecord, ...]:
    try:
        records = tuple(
            RawCandidateRecord.model_validate_json(line, strict=True)
            for line in path.read_bytes().splitlines()
            if line
        )
    except (ValueError, json.JSONDecodeError) as exc:
        raise H5CampaignValidationError(f"invalid raw candidate artifact: {path}") from exc
    if path.read_bytes() != _raw_candidate_bytes(records):
        raise H5CampaignValidationError("raw candidate artifact is not canonical JSONL")
    return records


def _validate_scenario_store(
    store: LineageStore,
    case: H5CaseResult,
) -> None:
    expected_target = _target_task(case.scenario, case.seed)
    if case.target_task != expected_target:
        raise H5CampaignValidationError("target task differs from campaign fixture")
    discovery = _discovery_records(case.scenario, seed=case.seed, target=expected_target)
    expected_tasks = [expected_target]
    expected_candidates: tuple[CandidateRecord, ...] = ()
    expected_transformations: tuple[TransformationRecord, ...] = ()
    expected_evidence: tuple[EvidenceRecord, ...] = ()
    if discovery is not None:
        source, candidate, transformation, evidence = discovery
        expected_tasks.append(source)
        expected_candidates = (candidate,)
        expected_transformations = (transformation,)
        expected_evidence = (evidence,)
    expected_invalidations: tuple[InvalidationEvent, ...] = ()
    expected_invalidated_count = 0
    if case.scenario is LineageScenario.STALE_AFTER_INVALIDATION:
        event = _invalidation(case.seed)
        expected_invalidations = (event,)
        expected_invalidated_count = 1
        expected_evidence = tuple(
            item.model_copy(
                update={
                    "freshness": EvidenceFreshness.STALE,
                    "invalidation_event_ids": (event.invalidation_id,),
                }
            )
            for item in expected_evidence
        )
    expected_task_tuple = tuple(sorted(expected_tasks, key=lambda item: item.task_id))
    if (
        store.list_tasks() != expected_task_tuple
        or store.list_candidates() != expected_candidates
        or store.list_transformations() != expected_transformations
        or store.list_evidence() != expected_evidence
        or store.list_invalidations() != expected_invalidations
        or store.list_transfers()
        or case.invalidated_evidence_count != expected_invalidated_count
    ):
        raise H5CampaignValidationError(
            "frozen lineage store does not match the deterministic scenario fixture"
        )


def _validate_evaluated_store(
    store: LineageStore,
    case: H5CaseResult,
    records: tuple[RawCandidateRecord, ...],
) -> None:
    expected_target = _target_task(case.scenario, case.seed)
    discovery = _discovery_records(case.scenario, seed=case.seed, target=expected_target)
    expected_tasks = [expected_target]
    expected_candidates: list[CandidateRecord] = []
    expected_transformations: list[TransformationRecord] = []
    expected_evidence: list[EvidenceRecord] = []
    if discovery is not None:
        source, candidate, transformation, evidence = discovery
        expected_tasks.append(source)
        expected_candidates.append(candidate)
        expected_transformations.append(transformation)
        if case.scenario is LineageScenario.STALE_AFTER_INVALIDATION:
            evidence = evidence.model_copy(
                update={
                    "freshness": EvidenceFreshness.STALE,
                    "invalidation_event_ids": (_invalidation(case.seed).invalidation_id,),
                }
            )
        expected_evidence.append(evidence)
    expected_transfers: list[TransferRecord] = []
    lineage_rank = 0
    for record in records:
        expected_candidates.append(_evaluated_candidate_record(record, expected_target))
        candidate_evidence = _evaluated_evidence_record(record, expected_target)
        if candidate_evidence is not None:
            expected_evidence.append(candidate_evidence)
        if record.proposal_kind == "lineage":
            lineage_rank += 1
        transfer = _evaluated_transfer_record(record, expected_target, rank=lineage_rank)
        if transfer is not None:
            expected_transfers.append(transfer)
    expected_invalidations = (
        (_invalidation(case.seed),)
        if case.scenario is LineageScenario.STALE_AFTER_INVALIDATION
        else ()
    )
    if (
        store.list_tasks() != tuple(sorted(expected_tasks, key=lambda item: item.task_id))
        or store.list_candidates()
        != tuple(sorted(expected_candidates, key=lambda item: item.candidate_id))
        or store.list_transformations()
        != tuple(sorted(expected_transformations, key=lambda item: item.transformation_id))
        or store.list_evidence()
        != tuple(sorted(expected_evidence, key=lambda item: item.evidence_id))
        or store.list_transfers()
        != tuple(sorted(expected_transfers, key=lambda item: item.transfer_id))
        or store.list_invalidations() != expected_invalidations
        or store.list_counterexamples()
        or store.list_constraints()
    ):
        raise H5CampaignValidationError(
            "evaluated lineage store does not match replayed candidate and transfer records"
        )


def validate_h5_lineage_campaign(
    root: Path,
    report: H5LineageCampaignReport | None = None,
) -> H5LineageCampaignReport:
    """Independently replay retrieval/evaluation and derive every reported metric."""

    resolved_root = root.resolve(strict=True)
    report_path = _resolve_artifact(resolved_root, "report.json")
    try:
        persisted = H5LineageCampaignReport.model_validate_json(
            report_path.read_bytes(), strict=True
        )
    except ValueError as exc:
        raise H5CampaignValidationError("campaign report does not match its strict schema") from exc
    if report is not None and report != persisted:
        raise H5CampaignValidationError("provided report differs from persisted campaign report")
    report = persisted
    if report.scenario_order != SCENARIO_ORDER:
        raise H5CampaignValidationError("campaign scenario order is incomplete")
    if (
        len(report.seeds) < 2
        or report.seeds != tuple(report.base_seed + index for index in range(len(report.seeds)))
        or len(set(report.seeds)) != len(report.seeds)
    ):
        raise H5CampaignValidationError("campaign seeds are not a consecutive multi-seed range")
    if (
        report.baseline_objective != BASELINE_OBJECTIVE
        or report.improvement_threshold != IMPROVEMENT_THRESHOLD
        or report.population_size != POPULATION_SIZE
        or report.lineage_fraction != LINEAGE_FRACTION
        or report.hardware_backed_runs != 0
    ):
        raise H5CampaignValidationError("campaign evaluator constants or hardware scope changed")
    expected_keys = {(seed, scenario) for seed in report.seeds for scenario in SCENARIO_ORDER}
    if {(item.seed, item.scenario) for item in report.cases} != expected_keys:
        raise H5CampaignValidationError("campaign cases do not cover the scenario/seed product")
    for case in report.cases:
        initial_path = _resolve_artifact(resolved_root, case.initial_lineage_store_path)
        evaluated_path = _resolve_artifact(resolved_root, case.evaluated_lineage_store_path)
        raw_path = _resolve_artifact(resolved_root, case.raw_candidates_path)
        if _sha256_file(initial_path) != case.initial_lineage_store_sha256:
            raise H5CampaignValidationError("initial lineage store digest changed")
        if _sha256_file(evaluated_path) != case.evaluated_lineage_store_sha256:
            raise H5CampaignValidationError("evaluated lineage store digest changed")
        if _sha256_file(raw_path) != case.raw_candidates_sha256:
            raise H5CampaignValidationError("raw candidate digest changed")
        records = _load_raw_candidates(raw_path)
        with LineageStore(initial_path) as initial_store:
            if initial_store.get_task(case.target_task.task_id) != case.target_task:
                raise H5CampaignValidationError("target task differs from frozen lineage input")
            _validate_scenario_store(initial_store, case)
            initialization = initialize_search_from_lineage(
                initial_store,
                case.target_task,
                seed=case.seed,
                as_of=_AS_OF,
                population_size=POPULATION_SIZE,
                lineage_fraction=LINEAGE_FRACTION,
            )
            if initialization != case.initialization:
                raise H5CampaignValidationError("lineage initialization is not reproducible")
            expected_records = _expected_raw_candidates(
                initial_store,
                case.scenario,
                seed=case.seed,
                target=case.target_task,
                initialization=initialization,
            )
        if records != expected_records:
            raise H5CampaignValidationError("raw candidates do not replay from frozen inputs")
        if _derive_case_metrics(records) != case.metrics:
            raise H5CampaignValidationError("case metrics are not derived from raw candidates")
        with LineageStore(evaluated_path) as evaluated_store:
            _validate_evaluated_store(evaluated_store, case, records)
    expected_aggregates = _scenario_aggregates(report.cases, report.seeds)
    expected_effects = _paired_effects(report.cases, report.seeds)
    expected_summary = _effect_summary(report.cases, expected_effects)
    if report.aggregates != expected_aggregates:
        raise H5CampaignValidationError("scenario aggregates are not derived from cases")
    if report.paired_seed_effects != expected_effects:
        raise H5CampaignValidationError("paired effects are not derived from cases")
    if report.effect_summary != expected_summary:
        raise H5CampaignValidationError("effect summary is not derived from paired effects")
    if not all(
        (
            expected_summary.related_faster_to_first_correct_every_seed,
            expected_summary.related_faster_to_first_improved_every_seed,
            expected_summary.related_better_final_objective_every_seed,
            expected_summary.stale_seed_suppressed_after_invalidation_every_seed,
        )
    ):
        raise H5CampaignValidationError("scoped H5 conclusion is not supported by the campaign")
    return report


def run_h5_lineage_campaign(
    output: Path,
    *,
    seed: int,
    count: int,
) -> H5LineageCampaignReport:
    """Run the deterministic synthetic H5 campaign and persist all raw evidence."""

    if seed < 0:
        raise ValueError("campaign seed must be non-negative")
    if not 2 <= count <= 100:
        raise ValueError("campaign requires between 2 and 100 seeds")
    if output.exists():
        raise FileExistsError(f"campaign output already exists: {output}")
    output.mkdir(parents=True)
    seeds = tuple(seed + index for index in range(count))
    cases = tuple(
        _run_case(
            output / "runs" / f"seed-{run_seed}" / scenario.value,
            output,
            scenario,
            seed=run_seed,
        )
        for run_seed in seeds
        for scenario in SCENARIO_ORDER
    )
    aggregates = _scenario_aggregates(cases, seeds)
    effects = _paired_effects(cases, seeds)
    summary = _effect_summary(cases, effects)
    report = H5LineageCampaignReport(
        base_seed=seed,
        seeds=seeds,
        scenario_order=SCENARIO_ORDER,
        baseline_objective=BASELINE_OBJECTIVE,
        improvement_threshold=IMPROVEMENT_THRESHOLD,
        population_size=POPULATION_SIZE,
        lineage_fraction=LINEAGE_FRACTION,
        metric_scope=(
            "deterministic synthetic evaluator; time units are declared candidate-cost units, "
            "not elapsed seconds; objective points are not hardware performance"
        ),
        cases=cases,
        aggregates=aggregates,
        paired_seed_effects=effects,
        effect_summary=summary,
        conclusion="supported_in_deterministic_synthetic_scope",
        limitations=(
            "the evaluator is deterministic and synthetic, not a wall-clock or hardware benchmark",
            "the fixture tests lineage retrieval, preconditions, mandatory reverification, "
            "negative transfer, and dependency invalidation rather than model quality",
            "results cannot establish production speedup or generalize beyond the declared fixture",
        ),
    )
    report_path = output / "report.json"
    report_path.write_bytes(canonical_json(report) + b"\n")
    return validate_h5_lineage_campaign(output, report)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=73129)
    parser.add_argument("--count", type=int, default=5)
    arguments = parser.parse_args(argv)
    report = run_h5_lineage_campaign(
        arguments.output,
        seed=arguments.seed,
        count=arguments.count,
    )
    print(json.dumps(report.model_dump(mode="json"), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "H5CampaignValidationError",
    "H5CaseMetrics",
    "H5CaseResult",
    "H5EffectSummary",
    "H5LineageCampaignReport",
    "H5PairedSeedEffect",
    "H5ScenarioAggregate",
    "LineageScenario",
    "RawCandidateRecord",
    "main",
    "run_h5_lineage_campaign",
    "validate_h5_lineage_campaign",
]
