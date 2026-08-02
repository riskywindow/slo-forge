"""Typed persistent records for Genesis optimization lineage."""

from __future__ import annotations

import re
from enum import StrEnum
from typing import Annotated, Literal, Self

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

LINEAGE_SCHEMA_VERSION: Literal["1.0.0"] = "1.0.0"
_VERSION_RANGE = re.compile(
    r"^(?:\*|(?:<=|>=|<|>)?\s*[0-9]+(?:\.(?:[0-9]+|[xX*])){0,2})"
    r"(?:\s*,\s*(?:<=|>=|<|>)?\s*[0-9]+(?:\.(?:[0-9]+|[xX*])){0,2})*$"
)

Identifier = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")]
NonEmpty = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
Digest = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
Probability = Annotated[float, Field(ge=0.0, le=1.0)]
PositiveInt = Annotated[int, Field(gt=0)]


class LineageModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        validate_default=True,
        allow_inf_nan=False,
    )


class DependencyKind(StrEnum):
    DRIVER = "driver"
    COMPILER = "compiler"
    RUNTIME = "runtime"
    LIBRARY = "library"
    HARDWARE = "hardware"
    MODEL_CONTRACT = "model_contract"
    WORKLOAD_CONTRACT = "workload_contract"


class DependencyVersion(LineageModel):
    kind: DependencyKind
    name: NonEmpty
    version: NonEmpty
    content_hash: Digest | None = None


class DependencySelector(LineageModel):
    kind: DependencyKind
    name: NonEmpty
    version_range: NonEmpty = "*"

    @field_validator("version_range")
    @classmethod
    def validate_range(cls, value: str) -> str:
        if _VERSION_RANGE.fullmatch(value) is None:
            raise ValueError("unsupported dependency version range")
        return value


class TaskFeatures(LineageModel):
    task_id: Identifier
    model_family: NonEmpty
    operator_families: tuple[NonEmpty, ...]
    workload_regimes: tuple[NonEmpty, ...]
    hardware_architecture: NonEmpty
    topology_features: tuple[NonEmpty, ...] = ()
    dependencies: tuple[DependencyVersion, ...]
    model_contract_hash: Digest
    workload_contract_hash: Digest

    @model_validator(mode="after")
    def validate_features(self) -> Self:
        for label, values in (
            ("operator_families", self.operator_families),
            ("workload_regimes", self.workload_regimes),
        ):
            if not values:
                raise ValueError(f"{label} must not be empty")
            if len(values) != len(set(values)):
                raise ValueError(f"{label} must be unique")
        dependency_keys = [(item.kind, item.name) for item in self.dependencies]
        if len(dependency_keys) != len(set(dependency_keys)):
            raise ValueError("task dependency identities must be unique")
        return self


class CandidateDisposition(StrEnum):
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    SUPERSEDED = "superseded"


class MetricDirection(StrEnum):
    MINIMIZE = "minimize"
    MAXIMIZE = "maximize"


class ObjectiveMeasurement(LineageModel):
    name: NonEmpty
    value: float
    unit: NonEmpty
    direction: MetricDirection
    evidence_id: Identifier


class CandidateRecord(LineageModel):
    candidate_id: Identifier
    task_id: Identifier
    genome_hash: Digest
    parent_candidate_ids: tuple[Identifier, ...] = ()
    disposition: CandidateDisposition
    transformation_ids: tuple[Identifier, ...] = ()
    objectives: tuple[ObjectiveMeasurement, ...] = ()
    causal_bottleneck: NonEmpty | None = None
    exposed_next_bottleneck: NonEmpty | None = None
    created_at: AwareDatetime

    @model_validator(mode="after")
    def unique_references(self) -> Self:
        if len(self.parent_candidate_ids) != len(set(self.parent_candidate_ids)):
            raise ValueError("candidate parent references must be unique")
        if len(self.transformation_ids) != len(set(self.transformation_ids)):
            raise ValueError("candidate transformation references must be unique")
        objective_names = [item.name for item in self.objectives]
        if len(objective_names) != len(set(objective_names)):
            raise ValueError("candidate objective names must be unique")
        return self


class TransformationOutcome(StrEnum):
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    EXPERIMENTAL = "experimental"


class SemanticCategory(StrEnum):
    EXACT = "semantics_preserving"
    APPROXIMATE = "approximate_within_budget"
    POLICY = "policy"
    RESOURCE = "resource_only"
    RUNTIME = "runtime_implementation"
    OPERATOR_REVIEW = "experimental_operator_review"


class TransformationRecord(LineageModel):
    transformation_id: Identifier
    family: NonEmpty
    semantic_category: SemanticCategory
    source_candidate_id: Identifier
    target_candidate_id: Identifier | None = None
    parent_transformation_ids: tuple[Identifier, ...] = ()
    affected_regions: tuple[NonEmpty, ...]
    preconditions: tuple[NonEmpty, ...]
    applicable_model_families: tuple[NonEmpty, ...]
    applicable_operations: tuple[NonEmpty, ...]
    applicable_hardware: tuple[NonEmpty, ...]
    applicable_workloads: tuple[NonEmpty, ...]
    dependency_preconditions: tuple[DependencySelector, ...] = ()
    expected_benefit: float
    outcome: TransformationOutcome
    proposal_source: NonEmpty
    created_at: AwareDatetime

    @model_validator(mode="after")
    def validate_transformation(self) -> Self:
        required = (
            self.affected_regions,
            self.preconditions,
            self.applicable_model_families,
            self.applicable_operations,
            self.applicable_hardware,
            self.applicable_workloads,
        )
        if any(not values for values in required):
            raise ValueError("transformation applicability and preconditions must be explicit")
        dependency_keys = [
            (item.kind, item.name, item.version_range) for item in self.dependency_preconditions
        ]
        if len(dependency_keys) != len(set(dependency_keys)):
            raise ValueError("transformation dependency preconditions must be unique")
        return self


class EvidenceTargetKind(StrEnum):
    CANDIDATE = "candidate"
    TRANSFORMATION = "transformation"


class EvidenceResult(StrEnum):
    PASS = "pass"
    FAIL = "fail"
    INCONCLUSIVE = "inconclusive"


class EvidenceFreshness(StrEnum):
    FRESH = "fresh"
    STALE = "stale"


class EvidenceRecord(LineageModel):
    evidence_id: Identifier
    target_kind: EvidenceTargetKind
    target_id: Identifier
    evidence_type: NonEmpty
    result: EvidenceResult
    content_hash: Digest
    model_family: NonEmpty
    workload_regimes: tuple[NonEmpty, ...]
    hardware_architecture: NonEmpty
    dependencies: tuple[DependencyVersion, ...]
    base_confidence: Probability
    observed_at: AwareDatetime
    valid_until: AwareDatetime
    freshness: EvidenceFreshness = EvidenceFreshness.FRESH
    invalidation_event_ids: tuple[Identifier, ...] = ()

    @model_validator(mode="after")
    def validate_evidence(self) -> Self:
        if self.valid_until <= self.observed_at:
            raise ValueError("evidence validity must extend beyond observation")
        if len(self.invalidation_event_ids) != len(set(self.invalidation_event_ids)):
            raise ValueError("invalidation references must be unique")
        if self.freshness is EvidenceFreshness.STALE and not self.invalidation_event_ids:
            raise ValueError("stale evidence requires an invalidation event")
        dependency_keys = [(item.kind, item.name) for item in self.dependencies]
        if len(dependency_keys) != len(set(dependency_keys)):
            raise ValueError("evidence dependency identities must be unique")
        return self


class CounterexampleScope(StrEnum):
    CANDIDATE = "candidate_specific"
    TRANSFORMATION_FAMILY = "transformation_family"
    HARDWARE = "hardware_specific"
    DEPENDENCY = "dependency_version"
    UNIVERSAL_PRECONDITION = "universal_precondition"


class CounterexampleRecord(LineageModel):
    counterexample_id: Identifier
    candidate_id: Identifier
    transformation_id: Identifier | None
    transformation_family: NonEmpty
    scope: CounterexampleScope
    violated_contract: NonEmpty
    minimized_input_hash: Digest
    reproduction_command: tuple[NonEmpty, ...]
    learned_precondition: NonEmpty
    hardware_architecture: NonEmpty | None = None
    dependencies: tuple[DependencyVersion, ...] = ()
    created_at: AwareDatetime

    @model_validator(mode="after")
    def require_reproducer(self) -> Self:
        if not self.reproduction_command:
            raise ValueError("counterexample requires a bounded reproduction command")
        if self.scope is CounterexampleScope.HARDWARE and self.hardware_architecture is None:
            raise ValueError("hardware-specific counterexample requires hardware architecture")
        if self.scope is CounterexampleScope.DEPENDENCY and not self.dependencies:
            raise ValueError("dependency-specific counterexample requires dependencies")
        return self


class ConstraintPredicate(LineageModel):
    model_families: tuple[NonEmpty, ...] = ()
    hardware_architectures: tuple[NonEmpty, ...] = ()
    workload_regimes: tuple[NonEmpty, ...] = ()
    dependency_selectors: tuple[DependencySelector, ...] = ()

    @model_validator(mode="after")
    def require_selector(self) -> Self:
        if not any(
            (
                self.model_families,
                self.hardware_architectures,
                self.workload_regimes,
                self.dependency_selectors,
            )
        ):
            raise ValueError("constraint predicate must restrict at least one feature")
        return self


class LearnedConstraintRecord(LineageModel):
    constraint_id: Identifier
    counterexample_id: Identifier
    transformation_family: NonEmpty
    transformation_id: Identifier | None
    predicate: ConstraintPredicate
    rationale: NonEmpty
    created_at: AwareDatetime


class InvalidationEvent(LineageModel):
    invalidation_id: Identifier
    selector: DependencySelector
    reason: NonEmpty
    occurred_at: AwareDatetime


class TransferOutcome(StrEnum):
    SEEDED = "seeded"
    REVERIFIED = "reverified"
    IMPROVED = "improved"
    REJECTED = "rejected"
    NEGATIVE_TRANSFER = "negative_transfer"


class TransferRecord(LineageModel):
    transfer_id: Identifier
    target_task_id: Identifier
    transformation_id: Identifier
    source_evidence_ids: tuple[Identifier, ...]
    retrieval_score: Annotated[float, Field(ge=0.0)]
    rank: PositiveInt
    seed: int
    outcome: TransferOutcome
    rationale: NonEmpty
    created_at: AwareDatetime

    @model_validator(mode="after")
    def unique_evidence(self) -> Self:
        if not self.source_evidence_ids:
            raise ValueError("transfer must retain its source evidence")
        if len(self.source_evidence_ids) != len(set(self.source_evidence_ids)):
            raise ValueError("transfer evidence references must be unique")
        return self


class RelatedTransformation(LineageModel):
    transformation_id: Identifier
    source_task_id: Identifier
    score: Annotated[float, Field(ge=0.0)]
    effective_confidence: Probability
    evidence_ids: tuple[Identifier, ...]
    rationale: tuple[NonEmpty, ...]
    requires_reverification: Literal[True] = True


class TransformationQuery(LineageModel):
    model_family: NonEmpty | None = None
    operation: NonEmpty | None = None
    hardware_architecture: NonEmpty | None = None
    workload_regime: NonEmpty | None = None
    family: NonEmpty | None = None
    outcome: TransformationOutcome | None = None
    limit: Annotated[int, Field(ge=1, le=1000)] = 100
    scan_limit: Annotated[int, Field(ge=1, le=100_000)] = 1000

    @model_validator(mode="after")
    def validate_query(self) -> Self:
        if self.scan_limit < self.limit:
            raise ValueError("scan_limit cannot be less than result limit")
        if not any(
            (
                self.model_family,
                self.operation,
                self.hardware_architecture,
                self.workload_regime,
                self.family,
                self.outcome,
            )
        ):
            raise ValueError("lineage query must include at least one filter")
        return self


class UnseededProposal(LineageModel):
    proposal_id: Identifier
    seed: int


class SearchInitialization(LineageModel):
    task_id: Identifier
    seed: int
    lineage_seeds: tuple[RelatedTransformation, ...]
    unseeded_proposals: tuple[UnseededProposal, ...]


class LineageSnapshot(LineageModel):
    schema_version: Literal["1.0.0"] = LINEAGE_SCHEMA_VERSION
    exported_at: AwareDatetime
    tasks: tuple[TaskFeatures, ...]
    candidates: tuple[CandidateRecord, ...]
    transformations: tuple[TransformationRecord, ...]
    evidence: tuple[EvidenceRecord, ...]
    counterexamples: tuple[CounterexampleRecord, ...]
    constraints: tuple[LearnedConstraintRecord, ...]
    transfers: tuple[TransferRecord, ...]
    invalidations: tuple[InvalidationEvent, ...]
