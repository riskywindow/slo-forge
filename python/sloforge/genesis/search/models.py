"""Typed deterministic search candidates, stages, events, and objectives."""

from __future__ import annotations

import math
from enum import StrEnum
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from sloforge.genesis.ir import (
    ArtifactDigest,
    BudgetUsage,
    CandidateFailureState,
    CandidateState,
    SearchBudget,
    TransformationFamily,
)

NonEmpty = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
PositiveInt = Annotated[int, Field(gt=0)]
NonNegativeInt = Annotated[int, Field(ge=0)]
Region = Literal[
    "workflow", "request", "serving", "state", "distributed", "tensor", "kernel", "recovery"
]


class SearchModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, validate_default=True)


class ParameterValue(SearchModel):
    key: NonEmpty
    value: NonEmpty


class MutationChoice(SearchModel):
    transformation_id: NonEmpty
    family: TransformationFamily
    regions: tuple[Region, ...]
    parameters: tuple[ParameterValue, ...]
    expected_upside: float
    invalidity_risk: Annotated[float, Field(ge=0.0, le=1.0)]
    feature_delta: tuple[float, ...]

    @model_validator(mode="after")
    def valid_mutation(self) -> Self:
        if not self.regions:
            raise ValueError("mutation must affect at least one genome region")
        if len(self.regions) != len(set(self.regions)):
            raise ValueError("mutation regions must be unique")
        keys = [parameter.key for parameter in self.parameters]
        if len(keys) != len(set(keys)):
            raise ValueError("mutation parameter keys must be unique")
        values = (self.expected_upside, *self.feature_delta)
        if not self.feature_delta or not all(math.isfinite(value) for value in values):
            raise ValueError("mutation estimates and features must be finite and non-empty")
        return self

    def parameter(self, key: str) -> str | None:
        return next((item.value for item in self.parameters if item.key == key), None)


class CandidateDesign(SearchModel):
    candidate_id: NonEmpty
    seed: NonNegativeInt
    genome_hash: ArtifactDigest
    parent_candidate_ids: tuple[NonEmpty, ...]
    mutations: tuple[MutationChoice, ...]
    feature_vector: tuple[float, ...]
    proposal_engine: Literal["beam", "evolutionary", "local", "novelty", "corrective", "fixture"]

    @model_validator(mode="after")
    def valid_design(self) -> Self:
        identifiers = [mutation.transformation_id for mutation in self.mutations]
        if not self.mutations or len(identifiers) != len(set(identifiers)):
            raise ValueError("candidate mutations must be non-empty and unique")
        if not self.feature_vector or len(self.feature_vector) > 64:
            raise ValueError("candidate feature vector must contain between 1 and 64 values")
        if not all(math.isfinite(value) for value in self.feature_vector):
            raise ValueError("candidate feature vector must be finite")
        return self

    @property
    def affected_regions(self) -> tuple[Region, ...]:
        return tuple(sorted({region for mutation in self.mutations for region in mutation.regions}))

    @property
    def cross_layer(self) -> bool:
        return len(self.affected_regions) >= 2


class FidelityStage(StrEnum):
    STATIC_PRUNING = "static_pruning"
    ANALYTICAL_BOUND = "analytical_lower_bound"
    COMPILE = "compile"
    DIGITAL_TWIN = "digital_twin"
    DETERMINISTIC_TESTS = "deterministic_tests"
    PROPERTY_VERIFICATION = "property_verification"
    MODEL_CHECK = "bounded_model_check"
    SIMULATION = "full_simulation"
    HARDWARE_MICROBENCHMARK = "hardware_microbenchmark"
    END_TO_END_BENCHMARK = "end_to_end_benchmark"
    SHADOW = "shadow_validation"
    CANARY = "canary_validation"


_FAILURE_STATES_BY_STAGE: dict[FidelityStage, frozenset[CandidateFailureState]] = {
    FidelityStage.STATIC_PRUNING: frozenset(
        {
            CandidateFailureState.STATIC_REJECTED,
            CandidateFailureState.RESOURCE_REJECTED,
            CandidateFailureState.SUPERSEDED,
        }
    ),
    FidelityStage.ANALYTICAL_BOUND: frozenset(
        {
            CandidateFailureState.RESOURCE_REJECTED,
            CandidateFailureState.PERFORMANCE_REJECTED,
            CandidateFailureState.SUPERSEDED,
        }
    ),
    FidelityStage.COMPILE: frozenset(
        {
            CandidateFailureState.COMPILE_REJECTED,
            CandidateFailureState.SANDBOX_VIOLATION,
            CandidateFailureState.SUPERSEDED,
        }
    ),
    FidelityStage.DIGITAL_TWIN: frozenset(
        {
            CandidateFailureState.RESOURCE_REJECTED,
            CandidateFailureState.PERFORMANCE_REJECTED,
            CandidateFailureState.SUPERSEDED,
        }
    ),
    FidelityStage.DETERMINISTIC_TESTS: frozenset(
        {
            CandidateFailureState.SEMANTIC_REJECTED,
            CandidateFailureState.QUALITY_REJECTED,
            CandidateFailureState.RESOURCE_REJECTED,
            CandidateFailureState.SANDBOX_VIOLATION,
            CandidateFailureState.SUPERSEDED,
        }
    ),
    FidelityStage.PROPERTY_VERIFICATION: frozenset(
        {
            CandidateFailureState.SEMANTIC_REJECTED,
            CandidateFailureState.QUALITY_REJECTED,
            CandidateFailureState.RESOURCE_REJECTED,
            CandidateFailureState.SANDBOX_VIOLATION,
            CandidateFailureState.SUPERSEDED,
        }
    ),
    FidelityStage.MODEL_CHECK: frozenset(
        {
            CandidateFailureState.MODEL_CHECK_REJECTED,
            CandidateFailureState.RESOURCE_REJECTED,
            CandidateFailureState.SANDBOX_VIOLATION,
            CandidateFailureState.SUPERSEDED,
        }
    ),
    FidelityStage.SIMULATION: frozenset(
        {
            CandidateFailureState.RESOURCE_REJECTED,
            CandidateFailureState.PERFORMANCE_REJECTED,
            CandidateFailureState.SUPERSEDED,
        }
    ),
    FidelityStage.HARDWARE_MICROBENCHMARK: frozenset(
        {
            CandidateFailureState.SEMANTIC_REJECTED,
            CandidateFailureState.QUALITY_REJECTED,
            CandidateFailureState.RESOURCE_REJECTED,
            CandidateFailureState.PERFORMANCE_REJECTED,
            CandidateFailureState.SANDBOX_VIOLATION,
            CandidateFailureState.SUPERSEDED,
        }
    ),
    FidelityStage.END_TO_END_BENCHMARK: frozenset(
        {
            CandidateFailureState.SEMANTIC_REJECTED,
            CandidateFailureState.QUALITY_REJECTED,
            CandidateFailureState.RESOURCE_REJECTED,
            CandidateFailureState.PERFORMANCE_REJECTED,
            CandidateFailureState.SANDBOX_VIOLATION,
            CandidateFailureState.SUPERSEDED,
        }
    ),
    FidelityStage.SHADOW: frozenset(
        {
            CandidateFailureState.SHADOW_REJECTED,
            CandidateFailureState.SANDBOX_VIOLATION,
            CandidateFailureState.SUPERSEDED,
        }
    ),
    FidelityStage.CANARY: frozenset(
        {
            CandidateFailureState.CANARY_REJECTED,
            CandidateFailureState.SANDBOX_VIOLATION,
            CandidateFailureState.SUPERSEDED,
        }
    ),
}


class ObjectiveVector(SearchModel):
    correctness_confidence: Annotated[float, Field(ge=0.0, le=1.0)]
    quality: Annotated[float, Field(ge=0.0, le=1.0)]
    ttft_ms: Annotated[float, Field(gt=0.0)]
    token_latency_ms: Annotated[float, Field(gt=0.0)]
    goodput: Annotated[float, Field(ge=0.0)]
    throughput: Annotated[float, Field(ge=0.0)]
    cost_usd_per_hour: Annotated[float, Field(ge=0.0)]
    energy_joules_per_token: Annotated[float, Field(ge=0.0)]
    startup_ms: Annotated[float, Field(ge=0.0)]
    memory_bytes: Annotated[float, Field(ge=0.0)]
    reliability: Annotated[float, Field(ge=0.0, le=1.0)]
    implementation_complexity: Annotated[float, Field(ge=0.0)]
    transition_cost: Annotated[float, Field(ge=0.0)]

    def desirability(self) -> tuple[float, ...]:
        return (
            self.correctness_confidence,
            self.quality,
            -self.ttft_ms,
            -self.token_latency_ms,
            self.goodput,
            self.throughput,
            -self.cost_usd_per_hour,
            -self.energy_joules_per_token,
            -self.startup_ms,
            -self.memory_bytes,
            self.reliability,
            -self.implementation_complexity,
            -self.transition_cost,
        )


class StageResult(SearchModel):
    stage: FidelityStage
    passed: bool
    reason: NonEmpty
    usage: BudgetUsage = Field(default_factory=BudgetUsage)
    objective: ObjectiveVector | None = None
    selection_utility: float | None = None
    failure_state: CandidateFailureState | None = None
    hardware_backed: bool = False
    evidence_ids: tuple[NonEmpty, ...] = ()

    @model_validator(mode="after")
    def result_is_consistent(self) -> Self:
        if self.passed and self.failure_state is not None:
            raise ValueError("passing stage cannot declare a failure state")
        if not self.passed and self.failure_state is None:
            raise ValueError("failed stage must declare a terminal failure state")
        if (
            self.failure_state is not None
            and self.failure_state not in _FAILURE_STATES_BY_STAGE[self.stage]
        ):
            raise ValueError("failure state is incompatible with the fidelity stage")
        if self.hardware_backed and self.stage not in {
            FidelityStage.HARDWARE_MICROBENCHMARK,
            FidelityStage.END_TO_END_BENCHMARK,
            FidelityStage.SHADOW,
            FidelityStage.CANARY,
        }:
            raise ValueError("hardware_backed is invalid for this fidelity stage")
        if self.selection_utility is not None and not math.isfinite(self.selection_utility):
            raise ValueError("selection utility must be finite")
        return self


class StageSpecification(SearchModel):
    stage: FidelityStage
    maximum_usage: BudgetUsage


class SearchConfiguration(SearchModel):
    seed: NonNegativeInt
    budget: SearchBudget
    stages: tuple[StageSpecification, ...]
    maximum_candidates: PositiveInt
    maximum_archive_size: PositiveInt
    maximum_events: PositiveInt = 10_000
    allow_hardware: bool = False

    @model_validator(mode="after")
    def valid_plan(self) -> Self:
        if self.maximum_candidates > self.budget.candidate_count:
            raise ValueError("maximum_candidates exceeds candidate budget")
        stages = [item.stage for item in self.stages]
        if not stages or len(stages) != len(set(stages)):
            raise ValueError("search stages must be non-empty and unique")
        order = tuple(FidelityStage)
        if stages != sorted(stages, key=order.index):
            raise ValueError("search stages must follow deterministic fidelity order")
        lifecycle_stages = (
            FidelityStage.STATIC_PRUNING,
            FidelityStage.COMPILE,
            FidelityStage.DETERMINISTIC_TESTS,
            FidelityStage.PROPERTY_VERIFICATION,
            FidelityStage.MODEL_CHECK,
            FidelityStage.SIMULATION,
            FidelityStage.HARDWARE_MICROBENCHMARK,
            FidelityStage.SHADOW,
            FidelityStage.CANARY,
        )
        included = tuple(stage for stage in lifecycle_stages if stage in stages)
        if included != lifecycle_stages[: len(included)]:
            raise ValueError("candidate lifecycle fidelity stages cannot be skipped")
        if FidelityStage.HARDWARE_MICROBENCHMARK in stages and not self.allow_hardware:
            raise ValueError("hardware evaluation requires explicit allow_hardware")
        return self


class SearchEvent(SearchModel):
    schema_version: Literal["1.0.0"] = "1.0.0"
    sequence: NonNegativeInt
    event_type: Literal[
        "candidate_proposed",
        "candidate_suppressed",
        "budget_consumed",
        "stage_result",
        "lifecycle_transition",
        "pareto_updated",
        "budget_exhausted",
    ]
    candidate_id: NonEmpty | None = None
    stage: FidelityStage | None = None
    from_state: CandidateState | None = None
    to_state: CandidateState | None = None
    passed: bool | None = None
    reason: NonEmpty
    usage: BudgetUsage
    evidence_ids: tuple[NonEmpty, ...] = ()


class CandidateSearchResult(SearchModel):
    design: CandidateDesign
    final_state: CandidateState
    stage_results: tuple[StageResult, ...]
    budget_exhausted: bool


class SearchRunResult(SearchModel):
    seed: NonNegativeInt
    candidates: tuple[CandidateSearchResult, ...]
    pareto_candidate_ids: tuple[NonEmpty, ...]
    usage: BudgetUsage
    events_path: NonEmpty
