"""Strict contracts for the deterministic Helix resource compiler.

The scheduler uses integer resource quantities and integer accounting units.  It
does not turn its discrete-tick predictions into measurements: every supplied
forecast, value estimate, and observation retains a reference to its source
artifact or raw samples.
"""

from __future__ import annotations

import hashlib
import json
import math
from enum import StrEnum
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

Identifier = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}$")]
Digest = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
NonEmpty = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=2048)]
FiniteFloat = Annotated[float, Field(allow_inf_nan=False)]
NonNegativeFloat = Annotated[float, Field(ge=0.0, allow_inf_nan=False)]


class SchedulerModel(BaseModel):
    """Fail-closed base model used at the scheduler JSON boundary."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, validate_default=True)


def canonical_digest(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class WorkClass(StrEnum):
    SERVING = "serving"
    ROLLOUT = "rollout"
    ENVIRONMENT = "environment"
    REWARD = "reward"
    VERIFIER = "verifier"
    TRAINING = "training"
    EVALUATION = "evaluation"


LEARNING_WORK_CLASSES: tuple[WorkClass, ...] = (
    WorkClass.ROLLOUT,
    WorkClass.ENVIRONMENT,
    WorkClass.REWARD,
    WorkClass.VERIFIER,
    WorkClass.TRAINING,
    WorkClass.EVALUATION,
)


class SchedulerPolicy(StrEnum):
    DEDICATED = "dedicated"
    STATIC = "static"
    UTILIZATION = "utilization"
    FIFO = "fifo"
    HELIX_VALUE_AWARE = "helix_value_aware"


class PrivacyClass(StrEnum):
    PUBLIC = "public"
    TENANT_PRIVATE = "tenant_private"
    RESTRICTED = "restricted"


class EffectClass(StrEnum):
    PURE = "pure"
    READ_ONLY = "read_only"
    REVERSIBLE = "reversible"
    EXTERNAL = "external"


class PreservationMode(StrEnum):
    RESTART = "restart"
    CHECKPOINT = "checkpoint"
    CONTINUUM = "continuum"


class FaultKind(StrEnum):
    TRAFFIC_SPIKE = "traffic_spike"
    GPU_LOSS = "gpu_loss"
    CPU_EXHAUSTION = "cpu_exhaustion"
    STORAGE_SLOWDOWN = "storage_slowdown"
    NETWORK_SLOWDOWN = "network_slowdown"
    VALUE_PREDICTION_ERROR = "value_prediction_error"


class DecisionKind(StrEnum):
    SELECT_BRANCH = "select_branch"
    REJECT = "reject"
    DEFER = "defer"
    START = "start"
    CONTINUE = "continue"
    PREEMPT = "preempt"
    COMPLETE = "complete"
    LEND_CAPACITY = "lend_capacity"
    RECLAIM_CAPACITY = "reclaim_capacity"
    APPLY_FAULT = "apply_fault"


class WorkStatus(StrEnum):
    COMPLETED = "completed"
    REJECTED = "rejected"
    DEFERRED = "deferred"


class EvidenceRef(SchedulerModel):
    """Reference to inputs or raw samples; it is never an evidence payload."""

    artifact_uri: Annotated[str, Field(min_length=1, max_length=2048)]
    artifact_sha256: Digest
    sample_ids: Annotated[tuple[Identifier, ...], Field(min_length=1, max_length=100_000)]

    @model_validator(mode="after")
    def validate_samples(self) -> Self:
        if len(self.sample_ids) != len(set(self.sample_ids)):
            raise ValueError("evidence sample identifiers must be unique")
        return self


class ResourceVector(SchedulerModel):
    """Integral capacity in portable, explicitly named scheduler units."""

    cpu_millicores: int = Field(ge=0, le=10**12)
    memory_mib: int = Field(ge=0, le=10**12)
    gpu_milliunits: int = Field(ge=0, le=10**9)
    storage_mib: int = Field(ge=0, le=10**15)
    storage_iops: int = Field(ge=0, le=10**12)
    network_mbps: int = Field(ge=0, le=10**12)

    @classmethod
    def zero(cls) -> ResourceVector:
        return cls(
            cpu_millicores=0,
            memory_mib=0,
            gpu_milliunits=0,
            storage_mib=0,
            storage_iops=0,
            network_mbps=0,
        )

    def as_tuple(self) -> tuple[int, int, int, int, int, int]:
        return (
            self.cpu_millicores,
            self.memory_mib,
            self.gpu_milliunits,
            self.storage_mib,
            self.storage_iops,
            self.network_mbps,
        )

    def is_zero(self) -> bool:
        return not any(self.as_tuple())

    def fits_within(self, capacity: ResourceVector) -> bool:
        return all(
            left <= right for left, right in zip(self.as_tuple(), capacity.as_tuple(), strict=True)
        )

    def add(self, other: ResourceVector) -> ResourceVector:
        values = tuple(
            left + right for left, right in zip(self.as_tuple(), other.as_tuple(), strict=True)
        )
        return ResourceVector._from_tuple(values)

    def subtract(self, other: ResourceVector) -> ResourceVector:
        if not other.fits_within(self):
            raise ValueError("resource subtraction would produce a negative quantity")
        values = tuple(
            left - right for left, right in zip(self.as_tuple(), other.as_tuple(), strict=True)
        )
        return ResourceVector._from_tuple(values)

    def positive_difference(self, other: ResourceVector) -> ResourceVector:
        values = tuple(
            max(0, left - right)
            for left, right in zip(self.as_tuple(), other.as_tuple(), strict=True)
        )
        return ResourceVector._from_tuple(values)

    def minimum(self, other: ResourceVector) -> ResourceVector:
        values = tuple(
            min(left, right) for left, right in zip(self.as_tuple(), other.as_tuple(), strict=True)
        )
        return ResourceVector._from_tuple(values)

    def maximum(self, other: ResourceVector) -> ResourceVector:
        values = tuple(
            max(left, right) for left, right in zip(self.as_tuple(), other.as_tuple(), strict=True)
        )
        return ResourceVector._from_tuple(values)

    def multiply(self, units: int) -> ResourceVector:
        if units < 0:
            raise ValueError("resource multiplier cannot be negative")
        values = tuple(value * units for value in self.as_tuple())
        return ResourceVector._from_tuple(values)

    def scaled_up(self, multiplier: float) -> ResourceVector:
        if not math.isfinite(multiplier) or multiplier < 0.0:
            raise ValueError("resource scale must be finite and non-negative")
        return ResourceVector._from_tuple(
            tuple(math.ceil(value * multiplier) for value in self.as_tuple())
        )

    def scaled_down(self, remaining_fraction: float) -> ResourceVector:
        if not math.isfinite(remaining_fraction) or not 0.0 <= remaining_fraction <= 1.0:
            raise ValueError("remaining resource fraction must be between zero and one")
        return ResourceVector._from_tuple(
            tuple(math.floor(value * remaining_fraction) for value in self.as_tuple())
        )

    @classmethod
    def _from_tuple(cls, values: tuple[int, ...]) -> ResourceVector:
        if len(values) != 6:
            raise ValueError("resource vectors have exactly six dimensions")
        return cls(
            cpu_millicores=values[0],
            memory_mib=values[1],
            gpu_milliunits=values[2],
            storage_mib=values[3],
            storage_iops=values[4],
            network_mbps=values[5],
        )


class ResourcePrices(SchedulerModel):
    """Integer micro-accounting units charged per resource unit per tick."""

    cpu_millicore_tick: int = Field(ge=0, le=10**12)
    memory_mib_tick: int = Field(ge=0, le=10**12)
    gpu_milliunit_tick: int = Field(ge=0, le=10**12)
    storage_mib_tick: int = Field(ge=0, le=10**12)
    storage_iop_tick: int = Field(ge=0, le=10**12)
    network_mbps_tick: int = Field(ge=0, le=10**12)

    def cost(self, resources: ResourceVector) -> int:
        prices = (
            self.cpu_millicore_tick,
            self.memory_mib_tick,
            self.gpu_milliunit_tick,
            self.storage_mib_tick,
            self.storage_iop_tick,
            self.network_mbps_tick,
        )
        return sum(
            quantity * price for quantity, price in zip(resources.as_tuple(), prices, strict=True)
        )


class ClassResourceVectors(SchedulerModel):
    """An explicit resource vector for every serving and learning activity."""

    serving: ResourceVector
    rollout: ResourceVector
    environment: ResourceVector
    reward: ResourceVector
    verifier: ResourceVector
    training: ResourceVector
    evaluation: ResourceVector

    def for_class(self, work_class: WorkClass) -> ResourceVector:
        return {
            WorkClass.SERVING: self.serving,
            WorkClass.ROLLOUT: self.rollout,
            WorkClass.ENVIRONMENT: self.environment,
            WorkClass.REWARD: self.reward,
            WorkClass.VERIFIER: self.verifier,
            WorkClass.TRAINING: self.training,
            WorkClass.EVALUATION: self.evaluation,
        }[work_class]

    @classmethod
    def zero(cls) -> ClassResourceVectors:
        zero = ResourceVector.zero()
        return cls(
            serving=zero,
            rollout=zero,
            environment=zero,
            reward=zero,
            verifier=zero,
            training=zero,
            evaluation=zero,
        )


class ValuePrediction(SchedulerModel):
    value: FiniteFloat
    model_id: Identifier
    model_version: NonEmpty
    evidence: EvidenceRef


class ServingDemandSample(SchedulerModel):
    tick: int = Field(ge=0, le=100_000)
    resource_units: int = Field(ge=0, le=10**9)
    predicted_latency_ms: NonNegativeFloat
    predicted_queue_depth: int = Field(ge=0, le=10**12)
    evidence: EvidenceRef


class ServingSLO(SchedulerModel):
    reserved_capacity: ResourceVector
    maximum_predicted_latency_ms: Annotated[float, Field(gt=0.0, allow_inf_nan=False)]
    maximum_predicted_queue_depth: int = Field(ge=0, le=10**12)


class PreservationOption(SchedulerModel):
    mode: PreservationMode
    pause_ticks: int = Field(ge=0, le=100_000)
    checkpoint_interval_ticks: int = Field(ge=0, le=100_000)
    storage_mib_written: int = Field(ge=0, le=10**12)
    network_mib_transferred: int = Field(ge=0, le=10**12)
    cost_microunits: int = Field(ge=0, le=10**18)
    method_evidence: EvidenceRef | None = None

    @model_validator(mode="after")
    def validate_mode(self) -> Self:
        if self.mode is PreservationMode.RESTART:
            if (
                self.checkpoint_interval_ticks != 0
                or self.storage_mib_written != 0
                or self.network_mib_transferred != 0
                or self.method_evidence is not None
            ):
                raise ValueError(
                    "restart preservation cannot claim checkpoint state, transfer, or method evidence"
                )
        elif self.mode is PreservationMode.CHECKPOINT:
            if self.checkpoint_interval_ticks == 0 or self.storage_mib_written == 0:
                raise ValueError("checkpoint preservation requires an interval and stored state")
            if self.method_evidence is None:
                raise ValueError("checkpoint preservation requires method evidence")
        else:
            if self.checkpoint_interval_ticks != 0:
                raise ValueError(
                    "Continuum preservation does not use a periodic checkpoint interval"
                )
            if self.method_evidence is None:
                raise ValueError("Continuum preservation requires adapter-contract evidence")
        return self


def restart_only() -> tuple[PreservationOption, ...]:
    return (
        PreservationOption(
            mode=PreservationMode.RESTART,
            pause_ticks=0,
            checkpoint_interval_ticks=0,
            storage_mib_written=0,
            network_mib_transferred=0,
            cost_microunits=0,
            method_evidence=None,
        ),
    )


class WorkUnit(SchedulerModel):
    work_id: Identifier
    branch_id: Identifier
    work_class: WorkClass
    tenant_id: Identifier
    privacy: PrivacyClass
    effect: EffectClass
    arrival_tick: int = Field(ge=0, le=100_000)
    duration_ticks: int = Field(gt=0, le=100_000)
    deadline_tick: int | None = Field(default=None, ge=0, le=100_000)
    policy_age_ticks: int = Field(ge=0, le=100_000)
    resource_units: int = Field(gt=0, le=10**9)
    predicted_learning_value: ValuePrediction
    preservation: Annotated[tuple[PreservationOption, ...], Field(min_length=1, max_length=3)] = (
        PreservationOption(
            mode=PreservationMode.RESTART,
            pause_ticks=0,
            checkpoint_interval_ticks=0,
            storage_mib_written=0,
            network_mib_transferred=0,
            cost_microunits=0,
            method_evidence=None,
        ),
    )

    @model_validator(mode="after")
    def validate_work(self) -> Self:
        if self.work_class is WorkClass.SERVING:
            raise ValueError("serving demand belongs in serving_forecast, not learning work")
        if self.deadline_tick is not None and self.deadline_tick <= self.arrival_tick:
            raise ValueError("work deadline must be after its arrival")
        modes = [option.mode for option in self.preservation]
        if len(modes) != len(set(modes)):
            raise ValueError("preservation modes must be unique per work item")
        return self


class SchedulerFault(SchedulerModel):
    fault_id: Identifier
    kind: FaultKind
    start_tick: int = Field(ge=0, le=100_000)
    end_tick: int = Field(gt=0, le=100_001)
    magnitude: Annotated[float, Field(gt=0.0, le=10.0, allow_inf_nan=False)]
    direction: Literal[-1, 1] = 1
    target_work_id: Identifier | None = None
    evidence: EvidenceRef

    @model_validator(mode="after")
    def validate_fault(self) -> Self:
        if self.end_tick <= self.start_tick:
            raise ValueError("fault end must be after its start")
        if self.kind not in {FaultKind.TRAFFIC_SPIKE, FaultKind.VALUE_PREDICTION_ERROR}:
            if self.magnitude > 1.0:
                raise ValueError("capacity-loss fault magnitude cannot exceed one")
            if self.direction != 1:
                raise ValueError("capacity-loss faults do not accept a direction")
        if self.kind is FaultKind.VALUE_PREDICTION_ERROR:
            return self
        if self.target_work_id is not None:
            raise ValueError("only value prediction faults may target a work item")
        if self.direction != 1:
            raise ValueError("only value prediction faults may be signed")
        return self

    def active(self, tick: int) -> bool:
        return self.start_tick <= tick < self.end_tick


class SchedulerConstraints(SchedulerModel):
    max_budget_microunits: int = Field(ge=0, le=10**30)
    prices: ResourcePrices
    max_policy_staleness_ticks: int = Field(ge=0, le=100_000)
    allowed_tenant_ids: Annotated[tuple[Identifier, ...], Field(min_length=1, max_length=10_000)]
    maximum_privacy: PrivacyClass
    allowed_effects: Annotated[tuple[EffectClass, ...], Field(min_length=1, max_length=4)]
    max_selected_branches: int = Field(gt=0, le=10_000)
    max_preemptions_per_work: int = Field(ge=0, le=1_000)
    max_total_preemptions: int = Field(ge=0, le=1_000_000)
    capacity_lending: bool

    @model_validator(mode="after")
    def validate_sets(self) -> Self:
        if len(self.allowed_tenant_ids) != len(set(self.allowed_tenant_ids)):
            raise ValueError("allowed tenant identifiers must be unique")
        if len(self.allowed_effects) != len(set(self.allowed_effects)):
            raise ValueError("allowed effects must be unique")
        return self


class SchedulerRequest(SchedulerModel):
    schema_version: Literal["sloforge.helix.scheduler-request/v1"] = (
        "sloforge.helix.scheduler-request/v1"
    )
    request_id: Identifier
    seed: int = Field(ge=0, le=2**64 - 1)
    policy: SchedulerPolicy
    horizon_ticks: int = Field(gt=0, le=100_000)
    capacity: ResourceVector
    resource_vectors: ClassResourceVectors
    serving_slo: ServingSLO
    serving_forecast: Annotated[
        tuple[ServingDemandSample, ...], Field(min_length=1, max_length=100_000)
    ]
    work: Annotated[tuple[WorkUnit, ...], Field(max_length=100_000)] = ()
    constraints: SchedulerConstraints
    static_limits: ClassResourceVectors | None = None
    faults: Annotated[tuple[SchedulerFault, ...], Field(max_length=10_000)] = ()
    max_audit_records: int = Field(gt=0, le=10_000_000)
    max_scheduler_events: int = Field(gt=0, le=10_000_000, default=5_000_000)

    @model_validator(mode="after")
    def validate_request(self) -> Self:
        if self.capacity.is_zero():
            raise ValueError("scheduler capacity cannot be zero")
        if not self.serving_slo.reserved_capacity.fits_within(self.capacity):
            raise ValueError("serving reservation exceeds total capacity")
        if self.resource_vectors.serving.is_zero():
            raise ValueError("the serving resource vector cannot be zero")
        ticks = tuple(sample.tick for sample in self.serving_forecast)
        if ticks != tuple(range(self.horizon_ticks)):
            raise ValueError("serving forecast ticks must exactly cover the scheduling horizon")
        work_ids = [item.work_id for item in self.work]
        if len(work_ids) != len(set(work_ids)):
            raise ValueError("scheduler work identifiers must be unique")
        fault_ids = [item.fault_id for item in self.faults]
        if len(fault_ids) != len(set(fault_ids)):
            raise ValueError("scheduler fault identifiers must be unique")
        unknown_targets = {
            fault.target_work_id
            for fault in self.faults
            if fault.target_work_id is not None and fault.target_work_id not in set(work_ids)
        }
        if unknown_targets:
            raise ValueError("value prediction fault references unknown work")
        if self.policy is SchedulerPolicy.STATIC and self.static_limits is None:
            raise ValueError("static scheduling requires explicit per-class limits")
        if self.policy is not SchedulerPolicy.STATIC and self.static_limits is not None:
            raise ValueError("static limits are only valid for static scheduling")
        event_upper_bound = self.horizon_ticks * max(1, len(self.work) + len(self.faults))
        if event_upper_bound > self.max_scheduler_events:
            raise ValueError("scheduler request exceeds max_scheduler_events")
        return self


class PreservationAccounting(SchedulerModel):
    mode: PreservationMode
    progress_before_ticks: int = Field(ge=0)
    preserved_work_ticks: int = Field(ge=0)
    lost_work_ticks: int = Field(ge=0)
    pause_ticks: int = Field(ge=0)
    storage_mib_written: int = Field(ge=0)
    network_mib_transferred: int = Field(ge=0)
    cost_microunits: int = Field(ge=0)
    method_evidence: EvidenceRef | None = None

    @model_validator(mode="after")
    def validate_conservation(self) -> Self:
        if self.preserved_work_ticks + self.lost_work_ticks != self.progress_before_ticks:
            raise ValueError("preservation accounting must conserve completed work")
        return self


class PreemptionRecord(SchedulerModel):
    sequence: int = Field(ge=0)
    tick: int = Field(ge=0)
    work_id: Identifier
    reason: NonEmpty
    selected_mode: PreservationMode
    selected: PreservationAccounting
    alternatives: Annotated[tuple[PreservationAccounting, ...], Field(min_length=1, max_length=3)]
    total_preemptions_for_work: int = Field(gt=0)


class AuditDecision(SchedulerModel):
    sequence: int = Field(ge=0)
    tick: int = Field(ge=0)
    kind: DecisionKind
    subject_id: Identifier
    reason: NonEmpty
    effective_capacity: ResourceVector
    requested_resources: ResourceVector
    effective_predicted_value: FiniteFloat | None = None
    fault_ids: tuple[Identifier, ...] = ()


class TickAllocation(SchedulerModel):
    tick: int = Field(ge=0)
    effective_capacity: ResourceVector
    allocations: ClassResourceVectors
    serving_resources: ResourceVector
    learning_resources: ResourceVector
    lent_capacity: ResourceVector
    reclaimed_capacity: ResourceVector
    running_work_ids: tuple[Identifier, ...]
    active_fault_ids: tuple[Identifier, ...]
    serving_predicted_latency_ms: NonNegativeFloat
    serving_predicted_queue_depth: int = Field(ge=0)
    serving_slo_satisfied: Literal[True] = True
    cost_microunits: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_capacity(self) -> Self:
        if self.serving_resources.add(self.learning_resources) != self.allocations_total():
            raise ValueError("tick class allocations disagree with serving and learning totals")
        if not self.allocations_total().fits_within(self.effective_capacity):
            raise ValueError("tick allocations exceed effective capacity")
        return self

    def allocations_total(self) -> ResourceVector:
        total = ResourceVector.zero()
        for work_class in WorkClass:
            total = total.add(self.allocations.for_class(work_class))
        return total


class WorkOutcome(SchedulerModel):
    work_id: Identifier
    branch_id: Identifier
    work_class: WorkClass
    status: WorkStatus
    reason: NonEmpty
    progress_ticks: int = Field(ge=0)
    executed_ticks: int = Field(ge=0)
    lost_work_ticks: int = Field(ge=0)
    preemptions: int = Field(ge=0)
    started_at_tick: int | None = Field(default=None, ge=0)
    completed_at_tick: int | None = Field(default=None, ge=0)
    predicted_learning_value: FiniteFloat
    prediction_evidence: EvidenceRef

    @model_validator(mode="after")
    def validate_status(self) -> Self:
        if self.executed_ticks != self.progress_ticks + self.lost_work_ticks:
            raise ValueError("executed work must equal retained progress plus lost work")
        if self.status is WorkStatus.COMPLETED:
            if self.completed_at_tick is None:
                raise ValueError("completed work requires a completion tick")
            if self.progress_ticks <= 0:
                raise ValueError("completed work must retain positive progress")
        elif self.completed_at_tick is not None:
            raise ValueError("incomplete work cannot claim a completion tick")
        if self.started_at_tick is None and self.executed_ticks != 0:
            raise ValueError("unstarted work cannot claim executed ticks")
        return self


class BudgetAccounting(SchedulerModel):
    limit_microunits: int = Field(ge=0)
    serving_microunits: int = Field(ge=0)
    learning_microunits: int = Field(ge=0)
    preservation_microunits: int = Field(ge=0)
    total_microunits: int = Field(ge=0)
    remaining_microunits: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_total(self) -> Self:
        expected = self.serving_microunits + self.learning_microunits + self.preservation_microunits
        if self.total_microunits != expected:
            raise ValueError("budget total must sum serving, learning, and preservation")
        if self.total_microunits + self.remaining_microunits != self.limit_microunits:
            raise ValueError("budget accounting must conserve the configured limit")
        return self


class SchedulerPlan(SchedulerModel):
    schema_version: Literal["sloforge.helix.scheduler-plan/v1"] = "sloforge.helix.scheduler-plan/v1"
    plan_id: Digest
    request_digest: Digest
    request_id: Identifier
    seed: int = Field(ge=0, le=2**64 - 1)
    policy: SchedulerPolicy
    selected_branch_ids: tuple[Identifier, ...]
    ticks: tuple[TickAllocation, ...]
    outcomes: tuple[WorkOutcome, ...]
    preemptions: tuple[PreemptionRecord, ...]
    decisions: tuple[AuditDecision, ...]
    budget: BudgetAccounting
    predicted_learning_value: FiniteFloat
    scheduler_adjusted_predicted_value: FiniteFloat
    completed_work_ids: tuple[Identifier, ...]
    limitations: Annotated[tuple[NonEmpty, ...], Field(min_length=1)]

    @model_validator(mode="after")
    def validate_plan(self) -> Self:
        if tuple(tick.tick for tick in self.ticks) != tuple(range(len(self.ticks))):
            raise ValueError("scheduler plan ticks must be dense and ordered")
        if tuple(item.sequence for item in self.decisions) != tuple(range(len(self.decisions))):
            raise ValueError("scheduler audit decisions must be dense and ordered")
        if tuple(item.sequence for item in self.preemptions) != tuple(range(len(self.preemptions))):
            raise ValueError("scheduler preemptions must be dense and ordered")
        if len(self.selected_branch_ids) != len(set(self.selected_branch_ids)):
            raise ValueError("selected branch identifiers must be unique")
        if len(self.completed_work_ids) != len(set(self.completed_work_ids)):
            raise ValueError("completed work identifiers must be unique")
        completed = tuple(
            item.work_id for item in self.outcomes if item.status is WorkStatus.COMPLETED
        )
        if tuple(sorted(completed)) != tuple(sorted(self.completed_work_ids)):
            raise ValueError("completed work index disagrees with work outcomes")
        predicted = math.fsum(
            item.predicted_learning_value
            for item in self.outcomes
            if item.status is WorkStatus.COMPLETED
        )
        if not math.isclose(predicted, self.predicted_learning_value, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("plan predicted value disagrees with completed work outcomes")
        adjusted_values = tuple(
            item.effective_predicted_value
            for item in self.decisions
            if item.kind is DecisionKind.COMPLETE
        )
        if any(value is None for value in adjusted_values):
            raise ValueError("completed audit decisions require an adjusted predicted value")
        adjusted = math.fsum(value for value in adjusted_values if value is not None)
        if not math.isclose(
            adjusted,
            self.scheduler_adjusted_predicted_value,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("scheduler-adjusted value disagrees with completion decisions")
        if self.budget.serving_microunits + self.budget.learning_microunits != sum(
            tick.cost_microunits for tick in self.ticks
        ):
            raise ValueError("tick costs disagree with serving and learning budget accounting")
        payload = self.model_dump(mode="json", exclude={"plan_id"})
        if canonical_digest(payload) != self.plan_id:
            raise ValueError("scheduler plan identifier is invalid")
        return self


class RawLearningValueSample(SchedulerModel):
    sample_id: Identifier
    work_id: Identifier
    seed: int = Field(ge=0, le=2**64 - 1)
    value: FiniteFloat
    observed_at_tick: int = Field(ge=0)
    evidence: EvidenceRef

    @model_validator(mode="after")
    def validate_identity(self) -> Self:
        if self.sample_id not in self.evidence.sample_ids:
            raise ValueError("raw value sample identifier must appear in its evidence reference")
        return self


class LearningValueComparison(SchedulerModel):
    work_id: Identifier
    predicted_value: FiniteFloat
    observed_value: FiniteFloat
    signed_error: FiniteFloat
    absolute_error: NonNegativeFloat
    observed_mean_lower_95: FiniteFloat
    observed_mean_upper_95: FiniteFloat
    observed_sample_standard_deviation: NonNegativeFloat
    sample_seeds: Annotated[tuple[int, ...], Field(min_length=2, max_length=32)]
    raw_sample_ids: Annotated[tuple[Identifier, ...], Field(min_length=2, max_length=32)]
    evidence_artifacts: Annotated[tuple[EvidenceRef, ...], Field(min_length=2, max_length=32)]

    @model_validator(mode="after")
    def validate_error(self) -> Self:
        expected = self.observed_value - self.predicted_value
        if not math.isclose(expected, self.signed_error, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("signed value error must equal observed minus predicted")
        if not math.isclose(abs(expected), self.absolute_error, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("absolute value error disagrees with signed error")
        if not self.observed_mean_lower_95 <= self.observed_value <= self.observed_mean_upper_95:
            raise ValueError("observed mean must lie inside its 95% interval")
        if tuple(sorted(set(self.sample_seeds))) != self.sample_seeds:
            raise ValueError("comparison seeds must be sorted and unique")
        if len(self.raw_sample_ids) != len(set(self.raw_sample_ids)):
            raise ValueError("comparison raw sample identifiers must be unique")
        if not (len(self.sample_seeds) == len(self.raw_sample_ids) == len(self.evidence_artifacts)):
            raise ValueError("comparison seed, sample, and evidence counts must agree")
        if any(
            sample_id not in evidence.sample_ids
            for sample_id, evidence in zip(
                self.raw_sample_ids, self.evidence_artifacts, strict=True
            )
        ):
            raise ValueError("comparison samples must retain matching raw evidence")
        return self


class LearningValueEvaluation(SchedulerModel):
    schema_version: Literal["sloforge.helix.learning-value-evaluation/v1"] = (
        "sloforge.helix.learning-value-evaluation/v1"
    )
    evaluation_id: Digest
    plan_id: Digest
    seed: int = Field(ge=0, le=2**64 - 1)
    comparisons: tuple[LearningValueComparison, ...]
    missing_observation_work_ids: tuple[Identifier, ...]
    predicted_total_for_compared_work: FiniteFloat
    observed_total: FiniteFloat
    signed_error_total: FiniteFloat
    mean_absolute_error: NonNegativeFloat
    observation_seeds: Annotated[tuple[int, ...], Field(max_length=32)]
    raw_sample_count: int = Field(ge=0)
    limitations: Annotated[tuple[NonEmpty, ...], Field(min_length=1)]

    @model_validator(mode="after")
    def validate_evaluation(self) -> Self:
        predicted = math.fsum(item.predicted_value for item in self.comparisons)
        observed = math.fsum(item.observed_value for item in self.comparisons)
        if not math.isclose(
            predicted, self.predicted_total_for_compared_work, rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError("evaluation predicted total disagrees with comparisons")
        if not math.isclose(observed, self.observed_total, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("evaluation observed total disagrees with comparisons")
        if not math.isclose(
            observed - predicted, self.signed_error_total, rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError("evaluation signed error total is invalid")
        expected_mae = (
            math.fsum(item.absolute_error for item in self.comparisons) / len(self.comparisons)
            if self.comparisons
            else 0.0
        )
        if not math.isclose(expected_mae, self.mean_absolute_error, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("evaluation mean absolute error is invalid")
        expected_seeds = tuple(
            sorted({seed for item in self.comparisons for seed in item.sample_seeds})
        )
        if self.observation_seeds != expected_seeds:
            raise ValueError("evaluation observation seeds disagree with comparisons")
        samples = sum(len(item.raw_sample_ids) for item in self.comparisons)
        if samples != self.raw_sample_count:
            raise ValueError("evaluation raw sample count disagrees with comparisons")
        payload = self.model_dump(mode="json", exclude={"evaluation_id"})
        if canonical_digest(payload) != self.evaluation_id:
            raise ValueError("learning-value evaluation identifier is invalid")
        return self
