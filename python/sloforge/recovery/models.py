"""Typed recovery policy and restart-safe execution state."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

NonEmpty = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class RecoveryModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, validate_default=True)


class ExecutionTarget(StrEnum):
    SIMULATED = "simulated"
    LOCAL = "local"
    DOCKER = "docker"
    EXTERNAL = "external"


class RecoveryPolicy(RecoveryModel):
    minimum_diagnosis_confidence: float = Field(default=0.70, ge=0.0, le=1.0)
    execution_target: ExecutionTarget = ExecutionTarget.SIMULATED
    external_mutation_authorized: bool = False
    allow_degraded_model: bool = False
    shadow_fraction: float = Field(default=0.10, gt=0.0, le=1.0)
    canary_fraction: float = Field(default=0.05, gt=0.0, le=1.0)
    minimum_shadow_samples: int = Field(default=20, gt=0)
    minimum_canary_samples: int = Field(default=40, gt=0)
    maximum_inflight_streams_at_drain: int = Field(default=0, ge=0)
    preserve_started_streams: bool = True
    target_p99_tpot_ms: float = Field(default=45.0, gt=0.0)
    target_p95_ttft_ms: float = Field(default=250.0, gt=0.0)
    maximum_error_rate: float = Field(default=0.01, ge=0.0, le=1.0)
    maximum_build_seconds: float = Field(default=300.0, gt=0.0)

    @model_validator(mode="after")
    def external_execution_is_explicit(self) -> Self:
        if (
            self.execution_target is ExecutionTarget.EXTERNAL
            and not self.external_mutation_authorized
        ):
            raise ValueError("external recovery execution requires explicit mutation authorization")
        if (
            self.external_mutation_authorized
            and self.execution_target is not ExecutionTarget.EXTERNAL
        ):
            raise ValueError("external mutation authorization is valid only for an external target")
        return self


class RecoveryState(StrEnum):
    PROPOSED = "PROPOSED"
    VALIDATED_IN_SIMULATION = "VALIDATED_IN_SIMULATION"
    BUILDING_REPLACEMENT = "BUILDING_REPLACEMENT"
    SHADOWING = "SHADOWING"
    CANARYING = "CANARYING"
    PROMOTING = "PROMOTING"
    DRAINING_OLD = "DRAINING_OLD"
    COMPLETED = "COMPLETED"
    REJECTED = "REJECTED"
    ABORTED = "ABORTED"
    ROLLED_BACK = "ROLLED_BACK"
    OPERATOR_REQUIRED = "OPERATOR_REQUIRED"

    @property
    def terminal(self) -> bool:
        return self in {
            RecoveryState.COMPLETED,
            RecoveryState.REJECTED,
            RecoveryState.ABORTED,
            RecoveryState.ROLLED_BACK,
            RecoveryState.OPERATOR_REQUIRED,
        }


class MetricObservation(RecoveryModel):
    name: NonEmpty
    value: float = Field(allow_inf_nan=False)
    window_seconds: float = Field(gt=0.0)


class RecoveryObservation(RecoveryModel):
    observed_at_ms: int = Field(ge=0)
    idempotency_key: NonEmpty
    simulation_validated: bool = False
    replacement_ready: bool = False
    shadow_samples: int = Field(default=0, ge=0)
    canary_samples: int = Field(default=0, ge=0)
    active_started_streams: int = Field(default=0, ge=0)
    traffic_migration_complete: bool = False
    metrics: tuple[MetricObservation, ...] = ()
    completed_action_ids: tuple[NonEmpty, ...] = ()
    failed_action_id: str | None = None
    failed_reason: str | None = None

    @model_validator(mode="after")
    def validate_observation(self) -> Self:
        names = [metric.name for metric in self.metrics]
        if len(names) != len(set(names)):
            raise ValueError("metric observations must have unique names")
        if len(self.completed_action_ids) != len(set(self.completed_action_ids)):
            raise ValueError("completed recovery action IDs must be unique")
        if (self.failed_action_id is None) != (self.failed_reason is None):
            raise ValueError("failed action and failure reason must be supplied together")
        return self


class RecoveryMachineConfig(RecoveryModel):
    external_mutation_authorized: bool = False
    minimum_plan_confidence: float = Field(default=0.70, ge=0.0, le=1.0)
    maximum_total_duration_ms: int = Field(default=900_000, gt=0)
    proposed_timeout_ms: int = Field(default=30_000, gt=0)
    validation_timeout_ms: int = Field(default=30_000, gt=0)
    build_timeout_ms: int = Field(default=300_000, gt=0)
    shadow_timeout_ms: int = Field(default=120_000, gt=0)
    canary_timeout_ms: int = Field(default=120_000, gt=0)
    promotion_timeout_ms: int = Field(default=60_000, gt=0)
    drain_timeout_ms: int = Field(default=300_000, gt=0)
    promotion_cooldown_ms: int = Field(default=5_000, ge=0)
    maximum_audit_records: int = Field(default=1_024, ge=32, le=65_536)
    maximum_idempotency_keys: int = Field(default=256, ge=16, le=4_096)


class AuditField(RecoveryModel):
    name: NonEmpty
    value: NonEmpty


class AuditRecord(RecoveryModel):
    record_id: NonEmpty
    sequence: int = Field(ge=0)
    at_ms: int = Field(ge=0)
    event: Literal["created", "transition", "guard", "action", "wait", "restored"]
    state_before: RecoveryState
    state_after: RecoveryState
    reason: NonEmpty
    idempotency_key: NonEmpty
    fields: tuple[AuditField, ...] = ()


class ActionAttempt(RecoveryModel):
    action_id: NonEmpty
    idempotency_key: NonEmpty
    attempted_at_ms: int = Field(ge=0)
    succeeded: bool
    detail: NonEmpty


class RecoverySnapshot(RecoveryModel):
    schema_version: Literal["sloforge.recovery.execution/v1"] = "sloforge.recovery.execution/v1"
    recovery_id: NonEmpty
    recovery_plan_hash: Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
    state: RecoveryState
    sequence: int = Field(ge=0)
    started_at_ms: int = Field(ge=0)
    state_entered_at_ms: int = Field(ge=0)
    deadline_ms: int = Field(ge=0)
    last_observed_at_ms: int = Field(ge=0)
    cooldown_until_ms: int | None = Field(default=None, ge=0)
    applied_action_ids: tuple[NonEmpty, ...] = ()
    action_attempts: tuple[ActionAttempt, ...] = ()
    processed_idempotency_keys: tuple[NonEmpty, ...] = ()
    audit: tuple[AuditRecord, ...]

    @model_validator(mode="after")
    def validate_state(self) -> Self:
        if self.state_entered_at_ms < self.started_at_ms:
            raise ValueError("state entry cannot precede recovery start")
        if self.last_observed_at_ms < self.started_at_ms:
            raise ValueError("last observation cannot precede recovery start")
        if self.deadline_ms <= self.started_at_ms:
            raise ValueError("recovery deadline must follow recovery start")
        if len(self.applied_action_ids) != len(set(self.applied_action_ids)):
            raise ValueError("applied recovery action IDs must be unique")
        if len(self.processed_idempotency_keys) != len(set(self.processed_idempotency_keys)):
            raise ValueError("processed idempotency keys must be unique")
        return self
