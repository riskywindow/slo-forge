"""Typed champion/challenger evolution contracts and persisted state."""

from __future__ import annotations

import re
from enum import StrEnum
from typing import Annotated, Final, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

from ..capsule.models import CapsuleValidationReport

EVOLUTION_SCHEMA_VERSION: Final = "sloforge.genesis.evolution/v1"
PERSISTENCE_SCHEMA_VERSION: Final = "sloforge.genesis.evolution.persistence/v1"
_DIGEST = re.compile(r"^[0-9a-f]{64}$")

NonEmpty = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
Identifier = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True, min_length=1, pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$"
    ),
]


class EvolutionModel(BaseModel):
    """Strict immutable values crossing the evolution-controller boundary."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        validate_default=True,
        allow_inf_nan=False,
    )


class ExecutionTarget(StrEnum):
    SIMULATED = "simulated"
    LOCAL = "local"
    EXTERNAL = "external"


class EvolutionTrigger(StrEnum):
    WORKLOAD_DRIFT = "workload_drift"
    MODEL_UPDATE = "model_update"
    HARDWARE_CHANGE = "hardware_change"
    DEPENDENCY_UPDATE = "dependency_update"
    SLO_CHANGE = "slo_change"
    COST_CHANGE = "cost_change"
    AUTOPSY_BOTTLENECK_CHANGE = "autopsy_bottleneck_change"
    PERFORMANCE_REGRESSION = "performance_regression"
    FABRIC_DEGRADATION = "fabric_degradation"


class EvolutionPhase(StrEnum):
    IDLE = "idle"
    EVOLVING = "evolving"
    CHALLENGER_READY = "challenger_ready"
    SHADOWING = "shadowing"
    SHADOW_VALIDATED = "shadow_validated"
    CANARYING = "canarying"
    READY_TO_PROMOTE = "ready_to_promote"
    PROMOTED = "promoted"
    ROLLED_BACK = "rolled_back"


class ChallengerStatus(StrEnum):
    CAPSULE_VALIDATED = "capsule_validated"
    CAPSULE_REJECTED = "capsule_rejected"
    SHADOWING = "shadowing"
    SHADOW_VALIDATED = "shadow_validated"
    SHADOW_REJECTED = "shadow_rejected"
    CANARYING = "canarying"
    CANARY_VALIDATED = "canary_validated"
    CANARY_REJECTED = "canary_rejected"
    PROMOTED = "promoted"
    ROLLED_BACK = "rolled_back"
    SUPERSEDED = "superseded"


class TransitionCategory(StrEnum):
    POLICY_ONLY_HOT_SWAP = "policy_only_hot_swap"
    REQUEST_BOUNDARY_SWAP = "request_boundary_swap"
    WORKER_RESTART = "worker_restart"
    NEW_REPLICA = "new_replica"
    STATE_COMPATIBLE_MIGRATION = "state_compatible_migration"
    STATE_CONVERSION_MIGRATION = "state_conversion_migration"
    FULL_DEPLOYMENT_REBUILD = "full_deployment_rebuild"
    OPERATOR_REQUIRED = "operator_required"


class GateStage(StrEnum):
    SHADOW = "shadow"
    CANARY = "canary"
    POST_PROMOTION = "post_promotion"


class IsolationMode(StrEnum):
    SANDBOX = "sandbox"
    LOCAL_PROCESS = "local_process"
    REPLAY_ONLY = "replay_only"


class CapsuleReference(EvolutionModel):
    capsule_id: Identifier
    capsule_digest: str
    genome_hash: str
    path: NonEmpty

    @field_validator("capsule_digest", "genome_hash")
    @classmethod
    def validate_digest(cls, value: str) -> str:
        if _DIGEST.fullmatch(value) is None:
            raise ValueError("capsule and genome hashes must be lowercase sha256 digests")
        return value


class IsolationContract(EvolutionModel):
    mode: IsolationMode
    writable_artifact_root: NonEmpty
    network_access: Literal[False] = False
    cloud_credentials_exposed: Literal[False] = False
    host_device_access: Literal[False] = False
    live_traffic_enabled: Literal[False] = False


class ChallengerSpec(EvolutionModel):
    candidate_id: Identifier
    capsule: CapsuleReference
    transition_category: TransitionCategory
    isolation: IsolationContract
    active_stream_compatible: bool
    state_conversion_verified: bool = False

    @model_validator(mode="after")
    def validate_transition_contract(self) -> Self:
        if (
            self.transition_category is TransitionCategory.POLICY_ONLY_HOT_SWAP
            and not self.active_stream_compatible
        ):
            raise ValueError("policy-only hot swaps must preserve active streams")
        if (
            self.transition_category is TransitionCategory.STATE_CONVERSION_MIGRATION
            and not self.state_conversion_verified
        ):
            raise ValueError("state-conversion migration requires verified conversion evidence")
        return self


class GateObservation(EvolutionModel):
    event_id: Identifier
    stage: GateStage
    observed_at_ms: int = Field(ge=0)
    deterministic_seed: int = Field(ge=0)
    sample_count: int = Field(ge=0)
    error_rate: float = Field(ge=0.0, le=1.0)
    p95_ttft_ratio: float = Field(gt=0.0)
    p99_tpot_ratio: float = Field(gt=0.0)
    quality_regression: float = Field(ge=0.0)
    interrupted_streams: int = Field(ge=0)


class TriggerObservation(EvolutionModel):
    event_id: Identifier
    observed_at_ms: int = Field(ge=0)
    workload_js_divergence: float = Field(default=0.0, ge=0.0)
    fabric_health_ratio: float = Field(default=1.0, ge=0.0, le=1.0)
    performance_regression_ratio: float = Field(default=0.0, ge=0.0)
    model_updated: bool = False
    hardware_changed: bool = False
    dependency_updated: bool = False
    slo_changed: bool = False
    cost_changed: bool = False
    autopsy_bottleneck_changed: bool = False
    detail: NonEmpty = "no material change"


class EvolutionConfig(EvolutionModel):
    execution_target: ExecutionTarget = ExecutionTarget.SIMULATED
    live_traffic: bool = False
    live_promotion_authorized: bool = False
    workload_drift_threshold: float = Field(default=0.20, gt=0.0)
    fabric_health_floor: float = Field(default=0.85, gt=0.0, le=1.0)
    performance_regression_threshold: float = Field(default=0.05, gt=0.0)
    minimum_shadow_samples: int = Field(default=20, gt=0)
    minimum_canary_samples: int = Field(default=40, gt=0)
    maximum_error_rate: float = Field(default=0.01, ge=0.0, le=1.0)
    maximum_p95_ttft_ratio: float = Field(default=1.05, gt=0.0)
    maximum_p99_tpot_ratio: float = Field(default=1.05, gt=0.0)
    maximum_quality_regression: float = Field(default=0.0, ge=0.0)
    canary_fraction: float = Field(default=0.05, gt=0.0, le=1.0)
    maximum_challengers: int = Field(default=64, ge=1, le=4_096)
    maximum_active_streams: int = Field(default=8_192, ge=1, le=1_000_000)
    maximum_audit_records: int = Field(default=2_048, ge=32, le=65_536)
    maximum_processed_events: int = Field(default=2_048, ge=32, le=65_536)

    @model_validator(mode="after")
    def validate_external_mutation(self) -> Self:
        if self.execution_target is ExecutionTarget.EXTERNAL and not self.live_traffic:
            raise ValueError("external evolution requires live_traffic=true")
        if self.live_promotion_authorized and not self.live_traffic:
            raise ValueError("live promotion authorization is valid only for live traffic")
        return self


class StreamLease(EvolutionModel):
    stream_id: Identifier
    capsule_id: Identifier
    opened_sequence: int = Field(ge=0)
    externally_visible_output: bool = False


class ChallengerRecord(EvolutionModel):
    spec: ChallengerSpec
    status: ChallengerStatus
    capsule_validation: CapsuleValidationReport
    shadow_observation: GateObservation | None = None
    canary_observation: GateObservation | None = None
    rejection_reason: str | None = None


class EvolutionAuditRecord(EvolutionModel):
    sequence: int = Field(ge=0)
    event_id: Identifier
    observed_at_ms: int = Field(ge=0)
    action: NonEmpty
    phase_before: EvolutionPhase
    phase_after: EvolutionPhase
    reason: NonEmpty
    candidate_id: str | None = None


class EvolutionSnapshot(EvolutionModel):
    schema_version: Literal["sloforge.genesis.evolution/v1"] = EVOLUTION_SCHEMA_VERSION
    deployment_id: Identifier
    seed: int = Field(ge=0)
    sequence: int = Field(ge=0)
    phase: EvolutionPhase
    champion: CapsuleReference
    previous_champion: CapsuleReference | None = None
    active_trigger: EvolutionTrigger | None = None
    selected_candidate_id: str | None = None
    challengers: tuple[ChallengerRecord, ...] = ()
    active_streams: tuple[StreamLease, ...] = ()
    retained_capsule_ids: tuple[Identifier, ...] = ()
    processed_event_ids: tuple[Identifier, ...] = ()
    audit: tuple[EvolutionAuditRecord, ...]
    last_observed_at_ms: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_snapshot(self) -> Self:
        candidate_ids = [item.spec.candidate_id for item in self.challengers]
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("challenger identifiers must be unique")
        stream_ids = [item.stream_id for item in self.active_streams]
        if len(stream_ids) != len(set(stream_ids)):
            raise ValueError("active stream identifiers must be unique")
        if len(self.processed_event_ids) != len(set(self.processed_event_ids)):
            raise ValueError("processed event identifiers must be unique")
        if len(self.retained_capsule_ids) != len(set(self.retained_capsule_ids)):
            raise ValueError("retained capsule identifiers must be unique")
        if (
            self.selected_candidate_id is not None
            and self.selected_candidate_id not in candidate_ids
        ):
            raise ValueError("selected candidate must exist in the challenger population")
        known_capsules = {
            self.champion.capsule_id,
            *(item.spec.capsule.capsule_id for item in self.challengers),
            *self.retained_capsule_ids,
        }
        if any(stream.capsule_id not in known_capsules for stream in self.active_streams):
            raise ValueError("active streams must reference a known retained runtime")
        return self


class PersistedEvolutionState(EvolutionModel):
    schema_version: Literal["sloforge.genesis.evolution.persistence/v1"] = (
        PERSISTENCE_SCHEMA_VERSION
    )
    payload_sha256: str
    payload: EvolutionSnapshot

    @field_validator("payload_sha256")
    @classmethod
    def validate_payload_digest(cls, value: str) -> str:
        if _DIGEST.fullmatch(value) is None:
            raise ValueError("persisted payload digest must be lowercase sha256")
        return value
