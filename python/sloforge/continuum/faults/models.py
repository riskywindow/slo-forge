"""Typed, deterministic fault ground truth for Continuum protocol exercises."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

NonEmpty = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=512)]


class FaultModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, validate_default=True)


class FaultKind(StrEnum):
    SOURCE_CRASH_BEFORE_INITIAL_SNAPSHOT = "source_crash_before_initial_snapshot"
    SOURCE_CRASH_DURING_PRECOPY = "source_crash_during_precopy"
    SOURCE_CRASH_AFTER_QUIESCE = "source_crash_after_quiesce"
    SOURCE_CRASH_AFTER_COMMIT_INTENT = "source_crash_after_commit_intent"
    DESTINATION_CRASH_DURING_IMPORT = "destination_crash_during_import"
    DESTINATION_CRASH_DURING_VALIDATION = "destination_crash_during_validation"
    DESTINATION_CRASH_AFTER_OWNERSHIP_COMMIT = "destination_crash_after_ownership_commit"
    GATEWAY_CRASH_DURING_SWITCH = "gateway_crash_during_switch"
    COORDINATOR_CRASH = "coordinator_crash"
    DELAYED_COORDINATOR_RESPONSE = "delayed_coordinator_response"
    STALE_SOURCE_WRITER = "stale_source_writer"
    STALE_DESTINATION_EPOCH = "stale_destination_epoch"
    NETWORK_PARTITION = "network_partition"
    TRANSFER_TIMEOUT = "transfer_timeout"
    CHUNK_LOSS = "chunk_loss"
    CHUNK_DUPLICATION = "chunk_duplication"
    CHUNK_CORRUPTION = "chunk_corruption"
    CHECKSUM_MISMATCH = "checksum_mismatch"
    OUT_OF_ORDER_DELTA = "out_of_order_delta"
    DIRTY_LOG_OVERFLOW = "dirty_log_overflow"
    HIGH_DIRTY_RATE = "high_dirty_rate_preventing_convergence"
    DESTINATION_OOM = "destination_oom"
    CONVERSION_KERNEL_FAILURE = "conversion_kernel_failure"
    DESTINATION_INCOMPATIBILITY = "destination_incompatibility"
    PAGE_TABLE_CORRUPTION = "page_table_corruption"
    TOKEN_GAP = "token_gap"
    DUPLICATE_TOKEN = "duplicate_token"
    CLIENT_DISCONNECT = "client_disconnect"
    CANCELLATION_DURING_CUTOVER = "cancellation_during_cutover"
    CLOCK_SKEW = "source_destination_clock_skew"
    WARM_DESTINATION_REGRESSION = "warm_destination_regression"


class FaultComponent(StrEnum):
    SOURCE_RUNTIME = "source_runtime"
    DESTINATION_RUNTIME = "destination_runtime"
    GATEWAY = "gateway"
    COORDINATOR = "coordinator"
    TRANSPORT = "transport"
    CONVERTER = "converter"
    DIRTY_TRACKER = "dirty_tracker"
    CLIENT = "client"
    WARMPATH = "warmpath"


class FaultDefinition(FaultModel):
    schema_version: Literal["1.0.0"] = "1.0.0"
    kind: FaultKind
    ground_truth_label: NonEmpty
    affected_component: FaultComponent
    transaction_phase: NonEmpty
    expected_protocol_response: NonEmpty
    host_wide: bool = False


class FaultActivation(FaultModel):
    schema_version: Literal["1.0.0"] = "1.0.0"
    definition: FaultDefinition
    transaction_id: NonEmpty
    activation_sequence: Annotated[int, Field(ge=0)]
    clear_sequence: Annotated[int, Field(ge=0)]
    observed_protocol_response: NonEmpty
    injected: bool

    @model_validator(mode="after")
    def validate_interval(self) -> Self:
        if self.clear_sequence < self.activation_sequence:
            raise ValueError("fault clear sequence cannot precede activation")
        return self
