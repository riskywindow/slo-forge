"""Strict wire records for Continuum's local transaction authority."""

from __future__ import annotations

import re
from enum import StrEnum
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

NonEmpty = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=256)]
Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
_TOKEN_ID_LIMIT = 2**31 - 1


class TransactionModel(BaseModel):
    """Closed and immutable model used for persisted coordinator records."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, validate_default=True)


class CutoverPhase(StrEnum):
    PROPOSED = "PROPOSED"
    COMPATIBILITY_VALIDATED = "COMPATIBILITY_VALIDATED"
    DESTINATION_PREPARING = "DESTINATION_PREPARING"
    PRECOPYING = "PRECOPYING"
    DELTA_SYNCING = "DELTA_SYNCING"
    CUTOVER_REQUESTED = "CUTOVER_REQUESTED"
    SOURCE_QUIESCING = "SOURCE_QUIESCING"
    SOURCE_FROZEN = "SOURCE_FROZEN"
    FINAL_DELTA_TRANSFERRING = "FINAL_DELTA_TRANSFERRING"
    DESTINATION_IMPORTING = "DESTINATION_IMPORTING"
    DESTINATION_VALIDATING = "DESTINATION_VALIDATING"
    COMMIT_INTENT_RECORDED = "COMMIT_INTENT_RECORDED"
    OWNERSHIP_COMMITTED = "OWNERSHIP_COMMITTED"
    GATEWAY_SWITCHING = "GATEWAY_SWITCHING"
    DESTINATION_ACTIVE = "DESTINATION_ACTIVE"
    SOURCE_DRAINING = "SOURCE_DRAINING"
    COMPLETED = "COMPLETED"
    REJECTED = "REJECTED"
    ABORTING = "ABORTING"
    ROLLED_BACK = "ROLLED_BACK"
    FAILED_BEFORE_COMMIT = "FAILED_BEFORE_COMMIT"
    FAILED_AFTER_COMMIT = "FAILED_AFTER_COMMIT"
    DESTINATION_LOST = "DESTINATION_LOST"
    SOURCE_LOST = "SOURCE_LOST"
    COORDINATOR_UNAVAILABLE = "COORDINATOR_UNAVAILABLE"
    OPERATOR_REQUIRED = "OPERATOR_REQUIRED"


TERMINAL_PHASES = frozenset(
    {
        CutoverPhase.COMPLETED,
        CutoverPhase.REJECTED,
        CutoverPhase.ROLLED_BACK,
        CutoverPhase.FAILED_BEFORE_COMMIT,
        CutoverPhase.FAILED_AFTER_COMMIT,
        CutoverPhase.OPERATOR_REQUIRED,
    }
)


class SessionLease(TransactionModel):
    session_id: NonEmpty
    owner_runtime: NonEmpty
    owner_epoch: Annotated[int, Field(ge=1)]
    fencing_token: Annotated[int, Field(ge=1)]
    expiration_ms: Annotated[int, Field(ge=0)]
    coordinator_version: Annotated[int, Field(ge=1)]
    last_committed_state_version: Annotated[int, Field(ge=0)] = 0
    last_committed_token_index: int = -1

    @model_validator(mode="after")
    def validate_watermark(self) -> Self:
        if self.last_committed_token_index < -1:
            raise ValueError("last committed token index must be at least -1")
        return self


class StateTransactionRecord(TransactionModel):
    transaction_id: Sha256
    session_id: NonEmpty
    source_owner: NonEmpty
    destination_candidate: NonEmpty
    source_epoch: Annotated[int, Field(ge=1)]
    proposed_destination_epoch: Annotated[int, Field(ge=2)]
    migration_plan_hash: Sha256
    phase: CutoverPhase = CutoverPhase.PROPOSED
    commit_watermark: int = -1
    rollback_watermark: int = -1
    state_hashes: tuple[Sha256, ...] = ()
    timeout_at_ms: Annotated[int, Field(ge=1)]
    failure_reason: str | None = None
    version: Annotated[int, Field(ge=1)] = 1

    @model_validator(mode="after")
    def validate_transaction(self) -> Self:
        if self.proposed_destination_epoch != self.source_epoch + 1:
            raise ValueError("destination epoch must be exactly one greater than source epoch")
        if self.commit_watermark < -1 or self.rollback_watermark < -1:
            raise ValueError("transaction watermarks must be at least -1")
        if len(self.state_hashes) > 4096:
            raise ValueError("transaction state hash list exceeds 4096 entries")
        if (
            self.phase
            not in {
                CutoverPhase.REJECTED,
                CutoverPhase.FAILED_BEFORE_COMMIT,
                CutoverPhase.FAILED_AFTER_COMMIT,
                CutoverPhase.OPERATOR_REQUIRED,
            }
            and self.failure_reason is not None
        ):
            raise ValueError("failure reason is only legal on a failure terminal phase")
        return self


class RecoveryEvidence(TransactionModel):
    """Version-scoped caller observations used for crash-recovery classification."""

    schema_version: Literal["1.0.0"] = "1.0.0"
    evidence_id: Sha256
    transaction_id: Sha256
    observed_owner_runtime: NonEmpty
    observed_owner_epoch: Annotated[int, Field(ge=1)]
    observed_fencing_token: Annotated[int, Field(ge=1)]
    source_available: bool
    source_fenced: bool
    source_resumable: bool
    source_commit_watermark: int
    destination_available: bool
    destination_validated: bool
    destination_aborted: bool
    gateway_owner_epoch: Annotated[int, Field(ge=1)]
    gateway_commit_watermark: int
    cleanup_completed: bool
    reason: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=512)]

    @model_validator(mode="after")
    def validate_observations(self) -> Self:
        if self.gateway_commit_watermark < -1:
            raise ValueError("gateway recovery watermark must be at least -1")
        if self.source_commit_watermark < -1:
            raise ValueError("source recovery watermark must be at least -1")
        if self.source_resumable and not self.source_available:
            raise ValueError("an unavailable source cannot be declared resumable")
        if self.destination_validated and not self.destination_available:
            raise ValueError("an unavailable destination cannot be declared validated")
        if self.destination_aborted and self.destination_available:
            raise ValueError("an aborted destination cannot remain available")
        return self


class RecoveryResult(TransactionModel):
    """Durable outcome of classifying one interrupted transaction."""

    transaction_id: Sha256
    evidence_id: Sha256
    ownership_committed: bool
    prior_phase: CutoverPhase
    terminal_phase: CutoverPhase
    lease_owner_runtime: NonEmpty
    lease_owner_epoch: Annotated[int, Field(ge=1)]
    recovered_at_ms: Annotated[int, Field(ge=0)]


class JournalEntry(TransactionModel):
    transaction_id: Sha256
    sequence: Annotated[int, Field(ge=0)]
    event_id: NonEmpty
    from_phase: CutoverPhase
    to_phase: CutoverPhase
    at_ms: Annotated[int, Field(ge=0)]
    payload_hash: Sha256


class TokenEvent(TransactionModel):
    session_id: NonEmpty
    owner_epoch: Annotated[int, Field(ge=1)]
    token_index: Annotated[int, Field(ge=0)]
    token_id: Annotated[int, Field(ge=0, le=_TOKEN_ID_LIMIT)]
    token_text: Annotated[str, StringConstraints(max_length=65536)] | None = None
    state_commit_version: Annotated[int, Field(ge=0)]
    transaction_id: Sha256 | None = None
    terminal: bool = False


class TokenAcceptance(TransactionModel):
    disposition: Literal["accepted", "duplicate"]
    session_id: NonEmpty
    owner_epoch: Annotated[int, Field(ge=1)]
    token_index: Annotated[int, Field(ge=0)]
    gateway_commit_watermark: Annotated[int, Field(ge=0)]
    delivery_semantics: Literal[
        "gateway_exactly_once",
        "client_exactly_once_acknowledged",
        "network_at_least_once_sequence_deduplicated",
    ]


def validate_transaction_id(value: str) -> str:
    """Validate IDs before using them in SQL lookups outside Pydantic parsing."""

    if re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError("transaction identifier must be lowercase SHA-256")
    return value
