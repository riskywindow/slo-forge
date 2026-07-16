"""Typed records and fail-closed errors for advanced Continuum state operations."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator

from sloforge.continuum.adapters import CapturedState, ImportValidation, SessionMetadata
from sloforge.continuum.ir import ExecutionStateCapsule, canonical_hash
from sloforge.continuum.storage import ChunkRef, StoredManifest
from sloforge.continuum.transaction import SessionLease, StateTransactionRecord


class OperationError(RuntimeError):
    """Base error for an advanced operation that failed closed."""

    code = "continuum_operation_error"


class AuthorizationError(OperationError):
    code = "state_authorization_failed"


class AncestryIntegrityError(OperationError):
    code = "checkpoint_ancestry_invalid"


class CapsuleRestoreError(OperationError):
    code = "capsule_restore_failed"


class UnsupportedReferenceABI(CapsuleRestoreError):
    code = "unsupported_reference_abi"


class RecomputeProofError(OperationError):
    code = "recomputation_proof_invalid"


class LazyMigrationRejected(OperationError):
    code = "lazy_migration_rejected"


class OperationModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, validate_default=True)


class AncestryKind(StrEnum):
    ROOT = "root"
    INCREMENTAL = "incremental"
    FORK = "fork"
    CLONE = "clone"
    RECOMPUTATION = "recomputation"


class AncestryProof(OperationModel):
    kind: AncestryKind
    generation: int = Field(ge=0, le=1024)
    parent_capsule_id: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    parent_manifest_id: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    current_manifest_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    digest: str = Field(pattern=r"^[0-9a-f]{64}$")


@dataclass(frozen=True, slots=True)
class CheckpointArtifact:
    """A capsule plus the exact authorized CAS publication needed to resolve it."""

    capsule: ExecutionStateCapsule
    store_manifest: StoredManifest
    chunk_references: tuple[ChunkRef, ...]
    ancestry: AncestryProof
    changed_segment_ids: tuple[str, ...]
    parent: CheckpointArtifact | None = field(default=None, repr=False)


@dataclass(frozen=True, slots=True)
class PauseCheckpointResult:
    before_pause: SessionMetadata
    paused: SessionMetadata
    checkpoint: CheckpointArtifact


@dataclass(frozen=True, slots=True)
class ResumeResult:
    captured: CapturedState
    validation: ImportValidation
    transaction: StateTransactionRecord
    lease: SessionLease
    active: SessionMetadata
    source_owner_epoch: int
    destination_owner_epoch: int
    source_fenced: bool


@dataclass(frozen=True, slots=True)
class ForkResult:
    parent_capsule_id: str
    branches: tuple[CheckpointArtifact, ...]
    shared_immutable_digests: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CloneResult:
    parent_capsule_id: str
    clone: CheckpointArtifact
    copied_bytes: int


class RecomputeEvidence(OperationModel):
    source_capsule_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    destination_model_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    recomputed_components: tuple[str, ...]
    dependency_edges: tuple[str, ...]
    token_count: int = Field(ge=0)
    continuation_horizon: int = Field(ge=1, le=64)
    seed: int = Field(ge=0, le=2**64 - 1)
    first_run_tokens: tuple[int, ...]
    independent_run_tokens: tuple[int, ...]
    verified: bool
    evidence_digest: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def verify_evidence_seal(self) -> RecomputeEvidence:
        payload = {
            "schema": "sloforge.continuum.recomputation-evidence/v1",
            "source_capsule_id": self.source_capsule_id,
            "destination_model_hash": self.destination_model_hash,
            "recomputed_components": self.recomputed_components,
            "dependency_edges": self.dependency_edges,
            "token_count": self.token_count,
            "continuation_horizon": self.continuation_horizon,
            "first_run_tokens": self.first_run_tokens,
            "independent_run_tokens": self.independent_run_tokens,
            "seed": self.seed,
        }
        if canonical_hash(payload) != self.evidence_digest:
            raise ValueError("recomputation evidence digest is invalid")
        if not self.verified or self.first_run_tokens != self.independent_run_tokens:
            raise ValueError("recomputation continuation verification did not pass")
        if len(self.first_run_tokens) != self.continuation_horizon:
            raise ValueError("recomputation evidence horizon does not match its token trace")
        return self


@dataclass(frozen=True, slots=True)
class RecomputeResult:
    captured: CapturedState
    evidence: RecomputeEvidence


class LazyReason(StrEnum):
    SLIDING_WINDOW = "sliding_window"
    LAYER_SEQUENTIAL = "layer_sequential"
    COLD = "cold"
    RECOMPUTABLE = "recomputable"


class LazySegmentDeclaration(OperationModel):
    segment_id: str = Field(min_length=1)
    reason: LazyReason
    available_before_first_use: bool
    explicit_stall_on_miss: bool
    failure_behavior: str = Field(min_length=1)


class LazyMigrationAssessment(OperationModel):
    legal: bool
    legal_segment_ids: tuple[str, ...]
    rejected_segment_ids: tuple[str, ...]
    obligations: tuple[str, ...]
    reasons: tuple[str, ...]
