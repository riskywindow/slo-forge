"""Typed records for a crash-recoverable Helix capture barrier."""

from __future__ import annotations

import hashlib
import json
from enum import StrEnum
from typing import Annotated, Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

NonEmpty = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=512)]
Identifier = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:@-]{0,255}$")]
Digest = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]


def canonical_digest(value: object) -> str:
    """Hash a bounded JSON-compatible value with the repository's canonical convention."""

    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class CaptureModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, validate_default=True)


class CapturePhase(StrEnum):
    """Durable phases. A BranchPoint is visible only in ``COMPLETED``."""

    CAPTURE_PROPOSED = "CAPTURE_PROPOSED"
    QUIESCING = "QUIESCING"
    QUIESCED = "QUIESCED"
    MODEL_CAPTURED = "MODEL_CAPTURED"
    ENVIRONMENT_CAPTURED = "ENVIRONMENT_CAPTURED"
    EFFECTS_CAPTURED = "EFFECTS_CAPTURED"
    VALIDATING = "VALIDATING"
    ABORTING = "ABORTING"
    ABORTED = "ABORTED"
    FAILED = "FAILED"
    COMPLETED = "COMPLETED"

    @property
    def terminal(self) -> bool:
        return self in {CapturePhase.ABORTED, CapturePhase.FAILED, CapturePhase.COMPLETED}


class CaptureBoundary(CaptureModel):
    """Independent source boundaries; equality between the four domains is not implied."""

    action_watermark: int = Field(ge=-1)
    model_token_watermark: int = Field(ge=-1)
    environment_event_watermark: int = Field(ge=-1)
    effect_watermark: int = Field(ge=-1)


class ArtifactWatermark(CaptureModel):
    artifact_id: Identifier
    watermark: int = Field(ge=-1)
    digest: Digest


class CoordinatedCaptureRequest(CaptureModel):
    schema_version: Literal["1.0.0"] = "1.0.0"
    capture_id: Identifier
    session_id: Identifier
    source_trajectory_id: Identifier
    policy_epoch_id: Identifier
    boundary: CaptureBoundary
    seed: int = Field(ge=0, le=2**64 - 1)
    max_quiescence_polls: int = Field(default=16, ge=2, le=4096)
    published_at_ms: int = Field(ge=0)
    capture_timestamp: NonEmpty
    git_commit: Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{40}$")]
    continuum_version: NonEmpty
    created_at: NonEmpty
    reason: NonEmpty = "counterfactual branch capture"


class CoordinatedBranchPoint(CaptureModel):
    """Atomic publication envelope binding Helix and Continuum capture artifacts."""

    schema_version: Literal["sloforge.helix.coordinated-branch-point/v1"] = (
        "sloforge.helix.coordinated-branch-point/v1"
    )
    branch_point_id: Digest
    capture_id: Identifier
    session_id: Identifier
    source_trajectory_id: Identifier
    policy_epoch_id: Identifier
    continuum_capsule_id: Digest
    environment: ArtifactWatermark
    effects: ArtifactWatermark
    boundary: CaptureBoundary
    seed: int = Field(ge=0, le=2**64 - 1)
    created_at: NonEmpty
    reason: NonEmpty
    consistency_digest: Digest

    @model_validator(mode="after")
    def verify_seals(self) -> Self:
        consistency = {
            "capture_id": self.capture_id,
            "continuum_capsule_id": self.continuum_capsule_id,
            "environment": self.environment.model_dump(mode="json"),
            "effects": self.effects.model_dump(mode="json"),
            "boundary": self.boundary.model_dump(mode="json"),
        }
        if canonical_digest(consistency) != self.consistency_digest:
            raise ValueError("coordinated capture consistency digest is invalid")
        identity = self.model_dump(mode="json", exclude={"branch_point_id"})
        if canonical_digest(identity) != self.branch_point_id:
            raise ValueError("coordinated BranchPoint identifier is invalid")
        return self


class CaptureJournalEntry(CaptureModel):
    capture_id: Identifier
    sequence: int = Field(ge=0)
    phase: CapturePhase
    attempt: int = Field(ge=1)
    detail: NonEmpty
    detail_digest: Digest

    @model_validator(mode="after")
    def verify_detail(self) -> Self:
        if canonical_digest({"detail": self.detail}) != self.detail_digest:
            raise ValueError("capture journal detail digest is invalid")
        return self


class CaptureStatus(CaptureModel):
    request: CoordinatedCaptureRequest
    phase: CapturePhase
    attempt: int = Field(ge=1)
    error_code: str | None = None
    error_message: str | None = None
    branch_point: CoordinatedBranchPoint | None = None

    @model_validator(mode="after")
    def validate_publication(self) -> Self:
        if (self.branch_point is not None) != (self.phase is CapturePhase.COMPLETED):
            raise ValueError("a BranchPoint may be published only by a completed capture")
        if self.phase is CapturePhase.FAILED and self.error_code is None:
            raise ValueError("failed capture requires an error code")
        return self


def make_branch_point(
    request: CoordinatedCaptureRequest,
    *,
    continuum_capsule_id: str,
    environment: ArtifactWatermark,
    effects: ArtifactWatermark,
) -> CoordinatedBranchPoint:
    consistency = {
        "capture_id": request.capture_id,
        "continuum_capsule_id": continuum_capsule_id,
        "environment": environment.model_dump(mode="json"),
        "effects": effects.model_dump(mode="json"),
        "boundary": request.boundary.model_dump(mode="json"),
    }
    draft: dict[str, Any] = {
        "schema_version": "sloforge.helix.coordinated-branch-point/v1",
        "branch_point_id": "0" * 64,
        "capture_id": request.capture_id,
        "session_id": request.session_id,
        "source_trajectory_id": request.source_trajectory_id,
        "policy_epoch_id": request.policy_epoch_id,
        "continuum_capsule_id": continuum_capsule_id,
        "environment": environment.model_dump(mode="json"),
        "effects": effects.model_dump(mode="json"),
        "boundary": request.boundary.model_dump(mode="json"),
        "seed": request.seed,
        "created_at": request.created_at,
        "reason": request.reason,
        "consistency_digest": canonical_digest(consistency),
    }
    draft["branch_point_id"] = canonical_digest(
        {key: value for key, value in draft.items() if key != "branch_point_id"}
    )
    return CoordinatedBranchPoint.model_validate(draft, strict=True)
