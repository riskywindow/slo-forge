"""Strict bridge artifacts for Helix, Autopsy, and ForgeCI."""

from __future__ import annotations

import hashlib
import json
from enum import StrEnum
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from sloforge.autopsy import AutopsyRun
from sloforge.forgeci import BenchmarkMatrix, ComparisonClassification
from sloforge.helix.ir import BranchWorkloadTrace
from sloforge.helix.transactions import (
    ArtifactReference,
    LearningState,
    TransactionEvent,
)

Identifier = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}$")]
Digest = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
NonEmpty = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=4096)]


def canonical_digest(value: object) -> str:
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class IntegrationModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, validate_default=True)


class BranchOperationKind(StrEnum):
    STATE = "state"
    FORK = "fork"
    TRANSFER = "transfer"
    CHECKSUM = "checksum"
    STORAGE = "storage"
    DIVERGENCE = "divergence"


class EvidenceClaimScope(StrEnum):
    DECLARED_ARTIFACT = "declared_artifact"
    DERIVED_CONTENT_IDENTITY = "derived_content_identity"


class BranchOperationEvidence(IntegrationModel):
    operation_id: Digest
    ordinal: int = Field(ge=0, le=1_000_000)
    kind: BranchOperationKind
    trajectory_id: Identifier | None = None
    source_artifact_id: NonEmpty
    target_artifact_id: NonEmpty
    logical_offset_ns: int = Field(ge=0, le=2**63 - 1)
    checksum_sha256: Digest
    byte_count: int | None = Field(default=None, ge=0, le=2**64 - 1)
    evidence_uri: NonEmpty
    evidence_sha256: Digest
    claim_scope: EvidenceClaimScope
    detail: NonEmpty

    @model_validator(mode="after")
    def validate_seal(self) -> Self:
        identity = self.model_dump(mode="json", exclude={"operation_id"})
        if canonical_digest(identity) != self.operation_id:
            raise ValueError("branch operation identifier is invalid")
        return self


class BranchTraceExport(IntegrationModel):
    schema_version: Literal["sloforge.helix.branch-trace-export/v1"] = (
        "sloforge.helix.branch-trace-export/v1"
    )
    export_id: Digest
    workload_trace: BranchWorkloadTrace
    operations: Annotated[
        tuple[BranchOperationEvidence, ...], Field(min_length=1, max_length=100_000)
    ]
    autopsy_run: AutopsyRun
    source_branch_group_sha256: Digest
    source_artifact_hashes: Annotated[tuple[Digest, ...], Field(min_length=1, max_length=100_000)]
    assumptions: Annotated[tuple[NonEmpty, ...], Field(min_length=1, max_length=32)]

    @model_validator(mode="after")
    def validate_export(self) -> Self:
        if tuple(item.ordinal for item in self.operations) != tuple(range(len(self.operations))):
            raise ValueError("branch operations must have dense ordinals")
        if {item.kind for item in self.operations} != set(BranchOperationKind):
            raise ValueError("branch trace must cover every structured operation kind")
        operation_ids = tuple(item.operation_id for item in self.operations)
        if len(operation_ids) != len(set(operation_ids)):
            raise ValueError("branch operation identifiers must be unique")
        if tuple(event.event_id for event in self.autopsy_run.events) != operation_ids:
            raise ValueError("Autopsy events must preserve branch operation ordering")
        if self.autopsy_run.workload_fingerprint != canonical_digest(self.workload_trace):
            raise ValueError("Autopsy workload fingerprint does not match the Helix trace")
        if tuple(sorted(set(self.source_artifact_hashes))) != self.source_artifact_hashes:
            raise ValueError("source artifact hashes must be sorted and unique")
        identity = self.model_dump(mode="json", exclude={"export_id"})
        if canonical_digest(identity) != self.export_id:
            raise ValueError("branch trace export identifier is invalid")
        return self


class ForgeCIRegressionArtifact(IntegrationModel):
    schema_version: Literal["sloforge.helix.forgeci-regression/v1"] = (
        "sloforge.helix.forgeci-regression/v1"
    )
    artifact_id: Digest
    transaction_id: Identifier
    failed_state: LearningState
    transaction_evidence_sha256: Digest
    seed: int = Field(ge=0, le=2**64 - 1)
    failure_event: TransactionEvent
    classification: Literal[ComparisonClassification.FAILED] = ComparisonClassification.FAILED
    matrix: BenchmarkMatrix
    matrix_sha256: Digest
    raw_evidence_artifacts: Annotated[
        tuple[ArtifactReference, ...], Field(min_length=1, max_length=10_000)
    ]
    raw_evidence_set_sha256: Digest
    raw_evidence_unchanged: Literal[True] = True
    limitations: Annotated[tuple[NonEmpty, ...], Field(min_length=1, max_length=32)]

    @model_validator(mode="after")
    def validate_artifact(self) -> Self:
        if not self.failed_state.terminal or self.failed_state is LearningState.COMPLETED:
            raise ValueError("ForgeCI bridge requires a failed terminal Helix transaction")
        if self.failure_event.state_after is not self.failed_state:
            raise ValueError("failure event does not match the failed transaction state")
        if self.matrix_sha256 != canonical_digest(self.matrix):
            raise ValueError("ForgeCI matrix digest is invalid")
        artifact_ids = tuple(item.artifact_id for item in self.raw_evidence_artifacts)
        if len(artifact_ids) != len(set(artifact_ids)):
            raise ValueError("raw evidence artifact identifiers must be unique")
        expected_set = canonical_digest(
            [item.model_dump(mode="json") for item in self.raw_evidence_artifacts]
        )
        if self.raw_evidence_set_sha256 != expected_set:
            raise ValueError("raw evidence artifact set digest is invalid")
        identity = self.model_dump(mode="json", exclude={"artifact_id"})
        if canonical_digest(identity) != self.artifact_id:
            raise ValueError("ForgeCI regression artifact identifier is invalid")
        return self


__all__ = [
    "BranchOperationEvidence",
    "BranchOperationKind",
    "BranchTraceExport",
    "EvidenceClaimScope",
    "ForgeCIRegressionArtifact",
    "canonical_digest",
]
