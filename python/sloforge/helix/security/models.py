"""Strict wire contracts for Helix security preflight and audit evidence.

The security package deliberately accepts only summaries and cryptographic digests at
authorization boundaries.  Raw production payloads, secrets, hidden expected values, and
credentials are not valid fields in these models.
"""

from __future__ import annotations

import hashlib
import json
import math
from enum import StrEnum
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator
from pydantic_core import to_jsonable_python

Digest = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
Identifier = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}$")]
NonEmpty = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=2048)]


def canonical_digest(value: object) -> str:
    """Return a deterministic SHA-256 digest for a bounded JSON-compatible value."""

    value = to_jsonable_python(value)
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class SecurityModel(BaseModel):
    """Fail-closed base for values which cross a Helix security boundary."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        validate_default=True,
        allow_inf_nan=False,
    )


class Severity(StrEnum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class SecurityViolation(SecurityModel):
    code: Annotated[str, StringConstraints(pattern=r"^[A-Z][A-Z0-9_]{2,63}$")]
    severity: Severity
    message: NonEmpty
    evidence_sha256: Digest | None = None


class SecurityReport(SecurityModel):
    schema_version: Literal["sloforge.helix.security-report/v1"] = (
        "sloforge.helix.security-report/v1"
    )
    subject_id: Identifier
    checked_at_ms: Annotated[int, Field(ge=0)]
    violations: Annotated[tuple[SecurityViolation, ...], Field(max_length=256)] = ()
    report_digest: Digest

    @property
    def passed(self) -> bool:
        return not self.violations

    @model_validator(mode="after")
    def validate_report(self) -> Self:
        codes = tuple(item.code for item in self.violations)
        if codes != tuple(sorted(codes)) or len(codes) != len(set(codes)):
            raise ValueError("security violations must be unique and sorted by code")
        expected = canonical_digest(self.model_dump(mode="json", exclude={"report_digest"}))
        if expected != self.report_digest:
            raise ValueError("security report digest is invalid")
        return self


class ProductionCaptureGrant(SecurityModel):
    schema_version: Literal["sloforge.helix.production-capture-grant/v1"] = (
        "sloforge.helix.production-capture-grant/v1"
    )
    grant_id: Identifier
    tenant_id: Identifier
    request_id: Identifier
    approver_id: Identifier
    issued_at_ms: Annotated[int, Field(ge=0)]
    expires_at_ms: Annotated[int, Field(ge=0)]
    approval_artifact_sha256: Digest
    grant_digest: Digest


class CaptureAccessContext(SecurityModel):
    request_id: Identifier
    actor_tenant_id: Identifier
    resource_tenant_id: Identifier
    production: bool = False
    explicit_production_opt_in: bool = False
    production_grant: ProductionCaptureGrant | None = None
    reuse_source_tenant_id: Identifier | None = None
    cross_tenant_reuse_enabled: bool = False


class RedactionEvidence(SecurityModel):
    schema_version: Literal["sloforge.helix.redaction-evidence/v1"] = (
        "sloforge.helix.redaction-evidence/v1"
    )
    source_sha256: Digest
    sanitized_sha256: Digest
    scanner_id: Identifier
    scanner_version: Identifier
    detected_categories: Annotated[tuple[Identifier, ...], Field(max_length=32)]
    detected_count: Annotated[int, Field(ge=0, le=1_000_000)]
    residual_count: Annotated[int, Field(ge=0, le=1_000_000)]
    replacement_count: Annotated[int, Field(ge=0, le=1_000_000)]
    evidence_digest: Digest


class RetentionPolicy(SecurityModel):
    policy_id: Identifier
    tenant_id: Identifier
    maximum_retention_ms: Annotated[int, Field(gt=0, le=10 * 365 * 24 * 60 * 60 * 1000)]
    deletion_grace_ms: Annotated[int, Field(ge=0, le=90 * 24 * 60 * 60 * 1000)]
    legal_hold_maximum_ms: Annotated[int, Field(gt=0, le=365 * 24 * 60 * 60 * 1000)]
    require_deletion_receipt: bool = True
    cross_tenant_reuse_enabled: bool = False


class DeletionReceipt(SecurityModel):
    schema_version: Literal["sloforge.helix.deletion-receipt/v1"] = (
        "sloforge.helix.deletion-receipt/v1"
    )
    artifact_id: Identifier
    tenant_id: Identifier
    requested_at_ms: Annotated[int, Field(ge=0)]
    completed_at_ms: Annotated[int, Field(ge=0)]
    tombstone_sha256: Digest
    replicas_deleted: Annotated[int, Field(ge=1, le=1_000_000)]
    receipt_digest: Digest


class ArtifactLifecycle(SecurityModel):
    artifact_id: Identifier
    tenant_id: Identifier
    created_at_ms: Annotated[int, Field(ge=0)]
    expires_at_ms: Annotated[int, Field(ge=0)]
    deletion_requested_at_ms: Annotated[int, Field(ge=0)] | None = None
    deleted_at_ms: Annotated[int, Field(ge=0)] | None = None
    replicas_remaining: Annotated[int, Field(ge=0, le=1_000_000)] = 1
    legal_hold_artifact_sha256: Digest | None = None
    legal_hold_expires_at_ms: Annotated[int, Field(ge=0)] | None = None
    deletion_receipt: DeletionReceipt | None = None


class ArtifactKind(StrEnum):
    CAPSULE = "capsule"
    CHECKPOINT = "checkpoint"
    ARTIFACT = "artifact"
    TRAJECTORY = "trajectory"
    REWARD = "reward"
    TRAINING_BATCH = "training_batch"
    POLICY = "policy"


class LineageLink(SecurityModel):
    artifact_id: Identifier
    content_sha256: Digest


class ArtifactManifest(SecurityModel):
    schema_version: Literal["sloforge.helix.security-artifact/v1"] = (
        "sloforge.helix.security-artifact/v1"
    )
    manifest_id: Digest
    artifact_id: Identifier
    tenant_id: Identifier
    kind: ArtifactKind
    content_sha256: Digest
    size_bytes: Annotated[int, Field(ge=0, le=2**63 - 1)]
    producer_id: Identifier
    producer_version: Identifier
    created_at_ms: Annotated[int, Field(ge=0)]
    deterministic_seed: Annotated[int, Field(ge=0, le=2**64 - 1)] | None = None
    lineage: Annotated[tuple[LineageLink, ...], Field(max_length=100_000)] = ()


class RewardSourceType(StrEnum):
    DETERMINISTIC = "deterministic"
    HIDDEN_TEST = "hidden_test"


class RewardComponentClaim(SecurityModel):
    component_id: Identifier
    source_type: RewardSourceType
    score: Annotated[float, Field(ge=-(10**12), le=10**12)]
    weight: Annotated[float, Field(ge=-(10**6), le=10**6)] = 1.0
    evidence_artifact_id: Identifier
    evidence_sha256: Digest
    hidden_boundary_digest: Digest | None = None

    @model_validator(mode="after")
    def validate_hidden_boundary(self) -> Self:
        if (self.hidden_boundary_digest is not None) != (
            self.source_type is RewardSourceType.HIDDEN_TEST
        ):
            raise ValueError("hidden-test reward components require one boundary digest")
        return self


class RewardClaim(SecurityModel):
    schema_version: Literal["sloforge.helix.security-reward-claim/v1"] = (
        "sloforge.helix.security-reward-claim/v1"
    )
    reward_id: Identifier
    tenant_id: Identifier
    trajectory_id: Identifier
    trajectory_sha256: Digest
    evaluator_sha256: Digest
    components: Annotated[tuple[RewardComponentClaim, ...], Field(min_length=1, max_length=4096)]
    aggregate_reward: Annotated[float, Field(ge=-(10**15), le=10**15)]
    claim_digest: Digest

    @model_validator(mode="after")
    def validate_components(self) -> Self:
        identifiers = tuple(item.component_id for item in self.components)
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("reward component identifiers must be unique")
        expected = math.fsum(item.score * item.weight for item in self.components)
        if not math.isclose(expected, self.aggregate_reward, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("aggregate reward differs from weighted components")
        return self


class HiddenTestBoundary(SecurityModel):
    schema_version: Literal["sloforge.helix.hidden-test-boundary/v1"] = (
        "sloforge.helix.hidden-test-boundary/v1"
    )
    boundary_id: Digest
    suite_id: Identifier
    tenant_id: Identifier
    policy_visible_roots: Annotated[tuple[NonEmpty, ...], Field(min_length=1, max_length=128)]
    verifier_only_roots: Annotated[tuple[NonEmpty, ...], Field(min_length=1, max_length=128)]
    expected_value_sha256: Annotated[tuple[Digest, ...], Field(min_length=1, max_length=100_000)]
    policy_environment_keys: Annotated[tuple[Identifier, ...], Field(max_length=256)] = ()
    raw_expected_values_present: Literal[False] = False


class ExecutionMode(StrEnum):
    CAPTURE = "capture"
    REWARD = "reward"
    TRAINING = "training"
    SPECULATIVE = "speculative"
    EVALUATION = "evaluation"


class ExecutionIsolation(SecurityModel):
    execution_id: Identifier
    tenant_id: Identifier
    mode: ExecutionMode
    seed: Annotated[int, Field(ge=0, le=2**64 - 1)]
    network_isolation_enforced: bool
    filesystem_isolation_enforced: bool
    environment_sanitized: bool
    external_side_effects_enabled: bool = False
    read_only_roots: Annotated[tuple[NonEmpty, ...], Field(min_length=1, max_length=128)]
    writable_roots: Annotated[tuple[NonEmpty, ...], Field(min_length=1, max_length=128)]
    inherited_environment_keys: Annotated[tuple[Identifier, ...], Field(max_length=256)] = ()
    wall_time_seconds: Annotated[float, Field(gt=0.0, le=3600.0)]
    output_bytes: Annotated[int, Field(gt=0, le=1024 * 1024 * 1024)]
    process_count: Annotated[int, Field(gt=0, le=1024)]


class RepositoryScanPolicy(SecurityModel):
    max_entries: Annotated[int, Field(gt=0, le=1_000_000)] = 100_000
    max_total_bytes: Annotated[int, Field(gt=0, le=2**63 - 1)] = 512 * 1024 * 1024
    max_file_bytes: Annotated[int, Field(gt=0, le=2**63 - 1)] = 64 * 1024 * 1024
    max_depth: Annotated[int, Field(gt=0, le=1024)] = 128
    allow_internal_symlinks: bool = False


class RepositoryFinding(SecurityModel):
    code: Annotated[str, StringConstraints(pattern=r"^[A-Z][A-Z0-9_]{2,63}$")]
    severity: Severity
    path_sha256: Digest


class RepositoryScanEvidence(SecurityModel):
    schema_version: Literal["sloforge.helix.repository-scan/v1"] = (
        "sloforge.helix.repository-scan/v1"
    )
    root_sha256: Digest
    entries_scanned: Annotated[int, Field(ge=0, le=1_000_000)]
    bytes_scanned: Annotated[int, Field(ge=0, le=2**63 - 1)]
    findings: Annotated[tuple[RepositoryFinding, ...], Field(max_length=100_000)]
    evidence_digest: Digest

    @property
    def passed(self) -> bool:
        return not self.findings


class ToolOutputEvidence(SecurityModel):
    schema_version: Literal["sloforge.helix.tool-output-evidence/v1"] = (
        "sloforge.helix.tool-output-evidence/v1"
    )
    raw_sha256: Digest
    sanitized_sha256: Digest
    observed_bytes: Annotated[int, Field(ge=0, le=2**63 - 1)]
    retained_bytes: Annotated[int, Field(ge=0, le=2**63 - 1)]
    truncated: bool
    invalid_utf8_replaced: bool
    control_characters_removed: Annotated[int, Field(ge=0, le=2**31 - 1)]
    redaction_evidence_digest: Digest
    evidence_digest: Digest
