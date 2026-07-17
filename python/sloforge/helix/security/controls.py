"""Offline, deterministic validators for Helix trust boundaries."""

from __future__ import annotations

import hashlib
import math
import os
import re
import stat
from collections.abc import Collection, Mapping, Sequence
from pathlib import Path, PurePosixPath

from .models import (
    ArtifactKind,
    ArtifactLifecycle,
    ArtifactManifest,
    CaptureAccessContext,
    DeletionReceipt,
    ExecutionIsolation,
    HiddenTestBoundary,
    LineageLink,
    ProductionCaptureGrant,
    RedactionEvidence,
    RepositoryFinding,
    RepositoryScanEvidence,
    RepositoryScanPolicy,
    RetentionPolicy,
    RewardClaim,
    RewardComponentClaim,
    RewardSourceType,
    SecurityReport,
    SecurityViolation,
    Severity,
    ToolOutputEvidence,
    canonical_digest,
)

_REDACTION_MARKER = "[REDACTED]"
_MAX_REDACTION_BYTES = 1024 * 1024
_EMAIL = re.compile(
    r"(?<![\w.+-])[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+(?![\w.-])"
)
_PHONE = re.compile(r"(?<!\w)(?:\+?\d[\s().-]*){10,15}(?!\w)")
_PAYMENT_CARD = re.compile(r"(?<!\d)(?:\d[ -]?){13,19}(?!\d)")
_ANSI_ESCAPE = re.compile(r"\x1b(?:\[[0-?]*[ -/]*[@-~]|\][^\x07]*(?:\x07|\x1b\\))")
_CREDENTIAL_MARKERS = (
    "TOKEN",
    "SECRET",
    "PASSWORD",
    "PASSWD",
    "CREDENTIAL",
    "API_KEY",
    "PRIVATE_KEY",
    "KUBECONFIG",
    "AWS_",
    "AZURE_",
    "GCP_",
    "GOOGLE_",
    "OPENAI_",
    "ANTHROPIC_",
    "SSH_",
)


class SecurityValidationError(ValueError):
    """Raised without embedding payload data when a report fails closed."""

    def __init__(self, report: SecurityReport) -> None:
        self.report = report
        codes = ", ".join(item.code for item in report.violations)
        super().__init__(f"security validation failed for {report.subject_id}: {codes}")


def _report(
    subject_id: str,
    checked_at_ms: int,
    violations: Sequence[SecurityViolation],
) -> SecurityReport:
    by_code = {item.code: item for item in violations}
    ordered = tuple(by_code[code] for code in sorted(by_code))
    draft: dict[str, object] = {
        "schema_version": "sloforge.helix.security-report/v1",
        "subject_id": subject_id,
        "checked_at_ms": checked_at_ms,
        "violations": ordered,
    }
    return SecurityReport.model_validate(
        {"report_digest": canonical_digest(draft), **draft}, strict=True
    )


def require_passed(report: SecurityReport) -> None:
    """Raise a non-sensitive error if any security violation is present."""

    if not report.passed:
        raise SecurityValidationError(report)


def _violation(code: str, severity: Severity, message: str) -> SecurityViolation:
    return SecurityViolation(code=code, severity=severity, message=message)


def build_production_capture_grant(
    *,
    grant_id: str,
    tenant_id: str,
    request_id: str,
    approver_id: str,
    issued_at_ms: int,
    expires_at_ms: int,
    approval_artifact_sha256: str,
) -> ProductionCaptureGrant:
    fields: dict[str, object] = {
        "schema_version": "sloforge.helix.production-capture-grant/v1",
        "grant_id": grant_id,
        "tenant_id": tenant_id,
        "request_id": request_id,
        "approver_id": approver_id,
        "issued_at_ms": issued_at_ms,
        "expires_at_ms": expires_at_ms,
        "approval_artifact_sha256": approval_artifact_sha256,
    }
    return ProductionCaptureGrant.model_validate(
        {"grant_digest": canonical_digest(fields), **fields}, strict=True
    )


def validate_capture_access(
    context: CaptureAccessContext,
    *,
    checked_at_ms: int,
    trusted_approval_digests: Collection[str] = (),
) -> SecurityReport:
    """Validate production consent, tenant authorization, and reuse isolation."""

    violations: list[SecurityViolation] = []
    if context.actor_tenant_id != context.resource_tenant_id:
        violations.append(
            _violation(
                "TENANT_AUTHORIZATION_DENIED",
                Severity.CRITICAL,
                "the actor tenant is not authorized for the resource tenant",
            )
        )
    if context.cross_tenant_reuse_enabled:
        violations.append(
            _violation(
                "CROSS_TENANT_REUSE_ENABLED",
                Severity.CRITICAL,
                "cross-tenant state and artifact reuse must remain disabled",
            )
        )
    if (
        context.reuse_source_tenant_id is not None
        and context.reuse_source_tenant_id != context.resource_tenant_id
    ):
        violations.append(
            _violation(
                "CROSS_TENANT_REUSE_ATTEMPT",
                Severity.CRITICAL,
                "a reuse source belongs to a different tenant",
            )
        )
    if context.production:
        if not context.explicit_production_opt_in:
            violations.append(
                _violation(
                    "PRODUCTION_OPT_IN_REQUIRED",
                    Severity.HIGH,
                    "production capture requires an explicit per-request opt-in",
                )
            )
        grant = context.production_grant
        if grant is None:
            violations.append(
                _violation(
                    "PRODUCTION_GRANT_REQUIRED",
                    Severity.HIGH,
                    "production capture requires trusted authorization evidence",
                )
            )
        else:
            expected = canonical_digest(grant.model_dump(mode="json", exclude={"grant_digest"}))
            if expected != grant.grant_digest:
                violations.append(
                    _violation(
                        "PRODUCTION_GRANT_TAMPERED",
                        Severity.CRITICAL,
                        "production authorization evidence failed its integrity seal",
                    )
                )
            if (
                grant.tenant_id != context.resource_tenant_id
                or grant.request_id != context.request_id
            ):
                violations.append(
                    _violation(
                        "PRODUCTION_GRANT_SCOPE_MISMATCH",
                        Severity.CRITICAL,
                        "production authorization evidence is outside the request scope",
                    )
                )
            if grant.issued_at_ms > checked_at_ms or grant.expires_at_ms < checked_at_ms:
                violations.append(
                    _violation(
                        "PRODUCTION_GRANT_EXPIRED",
                        Severity.HIGH,
                        "production authorization evidence is not valid at the requested time",
                    )
                )
            if grant.expires_at_ms <= grant.issued_at_ms:
                violations.append(
                    _violation(
                        "PRODUCTION_GRANT_WINDOW_INVALID",
                        Severity.HIGH,
                        "production authorization evidence has an invalid validity window",
                    )
                )
            if grant.approval_artifact_sha256 not in trusted_approval_digests:
                violations.append(
                    _violation(
                        "PRODUCTION_APPROVAL_UNTRUSTED",
                        Severity.CRITICAL,
                        "production approval is not present in the caller's trusted set",
                    )
                )
    elif context.production_grant is not None:
        violations.append(
            _violation(
                "UNEXPECTED_PRODUCTION_GRANT",
                Severity.LOW,
                "a production grant was supplied for a non-production capture",
            )
        )
    return _report(context.request_id, checked_at_ms, violations)


def _sensitive_counts(text: str, secret_values: Sequence[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    secret_count = sum(text.count(value) for value in set(secret_values) if value)
    if secret_count:
        counts["declared_secret"] = secret_count
    for name, pattern in (("email", _EMAIL), ("phone", _PHONE), ("payment_card", _PAYMENT_CARD)):
        count = len(pattern.findall(text))
        if count:
            counts[name] = count
    return counts


def redact_sensitive_text(
    value: str,
    *,
    secret_values: Sequence[str] = (),
    scanner_id: str = "helix-redactor",
    scanner_version: str = "1.0.0",
) -> tuple[str, RedactionEvidence]:
    """Redact declared secrets and common PII while emitting hash-only evidence."""

    raw = value.encode("utf-8")
    if len(raw) > _MAX_REDACTION_BYTES:
        raise ValueError("redaction input exceeds the 1 MiB bound")
    detected = _sensitive_counts(value, secret_values)
    sanitized = value
    replacements = 0
    for secret in sorted(set(filter(None, secret_values)), key=lambda item: (-len(item), item)):
        count = sanitized.count(secret)
        if count:
            sanitized = sanitized.replace(secret, _REDACTION_MARKER)
            replacements += count
    for pattern in (_EMAIL, _PHONE, _PAYMENT_CARD):
        sanitized, count = pattern.subn(_REDACTION_MARKER, sanitized)
        replacements += count
    residual = _sensitive_counts(sanitized, secret_values)
    fields: dict[str, object] = {
        "schema_version": "sloforge.helix.redaction-evidence/v1",
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "sanitized_sha256": hashlib.sha256(sanitized.encode("utf-8")).hexdigest(),
        "scanner_id": scanner_id,
        "scanner_version": scanner_version,
        "detected_categories": tuple(sorted(detected)),
        "detected_count": sum(detected.values()),
        "residual_count": sum(residual.values()),
        "replacement_count": replacements,
    }
    evidence = RedactionEvidence.model_validate(
        {"evidence_digest": canonical_digest(fields), **fields}, strict=True
    )
    return sanitized, evidence


def validate_redaction_evidence(
    evidence: RedactionEvidence,
    *,
    sanitized_value: str,
    checked_at_ms: int,
    source_value: str | None = None,
) -> SecurityReport:
    """Verify a redaction attestation without requiring the raw source to be retained."""

    violations: list[SecurityViolation] = []
    expected = canonical_digest(evidence.model_dump(mode="json", exclude={"evidence_digest"}))
    if evidence.evidence_digest != expected:
        violations.append(
            _violation(
                "REDACTION_EVIDENCE_TAMPERED",
                Severity.CRITICAL,
                "redaction evidence failed its integrity seal",
            )
        )
    if hashlib.sha256(sanitized_value.encode("utf-8")).hexdigest() != evidence.sanitized_sha256:
        violations.append(
            _violation(
                "SANITIZED_PAYLOAD_DIGEST_MISMATCH",
                Severity.CRITICAL,
                "the sanitized payload does not match the redaction evidence",
            )
        )
    if source_value is not None and (
        hashlib.sha256(source_value.encode("utf-8")).hexdigest() != evidence.source_sha256
    ):
        violations.append(
            _violation(
                "REDACTION_SOURCE_DIGEST_MISMATCH",
                Severity.CRITICAL,
                "the source payload does not match the redaction evidence",
            )
        )
    if evidence.residual_count:
        violations.append(
            _violation(
                "SENSITIVE_DATA_RESIDUAL",
                Severity.CRITICAL,
                "the redaction scan reports residual secret or PII matches",
            )
        )
    if evidence.detected_count and not evidence.replacement_count:
        violations.append(
            _violation(
                "SENSITIVE_DATA_NOT_REPLACED",
                Severity.CRITICAL,
                "sensitive values were detected without recorded replacements",
            )
        )
    return _report(f"redaction:{evidence.evidence_digest[:32]}", checked_at_ms, violations)


def build_deletion_receipt(
    *,
    artifact_id: str,
    tenant_id: str,
    requested_at_ms: int,
    completed_at_ms: int,
    tombstone_sha256: str,
    replicas_deleted: int,
) -> DeletionReceipt:
    fields: dict[str, object] = {
        "schema_version": "sloforge.helix.deletion-receipt/v1",
        "artifact_id": artifact_id,
        "tenant_id": tenant_id,
        "requested_at_ms": requested_at_ms,
        "completed_at_ms": completed_at_ms,
        "tombstone_sha256": tombstone_sha256,
        "replicas_deleted": replicas_deleted,
    }
    return DeletionReceipt.model_validate(
        {"receipt_digest": canonical_digest(fields), **fields}, strict=True
    )


def validate_retention(
    policy: RetentionPolicy,
    lifecycle: ArtifactLifecycle,
    *,
    checked_at_ms: int,
    trusted_legal_hold_digests: Collection[str] = (),
) -> SecurityReport:
    """Validate expiry, deletion deadlines, receipts, and bounded legal holds."""

    violations: list[SecurityViolation] = []
    if policy.tenant_id != lifecycle.tenant_id:
        violations.append(
            _violation(
                "RETENTION_TENANT_MISMATCH",
                Severity.CRITICAL,
                "retention policy and artifact tenants do not match",
            )
        )
    if policy.cross_tenant_reuse_enabled:
        violations.append(
            _violation(
                "RETENTION_CROSS_TENANT_REUSE_ENABLED",
                Severity.CRITICAL,
                "retained artifacts cannot be shared across tenants",
            )
        )
    if lifecycle.expires_at_ms < lifecycle.created_at_ms or (
        lifecycle.expires_at_ms - lifecycle.created_at_ms > policy.maximum_retention_ms
    ):
        violations.append(
            _violation(
                "RETENTION_WINDOW_INVALID",
                Severity.HIGH,
                "artifact expiry exceeds the bounded retention policy",
            )
        )
    hold_active = False
    if lifecycle.legal_hold_artifact_sha256 is not None:
        hold_expiry = lifecycle.legal_hold_expires_at_ms
        if (
            lifecycle.legal_hold_artifact_sha256 not in trusted_legal_hold_digests
            or hold_expiry is None
            or hold_expiry <= lifecycle.created_at_ms
            or hold_expiry - lifecycle.created_at_ms > policy.legal_hold_maximum_ms
        ):
            violations.append(
                _violation(
                    "LEGAL_HOLD_INVALID",
                    Severity.HIGH,
                    "legal-hold evidence is untrusted, incomplete, or exceeds its bound",
                )
            )
        else:
            hold_active = checked_at_ms <= hold_expiry
    elif lifecycle.legal_hold_expires_at_ms is not None:
        violations.append(
            _violation(
                "LEGAL_HOLD_EVIDENCE_REQUIRED",
                Severity.HIGH,
                "a legal-hold expiry requires trusted hold evidence",
            )
        )
    if not hold_active and checked_at_ms > lifecycle.expires_at_ms:
        if lifecycle.deletion_requested_at_ms is None:
            violations.append(
                _violation(
                    "EXPIRED_ARTIFACT_NOT_SCHEDULED",
                    Severity.HIGH,
                    "an expired artifact has no deletion request",
                )
            )
        elif (
            lifecycle.deleted_at_ms is None
            and checked_at_ms > lifecycle.deletion_requested_at_ms + policy.deletion_grace_ms
        ):
            violations.append(
                _violation(
                    "DELETION_DEADLINE_MISSED",
                    Severity.HIGH,
                    "artifact deletion did not complete within the configured grace period",
                )
            )
    if lifecycle.deleted_at_ms is not None:
        if lifecycle.deletion_requested_at_ms is None or (
            lifecycle.deleted_at_ms < lifecycle.deletion_requested_at_ms
        ):
            violations.append(
                _violation(
                    "DELETION_ORDER_INVALID",
                    Severity.HIGH,
                    "artifact deletion precedes its deletion request",
                )
            )
        if lifecycle.replicas_remaining != 0:
            violations.append(
                _violation(
                    "DELETION_INCOMPLETE",
                    Severity.CRITICAL,
                    "a deleted artifact still has retained replicas",
                )
            )
        receipt = lifecycle.deletion_receipt
        if policy.require_deletion_receipt and receipt is None:
            violations.append(
                _violation(
                    "DELETION_RECEIPT_REQUIRED",
                    Severity.HIGH,
                    "completed deletion requires a tamper-evident receipt",
                )
            )
        if receipt is not None:
            receipt_expected = canonical_digest(
                receipt.model_dump(mode="json", exclude={"receipt_digest"})
            )
            if receipt.receipt_digest != receipt_expected:
                violations.append(
                    _violation(
                        "DELETION_RECEIPT_TAMPERED",
                        Severity.CRITICAL,
                        "deletion receipt failed its integrity seal",
                    )
                )
            if (
                receipt.artifact_id != lifecycle.artifact_id
                or receipt.tenant_id != lifecycle.tenant_id
                or receipt.requested_at_ms != lifecycle.deletion_requested_at_ms
                or receipt.completed_at_ms != lifecycle.deleted_at_ms
            ):
                violations.append(
                    _violation(
                        "DELETION_RECEIPT_SCOPE_MISMATCH",
                        Severity.CRITICAL,
                        "deletion receipt does not match the artifact lifecycle",
                    )
                )
    elif lifecycle.deletion_receipt is not None:
        violations.append(
            _violation(
                "PREMATURE_DELETION_RECEIPT",
                Severity.HIGH,
                "a deletion receipt exists before deletion completion",
            )
        )
    return _report(lifecycle.artifact_id, checked_at_ms, violations)


def build_artifact_manifest(
    *,
    artifact_id: str,
    tenant_id: str,
    kind: ArtifactKind,
    payload: bytes,
    producer_id: str,
    producer_version: str,
    created_at_ms: int,
    deterministic_seed: int | None = None,
    lineage: tuple[LineageLink, ...] = (),
) -> ArtifactManifest:
    fields: dict[str, object] = {
        "schema_version": "sloforge.helix.security-artifact/v1",
        "artifact_id": artifact_id,
        "tenant_id": tenant_id,
        "kind": kind,
        "content_sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
        "producer_id": producer_id,
        "producer_version": producer_version,
        "created_at_ms": created_at_ms,
        "deterministic_seed": deterministic_seed,
        "lineage": lineage,
    }
    identity = {
        key: value.value if isinstance(value, ArtifactKind) else value
        for key, value in fields.items()
    }
    identity["lineage"] = [item.model_dump(mode="json") for item in lineage]
    return ArtifactManifest.model_validate(
        {"manifest_id": canonical_digest(identity), **fields}, strict=True
    )


def validate_artifact_payload(
    manifest: ArtifactManifest,
    payload: bytes,
    *,
    expected_tenant_id: str,
    checked_at_ms: int,
) -> SecurityReport:
    """Verify a capsule, checkpoint, or generic artifact before consumption."""

    violations: list[SecurityViolation] = []
    expected_manifest = canonical_digest(manifest.model_dump(mode="json", exclude={"manifest_id"}))
    if manifest.manifest_id != expected_manifest:
        violations.append(
            _violation(
                "ARTIFACT_MANIFEST_TAMPERED",
                Severity.CRITICAL,
                "artifact manifest failed its integrity seal",
            )
        )
    if manifest.tenant_id != expected_tenant_id:
        violations.append(
            _violation(
                "ARTIFACT_TENANT_MISMATCH",
                Severity.CRITICAL,
                "artifact tenant does not match the consuming tenant",
            )
        )
    if len(payload) != manifest.size_bytes:
        violations.append(
            _violation(
                "ARTIFACT_SIZE_MISMATCH",
                Severity.CRITICAL,
                "artifact payload size differs from its manifest",
            )
        )
    if hashlib.sha256(payload).hexdigest() != manifest.content_sha256:
        violations.append(
            _violation(
                "ARTIFACT_DIGEST_MISMATCH",
                Severity.CRITICAL,
                "artifact payload failed digest verification",
            )
        )
    if manifest.kind in {ArtifactKind.CAPSULE, ArtifactKind.CHECKPOINT} and (
        manifest.deterministic_seed is None
    ):
        violations.append(
            _violation(
                "ARTIFACT_SEED_REQUIRED",
                Severity.HIGH,
                "capsules and checkpoints require an explicit deterministic seed",
            )
        )
    lineage_ids = tuple(item.artifact_id for item in manifest.lineage)
    if len(lineage_ids) != len(set(lineage_ids)):
        violations.append(
            _violation(
                "ARTIFACT_LINEAGE_DUPLICATE",
                Severity.HIGH,
                "artifact lineage contains duplicate parent identities",
            )
        )
    return _report(manifest.artifact_id, checked_at_ms, violations)


def validate_artifact_graph(
    manifests: Sequence[ArtifactManifest],
    *,
    expected_tenant_id: str,
    checked_at_ms: int,
    trusted_roots: Mapping[str, str] | None = None,
) -> SecurityReport:
    """Validate exact parent digests, tenant scope, and acyclic lineage."""

    roots = trusted_roots or {}
    violations: list[SecurityViolation] = []
    by_id: dict[str, ArtifactManifest] = {}
    for manifest in manifests:
        existing = by_id.get(manifest.artifact_id)
        if existing is not None:
            code = (
                "ARTIFACT_ID_DIGEST_CONFLICT"
                if existing.content_sha256 != manifest.content_sha256
                else "ARTIFACT_ID_DUPLICATE"
            )
            violations.append(
                _violation(code, Severity.CRITICAL, "artifact identity is duplicated in lineage")
            )
        else:
            by_id[manifest.artifact_id] = manifest
        if manifest.tenant_id != expected_tenant_id:
            violations.append(
                _violation(
                    "LINEAGE_TENANT_MISMATCH",
                    Severity.CRITICAL,
                    "lineage contains an artifact from another tenant",
                )
            )
        expected_seal = canonical_digest(manifest.model_dump(mode="json", exclude={"manifest_id"}))
        if manifest.manifest_id != expected_seal:
            violations.append(
                _violation(
                    "LINEAGE_MANIFEST_TAMPERED",
                    Severity.CRITICAL,
                    "lineage contains a manifest with an invalid seal",
                )
            )
    edges: dict[str, tuple[str, ...]] = {}
    for manifest in manifests:
        parent_ids: list[str] = []
        for link in manifest.lineage:
            parent = by_id.get(link.artifact_id)
            expected_digest = (
                parent.content_sha256 if parent is not None else roots.get(link.artifact_id)
            )
            if expected_digest is None:
                violations.append(
                    _violation(
                        "LINEAGE_PARENT_MISSING",
                        Severity.CRITICAL,
                        "lineage references a parent outside the supplied trusted graph",
                    )
                )
            elif expected_digest != link.content_sha256:
                violations.append(
                    _violation(
                        "LINEAGE_PARENT_DIGEST_MISMATCH",
                        Severity.CRITICAL,
                        "lineage parent digest differs from the trusted artifact",
                    )
                )
            if parent is not None:
                parent_ids.append(parent.artifact_id)
        edges[manifest.artifact_id] = tuple(parent_ids)
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(artifact_id: str) -> bool:
        if artifact_id in visiting:
            return True
        if artifact_id in visited:
            return False
        visiting.add(artifact_id)
        cycle = any(visit(parent_id) for parent_id in edges.get(artifact_id, ()))
        visiting.remove(artifact_id)
        visited.add(artifact_id)
        return cycle

    if any(visit(artifact_id) for artifact_id in sorted(edges)):
        violations.append(
            _violation(
                "LINEAGE_CYCLE",
                Severity.CRITICAL,
                "artifact lineage contains a cycle",
            )
        )
    subject = f"lineage:{canonical_digest(sorted(by_id))[:32]}"
    return _report(subject, checked_at_ms, violations)


def build_reward_claim(
    *,
    reward_id: str,
    tenant_id: str,
    trajectory_id: str,
    trajectory_sha256: str,
    evaluator_sha256: str,
    components: tuple[RewardComponentClaim, ...],
) -> RewardClaim:
    aggregate = math.fsum(item.score * item.weight for item in components)
    fields: dict[str, object] = {
        "schema_version": "sloforge.helix.security-reward-claim/v1",
        "reward_id": reward_id,
        "tenant_id": tenant_id,
        "trajectory_id": trajectory_id,
        "trajectory_sha256": trajectory_sha256,
        "evaluator_sha256": evaluator_sha256,
        "components": components,
        "aggregate_reward": aggregate,
    }
    identity = dict(fields)
    identity["components"] = [item.model_dump(mode="json") for item in components]
    return RewardClaim.model_validate(
        {"claim_digest": canonical_digest(identity), **fields}, strict=True
    )


def validate_reward_claim(
    claim: RewardClaim,
    *,
    artifacts: Mapping[str, ArtifactManifest],
    expected_tenant_id: str,
    trusted_evaluator_digests: Collection[str],
    trusted_hidden_boundary_digests: Collection[str] = (),
    maximum_absolute_component_score: float = 1_000_000.0,
    checked_at_ms: int,
) -> SecurityReport:
    """Reject untrusted evaluators, forged lineage, and implausible reward values."""

    if not math.isfinite(maximum_absolute_component_score) or maximum_absolute_component_score <= 0:
        raise ValueError("maximum reward component magnitude must be finite and positive")
    violations: list[SecurityViolation] = []
    expected_claim = canonical_digest(claim.model_dump(mode="json", exclude={"claim_digest"}))
    if claim.claim_digest != expected_claim:
        violations.append(
            _violation(
                "REWARD_CLAIM_TAMPERED",
                Severity.CRITICAL,
                "reward claim failed its integrity seal",
            )
        )
    if claim.tenant_id != expected_tenant_id:
        violations.append(
            _violation(
                "REWARD_TENANT_MISMATCH",
                Severity.CRITICAL,
                "reward evidence belongs to another tenant",
            )
        )
    if claim.evaluator_sha256 not in trusted_evaluator_digests:
        violations.append(
            _violation(
                "REWARD_EVALUATOR_UNTRUSTED",
                Severity.CRITICAL,
                "reward evaluator is not present in the caller's trusted set",
            )
        )
    trajectory = artifacts.get(claim.trajectory_id)
    if (
        trajectory is None
        or trajectory.content_sha256 != claim.trajectory_sha256
        or trajectory.kind is not ArtifactKind.TRAJECTORY
    ):
        violations.append(
            _violation(
                "REWARD_TRAJECTORY_LINEAGE_INVALID",
                Severity.CRITICAL,
                "reward trajectory identity or digest is not trusted",
            )
        )
    elif trajectory.tenant_id != expected_tenant_id:
        violations.append(
            _violation(
                "REWARD_TRAJECTORY_TENANT_MISMATCH",
                Severity.CRITICAL,
                "reward trajectory belongs to another tenant",
            )
        )
    for component in claim.components:
        evidence = artifacts.get(component.evidence_artifact_id)
        if evidence is None or evidence.content_sha256 != component.evidence_sha256:
            violations.append(
                _violation(
                    "REWARD_COMPONENT_EVIDENCE_INVALID",
                    Severity.CRITICAL,
                    "a reward component does not match trusted source evidence",
                )
            )
        elif evidence.tenant_id != expected_tenant_id:
            violations.append(
                _violation(
                    "REWARD_COMPONENT_TENANT_MISMATCH",
                    Severity.CRITICAL,
                    "reward component evidence belongs to another tenant",
                )
            )
        if abs(component.score) > maximum_absolute_component_score:
            violations.append(
                _violation(
                    "REWARD_COMPONENT_OUT_OF_POLICY",
                    Severity.HIGH,
                    "a reward component exceeds the configured magnitude bound",
                )
            )
        if component.source_type is RewardSourceType.HIDDEN_TEST and (
            component.hidden_boundary_digest not in trusted_hidden_boundary_digests
        ):
            violations.append(
                _violation(
                    "HIDDEN_TEST_BOUNDARY_UNTRUSTED",
                    Severity.CRITICAL,
                    "hidden-test reward lacks a trusted isolation-boundary attestation",
                )
            )
    return _report(claim.reward_id, checked_at_ms, violations)


def _normalize_boundary_path(value: str) -> PurePosixPath | None:
    if "\x00" in value or "\\" in value:
        return None
    path = PurePosixPath(value)
    if ".." in path.parts or value in {"", "."}:
        return None
    return path


def _paths_overlap(left: PurePosixPath, right: PurePosixPath) -> bool:
    return left == right or left in right.parents or right in left.parents


def build_hidden_test_boundary(
    *,
    suite_id: str,
    tenant_id: str,
    policy_visible_roots: tuple[str, ...],
    verifier_only_roots: tuple[str, ...],
    expected_value_sha256: tuple[str, ...],
    policy_environment_keys: tuple[str, ...] = (),
) -> HiddenTestBoundary:
    fields: dict[str, object] = {
        "schema_version": "sloforge.helix.hidden-test-boundary/v1",
        "suite_id": suite_id,
        "tenant_id": tenant_id,
        "policy_visible_roots": policy_visible_roots,
        "verifier_only_roots": verifier_only_roots,
        "expected_value_sha256": expected_value_sha256,
        "policy_environment_keys": policy_environment_keys,
        "raw_expected_values_present": False,
    }
    return HiddenTestBoundary.model_validate(
        {"boundary_id": canonical_digest(fields), **fields}, strict=True
    )


def validate_hidden_test_boundary(
    boundary: HiddenTestBoundary,
    *,
    expected_tenant_id: str,
    checked_at_ms: int,
) -> SecurityReport:
    """Ensure hidden tests and answers are never policy-visible by path or environment."""

    violations: list[SecurityViolation] = []
    if boundary.boundary_id != canonical_digest(
        boundary.model_dump(mode="json", exclude={"boundary_id"})
    ):
        violations.append(
            _violation(
                "HIDDEN_BOUNDARY_TAMPERED",
                Severity.CRITICAL,
                "hidden-test boundary failed its integrity seal",
            )
        )
    if boundary.tenant_id != expected_tenant_id:
        violations.append(
            _violation(
                "HIDDEN_BOUNDARY_TENANT_MISMATCH",
                Severity.CRITICAL,
                "hidden-test boundary belongs to another tenant",
            )
        )
    visible = tuple(_normalize_boundary_path(item) for item in boundary.policy_visible_roots)
    hidden = tuple(_normalize_boundary_path(item) for item in boundary.verifier_only_roots)
    if any(item is None for item in (*visible, *hidden)):
        violations.append(
            _violation(
                "HIDDEN_BOUNDARY_PATH_INVALID",
                Severity.CRITICAL,
                "hidden-test boundary contains an unsafe path",
            )
        )
    else:
        typed_visible = tuple(item for item in visible if item is not None)
        typed_hidden = tuple(item for item in hidden if item is not None)
        if any(_paths_overlap(left, right) for left in typed_visible for right in typed_hidden):
            violations.append(
                _violation(
                    "HIDDEN_TEST_PATH_LEAKAGE",
                    Severity.CRITICAL,
                    "policy-visible and verifier-only path roots overlap",
                )
            )
    if len(boundary.expected_value_sha256) != len(set(boundary.expected_value_sha256)):
        violations.append(
            _violation(
                "HIDDEN_EXPECTED_DIGEST_DUPLICATE",
                Severity.MEDIUM,
                "hidden expected-value digests must be unique",
            )
        )
    forbidden_keys = [
        key
        for key in boundary.policy_environment_keys
        if any(
            marker in key.upper()
            for marker in (*_CREDENTIAL_MARKERS, "EXPECTED", "ANSWER", "HIDDEN")
        )
    ]
    if forbidden_keys:
        violations.append(
            _violation(
                "HIDDEN_TEST_ENVIRONMENT_LEAKAGE",
                Severity.CRITICAL,
                "policy-visible environment names could expose hidden answers or credentials",
            )
        )
    return _report(f"hidden:{boundary.suite_id}", checked_at_ms, violations)


def validate_execution_isolation(
    isolation: ExecutionIsolation,
    *,
    expected_tenant_id: str,
    checked_at_ms: int,
) -> SecurityReport:
    """Require kernel isolation and a single isolated write namespace."""

    violations: list[SecurityViolation] = []
    if isolation.tenant_id != expected_tenant_id:
        violations.append(
            _violation(
                "EXECUTION_TENANT_MISMATCH",
                Severity.CRITICAL,
                "execution tenant differs from the authorized tenant",
            )
        )
    if not isolation.network_isolation_enforced:
        violations.append(
            _violation(
                "NETWORK_ISOLATION_REQUIRED",
                Severity.CRITICAL,
                "untrusted Helix execution requires enforced network isolation",
            )
        )
    if not isolation.filesystem_isolation_enforced:
        violations.append(
            _violation(
                "FILESYSTEM_ISOLATION_REQUIRED",
                Severity.CRITICAL,
                "untrusted Helix execution requires enforced filesystem isolation",
            )
        )
    if not isolation.environment_sanitized:
        violations.append(
            _violation(
                "ENVIRONMENT_SANITIZATION_REQUIRED",
                Severity.CRITICAL,
                "untrusted Helix execution requires a sanitized environment",
            )
        )
    if isolation.external_side_effects_enabled:
        violations.append(
            _violation(
                "EXTERNAL_EFFECTS_NOT_ISOLATED",
                Severity.CRITICAL,
                "capture, reward, training, evaluation, and speculation cannot perform live effects",
            )
        )
    if any(
        marker in key.upper()
        for key in isolation.inherited_environment_keys
        for marker in _CREDENTIAL_MARKERS
    ):
        violations.append(
            _violation(
                "CREDENTIAL_ENVIRONMENT_INHERITED",
                Severity.CRITICAL,
                "credential-like environment variables cannot enter untrusted execution",
            )
        )
    read_only = tuple(_normalize_boundary_path(item) for item in isolation.read_only_roots)
    writable = tuple(_normalize_boundary_path(item) for item in isolation.writable_roots)
    if any(item is None or not item.is_absolute() for item in (*read_only, *writable)):
        violations.append(
            _violation(
                "ISOLATION_PATH_INVALID",
                Severity.CRITICAL,
                "isolation roots must be safe absolute paths",
            )
        )
    else:
        typed_read = tuple(item for item in read_only if item is not None)
        typed_write = tuple(item for item in writable if item is not None)
        if any(_paths_overlap(left, right) for left in typed_read for right in typed_write):
            violations.append(
                _violation(
                    "ISOLATION_WRITE_OVERLAP",
                    Severity.CRITICAL,
                    "writable output overlaps a read-only input",
                )
            )
        if len(typed_write) != 1:
            violations.append(
                _violation(
                    "ISOLATION_WRITE_NAMESPACE_INVALID",
                    Severity.HIGH,
                    "untrusted execution requires exactly one bounded write namespace",
                )
            )
    return _report(isolation.execution_id, checked_at_ms, violations)


def _path_digest(path: PurePosixPath) -> str:
    return hashlib.sha256(path.as_posix().encode("utf-8", errors="surrogateescape")).hexdigest()


def _repo_finding(code: str, severity: Severity, path: PurePosixPath) -> RepositoryFinding:
    return RepositoryFinding(code=code, severity=severity, path_sha256=_path_digest(path))


def scan_repository(
    root: Path, *, policy: RepositoryScanPolicy | None = None
) -> RepositoryScanEvidence:
    """Scan a hostile repository without invoking Git, hooks, interpreters, or tools."""

    chosen = policy or RepositoryScanPolicy()
    absolute = root.absolute()
    root_label = PurePosixPath(".")
    findings: list[RepositoryFinding] = []
    if absolute.is_symlink():
        findings.append(_repo_finding("REPOSITORY_ROOT_SYMLINK", Severity.CRITICAL, root_label))
        root_digest = hashlib.sha256(b"sloforge.helix.repository-scan/v1\0root-symlink").hexdigest()
        fields: dict[str, object] = {
            "schema_version": "sloforge.helix.repository-scan/v1",
            "root_sha256": root_digest,
            "entries_scanned": 0,
            "bytes_scanned": 0,
            "findings": tuple(findings),
        }
        return RepositoryScanEvidence.model_validate(
            {"evidence_digest": canonical_digest(fields), **fields}, strict=True
        )
    canonical_root = absolute.resolve(strict=True)
    if not canonical_root.is_dir():
        raise ValueError("repository root must be a directory")
    tree_hash = hashlib.sha256(b"sloforge.helix.repository-tree/v1\0")
    entries_scanned = 0
    bytes_scanned = 0
    pending: list[tuple[Path, int]] = [(canonical_root, 0)]
    stop = False
    while pending and not stop:
        directory, depth = pending.pop()
        try:
            children = sorted(os.scandir(directory), key=lambda item: item.name, reverse=True)
        except OSError as exc:
            del exc
            relative_directory = PurePosixPath(directory.relative_to(canonical_root).as_posix())
            findings.append(
                _repo_finding("REPOSITORY_ENTRY_UNREADABLE", Severity.HIGH, relative_directory)
            )
            continue
        for child in children:
            path = Path(child.path)
            relative = PurePosixPath(path.relative_to(canonical_root).as_posix())
            if entries_scanned >= chosen.max_entries:
                findings.append(_repo_finding("REPOSITORY_ENTRY_LIMIT", Severity.HIGH, relative))
                stop = True
                break
            entries_scanned += 1
            if depth + 1 > chosen.max_depth:
                findings.append(_repo_finding("REPOSITORY_DEPTH_LIMIT", Severity.HIGH, relative))
                continue
            if any(
                ord(character) < 32 or ord(character) == 127 for character in relative.as_posix()
            ):
                findings.append(_repo_finding("REPOSITORY_UNSAFE_PATH", Severity.HIGH, relative))
            try:
                status = child.stat(follow_symlinks=False)
            except OSError as exc:
                del exc
                findings.append(
                    _repo_finding("REPOSITORY_ENTRY_UNREADABLE", Severity.HIGH, relative)
                )
                continue
            tree_hash.update(relative.as_posix().encode("utf-8", errors="surrogateescape"))
            tree_hash.update(b"\0")
            tree_hash.update(str(stat.S_IFMT(status.st_mode)).encode("ascii"))
            tree_hash.update(b"\0")
            if child.is_symlink():
                unsafe = not chosen.allow_internal_symlinks
                try:
                    target = path.resolve(strict=True)
                    unsafe = unsafe or not target.is_relative_to(canonical_root)
                except (OSError, RuntimeError):
                    unsafe = True
                if unsafe:
                    findings.append(
                        _repo_finding("REPOSITORY_SYMLINK_REJECTED", Severity.CRITICAL, relative)
                    )
                continue
            if child.is_dir(follow_symlinks=False):
                pending.append((path, depth + 1))
                continue
            if not child.is_file(follow_symlinks=False):
                findings.append(
                    _repo_finding("REPOSITORY_SPECIAL_FILE", Severity.CRITICAL, relative)
                )
                continue
            if status.st_size > chosen.max_file_bytes:
                findings.append(_repo_finding("REPOSITORY_FILE_LIMIT", Severity.HIGH, relative))
                continue
            if bytes_scanned + status.st_size > chosen.max_total_bytes:
                findings.append(_repo_finding("REPOSITORY_BYTE_LIMIT", Severity.HIGH, relative))
                stop = True
                break
            try:
                data = path.read_bytes()
            except OSError as exc:
                del exc
                findings.append(
                    _repo_finding("REPOSITORY_ENTRY_UNREADABLE", Severity.HIGH, relative)
                )
                continue
            if len(data) != status.st_size:
                findings.append(_repo_finding("REPOSITORY_ENTRY_CHANGED", Severity.HIGH, relative))
                continue
            bytes_scanned += len(data)
            tree_hash.update(str(len(data)).encode("ascii"))
            tree_hash.update(b"\0")
            tree_hash.update(hashlib.sha256(data).digest())
            lower_path = relative.as_posix().lower()
            lower_data = data.lower()
            if lower_path.startswith(".git/hooks/") and not lower_path.endswith(".sample"):
                findings.append(_repo_finding("REPOSITORY_GIT_HOOK", Severity.CRITICAL, relative))
            if lower_path == ".git/config" and any(
                marker in lower_data
                for marker in (b"fsmonitor", b"hookspath", b"sshcommand", b"[filter ")
            ):
                findings.append(
                    _repo_finding("REPOSITORY_GIT_CONFIG_EXECUTION", Severity.CRITICAL, relative)
                )
            if lower_path == ".gitattributes" and b"filter=" in lower_data:
                findings.append(_repo_finding("REPOSITORY_GIT_FILTER", Severity.CRITICAL, relative))
            if lower_path == ".gitmodules" and b"url" in lower_data:
                findings.append(
                    _repo_finding("REPOSITORY_EXTERNAL_SUBMODULE", Severity.HIGH, relative)
                )
    ordered_findings = tuple(sorted(set(findings), key=lambda item: (item.code, item.path_sha256)))
    fields = {
        "schema_version": "sloforge.helix.repository-scan/v1",
        "root_sha256": tree_hash.hexdigest(),
        "entries_scanned": entries_scanned,
        "bytes_scanned": bytes_scanned,
        "findings": ordered_findings,
    }
    return RepositoryScanEvidence.model_validate(
        {"evidence_digest": canonical_digest(fields), **fields}, strict=True
    )


def sanitize_tool_output(
    output: bytes,
    *,
    secret_values: Sequence[str] = (),
    max_bytes: int = 256 * 1024,
    max_observed_bytes: int = 16 * 1024 * 1024,
) -> tuple[str, ToolOutputEvidence, RedactionEvidence]:
    """Bound, normalize, and redact hostile tool output before persistence or display."""

    if not 1 <= max_bytes <= 1024 * 1024:
        raise ValueError("tool output retention bound must be between 1 byte and 1 MiB")
    if max_observed_bytes < max_bytes or len(output) > max_observed_bytes:
        raise ValueError("observed tool output exceeds the configured processing bound")
    truncated = len(output) > max_bytes
    guard = min(
        max_bytes,
        max(256, *(len(item.encode("utf-8")) for item in secret_values if item), 0),
    )
    retained_limit = max_bytes - guard if truncated else max_bytes
    retained = output[:retained_limit]
    decoded = retained.decode("utf-8", errors="replace")
    invalid_utf8 = "\ufffd" in decoded
    without_ansi = _ANSI_ESCAPE.sub("", decoded)
    safe_characters: list[str] = []
    removed = len(decoded) - len(without_ansi)
    for character in without_ansi:
        if character in {"\n", "\r", "\t"} or (ord(character) >= 32 and ord(character) != 127):
            safe_characters.append(character)
        else:
            removed += 1
    normalized = "".join(safe_characters)
    redacted, redaction = redact_sensitive_text(
        normalized,
        secret_values=secret_values,
        scanner_id="helix-tool-output-redactor",
        scanner_version="1.0.0",
    )
    safe_bytes = redacted.encode("utf-8")[:max_bytes]
    safe = safe_bytes.decode("utf-8", errors="ignore")
    fields: dict[str, object] = {
        "schema_version": "sloforge.helix.tool-output-evidence/v1",
        "raw_sha256": hashlib.sha256(output).hexdigest(),
        "sanitized_sha256": hashlib.sha256(safe.encode("utf-8")).hexdigest(),
        "observed_bytes": len(output),
        "retained_bytes": len(safe.encode("utf-8")),
        "truncated": truncated,
        "invalid_utf8_replaced": invalid_utf8,
        "control_characters_removed": removed,
        "redaction_evidence_digest": redaction.evidence_digest,
    }
    evidence = ToolOutputEvidence.model_validate(
        {"evidence_digest": canonical_digest(fields), **fields}, strict=True
    )
    return safe, evidence, redaction


def validate_tool_output_evidence(
    evidence: ToolOutputEvidence,
    *,
    raw_output: bytes,
    sanitized_output: str,
    checked_at_ms: int,
    redaction_evidence: RedactionEvidence | None = None,
) -> SecurityReport:
    violations: list[SecurityViolation] = []
    if evidence.evidence_digest != canonical_digest(
        evidence.model_dump(mode="json", exclude={"evidence_digest"})
    ):
        violations.append(
            _violation(
                "TOOL_OUTPUT_EVIDENCE_TAMPERED",
                Severity.CRITICAL,
                "tool-output evidence failed its integrity seal",
            )
        )
    if hashlib.sha256(raw_output).hexdigest() != evidence.raw_sha256:
        violations.append(
            _violation(
                "TOOL_OUTPUT_SOURCE_MISMATCH",
                Severity.CRITICAL,
                "tool output does not match its source digest",
            )
        )
    safe_bytes = sanitized_output.encode("utf-8")
    if (
        hashlib.sha256(safe_bytes).hexdigest() != evidence.sanitized_sha256
        or len(safe_bytes) != evidence.retained_bytes
    ):
        violations.append(
            _violation(
                "TOOL_OUTPUT_SANITIZED_MISMATCH",
                Severity.CRITICAL,
                "sanitized tool output does not match its evidence",
            )
        )
    if redaction_evidence is None or (
        redaction_evidence.evidence_digest != evidence.redaction_evidence_digest
        or redaction_evidence.residual_count != 0
    ):
        violations.append(
            _violation(
                "TOOL_OUTPUT_REDACTION_UNVERIFIED",
                Severity.CRITICAL,
                "tool output lacks valid zero-residual redaction evidence",
            )
        )
    return _report(f"tool-output:{evidence.raw_sha256[:32]}", checked_at_ms, violations)
