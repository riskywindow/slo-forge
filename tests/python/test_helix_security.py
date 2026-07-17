from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from sloforge.helix.security import (
    ArtifactKind,
    ArtifactLifecycle,
    CaptureAccessContext,
    DuplicateSubmissionError,
    ExecutionIsolation,
    ExecutionMode,
    LineageLink,
    RepositoryScanPolicy,
    RetentionPolicy,
    RewardComponentClaim,
    RewardSourceType,
    SubmissionDisposition,
    SubmissionIdentityConflict,
    SubmissionRegistry,
    build_artifact_manifest,
    build_deletion_receipt,
    build_hidden_test_boundary,
    build_production_capture_grant,
    build_reward_claim,
    canonical_digest,
    redact_sensitive_text,
    sanitize_tool_output,
    scan_repository,
    validate_artifact_graph,
    validate_artifact_payload,
    validate_capture_access,
    validate_execution_isolation,
    validate_hidden_test_boundary,
    validate_redaction_evidence,
    validate_retention,
    validate_reward_claim,
    validate_tool_output_evidence,
)


def _digest(value: bytes | str) -> str:
    payload = value if isinstance(value, bytes) else value.encode()
    return hashlib.sha256(payload).hexdigest()


def _codes(report: object) -> set[str]:
    return {item.code for item in report.violations}  # type: ignore[attr-defined]


def test_production_capture_requires_scoped_trusted_opt_in() -> None:
    approval = _digest("approval")
    grant = build_production_capture_grant(
        grant_id="grant-1",
        tenant_id="tenant-a",
        request_id="capture-1",
        approver_id="privacy-reviewer",
        issued_at_ms=10,
        expires_at_ms=30,
        approval_artifact_sha256=approval,
    )
    context = CaptureAccessContext(
        request_id="capture-1",
        actor_tenant_id="tenant-a",
        resource_tenant_id="tenant-a",
        production=True,
        explicit_production_opt_in=True,
        production_grant=grant,
    )
    assert validate_capture_access(
        context, checked_at_ms=20, trusted_approval_digests={approval}
    ).passed

    denied = context.model_copy(
        update={
            "actor_tenant_id": "tenant-b",
            "explicit_production_opt_in": False,
            "cross_tenant_reuse_enabled": True,
            "reuse_source_tenant_id": "tenant-b",
        }
    )
    assert _codes(validate_capture_access(denied, checked_at_ms=20)) >= {
        "TENANT_AUTHORIZATION_DENIED",
        "PRODUCTION_OPT_IN_REQUIRED",
        "PRODUCTION_APPROVAL_UNTRUSTED",
        "CROSS_TENANT_REUSE_ENABLED",
        "CROSS_TENANT_REUSE_ATTEMPT",
    }


def test_secret_and_pii_redaction_emits_zero_residual_hash_only_evidence() -> None:
    declared_secret = "fixture-secret-value"
    source = f"token={declared_secret}; contact=person@example.test"
    sanitized, evidence = redact_sensitive_text(source, secret_values=(declared_secret,))
    assert declared_secret not in sanitized
    assert "person@example.test" not in sanitized
    assert evidence.detected_categories == ("declared_secret", "email")
    assert evidence.residual_count == 0
    assert declared_secret not in evidence.model_dump_json()
    assert validate_redaction_evidence(
        evidence, sanitized_value=sanitized, source_value=source, checked_at_ms=1
    ).passed

    tampered = evidence.model_copy(update={"residual_count": 1})
    assert _codes(
        validate_redaction_evidence(tampered, sanitized_value=sanitized, checked_at_ms=1)
    ) >= {"REDACTION_EVIDENCE_TAMPERED", "SENSITIVE_DATA_RESIDUAL"}


def test_retention_and_deletion_receipts_fail_closed() -> None:
    policy = RetentionPolicy(
        policy_id="retention-1",
        tenant_id="tenant-a",
        maximum_retention_ms=100,
        deletion_grace_ms=20,
        legal_hold_maximum_ms=50,
    )
    receipt = build_deletion_receipt(
        artifact_id="artifact-1",
        tenant_id="tenant-a",
        requested_at_ms=105,
        completed_at_ms=110,
        tombstone_sha256=_digest("tombstone"),
        replicas_deleted=2,
    )
    deleted = ArtifactLifecycle(
        artifact_id="artifact-1",
        tenant_id="tenant-a",
        created_at_ms=0,
        expires_at_ms=100,
        deletion_requested_at_ms=105,
        deleted_at_ms=110,
        replicas_remaining=0,
        deletion_receipt=receipt,
    )
    assert validate_retention(policy, deleted, checked_at_ms=120).passed

    overdue = deleted.model_copy(
        update={
            "deleted_at_ms": None,
            "replicas_remaining": 1,
            "deletion_receipt": None,
        }
    )
    assert "DELETION_DEADLINE_MISSED" in _codes(
        validate_retention(policy, overdue, checked_at_ms=130)
    )


def test_capsule_checkpoint_and_artifact_payloads_are_sealed() -> None:
    payload = b"checkpoint bytes"
    manifest = build_artifact_manifest(
        artifact_id="checkpoint-1",
        tenant_id="tenant-a",
        kind=ArtifactKind.CHECKPOINT,
        payload=payload,
        producer_id="trainer",
        producer_version="1.0.0",
        created_at_ms=10,
        deterministic_seed=7,
    )
    assert validate_artifact_payload(
        manifest, payload, expected_tenant_id="tenant-a", checked_at_ms=11
    ).passed
    report = validate_artifact_payload(
        manifest, b"changed", expected_tenant_id="tenant-b", checked_at_ms=11
    )
    assert _codes(report) >= {
        "ARTIFACT_TENANT_MISMATCH",
        "ARTIFACT_SIZE_MISMATCH",
        "ARTIFACT_DIGEST_MISMATCH",
    }


def test_lineage_rejects_missing_parent_digest_and_cycles() -> None:
    left_payload = b"left"
    right_payload = b"right"
    left = build_artifact_manifest(
        artifact_id="left",
        tenant_id="tenant-a",
        kind=ArtifactKind.TRAJECTORY,
        payload=left_payload,
        producer_id="rollout",
        producer_version="1",
        created_at_ms=1,
        lineage=(LineageLink(artifact_id="right", content_sha256=_digest(right_payload)),),
    )
    right = build_artifact_manifest(
        artifact_id="right",
        tenant_id="tenant-a",
        kind=ArtifactKind.ARTIFACT,
        payload=right_payload,
        producer_id="capture",
        producer_version="1",
        created_at_ms=1,
        lineage=(LineageLink(artifact_id="left", content_sha256=_digest(left_payload)),),
    )
    assert "LINEAGE_CYCLE" in _codes(
        validate_artifact_graph((left, right), expected_tenant_id="tenant-a", checked_at_ms=2)
    )
    missing = left.model_copy(
        update={"lineage": (LineageLink(artifact_id="absent", content_sha256=_digest("x")),)}
    )
    assert _codes(
        validate_artifact_graph((missing,), expected_tenant_id="tenant-a", checked_at_ms=2)
    ) >= {"LINEAGE_PARENT_MISSING", "LINEAGE_MANIFEST_TAMPERED"}


def test_reward_claim_rejects_poisoned_evaluator_lineage_and_hidden_boundary() -> None:
    trajectory = build_artifact_manifest(
        artifact_id="trajectory-1",
        tenant_id="tenant-a",
        kind=ArtifactKind.TRAJECTORY,
        payload=b"trajectory",
        producer_id="rollout",
        producer_version="1",
        created_at_ms=1,
    )
    evidence = build_artifact_manifest(
        artifact_id="evidence-1",
        tenant_id="tenant-a",
        kind=ArtifactKind.ARTIFACT,
        payload=b"evidence",
        producer_id="verifier",
        producer_version="1",
        created_at_ms=1,
    )
    boundary_digest = _digest("hidden-boundary")
    component = RewardComponentClaim(
        component_id="hidden-1",
        source_type=RewardSourceType.HIDDEN_TEST,
        score=1.0,
        evidence_artifact_id=evidence.artifact_id,
        evidence_sha256=evidence.content_sha256,
        hidden_boundary_digest=boundary_digest,
    )
    evaluator = _digest("trusted-evaluator")
    claim = build_reward_claim(
        reward_id="reward-1",
        tenant_id="tenant-a",
        trajectory_id=trajectory.artifact_id,
        trajectory_sha256=trajectory.content_sha256,
        evaluator_sha256=evaluator,
        components=(component,),
    )
    artifacts = {trajectory.artifact_id: trajectory, evidence.artifact_id: evidence}
    assert validate_reward_claim(
        claim,
        artifacts=artifacts,
        expected_tenant_id="tenant-a",
        trusted_evaluator_digests={evaluator},
        trusted_hidden_boundary_digests={boundary_digest},
        checked_at_ms=2,
    ).passed
    poisoned = claim.model_copy(update={"trajectory_sha256": _digest("forged")})
    codes = _codes(
        validate_reward_claim(
            poisoned,
            artifacts=artifacts,
            expected_tenant_id="tenant-a",
            trusted_evaluator_digests=(),
            checked_at_ms=2,
        )
    )
    assert codes >= {
        "REWARD_CLAIM_TAMPERED",
        "REWARD_EVALUATOR_UNTRUSTED",
        "REWARD_TRAJECTORY_LINEAGE_INVALID",
        "HIDDEN_TEST_BOUNDARY_UNTRUSTED",
    }


def test_hidden_tests_require_disjoint_paths_and_safe_environment() -> None:
    boundary = build_hidden_test_boundary(
        suite_id="suite-1",
        tenant_id="tenant-a",
        policy_visible_roots=("/sandbox/source",),
        verifier_only_roots=("/verifier/hidden",),
        expected_value_sha256=(_digest("answer"),),
        policy_environment_keys=("LANG",),
    )
    assert validate_hidden_test_boundary(
        boundary, expected_tenant_id="tenant-a", checked_at_ms=1
    ).passed
    leaking = build_hidden_test_boundary(
        suite_id="suite-2",
        tenant_id="tenant-a",
        policy_visible_roots=("/sandbox",),
        verifier_only_roots=("/sandbox/hidden",),
        expected_value_sha256=(_digest("answer"),),
        policy_environment_keys=("HIDDEN_EXPECTED_VALUE",),
    )
    assert _codes(
        validate_hidden_test_boundary(leaking, expected_tenant_id="tenant-a", checked_at_ms=1)
    ) >= {"HIDDEN_TEST_PATH_LEAKAGE", "HIDDEN_TEST_ENVIRONMENT_LEAKAGE"}


def test_untrusted_execution_requires_external_effect_isolation() -> None:
    isolation = ExecutionIsolation(
        execution_id="reward-run-1",
        tenant_id="tenant-a",
        mode=ExecutionMode.REWARD,
        seed=9,
        network_isolation_enforced=True,
        filesystem_isolation_enforced=True,
        environment_sanitized=True,
        read_only_roots=("/sandbox/source",),
        writable_roots=("/sandbox-output",),
        wall_time_seconds=10.0,
        output_bytes=1024,
        process_count=4,
    )
    assert validate_execution_isolation(
        isolation, expected_tenant_id="tenant-a", checked_at_ms=1
    ).passed
    unsafe = isolation.model_copy(
        update={
            "network_isolation_enforced": False,
            "external_side_effects_enabled": True,
            "writable_roots": ("/sandbox/source/out",),
            "inherited_environment_keys": ("API_TOKEN",),
        }
    )
    assert _codes(
        validate_execution_isolation(unsafe, expected_tenant_id="tenant-a", checked_at_ms=1)
    ) >= {
        "NETWORK_ISOLATION_REQUIRED",
        "EXTERNAL_EFFECTS_NOT_ISOLATED",
        "ISOLATION_WRITE_OVERLAP",
        "CREDENTIAL_ENVIRONMENT_INHERITED",
    }


def test_malicious_repository_scan_never_invokes_git_and_rejects_active_content(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    (repository / ".git" / "hooks").mkdir(parents=True)
    (repository / ".git" / "config").write_text("[core]\nfsmonitor = attack\n")
    (repository / ".git" / "hooks" / "post-checkout").write_text("#!/bin/sh\nexit 1\n")
    (repository / ".gitattributes").write_text("*.dat filter=attack\n")
    (repository / "escape").symlink_to(tmp_path)
    evidence = scan_repository(
        repository,
        policy=RepositoryScanPolicy(
            max_entries=100,
            max_total_bytes=1024 * 1024,
            max_file_bytes=1024,
            max_depth=10,
        ),
    )
    codes = {item.code for item in evidence.findings}
    assert codes >= {
        "REPOSITORY_SYMLINK_REJECTED",
        "REPOSITORY_GIT_HOOK",
        "REPOSITORY_GIT_CONFIG_EXECUTION",
        "REPOSITORY_GIT_FILTER",
    }
    assert evidence.evidence_digest == canonical_digest(
        evidence.model_dump(mode="json", exclude={"evidence_digest"})
    )


def test_tool_output_is_bounded_normalized_and_redacted() -> None:
    secret = "tool-secret-value"
    raw = (
        b"\x1b[31mstatus\x1b[0m\x00 " + secret.encode() + b" person@example.test\xff" + b"x" * 4096
    )
    safe, evidence, redaction = sanitize_tool_output(raw, secret_values=(secret,), max_bytes=512)
    assert secret not in safe
    assert "person@example.test" not in safe
    assert "\x1b" not in safe and "\x00" not in safe
    assert evidence.truncated and evidence.retained_bytes <= 512
    assert validate_tool_output_evidence(
        evidence,
        raw_output=raw,
        sanitized_output=safe,
        redaction_evidence=redaction,
        checked_at_ms=1,
    ).passed


def test_submission_registry_rejects_replays_across_restarts_and_aliases(
    tmp_path: Path,
) -> None:
    database = tmp_path / "submissions.sqlite"
    values = {
        "tenant_id": "tenant-a",
        "namespace": "rewards",
        "submission_id": "reward-1",
        "content_sha256": _digest("submission"),
        "created_at_ms": 1,
    }
    with SubmissionRegistry(database) as registry:
        assert registry.register(**values) is SubmissionDisposition.ACCEPTED
        assert registry.register(**values) is SubmissionDisposition.DUPLICATE_ID
        with pytest.raises(SubmissionIdentityConflict):
            registry.register(**{**values, "content_sha256": _digest("changed")})
    with SubmissionRegistry(database) as registry:
        assert (
            registry.register(**{**values, "submission_id": "reward-alias"})
            is SubmissionDisposition.DUPLICATE_CONTENT
        )
        with pytest.raises(DuplicateSubmissionError):
            registry.require_new(**values)
        assert (
            registry.register(**{**values, "tenant_id": "tenant-b"})
            is SubmissionDisposition.ACCEPTED
        )


def test_adversarial_scenario_corpus_covers_every_security_boundary() -> None:
    path = Path("scenarios/helix/security/adversarial-cases.json")
    document = json.loads(path.read_text())
    controls = {case["control"] for case in document["cases"]}
    assert controls == {
        "production_capture_opt_in",
        "tenant_authorization",
        "secret_pii_redaction",
        "retention_deletion",
        "cross_tenant_reuse",
        "artifact_integrity",
        "malicious_repository",
        "tool_output",
        "reward_lineage",
        "hidden_test_boundary",
        "duplicate_submission",
        "external_effect_isolation",
    }
