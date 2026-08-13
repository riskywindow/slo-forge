"""Offline authorization seal for BranchFabric Experiment 004 v10.

The live GPU controller must not wait for reviewers.  This module builds and
verifies a content-addressed authorization whose decisions are complete before
the Modal allocation starts.  It intentionally performs no network or GPU
operations.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

AUTHORIZATION_SCHEMA: Final = "sloforge.branchfabric.experiment-004-v10-authorization/v1"
SUBJECT_SCHEMA: Final = "sloforge.branchfabric.experiment-004-v10-authorization-subject/v1"
REVIEW_SCHEMA: Final = "sloforge.branchfabric.experiment-004-authorization-review/v1"
HASH_CONVENTION: Final = "sha256(canonical-json-utf8-with-lf, excluding top-level artifact_hash)"
REQUIRED_REVIEW_ROLES: Final = (
    "capacity_methodology",
    "queue_drain_methodology",
    "scientific_validity",
)
SOURCE_ROLES: Final = (
    "selected_load",
    "calibration_result",
    "lambda_1_plan",
    "lambda_1_raw",
    "lambda_1_result",
    "lambda_spike_plan",
    "lambda_spike_raw",
    "lambda_spike_result",
    "lambda_2_plan",
    "lambda_2_raw",
    "lambda_2_result",
)
SOURCE_REFERENCES: Final = (
    "artifacts/branchfabric/gpu-validation/experiment-004/raw/modal/"
    "exp004-v10-integrated-s41-v3/calibration/selected-load.json",
    "artifacts/branchfabric/gpu-validation/experiment-004/raw/modal/"
    "exp004-v10-integrated-s41-v3/calibration/calibration-result.json",
    "artifacts/branchfabric/gpu-validation/experiment-004/raw/modal/"
    "exp004-v10-integrated-s41-v3/calibration/one-gpu/"
    "capacity-04-gpu0-only/plan.json",
    "artifacts/branchfabric/gpu-validation/experiment-004/raw/modal/"
    "exp004-v10-integrated-s41-v3/calibration/one-gpu/"
    "capacity-04-gpu0-only/raw.json",
    "artifacts/branchfabric/gpu-validation/experiment-004/raw/modal/"
    "exp004-v10-integrated-s41-v3/calibration/one-gpu/"
    "capacity-04-gpu0-only/result.json",
    "artifacts/branchfabric/gpu-validation/experiment-004/raw/modal/"
    "exp004-v10-integrated-s41-v3/calibration/one-gpu/"
    "capacity-02-gpu0-only/plan.json",
    "artifacts/branchfabric/gpu-validation/experiment-004/raw/modal/"
    "exp004-v10-integrated-s41-v3/calibration/one-gpu/"
    "capacity-02-gpu0-only/raw.json",
    "artifacts/branchfabric/gpu-validation/experiment-004/raw/modal/"
    "exp004-v10-integrated-s41-v3/calibration/one-gpu/"
    "capacity-02-gpu0-only/result.json",
    "artifacts/branchfabric/gpu-validation/experiment-004/raw/modal/"
    "exp004-v10-integrated-s41-v3/calibration/two-gpu/"
    "capacity-05-two-gpu-round-robin/plan.json",
    "artifacts/branchfabric/gpu-validation/experiment-004/raw/modal/"
    "exp004-v10-integrated-s41-v3/calibration/two-gpu/"
    "capacity-05-two-gpu-round-robin/raw.json",
    "artifacts/branchfabric/gpu-validation/experiment-004/raw/modal/"
    "exp004-v10-integrated-s41-v3/calibration/two-gpu/"
    "capacity-05-two-gpu-round-robin/result.json",
)
_SUBJECT_FIELDS: Final = (
    "subject_schema_version",
    "experiment",
    "experiment_version",
    "calibration_attempt_id",
    "seed",
    "lambda_1_rps",
    "lambda_spike_rps",
    "lambda_2_rps",
    "one_gpu_margin_pct",
    "two_gpu_reserve_pct",
    "nominal_drain_headroom_rps",
    "calibration_evidence",
    "queue_drain_methodology",
    "source_calibration_artifact_hashes",
    "code_commit",
    "live_path_policy",
)
_AUTHORIZATION_FIELDS: Final = frozenset(
    {
        *_SUBJECT_FIELDS,
        "schema_version",
        "authorization_subject_sha256",
        "reviewer_decisions",
        "approval_timestamp",
        "hash_convention",
        "status",
        "artifact_hash",
    }
)


class AuthorizationError(ValueError):
    """The offline authorization or one of its content references is invalid."""


def canonical_json_bytes(value: Any) -> bytes:
    """Return the repository's deterministic JSON representation."""

    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n"
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def authorization_subject(payload: Mapping[str, Any]) -> dict[str, Any]:
    missing = set(_SUBJECT_FIELDS) - set(payload)
    if missing:
        raise AuthorizationError(f"authorization subject fields are missing: {sorted(missing)}")
    return {field: payload[field] for field in _SUBJECT_FIELDS}


def authorization_subject_sha256(payload: Mapping[str, Any]) -> str:
    return canonical_sha256(authorization_subject(payload))


def authorization_artifact_sha256(payload: Mapping[str, Any]) -> str:
    body = dict(payload)
    body.pop("artifact_hash", None)
    return canonical_sha256(body)


def _settled_calibration_evidence() -> dict[str, Any]:
    return {
        "lambda_1": {
            "probe_id": "capacity-04-gpu0-only",
            "completed_rate_rps": 11.9,
            "p95_ttft_ms": 44.20256505,
            "queue_depth_start": 11,
            "queue_depth_end": 12,
            "queue_slope_requests_per_second": 0.045454545454545456,
            "persistent_positive_drift": False,
            "verdict": "sustainable",
        },
        "lambda_spike_one_gpu": {
            "probe_id": "capacity-02-gpu0-only",
            "completed_rate_rps": 14.4,
            "p95_ttft_ms": 402.28929905,
            "queue_depth_start": 15,
            "queue_depth_end": 21,
            "queue_slope_requests_per_second": 0.5818181818181818,
            "persistent_positive_drift": True,
            "verdict": "unsustainable",
        },
        "lambda_2": {
            "probe_id": "capacity-05-two-gpu-round-robin",
            "completed_rate_rps": 19.9,
            "p95_ttft_ms": 42.4444484,
            "queue_depth_start": 18,
            "queue_depth_end": 19,
            "queue_slope_requests_per_second": 0.045454545454545456,
            "persistent_positive_drift": False,
            "gpu0_measurement_completions": 100,
            "gpu1_measurement_completions": 100,
            "verdict": "sustainable",
        },
    }


def _settled_queue_drain_methodology() -> dict[str, Any]:
    return {
        "offered_rate_rps": 15,
        "reference_two_gpu_service_rate_rps": 20,
        "acceptance_service_rate": "observed aggregate completed service rate > 15 rps",
        "acceptance_queue_behavior": "sustained negative queue slope and material drain",
        "predicted_drain_seconds": "trigger backlog / observed drain headroom",
        "bounded_trigger_queue_depth_requests": {"minimum": 10, "maximum": 25},
        "slo_stability_window_seconds": 5,
    }


def build_authorization_subject(
    *,
    code_commit: str,
    source_calibration_artifact_hashes: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build the settled v10 load-selection subject; no approval is implied."""

    subject: dict[str, Any] = {
        "subject_schema_version": SUBJECT_SCHEMA,
        "experiment": "branchfabric-gpu-validation-experiment-004",
        "experiment_version": 10,
        "calibration_attempt_id": "exp004-v10-integrated-s41-v3",
        "seed": 41,
        "lambda_1_rps": 12,
        "lambda_spike_rps": 15,
        "lambda_2_rps": 20,
        "one_gpu_margin_pct": 25,
        "two_gpu_reserve_pct": 25,
        "nominal_drain_headroom_rps": 5,
        "calibration_evidence": _settled_calibration_evidence(),
        "queue_drain_methodology": _settled_queue_drain_methodology(),
        "source_calibration_artifact_hashes": [
            dict(item) for item in source_calibration_artifact_hashes
        ],
        "code_commit": code_commit,
        "live_path_policy": {
            "reviews_complete_before_gpu_allocation": True,
            "remote_review_exchange_permitted": False,
            "asynchronous_review_wait_permitted": False,
            "controller_actions": [
                "load authorization artifact",
                "verify artifact hash",
                "verify status APPROVED",
                "verify source evidence hashes",
                "proceed",
            ],
        },
    }
    _validate_subject_values(subject)
    return subject


def seal_authorization(
    *, subject: Mapping[str, Any], reviews: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Seal a final APPROVED artifact from three already-completed reviews."""

    if set(subject) != set(_SUBJECT_FIELDS):
        raise AuthorizationError("authorization subject fields differ from the exact contract")
    subject_payload = authorization_subject(subject)
    _validate_subject_values(subject_payload)
    subject_digest = canonical_sha256(subject_payload)
    normalized_reviews = [dict(review) for review in reviews]
    _validate_review_payloads(normalized_reviews, subject_digest)
    approval_timestamp = str(
        max(normalized_reviews, key=lambda review: _parse_utc(str(review["decided_at_utc"])))[
            "decided_at_utc"
        ]
    )
    payload = {
        "schema_version": AUTHORIZATION_SCHEMA,
        **subject_payload,
        "authorization_subject_sha256": subject_digest,
        "reviewer_decisions": normalized_reviews,
        "approval_timestamp": approval_timestamp,
        "hash_convention": HASH_CONVENTION,
        "status": "APPROVED",
    }
    payload["artifact_hash"] = authorization_artifact_sha256(payload)
    return payload


def verify_authorization(
    payload: Mapping[str, Any], *, repository_root: Path, verify_review_artifacts: bool = True
) -> dict[str, Any]:
    """Fail closed unless the complete local authorization seal is valid."""

    if payload.get("schema_version") != AUTHORIZATION_SCHEMA:
        raise AuthorizationError("authorization schema is not the v10 schema")
    if set(payload) != _AUTHORIZATION_FIELDS:
        raise AuthorizationError("authorization fields differ from the exact contract")
    if payload.get("status") != "APPROVED":
        raise AuthorizationError("authorization status is not APPROVED")
    if payload.get("hash_convention") != HASH_CONVENTION:
        raise AuthorizationError("authorization hash convention is unsupported")
    _validate_subject_values(payload)
    subject_digest = authorization_subject_sha256(payload)
    if payload.get("authorization_subject_sha256") != subject_digest:
        raise AuthorizationError("authorization subject hash does not match")
    expected_artifact_hash = authorization_artifact_sha256(payload)
    if payload.get("artifact_hash") != expected_artifact_hash:
        raise AuthorizationError("authorization artifact hash does not match")
    _verify_source_artifacts(payload["source_calibration_artifact_hashes"], repository_root)
    reviews = payload.get("reviewer_decisions")
    if not isinstance(reviews, list):
        raise AuthorizationError("reviewer_decisions must be a JSON array")
    _validate_review_payloads(reviews, subject_digest)
    expected_approval_timestamp = str(
        max(reviews, key=lambda review: _parse_utc(str(review["decided_at_utc"])))["decided_at_utc"]
    )
    if payload.get("approval_timestamp") != expected_approval_timestamp:
        raise AuthorizationError("approval_timestamp is not the final review timestamp")
    if verify_review_artifacts:
        _verify_review_artifacts(reviews, repository_root, subject_digest)
    return dict(payload)


def verify_authorization_file(
    path: Path,
    *,
    repository_root: Path,
    expected_artifact_hash: str | None = None,
    verify_review_artifacts: bool = True,
) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise AuthorizationError("authorization must be a regular non-symlink file")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise AuthorizationError("authorization is not readable canonical JSON") from error
    if not isinstance(payload, dict):
        raise AuthorizationError("authorization must contain one JSON object")
    verified = verify_authorization(
        payload,
        repository_root=repository_root,
        verify_review_artifacts=verify_review_artifacts,
    )
    if expected_artifact_hash is not None and verified["artifact_hash"] != expected_artifact_hash:
        raise AuthorizationError("authorization differs from the controller-pinned hash")
    return verified


def _validate_subject_values(payload: Mapping[str, Any]) -> None:
    expected_scalars = {
        "subject_schema_version": SUBJECT_SCHEMA,
        "experiment": "branchfabric-gpu-validation-experiment-004",
        "experiment_version": 10,
        "calibration_attempt_id": "exp004-v10-integrated-s41-v3",
        "seed": 41,
        "lambda_1_rps": 12,
        "lambda_spike_rps": 15,
        "lambda_2_rps": 20,
        "one_gpu_margin_pct": 25,
        "two_gpu_reserve_pct": 25,
        "nominal_drain_headroom_rps": 5,
    }
    for field, expected in expected_scalars.items():
        if payload.get(field) != expected or isinstance(payload.get(field), bool):
            raise AuthorizationError(f"authorization {field} differs from settled v10 design")
    commit = payload.get("code_commit")
    if (
        not isinstance(commit, str)
        or len(commit) != 40
        or any(character not in "0123456789abcdef" for character in commit)
    ):
        raise AuthorizationError("code_commit must be a full lowercase Git object ID")
    source_rows = payload.get("source_calibration_artifact_hashes")
    if not isinstance(source_rows, list):
        raise AuthorizationError("source calibration hashes must be a JSON array")
    if tuple(row.get("role") for row in source_rows if isinstance(row, dict)) != SOURCE_ROLES:
        raise AuthorizationError("source calibration roles are incomplete or out of order")
    if len(source_rows) != len(SOURCE_ROLES):
        raise AuthorizationError("source calibration evidence count differs from contract")
    if (
        tuple(row.get("artifact_reference") for row in source_rows if isinstance(row, dict))
        != SOURCE_REFERENCES
    ):
        raise AuthorizationError("source calibration references differ from settled evidence")
    if payload.get("calibration_evidence") != _settled_calibration_evidence():
        raise AuthorizationError("calibration evidence values differ from settled measurements")
    if payload.get("queue_drain_methodology") != _settled_queue_drain_methodology():
        raise AuthorizationError("queue-drain methodology differs from settled v10 design")
    policy = payload.get("live_path_policy")
    if (
        not isinstance(policy, dict)
        or policy.get("reviews_complete_before_gpu_allocation") is not True
    ):
        raise AuthorizationError("authorization does not require offline reviews")
    if policy.get("remote_review_exchange_permitted") is not False:
        raise AuthorizationError("authorization permits a remote review exchange")
    if policy.get("asynchronous_review_wait_permitted") is not False:
        raise AuthorizationError("authorization permits an asynchronous review wait")


def _validate_review_payloads(reviews: Sequence[Mapping[str, Any]], subject_digest: str) -> None:
    if len(reviews) != len(REQUIRED_REVIEW_ROLES):
        raise AuthorizationError("exactly three independent reviewer decisions are required")
    roles: list[str] = []
    identities: list[str] = []
    for review in reviews:
        required = {
            "schema_version",
            "reviewer_id",
            "reviewer_role",
            "decision",
            "decided_at_utc",
            "authorization_subject_sha256",
            "source_evidence_hashes_verified",
            "findings",
            "artifact_reference",
            "artifact_sha256",
        }
        if set(review) != required:
            raise AuthorizationError("review decision fields differ from the exact contract")
        if review["schema_version"] != REVIEW_SCHEMA:
            raise AuthorizationError("review decision schema is invalid")
        if review["decision"] != "APPROVED":
            raise AuthorizationError("every authorization reviewer must approve")
        if review["authorization_subject_sha256"] != subject_digest:
            raise AuthorizationError("review decision addresses a different authorization subject")
        if review["source_evidence_hashes_verified"] is not True:
            raise AuthorizationError("reviewer did not verify source evidence hashes")
        if not isinstance(review["findings"], list):
            raise AuthorizationError("review findings must be a JSON array")
        _parse_utc(str(review["decided_at_utc"]))
        roles.append(str(review["reviewer_role"]))
        identities.append(str(review["reviewer_id"]))
    if tuple(roles) != REQUIRED_REVIEW_ROLES:
        raise AuthorizationError("review roles are incomplete or out of order")
    if len(set(identities)) != len(identities) or any(not identity for identity in identities):
        raise AuthorizationError("authorization reviewers must have unique non-empty identities")


def _safe_repo_file(repository_root: Path, reference: str) -> Path:
    relative = Path(reference)
    if relative.is_absolute() or ".." in relative.parts:
        raise AuthorizationError("artifact reference escapes the repository")
    root = repository_root.resolve(strict=True)
    candidate = root
    for part in relative.parts:
        candidate /= part
        if candidate.is_symlink():
            raise AuthorizationError("artifact reference traverses a symlink")
    path = candidate.resolve(strict=True)
    if not path.is_relative_to(root) or not path.is_file():
        raise AuthorizationError("artifact reference is not a regular repository file")
    return path


def _verify_source_artifacts(rows: Sequence[Mapping[str, Any]], repository_root: Path) -> None:
    for row in rows:
        if set(row) != {"role", "artifact_reference", "artifact_sha256"}:
            raise AuthorizationError("source calibration reference fields differ from contract")
        expected_hash = row["artifact_sha256"]
        if not isinstance(expected_hash, str) or len(expected_hash) != 64:
            raise AuthorizationError("source calibration hash is malformed")
        path = _safe_repo_file(repository_root, str(row["artifact_reference"]))
        if file_sha256(path) != expected_hash:
            raise AuthorizationError(f"source calibration hash mismatch for {row['role']}")


def _verify_review_artifacts(
    reviews: Sequence[Mapping[str, Any]], repository_root: Path, subject_digest: str
) -> None:
    for decision in reviews:
        path = _safe_repo_file(repository_root, str(decision["artifact_reference"]))
        if file_sha256(path) != decision["artifact_sha256"]:
            raise AuthorizationError(f"review artifact hash mismatch for {decision['reviewer_id']}")
        try:
            artifact = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise AuthorizationError("review artifact is not readable JSON") from error
        embedded = {
            key: value
            for key, value in decision.items()
            if key not in {"artifact_reference", "artifact_sha256"}
        }
        if artifact != embedded:
            raise AuthorizationError("embedded reviewer decision differs from its artifact")
        if artifact.get("authorization_subject_sha256") != subject_digest:
            raise AuthorizationError("review artifact addresses a different subject")


def _parse_utc(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise AuthorizationError("review timestamp is not ISO-8601") from error
    if parsed.tzinfo is None or parsed.utcoffset() != UTC.utcoffset(parsed):
        raise AuthorizationError("review timestamp must carry UTC")
    return parsed
