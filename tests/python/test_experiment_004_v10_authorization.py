from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from sloforge.helix.characterization.gpu_reclamation_v10_authorization import (
    HASH_CONVENTION,
    REQUIRED_REVIEW_ROLES,
    SOURCE_REFERENCES,
    SOURCE_ROLES,
    AuthorizationError,
    authorization_artifact_sha256,
    authorization_subject_sha256,
    build_authorization_subject,
    canonical_json_bytes,
    seal_authorization,
    verify_authorization,
)

COMMIT = "d5e581476404c60a4743a8c8b03ce185d5cb2b00"


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(value))


def _subject(repository_root: Path) -> dict[str, object]:
    sources = []
    for role, reference in zip(SOURCE_ROLES, SOURCE_REFERENCES, strict=True):
        path = repository_root / reference
        _write_json(path, {"role": role, "measurement": role})
        sources.append(
            {
                "role": role,
                "artifact_reference": reference,
                "artifact_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    return build_authorization_subject(
        code_commit=COMMIT,
        source_calibration_artifact_hashes=sources,
    )


def _reviews(repository_root: Path, subject_hash: str) -> list[dict[str, object]]:
    reviews = []
    for index, role in enumerate(REQUIRED_REVIEW_ROLES):
        review = {
            "schema_version": ("sloforge.branchfabric.experiment-004-authorization-review/v1"),
            "reviewer_id": f"reviewer-{index}",
            "reviewer_role": role,
            "decision": "APPROVED",
            "decided_at_utc": f"2026-08-13T04:0{index}:00Z",
            "authorization_subject_sha256": subject_hash,
            "source_evidence_hashes_verified": True,
            "findings": [],
        }
        reference = f"reviews/reviewer-{index}.json"
        path = repository_root / reference
        _write_json(path, review)
        reviews.append(
            {
                **review,
                "artifact_reference": reference,
                "artifact_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    return reviews


def _authorization(repository_root: Path) -> dict[str, object]:
    subject = _subject(repository_root)
    return seal_authorization(
        subject=subject,
        reviews=_reviews(repository_root, authorization_subject_sha256(subject)),
    )


def test_sealed_authorization_verifies_all_content_addresses(tmp_path: Path) -> None:
    authorization = _authorization(tmp_path)

    verified = verify_authorization(authorization, repository_root=tmp_path)

    assert verified["status"] == "APPROVED"
    assert verified["lambda_1_rps"] == 12
    assert verified["lambda_spike_rps"] == 15
    assert verified["lambda_2_rps"] == 20
    assert verified["one_gpu_margin_pct"] == 25
    assert verified["two_gpu_reserve_pct"] == 25
    assert verified["nominal_drain_headroom_rps"] == 5
    assert verified["hash_convention"] == HASH_CONVENTION
    assert verified["artifact_hash"] == authorization_artifact_sha256(verified)
    assert verified["approval_timestamp"] == "2026-08-13T04:02:00Z"


def test_authorization_fails_closed_on_source_evidence_mutation(tmp_path: Path) -> None:
    authorization = _authorization(tmp_path)
    first_source = authorization["source_calibration_artifact_hashes"][0]
    source_path = tmp_path / first_source["artifact_reference"]
    source_path.write_text("tampered\n", encoding="utf-8")

    with pytest.raises(AuthorizationError, match="source calibration hash mismatch"):
        verify_authorization(authorization, repository_root=tmp_path)


def test_authorization_rejects_symlinked_source_evidence(tmp_path: Path) -> None:
    authorization = _authorization(tmp_path)
    first_source = authorization["source_calibration_artifact_hashes"][0]
    source_path = tmp_path / first_source["artifact_reference"]
    real_path = source_path.with_name("real-selected-load.json")
    source_path.rename(real_path)
    source_path.symlink_to(real_path.name)

    with pytest.raises(AuthorizationError, match="traverses a symlink"):
        verify_authorization(authorization, repository_root=tmp_path)


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("status", "PENDING", "not APPROVED"),
        ("lambda_spike_rps", 14, "settled v10 design"),
        ("artifact_hash", "0" * 64, "artifact hash does not match"),
        ("unreviewed_extension", True, "fields differ from the exact contract"),
    ],
)
def test_authorization_fails_closed_on_tampering(
    tmp_path: Path, field: str, replacement: object, message: str
) -> None:
    authorization = _authorization(tmp_path)
    authorization[field] = replacement

    with pytest.raises(AuthorizationError, match=message):
        verify_authorization(authorization, repository_root=tmp_path)


def test_authorization_requires_three_unique_approved_reviewers(tmp_path: Path) -> None:
    subject = _subject(tmp_path)
    subject_hash = authorization_subject_sha256(subject)
    reviews = _reviews(tmp_path, subject_hash)
    reviews[1]["reviewer_id"] = reviews[0]["reviewer_id"]

    with pytest.raises(AuthorizationError, match="unique non-empty identities"):
        seal_authorization(subject=subject, reviews=reviews)

    reviews = _reviews(tmp_path / "second", subject_hash)
    reviews[2]["decision"] = "REJECTED"
    with pytest.raises(AuthorizationError, match="must approve"):
        seal_authorization(subject=subject, reviews=reviews)


def test_authorization_rejects_live_review_handoff(tmp_path: Path) -> None:
    subject = _subject(tmp_path)
    subject["live_path_policy"]["asynchronous_review_wait_permitted"] = True
    subject_hash = authorization_subject_sha256(subject)

    with pytest.raises(AuthorizationError, match="asynchronous review wait"):
        seal_authorization(
            subject=subject,
            reviews=_reviews(tmp_path, subject_hash),
        )
