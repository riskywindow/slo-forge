"""Trusted active-stream transition compatibility derived from capsule artifacts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from ..capsule import (
    ArtifactOrigin,
    ArtifactRole,
    ClaimCategory,
    EvidenceClass,
    EvidenceResult,
    GenesisCapsule,
    load_capsule,
)
from .models import CapsuleReference, TransitionCategory


@dataclass(frozen=True)
class TransitionCompatibility:
    compatible: bool
    behavior: str
    reason: str
    champion_recovery_digest: str
    challenger_recovery_digest: str
    state_conversion_digest: str | None = None


def _resolved_artifact(
    manifest_path: Path, capsule: GenesisCapsule, role: ArtifactRole
) -> tuple[bytes, str]:
    matches = [artifact for artifact in capsule.artifacts if artifact.role is role]
    if len(matches) != 1:
        raise ValueError(f"capsule must contain exactly one {role.value} artifact")
    artifact = matches[0]
    manifest_parent = manifest_path.parent.resolve(strict=True)
    root = manifest_parent.parent if manifest_parent.name == "manifests" else manifest_parent
    path = root.joinpath(*artifact.path.split("/"))
    if path.is_symlink():
        raise ValueError(f"{role.value} artifact must not be a symlink")
    resolved = path.resolve(strict=True)
    resolved.relative_to(root)
    if not resolved.is_file():
        raise ValueError(f"{role.value} artifact must be a regular file")
    payload = resolved.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if digest != artifact.digest.value:
        raise ValueError(f"{role.value} artifact digest changed after capsule validation")
    if role is ArtifactRole.ROLLBACK and artifact.origin is not ArtifactOrigin.TRUSTED:
        raise ValueError("rollback artifact must be issued by the trusted compiler boundary")
    if role is ArtifactRole.ROLLBACK and not any(
        record.evidence_class is EvidenceClass.OPERATIONAL
        and record.result is EvidenceResult.PASS
        and artifact.artifact_id in record.artifact_ids
        for record in capsule.evidence
    ):
        raise ValueError("rollback artifact is not bound to passing operational evidence")
    if role is ArtifactRole.STATE_CONVERSION and not any(
        record.result is EvidenceResult.PASS
        and artifact.artifact_id in record.artifact_ids
        for record in capsule.evidence
    ):
        raise ValueError("state conversion is not bound to passing verification evidence")
    return payload, digest


def _passed_claim(capsule: GenesisCapsule, category: ClaimCategory) -> bool:
    return any(
        claim.category is category
        and claim.result is EvidenceResult.PASS
        for claim in capsule.claims
    )


def _materialize(
    reference: CapsuleReference,
) -> tuple[Path, GenesisCapsule, dict[str, object], str]:
    manifest_path = Path(reference.path).resolve(strict=True)
    capsule = load_capsule(manifest_path)
    if capsule.capsule_digest is None or capsule.capsule_digest.value != reference.capsule_digest:
        raise ValueError("materialized capsule digest does not match the controller reference")
    if capsule.identity.candidate_genome_hash.value != reference.genome_hash:
        raise ValueError("materialized capsule genome does not match the controller reference")
    if not _passed_claim(capsule, ClaimCategory.SEMANTIC):
        raise ValueError("capsule lacks a passing, scoped semantic state claim")
    if not _passed_claim(capsule, ClaimCategory.OPERATIONAL):
        raise ValueError("capsule lacks a passing, scoped recovery protocol claim")
    rollback_payload, rollback_digest = _resolved_artifact(
        manifest_path, capsule, ArtifactRole.ROLLBACK
    )
    rollback = json.loads(rollback_payload)
    if not isinstance(rollback, dict):
        raise ValueError("rollback artifact must be an object")
    if rollback.get("action") != "restore-previous-champion":
        raise ValueError("rollback artifact does not restore the prior champion")
    return manifest_path, capsule, rollback, rollback_digest


def validate_transition_compatibility(
    champion: CapsuleReference,
    challenger: CapsuleReference,
    category: TransitionCategory,
) -> TransitionCompatibility:
    """Determine active-stream behavior without trusting proposal compatibility flags."""

    _, champion_capsule, champion_rollback, champion_recovery_digest = _materialize(champion)
    challenger_path, challenger_capsule, challenger_rollback, challenger_recovery_digest = (
        _materialize(challenger)
    )
    rollback_preserves = (
        champion_rollback.get("preserve_active_streams") is True
        and challenger_rollback.get("preserve_active_streams") is True
    )
    pinned_categories = {
        TransitionCategory.POLICY_ONLY_HOT_SWAP,
        TransitionCategory.REQUEST_BOUNDARY_SWAP,
        TransitionCategory.NEW_REPLICA,
    }
    if category in pinned_categories and rollback_preserves:
        return TransitionCompatibility(
            compatible=True,
            behavior="pin_existing_streams",
            reason=(
                "digest-bound rollback artifacts preserve existing leases; only new requests "
                "are assigned to the challenger"
            ),
            champion_recovery_digest=champion_recovery_digest,
            challenger_recovery_digest=challenger_recovery_digest,
        )

    same_state_identity = (
        champion_capsule.identity.source_model_hash == challenger_capsule.identity.source_model_hash
        and champion_capsule.identity.tokenizer_hash == challenger_capsule.identity.tokenizer_hash
    )
    deployment_payload, _ = _resolved_artifact(
        challenger_path, challenger_capsule, ArtifactRole.DEPLOYMENT
    )
    deployment = json.loads(deployment_payload)
    declared_transition = deployment.get("transition") if isinstance(deployment, dict) else None
    if (
        category is TransitionCategory.STATE_COMPATIBLE_MIGRATION
        and same_state_identity
        and declared_transition == "state-compatible"
        and rollback_preserves
    ):
        return TransitionCompatibility(
            compatible=True,
            behavior="state_compatible_migration",
            reason="state identity and digest-bound migration/recovery contracts agree",
            champion_recovery_digest=champion_recovery_digest,
            challenger_recovery_digest=challenger_recovery_digest,
        )
    if category is TransitionCategory.STATE_CONVERSION_MIGRATION:
        conversion = [
            artifact
            for artifact in challenger_capsule.artifacts
            if artifact.role is ArtifactRole.STATE_CONVERSION
        ]
        if (
            len(conversion) == 1
            and same_state_identity
            and declared_transition == "state-conversion"
            and rollback_preserves
        ):
            _, conversion_digest = _resolved_artifact(
                challenger_path, challenger_capsule, ArtifactRole.STATE_CONVERSION
            )
            return TransitionCompatibility(
                compatible=True,
                behavior="state_conversion_migration",
                reason="digest-bound state conversion and rollback artifacts agree",
                champion_recovery_digest=champion_recovery_digest,
                challenger_recovery_digest=challenger_recovery_digest,
                state_conversion_digest=conversion_digest,
            )
    return TransitionCompatibility(
        compatible=False,
        behavior="drain_required",
        reason=(
            "independently materialized capsule state/recovery artifacts do not authorize "
            f"{category.value} with active streams"
        ),
        champion_recovery_digest=champion_recovery_digest,
        challenger_recovery_digest=challenger_recovery_digest,
    )
