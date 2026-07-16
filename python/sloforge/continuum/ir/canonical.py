"""Canonical serialization, capsule sealing, and independent integrity checks."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from pydantic import BaseModel

from .models import (
    CapsuleIdentity,
    CapsuleIntegrity,
    CapsuleTransactionBinding,
    CapsuleType,
    CompatibilityConstraints,
    Digest,
    ExecutionStateCapsule,
    Extensions,
    LogicalStateSchema,
    MigrationVerificationEvidence,
    PhysicalStateLayout,
    SegmentManifest,
)

ZERO_SHA256 = "0" * 64


class CapsuleValidationError(ValueError):
    """An integrity, ownership, or manifest validation failure."""


def _json_value(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    return value


def canonical_json(document: BaseModel | dict[str, Any] | tuple[Any, ...]) -> bytes:
    """Encode the cross-language canonical JSON profile used by SLOForge."""

    value = _json_value(document)
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def canonical_hash(document: BaseModel | dict[str, Any] | tuple[Any, ...]) -> str:
    return hashlib.sha256(canonical_json(document)).hexdigest()


def digest(document: BaseModel | dict[str, Any] | tuple[Any, ...]) -> Digest:
    return Digest(value=canonical_hash(document))


def _integrity(
    identity: CapsuleIdentity,
    logical_state: LogicalStateSchema,
    physical_state: PhysicalStateLayout,
    segment_manifests: tuple[SegmentManifest, ...],
    compatibility: CompatibilityConstraints,
    transaction: CapsuleTransactionBinding,
    evidence: MigrationVerificationEvidence,
    extensions: Extensions,
) -> CapsuleIntegrity:
    identity_hash = digest(identity.model_dump(mode="json", exclude={"capsule_id"}))
    logical_hash = digest(logical_state)
    physical_hash = digest(physical_state)
    manifests_hash = digest(segment_manifests)
    compatibility_hash = digest(compatibility)
    transaction_hash = digest(transaction)
    evidence_hash = digest(evidence)
    extensions_hash = digest(extensions)
    root_input = {
        "compatibility_hash": compatibility_hash.value,
        "evidence_hash": evidence_hash.value,
        "extensions_hash": extensions_hash.value,
        "identity_hash": identity_hash.value,
        "logical_state_hash": logical_hash.value,
        "physical_layout_hash": physical_hash.value,
        "segment_manifests_hash": manifests_hash.value,
        "transaction_binding_hash": transaction_hash.value,
    }
    return CapsuleIntegrity(
        identity_hash=identity_hash,
        logical_state_hash=logical_hash,
        physical_layout_hash=physical_hash,
        segment_manifests_hash=manifests_hash,
        compatibility_hash=compatibility_hash,
        transaction_binding_hash=transaction_hash,
        evidence_hash=evidence_hash,
        extensions_hash=extensions_hash,
        merkle_root=digest(root_input),
    )


def build_capsule(
    *,
    capsule_type: CapsuleType,
    logical_state: LogicalStateSchema,
    physical_state: PhysicalStateLayout,
    segment_manifests: tuple[SegmentManifest, ...],
    compatibility: CompatibilityConstraints,
    transaction: CapsuleTransactionBinding,
    evidence: MigrationVerificationEvidence,
    capture_timestamp: str,
    git_commit: str,
    continuum_version: str,
    parent_capsule_id: str | None = None,
    extensions: Extensions | None = None,
) -> ExecutionStateCapsule:
    """Seal typed state parts into a content-addressed capsule.

    The capsule ID is the Merkle root over logical state, physical layout,
    manifests, compatibility constraints, transaction binding, and verification
    evidence. Transaction ownership consistency is checked before return.
    """

    capsule_extensions = extensions if extensions is not None else Extensions(root={})
    provisional_identity = CapsuleIdentity(
        capsule_id=ZERO_SHA256,
        capsule_type=capsule_type,
        session_id=logical_state.execution.session_id,
        tenant_id=logical_state.execution.tenant_id,
        model_hash=logical_state.execution.model_identity,
        tokenizer_hash=logical_state.execution.tokenizer_identity,
        adapter_hash=logical_state.execution.adapter_identity,
        source_runtime=physical_state.runtime,
        source_physical_plan=physical_state.physical_plan_hash,
        owner_epoch=logical_state.execution.current_owner_epoch,
        capture_timestamp=capture_timestamp,
        git_commit=git_commit,
        continuum_version=continuum_version,
        parent_capsule_id=parent_capsule_id,
    )
    integrity = _integrity(
        provisional_identity,
        logical_state,
        physical_state,
        segment_manifests,
        compatibility,
        transaction,
        evidence,
        capsule_extensions,
    )
    identity = provisional_identity.model_copy(update={"capsule_id": integrity.merkle_root.value})
    capsule = ExecutionStateCapsule(
        identity=identity,
        logical_state=logical_state,
        physical_state=physical_state,
        segment_manifests=segment_manifests,
        compatibility=compatibility,
        transaction=transaction,
        evidence=evidence,
        integrity=integrity,
        extensions=capsule_extensions,
    )
    validate_capsule(capsule)
    return capsule


def validate_capsule(capsule: ExecutionStateCapsule) -> None:
    """Independently recompute all capsule integrity commitments."""

    expected = _integrity(
        capsule.identity,
        capsule.logical_state,
        capsule.physical_state,
        capsule.segment_manifests,
        capsule.compatibility,
        capsule.transaction,
        capsule.evidence,
        capsule.extensions,
    )
    if capsule.integrity != expected:
        raise CapsuleValidationError("capsule Merkle integrity mismatch")
    if capsule.identity.capsule_id != expected.merkle_root.value:
        raise CapsuleValidationError("capsule ID does not match Merkle root")
    manifests = {manifest.segment_id: manifest for manifest in capsule.segment_manifests}
    for segment in capsule.physical_state.segments:
        manifest = manifests[segment.segment_id]
        if manifest.segment_hash != segment.checksum:
            raise CapsuleValidationError(f"altered segment manifest for {segment.segment_id}")
    if capsule.transaction.ownership_lease.session_id != capsule.identity.session_id:
        raise CapsuleValidationError("ownership lease names a different session")
    if (
        capsule.transaction.ownership_lease.last_committed_token_index
        != capsule.transaction.commit_watermark
    ):
        raise CapsuleValidationError("lease token watermark does not match transaction")
    if capsule.transaction.rollback_boundary > capsule.transaction.commit_watermark:
        raise CapsuleValidationError("rollback boundary exceeds commit watermark")
    if capsule.transaction.commit_watermark != (
        capsule.logical_state.client_delivery.last_gateway_committed_token_index
    ):
        raise CapsuleValidationError("transaction commit watermark does not match client state")
