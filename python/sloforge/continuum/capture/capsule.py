"""Publish exact adapter bytes and seal a proof-carrying execution capsule."""

from __future__ import annotations

from dataclasses import dataclass

from sloforge.continuum.adapters import CapturedState, captured_to_capsule_inputs
from sloforge.continuum.ir import (
    CapsuleTransactionBinding,
    CapsuleType,
    CompatibilityConstraints,
    CompressionKind,
    ConversionPermission,
    Digest,
    EncryptionKind,
    ExecutionStateCapsule,
    ExternalChunkReference,
    MigrationVerificationEvidence,
    OwnershipLease,
    RecomputationPermission,
    SegmentManifest,
    VerificationClaim,
    build_capsule,
    canonical_hash,
)
from sloforge.continuum.storage import ChunkRef, ContentStore, StoredManifest
from sloforge.continuum.transaction import JournalEntry, SessionLease, StateTransactionRecord


@dataclass(frozen=True, slots=True)
class PublishedCapsule:
    capsule: ExecutionStateCapsule
    store_manifest: StoredManifest
    chunk_references: tuple[ChunkRef, ...]


def _segment_manifests(
    inputs_manifests: tuple[SegmentManifest, ...],
    references: tuple[ChunkRef, ...],
) -> tuple[SegmentManifest, ...]:
    by_digest = {reference.digest: reference for reference in references}
    manifests: list[SegmentManifest] = []
    for manifest in inputs_manifests:
        chunks: list[ExternalChunkReference] = []
        for chunk in manifest.chunks:
            reference = by_digest.get(chunk.content_hash.value)
            if reference is None:
                raise ValueError(f"published store is missing segment {manifest.segment_id}")
            chunks.append(
                ExternalChunkReference(
                    chunk_id=chunk.chunk_id,
                    content_hash=chunk.content_hash,
                    size_bytes=reference.size_bytes,
                    tenant_security_domain=chunk.tenant_security_domain,
                    storage_uri=(f"cas://{chunk.tenant_security_domain}/{reference.digest}"),
                    compression=CompressionKind.NONE,
                    encryption=EncryptionKind.NONE,
                )
            )
        manifests.append(
            SegmentManifest(
                segment_id=manifest.segment_id,
                segment_hash=manifest.segment_hash,
                chunks=tuple(chunks),
            )
        )
    return tuple(manifests)


def publish_capture(
    captured: CapturedState,
    *,
    store: ContentStore,
    lease: SessionLease,
    transaction: StateTransactionRecord,
    journal: tuple[JournalEntry, ...],
    published_at_ms: int,
    capture_timestamp: str,
    git_commit: str,
    continuum_version: str,
) -> PublishedCapsule:
    """Persist exact chunks, publish a manifest, and seal authenticated metadata."""

    captured.verify()
    if captured.logical.session_id != lease.session_id != transaction.session_id:
        raise ValueError("capture, lease, and transaction sessions must match")
    if transaction.source_epoch != captured.logical.owner_epoch:
        raise ValueError("capture owner epoch differs from transaction source epoch")
    if transaction.commit_watermark != (
        captured.logical.client_delivery.last_gateway_committed_token_index
    ):
        raise ValueError("capture watermark differs from transaction commit boundary")
    inputs = captured_to_capsule_inputs(captured)
    references = tuple(
        store.put(captured.logical.tenant_id, chunk.payload, compression="none")
        for chunk in inputs.chunks
    )
    for chunk, reference in zip(inputs.chunks, references, strict=True):
        if reference.digest != chunk.content_hash.value:
            raise ValueError("content store changed a captured segment digest")
    store_manifest = store.publish(
        tenant_id=captured.logical.tenant_id,
        kind="migration",
        chunks=references,
        published_at_ms=published_at_ms,
    )
    segment_manifests = _segment_manifests(inputs.segment_manifests, references)
    journal_hash = Digest(value=canonical_hash(tuple(entry.model_dump() for entry in journal)))
    compatibility = CompatibilityConstraints(
        source_compatibility_fingerprint=Digest(
            value=canonical_hash(inputs.logical_state.dependency_graph)
        ),
        required_destination_capabilities=(
            "consistent_snapshot_import",
            "attention_kv",
            "recurrent_state",
            "sampler_state",
            "guided_decoding_state",
            "owner_epoch_fencing",
        ),
        prohibited_conversions=(ConversionPermission.QUANTIZATION,),
        recomputation_permissions=(RecomputationPermission.FROM_TOKEN_HISTORY,),
        architecture_restrictions=(
            f"state_producer={captured.logical.model.state_producer_hash}",
            f"recurrent_update={captured.logical.model.recurrent_update_hash}",
        ),
    )
    transaction_binding = CapsuleTransactionBinding(
        transaction_id=transaction.transaction_id,
        ownership_lease=OwnershipLease(
            session_id=lease.session_id,
            owner_runtime=lease.owner_runtime,
            owner_epoch=lease.owner_epoch,
            fencing_token=lease.fencing_token,
            expiration=f"coordinator-ms:{lease.expiration_ms}",
            coordinator_version=lease.coordinator_version,
            last_committed_state_version=lease.last_committed_state_version,
            last_committed_token_index=lease.last_committed_token_index,
        ),
        fencing_token=lease.fencing_token,
        source_epoch=transaction.source_epoch,
        destination_epoch=transaction.proposed_destination_epoch,
        commit_watermark=transaction.commit_watermark,
        rollback_boundary=transaction.rollback_watermark,
        transaction_journal_hash=journal_hash,
    )
    capture_digest = Digest(value=captured.logical.continuation_hash)
    evidence = MigrationVerificationEvidence(
        evidence_id=f"capture-{captured.handle.snapshot_id[:24]}",
        transaction_id=transaction.transaction_id,
        generated_at=capture_timestamp,
        capture_consistency=(
            VerificationClaim(
                claim_id="capture-consistency",
                property="logical state and physical bytes share one snapshot version",
                scope="adapter consistent-snapshot boundary",
                result="pass",
                evidence_digest=capture_digest,
            ),
        ),
        segment_integrity=tuple(
            VerificationClaim(
                claim_id=f"segment-{index}",
                property="published plaintext bytes match the captured segment digest",
                scope=manifest.segment_id,
                result="pass",
                evidence_digest=manifest.segment_hash,
            )
            for index, manifest in enumerate(segment_manifests)
        ),
        known_limitations=(
            "capture evidence proves this version-scoped reference adapter path only",
        ),
    )
    capsule = build_capsule(
        capsule_type=CapsuleType.MIGRATION,
        logical_state=inputs.logical_state,
        physical_state=inputs.physical_state,
        segment_manifests=segment_manifests,
        compatibility=compatibility,
        transaction=transaction_binding,
        evidence=evidence,
        capture_timestamp=capture_timestamp,
        git_commit=git_commit,
        continuum_version=continuum_version,
    )
    return PublishedCapsule(
        capsule=capsule,
        store_manifest=store_manifest,
        chunk_references=references,
    )
