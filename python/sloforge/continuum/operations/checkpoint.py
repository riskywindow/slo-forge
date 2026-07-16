"""Consistent full and incremental checkpoints over the canonical capsule ABI."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Literal

from sloforge.continuum.adapters import CapturedState, ContinuumRuntimeAdapter
from sloforge.continuum.adapters.abi import captured_to_capsule_inputs
from sloforge.continuum.ir import (
    CapsuleTransactionBinding,
    CapsuleType,
    CompatibilityConstraints,
    CompressionKind,
    ConversionPermission,
    Digest,
    EncryptionKind,
    ExternalChunkReference,
    MigrationVerificationEvidence,
    OwnershipLease,
    PhysicalStateLayout,
    RecomputationPermission,
    SegmentManifest,
    VerificationClaim,
    build_capsule,
    canonical_hash,
    validate_capsule,
)
from sloforge.continuum.storage import ChunkRef, ContentStore
from sloforge.continuum.transaction import SessionLease

from ._integrity import ANCESTRY_CLAIM_ID, ancestry_claim, make_ancestry_proof
from .models import (
    AncestryIntegrityError,
    AncestryKind,
    AuthorizationError,
    CheckpointArtifact,
    OperationError,
    PauseCheckpointResult,
)


def _unique_references(references: Iterable[ChunkRef]) -> tuple[ChunkRef, ...]:
    by_digest: dict[str, ChunkRef] = {}
    for reference in references:
        existing = by_digest.setdefault(reference.digest, reference)
        if existing != reference:
            raise AncestryIntegrityError("one digest resolved to conflicting chunk metadata")
    return tuple(by_digest[key] for key in sorted(by_digest))


def _external_manifests(
    source: tuple[SegmentManifest, ...], references: tuple[ChunkRef, ...]
) -> tuple[SegmentManifest, ...]:
    by_digest = {reference.digest: reference for reference in references}
    result: list[SegmentManifest] = []
    for manifest in source:
        chunks: list[ExternalChunkReference] = []
        for chunk in manifest.chunks:
            reference = by_digest.get(chunk.content_hash.value)
            if reference is None:
                raise AncestryIntegrityError(
                    f"published CAS lacks segment payload {manifest.segment_id}"
                )
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
        result.append(
            SegmentManifest(
                segment_id=manifest.segment_id,
                segment_hash=manifest.segment_hash,
                chunks=tuple(chunks),
            )
        )
    return tuple(result)


def _compatibility(captured: CapturedState) -> CompatibilityConstraints:
    return CompatibilityConstraints(
        source_compatibility_fingerprint=Digest(
            value=canonical_hash(
                captured_to_capsule_inputs(captured).logical_state.dependency_graph
            )
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


def _binding(
    captured: CapturedState, lease: SessionLease, proof_digest: str
) -> CapsuleTransactionBinding:
    return CapsuleTransactionBinding(
        transaction_id=None,
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
        source_epoch=lease.owner_epoch,
        destination_epoch=None,
        commit_watermark=(captured.logical.client_delivery.last_gateway_committed_token_index),
        rollback_boundary=(captured.logical.client_delivery.last_gateway_committed_token_index),
        transaction_journal_hash=Digest(value=proof_digest),
    )


def _publish_checkpoint(
    captured: CapturedState,
    *,
    store: ContentStore,
    lease: SessionLease,
    capsule_type: CapsuleType,
    store_kind: Literal["complete", "incremental", "fork"],
    ancestry_kind: AncestryKind,
    parent: CheckpointArtifact | None,
    published_at_ms: int,
    capture_timestamp: str,
    git_commit: str,
    continuum_version: str,
    link_store_parent: bool = True,
) -> CheckpointArtifact:
    captured.verify()
    watermark = captured.logical.client_delivery.last_gateway_committed_token_index
    if captured.logical.session_id != lease.session_id:
        raise AuthorizationError("lease and captured session differ")
    if captured.logical.owner_epoch != lease.owner_epoch:
        raise AuthorizationError("lease epoch does not authorize captured state")
    if watermark != lease.last_committed_token_index:
        raise AuthorizationError("lease token watermark differs from captured client state")
    if parent is not None:
        verify_checkpoint_artifact(parent)
        if (
            ancestry_kind is AncestryKind.INCREMENTAL
            and parent.capsule.identity.session_id != captured.logical.session_id
        ):
            raise AncestryIntegrityError("incremental parent belongs to another session")
        if parent.capsule.identity.tenant_id != captured.logical.tenant_id:
            raise AuthorizationError("incremental parent belongs to another tenant")
        if parent.capsule.identity.model_hash.value != captured.logical.model.model_hash:
            raise AncestryIntegrityError("incremental checkpoint changed model identity")
        if parent.capsule.identity.tokenizer_hash.value != captured.logical.model.tokenizer_hash:
            raise AncestryIntegrityError("incremental checkpoint changed tokenizer identity")
        generation = parent.ancestry.generation + 1
        parent_capsule_id = parent.capsule.identity.capsule_id
        parent_manifest_id = parent.store_manifest.manifest_id
    else:
        generation = 0
        parent_capsule_id = None
        parent_manifest_id = None

    inputs = captured_to_capsule_inputs(captured)
    cow_refs = {
        segment_id: page.copy_on_write_refs
        for page in captured.page_table
        for segment_id in page.segment_ids
    }
    page_tables = tuple(
        table.model_copy(
            update={
                "entries": tuple(
                    entry.model_copy(
                        update={
                            "copy_on_write_reference_count": cow_refs.get(
                                table.segment_id,
                                entry.copy_on_write_reference_count,
                            )
                        }
                    )
                    for entry in table.entries
                )
            }
        )
        for table in inputs.physical_state.page_tables
    )
    physical_document = inputs.physical_state.model_dump(mode="python")
    physical_document["page_tables"] = page_tables
    physical_state = PhysicalStateLayout.model_validate(physical_document, strict=True)
    references = _unique_references(
        store.put(captured.logical.tenant_id, chunk.payload, compression="none")
        for chunk in inputs.chunks
    )
    store_manifest = store.publish(
        tenant_id=captured.logical.tenant_id,
        kind=store_kind,
        chunks=references,
        published_at_ms=published_at_ms,
        parent_manifest_id=parent_manifest_id if link_store_parent else None,
    )
    proof = make_ancestry_proof(
        kind=ancestry_kind,
        generation=generation,
        parent_capsule_id=parent_capsule_id,
        parent_manifest_id=parent_manifest_id,
        current_manifest_id=store_manifest.manifest_id,
    )
    manifests = _external_manifests(inputs.segment_manifests, references)
    capture_digest = Digest(value=captured.logical.continuation_hash)
    evidence = MigrationVerificationEvidence(
        evidence_id=f"checkpoint-{captured.handle.snapshot_id[:24]}",
        transaction_id=None,
        generated_at=capture_timestamp,
        capture_consistency=(
            VerificationClaim(
                claim_id="checkpoint-consistency",
                property="checkpoint was captured at one adapter state version and token boundary",
                scope="reference adapter consistent snapshot",
                result="pass",
                evidence_digest=capture_digest,
            ),
            ancestry_claim(proof),
        ),
        segment_integrity=tuple(
            VerificationClaim(
                claim_id=f"checkpoint-segment-{index}",
                property="CAS plaintext digest equals runtime segment checksum",
                scope=manifest.segment_id,
                result="pass",
                evidence_digest=manifest.segment_hash,
            )
            for index, manifest in enumerate(manifests)
        ),
        known_limitations=("restore is version-scoped to the deterministic reference runtime ABI",),
    )
    capsule = build_capsule(
        capsule_type=capsule_type,
        logical_state=inputs.logical_state,
        physical_state=physical_state,
        segment_manifests=manifests,
        compatibility=_compatibility(captured),
        transaction=_binding(captured, lease, proof.digest),
        evidence=evidence,
        capture_timestamp=capture_timestamp,
        git_commit=git_commit,
        continuum_version=continuum_version,
        parent_capsule_id=parent_capsule_id,
    )
    previous = (
        {
            manifest.segment_id: manifest.segment_hash.value
            for manifest in parent.capsule.segment_manifests
        }
        if parent is not None
        else {}
    )
    changed = tuple(
        manifest.segment_id
        for manifest in manifests
        if previous.get(manifest.segment_id) != manifest.segment_hash.value
    )
    artifact = CheckpointArtifact(
        capsule=capsule,
        store_manifest=store_manifest,
        chunk_references=references,
        ancestry=proof,
        changed_segment_ids=changed,
        parent=parent,
    )
    verify_checkpoint_artifact(artifact)
    return artifact


def verify_checkpoint_artifact(artifact: CheckpointArtifact) -> None:
    """Verify capsule, CAS publication, and every authenticated ancestry edge."""

    try:
        validate_capsule(artifact.capsule)
    except ValueError as error:
        raise AncestryIntegrityError("capsule integrity validation failed") from error
    capsule = artifact.capsule
    manifest = artifact.store_manifest
    if capsule.identity.tenant_id != manifest.tenant_id:
        raise AuthorizationError("capsule and CAS manifest tenant differ")
    if artifact.ancestry.current_manifest_id != manifest.manifest_id:
        raise AncestryIntegrityError("ancestry names a different CAS publication")
    expected = make_ancestry_proof(
        kind=artifact.ancestry.kind,
        generation=artifact.ancestry.generation,
        parent_capsule_id=artifact.ancestry.parent_capsule_id,
        parent_manifest_id=artifact.ancestry.parent_manifest_id,
        current_manifest_id=artifact.ancestry.current_manifest_id,
    )
    if expected != artifact.ancestry:
        raise AncestryIntegrityError("ancestry digest is invalid")
    claims = {claim.claim_id: claim for claim in capsule.evidence.capture_consistency}
    claim = claims.get(ANCESTRY_CLAIM_ID)
    if claim is None or claim.result != "pass" or claim.evidence_digest.value != expected.digest:
        raise AncestryIntegrityError("capsule does not authenticate its ancestry proof")
    if capsule.identity.parent_capsule_id != artifact.ancestry.parent_capsule_id:
        raise AncestryIntegrityError("capsule parent is not the authenticated parent")
    reference_by_digest = {reference.digest: reference for reference in artifact.chunk_references}
    manifest_by_digest = {reference.digest: reference for reference in manifest.chunks}
    required = {
        chunk.content_hash.value
        for segment in capsule.segment_manifests
        for chunk in segment.chunks
    }
    if set(reference_by_digest) != required or set(manifest_by_digest) != required:
        raise AncestryIntegrityError("CAS publication does not exactly cover capsule chunks")
    if any(reference_by_digest[key] != manifest_by_digest[key] for key in required):
        raise AncestryIntegrityError("CAS reference metadata differs from publication")
    parent = artifact.parent
    if artifact.ancestry.kind is AncestryKind.ROOT:
        if parent is not None or artifact.ancestry.parent_capsule_id is not None:
            raise AncestryIntegrityError("root checkpoint cannot name a parent")
        return
    if parent is None:
        raise AncestryIntegrityError("non-root checkpoint must include its parent artifact")
    verify_checkpoint_artifact(parent)
    if artifact.ancestry.generation != parent.ancestry.generation + 1:
        raise AncestryIntegrityError("checkpoint generation is not monotonic")
    if artifact.ancestry.parent_capsule_id != parent.capsule.identity.capsule_id:
        raise AncestryIntegrityError("checkpoint parent capsule does not match ancestry")
    if artifact.ancestry.parent_manifest_id != parent.store_manifest.manifest_id:
        raise AncestryIntegrityError("checkpoint parent publication does not match ancestry")


def checkpoint_full(
    adapter: ContinuumRuntimeAdapter,
    session_id: str,
    *,
    store: ContentStore,
    lease: SessionLease,
    published_at_ms: int,
    capture_timestamp: str,
    git_commit: str,
    continuum_version: str,
) -> CheckpointArtifact:
    return _publish_checkpoint(
        adapter.capture_consistent(session_id),
        store=store,
        lease=lease,
        capsule_type=CapsuleType.COMPLETE,
        store_kind="complete",
        ancestry_kind=AncestryKind.ROOT,
        parent=None,
        published_at_ms=published_at_ms,
        capture_timestamp=capture_timestamp,
        git_commit=git_commit,
        continuum_version=continuum_version,
    )


def checkpoint_incremental(
    adapter: ContinuumRuntimeAdapter,
    session_id: str,
    *,
    store: ContentStore,
    lease: SessionLease,
    parent: CheckpointArtifact,
    published_at_ms: int,
    capture_timestamp: str,
    git_commit: str,
    continuum_version: str,
) -> CheckpointArtifact:
    return _publish_checkpoint(
        adapter.capture_consistent(session_id),
        store=store,
        lease=lease,
        capsule_type=CapsuleType.INCREMENTAL,
        store_kind="incremental",
        ancestry_kind=AncestryKind.INCREMENTAL,
        parent=parent,
        published_at_ms=published_at_ms,
        capture_timestamp=capture_timestamp,
        git_commit=git_commit,
        continuum_version=continuum_version,
    )


def pause_and_checkpoint(
    adapter: ContinuumRuntimeAdapter,
    session_id: str,
    *,
    store: ContentStore,
    lease: SessionLease,
    published_at_ms: int,
    capture_timestamp: str,
    git_commit: str,
    continuum_version: str,
) -> PauseCheckpointResult:
    before = adapter.inspect_session(session_id)
    paused = adapter.pause_session(session_id)
    if (
        before.committed_output_index != paused.committed_output_index
        or before.client_visible_index != paused.client_visible_index
        or before.owner_epoch != paused.owner_epoch
    ):
        raise OperationError("pause changed the client cursor or owner epoch")
    artifact = checkpoint_full(
        adapter,
        session_id,
        store=store,
        lease=lease,
        published_at_ms=published_at_ms,
        capture_timestamp=capture_timestamp,
        git_commit=git_commit,
        continuum_version=continuum_version,
    )
    return PauseCheckpointResult(before_pause=before, paused=paused, checkpoint=artifact)
