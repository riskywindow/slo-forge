"""Content-addressed fork and independent clone operations."""

from __future__ import annotations

from dataclasses import replace
from hashlib import sha256

from sloforge.continuum.adapters import (
    CapturedState,
    ModelContract,
    SessionLifecycle,
    SnapshotHandle,
)
from sloforge.continuum.ir import CapsuleType
from sloforge.continuum.reference.codec import decode_segments, encode_state
from sloforge.continuum.reference.models import HybridDecoderConfig
from sloforge.continuum.storage import ContentStore
from sloforge.continuum.transaction import SessionLease

from .checkpoint import _publish_checkpoint, verify_checkpoint_artifact
from .models import (
    AncestryKind,
    AuthorizationError,
    CheckpointArtifact,
    CloneResult,
    ForkResult,
)
from .restore import restore_reference_capture


def _config(captured: CapturedState, model: ModelContract) -> HybridDecoderConfig:
    logical = captured.logical
    if not logical.recurrent_state:
        raise ValueError("reference branch requires recurrent state")
    return HybridDecoderConfig(
        layout=captured.layout,
        layers=logical.attention_layer_count,
        kv_heads=logical.attention_kv_head_count,
        head_dimension=logical.attention_head_dimension,
        recurrent_width=len(logical.recurrent_state[0]),
        max_context_tokens=max(
            4096,
            len(logical.input_token_ids) + len(logical.committed_output_token_ids) + 64,
        ),
        model=model,
        automaton_id=logical.guided_decoding.automaton_id,
    )


def _branch_capture(
    source: CapturedState,
    *,
    model: ModelContract,
    session_id: str,
    owner_epoch: int,
    seed: int,
    branch_count: int,
) -> CapturedState:
    config = _config(source, model)
    state = decode_segments(
        source_layout=source.layout,
        source_segments=source.segments,
        manifest=source.logical,
        destination_config=config,
        destination_session_id=session_id,
    )
    state.session_id = session_id
    state.owner_epoch = owner_epoch
    state.lifecycle = SessionLifecycle.PAUSED
    encoded = encode_state(state, config)
    pages = tuple(replace(page, copy_on_write_refs=branch_count + 1) for page in encoded.page_table)
    snapshot_id = sha256(
        f"continuum-branch/v1\0{source.handle.snapshot_id}\0{session_id}\0{owner_epoch}\0{seed}".encode()
    ).hexdigest()
    captured = CapturedState(
        handle=SnapshotHandle(
            snapshot_id=snapshot_id,
            session_id=session_id,
            owner_epoch=owner_epoch,
            state_version=encoded.logical.state_version,
            dirty_epoch=encoded.logical.dirty_epoch,
            segment_count=len(encoded.segments),
        ),
        runtime=source.runtime,
        layout=source.layout,
        logical=encoded.logical,
        segments=encoded.segments,
        page_table=pages,
    )
    captured.verify()
    return captured


def fork_checkpoint(
    parent: CheckpointArtifact,
    *,
    store: ContentStore,
    expected_tenant_id: str,
    expected_model: ModelContract,
    branch_leases: tuple[SessionLease, ...],
    seed: int,
    published_at_ms: int,
    capture_timestamp: str,
    git_commit: str,
    continuum_version: str,
) -> ForkResult:
    """Fork immutable snapshot bytes while giving each descendant mutable runtime state."""

    verify_checkpoint_artifact(parent)
    if len(branch_leases) < 2:
        raise ValueError("fork requires at least two branches")
    if len(branch_leases) > 64:
        raise ValueError("fork branch count exceeds 64")
    session_ids = [lease.session_id for lease in branch_leases]
    if len(set(session_ids)) != len(session_ids):
        raise ValueError("fork session identifiers must be unique")
    if parent.capsule.identity.session_id in session_ids:
        raise ValueError("fork descendants require new session identifiers")
    source = restore_reference_capture(
        parent,
        store=store,
        expected_tenant_id=expected_tenant_id,
        expected_model=expected_model,
    )
    watermark = source.logical.client_delivery.last_gateway_committed_token_index
    artifacts: list[CheckpointArtifact] = []
    for index, lease in enumerate(branch_leases):
        if lease.owner_epoch < 1 or lease.last_committed_token_index != watermark:
            raise AuthorizationError("branch lease does not authorize the fork watermark")
        captured = _branch_capture(
            source,
            model=expected_model,
            session_id=lease.session_id,
            owner_epoch=lease.owner_epoch,
            seed=seed + index,
            branch_count=len(branch_leases),
        )
        artifacts.append(
            _publish_checkpoint(
                captured,
                store=store,
                lease=lease,
                capsule_type=CapsuleType.FORK,
                store_kind="fork",
                ancestry_kind=AncestryKind.FORK,
                parent=parent,
                published_at_ms=published_at_ms + index,
                capture_timestamp=capture_timestamp,
                git_commit=git_commit,
                continuum_version=continuum_version,
            )
        )
    mutable = {"logical:sampler", "logical:guided", "logical:delivery"}
    immutable_parent = {
        manifest.segment_hash.value
        for manifest in parent.capsule.segment_manifests
        if manifest.segment_id not in mutable
    }
    shared = set(immutable_parent)
    for artifact in artifacts:
        shared &= {
            manifest.segment_hash.value
            for manifest in artifact.capsule.segment_manifests
            if manifest.segment_id not in mutable
        }
    return ForkResult(
        parent_capsule_id=parent.capsule.identity.capsule_id,
        branches=tuple(artifacts),
        shared_immutable_digests=tuple(sorted(shared)),
    )


def clone_checkpoint(
    parent: CheckpointArtifact,
    *,
    source_store: ContentStore,
    destination_store: ContentStore,
    expected_tenant_id: str,
    expected_model: ModelContract,
    clone_lease: SessionLease,
    seed: int,
    published_at_ms: int,
    capture_timestamp: str,
    git_commit: str,
    continuum_version: str,
) -> CloneResult:
    """Create a descendant backed by an independent destination content store."""

    source = restore_reference_capture(
        parent,
        store=source_store,
        expected_tenant_id=expected_tenant_id,
        expected_model=expected_model,
    )
    if clone_lease.session_id == parent.capsule.identity.session_id:
        raise ValueError("clone requires an independent session identifier")
    if clone_lease.last_committed_token_index != (
        source.logical.client_delivery.last_gateway_committed_token_index
    ):
        raise AuthorizationError("clone lease does not authorize the checkpoint watermark")
    captured = _branch_capture(
        source,
        model=expected_model,
        session_id=clone_lease.session_id,
        owner_epoch=clone_lease.owner_epoch,
        seed=seed,
        branch_count=1,
    )
    clone = _publish_checkpoint(
        captured,
        store=destination_store,
        lease=clone_lease,
        capsule_type=CapsuleType.COMPLETE,
        store_kind="complete",
        ancestry_kind=AncestryKind.CLONE,
        parent=parent,
        published_at_ms=published_at_ms,
        capture_timestamp=capture_timestamp,
        git_commit=git_commit,
        continuum_version=continuum_version,
        link_store_parent=False,
    )
    return CloneResult(
        parent_capsule_id=parent.capsule.identity.capsule_id,
        clone=clone,
        copied_bytes=sum(reference.size_bytes for reference in clone.chunk_references),
    )
