from __future__ import annotations

from dataclasses import replace

import pytest

from sloforge.continuum.adapters import (
    FailurePoint,
    FailureRule,
    InjectedFailureError,
    ModelContract,
    ReferenceHeadMajorAdapter,
    ReferenceTokenMajorAdapter,
    SessionLifecycle,
)
from sloforge.continuum.ir import (
    AccessPatternKind,
    CapsuleType,
    PhysicalStateLayout,
    build_capsule,
)
from sloforge.continuum.operations import (
    AncestryIntegrityError,
    AuthorizationError,
    LazyMigrationRejected,
    LazyReason,
    LazySegmentDeclaration,
    assess_lazy_migration,
    checkpoint_full,
    checkpoint_incremental,
    clone_checkpoint,
    fork_checkpoint,
    pause_and_checkpoint,
    recompute_from_token_history,
    require_lazy_migration,
    restore_reference_capture,
    resume_checkpoint,
    verify_checkpoint_artifact,
)
from sloforge.continuum.storage import MemoryContentStore
from sloforge.continuum.transaction import (
    CutoverPhase,
    DurableCoordinator,
    GatewayCommitLedger,
    SessionLease,
    StateTransactionRecord,
    TokenEvent,
)

_STAMP = "2026-08-02T00:00:00Z"
_COMMIT = "7e51ea7f7338755d23f889820558a4e046d6c42e"


class _RecordingCoordinator(DurableCoordinator):
    last_transaction_id: str | None = None

    def begin_transaction(
        self,
        *,
        session_id: str,
        destination_candidate: str,
        migration_plan_hash: str,
        seed: int,
        now_ms: int,
        timeout_ms: int,
    ) -> StateTransactionRecord:
        record = super().begin_transaction(
            session_id=session_id,
            destination_candidate=destination_candidate,
            migration_plan_hash=migration_plan_hash,
            seed=seed,
            now_ms=now_ms,
            timeout_ms=timeout_ms,
        )
        self.last_transaction_id = record.transaction_id
        return record


def _source(*, seed: int = 19, tokens: int = 6) -> ReferenceTokenMajorAdapter:
    adapter = ReferenceTokenMajorAdapter(page_size_tokens=3)
    adapter.create_session(
        session_id="stateful-session",
        request_id="request-operations",
        tenant_id="tenant-a",
        input_token_ids=(2, 3, 5, 7),
        seed=seed,
    )
    for event in adapter.stream_tokens("stateful-session", count=tokens):
        adapter.acknowledge_gateway(
            "stateful-session",
            token_index=event.token_index,
            owner_epoch=event.owner_epoch,
        )
    return adapter


def _lease(
    adapter: ReferenceTokenMajorAdapter,
    *,
    session_id: str = "stateful-session",
    owner_epoch: int = 1,
    owner_runtime: str | None = None,
) -> SessionLease:
    metadata = adapter.inspect_session("stateful-session")
    return SessionLease(
        session_id=session_id,
        owner_runtime=owner_runtime or adapter.identity.runtime_name,
        owner_epoch=owner_epoch,
        fencing_token=owner_epoch,
        expiration_ms=120_000,
        coordinator_version=owner_epoch,
        last_committed_state_version=metadata.state_version,
        last_committed_token_index=metadata.committed_output_index,
    )


def _full_checkpoint(adapter: ReferenceTokenMajorAdapter, store: MemoryContentStore):
    return checkpoint_full(
        adapter,
        "stateful-session",
        store=store,
        lease=_lease(adapter),
        published_at_ms=1,
        capture_timestamp=_STAMP,
        git_commit=_COMMIT,
        continuum_version="0.1.0",
    )


def _transaction_event(runtime_event) -> TokenEvent:
    return TokenEvent(
        session_id=runtime_event.session_id,
        owner_epoch=runtime_event.owner_epoch,
        token_index=runtime_event.token_index,
        token_id=runtime_event.token_id,
        state_commit_version=runtime_event.state_commit_version,
    )


def test_pause_full_checkpoint_restore_and_authenticated_ancestry() -> None:
    source = _source(seed=23)
    store = MemoryContentStore()
    result = pause_and_checkpoint(
        source,
        "stateful-session",
        store=store,
        lease=_lease(source),
        published_at_ms=1,
        capture_timestamp=_STAMP,
        git_commit=_COMMIT,
        continuum_version="0.1.0",
    )
    assert result.before_pause.committed_output_index == result.paused.committed_output_index
    assert result.before_pause.client_visible_index == result.paused.client_visible_index
    assert result.paused.lifecycle is SessionLifecycle.PAUSED
    restored = restore_reference_capture(
        result.checkpoint,
        store=store,
        expected_tenant_id="tenant-a",
        expected_model=source.config.model,
    )
    assert (
        restored.logical.continuation_hash
        == source.capture_consistent("stateful-session").logical.continuation_hash
    )
    assert restored.logical.client_delivery.last_gateway_committed_token_index == 5

    tampered_proof = result.checkpoint.ancestry.model_copy(update={"current_manifest_id": "f" * 64})
    tampered = replace(result.checkpoint, ancestry=tampered_proof)
    with pytest.raises(AncestryIntegrityError):
        verify_checkpoint_artifact(tampered)
    with pytest.raises(AuthorizationError):
        restore_reference_capture(
            result.checkpoint,
            store=store,
            expected_tenant_id="tenant-b",
            expected_model=source.config.model,
        )


def test_incremental_checkpoint_tracks_parent_and_only_changed_segments() -> None:
    source = _source(seed=29, tokens=4)
    store = MemoryContentStore()
    parent = _full_checkpoint(source, store)
    for event in source.stream_tokens("stateful-session", count=2):
        source.acknowledge_gateway(
            "stateful-session", token_index=event.token_index, owner_epoch=event.owner_epoch
        )
    child = checkpoint_incremental(
        source,
        "stateful-session",
        store=store,
        lease=_lease(source),
        parent=parent,
        published_at_ms=2,
        capture_timestamp="2026-08-02T00:00:01Z",
        git_commit=_COMMIT,
        continuum_version="0.1.0",
    )
    verify_checkpoint_artifact(child)
    assert child.capsule.identity.capsule_type is CapsuleType.INCREMENTAL
    assert child.capsule.identity.parent_capsule_id == parent.capsule.identity.capsule_id
    assert child.store_manifest.parent_manifest_id == parent.store_manifest.manifest_id
    assert 0 < len(child.changed_segment_ids) < len(child.capsule.segment_manifests)
    parent_digests = {item.digest for item in parent.chunk_references}
    child_digests = {item.digest for item in child.chunk_references}
    assert parent_digests & child_digests
    restored = restore_reference_capture(
        child,
        store=store,
        expected_tenant_id="tenant-a",
        expected_model=source.config.model,
    )
    assert (
        restored.logical.committed_output_token_ids
        == source.capture_consistent("stateful-session").logical.committed_output_token_ids
    )


def test_resume_commits_new_owner_before_destination_output_without_gap() -> None:
    source = _source(seed=31)
    source.pause_session("stateful-session")
    store = MemoryContentStore()
    artifact = _full_checkpoint(source, store)
    destination = ReferenceHeadMajorAdapter(page_size_tokens=5)
    with DurableCoordinator(":memory:") as coordinator, GatewayCommitLedger(":memory:") as gateway:
        coordinator.create_lease(
            session_id="stateful-session",
            owner_runtime=source.identity.runtime_name,
            expiration_ms=120_000,
            initial_token_index=5,
        )
        gateway.register(session_id="stateful-session", owner_epoch=1, next_token_index=6)
        resumed = resume_checkpoint(
            artifact,
            store=store,
            destination=destination,
            source=source,
            expected_tenant_id="tenant-a",
            expected_model=source.config.model,
            coordinator=coordinator,
            gateway=gateway,
            seed=31,
            now_ms=10,
        )
        assert resumed.destination_owner_epoch == 2
        assert resumed.source_fenced
        assert source.inspect_session("stateful-session").lifecycle is SessionLifecycle.FENCED
        assert resumed.lease.owner_epoch == resumed.active.owner_epoch == 2
        assert resumed.transaction.phase.value == "COMPLETED"
        event = destination.generate_token("stateful-session")
        acceptance = gateway.accept(_transaction_event(event))
        assert acceptance.token_index == 6
        with pytest.raises(AuthorizationError):
            resume_checkpoint(
                artifact,
                store=store,
                destination=ReferenceHeadMajorAdapter(),
                source_release_confirmed=True,
                expected_tenant_id="tenant-a",
                expected_model=source.config.model,
                coordinator=coordinator,
                gateway=gateway,
                seed=32,
                now_ms=100,
            )


def test_resume_validation_failure_rolls_back_staging_and_can_retry() -> None:
    source = _source(seed=43)
    source.pause_session("stateful-session")
    store = MemoryContentStore()
    artifact = _full_checkpoint(source, store)
    destination = ReferenceHeadMajorAdapter(page_size_tokens=5)
    destination.inject_failure(
        FailureRule(
            point=FailurePoint.VALIDATE_IMPORT,
            trigger_on_call=1,
            session_id="stateful-session",
        )
    )
    with (
        _RecordingCoordinator(":memory:") as coordinator,
        GatewayCommitLedger(":memory:") as gateway,
    ):
        coordinator.create_lease(
            session_id="stateful-session",
            owner_runtime=source.identity.runtime_name,
            expiration_ms=120_000,
            initial_token_index=5,
        )
        gateway.register(session_id="stateful-session", owner_epoch=1, next_token_index=6)

        with pytest.raises(InjectedFailureError):
            resume_checkpoint(
                artifact,
                store=store,
                destination=destination,
                source=source,
                expected_tenant_id="tenant-a",
                expected_model=source.config.model,
                coordinator=coordinator,
                gateway=gateway,
                seed=43,
                now_ms=10,
            )

        assert coordinator.last_transaction_id is not None
        failed_transaction = coordinator.transaction(coordinator.last_transaction_id)
        assert failed_transaction.phase is CutoverPhase.ROLLED_BACK
        assert [entry.to_phase for entry in coordinator.journal(failed_transaction.transaction_id)][
            -2:
        ] == [CutoverPhase.ABORTING, CutoverPhase.ROLLED_BACK]
        assert coordinator.recoverable_transactions() == ()
        lease = coordinator.lease("stateful-session")
        assert lease.owner_epoch == 1
        assert lease.owner_runtime == source.identity.runtime_name
        assert gateway.watermark("stateful-session") == 5
        assert source.inspect_session("stateful-session").lifecycle is SessionLifecycle.PAUSED
        assert destination.prepared_session_count == 0
        assert destination.list_sessions() == ()

        destination.clear_failures()
        resumed = resume_checkpoint(
            artifact,
            store=store,
            destination=destination,
            source=source,
            expected_tenant_id="tenant-a",
            expected_model=source.config.model,
            coordinator=coordinator,
            gateway=gateway,
            seed=44,
            now_ms=100,
        )
        assert resumed.transaction.phase is CutoverPhase.COMPLETED
        assert resumed.destination_owner_epoch == 2


def test_fork_shares_immutable_chunks_and_clone_uses_independent_store() -> None:
    source = _source(seed=37)
    store = MemoryContentStore()
    parent = _full_checkpoint(source, store)
    watermark = source.inspect_session("stateful-session").committed_output_index
    leases = tuple(
        SessionLease(
            session_id=f"branch-{suffix}",
            owner_runtime=source.identity.runtime_name,
            owner_epoch=1,
            fencing_token=1,
            expiration_ms=120_000,
            coordinator_version=1,
            last_committed_state_version=source.inspect_session("stateful-session").state_version,
            last_committed_token_index=watermark,
        )
        for suffix in ("a", "b")
    )
    forked = fork_checkpoint(
        parent,
        store=store,
        expected_tenant_id="tenant-a",
        expected_model=source.config.model,
        branch_leases=leases,
        seed=37,
        published_at_ms=10,
        capture_timestamp=_STAMP,
        git_commit=_COMMIT,
        continuum_version="0.1.0",
    )
    assert {item.capsule.identity.session_id for item in forked.branches} == {
        "branch-a",
        "branch-b",
    }
    assert forked.shared_immutable_digests
    branch_states = tuple(
        restore_reference_capture(
            branch,
            store=store,
            expected_tenant_id="tenant-a",
            expected_model=source.config.model,
        )
        for branch in forked.branches
    )
    assert all(page.copy_on_write_refs == 3 for state in branch_states for page in state.page_table)
    assert branch_states[0].logical.sampler == branch_states[1].logical.sampler
    assert branch_states[0].logical.session_id != branch_states[1].logical.session_id

    clone_store = MemoryContentStore()
    clone_lease = SessionLease(
        session_id="clone-a",
        owner_runtime=source.identity.runtime_name,
        owner_epoch=1,
        fencing_token=1,
        expiration_ms=120_000,
        coordinator_version=1,
        last_committed_state_version=source.inspect_session("stateful-session").state_version,
        last_committed_token_index=watermark,
    )
    cloned = clone_checkpoint(
        parent,
        source_store=store,
        destination_store=clone_store,
        expected_tenant_id="tenant-a",
        expected_model=source.config.model,
        clone_lease=clone_lease,
        seed=37,
        published_at_ms=20,
        capture_timestamp=_STAMP,
        git_commit=_COMMIT,
        continuum_version="0.1.0",
    )
    assert cloned.copied_bytes > 0
    restored_clone = restore_reference_capture(
        cloned.clone,
        store=clone_store,
        expected_tenant_id="tenant-a",
        expected_model=source.config.model,
    )
    assert restored_clone.logical.session_id == "clone-a"


def _changed_model(source_model: ModelContract) -> ModelContract:
    return replace(
        source_model,
        model_id="continuum/hybrid-decoder-v2",
        model_hash="1" * 64,
        state_producer_hash="2" * 64,
        recurrent_update_hash="3" * 64,
    )


def test_recomputation_uses_only_proven_token_history_and_verifies_continuation() -> None:
    source = _source(seed=41)
    store = MemoryContentStore()
    artifact = _full_checkpoint(source, store)
    destination = ReferenceHeadMajorAdapter(model=_changed_model(source.config.model))
    result = recompute_from_token_history(
        artifact,
        store=store,
        destination=destination,
        expected_tenant_id="tenant-a",
        seed=41,
        continuation_horizon=5,
    )
    assert result.evidence.verified
    assert result.evidence.first_run_tokens == result.evidence.independent_run_tokens
    assert result.captured.logical.model.model_hash == "1" * 64
    assert result.captured.logical.committed_output_token_ids == (
        artifact.capsule.logical_state.token_history.committed_output_token_ids
    )
    destination.prepare_destination_session(
        result.captured,
        destination_session_id="stateful-session",
        proposed_owner_epoch=2,
    )
    destination.import_captured_state("stateful-session", result.captured)
    assert destination.validate_imported_state("stateful-session").continuation_valid

    incompatible = replace(_changed_model(source.config.model), tokenizer_hash="4" * 64)
    with pytest.raises(AuthorizationError):
        recompute_from_token_history(
            artifact,
            store=store,
            destination=ReferenceHeadMajorAdapter(model=incompatible),
            expected_tenant_id="tenant-a",
            seed=41,
        )


def _capsule_with_access(artifact, segment_id: str, *, kind: AccessPatternKind):
    physical = artifact.capsule.physical_state
    segment = next(item for item in physical.segments if item.segment_id == segment_id)
    patterns = tuple(
        item.model_copy(
            update={
                "kind": kind,
                "required_before_resume": False,
                "streamable_before_use": True,
            }
        )
        if item.access_pattern_id == segment.access_pattern_id
        else item
        for item in physical.access_patterns
    )
    document = physical.model_dump(mode="python")
    document["access_patterns"] = patterns
    rewritten = PhysicalStateLayout.model_validate(document, strict=True)
    return build_capsule(
        capsule_type=artifact.capsule.identity.capsule_type,
        logical_state=artifact.capsule.logical_state,
        physical_state=rewritten,
        segment_manifests=artifact.capsule.segment_manifests,
        compatibility=artifact.capsule.compatibility,
        transaction=artifact.capsule.transaction,
        evidence=artifact.capsule.evidence,
        capture_timestamp=artifact.capsule.identity.capture_timestamp,
        git_commit=artifact.capsule.identity.git_commit,
        continuum_version=artifact.capsule.identity.continuum_version,
        parent_capsule_id=artifact.capsule.identity.parent_capsule_id,
    )


def test_lazy_legality_rejects_dense_attention_and_allows_proven_layer_order() -> None:
    source = _source(seed=43)
    store = MemoryContentStore()
    artifact = _full_checkpoint(source, store)
    attention_id = next(
        item.segment_id
        for item in artifact.capsule.physical_state.segments
        if item.logical_state_reference == "state/attention-kv"
    )
    dense_capsule = _capsule_with_access(
        artifact, attention_id, kind=AccessPatternKind.SLIDING_WINDOW
    )
    dense = assess_lazy_migration(
        dense_capsule,
        (
            LazySegmentDeclaration(
                segment_id=attention_id,
                reason=LazyReason.SLIDING_WINDOW,
                available_before_first_use=False,
                explicit_stall_on_miss=True,
                failure_behavior="abort before emitting output",
            ),
        ),
    )
    assert not dense.legal
    assert attention_id in dense.rejected_segment_ids
    with pytest.raises(LazyMigrationRejected, match="dense full-attention"):
        require_lazy_migration(
            dense_capsule,
            (
                LazySegmentDeclaration(
                    segment_id=attention_id,
                    reason=LazyReason.SLIDING_WINDOW,
                    available_before_first_use=False,
                    explicit_stall_on_miss=True,
                    failure_behavior="abort before emitting output",
                ),
            ),
        )

    sampler_id = "logical:sampler"
    layer_capsule = _capsule_with_access(
        artifact, sampler_id, kind=AccessPatternKind.LAYER_SEQUENTIAL
    )
    legal = assess_lazy_migration(
        layer_capsule,
        (
            LazySegmentDeclaration(
                segment_id=sampler_id,
                reason=LazyReason.LAYER_SEQUENTIAL,
                available_before_first_use=True,
                explicit_stall_on_miss=False,
                failure_behavior="abort before sampler use",
            ),
        ),
    )
    assert legal.legal
    assert legal.legal_segment_ids == (sampler_id,)

    required = assess_lazy_migration(
        artifact.capsule,
        (
            LazySegmentDeclaration(
                segment_id=sampler_id,
                reason=LazyReason.RECOMPUTABLE,
                available_before_first_use=False,
                explicit_stall_on_miss=True,
                failure_behavior="recompute or abort",
            ),
        ),
    )
    assert not required.legal
    assert "required before resume" in required.reasons[0]
