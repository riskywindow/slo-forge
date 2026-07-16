from __future__ import annotations

from dataclasses import replace

import pytest

from sloforge.continuum.adapters import (
    AdapterUnavailableError,
    ContinuumRuntimeAdapter,
    FailurePoint,
    FailureRule,
    ImportValidationError,
    InjectedFailureError,
    LayoutKind,
    ReferenceHeadMajorAdapter,
    ReferenceTokenMajorAdapter,
    ResourceLimitError,
    RuntimeCapability,
    SessionLifecycle,
    SessionStateError,
    SnapshotConsistencyError,
    StaleOwnerEpochError,
    StateKind,
    TokenEvent,
    UnsupportedCapabilityError,
    captured_to_capsule_inputs,
)
from sloforge.continuum.ir import canonical_hash

ReferenceAdapter = ReferenceTokenMajorAdapter | ReferenceHeadMajorAdapter


def _create_source(*, seed: int = 7) -> ReferenceTokenMajorAdapter:
    source = ReferenceTokenMajorAdapter()
    source.create_session(
        session_id="session-001",
        request_id="request-001",
        tenant_id="tenant-a",
        input_token_ids=(11, 22, 33, 44),
        seed=seed,
    )
    return source


def _emit_and_ack(
    runtime: ReferenceAdapter,
    session_id: str,
    *,
    count: int,
    transaction_id: str | None = None,
) -> tuple[TokenEvent, ...]:
    events = runtime.stream_tokens(
        session_id,
        count=count,
        transaction_id=transaction_id,
    )
    for event in events:
        runtime.acknowledge_gateway(
            session_id,
            token_index=event.token_index,
            owner_epoch=event.owner_epoch,
        )
    return events


def test_capability_matrix_is_typed_and_layouts_are_physically_distinct() -> None:
    source = ReferenceTokenMajorAdapter()
    destination = ReferenceHeadMajorAdapter()

    assert source.capabilities.operations == frozenset(RuntimeCapability)
    assert StateKind.RECURRENT in source.capabilities.state_types
    assert StateKind.GUIDED_DECODING in source.capabilities.state_types
    assert source.capabilities.dirty_tracking_strategies
    source_layout = source.capabilities.layouts[0]
    destination_layout = destination.capabilities.layouts[0]
    assert source_layout.kind is LayoutKind.PAGED_TOKEN_MAJOR_SEPARATE_KV
    assert source_layout.tensor_parallel_degree == 4
    assert source_layout.page_size_tokens == 3
    assert source_layout.kv_packing == "separate-k-v"
    assert len(source_layout.simulated_devices) == 4
    assert destination_layout.kind is LayoutKind.PAGED_HEAD_MAJOR_PACKED_KV
    assert destination_layout.tensor_parallel_degree == 2
    assert destination_layout.page_size_tokens == 5
    assert destination_layout.kv_packing == "packed-k-v"
    assert len(destination_layout.simulated_devices) == 2
    assert source.identity.adapter_version != destination.identity.adapter_version


def test_base_adapter_fails_closed_for_unsupported_operation() -> None:
    adapter = ContinuumRuntimeAdapter()
    with pytest.raises(UnsupportedCapabilityError) as error:
        adapter.list_sessions()
    assert error.value.code == "unsupported_capability"
    assert error.value.operation == RuntimeCapability.INSPECT.value


@pytest.mark.parametrize(
    ("seed", "count"),
    [(0, 1), (1, 7), (7, 24), (2**31 - 1, 13), (2**63 - 1, 5)],
)
def test_generation_is_seeded_and_reproducible(seed: int, count: int) -> None:
    first = ReferenceTokenMajorAdapter()
    second = ReferenceTokenMajorAdapter()
    for adapter in (first, second):
        adapter.create_session(
            session_id="reproducible",
            request_id="request",
            tenant_id="tenant",
            input_token_ids=(1, 7, 19),
            seed=seed,
        )
    first_events = first.stream_tokens("reproducible", count=count)
    second_events = second.stream_tokens("reproducible", count=count)
    assert [event.token_id for event in first_events] == [event.token_id for event in second_events]
    assert [event.token_index for event in first_events] == list(range(count))
    assert all(event.owner_epoch == 1 for event in first_events)


def test_emission_and_gateway_commit_are_distinct_and_gap_checked() -> None:
    runtime = _create_source()
    event = runtime.generate_token("session-001")
    assert runtime.inspect_session("session-001").committed_output_index == -1
    with pytest.raises(SnapshotConsistencyError):
        runtime.capture_consistent("session-001")
    with pytest.raises(SessionStateError):
        runtime.acknowledge_gateway("session-001", token_index=1, owner_epoch=1)
    with pytest.raises(StaleOwnerEpochError):
        runtime.acknowledge_gateway("session-001", token_index=0, owner_epoch=2)
    committed = runtime.acknowledge_gateway(
        "session-001",
        token_index=event.token_index,
        owner_epoch=event.owner_epoch,
    )
    assert committed.committed_output_index == 0
    captured = runtime.capture_consistent("session-001")
    assert captured.logical.client_delivery.last_generated_token_index == 0
    assert captured.logical.client_delivery.last_gateway_committed_token_index == 0


def test_snapshot_is_consistent_while_live_session_continues() -> None:
    source = _create_source()
    _emit_and_ack(source, "session-001", count=4)
    handle = source.begin_consistent_snapshot("session-001")
    frozen_logical = source.read_logical_state(handle)
    _emit_and_ack(source, "session-001", count=3)

    assert source.read_logical_state(handle) == frozen_logical
    descriptors = source.enumerate_state_segments(handle)
    assert len(descriptors) == handle.segment_count
    assert {descriptor.state_kind for descriptor in descriptors} >= {
        StateKind.ATTENTION_KEY,
        StateKind.ATTENTION_VALUE,
        StateKind.RECURRENT,
        StateKind.SAMPLER,
        StateKind.GUIDED_DECODING,
        StateKind.TOKEN_HISTORY,
        StateKind.CLIENT_DELIVERY,
    }
    assert source.read_page_table(handle)
    source.end_consistent_snapshot(handle)
    assert source.open_snapshot_count == 0
    with pytest.raises(SnapshotConsistencyError):
        source.read_logical_state(handle)


def test_capture_detects_missing_page_segment() -> None:
    source = _create_source()
    _emit_and_ack(source, "session-001", count=2)
    captured = source.capture_consistent("session-001")
    missing_id = captured.page_table[0].segment_ids[0]
    remaining = tuple(
        segment for segment in captured.segments if segment.descriptor.segment_id != missing_id
    )
    altered = replace(
        captured,
        handle=replace(captured.handle, segment_count=len(remaining)),
        segments=remaining,
    )
    destination = ReferenceHeadMajorAdapter()
    with pytest.raises(SnapshotConsistencyError):
        destination.prepare_destination_session(
            altered,
            destination_session_id="session-001",
            proposed_owner_epoch=2,
        )


def test_capture_bridge_emits_canonical_abi_and_exact_tenant_scoped_chunks() -> None:
    source = _create_source()
    _emit_and_ack(source, "session-001", count=3)
    captured = source.capture_consistent("session-001")
    inputs = captured_to_capsule_inputs(captured)

    assert inputs.logical_state.execution.session_id == "session-001"
    assert inputs.logical_state.attention is not None
    assert inputs.logical_state.recurrent
    assert inputs.logical_state.guided_decoding is not None
    assert inputs.physical_state.owner_epoch == 1
    assert len(inputs.chunks) == len(captured.segments) == len(inputs.segment_manifests)
    assert sum(len(chunk.payload) for chunk in inputs.chunks) == sum(
        len(segment.payload) for segment in captured.segments
    )
    assert all(chunk.tenant_security_domain == "tenant-a" for chunk in inputs.chunks)
    assert canonical_hash(inputs.logical_state) == canonical_hash(inputs.logical_state)
    assert canonical_hash(inputs.physical_state) == canonical_hash(inputs.physical_state)


def test_capture_bridge_supports_empty_kv_and_disables_cross_tenant_deduplication() -> None:
    first = ReferenceTokenMajorAdapter()
    second = ReferenceTokenMajorAdapter()
    first.create_session(
        session_id="empty-a",
        request_id="request-a",
        tenant_id="tenant-a",
        input_token_ids=(),
        seed=3,
    )
    second.create_session(
        session_id="empty-b",
        request_id="request-b",
        tenant_id="tenant-b",
        input_token_ids=(),
        seed=3,
    )
    first_inputs = captured_to_capsule_inputs(first.capture_consistent("empty-a"))
    second_inputs = captured_to_capsule_inputs(second.capture_consistent("empty-b"))
    assert first_inputs.logical_state.attention is not None
    assert first_inputs.logical_state.attention.layers[0].logical_k_shape[0] == 0
    assert not any(
        segment.logical_state_reference == "state/attention-kv"
        for segment in first_inputs.physical_state.segments
    )
    # Equal empty scalar payloads receive different tenant-scoped chunk identifiers.
    first_hash_to_id = {chunk.content_hash.value: chunk.chunk_id for chunk in first_inputs.chunks}
    second_hash_to_id = {chunk.content_hash.value: chunk.chunk_id for chunk in second_inputs.chunks}
    shared_hashes = set(first_hash_to_id) & set(second_hash_to_id)
    assert shared_hashes
    assert all(first_hash_to_id[value] != second_hash_to_id[value] for value in shared_hashes)


def test_stop_and_copy_cross_adapter_cross_layout_preserves_stream_sequence() -> None:
    source = _create_source(seed=1234)
    source_events = _emit_and_ack(source, "session-001", count=12)
    source.pause_session("session-001")
    captured = source.capture_consistent("session-001")

    destination = ReferenceHeadMajorAdapter()
    prepared = destination.prepare_destination_session(
        captured,
        destination_session_id="session-001",
        proposed_owner_epoch=2,
    )
    assert prepared.lifecycle is SessionLifecycle.PREPARED
    destination.import_captured_state("session-001", captured)
    validation = destination.validate_imported_state("session-001")
    assert validation.structurally_valid and validation.continuation_valid
    assert validation.source_logical_hash == validation.imported_logical_hash
    assert source.dry_run_next_token("session-001") == validation.dry_run_next_token

    source.fence_source_writer("session-001", expected_owner_epoch=1)
    with pytest.raises(StaleOwnerEpochError):
        source.generate_token("session-001")
    active = destination.activate_destination(
        "session-001",
        committed_owner_epoch=2,
        fencing_token="coordinator-fence-2",
    )
    assert active.owner_epoch == 2
    destination_events = _emit_and_ack(destination, "session-001", count=8)
    all_events = (*source_events, *destination_events)
    assert [event.token_index for event in all_events] == list(range(20))
    assert len({event.token_index for event in all_events}) == 20
    assert {event.owner_epoch for event in source_events} == {1}
    assert {event.owner_epoch for event in destination_events} == {2}
    assert destination_events[0].token_id == validation.dry_run_next_token

    destination_capture = destination.capture_consistent("session-001")
    source_attention = {
        segment.descriptor.state_kind
        for segment in captured.segments
        if segment.descriptor.layer is not None
    }
    destination_attention = {
        segment.descriptor.state_kind
        for segment in destination_capture.segments
        if segment.descriptor.layer is not None
    }
    assert source_attention == {StateKind.ATTENTION_KEY, StateKind.ATTENTION_VALUE}
    assert destination_attention == {StateKind.ATTENTION_PACKED_KV}


def test_destination_must_validate_and_match_committed_epoch_before_activation() -> None:
    source = _create_source()
    _emit_and_ack(source, "session-001", count=3)
    captured = source.capture_consistent("session-001")
    destination = ReferenceHeadMajorAdapter()
    destination.prepare_destination_session(
        captured,
        destination_session_id="session-001",
        proposed_owner_epoch=4,
    )
    destination.import_captured_state("session-001", captured)
    with pytest.raises(ImportValidationError):
        destination.activate_destination(
            "session-001", committed_owner_epoch=4, fencing_token="fence"
        )
    destination.validate_imported_state("session-001")
    with pytest.raises(StaleOwnerEpochError):
        destination.activate_destination(
            "session-001", committed_owner_epoch=3, fencing_token="fence"
        )
    with pytest.raises(StaleOwnerEpochError):
        destination.activate_destination("session-001", committed_owner_epoch=4, fencing_token="")


def test_pause_resume_cancel_and_client_acknowledgment_are_guarded() -> None:
    runtime = _create_source()
    _emit_and_ack(runtime, "session-001", count=3)
    assert runtime.pause_session("session-001").lifecycle is SessionLifecycle.PAUSED
    with pytest.raises(SessionStateError):
        runtime.generate_token("session-001")
    with pytest.raises(StaleOwnerEpochError):
        runtime.resume_session("session-001", expected_owner_epoch=2)
    runtime.resume_session("session-001", expected_owner_epoch=1)
    acknowledged = runtime.acknowledge_client("session-001", token_index=1)
    assert acknowledged.state_version == 7
    with pytest.raises(SessionStateError):
        runtime.acknowledge_client("session-001", token_index=0)
    assert runtime.cancel_session("session-001").lifecycle is SessionLifecycle.CANCELLED
    with pytest.raises(SessionStateError):
        runtime.generate_token("session-001")


def test_precommit_fence_release_requires_current_uncommitted_epoch() -> None:
    runtime = _create_source()
    runtime.pause_session("session-001")
    runtime.fence_source_writer("session-001", expected_owner_epoch=1)
    with pytest.raises(StaleOwnerEpochError):
        runtime.release_source_fence(
            "session-001",
            expected_owner_epoch=1,
            coordinator_owner_epoch=2,
            ownership_committed=False,
        )
    with pytest.raises(StaleOwnerEpochError):
        runtime.release_source_fence(
            "session-001",
            expected_owner_epoch=1,
            coordinator_owner_epoch=1,
            ownership_committed=True,
        )
    rolled_back = runtime.release_source_fence(
        "session-001",
        expected_owner_epoch=1,
        coordinator_owner_epoch=1,
        ownership_committed=False,
    )
    assert rolled_back.lifecycle is SessionLifecycle.PAUSED
    runtime.resume_session("session-001", expected_owner_epoch=1)
    assert runtime.generate_token("session-001").owner_epoch == 1


def test_resource_bounds_are_enforced_without_silent_queue_growth() -> None:
    runtime = ReferenceTokenMajorAdapter(max_sessions=1)
    runtime.create_session(
        session_id="one",
        request_id="request-one",
        tenant_id="tenant",
        input_token_ids=(),
        seed=1,
    )
    with pytest.raises(ResourceLimitError):
        runtime.create_session(
            session_id="two",
            request_id="request-two",
            tenant_id="tenant",
            input_token_ids=(),
            seed=2,
        )
    with pytest.raises(ResourceLimitError):
        runtime.stream_tokens("one", count=runtime.capabilities.max_stream_buffer_tokens + 1)


def test_controlled_failures_are_deterministic_and_restart_clears_ephemera() -> None:
    runtime = _create_source()
    runtime.inject_failure(
        FailureRule(
            point=FailurePoint.GENERATE,
            trigger_on_call=2,
            session_id="session-001",
        )
    )
    first = runtime.generate_token("session-001")
    with pytest.raises(InjectedFailureError) as failure:
        runtime.generate_token("session-001")
    assert failure.value.point is FailurePoint.GENERATE
    second = runtime.generate_token("session-001")
    for event in (first, second):
        runtime.acknowledge_gateway(
            "session-001", token_index=event.token_index, owner_epoch=event.owner_epoch
        )
    handle = runtime.begin_consistent_snapshot("session-001")
    runtime.crash()
    with pytest.raises(AdapterUnavailableError):
        runtime.inspect_session("session-001")
    runtime.restart()
    assert runtime.inspect_session("session-001").state_version == 4
    with pytest.raises(SnapshotConsistencyError):
        runtime.read_logical_state(handle)
