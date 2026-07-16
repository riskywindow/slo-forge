from __future__ import annotations

import pytest

from sloforge.continuum.adapters import (
    DirtyLogOverflowError,
    ReferenceHeadMajorAdapter,
    ReferenceTokenMajorAdapter,
    SessionLifecycle,
    SessionStateError,
    StaleDeltaError,
    StateKind,
    TokenEvent,
)


def _emit_and_ack(
    runtime: ReferenceTokenMajorAdapter | ReferenceHeadMajorAdapter,
    *,
    count: int,
    transaction_id: str | None = None,
) -> tuple[TokenEvent, ...]:
    events = runtime.stream_tokens(
        "live-session",
        count=count,
        transaction_id=transaction_id,
    )
    for event in events:
        runtime.acknowledge_gateway(
            "live-session",
            token_index=event.token_index,
            owner_epoch=event.owner_epoch,
        )
    return events


def _source_with_history(*, max_dirty_events: int = 128) -> ReferenceTokenMajorAdapter:
    source = ReferenceTokenMajorAdapter(max_dirty_events=max_dirty_events)
    source.create_session(
        session_id="live-session",
        request_id="live-request",
        tenant_id="tenant-a",
        input_token_ids=(3, 5, 8),
        seed=99,
    )
    _emit_and_ack(source, count=5)
    return source


def test_precopy_transfers_incremental_pages_and_final_quiesced_delta() -> None:
    source = _source_with_history()
    tracking = source.start_dirty_tracking("live-session")
    initial = source.capture_consistent("live-session")

    destination = ReferenceHeadMajorAdapter()
    destination.prepare_destination_session(
        initial,
        destination_session_id="live-session",
        proposed_owner_epoch=2,
    )
    destination.import_captured_state("live-session", initial)

    continuing = _emit_and_ack(source, count=7, transaction_id="txn-precopy")
    delta_one = source.obtain_dirty_delta(tracking)
    assert delta_one.from_epoch == initial.logical.dirty_epoch
    assert delta_one.to_epoch == initial.logical.dirty_epoch + 14
    assert not delta_one.final
    assert len(delta_one.changed_segments) < len(source.capture_consistent("live-session").segments)
    assert {segment.descriptor.state_kind for segment in delta_one.changed_segments} >= {
        StateKind.RECURRENT,
        StateKind.SAMPLER,
        StateKind.GUIDED_DECODING,
        StateKind.TOKEN_HISTORY,
        StateKind.CLIENT_DELIVERY,
    }
    destination.apply_dirty_delta("live-session", delta_one)

    final_source_events = _emit_and_ack(source, count=3, transaction_id="txn-precopy")
    quiesced = source.quiesce_at_token_boundary("live-session")
    assert quiesced.lifecycle is SessionLifecycle.PAUSED
    final_delta = source.export_final_delta(tracking)
    assert final_delta.final
    assert final_delta.from_epoch == delta_one.to_epoch
    assert final_delta.to_epoch == delta_one.to_epoch + 6
    destination.apply_dirty_delta("live-session", final_delta)

    validation = destination.validate_imported_state("live-session")
    assert source.dry_run_next_token("live-session") == validation.dry_run_next_token
    source.fence_source_writer("live-session", expected_owner_epoch=1)
    destination.activate_destination(
        "live-session",
        committed_owner_epoch=2,
        fencing_token="fencing-token-2",
    )
    first_destination = destination.generate_token("live-session", transaction_id="txn-precopy")
    destination.acknowledge_gateway(
        "live-session",
        token_index=first_destination.token_index,
        owner_epoch=first_destination.owner_epoch,
    )

    assert first_destination.token_id == validation.dry_run_next_token
    visible_indices = [
        *range(5),
        *(event.token_index for event in continuing),
        *(event.token_index for event in final_source_events),
        first_destination.token_index,
    ]
    assert visible_indices == list(range(16))
    assert len(set(visible_indices)) == len(visible_indices)
    assert first_destination.owner_epoch == 2


def test_noop_delta_is_explicit_and_safe() -> None:
    source = _source_with_history()
    tracking = source.start_dirty_tracking("live-session")
    initial = source.capture_consistent("live-session")
    delta = source.obtain_dirty_delta(tracking)
    assert delta.from_epoch == delta.to_epoch == initial.logical.dirty_epoch
    assert delta.changed_segments == ()


def test_out_of_order_and_duplicate_delta_are_rejected() -> None:
    source = _source_with_history()
    tracking = source.start_dirty_tracking("live-session")
    initial = source.capture_consistent("live-session")
    destination = ReferenceHeadMajorAdapter()
    destination.prepare_destination_session(
        initial,
        destination_session_id="live-session",
        proposed_owner_epoch=2,
    )
    destination.import_captured_state("live-session", initial)

    _emit_and_ack(source, count=1)
    first = source.obtain_dirty_delta(tracking)
    _emit_and_ack(source, count=1)
    second = source.obtain_dirty_delta(tracking)
    with pytest.raises(StaleDeltaError):
        destination.apply_dirty_delta("live-session", second)
    destination.apply_dirty_delta("live-session", first)
    with pytest.raises(StaleDeltaError):
        destination.apply_dirty_delta("live-session", first)
    destination.apply_dirty_delta("live-session", second)


def test_dirty_log_overflow_fails_closed_and_requires_new_snapshot() -> None:
    source = _source_with_history(max_dirty_events=3)
    tracking = source.start_dirty_tracking("live-session")
    _emit_and_ack(source, count=5)
    with pytest.raises(DirtyLogOverflowError) as error:
        source.obtain_dirty_delta(tracking)
    assert "full snapshot" in str(error.value)


def test_final_delta_requires_token_boundary_quiescence() -> None:
    source = _source_with_history()
    tracking = source.start_dirty_tracking("live-session")
    with pytest.raises(SessionStateError):
        source.export_final_delta(tracking)


def test_abort_destination_releases_prepared_state() -> None:
    source = _source_with_history()
    captured = source.capture_consistent("live-session")
    destination = ReferenceHeadMajorAdapter()
    destination.prepare_destination_session(
        captured,
        destination_session_id="live-session",
        proposed_owner_epoch=2,
    )
    assert destination.prepared_session_count == 1
    destination.abort_destination("live-session")
    destination.abort_destination("live-session")
    assert destination.prepared_session_count == 0
