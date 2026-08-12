from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from sloforge.helix.characterization.gpu_reclamation_accounting import (
    AccountingBucket,
    LogicalStateSegment,
    MemoryDomain,
    MovementAccountingError,
    StateMovementReport,
    StatePassOperation,
    StatePassRecord,
    StateSegmentKind,
    TransferDirection,
    build_state_movement_report,
    canonical_movement_json,
)


def _segments() -> tuple[LogicalStateSegment, ...]:
    return (
        LogicalStateSegment(
            segment_id="private-b0",
            branch_group_id="group-004",
            branch_id="branch-0",
            kind=StateSegmentKind.PRIVATE_KV,
            logical_bytes=20,
        ),
        LogicalStateSegment(
            segment_id="shared-root",
            branch_group_id="group-004",
            kind=StateSegmentKind.SHARED_KV,
            logical_bytes=80,
        ),
    )


def _pass(
    record_id: str,
    *,
    operation: StatePassOperation,
    source: MemoryDomain,
    destination: MemoryDomain,
    read: int,
    written: int,
    start: int,
    direction: TransferDirection = TransferDirection.NONE,
    transfer: int = 0,
    checksum: int = 0,
    temporary: int = 0,
    temporary_memory: MemoryDomain | None = None,
    unavoidable: bool = False,
    state_segment: str = "shared-root",
    logical_bytes: int = 80,
) -> StatePassRecord:
    return StatePassRecord(
        record_id=record_id,
        state_segment=state_segment,
        branch_group="group-004",
        operation=operation,
        source_memory=source,
        destination_memory=destination,
        bytes_read=read,
        bytes_written=written,
        transfer_direction=direction,
        transfer_bytes=transfer,
        checksum_bytes=checksum,
        logical_bytes=logical_bytes,
        start_ns=start,
        end_ns=start + 10,
        device="cuda:1",
        temporary_allocation_bytes=temporary,
        temporary_allocation_memory=temporary_memory,
        temporary_allocation_id=f"allocation-{record_id}" if temporary else None,
        required_unavoidable=unavoidable,
        read_event_id=f"read-{record_id}" if read else None,
        write_event_id=f"write-{record_id}" if written else None,
        transfer_event_id=f"link-{record_id}" if transfer else None,
    )


def _passes() -> tuple[StatePassRecord, ...]:
    return (
        _pass(
            "source-transform",
            operation=StatePassOperation.TRANSFORM,
            source=MemoryDomain.SOURCE_GPU_NATIVE_PAGED,
            destination=MemoryDomain.GPU_TRANSFORM_BUFFER,
            read=100,
            written=100,
            start=0,
            temporary=100,
            temporary_memory=MemoryDomain.GPU_TRANSFORM_BUFFER,
            logical_bytes=80,
        ),
        _pass(
            "integrity",
            operation=StatePassOperation.CHECKSUM,
            source=MemoryDomain.GPU_TRANSFORM_BUFFER,
            destination=MemoryDomain.NONE,
            read=100,
            written=0,
            checksum=100,
            start=10,
            unavoidable=True,
            logical_bytes=80,
        ),
        _pass(
            "device-to-host",
            operation=StatePassOperation.D2H,
            source=MemoryDomain.GPU_TRANSFORM_BUFFER,
            destination=MemoryDomain.PINNED_HOST_TRANSPORT,
            read=100,
            written=100,
            direction=TransferDirection.D2H,
            transfer=100,
            start=20,
            temporary=100,
            temporary_memory=MemoryDomain.PINNED_HOST_TRANSPORT,
            unavoidable=True,
            logical_bytes=80,
        ),
        _pass(
            "host-to-device",
            operation=StatePassOperation.H2D,
            source=MemoryDomain.PINNED_HOST_TRANSPORT,
            destination=MemoryDomain.DESTINATION_GPU_NATIVE_PAGED,
            read=100,
            written=100,
            direction=TransferDirection.H2D,
            transfer=100,
            start=30,
            unavoidable=True,
            logical_bytes=80,
        ),
        _pass(
            "private-direct",
            operation=StatePassOperation.D2H,
            source=MemoryDomain.SOURCE_GPU_NATIVE_PAGED,
            destination=MemoryDomain.PINNED_HOST_TRANSPORT,
            read=20,
            written=20,
            direction=TransferDirection.D2H,
            transfer=20,
            start=40,
            unavoidable=True,
            state_segment="private-b0",
            logical_bytes=20,
        ),
        _pass(
            "private-restore",
            operation=StatePassOperation.H2D,
            source=MemoryDomain.PINNED_HOST_TRANSPORT,
            destination=MemoryDomain.DESTINATION_GPU_NATIVE_PAGED,
            read=20,
            written=20,
            direction=TransferDirection.H2D,
            transfer=20,
            start=50,
            unavoidable=True,
            state_segment="private-b0",
            logical_bytes=20,
        ),
    )


def test_exact_accounting_counts_endpoint_touches_and_links_once() -> None:
    report = build_state_movement_report(logical_segments=_segments(), passes=_passes())
    accounting = report.accounting

    assert accounting.logical_state_bytes == 100
    assert accounting.physical_bytes_read == 120
    assert accounting.physical_bytes_written == 120
    assert accounting.d2h_bytes == 120
    assert accounting.h2d_bytes == 120
    assert accounting.host_intermediate_bytes == 240
    assert accounting.gpu_intermediate_bytes == 300
    assert accounting.checksum_bytes == 100
    assert accounting.unavoidable_final_destination_write_bytes == 120
    assert accounting.amplification_numerator_bytes == 1020
    assert accounting.state_movement_amplification == 10.2
    assert accounting.amplification_numerator_excluding_unavoidable_final_writes_bytes == 900
    assert accounting.state_movement_amplification_excluding_unavoidable_final_writes == 9.0
    assert accounting.host_temporary_allocation_bytes == 100
    assert accounting.gpu_temporary_allocation_bytes == 100
    assert accounting.total_temporary_allocation_bytes == 200

    edge_totals = {bucket: 0 for bucket in AccountingBucket}
    for edge in report.edges:
        edge_totals[edge.accounting_bucket] += edge.bytes
    assert edge_totals[AccountingBucket.LOGICAL_STATE] == accounting.logical_state_bytes
    assert edge_totals[AccountingBucket.D2H] == accounting.d2h_bytes
    assert edge_totals[AccountingBucket.H2D] == accounting.h2d_bytes
    assert sum(edge.included_in_amplification for edge in report.edges) == 15


def test_shared_logical_payload_is_not_multiplied_by_pass_count() -> None:
    report = build_state_movement_report(logical_segments=_segments(), passes=_passes())

    assert sum(item.logical_bytes for item in report.logical_segments) == 100
    assert sum(item.logical_bytes for item in report.passes) > 100
    assert report.accounting.logical_state_bytes == 100


def test_duplicate_tracer_event_is_rejected_even_when_pass_ids_differ() -> None:
    original = _passes()[0]
    duplicate = original.model_copy(update={"record_id": "other-tracer-record"})

    with pytest.raises(MovementAccountingError, match="physical byte event"):
        build_state_movement_report(logical_segments=_segments(), passes=(original, duplicate))


def test_direct_report_json_revalidates_duplicate_physical_event_ids() -> None:
    original = build_state_movement_report(logical_segments=_segments(), passes=_passes())
    duplicate = original.passes[0].model_copy(
        update={"record_id": "other-tracer-record", "start_ns": 61, "end_ns": 71}
    )
    # Constructing with Pydantic's trusted-data escape hatch simulates a
    # self-consistent untrusted wire payload containing the duplicate pass.
    passes = tuple(sorted((*original.passes, duplicate), key=lambda item: item.start_ns))
    from sloforge.helix.characterization.gpu_reclamation_accounting import (
        _account,
        _build_edges,
    )

    edges = _build_edges(original.logical_segments, passes)
    raw = StateMovementReport.model_construct(
        logical_segments=original.logical_segments,
        passes=passes,
        edges=edges,
        accounting=_account(original.logical_segments, passes, edges),
    ).model_dump_json()
    with pytest.raises(ValidationError, match="physical byte event"):
        StateMovementReport.model_validate_json(raw, strict=True)


def test_exact_renamed_duplicate_without_event_ids_is_rejected() -> None:
    original = _passes()[0].model_copy(update={"read_event_id": None, "write_event_id": None})
    duplicate = original.model_copy(update={"record_id": "renamed-duplicate"})

    with pytest.raises(MovementAccountingError, match="duplicate state-pass observation"):
        build_state_movement_report(logical_segments=_segments(), passes=(original, duplicate))


def test_direction_and_temporary_allocation_validation_fail_closed() -> None:
    with pytest.raises(ValidationError, match="D2H traffic"):
        _pass(
            "backwards",
            operation=StatePassOperation.D2H,
            source=MemoryDomain.PINNED_HOST_TRANSPORT,
            destination=MemoryDomain.GPU_TRANSFORM_BUFFER,
            read=80,
            written=80,
            direction=TransferDirection.D2H,
            transfer=80,
            start=0,
        )

    with pytest.raises(ValidationError, match="stable allocation ID"):
        StatePassRecord(
            record_id="missing-allocation-id",
            state_segment="shared-root",
            branch_group="group-004",
            operation=StatePassOperation.TRANSFORM,
            source_memory=MemoryDomain.SOURCE_GPU_NATIVE_PAGED,
            destination_memory=MemoryDomain.GPU_TRANSFORM_BUFFER,
            bytes_read=80,
            bytes_written=80,
            logical_bytes=80,
            start_ns=0,
            end_ns=10,
            device="cuda:1",
            temporary_allocation_bytes=80,
            temporary_allocation_memory=MemoryDomain.GPU_TRANSFORM_BUFFER,
            required_unavoidable=False,
        )


def test_logical_range_and_branch_group_are_validated() -> None:
    too_wide = _passes()[0].model_copy(update={"logical_bytes": 81})
    with pytest.raises(MovementAccountingError, match="exceeds its logical segment"):
        build_state_movement_report(logical_segments=_segments(), passes=(too_wide,))

    wrong_group = _passes()[0].model_copy(update={"branch_group": "wrong"})
    with pytest.raises(MovementAccountingError, match="branch group differs"):
        build_state_movement_report(logical_segments=_segments(), passes=(wrong_group,))


def test_temporary_peak_uses_half_open_lifetimes() -> None:
    first = _passes()[0]
    simultaneous = first.model_copy(
        update={
            "record_id": "simultaneous",
            "start_ns": 5,
            "end_ns": 15,
            "read_event_id": "read-simultaneous",
            "write_event_id": "write-simultaneous",
            "temporary_allocation_id": "allocation-simultaneous",
        }
    )
    adjacent = first.model_copy(
        update={
            "record_id": "adjacent",
            "start_ns": 10,
            "end_ns": 20,
            "read_event_id": "read-adjacent",
            "write_event_id": "write-adjacent",
            "temporary_allocation_id": "allocation-adjacent",
        }
    )

    report = build_state_movement_report(
        logical_segments=_segments(), passes=(first, simultaneous, adjacent)
    )
    assert report.accounting.gpu_temporary_allocation_bytes == 300
    assert report.accounting.peak_gpu_temporary_allocation_bytes == 200


def test_canonical_serialization_is_input_order_independent_and_round_trips() -> None:
    forward = build_state_movement_report(logical_segments=_segments(), passes=_passes())
    reverse = build_state_movement_report(
        logical_segments=tuple(reversed(_segments())),
        passes=tuple(reversed(_passes())),
    )

    assert forward == reverse
    assert canonical_movement_json(forward) == canonical_movement_json(reverse)
    payload = json.loads(canonical_movement_json(forward))
    assert payload["schema_version"] == "sloforge.branchfabric.movement-report/v1"
    assert payload["accounting"]["state_movement_amplification"] == 10.2
    assert tuple(item["record_id"] for item in payload["passes"]) == tuple(
        item.record_id for item in forward.passes
    )
