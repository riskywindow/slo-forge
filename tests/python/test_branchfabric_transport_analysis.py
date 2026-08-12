from __future__ import annotations

import pytest
from pydantic import ValidationError

from sloforge.helix.characterization.analysis.transport import TransportAnalysis, analyze_transport
from sloforge.helix.characterization.trace.models import (
    ClockSource,
    MemoryLocation,
    OperationResult,
    StateOperationEventV1,
    StateOperationType,
    StateSegment,
    TimingMeasurementClass,
    TraceLevel,
    TransportType,
    WorkloadProvenance,
)


def _event(
    sequence: int,
    operation: StateOperationType,
    *,
    branch_id: str | None = "branch-a",
    dependencies: tuple[str, ...] = (),
    bytes_count: int = 1024,
    fanout: int = 1,
    destination_representation: str = "paged-kv/v1",
    transfer_time_ns: int = 100,
    chunk_size_bytes: int = 256,
    result: OperationResult = OperationResult.SUCCESS,
) -> StateOperationEventV1:
    return StateOperationEventV1(
        collection_level=TraceLevel.FULL,
        provenance=WorkloadProvenance.SYNTHETIC,
        timing_measurement_class=TimingMeasurementClass.HARDWARE_BACKED_REAL,
        trace_id="trace-41",
        session_id="session-41",
        branch_group_id="group-41",
        logical_state_id="state-shared",
        branch_id=branch_id,
        tenant_id="tenant-a",
        security_domain="domain-a",
        host="host-a",
        process_id=100,
        device=f"device-{branch_id}" if branch_id else None,
        monotonic_timestamp_ns=sequence * 1_000,
        normalized_timestamp_ns=sequence * 1_000,
        duration_ns=110,
        clock_source=ClockSource.PERF_COUNTER,
        alignment_confidence=1.0,
        operation_type=operation,
        state_segment=StateSegment.KV,
        source_physical_representation="paged-kv/source",
        destination_physical_representation=destination_representation,
        bytes=bytes_count,
        alignment_bytes=256,
        page_size_bytes=16 * 1024,
        chunk_size_bytes=chunk_size_bytes,
        fanout=fanout,
        dependency_event_ids=dependencies,
        concurrency=2,
        queue_delay_ns=10,
        operation_latency_ns=110,
        cpu_time_ns=20,
        gpu_time_ns=0,
        transfer_time_ns=transfer_time_ns,
        result=result,
        state_epoch=7,
        source_location=MemoryLocation.HOST_DRAM,
        destination_location=MemoryLocation.GPU_HBM,
        transport_type=TransportType.PCIE,
        event_sequence=sequence,
        content_hash="0" * 64,
    )


def _analyze(events: list[StateOperationEventV1]) -> TransportAnalysis:
    return analyze_transport(
        events,
        source_experiment="transport-seed-41",
        artifact_reference="artifacts/branchfabric/raw/transport-41.jsonl",
        artifact_sha256="a" * 64,
        seed=41,
        repetition=0,
    )


def test_transport_aggregation_preserves_counters_and_provenance() -> None:
    events = [
        _event(1, StateOperationType.STATE_SEND, branch_id="branch-a"),
        _event(2, StateOperationType.STATE_RECEIVE, branch_id="branch-a"),
        _event(
            3,
            StateOperationType.STATE_RETRY,
            branch_id="branch-a",
            result=OperationResult.RETRY,
            bytes_count=128,
        ),
    ]

    result = _analyze(events)

    assert result.evidence.input_event_count == 3
    assert result.evidence.workload_provenance == (WorkloadProvenance.SYNTHETIC,)
    assert result.evidence.timing_measurement_classes == (
        TimingMeasurementClass.HARDWARE_BACKED_REAL,
    )
    assert result.source_payload_bytes == 1024
    assert result.logical_delivery_bytes == 1024
    assert result.receive_payload_bytes == 1024
    assert result.retry_payload_bytes == 128
    send = next(
        item for item in result.aggregates if item.operation_type is StateOperationType.STATE_SEND
    )
    assert send.transfer_time_total_ns == 100
    assert send.observed_throughput_bytes_per_second == pytest.approx(10.24e9)
    assert send.chunk_size is not None and send.chunk_size.p99 == 256
    assert send.queue_delay is not None and send.queue_delay.p50 == 10
    assert send.cpu_time is not None and send.cpu_time.p50 == 20
    assert send.gpu_time is None
    assert send.concurrency.p50 == 2
    assert send.source_physical_representation == "paged-kv/source"
    assert send.destination_physical_representation == "paged-kv/v1"
    assert send.devices == ("device-branch-a",)


def test_repeated_unicast_requires_logical_identity_and_shared_dependencies() -> None:
    proven = [
        _event(
            sequence,
            StateOperationType.STATE_SEND,
            branch_id=f"branch-{sequence}",
            dependencies=("fork-7",),
        )
        for sequence in range(1, 4)
    ]
    unproven = [
        _event(10, StateOperationType.STATE_SEND, branch_id="unproven-a"),
        _event(11, StateOperationType.STATE_SEND, branch_id="unproven-b"),
    ]

    result = _analyze([*proven, *unproven])

    assert result.multicast.opportunity_count == 1
    opportunity = result.multicast.opportunities[0]
    assert opportunity.fanout == 3
    assert opportunity.payload_bytes_per_destination == 1024
    assert opportunity.repeated_unicast_source_bytes == 3072
    assert opportunity.exact_payload_reduction_bytes == 2048
    assert opportunity.tree_root_egress_reduction_bytes == 2048
    assert opportunity.host_assisted_multicast_reduction_bytes == 2048
    assert opportunity.network_multicast_reduction_bytes == 2048
    assert opportunity.destination_transform_byte_reduction_upper_bound == 2048
    assert result.multicast.by_minimum_fanout[0].minimum_fanout == 2
    assert result.multicast.by_minimum_fanout[0].opportunity_count == 1
    assert result.multicast.by_minimum_fanout[1].opportunity_count == 0
    assert result.multicast.ungrouped_send_event_count == 2
    assert result.multicast.event_content_hash_used_as_payload_identity is False


def test_explicit_multicast_and_mixed_layout_bounds_remain_distinct() -> None:
    explicit = _event(1, StateOperationType.STATE_MULTICAST, fanout=4, branch_id=None)
    mixed = [
        _event(
            2,
            StateOperationType.STATE_SEND,
            branch_id="branch-x",
            dependencies=("snapshot-7",),
            destination_representation="layout-a",
        ),
        _event(
            3,
            StateOperationType.STATE_SEND,
            branch_id="branch-y",
            dependencies=("snapshot-7",),
            destination_representation="layout-b",
        ),
    ]

    result = _analyze([explicit, *mixed])

    assert result.multicast.opportunity_count == 2
    mixed_result = next(item for item in result.multicast.opportunities if item.fanout == 2)
    assert mixed_result.exact_payload_layout_match is False
    assert mixed_result.exact_payload_multicast_source_bytes is None
    assert mixed_result.exact_payload_reduction_bytes is None
    assert mixed_result.source_transform_then_multicast_bytes is None
    assert mixed_result.multicast_then_destination_transform_bytes == 1024
    assert mixed_result.destination_transform_byte_reduction_upper_bound == 1024
    explicit_result = next(item for item in result.multicast.opportunities if item.fanout == 4)
    assert explicit_result.observed_delivery_mode.value == "STATE_MULTICAST"


def test_absent_transfer_and_chunk_counters_are_unknown_not_zero_measurements() -> None:
    result = _analyze(
        [
            _event(
                1,
                StateOperationType.STATE_SEND,
                transfer_time_ns=0,
                chunk_size_bytes=0,
            )
        ]
    )

    aggregate = result.aggregates[0]
    assert aggregate.transfer_time_observation_count == 0
    assert aggregate.transfer_time_total_ns is None
    assert aggregate.observed_throughput_bytes_per_second is None
    assert aggregate.transfer_latency is None
    assert aggregate.chunk_size is None
    assert any("transfer_time_ns" in item for item in result.unknown_counters)
    assert any("chunk_size_bytes" in item for item in result.unknown_counters)


def test_analysis_models_reject_unattributed_artifact_hashes() -> None:
    with pytest.raises(ValidationError):
        analyze_transport(
            [_event(1, StateOperationType.STATE_SEND)],
            source_experiment="transport-seed-41",
            artifact_reference="raw.jsonl",
            artifact_sha256="not-a-hash",
            seed=41,
            repetition=0,
        )
