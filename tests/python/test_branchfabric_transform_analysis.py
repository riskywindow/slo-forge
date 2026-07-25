from __future__ import annotations

from sloforge.helix.characterization.analysis.transform import TransformAnalysis, analyze_transforms
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
    dependencies: tuple[str, ...] = (),
    bytes_count: int = 4096,
    latency_ns: int = 100,
    cpu_time_ns: int = 25,
    gpu_time_ns: int = 0,
    source: str = "layout-a",
    destination: str = "layout-b",
) -> StateOperationEventV1:
    return StateOperationEventV1(
        collection_level=TraceLevel.FULL,
        provenance=WorkloadProvenance.SYNTHETIC,
        timing_measurement_class=TimingMeasurementClass.HARDWARE_BACKED_REAL,
        trace_id="trace-transform",
        session_id="session-41",
        branch_group_id="group-41",
        logical_state_id="state-kv",
        branch_id="branch-a",
        tenant_id="tenant-a",
        security_domain="domain-a",
        host="host-a",
        process_id=100,
        device="cpu",
        monotonic_timestamp_ns=sequence * 1_000,
        normalized_timestamp_ns=sequence * 1_000,
        duration_ns=latency_ns,
        clock_source=ClockSource.PERF_COUNTER,
        alignment_confidence=1.0,
        operation_type=operation,
        state_segment=StateSegment.KV,
        source_physical_representation=source,
        destination_physical_representation=destination,
        bytes=bytes_count,
        alignment_bytes=256,
        page_size_bytes=16 * 1024,
        chunk_size_bytes=1024,
        fanout=1,
        dependency_event_ids=dependencies,
        concurrency=1,
        queue_delay_ns=0,
        operation_latency_ns=latency_ns,
        cpu_time_ns=cpu_time_ns,
        gpu_time_ns=gpu_time_ns,
        transfer_time_ns=latency_ns if operation is StateOperationType.STATE_SEND else 0,
        result=OperationResult.SUCCESS,
        state_epoch=7,
        source_location=MemoryLocation.HOST_DRAM,
        destination_location=MemoryLocation.HOST_DRAM,
        transport_type=(
            TransportType.MEMORY_COPY
            if operation is StateOperationType.STATE_SEND
            else TransportType.NONE
        ),
        event_sequence=sequence,
        content_hash="0" * 64,
    )


def _analyze(events: list[StateOperationEventV1]) -> TransformAnalysis:
    return analyze_transforms(
        events,
        source_experiment="transform-seed-41",
        artifact_reference="artifacts/branchfabric/raw/transform-41.jsonl",
        artifact_sha256="b" * 64,
        seed=41,
        repetition=0,
    )


def test_explicit_dependencies_extract_common_transform_pipeline() -> None:
    operations = (
        StateOperationType.STATE_READ,
        StateOperationType.STATE_DEQUANTIZE,
        StateOperationType.STATE_RESHARD,
        StateOperationType.STATE_REPACK,
        StateOperationType.STATE_CHECKSUM,
        StateOperationType.STATE_SEND,
    )
    events = [
        _event(
            sequence,
            operation,
            dependencies=() if sequence == 1 else (f"trace-transform:{sequence - 1}",),
        )
        for sequence, operation in enumerate(operations, start=1)
    ]

    result = _analyze(events)

    assert result.transform_event_count == 3
    assert len(result.operation_aggregates) == 3
    assert len(result.sequence_aggregates) == 1
    sequence = result.sequence_aggregates[0]
    assert sequence.operations == operations
    assert sequence.occurrence_count == 1
    assert sequence.processed_bytes == 6 * 4096
    assert sequence.operation_latency_ns.total == 600
    assert sequence.temporary_bytes is None
    fusion = sequence.fusion_upper_bound
    assert fusion.analytical is True
    assert fusion.occurrence_count == 1
    assert fusion.transform_stage_count_per_occurrence == 3
    assert fusion.transform_stage_observation_count == 3
    assert fusion.observed_materialized_stage_bytes == 3 * 4096
    assert fusion.ideal_single_stream_bytes == 4096
    assert fusion.maximum_intermediate_bytes_avoided == 2 * 4096
    assert fusion.maximum_latency_reduction_ns == 300
    assert fusion.temporary_capacity_bytes is None
    assert result.sequence_ranking.by_frequency == (sequence.sequence_id,)


def test_missing_dependencies_do_not_infer_a_chain_from_timestamp_adjacency() -> None:
    events = [
        _event(1, StateOperationType.STATE_DEQUANTIZE),
        _event(2, StateOperationType.STATE_REPACK),
    ]

    result = _analyze(events)

    assert len(result.sequence_aggregates) == 2
    # Two standalone stages have the same aggregation count only if their
    # operation sequences are identical; these are intentionally separate.
    assert {item.operations for item in result.sequence_aggregates} == {
        (StateOperationType.STATE_DEQUANTIZE,),
        (StateOperationType.STATE_REPACK,),
    }


def test_unresolved_ambiguous_and_noncausal_dependencies_are_visible() -> None:
    events = [
        _event(1, StateOperationType.STATE_READ),
        _event(2, StateOperationType.STATE_DEQUANTIZE),
        _event(
            3,
            StateOperationType.STATE_REPACK,
            dependencies=("trace-transform:1", "trace-transform:2", "missing-event"),
        ),
        _event(
            4,
            StateOperationType.STATE_RESHARD,
            dependencies=("trace-transform:5",),
        ),
        _event(5, StateOperationType.STATE_CHECKSUM),
    ]

    result = _analyze(events)

    assert result.unresolved_dependency_count == 1
    assert result.ambiguous_pipeline_predecessor_count == 1
    assert result.noncausal_dependency_count == 1


def test_missing_transform_resource_counters_remain_unknown() -> None:
    result = _analyze(
        [
            _event(
                1,
                StateOperationType.STATE_COMPRESS,
                latency_ns=0,
                cpu_time_ns=0,
                gpu_time_ns=0,
            )
        ]
    )

    aggregate = result.operation_aggregates[0]
    assert aggregate.operation_latency is None
    assert aggregate.operation_latency_total_ns.total is None
    assert aggregate.cpu_time_ns.total is None
    assert aggregate.gpu_time_ns.total is None
    assert aggregate.temporary_bytes is None
    assert aggregate.memory_read_bytes is None
    assert aggregate.memory_write_bytes is None
    assert any("temporary" in item for item in result.unknown_counters)
    assert any("CPU time" in item for item in result.unknown_counters)


def test_producer_declared_fused_chain_is_preserved_without_fake_stage_timings() -> None:
    event = _event(1, StateOperationType.STATE_RESHARD).model_copy(
        update={
            "attributes": {
                "operation_chain_json": (
                    '["STATE_RESHARD","STATE_REPACK","STATE_CHECKSUM"]'
                )
            }
        }
    )
    result = _analyze([event])
    (declared,) = result.declared_fused_operation_chains
    assert declared.operations == (
        StateOperationType.STATE_RESHARD,
        StateOperationType.STATE_REPACK,
        StateOperationType.STATE_CHECKSUM,
    )
    assert declared.inclusive_latency_ns == 100
    assert declared.timing_decomposable is False
    assert result.transform_event_count == 1


def test_sequence_ranking_uses_frequency_bytes_and_observed_latency() -> None:
    first = [
        _event(1, StateOperationType.STATE_QUANTIZE),
        _event(
            2,
            StateOperationType.STATE_SEND,
            dependencies=("trace-transform:1",),
        ),
    ]
    # A second trace is needed because canonical event references are trace-local.
    second = [
        event.model_copy(
            update={
                "trace_id": "trace-transform-2",
                "event_sequence": event.event_sequence,
                "dependency_event_ids": ()
                if event.event_sequence == 1
                else ("trace-transform-2:1",),
            }
        )
        for event in first
    ]

    result = _analyze([*first, *second])

    sequence = result.sequence_aggregates[0]
    assert sequence.occurrence_count == 2
    assert sequence.fusion_upper_bound.occurrence_count == 2
    assert sequence.fusion_upper_bound.transform_stage_observation_count == 2
    assert sequence.fusion_upper_bound.observed_materialized_stage_bytes == 8192
    assert result.sequence_ranking.by_frequency[0] == sequence.sequence_id
    assert result.sequence_ranking.by_processed_bytes[0] == sequence.sequence_id
    assert result.sequence_ranking.by_observed_latency[0] == sequence.sequence_id
