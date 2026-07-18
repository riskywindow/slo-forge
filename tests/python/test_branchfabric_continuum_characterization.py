from __future__ import annotations

from sloforge.continuum.characterization import (
    ListRecorder,
    MeasurementKind,
    TraceLevel,
    measure_instrumentation_overhead,
    run_continuum_characterization,
)


class _FailingRecorder:
    def record(self, _observation) -> None:
        raise AssertionError("disabled tracing invoked the recorder")


class _CollectingRecorder:
    def __init__(self) -> None:
        self.events = []

    def record(self, observation) -> None:
        self.events.append(observation)


def test_disabled_characterization_preserves_semantics_and_never_calls_recorder() -> None:
    disabled = run_continuum_characterization(
        seed=101, trace_level=TraceLevel.DISABLED, recorder=_FailingRecorder()
    )
    full = run_continuum_characterization(seed=101, trace_level=TraceLevel.FULL)

    assert disabled.events == ()
    assert disabled.source_continuation_hash == full.source_continuation_hash
    assert disabled.divergent_continuation_hash == full.divergent_continuation_hash
    assert disabled.migration_transaction_id == full.migration_transaction_id
    assert disabled.source_continuation_hash != disabled.divergent_continuation_hash


def test_full_trace_covers_real_state_lifecycle_and_preserves_provenance() -> None:
    result = run_continuum_characterization(seed=211, trace_level=TraceLevel.FULL)
    operations = {event.operation for event in result.events}

    assert {
        "STATE_FORK",
        "STATE_COW",
        "STATE_APPEND",
        "STATE_SNAPSHOT",
        "STATE_DELTA",
        "STATE_RESHARD",
        "STATE_SEND",
        "STATE_RECEIVE",
        "STATE_COMMIT",
        "STATE_RECLAIM",
    } <= operations
    assert [event.sequence for event in result.events] == list(range(len(result.events)))
    assert all(event.duration_ns > 0 and event.cpu_time_ns > 0 for event in result.events)
    assert all(event.evidence_references for event in result.events)
    assert result.dropped_events == 0
    assert result.migration_transport_kind is MeasurementKind.SIMULATED_HARDWARE
    cow = next(event for event in result.events if event.operation == "STATE_COW")
    assert cow.metadata["cow_implementation"].endswith("not_an_os_or_gpu_page_fault")
    simulated = [
        event
        for event in result.events
        if event.measurement_kind is MeasurementKind.SIMULATED_HARDWARE
    ]
    assert simulated
    assert all(event.metadata["modeled_transport"] for event in simulated)
    assert all(event.transfer_time_ns > 0 for event in simulated)
    assert all(
        "gpu" not in event.source_physical_representation.lower()
        for event in result.events
        if event.measurement_kind is MeasurementKind.HARDWARE_BACKED_REAL
    )


def test_sharing_and_cow_metrics_are_derived_from_continuum_artifacts() -> None:
    result = run_continuum_characterization(seed=307)
    sharing = result.sharing

    assert sharing.branch_count == 2
    assert sharing.shared_logical_bytes > 0
    assert sharing.logical_unique_bytes == sharing.physical_allocated_bytes
    assert sharing.naive_independent_allocation_bytes > sharing.physical_allocated_bytes
    assert 0.0 < sharing.sharing_efficiency < 1.0
    assert sharing.physical_amplification == 1.0
    assert sharing.source_page_refcount == 3
    assert sharing.cow_page_count > 0
    assert sharing.cow_bytes > 0
    assert len(sharing.metadata_bytes_per_branch) == 2
    assert all(size > 0 for size in sharing.metadata_bytes_per_branch)


def test_recorder_bound_exposes_dropped_events() -> None:
    recorder = ListRecorder(maximum_events=3)
    result = run_continuum_characterization(
        seed=401, trace_level=TraceLevel.FULL, recorder=recorder
    )

    assert len(result.events) == 3
    assert result.dropped_events > 0


def test_generic_recorder_receives_result_derived_snapshot_and_delta_sizes() -> None:
    recorder = _CollectingRecorder()
    result = run_continuum_characterization(
        seed=449, trace_level=TraceLevel.FULL, recorder=recorder
    )

    assert result.events == ()
    initial_snapshot = recorder.events[0]
    delta = next(event for event in recorder.events if event.operation == "STATE_DELTA")
    assert initial_snapshot.operation == "STATE_SNAPSHOT"
    assert initial_snapshot.bytes > 0
    assert initial_snapshot.page_size > 0
    assert delta.bytes > 0
    assert delta.metadata["changed_segment_count"] > 0


def test_overhead_study_preserves_raw_samples_and_randomized_order() -> None:
    overhead = measure_instrumentation_overhead(seed=503, repetitions=1)

    assert len(overhead.raw_samples) == 3
    assert set(overhead.randomized_order) == {level.value for level in TraceLevel}
    assert set(overhead.median_duration_ns) == {level.value for level in TraceLevel}
    assert overhead.duration_overhead_fraction[TraceLevel.DISABLED.value] == 0.0
    assert all(sample.duration_ns > 0 and sample.cpu_time_ns > 0 for sample in overhead.raw_samples)
