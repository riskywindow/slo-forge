from __future__ import annotations

from sloforge.continuum.characterization import (
    ListRecorder,
    MeasurementKind,
    StateOperationObservation,
)
from sloforge.helix.characterization.lifecycle import TraceStream
from sloforge.helix.characterization.trace import (
    BoundedTraceBuffer,
    CanonicalContinuumRecorder,
    CanonicalLifecycleRecorder,
    StateOperationEventV1,
    StateSegment,
    TimingMeasurementClass,
    TraceLevel,
    WorkloadProvenance,
    verify_event,
)


def test_lifecycle_adapter_preserves_provenance_and_source_attributes() -> None:
    buffer = BoundedTraceBuffer(trace_id="trace-a", capacity_events=8, level=TraceLevel.FULL)
    recorder = CanonicalLifecycleRecorder(buffer)
    recorder.record(
        TraceStream.BRANCH_WORKLOAD,
        {
            "trace_id": "trace-a",
            "session_id": "session-a",
            "branch_group_id": None,
            "branch_id": None,
            "parent_branch_id": "production-session",
            "policy_epoch": "policy-a",
            "environment_id": "env-a",
            "transaction_id": None,
            "host": "host-a",
            "process": 4,
            "rank": 0,
            "device": "cpu",
            "monotonic_timestamp_ns": 100,
            "normalized_timestamp_ns": 0,
            "duration_ns": 20,
            "cpu_time_ns": 19,
            "clock_source": "time.perf_counter_ns",
            "alignment_confidence": 1.0,
            "operation_type": "BRANCH_POINT_CAPTURE",
            "logical_state_id": "state-a",
            "physical_state_id": "physical-a",
            "state_segment": "model",
            "logical_bytes": 128,
            "metadata_bytes": 64,
            "event_sequence": 17,
            "schema_version": "lifecycle-observation/v1",
            "trace_producer_version": "producer-a",
            "measurement_source": "SYNTHETIC",
            "workload_evidence_class": "SYNTHETIC",
            "timing_measurement_class": "HARDWARE_BACKED_REAL",
            "simulated_gpu_state": True,
            "seed": 41,
            "result": "success",
            "content_hash": "1" * 64,
            "generated_tokens": 3,
        },
    )
    (event,) = buffer.drain()
    assert event.provenance is WorkloadProvenance.SYNTHETIC
    assert event.timing_measurement_class is TimingMeasurementClass.HARDWARE_BACKED_REAL
    assert event.state_segment is StateSegment.MODEL
    assert event.branch_group_id is None
    assert event.attributes["generated_tokens"] == 3
    assert event.attributes["source_event_content_hash"] == "1" * 64
    verify_event(event)


def test_continuum_adapter_keeps_modeled_transport_separate() -> None:
    buffer = BoundedTraceBuffer(trace_id="trace-c", capacity_events=8, level=TraceLevel.FULL)
    recorder = CanonicalContinuumRecorder(
        buffer,
        trace_id="trace-c",
        session_id="session-c",
        branch_group_id="group-c",
    )
    observation = StateOperationObservation(
        sequence=2,
        operation="STATE_SEND",
        monotonic_start_ns=1000,
        duration_ns=50,
        cpu_time_ns=20,
        logical_state_id="state-c",
        branch_id="branch-c",
        tenant_security_domain="tenant-c",
        source_physical_representation="host_dram",
        destination_physical_representation="simulated_network",
        bytes=4096,
        alignment=64,
        page_size=4096,
        chunk_size=1024,
        fanout=1,
        dependencies=(1,),
        concurrency=1,
        queue_delay_ns=0,
        transfer_time_ns=400,
        result="success",
        failure=None,
        state_epoch=2,
        workload_evidence_class=MeasurementKind.SYNTHETIC,
        measurement_kind=MeasurementKind.SIMULATED_HARDWARE,
        seed=41,
        evidence_references=("artifact.json",),
        state_segment="model",
        metadata={"modeled_transport": True},
    )
    recorder.record(observation)
    (event,) = buffer.drain()
    assert isinstance(event, StateOperationEventV1)
    assert event.provenance is WorkloadProvenance.SYNTHETIC
    assert event.timing_measurement_class is TimingMeasurementClass.SIMULATED_HARDWARE
    assert event.transfer_time_ns == 400
    assert event.dependency_event_ids == ("trace-c:1",)
    assert event.attributes["modeled_transport"] is True
    verify_event(event)


def test_unrelated_list_recorder_remains_bounded() -> None:
    recorder = ListRecorder(maximum_events=1)
    assert recorder.maximum_events == 1
