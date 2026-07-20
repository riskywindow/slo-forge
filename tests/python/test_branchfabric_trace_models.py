from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import pytest
from pydantic import ValidationError

from sloforge.helix.characterization.trace import (
    BranchOperationType,
    BranchWorkloadEventV1,
    ClockSource,
    HardwareManifestV1,
    MemoryLocation,
    OperationResult,
    SamplingConfigurationV1,
    SoftwareManifestV1,
    StateOperationEventV1,
    StateOperationType,
    StateSegment,
    TimingMeasurementClass,
    TraceArtifactV1,
    TraceBufferStats,
    TraceIntegrityError,
    TraceLevel,
    WorkloadProvenance,
    build_manifest,
    canonical_hash,
    seal_event,
    trace_corpus_hash,
    verify_event,
)

ROOT = Path(__file__).parents[2]


def branch_event(**updates: object) -> BranchWorkloadEventV1:
    values: dict[str, object] = {
        "provenance": WorkloadProvenance.SYNTHETIC,
        "timing_measurement_class": TimingMeasurementClass.HARDWARE_BACKED_REAL,
        "trace_id": "trace-1",
        "session_id": "session-1",
        "branch_group_id": "group-1",
        "branch_id": "branch-1",
        "parent_branch_id": "root",
        "policy_epoch": "policy-3",
        "environment_id": "env-1",
        "transaction_id": "transaction-1",
        "host": "host-a",
        "process_id": 19,
        "rank": 0,
        "device": "cpu",
        "monotonic_timestamp_ns": 1_000,
        "normalized_timestamp_ns": 2_000,
        "duration_ns": 50,
        "clock_source": ClockSource.MONOTONIC,
        "alignment_confidence": 1.0,
        "operation_type": BranchOperationType.BRANCH_FORK,
        "logical_state_id": "logical-1",
        "physical_state_id": "physical-1",
        "state_segment": StateSegment.KV,
        "logical_bytes": 16_384,
        "physical_bytes": 4_096,
        "metadata_bytes": 128,
        "location": MemoryLocation.HOST_DRAM,
        "source_location": MemoryLocation.HOST_DRAM,
        "destination_location": MemoryLocation.HOST_DRAM,
        "shared_root": True,
        "queue_delay_ns": 5,
        "execution_latency_ns": 35,
        "wait_latency_ns": 10,
        "attributes": {"timing_span_id": "trace-1:fork", "generated_tokens": 4},
    }
    values.update(updates)
    return BranchWorkloadEventV1.model_validate(values)


def state_event(**updates: object) -> StateOperationEventV1:
    values: dict[str, object] = {
        "provenance": WorkloadProvenance.SYNTHETIC,
        "timing_measurement_class": TimingMeasurementClass.HARDWARE_BACKED_REAL,
        "trace_id": "trace-1",
        "session_id": "session-1",
        "branch_group_id": "group-1",
        "logical_state_id": "logical-1",
        "branch_id": "branch-1",
        "tenant_id": "tenant-1",
        "security_domain": "sandbox-1",
        "host": "host-a",
        "process_id": 19,
        "rank": 0,
        "device": "cpu",
        "monotonic_timestamp_ns": 1_050,
        "normalized_timestamp_ns": 2_050,
        "duration_ns": 25,
        "clock_source": ClockSource.MONOTONIC,
        "alignment_confidence": 1.0,
        "operation_type": StateOperationType.STATE_FORK,
        "state_segment": StateSegment.KV,
        "source_physical_representation": "continuum.kv/v1:host",
        "destination_physical_representation": "continuum.kv/v1:host-cow",
        "bytes": 4_096,
        "alignment_bytes": 64,
        "page_size_bytes": 4_096,
        "chunk_size_bytes": 4_096,
        "fanout": 4,
        "dependency_event_ids": ("capture-1",),
        "concurrency": 2,
        "queue_delay_ns": 4,
        "operation_latency_ns": 21,
        "cpu_time_ns": 20,
        "result": OperationResult.SUCCESS,
        "state_epoch": 3,
        "source_location": MemoryLocation.HOST_DRAM,
        "destination_location": MemoryLocation.HOST_DRAM,
        "attributes": {"dirty_bytes": 4096},
    }
    values.update(updates)
    return StateOperationEventV1.model_validate(values)


def _schema(name: str) -> dict[str, object]:
    return json.loads((ROOT / "schemas" / "branchfabric" / name).read_text())


def test_branch_and_state_events_conform_to_formal_schemas() -> None:
    branch = seal_event(branch_event(), event_sequence=7)
    state = seal_event(state_event(), event_sequence=8)
    jsonschema.validate(
        branch.model_dump(mode="json"), _schema("branch-workload-trace-v1.schema.json")
    )
    jsonschema.validate(
        state.model_dump(mode="json"), _schema("state-operation-trace-v1.schema.json")
    )


def test_state_operation_vocabulary_is_complete_and_exact() -> None:
    expected = {
        "STATE_ALLOC",
        "STATE_MAP",
        "STATE_PUBLISH",
        "STATE_FORK",
        "STATE_COW",
        "STATE_APPEND",
        "STATE_READ",
        "STATE_WRITE",
        "STATE_SNAPSHOT",
        "STATE_DELTA",
        "STATE_RESHARD",
        "STATE_TRANSPOSE",
        "STATE_REPACK",
        "STATE_QUANTIZE",
        "STATE_DEQUANTIZE",
        "STATE_COMPRESS",
        "STATE_DECOMPRESS",
        "STATE_HASH",
        "STATE_CHECKSUM",
        "STATE_ENCRYPT",
        "STATE_DECRYPT",
        "STATE_SEND",
        "STATE_RECEIVE",
        "STATE_MULTICAST",
        "STATE_ACK",
        "STATE_RETRY",
        "STATE_COMMIT",
        "STATE_ABORT",
        "STATE_RECLAIM",
        "STATE_FREE",
    }
    assert {operation.value for operation in StateOperationType} == expected
    assert expected <= {operation.value for operation in BranchOperationType}


def test_unresolved_branch_group_and_nonpaged_granularities_are_honest() -> None:
    branch = seal_event(branch_event(branch_group_id=None), event_sequence=0)
    state = seal_event(
        state_event(
            branch_group_id=None,
            alignment_bytes=0,
            page_size_bytes=0,
            chunk_size_bytes=0,
            state_segment=StateSegment.TRANSACTION,
        ),
        event_sequence=1,
    )
    assert branch.branch_group_id is None
    assert (state.alignment_bytes, state.page_size_bytes, state.chunk_size_bytes) == (0, 0, 0)
    assert state.provenance is WorkloadProvenance.SYNTHETIC
    assert state.timing_measurement_class is TimingMeasurementClass.HARDWARE_BACKED_REAL
    jsonschema.validate(
        branch.model_dump(mode="json"), _schema("branch-workload-trace-v1.schema.json")
    )
    jsonschema.validate(
        state.model_dump(mode="json"), _schema("state-operation-trace-v1.schema.json")
    )


def test_content_hash_is_deterministic_and_detects_mutation() -> None:
    first = seal_event(branch_event(), event_sequence=11)
    second = seal_event(branch_event(), event_sequence=11)
    assert first.content_hash == second.content_hash
    verify_event(first)

    tampered = first.model_copy(update={"physical_bytes": first.physical_bytes + 1})
    with pytest.raises(TraceIntegrityError, match="content hash mismatch"):
        verify_event(tampered)


def test_strict_models_reject_coercion_extra_fields_and_invalid_failure() -> None:
    with pytest.raises(ValidationError):
        branch_event(process_id="19")
    with pytest.raises(ValidationError):
        branch_event(unknown_field=True)
    with pytest.raises(ValidationError, match="failed operations require failure detail"):
        state_event(result=OperationResult.FAILURE)
    with pytest.raises(ValidationError, match="only failed operations"):
        state_event(failure="not allowed")
    with pytest.raises(ValidationError, match="64-entry bound"):
        state_event(attributes={f"key-{index}": index for index in range(65)})
    with pytest.raises(ValidationError, match="finite"):
        branch_event(attributes={"invalid": float("nan")})


def test_trace_manifest_balances_counts_and_conforms_to_schema() -> None:
    artifact = TraceArtifactV1(
        format="jsonl",
        uri="traces/trace-1.jsonl",
        byte_length=1234,
        sha256=canonical_hash({"raw": "fixture"}),
        event_count=2,
    )
    stats = TraceBufferStats(
        capacity_events=128,
        attempted_events=4,
        accepted_events=2,
        dropped_events=1,
        filtered_events=1,
        buffered_events=0,
        highest_event_sequence=3,
        event_counts={"BRANCH_FORK": 1, "STATE_FORK": 1},
    )
    hardware = HardwareManifestV1(
        host="host-a",
        machine="arm64",
        cpu_model="fixture-cpu",
        logical_cpu_count=8,
        memory_bytes=16 * 1024**3,
        numa_node_count=1,
    )
    software = SoftwareManifestV1(
        operating_system="fixture-os",
        kernel="fixture-kernel",
        python_version="3.12.0",
    )
    sampling = SamplingConfigurationV1()
    manifest = build_manifest(
        trace_id="trace-1",
        session_id="session-1",
        created_at="2026-08-09T00:00:00Z",
        seed=7,
        provenance=WorkloadProvenance.SYNTHETIC,
        collection_level=TraceLevel.FULL,
        stats=stats,
        sampling=sampling,
        hardware=hardware,
        software=software,
        artifacts=(artifact,),
    )
    assert manifest.trace_corpus_hash == trace_corpus_hash("trace-1", (artifact,))
    jsonschema.validate(manifest.model_dump(mode="json"), _schema("trace-manifest-v1.schema.json"))

    with pytest.raises(ValidationError, match="accounting does not balance"):
        manifest.model_copy(update={"attempted_events": 5}, deep=True).__class__.model_validate(
            {**manifest.model_dump(), "attempted_events": 5}
        )


def test_manifest_supports_raw_resource_and_analysis_artifacts() -> None:
    for artifact_format in ("raw", "resource", "analysis"):
        artifact = TraceArtifactV1(
            format=artifact_format,
            uri=f"artifacts/{artifact_format}.json",
            byte_length=1,
            sha256=canonical_hash({"format": artifact_format}),
            event_count=0,
        )
        assert artifact.format == artifact_format
