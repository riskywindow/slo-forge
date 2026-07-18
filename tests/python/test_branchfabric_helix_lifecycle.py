from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

from sloforge.helix.characterization.lifecycle import (
    InMemoryLifecycleRecorder,
    TraceLevel,
    TraceStream,
    analyze_branch_state_sharing,
    measure_cpu_demo_overhead,
)


def _event_hash(event: dict[str, object]) -> str:
    document = {key: value for key, value in event.items() if key != "content_hash"}
    payload = json.dumps(
        document,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()
    return sha256(payload).hexdigest()


def test_seed_41_real_demo_lifecycle_and_trace_levels_preserve_semantics(
    tmp_path: Path,
) -> None:
    recorders: list[InMemoryLifecycleRecorder] = []

    def recorder_factory() -> InMemoryLifecycleRecorder:
        recorder = InMemoryLifecycleRecorder()
        recorders.append(recorder)
        return recorder

    samples = measure_cpu_demo_overhead(
        tmp_path / "overhead",
        seed=41,
        recorder_factory=recorder_factory,
        repetitions=1,
    )

    assert {sample.trace_level for sample in samples} == set(TraceLevel)
    assert len({sample.semantic_digest for sample in samples}) == 1
    disabled = next(sample for sample in samples if sample.trace_level is TraceLevel.DISABLED)
    minimal = next(sample for sample in samples if sample.trace_level is TraceLevel.MINIMAL)
    full = next(sample for sample in samples if sample.trace_level is TraceLevel.FULL)
    assert disabled.branch_event_count == disabled.state_event_count == 0
    assert minimal.branch_event_count > 0
    assert minimal.state_event_count > 0
    assert full.branch_event_count > minimal.branch_event_count
    assert full.state_event_count > minimal.state_event_count
    assert all(sample.wall_time_ns > 0 and sample.cpu_time_ns > 0 for sample in samples)
    assert all(Path(sample.artifact_path, "summary.json").is_file() for sample in samples)

    full_recorder = next(
        recorder
        for recorder in recorders
        if any(event["operation_type"] == "STATE_COW" for event in recorder.state_events)
    )
    branch_operations = {event["operation_type"] for event in full_recorder.branch_events}
    assert {
        "BRANCH_POINT_CAPTURE",
        "BRANCH_FORK",
        "ENVIRONMENT_FORK",
        "ROLLOUT_COMPLETE",
        "REWARD_COMPLETE",
        "BRANCH_PRUNE",
        "BRANCH_COMPLETE",
        "TRAINING_COMPLETE",
        "LEARNING_TRANSACTION_STAGE",
        "PROMOTION_COMPLETE",
    } <= branch_operations
    state_operations = {event["operation_type"] for event in full_recorder.state_events}
    assert {"STATE_SNAPSHOT", "STATE_FORK", "STATE_COW", "STATE_FREE"} <= state_operations
    assert (
        len(
            [
                event
                for event in full_recorder.branch_events
                if event["operation_type"] == "BRANCH_FORK"
            ]
        )
        == 4
    )
    assert (
        len(
            [
                event
                for event in full_recorder.branch_events
                if event["operation_type"] == "BRANCH_PRUNE"
            ]
        )
        == 3
    )
    assert (
        len(
            [
                event
                for event in full_recorder.branch_events
                if event["operation_type"] == "BRANCH_COMPLETE"
            ]
        )
        == 1
    )

    all_events = (*full_recorder.branch_events, *full_recorder.state_events)
    assert all(event["measurement_source"] == "SYNTHETIC" for event in all_events)
    assert all(event["workload_evidence_class"] == "SYNTHETIC" for event in all_events)
    assert all(event["timing_measurement_class"] == "HARDWARE_BACKED_REAL" for event in all_events)
    assert all(event["simulated_gpu_state"] is False for event in all_events)
    assert all(int(event["duration_ns"]) >= 0 for event in all_events)
    assert all(event["content_hash"] == _event_hash(event) for event in all_events)
    assert len({int(event["event_sequence"]) for event in all_events}) == len(all_events)

    sharing = analyze_branch_state_sharing(full_recorder.branch_events, full_recorder.state_events)
    assert sharing["branch_count"] == 4
    model = sharing["model_state"]
    assert isinstance(model, dict)
    assert model["shared_logical_bytes"] > 0
    assert model["naive_independent_bytes"] > model["incremental_cas_physical_bytes"]
    environment = sharing["environment_state"]
    assert isinstance(environment, dict)
    assert environment["workspace_implementation"] == "eager_restore_then_atomic_replace"
    cow = sharing["copy_on_write"]
    assert isinstance(cow, dict)
    assert cow["events"] == 4
    assert cow["page_fault_measurement_available"] is False


def test_recorder_takes_mapping_snapshots() -> None:
    recorder = InMemoryLifecycleRecorder()
    event: dict[str, object] = {"operation_type": "STATE_FORK", "bytes": 17}
    recorder.record(TraceStream.STATE_OPERATION, event)  # type: ignore[arg-type]
    event["bytes"] = 99
    assert recorder.state_events == ({"operation_type": "STATE_FORK", "bytes": 17},)


def test_sharing_analysis_does_not_infer_filesystem_page_cow() -> None:
    analysis = analyze_branch_state_sharing(
        (
            {
                "operation_type": "ENVIRONMENT_FORK",
                "logical_bytes": 100,
                "physical_bytes": 100,
            },
        ),
        (
            {
                "operation_type": "STATE_COW",
                "bytes": 25,
                "old_bytes": 20,
            },
        ),
    )
    assert analysis["copy_on_write"] == {
        "events": 1,
        "bytes_written": 25,
        "old_bytes_replaced": 20,
        "page_faults_observed": 0,
        "page_fault_measurement_available": False,
    }
