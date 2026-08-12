from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from sloforge.helix.characterization.trace.models import (
    BranchOperationType,
    BranchWorkloadEventV1,
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
from sloforge.helix.characterization.workload_analysis import (
    ArtifactDescriptor,
    BranchOutcome,
    DivergenceCoverage,
    TrajectoryArtifactInput,
    WaterfallStage,
    analyze_workload,
    canonical_event_reference,
)


def _branch_event(
    sequence: int,
    operation: BranchOperationType,
    timestamp_ns: int,
    duration_ns: int,
    *,
    branch_id: str | None = None,
    parent_branch_id: str | None = "root",
    logical_bytes: int = 0,
    shared_root: bool = False,
    private_suffix: bool = False,
    state_segment: StateSegment = StateSegment.UNKNOWN,
) -> BranchWorkloadEventV1:
    return BranchWorkloadEventV1(
        collection_level=TraceLevel.FULL,
        provenance=WorkloadProvenance.SYNTHETIC,
        timing_measurement_class=TimingMeasurementClass.HARDWARE_BACKED_REAL,
        trace_id="trace-41",
        session_id="session-41",
        branch_group_id="group-41",
        branch_id=branch_id,
        parent_branch_id=parent_branch_id,
        policy_epoch="policy-0",
        environment_id="environment-0",
        transaction_id="transaction-0",
        host="host-a",
        process_id=100,
        rank=0,
        device="cpu",
        monotonic_timestamp_ns=1_000_000 + timestamp_ns,
        normalized_timestamp_ns=timestamp_ns,
        duration_ns=duration_ns,
        clock_source=ClockSource.PERF_COUNTER,
        alignment_confidence=1.0,
        operation_type=operation,
        logical_state_id="root-state" if branch_id is None else f"state-{branch_id}",
        physical_state_id=None,
        state_segment=state_segment,
        logical_bytes=logical_bytes,
        physical_bytes=logical_bytes,
        shared_root=shared_root,
        private_suffix=private_suffix,
        queue_delay_ns=0,
        execution_latency_ns=duration_ns,
        cpu_time_ns=max(1, duration_ns // 2),
        transport_type=TransportType.NONE,
        event_sequence=sequence,
        content_hash="0" * 64,
    )


def _state_event(
    sequence: int,
    operation: StateOperationType,
    timestamp_ns: int,
    duration_ns: int,
    *,
    branch_id: str,
    logical_state_id: str,
    segment: StateSegment,
    bytes_count: int,
    concurrency: int = 3,
    queue_delay_ns: int = 2,
    fanout: int = 1,
    destination_location: MemoryLocation = MemoryLocation.HOST_DRAM,
) -> StateOperationEventV1:
    return StateOperationEventV1(
        collection_level=TraceLevel.FULL,
        provenance=WorkloadProvenance.SYNTHETIC,
        timing_measurement_class=TimingMeasurementClass.HARDWARE_BACKED_REAL,
        trace_id="trace-41",
        session_id="session-41",
        branch_group_id="group-41",
        logical_state_id=logical_state_id,
        branch_id=branch_id,
        tenant_id="tenant-a",
        security_domain="domain-a",
        host="host-a",
        process_id=100,
        rank=0,
        device="cpu",
        monotonic_timestamp_ns=1_000_000 + timestamp_ns,
        normalized_timestamp_ns=timestamp_ns,
        duration_ns=duration_ns,
        clock_source=ClockSource.PERF_COUNTER,
        alignment_confidence=1.0,
        operation_type=operation,
        state_segment=segment,
        source_physical_representation="layout-a",
        destination_physical_representation="layout-b",
        bytes=bytes_count,
        alignment_bytes=256,
        page_size_bytes=4096,
        chunk_size_bytes=1024,
        fanout=fanout,
        dependency_event_ids=(),
        concurrency=concurrency,
        queue_delay_ns=queue_delay_ns,
        operation_latency_ns=duration_ns,
        cpu_time_ns=max(1, duration_ns // 2),
        gpu_time_ns=0,
        transfer_time_ns=(
            duration_ns
            if operation
            in {
                StateOperationType.STATE_SEND,
                StateOperationType.STATE_RECEIVE,
                StateOperationType.STATE_MULTICAST,
            }
            else 0
        ),
        result=OperationResult.SUCCESS,
        state_epoch=1,
        source_location=MemoryLocation.HOST_DRAM,
        destination_location=destination_location,
        transport_type=(
            TransportType.TCP
            if operation
            in {
                StateOperationType.STATE_SEND,
                StateOperationType.STATE_RECEIVE,
                StateOperationType.STATE_MULTICAST,
            }
            else TransportType.NONE
        ),
        event_sequence=sequence,
        content_hash="0" * 64,
    )


def _events() -> tuple[list[BranchWorkloadEventV1], list[StateOperationEventV1]]:
    branch = [
        _branch_event(0, BranchOperationType.BRANCH_POINT, 10, 10, parent_branch_id=None),
        *[
            _branch_event(
                index,
                BranchOperationType.BRANCH_FORK,
                100,
                50,
                branch_id=branch_id,
                logical_bytes=100,
                shared_root=True,
            )
            for index, branch_id in enumerate(("b1", "b2", "b3"), start=1)
        ],
        *[
            _branch_event(
                index,
                BranchOperationType.BRANCH_READY,
                160,
                10,
                branch_id=branch_id,
            )
            for index, branch_id in enumerate(("b1", "b2", "b3"), start=4)
        ],
        _branch_event(
            7,
            BranchOperationType.BRANCH_DIVERGENCE,
            200,
            5,
            branch_id="b1",
            private_suffix=True,
        ),
        _branch_event(8, BranchOperationType.ROLLOUT, 250, 100, branch_id="b1"),
        _branch_event(9, BranchOperationType.ROLLOUT, 260, 100, branch_id="b2"),
        _branch_event(10, BranchOperationType.ROLLOUT, 270, 100, branch_id="b3"),
        _branch_event(11, BranchOperationType.CHECKPOINT, 350, 20, branch_id="b1"),
        _branch_event(12, BranchOperationType.REWARD, 380, 10, branch_id="b1"),
        _branch_event(13, BranchOperationType.BRANCH_PRUNE, 400, 10, branch_id="b2"),
        _branch_event(14, BranchOperationType.BRANCH_ABORT, 410, 10, branch_id="b3"),
        _branch_event(15, BranchOperationType.TRAIN, 430, 20),
        _branch_event(16, BranchOperationType.EVALUATE, 460, 10),
        _branch_event(17, BranchOperationType.PROMOTE, 480, 10),
        _branch_event(18, BranchOperationType.BRANCH_COMPLETE, 500, 10, branch_id="b1"),
    ]
    state = [
        *[
            _state_event(
                index,
                StateOperationType.STATE_FORK,
                100,
                50,
                branch_id=branch_id,
                logical_state_id=f"state-{branch_id}",
                segment=StateSegment.KV,
                bytes_count=100,
            )
            for index, branch_id in enumerate(("b1", "b2", "b3"))
        ],
        _state_event(
            3,
            StateOperationType.STATE_COW,
            220,
            5,
            branch_id="b2",
            logical_state_id="state-b2",
            segment=StateSegment.KV,
            bytes_count=20,
            queue_delay_ns=7,
        ),
        _state_event(
            4,
            StateOperationType.STATE_SNAPSHOT,
            330,
            20,
            branch_id="b1",
            logical_state_id="environment-b1",
            segment=StateSegment.FILESYSTEM,
            bytes_count=40,
            destination_location=MemoryLocation.LOCAL_NVME,
        ),
        _state_event(
            5,
            StateOperationType.STATE_SEND,
            370,
            30,
            branch_id="b1",
            logical_state_id="state-b1",
            segment=StateSegment.KV,
            bytes_count=10,
            fanout=2,
        ),
    ]
    return branch, state


def _artifact(reference: str) -> ArtifactDescriptor:
    return ArtifactDescriptor(
        source_experiment="workload-seed-41",
        artifact_reference=reference,
        artifact_sha256=("a" if reference.startswith("branch") else "b") * 64,
        seed=41,
        repetition=0,
    )


def _trajectory(
    tmp_path: Path,
    branch_id: str,
    tokens: tuple[str, ...],
    actions: tuple[str, ...],
) -> TrajectoryArtifactInput:
    path = tmp_path / f"{branch_id}.json"
    payload = json.dumps(
        {
            "branch_id": branch_id,
            "branch_group_id": "group-41",
            "tokens": [
                {"token_index": index, "token": token} for index, token in enumerate(tokens)
            ],
            "actions": [
                {"action_index": index, "action": action} for index, action in enumerate(actions)
            ],
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    path.write_bytes(payload)
    return TrajectoryArtifactInput(
        source_experiment="trajectory-seed-41",
        artifact_reference=f"trajectories/{branch_id}.json",
        artifact_sha256=hashlib.sha256(payload).hexdigest(),
        seed=41,
        repetition=0,
        file_path=str(path),
        provenance=WorkloadProvenance.SYNTHETIC,
    )


def _span_sidecar(
    branch: list[BranchWorkloadEventV1], state: list[StateOperationEventV1]
) -> dict[str, str]:
    branch_forks = [
        event for event in branch if event.operation_type is BranchOperationType.BRANCH_FORK
    ]
    state_forks = [
        event for event in state if event.operation_type is StateOperationType.STATE_FORK
    ]
    return {
        **{canonical_event_reference(event): "combined-fork-span" for event in branch_forks},
        **{canonical_event_reference(event): "combined-fork-span" for event in state_forks},
    }


def test_workload_analysis_derives_dag_divergence_sharing_and_waterfall(
    tmp_path: Path,
) -> None:
    branch, state = _events()
    trajectories = (
        _trajectory(tmp_path, "b1", ("shared", "alpha"), ("edit",)),
        _trajectory(tmp_path, "b2", ("shared", "beta"), ("test",)),
        _trajectory(tmp_path, "b3", ("shared", "alpha"), ("edit",)),
    )

    result = analyze_workload(
        branch,
        state,
        branch_artifact=_artifact("branch.jsonl"),
        state_artifact=_artifact("state.jsonl"),
        trajectory_artifacts=trajectories,
        timing_span_ids=_span_sidecar(branch, state),
    )

    assert result.evidence.branch_trace.event_count == len(branch)
    assert result.evidence.state_trace.event_count == len(state)
    assert result.evidence.branch_trace.workload_provenance == (WorkloadProvenance.SYNTHETIC,)
    assert len(result.evidence.trajectories) == 3
    group = result.branch_dag.groups[0]
    assert group.fanout == 3
    assert (group.pruned_count, group.aborted_count, group.succeeded_count) == (1, 1, 1)
    assert group.shared_root_creation_timestamp_ns == 150
    assert group.shared_root_last_reference_timestamp_ns == 510
    assert group.shared_root_lifetime_ns == 360
    assert group.shared_root_logical_bytes == 100
    assert group.peak_refcount == 3
    assert [item.refcount for item in group.refcount_timeline] == [3, 2, 1, 0]
    by_branch = {item.branch_id: item for item in result.branch_dag.nodes}
    assert by_branch["b1"].outcome is BranchOutcome.SUCCEEDED
    assert by_branch["b1"].fork_to_first_private_ns == 50
    assert by_branch["b2"].fork_to_first_private_ns == 70
    assert by_branch["b3"].fork_to_first_private_ns is None
    assert by_branch["b1"].lifetime_ns == 410
    divergence = result.divergence[0]
    assert divergence.coverage is DivergenceCoverage.COMPLETE
    assert divergence.first_divergent_token_index == 1
    assert divergence.first_divergent_action_index == 0
    assert divergence.token_sequences_all_equal is False
    assert result.state_composition.model_state_bytes == 300
    assert result.state_composition.environment_state_bytes == 40
    queue = next(
        item
        for item in result.operation_queues
        if item.operation_type is StateOperationType.STATE_COW
    )
    assert queue.concurrency.p99 == 3
    assert queue.queue_delay_ns.p99 == 7
    fork_stage = next(item for item in result.waterfall.stages if item.stage is WaterfallStage.FORK)
    assert fork_stage.event_count == 6
    assert fork_stage.unique_timing_span_count == 1
    assert fork_stage.timing_deduplicated_event_count == 5
    assert fork_stage.inclusive_span_duration_sum_ns == 50
    assert fork_stage.inclusive_wall_union_ns == 50
    assert fork_stage.state_operation_bytes == 300
    assert fork_stage.branch_payload_bytes == 300
    rollout_stage = next(
        item for item in result.waterfall.stages if item.stage is WaterfallStage.ROLLOUT
    )
    assert rollout_stage.inclusive_span_duration_sum_ns == 300
    assert rollout_stage.inclusive_wall_union_ns == 120
    assert rollout_stage.exclusive_critical_path_ns is None
    production = next(
        item for item in result.waterfall.stages if item.stage is WaterfallStage.PRODUCTION
    )
    assert production.observed is False
    assert production.inclusive_wall_union_ns is None
    assert result.waterfall.exclusive_critical_path_ns is None
    assert any("not assigned to migration" in item for item in result.unknowns)


def test_divergence_is_unknown_without_exact_trajectory_artifacts() -> None:
    branch, state = _events()

    result = analyze_workload(
        branch,
        state,
        branch_artifact=_artifact("branch.jsonl"),
        state_artifact=_artifact("state.jsonl"),
    )

    comparison = result.divergence[0]
    assert comparison.coverage is DivergenceCoverage.UNAVAILABLE
    assert comparison.first_divergent_token_index is None
    assert comparison.first_divergent_action_index is None
    assert comparison.token_sequences_all_equal is None
    assert any("exact trajectories" in item for item in result.unknowns)
    assert any("timing_span_id" in item for item in result.unknowns)


def test_timing_sidecar_rejects_inconsistent_or_cross_stage_spans() -> None:
    branch, state = _events()
    inconsistent = _span_sidecar(branch, state)
    rollout = next(event for event in branch if event.operation_type is BranchOperationType.ROLLOUT)
    inconsistent[canonical_event_reference(rollout)] = "combined-fork-span"

    with pytest.raises(ValueError, match="cross waterfall stage"):
        analyze_workload(
            branch,
            state,
            branch_artifact=_artifact("branch.jsonl"),
            state_artifact=_artifact("state.jsonl"),
            timing_span_ids=inconsistent,
        )

    unknown_reference = {"branch:missing:999": "span"}
    with pytest.raises(ValueError, match="unknown events"):
        analyze_workload(
            branch,
            state,
            branch_artifact=_artifact("branch.jsonl"),
            state_artifact=_artifact("state.jsonl"),
            timing_span_ids=unknown_reference,
        )


def test_trajectory_sha_and_exact_indices_are_verified(tmp_path: Path) -> None:
    branch, state = _events()
    trajectory = _trajectory(tmp_path, "b1", ("token",), ("action",))
    bad_hash = trajectory.model_copy(update={"artifact_sha256": "f" * 64})
    with pytest.raises(ValueError, match="trajectory SHA-256 mismatch"):
        analyze_workload(
            branch,
            state,
            branch_artifact=_artifact("branch.jsonl"),
            state_artifact=_artifact("state.jsonl"),
            trajectory_artifacts=(bad_hash,),
        )

    path = Path(trajectory.file_path)
    document = json.loads(path.read_text())
    document["tokens"][0]["token_index"] = 9
    payload = json.dumps(document, sort_keys=True, separators=(",", ":")).encode()
    path.write_bytes(payload)
    bad_index = trajectory.model_copy(
        update={"artifact_sha256": hashlib.sha256(payload).hexdigest()}
    )
    with pytest.raises(ValueError, match="token indices"):
        analyze_workload(
            branch,
            state,
            branch_artifact=_artifact("branch.jsonl"),
            state_artifact=_artifact("state.jsonl"),
            trajectory_artifacts=(bad_index,),
        )


def test_branch_parent_cycles_are_rejected() -> None:
    branch, state = _events()
    cyclic = [
        event.model_copy(
            update={
                "parent_branch_id": (
                    "b2"
                    if event.branch_id == "b1"
                    else "b1"
                    if event.branch_id == "b2"
                    else event.parent_branch_id
                )
            }
        )
        for event in branch
    ]

    with pytest.raises(ValueError, match="cycle"):
        analyze_workload(
            cyclic,
            state,
            branch_artifact=_artifact("branch.jsonl"),
            state_artifact=_artifact("state.jsonl"),
        )
