from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import pytest
from pydantic import ValidationError

from sloforge.continuum.adapters import (
    DeterministicRuntimeFixtureAdapter,
    KvStateClass,
    LiveSessionPhase,
    PhysicalKvLayoutSnapshot,
    RealRuntimeCapability,
    RuntimeFixtureError,
)
from sloforge.helix.characterization.trace import (
    BranchOperationType,
    ClockSource,
    MemoryLocation,
    StateOperationType,
    TimingMeasurementClass,
    WorkloadProvenance,
    verify_event,
)

ROOT = Path(__file__).parents[2]
TIMEOUT_S = 1.0
PREFIX_TOKENS = tuple(range(64))
BLOCK_SIZE_TOKENS = 8
BLOCK_BYTES = 4_096
PREFIX_BLOCKS = len(PREFIX_TOKENS) // BLOCK_SIZE_TOKENS


def _adapter(*, seed: int = 41, pool_blocks: int = 512) -> DeterministicRuntimeFixtureAdapter:
    return DeterministicRuntimeFixtureAdapter(
        seed=seed,
        block_size_tokens=BLOCK_SIZE_TOKENS,
        block_bytes=BLOCK_BYTES,
        kv_pool_block_capacity=pool_blocks,
        layer_count=4,
        max_fanout=32,
        max_trace_events=20_000,
    )


def _shared_fixture(
    fanout: int, *, seed: int = 41
) -> tuple[DeterministicRuntimeFixtureAdapter, str, tuple[str, ...]]:
    adapter = _adapter(seed=seed)
    adapter.start_session("root", token_ids=PREFIX_TOKENS, seed=seed, timeout_s=TIMEOUT_S)
    adapter.prefill_session("root", timeout_s=TIMEOUT_S)
    adapter.pause_at_safe_decode_boundary("root", timeout_s=TIMEOUT_S)
    root = adapter.create_shared_root_reference(
        "root", prefix_token_count=len(PREFIX_TOKENS), timeout_s=TIMEOUT_S
    )
    branch_ids = tuple(f"branch-{index:02d}" for index in range(fanout))
    for index, branch_id in enumerate(branch_ids):
        ref = adapter.fork_same_policy_session(
            root.root_reference_id,
            branch_id,
            divergent_token_id=10_000 + index,
            seed=seed + index + 1,
            timeout_s=TIMEOUT_S,
        )
        assert ref.phase is LiveSessionPhase.READY
        assert ref.parent_session_id == "root"
        assert ref.root_reference_id == root.root_reference_id
    return adapter, root.root_reference_id, branch_ids


@pytest.mark.parametrize("fanout", [1, 8, 32])
def test_shared_root_accounting_invariants_across_required_fanouts(fanout: int) -> None:
    adapter, root_id, branch_ids = _shared_fixture(fanout)
    root_logical = adapter.inspect_logical_state("root", timeout_s=TIMEOUT_S)
    layout = adapter.inspect_physical_kv_layout(branch_ids, timeout_s=TIMEOUT_S)

    assert adapter.capability_matrix.capabilities == frozenset(
        capability
        for capability in RealRuntimeCapability
        if capability is not RealRuntimeCapability.RUN_CONCURRENT_INDEPENDENT_SESSIONS
    )
    assert adapter.capability_matrix.scope.same_runtime_only is True
    assert adapter.capability_matrix.scope.same_policy_only is True
    assert adapter.capability_matrix.scope.cross_runtime_portable is False
    assert root_logical.token_ids == PREFIX_TOKENS
    assert len(layout.shared_prefix_block_ids) == PREFIX_BLOCKS
    assert layout.shared_prefix_bytes == PREFIX_BLOCKS * BLOCK_BYTES
    assert len(layout.private_suffix_block_ids) == fanout
    assert layout.private_suffix_bytes == fanout * BLOCK_BYTES
    assert layout.physical_assigned_bytes == (PREFIX_BLOCKS + fanout) * BLOCK_BYTES
    assert layout.root_reference_count == fanout + 1

    shared_ids = adapter.identify_shared_prefix_blocks(branch_ids, timeout_s=TIMEOUT_S)
    assert shared_ids == layout.shared_prefix_block_ids
    shared = [block for block in layout.blocks if block.runtime_block_id in shared_ids]
    assert all(block.state_class is KvStateClass.SHARED_PREFIX for block in shared)
    assert all(block.refcount == fanout + 1 for block in shared)
    assert all(set(block.branch_ids) == {"root", *branch_ids} for block in shared)

    private = [
        block
        for block in layout.blocks
        if block.runtime_block_id in layout.private_suffix_block_ids
    ]
    assert all(block.state_class is KvStateClass.PRIVATE_SUFFIX for block in private)
    assert all(block.refcount == 1 and len(block.branch_ids) == 1 for block in private)
    assert {block.branch_ids[0] for block in private} == set(branch_ids)
    assert all(block.logical_token_start == len(PREFIX_TOKENS) for block in private)

    branchpoint = adapter.capture_branchpoint(root_id, branch_ids, timeout_s=TIMEOUT_S)
    assert branchpoint.parent_session_id == "root"
    assert branchpoint.branch_session_ids == branch_ids
    assert branchpoint.root_reference.adapter_owned_reference_count == fanout + 1
    assert branchpoint.scope.same_process_or_adapter_scope.endswith("instance")

    for index, branch_id in enumerate(branch_ids):
        logical = adapter.inspect_logical_state(branch_id, timeout_s=TIMEOUT_S)
        assert logical.token_ids[:-1] == PREFIX_TOKENS
        assert logical.token_ids[-1] == 10_000 + index
    adapter.cleanup_runtime(timeout_s=TIMEOUT_S)


@pytest.mark.parametrize("fanout", [1, 8, 32])
def test_shared_prefix_is_not_physically_multiplied_like_independent_prefill(
    fanout: int,
) -> None:
    independent = _adapter(seed=73)
    independent_ids = tuple(f"independent-{index:02d}" for index in range(fanout))
    for index, session_id in enumerate(independent_ids):
        independent.start_session(
            session_id,
            token_ids=PREFIX_TOKENS,
            seed=73 + index,
            timeout_s=TIMEOUT_S,
        )
        independent.prefill_session(session_id, timeout_s=TIMEOUT_S)
    independent_layout = independent.inspect_physical_kv_layout(
        independent_ids, timeout_s=TIMEOUT_S
    )
    assert independent_layout.shared_prefix_bytes == 0
    assert independent_layout.private_suffix_bytes == 0
    assert independent_layout.physical_assigned_bytes == fanout * PREFIX_BLOCKS * BLOCK_BYTES
    assert all(block.refcount == 1 for block in independent_layout.blocks)

    shared, _, branch_ids = _shared_fixture(fanout, seed=73)
    shared_layout = shared.inspect_physical_kv_layout(branch_ids, timeout_s=TIMEOUT_S)
    assert shared_layout.shared_prefix_bytes == PREFIX_BLOCKS * BLOCK_BYTES
    assert shared_layout.physical_assigned_bytes == (PREFIX_BLOCKS + fanout) * BLOCK_BYTES
    assert (
        shared_layout.shared_prefix_bytes < independent_layout.physical_assigned_bytes
        or fanout == 1
    )

    independent.cleanup_runtime(timeout_s=TIMEOUT_S)
    shared.cleanup_runtime(timeout_s=TIMEOUT_S)


def test_concurrent_divergence_allocates_only_branch_private_blocks() -> None:
    fanout = 8
    adapter, _, branch_ids = _shared_fixture(fanout, seed=113)
    before_states = {
        branch_id: adapter.inspect_logical_state(branch_id, timeout_s=TIMEOUT_S)
        for branch_id in branch_ids
    }
    for branch_id in branch_ids:
        adapter.resume_branch(branch_id, timeout_s=TIMEOUT_S)

    result = adapter.run_concurrent_branches(
        branch_ids,
        maximum_new_tokens=17,
        seed=149,
        timeout_s=TIMEOUT_S,
    )
    assert result.total_output_tokens == fanout * 17
    assert result.completed_branch_ids == branch_ids
    assert all(
        result.request_admitted_latency_ns[branch_id]
        < result.first_decode_token_started_latency_ns[branch_id]
        < result.branch_ready_latency_ns[branch_id]
        < result.first_token_latency_ns[branch_id]
        for branch_id in branch_ids
    )
    assert len({tokens for tokens in result.branch_output_token_ids.values()}) == fanout
    assert [snapshot.private_suffix_bytes for snapshot in result.allocation_snapshots] == sorted(
        snapshot.private_suffix_bytes for snapshot in result.allocation_snapshots
    )
    assert all(
        snapshot.shared_prefix_bytes == PREFIX_BLOCKS * BLOCK_BYTES
        for snapshot in result.allocation_snapshots
    )

    final = result.allocation_snapshots[-1]
    private_blocks_per_branch = 3  # divergent token + 17 decode tokens at 8 tokens/block
    assert len(final.private_suffix_block_ids) == fanout * private_blocks_per_branch
    assert final.private_suffix_bytes == fanout * private_blocks_per_branch * BLOCK_BYTES
    private_blocks = {
        block.runtime_block_id: block
        for block in final.blocks
        if block.state_class is KvStateClass.PRIVATE_SUFFIX
    }
    for branch_id in branch_ids:
        logical = adapter.inspect_logical_state(branch_id, timeout_s=TIMEOUT_S)
        assert (
            logical.token_ids[: len(before_states[branch_id].token_ids)]
            == before_states[branch_id].token_ids
        )
        assert logical.token_ids[-17:] == result.branch_output_token_ids[branch_id]
        owned = [block for block in private_blocks.values() if block.branch_ids == (branch_id,)]
        assert len(owned) == private_blocks_per_branch
    adapter.cleanup_runtime(timeout_s=TIMEOUT_S)


def test_branch_deletion_preserves_siblings_and_final_release_frees_root() -> None:
    adapter, root_id, branch_ids = _shared_fixture(8)
    branch_private_ids = {
        branch_id: {
            block.runtime_block_id
            for block in adapter.inspect_physical_kv_layout(
                (branch_id,), timeout_s=TIMEOUT_S
            ).blocks
            if block.state_class is KvStateClass.PRIVATE_SUFFIX
        }
        for branch_id in branch_ids
    }
    survivor_before = adapter.inspect_logical_state(branch_ids[-1], timeout_s=TIMEOUT_S)

    adapter.destroy_session("root", timeout_s=TIMEOUT_S)
    after_source = adapter.inspect_physical_kv_layout(branch_ids, timeout_s=TIMEOUT_S)
    assert after_source.root_reference_count == len(branch_ids)
    assert all(
        block.refcount == len(branch_ids)
        for block in after_source.blocks
        if block.state_class is KvStateClass.SHARED_PREFIX
    )

    adapter.destroy_branch(branch_ids[0], timeout_s=TIMEOUT_S)
    trace_count = len(adapter.emit_state_operation_trace())
    adapter.destroy_branch(branch_ids[0], timeout_s=TIMEOUT_S)
    assert len(adapter.emit_state_operation_trace()) == trace_count
    after_one = adapter.inspect_physical_kv_layout(branch_ids[1:], timeout_s=TIMEOUT_S)
    assert after_one.root_reference_count == len(branch_ids) - 1
    assert branch_private_ids[branch_ids[0]].isdisjoint(
        block.runtime_block_id for block in after_one.blocks
    )
    assert adapter.inspect_logical_state(branch_ids[-1], timeout_s=TIMEOUT_S) == survivor_before

    for branch_id in branch_ids[1:-1]:
        adapter.destroy_branch(branch_id, timeout_s=TIMEOUT_S)
    final_sibling = adapter.inspect_physical_kv_layout((branch_ids[-1],), timeout_s=TIMEOUT_S)
    assert final_sibling.root_reference_count == 1
    assert all(
        block.refcount == 1
        for block in final_sibling.blocks
        if block.state_class is KvStateClass.SHARED_PREFIX
    )
    adapter.destroy_branch(branch_ids[-1], timeout_s=TIMEOUT_S)

    memory = adapter.inspect_gpu_memory_state(timeout_s=TIMEOUT_S)
    empty = adapter.inspect_physical_kv_layout((), timeout_s=TIMEOUT_S)
    assert memory.kv_assigned_bytes == 0
    assert memory.kv_unassigned_bytes == memory.kv_pool_reserved_bytes
    assert empty.blocks == ()
    assert empty.root_reference_count == 0
    with pytest.raises(RuntimeFixtureError, match="unknown live shared root"):
        adapter.capture_branchpoint(root_id, (branch_ids[-1],), timeout_s=TIMEOUT_S)
    adapter.cleanup_runtime(timeout_s=TIMEOUT_S)
    adapter.cleanup_runtime(timeout_s=TIMEOUT_S)


def test_cleanup_is_idempotent_with_live_state_and_runtime_fails_closed_afterward() -> None:
    adapter, _, _ = _shared_fixture(8)
    adapter.cleanup_runtime(timeout_s=TIMEOUT_S)
    trace_lengths = (
        len(adapter.emit_branch_workload_trace()),
        len(adapter.emit_state_operation_trace()),
    )
    adapter.cleanup_runtime(timeout_s=TIMEOUT_S)
    assert trace_lengths == (
        len(adapter.emit_branch_workload_trace()),
        len(adapter.emit_state_operation_trace()),
    )
    with pytest.raises(RuntimeFixtureError, match="already been cleaned up"):
        adapter.start_session("late", token_ids=(1,), seed=1, timeout_s=TIMEOUT_S)


def test_fixture_gpu_shaped_traces_are_sealed_complete_and_explicitly_simulated() -> None:
    adapter, root_id, branch_ids = _shared_fixture(2)
    adapter.capture_branchpoint(root_id, branch_ids, timeout_s=TIMEOUT_S)
    for branch_id in branch_ids:
        adapter.resume_branch(branch_id, timeout_s=TIMEOUT_S)
    adapter.run_concurrent_branches(branch_ids, maximum_new_tokens=8, seed=7, timeout_s=TIMEOUT_S)
    adapter.destroy_session("root", timeout_s=TIMEOUT_S)
    for branch_id in branch_ids:
        adapter.destroy_branch(branch_id, timeout_s=TIMEOUT_S)

    branch_events = adapter.emit_branch_workload_trace()
    state_events = adapter.emit_state_operation_trace()
    assert tuple(event.event_sequence for event in branch_events) == tuple(
        range(len(branch_events))
    )
    assert tuple(event.event_sequence for event in state_events) == tuple(range(len(state_events)))
    for event in (*branch_events, *state_events):
        verify_event(event)
        assert event.provenance is WorkloadProvenance.SIMULATED_HARDWARE
        assert event.timing_measurement_class is TimingMeasurementClass.SIMULATED_HARDWARE
        assert event.clock_source is ClockSource.SYNTHETIC
        assert event.device == "cuda:0"
        assert event.attributes["fixture_only"] is True
        assert event.attributes["simulated_gpu_state"] is True
        assert event.attributes["real_gpu_measurement"] is False
    assert {event.operation_type for event in branch_events} >= {
        BranchOperationType.STATE_PUBLISH,
        BranchOperationType.BRANCH_FORK,
        BranchOperationType.BRANCH_DIVERGENCE,
        BranchOperationType.BRANCH_POINT,
        BranchOperationType.STATE_COW,
        BranchOperationType.ROLLOUT,
        BranchOperationType.BRANCH_PRUNE,
    }
    assert {event.operation_type for event in state_events} >= {
        StateOperationType.STATE_ALLOC,
        StateOperationType.STATE_PUBLISH,
        StateOperationType.STATE_FORK,
        StateOperationType.STATE_COW,
        StateOperationType.STATE_APPEND,
        StateOperationType.STATE_RECLAIM,
        StateOperationType.STATE_FREE,
    }
    for event in state_events:
        assert event.attributes["runtime_block_id"]
        assert event.attributes["runtime"]
        assert event.attributes["runtime_version"]
        assert event.attributes["state_class"]
        assert isinstance(event.attributes["refcount"], int)
        assert event.source_location in {MemoryLocation.UNKNOWN, MemoryLocation.GPU_HBM}
        assert event.destination_location in {MemoryLocation.UNKNOWN, MemoryLocation.GPU_HBM}

    branch_schema = json.loads(
        (ROOT / "schemas/branchfabric/branch-workload-trace-v1.schema.json").read_text()
    )
    state_schema = json.loads(
        (ROOT / "schemas/branchfabric/state-operation-trace-v1.schema.json").read_text()
    )
    for event in branch_events:
        jsonschema.validate(event.model_dump(mode="json"), branch_schema)
    for event in state_events:
        jsonschema.validate(event.model_dump(mode="json"), state_schema)
    adapter.cleanup_runtime(timeout_s=TIMEOUT_S)


def test_fixture_runs_are_deterministic_and_capacity_failures_are_atomic() -> None:
    def scenario() -> tuple[tuple[dict[str, object], ...], tuple[dict[str, object], ...]]:
        adapter, _, branch_ids = _shared_fixture(2, seed=191)
        for branch_id in branch_ids:
            adapter.resume_branch(branch_id, timeout_s=TIMEOUT_S)
        adapter.run_concurrent_branches(
            branch_ids, maximum_new_tokens=3, seed=193, timeout_s=TIMEOUT_S
        )
        adapter.cleanup_runtime(timeout_s=TIMEOUT_S)
        return (
            tuple(event.model_dump(mode="json") for event in adapter.emit_branch_workload_trace()),
            tuple(event.model_dump(mode="json") for event in adapter.emit_state_operation_trace()),
        )

    assert scenario() == scenario()

    bounded = _adapter(seed=211, pool_blocks=PREFIX_BLOCKS - 1)
    bounded.start_session("root", token_ids=PREFIX_TOKENS, seed=211, timeout_s=TIMEOUT_S)
    with pytest.raises(RuntimeFixtureError, match="complete operation"):
        bounded.prefill_session("root", timeout_s=TIMEOUT_S)
    assert bounded.inspect_physical_kv_layout(("root",), timeout_s=TIMEOUT_S).blocks == ()
    bounded.cleanup_runtime(timeout_s=TIMEOUT_S)


def test_physical_snapshot_rejects_corrupt_accounting_and_timeouts_are_bounded() -> None:
    adapter, _, branch_ids = _shared_fixture(1)
    snapshot = adapter.inspect_physical_kv_layout(branch_ids, timeout_s=TIMEOUT_S)
    payload = snapshot.model_dump(mode="python")
    payload["physical_assigned_bytes"] = snapshot.physical_assigned_bytes + 1
    with pytest.raises(ValidationError, match="assigned-byte accounting mismatch"):
        PhysicalKvLayoutSnapshot.model_validate(payload)
    with pytest.raises(ValueError, match="timeout_s"):
        adapter.inspect_gpu_memory_state(timeout_s=float("inf"))
    with pytest.raises(ValueError, match="timeout_s"):
        adapter.destroy_branch(branch_ids[0], timeout_s=0)
    adapter.cleanup_runtime(timeout_s=TIMEOUT_S)
