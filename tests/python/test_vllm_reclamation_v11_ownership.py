from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import pytest
from test_vllm_live_adapter import fake_engine, identity

from sloforge.continuum.adapters.real_runtime import GpuMemoryState, LiveSessionPhase
from sloforge.continuum.adapters.vllm_live import (
    VLLM_LIVE_RUNTIME_VERSION,
    VllmLiveStateAdapter,
)
from sloforge.continuum.adapters.vllm_reclamation_v11_ownership import (
    V11OwnershipProofError,
    capture_source_ownership_v11,
    release_source_and_prove_v11,
)
from sloforge.continuum.adapters.vllm_reclamation_v11_sync import (
    Vllm0230EngineStepBinding,
)


class _Queue:
    def __init__(self) -> None:
        self.rows: list[object] = []

    def add_request(self, request: object) -> None:
        self.rows.append(request)

    def remove_requests(self, requests: list[object]) -> None:
        for request in requests:
            if request in self.rows:
                self.rows.remove(request)


def _production_adapter_fixture(
    *, stale_native_tables_on_free: bool = False
) -> tuple[VllmLiveStateAdapter, dict[int, int]]:
    engine = fake_engine(prefix_cache=True)
    client = engine.llm_engine.engine_core
    core = client.engine_core
    scheduler = core.scheduler
    manager = scheduler.kv_cache_manager
    manager.coordinator = SimpleNamespace(
        single_type_managers=(SimpleNamespace(req_to_blocks=manager.tables),)
    )
    scheduler.scheduler_config = SimpleNamespace(async_scheduling=False)
    scheduler.waiting = _Queue()
    scheduler.skipped_waiting = _Queue()
    scheduler.running = []
    scheduler.add_request = lambda *_args, **_kwargs: None
    scheduler._enqueue_waiting_request = scheduler.waiting.add_request
    scheduler.schedule = lambda: object()
    scheduler.update_from_output = lambda *_args, **_kwargs: None
    scheduler.finish_requests = lambda *_args, **_kwargs: None
    manager.cache_blocks = lambda *_args, **_kwargs: None
    manager.take_new_block_ids = lambda: []

    def free(request: object) -> None:
        runtime_id = cast(Any, request).request_id
        groups = (
            manager.tables[runtime_id]
            if stale_native_tables_on_free
            else manager.tables.pop(runtime_id)
        )
        for group in groups:
            for block in group:
                block.ref_cnt -= 1
                assert block.ref_cnt >= 0

    manager.free = free

    def abort_request(request_ids: list[str]) -> None:
        for request_id in request_ids:
            request = scheduler.requests.pop(request_id)
            manager.free(request)

    engine.llm_engine.abort_request = abort_request
    adapter = VllmLiveStateAdapter(
        engine,
        identity=identity(),
        runtime_version=VLLM_LIVE_RUNTIME_VERSION,
    )
    adapter.start_session("root", token_ids=tuple(range(16)), seed=41, timeout_s=1.0)
    for branch_id, divergent in (("branch.0", 91), ("branch.1", 92)):
        adapter.start_session(
            branch_id,
            token_ids=(*range(16), divergent),
            seed=41,
            timeout_s=1.0,
        )

    root_session = adapter._sessions["root"]
    root_session.phase = LiveSessionPhase.PREFILLED
    root_session_id = "vllm-root:root:fixture"
    shared_block_id = "vllm:kv-group:0:block:1"
    adapter._roots[root_session_id] = SimpleNamespace(
        root_reference_id=root_session_id,
        source_session_id="root",
        block_ids=(shared_block_id,),
        prefix_token_count=16,
    )

    blocks = manager.block_pool.blocks
    requests = {
        "branch.0": SimpleNamespace(request_id="branch.0"),
        "branch.1": SimpleNamespace(request_id="branch.1"),
    }
    scheduler.requests.update(requests)
    manager.tables["branch.0"] = ((blocks[1], blocks[2]),)
    blocks[1].ref_cnt += 1
    blocks[2].ref_cnt += 1
    manager.allocate_slots(requests["branch.0"])
    manager.tables["branch.1"] = ((blocks[1], blocks[3]),)
    blocks[1].ref_cnt += 1
    blocks[3].ref_cnt += 1
    manager.allocate_slots(
        requests["branch.1"],
        num_new_computed_tokens=16,
        new_computed_blocks=SimpleNamespace(blocks=((blocks[1],),)),
    )
    blocks[1].block_hash = "shared-root-hash"
    for branch_id in requests:
        branch = adapter._sessions[branch_id]
        branch.phase = LiveSessionPhase.PAUSED
        branch.runtime_request_id = branch_id
        branch.parent_session_id = "root"
        branch.root_reference_id = root_session_id
        branch.branch_id = branch_id
        branch.submitted = True
    return adapter, dict(adapter._allocator_epoch_source.epoch_by_block())


def test_post_free_proof_covers_every_block_and_old_sessions_and_hbm() -> None:
    adapter, epochs = _production_adapter_fixture()
    binding = Vllm0230EngineStepBinding(adapter._view, acquire_timeout_seconds=1.0)
    try:
        with binding.critical_section(
            "EXPORT_CAPTURE", gate_id="ownership-release-seed-41"
        ) as gate:
            snapshot = capture_source_ownership_v11(
                adapter,
                root_session_id="root",
                branch_session_ids=("branch.0", "branch.1"),
                expected_allocation_epochs=epochs,
                expected_source_block_count=3,
                admission_gate=gate,
                timeout_s=1.0,
            )
            proof = release_source_and_prove_v11(
                adapter,
                snapshot,
                admission_gate=gate,
                timeout_s=1.0,
            )
        proof.require_passed()
        assert proof.source_block_count == 3
        assert proof.exact_source_coverage
        assert proof.all_old_sessions_destroyed
        assert proof.root_reference_released
        assert proof.old_session_layout_inaccessible
        assert proof.no_live_branch_reference
        assert proof.native_manager_request_tables_absent
        assert proof.all_allocator_epochs_tombstoned
        assert proof.allocator_hbm_agreement
        assert proof.kv_assigned_bytes_before == proof.source_physical_bytes
        assert proof.kv_assigned_bytes_after == 0
        assert (
            proof.kv_unassigned_bytes_after - proof.kv_unassigned_bytes_before
            == proof.source_physical_bytes
        )
        assert all(record.passed for record in proof.records)
        assert {record.pre_release_refcount for record in proof.records} == {1, 2}
        assert all(record.post_release_refcount == 0 for record in proof.records)
        assert all(record.post_release_owner == () for record in proof.records)
    finally:
        binding.close()
        adapter.cleanup_runtime(timeout_s=1.0)


def test_capture_rejects_missing_source_block_and_foreign_live_owner() -> None:
    adapter, epochs = _production_adapter_fixture()
    binding = Vllm0230EngineStepBinding(adapter._view, acquire_timeout_seconds=1.0)
    try:
        with binding.critical_section(
            "EXPORT_CAPTURE", gate_id="ownership-coverage-seed-41"
        ) as gate:
            incomplete = dict(epochs)
            incomplete.pop(max(incomplete))
            with pytest.raises(V11OwnershipProofError, match="exact expected source block count"):
                capture_source_ownership_v11(
                    adapter,
                    root_session_id="root",
                    branch_session_ids=("branch.0", "branch.1"),
                    expected_allocation_epochs=incomplete,
                    expected_source_block_count=2,
                    admission_gate=gate,
                    timeout_s=1.0,
                )

            scheduler = adapter._view.scheduler
            manager = adapter._view.manager
            foreign = SimpleNamespace(request_id="foreign")
            scheduler.requests[foreign.request_id] = foreign
            manager.tables[foreign.request_id] = ((manager.block_pool.blocks[1],),)
            manager.block_pool.blocks[1].ref_cnt += 1
            with pytest.raises(V11OwnershipProofError, match="owner/refcount/epoch proof"):
                capture_source_ownership_v11(
                    adapter,
                    root_session_id="root",
                    branch_session_ids=("branch.0", "branch.1"),
                    expected_allocation_epochs=epochs,
                    expected_source_block_count=3,
                    admission_gate=gate,
                    timeout_s=1.0,
                )
            manager.block_pool.blocks[1].ref_cnt -= 1
            manager.tables.pop(foreign.request_id)
            scheduler.requests.pop(foreign.request_id)
    finally:
        binding.close()
        adapter.cleanup_runtime(timeout_s=1.0)


def test_post_free_proof_rejects_allocator_hbm_disagreement() -> None:
    adapter, epochs = _production_adapter_fixture()
    binding = Vllm0230EngineStepBinding(adapter._view, acquire_timeout_seconds=1.0)
    try:
        with binding.critical_section("EXPORT_CAPTURE", gate_id="ownership-hbm-seed-41") as gate:
            snapshot = capture_source_ownership_v11(
                adapter,
                root_session_id="root",
                branch_session_ids=("branch.0", "branch.1"),
                expected_allocation_epochs=epochs,
                expected_source_block_count=3,
                admission_gate=gate,
                timeout_s=1.0,
            )
            original = adapter.inspect_gpu_memory_state

            def stale_hbm(*, timeout_s: float) -> GpuMemoryState:
                observed = original(timeout_s=timeout_s)
                page_bytes = adapter._view.page_size_bytes
                return observed.model_copy(
                    update={
                        "kv_assigned_bytes": page_bytes,
                        "kv_unassigned_bytes": observed.kv_pool_reserved_bytes - page_bytes,
                    }
                )

            adapter.inspect_gpu_memory_state = stale_hbm  # type: ignore[method-assign]
            with pytest.raises(V11OwnershipProofError, match="post-free ownership gate"):
                release_source_and_prove_v11(
                    adapter,
                    snapshot,
                    admission_gate=gate,
                    timeout_s=1.0,
                )
    finally:
        binding.close()
        adapter.cleanup_runtime(timeout_s=1.0)


def test_post_free_proof_rejects_stale_native_request_block_tables() -> None:
    adapter, epochs = _production_adapter_fixture(stale_native_tables_on_free=True)
    binding = Vllm0230EngineStepBinding(adapter._view, acquire_timeout_seconds=1.0)
    try:
        with binding.critical_section(
            "EXPORT_CAPTURE", gate_id="ownership-native-table-seed-41"
        ) as gate:
            snapshot = capture_source_ownership_v11(
                adapter,
                root_session_id="root",
                branch_session_ids=("branch.0", "branch.1"),
                expected_allocation_epochs=epochs,
                expected_source_block_count=3,
                admission_gate=gate,
                timeout_s=1.0,
            )
            manager = adapter._view.manager
            with pytest.raises(V11OwnershipProofError, match="post-free ownership gate"):
                release_source_and_prove_v11(
                    adapter,
                    snapshot,
                    admission_gate=gate,
                    timeout_s=1.0,
                )
            assert set(manager.tables) == {"branch.0", "branch.1"}
    finally:
        # Clear the deliberately corrupt test state before adapter teardown.
        adapter._view.manager.tables.clear()
        binding.close()
        adapter.cleanup_runtime(timeout_s=1.0)
