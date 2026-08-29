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
from sloforge.continuum.adapters.vllm_reclamation import (
    CanonicalCapturePlan,
    RuntimeBranchCaptureInput,
    build_canonical_capture_plan,
)
from sloforge.continuum.adapters.vllm_reclamation_v11_ownership import (
    V11OwnershipProofError,
    capture_source_ownership_v11,
    create_source_capture_commit_v11,
    expected_owner_set,
    expected_refcount,
    prove_allocator_quiescence_v11,
    release_source_and_prove_v11,
    validate_source_capture_commit_v11,
)
from sloforge.continuum.adapters.vllm_reclamation_v11_sync import (
    V11EngineStepBindingError,
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
        "branch.0": SimpleNamespace(
            request_id="branch.0",
            all_token_ids=[*range(16), 91],
            num_computed_tokens=17,
        ),
        "branch.1": SimpleNamespace(
            request_id="branch.1",
            all_token_ids=[*range(16), 92],
            num_computed_tokens=17,
        ),
    }
    scheduler.requests.update(requests)
    scheduler.running = list(requests.values())
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


def _capture_plan(
    adapter: VllmLiveStateAdapter,
    epochs: dict[int, int],
    *,
    branch_session_ids: tuple[str, ...] = ("branch.0", "branch.1"),
) -> CanonicalCapturePlan:
    runtime_inputs: list[RuntimeBranchCaptureInput] = []
    for branch_id in branch_session_ids:
        session = adapter._sessions[branch_id]
        assert session.runtime_request_id is not None
        request = adapter._view.request_object(session.runtime_request_id)
        assert request is not None
        runtime_inputs.append(
            RuntimeBranchCaptureInput(
                logical_branch_id=branch_id,
                parent_logical_branch_id="root",
                token_ids=tuple(int(item) for item in request.all_token_ids),
                computed_tokens=int(request.num_computed_tokens),
                source_block_indices=tuple(
                    int(block.block_id)
                    for block in adapter._view.request_blocks(session.runtime_request_id)[0]
                ),
            )
        )
    return build_canonical_capture_plan(
        branches=tuple(runtime_inputs),
        block_size_tokens=adapter._view.block_size_tokens,
        logical_token_bytes=(adapter._view.page_size_bytes // adapter._view.block_size_tokens),
        physical_page_bytes=adapter._view.page_size_bytes,
        gpu_uuid="GPU-fixture",
        allocation_epoch_by_block=epochs,
    )


def _capture_commit(
    adapter: VllmLiveStateAdapter,
    epochs: dict[int, int],
    gate: object,
) -> tuple[CanonicalCapturePlan, object, object]:
    plan = _capture_plan(adapter, epochs)
    snapshot = capture_source_ownership_v11(
        adapter,
        root_session_id="root",
        branch_session_ids=("branch.0", "branch.1"),
        expected_allocation_epochs=epochs,
        expected_source_block_count=3,
        admission_gate=gate,  # type: ignore[arg-type]
        timeout_s=2.0,
    )
    commit = create_source_capture_commit_v11(
        adapter,
        plan,
        snapshot,
        block_size_tokens=adapter._view.block_size_tokens,
        admission_gate=gate,  # type: ignore[arg-type]
    )
    return plan, snapshot, commit


def test_source_capture_commit_accepts_stable_semantic_identity() -> None:
    adapter, epochs = _production_adapter_fixture()
    binding = Vllm0230EngineStepBinding(adapter._view, acquire_timeout_seconds=2.0)
    try:
        with binding.critical_section(
            "EXPORT_CAPTURE", gate_id="source-commit-stable-seed-41"
        ) as gate:
            plan, _snapshot, commit = _capture_commit(adapter, epochs, gate)
            validation = validate_source_capture_commit_v11(
                adapter,
                commit,
                plan,
                block_size_tokens=adapter._view.block_size_tokens,
                admission_gate=gate,
                timeout_s=2.0,
            )
        assert validation.passed
        assert validation.allocation_count == 3
        assert validation.no_post_commit_mutation
        assert commit.semantic_sha256 == validation.semantic_sha256
        assert tuple(item.physical_block_id for item in commit.allocations) == (1, 2, 3)
        assert tuple(item.expected_refcount for item in commit.allocations) == (2, 1, 1)
        assert commit.allocator_quiescence.passed
    finally:
        binding.close()
        adapter.cleanup_runtime(timeout_s=2.0)


def test_expected_owners_derive_from_committed_branch_tables() -> None:
    adapter, epochs = _production_adapter_fixture()
    try:
        plan = _capture_plan(adapter, epochs)
        shared_page = plan.page_order[0][0]
        private_pages = tuple(item[0] for item in plan.page_order[1:])
        assert expected_owner_set(plan, shared_page) == ("branch.0", "branch.1")
        assert expected_refcount(plan, shared_page) == 2
        assert tuple(expected_owner_set(plan, page) for page in private_pages) == (
            ("branch.0",),
            ("branch.1",),
        )
        assert tuple(expected_refcount(plan, page) for page in private_pages) == (1, 1)

        first = plan.page_order[0]
        inconsistent = plan.model_copy(
            update={
                "page_order": (
                    (first[0], first[1], first[2], ("branch.0",)),
                    *plan.page_order[1:],
                )
            }
        )
        # page_order is an observation annotation; branch-table membership is
        # the normative owner relationship.
        assert expected_owner_set(inconsistent, shared_page) == ("branch.0", "branch.1")
    finally:
        adapter.cleanup_runtime(timeout_s=2.0)


def test_source_capture_commit_rejects_same_block_reused_with_new_epoch() -> None:
    adapter, epochs = _production_adapter_fixture()
    binding = Vllm0230EngineStepBinding(adapter._view, acquire_timeout_seconds=2.0)
    try:
        with binding.critical_section(
            "EXPORT_CAPTURE", gate_id="source-commit-new-epoch-seed-41"
        ) as gate:
            plan, _snapshot, commit = _capture_commit(adapter, epochs, gate)
            request = adapter._view.scheduler.requests["branch.0"]
            issued = adapter._allocator_epoch_source.issue_from_runtime_allocation(
                request,
                previous_block_ids=(1,),
                cached_block_ids=(),
                allocator_returned_block_ids=(2,),
            )
            assert issued[0].block_id == 2
            assert issued[0].allocation_epoch > epochs[2]
            with pytest.raises(V11OwnershipProofError, match=r"epoch|stale"):
                validate_source_capture_commit_v11(
                    adapter,
                    commit,
                    plan,
                    block_size_tokens=adapter._view.block_size_tokens,
                    admission_gate=gate,
                    timeout_s=2.0,
                )
    finally:
        binding.close()
        adapter.cleanup_runtime(timeout_s=2.0)


def test_source_capture_commit_rejects_unauthorized_private_page_remap() -> None:
    adapter, epochs = _production_adapter_fixture()
    binding = Vllm0230EngineStepBinding(adapter._view, acquire_timeout_seconds=2.0)
    manager = adapter._view.manager
    original_zero = manager.tables["branch.0"]
    original_one = manager.tables["branch.1"]
    try:
        with binding.critical_section(
            "EXPORT_CAPTURE", gate_id="source-commit-remap-seed-41"
        ) as gate:
            plan, _snapshot, commit = _capture_commit(adapter, epochs, gate)
            manager.tables["branch.0"] = (
                (manager.block_pool.blocks[1], manager.block_pool.blocks[3]),
            )
            manager.tables["branch.1"] = (
                (manager.block_pool.blocks[1], manager.block_pool.blocks[2]),
            )
            with pytest.raises(V11OwnershipProofError, match=r"identity|semantic"):
                validate_source_capture_commit_v11(
                    adapter,
                    commit,
                    plan,
                    block_size_tokens=adapter._view.block_size_tokens,
                    admission_gate=gate,
                    timeout_s=2.0,
                )
    finally:
        manager.tables["branch.0"] = original_zero
        manager.tables["branch.1"] = original_one
        binding.close()
        adapter.cleanup_runtime(timeout_s=2.0)


def test_source_capture_commit_rejects_unexpected_foreign_owner() -> None:
    adapter, epochs = _production_adapter_fixture()
    binding = Vllm0230EngineStepBinding(adapter._view, acquire_timeout_seconds=2.0)
    manager = adapter._view.manager
    shared = manager.block_pool.blocks[1]
    try:
        with binding.critical_section(
            "EXPORT_CAPTURE", gate_id="source-commit-foreign-owner-seed-41"
        ) as gate:
            plan, _snapshot, commit = _capture_commit(adapter, epochs, gate)
            manager.tables["foreign.request"] = ((shared,),)
            shared.ref_cnt += 1
            with pytest.raises(V11OwnershipProofError, match=r"owner|refcount"):
                validate_source_capture_commit_v11(
                    adapter,
                    commit,
                    plan,
                    block_size_tokens=adapter._view.block_size_tokens,
                    admission_gate=gate,
                    timeout_s=2.0,
                )
    finally:
        manager.tables.pop("foreign.request", None)
        if shared.ref_cnt == 3:
            shared.ref_cnt -= 1
        binding.close()
        adapter.cleanup_runtime(timeout_s=2.0)


def test_source_capture_commit_rejects_post_commit_refcount_mutation() -> None:
    adapter, epochs = _production_adapter_fixture()
    binding = Vllm0230EngineStepBinding(adapter._view, acquire_timeout_seconds=2.0)
    private = adapter._view.manager.block_pool.blocks[2]
    try:
        with binding.critical_section(
            "EXPORT_CAPTURE", gate_id="source-commit-refcount-mutation-seed-41"
        ) as gate:
            plan, _snapshot, commit = _capture_commit(adapter, epochs, gate)
            private.ref_cnt += 1
            with pytest.raises(V11OwnershipProofError, match=r"owner|refcount"):
                validate_source_capture_commit_v11(
                    adapter,
                    commit,
                    plan,
                    block_size_tokens=adapter._view.block_size_tokens,
                    admission_gate=gate,
                    timeout_s=2.0,
                )
    finally:
        if private.ref_cnt == 2:
            private.ref_cnt -= 1
        binding.close()
        adapter.cleanup_runtime(timeout_s=2.0)


def test_fresh_quiesced_commit_accepts_derived_owner_refcount_change() -> None:
    adapter, epochs = _production_adapter_fixture()
    binding = Vllm0230EngineStepBinding(adapter._view, acquire_timeout_seconds=2.0)
    scheduler = adapter._view.scheduler
    manager = adapter._view.manager
    request_one = scheduler.requests["branch.1"]
    table_one = manager.tables["branch.1"]
    try:
        scheduler.requests.pop("branch.1")
        scheduler.running = [scheduler.requests["branch.0"]]
        manager.tables.pop("branch.1")
        manager.block_pool.blocks[1].ref_cnt -= 1
        manager.block_pool.blocks[3].ref_cnt -= 1
        plan = _capture_plan(adapter, epochs, branch_session_ids=("branch.0",))
        expected_epochs = {
            int(page_binding.source.block_index): int(page_binding.source.allocation_epoch)
            for page_binding in plan.capture_evidence.bindings
        }
        with binding.critical_section(
            "EXPORT_CAPTURE", gate_id="source-commit-owner-transition-seed-41"
        ) as gate:
            snapshot = capture_source_ownership_v11(
                adapter,
                root_session_id="root",
                branch_session_ids=("branch.0",),
                expected_allocation_epochs=expected_epochs,
                expected_source_block_count=2,
                admission_gate=gate,
                timeout_s=2.0,
            )
            commit = create_source_capture_commit_v11(
                adapter,
                plan,
                snapshot,
                block_size_tokens=adapter._view.block_size_tokens,
                admission_gate=gate,
            )
            validation = validate_source_capture_commit_v11(
                adapter,
                commit,
                plan,
                block_size_tokens=adapter._view.block_size_tokens,
                admission_gate=gate,
                timeout_s=2.0,
            )
        assert validation.passed
        assert tuple(item.expected_refcount for item in commit.allocations) == (1, 1)
        assert all(not item.shared for item in commit.allocations)
    finally:
        manager.block_pool.blocks[1].ref_cnt += 1
        manager.block_pool.blocks[3].ref_cnt += 1
        manager.tables["branch.1"] = table_one
        scheduler.requests["branch.1"] = request_one
        scheduler.running = [scheduler.requests["branch.0"], request_one]
        binding.close()
        adapter.cleanup_runtime(timeout_s=2.0)


def test_source_capture_commit_rejects_freed_source_allocation() -> None:
    adapter, epochs = _production_adapter_fixture()
    binding = Vllm0230EngineStepBinding(adapter._view, acquire_timeout_seconds=2.0)
    block = adapter._view.manager.block_pool.blocks[2]
    allocation = (2, epochs[2])
    try:
        with binding.critical_section(
            "EXPORT_CAPTURE", gate_id="source-commit-freed-seed-41"
        ) as gate:
            plan, _snapshot, commit = _capture_commit(adapter, epochs, gate)
            block.ref_cnt = 0
            adapter._allocator_epoch_source._live.discard(allocation)
            with pytest.raises(V11OwnershipProofError, match=r"stale|refcount|epoch"):
                validate_source_capture_commit_v11(
                    adapter,
                    commit,
                    plan,
                    block_size_tokens=adapter._view.block_size_tokens,
                    admission_gate=gate,
                    timeout_s=2.0,
                )
    finally:
        block.ref_cnt = 1
        adapter._allocator_epoch_source._live.add(allocation)
        binding.close()
        adapter.cleanup_runtime(timeout_s=2.0)


def test_source_capture_commit_ignores_diagnostic_allocator_history() -> None:
    adapter, epochs = _production_adapter_fixture()
    binding = Vllm0230EngineStepBinding(adapter._view, acquire_timeout_seconds=2.0)
    try:
        with binding.critical_section(
            "EXPORT_CAPTURE", gate_id="source-commit-diagnostics-seed-41"
        ) as gate:
            plan, _snapshot, commit = _capture_commit(adapter, epochs, gate)
            adapter._observer.snapshot_epoch += 100
            adapter._allocator_epoch_source._next_epoch += 100
            adapter._view.manager.block_pool.blocks[1].block_hash = "changed-diagnostic-hash"
            validation = validate_source_capture_commit_v11(
                adapter,
                commit,
                plan,
                block_size_tokens=adapter._view.block_size_tokens,
                admission_gate=gate,
                timeout_s=2.0,
            )
        assert validation.passed
    finally:
        binding.close()
        adapter.cleanup_runtime(timeout_s=2.0)


def test_source_capture_commit_canonicalizes_layout_enumeration_order() -> None:
    adapter, epochs = _production_adapter_fixture()
    binding = Vllm0230EngineStepBinding(adapter._view, acquire_timeout_seconds=2.0)
    original_inspect = adapter.inspect_physical_kv_layout
    try:
        with binding.critical_section(
            "EXPORT_CAPTURE", gate_id="source-commit-layout-order-seed-41"
        ) as gate:
            plan, _snapshot, commit = _capture_commit(adapter, epochs, gate)

            def reversed_layout(session_ids: tuple[str, ...], *, timeout_s: float) -> object:
                observed = original_inspect(session_ids, timeout_s=timeout_s)
                return observed.model_copy(update={"blocks": tuple(reversed(observed.blocks))})

            adapter.inspect_physical_kv_layout = reversed_layout  # type: ignore[method-assign]
            validation = validate_source_capture_commit_v11(
                adapter,
                commit,
                plan,
                block_size_tokens=adapter._view.block_size_tokens,
                admission_gate=gate,
                timeout_s=2.0,
            )
        assert validation.passed
    finally:
        adapter.inspect_physical_kv_layout = original_inspect  # type: ignore[method-assign]
        binding.close()
        adapter.cleanup_runtime(timeout_s=2.0)


def test_stale_creation_plan_fails_and_quiesced_plan_passes() -> None:
    adapter, epochs = _production_adapter_fixture()
    binding = Vllm0230EngineStepBinding(adapter._view, acquire_timeout_seconds=2.0)
    try:
        stale = build_canonical_capture_plan(
            branches=(
                RuntimeBranchCaptureInput(
                    logical_branch_id="branch.0",
                    parent_logical_branch_id="root",
                    token_ids=tuple(range(16)),
                    computed_tokens=16,
                    source_block_indices=(1,),
                ),
                RuntimeBranchCaptureInput(
                    logical_branch_id="branch.1",
                    parent_logical_branch_id="root",
                    token_ids=tuple(range(16)),
                    computed_tokens=16,
                    source_block_indices=(1,),
                ),
            ),
            block_size_tokens=16,
            logical_token_bytes=64,
            physical_page_bytes=1_024,
            gpu_uuid="GPU-fixture",
            allocation_epoch_by_block=epochs,
        )
        with binding.critical_section(
            "EXPORT_CAPTURE", gate_id="source-commit-stale-plan-seed-41"
        ) as gate:
            quiesced, snapshot, commit = _capture_commit(adapter, epochs, gate)
            with pytest.raises(V11OwnershipProofError, match=r"cover|source commit"):
                create_source_capture_commit_v11(
                    adapter,
                    stale,
                    snapshot,
                    block_size_tokens=16,
                    admission_gate=gate,
                )
            validation = validate_source_capture_commit_v11(
                adapter,
                commit,
                quiesced,
                block_size_tokens=16,
                admission_gate=gate,
                timeout_s=2.0,
            )
        assert validation.passed
    finally:
        binding.close()
        adapter.cleanup_runtime(timeout_s=2.0)


def test_allocator_quiescence_rejects_active_branch_and_pending_cleanup() -> None:
    adapter, _epochs = _production_adapter_fixture()
    binding = Vllm0230EngineStepBinding(adapter._view, acquire_timeout_seconds=2.0)
    branch = adapter._sessions["branch.0"]
    original_running = adapter._view.scheduler.running
    try:
        with binding.critical_section(
            "EXPORT_CAPTURE", gate_id="allocator-quiescence-seed-41"
        ) as gate:
            evidence = prove_allocator_quiescence_v11(
                adapter,
                branch_session_ids=("branch.0", "branch.1"),
                admission_gate=gate,
            )
            assert evidence.passed
            branch.phase = LiveSessionPhase.DECODING
            with pytest.raises(V11OwnershipProofError, match="paused"):
                prove_allocator_quiescence_v11(
                    adapter,
                    branch_session_ids=("branch.0", "branch.1"),
                    admission_gate=gate,
                )
            branch.phase = LiveSessionPhase.PAUSED
            adapter._view.scheduler.waiting.rows.append(SimpleNamespace(request_id="deferred"))
            with pytest.raises(V11OwnershipProofError, match="empty pending queues"):
                prove_allocator_quiescence_v11(
                    adapter,
                    branch_session_ids=("branch.0", "branch.1"),
                    admission_gate=gate,
                )
            adapter._view.scheduler.waiting.rows.clear()
            adapter._view.scheduler.skipped_waiting.rows.append(
                SimpleNamespace(request_id="deferred-skipped")
            )
            with pytest.raises(V11OwnershipProofError, match="empty pending queues"):
                prove_allocator_quiescence_v11(
                    adapter,
                    branch_session_ids=("branch.0", "branch.1"),
                    admission_gate=gate,
                )
            adapter._view.scheduler.skipped_waiting.rows.clear()
            adapter._view.scheduler.scheduler_config.async_scheduling = True
            with pytest.raises(V11OwnershipProofError, match="empty pending queues"):
                prove_allocator_quiescence_v11(
                    adapter,
                    branch_session_ids=("branch.0", "branch.1"),
                    admission_gate=gate,
                )
            adapter._view.scheduler.scheduler_config.async_scheduling = False
            adapter._view.scheduler.running = original_running[:-1]
            with pytest.raises(V11OwnershipProofError, match="exact live source request tables"):
                prove_allocator_quiescence_v11(
                    adapter,
                    branch_session_ids=("branch.0", "branch.1"),
                    admission_gate=gate,
                )
            adapter._view.scheduler.running = original_running
    finally:
        branch.phase = LiveSessionPhase.PAUSED
        adapter._view.scheduler.waiting.rows.clear()
        adapter._view.scheduler.skipped_waiting.rows.clear()
        adapter._view.scheduler.scheduler_config.async_scheduling = False
        adapter._view.scheduler.running = original_running
        binding.close()
        adapter.cleanup_runtime(timeout_s=2.0)


def test_source_capture_commit_rejects_expired_witness() -> None:
    adapter, epochs = _production_adapter_fixture()
    binding = Vllm0230EngineStepBinding(adapter._view, acquire_timeout_seconds=2.0)
    try:
        with binding.critical_section(
            "EXPORT_CAPTURE", gate_id="source-commit-expired-gate-seed-41"
        ) as gate:
            plan, _snapshot, commit = _capture_commit(adapter, epochs, gate)
        with pytest.raises(V11EngineStepBindingError, match="stale, foreign, or not held"):
            validate_source_capture_commit_v11(
                adapter,
                commit,
                plan,
                block_size_tokens=adapter._view.block_size_tokens,
                admission_gate=gate,
                timeout_s=2.0,
            )
    finally:
        binding.close()
        adapter.cleanup_runtime(timeout_s=2.0)


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
