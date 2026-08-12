from __future__ import annotations

import importlib
import importlib.metadata
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from sloforge.continuum.adapters.external import UnsupportedRuntimeVersionError
from sloforge.continuum.adapters.real_runtime import (
    LiveSessionPhase,
    PhysicalKvLayoutSnapshot,
    RealRuntimeCapability,
    RuntimeModelIdentity,
)
from sloforge.continuum.adapters.vllm_live import (
    VLLM_LIVE_ADAPTER_VERSION,
    VLLM_LIVE_RUNTIME_VERSION,
    VllmLiveConfigurationError,
    VllmLiveInternalView,
    VllmLiveStateAdapter,
)
from sloforge.helix.characterization.trace import (
    BranchWorkloadEventV1,
    StateOperationEventV1,
    StateOperationType,
    TraceLevel,
    WorkloadProvenance,
    verify_event,
)


class FullAttentionSpec:
    def __init__(self, layer: str) -> None:
        self.block_size = 16
        self.page_size_bytes = 512
        self.dtype = "bfloat16"
        self.sliding_window = None
        self.attention_chunk_size = None
        self.layer = layer


class UniformTypeKVCacheSpecs:
    def __init__(self) -> None:
        self.block_size = 16
        self.kv_cache_specs = {
            "layer.0.attn": FullAttentionSpec("layer.0.attn"),
            "layer.1.attn": FullAttentionSpec("layer.1.attn"),
        }
        self.page_size_bytes = sum(spec.page_size_bytes for spec in self.kv_cache_specs.values())


class FakeDevice:
    type = "cuda"
    index = 0

    def __str__(self) -> str:
        return "cuda:0"


class FakeTensor:
    def __init__(self, *, device: object | None = None, nbytes: int = 4096) -> None:
        self.device = device or FakeDevice()
        self.nbytes = nbytes


class FakeBlock:
    def __init__(self, block_id: int) -> None:
        self.block_id = block_id
        self.ref_cnt = 0
        self.block_hash = None
        self.is_null = block_id == 0


class FakeBlockPool:
    def __init__(self) -> None:
        self.blocks = [FakeBlock(block_id) for block_id in range(4)]

    def get_num_free_blocks(self) -> int:
        return sum(block.ref_cnt == 0 and not block.is_null for block in self.blocks)


class FakeManager:
    def __init__(self) -> None:
        layer_names = ["layer.0.attn", "layer.1.attn"]
        spec = FullAttentionSpec("merged")
        self.block_pool = FakeBlockPool()
        self.kv_cache_config = SimpleNamespace(
            num_blocks=len(self.block_pool.blocks),
            kv_cache_groups=[SimpleNamespace(kv_cache_spec=spec, layer_names=layer_names)],
            kv_cache_tensors=[
                SimpleNamespace(size=2048, shared_by=[layer]) for layer in layer_names
            ],
        )
        self.tables: dict[str, tuple[tuple[FakeBlock, ...], ...]] = {}

    def get_blocks(self, request_id: str) -> object:
        return SimpleNamespace(blocks=self.tables[request_id])

    def allocate_slots(self, request: object, *args: object, **kwargs: object) -> object:
        del request, args, kwargs
        return object()

    def free(self, request: object) -> None:
        del request

    def reset_prefix_cache(self) -> bool:
        if any(block.ref_cnt for block in self.block_pool.blocks if not block.is_null):
            return False
        for block in self.block_pool.blocks:
            block.block_hash = None
        return True


class InprocClient:
    def __init__(self, engine_core: object) -> None:
        self.engine_core = engine_core
        self.shutdown_called = False

    def shutdown(self) -> None:
        self.shutdown_called = True


InprocClient.__module__ = "vllm.v1.engine.core_client"


class FakeLlmEngine:
    def __init__(self, client: InprocClient, config: object) -> None:
        self.engine_core = client
        self.vllm_config = config

    def add_request(self, request_id: str, prompt: object, params: object) -> str:
        del prompt, params
        return request_id

    def step(self) -> list[object]:
        return []

    def abort_request(self, request_ids: list[str]) -> None:
        del request_ids


def fake_engine(
    *, tensor_device: object | None = None, prefix_cache: bool = True
) -> SimpleNamespace:
    manager = FakeManager()
    runner = SimpleNamespace(
        device=FakeDevice(),
        kv_caches=[
            FakeTensor(device=tensor_device, nbytes=2048),
            FakeTensor(device=tensor_device, nbytes=2048),
        ],
    )
    core = SimpleNamespace(
        scheduler=SimpleNamespace(kv_cache_manager=manager, requests={}),
        model_executor=SimpleNamespace(
            driver_worker=SimpleNamespace(worker=SimpleNamespace(model_runner=runner))
        ),
    )
    client = InprocClient(core)
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            data_parallel_size=1,
        ),
        scheduler_config=SimpleNamespace(async_scheduling=False),
        speculative_config=None,
        lora_config=None,
        cache_config=SimpleNamespace(enable_prefix_caching=prefix_cache),
    )
    return SimpleNamespace(llm_engine=FakeLlmEngine(client, config))


def identity() -> RuntimeModelIdentity:
    return RuntimeModelIdentity(
        runtime="vllm",
        runtime_version=VLLM_LIVE_RUNTIME_VERSION,
        adapter_version=VLLM_LIVE_ADAPTER_VERSION,
        model_id="model",
        model_revision="revision",
        tokenizer_id="tokenizer",
        tokenizer_revision="tokenizer-revision",
        dtype="bfloat16",
        device="cuda:0",
        policy_epoch="policy-1",
    )


def local_snapshot(tmp_path: Path) -> Path:
    path = tmp_path / identity().model_revision
    path.mkdir()
    (path / "config.json").write_text("{}\n", encoding="utf-8")
    (path / "tokenizer_config.json").write_text("{}\n", encoding="utf-8")
    (path / "model.safetensors").write_bytes(b"weights")
    return path.resolve()


def test_version_gate_fails_before_internal_access() -> None:
    with pytest.raises(UnsupportedRuntimeVersionError):
        VllmLiveInternalView(object(), runtime_version="0.26.0")


def test_internal_view_requires_real_cuda_backing_and_exact_pool_bytes() -> None:
    view = VllmLiveInternalView(fake_engine(), runtime_version=VLLM_LIVE_RUNTIME_VERSION)
    assert view.device == "cuda:0"
    assert view.block_size_tokens == 16
    assert view.page_size_bytes == 1024
    assert view.kv_pool_reserved_bytes == 4096
    assert view.layer_ids == ("layer.0.attn", "layer.1.attn")

    cpu = SimpleNamespace(type="cpu")
    with pytest.raises(VllmLiveConfigurationError, match="not CUDA-backed"):
        VllmLiveInternalView(
            fake_engine(tensor_device=cpu), runtime_version=VLLM_LIVE_RUNTIME_VERSION
        )


def test_internal_view_rejects_per_layer_page_size_as_physical_block_bytes() -> None:
    engine = fake_engine()
    manager = engine.llm_engine.engine_core.engine_core.scheduler.kv_cache_manager
    manager.kv_cache_config.kv_cache_tensors.pop()
    engine.llm_engine.engine_core.engine_core.model_executor.driver_worker.worker.model_runner.kv_caches.pop()
    with pytest.raises(VllmLiveConfigurationError, match="per-layer page sizes"):
        VllmLiveInternalView(engine, runtime_version=VLLM_LIVE_RUNTIME_VERSION)


def test_adapter_is_importable_without_vllm_and_cleanup_is_idempotent() -> None:
    engine = fake_engine()
    adapter = VllmLiveStateAdapter(
        engine, identity=identity(), runtime_version=VLLM_LIVE_RUNTIME_VERSION
    )
    capabilities = adapter.capability_matrix.capabilities
    assert RealRuntimeCapability.FORK_SAME_POLICY_SESSION in capabilities
    assert RealRuntimeCapability.EMIT_STATE_OPERATION_TRACE not in capabilities

    adapter.cleanup_runtime(timeout_s=1.0)
    adapter.cleanup_runtime(timeout_s=1.0)
    assert engine.llm_engine.engine_core.shutdown_called is True


def test_cache_disabled_engine_does_not_advertise_shared_root() -> None:
    adapter = VllmLiveStateAdapter(
        fake_engine(prefix_cache=False),
        identity=identity(),
        runtime_version=VLLM_LIVE_RUNTIME_VERSION,
    )
    try:
        assert (
            RealRuntimeCapability.FORK_SAME_POLICY_SESSION
            not in adapter.capability_matrix.capabilities
        )
    finally:
        adapter.cleanup_runtime(timeout_s=1.0)


def test_empty_cache_groups_are_not_recorded_as_prefix_cache_hits() -> None:
    engine = fake_engine(prefix_cache=False)
    adapter = VllmLiveStateAdapter(
        engine,
        identity=identity(),
        runtime_version=VLLM_LIVE_RUNTIME_VERSION,
    )
    manager = engine.llm_engine.engine_core.engine_core.scheduler.kv_cache_manager
    request = SimpleNamespace(request_id="independent-request")
    manager.tables[request.request_id] = ((manager.block_pool.blocks[1],),)
    empty_computed_blocks = SimpleNamespace(blocks=((),))
    try:
        manager.allocate_slots(
            request,
            16,
            num_new_computed_tokens=0,
            new_computed_blocks=empty_computed_blocks,
        )
        assert adapter._observer.initial_cached_tokens[request.request_id] == 0
        assert adapter._observer.initial_cached_block_ids[request.request_id] == ()
    finally:
        adapter.cleanup_runtime(timeout_s=1.0)


def test_external_computed_pages_receive_allocation_epochs_when_return_is_empty() -> None:
    engine = fake_engine(prefix_cache=True)
    adapter = VllmLiveStateAdapter(
        engine,
        identity=identity(),
        runtime_version=VLLM_LIVE_RUNTIME_VERSION,
    )
    manager = engine.llm_engine.engine_core.engine_core.scheduler.kv_cache_manager
    request = SimpleNamespace(request_id="external-restore-request")
    manager.tables[request.request_id] = ((),)

    def allocate_external_pages(request_arg: object, *args: object, **kwargs: object) -> object:
        del args, kwargs
        manager.tables[cast(Any, request_arg).request_id] = (
            (manager.block_pool.blocks[1], manager.block_pool.blocks[2]),
        )
        # vLLM returns an empty group here because every token is externally
        # computed; the newly allocated pages exist only in the full table.
        return SimpleNamespace(blocks=((),))

    adapter._original_allocate_slots = allocate_external_pages
    try:
        manager.allocate_slots(
            request,
            num_new_tokens=0,
            num_new_computed_tokens=0,
            new_computed_blocks=SimpleNamespace(blocks=((),)),
            num_external_computed_tokens=32,
        )
        assert adapter._observer.block_epochs[1] > 0
        assert adapter._observer.block_epochs[2] > adapter._observer.block_epochs[1]
    finally:
        adapter.cleanup_runtime(timeout_s=1.0)


def test_cleanup_frees_untracked_staged_runtime_requests_before_prefix_reset() -> None:
    engine = fake_engine(prefix_cache=True)
    adapter = VllmLiveStateAdapter(
        engine,
        identity=identity(),
        runtime_version=VLLM_LIVE_RUNTIME_VERSION,
    )
    scheduler = engine.llm_engine.engine_core.engine_core.scheduler
    manager = scheduler.kv_cache_manager
    request = SimpleNamespace(request_id="internal-restored-request")
    block = manager.block_pool.blocks[1]
    block.ref_cnt = 1
    manager.tables[request.request_id] = ((block,),)

    class Queue:
        def remove_requests(self, requests: list[object]) -> None:
            assert requests == [request]

    scheduler.requests[request.request_id] = request
    scheduler.running = []
    scheduler.waiting = Queue()
    scheduler.skipped_waiting = Queue()

    def free_staged(request_arg: object) -> None:
        assert request_arg is request
        block.ref_cnt = 0
        manager.tables.pop(request.request_id)

    adapter._original_free = free_staged
    adapter.cleanup_runtime(timeout_s=1.0)
    assert scheduler.requests == {}
    assert block.ref_cnt == 0
    assert engine.llm_engine.engine_core.shutdown_called is True


def test_exact_block_release_evidence_uses_native_pool_state() -> None:
    engine = fake_engine()
    adapter = VllmLiveStateAdapter(
        engine, identity=identity(), runtime_version=VLLM_LIVE_RUNTIME_VERSION
    )
    try:
        block = (
            engine.llm_engine.engine_core.engine_core.scheduler.kv_cache_manager.block_pool.blocks[
                1
            ]
        )
        block.ref_cnt = 1
        block.block_hash = "cached"
        private = cast(Any, adapter)
        private._observer.block_epochs[1] = 7
        before = adapter.inspect_block_release_evidence(("vllm:kv-group:0:block:1",), timeout_s=1.0)
        assert before.blocks[0].native_refcount == 1
        assert before.blocks[0].allocator_available is False
        block.ref_cnt = 0
        after = adapter.inspect_block_release_evidence(("vllm:kv-group:0:block:1",), timeout_s=1.0)
        assert after.blocks[0].allocation_epoch == 7
        assert after.blocks[0].allocator_available is True
        assert after.blocks[0].block_hash_present is True
        assert after.pool_free_block_count == after.pool_usable_block_count
    finally:
        block.ref_cnt = 0
        adapter.cleanup_runtime(timeout_s=1.0)


def test_create_rejects_implicit_engine_defaults_before_importing_vllm(
    tmp_path: Path,
) -> None:
    snapshot = local_snapshot(tmp_path)
    with pytest.raises(ValueError, match="explicitly set"):
        VllmLiveStateAdapter.create(
            identity=identity(),
            execution_model_path=str(snapshot),
            seed=7,
            timeout_s=1.0,
            llm_kwargs={"model": str(snapshot)},
        )


def test_create_checks_distribution_version_before_import_or_model_load(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    snapshot = local_snapshot(tmp_path)
    imported = False

    def forbidden_import(name: str) -> object:
        nonlocal imported
        imported = True
        raise AssertionError(name)

    monkeypatch.setattr(importlib.metadata, "version", lambda name: "0.26.0")
    monkeypatch.setattr(importlib, "import_module", forbidden_import)
    with pytest.raises(UnsupportedRuntimeVersionError):
        VllmLiveStateAdapter.create(
            identity=identity(),
            execution_model_path=str(snapshot),
            seed=7,
            timeout_s=1.0,
            llm_kwargs={
                "model": str(snapshot),
                "revision": "revision",
                "tokenizer": str(snapshot),
                "tokenizer_revision": "tokenizer-revision",
                "dtype": "bfloat16",
                "max_model_len": 16_641,
                "enable_prefix_caching": True,
                "block_size": 16,
                "max_num_seqs": 32,
                "seed": 7,
                "tensor_parallel_size": 1,
                "async_scheduling": False,
                "cpu_offload_gb": 0.0,
            },
        )
    assert imported is False


def test_create_loads_local_snapshot_without_changing_logical_identity(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    snapshot = local_snapshot(tmp_path)
    captured: dict[str, object] = {}

    def factory(**kwargs: object) -> object:
        captured.update(kwargs)
        return fake_engine()

    original_import = importlib.import_module

    def fake_import(name: str) -> object:
        return SimpleNamespace(LLM=factory) if name == "vllm" else original_import(name)

    monkeypatch.setattr(importlib.metadata, "version", lambda name: VLLM_LIVE_RUNTIME_VERSION)
    monkeypatch.setattr(importlib, "import_module", fake_import)
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    adapter = VllmLiveStateAdapter.create(
        identity=identity(),
        execution_model_path=str(snapshot),
        seed=7,
        timeout_s=1.0,
        llm_kwargs={
            "model": str(snapshot),
            "revision": identity().model_revision,
            "tokenizer": str(snapshot),
            "tokenizer_revision": identity().tokenizer_revision,
            "dtype": "bfloat16",
            "max_model_len": 16_641,
            "enable_prefix_caching": True,
            "block_size": 16,
            "max_num_seqs": 32,
            "seed": 7,
            "tensor_parallel_size": 1,
            "async_scheduling": False,
            "cpu_offload_gb": 0.0,
        },
    )
    try:
        assert captured["model"] == str(snapshot)
        assert captured["tokenizer"] == str(snapshot)
        assert "device" not in captured
        assert adapter.capability_matrix.identity == identity()
    finally:
        adapter.cleanup_runtime(timeout_s=1.0)


def test_live_adapter_records_sealed_pointer_free_hardware_events() -> None:
    engine = fake_engine()
    adapter = VllmLiveStateAdapter(
        engine,
        identity=identity(),
        runtime_version=VLLM_LIVE_RUNTIME_VERSION,
        trace_level=TraceLevel.FULL,
        trace_id="live-test",
    )
    try:
        adapter.start_session("root", token_ids=tuple(range(16)), seed=1, timeout_s=1.0)
        private = cast(Any, adapter)
        session = private._sessions["root"]
        session.phase = LiveSessionPhase.PREFILLED
        session.runtime_request_id = "root-runtime"
        manager = engine.llm_engine.engine_core.engine_core.scheduler.kv_cache_manager
        block = manager.block_pool.blocks[1]
        block.block_hash = "content-hash"
        manager.tables["root-runtime"] = ((block,),)
        private._observer.last_request_blocks["root-runtime"] = ((1,),)
        root = adapter.create_shared_root_reference("root", prefix_token_count=16, timeout_s=1.0)
        adapter.fork_same_policy_session(
            root.root_reference_id,
            "branch-a",
            divergent_token_id=99,
            seed=2,
            timeout_s=1.0,
        )
        branch_events = adapter.emit_branch_workload_trace()
        state_events = adapter.emit_state_operation_trace()
        overhead = adapter.inspect_trace_self_overhead()
        assert branch_events and state_events
        assert overhead["event_record_calls"] == len(branch_events) + len(state_events)
        assert overhead["layout_observation_calls"] >= 1
        assert all(type(value) in {int, str} for value in overhead.values())
        assert not any("0x" in str(value) for value in overhead.values())
        events: tuple[BranchWorkloadEventV1 | StateOperationEventV1, ...] = (
            *branch_events,
            *state_events,
        )
        assert all(event.provenance is WorkloadProvenance.HARDWARE_BACKED_REAL for event in events)
        assert all(
            event.operation_type is not StateOperationType.STATE_COW for event in state_events
        )
        assert all(event.cpu_time_ns is None for event in branch_events)
        assert all(
            event.cpu_time_ns == 0
            and event.gpu_time_ns == 0
            and event.attributes["cpu_time_measured"] is False
            and event.attributes["gpu_time_measured"] is False
            for event in state_events
        )
        for event in events:
            verify_event(event)
            assert all(
                value is None or isinstance(value, (bool, int, float, str))
                for value in event.attributes.values()
            )
            assert not any("0x" in str(value) for value in event.attributes.values())
    finally:
        adapter.cleanup_runtime(timeout_s=1.0)
    assert not hasattr(adapter, "_engine")
    assert not hasattr(adapter, "_view")
    adapter.cleanup_runtime(timeout_s=1.0)


def test_trace_scope_disappearance_is_not_a_native_free() -> None:
    engine = fake_engine()
    adapter = VllmLiveStateAdapter(
        engine,
        identity=identity(),
        runtime_version=VLLM_LIVE_RUNTIME_VERSION,
        trace_level=TraceLevel.FULL,
        trace_id="release-proof-test",
    )
    try:
        adapter.start_session("root", token_ids=tuple(range(16)), seed=1, timeout_s=1.0)
        private = cast(Any, adapter)
        session = private._sessions["root"]
        session.phase = LiveSessionPhase.PREFILLED
        session.runtime_request_id = "root-runtime"
        manager = engine.llm_engine.engine_core.engine_core.scheduler.kv_cache_manager
        block = manager.block_pool.blocks[1]
        block.block_hash = "content-hash"
        manager.tables["root-runtime"] = ((block,),)
        private._observer.last_request_blocks["root-runtime"] = ((1,),)
        root = adapter.create_shared_root_reference("root", prefix_token_count=16, timeout_s=1.0)
        recorder = private._trace_recorder
        before_free = sum(
            event.operation_type is StateOperationType.STATE_FREE
            for event in adapter.emit_state_operation_trace()
        )
        recorder.observe_layout(
            PhysicalKvLayoutSnapshot(
                runtime="vllm",
                runtime_version="0.23.0",
                snapshot_epoch=99,
                observed_at_monotonic_ns=99,
                session_ids=("root",),
                blocks=(),
                shared_prefix_block_ids=(),
                private_suffix_block_ids=(),
                root_reference_count=0,
                physical_assigned_bytes=0,
                shared_prefix_bytes=0,
                private_suffix_bytes=0,
                kv_pool_reserved_bytes=4096,
            ),
            session_id="root",
            branch_group_id=root.root_reference_id,
            cause="selected_scope_disappearance",
        )
        assert (
            sum(
                event.operation_type is StateOperationType.STATE_FREE
                for event in adapter.emit_state_operation_trace()
            )
            == before_free
        )
        evidence = adapter.inspect_block_release_evidence(root.block_ids, timeout_s=1.0)
        recorder.observe_release_evidence(
            evidence,
            session_id="root",
            branch_group_id=root.root_reference_id,
            cause="native_release",
        )
        assert (
            adapter.emit_state_operation_trace()[-1].operation_type
            is StateOperationType.STATE_RECLAIM
        )
    finally:
        adapter.cleanup_runtime(timeout_s=1.0)
