from __future__ import annotations

import threading
from dataclasses import dataclass

import pytest

import sloforge.continuum.adapters.vllm_reclamation_v11 as v11
from sloforge.continuum.adapters.vllm_reclamation_v11_sync import (
    V11EngineStepBindingError,
    V11EngineStepTimeout,
    Vllm0230EngineStepBinding,
    require_production_gate_v11,
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


class _Manager:
    def __init__(self, mutations: list[str]) -> None:
        self.mutations = mutations

    def allocate_slots(self, *_args: object, **_kwargs: object) -> object:
        self.mutations.append("allocate_slots")
        return object()

    def free(self, *_args: object, **_kwargs: object) -> None:
        self.mutations.append("free")

    def cache_blocks(self, *_args: object, **_kwargs: object) -> None:
        self.mutations.append("cache_blocks")

    def reset_prefix_cache(self) -> bool:
        self.mutations.append("reset_prefix_cache")
        return True

    def take_new_block_ids(self) -> list[int]:
        self.mutations.append("take_new_block_ids")
        return []


class _Scheduler:
    def __init__(self, manager: _Manager, mutations: list[str]) -> None:
        self.kv_cache_manager = manager
        self.scheduler_config = type("SchedulerConfig", (), {"async_scheduling": False})()
        self.waiting = _Queue()
        self.skipped_waiting = _Queue()
        self.mutations = mutations

    def add_request(self, *_args: object, **_kwargs: object) -> None:
        self.mutations.append("scheduler.add_request")

    def _enqueue_waiting_request(self, request: object) -> None:
        self.mutations.append("scheduler.enqueue")
        self.waiting.add_request(request)

    def schedule(self) -> object:
        self.mutations.append("scheduler.schedule")
        return object()

    def update_from_output(self, *_args: object, **_kwargs: object) -> None:
        self.mutations.append("scheduler.update_from_output")

    def finish_requests(self, *_args: object, **_kwargs: object) -> None:
        self.mutations.append("scheduler.finish_requests")


class _EngineCore:
    def __init__(self, scheduler: _Scheduler, mutations: list[str]) -> None:
        self.scheduler = scheduler
        self.mutations = mutations

    def add_request(self, *_args: object, **_kwargs: object) -> None:
        self.mutations.append("core.add_request")

    def step(self) -> None:
        self.mutations.append("core.step")
        self.scheduler.schedule()


def _client_type() -> type:
    def add_request(self, *_args: object, **_kwargs: object) -> None:
        self.mutations.append("client.add_request")

    def abort_requests(self, *_args: object, **_kwargs: object) -> None:
        self.mutations.append("client.abort_requests")

    return type(
        "InprocClient",
        (),
        {
            "__module__": "vllm.v1.engine.core_client",
            "add_request": add_request,
            "abort_requests": abort_requests,
        },
    )


class _LlmEngine:
    def __init__(self, client: object, scheduler: _Scheduler, mutations: list[str]) -> None:
        self.engine_core = client
        self.scheduler = scheduler
        self.mutations = mutations

    def add_request(self, request: object) -> None:
        self.mutations.append("engine.add_request")
        self.engine_core.add_request(request)
        self.scheduler.add_request(request)

    def abort_request(self, request: object) -> None:
        self.mutations.append("engine.abort_request")
        self.engine_core.abort_requests([request])
        self.scheduler.finish_requests([request])

    def step(self) -> None:
        self.mutations.append("engine.step")
        self.engine_core.engine_core.step()
        self.scheduler.kv_cache_manager.allocate_slots()
        self.scheduler.update_from_output(object())


@dataclass
class _View:
    runtime_version: str
    llm_engine: object
    engine_core_client: object
    engine_core: object
    scheduler: object
    manager: object
    vllm_config: object


def _view(*, async_scheduling: bool = False) -> tuple[_View, list[str]]:
    mutations: list[str] = []
    manager = _Manager(mutations)
    scheduler = _Scheduler(manager, mutations)
    core = _EngineCore(scheduler, mutations)
    client = _client_type()()
    client.engine_core = core
    client.mutations = mutations
    engine = _LlmEngine(client, scheduler, mutations)
    config = type(
        "VllmConfig",
        (),
        {"scheduler_config": type("Config", (), {"async_scheduling": async_scheduling})()},
    )()
    return (
        _View(
            runtime_version="0.23.0",
            llm_engine=engine,
            engine_core_client=client,
            engine_core=core,
            scheduler=scheduler,
            manager=manager,
            vllm_config=config,
        ),
        mutations,
    )


def test_binding_wraps_real_step_scheduler_manager_and_queue_surfaces() -> None:
    view, mutations = _view()
    binding = Vllm0230EngineStepBinding(view, acquire_timeout_seconds=1.0)
    evidence = binding.evidence()

    assert evidence.passed
    assert {
        "llm_engine.step",
        "scheduler.schedule",
        "scheduler.update_from_output",
        "manager.allocate_slots",
        "manager.free",
        "manager.take_new_block_ids",
        "waiting.add_request",
        "waiting.remove_requests",
    }.issubset(evidence.wrapped_surfaces)

    view.llm_engine.step()
    assert mutations == [
        "engine.step",
        "core.step",
        "scheduler.schedule",
        "allocate_slots",
        "scheduler.update_from_output",
    ]
    binding.close()


@pytest.mark.parametrize(
    ("surface", "invoke"),
    [
        ("engine step", lambda view: view.llm_engine.step()),
        ("scheduler reassignment", lambda view: view.scheduler.schedule()),
        ("branch admission", lambda view: view.scheduler._enqueue_waiting_request(object())),
        ("block-table allocation", lambda view: view.manager.allocate_slots()),
        ("block release", lambda view: view.manager.free(object())),
        ("allocator zero-queue drain", lambda view: view.manager.take_new_block_ids()),
    ],
)
def test_export_capture_excludes_every_production_mutation_surface(
    surface: str,
    invoke: object,
) -> None:
    del surface
    view, mutations = _view()
    binding = Vllm0230EngineStepBinding(view, acquire_timeout_seconds=1.0)
    attempted = threading.Event()
    completed = threading.Event()

    def mutate() -> None:
        attempted.set()
        invoke(view)
        completed.set()

    with binding.critical_section("EXPORT_CAPTURE", gate_id="capture-gate-seed-41") as gate:
        require_production_gate_v11(
            gate,
            view.scheduler,
            view.manager,
            operation="EXPORT_CAPTURE",
        )
        thread = threading.Thread(target=mutate, daemon=False)
        thread.start()
        assert attempted.wait(timeout=1.0)
        assert not completed.wait(timeout=0.05)
        assert not mutations

    thread.join(timeout=1.0)
    assert not thread.is_alive()
    assert completed.is_set()
    assert mutations
    binding.close()


def test_import_witness_is_held_reentrant_and_expires_before_future_step() -> None:
    view, mutations = _view()
    binding = Vllm0230EngineStepBinding(view, acquire_timeout_seconds=1.0)
    with binding.critical_section("IMPORT_ADMISSION", gate_id="restore-gate-seed-41") as gate:
        require_production_gate_v11(
            gate,
            view.scheduler,
            view.manager,
            operation="IMPORT_ADMISSION",
        )
        view.manager.allocate_slots()
        view.manager.cache_blocks(object(), 16)
        view.scheduler._enqueue_waiting_request(object())
        require_production_gate_v11(
            gate,
            view.scheduler,
            view.manager,
            operation="IMPORT_ADMISSION",
        )

    assert mutations == ["allocate_slots", "cache_blocks", "scheduler.enqueue"]
    with pytest.raises(V11EngineStepBindingError, match="stale, foreign, or not held"):
        require_production_gate_v11(
            gate,
            view.scheduler,
            view.manager,
            operation="IMPORT_ADMISSION",
        )
    binding.close()


def test_witness_rejects_wrong_operation_foreign_runtime_and_displaced_wrapper() -> None:
    view, _mutations = _view()
    other_view, _other_mutations = _view()
    binding = Vllm0230EngineStepBinding(view, acquire_timeout_seconds=1.0)
    other_binding = Vllm0230EngineStepBinding(other_view, acquire_timeout_seconds=1.0)

    with binding.critical_section("EXPORT_CAPTURE", gate_id="capture-gate-seed-73") as gate:
        with pytest.raises(V11EngineStepBindingError, match="stale, foreign, or not held"):
            require_production_gate_v11(
                gate,
                view.scheduler,
                view.manager,
                operation="IMPORT_ADMISSION",
            )
        with pytest.raises(V11EngineStepBindingError, match="stale, foreign, or not held"):
            binding.require_witness(
                gate,
                scheduler=other_view.scheduler,
                manager=other_view.manager,
                operation="EXPORT_CAPTURE",
            )

    original_wrapper = view.llm_engine.step
    view.llm_engine.step = lambda: None
    with pytest.raises(V11EngineStepBindingError, match="binding was displaced"):
        binding.evidence()
        binding.assert_bound()
    view.llm_engine.step = original_wrapper
    binding.close()
    other_binding.close()


def test_binding_fails_closed_for_async_or_nonproduction_client() -> None:
    async_view, _mutations = _view(async_scheduling=True)
    with pytest.raises(V11EngineStepBindingError, match="async scheduling"):
        Vllm0230EngineStepBinding(async_view)

    wrong_view, _mutations = _view()
    type(wrong_view.engine_core_client).__module__ = "tests.fake_vllm"
    with pytest.raises(V11EngineStepBindingError, match="production V1 InprocClient"):
        Vllm0230EngineStepBinding(wrong_view)


def test_engine_step_times_out_while_another_thread_holds_capture_gate() -> None:
    view, _mutations = _view()
    binding = Vllm0230EngineStepBinding(view, acquire_timeout_seconds=0.05)
    entered = threading.Event()
    release = threading.Event()

    def capture() -> None:
        with binding.critical_section("EXPORT_CAPTURE", gate_id="capture-timeout-seed-113"):
            entered.set()
            assert release.wait(timeout=1.0)

    thread = threading.Thread(target=capture, daemon=False)
    thread.start()
    assert entered.wait(timeout=1.0)
    with pytest.raises(V11EngineStepTimeout):
        view.llm_engine.step()
    release.set()
    thread.join(timeout=1.0)
    assert not thread.is_alive()
    binding.close()


@pytest.mark.parametrize(
    "operation",
    ["capture", "write", "scrub"],
)
def test_cuda_state_helpers_reject_missing_production_gate(
    operation: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(v11, "_torch", lambda: object())
    config = v11.V11PipelineConfig(
        seed=41,
        expected_device_type="cuda",
        expected_cuda_device_index=0,
        expected_gpu_uuid="GPU-fixture",
        pin_memory=True,
        asynchronous_copy=True,
    )
    with pytest.raises(V11EngineStepBindingError, match="production"):
        if operation == "capture":
            v11.capture_native_to_transport_v11(  # type: ignore[arg-type]
                (),
                None,
                page_order=(),
                branch_tables=(),
                identity={},
                config=config,
            )
        elif operation == "write":
            v11.write_and_validate_native_subset_v11(  # type: ignore[arg-type]
                None,
                (),
                None,
                destination_block_indices={},
                expected_identity={},
                verified_transport=None,
                config=config,
            )
        else:
            v11.scrub_native_pages_v11(  # type: ignore[arg-type]
                (),
                None,
                state=None,
                verified_transport=None,
                config=config,
                destination_block_indices={},
            )


def test_cuda_capture_rejects_import_operation_witness(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(v11, "_torch", lambda: object())
    view, _mutations = _view()
    binding = Vllm0230EngineStepBinding(view, acquire_timeout_seconds=1.0)
    config = v11.V11PipelineConfig(
        seed=73,
        expected_device_type="cuda",
        expected_cuda_device_index=0,
        expected_gpu_uuid="GPU-fixture",
        pin_memory=True,
        asynchronous_copy=True,
    )
    with (
        binding.critical_section("IMPORT_ADMISSION", gate_id="wrong-operation-seed-73") as gate,
        pytest.raises(V11EngineStepBindingError, match="stale, foreign, or not held"),
    ):
        v11.capture_native_to_transport_v11(  # type: ignore[arg-type]
            (),
            None,
            page_order=(),
            branch_tables=(),
            identity={},
            config=config,
            engine_step_gate=gate,
        )
    binding.close()
