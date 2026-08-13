from __future__ import annotations

import importlib.util
import inspect
import json
import sys
import threading
import time
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]
WORKER_PATH = ROOT / "experiments/branchfabric/gpu_reclamation_worker.py"
SPEC = importlib.util.spec_from_file_location("gpu_reclamation_worker_integrated", WORKER_PATH)
assert SPEC is not None and SPEC.loader is not None
WORKER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = WORKER
SPEC.loader.exec_module(WORKER)


def test_integrated_worker_emits_model_load_clock_before_engine_creation() -> None:
    source = inspect.getsource(WORKER.main)

    integrated_mode = source.index(
        'integrated = config.get("execution_mode") == "integrated-calibration-v10"'
    )
    clock = source.index("model_load_started_ns = time.monotonic_ns()")
    signal = source.index('f"{args.role}.engine-started.json"')
    engine_creation = source.index("adapter = _create_adapter(")

    assert integrated_mode < clock < signal < engine_creation
    assert '"schema_version": "sloforge.branchfabric.retained-engine-start/v1"' in source


class _FakeEngine:
    def __init__(self) -> None:
        self.prompts: list[tuple[str, dict[str, Any]]] = []
        self.marker = "delegated"

    def add_request(self, request_id: str, prompt: dict[str, Any], *_args: Any) -> str:
        self.prompts.append((request_id, prompt))
        return request_id


def _wait_for(path: Path) -> None:
    deadline = time.monotonic() + 2.0
    while not path.is_file():
        if time.monotonic() >= deadline:
            raise TimeoutError(path)
        time.sleep(0.002)


def test_cache_salt_engine_is_unique_deterministic_and_delegates() -> None:
    engine = _FakeEngine()
    wrapped = WORKER._CacheSaltEngine(engine, namespace="attempt:gpu0")
    original = {"prompt_token_ids": [1, 2, 3]}

    wrapped.add_request("request-0", original, object())
    wrapped.add_request("request-1", original, object())

    assert original == {"prompt_token_ids": [1, 2, 3]}
    salts = [prompt["cache_salt"] for _request_id, prompt in engine.prompts]
    assert len(set(salts)) == 2
    assert all(len(salt) == 64 for salt in salts)
    assert wrapped.marker == "delegated"
    evidence = wrapped.evidence()
    assert evidence["request_count"] == evidence["unique_salt_count"] == 2
    assert evidence["passed"] is True
    with pytest.raises(RuntimeError, match="identity was reused"):
        wrapped.add_request("request-0", original, object())


def test_measured_transaction_compilation_observation_is_fail_closed(tmp_path: Path) -> None:
    class CleanCapture:
        def __init__(self) -> None:
            self.events: list[str] = []

        def __enter__(self):  # type: ignore[no-untyped-def]
            return self

        def __exit__(self, *_args: object) -> None:
            return None

    clean_root = tmp_path / "clean"
    result = WORKER._run_measured_v10_transaction(
        role="serving",
        work_root=clean_root,
        operation=lambda: {"transaction": "complete"},
        compilation_capture_factory=CleanCapture,
    )
    observation = result["measured_transaction_compilation_observation"]
    assert observation["passed"] is True
    assert observation["events"] == ()
    persisted = json.loads(
        (clean_root / "telemetry/measured-transaction-compilation.json").read_text()
    )
    assert persisted["events"] == []
    assert persisted["no_deferred_compilation_event"] is True

    class CompilingCapture(CleanCapture):
        def __enter__(self):  # type: ignore[no-untyped-def]
            self.events.append("Compiling a graph after transaction start")
            return self

    compiling_root = tmp_path / "compiling"
    with pytest.raises(RuntimeError, match="observed deferred compilation"):
        WORKER._run_measured_v10_transaction(
            role="rollout",
            work_root=compiling_root,
            operation=lambda: {"transaction": "invalid"},
            compilation_capture_factory=CompilingCapture,
        )
    failed = json.loads(
        (compiling_root / "telemetry/measured-transaction-compilation.json").read_text()
    )
    assert failed["passed"] is False
    assert failed["events"] == ["Compiling a graph after transaction start"]


def test_integrated_protocol_paths_match_controller_contract(tmp_path: Path) -> None:
    root = tmp_path / "barriers"
    assert WORKER._transaction_ready_path(root, "serving") == (
        root / "serving.transaction-ready.json"
    )
    assert WORKER._transaction_ready_path(root, "rollout") == (
        root / "rollout.transaction-ready.json"
    )
    with pytest.raises(ValueError, match="role"):
        WORKER._transaction_ready_path(root, "gpu0")


def test_integrated_phase_retains_one_engine_across_probe_and_final_reset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine = _FakeEngine()

    class Manager:
        def reset_prefix_cache(self) -> bool:
            return True

    adapter = SimpleNamespace(
        _view=SimpleNamespace(llm_engine=engine, manager=Manager()),
    )
    calibration = ModuleType("gpu_capacity_calibration_worker")

    class CompilationCapture:
        def __init__(self) -> None:
            self.events: list[str] = []

        def __enter__(self) -> CompilationCapture:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

    def collect(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        return {
            "schema_version": "sloforge.branchfabric.engine-readiness-evidence/v1",
            "passed": True,
        }

    def execute(salted: Any, **kwargs: Any) -> dict[str, Any]:
        salted.add_request(f"{kwargs['device']}-a", {"prompt_token_ids": [1]}, object())
        salted.add_request(f"{kwargs['device']}-b", {"prompt_token_ids": [1]}, object())
        return {"probe_id": kwargs["plan_payload"]["probe_id"]}

    def reset(_adapter: Any, *, probe_id: str) -> dict[str, Any]:
        state = {
            "request_count": 0,
            "running_requests": 0,
            "waiting_requests": 0,
            "skipped_waiting_requests": 0,
            "queue_depth": 0,
        }
        return {
            "schema_version": "sloforge.branchfabric.capacity-request-reset/v1",
            "probe_id": probe_id,
            "engine_reloaded": False,
            "state_before": state,
            "state_after": state,
            "passed": True,
        }

    calibration._collect_engine_readiness = collect  # type: ignore[attr-defined]
    calibration._CompilationLogCapture = CompilationCapture  # type: ignore[attr-defined]
    calibration._execute_probe = execute  # type: ignore[attr-defined]
    calibration._reset_probe_state = reset  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "gpu_capacity_calibration_worker", calibration)
    monkeypatch.setattr(WORKER.importlib.metadata, "version", lambda _name: "pinned")
    fake_torch = SimpleNamespace(
        version=SimpleNamespace(cuda="13.0"),
        cuda=SimpleNamespace(get_device_name=lambda _index: "NVIDIA A100-SXM4-80GB"),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    barrier_root = tmp_path / "barriers"
    capacity_root = barrier_root / "capacity"
    capacity_root.mkdir(parents=True)
    base_config = {
        "schema_version": "sloforge.branchfabric.experiment-004-modal-config/v1",
        "attempt_id": "integrated-test-attempt",
        "seed": 41,
        "gpu_memory_utilization": 0.7,
        "serving_prompt_tokens": 64,
        "serving_output_tokens": 64,
        "serving_methodology": "v10-global-capacity",
        "authorization_artifact_hash": "1" * 64,
        "probe_timeout_seconds": 2.0,
        "tail_drain_seconds": 0.1,
        "producer_queue_capacity": 8,
        "controller_timeout_seconds": 2.0,
        "maximum_wall_seconds": 2.0,
    }
    effective = dict(base_config)
    selected_hash = "a" * 64
    result: list[Any] = []
    failure: list[BaseException] = []

    def run() -> None:
        try:
            result.append(
                WORKER._run_integrated_calibration_phase(
                    adapter=adapter,
                    config=base_config,
                    inputs={"prefix_token_ids": list(range(128))},
                    role="serving",
                    physical_gpu_uuid="GPU-test",
                    barrier_root=barrier_root,
                    model_load_started_ns=100,
                    model_ready_ns=200,
                )
            )
        except BaseException as error:  # pragma: no cover - surfaced below
            failure.append(error)

    thread = threading.Thread(target=run)
    thread.start()
    _wait_for(barrier_root / "serving.ready.json")
    WORKER._write_new(
        capacity_root / "probe-00.command.json",
        {
            "schema_version": "sloforge.branchfabric.capacity-probe-command/v1",
            "plan": {"probe_id": "one-gpu-00"},
        },
    )
    _wait_for(capacity_root / "probe-00.gpu0.reset.json")
    WORKER._write_new(
        capacity_root / "final-reset.command.json",
        {
            "schema_version": "sloforge.branchfabric.integrated-final-reset-command/v1",
            "selected_load_sha256": selected_hash,
            "authorization_artifact_hash": "1" * 64,
            "issued_at_monotonic_ns": time.monotonic_ns(),
        },
    )
    _wait_for(capacity_root / "final-reset.gpu0.json")
    WORKER._write_new(
        barrier_root / "transaction.command.json",
        {
            "schema_version": "sloforge.branchfabric.integrated-transaction-command/v1",
            "effective_config": effective,
            "selection_sha256": selected_hash,
            "authorization_artifact_hash": "1" * 64,
            "sanity_result_sha256": "2" * 64,
            "issued_at_monotonic_ns": time.monotonic_ns(),
        },
    )
    thread.join(timeout=2.0)

    assert not thread.is_alive()
    assert failure == []
    returned_config, salted, handoff = result[0]
    assert returned_config == effective
    assert salted.underlying_engine is engine
    assert handoff["engine_reloaded"] is False
    assert handoff["engine_object_identity"] == id(engine)
    assert handoff["pid"] == WORKER.os.getpid()
    assert handoff["device"] == "gpu0"
    assert handoff["physical_gpu_uuid"] == "GPU-test"
    assert handoff["cache_salt_evidence"]["unique_salt_count"] == 2
    reset_payload = json.loads((capacity_root / "final-reset.gpu0.json").read_text())
    assert reset_payload["engine_object_identity"] == id(engine)
    assert reset_payload["engine_reloaded"] is False
    assert reset_payload["device"] == "gpu0"
    assert reset_payload["physical_gpu_uuid"] == "GPU-test"


def test_transaction_command_fails_closed_on_missing_pre_gpu_evidence() -> None:
    base = {
        "schema_version": "sloforge.branchfabric.experiment-004-modal-config/v1",
        "attempt_id": "attempt-004",
        "seed": 1,
        "gpu_memory_utilization": 0.7,
        "serving_prompt_tokens": 64,
        "serving_output_tokens": 64,
        "serving_methodology": "v10-global-capacity",
        "authorization_artifact_hash": "1" * 64,
    }
    command = {
        "schema_version": "sloforge.branchfabric.integrated-transaction-command/v1",
        "effective_config": dict(base),
        "selection_sha256": "a" * 64,
        "issued_at_monotonic_ns": 1,
    }
    with pytest.raises(ValueError, match="authorization artifact hash"):
        WORKER._validate_integrated_transaction_command(
            base_config=base,
            command=command,
            selected_load_sha256="a" * 64,
        )
