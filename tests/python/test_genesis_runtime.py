from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from threading import Event
from typing import cast

import pytest

from sloforge.genesis.frontend import inspect_reference_package
from sloforge.genesis.runtime import (
    BaselineStreamingRuntime,
    DecodeResult,
    EventKind,
    GeneratedRuntimeApplication,
    QueueSaturatedError,
    ReferenceRuntimeAdapter,
    RequestLifecycle,
    RuntimeLimits,
    RuntimeRequest,
    generate_baseline_runtime,
    load_generated_runtime,
)

ROOT = Path(__file__).resolve().parents[2]
HYBRID = ROOT / "models" / "reference_tasks" / "hybrid_decoder"


def _generated(tmp_path: Path, *, seed: int = 73129) -> Path:
    inspection = inspect_reference_package(HYBRID)
    output = tmp_path / "runtime"
    generate_baseline_runtime(HYBRID, inspection, output, seed=seed)
    return output


def _tokens(runtime: BaselineStreamingRuntime, request_id: str) -> list[int]:
    handle = runtime.submit_text(
        request_id=request_id,
        text="hybrid",
        maximum_new_tokens=5,
        seed=73,
        timeout_seconds=3.0,
    )
    events = list(handle.events(3.0))
    assert events[-1].kind == EventKind.COMPLETED
    assert [event.sequence for event in events] == list(range(6))
    return [event.token_id for event in events if event.token_id is not None]


def test_generated_runtime_streams_deterministically_and_releases_state(tmp_path: Path) -> None:
    output = _generated(tmp_path)
    config = output / "runtime_config.json"
    first = load_generated_runtime(config, seed=73129)
    second = load_generated_runtime(config, seed=73129)
    first.start()
    second.start()
    try:
        first_tokens = _tokens(first, "same-request")
        second_tokens = _tokens(second, "same-request")
    finally:
        first.shutdown()
        second.shutdown()

    assert first_tokens == second_tokens == [2, 0, 22, 11, 1]
    assert first.metrics()["state_allocations"] == first.metrics()["state_releases"] == 1
    assert first.health()["owned_state_count"] == 0
    assert first.health()["status"] == "stopped"


def test_generated_application_exposes_health_metrics_and_bounded_manifest(tmp_path: Path) -> None:
    output = _generated(tmp_path)
    runtime = load_generated_runtime(output / "runtime_config.json", seed=3)
    application = GeneratedRuntimeApplication(runtime)
    application.start()
    try:
        status, health = application.handle_get("/healthz")
        metrics_status, metrics = application.handle_get("/metrics")
        missing_status, _ = application.handle_get("/missing")
    finally:
        application.shutdown()

    deployment = json.loads((output / "deployment_manifest.json").read_text(encoding="utf-8"))
    artifacts = json.loads((output / "artifact_manifest.json").read_text(encoding="utf-8"))
    assert status == metrics_status == 200
    assert health["queue_capacity"] == 32
    assert metrics["submitted"] == 0
    assert missing_status == 404
    assert deployment["network_enabled"] is False
    assert deployment["request_timeout_required"] is True
    assert artifacts["artifacts"]
    compile((output / "runtime.py").read_text(encoding="utf-8"), "runtime.py", "exec")
    compile(
        (output / "correctness_harness.py").read_text(encoding="utf-8"),
        "correctness_harness.py",
        "exec",
    )


def test_generated_correctness_harness_executes_reference_fixture(tmp_path: Path) -> None:
    output = _generated(tmp_path)
    completed = subprocess.run(
        [
            sys.executable,
            "correctness_harness.py",
            "--samples",
            str(HYBRID / "final_samples.jsonl"),
            "--seed",
            "73129",
            "--timeout-seconds",
            "3",
        ],
        cwd=output,
        check=False,
        capture_output=True,
        text=True,
        timeout=15,
        env={**os.environ, "PYTHONPATH": str(ROOT / "python")},
    )

    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout) == {"failures": [], "passed": True}


def test_runtime_detects_reference_source_tampering(tmp_path: Path) -> None:
    package = tmp_path / "package"
    import shutil

    shutil.copytree(HYBRID, package)
    inspection = inspect_reference_package(package)
    output = tmp_path / "runtime"
    generate_baseline_runtime(package, inspection, output, seed=5)
    reference = package / "reference.py"
    reference.write_text(reference.read_text(encoding="utf-8") + "\n# tampered\n", encoding="utf-8")

    with pytest.raises(ValueError, match="changed after runtime generation"):
        load_generated_runtime(output / "runtime_config.json", seed=5)


class _BlockingAdapter:
    def __init__(self) -> None:
        self.prefill_entered = Event()
        self.release_prefill = Event()

    def tokenize(self, text: str) -> tuple[int, ...]:
        return tuple(ord(character) % 8 for character in text)

    def detokenize(self, token_id: int) -> str:
        return str(token_id)

    def allocate_state(self, request_id: str, prompt_tokens: tuple[int, ...], seed: int) -> object:
        return {"request_id": request_id, "tokens": prompt_tokens, "seed": seed}

    def prefill(self, prompt_tokens: tuple[int, ...], state: object, seed: int) -> object:
        self.prefill_entered.set()
        if not self.release_prefill.wait(2.0):
            raise TimeoutError("test prefill release was not signalled")
        return state

    def decode_step(
        self, previous_token: int, state: object, position: int, seed: int
    ) -> DecodeResult:
        return DecodeResult(logits=(0.0, 1.0), state=state)

    def sample(self, logits: tuple[float, ...], seed: int) -> int:
        return 1


def _blocking_runtime(adapter: _BlockingAdapter) -> BaselineStreamingRuntime:
    return BaselineStreamingRuntime(
        cast(ReferenceRuntimeAdapter, adapter),
        limits=RuntimeLimits(
            maximum_queue_depth=1,
            maximum_batch_size=1,
            maximum_prompt_tokens=8,
            maximum_generated_tokens=2,
            maximum_output_events_per_request=3,
            batch_wait_seconds=0.001,
            queue_operation_timeout_seconds=0.02,
            worker_poll_seconds=0.01,
            shutdown_timeout_seconds=2.0,
        ),
        runtime_seed=11,
    )


def test_cancellation_before_commit_releases_state_and_shuts_down() -> None:
    adapter = _BlockingAdapter()
    runtime = _blocking_runtime(adapter)
    runtime.start()
    first = runtime.submit(RuntimeRequest("cancel", (1,), 2, 7, 1.0))
    assert adapter.prefill_entered.wait(1.0)
    first.cancel()
    adapter.release_prefill.set()
    events = list(first.events(2.0))
    runtime.shutdown()

    assert [event.kind for event in events] == [EventKind.CANCELLED]
    assert first.lifecycle == RequestLifecycle.CANCELLED
    assert runtime.metrics()["emitted_tokens"] == 0
    assert runtime.metrics()["state_allocations"] == runtime.metrics()["state_releases"] == 1


def test_bounded_admission_queue_rejects_saturation() -> None:
    adapter = _BlockingAdapter()
    runtime = _blocking_runtime(adapter)
    runtime.start()
    first = runtime.submit(RuntimeRequest("active", (1,), 1, 1, 1.0))
    assert adapter.prefill_entered.wait(1.0)
    second = runtime.submit(RuntimeRequest("queued", (1,), 1, 2, 1.0))
    with pytest.raises(QueueSaturatedError):
        runtime.submit(RuntimeRequest("rejected", (1,), 1, 3, 1.0))
    first.cancel()
    second.cancel()
    adapter.release_prefill.set()
    list(first.events(2.0))
    list(second.events(2.0))
    runtime.shutdown()

    assert runtime.metrics()["rejected_queue_full"] == 1
    assert runtime.health()["queue_depth"] == 0
