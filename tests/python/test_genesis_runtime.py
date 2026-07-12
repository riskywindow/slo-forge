from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path
from threading import Event
from typing import cast

import pytest

from sloforge.genesis.evolution.runtime_evidence_runner import main as run_evolution_evidence
from sloforge.genesis.frontend import inspect_reference_package
from sloforge.genesis.policy_dsl import BytecodeProgram, Instruction, ScalarType, VariableSpec
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
    StateAllocatorConfig,
    StateAllocatorLayout,
    generate_baseline_runtime,
    load_generated_runtime,
)
from sloforge.genesis.runtime.models import RequestControl
from sloforge.genesis.synthesis import (
    cancellation_fixture_candidates,
    compiled_candidate_policy,
)

ROOT = Path(__file__).resolve().parents[2]
HYBRID = ROOT / "models" / "reference_tasks" / "hybrid_decoder"


def _generated(tmp_path: Path, *, seed: int = 73129, package: Path = HYBRID) -> Path:
    inspection = inspect_reference_package(package)
    output = tmp_path / "runtime"
    generate_baseline_runtime(package, inspection, output, seed=seed)
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
    first = load_generated_runtime(config, seed=73129, allow_untrusted_in_process=True)
    second = load_generated_runtime(config, seed=73129, allow_untrusted_in_process=True)
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
    runtime = load_generated_runtime(
        output / "runtime_config.json", seed=73129, allow_untrusted_in_process=True
    )
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
    assert deployment["direct_launch_supported"] is False
    assert deployment["trusted_launcher"] == "sloforge.genesis.sandbox.execute_sandboxed"
    assert artifacts["artifacts"]
    compile((output / "runtime.py").read_text(encoding="utf-8"), "runtime.py", "exec")
    compile(
        (output / "correctness_harness.py").read_text(encoding="utf-8"),
        "correctness_harness.py",
        "exec",
    )


def test_generated_runtime_rejects_seed_outside_verified_genome(tmp_path: Path) -> None:
    output = _generated(tmp_path, seed=73129)

    with pytest.raises(ValueError, match="seed bound to the generated runtime and genome"):
        load_generated_runtime(
            output / "runtime_config.json", seed=3, allow_untrusted_in_process=True
        )


def test_evolution_runner_separates_campaign_seed_from_runtime_seed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = _generated(tmp_path, seed=73129)
    trace = tmp_path / "trace.json"
    trace.write_text(
        json.dumps(
            {
                "requests": [
                    {
                        "request_id": "seed-separation",
                        "text": "hybrid",
                        "maximum_new_tokens": 1,
                        "seed": 17,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("SLOFORGE_GENESIS_SANDBOX_LAUNCH", "sandbox-executor-v1")

    for campaign_seed in (73129, 73130):
        result_path = tmp_path / f"observation-{campaign_seed}.json"
        assert (
            run_evolution_evidence(
                [
                    "--bundle",
                    str(output),
                    "--trace",
                    str(trace),
                    "--output",
                    str(result_path),
                    "--seed",
                    str(campaign_seed),
                ]
            )
            == 0
        )
        observation = json.loads(result_path.read_text(encoding="utf-8"))
        assert observation["seed"] == campaign_seed
        assert observation["runtime_seed"] == 73129
        assert observation["cases"][0]["request_seed"] == 17


def test_runtime_generation_recomputes_static_inspection(tmp_path: Path) -> None:
    inspection = inspect_reference_package(HYBRID)
    altered = inspection.model_copy(
        update={
            "supported_input_domain": inspection.supported_input_domain.model_copy(
                update={"maximum_generated_tokens": 1}
            )
        }
    )

    with pytest.raises(ValueError, match="independent static inspection"):
        generate_baseline_runtime(HYBRID, altered, tmp_path / "runtime", seed=73129)


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
        env={
            **os.environ,
            "PYTHONPATH": str(ROOT / "python"),
            "SLOFORGE_GENESIS_SANDBOX_LAUNCH": "sandbox-executor-v1",
        },
    )

    assert completed.returncode == 0, completed.stderr
    evidence = json.loads(completed.stdout)
    assert evidence["failures"] == []
    assert evidence["passed"] is True
    assert evidence["cases"]
    assert all(case["exact_match"] for case in evidence["cases"])


def test_generated_runtime_refuses_untrusted_direct_launch(tmp_path: Path) -> None:
    output = _generated(tmp_path)
    environment = {**os.environ, "PYTHONPATH": str(ROOT / "python")}
    environment.pop("SLOFORGE_GENESIS_SANDBOX_LAUNCH", None)
    completed = subprocess.run(
        [sys.executable, "runtime.py", "--seed", "73129"],
        cwd=output,
        check=False,
        capture_output=True,
        text=True,
        timeout=5,
        env=environment,
    )
    assert completed.returncode == 78
    assert "trusted Genesis sandbox executor" in completed.stderr


def test_generated_runtime_loader_requires_explicit_in_process_test_opt_in(
    tmp_path: Path,
) -> None:
    output = _generated(tmp_path)
    with pytest.raises(PermissionError, match="sandbox executor"):
        load_generated_runtime(output / "runtime_config.json", seed=73129)


def test_generated_subprocess_multiplexes_requests_and_cancellation(tmp_path: Path) -> None:
    package = tmp_path / "slow-reference"
    shutil.copytree(HYBRID, package)
    reference = package / "reference.py"
    source = reference.read_text(encoding="utf-8")
    marker = "    _advance(model, previous_token, state)\n"
    assert marker in source
    reference.write_text(
        source.replace(
            marker,
            "    for delay_index in range(200000):\n"
            "        _delay_value = delay_index * delay_index\n"
            "    _advance(model, previous_token, state)\n",
            1,
        ),
        encoding="utf-8",
    )
    output = _generated(tmp_path, package=package)
    process = subprocess.Popen(
        [sys.executable, "runtime.py", "--seed", "73129"],
        cwd=output,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env={
            **os.environ,
            "PYTHONPATH": str(ROOT / "python"),
            "SLOFORGE_GENESIS_SANDBOX_LAUNCH": "sandbox-executor-v1",
        },
    )
    assert process.stdin is not None
    requests = [
        {
            "operation": "submit",
            "request_id": f"multiplex-{index}",
            "text": "hybrid" * 8,
            "maximum_new_tokens": 16,
            "seed": 100 + index,
            "timeout_seconds": 3.0,
        }
        for index in range(8)
    ]
    for request in requests:
        process.stdin.write(json.dumps(request) + "\n")
    process.stdin.write(json.dumps({"operation": "cancel", "request_id": "multiplex-7"}) + "\n")
    process.stdin.write(json.dumps({"operation": "health"}) + "\n")
    process.stdin.write(json.dumps({"operation": "shutdown"}) + "\n")
    process.stdin.flush()
    stdout, stderr = process.communicate(timeout=15)

    assert process.returncode == 0, stderr
    events = [json.loads(line) for line in stdout.splitlines()]
    assert any(
        item.get("kind") == "control"
        and item.get("operation") == "cancel"
        and item.get("request_id") == "multiplex-7"
        for item in events
    )
    assert any(item.get("kind") == "control" and "body" in item for item in events)
    terminal = [
        item
        for item in events
        if item.get("request_id") == "multiplex-7"
        and item.get("kind") in {"cancelled", "completed", "error", "timed_out"}
    ]
    assert terminal and terminal[-1]["kind"] == "cancelled"


def test_runtime_detects_reference_source_tampering(tmp_path: Path) -> None:
    package = tmp_path / "package"
    shutil.copytree(HYBRID, package)
    inspection = inspect_reference_package(package)
    output = tmp_path / "runtime"
    generate_baseline_runtime(package, inspection, output, seed=5)
    reference = package / "reference.py"
    reference.write_text(reference.read_text(encoding="utf-8") + "\n# tampered\n", encoding="utf-8")

    with pytest.raises(ValueError, match="changed after runtime generation"):
        load_generated_runtime(
            output / "runtime_config.json", seed=5, allow_untrusted_in_process=True
        )


@pytest.mark.parametrize(
    "runtime_request",
    [
        RuntimeRequest("bad-timeout", (1,), 1, 7, math.inf),
        RuntimeRequest("bad-timeout", (1,), 1, 7, math.nan),
        RuntimeRequest("bad-count", (1,), True, 7, 1.0),
        RuntimeRequest("bad-seed", (1,), 1, True, 1.0),
        RuntimeRequest("bad-token", (True,), 1, 7, 1.0),
        RuntimeRequest("bad-batching", (1,), 1, 7, 1.0, batching_eligible=1),
    ],
)
def test_runtime_rejects_unbounded_or_coerced_request_fields(
    runtime_request: RuntimeRequest,
) -> None:
    with pytest.raises(ValueError):
        runtime_request.validate(maximum_prompt_tokens=8, maximum_generated_tokens=2)


def test_runtime_limits_reject_non_finite_timeouts_and_non_integer_bounds() -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        RuntimeLimits(worker_poll_seconds=math.inf)
    with pytest.raises(ValueError, match="integer limits"):
        RuntimeLimits(maximum_queue_depth=1.5)  # type: ignore[arg-type]


def test_paged_allocator_rejects_capacity_smaller_than_rounded_reservation() -> None:
    with pytest.raises(ValueError, match="rounded per-request state reservation"):
        StateAllocatorConfig(
            layout=StateAllocatorLayout.PAGED,
            page_bytes=64,
            maximum_bytes_per_request=65,
            maximum_total_bytes=65,
        )
    accepted = StateAllocatorConfig(
        layout=StateAllocatorLayout.PAGED,
        page_bytes=64,
        maximum_bytes_per_request=65,
        maximum_total_bytes=128,
    )
    assert accepted.maximum_total_bytes == 128


@pytest.mark.parametrize("timeout_seconds", [0.0, math.inf, math.nan])
def test_runtime_shutdown_rejects_unbounded_or_zero_timeout(timeout_seconds: float) -> None:
    runtime = BaselineStreamingRuntime(
        cast(ReferenceRuntimeAdapter, object()),
        limits=RuntimeLimits(),
        runtime_seed=1,
    )
    with pytest.raises(ValueError, match="shutdown timeout must be finite and positive"):
        runtime.shutdown(timeout_seconds)


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


class _EmissionRaceAdapter(_BlockingAdapter):
    def __init__(self) -> None:
        super().__init__()
        self.detokenize_entered = Event()
        self.release_detokenize = Event()

    def prefill(self, prompt_tokens: tuple[int, ...], state: object, seed: int) -> object:
        return state

    def detokenize(self, token_id: int) -> str:
        self.detokenize_entered.set()
        if not self.release_detokenize.wait(2.0):
            raise TimeoutError("test detokenize release was not signalled")
        return str(token_id)


class _RecordingBlockingAdapter(_BlockingAdapter):
    def __init__(self) -> None:
        super().__init__()
        self.allocation_order: list[str] = []

    def allocate_state(self, request_id: str, prompt_tokens: tuple[int, ...], seed: int) -> object:
        self.allocation_order.append(request_id)
        return super().allocate_state(request_id, prompt_tokens, seed)


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


def test_cancellation_racing_final_emit_check_cannot_publish_a_token() -> None:
    adapter = _EmissionRaceAdapter()
    runtime = _blocking_runtime(adapter)
    runtime.start()
    handle = runtime.submit(RuntimeRequest("emit-race", (1,), 1, 7, 1.0))
    assert adapter.detokenize_entered.wait(1.0)

    handle.cancel()
    adapter.release_detokenize.set()
    events = list(handle.events(2.0))
    runtime.shutdown()

    assert [event.kind for event in events] == [EventKind.CANCELLED]
    assert handle.lifecycle is RequestLifecycle.CANCELLED
    assert runtime.metrics()["emitted_tokens"] == 0


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


def test_runtime_rejects_policy_inputs_it_cannot_supply_before_start() -> None:
    policy = BytecodeProgram(
        name="unsupported-priority",
        inputs=(VariableSpec("priority", ScalarType.INT, 0, 7),),
        output=VariableSpec("batch", ScalarType.INT, 1, 1),
        instructions=(Instruction("literal", 1),),
        maximum_operations=1,
    )
    runtime = BaselineStreamingRuntime(
        cast(ReferenceRuntimeAdapter, _BlockingAdapter()),
        limits=RuntimeLimits(maximum_queue_depth=1, maximum_batch_size=1),
        runtime_seed=11,
        policy=policy,
    )

    with pytest.raises(RuntimeError, match="unsupported inputs"):
        runtime.start()
    assert runtime.health()["accepting_requests"] is False


def test_policy_execution_failure_terminalizes_request_without_killing_worker() -> None:
    policy = BytecodeProgram(
        name="divide-by-zero",
        inputs=(),
        output=VariableSpec("batch", ScalarType.INT, 1, 4),
        instructions=(
            Instruction("literal", 1),
            Instruction("literal", 0),
            Instruction("floor_div"),
        ),
        maximum_operations=3,
    )
    runtime = BaselineStreamingRuntime(
        cast(ReferenceRuntimeAdapter, _BlockingAdapter()),
        limits=RuntimeLimits(
            maximum_queue_depth=2,
            maximum_batch_size=1,
            maximum_prompt_tokens=8,
            maximum_generated_tokens=2,
            maximum_output_events_per_request=3,
            worker_poll_seconds=0.01,
        ),
        runtime_seed=11,
        policy=policy,
    )
    runtime.start()
    try:
        first = runtime.submit(RuntimeRequest("policy-error-1", (1,), 1, 1, 1.0))
        assert [event.kind for event in first.events(1.0)] == [EventKind.ERROR]
        assert first.wait(0.1)
        assert first.lifecycle is RequestLifecycle.FAILED
        assert runtime.health()["accepting_requests"] is True

        second = runtime.submit(RuntimeRequest("policy-error-2", (1,), 1, 2, 1.0))
        assert [event.kind for event in second.events(1.0)] == [EventKind.ERROR]
    finally:
        runtime.shutdown()

    assert runtime.metrics()["failed"] == 2
    assert runtime.health()["queue_depth"] == 0


def test_synthesized_deadline_policy_changes_request_order_and_batch_behavior() -> None:
    policy = BytecodeProgram(
        name="deadline_cancel_batch",
        inputs=(),
        output=VariableSpec("batch", ScalarType.INT, 1, 4),
        instructions=(Instruction("literal", 4),),
        maximum_operations=1,
    )
    adapter = _RecordingBlockingAdapter()
    runtime = BaselineStreamingRuntime(
        cast(ReferenceRuntimeAdapter, adapter),
        limits=RuntimeLimits(
            maximum_queue_depth=4,
            maximum_batch_size=4,
            maximum_prompt_tokens=8,
            maximum_generated_tokens=2,
            maximum_output_events_per_request=3,
            batch_wait_seconds=0.05,
            worker_poll_seconds=0.01,
            shutdown_timeout_seconds=2.0,
        ),
        runtime_seed=11,
        policy=policy,
    )
    runtime.start()
    long_deadline = runtime.submit(RuntimeRequest("long", (1,), 1, 1, 2.0))
    short_deadline = runtime.submit(RuntimeRequest("short", (1,), 1, 2, 0.5))
    assert adapter.prefill_entered.wait(1.0)
    assert adapter.allocation_order == ["short"]
    adapter.release_prefill.set()
    list(long_deadline.events(2.0))
    list(short_deadline.events(2.0))
    runtime.shutdown()

    metrics = runtime.metrics()
    assert metrics["maximum_observed_batch"] == 2
    assert metrics["deadline_reorders"] == 1


def test_runtime_executes_cegis_cancellation_policy_with_identical_zero_semantics() -> None:
    unsafe, _repeat, corrected = cancellation_fixture_candidates(73129)
    _source, unsafe_policy, _payload = compiled_candidate_policy(unsafe)
    _source, corrected_policy, _payload = compiled_candidate_policy(corrected)
    control = RequestControl(RuntimeRequest("cancelled-before-schedule", (1,), 1, 7, 1.0))
    control.cancellation.set()
    limits = RuntimeLimits(maximum_queue_depth=4, maximum_batch_size=4)
    unsafe_runtime = BaselineStreamingRuntime(
        cast(ReferenceRuntimeAdapter, object()),
        limits=limits,
        runtime_seed=1,
        policy=unsafe_policy,
    )
    corrected_runtime = BaselineStreamingRuntime(
        cast(ReferenceRuntimeAdapter, object()),
        limits=limits,
        runtime_seed=1,
        policy=corrected_policy,
    )
    assert unsafe_runtime._batch_limit(control) == 1
    assert corrected_runtime._batch_limit(control) == 0


def test_wait_cannot_observe_terminal_state_before_terminal_event_publication() -> None:
    adapter = _EmissionRaceAdapter()
    runtime = _blocking_runtime(adapter)
    adapter.release_detokenize.set()
    runtime.start()
    try:
        handle = runtime.submit(RuntimeRequest("terminal-order", (1,), 1, 7, 1.0))
        assert handle.wait(1.0)
        events = list(handle.events(0.1))
        assert events[-1].kind is EventKind.COMPLETED
        assert handle.lifecycle is RequestLifecycle.COMPLETED
    finally:
        runtime.shutdown()
