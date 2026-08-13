#!/usr/bin/env python3
"""Persistent single-GPU worker for Experiment 004 capacity calibration.

The process loads exactly one vLLM engine, then executes immutable probe plans
issued by the CUDA-clean controller.  It never chooses rates or routing.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import logging
import math
import os
import queue
import threading
import time
import traceback
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_READINESS_BATCH_SIZES = (1, 2, 4, 8, 16)
_HBM_STABILITY_TOLERANCE_BYTES = 16 * 1024 * 1024
_COMPILATION_LOG_MARKERS = (
    "jit compilation during inference",
    "compiling a graph",
    "capturing cuda graphs",
)


class _CompilationLogCapture(logging.Handler):
    """Bounded observation of deferred compilation during readiness work."""

    def __init__(self) -> None:
        super().__init__(level=logging.INFO)
        self.events: list[str] = []
        self._attached_loggers: list[logging.Logger] = []
        self._seen_records: set[int] = set()

    def emit(self, record: logging.LogRecord) -> None:
        record_identity = id(record)
        if record_identity in self._seen_records:
            return
        self._seen_records.add(record_identity)
        message = record.getMessage()
        if (
            any(marker in message.lower() for marker in _COMPILATION_LOG_MARKERS)
            and len(self.events) < 128
        ):
            self.events.append(message[:512])

    def __enter__(self) -> _CompilationLogCapture:
        candidates = [
            logging.getLogger(),
            *(
                logger
                for logger in logging.root.manager.loggerDict.values()
                if isinstance(logger, logging.Logger)
            ),
        ]
        for logger in candidates:
            if logger is logging.getLogger() or not logger.propagate:
                logger.addHandler(self)
                self._attached_loggers.append(logger)
        return self

    def __exit__(self, *_args: object) -> None:
        for logger in self._attached_loggers:
            logger.removeHandler(self)
        self._attached_loggers.clear()


def _canonical_bytes(value: Any) -> bytes:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n"
    ).encode()


def _write_new(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(_canonical_bytes(value))
        handle.flush()
        os.fsync(handle.fileno())


def _wait_for(path: Path, *, deadline_ns: int) -> dict[str, Any]:
    while not path.is_file():
        if time.monotonic_ns() >= deadline_ns:
            raise TimeoutError(f"timed out waiting for {path.name}")
        time.sleep(0.02)
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain a JSON object")
    return value


def _sleep_until(timestamp_ns: int) -> None:
    while True:
        remaining = timestamp_ns - time.monotonic_ns()
        if remaining <= 0:
            return
        time.sleep(min(remaining / 1e9, 0.02))


def _bounded_length(value: object, *, field: str) -> int:
    try:
        length = len(value)  # type: ignore[arg-type]
    except (TypeError, AttributeError) as error:
        raise RuntimeError(f"vLLM readiness cannot observe scheduler {field}") from error
    if isinstance(length, bool) or length < 0:
        raise RuntimeError(f"vLLM scheduler {field} length is invalid")
    return int(length)


def _runtime_queue_state(adapter: Any) -> dict[str, int]:
    scheduler = adapter._view.scheduler
    requests = getattr(scheduler, "requests", None)
    running = getattr(scheduler, "running", None)
    waiting = getattr(scheduler, "waiting", None)
    skipped_waiting = getattr(scheduler, "skipped_waiting", ())
    if requests is None or running is None or waiting is None:
        raise RuntimeError("vLLM readiness lacks scheduler request/queue state")
    state = {
        "request_count": _bounded_length(requests, field="requests"),
        "running_requests": _bounded_length(running, field="running"),
        "waiting_requests": _bounded_length(waiting, field="waiting"),
        "skipped_waiting_requests": _bounded_length(skipped_waiting, field="skipped_waiting"),
    }
    state["queue_depth"] = state["waiting_requests"] + state["skipped_waiting_requests"]
    return state


def _runtime_health(adapter: Any) -> tuple[bool, str]:
    candidates = (
        ("llm_engine", adapter._view.llm_engine),
        ("engine_core_client", adapter._view.engine_core_client),
        ("engine_core", adapter._view.engine_core),
    )
    for name, owner in candidates:
        checker = getattr(owner, "check_health", None)
        if not callable(checker):
            continue
        result = checker()
        if result is False:
            return False, f"{name}.check_health"
        if isinstance(result, dict):
            return result.get("status") in {"healthy", "ok"}, f"{name}.check_health"
        return True, f"{name}.check_health"

    # vLLM 0.23.0's supported synchronous V1 configuration uses an in-process
    # EngineCore and exposes no ``check_health`` method.  In that exact shape,
    # a successful probe is combined by the caller with zero scheduler state;
    # here, additionally require the public unfinished-request report to be
    # false and revalidate the adapter's InprocClient/Core object identity.
    view = adapter._view
    engine = view.llm_engine
    client = view.engine_core_client
    core = view.engine_core
    has_unfinished = getattr(engine, "has_unfinished_requests", None)
    client_type = type(client)
    core_type = type(core)
    if (
        not callable(has_unfinished)
        or client_type.__name__ != "InprocClient"
        or not client_type.__module__.startswith("vllm.v1.")
        or core_type.__name__ != "EngineCore"
        or not core_type.__module__.startswith("vllm.v1.")
        or getattr(engine, "engine_core", None) is not client
        or getattr(client, "engine_core", None) is not core
    ):
        return False, "vllm-0.23.inproc-structural-health-unavailable"
    try:
        unfinished = has_unfinished()
    except BaseException:
        return False, "vllm-0.23.llm_engine.has_unfinished_requests-failed"
    if unfinished is not False:
        return False, "vllm-0.23.llm_engine.has_unfinished_requests-nonidle"
    return True, "vllm-0.23.inproc-structure+has_unfinished_requests"


def _runtime_metrics(adapter: Any) -> tuple[bool, str]:
    engine = adapter._view.llm_engine
    # vLLM 0.23.0's V1 LLMEngine stores ``log_stats`` as its enablement
    # boolean and exposes metrics through ``get_metrics()``.  ``do_log_stats``
    # is a method, not the historical boolean, and ``stat_loggers`` is no
    # longer an engine attribute.  Keep this intentionally version-scoped so
    # API drift fails readiness instead of silently claiming telemetry.
    if getattr(engine, "log_stats", None) is not True:
        return False, "vllm-0.23.llm_engine.log_stats-disabled-or-unavailable"
    if getattr(engine, "logger_manager", None) is None:
        return False, "vllm-0.23.llm_engine.logger_manager-unavailable"
    get_metrics = getattr(engine, "get_metrics", None)
    if not callable(get_metrics):
        return False, "vllm-0.23.llm_engine.get_metrics-unavailable"
    metrics = get_metrics()
    if not isinstance(metrics, (list, tuple)):
        return False, "vllm-0.23.llm_engine.get_metrics-invalid-result"
    do_log_stats = getattr(engine, "do_log_stats", None)
    if do_log_stats is not None:
        if not callable(do_log_stats):
            return False, "vllm-0.23.llm_engine.do_log_stats-invalid"
        do_log_stats()
    return True, "vllm-0.23.llm_engine.get_metrics+logger_manager"


def _cudagraph_capture_sizes(adapter: Any) -> tuple[int, ...]:
    compilation = getattr(adapter._view.vllm_config, "compilation_config", None)
    values = getattr(compilation, "cudagraph_capture_sizes", None)
    if values is None:
        raise RuntimeError("vLLM readiness cannot inspect CUDA graph capture sizes")
    sizes = tuple(sorted({int(item) for item in values}))
    if not sizes or any(item <= 0 for item in sizes):
        raise RuntimeError("vLLM readiness observed invalid CUDA graph capture sizes")
    return sizes


def _rotated_prefix(prefix: tuple[int, ...], sequence: int) -> tuple[int, ...]:
    if not prefix:
        raise ValueError("readiness prefix must not be empty")
    offset = sequence % len(prefix)
    return prefix[offset:] + prefix[:offset]


def _calibration_prompt(prefix: tuple[int, ...], sequence: int) -> tuple[int, ...]:
    """Build a deterministic request-unique vLLM prefix-cache hash chain."""

    return _rotated_prefix(prefix, sequence)


def _readiness_sampling_params(*, output_tokens: int, seed: int) -> Any:
    from gpu_reclamation_worker import _sampling_params

    return _sampling_params(max_tokens=output_tokens, seed=seed)


def _run_readiness_batch(
    engine: Any,
    *,
    device: str,
    batch_size: int,
    request_offset: int,
    prefix: tuple[int, ...],
    sampling_params: Any,
    timeout_seconds: float,
) -> dict[str, Any]:
    if batch_size <= 0:
        raise ValueError("readiness batch size must be positive")
    started_ns = time.monotonic_ns()
    deadline_ns = started_ns + round(timeout_seconds * 1e9)
    request_ids = tuple(
        f"readiness-{device}-{request_offset + index:05d}" for index in range(batch_size)
    )
    first_token_ns: dict[str, int] = {}
    output_lengths: dict[str, int] = {request_id: 0 for request_id in request_ids}
    active = set(request_ids)
    try:
        for index, request_id in enumerate(request_ids):
            engine.add_request(
                request_id,
                {"prompt_token_ids": list(_rotated_prefix(prefix, request_offset + index))},
                sampling_params,
            )
        while active:
            if time.monotonic_ns() >= deadline_ns:
                raise TimeoutError("representative readiness batch did not complete")
            outputs = engine.step()
            observed_ns = time.monotonic_ns()
            progressed = False
            for output in outputs:
                request_id = str(getattr(output, "request_id", ""))
                if request_id not in active:
                    continue
                candidates = getattr(output, "outputs", ())
                token_ids = (
                    tuple(int(item) for item in getattr(candidates[0], "token_ids", ()))
                    if candidates
                    else ()
                )
                if len(token_ids) > output_lengths[request_id]:
                    progressed = True
                    output_lengths[request_id] = len(token_ids)
                    first_token_ns.setdefault(request_id, observed_ns)
                if bool(getattr(output, "finished", False)):
                    active.remove(request_id)
            if not progressed and active:
                time.sleep(0.001)
    except BaseException:
        if active:
            engine.abort_request(sorted(active))
        raise
    completed_ns = time.monotonic_ns()
    return {
        "batch_size": batch_size,
        "request_ids": request_ids,
        "started_ns": started_ns,
        "first_token_ns": min(first_token_ns.values()),
        "completed_ns": completed_ns,
        "output_lengths": tuple(output_lengths[request_id] for request_id in request_ids),
        "ttft_seconds": (min(first_token_ns.values()) - started_ns) / 1e9,
        "request_throughput_rps": batch_size / ((completed_ns - started_ns) / 1e9),
        "output_token_throughput_tps": sum(output_lengths.values())
        / ((completed_ns - started_ns) / 1e9),
    }


def _memory_sample(adapter: Any) -> dict[str, int | None]:
    state = adapter.inspect_gpu_memory_state(timeout_s=2.0)
    return {
        "observed_at_monotonic_ns": int(state.observed_at_monotonic_ns),
        "nvml_device_used_bytes": state.nvml_device_used_bytes,
        "torch_allocated_bytes": state.torch_allocated_bytes,
        "torch_reserved_bytes": state.torch_reserved_bytes,
        "kv_pool_reserved_bytes": int(state.kv_pool_reserved_bytes),
        "kv_assigned_bytes": int(state.kv_assigned_bytes),
    }


def _hbm_stable(samples: tuple[dict[str, int | None], ...]) -> bool:
    if len(samples) < 3:
        return False
    for field in (
        "nvml_device_used_bytes",
        "torch_allocated_bytes",
        "torch_reserved_bytes",
        "kv_pool_reserved_bytes",
        "kv_assigned_bytes",
    ):
        values = tuple(sample[field] for sample in samples)
        if any(value is None for value in values):
            return False
        numeric = tuple(int(value) for value in values if value is not None)
        tolerance = _HBM_STABILITY_TOLERANCE_BYTES if field == "nvml_device_used_bytes" else 0
        if max(numeric) - min(numeric) > tolerance:
            return False
    return True


def _reset_probe_state(adapter: Any, *, probe_id: str) -> dict[str, Any]:
    """Prove an idle, bounded request-state reset without reloading the engine."""

    started_ns = time.monotonic_ns()
    state_before = _runtime_queue_state(adapter)
    if any(state_before.values()):
        raise RuntimeError(f"probe {probe_id} retained live request state before reset")
    if not adapter._view.manager.reset_prefix_cache():
        raise RuntimeError(f"probe {probe_id} prefix cache did not reset")
    state_after = _runtime_queue_state(adapter)
    if any(state_after.values()):
        raise RuntimeError(f"probe {probe_id} retained request state after reset")
    memory_after = _memory_sample(adapter)
    ended_ns = time.monotonic_ns()
    return {
        "schema_version": "sloforge.branchfabric.capacity-request-reset/v1",
        "probe_id": probe_id,
        "started_ns": started_ns,
        "ended_ns": ended_ns,
        "duration_seconds": (ended_ns - started_ns) / 1e9,
        "engine_reloaded": False,
        "state_before": state_before,
        "prefix_cache_reset": True,
        "state_after": state_after,
        "memory_after": memory_after,
        "passed": True,
    }


def _collect_engine_readiness(
    adapter: Any,
    *,
    device: str,
    physical_gpu_uuid: str,
    prefix: tuple[int, ...],
    output_tokens: int,
    seed: int,
    engine_started_ns: int,
    model_ready_ns: int,
    timeout_seconds: float,
    hbm_sample_interval_seconds: float = 0.05,
) -> dict[str, Any]:
    """Warm all serving batch shapes and emit one fail-closed readiness object."""

    if output_tokens != 64:
        raise ValueError("readiness is pinned to the 64-token serving shape")
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0.0:
        raise ValueError("readiness timeout must be finite and positive")
    engine = adapter._view.llm_engine
    queue_before = _runtime_queue_state(adapter)
    if any(queue_before.values()):
        raise RuntimeError("engine was not idle before readiness warmup")
    graph_sizes = _cudagraph_capture_sizes(adapter)
    graph_coverage_pass = set(_READINESS_BATCH_SIZES).issubset(graph_sizes)
    params = _readiness_sampling_params(output_tokens=output_tokens, seed=seed)
    warmup_started_ns = time.monotonic_ns()
    warmup_rows: list[dict[str, Any]] = []
    request_offset = 0
    with _CompilationLogCapture() as warmup_compilation:
        for batch_size in _READINESS_BATCH_SIZES:
            row = _run_readiness_batch(
                engine,
                device=device,
                batch_size=batch_size,
                request_offset=request_offset,
                prefix=prefix,
                sampling_params=params,
                timeout_seconds=timeout_seconds,
            )
            warmup_rows.append(row)
            request_offset += batch_size
    warmup_ended_ns = time.monotonic_ns()
    warmup_output_pass = all(
        output_length == output_tokens
        for row in warmup_rows
        for output_length in row["output_lengths"]
    )
    verification_rows: list[dict[str, Any]] = []
    with _CompilationLogCapture() as probe_compilation:
        for batch_size in _READINESS_BATCH_SIZES:
            row = _run_readiness_batch(
                engine,
                device=device,
                batch_size=batch_size,
                request_offset=request_offset,
                prefix=prefix,
                sampling_params=params,
                timeout_seconds=timeout_seconds,
            )
            verification_rows.append(row)
            request_offset += batch_size
    probe = verification_rows[0]
    if not adapter._view.manager.reset_prefix_cache():
        raise RuntimeError("readiness prefix cache did not reset after warmup/probe")
    queue_after = _runtime_queue_state(adapter)
    health_pass, health_source = _runtime_health(adapter)
    metrics_pass, metrics_source = _runtime_metrics(adapter)
    try:
        import torch  # type: ignore[import-not-found]

        torch.cuda.synchronize()
    except (ImportError, RuntimeError, AttributeError) as error:
        raise RuntimeError("readiness could not synchronize its assigned CUDA device") from error
    hbm_samples: list[dict[str, int | None]] = []
    for index in range(3):
        hbm_samples.append(_memory_sample(adapter))
        if index != 2:
            time.sleep(hbm_sample_interval_seconds)
    hbm_rows = tuple(hbm_samples)
    hbm_stability_pass = _hbm_stable(hbm_rows)
    probe_output_pass = all(
        output_length == output_tokens
        for row in verification_rows
        for output_length in row["output_lengths"]
    )
    probe_ttft_pass = all(0.0 < float(row["ttft_seconds"]) <= 2.0 for row in verification_rows)
    queue_empty_pass = not any(queue_after.values())
    no_probe_compilation_pass = not probe_compilation.events
    passed = all(
        (
            graph_coverage_pass,
            warmup_output_pass,
            probe_output_pass,
            probe_ttft_pass,
            queue_empty_pass,
            hbm_stability_pass,
            health_pass,
            metrics_pass,
            no_probe_compilation_pass,
        )
    )
    return {
        "schema_version": "sloforge.branchfabric.engine-readiness-evidence/v1",
        "device": device,
        "pid": os.getpid(),
        "physical_gpu_uuid": physical_gpu_uuid,
        "engine_started_ns": engine_started_ns,
        "model_ready_ns": model_ready_ns,
        "compile_warmup_started_ns": warmup_started_ns,
        "compile_warmup_ended_ns": warmup_ended_ns,
        "required_cudagraph_batch_sizes": _READINESS_BATCH_SIZES,
        "observed_cudagraph_capture_sizes": graph_sizes,
        "cudagraph_coverage_pass": graph_coverage_pass,
        "warmup_request_count": sum(_READINESS_BATCH_SIZES),
        "warmup_output_target_verified": warmup_output_pass,
        "warmup_batches": tuple(warmup_rows),
        "verification_request_count": sum(_READINESS_BATCH_SIZES),
        "verification_output_target_verified": probe_output_pass,
        "verification_batches": tuple(verification_rows),
        "compilation_observation": {
            "source": "vllm-init-return+bounded-python-logging-handler",
            "engine_initialization_completed_before_warmup": model_ready_ns >= engine_started_ns,
            "warmup_compilation_events": tuple(warmup_compilation.events),
            "measured_probe_compilation_events": tuple(probe_compilation.events),
            "no_active_compilation_event": no_probe_compilation_pass,
            "fail_closed_limitation": (
                "vLLM 0.23.0 exposes no public compile-in-progress flag; any deferred "
                "JIT/graph-compilation log event during the measured probe fails readiness"
            ),
        },
        "probe": {
            "started_ns": probe["started_ns"],
            "first_token_ns": probe["first_token_ns"],
            "completed_ns": probe["completed_ns"],
            "requested_output_tokens": output_tokens,
            "emitted_output_tokens": probe["output_lengths"][0],
            "ttft_seconds": probe["ttft_seconds"],
            "request_throughput_rps": probe["request_throughput_rps"],
            "output_token_throughput_tps": probe["output_token_throughput_tps"],
            "output_target_verified": probe_output_pass,
            "ttft_sane": probe_ttft_pass,
        },
        "hbm": {
            "samples": hbm_rows,
            "nvml_stability_tolerance_bytes": _HBM_STABILITY_TOLERANCE_BYTES,
            "stable": hbm_stability_pass,
        },
        "runtime_state": {
            **queue_after,
            "prefix_cache_reset": True,
            "health": "healthy" if health_pass else "unhealthy-or-unavailable",
            "health_source": health_source,
            "metrics_available": metrics_pass,
            "metrics_source": metrics_source,
        },
        "queue_empty_pass": queue_empty_pass,
        "passed": passed,
    }


def _execute_probe(
    engine: Any,
    *,
    plan_payload: dict[str, Any],
    device: str,
    prefix: tuple[int, ...],
    output_tokens: int,
    seed: int,
    probe_timeout_seconds: float,
    tail_drain_seconds: float,
    producer_queue_capacity: int,
) -> dict[str, Any]:
    from gpu_reclamation_worker import _sampling_params

    from sloforge.helix.characterization.gpu_capacity_calibration import CapacityProbePlan

    plan = CapacityProbePlan.model_validate_json(
        json.dumps(plan_payload, sort_keys=True, separators=(",", ":")), strict=True
    )
    assigned = tuple(item for item in plan.arrivals if item.assigned_device == device)
    tail_drain_end_ns = plan.measurement_end_ns + round(tail_drain_seconds * 1e9)
    if not assigned:
        deadline_ns = time.monotonic_ns() + round(probe_timeout_seconds * 1e9)
        if tail_drain_end_ns > deadline_ns:
            raise TimeoutError(f"probe {plan.probe_id} idle shard exceeds its worker bound")
        _sleep_until(tail_drain_end_ns)
        probe_end_ns = time.monotonic_ns()
        return {
            "schema_version": "sloforge.branchfabric.capacity-worker-probe/v1",
            "probe_id": plan.probe_id,
            "device": device,
            "physical_gpu_uuid": os.environ["SLOFORGE_EXP004_PHYSICAL_GPU_UUID"],
            "observations": (),
            "tail_drain_end_ns": tail_drain_end_ns,
            "probe_end_ns": probe_end_ns,
            "completed_at_monotonic_ns": probe_end_ns,
        }
    if producer_queue_capacity <= 0:
        raise ValueError("producer queue capacity must be positive")
    admission_queue: queue.Queue[Any] = queue.Queue(maxsize=producer_queue_capacity)
    observations: dict[str, dict[str, Any]] = {}
    observations_lock = threading.Lock()
    producer_errors: list[BaseException] = []
    producer_done = threading.Event()
    active: set[str] = set()
    params = _sampling_params(max_tokens=output_tokens, seed=seed)
    deadline_ns = time.monotonic_ns() + round(probe_timeout_seconds * 1e9)

    def produce() -> None:
        try:
            for arrival in assigned:
                _sleep_until(arrival.scheduled_arrival_ns)
                offered_ns = time.monotonic_ns()
                row = {
                    "request_id": arrival.request_id,
                    "global_sequence": arrival.global_sequence,
                    "device": device,
                    "scheduled_arrival_ns": arrival.scheduled_arrival_ns,
                    "offered_ns": offered_ns,
                    "enqueued_ns": None,
                    "admitted_ns": None,
                    "first_token_ns": None,
                    "completed_ns": None,
                    "terminated_ns": tail_drain_end_ns,
                    "requested_output_tokens": output_tokens,
                    "emitted_tokens": 0,
                    "terminal_state": "not-admitted",
                }
                with observations_lock:
                    observations[arrival.request_id] = row
                while time.monotonic_ns() < tail_drain_end_ns:
                    enqueued_ns = time.monotonic_ns()
                    try:
                        admission_queue.put_nowait((arrival, enqueued_ns))
                    except queue.Full:
                        time.sleep(0.01)
                        continue
                    with observations_lock:
                        row["enqueued_ns"] = enqueued_ns
                    break
        except BaseException as caught:
            producer_errors.append(caught)
        finally:
            producer_done.set()

    producer = threading.Thread(
        target=produce,
        name=f"capacity-producer-{plan.probe_id}-{device}",
        daemon=False,
    )
    producer.start()

    while time.monotonic_ns() < tail_drain_end_ns:
        now = time.monotonic_ns()
        if now >= deadline_ns:
            raise TimeoutError(f"probe {plan.probe_id} exceeded its absolute worker bound")
        while True:
            try:
                arrival, enqueued_ns = admission_queue.get_nowait()
            except queue.Empty:
                break
            engine.add_request(
                arrival.request_id,
                {"prompt_token_ids": list(_calibration_prompt(prefix, arrival.global_sequence))},
                params,
            )
            admitted_ns = time.monotonic_ns()
            with observations_lock:
                row = observations[arrival.request_id]
                row["enqueued_ns"] = enqueued_ns
                row["admitted_ns"] = admitted_ns
                row["terminal_state"] = "aborted"
            active.add(arrival.request_id)
        if active:
            outputs = engine.step()
            observed_ns = time.monotonic_ns()
            for output in outputs:
                request_id = str(getattr(output, "request_id", ""))
                with observations_lock:
                    output_row = observations.get(request_id)
                    if output_row is None:
                        continue
                    candidates = getattr(output, "outputs", ())
                    token_ids = (
                        tuple(int(item) for item in getattr(candidates[0], "token_ids", ()))
                        if candidates
                        else ()
                    )
                    if len(token_ids) > int(output_row["emitted_tokens"]):
                        output_row["emitted_tokens"] = len(token_ids)
                        if output_row["first_token_ns"] is None:
                            output_row["first_token_ns"] = observed_ns
                    if bool(getattr(output, "finished", False)):
                        output_row["completed_ns"] = observed_ns
                        output_row["terminated_ns"] = observed_ns
                        output_row["terminal_state"] = "completed"
                        active.discard(request_id)
        else:
            time.sleep(0.001)

    producer.join(timeout=1.0)
    if producer.is_alive():
        raise TimeoutError("bounded capacity producer survived the fixed probe tail")
    if producer_errors:
        raise RuntimeError(f"capacity producer failed: {producer_errors[0]}")
    if active:
        engine.abort_request(sorted(active))
    terminal_ns = time.monotonic_ns()
    if terminal_ns > tail_drain_end_ns + 1_000_000_000:
        raise TimeoutError("capacity abort finalization exceeded its one-second allowance")
    with observations_lock:
        for request_id in active:
            observations[request_id]["terminated_ns"] = terminal_ns
            observations[request_id]["terminal_state"] = "aborted"
        while True:
            try:
                pending, enqueued_ns = admission_queue.get_nowait()
            except queue.Empty:
                break
            observations[pending.request_id]["enqueued_ns"] = enqueued_ns
            observations[pending.request_id]["terminated_ns"] = terminal_ns
            observations[pending.request_id]["terminal_state"] = "not-admitted"
        for row in observations.values():
            if row["terminal_state"] != "completed":
                row["terminated_ns"] = terminal_ns

    if set(observations) != {item.request_id for item in assigned}:
        raise RuntimeError("capacity producer omitted planned request accounting")
    rows = tuple(observations[item.request_id] for item in assigned)
    if any(
        row["terminal_state"] == "completed" and row["emitted_tokens"] != output_tokens
        for row in rows
    ):
        raise RuntimeError("completed calibration request violated the 64-token sampling target")
    return {
        "schema_version": "sloforge.branchfabric.capacity-worker-probe/v1",
        "probe_id": plan.probe_id,
        "device": device,
        "physical_gpu_uuid": os.environ["SLOFORGE_EXP004_PHYSICAL_GPU_UUID"],
        "observations": rows,
        "tail_drain_end_ns": tail_drain_end_ns,
        "probe_end_ns": terminal_ns,
        "completed_at_monotonic_ns": terminal_ns,
        "prefix_cache_policy": {
            "enabled_for_rollout_semantics": True,
            "calibration_reuse_prevented": True,
            "method": "request-unique-first-block-rotation-transitive-hash-chain",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("gpu0", "gpu1"), required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--command-root", type=Path, required=True)
    parser.add_argument("--model-snapshot", type=Path, required=True)
    parser.add_argument("--physical-gpu-uuid", required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    args.work_root.mkdir(parents=True, exist_ok=False)
    started_ns = time.monotonic_ns()
    adapter: Any | None = None
    error: dict[str, str] | None = None
    try:
        from gpu_reclamation_worker import _create_adapter

        adapter_config = {
            **config,
            "initialization_timeout_seconds": config["worker_initialization_timeout_seconds"],
            "disable_log_stats": False,
        }
        adapter = _create_adapter(
            adapter_config,
            args.model_snapshot.resolve(strict=True),
            args.physical_gpu_uuid,
        )
        inputs = json.loads((args.model_snapshot / "BRANCHFABRIC_INPUTS.json").read_text())
        prefix = tuple(
            int(item) for item in inputs["prefix_token_ids"][: int(config["serving_prompt_tokens"])]
        )
        import torch  # type: ignore[import-not-found]

        packages = {
            name: importlib.metadata.version(name) for name in ("vllm", "torch", "transformers")
        }
        gpu_name = torch.cuda.get_device_name(0)
        if packages["vllm"] != "0.23.0" or packages["torch"] != "2.11.0":
            raise RuntimeError(f"calibration runtime pins differ: {packages}")
        if "A100" not in gpu_name or torch.version.cuda is None:
            raise RuntimeError("calibration worker lacks the required A100/CUDA runtime")
        runtime_evidence = {
            "packages": packages,
            "torch_cuda": torch.version.cuda,
            "gpu_name": gpu_name,
        }
        model_ready_ns = time.monotonic_ns()
        readiness = _collect_engine_readiness(
            adapter,
            device=args.device,
            physical_gpu_uuid=args.physical_gpu_uuid,
            prefix=prefix,
            output_tokens=int(config["serving_output_tokens"]),
            seed=int(config["seed"]),
            engine_started_ns=started_ns,
            model_ready_ns=model_ready_ns,
            timeout_seconds=float(config["probe_timeout_seconds"]),
        )
        readiness["runtime_evidence"] = runtime_evidence
        if not readiness["passed"]:
            raise RuntimeError("strict engine readiness evidence failed")
        _write_new(
            args.command_root / f"{args.device}.ready.json",
            readiness,
        )
        overall_deadline_ns = started_ns + round(float(config["controller_timeout_seconds"]) * 1e9)
        for probe_index in range(int(config["maximum_probes"])):
            command_path = args.command_root / f"probe-{probe_index:02d}.json"
            shutdown_path = args.command_root / "shutdown.json"
            while not command_path.is_file() and not shutdown_path.is_file():
                if time.monotonic_ns() >= overall_deadline_ns:
                    raise TimeoutError("calibration worker exceeded its overall deadline")
                time.sleep(0.02)
            if shutdown_path.is_file():
                break
            command = json.loads(command_path.read_text())
            result = _execute_probe(
                adapter._view.llm_engine,
                plan_payload=command["plan"],
                device=args.device,
                prefix=prefix,
                output_tokens=int(config["serving_output_tokens"]),
                seed=int(config["seed"]),
                probe_timeout_seconds=float(config["probe_timeout_seconds"]),
                tail_drain_seconds=float(config["tail_drain_seconds"]),
                producer_queue_capacity=int(config["producer_queue_capacity"]),
            )
            _write_new(args.command_root / f"probe-{probe_index:02d}.{args.device}.json", result)
            reset = _reset_probe_state(adapter, probe_id=str(result["probe_id"]))
            _write_new(
                args.command_root / f"probe-{probe_index:02d}.{args.device}.reset.json",
                reset,
            )
        else:
            _wait_for(args.command_root / "shutdown.json", deadline_ns=overall_deadline_ns)
    except BaseException as caught:
        error = {
            "type": type(caught).__name__,
            "message": str(caught),
            "traceback": traceback.format_exc(),
        }
    finally:
        if adapter is not None:
            try:
                adapter.cleanup_runtime(timeout_s=float(config["worker_shutdown_timeout_seconds"]))
            except BaseException as caught:
                if error is None:
                    error = {
                        "type": type(caught).__name__,
                        "message": str(caught),
                        "traceback": traceback.format_exc(),
                    }
    result = {
        "schema_version": "sloforge.branchfabric.capacity-worker-completion/v1",
        "status": "succeeded" if error is None else "failed",
        "device": args.device,
        "physical_gpu_uuid": args.physical_gpu_uuid,
        "started_ns": started_ns,
        "ended_ns": time.monotonic_ns(),
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "runtime_evidence": locals().get("runtime_evidence"),
        "error": error,
    }
    _write_new(args.work_root / "result.json", result)
    return 0 if error is None else 1


if __name__ == "__main__":
    raise SystemExit(main())
