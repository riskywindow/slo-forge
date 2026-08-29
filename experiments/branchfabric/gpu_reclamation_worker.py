"""One-GPU worker for BranchFabric Experiment 004's two-GPU pilot.

The controller gives this process exactly one physical GPU UUID before Python
imports PyTorch or vLLM.  The serving role runs the deterministic load; the
rollout role executes one warm preserve -> serve -> restore transaction.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import math
import os
import re
import threading
import time
import traceback
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
MODEL_REVISION = "a09a35458c702b33eeacc393d103063234e8bc28"
POLICY_EPOCH = "branchfabric-modal-exp004-policy-v1"
RESOURCE_SAMPLE_INTERVAL_MS = 100
RESOURCE_SAMPLE_GRACE_SECONDS = 10
_V10_SANITY_GUARD_COUNT = 2


class _CacheSaltEngine:
    """Delegate to one live vLLM engine while isolating serving prefixes.

    vLLM 0.23.0 incorporates ``cache_salt`` into the first block hash and the
    resulting transitive hash chain.  A deterministic SHA-256 salt per request
    therefore prevents calibration/control requests from receiving accidental
    prefix-cache hits without disabling the prefix cache needed by Helix's
    shared-root rollout state.
    """

    def __init__(self, engine: Any, *, namespace: str) -> None:
        if not namespace:
            raise ValueError("cache-salt namespace must not be empty")
        self._engine = engine
        self._namespace = namespace
        self._salts: dict[str, str] = {}
        self._salt_values: set[str] = set()
        self._lock = threading.Lock()

    @property
    def underlying_engine(self) -> Any:
        return self._engine

    def add_request(self, request_id: str, prompt: Any, *args: Any, **kwargs: Any) -> Any:
        request_key = str(request_id)
        with self._lock:
            if request_key in self._salts:
                raise RuntimeError(f"cache-salt request identity was reused: {request_key}")
            salt = hashlib.sha256(f"{self._namespace}\0{request_key}".encode()).hexdigest()
            if salt in self._salt_values:
                raise RuntimeError("deterministic cache-salt collision")
            self._salts[request_key] = salt
            self._salt_values.add(salt)
        if not isinstance(prompt, dict) or "prompt_token_ids" not in prompt:
            raise TypeError("salted serving requires a TokensPrompt mapping")
        if "cache_salt" in prompt:
            raise ValueError("caller supplied an untracked cache_salt")
        salted_prompt = {**prompt, "cache_salt": salt}
        return self._engine.add_request(request_id, salted_prompt, *args, **kwargs)

    def evidence(self) -> dict[str, Any]:
        with self._lock:
            rows = tuple(sorted(self._salts.items()))
        encoded_rows = "".join(f"{request_id}\0{salt}\n" for request_id, salt in rows).encode()
        return {
            "schema_version": "sloforge.branchfabric.cache-salt-evidence/v1",
            "policy": "per-request-vllm-cache-salt-sha256",
            "namespace_sha256": hashlib.sha256(self._namespace.encode()).hexdigest(),
            "request_count": len(rows),
            "unique_request_count": len({request_id for request_id, _salt in rows}),
            "unique_salt_count": len({salt for _request_id, salt in rows}),
            "request_salt_map_sha256": hashlib.sha256(encoded_rows).hexdigest(),
            "prefix_cache_remained_enabled": True,
            "rollout_requests_unsalted_for_shared_root_semantics": True,
            "passed": len(rows) == len({salt for _request_id, salt in rows}),
        }

    def __getattr__(self, name: str) -> Any:
        return getattr(self._engine, name)


def _canonical_bytes(value: Any) -> bytes:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n"
    ).encode("utf-8")


def _write_new(path: Path, value: Any) -> None:
    """Atomically publish one immutable cross-worker JSON artifact."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(_canonical_bytes(value))
            handle.flush()
            os.fsync(handle.fileno())
        # A hard link publishes the complete inode without allowing an
        # existing immutable barrier to be replaced.
        os.link(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def _write_new_artifact(
    path: Path,
    value: Any,
    *,
    artifact_root: Path,
    sample_selector: str,
) -> dict[str, str]:
    """Write one immutable raw artifact and return exact local provenance."""

    resolved_root = artifact_root.resolve()
    resolved_path = path.resolve()
    try:
        reference = resolved_path.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError("raw telemetry artifact must remain below the worker root") from exc
    encoded = _canonical_bytes(value)
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    with resolved_path.open("xb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    return {
        "artifact_reference": reference.as_posix(),
        "artifact_sha256": hashlib.sha256(encoded).hexdigest(),
        "sample_selector": sample_selector,
    }


def _existing_artifact_provenance(
    path: Path, *, artifact_root: Path, sample_selector: str
) -> dict[str, str]:
    """Bind a raw artifact produced atomically by an external profiler."""

    resolved_root = artifact_root.resolve()
    resolved_path = path.resolve(strict=True)
    try:
        reference = resolved_path.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError("profile artifact must remain below the worker root") from exc
    return {
        "artifact_reference": reference.as_posix(),
        "artifact_sha256": hashlib.sha256(resolved_path.read_bytes()).hexdigest(),
        "sample_selector": sample_selector,
    }


def _resource_sampling_config(config: dict[str, Any]) -> Any:
    """Derive a deterministic bound covering readiness wait plus transaction."""

    from sloforge.helix.characterization.gpu_reclamation_instrumentation import SamplingConfig

    readiness_seconds = float(config["initialization_timeout_seconds"])
    transaction_seconds = float(config["maximum_wall_seconds"])
    if (
        not math.isfinite(readiness_seconds)
        or not math.isfinite(transaction_seconds)
        or readiness_seconds <= 0
        or transaction_seconds <= 0
    ):
        raise ValueError("telemetry requires finite positive worker time bounds")
    covered_ms = math.ceil(
        (readiness_seconds + transaction_seconds + RESOURCE_SAMPLE_GRACE_SECONDS) * 1_000
    )
    max_samples = math.ceil(covered_ms / RESOURCE_SAMPLE_INTERVAL_MS) + 1
    return SamplingConfig(
        interval_ms=RESOURCE_SAMPLE_INTERVAL_MS,
        max_samples=max_samples,
        require_pcie_counters=False,
    )


class _WorkerResourceTelemetry:
    """One worker's bounded NVML/host sampler and immutable raw artifact."""

    def __init__(
        self,
        *,
        role: str,
        physical_gpu_uuid: str,
        worker_pid: int,
        work_root: Path,
        config_sha256: str,
        inventory: Any,
        descriptor: Any,
        gpu_probe: Any,
        sampler: Any,
        started_ns: int,
    ) -> None:
        self.role = role
        self.physical_gpu_uuid = physical_gpu_uuid
        self.worker_pid = worker_pid
        self.work_root = work_root
        self.config_sha256 = config_sha256
        self.inventory = inventory
        self.descriptor = descriptor
        self.gpu_probe = gpu_probe
        self.sampler = sampler
        self.started_ns = started_ns
        self._stopped = False

    @property
    def is_running(self) -> bool:
        return bool(self.sampler.is_running)

    def stop(self, *, timeout_seconds: float) -> dict[str, Any]:
        if self._stopped:
            raise RuntimeError("worker resource telemetry has already been stopped")
        self._stopped = True
        try:
            sampling = self.sampler.stop(timeout_seconds=timeout_seconds)
        except BaseException:
            if not self.sampler.is_running:
                self.gpu_probe.close()
            raise
        else:
            self.gpu_probe.close()
        if sampling.reached_sample_bound:
            raise RuntimeError("worker telemetry exhausted its deterministic sample bound")
        if not sampling.samples:
            raise RuntimeError("worker telemetry produced no resource samples")
        for expected_sequence, sample in enumerate(sampling.samples):
            if sample.sequence != expected_sequence:
                raise RuntimeError("worker telemetry sample sequence is not contiguous")
            if tuple(item.uuid for item in sample.gpu_samples) != (self.physical_gpu_uuid,):
                raise RuntimeError("worker telemetry sampled a GPU outside its assigned role")
            if self.worker_pid not in {item.pid for item in sample.host_sample.processes}:
                raise RuntimeError("worker telemetry lost its experiment-owned process")
        stopped_ns = time.monotonic_ns()
        payload = {
            "schema_version": "sloforge.branchfabric.experiment-004-worker-telemetry/v1",
            "role": self.role,
            "physical_gpu_uuid": self.physical_gpu_uuid,
            "worker_pid": self.worker_pid,
            "config_sha256": self.config_sha256,
            "started_monotonic_ns": self.started_ns,
            "stopped_monotonic_ns": stopped_ns,
            "inventory": self.inventory.model_dump(mode="json"),
            "assigned_gpu": self.descriptor.model_dump(mode="json"),
            "sampling": sampling.model_dump(mode="json"),
            "sample_count": len(sampling.samples),
            "first_sample_trigger_monotonic_ns": (sampling.samples[0].sample_trigger_monotonic_ns),
            "last_sample_trigger_monotonic_ns": (sampling.samples[-1].sample_trigger_monotonic_ns),
        }
        provenance = _write_new_artifact(
            self.work_root / "telemetry/resource-sampling.json",
            payload,
            artifact_root=self.work_root,
            sample_selector="$.sampling.samples[*]",
        )
        return {
            "schema_version": "sloforge.branchfabric.experiment-004-telemetry-reference/v1",
            "status": "succeeded",
            "role": self.role,
            "physical_gpu_uuid": self.physical_gpu_uuid,
            "sample_count": len(sampling.samples),
            "interval_ms": sampling.config.interval_ms,
            "reached_sample_bound": sampling.reached_sample_bound,
            "raw_provenance": provenance,
        }


def _start_worker_resource_telemetry(
    *,
    role: str,
    physical_gpu_uuid: str,
    worker_pid: int,
    work_root: Path,
    config: dict[str, Any],
    gpu_probe_factory: Any = None,
    host_probe_factory: Any = None,
    sampler_factory: Any = None,
) -> _WorkerResourceTelemetry:
    """Start the existing low-overhead sampler for exactly one worker role."""

    if role not in {"serving", "rollout"}:
        raise ValueError("telemetry role must be serving or rollout")
    from sloforge.helix.characterization.gpu_reclamation_instrumentation import (
        LowOverheadResourceSampler,
        NvmlProbe,
        PsutilHostProbe,
    )

    probe_factory = gpu_probe_factory or NvmlProbe
    process_probe_factory = host_probe_factory or PsutilHostProbe
    resource_sampler_factory = sampler_factory or LowOverheadResourceSampler
    gpu_probe = probe_factory()
    try:
        inventory = gpu_probe.inventory()
        inventory.require_exact_gpu_count(2)
        matches = tuple(
            descriptor for descriptor in inventory.devices if descriptor.uuid == physical_gpu_uuid
        )
        if len(matches) != 1:
            raise RuntimeError(
                "worker physical GPU UUID is absent or duplicated in the NVML inventory"
            )
        descriptor = matches[0]
        host_probe = process_probe_factory(process_ids=(worker_pid,))
        sampler = resource_sampler_factory(
            gpu_probe=gpu_probe,
            host_probe=host_probe,
            descriptors=(descriptor,),
            config=_resource_sampling_config(config),
        )
        started_ns = time.monotonic_ns()
        sampler.start()
    except BaseException:
        gpu_probe.close()
        raise
    return _WorkerResourceTelemetry(
        role=role,
        physical_gpu_uuid=physical_gpu_uuid,
        worker_pid=worker_pid,
        work_root=work_root,
        config_sha256=hashlib.sha256(_canonical_bytes(config)).hexdigest(),
        inventory=inventory,
        descriptor=descriptor,
        gpu_probe=gpu_probe,
        sampler=sampler,
        started_ns=started_ns,
    )


def _arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", required=True, choices=("serving", "rollout"))
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--work-root", required=True, type=Path)
    parser.add_argument("--barrier-root", required=True, type=Path)
    parser.add_argument("--model-snapshot", required=True, type=Path)
    parser.add_argument("--physical-gpu-uuid", required=True)
    return parser.parse_args(argv)


def _wait_for(path: Path, *, timeout_s: float) -> dict[str, Any]:
    deadline = time.monotonic_ns() + int(timeout_s * 1e9)
    while not path.is_file():
        if time.monotonic_ns() >= deadline:
            raise TimeoutError(f"timed out waiting for barrier {path}")
        time.sleep(0.02)
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"barrier {path} is not a JSON object")
    return payload


def _sleep_until(timestamp_ns: int) -> None:
    while True:
        remaining = timestamp_ns - time.monotonic_ns()
        if remaining <= 0:
            return
        time.sleep(min(remaining / 1e9, 0.05))


def _identity() -> dict[str, str]:
    return {
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "tokenizer_id": MODEL_ID,
        "tokenizer_revision": MODEL_REVISION,
        "dtype": "bfloat16",
        "policy_epoch": POLICY_EPOCH,
    }


def _create_adapter(config: dict[str, Any], snapshot: Path, gpu_uuid: str) -> Any:
    from sloforge.continuum.adapters.real_runtime import RuntimeModelIdentity
    from sloforge.continuum.adapters.vllm_live import VllmLiveStateAdapter
    from sloforge.helix.characterization.trace import TraceLevel

    identity = RuntimeModelIdentity(
        runtime="vllm",
        runtime_version="0.23.0",
        adapter_version="1.0.0",
        model_id=MODEL_ID,
        model_revision=MODEL_REVISION,
        tokenizer_id=MODEL_ID,
        tokenizer_revision=MODEL_REVISION,
        dtype="bfloat16",
        device="cuda:0",
        policy_epoch=POLICY_EPOCH,
    )

    def read_memory(_device: str) -> dict[str, int | None]:
        import pynvml  # type: ignore[import-not-found]

        pynvml.nvmlInit()
        try:
            handle = pynvml.nvmlDeviceGetHandleByUUID(gpu_uuid)
            memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
            return {
                "nvml_process_bytes": None,
                "nvml_device_used_bytes": int(memory.used),
            }
        finally:
            pynvml.nvmlShutdown()

    kwargs: dict[str, object] = {
        "model": str(snapshot),
        "revision": MODEL_REVISION,
        "tokenizer": str(snapshot),
        "tokenizer_revision": MODEL_REVISION,
        "dtype": "bfloat16",
        "max_model_len": 16_384 + 256 + 16,
        "enable_prefix_caching": True,
        "block_size": 16,
        "max_num_seqs": 16,
        "seed": int(config["seed"]),
        "tensor_parallel_size": 1,
        "async_scheduling": False,
        "gpu_memory_utilization": float(config["gpu_memory_utilization"]),
        "download_dir": "/tmp/sloforge-exp004-download",
        "enforce_eager": False,
        # Capacity calibration must prove that runtime metrics are available at
        # its readiness gate.  Preserve the historical state-transaction
        # default, but let that dedicated worker opt in without changing v9.
        "disable_log_stats": bool(config.get("disable_log_stats", True)),
        "trust_remote_code": False,
        "enable_chunked_prefill": True,
        "cpu_offload_gb": 0.0,
    }
    return VllmLiveStateAdapter.create(
        identity=identity,
        execution_model_path=str(snapshot),
        seed=int(config["seed"]),
        timeout_s=float(config["initialization_timeout_seconds"]),
        llm_kwargs=kwargs,
        max_fanout=8,
        default_maximum_new_tokens=256,
        gpu_memory_reader=read_memory,
        trace_level=TraceLevel.DISABLED,
        trace_id=f"exp004:{config['attempt_id']}:{gpu_uuid}",
    )


def _sampling_params(*, max_tokens: int, seed: int) -> Any:
    from vllm import SamplingParams  # type: ignore[import-not-found]

    return SamplingParams(
        max_tokens=max_tokens,
        min_tokens=max_tokens,
        ignore_eos=True,
        temperature=0.0,
        seed=seed,
    )


def _run_raw_serving(
    engine: Any,
    *,
    prefix: tuple[int, ...],
    start_ns: int,
    phases: tuple[tuple[float, float, str], ...],
    output_tokens: int,
    seed: int,
    request_prefix: str,
    continuation_barrier: Path | None = None,
    continuation_rate_per_second: float | None = None,
    continuation_phase: str = "restore-interference",
    continuation_grace_seconds: float = 0.0,
    continuation_timeout_seconds: float = 180.0,
) -> dict[str, Any]:
    """Run one deterministic constant-cadence request plan on a warm engine."""

    arrivals = list(
        _deterministic_serving_arrivals(
            start_ns=start_ns,
            phases=phases,
            request_prefix=request_prefix,
        )
    )
    offset_s = sum(item[0] for item in phases)
    observations: dict[str, dict[str, Any]] = {}
    next_arrival = 0
    active: set[str] = set()
    continuation_enabled = continuation_barrier is not None
    if continuation_enabled != (continuation_rate_per_second is not None):
        raise ValueError("serving continuation requires both a barrier and request rate")
    if continuation_rate_per_second is not None and (
        not math.isfinite(continuation_rate_per_second)
        or continuation_rate_per_second <= 0
        or not math.isfinite(continuation_grace_seconds)
        or continuation_grace_seconds < 0
        or not math.isfinite(continuation_timeout_seconds)
        or continuation_timeout_seconds <= 0
    ):
        raise ValueError("serving continuation bounds are invalid")
    continuation_start_ns = start_ns + int(offset_s * 1e9)
    continuation_interval_ns = (
        None
        if continuation_rate_per_second is None
        else max(1, int(1e9 / continuation_rate_per_second))
    )
    continuation_next_ns = continuation_start_ns
    continuation_stop_after_ns: int | None = None
    continuation_barrier_observed_ns: int | None = None
    final_deadline = continuation_start_ns + int(
        (continuation_timeout_seconds + continuation_grace_seconds + 30.0) * 1e9
    )
    params = _sampling_params(max_tokens=output_tokens, seed=seed)
    while (
        next_arrival < len(arrivals)
        or active
        or (
            continuation_enabled
            and (
                continuation_stop_after_ns is None
                or continuation_next_ns < continuation_stop_after_ns
            )
        )
    ):
        now = time.monotonic_ns()
        if now >= final_deadline:
            raise TimeoutError("bounded serving plan did not drain")
        if continuation_enabled:
            assert continuation_barrier is not None
            assert continuation_interval_ns is not None
            if continuation_stop_after_ns is None and continuation_barrier.is_file():
                continuation_barrier_observed_ns = now
                continuation_stop_after_ns = now + int(continuation_grace_seconds * 1e9)
            while continuation_next_ns <= now and (
                continuation_stop_after_ns is None
                or continuation_next_ns < continuation_stop_after_ns
            ):
                sequence = len(arrivals)
                arrivals.append(
                    (
                        continuation_next_ns,
                        f"{request_prefix}-{sequence:05d}",
                        continuation_phase,
                    )
                )
                continuation_next_ns += continuation_interval_ns
        while next_arrival < len(arrivals) and arrivals[next_arrival][0] <= now:
            scheduled_ns, request_id, phase = arrivals[next_arrival]
            engine.add_request(
                request_id,
                {"prompt_token_ids": list(prefix)},
                params,
            )
            observations[request_id] = {
                "request_id": request_id,
                "phase": phase,
                "scheduled_arrival_ns": scheduled_ns,
                "admitted_ns": time.monotonic_ns(),
                "first_token_ns": None,
                "completed_ns": None,
                "token_timestamps_ns": [],
                "output_token_ids": [],
            }
            active.add(request_id)
            next_arrival += 1
        if active:
            outputs = engine.step()
            observed_ns = time.monotonic_ns()
            for output in outputs:
                request_id = str(getattr(output, "request_id", ""))
                row = observations.get(request_id)
                if row is None:
                    continue
                completions = getattr(output, "outputs", ())
                token_ids = (
                    tuple(int(item) for item in getattr(completions[0], "token_ids", ()))
                    if completions
                    else ()
                )
                old_count = len(row["output_token_ids"])
                if len(token_ids) > old_count:
                    row["token_timestamps_ns"].extend([observed_ns] * (len(token_ids) - old_count))
                    row["output_token_ids"] = list(token_ids)
                    if row["first_token_ns"] is None:
                        row["first_token_ns"] = observed_ns
                if bool(getattr(output, "finished", False)):
                    row["completed_ns"] = observed_ns
                    active.discard(request_id)
        elif next_arrival < len(arrivals):
            _sleep_until(arrivals[next_arrival][0])
        elif continuation_enabled and continuation_stop_after_ns is None:
            assert continuation_barrier is not None
            assert continuation_interval_ns is not None
            _sleep_until(min(continuation_next_ns, now + 20_000_000))
    rows = tuple(observations[item[1]] for item in arrivals)
    if any(
        row["first_token_ns"] is None
        or row["completed_ns"] is None
        or len(row["output_token_ids"]) != output_tokens
        for row in rows
    ):
        raise RuntimeError("serving workload produced an incomplete request")
    return {
        "schema_version": "sloforge.branchfabric.experiment-004-serving-raw/v1",
        "start_ns": start_ns,
        "end_ns": time.monotonic_ns(),
        "requests": rows,
        "continuation_barrier_observed_ns": continuation_barrier_observed_ns,
    }


def _deterministic_serving_arrivals(
    *,
    start_ns: int,
    phases: tuple[tuple[float, float, str], ...],
    request_prefix: str,
) -> tuple[tuple[int, str, str], ...]:
    """Build the immutable offered-load clock shared by both worker roles."""

    if start_ns < 0 or not request_prefix:
        raise ValueError("serving arrival plan identity is invalid")
    arrivals: list[tuple[int, str, str]] = []
    sequence = 0
    offset_s = 0.0
    for duration_s, rate_per_s, phase in phases:
        if (
            not math.isfinite(duration_s)
            or not math.isfinite(rate_per_s)
            or duration_s <= 0
            or rate_per_s <= 0
            or not phase
        ):
            raise ValueError("serving phases require finite positive duration/rate and a name")
        interval_s = 1.0 / rate_per_s
        local = 0.0
        while local < duration_s - 1e-12:
            arrivals.append(
                (
                    start_ns + int((offset_s + local) * 1e9),
                    f"{request_prefix}-{sequence:05d}",
                    phase,
                )
            )
            sequence += 1
            local += interval_s
        offset_s += duration_s
    return tuple(arrivals)


def _natural_layer_names(layer_names: tuple[str, ...]) -> tuple[str, ...]:
    def key(name: str) -> tuple[int, str]:
        match = re.search(r"(?:layers|h)\.(\d+)", name)
        return (int(match.group(1)) if match else 1 << 30, name)

    return tuple(sorted(layer_names, key=key))


def _geometry(adapter: Any) -> Any:
    from sloforge.continuum.adapters.vllm_reclamation import NativeKvGeometry

    view = adapter._view
    tensors = tuple(view.runner.kv_caches)
    layer_names = _natural_layer_names(tuple(view.layer_ids))
    if len(layer_names) != len(tensors):
        raise RuntimeError(
            f"vLLM KV tensor/layer count mismatch: {len(tensors)} != {len(layer_names)}"
        )
    block_dimensions: list[int] = []
    semantic_shapes: list[tuple[int, ...]] = []
    for tensor in tensors:
        candidates = [
            index for index, size in enumerate(tensor.shape) if size == view.num_pool_blocks
        ]
        if len(candidates) != 1:
            raise RuntimeError(f"cannot identify unique native KV block dimension: {tensor.shape}")
        block_dim = candidates[0]
        block_dimensions.append(block_dim)
        semantic_shapes.append(
            tuple(int(size) for index, size in enumerate(tensor.shape) if index != block_dim)
        )
    if len(set(semantic_shapes)) != 1:
        raise RuntimeError("vLLM KV layers expose nonuniform native semantic shapes")
    shape = semantic_shapes[0]
    expected_prefix = (2, view.block_size_tokens)
    if len(shape) != 4 or shape[:2] != expected_prefix:
        raise RuntimeError(f"unsupported vLLM KV semantic axes: {shape}")
    geometry = NativeKvGeometry(
        layer_names=layer_names,
        num_blocks=view.num_pool_blocks,
        block_size_tokens=view.block_size_tokens,
        kv_heads=shape[2],
        head_size=shape[3],
        element_size_bytes=tensors[0].element_size(),
        block_dimensions=tuple(block_dimensions),
        source_axis_labels=("kv", "token", "head", "dim"),
    )
    if geometry.physical_page_bytes != view.page_size_bytes:
        raise RuntimeError("canonical geometry does not equal vLLM's physical page bytes")
    return tensors, geometry


def _prepare_rollouts(
    adapter: Any, config: dict[str, Any], inputs: dict[str, Any]
) -> dict[str, Any]:
    prefix = tuple(int(item) for item in inputs["prefix_token_ids"][:16_384])
    divergent = tuple(int(item) for item in inputs["divergent_token_ids"][:8])
    if len(prefix) != 16_384 or len(divergent) != 8:
        raise ValueError("cached Experiment 003 tokenizer inputs are incomplete")
    root_id = f"{config['attempt_id']}-root"
    branches = tuple(f"branch.{index}" for index in range(8))
    adapter.start_session(root_id, token_ids=prefix, seed=int(config["seed"]), timeout_s=30.0)
    adapter.prefill_session(root_id, timeout_s=300.0)
    adapter.pause_at_safe_decode_boundary(root_id, timeout_s=5.0)
    root = adapter.create_shared_root_reference(root_id, prefix_token_count=16_384, timeout_s=30.0)
    for branch_id, token_id in zip(branches, divergent, strict=True):
        adapter.fork_same_policy_session(
            root.root_reference_id,
            branch_id,
            divergent_token_id=token_id,
            seed=int(config["seed"]),
            timeout_s=10.0,
        )
    decode = adapter.run_concurrent_branches(
        branches,
        maximum_new_tokens=256,
        seed=int(config["seed"]),
        timeout_s=300.0,
    )
    if set(decode.completed_branch_ids) != set(branches):
        raise RuntimeError("not all eight rollout branches reached the checkpoint boundary")
    return {"root_id": root_id, "root_reference_id": root.root_reference_id, "branches": branches}


def _wait_for_any(
    paths: tuple[Path, ...], *, timeout_s: float, phase: str
) -> tuple[Path, dict[str, Any]]:
    """Wait for exactly one immutable protocol command within a fixed bound."""

    if not paths or not math.isfinite(timeout_s) or timeout_s <= 0:
        raise ValueError("bounded command wait requires paths and a positive timeout")
    deadline_ns = time.monotonic_ns() + round(timeout_s * 1e9)
    while True:
        existing = tuple(path for path in paths if path.is_file())
        if len(existing) > 1:
            raise RuntimeError(f"conflicting commands observed during {phase}")
        if existing:
            payload = json.loads(existing[0].read_text())
            if not isinstance(payload, dict):
                raise ValueError(f"{phase} command must be a JSON object")
            return existing[0], payload
        if time.monotonic_ns() >= deadline_ns:
            raise TimeoutError(f"timed out during {phase}")
        time.sleep(0.02)


def _transaction_ready_path(barrier_root: Path, role: str) -> Path:
    if role not in {"serving", "rollout"}:
        raise ValueError("transaction-ready role must be serving or rollout")
    return barrier_root / f"{role}.transaction-ready.json"


def _engine_continuity(
    adapter: Any,
    *,
    engine_nonce: str,
    engine_started_ns: int,
    device: str,
    physical_gpu_uuid: str,
) -> dict[str, Any]:
    return {
        "device": device,
        "physical_gpu_uuid": physical_gpu_uuid,
        "pid": os.getpid(),
        "engine_nonce": engine_nonce,
        "engine_started_ns": engine_started_ns,
        "adapter_object_identity": id(adapter),
        "engine_object_identity": id(adapter._view.llm_engine),
        "engine_reloaded": False,
    }


def _assert_engine_continuity(adapter: Any, expected: dict[str, Any]) -> None:
    observed = {
        "pid": os.getpid(),
        "adapter_object_identity": id(adapter),
        "engine_object_identity": id(adapter._view.llm_engine),
    }
    if any(observed[field] != expected.get(field) for field in observed):
        raise RuntimeError("retained engine process/object continuity changed")


def _validate_sha256(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"integrated transaction {field} must be a lowercase SHA-256")
    return value


def _validate_integrated_transaction_command(
    *,
    base_config: dict[str, Any],
    command: dict[str, Any],
    selected_load_sha256: str,
) -> tuple[dict[str, Any], dict[str, str]]:
    """Bind v10 to its pre-GPU authorization and completed sanity guards."""

    if command.get("schema_version") != "sloforge.branchfabric.integrated-transaction-command/v1":
        raise ValueError("integrated transaction command schema differs")
    effective = command.get("effective_config")
    if not isinstance(effective, dict):
        raise ValueError("integrated transaction command lacks a complete effective_config")
    selection_sha256 = _validate_sha256(command.get("selection_sha256"), field="selection hash")
    if selection_sha256 != selected_load_sha256:
        raise ValueError("transaction selection hash differs from final-reset authorization")
    evidence_hashes = {
        "authorization_artifact_hash": _validate_sha256(
            command.get("authorization_artifact_hash"), field="authorization artifact hash"
        ),
        "sanity_result_sha256": _validate_sha256(
            command.get("sanity_result_sha256"), field="sanity result hash"
        ),
    }
    if evidence_hashes["authorization_artifact_hash"] != base_config.get(
        "authorization_artifact_hash"
    ):
        raise ValueError("transaction authorization hash differs from the base config")
    for field in (
        "schema_version",
        "attempt_id",
        "seed",
        "gpu_memory_utilization",
        "serving_prompt_tokens",
        "serving_output_tokens",
    ):
        if field not in base_config or effective.get(field) != base_config[field]:
            raise ValueError(f"late effective config changed immutable field {field}")
    if effective.get("serving_methodology") != "v10-global-capacity":
        raise ValueError("late effective config does not select the frozen v10 methodology")
    configured_selection = effective.get("calibration_selected_load_sha256")
    if configured_selection is not None and configured_selection != selection_sha256:
        raise ValueError("late effective config selected-load binding differs")
    issued_ns = command.get("issued_at_monotonic_ns")
    if isinstance(issued_ns, bool) or not isinstance(issued_ns, int) or issued_ns <= 0:
        raise ValueError("integrated transaction command timestamp is invalid")
    if effective != base_config:
        raise ValueError("preauthorized v10 effective config must equal its immutable base config")
    return effective, evidence_hashes


def _run_measured_v10_transaction(
    *,
    role: str,
    work_root: Path,
    operation: Callable[[], dict[str, Any]],
    compilation_capture_factory: Callable[[], Any],
) -> dict[str, Any]:
    """Observe both retained engines for deferred compilation during v10."""

    if role not in {"serving", "rollout"}:
        raise ValueError("measured v10 compilation observation requires a worker role")
    capture = compilation_capture_factory()
    started_ns = time.monotonic_ns()
    try:
        with capture:
            result = operation()
    finally:
        ended_ns = time.monotonic_ns()
        raw_events = getattr(capture, "events", None)
        capture_buffer_valid = isinstance(raw_events, list) and all(
            isinstance(event, str) for event in raw_events
        )
        events = tuple(raw_events) if capture_buffer_valid else ()
        observation = {
            "schema_version": (
                "sloforge.branchfabric.measured-transaction-compilation-observation/v1"
            ),
            "source": "bounded-python-logging-handler",
            "role": role,
            "interval_start_ns": started_ns,
            "interval_end_ns": ended_ns,
            "capture_buffer_valid": capture_buffer_valid,
            "events": events,
            "no_deferred_compilation_event": capture_buffer_valid and not events,
            "passed": capture_buffer_valid and not events,
        }
        _write_new(work_root / "telemetry/measured-transaction-compilation.json", observation)
    if not isinstance(result, dict):
        raise TypeError("measured v10 transaction did not return a JSON object")
    result["measured_transaction_compilation_observation"] = observation
    if observation["passed"] is not True:
        raise RuntimeError("measured v10 transaction observed deferred compilation")
    return result


def _run_integrated_calibration_phase(
    *,
    adapter: Any,
    config: dict[str, Any],
    inputs: dict[str, Any],
    role: str,
    physical_gpu_uuid: str,
    barrier_root: Path,
    model_load_started_ns: int,
    model_ready_ns: int,
) -> tuple[dict[str, Any], _CacheSaltEngine, dict[str, Any]]:
    """Warm, calibrate, and hand one unchanged engine into the v10 run."""

    from gpu_capacity_calibration_worker import (
        _collect_engine_readiness,
        _CompilationLogCapture,
        _execute_probe,
        _reset_probe_state,
    )

    device = "gpu0" if role == "serving" else "gpu1"
    prefix = tuple(
        int(item) for item in inputs["prefix_token_ids"][: int(config["serving_prompt_tokens"])]
    )
    if not prefix:
        raise ValueError("integrated calibration serving prefix is empty")
    engine_nonce = hashlib.sha256(
        (
            f"{config['attempt_id']}\0{role}\0{os.getpid()}\0"
            f"{model_load_started_ns}\0{id(adapter)}\0{id(adapter._view.llm_engine)}"
        ).encode()
    ).hexdigest()
    continuity = _engine_continuity(
        adapter,
        engine_nonce=engine_nonce,
        engine_started_ns=model_load_started_ns,
        device=device,
        physical_gpu_uuid=physical_gpu_uuid,
    )
    readiness = _collect_engine_readiness(
        adapter,
        device=device,
        physical_gpu_uuid=physical_gpu_uuid,
        prefix=prefix,
        output_tokens=int(config["serving_output_tokens"]),
        seed=int(config["seed"]),
        engine_started_ns=model_load_started_ns,
        model_ready_ns=model_ready_ns,
        timeout_seconds=float(config["probe_timeout_seconds"]),
    )
    readiness.update(
        {
            "role": role,
            **continuity,
            "runtime_evidence": {
                "packages": {
                    name: importlib.metadata.version(name)
                    for name in ("vllm", "torch", "transformers")
                },
                "torch_cuda": __import__("torch").version.cuda,
                "gpu_name": __import__("torch").cuda.get_device_name(0),
            },
            "rollouts_ready": False,
        }
    )
    if not readiness.get("passed"):
        raise RuntimeError("strict integrated engine readiness failed")
    _write_new(barrier_root / f"{role}.ready.json", readiness)

    salted_engine = _CacheSaltEngine(
        adapter._view.llm_engine,
        namespace=f"{config['attempt_id']}:{device}:calibration-and-serving",
    )
    capacity_root = barrier_root / "capacity"
    abort_path = barrier_root / "abort.json"
    final_reset_command = capacity_root / "final-reset.command.json"
    timeout_s = float(config.get("controller_timeout_seconds", config["maximum_wall_seconds"]))
    configured_probe_limit = _V10_SANITY_GUARD_COUNT
    final_reset_payload: dict[str, Any] | None = None
    for probe_index in range(configured_probe_limit):
        command_path = capacity_root / f"probe-{probe_index:02d}.command.json"
        selected, command = _wait_for_any(
            (command_path, final_reset_command, abort_path),
            timeout_s=timeout_s,
            phase=f"v10 stale-calibration sanity command {probe_index}",
        )
        if selected == abort_path:
            raise RuntimeError("integrated coordinator aborted before v10")
        if selected == final_reset_command:
            final_reset_payload = command
            break
        if command.get("schema_version") != "sloforge.branchfabric.capacity-probe-command/v1":
            raise ValueError("v10 sanity probe command schema differs")
        plan = command.get("plan")
        if not isinstance(plan, dict):
            raise ValueError("integrated capacity probe command lacks its plan")
        with _CompilationLogCapture() as probe_compilation:
            raw = _execute_probe(
                salted_engine,
                plan_payload=plan,
                device=device,
                prefix=prefix,
                output_tokens=int(config["serving_output_tokens"]),
                seed=int(config["seed"]),
                probe_timeout_seconds=float(config["probe_timeout_seconds"]),
                tail_drain_seconds=float(config["tail_drain_seconds"]),
                producer_queue_capacity=int(config["producer_queue_capacity"]),
            )
        raw["compilation_observation"] = {
            "source": "bounded-python-logging-handler",
            "events": tuple(probe_compilation.events),
            "no_deferred_compilation_event": not probe_compilation.events,
        }
        if probe_compilation.events:
            raise RuntimeError(
                f"probe {raw['probe_id']} observed deferred vLLM compilation after readiness"
            )
        _assert_engine_continuity(adapter, continuity)
        raw["engine_continuity"] = continuity
        raw["cache_salt_evidence"] = salted_engine.evidence()
        raw["prefix_cache_policy"] = {
            "enabled_for_rollout_semantics": True,
            "calibration_reuse_prevented": True,
            "method": "per-request-vllm-cache-salt-sha256",
            "evidence_sha256": raw["cache_salt_evidence"]["request_salt_map_sha256"],
        }
        _write_new(capacity_root / f"probe-{probe_index:02d}.{device}.raw.json", raw)
        reset = _reset_probe_state(adapter, probe_id=str(raw["probe_id"]))
        _assert_engine_continuity(adapter, continuity)
        reset.update(continuity)
        _write_new(capacity_root / f"probe-{probe_index:02d}.{device}.reset.json", reset)
    if final_reset_payload is None:
        selected, final_reset_payload = _wait_for_any(
            (final_reset_command, abort_path),
            timeout_s=timeout_s,
            phase="integrated final sanity reset command",
        )
        if selected == abort_path:
            raise RuntimeError("integrated coordinator aborted before final reset")
    if (
        final_reset_payload.get("schema_version")
        != "sloforge.branchfabric.integrated-final-reset-command/v1"
    ):
        raise ValueError("integrated final reset command schema differs")
    selected_load_sha256 = _validate_sha256(
        final_reset_payload.get("selected_load_sha256"), field="final reset selected-load hash"
    )
    authorization_artifact_hash = _validate_sha256(
        final_reset_payload.get("authorization_artifact_hash"),
        field="final reset authorization artifact hash",
    )
    if authorization_artifact_hash != config.get("authorization_artifact_hash"):
        raise ValueError("final reset authorization differs from the immutable config")
    final_reset = _reset_probe_state(adapter, probe_id="final-calibration-reset")
    _assert_engine_continuity(adapter, continuity)
    final_reset.update(continuity)
    final_reset["selected_load_sha256"] = selected_load_sha256
    final_reset["authorization_artifact_hash"] = authorization_artifact_hash
    final_reset["cache_salt_evidence"] = salted_engine.evidence()
    _write_new(capacity_root / f"final-reset.{device}.json", final_reset)

    selected, transaction_command = _wait_for_any(
        (barrier_root / "transaction.command.json", abort_path),
        timeout_s=timeout_s,
        phase="integrated transaction authorization",
    )
    if selected == abort_path:
        raise RuntimeError("integrated coordinator aborted before transaction authorization")
    effective_config, evidence_hashes = _validate_integrated_transaction_command(
        base_config=config,
        command=transaction_command,
        selected_load_sha256=selected_load_sha256,
    )
    _assert_engine_continuity(adapter, continuity)
    handoff = {
        "schema_version": "sloforge.branchfabric.retained-engine-handoff/v1",
        "device": device,
        "role": role,
        **continuity,
        "selected_load_sha256": selected_load_sha256,
        "authorization_artifact_hash": authorization_artifact_hash,
        "pre_gpu_evidence_hashes": evidence_hashes,
        "final_request_state": final_reset["state_after"],
        "cache_salt_evidence": salted_engine.evidence(),
        "passed": True,
    }
    return effective_config, salted_engine, handoff


def _stage_row(stage: str, start_ns: int, end_ns: int, **attributes: Any) -> dict[str, Any]:
    if end_ns < start_ns:
        raise ValueError("stage end precedes start")
    return {
        "stage": stage,
        "start_ns": start_ns,
        "end_ns": end_ns,
        "duration_ns": end_ns - start_ns,
        **attributes,
    }


def _causalize_stages(
    rows: Sequence[dict[str, Any]], *, timeline_start_ns: int, timeline_end_ns: int
) -> list[dict[str, Any]]:
    """Attribute orchestration gaps to adjacent stages without hiding them."""

    if timeline_end_ns < timeline_start_ns or not rows:
        raise ValueError("causal timeline requires a nonempty positive ordered range")
    ordered = sorted(rows, key=lambda item: (int(item["start_ns"]), int(item["end_ns"])))
    cursor = timeline_start_ns
    result: list[dict[str, Any]] = []
    for row in ordered:
        measured_start = int(row["start_ns"])
        measured_end = int(row["end_ns"])
        if measured_start < cursor or measured_end < measured_start:
            raise ValueError("measured critical-path stages overlap or run backwards")
        if measured_end > timeline_end_ns:
            raise ValueError("measured critical-path stage extends beyond its terminal event")
        attributes = {
            key: value
            for key, value in row.items()
            if key not in {"stage", "start_ns", "end_ns", "duration_ns"}
        }
        result.append(
            _stage_row(
                str(row["stage"]),
                cursor,
                measured_end,
                measured_start_ns=measured_start,
                measured_end_ns=measured_end,
                measured_duration_ns=measured_end - measured_start,
                orchestration_attributed_ns=measured_start - cursor,
                **attributes,
            )
        )
        cursor = measured_end
    if cursor != timeline_end_ns:
        raise ValueError("critical-path stages do not terminate at the declared causal event")
    return result


@dataclass(frozen=True, slots=True)
class _LedgerPage:
    logical_page_id: str
    segment_id: str
    logical_offset_bytes: int
    logical_bytes: int
    physical_bytes: int


class _WorkerMovementLedger:
    """Per-segment physical-touch ledger for the measured naive path.

    Copy records carry endpoint touches and exactly one link event. They are
    never accompanied by a second record for the same copy. Materialization
    records describe distinct buffers created before or after a link.
    """

    def __init__(
        self,
        *,
        page_order: tuple[tuple[str, int, int, tuple[str, ...]], ...],
        logical_token_bytes: int,
        physical_page_bytes: int,
        branch_group: str,
        device: str,
    ) -> None:
        from sloforge.helix.characterization.gpu_reclamation_accounting import (
            LogicalStateSegment,
            StateSegmentKind,
        )

        if min(logical_token_bytes, physical_page_bytes) <= 0:
            raise ValueError("movement ledger geometry must be positive")
        if not page_order:
            raise ValueError("movement ledger requires at least one canonical page")
        self.branch_group = branch_group
        self.device = device
        self._passes: list[Any] = []
        self._record_sequence = 0
        segment_totals: dict[str, int] = {}
        segment_branches: dict[str, str | None] = {}
        pages: dict[str, _LedgerPage] = {}
        for logical_page_id, _block_index, valid_tokens, owners in page_order:
            if logical_page_id in pages or not owners or len(owners) != len(set(owners)):
                raise ValueError("movement ledger page identity/ownership is invalid")
            if len(owners) > 1:
                segment_id = "kv.shared-root"
                branch_id = None
            else:
                branch_id = owners[0]
                segment_id = f"kv.private:{branch_id}"
            logical_bytes = valid_tokens * logical_token_bytes
            logical_offset = segment_totals.get(segment_id, 0)
            pages[logical_page_id] = _LedgerPage(
                logical_page_id=logical_page_id,
                segment_id=segment_id,
                logical_offset_bytes=logical_offset,
                logical_bytes=logical_bytes,
                physical_bytes=physical_page_bytes,
            )
            segment_totals[segment_id] = logical_offset + logical_bytes
            prior_branch = segment_branches.setdefault(segment_id, branch_id)
            if prior_branch != branch_id:
                raise ValueError("movement ledger segment has inconsistent branch ownership")
        self._pages = pages
        self.physical_source_bytes = sum(page.physical_bytes for page in pages.values())
        self.logical_segments = tuple(
            LogicalStateSegment(
                segment_id=segment_id,
                branch_group_id=branch_group,
                kind=(
                    StateSegmentKind.SHARED_KV if branch_id is None else StateSegmentKind.PRIVATE_KV
                ),
                logical_bytes=segment_totals[segment_id],
                branch_id=branch_id,
            )
            for segment_id, branch_id in segment_branches.items()
        )

    @staticmethod
    def _bytes(*, basis: str, logical_bytes: int, physical_bytes: int, multiplier: int) -> int:
        if multiplier < 0:
            raise ValueError("movement byte multiplier cannot be negative")
        if basis == "none":
            return 0
        if basis == "logical":
            return logical_bytes * multiplier
        if basis == "physical":
            return physical_bytes * multiplier
        raise ValueError(f"unknown movement byte basis {basis!r}")

    def _selected_ranges(
        self, logical_page_ids: set[str] | None
    ) -> tuple[tuple[str, int, int, int], ...]:
        selected_ids = set(self._pages) if logical_page_ids is None else logical_page_ids
        if not selected_ids or not selected_ids.issubset(self._pages):
            raise ValueError("movement pass selects an empty or unknown canonical page set")
        by_segment: dict[str, list[_LedgerPage]] = {}
        for page_id in self._pages:
            if page_id in selected_ids:
                page = self._pages[page_id]
                by_segment.setdefault(page.segment_id, []).append(page)
        ranges: list[tuple[str, int, int, int]] = []
        for segment_id, pages in by_segment.items():
            ordered = sorted(pages, key=lambda item: item.logical_offset_bytes)
            logical_offset = ordered[0].logical_offset_bytes
            logical_bytes = sum(item.logical_bytes for item in ordered)
            expected_end = logical_offset
            for page in ordered:
                if page.logical_offset_bytes != expected_end:
                    raise ValueError("movement pass selects a non-contiguous segment range")
                expected_end += page.logical_bytes
            ranges.append(
                (
                    segment_id,
                    logical_offset,
                    logical_bytes,
                    sum(item.physical_bytes for item in ordered),
                )
            )
        return tuple(sorted(ranges))

    def record(
        self,
        *,
        label: str,
        operation: Any,
        source_memory: Any,
        destination_memory: Any,
        start_ns: int,
        end_ns: int,
        logical_page_ids: set[str] | None = None,
        read_basis: str = "none",
        read_multiplier: int = 1,
        write_basis: str = "none",
        write_multiplier: int = 1,
        transfer_direction: Any = None,
        transfer_basis: str = "none",
        transfer_multiplier: int = 1,
        checksum_basis: str = "none",
        checksum_multiplier: int = 1,
        temporary_basis: str = "none",
        temporary_multiplier: int = 1,
        temporary_memory: Any = None,
        required_unavoidable: bool = False,
    ) -> None:
        from sloforge.helix.characterization.gpu_reclamation_accounting import (
            MemoryDomain,
            StatePassRecord,
            TransferDirection,
        )

        if end_ns < start_ns:
            raise ValueError("movement record end precedes start")
        direction = TransferDirection.NONE if transfer_direction is None else transfer_direction
        for segment_id, logical_offset, logical_bytes, physical_bytes in self._selected_ranges(
            logical_page_ids
        ):
            self._record_sequence += 1
            record_id = f"{self._record_sequence:04d}:{label}:{segment_id}"
            read_bytes = self._bytes(
                basis=read_basis,
                logical_bytes=logical_bytes,
                physical_bytes=physical_bytes,
                multiplier=read_multiplier,
            )
            write_bytes = self._bytes(
                basis=write_basis,
                logical_bytes=logical_bytes,
                physical_bytes=physical_bytes,
                multiplier=write_multiplier,
            )
            transfer_bytes = self._bytes(
                basis=transfer_basis,
                logical_bytes=logical_bytes,
                physical_bytes=physical_bytes,
                multiplier=transfer_multiplier,
            )
            checksum_bytes = self._bytes(
                basis=checksum_basis,
                logical_bytes=logical_bytes,
                physical_bytes=physical_bytes,
                multiplier=checksum_multiplier,
            )
            temporary_bytes = self._bytes(
                basis=temporary_basis,
                logical_bytes=logical_bytes,
                physical_bytes=physical_bytes,
                multiplier=temporary_multiplier,
            )
            self._passes.append(
                StatePassRecord(
                    record_id=record_id,
                    state_segment=segment_id,
                    branch_group=self.branch_group,
                    operation=operation,
                    source_memory=source_memory,
                    destination_memory=destination_memory,
                    bytes_read=read_bytes,
                    bytes_written=write_bytes,
                    transfer_direction=direction,
                    transfer_bytes=transfer_bytes,
                    checksum_bytes=checksum_bytes,
                    logical_offset_bytes=logical_offset,
                    logical_bytes=logical_bytes,
                    start_ns=start_ns,
                    end_ns=end_ns,
                    device=self.device,
                    temporary_allocation_bytes=temporary_bytes,
                    temporary_allocation_memory=(
                        None if temporary_bytes == 0 else temporary_memory
                    ),
                    temporary_allocation_id=(
                        None if temporary_bytes == 0 else f"allocation:{record_id}"
                    ),
                    required_unavoidable=required_unavoidable,
                    read_event_id=None if read_bytes == 0 else f"read:{record_id}",
                    write_event_id=None if write_bytes == 0 else f"write:{record_id}",
                    transfer_event_id=(None if transfer_bytes == 0 else f"transfer:{record_id}"),
                )
            )
        if temporary_basis != "none" and temporary_memory in (None, MemoryDomain.NONE):
            raise ValueError("temporary movement buffer lacks a concrete memory domain")

    def report(self) -> Any:
        from sloforge.helix.characterization.gpu_reclamation_accounting import (
            build_state_movement_report,
        )

        return build_state_movement_report(
            logical_segments=self.logical_segments,
            passes=tuple(self._passes),
        )


def _source_runtime_block_id(block_index: int) -> str:
    if block_index < 0:
        raise ValueError("source block index cannot be negative")
    return f"vllm:kv-group:0:block:{block_index}"


def _validate_exact_source_release(capture_evidence: Any, release_evidence: Any) -> dict[str, Any]:
    """Validate and summarize exact native BlockPool release evidence."""

    expected = {
        _source_runtime_block_id(int(binding.source.block_index)): binding.source
        for binding in capture_evidence.bindings
    }
    if not expected or len(expected) != len(capture_evidence.bindings):
        raise RuntimeError("capture evidence does not identify unique source runtime blocks")
    if set(release_evidence.requested_block_ids) != set(expected):
        raise RuntimeError("release evidence does not cover the exact captured source block set")
    observed = {block.runtime_block_id: block for block in release_evidence.blocks}
    if set(observed) != set(expected) or len(observed) != len(release_evidence.blocks):
        raise RuntimeError("release evidence contains missing or duplicate source blocks")
    failures: list[str] = []
    for runtime_block_id, source in expected.items():
        block = observed[runtime_block_id]
        if int(block.block_index) != int(source.block_index):
            failures.append(f"{runtime_block_id}:index")
        if int(block.allocation_epoch) != int(source.allocation_epoch):
            failures.append(f"{runtime_block_id}:allocation-epoch")
        if int(block.native_refcount) != 0:
            failures.append(f"{runtime_block_id}:refcount")
        if bool(block.block_hash_present):
            failures.append(f"{runtime_block_id}:hash")
        if not bool(block.allocator_available) or bool(block.is_null):
            failures.append(f"{runtime_block_id}:allocator")
    if failures:
        raise RuntimeError("captured source blocks were not fully released: " + ",".join(failures))
    if int(release_evidence.pool_free_block_count) != int(release_evidence.pool_usable_block_count):
        raise RuntimeError("source release did not recover the complete usable KV block pool")
    return {
        "exact_source_block_count": len(expected),
        "all_native_refcounts_zero": True,
        "all_allocator_available": True,
        "all_hashes_cleared": True,
        "allocation_epochs_match_capture": True,
        "pool_free_block_count": int(release_evidence.pool_free_block_count),
        "pool_usable_block_count": int(release_evidence.pool_usable_block_count),
        "full_free_pool_recovered": True,
    }


def _runtime_capture_inputs(
    adapter: Any, branches: tuple[str, ...], *, parent_logical_branch_id: str
) -> tuple[Any, ...]:
    """Read live scheduler state through each session's exact runtime identity."""

    from sloforge.continuum.adapters.vllm_reclamation import RuntimeBranchCaptureInput

    view = adapter._view
    result: list[RuntimeBranchCaptureInput] = []
    for branch_id in branches:
        session = adapter._sessions[branch_id]
        runtime_request_id = session.runtime_request_id
        if not runtime_request_id:
            raise RuntimeError(
                f"live rollout request lacks a runtime identity before capture: {branch_id}"
            )
        request = view.request_object(runtime_request_id)
        if request is None:
            raise RuntimeError(f"live rollout request disappeared before capture: {branch_id}")
        result.append(
            RuntimeBranchCaptureInput(
                logical_branch_id=branch_id,
                parent_logical_branch_id=parent_logical_branch_id,
                token_ids=tuple(int(item) for item in request.all_token_ids),
                computed_tokens=int(request.num_computed_tokens),
                source_block_indices=tuple(
                    block.block_id for block in view.request_blocks(runtime_request_id)[0]
                ),
            )
        )
    return tuple(result)


def _add_restore_requests(
    engine: Any,
    branch_tables: Sequence[Any],
    *,
    source_sampling_by_branch: dict[str, dict[str, Any]],
) -> tuple[dict[str, str], dict[str, str]]:
    """Return separate logical-to-internal and logical-to-external ID maps."""

    internal_ids: dict[str, str] = {}
    external_ids: dict[str, str] = {}
    for table in branch_tables:
        logical_id = str(table.logical_branch_id)
        external_id = f"{logical_id}@restore-1"
        params = _sampling_params(
            max_tokens=8,
            seed=int(source_sampling_by_branch[logical_id]["effective_seed"]),
        )
        internal_id = engine.add_request(
            external_id,
            {"prompt_token_ids": list(table.token_ids)},
            params,
        )
        if not isinstance(internal_id, str) or not internal_id:
            raise RuntimeError(f"restore request {logical_id!r} lacks an internal identity")
        if internal_id in internal_ids.values():
            raise RuntimeError("vLLM returned duplicate internal restore request IDs")
        internal_ids[logical_id] = internal_id
        external_ids[logical_id] = external_id
    return internal_ids, external_ids


def _run_rollout_transaction(
    adapter: Any,
    config: dict[str, Any],
    inputs: dict[str, Any],
    prepared: dict[str, Any],
    trigger_ns: int,
    restore_complete_barrier: Path,
    barrier_root: Path | None = None,
    serving_engine: Any | None = None,
) -> dict[str, Any]:
    import torch  # type: ignore[import-not-found]

    from sloforge.continuum.adapters.vllm_reclamation import (
        Vllm0230RestoreStager,
        build_canonical_capture_plan,
        build_host_transport_state,
        convert_canonical_device_to_native_pages,
        copy_canonical_device_to_host,
        copy_host_transport_to_canonical_device,
        read_native_pages_to_gpu_intermediate,
        transform_native_read_to_canonical_device,
        write_native_pages_to_destination,
    )
    from sloforge.helix.characterization.gpu_reclamation_accounting import (
        MemoryDomain,
        StatePassOperation,
        TransferDirection,
    )
    from sloforge.helix.characterization.gpu_reclamation_instrumentation import (
        CudaEventRecorder,
        CudaOperationKind,
        HostAllocationLedger,
        HostMemoryKind,
        summarize_cuda_operations,
    )

    stages: list[dict[str, Any]] = []
    phase_events: list[dict[str, Any]] = []
    tracing_level = str(config["tracing_level"])
    if tracing_level not in {"disabled", "minimal", "full"}:
        raise ValueError("unsupported Experiment 004 tracing level")
    cuda_recorder = CudaEventRecorder(
        logical_device=0,
        enabled=tracing_level != "disabled",
    )
    host_allocations = HostAllocationLedger()

    def phase(
        name: str,
        *,
        logical_bytes: int = 0,
        physical_bytes: int = 0,
        observed_ns: int | None = None,
    ) -> int:
        observed = time.monotonic_ns() if observed_ns is None else observed_ns
        phase_events.append(
            {
                "phase": name,
                "monotonic_timestamp_ns": observed,
                "logical_bytes": logical_bytes,
                "physical_bytes": physical_bytes,
            }
        )
        return observed

    v10_methodology = config.get("serving_methodology") == "v10-global-capacity"
    if v10_methodology:
        if barrier_root is None:
            raise ValueError("v10 rollout transaction requires its barrier root")
        trigger_payload = _wait_for(
            barrier_root / "v10-reclaim-trigger.json",
            timeout_s=float(config["maximum_wall_seconds"]),
        )
        if not bool(trigger_payload.get("overload_confirmed")):
            raise RuntimeError("v10 reclaim trigger lacks measured GPU0 overload evidence")
        trigger_ns = int(trigger_payload["triggered_ns"])
    else:
        _sleep_until(trigger_ns)
    reclaim_trigger_observed_ns = phase(
        "RECLAIM_TRIGGER" if v10_methodology else "HELIX_RECLAIM_TRIGGER",
        observed_ns=trigger_ns,
    )
    start = time.monotonic_ns()
    phase("ROLLOUT_ADMISSION_STOP")
    stages.append(_stage_row("admission_stop", start, time.monotonic_ns()))
    start = time.monotonic_ns()
    # Exclusive engine-step gate: run_concurrent_branches returned at a valid
    # token boundary and this worker performs no engine.step until release.
    if v10_methodology:
        phase("BRANCH_QUIESCE_BEGIN", observed_ns=start)
    else:
        phase("BRANCH_QUIESCE")
    quiesce_end = time.monotonic_ns()
    if v10_methodology:
        phase("BRANCH_QUIESCE_END", observed_ns=quiesce_end)
    stages.append(_stage_row("branch_quiesce", start, quiesce_end))

    branches = tuple(str(item) for item in prepared["branches"])
    view = adapter._view
    live_serving_engine = view.llm_engine if serving_engine is None else serving_engine
    if (
        isinstance(live_serving_engine, _CacheSaltEngine)
        and live_serving_engine.underlying_engine is not view.llm_engine
    ):
        raise RuntimeError("GPU1 serving proxy does not wrap the retained rollout engine")
    raw_source_runtime_request_ids = {
        branch_id: adapter._sessions[branch_id].runtime_request_id for branch_id in branches
    }
    if any(
        runtime_id is None or str(runtime_id) == ""
        for runtime_id in raw_source_runtime_request_ids.values()
    ):
        raise RuntimeError("source rollout runtime request identities are absent")
    source_runtime_request_ids = {
        branch_id: str(runtime_id)
        for branch_id, runtime_id in raw_source_runtime_request_ids.items()
    }
    if len(set(source_runtime_request_ids.values())) != len(source_runtime_request_ids):
        raise RuntimeError("source rollout runtime request identities are duplicated")
    source_sampling_by_branch: dict[str, dict[str, Any]] = {}
    for branch_id in branches:
        session = adapter._sessions[branch_id]
        source_sampling_by_branch[branch_id] = {
            "effective_seed": int(adapter._effective_seed(int(config["seed"]), session)),
            "temperature": 0.0,
            "ignore_eos": True,
        }
    tensors, geometry = _geometry(adapter)
    phase("STATE_CAPTURE_BEGIN")
    start = time.monotonic_ns()
    runtime_inputs = _runtime_capture_inputs(
        adapter,
        branches,
        parent_logical_branch_id=str(prepared["root_id"]),
    )
    stages.append(_stage_row("final_state_capture", start, time.monotonic_ns()))
    start = time.monotonic_ns()
    plan = build_canonical_capture_plan(
        branches=runtime_inputs,
        block_size_tokens=geometry.block_size_tokens,
        logical_token_bytes=geometry.logical_token_bytes,
        physical_page_bytes=geometry.physical_page_bytes,
        gpu_uuid=os.environ["SLOFORGE_EXP004_PHYSICAL_GPU_UUID"],
        allocation_epoch_by_block=dict(adapter._observer.block_epochs),
    )
    stages.append(_stage_row("delta_extraction", start, time.monotonic_ns()))
    logical_bytes = plan.logical_state_bytes
    physical_bytes = plan.physical_source_bytes
    ledger = _WorkerMovementLedger(
        page_order=plan.page_order,
        logical_token_bytes=geometry.logical_token_bytes,
        physical_page_bytes=geometry.physical_page_bytes,
        branch_group=str(prepared["root_reference_id"]),
        device=os.environ["SLOFORGE_EXP004_PHYSICAL_GPU_UUID"],
    )
    if sum(item.logical_bytes for item in ledger.logical_segments) != logical_bytes:
        raise RuntimeError("movement ledger logical segments differ from capture plan")
    if ledger.physical_source_bytes != physical_bytes:
        raise RuntimeError("movement ledger physical pages differ from capture plan")
    if {item.branch_id for item in ledger.logical_segments if item.branch_id is not None} != set(
        branches
    ) or sum(item.branch_id is None for item in ledger.logical_segments) != 1:
        raise RuntimeError(
            "movement ledger does not cover one shared root and every private branch"
        )

    start = time.monotonic_ns()
    native_read = cuda_recorder.measure(
        operation_id="capture-source-native-read",
        kind=CudaOperationKind.TRANSFORM,
        operation=lambda: read_native_pages_to_gpu_intermediate(
            tensors, geometry, page_order=plan.page_order
        ),
        stream_id="cuda:0/default",
        bytes=physical_bytes,
        byte_semantics="physical source-page extent",
    )
    ended = time.monotonic_ns()
    stages.append(_stage_row("source_layout_read", start, ended))
    ledger.record(
        label="capture-source-native-read",
        operation=StatePassOperation.READ,
        source_memory=MemoryDomain.SOURCE_GPU_NATIVE_PAGED,
        destination_memory=MemoryDomain.GPU_TRANSFORM_BUFFER,
        start_ns=start,
        end_ns=ended,
        read_basis="physical",
        write_basis="physical",
        temporary_basis="physical",
        temporary_memory=MemoryDomain.GPU_TRANSFORM_BUFFER,
        required_unavoidable=True,
    )
    # ``read_native_pages_to_gpu_intermediate`` first materializes the
    # index_select result, then movedim exposes the requested page-major view,
    # and finally contiguous() performs a second full physical materialization.
    # Keep that repack separate from the native-pool read so the naive path's
    # avoidable traffic is visible to the optimizer and Amdahl analysis.
    ledger.record(
        label="capture-native-axis-contiguous",
        operation=StatePassOperation.REPACK,
        source_memory=MemoryDomain.GPU_TRANSFORM_BUFFER,
        destination_memory=MemoryDomain.GPU_TRANSFORM_BUFFER,
        start_ns=start,
        end_ns=ended,
        read_basis="physical",
        write_basis="physical",
        temporary_basis="physical",
        temporary_memory=MemoryDomain.GPU_TRANSFORM_BUFFER,
    )
    phase("STATE_CAPTURE_END", logical_bytes=logical_bytes, physical_bytes=physical_bytes)
    phase("STATE_TRANSFORM_BEGIN", logical_bytes=logical_bytes, physical_bytes=physical_bytes)
    start = time.monotonic_ns()
    canonical_device = cuda_recorder.measure(
        operation_id="capture-native-to-canonical-transform",
        kind=CudaOperationKind.TRANSFORM,
        operation=lambda: transform_native_read_to_canonical_device(native_read, geometry),
        stream_id="cuda:0/default",
        bytes=logical_bytes,
        byte_semantics="canonical logical output extent",
    )
    ended = time.monotonic_ns()
    stages.append(_stage_row("state_transform", start, ended))
    # The straightforward converter performs three logical-byte
    # materializations: valid-token contiguous views, per-page layer stacks,
    # and the final flat canonical concatenation.
    ledger.record(
        label="capture-unpage-valid-tokens",
        operation=StatePassOperation.UNPAGE,
        source_memory=MemoryDomain.GPU_TRANSFORM_BUFFER,
        destination_memory=MemoryDomain.GPU_TRANSPORT_BUFFER,
        start_ns=start,
        end_ns=ended,
        read_basis="logical",
        write_basis="logical",
        temporary_basis="logical",
        temporary_memory=MemoryDomain.GPU_TRANSPORT_BUFFER,
    )
    ledger.record(
        label="capture-stack-layers",
        operation=StatePassOperation.REPACK,
        source_memory=MemoryDomain.GPU_TRANSPORT_BUFFER,
        destination_memory=MemoryDomain.GPU_TRANSFORM_BUFFER,
        start_ns=start,
        end_ns=ended,
        read_basis="logical",
        write_basis="logical",
        temporary_basis="logical",
        temporary_memory=MemoryDomain.GPU_TRANSFORM_BUFFER,
    )
    ledger.record(
        label="capture-concatenate-pages",
        operation=StatePassOperation.TRANSFORM,
        source_memory=MemoryDomain.GPU_TRANSFORM_BUFFER,
        destination_memory=MemoryDomain.GPU_TRANSPORT_BUFFER,
        start_ns=start,
        end_ns=ended,
        read_basis="logical",
        write_basis="logical",
        temporary_basis="logical",
        temporary_memory=MemoryDomain.GPU_TRANSPORT_BUFFER,
    )
    phase("STATE_TRANSFORM_END", logical_bytes=logical_bytes, physical_bytes=physical_bytes)
    phase("D2H_BEGIN", logical_bytes=logical_bytes, physical_bytes=logical_bytes)
    start = time.monotonic_ns()
    host_payload = cuda_recorder.measure(
        operation_id="capture-canonical-d2h",
        kind=CudaOperationKind.D2H,
        operation=lambda: copy_canonical_device_to_host(canonical_device, pin_memory=True),
        stream_id="cuda:0/default",
        bytes=logical_bytes,
        byte_semantics="D2H link payload",
        copy_sizes_bytes=(logical_bytes,),
    )
    ended = time.monotonic_ns()
    pinned_transport_allocated_ns = ended
    host_allocations.allocate(
        allocation_id="capture-pinned-transport",
        kind=HostMemoryKind.PINNED,
        purpose="canonical Continuum transport payload",
        bytes=logical_bytes,
        timestamp_ns=pinned_transport_allocated_ns,
    )
    stages.append(_stage_row("device_to_host", start, ended))
    ledger.record(
        label="capture-d2h",
        operation=StatePassOperation.D2H,
        source_memory=MemoryDomain.GPU_TRANSPORT_BUFFER,
        destination_memory=MemoryDomain.PINNED_HOST_TRANSPORT,
        start_ns=start,
        end_ns=ended,
        read_basis="logical",
        write_basis="logical",
        transfer_direction=TransferDirection.D2H,
        transfer_basis="logical",
        required_unavoidable=True,
    )
    phase("D2H_END", logical_bytes=logical_bytes, physical_bytes=logical_bytes)
    canonical_device = None
    native_read = None
    torch.cuda.empty_cache()
    phase("INTEGRITY_BEGIN", logical_bytes=logical_bytes, physical_bytes=physical_bytes)
    start = time.monotonic_ns()
    transport = build_host_transport_state(
        host_payload,
        geometry,
        page_order=plan.page_order,
        branch_tables=plan.branch_tables,
        identity=_identity(),
        verify=False,
    )
    ended = time.monotonic_ns()
    # build_host_transport_state creates one full payload bytes object. Page
    # hashing uses zero-copy memoryview slices, so there is no hidden page-sized
    # allocation.
    integrity_payload_allocation_id = "integrity-payload-bytes"
    host_allocations.allocate(
        allocation_id=integrity_payload_allocation_id,
        kind=HostMemoryKind.PAGEABLE,
        purpose="integrity aggregate payload bytes materialization",
        bytes=logical_bytes,
        timestamp_ns=start,
    )
    host_allocations.free(integrity_payload_allocation_id, timestamp_ns=ended)
    stages.append(_stage_row("integrity_generation", start, ended))
    # Manifest construction materializes one full pageable bytes object and
    # hashes the aggregate payload and every page once.
    ledger.record(
        label="capture-integrity-manifest",
        operation=StatePassOperation.CHECKSUM,
        source_memory=MemoryDomain.PINNED_HOST_TRANSPORT,
        destination_memory=MemoryDomain.PAGEABLE_HOST_BUFFER,
        start_ns=start,
        end_ns=ended,
        read_basis="logical",
        write_basis="logical",
        temporary_basis="logical",
        temporary_memory=MemoryDomain.PAGEABLE_HOST_BUFFER,
        required_unavoidable=True,
    )
    ledger.record(
        label="capture-integrity-hash-reads",
        operation=StatePassOperation.CHECKSUM,
        source_memory=MemoryDomain.PAGEABLE_HOST_BUFFER,
        destination_memory=MemoryDomain.NONE,
        start_ns=start,
        end_ns=ended,
        read_basis="logical",
        read_multiplier=2,
        checksum_basis="logical",
        checksum_multiplier=2,
        required_unavoidable=True,
    )
    phase("INTEGRITY_END", logical_bytes=logical_bytes, physical_bytes=physical_bytes)
    start = time.monotonic_ns()
    transport.verify()
    phase("STATE_PUBLISH", logical_bytes=logical_bytes, physical_bytes=physical_bytes)
    ended = time.monotonic_ns()
    publish_payload_allocation_id = "publish-payload-bytes"
    host_allocations.allocate(
        allocation_id=publish_payload_allocation_id,
        kind=HostMemoryKind.PAGEABLE,
        purpose="publish verification aggregate payload bytes",
        bytes=logical_bytes,
        timestamp_ns=start,
    )
    host_allocations.free(publish_payload_allocation_id, timestamp_ns=ended)
    stages.append(_stage_row("transport_publish", start, ended))
    # Publication aliases the pinned payload; it does not copy it into a
    # second store. Only the mandatory aggregate+per-page validation reads.
    ledger.record(
        label="transport-publish-validation",
        operation=StatePassOperation.VALIDATE,
        source_memory=MemoryDomain.PINNED_HOST_TRANSPORT,
        destination_memory=MemoryDomain.PAGEABLE_HOST_BUFFER,
        start_ns=start,
        end_ns=ended,
        read_basis="logical",
        write_basis="logical",
        temporary_basis="logical",
        temporary_memory=MemoryDomain.PAGEABLE_HOST_BUFFER,
        required_unavoidable=True,
    )
    ledger.record(
        label="transport-publish-hash-reads",
        operation=StatePassOperation.VALIDATE,
        source_memory=MemoryDomain.PAGEABLE_HOST_BUFFER,
        destination_memory=MemoryDomain.NONE,
        start_ns=start,
        end_ns=ended,
        read_basis="logical",
        read_multiplier=2,
        checksum_basis="logical",
        checksum_multiplier=2,
        required_unavoidable=True,
    )

    memory_before = adapter.inspect_gpu_memory_state(timeout_s=10.0)
    source_allocations = {
        (binding.source.block_index, binding.source.allocation_epoch)
        for binding in plan.capture_evidence.bindings
    }
    phase(
        "GPU1_STATE_RELEASE_BEGIN" if v10_methodology else "GPU_STATE_RELEASE_BEGIN",
        logical_bytes=logical_bytes,
        physical_bytes=physical_bytes,
    )
    start = time.monotonic_ns()
    for branch_id in branches:
        adapter.destroy_branch(branch_id, timeout_s=30.0)
    adapter.destroy_session(str(prepared["root_id"]), timeout_s=30.0)
    release_end_ns = phase(
        "GPU1_STATE_RELEASE_END" if v10_methodology else "GPU_STATE_RELEASE_END",
        logical_bytes=logical_bytes,
        physical_bytes=physical_bytes,
    )
    stages.append(_stage_row("runtime_state_release", start, release_end_ns))
    start = time.monotonic_ns()
    source_runtime_block_ids = tuple(
        sorted(
            _source_runtime_block_id(int(binding.source.block_index))
            for binding in plan.capture_evidence.bindings
        )
    )
    source_release_evidence = adapter.inspect_block_release_evidence(
        source_runtime_block_ids, timeout_s=30.0
    )
    source_release_summary = _validate_exact_source_release(
        plan.capture_evidence, source_release_evidence
    )
    memory_after = adapter.inspect_gpu_memory_state(timeout_s=10.0)
    if int(memory_after.kv_assigned_bytes) != 0:
        raise RuntimeError("warm vLLM allocator retained rollout-owned KV bytes")
    if int(memory_after.kv_pool_reserved_bytes) != int(memory_before.kv_pool_reserved_bytes):
        raise RuntimeError("warm role switch unexpectedly changed the vLLM KV pool reservation")
    reclaim_confirmed_ns = phase(
        "GPU1_HBM_RECLAIM_CONFIRMED" if v10_methodology else "HBM_RECLAIM_CONFIRMED",
        logical_bytes=logical_bytes,
        physical_bytes=physical_bytes,
    )
    stages.append(_stage_row("capacity_reclaim_confirmation", start, reclaim_confirmed_ns))
    start = time.monotonic_ns()

    if v10_methodology:
        assert barrier_root is not None
        from gpu_capacity_calibration_worker import _runtime_queue_state
        from gpu_reclamation_v10_serving import LiveV10Config, run_v10_gpu1

        serving = run_v10_gpu1(
            live_serving_engine,
            prefix=tuple(int(item) for item in inputs["prefix_token_ids"][:256]),
            params=_sampling_params(
                max_tokens=int(config["serving_output_tokens"]), seed=int(config["seed"])
            ),
            config=LiveV10Config.from_mapping(config),
            barriers=barrier_root,
            write_new=_write_new,
            runtime_queue_state=lambda: _runtime_queue_state(adapter),
        )
        serving_enable_ns = phase(
            "GPU1_SERVING_ENABLE",
            observed_ns=int(serving["enable_ack"]["observed_ns"]),
        )
    else:
        # v9 compatibility path: this independent GPU1 stream is intentionally
        # retained only for immutable historical reprocessing. It is not a
        # scientifically valid v10 serving methodology.
        serving = _run_raw_serving(
            live_serving_engine,
            prefix=tuple(int(item) for item in inputs["prefix_token_ids"][:256]),
            start_ns=trigger_ns,
            phases=(
                (
                    float(config["serving_spike_seconds"]),
                    float(config["gpu1_spike_request_rate_per_second"]),
                    "spike",
                ),
            ),
            output_tokens=int(config["serving_output_tokens"]),
            seed=int(config["seed"]),
            request_prefix="gpu1-serving",
        )
        serving_enable_ns = phase("SERVING_SECONDARY_ENABLE")
    stages.append(_stage_row("serving_secondary_enable", start, serving_enable_ns))
    stages = _causalize_stages(
        stages,
        timeline_start_ns=reclaim_trigger_observed_ns,
        timeline_end_ns=serving_enable_ns,
    )
    first_serving_ns = min(int(row["first_token_ns"]) for row in serving["requests"])
    phase_events.append(
        {
            "phase": (
                "GPU1_FIRST_USEFUL_SERVING_REQUEST"
                if v10_methodology
                else "GPU1_FIRST_SERVING_REQUEST"
            ),
            "monotonic_timestamp_ns": first_serving_ns,
            "logical_bytes": 0,
            "physical_bytes": 0,
        }
    )
    stages.append(_stage_row("first_useful_serving_request", serving_enable_ns, first_serving_ns))
    if not view.manager.reset_prefix_cache():
        raise RuntimeError("GPU1 temporary serving prefix cache did not drain before restore")
    if v10_methodology:
        assert barrier_root is not None
        restore_start = _wait_for(
            barrier_root / "v10-restore-start.json",
            timeout_s=float(config["maximum_wall_seconds"]),
        )
        restore_trigger_observed_ns = int(restore_start["observed_ns"])
    else:
        restore_trigger_observed_ns = None
    restore_trigger_ns = phase(
        "RESTORE_TRIGGER" if v10_methodology else "ROLLOUT_RESTORE_TRIGGER",
        logical_bytes=logical_bytes,
        physical_bytes=physical_bytes,
        observed_ns=restore_trigger_observed_ns,
    )

    restore_stages: list[dict[str, Any]] = []
    start = time.monotonic_ns()
    runtime_ids, output_ids = _add_restore_requests(
        view.llm_engine,
        transport.manifest.branches,
        source_sampling_by_branch=source_sampling_by_branch,
    )
    restore_stages.append(
        _stage_row("destination_request_construction", start, time.monotonic_ns())
    )

    start = time.monotonic_ns()
    transport.verify()
    ended = time.monotonic_ns()
    restore_verify_payload_id = "restore-verify-payload-bytes"
    host_allocations.allocate(
        allocation_id=restore_verify_payload_id,
        kind=HostMemoryKind.PAGEABLE,
        purpose="restore verification aggregate payload bytes",
        bytes=logical_bytes,
        timestamp_ns=start,
    )
    host_allocations.free(restore_verify_payload_id, timestamp_ns=ended)
    restore_stages.append(_stage_row("transport_layout_read_validation", start, ended))
    ledger.record(
        label="restore-transport-validation",
        operation=StatePassOperation.VALIDATE,
        source_memory=MemoryDomain.PINNED_HOST_TRANSPORT,
        destination_memory=MemoryDomain.PAGEABLE_HOST_BUFFER,
        start_ns=start,
        end_ns=ended,
        read_basis="logical",
        write_basis="logical",
        temporary_basis="logical",
        temporary_memory=MemoryDomain.PAGEABLE_HOST_BUFFER,
        required_unavoidable=True,
    )
    ledger.record(
        label="restore-transport-hash-reads",
        operation=StatePassOperation.VALIDATE,
        source_memory=MemoryDomain.PAGEABLE_HOST_BUFFER,
        destination_memory=MemoryDomain.NONE,
        start_ns=start,
        end_ns=ended,
        read_basis="logical",
        read_multiplier=2,
        checksum_basis="logical",
        checksum_multiplier=2,
        required_unavoidable=True,
    )

    phase("H2D_BEGIN", logical_bytes=logical_bytes, physical_bytes=logical_bytes)
    start = time.monotonic_ns()
    canonical_restore_device = cuda_recorder.measure(
        operation_id="restore-canonical-h2d",
        kind=CudaOperationKind.H2D,
        operation=lambda: copy_host_transport_to_canonical_device(transport, verify=False),
        stream_id="cuda:0/default",
        bytes=logical_bytes,
        byte_semantics="H2D link payload",
        copy_sizes_bytes=(logical_bytes,),
    )
    ended = time.monotonic_ns()
    restore_stages.append(_stage_row("host_to_device", start, ended))
    ledger.record(
        label="restore-h2d",
        operation=StatePassOperation.H2D,
        source_memory=MemoryDomain.PINNED_HOST_TRANSPORT,
        destination_memory=MemoryDomain.GPU_TRANSPORT_BUFFER,
        start_ns=start,
        end_ns=ended,
        read_basis="logical",
        write_basis="logical",
        transfer_direction=TransferDirection.H2D,
        transfer_basis="logical",
        temporary_basis="logical",
        temporary_memory=MemoryDomain.GPU_TRANSPORT_BUFFER,
        required_unavoidable=True,
    )
    phase("H2D_END", logical_bytes=logical_bytes, physical_bytes=logical_bytes)

    imported_subsets: set[str] = set()
    allocation_intervals: list[dict[str, Any]] = []
    write_intervals: list[dict[str, Any]] = []
    validate_intervals: list[dict[str, Any]] = []
    write_callback_sequence = 0
    validation_callback_sequence = 0
    import_validation_started_ns: int | None = None
    import_validation_ended_ns: int | None = None
    import_validation_allocation_id = "restore-import-verify-payload-bytes"

    def observe_allocation(logical_branch_id: str, started_ns: int, ended_ns: int) -> None:
        nonlocal import_validation_ended_ns
        if import_validation_ended_ns is None:
            import_validation_ended_ns = started_ns
            host_allocations.free(
                import_validation_allocation_id,
                timestamp_ns=max(started_ns, int(import_validation_started_ns or 0) + 1),
            )
        allocation_intervals.append(
            _stage_row(
                "destination_allocation_subset",
                started_ns,
                ended_ns,
                logical_branch_id=logical_branch_id,
            )
        )

    def write_pages(destinations: dict[str, int]) -> None:
        nonlocal write_callback_sequence
        callback_sequence = write_callback_sequence
        write_callback_sequence += 1
        selected_ids = set(destinations)
        subset_physical_bytes = len(destinations) * geometry.physical_page_bytes
        started = time.monotonic_ns()
        native_destination = cuda_recorder.measure(
            operation_id=f"restore-destination-conversion-{callback_sequence:02d}",
            kind=CudaOperationKind.TRANSFORM,
            operation=lambda: convert_canonical_device_to_native_pages(
                canonical_restore_device,
                transport,
                geometry,
                destination_block_indices=destinations,
                require_complete=False,
            ),
            stream_id="cuda:0/default",
            bytes=subset_physical_bytes,
            byte_semantics="native physical output extent",
        )
        converted = time.monotonic_ns()
        write_intervals.append(
            _stage_row(
                "destination_conversion_subset",
                started,
                converted,
                pages=len(destinations),
            )
        )
        # zeros + valid-token overlay, native-axis contiguous materialization,
        # and per-layer stacking are distinct physical touches.
        ledger.record(
            label="restore-zero-native-pages",
            operation=StatePassOperation.ALLOCATE,
            source_memory=MemoryDomain.NONE,
            destination_memory=MemoryDomain.GPU_TRANSFORM_BUFFER,
            start_ns=started,
            end_ns=converted,
            logical_page_ids=selected_ids,
            write_basis="physical",
            temporary_basis="physical",
            temporary_memory=MemoryDomain.GPU_TRANSFORM_BUFFER,
        )
        ledger.record(
            label="restore-overlay-valid-tokens",
            operation=StatePassOperation.UNPACK,
            source_memory=MemoryDomain.GPU_TRANSPORT_BUFFER,
            destination_memory=MemoryDomain.GPU_TRANSFORM_BUFFER,
            start_ns=started,
            end_ns=converted,
            logical_page_ids=selected_ids,
            read_basis="logical",
            write_basis="logical",
        )
        ledger.record(
            label="restore-native-axis-contiguous",
            operation=StatePassOperation.REPACK,
            source_memory=MemoryDomain.GPU_TRANSFORM_BUFFER,
            destination_memory=MemoryDomain.GPU_TRANSFORM_BUFFER,
            start_ns=started,
            end_ns=converted,
            logical_page_ids=selected_ids,
            read_basis="physical",
            write_basis="physical",
            temporary_basis="physical",
            temporary_memory=MemoryDomain.GPU_TRANSFORM_BUFFER,
        )
        ledger.record(
            label="restore-stack-native-pages",
            operation=StatePassOperation.REPAGE,
            source_memory=MemoryDomain.GPU_TRANSFORM_BUFFER,
            destination_memory=MemoryDomain.GPU_TRANSFORM_BUFFER,
            start_ns=started,
            end_ns=converted,
            logical_page_ids=selected_ids,
            read_basis="physical",
            write_basis="physical",
            temporary_basis="physical",
            temporary_memory=MemoryDomain.GPU_TRANSFORM_BUFFER,
        )
        write_started = time.monotonic_ns()
        cuda_recorder.measure(
            operation_id=f"restore-destination-write-{callback_sequence:02d}",
            kind=CudaOperationKind.OTHER,
            operation=lambda: write_native_pages_to_destination(
                native_destination, tensors, geometry
            ),
            stream_id="cuda:0/default",
            bytes=subset_physical_bytes,
            byte_semantics="native physical destination-write extent",
        )
        ended = time.monotonic_ns()
        write_intervals.append(
            _stage_row(
                "destination_native_write_subset", write_started, ended, pages=len(destinations)
            )
        )
        ledger.record(
            label="restore-destination-native-write",
            operation=StatePassOperation.WRITE,
            source_memory=MemoryDomain.GPU_TRANSFORM_BUFFER,
            destination_memory=MemoryDomain.DESTINATION_GPU_NATIVE_PAGED,
            start_ns=write_started,
            end_ns=ended,
            logical_page_ids=selected_ids,
            read_basis="physical",
            write_basis="physical",
            required_unavoidable=True,
        )
        native_destination = None
        imported_subsets.update(destinations)

    def validate_pages(destinations: dict[str, int]) -> bool:
        nonlocal validation_callback_sequence
        callback_sequence = validation_callback_sequence
        validation_callback_sequence += 1
        phase("STATE_VALIDATE_BEGIN", logical_bytes=logical_bytes, physical_bytes=physical_bytes)
        selected_ids = set(destinations)
        subset_descriptors = tuple(
            descriptor
            for descriptor in transport.manifest.pages
            if descriptor.logical_page_id in selected_ids
        )
        subset_logical_bytes = sum(item.payload_bytes for item in subset_descriptors)
        subset_physical_bytes = len(destinations) * geometry.physical_page_bytes
        started = time.monotonic_ns()
        selected = tuple(item for item in plan.page_order if item[0] in selected_ids)
        validation_page_order = tuple(
            (logical, destinations[logical], valid, owners)
            for logical, _source, valid, owners in selected
        )
        recaptured_native = cuda_recorder.measure(
            operation_id=f"validation-native-read-{callback_sequence:02d}",
            kind=CudaOperationKind.TRANSFORM,
            operation=lambda: read_native_pages_to_gpu_intermediate(
                tensors,
                geometry,
                page_order=validation_page_order,
            ),
            stream_id="cuda:0/default",
            bytes=subset_physical_bytes,
            byte_semantics="native physical validation-read extent",
        )
        native_read_ended = time.monotonic_ns()
        validate_intervals.append(
            _stage_row(
                "destination_validation_native_read_subset",
                started,
                native_read_ended,
                pages=len(destinations),
            )
        )
        ledger.record(
            label="validation-destination-native-read",
            operation=StatePassOperation.READ,
            source_memory=MemoryDomain.DESTINATION_GPU_NATIVE_PAGED,
            destination_memory=MemoryDomain.GPU_TRANSFORM_BUFFER,
            start_ns=started,
            end_ns=native_read_ended,
            logical_page_ids=selected_ids,
            read_basis="physical",
            write_basis="physical",
            temporary_basis="physical",
            temporary_memory=MemoryDomain.GPU_TRANSFORM_BUFFER,
            required_unavoidable=True,
        )
        # The validation gather has the same index_select -> movedim ->
        # contiguous implementation as source capture.  Account the second
        # physical materialization instead of hiding it in the native read.
        ledger.record(
            label="validation-native-axis-contiguous",
            operation=StatePassOperation.REPACK,
            source_memory=MemoryDomain.GPU_TRANSFORM_BUFFER,
            destination_memory=MemoryDomain.GPU_TRANSFORM_BUFFER,
            start_ns=started,
            end_ns=native_read_ended,
            logical_page_ids=selected_ids,
            read_basis="physical",
            write_basis="physical",
            temporary_basis="physical",
            temporary_memory=MemoryDomain.GPU_TRANSFORM_BUFFER,
        )
        transform_started = time.monotonic_ns()
        recaptured_device = cuda_recorder.measure(
            operation_id=f"validation-canonical-transform-{callback_sequence:02d}",
            kind=CudaOperationKind.TRANSFORM,
            operation=lambda: transform_native_read_to_canonical_device(
                recaptured_native, geometry
            ),
            stream_id="cuda:0/default",
            bytes=subset_logical_bytes,
            byte_semantics="canonical logical validation extent",
        )
        transform_ended = time.monotonic_ns()
        validate_intervals.append(
            _stage_row(
                "destination_validation_transform_subset",
                transform_started,
                transform_ended,
                pages=len(destinations),
            )
        )
        for label, operation, source, destination in (
            (
                "validation-unpage-valid-tokens",
                StatePassOperation.UNPAGE,
                MemoryDomain.GPU_TRANSFORM_BUFFER,
                MemoryDomain.GPU_TRANSPORT_BUFFER,
            ),
            (
                "validation-stack-layers",
                StatePassOperation.REPACK,
                MemoryDomain.GPU_TRANSPORT_BUFFER,
                MemoryDomain.GPU_TRANSFORM_BUFFER,
            ),
            (
                "validation-concatenate-pages",
                StatePassOperation.TRANSFORM,
                MemoryDomain.GPU_TRANSFORM_BUFFER,
                MemoryDomain.GPU_TRANSPORT_BUFFER,
            ),
        ):
            ledger.record(
                label=label,
                operation=operation,
                source_memory=source,
                destination_memory=destination,
                start_ns=transform_started,
                end_ns=transform_ended,
                logical_page_ids=selected_ids,
                read_basis="logical",
                write_basis="logical",
                temporary_basis="logical",
                temporary_memory=destination,
            )
        d2h_started = time.monotonic_ns()
        recaptured = cuda_recorder.measure(
            operation_id=f"validation-d2h-{callback_sequence:02d}",
            kind=CudaOperationKind.D2H,
            operation=lambda: copy_canonical_device_to_host(recaptured_device, pin_memory=False),
            stream_id="cuda:0/default",
            bytes=subset_logical_bytes,
            byte_semantics="validation D2H link payload",
            copy_sizes_bytes=(subset_logical_bytes,),
        )
        d2h_ended = time.monotonic_ns()
        recapture_allocation_id = f"validation-recapture-{callback_sequence:02d}"
        host_allocations.allocate(
            allocation_id=recapture_allocation_id,
            kind=HostMemoryKind.PAGEABLE,
            purpose="destination validation D2H recapture",
            bytes=subset_logical_bytes,
            timestamp_ns=d2h_ended,
        )
        validate_intervals.append(
            _stage_row(
                "destination_validation_d2h_subset",
                d2h_started,
                d2h_ended,
                pages=len(destinations),
            )
        )
        ledger.record(
            label="validation-d2h",
            operation=StatePassOperation.D2H,
            source_memory=MemoryDomain.GPU_TRANSPORT_BUFFER,
            destination_memory=MemoryDomain.PAGEABLE_HOST_BUFFER,
            start_ns=d2h_started,
            end_ns=d2h_ended,
            logical_page_ids=selected_ids,
            read_basis="logical",
            write_basis="logical",
            transfer_direction=TransferDirection.D2H,
            transfer_basis="logical",
            required_unavoidable=True,
        )
        host_validate_started = time.monotonic_ns()
        expected_parts = tuple(
            transport.payload[
                descriptor.payload_offset_bytes : descriptor.payload_offset_bytes
                + descriptor.payload_bytes
            ]
            for descriptor in transport.manifest.pages
            if descriptor.logical_page_id in destinations
        )
        expected = torch.cat(expected_parts)
        expected_allocated_ns = time.monotonic_ns()
        expected_allocation_id = f"validation-expected-{callback_sequence:02d}"
        host_allocations.allocate(
            allocation_id=expected_allocation_id,
            kind=HostMemoryKind.PAGEABLE,
            purpose="validation expected-byte materialization",
            bytes=len(expected),
            timestamp_ns=expected_allocated_ns,
        )
        valid = bool(torch.equal(recaptured, expected))
        expected = None
        expected_parts = None
        recaptured = None
        recaptured_device = None
        recaptured_native = None
        ended = max(time.monotonic_ns(), expected_allocated_ns + 1)
        host_allocations.free(
            expected_allocation_id,
            timestamp_ns=max(ended, expected_allocated_ns + 1),
        )
        host_allocations.free(
            recapture_allocation_id,
            timestamp_ns=max(ended, d2h_ended + 1),
        )
        validate_intervals.append(
            _stage_row(
                "destination_validation_host_compare_subset",
                host_validate_started,
                ended,
                pages=len(destinations),
            )
        )
        ledger.record(
            label="validation-expected-page-concatenation",
            operation=StatePassOperation.REPACK,
            source_memory=MemoryDomain.PINNED_HOST_TRANSPORT,
            destination_memory=MemoryDomain.PAGEABLE_HOST_BUFFER,
            start_ns=host_validate_started,
            end_ns=ended,
            logical_page_ids=selected_ids,
            read_basis="logical",
            write_basis="logical",
            temporary_basis="logical",
            temporary_memory=MemoryDomain.PAGEABLE_HOST_BUFFER,
        )
        ledger.record(
            label="validation-host-tensor-compare",
            operation=StatePassOperation.VALIDATE,
            source_memory=MemoryDomain.PAGEABLE_HOST_BUFFER,
            destination_memory=MemoryDomain.NONE,
            start_ns=host_validate_started,
            end_ns=ended,
            logical_page_ids=selected_ids,
            read_basis="logical",
            read_multiplier=2,
        )
        ledger.record(
            label="validation-recapture-host-lifetime",
            operation=StatePassOperation.ALLOCATE,
            source_memory=MemoryDomain.NONE,
            destination_memory=MemoryDomain.NONE,
            start_ns=d2h_ended,
            end_ns=ended,
            logical_page_ids=selected_ids,
            temporary_basis="logical",
            temporary_memory=MemoryDomain.PAGEABLE_HOST_BUFFER,
        )
        phase("STATE_VALIDATE_END", logical_bytes=logical_bytes, physical_bytes=physical_bytes)
        return valid

    phase("STATE_IMPORT_BEGIN", logical_bytes=logical_bytes, physical_bytes=physical_bytes)
    import_validation_started_ns = time.monotonic_ns()
    host_allocations.allocate(
        allocation_id=import_validation_allocation_id,
        kind=HostMemoryKind.PAGEABLE,
        purpose="restore import duplicate verification payload bytes",
        bytes=logical_bytes,
        timestamp_ns=import_validation_started_ns,
    )
    stager = Vllm0230RestoreStager(view.scheduler, view.manager)
    imported = stager.import_group(
        transport,
        runtime_request_ids=runtime_ids,
        write_pages=write_pages,
        validate_pages=validate_pages,
        expected_identity=_identity(),
        allocation_observer=observe_allocation,
    )
    if import_validation_ended_ns is None:
        raise RuntimeError("restore import did not expose its verification/allocation boundary")
    native_write_intervals = tuple(
        row for row in write_intervals if row.get("stage") == "destination_native_write_subset"
    )
    if not native_write_intervals:
        raise RuntimeError("restore import performed no destination native writes")
    phase(
        "DESTINATION_NATIVE_WRITE_BEGIN",
        logical_bytes=logical_bytes,
        physical_bytes=physical_bytes,
        observed_ns=min(int(row["start_ns"]) for row in native_write_intervals),
    )
    phase(
        "DESTINATION_NATIVE_WRITE_END",
        logical_bytes=logical_bytes,
        physical_bytes=physical_bytes,
        observed_ns=max(int(row["end_ns"]) for row in native_write_intervals),
    )
    restore_stages.append(
        _stage_row(
            "restore_import_validation",
            import_validation_started_ns,
            import_validation_ended_ns,
        )
    )
    ledger.record(
        label="restore-import-validation",
        operation=StatePassOperation.VALIDATE,
        source_memory=MemoryDomain.PINNED_HOST_TRANSPORT,
        destination_memory=MemoryDomain.PAGEABLE_HOST_BUFFER,
        start_ns=import_validation_started_ns,
        end_ns=import_validation_ended_ns,
        read_basis="logical",
        write_basis="logical",
        temporary_basis="logical",
        temporary_memory=MemoryDomain.PAGEABLE_HOST_BUFFER,
    )
    ledger.record(
        label="restore-import-hash-reads",
        operation=StatePassOperation.VALIDATE,
        source_memory=MemoryDomain.PAGEABLE_HOST_BUFFER,
        destination_memory=MemoryDomain.NONE,
        start_ns=import_validation_started_ns,
        end_ns=import_validation_ended_ns,
        read_basis="logical",
        read_multiplier=2,
        checksum_basis="logical",
        checksum_multiplier=2,
    )
    phase("STATE_IMPORT_END", logical_bytes=logical_bytes, physical_bytes=physical_bytes)
    if imported_subsets != {page.logical_page_id for page in transport.manifest.pages}:
        raise RuntimeError("restore callbacks did not import every logical page")
    restore_stages.extend(allocation_intervals)
    restore_stages.extend(write_intervals)
    restore_stages.extend(validate_intervals)
    destination_allocations = {
        (
            destination,
            int(adapter._observer.block_epochs.get(destination, -1)),
        )
        for destination in imported.logical_page_destinations.values()
    }
    if any(epoch < 0 for _block, epoch in destination_allocations):
        raise RuntimeError("restored pages lack runtime allocation generations")
    if source_allocations & destination_allocations:
        raise RuntimeError("restored pages reused a stale source allocation generation")

    phase("BRANCH_RESUME_BEGIN")
    first_tokens: dict[str, int] = {}
    first_token_timestamps_ns: dict[str, int] = {}
    continuation_counts: dict[str, int] = {output_id: 0 for output_id in output_ids.values()}
    all_branches_resumed_ns: int | None = None
    first_forward_start = time.monotonic_ns()
    deadline = first_forward_start + 120_000_000_000
    while min(continuation_counts.values()) < 8:
        if time.monotonic_ns() >= deadline:
            raise TimeoutError("restored rollout continuation did not complete")
        outputs = view.llm_engine.step()
        resumed_observed_ns = time.monotonic_ns()
        for output in outputs:
            runtime_id = str(getattr(output, "request_id", ""))
            if runtime_id not in continuation_counts:
                continue
            completions = getattr(output, "outputs", ())
            token_ids = tuple(int(item) for item in getattr(completions[0], "token_ids", ()))
            continuation_counts[runtime_id] = len(token_ids)
            if token_ids and runtime_id not in first_tokens:
                first_tokens[runtime_id] = token_ids[0]
                first_token_timestamps_ns[runtime_id] = resumed_observed_ns
                if len(first_tokens) == 1:
                    phase("FIRST_RESUMED_TOKEN", observed_ns=resumed_observed_ns)
                if len(first_tokens) == len(output_ids):
                    all_branches_resumed_ns = resumed_observed_ns
                    if v10_methodology:
                        phase("ALL_BRANCHES_RESUMED", observed_ns=resumed_observed_ns)
    if all_branches_resumed_ns is None:
        raise RuntimeError("not every restored branch emitted its first token")
    resume_complete_ns = phase(
        "ROLLOUT_CONTINUATION_COMPLETE" if v10_methodology else "ROLLOUT_RESUME_COMPLETE"
    )
    _write_new(
        restore_complete_barrier,
        {
            "schema_version": "sloforge.branchfabric.experiment-004-restore-complete/v1",
            "observed_at_monotonic_ns": resume_complete_ns,
        },
    )
    if len(first_tokens) != 8 or min(continuation_counts.values()) < 8:
        raise RuntimeError("not all restored rollout branches resumed")
    first_resumed_ns = min(first_token_timestamps_ns.values())
    restore_stages.append(
        _stage_row("scheduler_first_forward_and_token", first_forward_start, first_resumed_ns)
    )
    restore_stages = _causalize_stages(
        restore_stages,
        timeline_start_ns=restore_trigger_ns,
        timeline_end_ns=first_resumed_ns,
    )

    # A bounded greedy recompute control supplies an exact first-token oracle
    # without advancing or freeing the source checkpoint before reclamation.
    if not view.manager.reset_prefix_cache():
        raise RuntimeError("restored rollout cache did not drain before recompute control")
    recompute_started_ns = time.monotonic_ns()
    control_by_runtime: dict[str, str] = {}
    for table in transport.manifest.branches:
        control_id = f"{table.logical_branch_id}@recompute-control"
        control_params = _sampling_params(
            max_tokens=1,
            seed=int(source_sampling_by_branch[table.logical_branch_id]["effective_seed"]),
        )
        view.llm_engine.add_request(
            control_id,
            {"prompt_token_ids": list(table.token_ids)},
            control_params,
        )
        control_by_runtime[control_id] = table.logical_branch_id
    expected_first_tokens: dict[str, int] = {}
    control_deadline = recompute_started_ns + 180_000_000_000
    while len(expected_first_tokens) < 8:
        if time.monotonic_ns() >= control_deadline:
            raise TimeoutError("recompute first-token control did not complete")
        for output in view.llm_engine.step():
            runtime_id = str(getattr(output, "request_id", ""))
            logical_id = control_by_runtime.get(runtime_id)
            if logical_id is None:
                continue
            completions = getattr(output, "outputs", ())
            token_ids = tuple(int(item) for item in getattr(completions[0], "token_ids", ()))
            if token_ids:
                expected_first_tokens[logical_id] = token_ids[0]
    observed_first_tokens = {
        logical_id: first_tokens[output_id] for logical_id, output_id in output_ids.items()
    }
    if observed_first_tokens != expected_first_tokens:
        raise RuntimeError("restored first tokens differ from the greedy recompute control")
    recompute_ended_ns = time.monotonic_ns()
    from sloforge.helix.characterization.gpu_reclamation import (
        ExperimentPhase,
        ExperimentPhaseMarker,
    )
    from sloforge.helix.characterization.gpu_reclamation_trace import (
        Experiment004TraceIdentity,
        phase_markers_to_trace_events,
    )

    # v10 adds a richer phase vocabulary while the stable v1 Perfetto schema
    # remains backwards compatible. Keep the v1 subset in this legacy trace;
    # the v10 postprocessor writes the complete v10 timeline separately.
    known_experiment_phases = {phase.value for phase in ExperimentPhase}
    markers = tuple(
        ExperimentPhaseMarker(
            phase=ExperimentPhase(str(item["phase"])),
            monotonic_timestamp_ns=int(item["monotonic_timestamp_ns"]),
            logical_bytes=int(item["logical_bytes"]),
            physical_bytes=int(item["physical_bytes"]),
            attributes={"attempt_id": str(config["attempt_id"])},
        )
        for item in phase_events
        if str(item["phase"]) in known_experiment_phases
    )
    minimal_trace_events = phase_markers_to_trace_events(
        markers,
        identity=Experiment004TraceIdentity(
            trace_id=f"exp004:{config['attempt_id']}:rollout",
            session_id=str(prepared["root_id"]),
            branch_group_id=str(prepared["root_reference_id"]),
            logical_state_id=f"exp004-state:{config['attempt_id']}",
            tenant_id="branchfabric-exp004",
            security_domain="branchfabric-exp004",
            device=os.environ["SLOFORGE_EXP004_PHYSICAL_GPU_UUID"],
            process_id=os.getpid(),
        ),
    )
    minimal_trace_jsonl = "".join(
        _canonical_bytes(event).decode("utf-8") for event in minimal_trace_events
    )
    transport_manifest = transport.manifest.model_dump(mode="json")
    host_resident_transport_metadata_bytes = len(_canonical_bytes(transport_manifest))
    transport_payload_sha256 = transport.manifest.payload_sha256
    recompute_tokens_prefilled = sum(table.computed_tokens for table in transport.manifest.branches)
    canonical_restore_device = None
    transport = None
    host_payload = None
    pinned_freed_ns = time.monotonic_ns()
    host_allocations.free(
        "capture-pinned-transport",
        timestamp_ns=max(pinned_freed_ns, pinned_transport_allocated_ns + 1),
    )
    ledger.record(
        label="capture-pinned-transport-lifetime",
        operation=StatePassOperation.ALLOCATE,
        source_memory=MemoryDomain.NONE,
        destination_memory=MemoryDomain.NONE,
        start_ns=pinned_transport_allocated_ns,
        end_ns=max(pinned_freed_ns, pinned_transport_allocated_ns + 1),
        temporary_basis="logical",
        temporary_memory=MemoryDomain.PINNED_HOST_TRANSPORT,
        required_unavoidable=True,
    )
    movement_report = ledger.report()
    if movement_report.accounting.logical_state_bytes != logical_bytes:
        raise RuntimeError("movement accounting denominator differs from transport payload")
    host_allocation_summary = host_allocations.summarize(require_all_freed=True)
    cuda_operation_summary = summarize_cuda_operations(cuda_recorder.records)
    restored_runtime_identity_injective = len(set(runtime_ids.values())) == len(runtime_ids)
    runtime_incarnations_fresh = restored_runtime_identity_injective and set(
        source_runtime_request_ids.values()
    ).isdisjoint(runtime_ids.values())
    if not runtime_incarnations_fresh:
        raise RuntimeError("restored runtime request incarnations are not fresh and injective")

    return {
        "schema_version": "sloforge.branchfabric.experiment-004-rollout-result/v1",
        "trigger_ns": trigger_ns,
        "logical_state_bytes": logical_bytes,
        "kv_logical_state_bytes": logical_bytes,
        "logical_state_accounting_scope": (
            "gpu-resident KV payload; required host-resident transport metadata is "
            "reported separately and excluded from movement amplification"
        ),
        "host_resident_transport_metadata_bytes": host_resident_transport_metadata_bytes,
        "helix_environment_state_bytes": 0,
        "helix_environment_state_scope": (
            "bounded model-only greedy rollout has no external environment or trajectory payload"
        ),
        "shared_bytes": plan.shared_logical_bytes,
        "private_bytes": plan.private_logical_bytes,
        "physical_source_bytes": physical_bytes,
        "block_count": len(plan.page_order),
        "branch_count": len(branches),
        "transport_manifest": transport_manifest,
        "source_capture_evidence": plan.capture_evidence.model_dump(mode="json"),
        "source_memory_before_release": memory_before.model_dump(mode="json"),
        "source_memory_after_release": memory_after.model_dump(mode="json"),
        "source_block_release_evidence": source_release_evidence.model_dump(mode="json"),
        "source_block_release_summary": source_release_summary,
        "source_allocations": sorted(source_allocations),
        "source_runtime_request_ids": source_runtime_request_ids,
        "destination_allocations": sorted(destination_allocations),
        "destination_page_map": imported.logical_page_destinations,
        "restored_internal_runtime_request_ids": runtime_ids,
        "restored_external_runtime_request_ids": output_ids,
        "restored_runtime_identity_injective": restored_runtime_identity_injective,
        "runtime_incarnations_fresh": runtime_incarnations_fresh,
        "reclamation_stages": stages,
        "restore_stages": restore_stages,
        "phase_events": phase_events,
        "temporary_serving": serving,
        "first_serving_ns": first_serving_ns,
        "first_resumed_tokens": observed_first_tokens,
        "first_resumed_token_timestamps_ns": {
            logical_id: first_token_timestamps_ns[output_id]
            for logical_id, output_id in output_ids.items()
        },
        "expected_first_tokens": expected_first_tokens,
        "continuation_token_counts": continuation_counts,
        "sampling_semantics_by_branch": source_sampling_by_branch,
        "recompute_control": {
            "start_ns": recompute_started_ns,
            "end_ns": recompute_ended_ns,
            "duration_ns": recompute_ended_ns - recompute_started_ns,
            "tokens_prefilled": recompute_tokens_prefilled,
        },
        "movement_report": movement_report.model_dump(mode="json"),
        "minimal_trace_events": tuple(
            event.model_dump(mode="json") for event in minimal_trace_events
        ),
        "minimal_trace_jsonl": minimal_trace_jsonl,
        "restore_resume_complete_ns": resume_complete_ns,
        "all_branches_resumed_ns": all_branches_resumed_ns,
        "rollout_continuation_complete_ns": resume_complete_ns,
        "_operation_telemetry": {
            "schema_version": ("sloforge.branchfabric.experiment-004-operation-telemetry/v1"),
            "collection_level": tracing_level,
            "cuda_operations": [record.model_dump(mode="json") for record in cuda_recorder.records],
            "cuda_summary": cuda_operation_summary.model_dump(mode="json"),
            "host_allocations": host_allocation_summary.model_dump(mode="json"),
        },
        "transport_payload_sha256": transport_payload_sha256,
        "transport_integrity_valid": True,
        "source_destination_fresh": True,
        "all_branches_resumed": True,
        "warm_pool_driver_hbm_released": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _arguments(argv)
    work_root = args.work_root.resolve()
    work_root.mkdir(parents=True, exist_ok=True)
    status = "failed"
    adapter: Any = None
    telemetry: _WorkerResourceTelemetry | None = None
    config: dict[str, Any] | None = None
    error: dict[str, Any] | None = None
    result: dict[str, Any] = {}
    telemetry_cleanup_complete = True
    started_ns = time.monotonic_ns()
    try:
        config = json.loads(args.config.resolve(strict=True).read_text())
        if config.get("schema_version") != "sloforge.branchfabric.experiment-004-modal-config/v1":
            raise ValueError("worker received an unsupported Experiment 004 config")
        visible_index = os.environ.get("SLOFORGE_EXP004_VISIBLE_GPU_INDEX")
        if (
            visible_index is None
            or os.environ.get("CUDA_VISIBLE_DEVICES") != visible_index
            or os.environ.get("SLOFORGE_EXP004_PHYSICAL_GPU_UUID") != args.physical_gpu_uuid
        ):
            raise RuntimeError("worker CUDA ordinal/physical UUID binding is inconsistent")
        os.environ.update(
            {
                "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
                "VLLM_USE_FLASHINFER_SAMPLER": "0",
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "TOKENIZERS_PARALLELISM": "false",
                "TRITON_CACHE_DIR": str(work_root / "triton"),
            }
        )
        (work_root / "triton").mkdir(exist_ok=True)
        import torch  # type: ignore[import-not-found]

        if (
            importlib.metadata.version("vllm") != "0.23.0"
            or importlib.metadata.version("torch") != "2.11.0"
            or torch.version.cuda != "13.0"
            or not torch.cuda.is_available()
            or torch.cuda.device_count() != 1
            or torch.cuda.get_device_capability(0) != (8, 0)
        ):
            raise RuntimeError("worker runtime pins or A100 visibility are invalid")
        inputs = json.loads((args.model_snapshot / "BRANCHFABRIC_INPUTS.json").read_text())
        integrated = config.get("execution_mode") == "integrated-calibration-v10"
        model_load_started_ns = time.monotonic_ns()
        if integrated:
            _write_new(
                args.barrier_root / f"{args.role}.engine-started.json",
                {
                    "schema_version": "sloforge.branchfabric.retained-engine-start/v1",
                    "role": args.role,
                    "device": "gpu0" if args.role == "serving" else "gpu1",
                    "pid": os.getpid(),
                    "physical_gpu_uuid": args.physical_gpu_uuid,
                    "engine_started_ns": model_load_started_ns,
                },
            )
        adapter = _create_adapter(
            config, args.model_snapshot.resolve(strict=True), args.physical_gpu_uuid
        )
        model_ready_ns = time.monotonic_ns()
        telemetry = _start_worker_resource_telemetry(
            role=args.role,
            physical_gpu_uuid=args.physical_gpu_uuid,
            worker_pid=os.getpid(),
            work_root=work_root,
            config=config,
        )
        prepared = None
        serving_engine: Any = adapter._view.llm_engine
        retained_engine_handoff: dict[str, Any] | None = None
        if integrated:
            config, serving_engine, retained_engine_handoff = _run_integrated_calibration_phase(
                adapter=adapter,
                config=config,
                inputs=inputs,
                role=args.role,
                physical_gpu_uuid=args.physical_gpu_uuid,
                barrier_root=args.barrier_root,
                model_load_started_ns=model_load_started_ns,
                model_ready_ns=model_ready_ns,
            )
            if args.role == "rollout":
                prepared = _prepare_rollouts(adapter, config, inputs)
            transaction_ready = {
                **retained_engine_handoff,
                "schema_version": "sloforge.branchfabric.transaction-ready/v1",
                "rollouts_ready": prepared is not None,
                "branch_count": 0 if prepared is None else len(prepared["branches"]),
                "observed_at_monotonic_ns": time.monotonic_ns(),
            }
            _write_new(
                _transaction_ready_path(args.barrier_root, args.role),
                transaction_ready,
            )
        else:
            if args.role == "rollout":
                prepared = _prepare_rollouts(adapter, config, inputs)
            else:
                warm_start = time.monotonic_ns()
                _run_raw_serving(
                    adapter._view.llm_engine,
                    prefix=tuple(int(item) for item in inputs["prefix_token_ids"][:64]),
                    start_ns=warm_start,
                    phases=((0.1, 10.0, "warmup"),),
                    output_tokens=1,
                    seed=int(config["seed"]),
                    request_prefix="gpu0-warmup",
                )
                adapter._view.manager.reset_prefix_cache()
            _write_new(
                args.barrier_root / f"{args.role}.ready.json",
                {
                    "role": args.role,
                    "pid": os.getpid(),
                    "physical_gpu_uuid": args.physical_gpu_uuid,
                    "model_load_started_ns": model_load_started_ns,
                    "model_ready_ns": model_ready_ns,
                    "rollouts_ready": prepared is not None,
                },
            )
        start_barrier = _wait_for(
            args.barrier_root / "start.json",
            timeout_s=float(config["maximum_wall_seconds"]),
        )
        common_start_ns = int(start_barrier["start_monotonic_ns"])
        if args.role == "serving":
            serving_prefix = tuple(
                int(item)
                for item in inputs["prefix_token_ids"][: int(config["serving_prompt_tokens"])]
            )

            def execute_transaction() -> dict[str, Any]:
                if config.get("serving_methodology") == "v10-global-capacity":
                    from gpu_capacity_calibration_worker import _runtime_queue_state
                    from gpu_reclamation_v10_serving import LiveV10Config, run_v10_gpu0

                    return run_v10_gpu0(
                        serving_engine,
                        prefix=serving_prefix,
                        params=_sampling_params(
                            max_tokens=int(config["serving_output_tokens"]),
                            seed=int(config["seed"]),
                        ),
                        config=LiveV10Config.from_mapping(config),
                        start_ns=common_start_ns,
                        barriers=args.barrier_root,
                        write_new=_write_new,
                        runtime_queue_state=lambda: _runtime_queue_state(adapter),
                    )
                return _run_raw_serving(
                    adapter._view.llm_engine,
                    prefix=serving_prefix,
                    start_ns=common_start_ns,
                    phases=(
                        (
                            float(config["baseline_seconds"]),
                            float(config["gpu0_control_request_rate_per_second"]),
                            "control",
                        ),
                        (
                            float(config["serving_spike_seconds"]),
                            float(config["gpu0_spike_request_rate_per_second"]),
                            "spike",
                        ),
                    ),
                    output_tokens=int(config["serving_output_tokens"]),
                    seed=int(config["seed"]),
                    request_prefix="gpu0-serving",
                    continuation_barrier=args.barrier_root / "rollout-restore-complete.json",
                    continuation_rate_per_second=float(
                        config["gpu0_restore_request_rate_per_second"]
                    ),
                    continuation_grace_seconds=float(config["temporary_serving_seconds"]),
                    continuation_timeout_seconds=float(config["maximum_wall_seconds"]) / 2.0,
                )

        else:
            assert prepared is not None
            trigger_ns = common_start_ns + int(float(config["baseline_seconds"]) * 1e9)

            def execute_transaction() -> dict[str, Any]:
                if str(config["tracing_level"]) == "full":
                    # FULL is a bounded diagnostic mode and is never eligible as
                    # a causal performance trial. Minimal/disabled avoid Kineto.
                    profile_path = work_root / "profiles/torch-profiler-chrome-trace.json"
                    profile_path.parent.mkdir(parents=True, exist_ok=True)
                    with torch.profiler.profile(
                        activities=(
                            torch.profiler.ProfilerActivity.CPU,
                            torch.profiler.ProfilerActivity.CUDA,
                        ),
                        record_shapes=False,
                        profile_memory=True,
                        with_stack=False,
                        with_flops=False,
                    ) as profiler:
                        transaction_result = _run_rollout_transaction(
                            adapter,
                            config,
                            inputs,
                            prepared,
                            trigger_ns,
                            args.barrier_root / "rollout-restore-complete.json",
                            args.barrier_root,
                            serving_engine,
                        )
                    profiler.export_chrome_trace(str(profile_path))
                    transaction_result["full_profile"] = {
                        "schema_version": (
                            "sloforge.branchfabric.experiment-004-full-profile-reference/v1"
                        ),
                        "status": "succeeded",
                        "causal_trial_eligible": False,
                        "raw_provenance": _existing_artifact_provenance(
                            profile_path,
                            artifact_root=work_root,
                            sample_selector="$.traceEvents[*]",
                        ),
                    }
                    return transaction_result
                return _run_rollout_transaction(
                    adapter,
                    config,
                    inputs,
                    prepared,
                    trigger_ns,
                    args.barrier_root / "rollout-restore-complete.json",
                    args.barrier_root,
                    serving_engine,
                )

        if integrated:
            from gpu_capacity_calibration_worker import _CompilationLogCapture

            result = _run_measured_v10_transaction(
                role=args.role,
                work_root=work_root,
                operation=execute_transaction,
                compilation_capture_factory=_CompilationLogCapture,
            )
        else:
            result = execute_transaction()
        if isinstance(serving_engine, _CacheSaltEngine):
            cache_salt_evidence = serving_engine.evidence()
            if cache_salt_evidence["passed"] is not True:
                raise RuntimeError("serving cache-salt uniqueness evidence failed")
            result["cache_salt_evidence"] = cache_salt_evidence
            result["prefix_cache_policy"] = {
                "enabled_for_rollout_semantics": True,
                "serving_reuse_prevented": True,
                "method": "per-request-vllm-cache-salt-sha256",
                "evidence_sha256": cache_salt_evidence["request_salt_map_sha256"],
                "rollout_requests_unsalted_for_shared_root_semantics": True,
            }
        if retained_engine_handoff is not None:
            result["retained_engine_handoff"] = retained_engine_handoff
        operation_telemetry = result.pop("_operation_telemetry", None)
        if operation_telemetry is not None:
            operation_provenance = _write_new_artifact(
                work_root / "telemetry/cuda-and-host-operations.json",
                operation_telemetry,
                artifact_root=work_root,
                sample_selector="$.cuda_operations[*]",
            )
            result["operation_telemetry"] = {
                "schema_version": (
                    "sloforge.branchfabric.experiment-004-operation-telemetry-reference/v1"
                ),
                "status": "succeeded",
                "collection_level": str(config["tracing_level"]),
                "cuda_summary": operation_telemetry["cuda_summary"],
                "host_allocation_summary": {
                    key: operation_telemetry["host_allocations"][key]
                    for key in (
                        "allocated_bytes_by_kind",
                        "peak_live_bytes_by_kind",
                        "active_bytes_by_kind",
                    )
                },
                "raw_provenance": operation_provenance,
            }
        else:
            result["operation_telemetry"] = {
                "schema_version": (
                    "sloforge.branchfabric.experiment-004-operation-telemetry-reference/v1"
                ),
                "status": "not_applicable",
                "reason": "serving worker performs no checkpoint state operations",
            }
        result.update(
            {
                "role": args.role,
                "physical_gpu_uuid": args.physical_gpu_uuid,
                "model_load_started_ns": model_load_started_ns,
                "model_ready_ns": model_ready_ns,
                "packages": {
                    name: importlib.metadata.version(name)
                    for name in ("vllm", "torch", "transformers")
                },
                "torch_cuda": torch.version.cuda,
                "gpu_name": torch.cuda.get_device_name(0),
            }
        )
        status = "succeeded"
    except BaseException as caught:
        error = {
            "type": type(caught).__name__,
            "message": str(caught),
            "traceback": traceback.format_exc(),
        }
        _write_new(work_root / "failure.json", error)
    finally:
        if telemetry is not None:
            try:
                cleanup_timeout = (
                    min(5.0, float(config["cleanup_timeout_seconds"]))
                    if config is not None
                    else 5.0
                )
                result["resource_telemetry"] = telemetry.stop(timeout_seconds=cleanup_timeout)
            except BaseException as telemetry_error:
                status = "failed"
                telemetry_cleanup_complete = not telemetry.is_running
                telemetry_failure = {
                    "type": type(telemetry_error).__name__,
                    "message": str(telemetry_error),
                    "traceback": traceback.format_exc(),
                    "sampler_thread_stopped": telemetry_cleanup_complete,
                }
                provenance: dict[str, str] | None = None
                try:
                    provenance = _write_new_artifact(
                        work_root / "telemetry/telemetry-failure.json",
                        telemetry_failure,
                        artifact_root=work_root,
                        sample_selector="$",
                    )
                except BaseException as artifact_error:
                    telemetry_failure["artifact_write_error"] = {
                        "type": type(artifact_error).__name__,
                        "message": str(artifact_error),
                    }
                result["resource_telemetry"] = {
                    "schema_version": (
                        "sloforge.branchfabric.experiment-004-telemetry-reference/v1"
                    ),
                    "status": "failed",
                    "role": args.role,
                    "physical_gpu_uuid": args.physical_gpu_uuid,
                    "raw_failure_provenance": provenance,
                    "failure": telemetry_failure,
                }
                if error is None:
                    error = telemetry_failure
        if adapter is not None:
            try:
                adapter.cleanup_runtime(timeout_s=60.0)
            except BaseException as cleanup_error:
                status = "failed"
                cleanup = {
                    "type": type(cleanup_error).__name__,
                    "message": str(cleanup_error),
                    "traceback": traceback.format_exc(),
                }
                _write_new(work_root / "cleanup-failure.json", cleanup)
                if error is None:
                    error = cleanup
        try:
            import torch  # type: ignore[import-not-found]

            torch.cuda.synchronize()
            gc.collect()
            torch.cuda.empty_cache()
        except BaseException as cuda_cleanup_error:
            status = "failed"
            cuda_cleanup = {
                "type": type(cuda_cleanup_error).__name__,
                "message": str(cuda_cleanup_error),
                "traceback": traceback.format_exc(),
            }
            _write_new(work_root / "cuda-cleanup-failure.json", cuda_cleanup)
            if error is None:
                error = cuda_cleanup
    result.update(
        {
            "status": status,
            "worker_pid": os.getpid(),
            "started_ns": started_ns,
            "ended_ns": time.monotonic_ns(),
            "error": error,
            "completed_at_utc": datetime.now(UTC).isoformat(),
        }
    )
    if not telemetry_cleanup_complete:
        # Publishing a nominal worker result while its sampler thread remains
        # alive would make cleanup evidence false.  Returning terminates the
        # daemon thread with this worker and lets the controller fail closed on
        # the nonzero exit code.
        return 1
    _write_new(work_root / "result.json", result)
    return 0 if status == "succeeded" else 1


if __name__ == "__main__":
    raise SystemExit(main())
