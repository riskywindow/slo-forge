from __future__ import annotations

import importlib
import json
import os
import platform
import re
import shutil
import statistics
import subprocess
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from sloforge.util import canonical_json, percentile, sha256_bytes, utc_now


class BenchmarkSamples(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    unit: str
    device: str
    warmup_count: int
    samples: list[float]
    median: float
    p95: float


class HardwareProbe(BaseModel):
    model_config = ConfigDict(extra="forbid")

    hostname: str
    os: str
    architecture: str
    cpu_model: str
    logical_cpu_count: int
    memory_bytes: int
    cgroup_memory_limit_bytes: int | None = None
    numa_nodes: int | None = None
    gpu: dict[str, str | int | float | None] | None = None
    container: str | None = None


class ProbeResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["sloforge.hardware/v1"] = "sloforge.hardware/v1"
    captured_at: str
    requested_device: Literal["cpu", "cuda"]
    hardware: HardwareProbe
    benchmarks: list[BenchmarkSamples]
    fingerprint: str
    warnings: list[str] = Field(default_factory=list)


def _memory_bytes() -> int:
    if platform.system() == "Darwin":
        completed = subprocess.run(
            ["sysctl", "-n", "hw.memsize"], check=True, capture_output=True, text=True, timeout=5
        )
        return int(completed.stdout.strip())
    meminfo = Path("/proc/meminfo")
    if meminfo.exists():
        for line in meminfo.read_text(encoding="utf-8").splitlines():
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) * 1024
    raise RuntimeError("unable to determine system memory without silently estimating it")


def _cpu_model() -> str:
    if platform.system() == "Darwin":
        completed = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if completed.returncode == 0 and completed.stdout.strip():
            return completed.stdout.strip()
        completed = subprocess.run(
            ["sysctl", "-n", "hw.model"], check=True, capture_output=True, text=True, timeout=5
        )
        return completed.stdout.strip()
    return platform.processor() or "unknown"


def _cgroup_limit() -> int | None:
    for candidate in (
        Path("/sys/fs/cgroup/memory.max"),
        Path("/sys/fs/cgroup/memory/memory.limit_in_bytes"),
    ):
        if candidate.exists():
            value = candidate.read_text(encoding="utf-8").strip()
            if value != "max":
                return int(value)
    return None


def _nvidia_query(executable: str, field: str, *, required: bool) -> str | None:
    completed = subprocess.run(
        [
            executable,
            f"--query-gpu={field}",
            "--format=csv,noheader,nounits",
            "--id=0",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    value = completed.stdout.strip().splitlines()[0].strip() if completed.stdout.strip() else ""
    if completed.returncode != 0 or not value or value in {"N/A", "[N/A]"}:
        if required:
            raise RuntimeError(
                f"nvidia-smi could not query required device-0 field {field}: "
                f"{completed.stderr.strip()}"
            )
        return None
    return value


def _as_int(value: str | None) -> int | None:
    if value is None:
        return None
    match = re.search(r"-?\d+", value)
    return int(match.group()) if match else None


def _as_float(value: str | None) -> float | None:
    if value is None:
        return None
    match = re.search(r"-?\d+(?:\.\d+)?", value)
    return float(match.group()) if match else None


def _gpu_info(*, hourly_price_usd: float | None) -> dict[str, str | int | float | None]:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        raise RuntimeError("CUDA was explicitly requested, but nvidia-smi is not installed")
    required = {
        field: _nvidia_query(executable, field, required=True)
        for field in ("name", "uuid", "memory.total", "driver_version", "pci.bus_id")
    }
    optional = {
        field: _nvidia_query(executable, field, required=False)
        for field in (
            "pstate",
            "temperature.gpu",
            "utilization.gpu",
            "clocks.sm",
            "clocks.mem",
            "pcie.link.gen.current",
            "pcie.link.width.current",
            "ecc.errors.uncorrected.volatile.total",
            "compute_cap",
            "memory.bus_width",
        )
    }
    inventory = subprocess.run(
        [executable, "--list-gpus"],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    topology = subprocess.run(
        [executable, "topo", "-m"],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    version = subprocess.run([executable], check=False, capture_output=True, text=True, timeout=10)
    cuda_match = re.search(r"CUDA Version:\s*([0-9.]+)", version.stdout)
    pci_bus_id = required["pci.bus_id"] or ""
    sysfs_pci = Path("/sys/bus/pci/devices") / pci_bus_id.lower()
    numa_node = None
    if (sysfs_pci / "numa_node").is_file():
        numa_node = _as_int((sysfs_pci / "numa_node").read_text(encoding="utf-8").strip())
    return {
        "index": 0,
        "gpu_count": len(inventory.stdout.splitlines()) if inventory.returncode == 0 else None,
        "name": required["name"],
        "uuid": required["uuid"],
        "vram_mib": _as_int(required["memory.total"]),
        "driver_version": required["driver_version"],
        "cuda_version": cuda_match.group(1) if cuda_match else None,
        "pci_bus_id": pci_bus_id,
        "pcie_generation": _as_int(optional["pcie.link.gen.current"]),
        "pcie_width": _as_int(optional["pcie.link.width.current"]),
        "numa_node": numa_node,
        "performance_state": optional["pstate"],
        "temperature_c": _as_float(optional["temperature.gpu"]),
        "utilization_percent": _as_float(optional["utilization.gpu"]),
        "sm_clock_mhz": _as_float(optional["clocks.sm"]),
        "memory_clock_mhz": _as_float(optional["clocks.mem"]),
        "memory_bus_width_bits": _as_int(optional["memory.bus_width"]),
        "compute_capability": optional["compute_cap"],
        "uncorrected_ecc_errors": _as_int(optional["ecc.errors.uncorrected.volatile.total"]),
        "topology": topology.stdout.strip() if topology.returncode == 0 else None,
        "hourly_price_usd": hourly_price_usd,
    }


def _bench_memory(*, warmups: int, samples: int) -> BenchmarkSamples:
    source = np.arange(16 * 1024 * 1024, dtype=np.uint8)
    target = np.empty_like(source)
    measured: list[float] = []
    for index in range(warmups + samples):
        started = time.perf_counter_ns()
        np.copyto(target, source)
        elapsed_s = (time.perf_counter_ns() - started) / 1e9
        if index >= warmups:
            measured.append((source.nbytes / elapsed_s) / 1e9)
    return BenchmarkSamples(
        name="host_memory_copy_bandwidth",
        unit="GB/s",
        device="cpu",
        warmup_count=warmups,
        samples=measured,
        median=statistics.median(measured),
        p95=percentile(measured, 0.95),
    )


def _bench_gemm(*, warmups: int, samples: int) -> BenchmarkSamples:
    rng = np.random.default_rng(0)
    left = rng.standard_normal((384, 384), dtype=np.float32)
    right = rng.standard_normal((384, 384), dtype=np.float32)
    measured: list[float] = []
    for index in range(warmups + samples):
        started = time.perf_counter_ns()
        result = left @ right
        elapsed_s = (time.perf_counter_ns() - started) / 1e9
        if float(result[0, 0]) == float("inf"):
            raise RuntimeError("GEMM produced an invalid result")
        if index >= warmups:
            operations = 2 * left.shape[0] * left.shape[1] * right.shape[1]
            measured.append(operations / elapsed_s / 1e9)
    return BenchmarkSamples(
        name="fp32_gemm_384",
        unit="GFLOP/s",
        device="cpu",
        warmup_count=warmups,
        samples=measured,
        median=statistics.median(measured),
        p95=percentile(measured, 0.95),
    )


def _cuda_timed_samples(
    torch: Any,
    operation: Callable[[], None],
    *,
    warmups: int,
    samples: int,
) -> list[float]:
    measured: list[float] = []
    for index in range(warmups + samples):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        operation()
        end.record()
        end.synchronize()
        if index >= warmups:
            measured.append(float(start.elapsed_time(end)))
    return measured


def _cuda_benchmark(
    *,
    name: str,
    unit: str,
    raw_ms: list[float],
    transform: Callable[[float], float],
    warmups: int,
) -> BenchmarkSamples:
    values = [transform(value) for value in raw_ms]
    return BenchmarkSamples(
        name=name,
        unit=unit,
        device="cuda:0",
        warmup_count=warmups,
        samples=values,
        median=statistics.median(values),
        p95=percentile(values, 0.95),
    )


def _bench_cuda(*, warmups: int, samples: int) -> list[BenchmarkSamples]:
    try:
        torch = importlib.import_module("torch")
    except ImportError as exc:
        raise RuntimeError(
            "CUDA microbenchmarks require PyTorch; install the gpu-transformers extra"
        ) from exc
    if not bool(torch.cuda.is_available()) or int(torch.cuda.device_count()) < 1:
        raise RuntimeError("CUDA was requested but PyTorch cannot access cuda:0")
    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    byte_count = 16 * 1024 * 1024
    element_count = byte_count // 4
    pinned_source = torch.arange(element_count, dtype=torch.float32, pin_memory=True)
    pinned_target = torch.empty_like(pinned_source, pin_memory=True)
    device_source = torch.empty(element_count, dtype=torch.float32, device=device)
    device_target = torch.empty_like(device_source)
    tiny = torch.ones(1, dtype=torch.float32, device=device)

    launch_ms = _cuda_timed_samples(torch, lambda: tiny.add_(1.0), warmups=warmups, samples=samples)
    h2d_ms = _cuda_timed_samples(
        torch,
        lambda: device_source.copy_(pinned_source, non_blocking=True),
        warmups=warmups,
        samples=samples,
    )
    d2h_ms = _cuda_timed_samples(
        torch,
        lambda: pinned_target.copy_(device_source, non_blocking=True),
        warmups=warmups,
        samples=samples,
    )
    bandwidth_ms = _cuda_timed_samples(
        torch,
        lambda: device_target.copy_(device_source),
        warmups=warmups,
        samples=samples,
    )

    gemm_results: list[BenchmarkSamples] = []
    for size in (512, 1024):
        left = torch.randn((size, size), dtype=torch.float16, device=device)
        right = torch.randn((size, size), dtype=torch.float16, device=device)
        result = torch.empty((size, size), dtype=torch.float16, device=device)

        def gemm_operation(
            left_operand: Any = left,
            right_operand: Any = right,
            output: Any = result,
        ) -> None:
            torch.mm(left_operand, right_operand, out=output)

        gemm_ms = _cuda_timed_samples(
            torch,
            gemm_operation,
            warmups=warmups,
            samples=samples,
        )
        operations = float(2 * size**3)

        def to_tflops(elapsed_ms: float, operation_count: float = operations) -> float:
            return operation_count / (elapsed_ms / 1000.0) / 1e12

        gemm_results.append(
            _cuda_benchmark(
                name=f"cuda_fp16_gemm_{size}",
                unit="TFLOP/s",
                raw_ms=gemm_ms,
                transform=to_tflops,
                warmups=warmups,
            )
        )

    hidden = 1024
    weights = torch.randn((hidden, hidden), dtype=torch.float16, device=device)
    prefill = torch.randn((512, hidden), dtype=torch.float16, device=device)
    decode = torch.randn((16, hidden), dtype=torch.float16, device=device)
    prefill_ms = _cuda_timed_samples(
        torch, lambda: torch.mm(prefill, weights), warmups=warmups, samples=samples
    )
    decode_ms = _cuda_timed_samples(
        torch, lambda: torch.mm(decode, weights), warmups=warmups, samples=samples
    )
    sync_us: list[float] = []
    for index in range(warmups + samples):
        started = time.perf_counter_ns()
        torch.cuda.synchronize(device)
        elapsed_us = (time.perf_counter_ns() - started) / 1000.0
        if index >= warmups:
            sync_us.append(elapsed_us)

    def bandwidth(elapsed_ms: float) -> float:
        return byte_count / (elapsed_ms / 1000.0) / 1e9

    return [
        _cuda_benchmark(
            name="cuda_kernel_launch_overhead",
            unit="us",
            raw_ms=launch_ms,
            transform=lambda elapsed_ms: elapsed_ms * 1000.0,
            warmups=warmups,
        ),
        _cuda_benchmark(
            name="cuda_host_to_device_bandwidth",
            unit="GB/s",
            raw_ms=h2d_ms,
            transform=bandwidth,
            warmups=warmups,
        ),
        _cuda_benchmark(
            name="cuda_device_to_host_bandwidth",
            unit="GB/s",
            raw_ms=d2h_ms,
            transform=bandwidth,
            warmups=warmups,
        ),
        _cuda_benchmark(
            name="cuda_device_memory_copy_bandwidth",
            unit="GB/s",
            raw_ms=bandwidth_ms,
            transform=bandwidth,
            warmups=warmups,
        ),
        *gemm_results,
        _cuda_benchmark(
            name="cuda_representative_prefill_512x1024",
            unit="ms",
            raw_ms=prefill_ms,
            transform=float,
            warmups=warmups,
        ),
        _cuda_benchmark(
            name="cuda_representative_decode_16x1024",
            unit="ms",
            raw_ms=decode_ms,
            transform=float,
            warmups=warmups,
        ),
        BenchmarkSamples(
            name="cuda_synchronization_overhead",
            unit="us",
            device="cuda:0",
            warmup_count=warmups,
            samples=sync_us,
            median=statistics.median(sync_us),
            p95=percentile(sync_us, 0.95),
        ),
    ]


def run_probe(
    *,
    device: Literal["cpu", "cuda"] = "cpu",
    warmups: int = 2,
    samples: int = 7,
    hourly_price_usd: float | None = None,
) -> ProbeResult:
    if warmups < 0 or samples < 3:
        raise ValueError("probe requires nonnegative warmups and at least three samples")
    if hourly_price_usd is not None and hourly_price_usd < 0:
        raise ValueError("hourly_price_usd cannot be negative")
    if device == "cpu" and hourly_price_usd is not None:
        raise ValueError("hourly_price_usd currently applies only to an explicitly probed GPU")
    gpu = _gpu_info(hourly_price_usd=hourly_price_usd) if device == "cuda" else None
    hardware = HardwareProbe(
        hostname=platform.node(),
        os=platform.platform(),
        architecture=platform.machine(),
        cpu_model=_cpu_model(),
        logical_cpu_count=os.cpu_count() or 1,
        memory_bytes=_memory_bytes(),
        cgroup_memory_limit_bytes=_cgroup_limit(),
        numa_nodes=None,
        gpu=gpu,
        container="docker" if Path("/.dockerenv").exists() else None,
    )
    benchmarks = [
        _bench_memory(warmups=warmups, samples=samples),
        _bench_gemm(warmups=warmups, samples=samples),
    ]
    if device == "cuda":
        benchmarks.extend(_bench_cuda(warmups=warmups, samples=samples))
    identity = {
        "hardware": hardware.model_dump(),
        "benchmarks": [item.model_dump(exclude={"samples"}) for item in benchmarks],
    }
    warnings: list[str] = []
    if device == "cpu":
        warnings.append("GPU microbenchmarks were not requested; CUDA capabilities are unmeasured.")
    elif hourly_price_usd is None:
        warnings.append(
            "GPU hourly price was not supplied; real profiling will reject this catalog until "
            "gpu.hourly_price_usd is explicit."
        )
    if device == "cuda" and gpu is not None:
        gpu_count = gpu.get("gpu_count")
        if isinstance(gpu_count, int) and gpu_count > 1:
            warnings.append(
                "Multiple GPUs were detected; point-to-point and collective benchmarks require "
                "the optional distributed benchmark harness and were not run by this probe."
            )
    return ProbeResult(
        captured_at=utc_now(),
        requested_device=device,
        hardware=hardware,
        benchmarks=benchmarks,
        fingerprint=sha256_bytes(canonical_json(identity).encode()),
        warnings=warnings,
    )


def save_probe(path: Path, result: ProbeResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(result.model_dump(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
