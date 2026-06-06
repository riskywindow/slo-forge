"""Version-checked command builders for optional fabric benchmark tools.

Builders never execute and never substitute another transport.  Callers must
explicitly handle an unavailable adapter before invoking the returned command.
"""

from __future__ import annotations

import importlib.metadata
import re
import shutil
import subprocess
from enum import StrEnum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class Availability(StrEnum):
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    VERSION_UNPARSED = "version_unparsed"


class AdapterAvailability(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    name: str
    availability: Availability
    executable: str | None
    version: str | None
    reason: str | None
    probe_argv: tuple[str, ...]


class AdapterCommand(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    adapter: str
    executable: str
    argv: tuple[str, ...]
    environment: tuple[tuple[str, str], ...]
    timeout_seconds: float = Field(gt=0.0)
    expected_transport: str
    requires_gpu: bool
    requires_multi_process: bool


_TOOL_PROBES: tuple[tuple[str, tuple[str, ...], str], ...] = (
    ("cuda", ("nvcc", "--version"), r"release\s+([0-9]+(?:\.[0-9]+)+)"),
    ("nvml", ("nvidia-smi", "--version"), r"NVIDIA-SMI\s+([0-9]+(?:\.[0-9]+)+)"),
    (
        "nccl-all-reduce",
        ("all_reduce_perf", "--help"),
        r"NCCL(?: VERSION)?\s*([0-9]+(?:\.[0-9]+)+)",
    ),
    ("ibverbs", ("ibv_devinfo", "--version"), r"([0-9]+(?:\.[0-9]+)+)"),
    ("ib-perftest", ("ib_write_bw", "--version"), r"([0-9]+(?:\.[0-9]+)+)"),
    ("ucx", ("ucx_info", "-v"), r"version:\s*([0-9]+(?:\.[0-9]+)+)"),
)


def _probe_tool(name: str, argv: tuple[str, ...], pattern: str) -> AdapterAvailability:
    executable = shutil.which(argv[0])
    if executable is None:
        return AdapterAvailability(
            name=name,
            availability=Availability.UNAVAILABLE,
            executable=None,
            version=None,
            reason=f"{argv[0]} is not installed or not visible on PATH",
            probe_argv=argv,
        )
    try:
        completed = subprocess.run(
            [executable, *argv[1:]],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return AdapterAvailability(
            name=name,
            availability=Availability.UNAVAILABLE,
            executable=executable,
            version=None,
            reason=f"version probe failed: {error}",
            probe_argv=argv,
        )
    if completed.returncode not in {0, 1}:
        return AdapterAvailability(
            name=name,
            availability=Availability.UNAVAILABLE,
            executable=executable,
            version=None,
            reason=f"version probe exited with {completed.returncode}",
            probe_argv=argv,
        )
    match = re.search(pattern, completed.stdout + completed.stderr, re.IGNORECASE)
    return AdapterAvailability(
        name=name,
        availability=Availability.AVAILABLE if match else Availability.VERSION_UNPARSED,
        executable=executable,
        version=match.group(1) if match else None,
        reason=None if match else "tool is executable but its version output was not recognized",
        probe_argv=argv,
    )


def _python_adapter(name: str, distribution: str) -> AdapterAvailability:
    try:
        version = importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return AdapterAvailability(
            name=name,
            availability=Availability.UNAVAILABLE,
            executable=None,
            version=None,
            reason=f"Python distribution {distribution} is not installed",
            probe_argv=("python-package-metadata", distribution),
        )
    return AdapterAvailability(
        name=name,
        availability=Availability.AVAILABLE,
        executable=None,
        version=version,
        reason=None,
        probe_argv=("python-package-metadata", distribution),
    )


def adapter_inventory() -> tuple[AdapterAvailability, ...]:
    native = tuple(_probe_tool(*specification) for specification in _TOOL_PROBES)
    python_adapters = tuple(
        _python_adapter(name, distribution)
        for name, distribution in (
            ("nvml-python", "nvidia-ml-py"),
            ("torch", "torch"),
            ("vllm", "vllm"),
            ("sglang", "sglang"),
            ("nvidia-dynamo", "ai-dynamo"),
            ("nixl", "nixl"),
            ("deepep", "deepep"),
        )
    )
    return native + python_adapters


def _required_executable(path: Path, adapter: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"{adapter} executable does not exist: {path}")
    if not path.stat().st_mode & 0o111:
        raise PermissionError(f"{adapter} executable is not executable: {path}")
    return str(path.resolve())


def build_nccl_tests_command(
    *,
    executable: Path,
    operation: Literal[
        "all_reduce", "all_gather", "reduce_scatter", "broadcast", "send_receive", "all_to_all"
    ],
    minimum_bytes: int,
    maximum_bytes: int,
    step_factor: int,
    gpus_per_process: int,
    visible_devices: tuple[str, ...],
    iterations: int,
    warmups: int,
    timeout_seconds: float = 120.0,
    algorithm: Literal["Ring", "Tree", "CollNetDirect", "CollNetChain"] | None = None,
    protocol: Literal["Simple", "LL", "LL128"] | None = None,
    channels: int | None = None,
) -> AdapterCommand:
    if minimum_bytes <= 0 or maximum_bytes < minimum_bytes:
        raise ValueError("NCCL test message bounds are invalid")
    if step_factor < 2 or gpus_per_process < 1 or iterations < 1 or warmups < 0:
        raise ValueError("NCCL test counts are invalid")
    if (
        len(visible_devices) != gpus_per_process
        or len(set(visible_devices)) != len(visible_devices)
        or any(not device.strip() or "," in device for device in visible_devices)
    ):
        raise ValueError("NCCL visible_devices must contain one unique explicit identifier per GPU")
    expected_name = {
        "all_reduce": "all_reduce_perf",
        "all_gather": "all_gather_perf",
        "reduce_scatter": "reduce_scatter_perf",
        "broadcast": "broadcast_perf",
        "send_receive": "sendrecv_perf",
        "all_to_all": "alltoall_perf",
    }[operation]
    if executable.name != expected_name:
        raise ValueError(f"{operation} requires the {expected_name} binary, got {executable.name}")
    resolved = _required_executable(executable, "nccl-tests")
    environment: list[tuple[str, str]] = [
        ("CUDA_VISIBLE_DEVICES", ",".join(visible_devices)),
    ]
    if algorithm:
        environment.append(("NCCL_ALGO", algorithm))
    if protocol:
        environment.append(("NCCL_PROTO", protocol))
    if channels is not None:
        if channels < 1:
            raise ValueError("NCCL channel count must be positive")
        environment.extend(
            (("NCCL_MIN_NCHANNELS", str(channels)), ("NCCL_MAX_NCHANNELS", str(channels)))
        )
    return AdapterCommand(
        adapter="nccl-tests",
        executable=resolved,
        argv=(
            resolved,
            "-b",
            str(minimum_bytes),
            "-e",
            str(maximum_bytes),
            "-f",
            str(step_factor),
            "-g",
            str(gpus_per_process),
            "-n",
            str(iterations),
            "-w",
            str(warmups),
            "-c",
            "1",
        ),
        environment=tuple(environment),
        timeout_seconds=timeout_seconds,
        expected_transport="nccl",
        requires_gpu=True,
        requires_multi_process=False,
    )


def build_ibverbs_command(
    *,
    executable: Path,
    role: Literal["server", "client"],
    server_address: str | None,
    device: str,
    port: int,
    message_bytes: int,
    iterations: int,
    use_rdma_cm: bool = False,
    timeout_seconds: float = 60.0,
) -> AdapterCommand:
    if executable.name not in {"ib_write_bw", "ib_read_bw", "ib_send_bw"}:
        raise ValueError("only the bounded perftest bandwidth adapters are supported")
    if role == "client" and not server_address:
        raise ValueError("an ibverbs client requires a server address")
    if role == "server" and server_address is not None:
        raise ValueError("an ibverbs server must not specify a server address")
    if not 1 <= port <= 65535 or message_bytes <= 0 or iterations < 1:
        raise ValueError("ibverbs command arguments are outside valid bounds")
    resolved = _required_executable(executable, "ib-perftest")
    argv = [
        resolved,
        "--ib-dev",
        device,
        "--port",
        str(port),
        "--size",
        str(message_bytes),
        "--iters",
        str(iterations),
        "--report_gbits",
    ]
    if use_rdma_cm:
        argv.append("--rdma_cm")
    if role == "client":
        argv.append(server_address or "")
    return AdapterCommand(
        adapter="ib-perftest",
        executable=resolved,
        argv=tuple(argv),
        environment=(),
        timeout_seconds=timeout_seconds,
        expected_transport="ibverbs",
        requires_gpu=False,
        requires_multi_process=True,
    )


def build_nvidia_smi_command(
    *,
    executable: Path,
    gpu_id: str,
    fields: tuple[str, ...],
    timeout_seconds: float = 10.0,
) -> AdapterCommand:
    allowed = {
        "uuid",
        "name",
        "pci.bus_id",
        "memory.total",
        "utilization.gpu",
        "clocks.sm",
        "clocks.mem",
        "power.draw",
        "temperature.gpu",
        "ecc.errors.uncorrected.volatile.total",
    }
    if not fields or any(field not in allowed for field in fields):
        raise ValueError("nvidia-smi fields must come from the read-only allowlist")
    resolved = _required_executable(executable, "nvidia-smi")
    return AdapterCommand(
        adapter="nvidia-smi-query",
        executable=resolved,
        argv=(
            resolved,
            f"--id={gpu_id}",
            f"--query-gpu={','.join(fields)}",
            "--format=csv,noheader,nounits",
        ),
        environment=(),
        timeout_seconds=timeout_seconds,
        expected_transport="nvml",
        requires_gpu=True,
        requires_multi_process=False,
    )
