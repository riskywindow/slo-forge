"""Bounded host-resource sampling for Helix characterization.

The sampler deliberately limits itself to portable process and operating-system
counters.  Hardware performance counters are capability-gated external profiler
inputs; an unavailable counter is never replaced with an estimate.  Workload
evidence and timing provenance are separate fields so a real host measurement of
a synthetic workload cannot be mislabeled as a real production workload.
"""

from __future__ import annotations

import importlib
import os
import platform
import shlex
import shutil
import subprocess
import threading
import time
from collections.abc import Callable, Sequence
from contextlib import AbstractContextManager
from enum import StrEnum
from pathlib import Path
from types import TracebackType
from typing import Literal, Protocol, cast

from pydantic import BaseModel, ConfigDict, Field, model_validator

from sloforge.helix.characterization.matrix import EvidenceClass, TraceLevel

MAX_ERROR_RECORDS = 32
MAX_PROFILER_ARGUMENTS = 4096
MAX_PROFILER_ARGUMENT_BYTES = 64 * 1024


class ResourceModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, validate_default=True)


class ResourceSamplerState(StrEnum):
    NEW = "new"
    RUNNING = "running"
    STOPPED = "stopped"


class TimingProvenance(StrEnum):
    HOST_MONOTONIC_OBSERVED = "HOST_MONOTONIC_OBSERVED"


class CounterCapability(ResourceModel):
    available: bool
    source: str | None = Field(default=None, min_length=1, max_length=1024)
    reason: str | None = Field(default=None, min_length=1, max_length=2048)
    requires_privileges: bool = False

    @model_validator(mode="after")
    def explain_availability(self) -> CounterCapability:
        if self.available and self.source is None:
            raise ValueError("an available capability requires a source")
        if not self.available and self.reason is None:
            raise ValueError("an unavailable capability requires a reason")
        return self


class PlatformCapabilities(ResourceModel):
    schema_version: Literal["sloforge.branchfabric.resource-capabilities/v1"]
    hostname: str = Field(min_length=1, max_length=255)
    process_id: int = Field(ge=1)
    operating_system: str = Field(min_length=1, max_length=128)
    machine: str = Field(min_length=1, max_length=128)
    python_implementation: str = Field(min_length=1, max_length=128)
    logical_cpu_count: int | None = Field(default=None, ge=1)
    physical_cpu_count: int | None = Field(default=None, ge=1)
    counters: dict[str, CounterCapability]
    profiler_tools: dict[str, CounterCapability]


class ResourceSamplerConfig(ResourceModel):
    schema_version: Literal["sloforge.branchfabric.resource-sampler-config/v1"] = (
        "sloforge.branchfabric.resource-sampler-config/v1"
    )
    trace_level: TraceLevel
    workload_evidence: EvidenceClass
    seed: int = Field(ge=0, le=2**64 - 1)
    sample_interval_ms: int = Field(default=100, ge=1, le=60_000)
    max_samples: int = Field(default=100_000, ge=1, le=1_000_000)
    max_duration_seconds: float | None = Field(
        default=None, gt=0.0, le=86_400.0, allow_inf_nan=False
    )


class ResourceSample(ResourceModel):
    sequence: int = Field(ge=0)
    monotonic_ns: int = Field(ge=0)
    relative_ns: int = Field(ge=0)
    timing_provenance: Literal[TimingProvenance.HOST_MONOTONIC_OBSERVED]
    collector: Literal["psutil", "stdlib-degraded"]
    process_cpu_user_seconds: float = Field(ge=0.0, allow_inf_nan=False)
    process_cpu_system_seconds: float = Field(ge=0.0, allow_inf_nan=False)
    process_rss_bytes: int | None = Field(default=None, ge=0)
    process_vms_bytes: int | None = Field(default=None, ge=0)
    process_thread_count: int | None = Field(default=None, ge=1)
    process_read_bytes: int | None = Field(default=None, ge=0)
    process_write_bytes: int | None = Field(default=None, ge=0)
    load_average_1m: float | None = Field(default=None, ge=0.0, allow_inf_nan=False)
    load_average_5m: float | None = Field(default=None, ge=0.0, allow_inf_nan=False)
    load_average_15m: float | None = Field(default=None, ge=0.0, allow_inf_nan=False)
    system_memory_total_bytes: int | None = Field(default=None, ge=0)
    system_memory_available_bytes: int | None = Field(default=None, ge=0)
    system_memory_used_bytes: int | None = Field(default=None, ge=0)
    system_disk_read_bytes: int | None = Field(default=None, ge=0)
    system_disk_write_bytes: int | None = Field(default=None, ge=0)
    system_network_receive_bytes: int | None = Field(default=None, ge=0)
    system_network_send_bytes: int | None = Field(default=None, ge=0)
    cpu_cycles: int | None = Field(default=None, ge=0)
    cache_misses: int | None = Field(default=None, ge=0)
    gpu_utilization_percent: float | None = Field(
        default=None, ge=0.0, le=100.0, allow_inf_nan=False
    )
    pcie_receive_bytes: int | None = Field(default=None, ge=0)
    pcie_send_bytes: int | None = Field(default=None, ge=0)


class ResourceTrace(ResourceModel):
    schema_version: Literal["sloforge.branchfabric.resource-trace/v1"]
    config: ResourceSamplerConfig
    capabilities: PlatformCapabilities
    timing_provenance: Literal[TimingProvenance.HOST_MONOTONIC_OBSERVED]
    started_monotonic_ns: int | None = Field(default=None, ge=0)
    ended_monotonic_ns: int | None = Field(default=None, ge=0)
    samples: tuple[ResourceSample, ...]
    sample_attempts: int = Field(ge=0)
    samples_recorded: int = Field(ge=0)
    samples_dropped: int = Field(ge=0)
    buffer_overflowed: bool
    collection_error_count: int = Field(ge=0)
    collection_errors: tuple[str, ...]
    error_records_dropped: int = Field(ge=0)

    @model_validator(mode="after")
    def accounting_is_complete(self) -> ResourceTrace:
        if self.samples_recorded != len(self.samples):
            raise ValueError("samples_recorded must equal the preserved sample count")
        if self.sample_attempts != (
            self.samples_recorded + self.samples_dropped + self.collection_error_count
        ):
            raise ValueError("every sample attempt must be recorded, dropped, or report an error")
        if self.buffer_overflowed != (self.samples_dropped > 0):
            raise ValueError("buffer_overflowed must reflect dropped samples")
        if self.collection_error_count < len(self.collection_errors):
            raise ValueError("collection error count cannot be smaller than its records")
        if self.error_records_dropped != self.collection_error_count - len(self.collection_errors):
            raise ValueError("error record drop accounting is inconsistent")
        return self


class ProfilerCommand(ResourceModel):
    tool: Literal["powermetrics", "xctrace", "nsys", "ncu"]
    capture_mode: Literal["sidecar", "wrap"]
    argv: tuple[str, ...] = Field(min_length=1, max_length=MAX_PROFILER_ARGUMENTS)
    rendered_command: str = Field(min_length=1, max_length=MAX_PROFILER_ARGUMENT_BYTES)
    output_artifact: str = Field(min_length=1, max_length=4096)
    requires_privileges: bool
    executes_target: bool
    note: str = Field(min_length=1, max_length=2048)


class _CpuTimes(Protocol):
    user: float
    system: float


class _MemoryInfo(Protocol):
    rss: int
    vms: int


class _IoCounters(Protocol):
    read_bytes: int
    write_bytes: int


class _VirtualMemory(Protocol):
    total: int
    available: int
    used: int


class _NetIoCounters(Protocol):
    bytes_recv: int
    bytes_sent: int


class _Process(Protocol):
    def cpu_times(self) -> _CpuTimes: ...

    def memory_info(self) -> _MemoryInfo: ...

    def num_threads(self) -> int: ...

    def io_counters(self) -> _IoCounters: ...


class _PsutilModule(Protocol):
    def Process(self, pid: int) -> _Process: ...

    def cpu_count(self, *, logical: bool = True) -> int | None: ...

    def virtual_memory(self) -> _VirtualMemory: ...

    def disk_io_counters(self) -> _IoCounters | None: ...

    def net_io_counters(self) -> _NetIoCounters | None: ...


def _load_psutil() -> _PsutilModule | None:
    try:
        return cast(_PsutilModule, importlib.import_module("psutil"))
    except ImportError:
        return None


def _capability(
    available: bool,
    *,
    source: str | None = None,
    reason: str | None = None,
    requires_privileges: bool = False,
) -> CounterCapability:
    return CounterCapability(
        available=available,
        source=source,
        reason=reason,
        requires_privileges=requires_privileges,
    )


def _xctrace_is_operational(xctrace_path: str | None) -> bool:
    if xctrace_path is None or platform.system() != "Darwin":
        return False
    developer_dir = os.environ.get("DEVELOPER_DIR")
    if developer_dir:
        return "Xcode" in developer_dir and Path(developer_dir).is_dir()
    xcode_select = shutil.which("xcode-select")
    if xcode_select is None:
        return False
    try:
        result = subprocess.run(
            (xcode_select, "-p"),
            check=False,
            capture_output=True,
            text=True,
            timeout=2.0,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return result.returncode == 0 and "Xcode" in result.stdout


def detect_platform_capabilities() -> PlatformCapabilities:
    """Detect counters and tools without starting a profiler or requiring privileges."""

    psutil_module = _load_psutil()
    operating_system = platform.system() or "unknown"
    process_source = "os.times"
    psutil_reason = "optional psutil dependency is not installed"
    powermetrics = shutil.which("powermetrics") if operating_system == "Darwin" else None
    xctrace = shutil.which("xctrace") if operating_system == "Darwin" else None
    xctrace_operational = _xctrace_is_operational(xctrace)
    nvidia_smi = shutil.which("nvidia-smi")
    nsys = shutil.which("nsys") if nvidia_smi is not None else None
    ncu = shutil.which("ncu") if nvidia_smi is not None else None

    counters = {
        "process_cpu_time": _capability(True, source=process_source),
        "process_memory": _capability(
            psutil_module is not None,
            source="psutil.Process.memory_info" if psutil_module is not None else None,
            reason=None if psutil_module is not None else psutil_reason,
        ),
        "process_threads": _capability(
            psutil_module is not None,
            source="psutil.Process.num_threads" if psutil_module is not None else None,
            reason=None if psutil_module is not None else psutil_reason,
        ),
        "process_io": _capability(
            psutil_module is not None,
            source="psutil.Process.io_counters" if psutil_module is not None else None,
            reason=None if psutil_module is not None else psutil_reason,
        ),
        "system_load": _capability(
            hasattr(os, "getloadavg"),
            source="os.getloadavg" if hasattr(os, "getloadavg") else None,
            reason=None if hasattr(os, "getloadavg") else "os.getloadavg is unavailable",
        ),
        "system_memory": _capability(
            psutil_module is not None,
            source="psutil.virtual_memory" if psutil_module is not None else None,
            reason=None if psutil_module is not None else psutil_reason,
        ),
        "system_disk_io": _capability(
            psutil_module is not None,
            source="psutil.disk_io_counters" if psutil_module is not None else None,
            reason=None if psutil_module is not None else psutil_reason,
        ),
        "system_network_io": _capability(
            psutil_module is not None,
            source="psutil.net_io_counters" if psutil_module is not None else None,
            reason=None if psutil_module is not None else psutil_reason,
        ),
        "cpu_cycles": _capability(
            False,
            reason="not collected by the portable periodic sampler; use an external profiler",
        ),
        "cache_misses": _capability(
            False,
            reason="not collected by the portable periodic sampler; use an external profiler",
        ),
        "gpu_utilization": _capability(
            False,
            reason="not collected by the portable periodic sampler; use a GPU profiler",
        ),
        "pcie_bytes": _capability(
            False,
            reason="no portable in-process PCIe byte counter is available",
        ),
        "nic_hardware_counters": _capability(
            False,
            reason="psutil network bytes are OS totals, not NIC hardware counters",
        ),
    }
    profiler_tools = {
        "powermetrics": _capability(
            powermetrics is not None,
            source=powermetrics,
            reason=None if powermetrics is not None else "powermetrics executable is unavailable",
            requires_privileges=True,
        ),
        "xctrace": _capability(
            xctrace_operational,
            source=xctrace if xctrace_operational else None,
            reason=(
                None
                if xctrace_operational
                else "xctrace requires an installed and selected full Xcode developer directory"
            ),
        ),
        "nsys": _capability(
            nsys is not None,
            source=nsys,
            reason=None if nsys is not None else "NVIDIA Nsight Systems and GPU are unavailable",
        ),
        "ncu": _capability(
            ncu is not None,
            source=ncu,
            reason=None if ncu is not None else "NVIDIA Nsight Compute and GPU are unavailable",
        ),
    }
    logical_cpu_count = (
        psutil_module.cpu_count(logical=True) if psutil_module is not None else os.cpu_count()
    )
    physical_cpu_count = (
        psutil_module.cpu_count(logical=False) if psutil_module is not None else None
    )
    return PlatformCapabilities(
        schema_version="sloforge.branchfabric.resource-capabilities/v1",
        hostname=platform.node() or "unknown",
        process_id=os.getpid(),
        operating_system=operating_system,
        machine=platform.machine() or "unknown",
        python_implementation=platform.python_implementation() or "unknown",
        logical_cpu_count=logical_cpu_count,
        physical_cpu_count=physical_cpu_count,
        counters=counters,
        profiler_tools=profiler_tools,
    )


def _validate_target_argv(target_argv: Sequence[str]) -> tuple[str, ...]:
    target = tuple(target_argv)
    if len(target) > MAX_PROFILER_ARGUMENTS:
        raise ValueError("profiler target has too many arguments")
    if any(not argument or "\x00" in argument for argument in target):
        raise ValueError("profiler target arguments must be non-empty and NUL-free")
    if sum(len(argument.encode("utf-8")) for argument in target) > MAX_PROFILER_ARGUMENT_BYTES:
        raise ValueError("profiler target arguments exceed 64 KiB")
    return target


def _profiler_command(
    *,
    tool: Literal["powermetrics", "xctrace", "nsys", "ncu"],
    capture_mode: Literal["sidecar", "wrap"],
    argv: tuple[str, ...],
    output_artifact: Path,
    requires_privileges: bool,
    executes_target: bool,
    note: str,
) -> ProfilerCommand:
    return ProfilerCommand(
        tool=tool,
        capture_mode=capture_mode,
        argv=argv,
        rendered_command=shlex.join(argv),
        output_artifact=output_artifact.as_posix(),
        requires_privileges=requires_privileges,
        executes_target=executes_target,
        note=note,
    )


def generate_profiler_commands(
    capabilities: PlatformCapabilities,
    *,
    output_dir: Path,
    target_argv: Sequence[str],
    sample_interval_ms: int = 100,
    sample_count: int = 100,
) -> tuple[ProfilerCommand, ...]:
    """Generate, but never execute, capability-gated external profiler commands."""

    if not 1 <= sample_interval_ms <= 60_000:
        raise ValueError("sample_interval_ms must be between 1 and 60000")
    if not 1 <= sample_count <= 1_000_000:
        raise ValueError("sample_count must be between 1 and 1000000")
    target = _validate_target_argv(target_argv)
    commands: list[ProfilerCommand] = []
    powermetrics = capabilities.profiler_tools["powermetrics"]
    if powermetrics.available:
        assert powermetrics.source is not None
        output = output_dir / "powermetrics.plist"
        powermetrics_argv: tuple[str, ...] = (
            powermetrics.source,
            "--sample-rate",
            str(sample_interval_ms),
            "--sample-count",
            str(sample_count),
            "--format",
            "plist",
            "--samplers",
            "tasks,cpu_power,gpu_power,network,disk",
            "--show-process-energy",
            "--show-process-io",
            "--show-process-gpu",
            "--show-process-netstats",
            "--output-file",
            output.as_posix(),
        )
        commands.append(
            _profiler_command(
                tool="powermetrics",
                capture_mode="sidecar",
                argv=powermetrics_argv,
                output_artifact=output,
                requires_privileges=True,
                executes_target=False,
                note=(
                    "Run as a privileged sidecar while the workload executes; energy estimates "
                    "must not be treated as hardware performance-counter measurements."
                ),
            )
        )
    if target:
        xctrace = capabilities.profiler_tools["xctrace"]
        if xctrace.available:
            assert xctrace.source is not None
            output = output_dir / "helix-time-profiler.trace"
            xctrace_argv: tuple[str, ...] = (
                xctrace.source,
                "record",
                "--template",
                "Time Profiler",
                "--output",
                output.as_posix(),
                "--launch",
                "--",
                *target,
            )
            commands.append(
                _profiler_command(
                    tool="xctrace",
                    capture_mode="wrap",
                    argv=xctrace_argv,
                    output_artifact=output,
                    requires_privileges=False,
                    executes_target=True,
                    note="Wraps the target with the Xcode Time Profiler template.",
                )
            )
        nsys = capabilities.profiler_tools["nsys"]
        if nsys.available:
            assert nsys.source is not None
            output = output_dir / "helix-nsys"
            nsys_argv: tuple[str, ...] = (
                nsys.source,
                "profile",
                "--trace=cuda,nvtx,osrt",
                "--sample=cpu",
                "--cpuctxsw=true",
                "--force-overwrite=true",
                "--output",
                output.as_posix(),
                *target,
            )
            commands.append(
                _profiler_command(
                    tool="nsys",
                    capture_mode="wrap",
                    argv=nsys_argv,
                    output_artifact=output.with_suffix(".nsys-rep"),
                    requires_privileges=False,
                    executes_target=True,
                    note="Captures CUDA, NVTX, OS-runtime, CPU sampling, and context switches.",
                )
            )
        ncu = capabilities.profiler_tools["ncu"]
        if ncu.available:
            assert ncu.source is not None
            output = output_dir / "helix-ncu"
            ncu_argv: tuple[str, ...] = (
                ncu.source,
                "--set",
                "full",
                "--target-processes",
                "all",
                "--export",
                output.as_posix(),
                *target,
            )
            commands.append(
                _profiler_command(
                    tool="ncu",
                    capture_mode="wrap",
                    argv=ncu_argv,
                    output_artifact=output.with_suffix(".ncu-rep"),
                    requires_privileges=False,
                    executes_target=True,
                    note="Captures the full Nsight Compute metric set for target CUDA kernels.",
                )
            )
    return tuple(commands)


class ResourceSampler(AbstractContextManager["ResourceSampler"]):
    """A one-shot periodic sampler with bounded storage and bounded shutdown."""

    def __init__(
        self,
        config: ResourceSamplerConfig,
        *,
        capabilities: PlatformCapabilities | None = None,
        monotonic_ns: Callable[[], int] = time.monotonic_ns,
    ) -> None:
        self.config = config
        self.capabilities = capabilities or detect_platform_capabilities()
        self._monotonic_ns = monotonic_ns
        self._psutil = _load_psutil()
        self._process = self._psutil.Process(os.getpid()) if self._psutil is not None else None
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._state = ResourceSamplerState.NEW
        self._samples: list[ResourceSample] = []
        self._sample_attempts = 0
        self._samples_dropped = 0
        self._collection_error_count = 0
        self._collection_errors: list[str] = []
        self._started_monotonic_ns: int | None = None
        self._ended_monotonic_ns: int | None = None

    @property
    def state(self) -> ResourceSamplerState:
        with self._lock:
            return self._state

    def __enter__(self) -> ResourceSampler:
        return self.start()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.stop()

    def start(self) -> ResourceSampler:
        with self._lock:
            if self._state is not ResourceSamplerState.NEW:
                raise RuntimeError("resource sampler is one-shot and has already been started")
            self._started_monotonic_ns = self._monotonic_ns()
            if self.config.trace_level is TraceLevel.DISABLED:
                self._ended_monotonic_ns = self._started_monotonic_ns
                self._state = ResourceSamplerState.STOPPED
                return self
            self._state = ResourceSamplerState.RUNNING
            self._thread = threading.Thread(
                target=self._run,
                name="sloforge-resource-sampler",
                daemon=True,
            )
            self._thread.start()
        return self

    def wait(self, *, timeout_seconds: float) -> bool:
        if timeout_seconds <= 0.0:
            raise ValueError("timeout_seconds must be positive")
        thread = self._thread
        if thread is None:
            return self.state is ResourceSamplerState.STOPPED
        thread.join(timeout_seconds)
        return not thread.is_alive()

    def stop(self, *, timeout_seconds: float = 5.0) -> ResourceTrace:
        if timeout_seconds <= 0.0:
            raise ValueError("timeout_seconds must be positive")
        with self._lock:
            if self._state is ResourceSamplerState.NEW:
                raise RuntimeError("resource sampler has not been started")
            thread = self._thread
            self._stop_event.set()
        if thread is not None:
            thread.join(timeout_seconds)
            if thread.is_alive():
                raise TimeoutError("resource sampler did not stop before its timeout")
        return self.snapshot()

    def snapshot(self) -> ResourceTrace:
        with self._lock:
            return ResourceTrace(
                schema_version="sloforge.branchfabric.resource-trace/v1",
                config=self.config,
                capabilities=self.capabilities,
                timing_provenance=TimingProvenance.HOST_MONOTONIC_OBSERVED,
                started_monotonic_ns=self._started_monotonic_ns,
                ended_monotonic_ns=self._ended_monotonic_ns,
                samples=tuple(self._samples),
                sample_attempts=self._sample_attempts,
                samples_recorded=len(self._samples),
                samples_dropped=self._samples_dropped,
                buffer_overflowed=self._samples_dropped > 0,
                collection_error_count=self._collection_error_count,
                collection_errors=tuple(self._collection_errors),
                error_records_dropped=(self._collection_error_count - len(self._collection_errors)),
            )

    def _run(self) -> None:
        assert self._started_monotonic_ns is not None
        interval_ns = self.config.sample_interval_ms * 1_000_000
        deadline_ns = (
            self._started_monotonic_ns + int(self.config.max_duration_seconds * 1_000_000_000)
            if self.config.max_duration_seconds is not None
            else None
        )
        next_sample_ns = self._started_monotonic_ns
        try:
            while not self._stop_event.is_set():
                now_ns = self._monotonic_ns()
                if deadline_ns is not None and now_ns >= deadline_ns:
                    break
                self._sample(now_ns)
                next_sample_ns += interval_ns
                now_after_sample_ns = self._monotonic_ns()
                if next_sample_ns <= now_after_sample_ns:
                    next_sample_ns = now_after_sample_ns + interval_ns
                wait_seconds = (next_sample_ns - now_after_sample_ns) / 1_000_000_000
                self._stop_event.wait(wait_seconds)
        finally:
            ended_ns = self._monotonic_ns()
            with self._lock:
                self._ended_monotonic_ns = ended_ns
                self._state = ResourceSamplerState.STOPPED

    def _sample(self, monotonic_ns: int) -> None:
        assert self._started_monotonic_ns is not None
        with self._lock:
            sequence = self._sample_attempts
            self._sample_attempts += 1
        try:
            sample = self._capture_sample(monotonic_ns, sequence=sequence)
        except Exception as exc:  # sampler failures must be visible without killing the workload
            self._record_error(exc)
            return
        with self._lock:
            if len(self._samples) >= self.config.max_samples:
                self._samples_dropped += 1
                return
            self._samples.append(sample)

    def _record_error(self, exc: Exception) -> None:
        record = f"{type(exc).__name__}: {exc}"[:2048]
        with self._lock:
            self._collection_error_count += 1
            if len(self._collection_errors) < MAX_ERROR_RECORDS:
                self._collection_errors.append(record)

    def _capture_sample(self, monotonic_ns: int, *, sequence: int) -> ResourceSample:
        started_monotonic_ns = self._started_monotonic_ns
        assert started_monotonic_ns is not None
        fallback_times = os.times()
        process_cpu_user_seconds = float(fallback_times.user)
        process_cpu_system_seconds = float(fallback_times.system)
        rss: int | None = None
        vms: int | None = None
        thread_count: int | None = None
        read_bytes: int | None = None
        write_bytes: int | None = None
        memory_total: int | None = None
        memory_available: int | None = None
        memory_used: int | None = None
        disk_read_bytes: int | None = None
        disk_write_bytes: int | None = None
        net_receive_bytes: int | None = None
        net_send_bytes: int | None = None
        collector: Literal["psutil", "stdlib-degraded"] = "stdlib-degraded"

        if self._process is not None and self._psutil is not None:
            collector = "psutil"
            cpu_times = self._process.cpu_times()
            process_cpu_user_seconds = float(cpu_times.user)
            process_cpu_system_seconds = float(cpu_times.system)
            memory = self._process.memory_info()
            rss = int(memory.rss)
            vms = int(memory.vms)
            thread_count = int(self._process.num_threads())
            if self.config.trace_level is TraceLevel.FULL:
                try:
                    process_io = self._process.io_counters()
                except (AttributeError, NotImplementedError, PermissionError):
                    process_io = None
                if process_io is not None:
                    read_bytes = int(process_io.read_bytes)
                    write_bytes = int(process_io.write_bytes)
                virtual_memory = self._psutil.virtual_memory()
                memory_total = int(virtual_memory.total)
                memory_available = int(virtual_memory.available)
                memory_used = int(virtual_memory.used)
                disk_io = self._psutil.disk_io_counters()
                if disk_io is not None:
                    disk_read_bytes = int(disk_io.read_bytes)
                    disk_write_bytes = int(disk_io.write_bytes)
                network_io = self._psutil.net_io_counters()
                if network_io is not None:
                    net_receive_bytes = int(network_io.bytes_recv)
                    net_send_bytes = int(network_io.bytes_sent)

        load_1m: float | None
        load_5m: float | None
        load_15m: float | None
        try:
            load_1m, load_5m, load_15m = os.getloadavg()
        except (AttributeError, OSError):
            load_1m = load_5m = load_15m = None
        return ResourceSample(
            sequence=sequence,
            monotonic_ns=monotonic_ns,
            relative_ns=monotonic_ns - started_monotonic_ns,
            timing_provenance=TimingProvenance.HOST_MONOTONIC_OBSERVED,
            collector=collector,
            process_cpu_user_seconds=process_cpu_user_seconds,
            process_cpu_system_seconds=process_cpu_system_seconds,
            process_rss_bytes=rss,
            process_vms_bytes=vms,
            process_thread_count=thread_count,
            process_read_bytes=read_bytes,
            process_write_bytes=write_bytes,
            load_average_1m=load_1m,
            load_average_5m=load_5m,
            load_average_15m=load_15m,
            system_memory_total_bytes=memory_total,
            system_memory_available_bytes=memory_available,
            system_memory_used_bytes=memory_used,
            system_disk_read_bytes=disk_read_bytes,
            system_disk_write_bytes=disk_write_bytes,
            system_network_receive_bytes=net_receive_bytes,
            system_network_send_bytes=net_send_bytes,
            cpu_cycles=None,
            cache_misses=None,
            gpu_utilization_percent=None,
            pcie_receive_bytes=None,
            pcie_send_bytes=None,
        )


def collect_resource_trace(
    config: ResourceSamplerConfig,
    *,
    duration_seconds: float,
    stop_timeout_seconds: float = 5.0,
) -> ResourceTrace:
    """Collect for a bounded duration using the periodic sampler."""

    if duration_seconds < 0.0 or duration_seconds > 86_400.0:
        raise ValueError("duration_seconds must be between 0 and 86400")
    sampler = ResourceSampler(config).start()
    if duration_seconds > 0.0:
        sampler._stop_event.wait(duration_seconds)
    return sampler.stop(timeout_seconds=stop_timeout_seconds)
