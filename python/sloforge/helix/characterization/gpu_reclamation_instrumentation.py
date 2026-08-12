"""Low-overhead hardware instrumentation for BranchFabric Experiment 004.

This module contains no synthetic hardware backend and never launches GPU work.
The production probes are thin wrappers around NVML, psutil, CUDA events, and
bounded ``nvidia-smi`` commands.  Tests inject explicit fixture probes; those
records must not be labelled as hardware evidence by an experiment runner.

Two rules shape the API:

* a missing optional counter is represented by :class:`UnavailableMetric`, not
  by a zero; and
* topology capture and required counter policies are validated separately so
  raw command output can still be persisted when validation fails.

CUDA copy byte counts and host pinned/pageable allocations are application
observations.  NVML cannot recover those exact values after the fact, so callers
must register them at the operation/allocation boundary.
"""

from __future__ import annotations

import importlib
import math
import os
import statistics
import subprocess
import threading
import time
from collections.abc import Callable, Iterable, Sequence
from enum import StrEnum
from pathlib import Path
from types import ModuleType
from typing import Any, Literal, Protocol, Self, TypeVar, cast

from pydantic import BaseModel, ConfigDict, Field, model_validator

MAX_GPU_DEVICES = 16
MAX_PROCESSES = 4096
MAX_SAMPLES = 1_000_000
MAX_CUDA_OPERATION_RECORDS = 1_000_000
MAX_COPY_SIZES_PER_OPERATION = 65_536
MAX_HOST_ALLOCATIONS = 1_000_000
DEFAULT_COMMAND_TIMEOUT_SECONDS = 10.0

T = TypeVar("T")


class InstrumentationUnavailable(RuntimeError):
    """A requested real instrumentation source cannot provide valid evidence."""


class TopologyValidationError(ValueError):
    """Raw topology output exists but does not satisfy the experiment contract."""


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, validate_default=True)


class MetricName(StrEnum):
    PCIE_RX_BYTES_PER_SECOND = "pcie_rx_bytes_per_second"
    PCIE_TX_BYTES_PER_SECOND = "pcie_tx_bytes_per_second"
    PROCESS_GPU_MEMORY_BYTES = "process_gpu_memory_bytes"
    PROCESS_PEAK_RSS_BYTES = "process_peak_rss_bytes"
    HOST_MEMORY_READ_BYTES_PER_SECOND = "host_memory_read_bytes_per_second"
    HOST_MEMORY_WRITE_BYTES_PER_SECOND = "host_memory_write_bytes_per_second"


class UnavailabilityKind(StrEnum):
    PROVIDER_UNAVAILABLE = "provider_unavailable"
    UNSUPPORTED = "unsupported"
    PERMISSION_DENIED = "permission_denied"
    QUERY_FAILED = "query_failed"


class UnavailableMetric(_StrictModel):
    metric: MetricName
    provider: str = Field(min_length=1, max_length=128)
    kind: UnavailabilityKind
    reason: str = Field(min_length=1, max_length=1024)


def _unavailability(
    metric: MetricName,
    provider: str,
    error: BaseException,
) -> UnavailableMetric:
    name = type(error).__name__.lower()
    message = str(error).strip() or type(error).__name__
    if "notsupported" in name or "not supported" in message.lower():
        kind = UnavailabilityKind.UNSUPPORTED
    elif "permission" in name or "permission" in message.lower():
        kind = UnavailabilityKind.PERMISSION_DENIED
    else:
        kind = UnavailabilityKind.QUERY_FAILED
    return UnavailableMetric(metric=metric, provider=provider, kind=kind, reason=message[:1024])


class GpuDescriptor(_StrictModel):
    nvml_index: int = Field(ge=0, lt=MAX_GPU_DEVICES)
    uuid: str = Field(pattern=r"^GPU-[0-9A-Za-z-]+$")
    model: str = Field(min_length=1, max_length=256)
    pci_bus_id: str = Field(min_length=1, max_length=64)
    memory_total_bytes: int = Field(gt=0)


class NvmlInventory(_StrictModel):
    schema_version: Literal["sloforge.branchfabric.nvml-inventory/v1"] = (
        "sloforge.branchfabric.nvml-inventory/v1"
    )
    observed_at_monotonic_ns: int = Field(ge=0)
    provider: Literal["pynvml"] = "pynvml"
    driver_version: str = Field(min_length=1, max_length=128)
    cuda_driver_version_raw: int = Field(ge=0)
    cuda_driver_version: str = Field(pattern=r"^[0-9]+\.[0-9]+$")
    cuda_visible_devices: str | None = Field(default=None, max_length=4096)
    devices: tuple[GpuDescriptor, ...] = Field(min_length=1, max_length=MAX_GPU_DEVICES)

    @model_validator(mode="after")
    def unique_physical_devices(self) -> Self:
        if len({device.uuid for device in self.devices}) != len(self.devices):
            raise ValueError("NVML inventory contains duplicate GPU UUIDs")
        if len({device.nvml_index for device in self.devices}) != len(self.devices):
            raise ValueError("NVML inventory contains duplicate GPU indices")
        return self

    def require_exact_gpu_count(self, expected: int) -> None:
        if expected <= 0:
            raise ValueError("expected GPU count must be positive")
        if len(self.devices) != expected:
            raise InstrumentationUnavailable(
                f"expected exactly {expected} NVML-visible GPUs, observed {len(self.devices)}"
            )


class GpuProcessMemorySample(_StrictModel):
    pid: int = Field(gt=0)
    used_gpu_memory_bytes: int | None = Field(default=None, ge=0)
    unavailable_metrics: tuple[UnavailableMetric, ...] = ()

    @model_validator(mode="after")
    def memory_or_reason(self) -> Self:
        names = {item.metric for item in self.unavailable_metrics}
        missing = self.used_gpu_memory_bytes is None
        explained = MetricName.PROCESS_GPU_MEMORY_BYTES in names
        if missing != explained:
            raise ValueError("missing process GPU memory must have exactly one availability state")
        return self


class GpuSample(_StrictModel):
    schema_version: Literal["sloforge.branchfabric.gpu-sample/v1"] = (
        "sloforge.branchfabric.gpu-sample/v1"
    )
    sequence: int = Field(ge=0, lt=MAX_SAMPLES)
    query_start_monotonic_ns: int = Field(ge=0)
    query_end_monotonic_ns: int = Field(ge=0)
    nvml_index: int = Field(ge=0, lt=MAX_GPU_DEVICES)
    uuid: str = Field(pattern=r"^GPU-[0-9A-Za-z-]+$")
    gpu_utilization_percent: int = Field(ge=0, le=100)
    memory_utilization_percent: int = Field(ge=0, le=100)
    memory_used_bytes: int = Field(ge=0)
    memory_free_bytes: int = Field(ge=0)
    memory_total_bytes: int = Field(gt=0)
    pcie_rx_bytes_per_second: int | None = Field(default=None, ge=0)
    pcie_tx_bytes_per_second: int | None = Field(default=None, ge=0)
    compute_processes: tuple[GpuProcessMemorySample, ...] = Field(
        default=(), max_length=MAX_PROCESSES
    )
    unavailable_metrics: tuple[UnavailableMetric, ...] = ()

    @model_validator(mode="after")
    def sample_is_consistent(self) -> Self:
        if self.query_end_monotonic_ns < self.query_start_monotonic_ns:
            raise ValueError("GPU query ended before it started")
        if self.memory_used_bytes + self.memory_free_bytes > self.memory_total_bytes:
            raise ValueError("NVML used plus free memory exceeds total memory")
        unavailable = [item.metric for item in self.unavailable_metrics]
        if len(unavailable) != len(set(unavailable)):
            raise ValueError("GPU sample has duplicate metric-unavailability records")
        for metric, value in (
            (MetricName.PCIE_RX_BYTES_PER_SECOND, self.pcie_rx_bytes_per_second),
            (MetricName.PCIE_TX_BYTES_PER_SECOND, self.pcie_tx_bytes_per_second),
        ):
            if (value is None) != (metric in unavailable):
                raise ValueError(f"{metric} must contain a value or an unavailability record")
        return self


def _decode_nvml_text(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="strict")
    return str(value)


def _format_cuda_driver_version(raw: int) -> str:
    # CUDA driver API integer encoding: 1000 * major + 10 * minor.
    return f"{raw // 1000}.{(raw % 1000) // 10}"


class NvmlProbe:
    """Explicit pynvml probe; construction fails if NVML cannot initialize."""

    def __init__(
        self,
        *,
        nvml_module: ModuleType | object | None = None,
        clock_ns: Callable[[], int] = time.perf_counter_ns,
    ) -> None:
        try:
            module = nvml_module or importlib.import_module("pynvml")
        except (ImportError, OSError) as exc:
            raise InstrumentationUnavailable(f"pynvml provider unavailable: {exc}") from exc
        self._nvml = cast(Any, module)
        self._clock_ns = clock_ns
        self._closed = False
        try:
            self._nvml.nvmlInit()
        except Exception as exc:
            raise InstrumentationUnavailable(f"NVML initialization failed: {exc}") from exc

    def __enter__(self) -> NvmlProbe:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def close(self) -> None:
        if not self._closed:
            self._nvml.nvmlShutdown()
            self._closed = True

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("NVML probe is closed")

    def inventory(self) -> NvmlInventory:
        self._require_open()
        observed_ns = self._clock_ns()
        try:
            count = int(self._nvml.nvmlDeviceGetCount())
            driver = _decode_nvml_text(self._nvml.nvmlSystemGetDriverVersion())
            cuda_raw = int(self._nvml.nvmlSystemGetCudaDriverVersion_v2())
            devices = tuple(self._descriptor(index) for index in range(count))
        except Exception as exc:
            raise InstrumentationUnavailable(f"NVML inventory query failed: {exc}") from exc
        return NvmlInventory(
            observed_at_monotonic_ns=observed_ns,
            driver_version=driver,
            cuda_driver_version_raw=cuda_raw,
            cuda_driver_version=_format_cuda_driver_version(cuda_raw),
            cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
            devices=devices,
        )

    def _descriptor(self, index: int) -> GpuDescriptor:
        handle = self._nvml.nvmlDeviceGetHandleByIndex(index)
        pci = self._nvml.nvmlDeviceGetPciInfo(handle)
        memory = self._nvml.nvmlDeviceGetMemoryInfo(handle)
        return GpuDescriptor(
            nvml_index=index,
            uuid=_decode_nvml_text(self._nvml.nvmlDeviceGetUUID(handle)),
            model=_decode_nvml_text(self._nvml.nvmlDeviceGetName(handle)),
            pci_bus_id=_decode_nvml_text(pci.busId),
            memory_total_bytes=int(memory.total),
        )

    def sample(
        self,
        *,
        sequence: int,
        descriptors: Sequence[GpuDescriptor],
        require_pcie: bool = False,
    ) -> tuple[GpuSample, ...]:
        self._require_open()
        if not descriptors:
            raise ValueError("at least one GPU descriptor is required")
        samples = tuple(
            self._sample_device(sequence=sequence, descriptor=descriptor)
            for descriptor in descriptors
        )
        if require_pcie:
            missing = [
                f"{sample.uuid}:{item.metric}"
                for sample in samples
                for item in sample.unavailable_metrics
                if item.metric
                in {
                    MetricName.PCIE_RX_BYTES_PER_SECOND,
                    MetricName.PCIE_TX_BYTES_PER_SECOND,
                }
            ]
            if missing:
                raise InstrumentationUnavailable(
                    "required NVML PCIe counters unavailable: " + ", ".join(missing)
                )
        return samples

    def _sample_device(self, *, sequence: int, descriptor: GpuDescriptor) -> GpuSample:
        start_ns = self._clock_ns()
        handle = self._nvml.nvmlDeviceGetHandleByIndex(descriptor.nvml_index)
        try:
            utilization = self._nvml.nvmlDeviceGetUtilizationRates(handle)
            memory = self._nvml.nvmlDeviceGetMemoryInfo(handle)
            processes = self._compute_processes(handle)
        except Exception as exc:
            raise InstrumentationUnavailable(
                f"required NVML sample failed for {descriptor.uuid}: {exc}"
            ) from exc

        unavailable: list[UnavailableMetric] = []
        pcie: dict[MetricName, int | None] = {}
        for metric, counter_name in (
            (MetricName.PCIE_RX_BYTES_PER_SECOND, "NVML_PCIE_UTIL_RX_BYTES"),
            (MetricName.PCIE_TX_BYTES_PER_SECOND, "NVML_PCIE_UTIL_TX_BYTES"),
        ):
            try:
                counter = getattr(self._nvml, counter_name)
                # NVML reports PCIe throughput in KiB/s over a short sampling window.
                pcie[metric] = int(self._nvml.nvmlDeviceGetPcieThroughput(handle, counter)) * 1024
            except Exception as exc:
                pcie[metric] = None
                unavailable.append(_unavailability(metric, "pynvml", exc))

        end_ns = self._clock_ns()
        return GpuSample(
            sequence=sequence,
            query_start_monotonic_ns=start_ns,
            query_end_monotonic_ns=end_ns,
            nvml_index=descriptor.nvml_index,
            uuid=descriptor.uuid,
            gpu_utilization_percent=int(utilization.gpu),
            memory_utilization_percent=int(utilization.memory),
            memory_used_bytes=int(memory.used),
            memory_free_bytes=int(memory.free),
            memory_total_bytes=int(memory.total),
            pcie_rx_bytes_per_second=pcie[MetricName.PCIE_RX_BYTES_PER_SECOND],
            pcie_tx_bytes_per_second=pcie[MetricName.PCIE_TX_BYTES_PER_SECOND],
            compute_processes=processes,
            unavailable_metrics=tuple(unavailable),
        )

    def _compute_processes(self, handle: object) -> tuple[GpuProcessMemorySample, ...]:
        processes = self._nvml.nvmlDeviceGetComputeRunningProcesses(handle)
        if len(processes) > MAX_PROCESSES:
            raise InstrumentationUnavailable(
                f"NVML returned {len(processes)} processes; bounded limit is {MAX_PROCESSES}"
            )
        unavailable_value = getattr(self._nvml, "NVML_VALUE_NOT_AVAILABLE", None)
        result: list[GpuProcessMemorySample] = []
        for process in processes:
            used = getattr(process, "usedGpuMemory", None)
            if used is None or used == unavailable_value:
                result.append(
                    GpuProcessMemorySample(
                        pid=int(process.pid),
                        used_gpu_memory_bytes=None,
                        unavailable_metrics=(
                            UnavailableMetric(
                                metric=MetricName.PROCESS_GPU_MEMORY_BYTES,
                                provider="pynvml",
                                kind=UnavailabilityKind.UNSUPPORTED,
                                reason="NVML did not expose per-process GPU memory",
                            ),
                        ),
                    )
                )
            else:
                result.append(
                    GpuProcessMemorySample(pid=int(process.pid), used_gpu_memory_bytes=int(used))
                )
        return tuple(result)


class ProcessResourceSample(_StrictModel):
    pid: int = Field(gt=0)
    rss_bytes: int = Field(ge=0)
    peak_rss_bytes: int | None = Field(default=None, ge=0)
    cpu_user_ns: int = Field(ge=0)
    cpu_system_ns: int = Field(ge=0)
    thread_count: int = Field(ge=0)
    unavailable_metrics: tuple[UnavailableMetric, ...] = ()

    @model_validator(mode="after")
    def peak_or_reason(self) -> Self:
        missing = self.peak_rss_bytes is None
        explained = any(
            item.metric is MetricName.PROCESS_PEAK_RSS_BYTES for item in self.unavailable_metrics
        )
        if missing != explained:
            raise ValueError("missing peak RSS must have exactly one availability state")
        return self


class HostResourceSample(_StrictModel):
    schema_version: Literal["sloforge.branchfabric.host-resource-sample/v1"] = (
        "sloforge.branchfabric.host-resource-sample/v1"
    )
    sequence: int = Field(ge=0, lt=MAX_SAMPLES)
    query_start_monotonic_ns: int = Field(ge=0)
    query_end_monotonic_ns: int = Field(ge=0)
    provider: Literal["psutil"] = "psutil"
    logical_cpu_count: int = Field(gt=0)
    system_cpu_user_ns: int = Field(ge=0)
    system_cpu_system_ns: int = Field(ge=0)
    system_cpu_idle_ns: int = Field(ge=0)
    host_memory_total_bytes: int = Field(gt=0)
    host_memory_available_bytes: int = Field(ge=0)
    processes: tuple[ProcessResourceSample, ...] = Field(max_length=MAX_PROCESSES)
    host_memory_read_bytes_per_second: int | None = Field(default=None, ge=0)
    host_memory_write_bytes_per_second: int | None = Field(default=None, ge=0)
    unavailable_metrics: tuple[UnavailableMetric, ...] = ()

    @model_validator(mode="after")
    def sample_is_consistent(self) -> Self:
        if self.query_end_monotonic_ns < self.query_start_monotonic_ns:
            raise ValueError("host query ended before it started")
        if self.host_memory_available_bytes > self.host_memory_total_bytes:
            raise ValueError("available host memory exceeds total memory")
        unavailable = {item.metric for item in self.unavailable_metrics}
        for metric, value in (
            (
                MetricName.HOST_MEMORY_READ_BYTES_PER_SECOND,
                self.host_memory_read_bytes_per_second,
            ),
            (
                MetricName.HOST_MEMORY_WRITE_BYTES_PER_SECOND,
                self.host_memory_write_bytes_per_second,
            ),
        ):
            if (value is None) != (metric in unavailable):
                raise ValueError(f"{metric} must contain a value or an unavailability record")
        return self


class PsutilHostProbe:
    """Cumulative CPU and memory probe for an explicit, bounded PID set."""

    def __init__(
        self,
        *,
        process_ids: Sequence[int],
        psutil_module: ModuleType | object | None = None,
        clock_ns: Callable[[], int] = time.perf_counter_ns,
    ) -> None:
        if not process_ids:
            raise ValueError("at least one experiment-owned PID is required")
        if len(process_ids) > MAX_PROCESSES:
            raise ValueError(f"process count exceeds bounded limit {MAX_PROCESSES}")
        if any(pid <= 0 for pid in process_ids) or len(set(process_ids)) != len(process_ids):
            raise ValueError("process IDs must be unique positive integers")
        try:
            module = psutil_module or importlib.import_module("psutil")
        except (ImportError, OSError) as exc:
            raise InstrumentationUnavailable(f"psutil provider unavailable: {exc}") from exc
        self._psutil = cast(Any, module)
        self._process_ids = tuple(process_ids)
        self._clock_ns = clock_ns

    @staticmethod
    def _peak_rss(pid: int) -> tuple[int | None, UnavailableMetric | None]:
        status = Path(f"/proc/{pid}/status")
        try:
            for line in status.read_text(encoding="utf-8").splitlines():
                if line.startswith("VmHWM:"):
                    fields = line.split()
                    if len(fields) != 3 or fields[2] != "kB":
                        raise ValueError(f"unexpected VmHWM format: {line}")
                    return int(fields[1]) * 1024, None
        except (OSError, ValueError) as exc:
            return None, _unavailability(MetricName.PROCESS_PEAK_RSS_BYTES, "procfs", exc)
        return None, UnavailableMetric(
            metric=MetricName.PROCESS_PEAK_RSS_BYTES,
            provider="procfs",
            kind=UnavailabilityKind.UNSUPPORTED,
            reason="VmHWM is not present in procfs status",
        )

    def sample(self, *, sequence: int) -> HostResourceSample:
        start_ns = self._clock_ns()
        process_samples: list[ProcessResourceSample] = []
        try:
            for pid in self._process_ids:
                process = self._psutil.Process(pid)
                memory = process.memory_info()
                cpu = process.cpu_times()
                peak, unavailable = self._peak_rss(pid)
                process_samples.append(
                    ProcessResourceSample(
                        pid=pid,
                        rss_bytes=int(memory.rss),
                        peak_rss_bytes=peak,
                        cpu_user_ns=int(float(cpu.user) * 1_000_000_000),
                        cpu_system_ns=int(float(cpu.system) * 1_000_000_000),
                        thread_count=int(process.num_threads()),
                        unavailable_metrics=() if unavailable is None else (unavailable,),
                    )
                )
            system_cpu = self._psutil.cpu_times()
            virtual_memory = self._psutil.virtual_memory()
            cpu_count = self._psutil.cpu_count(logical=True)
        except Exception as exc:
            raise InstrumentationUnavailable(f"psutil resource query failed: {exc}") from exc
        if cpu_count is None or int(cpu_count) <= 0:
            raise InstrumentationUnavailable("psutil did not report a positive logical CPU count")
        unavailable_bandwidth = (
            UnavailableMetric(
                metric=MetricName.HOST_MEMORY_READ_BYTES_PER_SECOND,
                provider="psutil",
                kind=UnavailabilityKind.UNSUPPORTED,
                reason="psutil does not expose host DRAM-controller bandwidth",
            ),
            UnavailableMetric(
                metric=MetricName.HOST_MEMORY_WRITE_BYTES_PER_SECOND,
                provider="psutil",
                kind=UnavailabilityKind.UNSUPPORTED,
                reason="psutil does not expose host DRAM-controller bandwidth",
            ),
        )
        return HostResourceSample(
            sequence=sequence,
            query_start_monotonic_ns=start_ns,
            query_end_monotonic_ns=self._clock_ns(),
            logical_cpu_count=int(cpu_count),
            system_cpu_user_ns=int(float(system_cpu.user) * 1_000_000_000),
            system_cpu_system_ns=int(float(system_cpu.system) * 1_000_000_000),
            system_cpu_idle_ns=int(float(system_cpu.idle) * 1_000_000_000),
            host_memory_total_bytes=int(virtual_memory.total),
            host_memory_available_bytes=int(virtual_memory.available),
            processes=tuple(process_samples),
            unavailable_metrics=unavailable_bandwidth,
        )


class ResourceSample(_StrictModel):
    schema_version: Literal["sloforge.branchfabric.resource-sample/v1"] = (
        "sloforge.branchfabric.resource-sample/v1"
    )
    sequence: int = Field(ge=0, lt=MAX_SAMPLES)
    sample_trigger_monotonic_ns: int = Field(ge=0)
    gpu_samples: tuple[GpuSample, ...] = Field(min_length=1, max_length=MAX_GPU_DEVICES)
    host_sample: HostResourceSample

    @model_validator(mode="after")
    def sequences_align(self) -> Self:
        if self.host_sample.sequence != self.sequence or any(
            sample.sequence != self.sequence for sample in self.gpu_samples
        ):
            raise ValueError("resource sample sequence numbers do not align")
        return self


class GpuSampleProbe(Protocol):
    def sample(
        self,
        *,
        sequence: int,
        descriptors: Sequence[GpuDescriptor],
        require_pcie: bool = False,
    ) -> tuple[GpuSample, ...]: ...


class HostSampleProbe(Protocol):
    def sample(self, *, sequence: int) -> HostResourceSample: ...


class SamplingConfig(_StrictModel):
    interval_ms: int = Field(default=50, ge=10, le=60_000)
    max_samples: int = Field(default=72_000, ge=1, le=MAX_SAMPLES)
    require_pcie_counters: bool = False


class ResourceSamplingResult(_StrictModel):
    schema_version: Literal["sloforge.branchfabric.resource-sampling/v1"] = (
        "sloforge.branchfabric.resource-sampling/v1"
    )
    config: SamplingConfig
    samples: tuple[ResourceSample, ...] = Field(max_length=MAX_SAMPLES)
    reached_sample_bound: bool


class LowOverheadResourceSampler:
    """Bounded periodic sampler with explicit background-thread failures."""

    def __init__(
        self,
        *,
        gpu_probe: GpuSampleProbe,
        host_probe: HostSampleProbe,
        descriptors: Sequence[GpuDescriptor],
        config: SamplingConfig,
        clock_ns: Callable[[], int] = time.perf_counter_ns,
    ) -> None:
        if not descriptors:
            raise ValueError("at least one GPU descriptor is required")
        self._gpu_probe = gpu_probe
        self._host_probe = host_probe
        self._descriptors = tuple(descriptors)
        self._config = config
        self._clock_ns = clock_ns
        self._samples: list[ResourceSample] = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._failure: BaseException | None = None
        self._reached_bound = False

    @property
    def is_running(self) -> bool:
        """Whether the bounded sampler thread is still alive."""

        return self._thread is not None and self._thread.is_alive()

    def sample_once(self) -> ResourceSample:
        with self._lock:
            sequence = len(self._samples)
            if sequence >= self._config.max_samples:
                self._reached_bound = True
                raise InstrumentationUnavailable("resource sampler reached its configured bound")
        trigger_ns = self._clock_ns()
        gpu_samples = self._gpu_probe.sample(
            sequence=sequence,
            descriptors=self._descriptors,
            require_pcie=self._config.require_pcie_counters,
        )
        if tuple(sample.uuid for sample in gpu_samples) != tuple(
            descriptor.uuid for descriptor in self._descriptors
        ):
            raise InstrumentationUnavailable("GPU probe sample order/UUIDs changed during trial")
        sample = ResourceSample(
            sequence=sequence,
            sample_trigger_monotonic_ns=trigger_ns,
            gpu_samples=gpu_samples,
            host_sample=self._host_probe.sample(sequence=sequence),
        )
        with self._lock:
            if len(self._samples) != sequence:
                raise RuntimeError("resource sampler does not support concurrent sample_once calls")
            self._samples.append(sample)
        return sample

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("resource sampler has already been started")
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="sloforge-exp004-resource-sampler",
            daemon=True,
        )
        self._thread.start()

    def _run(self) -> None:
        interval_seconds = self._config.interval_ms / 1000.0
        try:
            while not self._stop.is_set():
                self.sample_once()
                if self._stop.wait(interval_seconds):
                    return
        except BaseException as exc:
            self._failure = exc
            self._stop.set()

    def stop(self, *, timeout_seconds: float = 5.0) -> ResourceSamplingResult:
        if timeout_seconds <= 0 or not math.isfinite(timeout_seconds):
            raise ValueError("sampler stop timeout must be positive and finite")
        thread = self._thread
        if thread is None:
            raise RuntimeError("resource sampler has not been started")
        self._stop.set()
        thread.join(timeout=timeout_seconds)
        if thread.is_alive():
            raise TimeoutError("resource sampler did not stop before its bounded deadline")
        if self._failure is not None:
            raise InstrumentationUnavailable(
                f"resource sampler failed: {self._failure}"
            ) from self._failure
        with self._lock:
            return ResourceSamplingResult(
                config=self._config,
                samples=tuple(self._samples),
                reached_sample_bound=self._reached_bound,
            )


class TopologyCommandName(StrEnum):
    GPU_INVENTORY = "gpu_inventory"
    TOPOLOGY_MATRIX = "topology_matrix"
    P2P_READ = "p2p_read"
    P2P_WRITE = "p2p_write"
    NVLINK_STATUS = "nvlink_status"


class TopologyCommand(_StrictModel):
    name: TopologyCommandName
    argv: tuple[str, ...] = Field(min_length=1, max_length=32)
    timeout_seconds: float = Field(gt=0.0, le=60.0, allow_inf_nan=False)
    required: bool


def topology_command_plan(
    *, timeout_seconds: float = DEFAULT_COMMAND_TIMEOUT_SECONDS
) -> tuple[TopologyCommand, ...]:
    """Return the exact bounded command plan; this function executes nothing."""

    if timeout_seconds <= 0 or timeout_seconds > 60 or not math.isfinite(timeout_seconds):
        raise ValueError("topology command timeout must be finite and in (0, 60]")
    return (
        TopologyCommand(
            name=TopologyCommandName.GPU_INVENTORY,
            argv=(
                "nvidia-smi",
                "--query-gpu=index,uuid,name,pci.bus_id,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ),
            timeout_seconds=timeout_seconds,
            required=True,
        ),
        TopologyCommand(
            name=TopologyCommandName.TOPOLOGY_MATRIX,
            argv=("nvidia-smi", "topo", "-m"),
            timeout_seconds=timeout_seconds,
            required=True,
        ),
        TopologyCommand(
            name=TopologyCommandName.P2P_READ,
            argv=("nvidia-smi", "topo", "-p2p", "r"),
            timeout_seconds=timeout_seconds,
            required=True,
        ),
        TopologyCommand(
            name=TopologyCommandName.P2P_WRITE,
            argv=("nvidia-smi", "topo", "-p2p", "w"),
            timeout_seconds=timeout_seconds,
            required=True,
        ),
        TopologyCommand(
            name=TopologyCommandName.NVLINK_STATUS,
            argv=("nvidia-smi", "nvlink", "--status"),
            timeout_seconds=timeout_seconds,
            required=False,
        ),
    )


class CommandStatus(StrEnum):
    SUCCESS = "success"
    EXECUTABLE_UNAVAILABLE = "executable_unavailable"
    TIMED_OUT = "timed_out"
    NONZERO_EXIT = "nonzero_exit"


class TopologyCommandResult(_StrictModel):
    command: TopologyCommand
    status: CommandStatus
    start_monotonic_ns: int = Field(ge=0)
    duration_ns: int = Field(ge=0)
    returncode: int | None = None
    stdout: str = Field(max_length=4_000_000)
    stderr: str = Field(max_length=4_000_000)


CommandRunner = Callable[..., subprocess.CompletedProcess[str]]


def capture_topology_commands(
    *,
    plan: Sequence[TopologyCommand] | None = None,
    runner: CommandRunner = subprocess.run,
    clock_ns: Callable[[], int] = time.perf_counter_ns,
) -> tuple[TopologyCommandResult, ...]:
    """Capture raw command evidence without a shell and with per-command timeouts."""

    commands = tuple(plan or topology_command_plan())
    results: list[TopologyCommandResult] = []
    for command in commands:
        start_ns = clock_ns()
        try:
            completed = runner(
                command.argv,
                capture_output=True,
                check=False,
                text=True,
                timeout=command.timeout_seconds,
            )
            status = (
                CommandStatus.SUCCESS if completed.returncode == 0 else CommandStatus.NONZERO_EXIT
            )
            returncode: int | None = completed.returncode
            stdout = completed.stdout
            stderr = completed.stderr
        except FileNotFoundError as exc:
            status = CommandStatus.EXECUTABLE_UNAVAILABLE
            returncode = None
            stdout = ""
            stderr = str(exc)
        except subprocess.TimeoutExpired as exc:
            status = CommandStatus.TIMED_OUT
            returncode = None
            stdout = _decode_subprocess_stream(exc.stdout)
            stderr = _decode_subprocess_stream(exc.stderr)
        end_ns = clock_ns()
        results.append(
            TopologyCommandResult(
                command=command,
                status=status,
                start_monotonic_ns=start_ns,
                duration_ns=end_ns - start_ns,
                returncode=returncode,
                stdout=stdout,
                stderr=stderr,
            )
        )
    return tuple(results)


def _decode_subprocess_stream(value: str | bytes | None) -> str:
    if value is None:
        return ""
    return value.decode("utf-8", errors="replace") if isinstance(value, bytes) else value


def validate_topology_capture(
    results: Sequence[TopologyCommandResult], *, expected_gpu_count: int
) -> None:
    """Fail if required commands failed or inventory is not exactly as requested."""

    if expected_gpu_count <= 0:
        raise ValueError("expected GPU count must be positive")
    by_name = {result.command.name: result for result in results}
    # The matrix command must always be attempted and preserved, but some
    # containerized A100 hosts hide PCI bus IDs and make only `topo -m` return
    # 255.  Bidirectional P2P queries plus NVLink status remain scientifically
    # useful topology evidence in that environment.
    required_names = {command.name for command in topology_command_plan() if command.required}
    missing = sorted(name.value for name in required_names - set(by_name))
    strict_success_names = required_names - {TopologyCommandName.TOPOLOGY_MATRIX}
    failed = sorted(
        name.value
        for name in strict_success_names
        if name in by_name and by_name[name].status is not CommandStatus.SUCCESS
    )
    if missing or failed:
        raise TopologyValidationError(
            f"required topology evidence missing={missing} failed={failed}"
        )
    matrix = by_name[TopologyCommandName.TOPOLOGY_MATRIX]
    if matrix.status is not CommandStatus.SUCCESS:
        nvlink = by_name.get(TopologyCommandName.NVLINK_STATUS)
        fallback_valid = (
            matrix.status is CommandStatus.NONZERO_EXIT
            and matrix.returncode is not None
            and bool(matrix.stdout.strip())
            and by_name[TopologyCommandName.P2P_READ].status is CommandStatus.SUCCESS
            and by_name[TopologyCommandName.P2P_WRITE].status is CommandStatus.SUCCESS
            and nvlink is not None
            and nvlink.status is CommandStatus.SUCCESS
            and bool(nvlink.stdout.strip())
        )
        if not fallback_valid:
            raise TopologyValidationError(
                "topology matrix failed without complete P2P/NVLink fallback evidence"
            )
    inventory = by_name[TopologyCommandName.GPU_INVENTORY].stdout
    rows = [line for line in inventory.splitlines() if line.strip()]
    if len(rows) != expected_gpu_count:
        raise TopologyValidationError(
            f"expected exactly {expected_gpu_count} nvidia-smi inventory rows, observed {len(rows)}"
        )
    for row in rows:
        fields = [field.strip() for field in row.split(",")]
        if len(fields) != 6 or not fields[1].startswith("GPU-"):
            raise TopologyValidationError(f"invalid nvidia-smi inventory row: {row!r}")


class PeerAccessRecord(_StrictModel):
    source_logical_device: int = Field(ge=0, lt=MAX_GPU_DEVICES)
    destination_logical_device: int = Field(ge=0, lt=MAX_GPU_DEVICES)
    can_access_peer: bool


class CudaPeerAccessMatrix(_StrictModel):
    schema_version: Literal["sloforge.branchfabric.cuda-peer-access/v1"] = (
        "sloforge.branchfabric.cuda-peer-access/v1"
    )
    observed_at_monotonic_ns: int = Field(ge=0)
    device_count: int = Field(gt=0, le=MAX_GPU_DEVICES)
    device_names: tuple[str, ...] = Field(min_length=1, max_length=MAX_GPU_DEVICES)
    access: tuple[PeerAccessRecord, ...]


def capture_cuda_peer_access(
    *,
    expected_gpu_count: int,
    torch_module: ModuleType | object | None = None,
    clock_ns: Callable[[], int] = time.perf_counter_ns,
) -> CudaPeerAccessMatrix:
    """Query CUDA peer capability explicitly in the CUDA-owning child process."""

    try:
        torch = cast(Any, torch_module or importlib.import_module("torch"))
    except (ImportError, OSError) as exc:
        raise InstrumentationUnavailable(f"PyTorch CUDA provider unavailable: {exc}") from exc
    if not bool(torch.cuda.is_available()):
        raise InstrumentationUnavailable("PyTorch reports CUDA unavailable")
    count = int(torch.cuda.device_count())
    if count != expected_gpu_count:
        raise InstrumentationUnavailable(
            f"expected exactly {expected_gpu_count} CUDA devices, observed {count}"
        )
    names = tuple(str(torch.cuda.get_device_name(index)) for index in range(count))
    try:
        access = tuple(
            PeerAccessRecord(
                source_logical_device=source,
                destination_logical_device=destination,
                can_access_peer=bool(torch.cuda.can_device_access_peer(source, destination)),
            )
            for source in range(count)
            for destination in range(count)
            if source != destination
        )
    except Exception as exc:
        raise InstrumentationUnavailable(f"CUDA peer-access query failed: {exc}") from exc
    return CudaPeerAccessMatrix(
        observed_at_monotonic_ns=clock_ns(),
        device_count=count,
        device_names=names,
        access=access,
    )


class CudaOperationKind(StrEnum):
    D2H = "d2h"
    H2D = "h2d"
    D2D = "d2d"
    TRANSFORM = "transform"
    CHECKSUM = "checksum"
    OTHER = "other"


_COPY_KINDS = frozenset({CudaOperationKind.D2H, CudaOperationKind.H2D, CudaOperationKind.D2D})


class CudaOperationRecord(_StrictModel):
    schema_version: Literal["sloforge.branchfabric.cuda-operation/v1"] = (
        "sloforge.branchfabric.cuda-operation/v1"
    )
    operation_id: str = Field(min_length=1, max_length=256)
    kind: CudaOperationKind
    logical_device: int = Field(ge=0, lt=MAX_GPU_DEVICES)
    stream_id: str = Field(min_length=1, max_length=128)
    cpu_start_monotonic_ns: int = Field(ge=0)
    cpu_end_monotonic_ns: int = Field(ge=0)
    cpu_launch_ns: int = Field(ge=0)
    cpu_launch_ns_semantics: Literal[
        "python-operation-call; synchronous APIs may include completion"
    ] = "python-operation-call; synchronous APIs may include completion"
    synchronization_wait_ns: int = Field(ge=0)
    cuda_event_elapsed_ns: int = Field(ge=0)
    bytes: int = Field(ge=0)
    byte_semantics: str = Field(
        default="operation payload extent",
        min_length=1,
        max_length=256,
    )
    copy_sizes_bytes: tuple[int, ...] = Field(max_length=MAX_COPY_SIZES_PER_OPERATION)

    @model_validator(mode="after")
    def operation_is_consistent(self) -> Self:
        if self.cpu_end_monotonic_ns < self.cpu_start_monotonic_ns:
            raise ValueError("CUDA operation end precedes start")
        if self.kind in _COPY_KINDS:
            if not self.copy_sizes_bytes or any(size <= 0 for size in self.copy_sizes_bytes):
                raise ValueError("copy operations require positive per-copy sizes")
            if sum(self.copy_sizes_bytes) != self.bytes:
                raise ValueError("copy sizes must sum exactly to operation bytes")
        elif self.copy_sizes_bytes:
            raise ValueError("non-copy CUDA operations cannot contain copy sizes")
        return self

    @property
    def copy_count(self) -> int:
        return len(self.copy_sizes_bytes)


class CudaOperationSummary(_StrictModel):
    schema_version: Literal["sloforge.branchfabric.cuda-operation-summary/v1"] = (
        "sloforge.branchfabric.cuda-operation-summary/v1"
    )
    operation_count: int = Field(ge=0)
    copy_count: int = Field(ge=0)
    bytes_by_kind: dict[CudaOperationKind, int]
    copy_count_by_kind: dict[CudaOperationKind, int]
    copy_sizes_bytes_by_kind: dict[CudaOperationKind, tuple[int, ...]]
    copy_size_min_bytes: int | None = Field(default=None, ge=0)
    copy_size_p50_bytes: int | None = Field(default=None, ge=0)
    copy_size_max_bytes: int | None = Field(default=None, ge=0)
    cuda_event_elapsed_ns: int = Field(ge=0)
    cpu_launch_ns: int = Field(ge=0)
    cpu_launch_ns_semantics: Literal[
        "sum of Python operation calls; synchronous APIs may include completion"
    ] = "sum of Python operation calls; synchronous APIs may include completion"
    synchronization_wait_ns: int = Field(ge=0)

    @model_validator(mode="after")
    def copy_distribution_is_consistent(self) -> Self:
        for kind in _COPY_KINDS:
            sizes = self.copy_sizes_bytes_by_kind.get(kind)
            if sizes is None or any(size <= 0 for size in sizes):
                raise ValueError("every copy kind requires an explicit nonzero size tuple")
            if len(sizes) != self.copy_count_by_kind.get(kind):
                raise ValueError("copy size distribution and per-kind count disagree")
            if sum(sizes) != self.bytes_by_kind.get(kind):
                raise ValueError("copy size distribution and per-kind bytes disagree")
        if self.copy_count != sum(self.copy_count_by_kind.values()):
            raise ValueError("aggregate copy count and per-kind counts disagree")
        return self


def summarize_cuda_operations(records: Iterable[CudaOperationRecord]) -> CudaOperationSummary:
    materialized = tuple(records)
    if len(materialized) > MAX_CUDA_OPERATION_RECORDS:
        raise ValueError(
            f"CUDA operation records exceed bounded limit {MAX_CUDA_OPERATION_RECORDS}"
        )
    operation_ids = [record.operation_id for record in materialized]
    if len(operation_ids) != len(set(operation_ids)):
        raise ValueError("CUDA operation IDs must be unique")
    bytes_by_kind = {kind: 0 for kind in CudaOperationKind}
    copy_count_by_kind = {kind: 0 for kind in _COPY_KINDS}
    sizes: list[int] = []
    for record in materialized:
        bytes_by_kind[record.kind] += record.bytes
        if record.kind in _COPY_KINDS:
            copy_count_by_kind[record.kind] += record.copy_count
            sizes.extend(record.copy_sizes_bytes)
    sizes.sort()
    p50 = int(statistics.median(sizes)) if sizes else None
    return CudaOperationSummary(
        operation_count=len(materialized),
        copy_count=len(sizes),
        bytes_by_kind=bytes_by_kind,
        copy_count_by_kind=copy_count_by_kind,
        copy_sizes_bytes_by_kind={
            kind: tuple(
                sorted(
                    size
                    for record in materialized
                    if record.kind is kind
                    for size in record.copy_sizes_bytes
                )
            )
            for kind in _COPY_KINDS
        },
        copy_size_min_bytes=sizes[0] if sizes else None,
        copy_size_p50_bytes=p50,
        copy_size_max_bytes=sizes[-1] if sizes else None,
        cuda_event_elapsed_ns=sum(record.cuda_event_elapsed_ns for record in materialized),
        cpu_launch_ns=sum(record.cpu_launch_ns for record in materialized),
        synchronization_wait_ns=sum(record.synchronization_wait_ns for record in materialized),
    )


class CudaEventRecorder:
    """Time one explicitly selected CUDA stream and retain exact copy metadata."""

    def __init__(
        self,
        *,
        logical_device: int,
        torch_module: ModuleType | object | None = None,
        clock_ns: Callable[[], int] = time.perf_counter_ns,
        enabled: bool = True,
    ) -> None:
        try:
            torch = cast(Any, torch_module or importlib.import_module("torch"))
        except (ImportError, OSError) as exc:
            raise InstrumentationUnavailable(f"PyTorch CUDA provider unavailable: {exc}") from exc
        if not bool(torch.cuda.is_available()):
            raise InstrumentationUnavailable("PyTorch reports CUDA unavailable")
        count = int(torch.cuda.device_count())
        if logical_device < 0 or logical_device >= count:
            raise InstrumentationUnavailable(
                f"requested CUDA device {logical_device}, but CUDA device count is {count}"
            )
        self._torch = torch
        self._logical_device = logical_device
        self._clock_ns = clock_ns
        self._enabled = enabled
        self._records: list[CudaOperationRecord] = []

    @property
    def records(self) -> tuple[CudaOperationRecord, ...]:
        return tuple(self._records)

    def measure(
        self,
        *,
        operation_id: str,
        kind: CudaOperationKind,
        operation: Callable[[], T],
        stream: object | None = None,
        stream_id: str,
        bytes: int = 0,
        byte_semantics: str = "operation payload extent",
        copy_sizes_bytes: Sequence[int] = (),
    ) -> T:
        if not self._enabled:
            return operation()
        if len(self._records) >= MAX_CUDA_OPERATION_RECORDS:
            raise InstrumentationUnavailable("CUDA event recorder reached its configured bound")
        if any(record.operation_id == operation_id for record in self._records):
            raise ValueError(f"duplicate CUDA operation ID {operation_id!r}")
        cuda = self._torch.cuda
        with cuda.device(self._logical_device):
            selected_stream = stream or cuda.current_stream(self._logical_device)
            start_event = cuda.Event(enable_timing=True)
            end_event = cuda.Event(enable_timing=True)
            cpu_start = self._clock_ns()
            start_event.record(selected_stream)
            launch_start = self._clock_ns()
            result = operation()
            launch_end = self._clock_ns()
            end_event.record(selected_stream)
            sync_start = self._clock_ns()
            end_event.synchronize()
            sync_end = self._clock_ns()
            elapsed_ns = round(float(start_event.elapsed_time(end_event)) * 1_000_000)
        record = CudaOperationRecord(
            operation_id=operation_id,
            kind=kind,
            logical_device=self._logical_device,
            stream_id=stream_id,
            cpu_start_monotonic_ns=cpu_start,
            cpu_end_monotonic_ns=sync_end,
            cpu_launch_ns=launch_end - launch_start,
            synchronization_wait_ns=sync_end - sync_start,
            cuda_event_elapsed_ns=elapsed_ns,
            bytes=bytes,
            byte_semantics=byte_semantics,
            copy_sizes_bytes=tuple(copy_sizes_bytes),
        )
        self._records.append(record)
        return result


class HostMemoryKind(StrEnum):
    PAGEABLE = "pageable"
    PINNED = "pinned"


class HostAllocationRecord(_StrictModel):
    schema_version: Literal["sloforge.branchfabric.host-allocation/v1"] = (
        "sloforge.branchfabric.host-allocation/v1"
    )
    allocation_id: str = Field(min_length=1, max_length=256)
    kind: HostMemoryKind
    purpose: str = Field(min_length=1, max_length=256)
    bytes: int = Field(gt=0)
    allocated_at_monotonic_ns: int = Field(ge=0)
    freed_at_monotonic_ns: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def valid_lifetime(self) -> Self:
        if (
            self.freed_at_monotonic_ns is not None
            and self.freed_at_monotonic_ns <= self.allocated_at_monotonic_ns
        ):
            raise ValueError("host allocation requires a positive measured lifetime")
        return self


class HostAllocationSummary(_StrictModel):
    schema_version: Literal["sloforge.branchfabric.host-allocation-summary/v1"] = (
        "sloforge.branchfabric.host-allocation-summary/v1"
    )
    records: tuple[HostAllocationRecord, ...] = Field(max_length=MAX_HOST_ALLOCATIONS)
    allocated_bytes_by_kind: dict[HostMemoryKind, int]
    peak_live_bytes_by_kind: dict[HostMemoryKind, int]
    active_bytes_by_kind: dict[HostMemoryKind, int]


class HostAllocationLedger:
    """Exact application-side ledger for pinned/pageable checkpoint buffers."""

    def __init__(self) -> None:
        self._records: dict[str, HostAllocationRecord] = {}

    def allocate(
        self,
        *,
        allocation_id: str,
        kind: HostMemoryKind,
        purpose: str,
        bytes: int,
        timestamp_ns: int,
    ) -> None:
        if len(self._records) >= MAX_HOST_ALLOCATIONS:
            raise InstrumentationUnavailable("host allocation ledger reached its bound")
        if allocation_id in self._records:
            raise ValueError(f"duplicate host allocation ID {allocation_id!r}")
        self._records[allocation_id] = HostAllocationRecord(
            allocation_id=allocation_id,
            kind=kind,
            purpose=purpose,
            bytes=bytes,
            allocated_at_monotonic_ns=timestamp_ns,
        )

    def free(self, allocation_id: str, *, timestamp_ns: int) -> None:
        try:
            record = self._records[allocation_id]
        except KeyError as exc:
            raise ValueError(f"unknown host allocation ID {allocation_id!r}") from exc
        if record.freed_at_monotonic_ns is not None:
            raise ValueError(f"host allocation {allocation_id!r} was already freed")
        self._records[allocation_id] = record.model_copy(
            update={"freed_at_monotonic_ns": timestamp_ns}
        )
        # model_copy(update=...) intentionally skips validation in Pydantic, so
        # validate the replaced immutable record explicitly.
        self._records[allocation_id] = HostAllocationRecord.model_validate(
            self._records[allocation_id].model_dump()
        )

    def summarize(self, *, require_all_freed: bool = False) -> HostAllocationSummary:
        records = tuple(self._records.values())
        if require_all_freed:
            active = [
                record.allocation_id for record in records if record.freed_at_monotonic_ns is None
            ]
            if active:
                raise InstrumentationUnavailable(
                    "host allocations remain live after cleanup: " + ", ".join(active)
                )
        allocated = {kind: 0 for kind in HostMemoryKind}
        active_bytes = {kind: 0 for kind in HostMemoryKind}
        peak = {kind: 0 for kind in HostMemoryKind}
        events: list[tuple[int, int, HostMemoryKind, int]] = []
        for record in records:
            allocated[record.kind] += record.bytes
            if record.freed_at_monotonic_ns is None:
                active_bytes[record.kind] += record.bytes
            events.append((record.allocated_at_monotonic_ns, 1, record.kind, record.bytes))
            if record.freed_at_monotonic_ns is not None:
                # Free before allocate at equal timestamps models half-open lifetimes.
                events.append((record.freed_at_monotonic_ns, 0, record.kind, -record.bytes))
        live = {kind: 0 for kind in HostMemoryKind}
        for _timestamp, _order, kind, delta in sorted(events):
            live[kind] += delta
            if live[kind] < 0:
                raise ValueError("host allocation ledger produced a negative live-byte count")
            peak[kind] = max(peak[kind], live[kind])
        return HostAllocationSummary(
            records=records,
            allocated_bytes_by_kind=allocated,
            peak_live_bytes_by_kind=peak,
            active_bytes_by_kind=active_bytes,
        )


class TraceCollectionLevel(StrEnum):
    DISABLED = "disabled"
    MINIMAL = "minimal"
    FULL = "full"


class TraceOverheadTrial(_StrictModel):
    schema_version: Literal["sloforge.branchfabric.trace-overhead-trial/v1"] = (
        "sloforge.branchfabric.trace-overhead-trial/v1"
    )
    trial_id: str = Field(min_length=1, max_length=256)
    seed: int = Field(ge=0, le=2**64 - 1)
    repetition: int = Field(ge=0)
    collection_level: TraceCollectionLevel
    workload_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    reclamation_interruption_ns: int = Field(gt=0)
    serving_throughput_tokens_per_second: float = Field(gt=0.0, allow_inf_nan=False)
    state_copy_count: int = Field(ge=0)
    state_copy_bytes: int = Field(ge=0)


class TraceOverheadAssessment(_StrictModel):
    schema_version: Literal["sloforge.branchfabric.trace-overhead-assessment/v1"] = (
        "sloforge.branchfabric.trace-overhead-assessment/v1"
    )
    reference_trial_id: str
    instrumented_trial_id: str
    collection_level: TraceCollectionLevel
    reclamation_latency_overhead_fraction: float = Field(allow_inf_nan=False)
    serving_throughput_degradation_fraction: float = Field(allow_inf_nan=False)
    copy_behavior_changed: bool
    materiality_threshold_fraction: float = Field(gt=0.0, lt=1.0, allow_inf_nan=False)
    materially_changes_trial: bool


def assess_trace_overhead(
    reference: TraceOverheadTrial,
    instrumented: TraceOverheadTrial,
    *,
    materiality_threshold_fraction: float = 0.05,
) -> TraceOverheadAssessment:
    """Compare paired raw trials; never infer percentiles from the pair."""

    if reference.collection_level is not TraceCollectionLevel.DISABLED:
        raise ValueError("trace-overhead reference must have tracing disabled")
    if instrumented.collection_level is TraceCollectionLevel.DISABLED:
        raise ValueError("instrumented trace-overhead trial must enable tracing")
    if not 0 < materiality_threshold_fraction < 1 or not math.isfinite(
        materiality_threshold_fraction
    ):
        raise ValueError("materiality threshold must be finite and in (0, 1)")
    if (
        reference.seed,
        reference.repetition,
        reference.workload_fingerprint,
    ) != (
        instrumented.seed,
        instrumented.repetition,
        instrumented.workload_fingerprint,
    ):
        raise ValueError("trace-overhead trials must be paired by seed/repetition/workload")
    latency_overhead = (
        instrumented.reclamation_interruption_ns / reference.reclamation_interruption_ns - 1.0
    )
    throughput_degradation = 1.0 - (
        instrumented.serving_throughput_tokens_per_second
        / reference.serving_throughput_tokens_per_second
    )
    behavior_changed = (
        instrumented.state_copy_count != reference.state_copy_count
        or instrumented.state_copy_bytes != reference.state_copy_bytes
    )
    material = behavior_changed or any(
        abs(value) >= materiality_threshold_fraction
        for value in (latency_overhead, throughput_degradation)
    )
    return TraceOverheadAssessment(
        reference_trial_id=reference.trial_id,
        instrumented_trial_id=instrumented.trial_id,
        collection_level=instrumented.collection_level,
        reclamation_latency_overhead_fraction=latency_overhead,
        serving_throughput_degradation_fraction=throughput_degradation,
        copy_behavior_changed=behavior_changed,
        materiality_threshold_fraction=materiality_threshold_fraction,
        materially_changes_trial=material,
    )


def process_cpu_fractions(
    before: HostResourceSample, after: HostResourceSample
) -> dict[int, float]:
    """Return CPU cores consumed per process between two cumulative samples."""

    elapsed_ns = after.query_start_monotonic_ns - before.query_start_monotonic_ns
    if elapsed_ns <= 0:
        raise ValueError("host samples require a positive elapsed interval")
    if before.logical_cpu_count != after.logical_cpu_count:
        raise ValueError("logical CPU count changed between host samples")
    old = {process.pid: process for process in before.processes}
    new = {process.pid: process for process in after.processes}
    if set(old) != set(new):
        raise ValueError("process set changed between host samples")
    fractions: dict[int, float] = {}
    for pid in sorted(old):
        cpu_delta = (
            new[pid].cpu_user_ns
            + new[pid].cpu_system_ns
            - old[pid].cpu_user_ns
            - old[pid].cpu_system_ns
        )
        if cpu_delta < 0:
            raise ValueError(f"process {pid} cumulative CPU time regressed")
        fractions[pid] = cpu_delta / elapsed_ns
    return fractions
