"""Bounded BranchFabric real-runtime GPU validation runner.

The runner contains no CUDA or vLLM implementation. It sequences a
``RealRuntimeStateAdapter``, enforces independent/shared scientific gates, and
writes canonical JSON/JSONL with raw provenance. Runtime imports and GPU
creation belong to the coordinator-provided adapter factory, so the runner and
its invariants execute in CPU-only unit tests.
"""

from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import subprocess
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any, Literal, Protocol, TypeVar, cast

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from sloforge.continuum.adapters.real_runtime import (
    ConcurrentDecodeResult,
    GpuMemoryState,
    KvStateClass,
    LogicalRuntimeState,
    PhysicalKvLayoutSnapshot,
    PhysicalKvReleaseEvidence,
    RealRuntimeCapability,
    RealRuntimeStateAdapter,
    RuntimeCapabilityMatrix,
    RuntimeModelIdentity,
)
from sloforge.continuum.adapters.sdk import SnapshotConsistencyError, UnsupportedCapabilityError
from sloforge.helix.characterization.trace import TraceLevel, write_jsonl

SCHEMA_VERSION = "sloforge.branchfabric.real-gpu-validation-run/v1"
RUNNER_VERSION = "1.0.0"
MAX_TIMEOUT_S = 4 * 60 * 60
MAX_GPU_SAMPLES = 1_000_000
_NonEmpty = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=512)]
_GpuSelector = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        pattern=(
            r"^(?:[0-9]{1,3}|GPU-[A-Za-z0-9-]{8,128}|"
            r"[0-9A-Fa-f]{4}:[0-9A-Fa-f]{2}:[0-9A-Fa-f]{2}\.[0-7])$"
        ),
    ),
]
_T = TypeVar("_T")


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, validate_default=True)


class ExperimentProfile(StrEnum):
    SMOKE = "smoke"
    CONTEXT_16K = "context_16k"


class TrialPath(StrEnum):
    INDEPENDENT_PREFILL = "independent_prefill"
    SHARED_ROOT = "shared_root"


class SnapshotPhase(StrEnum):
    ROOT = "root"
    FORK = "fork"
    DIVERGENCE = "divergence"
    SUFFIX_16 = "suffix_16"
    SUFFIX_64 = "suffix_64"
    SUFFIX_256 = "suffix_256"
    PRE_BRANCH_DELETE = "pre_branch_delete"
    POST_BRANCH_DELETE = "post_branch_delete"
    POST_ROOT_DELETE = "post_root_delete"


class GpuSamplingConfiguration(_StrictModel):
    interval_s: float = Field(default=0.25, ge=0.05, le=60.0)
    command_timeout_s: float = Field(default=2.0, ge=0.05, le=30.0)
    maximum_samples: int = Field(default=100_000, ge=1, le=MAX_GPU_SAMPLES)
    physical_device_selector: _GpuSelector | None = None


class ExperimentSpecification(_StrictModel):
    schema_version: str = SCHEMA_VERSION
    profile: ExperimentProfile
    seed: int = Field(ge=0, lt=1 << 64)
    timeout_s: float = Field(gt=0.0, le=MAX_TIMEOUT_S)
    cleanup_timeout_s: float = Field(default=60.0, gt=0.0, le=300.0)
    prefix_token_ids: tuple[int, ...]
    divergent_token_ids: tuple[int, ...]
    fanout: int = Field(ge=1, le=32)
    suffix_tokens: int = Field(ge=1, le=256)
    expected_identity: RuntimeModelIdentity
    trace_level: TraceLevel = TraceLevel.FULL
    gpu_sampling: GpuSamplingConfiguration = GpuSamplingConfiguration()

    @model_validator(mode="after")
    def bounded_matrix(self) -> ExperimentSpecification:
        if any(token < 0 for token in self.prefix_token_ids):
            raise ValueError("prefix token IDs must be non-negative")
        if len(self.divergent_token_ids) != self.fanout:
            raise ValueError("one divergent token ID is required per branch")
        if any(token < 0 for token in self.divergent_token_ids):
            raise ValueError("divergent token IDs must be non-negative")
        if len(set(self.divergent_token_ids)) != self.fanout:
            raise ValueError("branches must have distinct first divergent tokens")
        if self.profile is ExperimentProfile.SMOKE:
            if len(self.prefix_token_ids) not in {2048, 4096}:
                raise ValueError("smoke prefix must contain exactly 2K or 4K tokens")
            if self.fanout != 2 or self.suffix_tokens != 16:
                raise ValueError("smoke requires fanout 2 and a 16-token suffix")
        else:
            if len(self.prefix_token_ids) != 16_384:
                raise ValueError("16K validation requires exactly 16384 prefix tokens")
            if self.fanout not in {1, 8, 16, 32}:
                raise ValueError("16K validation fanout must be 1, 8, 16, or 32")
            if self.suffix_tokens != 256:
                raise ValueError("16K validation requires a 256-token suffix")
        return self


class VllmEngineConfiguration(_StrictModel):
    """Pinned, bounded engine knobs shared by both comparison paths."""

    execution_model_path: _NonEmpty
    download_dir: _NonEmpty
    max_model_len: int = Field(ge=4_096, le=131_072)
    block_size: Literal[16, 32] = 16
    max_num_seqs: int = Field(ge=2, le=64)
    gpu_memory_utilization: float = Field(gt=0.0, le=0.95)
    initialization_timeout_s: float = Field(default=900.0, gt=0.0, le=3_600.0)
    maximum_trace_events: int = Field(default=262_144, ge=1, le=1_000_000)
    enforce_eager: bool = False
    disable_log_stats: bool = True
    trust_remote_code: Literal[False] = False
    enable_chunked_prefill: bool = True


class VllmGpuValidationInvocation(_StrictModel):
    """Complete executable input, including physical-engine configuration."""

    schema_version: Literal["sloforge.branchfabric.vllm-gpu-validation-invocation/v1"] = (
        "sloforge.branchfabric.vllm-gpu-validation-invocation/v1"
    )
    experiment: ExperimentSpecification
    engine: VllmEngineConfiguration

    @model_validator(mode="after")
    def engine_covers_workload(self) -> VllmGpuValidationInvocation:
        required_context = len(self.experiment.prefix_token_ids) + self.experiment.suffix_tokens + 2
        if self.engine.max_model_len < required_context:
            raise ValueError(
                "max_model_len must cover prefix, divergence, measured suffix, and probe headroom"
            )
        if self.engine.max_num_seqs < self.experiment.fanout:
            raise ValueError("max_num_seqs must cover the requested fanout")
        if self.experiment.gpu_sampling.physical_device_selector is None:
            raise ValueError(
                "executable vLLM invocation requires an explicit physical GPU selector"
            )
        return self


class ArtifactRecord(_StrictModel):
    relative_path: _NonEmpty
    bytes: int = Field(ge=0)
    sha256: Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
    records: int | None = Field(default=None, ge=0)


class InstrumentationAvailability(_StrictModel):
    """Counters this runner cannot infer from nvidia-smi utilization samples."""

    copy_engine_activity_available: Literal[False] = False
    copy_engine_activity_unavailable_reason: Literal[
        "requires CUPTI, DCGM, or profiler copy-engine counters"
    ] = "requires CUPTI, DCGM, or profiler copy-engine counters"
    hbm_bandwidth_available: Literal[False] = False
    hbm_bandwidth_unavailable_reason: Literal[
        "requires CUPTI, DCGM, or profiler HBM bandwidth counters"
    ] = "requires CUPTI, DCGM, or profiler HBM bandwidth counters"
    nvidia_smi_memory_utilization_is_hbm_bandwidth: Literal[False] = False


class TraceCaptureMetrics(_StrictModel):
    """Extraction cost and bounded recorder work; neither is causal overhead."""

    extraction_wall_time_ns: int = Field(ge=0)
    extraction_cpu_time_ns: int = Field(ge=0)
    branch_events: int = Field(ge=0)
    state_events: int = Field(ge=0)
    live_self_measurement_available: bool
    live_self_measurement_unavailable_reason: _NonEmpty | None = None
    live_event_record_calls: int | None = Field(default=None, ge=0)
    live_layout_observation_calls: int | None = Field(default=None, ge=0)
    live_event_record_wall_time_ns: int | None = Field(default=None, ge=0)
    live_event_record_cpu_time_ns: int | None = Field(default=None, ge=0)
    live_layout_diff_wall_time_ns: int | None = Field(default=None, ge=0)
    live_layout_diff_cpu_time_ns: int | None = Field(default=None, ge=0)
    overhead_fraction_available: Literal[False] = False
    overhead_fraction_unavailable_reason: Literal[
        "requires paired disabled/full trials with identical seed and workload"
    ] = "requires paired disabled/full trials with identical seed and workload"

    @model_validator(mode="after")
    def valid_live_measurement(self) -> TraceCaptureMetrics:
        live_values = (
            self.live_event_record_calls,
            self.live_layout_observation_calls,
            self.live_event_record_wall_time_ns,
            self.live_event_record_cpu_time_ns,
            self.live_layout_diff_wall_time_ns,
            self.live_layout_diff_cpu_time_ns,
        )
        if self.live_self_measurement_available:
            if self.live_self_measurement_unavailable_reason is not None or any(
                value is None for value in live_values
            ):
                raise ValueError("available trace self-measurement requires every live counter")
        elif self.live_self_measurement_unavailable_reason is None or any(
            value is not None for value in live_values
        ):
            raise ValueError("unavailable trace self-measurement must provide only a reason")
        return self


class TrialSummary(_StrictModel):
    schema_version: str = SCHEMA_VERSION
    runner_version: str = RUNNER_VERSION
    run_id: _NonEmpty
    path: TrialPath
    profile: ExperimentProfile
    seed: int = Field(ge=0, lt=1 << 64)
    fanout: int = Field(ge=1, le=32)
    prefix_tokens: int = Field(gt=0)
    suffix_tokens: int = Field(gt=0)
    identity: RuntimeModelIdentity
    branchpoint_id: _NonEmpty
    request_admitted_latency_ns: dict[str, int]
    first_decode_token_started_latency_ns: dict[str, int]
    branch_ready_latency_ns: dict[str, int]
    first_token_latency_ns: dict[str, int]
    first_request_admitted_ns: int = Field(ge=0)
    all_requests_admitted_ns: int = Field(ge=0)
    first_decode_token_started_ns: int = Field(ge=0)
    all_branches_decode_started_ns: int = Field(ge=0)
    first_branch_ready_ns: int = Field(ge=0)
    all_branches_ready_ns: int = Field(ge=0)
    readiness_and_decode_elapsed_ns: int = Field(gt=0)
    readiness_and_decode_throughput_tokens_per_second: float = Field(ge=0.0)
    decode_elapsed_ns: int = Field(gt=0)
    decode_active_boundary_output_counts: dict[str, int]
    decode_tokens: int = Field(ge=0)
    decode_throughput_tokens_per_second: float = Field(ge=0.0)
    root_physical_bytes: int = Field(ge=0)
    final_physical_bytes: int = Field(ge=0)
    final_shared_prefix_bytes: int = Field(ge=0)
    final_private_suffix_bytes: int = Field(ge=0)
    trace_extraction_wall_time_ns: int = Field(ge=0)
    trace_extraction_cpu_time_ns: int = Field(ge=0)
    trace_live_self_measurement_available: bool
    trace_live_self_measurement_unavailable_reason: _NonEmpty | None = None
    trace_live_event_record_calls: int | None = Field(default=None, ge=0)
    trace_live_layout_observation_calls: int | None = Field(default=None, ge=0)
    trace_live_event_record_wall_time_ns: int | None = Field(default=None, ge=0)
    trace_live_event_record_cpu_time_ns: int | None = Field(default=None, ge=0)
    trace_live_layout_diff_wall_time_ns: int | None = Field(default=None, ge=0)
    trace_live_layout_diff_cpu_time_ns: int | None = Field(default=None, ge=0)
    tracing_overhead_fraction_available: Literal[False] = False
    tracing_overhead_fraction_unavailable_reason: Literal[
        "requires paired disabled/full trials with identical seed and workload"
    ] = "requires paired disabled/full trials with identical seed and workload"
    tracing_branch_events: int = Field(ge=0)
    tracing_state_events: int = Field(ge=0)
    instrumentation_availability: InstrumentationAvailability = Field(
        default_factory=InstrumentationAvailability
    )
    assertions: tuple[_NonEmpty, ...]
    phase_metrics: tuple[dict[str, Any], ...]
    artifacts: tuple[ArtifactRecord, ...] = ()

    @model_validator(mode="after")
    def valid_ready_range(self) -> TrialSummary:
        branch_ids = set(self.branch_ready_latency_ns)
        if (
            set(self.request_admitted_latency_ns) != branch_ids
            or set(self.first_decode_token_started_latency_ns) != branch_ids
            or set(self.first_token_latency_ns) != branch_ids
        ):
            raise ValueError("every readiness milestone must cover every branch")
        if any(
            not (
                self.request_admitted_latency_ns[branch_id]
                <= self.first_decode_token_started_latency_ns[branch_id]
                <= self.branch_ready_latency_ns[branch_id]
                <= self.first_token_latency_ns[branch_id]
            )
            for branch_id in branch_ids
        ):
            raise ValueError("readiness milestone timings are not causally ordered")
        admitted = tuple(self.request_admitted_latency_ns.values())
        decode_started = tuple(self.first_decode_token_started_latency_ns.values())
        if (
            not admitted
            or self.first_request_admitted_ns != min(admitted)
            or self.all_requests_admitted_ns != max(admitted)
        ):
            raise ValueError("request-admission aggregates do not match raw timings")
        if (
            not decode_started
            or self.first_decode_token_started_ns != min(decode_started)
            or self.all_branches_decode_started_ns != max(decode_started)
        ):
            raise ValueError("decode-start aggregates do not match raw timings")
        if set(self.decode_active_boundary_output_counts) != set(
            self.branch_ready_latency_ns
        ) or any(value < 1 for value in self.decode_active_boundary_output_counts.values()):
            raise ValueError(
                "active decode boundary counts must cover every ready branch and be positive"
            )
        if self.branch_ready_latency_ns:
            values = tuple(self.branch_ready_latency_ns.values())
            if any(value < 0 for value in values):
                raise ValueError("branch readiness latencies must be non-negative")
            if self.first_branch_ready_ns != min(values):
                raise ValueError("first-branch-ready does not match raw readiness")
            if self.all_branches_ready_ns != max(values):
                raise ValueError("all-branches-ready does not match raw readiness")
        if any(value < 0 for value in self.first_token_latency_ns.values()):
            raise ValueError("first-token latencies must be non-negative")
        if not all(
            math.isfinite(value)
            for value in (
                self.readiness_and_decode_throughput_tokens_per_second,
                self.decode_throughput_tokens_per_second,
            )
        ):
            raise ValueError("request and active decode throughput must be finite")
        return self


class ComparisonSummary(_StrictModel):
    schema_version: str = SCHEMA_VERSION
    runner_version: str = RUNNER_VERSION
    run_id: _NonEmpty
    profile: ExperimentProfile
    seed: int = Field(ge=0, lt=1 << 64)
    independent: TrialSummary
    shared_root: TrialSummary
    prefill_work_avoided_tokens: int = Field(ge=0)
    sharing_efficiency: float | None = None
    readiness_speedup: float | None = None
    decode_interference_fraction: float | None = None
    assertions: tuple[_NonEmpty, ...]
    artifacts: tuple[ArtifactRecord, ...] = ()


class IndependentDecodeAdapter(Protocol):
    """Baseline extension: independent sessions must not be represented as forks."""

    @property
    def capability_matrix(self) -> RuntimeCapabilityMatrix: ...

    def start_session(
        self, session_id: str, *, token_ids: tuple[int, ...], seed: int, timeout_s: float
    ) -> object: ...

    def inspect_logical_state(
        self, session_id: str, *, timeout_s: float
    ) -> LogicalRuntimeState: ...

    def run_concurrent_independent_sessions(
        self,
        session_ids: tuple[str, ...],
        *,
        maximum_new_tokens: int,
        seed: int,
        timeout_s: float,
    ) -> ConcurrentDecodeResult: ...

    def inspect_gpu_memory_state(self, *, timeout_s: float) -> GpuMemoryState: ...

    def destroy_session(self, session_id: str, *, timeout_s: float) -> None: ...

    def cleanup_runtime(self, *, timeout_s: float) -> None: ...

    def emit_branch_workload_trace(self) -> tuple[object, ...]: ...

    def emit_state_operation_trace(self) -> tuple[object, ...]: ...


@dataclass(frozen=True, slots=True)
class CommandResult:
    returncode: int
    stdout: str
    stderr: str


CommandRunner = Callable[[Sequence[str], float], CommandResult]
AdapterFactory = Callable[[TrialPath], RealRuntimeStateAdapter | IndependentDecodeAdapter]


def _default_command_runner(argv: Sequence[str], timeout_s: float) -> CommandResult:
    completed = subprocess.run(
        list(argv), check=False, capture_output=True, text=True, timeout=timeout_s
    )
    return CommandResult(completed.returncode, completed.stdout, completed.stderr)


class NvidiaSmiSampler:
    """Bounded, read-only nvidia-smi sampler retaining raw command provenance."""

    _FIELDS = (
        "timestamp",
        "index",
        "uuid",
        "memory.used",
        "memory.total",
        "utilization.gpu",
        "utilization.memory",
        "power.draw",
        "clocks.sm",
        "clocks.mem",
    )

    def __init__(
        self,
        *,
        device: str,
        config: GpuSamplingConfiguration,
        command_runner: CommandRunner = _default_command_runner,
        clock_ns: Callable[[], int] = time.monotonic_ns,
    ) -> None:
        self._device = device
        self._config = config
        self._run = command_runner
        self._clock_ns = clock_ns
        self._rows: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def rows(self) -> tuple[dict[str, Any], ...]:
        with self._lock:
            return tuple(dict(row) for row in self._rows)

    def _selector(self) -> str:
        if self._config.physical_device_selector is not None:
            return self._config.physical_device_selector
        if self._device.startswith("cuda:"):
            suffix = self._device.split(":", 1)[1]
            if suffix.isdigit():
                return suffix
        return self._device

    def _unavailable_row(
        self,
        sequence: int,
        observed_ns: int,
        reason: str,
        *,
        command: Sequence[str],
        returncode: int | None,
        stdout: str,
        stderr: str,
    ) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "sequence": sequence,
            "observed_at_monotonic_ns": observed_ns,
            "source": "nvidia-smi",
            "available": False,
            "unavailable_reason": reason,
            "command": list(command),
            "returncode": returncode,
            "raw_stdout": stdout,
            "raw_stderr": stderr,
            "device": self._device,
            "driver_timestamp": None,
            "gpu_index": None,
            "gpu_uuid": None,
            "memory_used_bytes": None,
            "memory_total_bytes": None,
            "gpu_utilization_percent": None,
            "memory_utilization_percent": None,
            "power_watts": None,
            "sm_clock_mhz": None,
            "memory_clock_mhz": None,
        }

    @staticmethod
    def _int_value(value: str, *, scale: int = 1) -> int | None:
        stripped = value.strip()
        if stripped in {"", "N/A", "[N/A]"}:
            return None
        try:
            return int(float(stripped) * scale)
        except ValueError:
            return None

    @staticmethod
    def _float_value(value: str) -> float | None:
        stripped = value.strip()
        if stripped in {"", "N/A", "[N/A]"}:
            return None
        try:
            parsed = float(stripped)
        except ValueError:
            return None
        return parsed if math.isfinite(parsed) else None

    def sample_once(self) -> dict[str, Any]:
        with self._lock:
            sequence = len(self._rows)
        observed_ns = self._clock_ns()
        command = (
            "nvidia-smi",
            f"--query-gpu={','.join(self._FIELDS)}",
            "--format=csv,noheader,nounits",
            "-i",
            self._selector(),
        )
        try:
            result = self._run(command, self._config.command_timeout_s)
        except FileNotFoundError as error:
            row = self._unavailable_row(
                sequence,
                observed_ns,
                "nvidia-smi executable unavailable",
                command=command,
                returncode=None,
                stdout="",
                stderr=str(error),
            )
        except OSError as error:
            row = self._unavailable_row(
                sequence,
                observed_ns,
                "nvidia-smi invocation failed",
                command=command,
                returncode=None,
                stdout="",
                stderr=str(error),
            )
        except subprocess.TimeoutExpired as error:
            row = self._unavailable_row(
                sequence,
                observed_ns,
                "nvidia-smi timed out",
                command=command,
                returncode=None,
                stdout=str(error.stdout or ""),
                stderr=str(error.stderr or ""),
            )
        else:
            lines = [line for line in result.stdout.splitlines() if line.strip()]
            values = [value.strip() for value in lines[0].split(",")] if lines else []
            if result.returncode != 0 or len(values) != len(self._FIELDS):
                reason = (
                    f"nvidia-smi exited {result.returncode}"
                    if result.returncode != 0
                    else "nvidia-smi returned an unexpected field count"
                )
                row = self._unavailable_row(
                    sequence,
                    observed_ns,
                    reason,
                    command=command,
                    returncode=result.returncode,
                    stdout=result.stdout,
                    stderr=result.stderr,
                )
            else:
                row = {
                    "schema_version": SCHEMA_VERSION,
                    "sequence": sequence,
                    "observed_at_monotonic_ns": observed_ns,
                    "source": "nvidia-smi",
                    "available": True,
                    "unavailable_reason": None,
                    "command": list(command),
                    "returncode": result.returncode,
                    "raw_stdout": result.stdout,
                    "raw_stderr": result.stderr,
                    "device": self._device,
                    "driver_timestamp": values[0],
                    "gpu_index": self._int_value(values[1]),
                    "gpu_uuid": values[2] or None,
                    "memory_used_bytes": self._int_value(values[3], scale=1024 * 1024),
                    "memory_total_bytes": self._int_value(values[4], scale=1024 * 1024),
                    "gpu_utilization_percent": self._float_value(values[5]),
                    "memory_utilization_percent": self._float_value(values[6]),
                    "power_watts": self._float_value(values[7]),
                    "sm_clock_mhz": self._int_value(values[8]),
                    "memory_clock_mhz": self._int_value(values[9]),
                }
        with self._lock:
            if len(self._rows) < self._config.maximum_samples:
                self._rows.append(row)
        return row

    def _loop(self) -> None:
        while not self._stop.is_set():
            self.sample_once()
            if len(self.rows) >= self._config.maximum_samples:
                return
            self._stop.wait(self._config.interval_s)

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("GPU sampler has already been started")
        self._thread = threading.Thread(
            target=self._loop, name="sloforge-nvidia-smi-sampler", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self._config.command_timeout_s + 1.0)
            if self._thread.is_alive():
                raise TimeoutError("nvidia-smi sampler did not stop within its bound")


class NvidiaSmiProcessMemoryReader:
    """Bounded NVML-backed phase reader for device and current-process HBM."""

    def __init__(
        self,
        *,
        physical_device_selector: _GpuSelector,
        command_timeout_s: float,
        command_runner: CommandRunner = _default_command_runner,
        process_id: int | None = None,
    ) -> None:
        if not math.isfinite(command_timeout_s) or not 0 < command_timeout_s <= 30.0:
            raise ValueError("command_timeout_s must be finite and in (0, 30]")
        self._selector = physical_device_selector
        self._timeout_s = command_timeout_s
        self._run = command_runner
        self._process_id = os.getpid() if process_id is None else process_id
        if self._process_id < 0:
            raise ValueError("process_id must be non-negative")

    @staticmethod
    def _mib(value: str) -> int | None:
        parsed = NvidiaSmiSampler._int_value(value, scale=1024 * 1024)
        return parsed if parsed is not None and parsed >= 0 else None

    def _query(self, fields: str) -> CommandResult | None:
        command = (
            "nvidia-smi",
            fields,
            "--format=csv,noheader,nounits",
            "-i",
            self._selector,
        )
        try:
            result = self._run(command, self._timeout_s)
        except (OSError, subprocess.TimeoutExpired):
            return None
        return result if result.returncode == 0 else None

    def __call__(self, device: str) -> Mapping[str, int | None]:
        del device  # Selection is deliberately physical, not CUDA-remapped.
        device_result = self._query("--query-gpu=memory.used")
        device_bytes = None
        if device_result is not None:
            lines = [line.strip() for line in device_result.stdout.splitlines() if line.strip()]
            if len(lines) == 1:
                device_bytes = self._mib(lines[0])

        process_result = self._query("--query-compute-apps=pid,used_gpu_memory")
        process_bytes = None
        if process_result is not None:
            nonempty_lines = [line for line in process_result.stdout.splitlines() if line.strip()]
            valid_records = 0
            current_process_unavailable = False
            current_process_found = False
            matched_process_bytes = 0
            for line in nonempty_lines:
                values = [value.strip() for value in line.split(",")]
                if len(values) != 2:
                    continue
                try:
                    pid = int(values[0])
                except ValueError:
                    continue
                used_bytes = self._mib(values[1])
                if used_bytes is None:
                    current_process_unavailable |= pid == self._process_id
                    continue
                valid_records += 1
                if pid == self._process_id:
                    current_process_found = True
                    matched_process_bytes += used_bytes
            if (
                not current_process_unavailable
                and current_process_found
                and (not nonempty_lines or valid_records > 0)
            ):
                process_bytes = matched_process_bytes
        return {
            "nvml_process_bytes": process_bytes,
            "nvml_device_used_bytes": device_bytes,
        }


def build_vllm_adapter_factory(
    invocation: VllmGpuValidationInvocation,
    *,
    command_runner: CommandRunner = _default_command_runner,
    metadata_instrumentation_level: str = "disabled",
    metadata_optimization: str = "baseline",
) -> AdapterFactory:
    """Create path-scoped vLLM factories without importing vLLM eagerly."""

    selector = invocation.experiment.gpu_sampling.physical_device_selector
    if selector is None:  # Also enforced by the invocation validator; keeps typing fail closed.
        raise ValueError("a physical GPU selector is required")
    memory_reader = NvidiaSmiProcessMemoryReader(
        physical_device_selector=selector,
        command_timeout_s=invocation.experiment.gpu_sampling.command_timeout_s,
        command_runner=command_runner,
    )

    def create(path: TrialPath) -> RealRuntimeStateAdapter | IndependentDecodeAdapter:
        from sloforge.continuum.adapters.vllm_live import VllmLiveStateAdapter
        from sloforge.continuum.adapters.vllm_metadata_0230 import (
            MetadataInstrumentationLevel,
            MetadataOptimization,
        )

        experiment = invocation.experiment
        engine = invocation.engine
        local_snapshot = str(Path(engine.execution_model_path).resolve(strict=True))
        llm_kwargs: dict[str, object] = {
            "model": local_snapshot,
            "revision": experiment.expected_identity.model_revision,
            "tokenizer": local_snapshot,
            "tokenizer_revision": experiment.expected_identity.tokenizer_revision,
            "dtype": experiment.expected_identity.dtype,
            "max_model_len": engine.max_model_len,
            "enable_prefix_caching": path is TrialPath.SHARED_ROOT,
            "block_size": engine.block_size,
            "max_num_seqs": engine.max_num_seqs,
            "seed": experiment.seed,
            "tensor_parallel_size": 1,
            "async_scheduling": False,
            "gpu_memory_utilization": engine.gpu_memory_utilization,
            "download_dir": engine.download_dir,
            "enforce_eager": engine.enforce_eager,
            "disable_log_stats": engine.disable_log_stats,
            "trust_remote_code": engine.trust_remote_code,
            "enable_chunked_prefill": engine.enable_chunked_prefill,
            "cpu_offload_gb": 0.0,
        }
        return VllmLiveStateAdapter.create(
            identity=experiment.expected_identity,
            execution_model_path=local_snapshot,
            seed=experiment.seed,
            timeout_s=engine.initialization_timeout_s,
            llm_kwargs=llm_kwargs,
            max_fanout=experiment.fanout,
            default_maximum_new_tokens=experiment.suffix_tokens,
            gpu_memory_reader=memory_reader,
            trace_level=experiment.trace_level,
            trace_id=f"{_run_id(experiment)}:{path.value}",
            max_trace_events=engine.maximum_trace_events,
            metadata_instrumentation_level=MetadataInstrumentationLevel(
                metadata_instrumentation_level
            ),
            metadata_optimization=MetadataOptimization(metadata_optimization),
        )

    return create


def _canonical_bytes(value: Any) -> bytes:
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n"
    ).encode("utf-8")


def _write_new(path: Path, payload: bytes) -> ArtifactRecord:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(payload)
    return ArtifactRecord(
        relative_path=str(path), bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest()
    )


def _write_jsonl_new(path: Path, rows: Sequence[Any]) -> ArtifactRecord:
    payload = b"".join(_canonical_bytes(row) for row in rows)
    artifact = _write_new(path, payload)
    return artifact.model_copy(update={"records": len(rows)})


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _artifact_for_existing(path: Path, records: int | None = None) -> ArtifactRecord:
    return ArtifactRecord(
        relative_path=str(path),
        bytes=path.stat().st_size,
        sha256=_hash_file(path),
        records=records,
    )


def _run_id(spec: ExperimentSpecification) -> str:
    return "gpu-validation:" + hashlib.sha256(_canonical_bytes(spec)).hexdigest()[:24]


def _remaining(deadline_ns: int) -> float:
    remaining = (deadline_ns - time.monotonic_ns()) / 1_000_000_000
    if remaining <= 0:
        raise TimeoutError("GPU validation runner exceeded the experiment timeout")
    return remaining


def _assert_identity(
    matrix: RuntimeCapabilityMatrix, expected: RuntimeModelIdentity, path: TrialPath
) -> tuple[str, ...]:
    if matrix.identity != expected:
        raise SnapshotConsistencyError(
            "runtime/model/tokenizer/dtype/device/policy identity does not match the experiment",
            operation=f"validate_{path.value}_identity",
        )
    if not matrix.scope.same_runtime_only or not matrix.scope.same_policy_only:
        raise SnapshotConsistencyError(
            "live adapter did not declare same-runtime and same-policy scope",
            operation=f"validate_{path.value}_scope",
        )
    if matrix.scope.cross_runtime_portable:
        raise SnapshotConsistencyError(
            "live experiment cannot use a cross-runtime-portable scope claim",
            operation=f"validate_{path.value}_scope",
        )
    required = {
        RealRuntimeCapability.START_SESSION,
        RealRuntimeCapability.INSPECT_LOGICAL_STATE,
        RealRuntimeCapability.INSPECT_PHYSICAL_KV_LAYOUT,
        RealRuntimeCapability.INSPECT_GPU_MEMORY_STATE,
        RealRuntimeCapability.DESTROY_SESSION,
        RealRuntimeCapability.CLEANUP_RUNTIME,
    }
    if path is TrialPath.SHARED_ROOT:
        required.update(
            {
                RealRuntimeCapability.PREFILL_SESSION,
                RealRuntimeCapability.PAUSE_AT_SAFE_DECODE_BOUNDARY,
                RealRuntimeCapability.CREATE_SHARED_ROOT_REFERENCE,
                RealRuntimeCapability.FORK_SAME_POLICY_SESSION,
                RealRuntimeCapability.RUN_CONCURRENT_BRANCHES,
                RealRuntimeCapability.DESTROY_BRANCH,
            }
        )
    else:
        required.add(RealRuntimeCapability.RUN_CONCURRENT_INDEPENDENT_SESSIONS)
    missing = required - matrix.capabilities
    if missing:
        raise UnsupportedCapabilityError(
            "live adapter lacks required experiment capabilities: "
            + ", ".join(sorted(capability.value for capability in missing)),
            operation=f"validate_{path.value}_capabilities",
        )
    return (
        "runtime_model_tokenizer_dtype_device_policy_identity_equal",
        "same_runtime_same_policy_process_local_scope",
        "required_adapter_capabilities_present",
    )


def _assert_trace_capabilities(
    matrix: RuntimeCapabilityMatrix, level: TraceLevel
) -> tuple[str, ...]:
    if level is TraceLevel.DISABLED:
        return ("runtime_trace_extraction_disabled_for_overhead_control",)
    required = {
        RealRuntimeCapability.EMIT_BRANCH_WORKLOAD_TRACE,
        RealRuntimeCapability.EMIT_STATE_OPERATION_TRACE,
    }
    missing = required - matrix.capabilities
    if missing:
        raise UnsupportedCapabilityError(
            "requested trace condition lacks adapter capabilities: "
            + ", ".join(sorted(item.value for item in missing)),
            operation="validate_trace_capabilities",
        )
    return (f"runtime_trace_level_{level.value}_explicitly_selected",)


def _assert_logical_prefix(
    state: LogicalRuntimeState,
    spec: ExperimentSpecification,
    *,
    divergent_token: int | None,
) -> None:
    expected = spec.prefix_token_ids
    if divergent_token is not None:
        expected += (divergent_token,)
    if state.model != spec.expected_identity or state.token_ids[: len(expected)] != expected:
        raise SnapshotConsistencyError(
            "logical branch state changed model/policy or token prefix",
            operation="validate_logical_prefix",
            session_id=state.session_id,
        )


def _measure_phase(
    phase: str,
    operation: Callable[[], _T],
    phase_metrics: list[dict[str, Any]],
) -> _T:
    wall_start = time.monotonic_ns()
    cpu_start = time.process_time_ns()
    result = operation()
    cpu_end = time.process_time_ns()
    wall_end = time.monotonic_ns()
    phase_metrics.append(
        {
            "schema_version": SCHEMA_VERSION,
            "phase": phase,
            "wall_start_monotonic_ns": wall_start,
            "wall_end_monotonic_ns": wall_end,
            "wall_duration_ns": wall_end - wall_start,
            "process_cpu_start_ns": cpu_start,
            "process_cpu_end_ns": cpu_end,
            "process_cpu_duration_ns": cpu_end - cpu_start,
        }
    )
    return result


def _phase_gpu_sample(
    adapter: RealRuntimeStateAdapter | IndependentDecodeAdapter,
    phase: str,
    deadline_ns: int,
    rows: list[dict[str, Any]],
) -> GpuMemoryState:
    sample = adapter.inspect_gpu_memory_state(timeout_s=_remaining(deadline_ns))
    rows.append(
        {
            "schema_version": SCHEMA_VERSION,
            "phase": phase,
            "source": "RealRuntimeStateAdapter.inspect_gpu_memory_state",
            "state": sample.model_dump(mode="json"),
        }
    )
    return sample


def _decode_phases(spec: ExperimentSpecification) -> tuple[SnapshotPhase, ...]:
    phases = [SnapshotPhase.DIVERGENCE]
    if spec.suffix_tokens >= 16:
        phases.append(SnapshotPhase.SUFFIX_16)
    if spec.suffix_tokens >= 64:
        phases.append(SnapshotPhase.SUFFIX_64)
    if spec.suffix_tokens >= 256:
        phases.append(SnapshotPhase.SUFFIX_256)
    return tuple(phases)


def _label_decode_snapshots(
    spec: ExperimentSpecification, result: ConcurrentDecodeResult
) -> tuple[tuple[SnapshotPhase, PhysicalKvLayoutSnapshot], ...]:
    phases = _decode_phases(spec)
    if len(result.allocation_snapshots) != len(phases):
        raise SnapshotConsistencyError(
            f"decode returned {len(result.allocation_snapshots)} snapshots; "
            f"required phases are {[phase.value for phase in phases]}",
            operation="validate_decode_snapshots",
        )
    return tuple(zip(phases, result.allocation_snapshots, strict=True))


def _snapshot_row(phase: SnapshotPhase, snapshot: PhysicalKvLayoutSnapshot) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "phase": phase.value,
        "snapshot": snapshot.model_dump(mode="json"),
    }


def _validate_shared_snapshots(
    root_block_ids: tuple[str, ...],
    root_bytes: int,
    fanout: int,
    labeled: Sequence[tuple[SnapshotPhase, PhysicalKvLayoutSnapshot]],
) -> tuple[str, ...]:
    expected = set(root_block_ids)
    if not expected:
        raise SnapshotConsistencyError("root contains no physical blocks", operation="shared_root")
    for phase, snapshot in labeled:
        if set(snapshot.shared_prefix_block_ids) != expected:
            raise SnapshotConsistencyError(
                f"shared-root IDs changed at {phase.value}", operation="shared_root"
            )
        if snapshot.shared_prefix_bytes != root_bytes:
            raise SnapshotConsistencyError(
                f"physical root bytes were multiplied at {phase.value}", operation="shared_root"
            )
        blocks = {block.runtime_block_id: block for block in snapshot.blocks}
        for block_id in expected:
            block = blocks[block_id]
            if len(block.branch_ids) != fanout or block.refcount < fanout:
                raise SnapshotConsistencyError(
                    f"root block lacks all branch owners at {phase.value}",
                    operation="shared_root",
                )
        if phase is SnapshotPhase.DIVERGENCE and not snapshot.private_suffix_block_ids:
            raise SnapshotConsistencyError(
                "no private GPU suffix allocation appeared at first divergence",
                operation="shared_root",
            )
    return (
        "one_physical_shared_root_not_multiplied_by_fanout",
        "all_branches_reference_exact_root_block_ids",
        "private_suffix_allocation_observed_after_divergence",
        "runtime_native_refcounts_cover_live_branches",
    )


def _validate_zero_copy_fork(
    root_block_ids: tuple[str, ...],
    root_bytes: int,
    branch_ids: tuple[str, ...],
    root_snapshot: PhysicalKvLayoutSnapshot,
    fork_snapshot: PhysicalKvLayoutSnapshot,
) -> tuple[str, ...]:
    """Prove that fork created logical branches without allocating KV state.

    vLLM materializes native request ownership only when the forked requests enter
    the scheduler.  The pre-decode snapshot therefore proves the zero-copy fork
    through exact physical block identity and allocator accounting; the decode
    snapshots validate native branch ownership and refcounts separately.
    """

    expected_ids = set(root_block_ids)
    root_blocks = {block.runtime_block_id: block for block in root_snapshot.blocks}
    fork_blocks = {block.runtime_block_id: block for block in fork_snapshot.blocks}
    if (
        set(root_snapshot.shared_prefix_block_ids) != expected_ids
        or set(root_blocks) != expected_ids
        or root_snapshot.shared_prefix_bytes != root_bytes
        or root_snapshot.physical_assigned_bytes != root_bytes
    ):
        raise SnapshotConsistencyError(
            "published root snapshot is not the exact referenced physical root",
            operation="validate_zero_copy_fork",
        )
    if set(fork_snapshot.shared_prefix_block_ids) != expected_ids:
        raise SnapshotConsistencyError(
            "fork does not reference the exact published root block IDs",
            operation="validate_zero_copy_fork",
        )
    if (
        any(fork_blocks[block_id].bytes != root_blocks[block_id].bytes for block_id in expected_ids)
        or fork_snapshot.shared_prefix_bytes != root_bytes
    ):
        raise SnapshotConsistencyError(
            "fork changed or duplicated published root bytes",
            operation="validate_zero_copy_fork",
        )
    if fork_snapshot.private_suffix_block_ids or fork_snapshot.private_suffix_bytes != 0:
        raise SnapshotConsistencyError(
            "fork allocated private KV state before divergence",
            operation="validate_zero_copy_fork",
        )
    if fork_snapshot.physical_assigned_bytes != root_snapshot.physical_assigned_bytes:
        raise SnapshotConsistencyError(
            "fork changed runtime-assigned KV bytes before divergence",
            operation="validate_zero_copy_fork",
        )
    if set(fork_blocks) != expected_ids:
        raise SnapshotConsistencyError(
            "fork exposed physical KV blocks beyond the exact published root",
            operation="validate_zero_copy_fork",
        )
    return (
        "fork_references_exact_published_root_ids_and_bytes",
        "fork_allocates_zero_private_or_additional_assigned_kv_bytes",
        "fork_logical_branches_validated_before_native_scheduler_ownership",
    )


def _validate_independent_snapshots(
    labeled: Sequence[tuple[SnapshotPhase, PhysicalKvLayoutSnapshot]],
) -> tuple[str, ...]:
    for phase, snapshot in labeled:
        if snapshot.shared_prefix_block_ids or snapshot.shared_prefix_bytes:
            raise SnapshotConsistencyError(
                f"independent baseline reported a shared prefix at {phase.value}",
                operation="independent_cache_isolation",
            )
        for block in snapshot.blocks:
            if len(block.branch_ids) > 1:
                raise SnapshotConsistencyError(
                    f"independent branches alias block {block.runtime_block_id} at {phase.value}",
                    operation="independent_cache_isolation",
                )
            if block.state_class is KvStateClass.SHARED_PREFIX:
                raise SnapshotConsistencyError(
                    "independent baseline classified a block as shared prefix",
                    operation="independent_cache_isolation",
                )
    return (
        "independent_prefix_cache_hits_absent",
        "independent_branch_physical_block_sets_disjoint",
        "independent_prefix_physical_bytes_scale_with_fanout",
    )


def _capture_traces(
    adapter: RealRuntimeStateAdapter | IndependentDecodeAdapter,
    level: TraceLevel,
    output: Path,
) -> tuple[TraceCaptureMetrics, tuple[ArtifactRecord, ...]]:
    if level is TraceLevel.DISABLED:
        return (
            TraceCaptureMetrics(
                extraction_wall_time_ns=0,
                extraction_cpu_time_ns=0,
                branch_events=0,
                state_events=0,
                live_self_measurement_available=False,
                live_self_measurement_unavailable_reason="tracing disabled",
            ),
            (),
        )
    overhead_inspector = getattr(adapter, "inspect_trace_self_overhead", None)
    live: dict[str, int | str] | None = None
    if callable(overhead_inspector):
        candidate = overhead_inspector()
        if not isinstance(candidate, dict) or any(
            type(value) not in {int, str} for value in candidate.values()
        ):
            raise SnapshotConsistencyError(
                "adapter trace self-overhead counters are not pointer-free primitives",
                operation="capture_traces",
            )
        live = cast(dict[str, int | str], candidate)
    wall_start = time.monotonic_ns()
    cpu_start = time.process_time_ns()
    branch_events = tuple(adapter.emit_branch_workload_trace())
    state_events = tuple(adapter.emit_state_operation_trace())
    for event in branch_events + state_events:
        if getattr(event, "collection_level", None) is not level:
            raise SnapshotConsistencyError(
                "adapter trace level does not match the requested overhead condition",
                operation="capture_traces",
            )
    branch_path = output / "traces" / "branch-workload-trace-v1.jsonl"
    state_path = output / "traces" / "state-operation-trace-v1.jsonl"
    branch_path.parent.mkdir(parents=True, exist_ok=True)
    branch_artifact = write_jsonl(branch_path, cast(Sequence[Any], branch_events))
    state_artifact = write_jsonl(state_path, cast(Sequence[Any], state_events))
    cpu_elapsed = time.process_time_ns() - cpu_start
    wall_elapsed = time.monotonic_ns() - wall_start
    artifacts = (
        _artifact_for_existing(branch_path, branch_artifact.event_count),
        _artifact_for_existing(state_path, state_artifact.event_count),
    )
    if live is None:
        metrics = TraceCaptureMetrics(
            extraction_wall_time_ns=wall_elapsed,
            extraction_cpu_time_ns=cpu_elapsed,
            branch_events=len(branch_events),
            state_events=len(state_events),
            live_self_measurement_available=False,
            live_self_measurement_unavailable_reason=(
                "adapter did not expose bounded live trace self-measurement"
            ),
        )
    else:
        expected_level = live.get("trace_level")
        if expected_level != level.value:
            raise SnapshotConsistencyError(
                "adapter trace self-overhead level does not match requested trace level",
                operation="capture_traces",
            )

        def live_counter(key: str) -> int:
            value = live.get(key)
            if type(value) is not int or value < 0:
                raise SnapshotConsistencyError(
                    f"adapter trace self-overhead counter {key!r} is invalid",
                    operation="capture_traces",
                )
            return value

        event_record_calls = live_counter("event_record_calls")
        if event_record_calls != len(branch_events) + len(state_events):
            raise SnapshotConsistencyError(
                "adapter live trace self-measurement event count disagrees with extraction",
                operation="capture_traces",
            )
        metrics = TraceCaptureMetrics(
            extraction_wall_time_ns=wall_elapsed,
            extraction_cpu_time_ns=cpu_elapsed,
            branch_events=len(branch_events),
            state_events=len(state_events),
            live_self_measurement_available=True,
            live_event_record_calls=event_record_calls,
            live_layout_observation_calls=live_counter("layout_observation_calls"),
            live_event_record_wall_time_ns=live_counter("live_event_record_wall_time_ns"),
            live_event_record_cpu_time_ns=live_counter("live_event_record_cpu_time_ns"),
            live_layout_diff_wall_time_ns=live_counter("live_layout_diff_wall_time_ns"),
            live_layout_diff_cpu_time_ns=live_counter("live_layout_diff_cpu_time_ns"),
        )
    return metrics, artifacts


def _trial_session_ids(run_id: str, path: TrialPath, fanout: int) -> tuple[str, ...]:
    prefix = run_id.replace(":", "-")
    return tuple(f"{prefix}-{path.value}-branch-{index:02d}" for index in range(fanout))


def _build_trial_summary(
    *,
    run_id: str,
    path: TrialPath,
    spec: ExperimentSpecification,
    result: ConcurrentDecodeResult,
    root_bytes: int,
    final: PhysicalKvLayoutSnapshot,
    trace_metrics: TraceCaptureMetrics,
    assertions: Sequence[str],
    phase_metrics: Sequence[dict[str, Any]],
    readiness_offset_ns: int = 0,
) -> TrialSummary:
    readiness = dict(
        sorted(
            (branch_id, latency + readiness_offset_ns)
            for branch_id, latency in result.branch_ready_latency_ns.items()
        )
    )
    admitted = dict(
        sorted(
            (branch_id, latency + readiness_offset_ns)
            for branch_id, latency in result.request_admitted_latency_ns.items()
        )
    )
    decode_started = dict(
        sorted(
            (branch_id, latency + readiness_offset_ns)
            for branch_id, latency in result.first_decode_token_started_latency_ns.items()
        )
    )
    first_tokens = dict(
        sorted(
            (branch_id, latency + readiness_offset_ns)
            for branch_id, latency in result.first_token_latency_ns.items()
        )
    )
    active_throughput = result.decode_active_tokens / (
        result.decode_active_elapsed_ns / 1_000_000_000
    )
    return TrialSummary(
        run_id=f"{run_id}:{path.value}",
        path=path,
        profile=spec.profile,
        seed=spec.seed,
        fanout=spec.fanout,
        prefix_tokens=len(spec.prefix_token_ids),
        suffix_tokens=spec.suffix_tokens,
        identity=spec.expected_identity,
        branchpoint_id=result.branchpoint_id,
        request_admitted_latency_ns=admitted,
        first_decode_token_started_latency_ns=decode_started,
        branch_ready_latency_ns=readiness,
        first_token_latency_ns=first_tokens,
        first_request_admitted_ns=min(admitted.values()),
        all_requests_admitted_ns=max(admitted.values()),
        first_decode_token_started_ns=min(decode_started.values()),
        all_branches_decode_started_ns=max(decode_started.values()),
        first_branch_ready_ns=min(readiness.values()),
        all_branches_ready_ns=max(readiness.values()),
        readiness_and_decode_elapsed_ns=result.elapsed_ns + readiness_offset_ns,
        readiness_and_decode_throughput_tokens_per_second=(
            result.total_output_tokens / ((result.elapsed_ns + readiness_offset_ns) / 1_000_000_000)
        ),
        decode_elapsed_ns=result.decode_active_elapsed_ns,
        decode_active_boundary_output_counts=result.decode_active_boundary_output_counts,
        decode_tokens=result.decode_active_tokens,
        decode_throughput_tokens_per_second=active_throughput,
        root_physical_bytes=root_bytes,
        final_physical_bytes=final.physical_assigned_bytes,
        final_shared_prefix_bytes=final.shared_prefix_bytes,
        final_private_suffix_bytes=final.private_suffix_bytes,
        trace_extraction_wall_time_ns=trace_metrics.extraction_wall_time_ns,
        trace_extraction_cpu_time_ns=trace_metrics.extraction_cpu_time_ns,
        trace_live_self_measurement_available=trace_metrics.live_self_measurement_available,
        trace_live_self_measurement_unavailable_reason=(
            trace_metrics.live_self_measurement_unavailable_reason
        ),
        trace_live_event_record_calls=trace_metrics.live_event_record_calls,
        trace_live_layout_observation_calls=trace_metrics.live_layout_observation_calls,
        trace_live_event_record_wall_time_ns=trace_metrics.live_event_record_wall_time_ns,
        trace_live_event_record_cpu_time_ns=trace_metrics.live_event_record_cpu_time_ns,
        trace_live_layout_diff_wall_time_ns=trace_metrics.live_layout_diff_wall_time_ns,
        trace_live_layout_diff_cpu_time_ns=trace_metrics.live_layout_diff_cpu_time_ns,
        tracing_branch_events=trace_metrics.branch_events,
        tracing_state_events=trace_metrics.state_events,
        assertions=(
            *assertions,
            "request_admission_branch_readiness_decode_start_and_token_emission_are_distinct",
            "decode_throughput_excludes_prefill_and_branch_readiness_window",
            "decode_active_tokens_use_exact_common_boundary_counts",
            "trace_extraction_cost_is_not_labeled_as_runtime_tracing_overhead",
            "tracing_overhead_fraction_requires_paired_disabled_full_trials",
            "copy_engine_and_hbm_bandwidth_unavailable_without_profiler_counters",
        ),
        phase_metrics=tuple(phase_metrics),
    )


def _persist_trial(
    output: Path,
    spec: ExperimentSpecification,
    summary: TrialSummary,
    physical_rows: Sequence[dict[str, Any]],
    gpu_state_rows: Sequence[dict[str, Any]],
    gpu_sampler_rows: Sequence[dict[str, Any]],
    release_evidence_rows: Sequence[dict[str, Any]],
    trace_artifacts: Sequence[ArtifactRecord],
) -> TrialSummary:
    artifacts: list[ArtifactRecord] = list(trace_artifacts)
    artifacts.append(_write_new(output / "configuration.json", _canonical_bytes(spec)))
    artifacts.append(_write_jsonl_new(output / "raw" / "physical-snapshots.jsonl", physical_rows))
    artifacts.append(_write_jsonl_new(output / "raw" / "gpu-memory-states.jsonl", gpu_state_rows))
    artifacts.append(
        _write_jsonl_new(output / "raw" / "nvidia-smi-samples.jsonl", gpu_sampler_rows)
    )
    artifacts.append(
        _write_jsonl_new(output / "raw" / "block-release-evidence.jsonl", release_evidence_rows)
    )
    artifacts.append(
        _write_jsonl_new(output / "raw" / "phase-timings.jsonl", summary.phase_metrics)
    )
    relative_artifacts = tuple(
        sorted(
            (
                artifact.model_copy(
                    update={"relative_path": str(Path(artifact.relative_path).relative_to(output))}
                )
                for artifact in artifacts
            ),
            key=lambda item: item.relative_path,
        )
    )
    finalized = summary.model_copy(update={"artifacts": relative_artifacts})
    summary_record = _write_new(
        output / "metrics" / "trial-summary.json", _canonical_bytes(finalized)
    ).model_copy(update={"relative_path": "metrics/trial-summary.json"})
    _write_new(
        output / "artifact-manifest.json",
        _canonical_bytes(
            {
                "schema_version": SCHEMA_VERSION,
                "run_id": finalized.run_id,
                "artifacts": [
                    item.model_dump(mode="json") for item in (*relative_artifacts, summary_record)
                ],
            }
        ),
    )
    return finalized


def _release_evidence_row(phase: str, evidence: PhysicalKvReleaseEvidence) -> dict[str, Any]:
    return {"phase": phase, **evidence.model_dump(mode="json")}


def _assert_exact_native_release(
    evidence: PhysicalKvReleaseEvidence,
    expected_epochs: Mapping[str, int],
    *,
    operation: str,
    require_hash_cleared: bool,
) -> None:
    if set(evidence.requested_block_ids) != set(expected_epochs):
        raise SnapshotConsistencyError(
            "native release evidence did not cover the exact former block IDs",
            operation=operation,
        )
    observed = {block.runtime_block_id: block for block in evidence.blocks}
    for block_id, allocation_epoch in expected_epochs.items():
        block = observed[block_id]
        if block.allocation_epoch != allocation_epoch:
            raise SnapshotConsistencyError(
                f"block {block_id} was reused before release could be proven",
                operation=operation,
            )
        if block.native_refcount != 0 or not block.allocator_available:
            raise SnapshotConsistencyError(
                f"block {block_id} remains owned after release",
                operation=operation,
            )
        if require_hash_cleared and block.block_hash_present:
            raise SnapshotConsistencyError(
                f"block {block_id} retained a cache hash after final reset",
                operation=operation,
            )


def run_shared_trial(
    adapter: RealRuntimeStateAdapter,
    spec: ExperimentSpecification,
    output: Path,
    *,
    command_runner: CommandRunner = _default_command_runner,
) -> TrialSummary:
    """Run one physically shared-root trial and always clean its owned runtime."""

    run_id = _run_id(spec)
    deadline_ns = time.monotonic_ns() + int(spec.timeout_s * 1_000_000_000)
    assertions: list[str] = []
    root_session_id = run_id.replace(":", "-") + "-root"
    branch_ids = _trial_session_ids(run_id, TrialPath.SHARED_ROOT, spec.fanout)
    phase_metrics: list[dict[str, Any]] = []
    physical_rows: list[dict[str, Any]] = []
    gpu_rows: list[dict[str, Any]] = []
    release_evidence_rows: list[dict[str, Any]] = []
    sampler = NvidiaSmiSampler(
        device=spec.expected_identity.device,
        config=spec.gpu_sampling,
        command_runner=command_runner,
    )
    trace_artifacts: tuple[ArtifactRecord, ...] = ()
    sampler_started = False
    try:
        sampler.start()
        sampler_started = True
        assertions.extend(
            _assert_identity(
                adapter.capability_matrix,
                spec.expected_identity,
                TrialPath.SHARED_ROOT,
            )
        )
        assertions.extend(_assert_trace_capabilities(adapter.capability_matrix, spec.trace_level))
        _phase_gpu_sample(adapter, "model_only", deadline_ns, gpu_rows)
        _measure_phase(
            "root_session_create",
            lambda: adapter.start_session(
                root_session_id,
                token_ids=spec.prefix_token_ids,
                seed=spec.seed,
                timeout_s=_remaining(deadline_ns),
            ),
            phase_metrics,
        )
        logical_root = adapter.inspect_logical_state(
            root_session_id, timeout_s=_remaining(deadline_ns)
        )
        _assert_logical_prefix(logical_root, spec, divergent_token=None)
        _measure_phase(
            "root_prefill",
            lambda: adapter.prefill_session(root_session_id, timeout_s=_remaining(deadline_ns)),
            phase_metrics,
        )
        adapter.pause_at_safe_decode_boundary(root_session_id, timeout_s=_remaining(deadline_ns))
        root = _measure_phase(
            "root_publish",
            lambda: adapter.create_shared_root_reference(
                root_session_id,
                prefix_token_count=len(spec.prefix_token_ids),
                timeout_s=_remaining(deadline_ns),
            ),
            phase_metrics,
        )
        root_snapshot = adapter.inspect_physical_kv_layout(
            (root_session_id,), timeout_s=_remaining(deadline_ns)
        )
        physical_rows.append(_snapshot_row(SnapshotPhase.ROOT, root_snapshot))
        if set(root_snapshot.shared_prefix_block_ids) != set(root.block_ids):
            raise SnapshotConsistencyError(
                "published root does not match its physical snapshot", operation="root_publish"
            )
        _phase_gpu_sample(adapter, SnapshotPhase.ROOT.value, deadline_ns, gpu_rows)

        fork_started_ns = time.monotonic_ns()
        fork_cpu_started_ns = time.process_time_ns()
        for branch_id, divergent in zip(branch_ids, spec.divergent_token_ids, strict=True):
            branch_started_ns = time.monotonic_ns()
            branch_cpu_started_ns = time.process_time_ns()
            adapter.fork_same_policy_session(
                root.root_reference_id,
                branch_id,
                divergent_token_id=divergent,
                seed=spec.seed,
                timeout_s=_remaining(deadline_ns),
            )
            branch_cpu_end_ns = time.process_time_ns()
            branch_end_ns = time.monotonic_ns()
            phase_metrics.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "phase": "branch_metadata_created",
                    "branch_id": branch_id,
                    "wall_start_monotonic_ns": branch_started_ns,
                    "wall_end_monotonic_ns": branch_end_ns,
                    "wall_duration_ns": branch_end_ns - branch_started_ns,
                    "process_cpu_start_ns": branch_cpu_started_ns,
                    "process_cpu_end_ns": branch_cpu_end_ns,
                    "process_cpu_duration_ns": branch_cpu_end_ns - branch_cpu_started_ns,
                }
            )
            branch_logical = adapter.inspect_logical_state(
                branch_id, timeout_s=_remaining(deadline_ns)
            )
            _assert_logical_prefix(branch_logical, spec, divergent_token=divergent)
        fork_cpu_end_ns = time.process_time_ns()
        fork_end_ns = time.monotonic_ns()
        phase_metrics.append(
            {
                "schema_version": SCHEMA_VERSION,
                "phase": "fork_metadata",
                "wall_start_monotonic_ns": fork_started_ns,
                "wall_end_monotonic_ns": fork_end_ns,
                "wall_duration_ns": fork_end_ns - fork_started_ns,
                "process_cpu_start_ns": fork_cpu_started_ns,
                "process_cpu_end_ns": fork_cpu_end_ns,
                "process_cpu_duration_ns": fork_cpu_end_ns - fork_cpu_started_ns,
            }
        )
        branchpoint = adapter.capture_branchpoint(
            root.root_reference_id, branch_ids, timeout_s=_remaining(deadline_ns)
        )
        if (
            branchpoint.model != spec.expected_identity
            or branchpoint.policy_epoch != spec.expected_identity.policy_epoch
        ):
            raise SnapshotConsistencyError(
                "branchpoint model or policy epoch changed", operation="capture_branchpoint"
            )
        fork_snapshot = adapter.inspect_physical_kv_layout(
            (root_session_id, *branch_ids), timeout_s=_remaining(deadline_ns)
        )
        physical_rows.append(_snapshot_row(SnapshotPhase.FORK, fork_snapshot))
        assertions.extend(
            _validate_zero_copy_fork(
                root.block_ids,
                root.physical_bytes,
                branch_ids,
                root_snapshot,
                fork_snapshot,
            )
        )
        _phase_gpu_sample(adapter, SnapshotPhase.FORK.value, deadline_ns, gpu_rows)

        result = _measure_phase(
            "concurrent_decode",
            lambda: adapter.run_concurrent_branches(
                branch_ids,
                maximum_new_tokens=spec.suffix_tokens,
                seed=spec.seed,
                timeout_s=_remaining(deadline_ns),
            ),
            phase_metrics,
        )
        if result.branchpoint_id != branchpoint.branchpoint_id:
            raise SnapshotConsistencyError(
                "decode result refers to a different BranchPoint", operation="concurrent_decode"
            )
        if set(result.completed_branch_ids) != set(branch_ids):
            raise SnapshotConsistencyError(
                "not every concurrent branch completed", operation="concurrent_decode"
            )
        labeled = _label_decode_snapshots(spec, result)
        physical_rows.extend(_snapshot_row(phase, snapshot) for phase, snapshot in labeled)
        assertions.extend(
            _validate_shared_snapshots(root.block_ids, root.physical_bytes, spec.fanout, labeled)
        )
        assertions.extend(
            (
                "prefix_prefilled_once_before_branch_admission",
                "branch_token_histories_diverge_after_identical_prefix",
                "branches_decode_concurrently_under_one_branchpoint",
            )
        )
        _phase_gpu_sample(adapter, "post_decode", deadline_ns, gpu_rows)

        if spec.profile is ExperimentProfile.SMOKE:
            former_blocks = {
                block.runtime_block_id: block.allocation_epoch
                for _, snapshot in labeled
                for block in snapshot.blocks
                if block.state_class is KvStateClass.PRIVATE_SUFFIX
                and branch_ids[0] in block.branch_ids
            }
            if not former_blocks:
                raise SnapshotConsistencyError(
                    "smoke branch A produced no attributable private KV blocks",
                    operation="smoke_branch_delete",
                )
            before_delete = adapter.observe_block_release(
                branch_ids, timeout_s=_remaining(deadline_ns)
            )
            physical_rows.append(_snapshot_row(SnapshotPhase.PRE_BRANCH_DELETE, before_delete))
            before_native = adapter.inspect_block_release_evidence(
                tuple(sorted(former_blocks)), timeout_s=_remaining(deadline_ns)
            )
            release_evidence_rows.append(
                _release_evidence_row("pre_branch_a_destroy", before_native)
            )
            adapter.destroy_branch(branch_ids[0], timeout_s=_remaining(deadline_ns))
            after_native = adapter.inspect_block_release_evidence(
                tuple(sorted(former_blocks)), timeout_s=_remaining(deadline_ns)
            )
            release_evidence_rows.append(
                _release_evidence_row("post_branch_a_destroy", after_native)
            )
            _assert_exact_native_release(
                after_native,
                former_blocks,
                operation="smoke_branch_delete",
                require_hash_cleared=False,
            )
            if after_native.pool_free_block_count < before_native.pool_free_block_count:
                raise SnapshotConsistencyError(
                    "destroying smoke branch A reduced allocator availability",
                    operation="smoke_branch_delete",
                )
            after_delete = adapter.observe_block_release(
                (branch_ids[1],), timeout_s=_remaining(deadline_ns)
            )
            physical_rows.append(_snapshot_row(SnapshotPhase.POST_BRANCH_DELETE, after_delete))
            if set(after_delete.shared_prefix_block_ids) != set(root.block_ids):
                raise SnapshotConsistencyError(
                    "deleting smoke branch A corrupted branch B's root",
                    operation="smoke_branch_delete",
                )
            assertions.extend(
                (
                    "deleting_branch_a_preserves_branch_b_and_shared_root",
                    "branch_a_former_private_ids_native_refcount_zero_and_allocator_available",
                )
            )
            remaining = branch_ids[1:]
        else:
            remaining = branch_ids
        for branch_id in remaining:
            adapter.destroy_branch(branch_id, timeout_s=_remaining(deadline_ns))
        adapter.destroy_session(root_session_id, timeout_s=_remaining(deadline_ns))
        experiment_blocks = {
            block.runtime_block_id: block.allocation_epoch
            for _, snapshot in labeled
            for block in snapshot.blocks
            if block.state_class in {KvStateClass.SHARED_PREFIX, KvStateClass.PRIVATE_SUFFIX}
        }
        experiment_blocks.update(
            {
                block.runtime_block_id: block.allocation_epoch
                for block in root_snapshot.blocks
                if block.runtime_block_id in set(root.block_ids)
            }
        )
        if set(root.block_ids) - set(experiment_blocks):
            raise SnapshotConsistencyError(
                "root block allocation epochs were not retained for cleanup proof",
                operation="root_release",
            )
        final_native = adapter.inspect_block_release_evidence(
            tuple(sorted(experiment_blocks)), timeout_s=_remaining(deadline_ns)
        )
        release_evidence_rows.append(_release_evidence_row("post_root_destroy", final_native))
        _assert_exact_native_release(
            final_native,
            experiment_blocks,
            operation="root_release",
            require_hash_cleared=True,
        )
        if final_native.pool_free_block_count != final_native.pool_usable_block_count:
            raise SnapshotConsistencyError(
                "final root reset did not recover the complete usable KV pool",
                operation="root_release",
            )
        post_root = adapter.observe_block_release(
            (root_session_id,), timeout_s=_remaining(deadline_ns)
        )
        physical_rows.append(_snapshot_row(SnapshotPhase.POST_ROOT_DELETE, post_root))
        if post_root.shared_prefix_block_ids or post_root.private_suffix_block_ids:
            raise SnapshotConsistencyError(
                "final root destruction did not release experiment state",
                operation="root_release",
            )
        final_gpu = adapter.inspect_gpu_memory_state(timeout_s=_remaining(deadline_ns))
        if final_gpu.kv_assigned_bytes != 0:
            raise SnapshotConsistencyError(
                "final root reset left assigned or cache-resident KV bytes",
                operation="root_release",
            )
        assertions.extend(
            (
                "final_branch_and_root_release_physical_state",
                "final_root_exact_ids_hash_cleared_and_complete_kv_pool_recovered",
            )
        )
        _phase_gpu_sample(adapter, SnapshotPhase.POST_ROOT_DELETE.value, deadline_ns, gpu_rows)
        captured = _capture_traces(adapter, spec.trace_level, output)
        trace_artifacts = captured[1]
        final_snapshot = labeled[-1][1]
        summary = _build_trial_summary(
            run_id=run_id,
            path=TrialPath.SHARED_ROOT,
            spec=spec,
            result=result,
            root_bytes=root.physical_bytes,
            final=final_snapshot,
            trace_metrics=captured[0],
            assertions=assertions,
            phase_metrics=phase_metrics,
            readiness_offset_ns=fork_end_ns - fork_started_ns,
        )
    finally:
        try:
            adapter.cleanup_runtime(timeout_s=spec.cleanup_timeout_s)
        finally:
            if sampler_started:
                sampler.stop()
    return _persist_trial(
        output,
        spec,
        summary,
        physical_rows,
        gpu_rows,
        sampler.rows,
        release_evidence_rows,
        trace_artifacts,
    )


def run_independent_trial(
    adapter: IndependentDecodeAdapter,
    spec: ExperimentSpecification,
    output: Path,
    *,
    command_runner: CommandRunner = _default_command_runner,
) -> TrialSummary:
    """Run a cache-isolated independent-prefill baseline."""

    run_id = _run_id(spec)
    deadline_ns = time.monotonic_ns() + int(spec.timeout_s * 1_000_000_000)
    assertions: list[str] = []
    session_ids = _trial_session_ids(run_id, TrialPath.INDEPENDENT_PREFILL, spec.fanout)
    phase_metrics: list[dict[str, Any]] = []
    physical_rows: list[dict[str, Any]] = []
    gpu_rows: list[dict[str, Any]] = []
    sampler = NvidiaSmiSampler(
        device=spec.expected_identity.device,
        config=spec.gpu_sampling,
        command_runner=command_runner,
    )
    trace_artifacts: tuple[ArtifactRecord, ...] = ()
    sampler_started = False
    try:
        sampler.start()
        sampler_started = True
        assertions.extend(
            _assert_identity(
                adapter.capability_matrix,
                spec.expected_identity,
                TrialPath.INDEPENDENT_PREFILL,
            )
        )
        assertions.extend(_assert_trace_capabilities(adapter.capability_matrix, spec.trace_level))
        if (
            RealRuntimeCapability.CREATE_SHARED_ROOT_REFERENCE
            in adapter.capability_matrix.capabilities
        ):
            raise SnapshotConsistencyError(
                "independent adapter still exposes shared-root prefix caching",
                operation="independent_cache_isolation",
            )
        _phase_gpu_sample(adapter, "model_only", deadline_ns, gpu_rows)
        metadata_started_ns = time.monotonic_ns()
        metadata_cpu_started_ns = time.process_time_ns()
        for session_id, divergent in zip(session_ids, spec.divergent_token_ids, strict=True):
            session_started_ns = time.monotonic_ns()
            session_cpu_started_ns = time.process_time_ns()
            adapter.start_session(
                session_id,
                token_ids=(*spec.prefix_token_ids, divergent),
                seed=spec.seed,
                timeout_s=_remaining(deadline_ns),
            )
            session_cpu_end_ns = time.process_time_ns()
            session_end_ns = time.monotonic_ns()
            phase_metrics.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "phase": "independent_session_metadata_created",
                    "branch_id": session_id,
                    "wall_start_monotonic_ns": session_started_ns,
                    "wall_end_monotonic_ns": session_end_ns,
                    "wall_duration_ns": session_end_ns - session_started_ns,
                    "process_cpu_start_ns": session_cpu_started_ns,
                    "process_cpu_end_ns": session_cpu_end_ns,
                    "process_cpu_duration_ns": session_cpu_end_ns - session_cpu_started_ns,
                }
            )
            logical = adapter.inspect_logical_state(session_id, timeout_s=_remaining(deadline_ns))
            _assert_logical_prefix(logical, spec, divergent_token=divergent)
        metadata_cpu_end_ns = time.process_time_ns()
        metadata_end_ns = time.monotonic_ns()
        phase_metrics.append(
            {
                "schema_version": SCHEMA_VERSION,
                "phase": "independent_session_metadata",
                "wall_start_monotonic_ns": metadata_started_ns,
                "wall_end_monotonic_ns": metadata_end_ns,
                "wall_duration_ns": metadata_end_ns - metadata_started_ns,
                "process_cpu_start_ns": metadata_cpu_started_ns,
                "process_cpu_end_ns": metadata_cpu_end_ns,
                "process_cpu_duration_ns": metadata_cpu_end_ns - metadata_cpu_started_ns,
            }
        )
        result = _measure_phase(
            "independent_prefill_and_concurrent_decode",
            lambda: adapter.run_concurrent_independent_sessions(
                session_ids,
                maximum_new_tokens=spec.suffix_tokens,
                seed=spec.seed,
                timeout_s=_remaining(deadline_ns),
            ),
            phase_metrics,
        )
        if set(result.completed_branch_ids) != set(session_ids):
            raise SnapshotConsistencyError(
                "not every independent branch completed", operation="independent_decode"
            )
        labeled = _label_decode_snapshots(spec, result)
        physical_rows.extend(_snapshot_row(phase, snapshot) for phase, snapshot in labeled)
        assertions.extend(_validate_independent_snapshots(labeled))
        assertions.extend(
            (
                "same_model_tokenizer_dtype_gpu_scheduler_and_policy_as_shared_path",
                "every_branch_executes_complete_prefix_prefill",
                "independent_branches_decode_concurrently",
            )
        )
        _phase_gpu_sample(adapter, "post_decode", deadline_ns, gpu_rows)
        for session_id in session_ids:
            adapter.destroy_session(session_id, timeout_s=_remaining(deadline_ns))
        post_destroy_gpu = _phase_gpu_sample(adapter, "post_destroy", deadline_ns, gpu_rows)
        if post_destroy_gpu.kv_assigned_bytes != 0:
            raise SnapshotConsistencyError(
                "independent session destruction left runtime-assigned KV bytes",
                operation="independent_cleanup",
            )
        assertions.append("independent_post_destroy_kv_assigned_bytes_zero")
        captured = _capture_traces(adapter, spec.trace_level, output)
        trace_artifacts = captured[1]
        final_snapshot = labeled[-1][1]
        prefix_bytes = sum(
            block.bytes
            for block in final_snapshot.blocks
            if block.logical_token_start < len(spec.prefix_token_ids)
        )
        summary = _build_trial_summary(
            run_id=run_id,
            path=TrialPath.INDEPENDENT_PREFILL,
            spec=spec,
            result=result,
            root_bytes=prefix_bytes,
            final=final_snapshot,
            trace_metrics=captured[0],
            assertions=assertions,
            phase_metrics=phase_metrics,
            readiness_offset_ns=metadata_end_ns - metadata_started_ns,
        )
    finally:
        try:
            adapter.cleanup_runtime(timeout_s=spec.cleanup_timeout_s)
        finally:
            if sampler_started:
                sampler.stop()
    return _persist_trial(
        output,
        spec,
        summary,
        physical_rows,
        gpu_rows,
        sampler.rows,
        (),
        trace_artifacts,
    )


def _safe_fraction(numerator: float, denominator: float) -> float | None:
    if denominator == 0:
        return None
    value = numerator / denominator
    return value if math.isfinite(value) else None


def run_experiment(
    spec: ExperimentSpecification,
    adapter_factory: AdapterFactory,
    output: Path,
    *,
    command_runner: CommandRunner = _default_command_runner,
    launcher_configuration: BaseModel | None = None,
) -> TrialSummary | ComparisonSummary:
    """Run smoke or an independent/shared 16K comparison sequentially.

    The factory is invoked only when its path begins, and the runner cleans that
    adapter before requesting the next path. Independent and optimized engines
    therefore never occupy the GPU concurrently.
    """

    output.mkdir(parents=True, exist_ok=False)
    run_id = _run_id(spec)
    configuration_artifacts = [
        _write_new(output / "experiment-configuration.json", _canonical_bytes(spec)).model_copy(
            update={"relative_path": "experiment-configuration.json"}
        )
    ]
    if launcher_configuration is not None:
        configuration_artifacts.append(
            _write_new(
                output / "launcher-configuration.json",
                _canonical_bytes(launcher_configuration),
            ).model_copy(update={"relative_path": "launcher-configuration.json"})
        )
    if spec.profile is ExperimentProfile.SMOKE:
        adapter = adapter_factory(TrialPath.SHARED_ROOT)
        result = run_shared_trial(
            cast(RealRuntimeStateAdapter, adapter),
            spec,
            output / TrialPath.SHARED_ROOT.value,
            command_runner=command_runner,
        )
        trial_manifest = _artifact_for_existing(
            output / TrialPath.SHARED_ROOT.value / "artifact-manifest.json"
        ).model_copy(
            update={"relative_path": f"{TrialPath.SHARED_ROOT.value}/artifact-manifest.json"}
        )
        _write_new(
            output / "artifact-manifest.json",
            _canonical_bytes(
                {
                    "schema_version": SCHEMA_VERSION,
                    "run_id": run_id,
                    "artifacts": [
                        item.model_dump(mode="json")
                        for item in (*configuration_artifacts, trial_manifest)
                    ],
                }
            ),
        )
        return result

    independent_adapter = adapter_factory(TrialPath.INDEPENDENT_PREFILL)
    if not callable(getattr(independent_adapter, "run_concurrent_independent_sessions", None)):
        try:
            independent_adapter.cleanup_runtime(timeout_s=spec.cleanup_timeout_s)
        finally:
            raise UnsupportedCapabilityError(
                "independent baseline requires run_concurrent_independent_sessions; "
                "shared fork APIs may not conceal prefix reuse",
                operation="independent_prefill_baseline",
            )
    independent = run_independent_trial(
        cast(IndependentDecodeAdapter, independent_adapter),
        spec,
        output / TrialPath.INDEPENDENT_PREFILL.value,
        command_runner=command_runner,
    )
    # cleanup_runtime() drops all engine/view bound references.  Remove the
    # adapter itself and force collection before the comparison path creates a
    # second CUDA engine in this process.
    del independent_adapter
    gc.collect()
    try:
        import torch  # type: ignore[import-not-found]

        torch.cuda.empty_cache()
    except (ImportError, RuntimeError):
        pass
    shared_adapter = adapter_factory(TrialPath.SHARED_ROOT)
    shared = run_shared_trial(
        cast(RealRuntimeStateAdapter, shared_adapter),
        spec,
        output / TrialPath.SHARED_ROOT.value,
        command_runner=command_runner,
    )
    if independent.identity != shared.identity or independent.identity != spec.expected_identity:
        raise SnapshotConsistencyError(
            "baseline and optimized adapters used different runtime/model policies",
            operation="compare_trials",
        )
    sharing = (
        1.0 - shared.final_physical_bytes / independent.final_physical_bytes
        if independent.final_physical_bytes
        else None
    )
    readiness_speedup = _safe_fraction(
        float(independent.all_branches_ready_ns), float(shared.all_branches_ready_ns)
    )
    interference = (
        1.0
        - shared.decode_throughput_tokens_per_second
        / independent.decode_throughput_tokens_per_second
        if independent.decode_throughput_tokens_per_second
        else None
    )
    child_manifests = tuple(
        _artifact_for_existing(output / path.value / "artifact-manifest.json").model_copy(
            update={"relative_path": f"{path.value}/artifact-manifest.json"}
        )
        for path in (TrialPath.INDEPENDENT_PREFILL, TrialPath.SHARED_ROOT)
    )
    summary = ComparisonSummary(
        run_id=run_id,
        profile=spec.profile,
        seed=spec.seed,
        independent=independent,
        shared_root=shared,
        prefill_work_avoided_tokens=len(spec.prefix_token_ids) * (spec.fanout - 1),
        sharing_efficiency=sharing,
        readiness_speedup=readiness_speedup,
        decode_interference_fraction=interference,
        assertions=(
            "independent_and_shared_identity_equal",
            "independent_engine_cleaned_before_shared_engine_created",
            "independent_prefix_cache_isolation_asserted_from_physical_ids",
            "optimized_no_hidden_reprefill_asserted_from_shared_physical_ids",
        ),
        artifacts=(*configuration_artifacts, *child_manifests),
    )
    comparison_path = output / "metrics" / "comparison-summary.json"
    comparison_artifact = _write_new(comparison_path, _canonical_bytes(summary)).model_copy(
        update={"relative_path": "metrics/comparison-summary.json"}
    )
    _write_new(
        output / "artifact-manifest.json",
        _canonical_bytes(
            {
                "schema_version": SCHEMA_VERSION,
                "run_id": run_id,
                "artifacts": [
                    item.model_dump(mode="json")
                    for item in (*summary.artifacts, comparison_artifact)
                ],
            }
        ),
    )
    return summary


__all__ = [
    "RUNNER_VERSION",
    "SCHEMA_VERSION",
    "AdapterFactory",
    "ArtifactRecord",
    "CommandResult",
    "ComparisonSummary",
    "ExperimentProfile",
    "ExperimentSpecification",
    "GpuSamplingConfiguration",
    "IndependentDecodeAdapter",
    "InstrumentationAvailability",
    "NvidiaSmiProcessMemoryReader",
    "NvidiaSmiSampler",
    "SnapshotPhase",
    "TraceCaptureMetrics",
    "TrialPath",
    "TrialSummary",
    "VllmEngineConfiguration",
    "VllmGpuValidationInvocation",
    "build_vllm_adapter_factory",
    "run_experiment",
    "run_independent_trial",
    "run_shared_trial",
]
