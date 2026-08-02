"""Bounded execution and ingestion for explicitly selected hardware adapters."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import selectors
import signal
import subprocess
import time
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from sloforge.fabric.profiling.adapters import AdapterCommand
from sloforge.fabric.profiling.benchmark import summarize_samples
from sloforge.fabric.profiling.models import (
    BenchmarkCase,
    BenchmarkResult,
    BenchmarkStatus,
    Direction,
    EnvironmentFact,
    FabricProfile,
    Invocation,
    MeasurementMode,
    Placement,
    Primitive,
    RawSample,
    finalize_profile,
    finalize_result,
)
from sloforge.util import sha256_file, utc_now, write_json

NcclOperation = Literal[
    "all_reduce", "all_gather", "reduce_scatter", "broadcast", "send_receive", "all_to_all"
]

_OPERATION_PRIMITIVE: dict[NcclOperation, Primitive] = {
    "all_reduce": Primitive.ALL_REDUCE,
    "all_gather": Primitive.ALL_GATHER,
    "reduce_scatter": Primitive.REDUCE_SCATTER,
    "broadcast": Primitive.BROADCAST,
    "send_receive": Primitive.SEND_RECV,
    "all_to_all": Primitive.ALL_TO_ALL,
}
_PASSTHROUGH_ENVIRONMENT = (
    "PATH",
    "LD_LIBRARY_PATH",
    "DYLD_LIBRARY_PATH",
    "SYSTEMROOT",
    "TMPDIR",
    "TMP",
    "TEMP",
)


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class CommandCapture(_StrictModel):
    """Bounded process evidence; stdout and stderr never exceed the configured cap."""

    repetition: int = Field(ge=0)
    return_code: int
    duration_seconds: float = Field(ge=0.0)
    stdout: str
    stderr: str
    timed_out: bool
    output_limited: bool


class NcclTestsRow(_StrictModel):
    """One standard nccl-tests result row, retaining both buffer modes."""

    message_bytes: int = Field(gt=0)
    element_count: int = Field(gt=0)
    datatype: str = Field(min_length=1)
    reduction: str = Field(min_length=1)
    root: int
    out_of_place_time_us: float = Field(gt=0.0)
    out_of_place_algorithm_gbps: float = Field(ge=0.0)
    out_of_place_bus_gbps: float = Field(ge=0.0)
    out_of_place_wrong: str
    in_place_time_us: float = Field(gt=0.0)
    in_place_algorithm_gbps: float = Field(ge=0.0)
    in_place_bus_gbps: float = Field(ge=0.0)
    in_place_wrong: str


class NvidiaInventoryRecord(_StrictModel):
    """One explicitly requested read-only nvidia-smi query result."""

    gpu_id: str = Field(min_length=1)
    fields: tuple[tuple[str, str], ...]
    capture_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class AdapterExecutionError(RuntimeError):
    """Raised when an explicitly selected measured adapter cannot be trusted."""


def _sanitized_environment(command: AdapterCommand) -> dict[str, str]:
    environment = {
        name: os.environ[name] for name in _PASSTHROUGH_ENVIRONMENT if name in os.environ
    }
    # In particular, do not inherit CUDA_VISIBLE_DEVICES or any ambient NCCL
    # tuning/transport knob. The typed command is the complete adapter overlay.
    environment.update(dict(command.environment))
    environment["SLOFORGE_MEASURED_ADAPTER"] = "1"
    return environment


def _terminate_process_group(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    if os.name == "posix":
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
    else:  # pragma: no cover - Windows is not a supported GPU benchmark host.
        process.terminate()
    try:
        process.wait(timeout=0.5)
        return
    except subprocess.TimeoutExpired:
        if os.name == "posix":
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                return
        else:  # pragma: no cover
            process.kill()


def execute_bounded(
    command: AdapterCommand,
    *,
    repetition: int,
    maximum_output_bytes: int = 1 << 20,
) -> CommandCapture:
    """Execute one adapter command with a deadline and capped pipe ingestion."""

    if not 4_096 <= maximum_output_bytes <= 16 * 1024 * 1024:
        raise ValueError("maximum_output_bytes must be in [4096, 16 MiB]")
    started = time.monotonic()
    process = subprocess.Popen(
        command.argv,
        executable=command.executable,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=_sanitized_environment(command),
        start_new_session=True,
    )
    assert process.stdout is not None
    assert process.stderr is not None
    selector = selectors.DefaultSelector()
    selector.register(process.stdout, selectors.EVENT_READ, "stdout")
    selector.register(process.stderr, selectors.EVENT_READ, "stderr")
    buffers: dict[str, bytearray] = {"stdout": bytearray(), "stderr": bytearray()}
    timed_out = False
    output_limited = False
    deadline = started + command.timeout_seconds
    try:
        while selector.get_map():
            remaining = deadline - time.monotonic()
            if remaining <= 0.0 and process.poll() is None:
                timed_out = True
                _terminate_process_group(process)
            events = selector.select(timeout=max(0.0, min(0.05, remaining)))
            for key, _ in events:
                chunk = os.read(key.fd, 65_536)
                if not chunk:
                    selector.unregister(key.fileobj)
                    continue
                target = buffers[str(key.data)]
                if len(target) + len(chunk) > maximum_output_bytes:
                    output_limited = True
                    keep = max(0, maximum_output_bytes - len(target))
                    target.extend(chunk[:keep])
                    _terminate_process_group(process)
                else:
                    target.extend(chunk)
            if process.poll() is not None and not events:
                # Pipes reach EOF on the following selector iteration.
                continue
        return_code = process.wait(timeout=1.0)
    finally:
        selector.close()
        _terminate_process_group(process)
        if process.poll() is None:  # pragma: no cover - kill fallback is defensive.
            process.kill()
            process.wait(timeout=1.0)
    return CommandCapture(
        repetition=repetition,
        return_code=return_code,
        duration_seconds=time.monotonic() - started,
        stdout=bytes(buffers["stdout"]).decode("utf-8", errors="replace"),
        stderr=bytes(buffers["stderr"]).decode("utf-8", errors="replace"),
        timed_out=timed_out,
        output_limited=output_limited,
    )


def parse_nccl_tests_output(output: str) -> tuple[NcclTestsRow, ...]:
    """Parse the stable 13-column table emitted by standard nccl-tests tools."""

    rows: list[NcclTestsRow] = []
    seen_sizes: set[int] = set()
    for line in output.splitlines():
        fields = line.split()
        if len(fields) < 13 or line.lstrip().startswith("#"):
            continue
        try:
            row = NcclTestsRow(
                message_bytes=int(fields[0]),
                element_count=int(fields[1]),
                datatype=fields[2],
                reduction=fields[3],
                root=int(fields[4]),
                out_of_place_time_us=float(fields[5]),
                out_of_place_algorithm_gbps=float(fields[6]),
                out_of_place_bus_gbps=float(fields[7]),
                out_of_place_wrong=fields[8],
                in_place_time_us=float(fields[9]),
                in_place_algorithm_gbps=float(fields[10]),
                in_place_bus_gbps=float(fields[11]),
                in_place_wrong=fields[12],
            )
        except (ValueError, TypeError):
            continue
        if row.message_bytes in seen_sizes:
            raise AdapterExecutionError(
                f"nccl-tests output repeated message size {row.message_bytes}"
            )
        seen_sizes.add(row.message_bytes)
        rows.append(row)
    if not rows:
        raise AdapterExecutionError("nccl-tests output contained no recognized measurement rows")
    return tuple(rows)


def _capture_digest(capture: CommandCapture) -> str:
    encoded = json.dumps(
        capture.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def read_nvidia_inventory(
    command: AdapterCommand,
    *,
    gpu_id: str,
    fields: tuple[str, ...],
    maximum_output_bytes: int = 64 * 1024,
) -> NvidiaInventoryRecord:
    """Execute one allowlisted nvidia-smi query; mutation commands are impossible here."""

    if command.adapter != "nvidia-smi-query" or command.expected_transport != "nvml":
        raise ValueError("inventory execution requires a typed nvidia-smi query command")
    capture = execute_bounded(command, repetition=0, maximum_output_bytes=maximum_output_bytes)
    if capture.timed_out or capture.output_limited or capture.return_code != 0:
        raise AdapterExecutionError(
            "nvidia-smi inventory query failed "
            f"(return_code={capture.return_code}, timeout={capture.timed_out}, "
            f"output_limited={capture.output_limited})"
        )
    parsed = list(csv.reader(capture.stdout.splitlines(), skipinitialspace=True))
    rows = [
        tuple(value.strip() for value in row)
        for row in parsed
        if any(value.strip() for value in row)
    ]
    if len(rows) != 1 or len(rows[0]) != len(fields):
        raise AdapterExecutionError(
            "nvidia-smi inventory output did not contain exactly one requested row"
        )
    return NvidiaInventoryRecord(
        gpu_id=gpu_id,
        fields=tuple(zip(fields, rows[0], strict=True)),
        capture_sha256=_capture_digest(capture),
    )


def _expected_message_sizes(minimum: int, maximum: int, factor: int) -> tuple[int, ...]:
    sizes: list[int] = []
    current = minimum
    while current <= maximum:
        sizes.append(current)
        if current > maximum // factor:
            break
        current *= factor
    if not sizes or sizes[-1] != maximum:
        raise ValueError("NCCL message range must end exactly on a step-factor boundary")
    return tuple(sizes)


def run_nccl_tests_profile(
    *,
    command: AdapterCommand,
    operation: NcclOperation,
    topology_fingerprint: str,
    suite: Literal["quick", "full"],
    minimum_bytes: int,
    maximum_bytes: int,
    step_factor: int,
    repetitions: int,
    warmup_count: int,
    seed: int,
    adapter_version: str | None = None,
    inventory: tuple[NvidiaInventoryRecord, ...] = (),
    output_dir: Path | None = None,
    maximum_output_bytes: int = 1 << 20,
) -> FabricProfile:
    """Run independent nccl-tests processes and retain each table row as evidence."""

    if command.adapter != "nccl-tests" or command.expected_transport != "nccl-local":
        raise ValueError("measured NCCL execution requires the explicit nccl-tests adapter")
    if not command.requires_gpu or command.requires_multi_process:
        raise ValueError("this runner supports one explicit local NCCL GPU process only")
    if not 3 <= repetitions <= 100:
        raise ValueError("NCCL repetitions must be in [3, 100]")
    expected_sizes = _expected_message_sizes(minimum_bytes, maximum_bytes, step_factor)
    environment_overlay = dict(command.environment)
    visible_devices = tuple(environment_overlay.get("CUDA_VISIBLE_DEVICES", "").split(","))
    if not visible_devices or any(not device for device in visible_devices):
        raise ValueError("NCCL execution requires explicit CUDA_VISIBLE_DEVICES")

    environment: list[EnvironmentFact] = [
        EnvironmentFact(name="platform", value=platform.platform(), source="python-platform"),
        EnvironmentFact(name="machine", value=platform.machine(), source="python-platform"),
        EnvironmentFact(name="adapter", value="nccl-tests", source="explicit-cli-selection"),
        EnvironmentFact(
            name="adapter_executable_sha256",
            value=sha256_file(Path(command.executable)),
            source="resolved-executable",
        ),
        EnvironmentFact(
            name="expected_transport", value=command.expected_transport, source="typed-command"
        ),
        EnvironmentFact(
            name="visible_devices", value=",".join(visible_devices), source="typed-command"
        ),
    ]
    for record in inventory:
        if record.gpu_id not in visible_devices:
            raise ValueError(f"inventory GPU {record.gpu_id} is outside the explicit device set")
        environment.append(
            EnvironmentFact(
                name=f"nvidia.{record.gpu_id}.capture_sha256",
                value=record.capture_sha256,
                source="bounded-read-only-nvidia-smi",
            )
        )
        environment.extend(
            EnvironmentFact(
                name=f"nvidia.{record.gpu_id}.{name}",
                value=value,
                source="bounded-read-only-nvidia-smi",
            )
            for name, value in record.fields
        )

    captures: list[CommandCapture] = []
    parsed_runs: list[dict[int, NcclTestsRow]] = []
    failure_reason: str | None = None
    for repetition in range(repetitions):
        capture = execute_bounded(
            command, repetition=repetition, maximum_output_bytes=maximum_output_bytes
        )
        captures.append(capture)
        environment.append(
            EnvironmentFact(
                name=f"capture.{repetition}.sha256",
                value=_capture_digest(capture),
                source="bounded-subprocess",
            )
        )
        if capture.timed_out or capture.output_limited or capture.return_code != 0:
            failure_reason = (
                f"nccl-tests repetition {repetition} failed: return_code={capture.return_code}, "
                f"timeout={capture.timed_out}, output_limited={capture.output_limited}"
            )
            break
        try:
            rows = parse_nccl_tests_output(capture.stdout)
        except AdapterExecutionError as error:
            failure_reason = f"nccl-tests repetition {repetition} parse failed: {error}"
            break
        incorrect = tuple(
            row.message_bytes
            for row in rows
            if row.out_of_place_wrong not in {"0", "0.0"} or row.in_place_wrong not in {"0", "0.0"}
        )
        if incorrect:
            failure_reason = (
                f"nccl-tests repetition {repetition} reported incorrect output for "
                f"message sizes {list(incorrect)}"
            )
            break
        by_size = {row.message_bytes: row for row in rows}
        if set(by_size) != set(expected_sizes):
            failure_reason = (
                f"nccl-tests repetition {repetition} returned sizes {sorted(by_size)}, "
                f"expected {list(expected_sizes)}"
            )
            break
        parsed_runs.append(by_size)

    if output_dir is not None:
        capture_dir = output_dir / "captures"
        capture_dir.mkdir(parents=True, exist_ok=True)
        for capture in captures:
            write_json(
                capture_dir / f"nccl-{operation}-{capture.repetition:03d}.json",
                capture.model_dump(mode="json"),
            )

    primitive = _OPERATION_PRIMITIVE[operation]
    results: list[BenchmarkResult] = []
    for message_bytes in expected_sizes:
        case_id = f"measured-nccl-{operation}-b{message_bytes}-r{len(visible_devices)}-c1"
        case = BenchmarkCase(
            case_id=case_id,
            primitive=primitive,
            message_bytes=message_bytes,
            rank_count=len(visible_devices),
            concurrency=1,
            direction=Direction.NOT_APPLICABLE,
            topology_path=(),
            contention_domains=("nccl-explicit-device-set",),
            placement=Placement(
                hosts=(platform.node() or "localhost",),
                ranks=tuple(range(len(visible_devices))),
                gpu_ids=visible_devices,
                numa_domains=(),
                nic_ids=(),
            ),
            warmup_count=warmup_count,
            sample_count=repetitions,
            invocation=Invocation(
                adapter="nccl-tests",
                adapter_version=adapter_version,
                argv=command.argv,
                timeout_seconds=command.timeout_seconds,
                environment=command.environment,
            ),
        )
        raw_artifact = str(Path("raw") / f"{case_id}.json")
        if failure_reason is not None:
            result = finalize_result(
                schema_version="sloforge.fabric.benchmark-result/v1",
                case=case,
                mode=MeasurementMode.MEASURED,
                status=BenchmarkStatus.FAILED,
                raw_samples=(),
                summary=None,
                environment=tuple(environment),
                failure_reason=failure_reason,
                raw_artifact=raw_artifact,
            )
        else:
            samples = tuple(
                RawSample(
                    sample_index=index,
                    duration_microseconds=run[message_bytes].out_of_place_time_us,
                    throughput_bytes_per_second=(
                        run[message_bytes].out_of_place_algorithm_gbps * 1_000_000_000.0
                    ),
                    synthetic=False,
                    seed=None,
                )
                for index, run in enumerate(parsed_runs)
            )
            result = finalize_result(
                schema_version="sloforge.fabric.benchmark-result/v1",
                case=case,
                mode=MeasurementMode.MEASURED,
                status=BenchmarkStatus.SUCCESS,
                raw_samples=samples,
                summary=summarize_samples(samples, seed=seed ^ message_bytes),
                environment=tuple(environment),
                failure_reason=None,
                raw_artifact=raw_artifact,
            )
        results.append(result)

    profile = finalize_profile(
        schema_version="sloforge.fabric.profile/v1",
        profile_id=f"measured-nccl-{operation}-{hashlib.sha256(command.executable.encode()).hexdigest()[:12]}",
        captured_at=utc_now(),
        topology_fingerprint=topology_fingerprint,
        seed=seed,
        suite=f"{suite}:nccl-tests:{operation}",
        results=tuple(results),
        environment=tuple(environment),
    )
    return profile
