"""Restartable characterization runners that preserve raw and canonical evidence."""

from __future__ import annotations

import json
import platform
import shutil
import socket
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import psutil  # type: ignore[import-untyped]
from pydantic import BaseModel, ConfigDict, Field

from sloforge.helix.characterization.lifecycle import (
    CharacterizedRun,
    InMemoryLifecycleRecorder,
    TraceStream,
    run_characterized_cpu_demo,
)
from sloforge.helix.characterization.lifecycle import (
    TraceLevel as LifecycleTraceLevel,
)
from sloforge.helix.characterization.lifecycle.recorder import JsonValue, LifecycleRecorder
from sloforge.helix.characterization.trace import (
    BoundedTraceBuffer,
    BranchWorkloadEventV1,
    CanonicalLifecycleRecorder,
    HardwareDevice,
    HardwareManifestV1,
    ParquetUnavailableError,
    SamplingConfigurationV1,
    SoftwareManifestV1,
    StateOperationEventV1,
    TraceArtifactV1,
    TraceLevel,
    VersionedComponent,
    WorkloadProvenance,
    artifact_from_file,
    build_manifest,
    canonical_hash,
    write_jsonl,
    write_manifest,
    write_parquet,
    write_perfetto,
)

MAX_TRACE_CAPACITY = 10_000_000


class RunnerModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, validate_default=True)


class VerticalTraceResult(RunnerModel):
    schema_version: Literal["sloforge.branchfabric.vertical-trace-run/v1"]
    workload_evidence_class: Literal["SYNTHETIC"]
    timing_measurement_class: Literal["HARDWARE_BACKED_REAL"]
    simulated_gpu_state: Literal[True]
    seed: int = Field(ge=0)
    trace_id: str
    output: str
    semantic_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    branch_event_count: int = Field(ge=0)
    state_event_count: int = Field(ge=0)
    attempted_events: int = Field(ge=0)
    dropped_events: int = Field(ge=0)
    filtered_events: int = Field(ge=0)
    wall_time_ns: int = Field(ge=0)
    cpu_time_ns: int = Field(ge=0)
    migration_observed: bool
    trace_corpus_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    parquet_status: Literal["written", "unavailable"]
    artifacts: tuple[str, ...]
    sharing_analysis: dict[str, object]
    limitations: tuple[str, ...]


class _TeeLifecycleRecorder:
    def __init__(self, *recorders: LifecycleRecorder) -> None:
        self._recorders = recorders

    def record(self, stream: TraceStream, event: Mapping[str, JsonValue]) -> None:
        for recorder in self._recorders:
            recorder.record(stream, event)


def _read_baseline_manifest(path: Path) -> dict[str, object]:
    document = json.loads(path.read_bytes())
    if not isinstance(document, dict):
        raise ValueError(f"baseline manifest {path} must be an object")
    return document


def _hardware_manifest(baseline: Path) -> HardwareManifestV1:
    document = _read_baseline_manifest(baseline)
    host = document.get("host")
    gpu = document.get("gpu")
    fabric = document.get("fabric")
    if not isinstance(host, dict) or not isinstance(gpu, dict) or not isinstance(fabric, dict):
        raise ValueError("baseline hardware manifest is incomplete")
    apple = gpu.get("apple_integrated")
    devices: tuple[HardwareDevice, ...] = ()
    if isinstance(apple, dict):
        devices = (
            HardwareDevice(
                kind="integrated_gpu",
                name=str(apple.get("model", "Apple integrated GPU")),
                attributes={
                    "cores": int(apple.get("cores", 0)),
                    "metal": str(apple.get("metal", "unknown")),
                    "memory_model": str(apple.get("memory_model", "unknown")),
                },
            ),
        )
    return HardwareManifestV1(
        host=socket.gethostname(),
        machine=str(host.get("model_identifier", platform.machine())),
        cpu_model=str(host.get("chip", platform.processor() or "unknown")),
        logical_cpu_count=int(host.get("logical_cpus", psutil.cpu_count() or 1)),
        memory_bytes=int(host.get("memory_bytes", psutil.virtual_memory().total)),
        numa_node_count=int(host.get("numa_nodes", 1)),
        devices=devices,
        pcie_paths=(str(fabric.get("pcie_gpu_path", "unknown")),),
        network_rails=("none-measured",),
        clock_settings=("not-locked",),
    )


def _software_manifest(baseline: Path) -> SoftwareManifestV1:
    document = _read_baseline_manifest(baseline)
    host = document.get("host")
    tools = document.get("tools")
    packages = document.get("packages")
    if not isinstance(host, dict) or not isinstance(tools, dict) or not isinstance(packages, dict):
        raise ValueError("baseline software manifest is incomplete")
    components = tuple(
        VersionedComponent(name=name, version=str(version), source=baseline.as_posix())
        for name, version in sorted({**tools, **packages}.items())
        if version is not None
    )
    return SoftwareManifestV1(
        operating_system=str(host.get("operating_system", platform.platform())),
        kernel=str(host.get("kernel", platform.release())),
        python_version=platform.python_version(),
        rust_version=(str(tools["rustc"]) if tools.get("rustc") else None),
        components=components,
    )


def _prepare_output(output: Path, *, replace: bool) -> None:
    if output.exists() and (not output.is_dir() or output.is_symlink()):
        raise FileExistsError("vertical trace output must be a regular directory")
    if output.exists() and any(output.iterdir()):
        marker = output / "vertical-run.json"
        if not replace or not marker.is_file():
            raise FileExistsError(
                "vertical trace output must be empty; replace requires vertical-run.json"
            )
        prior = json.loads(marker.read_bytes())
        if prior.get("schema_version") != "sloforge.branchfabric.vertical-trace-run/v1":
            raise ValueError("refusing to replace an unrecognized output directory")
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _trace_artifact(path: Path, *, format: str, event_count: int) -> TraceArtifactV1:
    return artifact_from_file(path, format=format, event_count=event_count)


def run_vertical_trace(
    output: Path,
    *,
    seed: int,
    trace_level: TraceLevel = TraceLevel.FULL,
    buffer_capacity_events: int = 100_000,
    replace: bool = False,
    hardware_baseline: Path = Path("artifacts/branchfabric/manifests/hardware-baseline.json"),
    software_baseline: Path = Path("artifacts/branchfabric/manifests/software-baseline.json"),
) -> VerticalTraceResult:
    """Trace one real Helix CPU demo lifecycle into both canonical formats."""

    if not 0 <= seed <= 2**63 - 1:
        raise ValueError("seed must fit signed 64-bit")
    if not 1 <= buffer_capacity_events <= MAX_TRACE_CAPACITY:
        raise ValueError(f"buffer capacity must be within 1..{MAX_TRACE_CAPACITY}")
    _prepare_output(output, replace=replace)
    trace_id = canonical_hash(
        {
            "schema": "sloforge.branchfabric.vertical-trace-run/v1",
            "seed": seed,
            "trace_level": trace_level.value,
        }
    )
    sampling = SamplingConfigurationV1()
    buffer = BoundedTraceBuffer(
        trace_id=trace_id,
        capacity_events=buffer_capacity_events,
        level=trace_level,
        sampling=sampling,
    )
    raw_recorder = InMemoryLifecycleRecorder()
    canonical_recorder = CanonicalLifecycleRecorder(buffer)
    run: CharacterizedRun = run_characterized_cpu_demo(
        output / "helix-demo",
        seed=seed,
        recorder=_TeeLifecycleRecorder(raw_recorder, canonical_recorder),
        trace_level=LifecycleTraceLevel(trace_level.value),
        trace_id=trace_id,
    )
    events = buffer.drain()
    branch_events = tuple(event for event in events if isinstance(event, BranchWorkloadEventV1))
    state_events = tuple(event for event in events if isinstance(event, StateOperationEventV1))
    branch_path = output / "branch-workload-trace-v1.jsonl"
    state_path = output / "state-operation-trace-v1.jsonl"
    perfetto_path = output / "helix-lifecycle.perfetto.json"
    branch_artifact = write_jsonl(branch_path, branch_events)
    state_artifact = write_jsonl(state_path, state_events)
    write_perfetto(perfetto_path, events)
    perfetto_artifact = _trace_artifact(perfetto_path, format="perfetto", event_count=len(events))

    parquet_status: Literal["written", "unavailable"] = "unavailable"
    trace_artifacts: tuple[TraceArtifactV1, ...] = (
        branch_artifact,
        state_artifact,
        perfetto_artifact,
    )
    parquet_path = output / "trace-v1.parquet"
    try:
        parquet_count = write_parquet(parquet_path, events)
    except ParquetUnavailableError as error:
        _write_json(
            output / "parquet-status.json",
            {
                "status": "unavailable",
                "reason": str(error),
                "authoritative_format": "jsonl",
            },
        )
    else:
        parquet_status = "written"
        trace_artifacts += (
            _trace_artifact(parquet_path, format="parquet", event_count=parquet_count),
        )

    _write_json(
        output / "raw-lifecycle-events.json",
        {
            "schema_version": "sloforge.branchfabric.raw-lifecycle-events/v1",
            "workload_evidence_class": "SYNTHETIC",
            "timing_measurement_class": "HARDWARE_BACKED_REAL",
            "branch_events": raw_recorder.branch_events,
            "state_events": raw_recorder.state_events,
        },
    )
    sharing = run.sharing_analysis or {}
    _write_json(output / "sharing-analysis.json", sharing)
    stats = buffer.stats()
    manifest = build_manifest(
        trace_id=trace_id,
        session_id="production-session",
        created_at=datetime.now(UTC).isoformat(),
        seed=seed,
        provenance=WorkloadProvenance.SYNTHETIC,
        collection_level=trace_level,
        stats=stats,
        sampling=sampling,
        hardware=_hardware_manifest(hardware_baseline),
        software=_software_manifest(software_baseline),
        artifacts=trace_artifacts,
    )
    write_manifest(output / "trace-manifest-v1.json", manifest)
    result = VerticalTraceResult(
        schema_version="sloforge.branchfabric.vertical-trace-run/v1",
        workload_evidence_class="SYNTHETIC",
        timing_measurement_class="HARDWARE_BACKED_REAL",
        simulated_gpu_state=True,
        seed=seed,
        trace_id=trace_id,
        output=output.as_posix(),
        semantic_digest=run.semantic_digest,
        branch_event_count=len(branch_events),
        state_event_count=len(state_events),
        attempted_events=stats.attempted_events,
        dropped_events=stats.dropped_events,
        filtered_events=stats.filtered_events,
        wall_time_ns=run.wall_time_ns,
        cpu_time_ns=run.cpu_time_ns,
        migration_observed=run.migration_observed,
        trace_corpus_hash=manifest.trace_corpus_hash,
        parquet_status=parquet_status,
        artifacts=tuple(artifact.uri for artifact in trace_artifacts),
        sharing_analysis=sharing,
        limitations=(
            "The exercised coding-agent workload is a deterministic synthetic CPU fixture.",
            "Model state sizes are simulated; no GPU HBM, PCIe, or GPU kernel timing is claimed.",
            "The reference environment fork eagerly restores files; live filesystem COW is absent.",
            "This demo does not migrate a post-fork branch; migration is measured separately.",
        ),
    )
    _write_json(output / "vertical-run.json", result.model_dump(mode="json"))
    return result


__all__ = ["MAX_TRACE_CAPACITY", "VerticalTraceResult", "run_vertical_trace"]
