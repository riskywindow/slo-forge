"""Restartable characterization runners that preserve raw and canonical evidence."""

from __future__ import annotations

import json
import platform
import shutil
import socket
from collections.abc import Mapping
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import psutil  # type: ignore[import-untyped]
from pydantic import BaseModel, ConfigDict, Field

from sloforge.continuum.characterization import (
    ListRecorder,
    StateOperationObservation,
    StateOperationRecorder,
    run_continuum_characterization,
)
from sloforge.continuum.characterization import (
    MeasurementKind as ContinuumMeasurementKind,
)
from sloforge.continuum.characterization import (
    TraceLevel as ContinuumTraceLevel,
)
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
from sloforge.helix.characterization.matrix import EvidenceClass
from sloforge.helix.characterization.matrix import TraceLevel as MatrixTraceLevel
from sloforge.helix.characterization.resources import ResourceSampler, ResourceSamplerConfig
from sloforge.helix.characterization.trace import (
    BoundedTraceBuffer,
    BranchWorkloadEventV1,
    CanonicalContinuumRecorder,
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
    resource_sample_count: int = Field(ge=0)
    resource_samples_dropped: int = Field(ge=0)
    wall_time_ns: int = Field(ge=0)
    cpu_time_ns: int = Field(ge=0)
    migration_observed: bool
    trace_corpus_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    parquet_status: Literal["written", "unavailable"]
    artifacts: tuple[str, ...]
    sharing_analysis: dict[str, object]
    limitations: tuple[str, ...]


class ContinuumTraceResult(RunnerModel):
    schema_version: Literal["sloforge.branchfabric.continuum-trace-run/v1"]
    workload_evidence_class: Literal["SYNTHETIC"]
    seed: int = Field(ge=0)
    trace_id: str
    output: str
    state_event_count: int = Field(ge=0)
    attempted_events: int = Field(ge=0)
    dropped_events: int = Field(ge=0)
    filtered_events: int = Field(ge=0)
    resource_sample_count: int = Field(ge=0)
    resource_samples_dropped: int = Field(ge=0)
    timing_measurement_counts: dict[str, int]
    migration_transaction_id: str
    migration_transport_kind: str
    source_continuation_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    divergent_continuation_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
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


class _TeeContinuumRecorder:
    def __init__(self, *recorders: StateOperationRecorder) -> None:
        self._recorders = recorders

    def record(self, observation: StateOperationObservation) -> None:
        for recorder in self._recorders:
            recorder.record(observation)


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
    tools = document.get("tools")
    packages = document.get("packages")
    if not isinstance(tools, dict) or not isinstance(packages, dict):
        raise ValueError("baseline software manifest is incomplete")
    components = tuple(
        VersionedComponent(name=name, version=str(version), source=baseline.as_posix())
        for name, version in sorted({**tools, **packages}.items())
        if version is not None
    )
    return SoftwareManifestV1(
        operating_system=platform.platform(),
        kernel=platform.release(),
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
    resource_sampler = ResourceSampler(
        ResourceSamplerConfig(
            trace_level=MatrixTraceLevel(trace_level.value),
            workload_evidence=EvidenceClass.SYNTHETIC,
            seed=seed,
            sample_interval_ms=100,
            max_samples=5000,
            max_duration_seconds=300.0,
        )
    ).start()
    try:
        run: CharacterizedRun = run_characterized_cpu_demo(
            output / "helix-demo",
            seed=seed,
            recorder=_TeeLifecycleRecorder(raw_recorder, canonical_recorder),
            trace_level=LifecycleTraceLevel(trace_level.value),
            trace_id=trace_id,
        )
    finally:
        resource_trace = resource_sampler.stop(timeout_seconds=5.0)
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

    raw_path = output / "raw-lifecycle-events.json"
    _write_json(
        raw_path,
        {
            "schema_version": "sloforge.branchfabric.raw-lifecycle-events/v1",
            "workload_evidence_class": "SYNTHETIC",
            "timing_measurement_class": "HARDWARE_BACKED_REAL",
            "branch_events": raw_recorder.branch_events,
            "state_events": raw_recorder.state_events,
        },
    )
    resource_path = output / "resource-trace-v1.json"
    _write_json(resource_path, resource_trace.model_dump(mode="json"))
    sharing = run.sharing_analysis or {}
    sharing_path = output / "sharing-analysis.json"
    _write_json(sharing_path, sharing)
    trace_artifacts += (
        _trace_artifact(raw_path, format="raw", event_count=len(events)),
        _trace_artifact(
            resource_path,
            format="resource",
            event_count=resource_trace.samples_recorded,
        ),
        _trace_artifact(sharing_path, format="analysis", event_count=0),
    )
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
        resource_sample_count=resource_trace.samples_recorded,
        resource_samples_dropped=resource_trace.samples_dropped,
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


def run_continuum_trace(
    output: Path,
    *,
    seed: int,
    buffer_capacity_events: int = 100_000,
    hardware_baseline: Path = Path("artifacts/branchfabric/manifests/hardware-baseline.json"),
    software_baseline: Path = Path("artifacts/branchfabric/manifests/software-baseline.json"),
) -> ContinuumTraceResult:
    """Capture the real reference Continuum lifecycle into StateOperationTrace v1.

    The workload state is a deterministic synthetic fixture. Host operation timings are
    measured; the deterministic migration transport retains its independent simulated-
    hardware timing class in every affected event.
    """

    if not 0 <= seed <= 2**63 - 1:
        raise ValueError("seed must fit signed 64-bit")
    if not 1 <= buffer_capacity_events <= MAX_TRACE_CAPACITY:
        raise ValueError(f"buffer capacity must be within 1..{MAX_TRACE_CAPACITY}")
    if output.exists() and (not output.is_dir() or output.is_symlink() or any(output.iterdir())):
        raise FileExistsError("Continuum trace output must be a new or empty directory")
    output.mkdir(parents=True, exist_ok=True)
    trace_id = canonical_hash(
        {
            "schema": "sloforge.branchfabric.continuum-trace-run/v1",
            "seed": seed,
            "trace_level": "full",
        }
    )
    sampling = SamplingConfigurationV1()
    buffer = BoundedTraceBuffer(
        trace_id=trace_id,
        capacity_events=buffer_capacity_events,
        level=TraceLevel.FULL,
        sampling=sampling,
    )
    recorder = CanonicalContinuumRecorder(
        buffer,
        trace_id=trace_id,
        session_id="continuum-characterization",
        branch_group_id="continuum-characterization-branches",
    )
    raw_recorder = ListRecorder(maximum_events=buffer_capacity_events)
    resource_sampler = ResourceSampler(
        ResourceSamplerConfig(
            trace_level=MatrixTraceLevel.FULL,
            workload_evidence=EvidenceClass.SYNTHETIC,
            seed=seed,
            sample_interval_ms=10,
            max_samples=5000,
            max_duration_seconds=300.0,
        )
    ).start()
    try:
        characterization = run_continuum_characterization(
            seed=seed,
            trace_level=ContinuumTraceLevel.FULL,
            recorder=_TeeContinuumRecorder(raw_recorder, recorder),
        )
    finally:
        resource_trace = resource_sampler.stop(timeout_seconds=5.0)
    events = buffer.drain()
    state_events = tuple(event for event in events if isinstance(event, StateOperationEventV1))
    state_path = output / "state-operation-trace-v1.jsonl"
    perfetto_path = output / "continuum-lifecycle.perfetto.json"
    state_artifact = write_jsonl(state_path, state_events)
    write_perfetto(perfetto_path, state_events)
    trace_artifacts: tuple[TraceArtifactV1, ...] = (
        state_artifact,
        _trace_artifact(perfetto_path, format="perfetto", event_count=len(state_events)),
    )

    raw_path = output / "raw-continuum-observations.json"
    _write_json(
        raw_path,
        {
            "schema_version": "sloforge.branchfabric.raw-continuum-observations/v1",
            "workload_evidence_class": "SYNTHETIC",
            "events": [event.as_dict() for event in raw_recorder.events],
            "dropped_events": raw_recorder.dropped_events,
        },
    )
    sharing_path = output / "sharing-analysis.json"
    sharing = asdict(characterization.sharing)
    _write_json(sharing_path, sharing)
    resource_path = output / "resource-trace-v1.json"
    _write_json(resource_path, resource_trace.model_dump(mode="json"))
    trace_artifacts += (
        _trace_artifact(raw_path, format="raw", event_count=len(raw_recorder.events)),
        _trace_artifact(sharing_path, format="analysis", event_count=0),
        _trace_artifact(
            resource_path,
            format="resource",
            event_count=resource_trace.samples_recorded,
        ),
    )

    parquet_status: Literal["written", "unavailable"] = "unavailable"
    parquet_path = output / "state-operation-trace-v1.parquet"
    try:
        parquet_count = write_parquet(parquet_path, state_events)
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

    stats = buffer.stats()
    manifest = build_manifest(
        trace_id=trace_id,
        session_id="continuum-characterization",
        created_at=datetime.now(UTC).isoformat(),
        seed=seed,
        provenance=WorkloadProvenance.SYNTHETIC,
        collection_level=TraceLevel.FULL,
        stats=stats,
        sampling=sampling,
        hardware=_hardware_manifest(hardware_baseline),
        software=_software_manifest(software_baseline),
        artifacts=trace_artifacts,
    )
    write_manifest(output / "trace-manifest-v1.json", manifest)
    timing_counts = {
        kind.value: sum(
            event.measurement_kind is kind for event in raw_recorder.events
        )
        for kind in ContinuumMeasurementKind
    }
    result = ContinuumTraceResult(
        schema_version="sloforge.branchfabric.continuum-trace-run/v1",
        workload_evidence_class="SYNTHETIC",
        seed=seed,
        trace_id=trace_id,
        output=output.as_posix(),
        state_event_count=len(state_events),
        attempted_events=stats.attempted_events,
        dropped_events=stats.dropped_events + raw_recorder.dropped_events,
        filtered_events=stats.filtered_events,
        resource_sample_count=resource_trace.samples_recorded,
        resource_samples_dropped=resource_trace.samples_dropped,
        timing_measurement_counts=timing_counts,
        migration_transaction_id=characterization.migration_transaction_id,
        migration_transport_kind=characterization.migration_transport_kind.value,
        source_continuation_hash=characterization.source_continuation_hash,
        divergent_continuation_hash=characterization.divergent_continuation_hash,
        trace_corpus_hash=manifest.trace_corpus_hash,
        parquet_status=parquet_status,
        artifacts=tuple(artifact.uri for artifact in trace_artifacts),
        sharing_analysis=sharing,
        limitations=(
            "The Continuum workload is a deterministic synthetic state fixture.",
            "Reference runtime state names simulated GPU layouts but every byte executes in host memory.",
            "Pre-copy network latency and bandwidth are deterministic simulated-hardware values.",
            "No GPU, PCIe, NVLink, RDMA, or multi-node timing is inferred.",
        ),
    )
    _write_json(output / "continuum-trace-run.json", result.model_dump(mode="json"))
    return result


__all__ = [
    "MAX_TRACE_CAPACITY",
    "ContinuumTraceResult",
    "VerticalTraceResult",
    "run_continuum_trace",
    "run_vertical_trace",
]
