"""Integrated, evidence-preserving Helix characterization workflow.

This module is the stable backend used by the CLI.  It coordinates bounded
measurements and analysis; it never provisions hardware or fills unavailable
counters with estimates.
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from sloforge.helix.characterization.analysis.amdahl import (
    AmdahlTimingSample,
    CandidatePrimitive,
    EndToEndObjective,
    analyze_amdahl,
)
from sloforge.helix.characterization.matrix import (
    DivergencePattern,
    EvidenceClass,
    WorkloadClass,
    expand_matrix,
    load_matrix,
)
from sloforge.helix.characterization.orchestration import (
    CPU_STAGES,
    DEFAULT_STAGES,
    CharacterizationRunManifest,
    OrchestrationConfig,
    RunState,
    characterization_run_result,
    load_characterization_run,
    resume_characterization_run,
)
from sloforge.helix.characterization.orchestration import (
    run_characterization as run_orchestrated_characterization,
)
from sloforge.helix.characterization.projection import (
    calibrate_state,
    project_attention_cow,
    project_state_composition,
)
from sloforge.helix.characterization.requirements import (
    ArtifactBinding,
    Availability,
    BandwidthRequirement,
    BranchFabricRequirementsV1,
    ConfidenceOrPercentile,
    CowRequirements,
    DistributionRequirement,
    EnumRequirement,
    EvidenceReference,
    InterfaceKind,
    IsaClassification,
    IsaClassificationRequirement,
    IsaOperationRecommendation,
    LatencyRequirement,
    MemoryCapacityRequirement,
    MetadataRequirements,
    NetworkRequirements,
    NumericRequirement,
    PageSizeRequirement,
    QueueRequirement,
    RequirementsCompilationInput,
    RequirementsDraft,
    RequirementUnit,
    StateRequirements,
    TransactionRequirements,
    TransformRequirements,
    TransformSequenceRequirement,
    WorkloadRequirement,
    compile_requirements,
    write_requirements,
)
from sloforge.helix.characterization.trace import (
    BranchOperationType,
    BranchWorkloadEventV1,
    StateOperationEventV1,
    StateOperationType,
    StateSegment,
    TimingMeasurementClass,
    TraceEventV1,
    WorkloadProvenance,
    iter_jsonl,
)
from sloforge.helix.characterization.trace.manifest import trace_corpus_hash
from sloforge.helix.characterization.trace.models import TraceArtifactV1, TraceManifestV1

MAX_WORKFLOW_EVENTS = 10_000_000


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: object, *, replace: bool) -> Path:
    if path.exists() and not replace:
        raise FileExistsError(f"output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return path


def _output_file(output: Path, default_name: str) -> Path:
    return output if output.suffix == ".json" else output / default_name


def _safe_replace_run(output: Path) -> None:
    if not output.exists():
        return
    marker = output / "run-manifest.json"
    if not output.is_dir() or output.is_symlink() or not marker.is_file():
        raise FileExistsError("replace requires a recognized characterization run directory")
    load_characterization_run(output)
    shutil.rmtree(output)


def run_characterization(
    *,
    matrix: Path,
    output: Path,
    hardware: str,
    seed: int,
    max_experiments: int,
    timeout_seconds: float,
    replace: bool,
) -> object:
    """Run the bounded CPU workflow and explicitly skip unavailable hardware stages."""

    loaded = load_matrix(matrix)
    cases = expand_matrix(loaded)
    if len(cases) > max_experiments:
        raise ValueError(
            f"matrix expands to {len(cases)} cases, above max_experiments={max_experiments}"
        )
    if hardware not in {"cpu", "gpu"}:
        raise ValueError("hardware must be cpu or gpu")
    if replace:
        _safe_replace_run(output)
    elif output.exists():
        raise FileExistsError("characterization output already exists")
    stages = CPU_STAGES if hardware == "cpu" else DEFAULT_STAGES
    config = OrchestrationConfig(
        seed=seed,
        stages=stages,
        stage_timeout_seconds=timeout_seconds,
        environment_repetitions=1,
        environment_warmups=0,
        metadata_operations_per_thread=8,
        metadata_repetitions=3,
        metadata_warmups=1,
        metadata_thread_counts=(1, 2, 4),
        overhead_repetitions=1,
    )
    return run_orchestrated_characterization(
        output,
        matrix_path=matrix,
        config=config,
    )


def resume_characterization(*, run: Path, max_experiments: int, timeout_seconds: float) -> object:
    manifest = load_characterization_run(run)
    if manifest.matrix_case_count > max_experiments:
        raise ValueError(
            f"run contains {manifest.matrix_case_count} cases, above {max_experiments}"
        )
    if timeout_seconds != manifest.config.stage_timeout_seconds:
        raise ValueError(
            "resume timeout must equal the immutable run timeout "
            f"({manifest.config.stage_timeout_seconds:g} seconds)"
        )
    return resume_characterization_run(run)


def _load_trace(path: Path, *, max_events: int) -> tuple[TraceEventV1, ...]:
    if not 1 <= max_events <= MAX_WORKFLOW_EVENTS:
        raise ValueError(f"max_events must be within 1..{MAX_WORKFLOW_EVENTS}")
    events: list[TraceEventV1] = []
    for event in iter_jsonl(path):
        if len(events) >= max_events:
            raise ValueError(f"trace exceeds the {max_events}-event analysis bound")
        events.append(event)
    if not events:
        raise ValueError("trace contains no events")
    return tuple(events)


def _percentile(values: Sequence[int | float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _distribution(values: Sequence[int | float]) -> dict[str, float | int | None]:
    return {
        "sample_count": len(values),
        "mean": sum(values) / len(values) if values else None,
        "p50": _percentile(values, 0.50),
        "p90": _percentile(values, 0.90),
        "p95": _percentile(values, 0.95),
        "p99": _percentile(values, 0.99),
        "maximum": max(values) if values else None,
    }


def analyze_workload(
    *, trace: Path, output: Path, seed: int, max_events: int, replace: bool
) -> dict[str, object]:
    """Produce a compact workload summary when only one canonical stream is supplied."""

    events = _load_trace(trace, max_events=max_events)
    branches = [event for event in events if isinstance(event, BranchWorkloadEventV1)]
    states = [event for event in events if isinstance(event, StateOperationEventV1)]
    groups: dict[str, set[str]] = defaultdict(set)
    for event in branches:
        if event.branch_group_id and event.branch_id:
            groups[event.branch_group_id].add(event.branch_id)
    fork_by_branch = {
        event.branch_id: event
        for event in branches
        if event.operation_type.value == "BRANCH_FORK" and event.branch_id
    }
    terminal_by_branch = {
        event.branch_id: event
        for event in branches
        if event.operation_type.value in {"BRANCH_COMPLETE", "BRANCH_PRUNE", "BRANCH_ABORT"}
        and event.branch_id
    }
    lifetimes = [
        terminal.normalized_timestamp_ns + terminal.duration_ns - fork.normalized_timestamp_ns
        for branch_id, fork in fork_by_branch.items()
        if (terminal := terminal_by_branch.get(branch_id)) is not None
    ]
    result: dict[str, object] = {
        "schema_version": "sloforge.branchfabric.workload-analysis/v1",
        "source_experiment": "cli.workload",
        "artifact_reference": trace.as_posix(),
        "artifact_sha256": _sha256(trace),
        "seed": seed,
        "input_event_count": len(events),
        "branch_event_count": len(branches),
        "state_event_count": len(states),
        "workload_provenance": sorted({event.provenance.value for event in events}),
        "timing_measurement_classes": sorted(
            {event.timing_measurement_class.value for event in events}
        ),
        "branch_group_count": len(groups),
        "branch_fanout": _distribution([len(value) for value in groups.values()]),
        "branch_lifetime_ns": _distribution(lifetimes),
        "operation_counts": dict(
            sorted(Counter(event.operation_type.value for event in events).items())
        ),
        "limitations": [
            "A single stream cannot recover sibling token divergence unless trajectory artifacts are supplied.",
            "Nested event durations are not summed as exclusive critical-path time.",
        ],
    }
    target = _write_json(_output_file(output, "workload-analysis.json"), result, replace=replace)
    return {"output": target.as_posix(), "artifact_sha256": _sha256(target), **result}


def analyze_cow(
    *,
    trace: Path,
    output: Path,
    page_sizes: tuple[int, ...],
    seed: int,
    max_events: int,
    replace: bool,
) -> dict[str, object]:
    """Run a controlled COW sweep calibrated to the captured CPU fixture."""

    _load_trace(trace, max_events=max_events)
    if not page_sizes or len(page_sizes) != len(set(page_sizes)):
        raise ValueError("page_sizes must be non-empty and unique")
    calibration_path = (
        trace.parent / "helix-demo" / "capture" / "coding-failure-capture.continuum.json"
    )
    if not calibration_path.is_file():
        raise ValueError(
            "COW analysis requires a vertical trace beside helix-demo/capture/"
            "coding-failure-capture.continuum.json"
        )
    calibration = calibrate_state(calibration_path)
    fanouts = (1, 2, 4, 8, 16, 32, 64, 128)
    projections = tuple(
        project_attention_cow(
            calibration,
            branch_fanout=fanout,
            prefix_tokens=8192,
            suffix_tokens=256,
            divergence_pattern=pattern,
            page_size_bytes=page_size,
        )
        for fanout in fanouts
        for pattern in DivergencePattern
        for page_size in page_sizes
    )
    winners: Counter[int] = Counter()
    for fanout in fanouts:
        for pattern in DivergencePattern:
            candidates = [
                projection
                for projection in projections
                if projection.branch_fanout == fanout and projection.divergence_pattern is pattern
            ]
            minimum = min(item.physical_allocated_bytes for item in candidates)
            for item in candidates:
                if item.physical_allocated_bytes == minimum:
                    winners[item.page_size_bytes] += 1
    result: dict[str, object] = {
        "schema_version": "sloforge.branchfabric.cow-study/v1",
        "source_experiment": "controlled-cow-page-sweep",
        "seed": seed,
        "workload_evidence_class": "SYNTHETIC",
        "timing_measurement_class": "NOT_MEASURED_ANALYTICAL_PROJECTION",
        "distribution_claim": False,
        "trace_reference": trace.as_posix(),
        "trace_sha256": _sha256(trace),
        "calibration": calibration.model_dump(mode="json"),
        "state_composition": [
            project_state_composition(calibration, tokens=tokens).model_dump(mode="json")
            for tokens in (1024, 4096, 8192, 16384, 32768, 65536, 131072)
        ],
        "projection_count": len(projections),
        "projections": [item.model_dump(mode="json") for item in projections],
        "allocation_minimizing_page_size_wins": dict(sorted(winners.items())),
        "recommendation": {
            "allocation_only": min(winners, key=lambda size: (-winners[size], size)),
            "universal_page_size_supported": False,
            "reason": (
                "metadata cost and real GPU page-fault latency are unmeasured; allocation-only "
                "projections cannot establish a universal hardware page size"
            ),
        },
    }
    target = _write_json(_output_file(output, "cow-analysis.json"), result, replace=replace)
    return {"output": target.as_posix(), "artifact_sha256": _sha256(target), **result}


_AMDAHL_OPERATION_MAP = {
    StateOperationType.STATE_COW: CandidatePrimitive.COW_HANDLING,
    StateOperationType.STATE_ALLOC: CandidatePrimitive.ALLOCATION,
    StateOperationType.STATE_DELTA: CandidatePrimitive.DELTA_EXTRACTION,
    StateOperationType.STATE_CHECKSUM: CandidatePrimitive.CHECKSUM,
    StateOperationType.STATE_HASH: CandidatePrimitive.HASH,
    StateOperationType.STATE_RESHARD: CandidatePrimitive.RESHARD,
    StateOperationType.STATE_REPACK: CandidatePrimitive.REPACK,
    StateOperationType.STATE_QUANTIZE: CandidatePrimitive.QUANTIZATION,
    StateOperationType.STATE_COMPRESS: CandidatePrimitive.COMPRESSION,
    StateOperationType.STATE_SEND: CandidatePrimitive.TRANSFER,
    StateOperationType.STATE_MULTICAST: CandidatePrimitive.MULTICAST,
    StateOperationType.STATE_COMMIT: CandidatePrimitive.TRANSACTION_COMMIT,
    StateOperationType.STATE_RECLAIM: CandidatePrimitive.RECLAMATION,
    StateOperationType.STATE_FREE: CandidatePrimitive.RECLAMATION,
}


def _deduplicated_duration(events: list[StateOperationEventV1]) -> int:
    spans: dict[str, int] = {}
    unshared = 0
    for event in events:
        span = event.attributes.get("timing_span_id")
        if isinstance(span, str):
            spans[span] = max(spans.get(span, 0), event.operation_latency_ns)
        else:
            unshared += event.operation_latency_ns
    return unshared + sum(spans.values())


def analyze_run_amdahl(*, run: Path, output: Path, replace: bool) -> dict[str, object]:
    """Calculate non-additive Amdahl bounds from direct measured operation spans."""

    load_characterization_run(run)
    samples: list[AmdahlTimingSample] = []
    for trace in sorted(run.rglob("state-operation-trace-v1.jsonl")):
        state_events = [
            event
            for event in _load_trace(trace, max_events=MAX_WORKFLOW_EVENTS)
            if isinstance(event, StateOperationEventV1)
        ]
        if not state_events:
            continue
        is_continuum = "continuum-trace" in trace.parts
        objective = (
            EndToEndObjective.MIGRATION_LATENCY
            if is_continuum
            else EndToEndObjective.FULL_HELIX_TRANSACTION
        )
        total = max(
            event.normalized_timestamp_ns + event.duration_ns for event in state_events
        ) - min(event.normalized_timestamp_ns for event in state_events)
        total = max(1, total)
        by_primitive: dict[CandidatePrimitive, list[StateOperationEventV1]] = defaultdict(list)
        for event in state_events:
            primitive = _AMDAHL_OPERATION_MAP.get(event.operation_type)
            if primitive is None:
                continue
            if event.attributes.get("timing_scope") == "combined_model_and_environment_group_fork":
                continue
            by_primitive[primitive].append(event)
        for primitive, primitive_events in sorted(
            by_primitive.items(), key=lambda item: item[0].value
        ):
            duration = min(total, _deduplicated_duration(primitive_events))
            timing_classes = {event.timing_measurement_class for event in primitive_events}
            timing_class = (
                timing_classes.pop()
                if len(timing_classes) == 1
                else TimingMeasurementClass.SIMULATED_HARDWARE
            )
            samples.append(
                AmdahlTimingSample(
                    sample_id=f"{_sha256(trace)[:16]}-{primitive.value}",
                    experiment_id=trace.parent.name,
                    artifact_reference=trace.as_posix(),
                    provenance=WorkloadProvenance.SYNTHETIC,
                    timing_measurement_class=timing_class,
                    objective=objective,
                    primitive=primitive,
                    total_duration_ns=total,
                    primitive_exclusive_duration_ns=duration,
                    operation_count=len(primitive_events),
                )
            )
    if not samples:
        raise ValueError("run contains no state operation spans usable for Amdahl analysis")
    report = analyze_amdahl(tuple(samples))
    payload = report.model_dump(mode="json")
    payload["interpretation"] = (
        "Each primitive bound is independent and non-additive. Direct operation spans are treated "
        "as removable work; combined fork spans and nested duplicate timing spans are excluded."
    )
    target = _write_json(_output_file(output, "amdahl-analysis.json"), payload, replace=replace)
    return {"output": target.as_posix(), "artifact_sha256": _sha256(target), **payload}


@dataclass(frozen=True)
class _BoundArtifact:
    reference: str
    path: Path
    sha256: str
    trace_format: Literal["jsonl", "parquet", "perfetto", "manifest"] | None = None
    event_count: int | None = None

    def binding(self) -> ArtifactBinding:
        return ArtifactBinding(
            artifact_reference=self.reference,
            file_path=self.path.as_posix(),
            expected_sha256=self.sha256,
            trace_format=self.trace_format,
            event_count=self.event_count,
        )


def _artifact_reference(run: Path, path: Path) -> str:
    try:
        return path.relative_to(run).as_posix()
    except ValueError:
        return path.as_posix()


def _bind_artifact(
    run: Path,
    path: Path,
    *,
    trace_format: Literal["jsonl", "parquet", "perfetto", "manifest"] | None = None,
    event_count: int | None = None,
) -> _BoundArtifact:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"requirements evidence must be a regular file: {path}")
    return _BoundArtifact(
        reference=_artifact_reference(run, path),
        path=path,
        sha256=_sha256(path),
        trace_format=trace_format,
        event_count=event_count,
    )


def _successful_stage_attempt(run: Path, stage: str) -> Path:
    candidates: list[Path] = []
    for attempt in sorted((run / "attempts" / stage).glob("attempt-*")):
        worker_result = attempt / "worker-result.json"
        if not worker_result.is_file():
            continue
        document = json.loads(worker_result.read_text(encoding="utf-8"))
        if isinstance(document, dict) and document.get("status") == "SUCCEEDED":
            candidates.append(attempt)
    if not candidates:
        raise ValueError(f"completed run has no successful {stage} attempt")
    return candidates[-1]


def _verified_trace_streams(
    run: Path, trace_root: Path
) -> tuple[TraceManifestV1, tuple[tuple[_BoundArtifact, tuple[TraceEventV1, ...]], ...]]:
    manifest_path = trace_root / "trace-manifest-v1.json"
    manifest = TraceManifestV1.model_validate_json(manifest_path.read_bytes())
    by_name = {Path(item.uri).name: item for item in manifest.artifacts}
    streams: list[tuple[_BoundArtifact, tuple[TraceEventV1, ...]]] = []
    for name in ("branch-workload-trace-v1.jsonl", "state-operation-trace-v1.jsonl"):
        path = trace_root / name
        if not path.is_file():
            continue
        recorded = by_name.get(name)
        if recorded is None or recorded.format != "jsonl":
            raise ValueError(f"trace manifest does not bind canonical stream {name}")
        events = _load_trace(path, max_events=MAX_WORKFLOW_EVENTS)
        if (
            recorded.sha256 != _sha256(path)
            or recorded.byte_length != path.stat().st_size
            or recorded.event_count != len(events)
        ):
            raise ValueError(f"trace stream differs from its immutable manifest: {path}")
        streams.append(
            (
                _bind_artifact(
                    run,
                    path,
                    trace_format="jsonl",
                    event_count=len(events),
                ),
                events,
            )
        )
    if not streams:
        raise ValueError(f"trace manifest contains no canonical JSONL stream: {manifest_path}")
    return manifest, tuple(streams)


def _evidence(
    artifact: _BoundArtifact,
    *,
    source: str,
    sample_count: int,
    confidence: ConfidenceOrPercentile,
    evidence_class: EvidenceClass = EvidenceClass.SYNTHETIC,
) -> EvidenceReference:
    return EvidenceReference(
        source_experiment=source,
        sample_count=sample_count,
        evidence_class=evidence_class,
        confidence_or_percentile=confidence,
        artifact_reference=artifact.reference,
        artifact_sha256=artifact.sha256,
    )


def _available_number(
    artifact: _BoundArtifact,
    value: int | float,
    unit: RequirementUnit,
    *,
    source: str,
    sample_count: int,
    confidence: ConfidenceOrPercentile = ConfidenceOrPercentile.POINT_ESTIMATE,
    rationale: str,
    evidence_class: EvidenceClass = EvidenceClass.SYNTHETIC,
) -> NumericRequirement:
    return NumericRequirement(
        availability=Availability.AVAILABLE,
        value=value,
        unit=unit,
        evidence=_evidence(
            artifact,
            source=source,
            sample_count=sample_count,
            confidence=confidence,
            evidence_class=evidence_class,
        ),
        rationale=rationale,
    )


def _unknown_number(
    artifact: _BoundArtifact,
    unit: RequirementUnit,
    *,
    source: str,
    rationale: str,
) -> NumericRequirement:
    return NumericRequirement(
        availability=Availability.UNKNOWN,
        value=None,
        unit=unit,
        evidence=_evidence(
            artifact,
            source=source,
            sample_count=0,
            confidence=ConfidenceOrPercentile.COUNTER_ABSENT,
        ),
        rationale=rationale,
    )


def _unavailable_number(
    artifact: _BoundArtifact,
    unit: RequirementUnit,
    *,
    source: str,
    rationale: str,
) -> NumericRequirement:
    return NumericRequirement(
        availability=Availability.UNAVAILABLE,
        value=None,
        unit=unit,
        evidence=_evidence(
            artifact,
            source=source,
            sample_count=0,
            confidence=ConfidenceOrPercentile.CAPABILITY_UNAVAILABLE,
            evidence_class=EvidenceClass.HARDWARE_BACKED_REAL,
        ),
        rationale=rationale,
    )


def _available_distribution(
    artifact: _BoundArtifact,
    values: Sequence[int | float],
    unit: RequirementUnit,
    *,
    source: str,
    rationale: str,
    evidence_class: EvidenceClass = EvidenceClass.SYNTHETIC,
) -> DistributionRequirement:
    if not values:
        raise ValueError(f"available distribution {source} requires observations")
    metrics = (
        (0.50, ConfidenceOrPercentile.P50),
        (0.95, ConfidenceOrPercentile.P95),
        (0.99, ConfidenceOrPercentile.P99),
    )
    numbers: list[NumericRequirement] = []
    for probability, confidence in metrics:
        value = _percentile(values, probability)
        if value is None:
            raise ValueError(f"available distribution {source} has no percentile value")
        numbers.append(
            _available_number(
                artifact,
                value,
                unit,
                source=source,
                sample_count=len(values),
                confidence=confidence,
                rationale=rationale,
                evidence_class=evidence_class,
            )
        )
    maximum = _available_number(
        artifact,
        float(max(values)),
        unit,
        source=source,
        sample_count=len(values),
        confidence=ConfidenceOrPercentile.MAXIMUM,
        rationale=rationale,
        evidence_class=evidence_class,
    )
    return DistributionRequirement(
        p50=numbers[0],
        p95=numbers[1],
        p99=numbers[2],
        maximum=maximum,
    )


def _unknown_distribution(
    artifact: _BoundArtifact,
    unit: RequirementUnit,
    *,
    source: str,
    rationale: str,
) -> DistributionRequirement:
    return DistributionRequirement(
        p50=_unknown_number(artifact, unit, source=source, rationale=rationale),
        p95=_unknown_number(artifact, unit, source=source, rationale=rationale),
        p99=_unknown_number(artifact, unit, source=source, rationale=rationale),
        maximum=_unknown_number(artifact, unit, source=source, rationale=rationale),
    )


def _unknown_enum(
    artifact: _BoundArtifact,
    *,
    source: str,
    rationale: str,
) -> EnumRequirement:
    return EnumRequirement(
        availability=Availability.UNKNOWN,
        value=None,
        evidence=_evidence(
            artifact,
            source=source,
            sample_count=0,
            confidence=ConfidenceOrPercentile.COUNTER_ABSENT,
        ),
        rationale=rationale,
    )


def _trace_span_seconds(events: Sequence[TraceEventV1]) -> float:
    start = min(event.normalized_timestamp_ns for event in events)
    finish = max(event.normalized_timestamp_ns + event.duration_ns for event in events)
    return max(1, finish - start) / 1_000_000_000


def _compile_requirements_from_run(
    run: Path,
) -> tuple[BranchFabricRequirementsV1, CharacterizationRunManifest]:
    manifest = load_characterization_run(run)
    result = characterization_run_result(run)
    if result.run_state is not RunState.COMPLETE:
        raise ValueError(
            "requirements require a completed characterization run without failed stages"
        )

    vertical_attempt = _successful_stage_attempt(run, "vertical_trace")
    continuum_attempt = _successful_stage_attempt(run, "continuum_state")
    vertical_root = vertical_attempt / "vertical-trace"
    continuum_root = continuum_attempt / "continuum-trace"
    vertical_manifest, vertical_streams = _verified_trace_streams(run, vertical_root)
    _continuum_manifest, continuum_streams = _verified_trace_streams(run, continuum_root)
    streams = (*vertical_streams, *continuum_streams)
    bindings: dict[str, _BoundArtifact] = {item.reference: item for item, _events in streams}

    def bind(path: Path) -> _BoundArtifact:
        artifact = _bind_artifact(run, path)
        bindings[artifact.reference] = artifact
        return artifact

    run_artifact = bind(run / "run-manifest.json")
    matrix_artifact = bind(run / manifest.preserved_matrix_reference)
    capability_path = (
        run / manifest.capability_evidence.preserved_reference
        if manifest.capability_evidence.preserved_reference is not None
        else run / "run-manifest.json"
    )
    capability_artifact = bindings.get(_artifact_reference(run, capability_path)) or bind(
        capability_path
    )
    sharing_artifact = bind(vertical_root / "sharing-analysis.json")
    sharing = json.loads(sharing_artifact.path.read_text(encoding="utf-8"))

    metadata_path = _successful_stage_attempt(run, "metadata") / "metadata-study.json"
    metadata_artifact = bind(metadata_path)
    metadata_document = json.loads(metadata_path.read_text(encoding="utf-8"))

    branch_artifact, branch_trace = next(
        (artifact, events)
        for artifact, events in streams
        if any(isinstance(event, BranchWorkloadEventV1) for event in events)
    )
    branches = tuple(event for event in branch_trace if isinstance(event, BranchWorkloadEventV1))
    state_streams = tuple(
        (
            artifact,
            tuple(event for event in events if isinstance(event, StateOperationEventV1)),
        )
        for artifact, events in streams
        if any(isinstance(event, StateOperationEventV1) for event in events)
    )
    all_states = tuple(event for _artifact, events in state_streams for event in events)

    groups: dict[str, set[str]] = defaultdict(set)
    for event in branches:
        if event.branch_group_id and event.branch_id:
            groups[event.branch_group_id].add(event.branch_id)
    fanouts = tuple(len(branch_ids) for branch_ids in groups.values())
    if not fanouts:
        raise ValueError("vertical trace has no measured branch group")
    fork_by_branch = {
        event.branch_id: event
        for event in branches
        if event.operation_type is BranchOperationType.BRANCH_FORK and event.branch_id
    }
    terminal_by_branch = {
        event.branch_id: event
        for event in branches
        if event.operation_type
        in {
            BranchOperationType.BRANCH_COMPLETE,
            BranchOperationType.BRANCH_PRUNE,
            BranchOperationType.BRANCH_ABORT,
        }
        and event.branch_id
    }
    lifetimes_ms = tuple(
        (terminal.normalized_timestamp_ns + terminal.duration_ns - fork.normalized_timestamp_ns)
        / 1_000_000
        for branch_id, fork in fork_by_branch.items()
        if (terminal := terminal_by_branch.get(branch_id)) is not None
    )
    fork_states = tuple(
        event for event in all_states if event.operation_type is StateOperationType.STATE_FORK
    )
    shared_root_bytes = tuple(
        int(value)
        for event in fork_states
        if isinstance((value := event.attributes.get("shared_logical_bytes")), int)
    )
    private_suffix_bytes = tuple(
        int(value)
        for event in fork_states
        if isinstance((value := event.attributes.get("private_logical_bytes")), int)
    )
    if not shared_root_bytes or not private_suffix_bytes or not lifetimes_ms:
        raise ValueError("vertical trace lacks sharing or branch-lifetime observations")

    unknown_reason = (
        "the completed canonical trace does not contain the counter or sensitivity experiment "
        "needed to derive this requirement"
    )
    workload = WorkloadRequirement(
        workload_class=WorkloadClass.CODING_AGENT,
        evidence=_evidence(
            branch_artifact,
            source="helix-coding-agent-vertical-trace",
            sample_count=len(branches),
            confidence=ConfidenceOrPercentile.POINT_ESTIMATE,
        ),
        experiment_count=_available_number(
            branch_artifact,
            1,
            RequirementUnit.COUNT,
            source="helix-coding-agent-vertical-trace",
            sample_count=len(branches),
            rationale="one complete deterministic coding-agent vertical trace was observed",
        ),
        branch_fanout=_available_distribution(
            branch_artifact,
            fanouts,
            RequirementUnit.COUNT,
            source="helix-coding-agent-branch-groups",
            rationale="unique branch IDs were counted within each observed branch group",
        ),
        prefix_tokens=_unknown_distribution(
            branch_artifact,
            RequirementUnit.TOKENS,
            source="helix-coding-agent-vertical-trace",
            rationale="the fixture trace does not expose a production prefix-token distribution",
        ),
        suffix_tokens=_unknown_distribution(
            branch_artifact,
            RequirementUnit.TOKENS,
            source="helix-coding-agent-vertical-trace",
            rationale="the fixture trace does not expose a production suffix-token distribution",
        ),
    )
    state = StateRequirements(
        branch_fanout=workload.branch_fanout,
        shared_root_bytes=_available_distribution(
            next(
                artifact
                for artifact, events in state_streams
                if any(e in events for e in fork_states)
            ),
            shared_root_bytes,
            RequirementUnit.BYTES,
            source="state-fork-shared-logical-bytes",
            rationale="STATE_FORK records directly report shared logical bytes",
        ),
        private_suffix_bytes=_available_distribution(
            next(
                artifact
                for artifact, events in state_streams
                if any(e in events for e in fork_states)
            ),
            private_suffix_bytes,
            RequirementUnit.BYTES,
            source="state-fork-private-logical-bytes",
            rationale="STATE_FORK records directly report private logical bytes",
        ),
        divergence_rate=_unknown_distribution(
            branch_artifact,
            RequirementUnit.RATIO,
            source="trajectory-divergence",
            rationale=(
                "private-byte share is not substituted for token, action, page, or environment "
                "divergence rate"
            ),
        ),
        branch_lifetime_ms=_available_distribution(
            branch_artifact,
            lifetimes_ms,
            RequirementUnit.MILLISECONDS,
            source="branch-fork-to-terminal",
            rationale="each lifetime is the monotonic fork-to-prune/complete/abort interval",
        ),
    )

    amplification = float(sharing["model_state"]["physical_amplification"])
    cow = CowRequirements(
        recommended_page_sizes=(
            PageSizeRequirement(
                state_segment=StateSegment.KV,
                recommended_page_size_bytes=_unknown_number(
                    run_artifact,
                    RequirementUnit.BYTES,
                    source="cow-page-size-requirement",
                    rationale=(
                        "no real GPU or operating-system page-fault sweep exists in this run; "
                        "a universal page size is not selected"
                    ),
                ),
                physical_amplification=_available_distribution(
                    sharing_artifact,
                    (amplification,),
                    RequirementUnit.RATIO,
                    source="vertical-model-sharing-analysis",
                    rationale="physical allocated bytes divided by logical unique bytes",
                ),
                cow_fault_rate=_unknown_distribution(
                    sharing_artifact,
                    RequirementUnit.FAULTS_PER_SECOND,
                    source="cow-page-fault-rate",
                    rationale="logical whole-file COW writes are not hardware page faults",
                ),
            ),
        ),
        fault_rate=_unknown_distribution(
            sharing_artifact,
            RequirementUnit.FAULTS_PER_SECOND,
            source="cow-page-fault-rate",
            rationale="the reference environment reports no page-fault counter",
        ),
        amplification=_available_distribution(
            sharing_artifact,
            (amplification,),
            RequirementUnit.RATIO,
            source="vertical-model-sharing-analysis",
            rationale="measured content-addressed model-state physical amplification",
        ),
    )

    summaries = metadata_document.get("summaries", [])
    current_throughput = tuple(
        float(item["throughput_ops_per_second_median"])
        for item in summaries
        if isinstance(item, dict) and item.get("implementation") == "current_software"
    )
    metadata_rate = (
        _available_distribution(
            metadata_artifact,
            current_throughput,
            RequirementUnit.OPERATIONS_PER_SECOND,
            source="metadata-current-software-summaries",
            rationale=(
                "distribution across measured current-software operation/thread summary medians"
            ),
            evidence_class=EvidenceClass.SYNTHETIC,
        )
        if current_throughput
        else _unknown_distribution(
            metadata_artifact,
            RequirementUnit.OPERATIONS_PER_SECOND,
            source="metadata-current-software-summaries",
            rationale="metadata artifact contains no current-software throughput summary",
        )
    )
    metadata = MetadataRequirements(
        operations_per_second=metadata_rate,
        queue_depth=_unknown_distribution(
            metadata_artifact,
            RequirementUnit.COUNT,
            source="metadata-outstanding-queue-depth",
            rationale="thread count is not substituted for outstanding metadata queue depth",
        ),
        working_set_bytes=_unknown_distribution(
            metadata_artifact,
            RequirementUnit.BYTES,
            source="metadata-resident-working-set",
            rationale="tracemalloc peak deltas do not establish the resident metadata working set",
        ),
    )

    transform_types = {
        StateOperationType.STATE_RESHARD,
        StateOperationType.STATE_TRANSPOSE,
        StateOperationType.STATE_REPACK,
        StateOperationType.STATE_QUANTIZE,
        StateOperationType.STATE_DEQUANTIZE,
        StateOperationType.STATE_COMPRESS,
        StateOperationType.STATE_DECOMPRESS,
    }
    transform_sequences: list[TransformSequenceRequirement] = []
    for artifact, events in state_streams:
        grouped = {
            operation: tuple(event for event in events if event.operation_type is operation)
            for operation in transform_types
        }
        for operation, observed in sorted(grouped.items(), key=lambda item: item[0].value):
            if not observed:
                continue
            transform_sequences.append(
                TransformSequenceRequirement(
                    operations=(operation,),
                    evidence=_evidence(
                        artifact,
                        source=f"observed-{operation.value.lower()}",
                        sample_count=len(observed),
                        confidence=ConfidenceOrPercentile.POINT_ESTIMATE,
                    ),
                    frequency=_available_number(
                        artifact,
                        len(observed),
                        RequirementUnit.COUNT,
                        source=f"observed-{operation.value.lower()}",
                        sample_count=len(observed),
                        rationale="count of explicit canonical state-operation events",
                    ),
                    bytes_processed=_available_number(
                        artifact,
                        sum(event.bytes for event in observed),
                        RequirementUnit.BYTES,
                        source=f"observed-{operation.value.lower()}",
                        sample_count=len(observed),
                        rationale="sum of bytes recorded by explicit transform events",
                    ),
                    latency=_available_number(
                        artifact,
                        sum(event.operation_latency_ns for event in observed),
                        RequirementUnit.NANOSECONDS,
                        source=f"observed-{operation.value.lower()}",
                        sample_count=len(observed),
                        rationale="sum of directly recorded host operation latencies",
                    ),
                    temporary_memory_bytes=_unknown_number(
                        artifact,
                        RequirementUnit.BYTES,
                        source=f"observed-{operation.value.lower()}",
                        rationale="StateOperationTrace v1 has no temporary-allocation counter",
                    ),
                )
            )
    transform = TransformRequirements(
        top_sequences=tuple(transform_sequences),
        bandwidth_requirement=_unknown_distribution(
            run_artifact,
            RequirementUnit.BYTES_PER_SECOND,
            source="transform-bandwidth-target",
            rationale="observed fixture throughput is not a target bandwidth requirement",
        ),
        temporary_memory=_unknown_distribution(
            run_artifact,
            RequirementUnit.BYTES,
            source="transform-temporary-memory",
            rationale=unknown_reason,
        ),
    )

    network_artifact, network_events = next(
        (
            artifact,
            tuple(
                event for event in events if event.operation_type is StateOperationType.STATE_SEND
            ),
        )
        for artifact, events in state_streams
        if any(event.operation_type is StateOperationType.STATE_SEND for event in events)
    )
    network = NetworkRequirements(
        unicast_bytes=_available_distribution(
            network_artifact,
            tuple(event.bytes for event in network_events),
            RequirementUnit.BYTES,
            source="continuum-state-send-events",
            rationale="payload bytes from explicit STATE_SEND events; transport classes remain labeled",
        ),
        multicast_opportunity_bytes=_unknown_distribution(
            network_artifact,
            RequirementUnit.BYTES,
            source="multicast-opportunity",
            rationale=(
                "fanout-one sends cannot establish a production multicast opportunity distribution"
            ),
        ),
        fanout=_available_distribution(
            network_artifact,
            tuple(event.fanout for event in network_events),
            RequirementUnit.COUNT,
            source="continuum-state-send-events",
            rationale="fanout recorded by explicit STATE_SEND events",
        ),
        required_bandwidth=_unknown_distribution(
            capability_artifact,
            RequirementUnit.BYTES_PER_SECOND,
            source="network-bandwidth-requirement",
            rationale="no physical network rail was measured on the preserved host",
        ),
    )

    transaction_artifact, transaction_events = next(
        (artifact, events)
        for artifact, events in state_streams
        if any(event.operation_type is StateOperationType.STATE_COMMIT for event in events)
    )
    observation_seconds = _trace_span_seconds(transaction_events)
    commit_count = sum(
        event.operation_type is StateOperationType.STATE_COMMIT for event in transaction_events
    )
    abort_count = sum(
        event.operation_type is StateOperationType.STATE_ABORT for event in transaction_events
    )
    transactions = TransactionRequirements(
        commit_rate=_available_distribution(
            transaction_artifact,
            (commit_count / observation_seconds,),
            RequirementUnit.EVENTS_PER_SECOND,
            source="continuum-transaction-observation-window",
            rationale="STATE_COMMIT count divided by the canonical state-trace observation span",
        ),
        abort_rate=_available_distribution(
            transaction_artifact,
            (abort_count / observation_seconds,),
            RequirementUnit.EVENTS_PER_SECOND,
            source="continuum-transaction-observation-window",
            rationale="zero or more STATE_ABORT events divided by the same observation span",
        ),
        epoch_checks_per_second=_unknown_distribution(
            transaction_artifact,
            RequirementUnit.OPERATIONS_PER_SECOND,
            source="transaction-epoch-checks",
            rationale="epoch checks are not emitted as independent canonical operations",
        ),
    )

    amdahl_free_speedup: dict[tuple[str, CandidatePrimitive], tuple[float, int, str]] = {}
    for artifact, events in state_streams:
        total_duration = max(
            event.normalized_timestamp_ns + event.duration_ns for event in events
        ) - min(event.normalized_timestamp_ns for event in events)
        total_duration = max(1, total_duration)
        by_primitive: dict[CandidatePrimitive, list[StateOperationEventV1]] = defaultdict(list)
        for state_event in events:
            primitive = _AMDAHL_OPERATION_MAP.get(state_event.operation_type)
            if primitive is None:
                continue
            if (
                state_event.attributes.get("timing_scope")
                == "combined_model_and_environment_group_fork"
            ):
                continue
            by_primitive[primitive].append(state_event)
        for primitive, primitive_events in by_primitive.items():
            removable_duration = min(total_duration, _deduplicated_duration(primitive_events))
            remaining_duration = total_duration - removable_duration
            speedup = (
                float("inf") if remaining_duration == 0 else total_duration / remaining_duration
            )
            timing_classes = ",".join(
                sorted({event.timing_measurement_class.value for event in primitive_events})
            )
            amdahl_free_speedup[(artifact.reference, primitive)] = (
                speedup,
                len(primitive_events),
                timing_classes,
            )

    def unresolved_isa(
        operation: StateOperationType,
        artifact: _BoundArtifact,
        observed: tuple[StateOperationEventV1, ...],
    ) -> IsaOperationRecommendation:
        span_events = next(events for bound, events in state_streams if bound == artifact)
        rate = len(observed) / _trace_span_seconds(span_events)
        state_types = tuple(
            sorted({event.state_segment for event in observed}, key=lambda item: item.value)
        ) or (StateSegment.UNKNOWN,)
        primitive = _AMDAHL_OPERATION_MAP.get(operation)
        amdahl = (
            amdahl_free_speedup.get((artifact.reference, primitive))
            if primitive is not None
            else None
        )
        if amdahl is not None and math.isfinite(amdahl[0]):
            assert primitive is not None
            expected_speedup = _available_number(
                artifact,
                amdahl[0],
                RequirementUnit.RATIO,
                source=f"amdahl-free-{primitive.value}",
                sample_count=amdahl[1],
                rationale=(
                    "independent, non-additive maximum speedup if the directly recorded primitive "
                    f"span becomes free; timing classes: {amdahl[2]}"
                ),
            )
        else:
            expected_speedup = _unknown_number(
                artifact,
                RequirementUnit.RATIO,
                source=f"isa-{operation.value.lower()}",
                rationale="no finite exclusive end-to-end Amdahl bound is available",
            )
        return IsaOperationRecommendation(
            operation=operation,
            classification=IsaClassificationRequirement(
                availability=Availability.UNKNOWN,
                value=None,
                evidence=_evidence(
                    artifact,
                    source=f"isa-{operation.value.lower()}",
                    sample_count=0,
                    confidence=ConfidenceOrPercentile.COUNTER_ABSENT,
                ),
                rationale=(
                    "operation frequency is observed, but hardware target latency, strongest "
                    "software comparison, and end-to-end leverage are incomplete"
                ),
            ),
            measured_frequency=_available_number(
                artifact,
                rate,
                RequirementUnit.EVENTS_PER_SECOND,
                source=f"isa-{operation.value.lower()}",
                sample_count=len(observed),
                rationale="event count divided by its canonical trace observation span",
            ),
            size=_available_distribution(
                artifact,
                tuple(event.bytes for event in observed),
                RequirementUnit.BYTES,
                source=f"isa-{operation.value.lower()}",
                rationale="bytes recorded by explicit canonical operation events",
            ),
            latency_target=_unknown_number(
                artifact,
                RequirementUnit.NANOSECONDS,
                source=f"isa-{operation.value.lower()}",
                rationale="an observed latency is not substituted for a hardware latency target",
            ),
            throughput_target=_unknown_number(
                artifact,
                RequirementUnit.OPERATIONS_PER_SECOND,
                source=f"isa-{operation.value.lower()}",
                rationale="no sensitivity experiment establishes a throughput target",
            ),
            concurrency=_available_distribution(
                artifact,
                tuple(event.concurrency for event in observed),
                RequirementUnit.COUNT,
                source=f"isa-{operation.value.lower()}",
                rationale="concurrency recorded by canonical operation events",
            ),
            fanout=_available_distribution(
                artifact,
                tuple(event.fanout for event in observed),
                RequirementUnit.COUNT,
                source=f"isa-{operation.value.lower()}",
                rationale="fanout recorded by canonical operation events",
            ),
            state_types=state_types,
            consistency=_unknown_enum(
                artifact,
                source=f"isa-{operation.value.lower()}",
                rationale="future hardware consistency semantics were not exercised",
            ),
            failure_behavior=_unknown_enum(
                artifact,
                source=f"isa-{operation.value.lower()}",
                rationale="future hardware command failure behavior was not exercised",
            ),
            expected_end_to_end_speedup=expected_speedup,
            dependencies=(),
        )

    candidate_operations = (
        StateOperationType.STATE_ALLOC,
        StateOperationType.STATE_MAP,
        StateOperationType.STATE_PUBLISH,
        StateOperationType.STATE_FORK,
        StateOperationType.STATE_COW,
        StateOperationType.STATE_APPEND,
        StateOperationType.STATE_SNAPSHOT,
        StateOperationType.STATE_DELTA,
        StateOperationType.STATE_RESHARD,
        StateOperationType.STATE_REPACK,
        StateOperationType.STATE_QUANTIZE,
        StateOperationType.STATE_DEQUANTIZE,
        StateOperationType.STATE_COMPRESS,
        StateOperationType.STATE_DECOMPRESS,
        StateOperationType.STATE_HASH,
        StateOperationType.STATE_CHECKSUM,
        StateOperationType.STATE_ENCRYPT,
        StateOperationType.STATE_DECRYPT,
        StateOperationType.STATE_SEND,
        StateOperationType.STATE_MULTICAST,
        StateOperationType.STATE_COMMIT,
        StateOperationType.STATE_ABORT,
        StateOperationType.STATE_RECLAIM,
        StateOperationType.STATE_FREE,
    )
    absent_not_justified = {
        StateOperationType.STATE_QUANTIZE,
        StateOperationType.STATE_DEQUANTIZE,
        StateOperationType.STATE_COMPRESS,
        StateOperationType.STATE_DECOMPRESS,
        StateOperationType.STATE_ENCRYPT,
        StateOperationType.STATE_DECRYPT,
        StateOperationType.STATE_MULTICAST,
    }
    unresolved: list[IsaOperationRecommendation] = []
    not_justified: list[IsaOperationRecommendation] = []
    fallback_artifact, fallback_states = max(state_streams, key=lambda item: len(item[1]))
    for operation in candidate_operations:
        selected: tuple[_BoundArtifact, tuple[StateOperationEventV1, ...]] | None = None
        for artifact, events in state_streams:
            observed = tuple(event for event in events if event.operation_type is operation)
            if observed and (selected is None or len(observed) > len(selected[1])):
                selected = (artifact, observed)
        if selected is not None:
            unresolved.append(unresolved_isa(operation, *selected))
            continue
        if operation not in absent_not_justified:
            continue
        span = _trace_span_seconds(fallback_states)
        classification_evidence = _evidence(
            fallback_artifact,
            source=f"isa-{operation.value.lower()}-absence",
            sample_count=len(fallback_states),
            confidence=ConfidenceOrPercentile.POINT_ESTIMATE,
        )
        not_justified.append(
            IsaOperationRecommendation(
                operation=operation,
                classification=IsaClassificationRequirement(
                    availability=Availability.AVAILABLE,
                    value=IsaClassification.NOT_JUSTIFIED,
                    evidence=classification_evidence,
                    rationale=(
                        "no explicit operation occurred in the bounded state-lifecycle corpus; "
                        "this is a current-workload negative finding, not a universal claim"
                    ),
                ),
                measured_frequency=_available_number(
                    fallback_artifact,
                    0.0 / span,
                    RequirementUnit.EVENTS_PER_SECOND,
                    source=f"isa-{operation.value.lower()}-absence",
                    sample_count=len(fallback_states),
                    rationale="zero explicit events in the complete bounded state trace",
                ),
                size=_unknown_distribution(
                    fallback_artifact,
                    RequirementUnit.BYTES,
                    source=f"isa-{operation.value.lower()}-absence",
                    rationale="no event exists from which to derive an operation size",
                ),
                latency_target=_unknown_number(
                    fallback_artifact,
                    RequirementUnit.NANOSECONDS,
                    source=f"isa-{operation.value.lower()}-absence",
                    rationale=unknown_reason,
                ),
                throughput_target=_unknown_number(
                    fallback_artifact,
                    RequirementUnit.OPERATIONS_PER_SECOND,
                    source=f"isa-{operation.value.lower()}-absence",
                    rationale=unknown_reason,
                ),
                concurrency=_unknown_distribution(
                    fallback_artifact,
                    RequirementUnit.COUNT,
                    source=f"isa-{operation.value.lower()}-absence",
                    rationale="no event exists from which to derive concurrency",
                ),
                fanout=_unknown_distribution(
                    fallback_artifact,
                    RequirementUnit.COUNT,
                    source=f"isa-{operation.value.lower()}-absence",
                    rationale="no event exists from which to derive fanout",
                ),
                state_types=(StateSegment.UNKNOWN,),
                consistency=_unknown_enum(
                    fallback_artifact,
                    source=f"isa-{operation.value.lower()}-absence",
                    rationale=unknown_reason,
                ),
                failure_behavior=_unknown_enum(
                    fallback_artifact,
                    source=f"isa-{operation.value.lower()}-absence",
                    rationale=unknown_reason,
                ),
                expected_end_to_end_speedup=_unknown_number(
                    fallback_artifact,
                    RequirementUnit.RATIO,
                    source=f"isa-{operation.value.lower()}-absence",
                    rationale="an unobserved operation has no measured acceleration leverage",
                ),
                dependencies=(),
            )
        )

    matrix = load_matrix(run / manifest.preserved_matrix_reference)
    cases = expand_matrix(matrix)
    memory_requirements: list[MemoryCapacityRequirement] = []
    for branch_count in (8, 16, 32, 64, 128):
        case_count = sum(case.spec.branch_fanout == branch_count for case in cases)
        memory_requirements.append(
            MemoryCapacityRequirement(
                workload_class=WorkloadClass.CODING_AGENT,
                branch_count=_available_number(
                    matrix_artifact,
                    branch_count,
                    RequirementUnit.COUNT,
                    source="controlled-characterization-matrix",
                    sample_count=max(1, case_count),
                    rationale=(
                        "representative branch count is a declared controlled sweep input, "
                        "not an observed production percentile"
                    ),
                ),
                low_latency_metadata_bytes=_unknown_number(
                    matrix_artifact,
                    RequirementUnit.BYTES,
                    source=f"memory-capacity-{branch_count}-branches",
                    rationale="resident metadata capacity was not measured at this branch count",
                ),
                hbm_bytes=(
                    _unknown_number(
                        capability_artifact,
                        RequirementUnit.BYTES,
                        source=f"memory-capacity-{branch_count}-branches",
                        rationale="compatible GPU exists but HBM capacity demand was not measured",
                    )
                    if manifest.capability_evidence.compatible_gpu_available
                    else _unavailable_number(
                        capability_artifact,
                        RequirementUnit.BYTES,
                        source=f"memory-capacity-{branch_count}-branches",
                        rationale="the preserved capability evidence has no compatible GPU HBM",
                    )
                ),
                ddr_bytes=_unknown_number(
                    matrix_artifact,
                    RequirementUnit.BYTES,
                    source=f"memory-capacity-{branch_count}-branches",
                    rationale="DDR capacity was not measured at this branch count",
                ),
                cxl_bytes=_unknown_number(
                    capability_artifact,
                    RequirementUnit.BYTES,
                    source=f"memory-capacity-{branch_count}-branches",
                    rationale="no CXL device or CXL capacity sensitivity experiment was recorded",
                ),
                host_or_storage_bytes=_unknown_number(
                    matrix_artifact,
                    RequirementUnit.BYTES,
                    source=f"memory-capacity-{branch_count}-branches",
                    rationale=(
                        "the matrix declares this fanout but does not execute or extrapolate a "
                        "capacity measurement"
                    ),
                ),
            )
        )

    queue_requirements: list[QueueRequirement] = []
    latency_targets: list[LatencyRequirement] = []
    for operation in sorted(
        {event.operation_type for event in all_states}, key=lambda item: item.value
    ):
        artifact, observed = next(
            (
                bound,
                tuple(event for event in events if event.operation_type is operation),
            )
            for bound, events in state_streams
            if any(event.operation_type is operation for event in events)
        )
        queue_requirements.append(
            QueueRequirement(
                operation=operation,
                minimum_depth=_available_number(
                    artifact,
                    max(event.concurrency for event in observed),
                    RequirementUnit.COUNT,
                    source=f"observed-concurrency-{operation.value.lower()}",
                    sample_count=len(observed),
                    confidence=ConfidenceOrPercentile.MAXIMUM,
                    rationale="minimum depth needed not to reject the maximum observed concurrency",
                ),
                recommended_depth=_unknown_number(
                    artifact,
                    RequirementUnit.COUNT,
                    source=f"queue-depth-{operation.value.lower()}",
                    rationale="no queue-depth sensitivity experiment establishes headroom",
                ),
                pathological_maximum=_unknown_number(
                    artifact,
                    RequirementUnit.COUNT,
                    source=f"queue-depth-{operation.value.lower()}",
                    rationale="the bounded trace does not establish a pathological maximum",
                ),
                backpressure_policy=_unknown_enum(
                    artifact,
                    source=f"queue-depth-{operation.value.lower()}",
                    rationale="hardware queue backpressure policy was not exercised",
                ),
            )
        )
        latency_targets.append(
            LatencyRequirement(
                operation=operation,
                target_p50=_unknown_number(
                    artifact,
                    RequirementUnit.NANOSECONDS,
                    source=f"latency-target-{operation.value.lower()}",
                    rationale="observed latency is not substituted for a target",
                ),
                target_p99=_unknown_number(
                    artifact,
                    RequirementUnit.NANOSECONDS,
                    source=f"latency-target-{operation.value.lower()}",
                    rationale="insufficient repetitions and no SLO sensitivity experiment",
                ),
                maximum_tolerable=_unknown_number(
                    artifact,
                    RequirementUnit.NANOSECONDS,
                    source=f"latency-target-{operation.value.lower()}",
                    rationale="maximum tolerable latency was not measured",
                ),
            )
        )

    bandwidth_targets: list[BandwidthRequirement] = []
    for interface in InterfaceKind:
        gpu_interface = interface in {
            InterfaceKind.GPU_TO_BRANCHFABRIC,
            InterfaceKind.BRANCHFABRIC_TO_GPU,
        }
        unavailable = gpu_interface and not manifest.capability_evidence.compatible_gpu_available
        make_number = _unavailable_number if unavailable else _unknown_number
        rationale = (
            "the preserved capability evidence has no compatible discrete GPU interface"
            if unavailable
            else "no end-to-end sensitivity experiment establishes this interface requirement"
        )
        kwargs = {
            "source": f"bandwidth-target-{interface.value}",
            "rationale": rationale,
        }
        bandwidth_targets.append(
            BandwidthRequirement(
                interface=interface,
                mean=make_number(
                    capability_artifact,
                    RequirementUnit.BYTES_PER_SECOND,
                    **kwargs,
                ),
                p95=make_number(
                    capability_artifact,
                    RequirementUnit.BYTES_PER_SECOND,
                    **kwargs,
                ),
                p99=make_number(
                    capability_artifact,
                    RequirementUnit.BYTES_PER_SECOND,
                    **kwargs,
                ),
                burst_peak=make_number(
                    capability_artifact,
                    RequirementUnit.BYTES_PER_SECOND,
                    **kwargs,
                ),
                burst_duration=make_number(
                    capability_artifact,
                    RequirementUnit.MILLISECONDS,
                    **kwargs,
                ),
            )
        )

    draft = RequirementsDraft(
        generated_at=manifest.created_at,
        limitations=(
            "The workload corpus is a deterministic synthetic CPU coding-agent fixture, not a production distribution.",
            "Model-state byte sizes are simulated; environment and host-operation timings are measured on the recorded CPU host.",
            "No compatible NVIDIA GPU, HBM, PCIe GPU link, multi-GPU fabric, or multi-node network was measured.",
            "Unknown values are not replaced with matrix sweep points, analytical projections, zeroes, or targets.",
            "NOT_JUSTIFIED ISA labels mean absent from this bounded corpus and must be revisited with broader real workloads.",
        ),
        workloads=(workload,),
        state=state,
        cow=cow,
        metadata=metadata,
        transform=transform,
        network=network,
        transactions=transactions,
        recommended_isa=(),
        software_only_operations=(),
        not_justified_operations=tuple(not_justified),
        unresolved_isa_operations=tuple(unresolved),
        memory_requirements=tuple(memory_requirements),
        queue_requirements=tuple(queue_requirements),
        latency_targets=tuple(latency_targets),
        bandwidth_targets=tuple(bandwidth_targets),
    )
    trace_artifact_list: list[TraceArtifactV1] = []
    for artifact, _events in streams:
        if artifact.trace_format is None or artifact.event_count is None:
            continue
        trace_artifact_list.append(
            TraceArtifactV1(
                format=artifact.trace_format,
                uri=artifact.reference,
                byte_length=artifact.path.stat().st_size,
                sha256=artifact.sha256,
                event_count=artifact.event_count,
            )
        )
    trace_artifacts = tuple(trace_artifact_list)
    expected_corpus_hash = trace_corpus_hash(manifest.run_id, trace_artifacts)
    compilation = RequirementsCompilationInput(
        trace_id=manifest.run_id,
        expected_trace_corpus_hash=expected_corpus_hash,
        artifact_bindings=tuple(
            item.binding() for item in sorted(bindings.values(), key=lambda x: x.reference)
        ),
        trace_artifact_references=tuple(item.uri for item in trace_artifacts),
        draft=draft,
    )
    requirements = compile_requirements(compilation)
    if requirements.trace_corpus_hash != expected_corpus_hash:
        raise ValueError("compiled requirements trace corpus identity changed unexpectedly")
    if requirements.trace_corpus_hash == vertical_manifest.trace_corpus_hash:
        # The combined characterization corpus intentionally differs from either individual
        # trace manifest.  Equality would indicate that Continuum evidence was omitted.
        raise ValueError("requirements corpus unexpectedly contains only the vertical trace")
    return requirements, manifest


def derive_requirements(*, run: Path, output: Path, replace: bool) -> object:
    """Compile a machine-readable requirement set from immutable completed-run evidence."""

    requirements, manifest = _compile_requirements_from_run(run)
    target = _output_file(output, "branchfabric_requirements.json")
    if target.exists() and not replace:
        raise FileExistsError(f"output already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    write_requirements(target, requirements, overwrite=replace)
    availability: Counter[str] = Counter()
    stack: list[Any] = [requirements.model_dump(mode="json")]
    while stack:
        value = stack.pop()
        if isinstance(value, dict):
            state_value = value.get("availability")
            if state_value in {item.value for item in Availability}:
                availability[str(state_value)] += 1
            stack.extend(value.values())
        elif isinstance(value, list):
            stack.extend(value)
    return {
        "schema_version": requirements.schema_version,
        "output": target.as_posix(),
        "artifact_sha256": _sha256(target),
        "run_id": manifest.run_id,
        "trace_corpus_hash": requirements.trace_corpus_hash,
        "verified_artifact_count": len(requirements.verified_artifacts),
        "availability_counts": dict(sorted(availability.items())),
        "recommended_isa_count": len(requirements.recommended_isa),
        "software_only_isa_count": len(requirements.software_only_operations),
        "not_justified_isa_count": len(requirements.not_justified_operations),
        "unresolved_isa_count": len(requirements.unresolved_isa_operations),
    }


def _metric_text(metric: NumericRequirement) -> str:
    if metric.availability is not Availability.AVAILABLE:
        return metric.availability.value
    return f"{metric.value:g} {metric.unit.value}" if metric.value is not None else "UNKNOWN"


def _distribution_text(distribution: DistributionRequirement) -> str:
    if distribution.p50.availability is not Availability.AVAILABLE:
        return distribution.p50.availability.value
    return (
        f"p50 {_metric_text(distribution.p50)}; p95 {_metric_text(distribution.p95)}; "
        f"p99 {_metric_text(distribution.p99)}; max {_metric_text(distribution.maximum)}"
    )


def _render_report(
    requirements: BranchFabricRequirementsV1, manifest: CharacterizationRunManifest
) -> str:
    capability = manifest.capability_evidence
    amdahl_rows = sorted(
        (
            (item.operation.value, item.expected_end_to_end_speedup)
            for item in requirements.unresolved_isa_operations
            if item.expected_end_to_end_speedup.availability is Availability.AVAILABLE
        ),
        key=lambda item: float(item[1].value or 0),
        reverse=True,
    )
    lines = [
        "# SLOForge Helix BranchFabric Characterization Report",
        "",
        "## Evidence scope",
        "",
        f"- Run ID: `{manifest.run_id}`",
        f"- Combined canonical trace corpus: `{requirements.trace_corpus_hash}`",
        f"- Verified evidence artifacts: {len(requirements.verified_artifacts)}",
        "- Workload evidence: deterministic synthetic coding-agent fixture",
        "- Timing evidence: measured host CPU monotonic/process clocks where recorded",
        (
            "- GPU status: hardware-backed GPU measurements available"
            if capability.compatible_gpu_available
            else "- GPU status: unavailable; no compatible NVIDIA/CUDA Helix GPU was recorded"
        ),
        "",
        "Controlled matrix points are design sweeps, not empirical production distributions. "
        "Every number below remains linked to its immutable source artifact in the companion "
        "requirements JSON.",
        "",
        "## Measured branch state",
        "",
        "| Characteristic | Result |",
        "|---|---|",
        f"| Branch fanout | {_distribution_text(requirements.state.branch_fanout)} |",
        f"| Shared root bytes per fork | {_distribution_text(requirements.state.shared_root_bytes)} |",
        f"| Private suffix bytes per fork | {_distribution_text(requirements.state.private_suffix_bytes)} |",
        f"| Branch lifetime | {_distribution_text(requirements.state.branch_lifetime_ms)} |",
        f"| Divergence rate | {_distribution_text(requirements.state.divergence_rate)} |",
        "",
        "## Copy-on-write and metadata",
        "",
        f"- Physical model-state amplification: {_distribution_text(requirements.cow.amplification)}.",
        f"- Hardware page-fault rate: {_distribution_text(requirements.cow.fault_rate)}.",
        (
            "- Page-size decision: UNKNOWN. The trace records logical whole-file and logical-page "
            "COW, not a GPU/OS page-fault sweep."
        ),
        f"- Current-software metadata throughput summaries: {_distribution_text(requirements.metadata.operations_per_second)}.",
        f"- Outstanding metadata queue depth: {_distribution_text(requirements.metadata.queue_depth)}.",
        "",
        "## Transform, transport, and transactions",
        "",
        f"- Explicit transform sequences retained: {len(requirements.transform.top_sequences)}.",
        f"- STATE_SEND payload bytes: {_distribution_text(requirements.network.unicast_bytes)}.",
        f"- Observed send fanout: {_distribution_text(requirements.network.fanout)}.",
        f"- Multicast opportunity bytes: {_distribution_text(requirements.network.multicast_opportunity_bytes)}.",
        f"- Commit rate in the bounded Continuum trace: {_distribution_text(requirements.transactions.commit_rate)}.",
        f"- Abort rate in the same observation window: {_distribution_text(requirements.transactions.abort_rate)}.",
        "",
        "## Amdahl bounds",
        "",
        (
            "These are independent, non-additive upper bounds from directly recorded spans. "
            "They are not hardware speedup forecasts; simulated transport timing remains labeled."
        ),
        "",
        "| Operation | Maximum speedup if recorded primitive becomes free |",
        "|---|---|",
    ]
    lines.extend(f"| {operation} | {_metric_text(metric)} |" for operation, metric in amdahl_rows)
    if not amdahl_rows:
        lines.append("| none | UNKNOWN |")
    lines.extend(
        [
            "",
            "## Hardware conclusions supported by this run",
            "",
            f"- REQUIRED/HIGH_VALUE/OPTIONAL ISA recommendations: {len(requirements.recommended_isa)}.",
            f"- SOFTWARE_ONLY classifications: {len(requirements.software_only_operations)}.",
            f"- NOT_JUSTIFIED for the current bounded corpus: {len(requirements.not_justified_operations)}.",
            f"- Unresolved pending stronger measurements: {len(requirements.unresolved_isa_operations)}.",
            "",
            "No hardware primitive is promoted from this run alone. Unobserved quantization, "
            "compression, encryption, and multicast operations are negative findings for this "
            "bounded corpus; observed operations remain unresolved until target sensitivity, Amdahl "
            "leverage, and strongest-software-baseline evidence are all available.",
            "",
            "## Queue, memory, bandwidth, and latency requirements",
            "",
            "Observed concurrency establishes only lower bounds for queue depth. Recommended and "
            "pathological depths remain UNKNOWN. The controlled matrix covers 8--128 branches, but "
            "does not execute those fanouts, so capacity fields remain UNKNOWN rather than extrapolated. "
            "GPU interface targets are UNAVAILABLE on this host; all other bandwidth and maximum "
            "tolerable latency targets remain UNKNOWN without sensitivity experiments.",
            "",
            "## Limitations",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in requirements.limitations)
    lines.extend(
        [
            "",
            "## Evidence artifacts",
            "",
            "| Included in trace corpus | SHA-256 | Artifact |",
            "|---|---|---|",
        ]
    )
    for artifact in requirements.verified_artifacts:
        lines.append(
            f"| {'yes' if artifact.included_in_trace_corpus else 'no'} | "
            f"`{artifact.sha256}` | `{artifact.artifact_reference}` |"
        )
    return "\n".join(lines) + "\n"


def write_characterization_report(*, run: Path, output: Path, replace: bool) -> object:
    """Render a self-contained report and its independently verified requirements JSON."""

    requirements, manifest = _compile_requirements_from_run(run)
    if output.suffix.lower() == ".md":
        report_path = output
        requirements_path = output.with_suffix(".requirements.json")
        report_manifest_path = output.with_suffix(".manifest.json")
    else:
        report_path = output / "CHARACTERIZATION_REPORT.md"
        requirements_path = output / "branchfabric_requirements.json"
        report_manifest_path = output / "report-manifest.json"
    targets = (report_path, requirements_path, report_manifest_path)
    existing = [path for path in targets if path.exists()]
    if existing and not replace:
        raise FileExistsError(f"report output already exists: {existing[0]}")
    for path in targets:
        path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(_render_report(requirements, manifest), encoding="utf-8")
    write_requirements(requirements_path, requirements, overwrite=replace)
    report_manifest = {
        "schema_version": "sloforge.branchfabric.characterization-report-manifest/v1",
        "run_id": manifest.run_id,
        "trace_corpus_hash": requirements.trace_corpus_hash,
        "report": {
            "path": report_path.as_posix(),
            "sha256": _sha256(report_path),
        },
        "requirements": {
            "path": requirements_path.as_posix(),
            "sha256": _sha256(requirements_path),
        },
        "hardware_backed_gpu_measurements": manifest.capability_evidence.compatible_gpu_available,
        "synthetic_workload": True,
    }
    _write_json(report_manifest_path, report_manifest, replace=replace)
    return {
        "schema_version": report_manifest["schema_version"],
        "output": report_path.as_posix(),
        "artifact_sha256": _sha256(report_path),
        "requirements": requirements_path.as_posix(),
        "requirements_sha256": _sha256(requirements_path),
        "manifest": report_manifest_path.as_posix(),
        "trace_corpus_hash": requirements.trace_corpus_hash,
        "run_id": manifest.run_id,
    }


__all__ = [
    "MAX_WORKFLOW_EVENTS",
    "analyze_cow",
    "analyze_run_amdahl",
    "analyze_workload",
    "derive_requirements",
    "resume_characterization",
    "run_characterization",
    "write_characterization_report",
]
