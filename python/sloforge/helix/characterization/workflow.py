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
from pathlib import Path

from sloforge.helix.characterization.analysis.amdahl import (
    AmdahlTimingSample,
    CandidatePrimitive,
    EndToEndObjective,
    analyze_amdahl,
)
from sloforge.helix.characterization.matrix import (
    DivergencePattern,
    expand_matrix,
    load_matrix,
)
from sloforge.helix.characterization.orchestration import (
    CPU_STAGES,
    DEFAULT_STAGES,
    OrchestrationConfig,
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
from sloforge.helix.characterization.trace import (
    BranchWorkloadEventV1,
    StateOperationEventV1,
    StateOperationType,
    TimingMeasurementClass,
    TraceEventV1,
    WorkloadProvenance,
    iter_jsonl,
)

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


def derive_requirements(*, run: Path, output: Path, replace: bool) -> object:
    raise ValueError(
        "requirements synthesis requires the completed workload, COW, metadata, transport, "
        "transform, and Amdahl artifacts; run make branchfabric-characterization first"
    )


def write_characterization_report(*, run: Path, output: Path, replace: bool) -> object:
    raise ValueError(
        "report synthesis requires compiled requirements and completed analysis artifacts"
    )


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
