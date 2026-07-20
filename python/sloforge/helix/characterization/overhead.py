"""Repeated disabled/minimal/full instrumentation-overhead characterization."""

from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from sloforge.helix.characterization.analysis import (
    ArtifactEvidence,
    BootstrapStatistic,
    RawSampleSeries,
    WarmupPolicy,
    bootstrap_confidence_interval,
    identify_mad_outliers,
    noise_floor,
    paired_effect_size,
    summarize,
)
from sloforge.helix.characterization.lifecycle import (
    InMemoryLifecycleRecorder,
    run_characterized_cpu_demo,
)
from sloforge.helix.characterization.lifecycle import (
    TraceLevel as LifecycleTraceLevel,
)
from sloforge.helix.characterization.lifecycle.recorder import JsonValue
from sloforge.helix.characterization.matrix import EvidenceClass, TraceLevel
from sloforge.helix.characterization.resources import ResourceSampler, ResourceSamplerConfig

MAX_OVERHEAD_TRIALS = 300


class OverheadModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, validate_default=True)


class OverheadTrial(OverheadModel):
    workload_evidence_class: Literal[EvidenceClass.SYNTHETIC]
    timing_measurement_class: Literal[EvidenceClass.HARDWARE_BACKED_REAL]
    trace_level: TraceLevel
    seed: int = Field(ge=0)
    repetition: int = Field(ge=0)
    warmup: bool
    order_index: int = Field(ge=0)
    artifact_path: str
    wall_time_ns: int = Field(gt=0)
    cpu_time_ns: int = Field(gt=0)
    cpu_core_equivalents: float = Field(ge=0.0, allow_inf_nan=False)
    branch_event_count: int = Field(ge=0)
    state_event_count: int = Field(ge=0)
    semantic_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    branch_readiness_ns: int | None = Field(default=None, ge=0)
    generated_tokens: int | None = Field(default=None, ge=0)
    rollout_throughput_tokens_per_second: float | None = Field(
        default=None, ge=0.0, allow_inf_nan=False
    )
    workload_artifact_bytes: int = Field(ge=0)
    workload_storage_bytes_per_second: float = Field(ge=0.0, allow_inf_nan=False)
    process_write_bytes_delta: int | None = Field(default=None, ge=0)
    process_read_bytes_delta: int | None = Field(default=None, ge=0)
    peak_sampled_rss_bytes: int | None = Field(default=None, ge=0)
    peak_sampled_vms_bytes: int | None = Field(default=None, ge=0)
    resource_sample_count: int = Field(ge=0)
    resource_samples_dropped: int = Field(ge=0)
    ttft_ns: None = None
    per_token_latency_ns: None = None
    gpu_utilization_percent: None = None


class InstrumentationOverheadArtifact(OverheadModel):
    schema_version: Literal["sloforge.branchfabric.instrumentation-overhead/v1"]
    workload_evidence_class: Literal[EvidenceClass.SYNTHETIC]
    timing_measurement_class: Literal[EvidenceClass.HARDWARE_BACKED_REAL]
    seeds: tuple[int, ...] = Field(min_length=1)
    repetitions: int = Field(ge=1)
    warmups_per_level: int = Field(ge=0)
    run_order_seed: int = Field(ge=0)
    raw_artifact: str
    raw_artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    trials: tuple[OverheadTrial, ...] = Field(min_length=1, max_length=MAX_OVERHEAD_TRIALS)
    wall_time_statistics: dict[str, dict[str, object]]
    cpu_time_statistics: dict[str, dict[str, object]]
    wall_time_paired_effects: dict[str, dict[str, object]]
    semantic_equivalence_verified: bool
    metric_availability: dict[str, str]
    limitations: tuple[str, ...]

    @model_validator(mode="after")
    def has_each_trace_level(self) -> InstrumentationOverheadArtifact:
        measured_levels = {trial.trace_level for trial in self.trials if not trial.warmup}
        if measured_levels != set(TraceLevel):
            raise ValueError("overhead artifact requires measured disabled/minimal/full trials")
        return self


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _directory_bytes(root: Path) -> int:
    return sum(
        path.stat().st_size for path in root.rglob("*") if path.is_file() and not path.is_symlink()
    )


def _resource_delta(values: tuple[int | None, ...]) -> int | None:
    present = tuple(value for value in values if value is not None)
    if len(present) < 2:
        return None
    return max(0, present[-1] - present[0])


def _event_int(event: Mapping[str, JsonValue], key: str, *, default: int = 0) -> int:
    value = event.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"lifecycle event field {key!r} must be an integer")
    return value


def _run_trial(
    output: Path,
    *,
    level: TraceLevel,
    seed: int,
    repetition: int,
    warmup: bool,
    order_index: int,
) -> OverheadTrial:
    recorder = InMemoryLifecycleRecorder()
    sampler = ResourceSampler(
        ResourceSamplerConfig(
            trace_level=level,
            workload_evidence=EvidenceClass.SYNTHETIC,
            seed=seed,
            sample_interval_ms=100,
            max_samples=5000,
            max_duration_seconds=300.0,
        )
    ).start()
    try:
        run = run_characterized_cpu_demo(
            output,
            seed=seed,
            recorder=recorder,
            trace_level=LifecycleTraceLevel(level.value),
        )
    finally:
        resource = sampler.stop(timeout_seconds=5.0)
    workload_bytes = _directory_bytes(output)
    resource_path = output / "resource-trace-v1.json"
    _write_json(resource_path, resource.model_dump(mode="json"))

    branch_forks = tuple(
        event for event in recorder.branch_events if event.get("operation_type") == "BRANCH_FORK"
    )
    branch_readiness = _event_int(branch_forks[0], "duration_ns") if branch_forks else None
    rollouts = tuple(
        event
        for event in recorder.branch_events
        if event.get("operation_type") == "ROLLOUT_COMPLETE"
    )
    generated_tokens = (
        sum(_event_int(event, "generated_tokens") for event in rollouts) if rollouts else None
    )
    rollout_duration_ns = sum(_event_int(event, "duration_ns") for event in rollouts)
    throughput = (
        generated_tokens * 1_000_000_000 / rollout_duration_ns
        if generated_tokens is not None and rollout_duration_ns
        else None
    )
    rss_values = tuple(sample.process_rss_bytes for sample in resource.samples)
    vms_values = tuple(sample.process_vms_bytes for sample in resource.samples)
    read_values = tuple(sample.process_read_bytes for sample in resource.samples)
    write_values = tuple(sample.process_write_bytes for sample in resource.samples)
    return OverheadTrial(
        workload_evidence_class=EvidenceClass.SYNTHETIC,
        timing_measurement_class=EvidenceClass.HARDWARE_BACKED_REAL,
        trace_level=level,
        seed=seed,
        repetition=repetition,
        warmup=warmup,
        order_index=order_index,
        artifact_path=output.as_posix(),
        wall_time_ns=run.wall_time_ns,
        cpu_time_ns=run.cpu_time_ns,
        cpu_core_equivalents=run.cpu_time_ns / run.wall_time_ns,
        branch_event_count=run.branch_event_count,
        state_event_count=run.state_event_count,
        semantic_digest=run.semantic_digest,
        branch_readiness_ns=branch_readiness,
        generated_tokens=generated_tokens,
        rollout_throughput_tokens_per_second=throughput,
        workload_artifact_bytes=workload_bytes,
        workload_storage_bytes_per_second=workload_bytes * 1_000_000_000 / run.wall_time_ns,
        process_write_bytes_delta=_resource_delta(write_values),
        process_read_bytes_delta=_resource_delta(read_values),
        peak_sampled_rss_bytes=max(
            (value for value in rss_values if value is not None), default=None
        ),
        peak_sampled_vms_bytes=max(
            (value for value in vms_values if value is not None), default=None
        ),
        resource_sample_count=resource.samples_recorded,
        resource_samples_dropped=resource.samples_dropped,
    )


def _series(
    trials: tuple[OverheadTrial, ...],
    *,
    level: TraceLevel,
    metric: Literal["wall_time_ns", "cpu_time_ns"],
    raw_path: Path,
    raw_sha256: str,
    run_order_seed: int,
) -> RawSampleSeries:
    warmups = tuple(float(getattr(trial, metric)) for trial in trials if trial.warmup)
    measured = tuple(
        float(getattr(trial, metric))
        for trial in sorted(
            (item for item in trials if not item.warmup),
            key=lambda item: (item.seed, item.repetition),
        )
    )
    evidence = ArtifactEvidence(
        schema_version="sloforge.branchfabric.artifact-evidence/v1",
        source_experiment="helix-instrumentation-overhead",
        artifact_reference=raw_path.as_posix(),
        artifact_sha256=raw_sha256,
        evidence_class=EvidenceClass.SYNTHETIC,
        sample_selector=f"trials[trace_level={level.value}].{metric}",
        sample_count=len(warmups) + len(measured),
        seed=run_order_seed,
        repetition=0,
    )
    return RawSampleSeries(
        schema_version="sloforge.branchfabric.raw-sample-series/v1",
        series_id=f"helix-overhead/{level.value}/{metric}",
        metric=metric,
        unit="nanoseconds",
        provenance=evidence,
        warmup_policy=WarmupPolicy(
            method="fixed_count" if warmups else "none",
            declared_warmup_count=len(warmups),
            rationale="one explicit warmup per trace level" if warmups else "no warmup requested",
        ),
        warmup_samples=warmups,
        samples=measured,
    )


def run_instrumentation_overhead_study(
    output: Path,
    *,
    seeds: tuple[int, ...],
    repetitions: int,
    warmups_per_level: int,
    run_order_seed: int,
) -> InstrumentationOverheadArtifact:
    """Run randomized trials and retain every warmup, sample, and derived statistic."""

    if not seeds or len(seeds) != len(set(seeds)) or any(seed < 0 for seed in seeds):
        raise ValueError("overhead seeds must be unique non-negative integers")
    if not 1 <= repetitions <= 20 or not 0 <= warmups_per_level <= 3:
        raise ValueError("repetitions must be 1..20 and warmups per level 0..3")
    trial_count = len(TraceLevel) * (warmups_per_level + len(seeds) * repetitions)
    if trial_count > MAX_OVERHEAD_TRIALS:
        raise ValueError(f"overhead study exceeds the {MAX_OVERHEAD_TRIALS}-trial bound")
    if output.exists() and (not output.is_dir() or output.is_symlink() or any(output.iterdir())):
        raise FileExistsError("overhead output must be a new or empty directory")
    output.mkdir(parents=True, exist_ok=True)
    schedule: list[tuple[TraceLevel, int, int, bool]] = []
    for level in TraceLevel:
        schedule.extend(
            (level, seeds[0], repetition, True) for repetition in range(warmups_per_level)
        )
        schedule.extend(
            (level, seed, repetition, False) for seed in seeds for repetition in range(repetitions)
        )
    random.Random(run_order_seed).shuffle(schedule)
    trials = tuple(
        _run_trial(
            output / "trials" / f"{order_index:03d}-{level.value}-s{seed}-r{repetition}",
            level=level,
            seed=seed,
            repetition=repetition,
            warmup=warmup,
            order_index=order_index,
        )
        for order_index, (level, seed, repetition, warmup) in enumerate(schedule)
    )
    semantics: dict[tuple[int, int, bool], set[str]] = defaultdict(set)
    for trial in trials:
        semantics[(trial.seed, trial.repetition, trial.warmup)].add(trial.semantic_digest)
    semantic_equivalence = all(len(values) == 1 for values in semantics.values())
    if not semantic_equivalence:
        raise RuntimeError("trace levels changed Helix semantic outputs")

    raw_path = output / "overhead-raw-samples.json"
    _write_json(
        raw_path,
        {
            "schema_version": "sloforge.branchfabric.instrumentation-overhead-raw/v1",
            "workload_evidence_class": "SYNTHETIC",
            "timing_measurement_class": "HARDWARE_BACKED_REAL",
            "run_order_seed": run_order_seed,
            "schedule": [
                {
                    "order_index": trial.order_index,
                    "trace_level": trial.trace_level.value,
                    "seed": trial.seed,
                    "repetition": trial.repetition,
                    "warmup": trial.warmup,
                }
                for trial in trials
            ],
            "trials": [trial.model_dump(mode="json") for trial in trials],
        },
    )
    raw_hash = _sha256(raw_path)
    wall_series = {
        level.value: _series(
            trials,
            level=level,
            metric="wall_time_ns",
            raw_path=raw_path,
            raw_sha256=raw_hash,
            run_order_seed=run_order_seed,
        )
        for level in TraceLevel
    }
    cpu_series = {
        level.value: _series(
            trials,
            level=level,
            metric="cpu_time_ns",
            raw_path=raw_path,
            raw_sha256=raw_hash,
            run_order_seed=run_order_seed,
        )
        for level in TraceLevel
    }

    def statistics_for(series: RawSampleSeries, bootstrap_seed: int) -> dict[str, object]:
        return {
            "series": series.model_dump(mode="json"),
            "summary": summarize(series).model_dump(mode="json"),
            "median_confidence_interval": bootstrap_confidence_interval(
                series,
                BootstrapStatistic.MEDIAN,
                seed=bootstrap_seed,
                repetitions=2000,
            ).model_dump(mode="json"),
            "noise_floor": noise_floor(series).model_dump(mode="json"),
            "outliers": identify_mad_outliers(series).model_dump(mode="json"),
        }

    wall_stats = {
        level.value: statistics_for(wall_series[level.value], run_order_seed + index)
        for index, level in enumerate(TraceLevel)
    }
    cpu_stats = {
        level.value: statistics_for(cpu_series[level.value], run_order_seed + 100 + index)
        for index, level in enumerate(TraceLevel)
    }
    disabled = wall_series[TraceLevel.DISABLED.value]
    paired = {
        level.value: paired_effect_size(disabled, wall_series[level.value]).model_dump(mode="json")
        for level in (TraceLevel.MINIMAL, TraceLevel.FULL)
    }
    artifact = InstrumentationOverheadArtifact(
        schema_version="sloforge.branchfabric.instrumentation-overhead/v1",
        workload_evidence_class=EvidenceClass.SYNTHETIC,
        timing_measurement_class=EvidenceClass.HARDWARE_BACKED_REAL,
        seeds=seeds,
        repetitions=repetitions,
        warmups_per_level=warmups_per_level,
        run_order_seed=run_order_seed,
        raw_artifact=raw_path.as_posix(),
        raw_artifact_sha256=raw_hash,
        trials=trials,
        wall_time_statistics=wall_stats,
        cpu_time_statistics=cpu_stats,
        wall_time_paired_effects=paired,
        semantic_equivalence_verified=True,
        metric_availability={
            "ttft_ns": "UNAVAILABLE: reference CPU policy demo has no model-server TTFT",
            "per_token_latency_ns": (
                "UNAVAILABLE: fixture emits policy decisions, not model-server token timings"
            ),
            "branch_readiness_ns": "AVAILABLE for minimal/full; disabled mode has no wrappers",
            "rollout_throughput": "AVAILABLE for minimal/full; disabled mode has no wrappers",
            "cpu_utilization": "AVAILABLE as process CPU time divided by wall time",
            "memory": "AVAILABLE as 100 ms sampled process RSS/VMS",
            "storage_writes": "AVAILABLE as psutil process I/O where supported plus artifact bytes",
            "gpu_utilization": "UNAVAILABLE: no compatible GPU execution occurred",
            "learning_transaction_latency": "AVAILABLE as full run wall time",
        },
        limitations=(
            "The workload is the deterministic synthetic Helix CPU demo, not production traffic.",
            "A negative overhead estimate inside the measured noise floor is not a speedup claim.",
            "Resource sampling is periodic and can miss peaks between 100 ms samples.",
            "Lifecycle events are buffered in memory during the timed run; canonical adaptation and "
            "trace persistence are measured separately by the vertical trace workflow.",
        ),
    )
    _write_json(output / "instrumentation-overhead.json", artifact.model_dump(mode="json"))
    return artifact


__all__ = [
    "MAX_OVERHEAD_TRIALS",
    "InstrumentationOverheadArtifact",
    "OverheadTrial",
    "run_instrumentation_overhead_study",
]
