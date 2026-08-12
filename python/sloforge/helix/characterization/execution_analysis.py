"""Evidence-bound statistics for the gated BranchFabric execution verticals."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from sloforge.helix.characterization.analysis.statistics import (
    ArtifactEvidence,
    BootstrapStatistic,
    RawSampleSeries,
    WarmupPolicy,
    bootstrap_confidence_interval,
    paired_effect_size,
)
from sloforge.helix.characterization.matrix import EvidenceClass

BASELINES = ("existing_serial", "optimized_bounded_parallel")
TRACE_LEVELS = ("disabled", "minimal", "full")
PHASES = (
    "branch_readiness_ns",
    "pause_checkpoint_ns",
    "migration_ns",
    "resume_ns",
    "total_interruption_ns",
    "total_wall_ns",
)
BOOTSTRAP_SEED = 202_608_091
BOOTSTRAP_REPETITIONS = 5_000


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid JSON at {path}:{line_number}") from error
        if not isinstance(value, dict):
            raise ValueError(f"expected an object at {path}:{line_number}")
        rows.append(value)
    if not rows:
        raise ValueError("reclamation raw evidence is empty")
    return rows


def _series(
    values: Sequence[float],
    *,
    series_id: str,
    metric: str,
    unit: str,
    raw_path: Path,
    selector: str,
) -> RawSampleSeries:
    return RawSampleSeries(
        schema_version="sloforge.branchfabric.raw-sample-series/v1",
        series_id=series_id,
        metric=metric,
        unit=unit,
        provenance=ArtifactEvidence(
            schema_version="sloforge.branchfabric.artifact-evidence/v1",
            source_experiment="branchfabric-execution-reclamation",
            artifact_reference=raw_path.as_posix(),
            artifact_sha256=_sha256(raw_path),
            evidence_class=EvidenceClass.SYNTHETIC,
            sample_selector=selector,
            sample_count=len(values),
            seed=BOOTSTRAP_SEED,
            repetition=0,
        ),
        warmup_policy=WarmupPolicy(
            method="none",
            declared_warmup_count=0,
            rationale="campaign uses complete randomized measurement trials with no implicit removal",
        ),
        samples=tuple(values),
    )


def _ordered(rows: Sequence[dict[str, Any]], baseline: str) -> list[dict[str, Any]]:
    selected = [row for row in rows if row.get("software_baseline") == baseline]
    return sorted(
        selected,
        key=lambda row: (int(row["seed"]), str(row["trace_level"]), int(row["repetition"])),
    )


def _median_ci(series: RawSampleSeries, *, seed_offset: int) -> dict[str, Any]:
    result = bootstrap_confidence_interval(
        series,
        BootstrapStatistic.MEDIAN,
        seed=BOOTSTRAP_SEED + seed_offset,
        repetitions=BOOTSTRAP_REPETITIONS,
    )
    return result.model_dump(mode="json")


def analyze_reclamation(raw_path: Path) -> dict[str, Any]:
    """Recompute paired effects, confidence intervals, and critical-path fractions."""

    rows = _read_jsonl(raw_path)
    required_count = 3 * len(BASELINES) * len(TRACE_LEVELS) * 2
    if len(rows) != required_count:
        raise ValueError(f"expected {required_count} complete reclamation trials")
    keys = [(int(row["seed"]), str(row["trace_level"]), int(row["repetition"])) for row in rows]
    if len(keys) != len(
        set((key, str(row["software_baseline"])) for key, row in zip(keys, rows, strict=True))
    ):
        raise ValueError("duplicate reclamation trial key")
    if {int(row["seed"]) for row in rows} != {41, 73, 113}:
        raise ValueError("reclamation evidence must contain seeds 41, 73, and 113")
    if {str(row["trace_level"]) for row in rows} != set(TRACE_LEVELS):
        raise ValueError("reclamation evidence must contain disabled, minimal, and full controls")
    if any(
        row.get("hidden_fallback")
        or row.get("actual_engine") != row.get("requested_engine")
        or int(row.get("physical_gpu_capacity_reclaimed", -1)) != 0
        for row in rows
    ):
        raise ValueError("fallback or a physical-GPU claim appeared in CPU-reference evidence")

    serial = _ordered(rows, BASELINES[0])
    optimized = _ordered(rows, BASELINES[1])
    serial_keys = [
        (int(row["seed"]), str(row["trace_level"]), int(row["repetition"])) for row in serial
    ]
    optimized_keys = [
        (int(row["seed"]), str(row["trace_level"]), int(row["repetition"])) for row in optimized
    ]
    if serial_keys != optimized_keys:
        raise ValueError("software baselines do not form matched trial pairs")

    effects: dict[str, Any] = {}
    for index, metric in enumerate(PHASES):
        baseline_series = _series(
            [float(row[metric]) for row in serial],
            series_id=f"reclamation.serial.{metric}",
            metric=metric,
            unit="ns",
            raw_path=raw_path,
            selector=f"software_baseline=existing_serial; metric={metric}",
        )
        optimized_series = _series(
            [float(row[metric]) for row in optimized],
            series_id=f"reclamation.optimized.{metric}",
            metric=metric,
            unit="ns",
            raw_path=raw_path,
            selector=f"software_baseline=optimized_bounded_parallel; metric={metric}",
        )
        ratios = _series(
            [
                float(before[metric]) / float(after[metric])
                for before, after in zip(serial, optimized, strict=True)
            ],
            series_id=f"reclamation.paired-speedup.{metric}",
            metric=f"{metric}_speedup",
            unit="ratio",
            raw_path=raw_path,
            selector=f"matched by seed, trace level, repetition; serial/optimized {metric}",
        )
        effects[metric] = {
            "baseline_median_ci": _median_ci(baseline_series, seed_offset=index * 3),
            "optimized_median_ci": _median_ci(optimized_series, seed_offset=index * 3 + 1),
            "paired_speedup_median_ci": _median_ci(ratios, seed_offset=index * 3 + 2),
            "paired_effect_size": paired_effect_size(baseline_series, optimized_series).model_dump(
                mode="json"
            ),
        }

    optimized_full = [row for row in optimized if row["trace_level"] == "full"]
    phase_fractions: dict[str, Any] = {}
    for index, metric in enumerate(PHASES[:-1]):
        denominator = "total_interruption_ns" if metric == "migration_ns" else "total_wall_ns"
        values = [float(row[metric]) / float(row[denominator]) for row in optimized_full]
        series = _series(
            values,
            series_id=f"reclamation.optimized-full.fraction.{metric}",
            metric=f"{metric}_fraction_of_{denominator}",
            unit="fraction",
            raw_path=raw_path,
            selector=f"optimized_bounded_parallel, full trace, {metric}/{denominator}",
        )
        phase_fractions[metric] = {
            "denominator": denominator,
            "median": statistics.median(values),
            "median_ci": _median_ci(series, seed_offset=100 + index),
            "minimum": min(values),
            "maximum": max(values),
        }

    tracing_controls: dict[str, Any] = {}
    for baseline_index, baseline in enumerate(BASELINES):
        selected = _ordered(rows, baseline)
        by_key = {
            (int(row["seed"]), int(row["repetition"]), str(row["trace_level"])): row
            for row in selected
        }
        changes = [
            (
                float(by_key[(seed, repetition, "full")]["total_wall_ns"])
                - float(by_key[(seed, repetition, "disabled")]["total_wall_ns"])
            )
            / float(by_key[(seed, repetition, "disabled")]["total_wall_ns"])
            for seed in (41, 73, 113)
            for repetition in (0, 1)
        ]
        series = _series(
            changes,
            series_id=f"reclamation.trace-overhead.{baseline}",
            metric="full_minus_disabled_total_wall_relative_change",
            unit="fraction",
            raw_path=raw_path,
            selector=f"matched {baseline} full-minus-disabled by seed and repetition",
        )
        tracing_controls[baseline] = {
            "median_relative_change": statistics.median(changes),
            "median_ci": _median_ci(series, seed_offset=200 + baseline_index),
            "pair_count": len(changes),
        }

    return {
        "schema_version": "sloforge.branchfabric.reclamation-execution-analysis/v1",
        "evidence_class": "CPU_REFERENCE_LOCAL_TRANSACTION",
        "raw_artifact": raw_path.as_posix(),
        "raw_sha256": _sha256(raw_path),
        "raw_trial_count": len(rows),
        "seeds": [41, 73, 113],
        "workload_classes": ["capacity_reclamation_reference"],
        "workload_class_note": (
            "one deterministic reference reclamation workload was exercised; it is not a real "
            "transformer workload and does not independently satisfy the two-class gate"
        ),
        "paired_software_effects": effects,
        "optimized_full_trace_phase_fractions": phase_fractions,
        "tracing_controls": tracing_controls,
        "state_movement_relevance_metric": {
            "metric": "migration_ns / total_interruption_ns",
            "value": phase_fractions["migration_ns"]["median"],
            "raw_bound": True,
            "hardware_gate_interpretation": (
                "end-to-end relevance context only; no target-hardware service curve or "
                "confidence-backed hardware headroom is available"
            ),
        },
        "outliers_removed": False,
        "target_hardware_measured": False,
        "calibrated_hardware_model_available": False,
    }


def write_analysis(raw_path: Path, output_path: Path, *, replace: bool) -> dict[str, Any]:
    result = analyze_reclamation(raw_path)
    if output_path.exists() and not replace:
        raise FileExistsError(f"output exists: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replace", action="store_true")
    arguments = parser.parse_args()
    result = write_analysis(arguments.raw, arguments.output, replace=arguments.replace)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
