"""Deterministic statistical primitives for BranchFabric Experiment 003.

The functions in this module are local-only and preserve every raw observation
and artifact reference used by a derived result.  In particular, the three-pair
fanout-8 design is never presented as supporting p95 or p99 estimates.
"""

from __future__ import annotations

import json
import math
import random
import statistics
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal, TypeVar, cast

MAX_BOOTSTRAP_REPETITIONS = 100_000
MAX_BOOTSTRAP_DRAWS = 20_000_000
TAIL_UNAVAILABLE = {
    "p95": None,
    "p99": None,
    "status": "TAIL_INSUFFICIENT",
    "reason": "three paired trials do not support tail-percentile inference",
}
_JsonValue = TypeVar("_JsonValue")
_BOOTSTRAP_STATISTICS: tuple[Literal["mean", "median"], ...] = ("mean", "median")


def _unavailable_tails(reason: str) -> dict[str, Any]:
    return {
        "p95": None,
        "p99": None,
        "status": "TAIL_INSUFFICIENT",
        "reason": reason,
    }


def _finite(value: float, *, field: str, nonnegative: bool = False) -> float:
    converted = float(value)
    if not math.isfinite(converted) or (nonnegative and converted < 0.0):
        qualifier = "finite and nonnegative" if nonnegative else "finite"
        raise ValueError(f"{field} must be {qualifier}")
    return converted


def _nonempty(value: str, *, field: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be nonempty")


def _json_clone(value: _JsonValue) -> _JsonValue:
    return cast(
        _JsonValue,
        json.loads(json.dumps(value, sort_keys=True, separators=(",", ":"))),
    )


@dataclass(frozen=True, slots=True)
class ArtifactProvenance:
    artifact_reference: str
    artifact_sha256: str
    sample_selector: str

    def __post_init__(self) -> None:
        _nonempty(self.artifact_reference, field="artifact_reference")
        _nonempty(self.sample_selector, field="sample_selector")
        if len(self.artifact_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in self.artifact_sha256
        ):
            raise ValueError("artifact_sha256 must be a lowercase SHA-256 digest")

    def to_dict(self) -> dict[str, str]:
        return {
            "artifact_reference": self.artifact_reference,
            "artifact_sha256": self.artifact_sha256,
            "sample_selector": self.sample_selector,
        }


@dataclass(frozen=True, slots=True)
class PairKey:
    pair_id: str
    seed: int
    repetition: int
    workload_digest: str
    function_call_id: str
    gpu_uuid: str

    def __post_init__(self) -> None:
        for field, value in (
            ("pair_id", self.pair_id),
            ("workload_digest", self.workload_digest),
            ("function_call_id", self.function_call_id),
            ("gpu_uuid", self.gpu_uuid),
        ):
            _nonempty(value, field=field)
        if isinstance(self.seed, bool) or self.seed < 0:
            raise ValueError("seed must be nonnegative")
        if isinstance(self.repetition, bool) or self.repetition < 0:
            raise ValueError("repetition must be nonnegative")

    def to_dict(self) -> dict[str, Any]:
        return {
            "pair_id": self.pair_id,
            "seed": self.seed,
            "repetition": self.repetition,
            "workload_digest": self.workload_digest,
            "function_call_id": self.function_call_id,
            "gpu_uuid": self.gpu_uuid,
        }


@dataclass(frozen=True, slots=True)
class PairedObservation:
    key: PairKey
    baseline_value: float
    candidate_value: float
    baseline_provenance: ArtifactProvenance
    candidate_provenance: ArtifactProvenance

    def __post_init__(self) -> None:
        _finite(self.baseline_value, field="baseline_value", nonnegative=True)
        _finite(self.candidate_value, field="candidate_value", nonnegative=True)


@dataclass(frozen=True, slots=True)
class TraceOverheadObservation:
    key: PairKey
    disabled_value: float
    minimal_value: float
    full_value: float
    disabled_provenance: ArtifactProvenance
    minimal_provenance: ArtifactProvenance
    full_provenance: ArtifactProvenance

    def __post_init__(self) -> None:
        if _finite(self.disabled_value, field="disabled_value", nonnegative=True) <= 0.0:
            raise ValueError("disabled_value must be positive for an overhead ratio")
        _finite(self.minimal_value, field="minimal_value", nonnegative=True)
        _finite(self.full_value, field="full_value", nonnegative=True)


@dataclass(frozen=True, slots=True)
class ScalingObservation:
    fanout: int
    prefix_blocks: int
    value: float
    repetition: int
    provenance: ArtifactProvenance

    def __post_init__(self) -> None:
        if isinstance(self.fanout, bool) or self.fanout <= 0:
            raise ValueError("fanout must be a positive integer")
        if isinstance(self.prefix_blocks, bool) or self.prefix_blocks <= 0:
            raise ValueError("prefix_blocks must be a positive integer")
        if isinstance(self.repetition, bool) or self.repetition < 0:
            raise ValueError("repetition must be nonnegative")
        _finite(self.value, field="value", nonnegative=True)


def _percentile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise ValueError("percentile requires at least one value")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be in [0, 1]")
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


def _bootstrap_ci(
    values: Sequence[float],
    *,
    statistic: Literal["mean", "median"],
    seed: int,
    repetitions: int,
    confidence_level: float,
) -> dict[str, Any]:
    if not 0 < repetitions <= MAX_BOOTSTRAP_REPETITIONS:
        raise ValueError("bootstrap repetitions are outside the bounded range")
    if repetitions * len(values) > MAX_BOOTSTRAP_DRAWS:
        raise ValueError("bootstrap draw count exceeds the bounded limit")
    if isinstance(seed, bool) or seed < 0 or seed >= 1 << 64:
        raise ValueError("bootstrap seed must be an unsigned 64-bit integer")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be in (0, 1)")
    evaluate = cast(
        Callable[[Sequence[float]], float],
        statistics.fmean if statistic == "mean" else statistics.median,
    )
    generator = random.Random(seed)
    count = len(values)
    estimates = [
        float(evaluate([values[generator.randrange(count)] for _ in range(count)]))
        for _ in range(repetitions)
    ]
    tail = (1.0 - confidence_level) / 2.0
    return {
        "statistic": statistic,
        "observed": float(evaluate(values)),
        "lower": _percentile(estimates, tail),
        "upper": _percentile(estimates, 1.0 - tail),
        "confidence_level": confidence_level,
        "repetitions": repetitions,
        "seed": seed,
        "method": "deterministic paired percentile bootstrap",
        "resampling_unit": "whole matched pair difference",
        "percentile_method": "Hyndman-Fan type 7",
    }


def _average_ranks(values: Sequence[float]) -> list[float]:
    ordered = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and ordered[end][1] == ordered[start][1]:
            end += 1
        average_rank = ((start + 1) + end) / 2.0
        for original_index, _ in ordered[start:end]:
            ranks[original_index] = average_rank
        start = end
    return ranks


def _matched_rank_biserial(differences: Sequence[float]) -> tuple[float | None, int]:
    nonzero = [difference for difference in differences if difference != 0.0]
    zero_count = len(differences) - len(nonzero)
    if not nonzero:
        return None, zero_count
    ranks = _average_ranks([abs(difference) for difference in nonzero])
    positive = sum(rank for rank, difference in zip(ranks, nonzero, strict=True) if difference > 0)
    negative = sum(rank for rank, difference in zip(ranks, nonzero, strict=True) if difference < 0)
    return (positive - negative) / (positive + negative), zero_count


def analyze_three_pair_effect(
    observations: Sequence[PairedObservation],
    *,
    metric: str,
    unit: str,
    baseline_label: str,
    candidate_label: str,
    bootstrap_seed: int,
    bootstrap_repetitions: int = 10_000,
    confidence_level: float = 0.95,
) -> dict[str, Any]:
    """Analyze exactly three pre-matched pairs without inventing tail statistics."""

    for field, value in (
        ("metric", metric),
        ("unit", unit),
        ("baseline_label", baseline_label),
        ("candidate_label", candidate_label),
    ):
        _nonempty(value, field=field)
    if len(observations) != 3:
        raise ValueError("fanout-8 primary analysis requires exactly three matched pairs")
    keys = [observation.key for observation in observations]
    if len(set(keys)) != len(keys):
        raise ValueError("paired observations contain a duplicate typed pair key")

    differences = [
        float(observation.candidate_value - observation.baseline_value)
        for observation in observations
    ]
    relative_changes = [
        (
            None
            if observation.baseline_value == 0.0
            else (observation.candidate_value - observation.baseline_value)
            / abs(observation.baseline_value)
        )
        for observation in observations
    ]
    raw_pairs = [
        {
            "pair_key": observation.key.to_dict(),
            "baseline_value": observation.baseline_value,
            "candidate_value": observation.candidate_value,
            "difference": difference,
            "relative_change": relative,
            "baseline_provenance": observation.baseline_provenance.to_dict(),
            "candidate_provenance": observation.candidate_provenance.to_dict(),
        }
        for observation, difference, relative in zip(
            observations, differences, relative_changes, strict=True
        )
    ]
    sample_stdev = statistics.stdev(differences)
    cohens_dz = None if sample_stdev == 0.0 else statistics.fmean(differences) / sample_stdev
    rank_biserial, zero_differences = _matched_rank_biserial(differences)
    nonzero_relative = [value for value in relative_changes if value is not None]
    result = {
        "schema_version": "sloforge.branchfabric.experiment-003-paired-analysis/v1",
        "metric": metric,
        "unit": unit,
        "baseline_label": baseline_label,
        "candidate_label": candidate_label,
        "direction": "candidate minus baseline",
        "pair_count": 3,
        "pair_order": "explicit typed pair keys; input order preserved",
        "raw_pairs": raw_pairs,
        "raw_differences": differences,
        "summary": {
            "minimum_difference": min(differences),
            "maximum_difference": max(differences),
            "range_difference": max(differences) - min(differences),
            "mean_difference": statistics.fmean(differences),
            "median_difference": statistics.median(differences),
            "median_relative_change": (
                None if not nonzero_relative else statistics.median(nonzero_relative)
            ),
            "zero_baseline_pairs": len(relative_changes) - len(nonzero_relative),
            "outlier_policy": "none removed",
        },
        "bootstrap_confidence_intervals": [
            _bootstrap_ci(
                differences,
                statistic=statistic,
                seed=bootstrap_seed,
                repetitions=bootstrap_repetitions,
                confidence_level=confidence_level,
            )
            for statistic in _BOOTSTRAP_STATISTICS
        ],
        "effect_sizes": {
            "paired_cohens_dz": cohens_dz,
            "cohens_dz_method": "mean(pair differences) / sample SD(pair differences)",
            "cohens_dz_caveat": (
                "descriptive only: n=3 gives an unstable standardized effect and does not "
                "establish population normality"
            ),
            "matched_pairs_rank_biserial": rank_biserial,
            "rank_biserial_method": (
                "Wilcoxon signed ranks with average absolute-difference tie ranks"
            ),
            "zero_differences_excluded_from_rank_sum": zero_differences,
            "rank_biserial_caveat": "descriptive only with three matched pairs",
        },
        "tail_statistics": dict(TAIL_UNAVAILABLE),
    }
    return _json_clone(result)


def analyze_trace_overhead(
    observations: Sequence[TraceOverheadObservation],
    *,
    metric: str,
    unit: str,
) -> dict[str, Any]:
    """Calculate paired disabled/minimal/full tracing ratios from raw controls."""

    _nonempty(metric, field="metric")
    _nonempty(unit, field="unit")
    if not observations:
        raise ValueError("trace overhead analysis requires at least one paired control")
    if len({observation.key for observation in observations}) != len(observations):
        raise ValueError("trace controls contain a duplicate typed pair key")
    raw: list[dict[str, Any]] = []
    for observation in observations:
        minimal_ratio = observation.minimal_value / observation.disabled_value
        full_ratio = observation.full_value / observation.disabled_value
        raw.append(
            {
                "pair_key": observation.key.to_dict(),
                "disabled_value": observation.disabled_value,
                "minimal_value": observation.minimal_value,
                "full_value": observation.full_value,
                "minimal_to_disabled_ratio": minimal_ratio,
                "full_to_disabled_ratio": full_ratio,
                "minimal_overhead_fraction": minimal_ratio - 1.0,
                "full_overhead_fraction": full_ratio - 1.0,
                "provenance": {
                    "disabled": observation.disabled_provenance.to_dict(),
                    "minimal": observation.minimal_provenance.to_dict(),
                    "full": observation.full_provenance.to_dict(),
                },
            }
        )

    minimal_fractions = [float(item["minimal_overhead_fraction"]) for item in raw]
    full_fractions = [float(item["full_overhead_fraction"]) for item in raw]

    def summarize(values: Sequence[float]) -> dict[str, float]:
        return {
            "median": statistics.median(values),
            "minimum": min(values),
            "maximum": max(values),
            "range": max(values) - min(values),
        }

    return _json_clone(
        {
            "schema_version": "sloforge.branchfabric.experiment-003-trace-overhead/v1",
            "metric": metric,
            "unit": unit,
            "control": "tracing disabled",
            "sample_count": len(observations),
            "raw_controls": raw,
            "minimal_trace_overhead": summarize(minimal_fractions),
            "full_trace_overhead": summarize(full_fractions),
            "ratio_definition": "traced POST_ROOT_READY / disabled POST_ROOT_READY",
            "overhead_fraction_definition": "ratio - 1",
            "tail_statistics": _unavailable_tails(
                "trace-control sample count does not support tail-percentile inference"
            ),
        }
    )


def _fit_constant(values: Sequence[float]) -> tuple[float, float, list[float]]:
    intercept = statistics.fmean(values)
    return intercept, 0.0, [intercept] * len(values)


def _fit_affine(
    features: Sequence[float], values: Sequence[float]
) -> tuple[float, float, list[float]]:
    feature_mean = statistics.fmean(features)
    value_mean = statistics.fmean(values)
    denominator = sum((feature - feature_mean) ** 2 for feature in features)
    if denominator == 0.0:
        raise ValueError("candidate scaling feature is constant")
    slope = (
        sum(
            (feature - feature_mean) * (value - value_mean)
            for feature, value in zip(features, values, strict=True)
        )
        / denominator
    )
    intercept = value_mean - slope * feature_mean
    return intercept, slope, [intercept + slope * feature for feature in features]


def _proportional(left: Sequence[float], right: Sequence[float]) -> bool:
    if len(left) != len(right) or not left:
        return False
    ratios = [right_value / left_value for left_value, right_value in zip(left, right, strict=True)]
    first = ratios[0]
    return all(math.isclose(ratio, first, rel_tol=1e-12, abs_tol=1e-12) for ratio in ratios[1:])


def select_scaling_model(
    observations: Sequence[ScalingObservation],
    *,
    metric: str,
    unit: str,
    relative_uncertainty: float = 0.05,
) -> dict[str, Any]:
    """Choose the simplest candidate consistent with an explicit error tolerance."""

    _nonempty(metric, field="metric")
    _nonempty(unit, field="unit")
    if len(observations) < 3:
        raise ValueError("scaling analysis requires at least three observations")
    if not 0.0 <= relative_uncertainty < 1.0:
        raise ValueError("relative_uncertainty must be in [0, 1)")
    identity_keys = [(item.fanout, item.prefix_blocks, item.repetition) for item in observations]
    if len(set(identity_keys)) != len(identity_keys):
        raise ValueError("scaling observations contain a duplicate typed observation key")

    fanout = [float(item.fanout) for item in observations]
    values = [float(item.value) for item in observations]
    features = {
        "O(1)": [1.0] * len(observations),
        "O(N)": fanout,
        "O(N log N)": [value * math.log2(value) for value in fanout],
        "O(N*prefix_blocks)": [float(item.fanout * item.prefix_blocks) for item in observations],
    }
    complexity = {
        "O(1)": 0,
        "O(N)": 1,
        "O(N log N)": 2,
        "O(N*prefix_blocks)": 3,
    }
    response_scale = max(abs(statistics.median(values)), 1e-12)
    candidates: list[dict[str, Any]] = []
    for name, feature in features.items():
        aliased_with = None
        identifiable = True
        if name == "O(N*prefix_blocks)" and _proportional(features["O(N)"], feature):
            aliased_with = "O(N)"
            identifiable = False
        try:
            if name == "O(1)":
                intercept, slope, predicted = _fit_constant(values)
            else:
                intercept, slope, predicted = _fit_affine(feature, values)
            residuals = [
                value - prediction for value, prediction in zip(values, predicted, strict=True)
            ]
            rmse = math.sqrt(statistics.fmean([residual**2 for residual in residuals]))
            nrmse = rmse / response_scale
            total = sum((value - statistics.fmean(values)) ** 2 for value in values)
            residual_sum = sum(residual**2 for residual in residuals)
            r_squared = None if total == 0.0 else 1.0 - residual_sum / total
            candidate = {
                "model": name,
                "identifiable": identifiable,
                "aliased_with": aliased_with,
                "intercept": intercept,
                "slope": slope,
                "rmse": rmse,
                "normalized_rmse": nrmse,
                "r_squared": r_squared,
                "predicted_values": predicted,
                "residuals": residuals,
                "complexity_rank": complexity[name],
            }
        except ValueError as error:
            candidate = {
                "model": name,
                "identifiable": False,
                "aliased_with": aliased_with,
                "unavailable_reason": str(error),
                "complexity_rank": complexity[name],
            }
        candidates.append(candidate)

    estimable = [
        candidate
        for candidate in candidates
        if candidate.get("identifiable") is True
        and isinstance(candidate.get("normalized_rmse"), float)
    ]
    if not estimable:
        raise ValueError("no candidate scaling model was identifiable")
    best_error = min(float(candidate["normalized_rmse"]) for candidate in estimable)
    acceptance_limit = best_error + relative_uncertainty
    consistent = [
        candidate
        for candidate in estimable
        if float(candidate["normalized_rmse"]) <= acceptance_limit
    ]
    selected = min(consistent, key=lambda candidate: int(candidate["complexity_rank"]))
    raw = [
        {
            "fanout": item.fanout,
            "prefix_blocks": item.prefix_blocks,
            "value": item.value,
            "repetition": item.repetition,
            "provenance": item.provenance.to_dict(),
        }
        for item in observations
    ]
    return _json_clone(
        {
            "schema_version": "sloforge.branchfabric.experiment-003-scaling-model/v1",
            "metric": metric,
            "unit": unit,
            "raw_observations": raw,
            "candidate_models": candidates,
            "selected_model": selected["model"],
            "best_normalized_rmse": best_error,
            "acceptance_normalized_rmse": acceptance_limit,
            "relative_uncertainty": relative_uncertainty,
            "selection_rule": (
                "lowest-complexity identifiable candidate within relative_uncertainty "
                "normalized RMSE of the best fit"
            ),
            "modeling_caveat": (
                "descriptive small-sample scaling fit; O(N*prefix_blocks) is not "
                "identifiable separately from O(N) when prefix_blocks is constant"
            ),
            "tail_statistics": _unavailable_tails(
                "the bounded fanout-scaling design does not support tail-percentile inference"
            ),
        }
    )


def amdahl_metadata_projections(
    *,
    post_root_ready_ms: float,
    metadata_ms: float,
    fanout: int,
    root_inclusive_ready_ms: float | None = None,
    post_root_process_cpu_ms: float | None = None,
    metadata_process_cpu_ms: float | None = None,
    provenance: ArtifactProvenance,
) -> dict[str, Any]:
    """Project readiness for 2x/5x/10x and effectively-free metadata."""

    total = _finite(post_root_ready_ms, field="post_root_ready_ms", nonnegative=True)
    metadata = _finite(metadata_ms, field="metadata_ms", nonnegative=True)
    if total <= 0.0:
        raise ValueError("post_root_ready_ms must be positive")
    if metadata > total:
        raise ValueError("metadata_ms cannot exceed post_root_ready_ms")
    if isinstance(fanout, bool) or fanout <= 0:
        raise ValueError("fanout must be a positive integer")
    root_inclusive = None
    if root_inclusive_ready_ms is not None:
        root_inclusive = _finite(
            root_inclusive_ready_ms,
            field="root_inclusive_ready_ms",
            nonnegative=True,
        )
        if root_inclusive < total:
            raise ValueError("root-inclusive readiness cannot be less than post-root readiness")
    process_cpu = None
    metadata_cpu = None
    if post_root_process_cpu_ms is not None or metadata_process_cpu_ms is not None:
        if post_root_process_cpu_ms is None or metadata_process_cpu_ms is None:
            raise ValueError("post-root and metadata process CPU must be supplied together")
        process_cpu = _finite(
            post_root_process_cpu_ms,
            field="post_root_process_cpu_ms",
            nonnegative=True,
        )
        metadata_cpu = _finite(
            metadata_process_cpu_ms,
            field="metadata_process_cpu_ms",
            nonnegative=True,
        )
        if metadata_cpu > process_cpu:
            raise ValueError("metadata process CPU cannot exceed post-root process CPU")

    projections: list[dict[str, Any]] = []
    for label, speedup in (("2x", 2.0), ("5x", 5.0), ("10x", 10.0), ("free", None)):
        optimized_metadata = 0.0 if speedup is None else metadata / speedup
        optimized_total = total - metadata + optimized_metadata
        total_speedup = None if optimized_total == 0.0 else total / optimized_total
        branch_rate = None if optimized_total == 0.0 else fanout * 1000.0 / optimized_total
        if root_inclusive is None:
            optimized_root_inclusive = None
            root_inclusive_speedup = None
        else:
            optimized_root_inclusive = root_inclusive - metadata + optimized_metadata
            root_inclusive_speedup = (
                None
                if optimized_root_inclusive == 0.0
                else root_inclusive / optimized_root_inclusive
            )
        if process_cpu is None or metadata_cpu is None:
            optimized_process_cpu = None
            process_cpu_reduction = None
            cpu_utilization_proxy = None
        else:
            optimized_metadata_cpu = 0.0 if speedup is None else metadata_cpu / speedup
            optimized_process_cpu = process_cpu - metadata_cpu + optimized_metadata_cpu
            process_cpu_reduction = (
                0.0 if process_cpu == 0.0 else (process_cpu - optimized_process_cpu) / process_cpu
            )
            cpu_utilization_proxy = optimized_process_cpu / optimized_total
        projections.append(
            {
                "metadata_speedup": label,
                "optimized_metadata_ms": optimized_metadata,
                "projected_post_root_ready_ms": optimized_total,
                "post_root_ready_speedup": total_speedup,
                "post_root_ready_reduction_fraction": (total - optimized_total) / total,
                "projected_valid_branches_per_second": branch_rate,
                "projected_root_inclusive_ready_ms": optimized_root_inclusive,
                "root_inclusive_ready_speedup": root_inclusive_speedup,
                "projected_post_root_process_cpu_ms": optimized_process_cpu,
                "post_root_process_cpu_reduction_fraction": process_cpu_reduction,
                "projected_process_cpu_to_wall_ratio": cpu_utilization_proxy,
            }
        )

    baseline_rate = fanout * 1000.0 / total
    return _json_clone(
        {
            "schema_version": "sloforge.branchfabric.experiment-003-amdahl/v1",
            "fanout": fanout,
            "post_root_ready_ms": total,
            "metadata_ms": metadata,
            "metadata_fraction": metadata / total,
            "root_inclusive_ready_ms": root_inclusive,
            "post_root_process_cpu_ms": process_cpu,
            "metadata_process_cpu_ms": metadata_cpu,
            "baseline_process_cpu_to_wall_ratio": (
                None if process_cpu is None else process_cpu / total
            ),
            "baseline_valid_branches_per_second": baseline_rate,
            "provenance": provenance.to_dict(),
            "projections": projections,
            "ideal_metadata_free_speedup": projections[-1]["post_root_ready_speedup"],
            "assumption": (
                "only measured metadata time changes; all orchestration, scheduler, GPU, "
                "output, and residual time remains fixed"
            ),
        }
    )


__all__ = [
    "MAX_BOOTSTRAP_DRAWS",
    "MAX_BOOTSTRAP_REPETITIONS",
    "TAIL_UNAVAILABLE",
    "ArtifactProvenance",
    "PairKey",
    "PairedObservation",
    "ScalingObservation",
    "TraceOverheadObservation",
    "amdahl_metadata_projections",
    "analyze_three_pair_effect",
    "analyze_trace_overhead",
    "select_scaling_model",
]
