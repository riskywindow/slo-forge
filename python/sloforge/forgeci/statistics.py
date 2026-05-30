"""Robust, deterministic statistical methods for ForgeCI."""

from __future__ import annotations

import math
import random
import statistics

from sloforge.forgeci.models import (
    ComparisonClassification,
    MetricComparison,
    MetricDirection,
    MetricSpec,
    MetricSummary,
)


def percentile(values: list[float], quantile: float) -> float:
    """Return a linearly interpolated percentile for non-empty samples."""

    if not values:
        raise ValueError("percentile requires at least one sample")
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be between zero and one")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = quantile * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def summarize_metric(
    metric: MetricSpec,
    samples: tuple[float, ...],
    *,
    bootstrap_rounds: int,
    seed: int,
    significance_level: float | None = None,
) -> MetricSummary:
    """Summarize raw measurements while preserving their full provenance."""

    if len(samples) < 3:
        raise ValueError("metric summary requires at least three samples")
    if any(not math.isfinite(sample) for sample in samples):
        raise ValueError("metric samples must be finite")
    if bootstrap_rounds < 200:
        raise ValueError("bootstrap_rounds must be at least 200")
    alpha = significance_level if significance_level is not None else metric.significance_level
    median = statistics.median(samples)
    deviations = tuple(abs(sample - median) for sample in samples)
    rng = random.Random(seed)
    bootstrapped = sorted(
        statistics.median(rng.choices(samples, k=len(samples))) for _ in range(bootstrap_rounds)
    )
    return MetricSummary(
        name=metric.name,
        unit=metric.unit,
        sample_count=len(samples),
        median=median,
        median_absolute_deviation=statistics.median(deviations),
        p95=percentile(list(samples), 0.95),
        confidence_level=1.0 - alpha,
        median_ci_low=percentile(bootstrapped, alpha / 2.0),
        median_ci_high=percentile(bootstrapped, 1.0 - alpha / 2.0),
        raw_samples=samples,
    )


def _degradation_percent(baseline: float, candidate: float, direction: MetricDirection) -> float:
    denominator = max(abs(baseline), 1e-12)
    raw = 100.0 * (candidate - baseline) / denominator
    return raw if direction == MetricDirection.LOWER_IS_BETTER else -raw


def _cliffs_delta(
    baseline: tuple[float, ...], candidate: tuple[float, ...], direction: MetricDirection
) -> float:
    worse = 0
    better = 0
    for base in baseline:
        for current in candidate:
            if current > base:
                worse += 1
            elif current < base:
                better += 1
    delta = (worse - better) / (len(baseline) * len(candidate))
    return delta if direction == MetricDirection.LOWER_IS_BETTER else -delta


def _noise_floor_percent(samples: tuple[float, ...]) -> float:
    median = statistics.median(samples)
    if abs(median) < 1e-12:
        return 0.0 if all(abs(value) < 1e-12 for value in samples) else math.inf
    mad = statistics.median(abs(value - median) for value in samples)
    return 100.0 * 1.4826 * mad / abs(median)


def compare_metric(
    spec: MetricSpec,
    baseline: MetricSummary,
    candidate: MetricSummary,
    *,
    corrected_alpha: float,
    bootstrap_rounds: int,
    seed: int,
) -> MetricComparison:
    """Compare two independent samples using bootstrap uncertainty and robust effects."""

    if baseline.name != spec.name or candidate.name != spec.name:
        raise ValueError(f"metric summary does not match {spec.name}")
    rng = random.Random(seed)
    effects = sorted(
        _degradation_percent(
            statistics.median(rng.choices(baseline.raw_samples, k=baseline.sample_count)),
            statistics.median(rng.choices(candidate.raw_samples, k=candidate.sample_count)),
            spec.direction,
        )
        for _ in range(bootstrap_rounds)
    )
    effect = _degradation_percent(baseline.median, candidate.median, spec.direction)
    low = percentile(effects, corrected_alpha / 2.0)
    high = percentile(effects, 1.0 - corrected_alpha / 2.0)
    noise_floor = max(
        _noise_floor_percent(baseline.raw_samples),
        _noise_floor_percent(candidate.raw_samples),
    )
    practical = max(spec.practical_threshold_percent, noise_floor)
    delta = _cliffs_delta(baseline.raw_samples, candidate.raw_samples, spec.direction)

    if noise_floor > spec.maximum_noise_percent:
        classification = ComparisonClassification.FLAKY
        rationale = (
            f"robust noise floor {noise_floor:.3f}% exceeds allowed "
            f"{spec.maximum_noise_percent:.3f}%"
        )
    elif low > 0.0 and effect >= practical:
        classification = ComparisonClassification.REGRESSION
        rationale = (
            f"degradation {effect:.3f}% exceeds practical/noise threshold {practical:.3f}% "
            f"and corrected {(1.0 - corrected_alpha):.1%} interval excludes zero"
        )
    elif high < 0.0 and -effect >= practical:
        classification = ComparisonClassification.IMPROVEMENT
        rationale = (
            f"improvement {-effect:.3f}% exceeds practical/noise threshold {practical:.3f}% "
            f"and corrected {(1.0 - corrected_alpha):.1%} interval excludes zero"
        )
    elif low <= 0.0 <= high and abs(effect) >= practical:
        classification = ComparisonClassification.INCONCLUSIVE
        rationale = (
            f"effect {effect:.3f}% is practically meaningful but interval "
            f"[{low:.3f}%, {high:.3f}%] crosses zero"
        )
    else:
        classification = ComparisonClassification.UNCHANGED
        rationale = (
            f"effect {effect:.3f}% does not exceed practical/noise threshold {practical:.3f}%"
        )
    return MetricComparison(
        metric=spec.name,
        unit=spec.unit,
        baseline_median=baseline.median,
        candidate_median=candidate.median,
        degradation_percent=effect,
        degradation_ci_low_percent=low,
        degradation_ci_high_percent=high,
        cliffs_delta=delta,
        noise_floor_percent=noise_floor,
        corrected_significance_level=corrected_alpha,
        classification=classification,
        rationale=rationale,
    )
