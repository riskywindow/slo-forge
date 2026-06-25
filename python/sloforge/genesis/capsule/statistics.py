"""Deterministic statistical helpers shared by capsule builders and validators."""

from __future__ import annotations

import math
import random
import statistics


def _quantile(values: tuple[float, ...], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def bootstrap_median_interval(
    values: tuple[float, ...],
    *,
    seed: int,
    rounds: int,
    confidence: float,
) -> tuple[float, float]:
    """Return a seeded percentile-bootstrap interval for the sample median."""

    if len(values) < 2 or any(not math.isfinite(value) or value < 0 for value in values):
        raise ValueError("bootstrap values must be finite, non-negative, and repeated")
    if seed < 0 or not 100 <= rounds <= 1_000_000 or not 0.5 < confidence < 1.0:
        raise ValueError("bootstrap configuration is outside its bounded domain")
    generator = random.Random(seed)
    medians = tuple(
        float(statistics.median(generator.choices(values, k=len(values)))) for _ in range(rounds)
    )
    alpha = 1.0 - confidence
    return _quantile(medians, alpha / 2.0), _quantile(medians, 1.0 - alpha / 2.0)


def paired_regression_probability(
    baseline: tuple[float, ...],
    candidate: tuple[float, ...],
    *,
    objective: str,
) -> float:
    """Return the observed paired fraction in which the candidate regressed."""

    if len(baseline) != len(candidate) or not baseline:
        raise ValueError("paired samples must be non-empty and aligned")
    if objective == "minimize":
        regressions = sum(right >= left for left, right in zip(baseline, candidate, strict=True))
    elif objective == "maximize":
        regressions = sum(right <= left for left, right in zip(baseline, candidate, strict=True))
    else:
        raise ValueError("benchmark objective must be minimize or maximize")
    return regressions / len(candidate)


__all__ = ["bootstrap_median_interval", "paired_regression_probability"]
