"""Deterministic robust summaries used by WarmPath profiling."""

from __future__ import annotations

import math
import random
import statistics
from collections.abc import Sequence


def percentile(samples: Sequence[float], quantile: float) -> float:
    if not samples:
        raise ValueError("percentile requires samples")
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be between zero and one")
    values = sorted(samples)
    if len(values) == 1:
        return values[0]
    position = quantile * (len(values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    fraction = position - lower
    return values[lower] * (1.0 - fraction) + values[upper] * fraction


def bootstrap_median_interval(
    samples: tuple[float, ...],
    *,
    seed: int,
    rounds: int = 1_000,
    confidence_level: float = 0.95,
) -> tuple[float, float]:
    if len(samples) < 3:
        raise ValueError("bootstrap interval requires at least three samples")
    if rounds < 200:
        raise ValueError("bootstrap requires at least 200 rounds")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence level must be between zero and one")
    rng = random.Random(seed)
    medians = sorted(statistics.median(rng.choices(samples, k=len(samples))) for _ in range(rounds))
    alpha = 1.0 - confidence_level
    return percentile(medians, alpha / 2.0), percentile(medians, 1.0 - alpha / 2.0)


def robust_summary(
    samples: tuple[float, ...], *, seed: int
) -> tuple[float, float, float, float, float]:
    if len(samples) < 3:
        raise ValueError("summary requires at least three samples")
    if any(not math.isfinite(sample) or sample < 0.0 for sample in samples):
        raise ValueError("timing samples must be finite and non-negative")
    median = statistics.median(samples)
    mad = statistics.median(abs(sample - median) for sample in samples)
    low, high = bootstrap_median_interval(samples, seed=seed)
    return median, percentile(samples, 0.95), mad, low, high
