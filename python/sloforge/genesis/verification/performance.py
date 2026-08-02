"""Raw-sample performance acceptance with seeded bootstrap uncertainty."""

from __future__ import annotations

import random
import statistics

import numpy as np

from .model import (
    BenchmarkContract,
    EvidenceStatus,
    MetricDirection,
    PerformanceEvidence,
    VerificationError,
)


def _improvement(baseline: float, candidate: float, direction: MetricDirection) -> float:
    if baseline <= 0 or candidate <= 0:
        raise VerificationError("performance samples must be positive")
    if direction is MetricDirection.LOWER_IS_BETTER:
        return (baseline - candidate) / baseline * 100
    return (candidate - baseline) / baseline * 100


def _effect_size(
    baseline: tuple[float, ...], candidate: tuple[float, ...], direction: MetricDirection
) -> float:
    favorable = 0.0
    total = len(baseline) * len(candidate)
    for left in baseline:
        for right in candidate:
            difference = (
                left - right if direction is MetricDirection.LOWER_IS_BETTER else right - left
            )
            favorable += 1 if difference > 0 else 0.5 if difference == 0 else 0
    return favorable / total * 2 - 1


def evaluate_performance(
    contract: BenchmarkContract,
    baseline_samples: tuple[float, ...],
    candidate_samples: tuple[float, ...],
    *,
    seed: int,
) -> PerformanceEvidence:
    if not contract.benchmark_id or not contract.workload_fingerprint:
        raise VerificationError("benchmark identity and workload fingerprint are required")
    if not contract.hardware_fingerprint or not contract.software_manifest_hash:
        raise VerificationError("hardware and software provenance are required")
    if len(baseline_samples) < 7 or len(candidate_samples) < 7:
        raise VerificationError("at least seven measured samples per alternative are required")
    if contract.warmup_count <= 0:
        raise VerificationError("warmup policy must be explicit and positive")
    if not 100 <= contract.bootstrap_rounds <= 1_000_000:
        raise VerificationError("bootstrap rounds must be in [100, 1000000]")
    if not 0.5 < contract.confidence < 1:
        raise VerificationError("confidence must be in (0.5, 1)")
    baseline = float(statistics.median(baseline_samples))
    candidate = float(statistics.median(candidate_samples))
    improvement = _improvement(baseline, candidate, contract.direction)
    generator = np.random.default_rng(seed)
    base_array = np.asarray(baseline_samples, dtype=np.float64)
    candidate_array = np.asarray(candidate_samples, dtype=np.float64)
    draws = np.empty(contract.bootstrap_rounds, dtype=np.float64)
    for index in range(contract.bootstrap_rounds):
        base_draw = generator.choice(base_array, size=base_array.size, replace=True)
        candidate_draw = generator.choice(candidate_array, size=candidate_array.size, replace=True)
        draws[index] = _improvement(
            float(np.median(base_draw)), float(np.median(candidate_draw)), contract.direction
        )
    alpha = 1 - contract.confidence
    low, high = np.quantile(draws, (alpha / 2, 1 - alpha / 2)).tolist()
    threshold = max(contract.practical_significance_percent, contract.noise_floor_percent)
    if low > threshold:
        status = EvidenceStatus.PASSED
        rationale = "confidence interval exceeds practical and measured noise thresholds"
    elif high < -threshold:
        status = EvidenceStatus.FAILED
        rationale = "confidence interval establishes a practically significant regression"
    else:
        status = EvidenceStatus.INCONCLUSIVE
        rationale = "confidence interval overlaps the practical or noise threshold"
    run_order = ["baseline"] * len(baseline_samples) + ["candidate"] * len(candidate_samples)
    random.Random(seed).shuffle(run_order)
    return PerformanceEvidence(
        status=status,
        seed=seed,
        contract=contract,
        baseline_samples=baseline_samples,
        candidate_samples=candidate_samples,
        run_order=tuple(run_order),
        baseline_median=baseline,
        candidate_median=candidate,
        improvement_percent=improvement,
        interval_low_percent=float(low),
        interval_high_percent=float(high),
        effect_size=_effect_size(baseline_samples, candidate_samples, contract.direction),
        rationale=rationale,
    )
