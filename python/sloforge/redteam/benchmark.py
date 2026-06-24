"""Independent benchmark-integrity checks for red-team comparisons."""

from __future__ import annotations

import statistics

from .models import (
    BenchmarkComparison,
    BenchmarkIntegrityCode,
    BenchmarkIntegrityIssue,
    TimerKind,
)


def _issue(
    comparison: BenchmarkComparison,
    code: BenchmarkIntegrityCode,
    message: str,
) -> BenchmarkIntegrityIssue:
    return BenchmarkIntegrityIssue(
        code=code,
        message=message,
        baseline_run_id=comparison.baseline.run_id,
        candidate_run_id=comparison.candidate.run_id,
    )


def audit_benchmark_integrity(
    comparison: BenchmarkComparison,
) -> tuple[BenchmarkIntegrityIssue, ...]:
    """Return all independently observed integrity failures in stable order."""

    baseline = comparison.baseline
    candidate = comparison.candidate
    issues: list[BenchmarkIntegrityIssue] = []

    if not baseline.synchronized or not candidate.synchronized:
        issues.append(
            _issue(
                comparison,
                BenchmarkIntegrityCode.MISSING_SYNCHRONIZATION,
                "every timed benchmark region must synchronize before recording completion",
            )
        )
    if (
        baseline.timer_kind is TimerKind.WALL_CLOCK
        or candidate.timer_kind is TimerKind.WALL_CLOCK
        or (
            (
                baseline.timer_kind is TimerKind.CUDA_EVENT
                or candidate.timer_kind is TimerKind.CUDA_EVENT
            )
            and (not baseline.synchronized or not candidate.synchronized)
        )
    ):
        issues.append(
            _issue(
                comparison,
                BenchmarkIntegrityCode.TIMER_MISUSE,
                "wall-clock timing is not accepted for asynchronous device work",
            )
        )
    if (
        baseline.warmup_iterations == 0
        or candidate.warmup_iterations == 0
        or baseline.warmup_iterations != candidate.warmup_iterations
    ):
        issues.append(
            _issue(
                comparison,
                BenchmarkIntegrityCode.WARMUP_MISMATCH,
                "baseline and candidate require the same nonzero warmup policy",
            )
        )
    if baseline.input_fingerprints != candidate.input_fingerprints:
        issues.append(
            _issue(
                comparison,
                BenchmarkIntegrityCode.INPUT_DISTRIBUTION_MISMATCH,
                "baseline and candidate input fingerprints differ",
            )
        )
    if (
        baseline.cache_regime != candidate.cache_regime
        or not baseline.cache_reset_between_trials
        or not candidate.cache_reset_between_trials
    ):
        issues.append(
            _issue(
                comparison,
                BenchmarkIntegrityCode.CACHE_CONTAMINATION,
                "cache regime or per-trial reset differs between compared runs",
            )
        )
    if candidate.fallback_invocations != 0:
        issues.append(
            _issue(
                comparison,
                BenchmarkIntegrityCode.HIDDEN_FALLBACK,
                "candidate invoked a fallback implementation during measurement",
            )
        )
    if (
        baseline.precision is not comparison.required_precision
        or candidate.precision is not comparison.required_precision
        or baseline.precision is not candidate.precision
    ):
        issues.append(
            _issue(
                comparison,
                BenchmarkIntegrityCode.PRECISION_MISMATCH,
                "measured precision does not match the declared comparison precision",
            )
        )
    if (
        baseline.quality_contract_hash != comparison.quality_contract_hash
        or candidate.quality_contract_hash != comparison.quality_contract_hash
        or baseline.quality_score < comparison.minimum_quality_score
        or candidate.quality_score < comparison.minimum_quality_score
    ):
        issues.append(
            _issue(
                comparison,
                BenchmarkIntegrityCode.QUALITY_MISMATCH,
                "quality contract identity or minimum accepted quality is not satisfied",
            )
        )
    if not baseline.failures_included or not candidate.failures_included:
        issues.append(
            _issue(
                comparison,
                BenchmarkIntegrityCode.OMITTED_FAILURES,
                "all failures must remain in the benchmark evidence",
            )
        )
    if (
        abs(baseline.hardware_clock_mhz - candidate.hardware_clock_mhz)
        > comparison.maximum_clock_delta_mhz
    ):
        issues.append(
            _issue(
                comparison,
                BenchmarkIntegrityCode.HARDWARE_CLOCK_CHANGE,
                "hardware clocks changed beyond the declared comparison tolerance",
            )
        )
    if baseline.cpu_affinity != candidate.cpu_affinity:
        issues.append(
            _issue(
                comparison,
                BenchmarkIntegrityCode.CPU_AFFINITY_CHANGE,
                "CPU affinity differs between baseline and candidate",
            )
        )
    allowed = set(comparison.allowed_background_processes)
    baseline_background = set(baseline.background_processes) - allowed
    candidate_background = set(candidate.background_processes) - allowed
    if baseline_background != candidate_background:
        issues.append(
            _issue(
                comparison,
                BenchmarkIntegrityCode.BACKGROUND_PROCESS_CHANGE,
                "unallowlisted background processes differ between runs",
            )
        )
    if candidate.discarded_sample_indices:
        kept_samples = tuple(
            sample
            for index, sample in enumerate(candidate.raw_samples)
            if index not in candidate.discarded_sample_indices
        )
        discarded_samples = tuple(
            candidate.raw_samples[index] for index in candidate.discarded_sample_indices
        )
        retained_median = statistics.median(kept_samples) if kept_samples else 0.0
        description = (
            "candidate discarded raw samples, including samples at or above the retained median"
            if any(sample >= retained_median for sample in discarded_samples)
            else "candidate discarded raw samples without a declared independent exclusion policy"
        )
        issues.append(
            _issue(
                comparison,
                BenchmarkIntegrityCode.DISCARDED_SLOW_SAMPLES,
                description,
            )
        )
    if baseline.benchmark_definition_hash != candidate.benchmark_definition_hash:
        issues.append(
            _issue(
                comparison,
                BenchmarkIntegrityCode.BENCHMARK_DEFINITION_MISMATCH,
                "benchmark definition hashes differ",
            )
        )
    return tuple(issues)


__all__ = ["audit_benchmark_integrity"]
