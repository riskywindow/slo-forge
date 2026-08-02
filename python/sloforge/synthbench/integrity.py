"""Benchmark-integrity checks over complete raw CPU sample artifacts."""

from __future__ import annotations

from collections import Counter, defaultdict

from .models import BaselineKind, IntegrityReport, RawCpuSample


def audit_raw_samples(
    samples: tuple[RawCpuSample, ...],
    *,
    measured_baselines: tuple[BaselineKind, ...],
    workload_fingerprint: str,
    environment_fingerprint: str,
    repetitions: int,
    request_ids: tuple[str, ...],
) -> IntegrityReport:
    violations: list[str] = []
    expected_pairs = {
        (repetition, request_id) for repetition in range(repetitions) for request_id in request_ids
    }
    by_baseline: dict[BaselineKind, list[RawCpuSample]] = defaultdict(list)
    for sample in samples:
        by_baseline[sample.baseline].append(sample)
        if sample.workload_fingerprint != workload_fingerprint:
            violations.append(f"{sample.baseline}:workload_fingerprint_mismatch")
        if sample.environment_fingerprint != environment_fingerprint:
            violations.append(f"{sample.baseline}:environment_fingerprint_mismatch")
        if sample.source != "measured_cpu_monotonic_clock":
            violations.append(f"{sample.baseline}:untrusted_timer_source")
        if sample.precision != "python_float64_reference":
            violations.append(f"{sample.baseline}:precision_mismatch")
    for baseline in measured_baselines:
        baseline_samples = by_baseline.get(baseline, [])
        observed_pairs = Counter(
            (sample.repetition, sample.request_id) for sample in baseline_samples
        )
        if set(observed_pairs) != expected_pairs:
            violations.append(f"{baseline}:input_distribution_mismatch")
        if any(count != 1 for count in observed_pairs.values()):
            violations.append(f"{baseline}:duplicate_or_discarded_samples")
        ordinals = sorted(sample.execution_ordinal for sample in baseline_samples)
        if ordinals != list(range(len(baseline_samples))):
            violations.append(f"{baseline}:non_contiguous_execution_ordinals")
    return IntegrityReport(
        passed=not violations,
        violations=tuple(sorted(set(violations))),
        workload_fingerprint=workload_fingerprint,
        environment_fingerprint=environment_fingerprint,
        expected_repetitions=repetitions,
        expected_request_ids=request_ids,
    )
