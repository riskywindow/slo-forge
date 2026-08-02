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
    task_hash: str,
    run_seed: int,
    repetitions: int,
    request_ids: tuple[str, ...],
    measurement_run_order: tuple[tuple[int, BaselineKind], ...],
) -> IntegrityReport:
    violations: list[str] = []
    expected_pairs = {
        (repetition, request_id) for repetition in range(repetitions) for request_id in request_ids
    }
    by_baseline: dict[BaselineKind, list[RawCpuSample]] = defaultdict(list)
    by_measurement_block: dict[int, list[RawCpuSample]] = defaultdict(list)
    expected_surfaces = {
        baseline: (
            "python_eager_reference"
            if baseline is BaselineKind.PYTHON_EAGER_REFERENCE
            else "genesis_generated_runtime"
            if baseline is BaselineKind.GENESIS_FULL
            else "reference_order_surrogate"
        )
        for baseline in measured_baselines
    }
    for sample in samples:
        by_baseline[sample.baseline].append(sample)
        by_measurement_block[sample.measurement_order_ordinal].append(sample)
        if sample.baseline not in measured_baselines:
            violations.append(f"{sample.baseline}:undeclared_measured_baseline")
        if sample.workload_fingerprint != workload_fingerprint:
            violations.append(f"{sample.baseline}:workload_fingerprint_mismatch")
        if sample.environment_fingerprint != environment_fingerprint:
            violations.append(f"{sample.baseline}:environment_fingerprint_mismatch")
        expected_source = (
            "measured_cpu_monotonic_clock"
            if sample.baseline in {BaselineKind.PYTHON_EAGER_REFERENCE, BaselineKind.GENESIS_FULL}
            else "replayed_cpu_reference_observation"
        )
        if sample.source != expected_source:
            violations.append(f"{sample.baseline}:untrusted_timer_source")
        if sample.precision != "python_float64_reference":
            violations.append(f"{sample.baseline}:precision_mismatch")
        if sample.task_hash != task_hash:
            violations.append(f"{sample.baseline}:task_hash_mismatch")
        if sample.run_seed != run_seed:
            violations.append(f"{sample.baseline}:run_seed_mismatch")
        if sample.execution_surface != expected_surfaces.get(sample.baseline):
            violations.append(f"{sample.baseline}:execution_surface_mismatch")
        if sample.exact_match != (sample.observed_tokens == sample.expected_tokens):
            violations.append(f"{sample.baseline}:false_exact_match")
        if sample.ttft_ns > sample.latency_ns:
            violations.append(f"{sample.baseline}:ttft_exceeds_latency")
        if len(sample.inter_token_ns) != max(0, len(sample.observed_tokens) - 1):
            violations.append(f"{sample.baseline}:inter_token_count_mismatch")
        if sample.ttft_ns + sum(sample.inter_token_ns) > sample.latency_ns:
            violations.append(f"{sample.baseline}:token_timeline_exceeds_latency")
    if set(by_measurement_block) != set(range(len(measurement_run_order))):
        violations.append("global_measurement_order:ordinal_set_mismatch")
    for ordinal, (expected_repetition, expected_baseline) in enumerate(measurement_run_order):
        block = by_measurement_block.get(ordinal, [])
        if len(block) != len(request_ids):
            violations.append("global_measurement_order:block_size_mismatch")
            continue
        if {(sample.repetition, sample.baseline) for sample in block} != {
            (expected_repetition, expected_baseline)
        }:
            violations.append("global_measurement_order:block_identity_mismatch")
        if {sample.request_id for sample in block} != set(request_ids):
            violations.append("global_measurement_order:block_request_set_mismatch")
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
        ordered = sorted(baseline_samples, key=lambda sample: sample.execution_ordinal)
        request_count = len(request_ids)
        observed_repetition_order: list[int] = []
        if request_count:
            for offset in range(0, len(ordered), request_count):
                chunk = ordered[offset : offset + request_count]
                if len(chunk) != request_count or len({item.repetition for item in chunk}) != 1:
                    violations.append(f"{baseline}:measurement_chunk_mismatch")
                    continue
                observed_repetition_order.append(chunk[0].repetition)
        expected_repetition_order = [
            repetition
            for repetition, ordered_baseline in measurement_run_order
            if ordered_baseline is baseline
        ]
        if observed_repetition_order != expected_repetition_order:
            violations.append(f"{baseline}:measurement_run_order_mismatch")
    return IntegrityReport(
        passed=not violations,
        violations=tuple(sorted(set(violations))),
        workload_fingerprint=workload_fingerprint,
        environment_fingerprint=environment_fingerprint,
        expected_repetitions=repetitions,
        expected_request_ids=request_ids,
    )
