"""Raw-sample CPU benchmarking and conservative full-stack acceptance gates."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
from pathlib import Path

from ..sandbox import SandboxTermination
from ..verification import (
    BenchmarkContract,
    EvidenceStatus,
    MetricDirection,
    evaluate_performance,
)
from .cases import generate_correctness_cases
from .executor import execute_benchmark_payload
from .models import (
    AcceptanceStatus,
    BenchmarkRegimeEvidence,
    BenchmarkSample,
    CandidateDecision,
    CorrectnessEvidence,
    KernelBenchmarkConfig,
    KernelBenchmarkReport,
    KernelCandidate,
    LabStatus,
)


def _finite_case(case: object) -> bool:
    return getattr(case, "expected_error", None) is None


def _benchmark_configuration(config: KernelBenchmarkConfig) -> dict[str, object]:
    cases = tuple(
        case
        for case in generate_correctness_cases(seed=config.deterministic_seed, randomized_cases=8)
        if _finite_case(case)
    )
    batch_one = next(case for case in cases if case.count == 1)
    noncontiguous = next(case for case in cases if case.category == "noncontiguous")
    batch_32 = next(case for case in cases if case.count >= 32)
    regimes = (
        {
            "regime": "micro_batch_1",
            "case": batch_one.model_dump(mode="json"),
            "iterations": config.micro_iterations,
            "token_steps": 0,
        },
        {
            "regime": "micro_noncontiguous",
            "case": noncontiguous.model_dump(mode="json"),
            "iterations": config.micro_iterations,
            "token_steps": 0,
        },
        {
            "regime": "token_loop_batch_32",
            "case": batch_32.model_dump(mode="json"),
            "iterations": config.token_loop_iterations,
            "token_steps": config.token_steps,
        },
    )
    return {
        "mode": "benchmark",
        "seed": config.deterministic_seed,
        "warmup_count": config.warmup_count,
        "repetitions": config.repetitions,
        "regimes": regimes,
    }


def _fingerprints(configuration: dict[str, object]) -> tuple[str, str, tuple[str, ...]]:
    encoded = json.dumps(configuration, sort_keys=True, separators=(",", ":")).encode()
    workload = hashlib.sha256(encoded).hexdigest()
    affinity = (
        ",".join(str(cpu) for cpu in sorted(os.sched_getaffinity(0)))
        if hasattr(os, "sched_getaffinity")
        else "unavailable"
    )
    manifest = (
        f"python={platform.python_version()}",
        f"implementation={platform.python_implementation()}",
        f"machine={platform.machine() or 'unknown'}",
        f"system={platform.system()}",
        f"executable={sys.executable}",
        f"logical_cpu_count={os.cpu_count() or 0}",
        f"cpu_affinity={affinity}",
        "timer=perf_counter_ns",
    )
    hardware = "cpu:" + hashlib.sha256("|".join(manifest[2:4]).encode()).hexdigest()
    return workload, hardware, manifest


def raw_samples_bytes(samples: tuple[BenchmarkSample, ...]) -> bytes:
    return "".join(
        json.dumps(sample.model_dump(mode="json"), sort_keys=True) + "\n" for sample in samples
    ).encode()


def benchmark_candidate(
    candidate: KernelCandidate,
    source: str,
    correctness: CorrectnessEvidence,
    *,
    output_root: Path,
    config: KernelBenchmarkConfig,
) -> KernelBenchmarkReport:
    configuration = _benchmark_configuration(config)
    workload, hardware, manifest = _fingerprints(configuration)
    if correctness.status is not LabStatus.PASSED:
        return KernelBenchmarkReport(
            status=(
                LabStatus.UNAVAILABLE
                if correctness.status is LabStatus.UNAVAILABLE
                else LabStatus.FAILED
            ),
            candidate=candidate,
            deterministic_seed=config.deterministic_seed,
            hardware_fingerprint=hardware,
            software_manifest=manifest,
            workload_fingerprint=workload,
            raw_samples=(),
            raw_samples_sha256=None,
            regimes=(),
            sandbox_termination="not_run_correctness_gate",
            sandbox_backend=correctness.sandbox_backend,
        )
    termination, backend, raw = execute_benchmark_payload(
        candidate,
        source,
        configuration,
        output_root=output_root,
        seed=config.deterministic_seed,
        timeout_seconds=config.wall_time_seconds,
    )
    if termination is not SandboxTermination.SUCCESS:
        return KernelBenchmarkReport(
            status=(
                LabStatus.UNAVAILABLE
                if termination is SandboxTermination.POLICY_UNAVAILABLE
                else LabStatus.FAILED
            ),
            candidate=candidate,
            deterministic_seed=config.deterministic_seed,
            hardware_fingerprint=hardware,
            software_manifest=manifest,
            workload_fingerprint=workload,
            raw_samples=(),
            raw_samples_sha256=None,
            regimes=(),
            sandbox_termination=termination.value,
            sandbox_backend=backend,
        )
    try:
        samples = tuple(BenchmarkSample.model_validate(item, strict=True) for item in raw)
    except ValueError:
        samples = ()
    if not samples:
        return KernelBenchmarkReport(
            status=LabStatus.FAILED,
            candidate=candidate,
            deterministic_seed=config.deterministic_seed,
            hardware_fingerprint=hardware,
            software_manifest=manifest,
            workload_fingerprint=workload,
            raw_samples=(),
            raw_samples_sha256=None,
            regimes=(),
            sandbox_termination="invalid_benchmark_output",
            sandbox_backend=backend,
        )
    evidence: list[BenchmarkRegimeEvidence] = []
    for regime in ("micro_batch_1", "micro_noncontiguous", "token_loop_batch_32"):
        regime_samples = tuple(sample for sample in samples if sample.regime == regime)
        reference = tuple(
            float(sample.duration_ns)
            for sample in regime_samples
            if sample.alternative == "reference"
        )
        current = tuple(
            float(sample.duration_ns)
            for sample in regime_samples
            if sample.alternative == "candidate"
        )
        contract = BenchmarkContract(
            benchmark_id=f"hybrid-qstate-{regime}",
            metric="duration_ns",
            unit="ns",
            direction=MetricDirection.LOWER_IS_BETTER,
            workload_fingerprint=workload,
            hardware_fingerprint=hardware,
            software_manifest_hash=hashlib.sha256("|".join(manifest).encode()).hexdigest(),
            warmup_count=config.warmup_count,
            practical_significance_percent=config.practical_significance_percent,
            noise_floor_percent=config.noise_floor_percent,
            bootstrap_rounds=config.bootstrap_rounds,
            confidence=config.confidence,
        )
        statistical = evaluate_performance(
            contract,
            reference,
            current,
            seed=config.deterministic_seed + len(evidence),
            run_order=tuple(
                "baseline" if sample.alternative == "reference" else "candidate"
                for sample in regime_samples
            ),
        )
        status = {
            EvidenceStatus.PASSED: LabStatus.PASSED,
            EvidenceStatus.FAILED: LabStatus.FAILED,
            EvidenceStatus.INCONCLUSIVE: LabStatus.INCONCLUSIVE,
            EvidenceStatus.UNAVAILABLE: LabStatus.UNAVAILABLE,
        }[statistical.status]
        evidence.append(
            BenchmarkRegimeEvidence(
                regime=regime,
                status=status,
                warmup_count=config.warmup_count,
                repetitions=config.repetitions,
                iterations=regime_samples[0].iterations,
                run_order=tuple(sample.alternative for sample in regime_samples),
                reference_samples_ns=reference,
                candidate_samples_ns=current,
                reference_median_ns=statistical.baseline_median,
                candidate_median_ns=statistical.candidate_median,
                improvement_percent=statistical.improvement_percent,
                confidence_interval_low_percent=statistical.interval_low_percent,
                confidence_interval_high_percent=statistical.interval_high_percent,
                effect_size=statistical.effect_size,
                rationale=statistical.rationale,
            )
        )
    aggregate = (
        LabStatus.FAILED
        if any(item.status is LabStatus.FAILED for item in evidence)
        else LabStatus.PASSED
        if all(item.status is LabStatus.PASSED for item in evidence)
        else LabStatus.INCONCLUSIVE
    )
    return KernelBenchmarkReport(
        status=aggregate,
        candidate=candidate,
        deterministic_seed=config.deterministic_seed,
        hardware_fingerprint=hardware,
        software_manifest=manifest,
        workload_fingerprint=workload,
        raw_samples=samples,
        raw_samples_sha256=hashlib.sha256(raw_samples_bytes(samples)).hexdigest(),
        regimes=tuple(evidence),
        sandbox_termination=termination.value,
        sandbox_backend=backend,
    )


def decide_candidate(
    correctness: CorrectnessEvidence, benchmark: KernelBenchmarkReport
) -> CandidateDecision:
    by_name = {item.regime: item for item in benchmark.regimes}
    micro = by_name.get("micro_noncontiguous")
    end_to_end = by_name.get("token_loop_batch_32")
    micro_status = micro.status if micro is not None else LabStatus.UNAVAILABLE
    end_to_end_status = end_to_end.status if end_to_end is not None else LabStatus.UNAVAILABLE
    reasons = []
    if correctness.status is not LabStatus.PASSED:
        reasons.append("independent correctness gate did not pass")
    if micro_status is not LabStatus.PASSED:
        reasons.append("intended microbenchmark confidence gate did not pass")
    if end_to_end_status is not LabStatus.PASSED:
        reasons.append("end-to-end token-loop confidence gate did not pass")
    if any(item.status is not LabStatus.PASSED for item in benchmark.regimes):
        reasons.append("not every declared supported benchmark regime passed")
    if any(item.status is LabStatus.FAILED for item in benchmark.regimes):
        reasons.append("at least one declared supported benchmark regime established a regression")
    if correctness.status is LabStatus.UNAVAILABLE or benchmark.status is LabStatus.UNAVAILABLE:
        status = AcceptanceStatus.UNAVAILABLE
    elif correctness.status is LabStatus.FAILED or benchmark.status is LabStatus.FAILED:
        status = AcceptanceStatus.REJECTED
    elif not reasons:
        status = AcceptanceStatus.ACCEPTED
    else:
        status = AcceptanceStatus.INCONCLUSIVE
    claim = (
        "accepted within the declared CPU, shape, numerical, and measured workload scope"
        if status is AcceptanceStatus.ACCEPTED
        else "no speedup claim; candidate retained with negative or inconclusive evidence"
    )
    return CandidateDecision(
        candidate_id=correctness.candidate.candidate_id,
        status=status,
        correctness_status=correctness.status,
        microbenchmark_status=micro_status,
        end_to_end_status=end_to_end_status,
        claim=claim,
        reasons=tuple(reasons) or ("all promotion gates passed",),
    )
