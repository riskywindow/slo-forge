"""Raw-sample CPU benchmarking and conservative full-stack acceptance gates."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
from pathlib import Path
from typing import Literal

from ..sandbox import SandboxTermination
from ..verification import (
    BenchmarkContract,
    EvidenceStatus,
    MetricDirection,
    evaluate_performance,
)
from .cases import generate_correctness_cases
from .executor import execute_benchmark_payload, validate_correctness_evidence
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

_REGIMES: tuple[
    tuple[
        str,
        Literal[
            "isolated_operator_microbenchmark",
            "repeated_operator_loop_not_serving_end_to_end",
        ],
    ],
    ...,
] = (
    ("micro_batch_1", "isolated_operator_microbenchmark"),
    ("micro_noncontiguous", "isolated_operator_microbenchmark"),
    ("operator_loop_batch_32", "repeated_operator_loop_not_serving_end_to_end"),
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
            "regime": "operator_loop_batch_32",
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


def _fingerprints(
    configuration: dict[str, object], config: KernelBenchmarkConfig
) -> tuple[str, str, tuple[str, ...]]:
    encoded = json.dumps(
        {
            "execution": configuration,
            "analysis": config.model_dump(mode="json"),
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
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
    hardware = _hardware_fingerprint_from_manifest(manifest)
    return workload, hardware, manifest


def _hardware_fingerprint_from_manifest(manifest: tuple[str, ...]) -> str:
    hardware_items = tuple(
        item
        for item in manifest
        if item.startswith(("machine=", "system=", "logical_cpu_count=", "cpu_affinity="))
    )
    if len(hardware_items) != 4:
        raise ValueError("CPU benchmark manifest is missing hardware provenance")
    return "cpu:" + hashlib.sha256("|".join(hardware_items).encode()).hexdigest()


def raw_samples_bytes(samples: tuple[BenchmarkSample, ...]) -> bytes:
    return "".join(
        json.dumps(sample.model_dump(mode="json"), sort_keys=True) + "\n" for sample in samples
    ).encode()


def _regime_samples(
    samples: tuple[BenchmarkSample, ...],
    *,
    regime: str,
    config: KernelBenchmarkConfig,
) -> tuple[tuple[BenchmarkSample, ...], tuple[float, ...], tuple[float, ...]]:
    selected = tuple(
        sorted(
            (item for item in samples if item.regime == regime),
            key=lambda item: item.order_index,
        )
    )
    expected_iterations = (
        config.token_loop_iterations
        if regime == "operator_loop_batch_32"
        else config.micro_iterations
    )
    if len(selected) != config.repetitions * 2:
        raise ValueError(f"{regime} does not contain every required raw sample")
    if [item.order_index for item in selected] != list(range(config.repetitions * 2)):
        raise ValueError(f"{regime} raw sample order is incomplete or duplicated")
    if any(item.iterations != expected_iterations for item in selected):
        raise ValueError(f"{regime} raw sample iteration count changed")
    for alternative in ("reference", "candidate"):
        trials = sorted(item.trial_index for item in selected if item.alternative == alternative)
        if trials != list(range(config.repetitions)):
            raise ValueError(f"{regime} {alternative} trial set is incomplete or duplicated")
    reference = tuple(
        float(item.duration_ns) for item in selected if item.alternative == "reference"
    )
    candidate = tuple(
        float(item.duration_ns) for item in selected if item.alternative == "candidate"
    )
    return selected, reference, candidate


def _build_regimes(
    samples: tuple[BenchmarkSample, ...],
    *,
    config: KernelBenchmarkConfig,
    workload: str,
    hardware: str,
    manifest: tuple[str, ...],
) -> tuple[BenchmarkRegimeEvidence, ...]:
    declared = {name for name, _scope in _REGIMES}
    if {item.regime for item in samples} != declared:
        raise ValueError("raw samples do not contain exactly the declared operator regimes")
    evidence: list[BenchmarkRegimeEvidence] = []
    for regime, scope in _REGIMES:
        regime_samples, reference, current = _regime_samples(samples, regime=regime, config=config)
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
                measurement_scope=scope,
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
    return tuple(evidence)


def _aggregate_status(regimes: tuple[BenchmarkRegimeEvidence, ...]) -> LabStatus:
    return (
        LabStatus.FAILED
        if any(item.status is LabStatus.FAILED for item in regimes)
        else LabStatus.PASSED
        if regimes and all(item.status is LabStatus.PASSED for item in regimes)
        else LabStatus.INCONCLUSIVE
    )


def validate_benchmark_report(report: KernelBenchmarkReport) -> None:
    """Independently reconstruct all statistical claims from embedded raw samples."""

    config = report.benchmark_config
    if report.deterministic_seed != config.deterministic_seed:
        raise ValueError("benchmark report and analysis seeds differ")
    configuration = _benchmark_configuration(config)
    expected_workload, _current_hardware, _current_manifest = _fingerprints(configuration, config)
    if report.workload_fingerprint != expected_workload:
        raise ValueError("benchmark workload fingerprint is not reproducible")
    recorded_hardware = _hardware_fingerprint_from_manifest(report.software_manifest)
    if report.hardware_fingerprint != recorded_hardware:
        raise ValueError("benchmark hardware fingerprint is not derived from its manifest")
    if not report.raw_samples:
        if report.regimes or report.raw_samples_sha256 is not None:
            raise ValueError("unmeasured benchmark contains derived evidence")
        if report.status in {LabStatus.PASSED, LabStatus.INCONCLUSIVE}:
            raise ValueError("unmeasured benchmark cannot pass or be inconclusive")
        return
    digest = hashlib.sha256(raw_samples_bytes(report.raw_samples)).hexdigest()
    if digest != report.raw_samples_sha256:
        raise ValueError("benchmark raw sample digest mismatch")
    rebuilt = _build_regimes(
        report.raw_samples,
        config=config,
        workload=report.workload_fingerprint,
        hardware=report.hardware_fingerprint,
        manifest=report.software_manifest,
    )
    if rebuilt != report.regimes:
        raise ValueError("benchmark regime claims are not derived from raw samples")
    if _aggregate_status(rebuilt) is not report.status:
        raise ValueError("benchmark aggregate status is not derived from its regimes")


def benchmark_candidate(
    candidate: KernelCandidate,
    source: str,
    correctness: CorrectnessEvidence,
    *,
    output_root: Path,
    config: KernelBenchmarkConfig,
) -> KernelBenchmarkReport:
    configuration = _benchmark_configuration(config)
    workload, hardware, manifest = _fingerprints(configuration, config)
    if correctness.status is not LabStatus.PASSED:
        return KernelBenchmarkReport(
            status=(
                LabStatus.UNAVAILABLE
                if correctness.status is LabStatus.UNAVAILABLE
                else LabStatus.FAILED
            ),
            candidate=candidate,
            deterministic_seed=config.deterministic_seed,
            benchmark_config=config,
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
            benchmark_config=config,
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
            benchmark_config=config,
            hardware_fingerprint=hardware,
            software_manifest=manifest,
            workload_fingerprint=workload,
            raw_samples=(),
            raw_samples_sha256=None,
            regimes=(),
            sandbox_termination="invalid_benchmark_output",
            sandbox_backend=backend,
        )
    try:
        evidence = _build_regimes(
            samples,
            config=config,
            workload=workload,
            hardware=hardware,
            manifest=manifest,
        )
    except ValueError:
        return KernelBenchmarkReport(
            status=LabStatus.FAILED,
            candidate=candidate,
            deterministic_seed=config.deterministic_seed,
            benchmark_config=config,
            hardware_fingerprint=hardware,
            software_manifest=manifest,
            workload_fingerprint=workload,
            raw_samples=(),
            raw_samples_sha256=None,
            regimes=(),
            sandbox_termination="invalid_benchmark_output",
            sandbox_backend=backend,
        )
    report = KernelBenchmarkReport(
        status=_aggregate_status(evidence),
        candidate=candidate,
        deterministic_seed=config.deterministic_seed,
        benchmark_config=config,
        hardware_fingerprint=hardware,
        software_manifest=manifest,
        workload_fingerprint=workload,
        raw_samples=samples,
        raw_samples_sha256=hashlib.sha256(raw_samples_bytes(samples)).hexdigest(),
        regimes=evidence,
        sandbox_termination=termination.value,
        sandbox_backend=backend,
    )
    validate_benchmark_report(report)
    return report


def decide_candidate(
    correctness: CorrectnessEvidence, benchmark: KernelBenchmarkReport
) -> CandidateDecision:
    if correctness.candidate != benchmark.candidate:
        raise ValueError("correctness and benchmark evidence refer to different candidates")
    validate_correctness_evidence(correctness)
    validate_benchmark_report(benchmark)
    by_name = {item.regime: item for item in benchmark.regimes}
    micro = by_name.get("micro_noncontiguous")
    operator_loop = by_name.get("operator_loop_batch_32")
    micro_status = micro.status if micro is not None else LabStatus.UNAVAILABLE
    operator_loop_status = (
        operator_loop.status if operator_loop is not None else LabStatus.UNAVAILABLE
    )
    full_stack_status = LabStatus.UNAVAILABLE
    reasons = []
    if correctness.status is not LabStatus.PASSED:
        reasons.append("independent correctness gate did not pass")
    if micro_status is not LabStatus.PASSED:
        reasons.append("intended microbenchmark confidence gate did not pass")
    if operator_loop_status is not LabStatus.PASSED:
        reasons.append("repeated operator-loop confidence gate did not pass")
    if any(item.status is not LabStatus.PASSED for item in benchmark.regimes):
        reasons.append("not every declared supported benchmark regime passed")
    if any(item.status is LabStatus.FAILED for item in benchmark.regimes):
        reasons.append("at least one declared supported benchmark regime established a regression")
    reasons.append(
        "no end-to-end serving benchmark was executed; isolated operator evidence cannot promote"
    )
    if correctness.status is LabStatus.UNAVAILABLE or benchmark.status is LabStatus.UNAVAILABLE:
        status = AcceptanceStatus.UNAVAILABLE
    elif correctness.status is LabStatus.FAILED or benchmark.status is LabStatus.FAILED:
        status = AcceptanceStatus.REJECTED
    else:
        status = AcceptanceStatus.INCONCLUSIVE
    claim = "no speedup claim; isolated operator evidence is not end-to-end serving evidence"
    return CandidateDecision(
        candidate_id=correctness.candidate.candidate_id,
        status=status,
        correctness_status=correctness.status,
        microbenchmark_status=micro_status,
        operator_loop_status=operator_loop_status,
        full_stack_status=full_stack_status,
        claim=claim,
        reasons=tuple(reasons),
    )
