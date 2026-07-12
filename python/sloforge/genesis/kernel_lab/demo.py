"""Artifact-producing deterministic CPU kernel-lab demonstration."""

from __future__ import annotations

import hashlib
from pathlib import Path

from .benchmark import (
    benchmark_candidate,
    decide_candidate,
    raw_samples_bytes,
    validate_benchmark_report,
)
from .cases import generate_correctness_cases
from .evidence import validate_bottleneck_evidence
from .executor import execute_correctness
from .generator import generate_candidates, hybrid_quantized_state_schema
from .models import (
    BottleneckEvidence,
    KernelBenchmarkConfig,
    KernelLabReport,
    RuntimeImpactConfig,
)
from .runtime_impact import (
    benchmark_generated_runtime_impact,
    decide_with_runtime_impact,
    derive_runtime_impact_config,
)
from .triton import triton_adapter_status


def run_kernel_lab_demo(
    *,
    evidence: BottleneckEvidence,
    output_root: Path,
    seed: int,
    candidate_limit: int = 2,
    benchmark_config: KernelBenchmarkConfig | None = None,
    reference_package_root: Path | None = None,
    runtime_impact_config: RuntimeImpactConfig | None = None,
) -> KernelLabReport:
    """Generate, sandbox, verify, benchmark, decide, and persist all raw evidence."""

    validate_bottleneck_evidence(evidence)
    if seed < 0 or not 1 <= candidate_limit <= 12:
        raise ValueError("demo seed or candidate limit is invalid")
    config = benchmark_config or KernelBenchmarkConfig(deterministic_seed=seed)
    if config.deterministic_seed != seed:
        raise ValueError("demo and benchmark seeds must match")
    if output_root.exists() and output_root.is_symlink():
        raise ValueError("kernel-lab output directory must not be a symlink")
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError("kernel-lab output directory must be empty")
    output_root.mkdir(parents=True, exist_ok=True)
    generated = generate_candidates(seed=seed)[:candidate_limit]
    cases = generate_correctness_cases(seed=seed)
    correctness = []
    benchmarks = []
    runtime_impacts = []
    decisions = []
    package_root = reference_package_root or (
        Path(__file__).resolve().parents[4] / "models/reference_tasks/hybrid_decoder"
    )
    impact_config = runtime_impact_config or derive_runtime_impact_config(seed)
    if impact_config.synthesis_seed != seed:
        raise ValueError("kernel demo and runtime-impact synthesis seeds must match")
    for candidate_index, (candidate, source) in enumerate(generated):
        candidate_root = output_root / "candidates" / candidate.candidate_id
        verification = execute_correctness(
            candidate,
            source,
            cases,
            output_root=candidate_root / "correctness",
            seed=seed,
            timeout_seconds=min(config.wall_time_seconds, 15.0),
        )
        benchmark = benchmark_candidate(
            candidate,
            source,
            verification,
            output_root=candidate_root / "benchmark",
            config=config,
        )
        if candidate_index == 0 and verification.status.value == "passed":
            impact = benchmark_generated_runtime_impact(
                candidate,
                source,
                verification,
                reference_package_root=package_root,
                output_root=candidate_root / "runtime-impact",
                config=impact_config,
            )
            runtime_impacts.append(impact)
            decision = decide_with_runtime_impact(
                verification,
                benchmark,
                impact,
                artifact_root=candidate_root / "runtime-impact",
            )
        else:
            decision = decide_candidate(verification, benchmark)
        correctness.append(verification)
        benchmarks.append(benchmark)
        decisions.append(decision)
        (candidate_root / "candidate.json").write_text(
            candidate.model_dump_json(indent=2), encoding="utf-8"
        )
        (candidate_root / "correctness.json").write_text(
            verification.model_dump_json(indent=2), encoding="utf-8"
        )
        (candidate_root / "benchmark.json").write_text(
            benchmark.model_dump_json(indent=2), encoding="utf-8"
        )
        raw_path = candidate_root / "raw_samples.jsonl"
        raw_path.write_bytes(raw_samples_bytes(benchmark.raw_samples))
        if (
            benchmark.raw_samples_sha256 is not None
            and hashlib.sha256(raw_path.read_bytes()).hexdigest() != benchmark.raw_samples_sha256
        ):
            raise RuntimeError("persisted kernel raw samples do not match their report")
        validate_benchmark_report(benchmark)
    report = KernelLabReport(
        evidence=evidence,
        operator_schema=hybrid_quantized_state_schema(),
        candidates=tuple(item[0] for item in generated),
        correctness=tuple(correctness),
        benchmarks=tuple(benchmarks),
        runtime_impacts=tuple(runtime_impacts),
        decisions=tuple(decisions),
        triton_adapter=triton_adapter_status(),
    )
    (output_root / "kernel_lab_report.json").write_text(
        report.model_dump_json(indent=2), encoding="utf-8"
    )
    return report
