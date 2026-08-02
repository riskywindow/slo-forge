"""Artifact-producing deterministic CPU kernel-lab demonstration."""

from __future__ import annotations

from pathlib import Path

from .benchmark import benchmark_candidate, decide_candidate, raw_samples_bytes
from .cases import generate_correctness_cases
from .evidence import validate_bottleneck_evidence
from .executor import execute_correctness
from .generator import generate_candidates, hybrid_quantized_state_schema
from .models import BottleneckEvidence, KernelBenchmarkConfig, KernelLabReport
from .triton import triton_adapter_status


def run_kernel_lab_demo(
    *,
    evidence: BottleneckEvidence,
    output_root: Path,
    seed: int,
    candidate_limit: int = 2,
    benchmark_config: KernelBenchmarkConfig | None = None,
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
    decisions = []
    for candidate, source in generated:
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
    report = KernelLabReport(
        evidence=evidence,
        operator_schema=hybrid_quantized_state_schema(),
        candidates=tuple(item[0] for item in generated),
        correctness=tuple(correctness),
        benchmarks=tuple(benchmarks),
        decisions=tuple(decisions),
        triton_adapter=triton_adapter_status(),
    )
    (output_root / "kernel_lab_report.json").write_text(
        report.model_dump_json(indent=2), encoding="utf-8"
    )
    return report
