"""Deterministic executable Genesis red-team demonstration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sloforge.genesis.ir import write_canonical

from .fixture import UnsafeStreamingCandidate, unsafe_benchmark_comparison
from .models import RedTeamConfiguration, RedTeamDemoResult
from .runner import corpus_from_report, replay_regression_corpus, run_red_team


def run_demo(output_directory: Path, *, seed: int = 73129) -> RedTeamDemoResult:
    target = UnsafeStreamingCandidate()
    report = run_red_team(
        target=target,
        configuration=RedTeamConfiguration(
            seed=seed,
            maximum_findings=32,
            maximum_minimization_evaluations=128,
            minimization_timeout_seconds=2.0,
            run_timeout_seconds=15.0,
            tensor_cases=12,
            schedule_cases=12,
            topology_cases=8,
            resource_cases=8,
        ),
        benchmark_comparison=unsafe_benchmark_comparison(),
    )
    corpus = corpus_from_report(report)
    replay = replay_regression_corpus(
        target=target,
        corpus=corpus,
        benchmark_comparison=unsafe_benchmark_comparison(),
    )
    output_directory.mkdir(parents=True, exist_ok=True)
    report_path = output_directory / "redteam-report.json"
    corpus_path = output_directory / "regression-corpus.json"
    write_canonical(report, report_path)
    write_canonical(corpus, corpus_path)
    counterexample_directory = output_directory / "counterexamples"
    constraint_directory = output_directory / "constraints"
    counterexample_paths: list[str] = []
    constraint_paths: list[str] = []
    for finding in report.findings:
        counterexample_path = (
            counterexample_directory / f"{finding.counterexample.counterexample_id}.json"
        )
        constraint_path = constraint_directory / f"{finding.learned_constraint.constraint_id}.json"
        write_canonical(finding.counterexample, counterexample_path)
        write_canonical(finding.learned_constraint, constraint_path)
        counterexample_paths.append(str(counterexample_path))
        constraint_paths.append(str(constraint_path))
    return RedTeamDemoResult(
        seed=seed,
        report_path=str(report_path),
        corpus_path=str(corpus_path),
        counterexample_paths=tuple(counterexample_paths),
        constraint_paths=tuple(constraint_paths),
        finding_count=len(report.findings),
        reproduced_regressions=sum(result.reproduced for result in replay),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=73129)
    parser.add_argument("--candidate", default="unsafe-fastpath-v1")
    arguments = parser.parse_args(argv)
    if arguments.seed < 0:
        parser.error("--seed must be non-negative")
    if arguments.candidate != UnsafeStreamingCandidate.descriptor.candidate_id:
        parser.error("the deterministic fixture supports only unsafe-fastpath-v1")
    result = run_demo(arguments.output, seed=arguments.seed)
    print(json.dumps(result.model_dump(mode="json"), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main", "run_demo"]
