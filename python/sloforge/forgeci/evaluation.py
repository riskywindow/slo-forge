"""CPU-only ForgeCI fixture evaluation derived entirely from live artifacts."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from sloforge.forgeci.bisect import bisect_regression
from sloforge.forgeci.fixture import create_regression_fixture
from sloforge.forgeci.minimize import minimize_benchmark
from sloforge.forgeci.models import (
    BenchmarkInput,
    BenchmarkSpec,
    BisectStep,
    CommandSpec,
    ComparisonClassification,
    ComparisonRecord,
    ForgeCIEvaluation,
    HardwareRequirement,
    MatrixCase,
    MetricDirection,
    MetricSpec,
)
from sloforge.forgeci.report import render_upstream_issue
from sloforge.forgeci.runner import compare_runs, run_case


def _git(repository: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repository), *arguments],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=10.0,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip())
    return completed.stdout.strip()


def fixture_case(repository: Path, revision: str) -> MatrixCase:
    """Return the documented two-metric fixture benchmark."""

    return MatrixCase(
        case_id="forgeci-intentional-regression",
        repository=str(repository),
        revision=revision,
        hardware=HardwareRequirement(
            architecture="cpu", minimum_cpu_cores=1, minimum_memory_gib=0.25
        ),
        benchmark=BenchmarkSpec(
            command=CommandSpec(
                executable=sys.executable,
                arguments=("benchmark.py",),
                timeout_seconds=5.0,
            ),
            metrics=(
                MetricSpec(
                    name="p99_ttft_ms",
                    unit="ms",
                    direction=MetricDirection.LOWER_IS_BETTER,
                    practical_threshold_percent=5.0,
                    maximum_noise_percent=3.0,
                ),
                MetricSpec(
                    name="throughput_rps",
                    unit="request/s",
                    direction=MetricDirection.HIGHER_IS_BETTER,
                    practical_threshold_percent=5.0,
                    maximum_noise_percent=3.0,
                ),
            ),
            input=BenchmarkInput(
                model="synthetic-moe",
                workload="long-context",
                prompt_tokens=256,
                output_tokens=64,
                concurrency=8,
                runtime_arguments=("--collective", "ring", "--diagnostics"),
            ),
            warmup_trials=2,
            repetitions=7,
            maximum_repetitions=21,
            failed_trial_retries=1,
            bootstrap_rounds=500,
            seed=2026,
        ),
        budget_seconds=120.0,
    )


def _comparison_for_commit(
    bisection_steps: tuple[BisectStep, ...], commit: str
) -> ComparisonRecord:
    matching = [step for step in bisection_steps if step.commit == commit]
    if not matching:
        raise RuntimeError("bisection did not preserve comparison evidence for identified commit")
    return ComparisonRecord.model_validate_json(
        Path(matching[-1].comparison_artifact).read_text(encoding="utf-8")
    )


def run_fixture_evaluation(output_directory: Path, report_path: Path) -> ForgeCIEvaluation:
    """Execute detection, bisection, minimization, and issue generation locally."""

    output_directory.mkdir(parents=True, exist_ok=False)
    repository = output_directory / "fixture-repository"
    good, expected_first_bad, bad = create_regression_fixture(repository)
    case = fixture_case(repository, good)
    bisection = bisect_regression(
        repository=repository,
        good_revision=good,
        bad_revision=bad,
        case=case,
        output_directory=output_directory / "bisect",
    )
    comparison = _comparison_for_commit(bisection.steps, expected_first_bad)

    minimization_dir = output_directory / "minimization"
    minimization_dir.mkdir()
    counter = 0

    def preserves(benchmark: BenchmarkSpec) -> bool:
        nonlocal counter
        trial_dir = minimization_dir / f"candidate-{counter:03d}"
        counter += 1
        _git(repository, "checkout", "--detach", "--force", good)
        baseline = run_case(
            case.model_copy(update={"benchmark": benchmark, "revision": good}),
            checkout=repository,
            output_directory=trial_dir / "runs",
            repetitions=3,
        )
        _git(repository, "checkout", "--detach", "--force", bad)
        candidate = run_case(
            case.model_copy(update={"benchmark": benchmark, "revision": bad}),
            checkout=repository,
            output_directory=trial_dir / "runs",
            repetitions=3,
        )
        result = compare_runs(baseline, candidate, benchmark)
        (trial_dir / "comparison.json").write_text(
            result.model_dump_json(indent=2) + "\n", encoding="utf-8"
        )
        return result.classification == ComparisonClassification.REGRESSION

    reproducer = minimize_benchmark(
        case.benchmark,
        good_commit=good,
        bad_commit=bad,
        hardware=case.hardware,
        preserves_regression=preserves,
        expected_regression="both p99 TTFT and throughput degrade by at least 5%",
        confidence_interval="family-wise 95% bootstrap intervals exclude zero",
        artifact_references=(
            str(output_directory / "bisect/bisect-result.json"),
            str(Path(bisection.steps[-1].comparison_artifact)),
            str(minimization_dir),
        ),
    )
    issue = render_upstream_issue(
        title="Fixture runtime performance regression",
        comparison=comparison,
        bisection=bisection,
        reproducer=reproducer,
    )
    issue_path = output_directory / "upstream-issue.md"
    issue_path.write_text(issue, encoding="utf-8")
    commits_in_range = len(_git(repository, "rev-list", f"{good}..{bad}").splitlines()) + 1
    unique_evaluated = len({step.commit for step in bisection.steps})
    inconclusive_rate = len(set(bisection.inconclusive_commits)) / max(unique_evaluated, 1)
    evaluation = ForgeCIEvaluation(
        expected_first_regressing_commit=expected_first_bad,
        identified_first_regressing_commit=bisection.first_regressing_commit,
        bisection_correct=bisection.first_regressing_commit == expected_first_bad,
        commits_in_range=commits_in_range,
        unique_commits_evaluated=unique_evaluated,
        inconclusive_rate=inconclusive_rate,
        comparison=comparison,
        bisection=bisection,
        reproducer=reproducer,
        report_path=str(report_path),
        issue_path=str(issue_path),
    )
    evaluation_path = output_directory / "evaluation.json"
    evaluation_path.write_text(evaluation.model_dump_json(indent=2) + "\n", encoding="utf-8")
    latency = next(metric for metric in comparison.metrics if metric.metric == "p99_ttft_ms")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        f"""# ForgeCI fixture evaluation

This CPU-only evaluation was generated from `{evaluation_path}`. It creates and measures
a local four-commit repository; no expected performance samples are embedded in ForgeCI.

| Result | Measured value |
|---|---:|
| First-regression bisection | {"correct" if evaluation.bisection_correct else "incorrect"} |
| Unique candidate commits evaluated | {evaluation.unique_commits_evaluated} |
| Inconclusive rate | {evaluation.inconclusive_rate:.1%} |
| P99 TTFT change | {latency.degradation_percent:+.2f}% |
| P99 TTFT corrected interval | [{latency.degradation_ci_low_percent:+.2f}%, {latency.degradation_ci_high_percent:+.2f}%] |
| Noise floor | {latency.noise_floor_percent:.2f}% |

Warmups are excluded, trials are repeated, and regression classification requires both
practical significance and a family-wise corrected bootstrap interval excluding zero.
The deterministic fixture is an integration validation, not a claim about external GPU
runtimes or hardware.
""",
        encoding="utf-8",
    )
    return evaluation
