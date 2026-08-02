from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import pytest
from pydantic import ValidationError

from sloforge.forgeci import (
    BenchmarkInput,
    BenchmarkMatrix,
    BenchmarkSpec,
    CommandSpec,
    ComparisonClassification,
    HardwareRequirement,
    MatrixCase,
    MetricDirection,
    MetricSpec,
    bisect_regression,
    compare_runs,
    create_regression_fixture,
    load_matrix,
    minimize_benchmark,
    render_upstream_issue,
    run_case,
    run_matrix,
    verify_run_artifact,
)


def _git(repository: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=True,
        capture_output=True,
        text=True,
        timeout=10.0,
    )
    return completed.stdout.strip()


def _case(repository: Path, revision: str, *, repetitions: int = 7) -> MatrixCase:
    return MatrixCase(
        case_id="synthetic-latency",
        repository=str(repository),
        revision=revision,
        hardware=HardwareRequirement(
            architecture="cpu",
            minimum_cpu_cores=1,
            minimum_memory_gib=0.5,
            gpu_count=0,
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
                workload="mixed-bursty",
                physical_plan="physical-plan.json",
                prompt_tokens=256,
                output_tokens=64,
                concurrency=8,
                runtime_arguments=("--use-collective", "ring", "--verbose"),
            ),
            warmup_trials=2,
            repetitions=repetitions,
            maximum_repetitions=21,
            failed_trial_retries=1,
            bootstrap_rounds=500,
            seed=2026,
        ),
        budget_seconds=120.0,
    )


def test_matrix_loader_is_versioned_and_strict() -> None:
    root = Path(__file__).parents[2]
    matrix = load_matrix(root / "tests/fixtures/forgeci/runtime-compatibility.yaml")
    assert matrix.schema_version == "1.0.0"
    assert matrix.cases[0].benchmark.warmup_trials == 2
    invalid = matrix.model_dump()
    invalid["unexpected"] = True
    with pytest.raises(ValidationError):
        type(matrix).model_validate(invalid)


def test_runner_separates_warmup_and_detects_two_metric_regression(tmp_path: Path) -> None:
    repository = tmp_path / "fixture"
    good, _, bad = create_regression_fixture(repository)
    case = _case(repository, good)

    _git(repository, "checkout", "--detach", "--force", good)
    baseline = run_case(case, checkout=repository, output_directory=tmp_path / "runs")
    _git(repository, "checkout", "--detach", "--force", bad)
    candidate = run_case(
        case.model_copy(update={"revision": bad}),
        checkout=repository,
        output_directory=tmp_path / "runs",
    )
    comparison = compare_runs(baseline, candidate, case.benchmark)

    assert baseline.success
    assert len(baseline.warmups) == 2
    assert len(baseline.trials) == 7
    assert all(summary.sample_count == 7 for summary in baseline.summaries)
    assert comparison.classification == ComparisonClassification.REGRESSION
    assert {metric.classification for metric in comparison.metrics} == {
        ComparisonClassification.REGRESSION
    }
    assert comparison.metrics[0].degradation_ci_low_percent > 5.0
    assert Path(baseline.artifact_directory, "run.json").is_file()
    assert len(baseline.artifact_hash) == 64
    assert verify_run_artifact(baseline)
    raw_log = next(Path(baseline.artifact_directory).rglob("*.stdout"))
    raw_log.write_text(raw_log.read_text(encoding="utf-8") + "tampered\n", encoding="utf-8")
    assert not verify_run_artifact(baseline)


def test_runner_retries_and_redacts_sensitive_values(tmp_path: Path) -> None:
    repository = tmp_path / "retry"
    repository.mkdir()
    script = repository / "benchmark.py"
    script.write_text(
        """import json
import os
import pathlib
import sys

marker = pathlib.Path(f"attempt-{os.environ['SLOFORGE_FORGECI_PHASE']}-{os.environ['SLOFORGE_FORGECI_TRIAL']}")
print(os.environ["FIXTURE_SECRET"])
if not marker.exists():
    marker.write_text("failed", encoding="utf-8")
    sys.exit(3)
print(json.dumps({"p99_ttft_ms": 10.0}))
""",
        encoding="utf-8",
    )
    from sloforge.forgeci.models import EnvironmentSpec, EnvironmentVariable

    case = MatrixCase(
        case_id="retry-redaction",
        repository=str(repository),
        revision="fixture",
        environment=EnvironmentSpec(
            variables=(
                EnvironmentVariable(name="FIXTURE_SECRET", value="never-log-me", sensitive=True),
            )
        ),
        hardware=HardwareRequirement(
            architecture="cpu", minimum_cpu_cores=1, minimum_memory_gib=0.5
        ),
        benchmark=BenchmarkSpec(
            command=CommandSpec(executable=sys.executable, arguments=("benchmark.py",)),
            metrics=(
                MetricSpec(
                    name="p99_ttft_ms",
                    unit="ms",
                    direction=MetricDirection.LOWER_IS_BETTER,
                ),
            ),
            warmup_trials=1,
            repetitions=3,
            maximum_repetitions=3,
            failed_trial_retries=1,
            bootstrap_rounds=200,
        ),
    )
    run = run_case(case, checkout=repository, output_directory=tmp_path / "runs")
    assert run.success
    assert len(run.trials) == 6
    assert any("trial attempts failed" in warning for warning in run.warnings)
    logged = "\n".join(
        path.read_text(encoding="utf-8")
        for path in Path(run.artifact_directory).rglob("*")
        if path.is_file()
    )
    assert "never-log-me" not in logged
    assert "<redacted>" in logged
    assert run.environment_manifest.environment[0].value == "<redacted>"


def test_bisection_finds_intentional_fixture_commit(tmp_path: Path) -> None:
    repository = tmp_path / "fixture"
    good, first_bad, bad = create_regression_fixture(repository)
    result = bisect_regression(
        repository=repository,
        good_revision=good,
        bad_revision=bad,
        case=_case(repository, good),
        output_directory=tmp_path / "bisect",
    )
    assert result.first_regressing_commit == first_bad
    assert result.confidence >= 0.85
    assert result.steps
    assert Path(result.artifact_directory, "bisect-result.json").is_file()
    assert _git(repository, "rev-parse", "HEAD") == bad


def test_minimizer_and_upstream_report_are_evidence_linked(tmp_path: Path) -> None:
    repository = tmp_path / "fixture"
    good, first_bad, bad = create_regression_fixture(repository)
    case = _case(repository, good, repetitions=3)
    result = bisect_regression(
        repository=repository,
        good_revision=good,
        bad_revision=bad,
        case=case,
        output_directory=tmp_path / "bisect",
    )
    comparison_path = Path(result.steps[0].comparison_artifact)
    from sloforge.forgeci.models import ComparisonRecord

    comparison = ComparisonRecord.model_validate_json(comparison_path.read_text(encoding="utf-8"))
    reproducer = minimize_benchmark(
        case.benchmark,
        good_commit=good,
        bad_commit=bad,
        hardware=case.hardware,
        preserves_regression=lambda benchmark: (
            benchmark.input.prompt_tokens >= 32
            and benchmark.input.output_tokens >= 8
            and benchmark.input.concurrency >= 2
            and "ring" in benchmark.input.runtime_arguments
        ),
        expected_regression="p99 TTFT increases by more than 5%",
        confidence_interval="95% bootstrap interval excludes zero",
        artifact_references=(str(comparison_path), str(tmp_path / "bisect/bisect-result.json")),
    )
    assert reproducer.benchmark.input.prompt_tokens == 32
    assert reproducer.benchmark.input.output_tokens == 8
    assert reproducer.benchmark.input.concurrency == 2
    assert reproducer.benchmark.input.runtime_arguments == ("ring",)
    report = render_upstream_issue(
        title="Deterministic fixture latency regression",
        comparison=comparison,
        bisection=result,
        reproducer=reproducer,
    )
    assert first_bad in report
    assert "change point, not a proven source-code cause" in report
    assert str(comparison_path) in report
    assert "Cliff's δ" in report


def test_metric_parser_rejects_hard_coded_or_malformed_output(tmp_path: Path) -> None:
    repository = tmp_path / "bad-output"
    repository.mkdir()
    (repository / "benchmark.py").write_text("print('not-json')\n", encoding="utf-8")
    case = _case(repository, "fixture", repetitions=3)
    failed = run_case(case, checkout=repository, output_directory=tmp_path / "runs")
    assert not failed.success
    assert not failed.summaries
    errors = [trial.error for trial in failed.trials]
    assert all(error is not None for error in errors)
    payload = json.loads(Path(failed.artifact_directory, "run.json").read_text())
    assert payload["success"] is False


def test_matrix_runner_uses_isolated_local_checkout(tmp_path: Path) -> None:
    repository = tmp_path / "fixture"
    good, _, bad = create_regression_fixture(repository)
    matrix = BenchmarkMatrix(
        matrix_id="isolated-fixture",
        cases=(_case(repository, good, repetitions=3),),
    )
    result = run_matrix(
        matrix,
        output_directory=tmp_path / "matrix-run",
        repository_base=tmp_path,
    )
    assert result.success
    assert result.runs[0].revision == good
    assert Path(result.artifact_directory, "matrix-run.json").is_file()
    assert _git(repository, "rev-parse", "HEAD") == bad


def test_timeout_terminates_benchmark_process_group(tmp_path: Path) -> None:
    repository = tmp_path / "timeout"
    repository.mkdir()
    (repository / "benchmark.py").write_text(
        """import subprocess
import sys
import time

subprocess.Popen([sys.executable, "-c", "import time; from pathlib import Path; time.sleep(.5); Path('orphan-marker').write_text('orphan')"])
time.sleep(5)
""",
        encoding="utf-8",
    )
    base = _case(repository, "fixture", repetitions=3)
    command = base.benchmark.command.model_copy(update={"timeout_seconds": 0.05})
    benchmark = base.benchmark.model_copy(
        update={"command": command, "failed_trial_retries": 0, "warmup_trials": 0}
    )
    run = run_case(
        base.model_copy(update={"benchmark": benchmark}),
        checkout=repository,
        output_directory=tmp_path / "runs",
    )
    assert not run.success
    assert all(trial.status.value == "timed_out" for trial in run.trials)
    time.sleep(0.6)
    assert not (repository / "orphan-marker").exists()


def test_gpu_requirement_never_falls_back_to_cpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sloforge.forgeci.hardware as hardware_module

    monkeypatch.setattr(hardware_module, "_gpus", lambda: ())
    from sloforge.forgeci import EnvironmentSpec, RequirementMismatch, validate_requirements

    with pytest.raises(RequirementMismatch, match="CPU fallback disabled"):
        validate_requirements(
            HardwareRequirement(
                architecture="cpu",
                minimum_cpu_cores=1,
                minimum_memory_gib=0.1,
                gpu_count=1,
                gpu_model="fixture-gpu",
            ),
            EnvironmentSpec(),
        )
