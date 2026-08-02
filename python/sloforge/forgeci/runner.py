"""Bounded local ForgeCI runner with artifact-complete trial recording."""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import signal
import subprocess
import sys
import time
import uuid
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path

from sloforge.forgeci.hardware import validate_requirements
from sloforge.forgeci.models import (
    BenchmarkSpec,
    CommandSpec,
    ComparisonClassification,
    ComparisonRecord,
    EnvironmentManifest,
    EnvironmentVariable,
    MatrixCase,
    MetricComparison,
    MetricSummary,
    MetricValue,
    RunRecord,
    TrialRecord,
    TrialStatus,
)
from sloforge.forgeci.statistics import compare_metric, summarize_metric


class CommandOutcome:
    """Internal immutable command result."""

    __slots__ = ("duration_seconds", "exit_code", "stderr", "stdout", "timed_out")

    def __init__(
        self,
        *,
        stdout: str,
        stderr: str,
        exit_code: int | None,
        duration_seconds: float,
        timed_out: bool,
    ) -> None:
        self.stdout = stdout
        self.stderr = stderr
        self.exit_code = exit_code
        self.duration_seconds = duration_seconds
        self.timed_out = timed_out


def _bounded_command(
    command: CommandSpec,
    *,
    cwd: Path,
    environment: Mapping[str, str],
) -> CommandOutcome:
    """Run without a shell and terminate the entire process group on timeout."""

    started = time.monotonic()
    creationflags = 0
    start_new_session = os.name != "nt"
    if os.name == "nt":
        creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    process = subprocess.Popen(
        [command.executable, *command.arguments],
        cwd=cwd,
        env=dict(environment),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=start_new_session,
        creationflags=creationflags,
    )
    try:
        stdout, stderr = process.communicate(timeout=command.timeout_seconds)
        return CommandOutcome(
            stdout=stdout,
            stderr=stderr,
            exit_code=process.returncode,
            duration_seconds=time.monotonic() - started,
            timed_out=False,
        )
    except subprocess.TimeoutExpired:
        if os.name != "nt":
            os.killpg(process.pid, signal.SIGTERM)
        else:
            process.terminate()
        try:
            stdout, stderr = process.communicate(timeout=2.0)
        except subprocess.TimeoutExpired:
            if os.name != "nt":
                os.killpg(process.pid, signal.SIGKILL)
            else:
                process.kill()
            stdout, stderr = process.communicate()
        return CommandOutcome(
            stdout=stdout,
            stderr=stderr,
            exit_code=None,
            duration_seconds=time.monotonic() - started,
            timed_out=True,
        )


def _parse_metrics(stdout: str, benchmark: BenchmarkSpec) -> tuple[MetricValue, ...]:
    lines = [line for line in stdout.splitlines() if line.strip()]
    if not lines:
        raise ValueError("benchmark emitted no JSON metric record")
    decoded = json.loads(lines[-1])
    if not isinstance(decoded, dict):
        raise ValueError("benchmark's final output line must be a JSON object")
    expected = {spec.name for spec in benchmark.metrics}
    if set(decoded) != expected:
        raise ValueError(
            f"benchmark metrics mismatch: expected {sorted(expected)}, got {sorted(decoded)}"
        )
    values: list[MetricValue] = []
    for name in sorted(expected):
        value = decoded[name]
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ValueError(f"metric {name} must be numeric")
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"metric {name} must be finite")
        values.append(MetricValue(name=name, value=number))
    return tuple(values)


def _environment(case: MatrixCase, phase: str, trial: int, seed: int) -> dict[str, str]:
    environment = dict(os.environ)
    environment.update(
        {
            "SLOFORGE_FORGECI_PHASE": phase,
            "SLOFORGE_FORGECI_TRIAL": str(trial),
            "SLOFORGE_FORGECI_SEED": str(seed),
            "SLOFORGE_FORGECI_PROMPT_TOKENS": str(case.benchmark.input.prompt_tokens),
            "SLOFORGE_FORGECI_OUTPUT_TOKENS": str(case.benchmark.input.output_tokens),
            "SLOFORGE_FORGECI_CONCURRENCY": str(case.benchmark.input.concurrency),
            "SLOFORGE_FORGECI_MODEL": case.benchmark.input.model,
            "SLOFORGE_FORGECI_WORKLOAD": case.benchmark.input.workload,
            "SLOFORGE_FORGECI_RUNTIME_ARGUMENTS": json.dumps(
                case.benchmark.input.runtime_arguments, separators=(",", ":")
            ),
        }
    )
    if case.benchmark.input.physical_plan is not None:
        environment["SLOFORGE_FORGECI_PHYSICAL_PLAN"] = case.benchmark.input.physical_plan
    environment.update({entry.name: entry.value for entry in case.environment.variables})
    return environment


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


def _redact_output(value: str, case: MatrixCase) -> str:
    redacted = value
    for entry in case.environment.variables:
        if entry.sensitive and entry.value:
            redacted = redacted.replace(entry.value, "<redacted>")
    return redacted


def _run_trial(
    *,
    case: MatrixCase,
    checkout: Path,
    artifact_dir: Path,
    phase: str,
    trial_index: int,
    seed: int,
    deadline: float,
) -> tuple[TrialRecord, ...]:
    records: list[TrialRecord] = []
    for attempt in range(case.benchmark.failed_trial_retries + 1):
        remaining = deadline - time.monotonic()
        if remaining <= 0.0:
            raise TimeoutError(f"case {case.case_id} exceeded benchmark budget")
        command = case.benchmark.command.model_copy(
            update={"timeout_seconds": min(case.benchmark.command.timeout_seconds, remaining)}
        )
        outcome = _bounded_command(
            command,
            cwd=checkout,
            environment=_environment(case, phase, trial_index, seed),
        )
        stem = f"{phase}-{trial_index:04d}-attempt-{attempt}"
        stdout_path = artifact_dir / "trials" / f"{stem}.stdout"
        stderr_path = artifact_dir / "trials" / f"{stem}.stderr"
        _write_text(stdout_path, _redact_output(outcome.stdout, case))
        _write_text(stderr_path, _redact_output(outcome.stderr, case))
        metrics: tuple[MetricValue, ...] = ()
        error: str | None = None
        status = TrialStatus.TIMED_OUT if outcome.timed_out else TrialStatus.FAILED
        if outcome.exit_code == 0 and not outcome.timed_out:
            try:
                metrics = _parse_metrics(outcome.stdout, case.benchmark)
                status = TrialStatus.SUCCEEDED
            except (ValueError, json.JSONDecodeError) as failure:
                error = str(failure)
        elif outcome.timed_out:
            error = f"timed out after {case.benchmark.command.timeout_seconds:.3f}s"
        else:
            error = f"benchmark exited with status {outcome.exit_code}"
        record = TrialRecord(
            trial_index=trial_index,
            attempt=attempt,
            phase="warmup" if phase == "warmup" else "measurement",
            seed=seed,
            status=status,
            duration_seconds=outcome.duration_seconds,
            metrics=metrics,
            stdout_path=str(stdout_path),
            stderr_path=str(stderr_path),
            exit_code=outcome.exit_code,
            error=error,
        )
        records.append(record)
        if status == TrialStatus.SUCCEEDED:
            return tuple(records)
    return tuple(records)


def _tree_hash(directory: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in directory.rglob("*") if item.is_file()):
        if path.name == "run.json":
            continue
        digest.update(path.relative_to(directory).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _manifest(case: MatrixCase, revision: str, hardware_fingerprint: str) -> EnvironmentManifest:
    recorded = tuple(
        EnvironmentVariable(
            name=entry.name,
            value="<redacted>" if entry.sensitive else entry.value,
            sensitive=entry.sensitive,
        )
        for entry in case.environment.variables
    )
    return EnvironmentManifest(
        platform=platform.platform(),
        python_version=sys.version.split()[0],
        machine=platform.machine() or "unknown",
        processor=platform.processor(),
        git_commit=revision,
        environment=recorded,
        hardware_fingerprint=hardware_fingerprint,
    )


def _redacted_case(case: MatrixCase) -> MatrixCase:
    variables = tuple(
        entry.model_copy(update={"value": "<redacted>"}) if entry.sensitive else entry
        for entry in case.environment.variables
    )
    environment = case.environment.model_copy(update={"variables": variables})
    return case.model_copy(update={"environment": environment})


def run_case(
    case: MatrixCase,
    *,
    checkout: Path,
    output_directory: Path,
    repetitions: int | None = None,
) -> RunRecord:
    """Build and run one matrix case under a hard wall-time budget."""

    started_at = datetime.now(UTC)
    observed_hardware = validate_requirements(case.hardware, case.environment)
    started = time.monotonic()
    deadline = started + case.budget_seconds
    count = repetitions if repetitions is not None else case.benchmark.repetitions
    if count < 3 or count > case.benchmark.maximum_repetitions:
        raise ValueError("repetitions outside benchmark limits")
    run_id = f"{case.case_id}-{case.revision[:12]}-{uuid.uuid4().hex[:10]}"
    artifact_dir = output_directory / run_id
    artifact_dir.mkdir(parents=True, exist_ok=False)

    build_environment = _environment(case, "warmup", 0, case.benchmark.seed)
    warnings: list[str] = []
    build_failed = False
    for index, command in enumerate(case.build):
        remaining = deadline - time.monotonic()
        if remaining <= 0.0:
            raise TimeoutError(f"case {case.case_id} exceeded budget during build")
        bounded = command.model_copy(
            update={"timeout_seconds": min(command.timeout_seconds, remaining)}
        )
        outcome = _bounded_command(bounded, cwd=checkout, environment=build_environment)
        _write_text(
            artifact_dir / "build" / f"{index:03d}.stdout",
            _redact_output(outcome.stdout, case),
        )
        _write_text(
            artifact_dir / "build" / f"{index:03d}.stderr",
            _redact_output(outcome.stderr, case),
        )
        if outcome.timed_out or outcome.exit_code != 0:
            warnings.append(f"build command {index} failed with status {outcome.exit_code}")
            build_failed = True
            break

    warmups: list[TrialRecord] = []
    trials: list[TrialRecord] = []
    if not build_failed:
        for phase, total, target in (
            ("warmup", case.benchmark.warmup_trials, warmups),
            ("measurement", count, trials),
        ):
            for index in range(total):
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"case {case.case_id} exceeded benchmark budget")
                attempts = _run_trial(
                    case=case,
                    checkout=checkout,
                    artifact_dir=artifact_dir,
                    phase=phase,
                    trial_index=index,
                    seed=case.benchmark.seed + index,
                    deadline=deadline,
                )
                target.extend(attempts)

    successful = [trial for trial in trials if trial.status == TrialStatus.SUCCEEDED]
    summaries: list[MetricSummary] = []
    if len(successful) >= 3:
        for metric_index, metric in enumerate(case.benchmark.metrics):
            samples = tuple(
                next(value.value for value in trial.metrics if value.name == metric.name)
                for trial in successful
            )
            summaries.append(
                summarize_metric(
                    metric,
                    samples,
                    bootstrap_rounds=case.benchmark.bootstrap_rounds,
                    seed=case.benchmark.seed + metric_index * 1009,
                )
            )
    else:
        warnings.append(
            f"only {len(successful)} successful measurement trials; at least three are required"
        )
    failed_attempts = sum(trial.status != TrialStatus.SUCCEEDED for trial in (*warmups, *trials))
    if failed_attempts:
        warnings.append(f"{failed_attempts} trial attempts failed or timed out before completion")

    _write_text(
        artifact_dir / "case.json",
        _redacted_case(case).model_dump_json(indent=2),
    )
    record = RunRecord(
        run_id=run_id,
        case_id=case.case_id,
        revision=case.revision,
        started_at=started_at,
        completed_at=datetime.now(UTC),
        success=len(successful) == count and bool(summaries),
        warmups=tuple(warmups),
        trials=tuple(trials),
        summaries=tuple(summaries),
        environment_manifest=_manifest(case, case.revision, observed_hardware.fingerprint),
        artifact_directory=str(artifact_dir),
        artifact_hash=_tree_hash(artifact_dir),
        warnings=tuple(warnings),
    )
    _write_text(artifact_dir / "run.json", record.model_dump_json(indent=2))
    return record


def compare_runs(
    baseline: RunRecord,
    candidate: RunRecord,
    benchmark: BenchmarkSpec,
) -> ComparisonRecord:
    """Classify all metrics with a family-wise Bonferroni correction."""

    if not baseline.success or not candidate.success:
        classification = ComparisonClassification.FAILED
        return ComparisonRecord(
            comparison_id=f"{baseline.run_id}--{candidate.run_id}",
            baseline_run_id=baseline.run_id,
            candidate_run_id=candidate.run_id,
            classification=classification,
            metrics=(),
            warnings=("one or both benchmark runs failed",),
        )
    baseline_by_name = {summary.name: summary for summary in baseline.summaries}
    candidate_by_name = {summary.name: summary for summary in candidate.summaries}
    comparisons: list[MetricComparison] = []
    for index, metric in enumerate(benchmark.metrics):
        corrected_alpha = metric.significance_level / len(benchmark.metrics)
        comparisons.append(
            compare_metric(
                metric,
                baseline_by_name[metric.name],
                candidate_by_name[metric.name],
                corrected_alpha=corrected_alpha,
                bootstrap_rounds=benchmark.bootstrap_rounds,
                seed=benchmark.seed + 7919 * index,
            )
        )
    classes = {item.classification for item in comparisons}
    if ComparisonClassification.FAILED in classes:
        overall = ComparisonClassification.FAILED
    elif ComparisonClassification.FLAKY in classes:
        overall = ComparisonClassification.FLAKY
    elif ComparisonClassification.REGRESSION in classes:
        overall = ComparisonClassification.REGRESSION
    elif ComparisonClassification.INCONCLUSIVE in classes:
        overall = ComparisonClassification.INCONCLUSIVE
    elif classes == {ComparisonClassification.IMPROVEMENT} or (
        ComparisonClassification.IMPROVEMENT in classes
        and classes <= {ComparisonClassification.IMPROVEMENT, ComparisonClassification.UNCHANGED}
    ):
        overall = ComparisonClassification.IMPROVEMENT
    else:
        overall = ComparisonClassification.UNCHANGED
    return ComparisonRecord(
        comparison_id=f"{baseline.run_id}--{candidate.run_id}",
        baseline_run_id=baseline.run_id,
        candidate_run_id=candidate.run_id,
        classification=overall,
        metrics=tuple(comparisons),
    )


def write_comparison(record: ComparisonRecord, path: Path) -> None:
    _write_text(path, record.model_dump_json(indent=2))


def verify_run_artifact(record: RunRecord) -> bool:
    """Verify that raw logs and case inputs still match the recorded evidence hash."""

    directory = Path(record.artifact_directory)
    return directory.is_dir() and _tree_hash(directory) == record.artifact_hash
