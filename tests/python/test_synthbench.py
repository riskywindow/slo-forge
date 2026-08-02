from __future__ import annotations

import json
import statistics
from pathlib import Path

import pytest
from typer.testing import CliRunner

import sloforge.synthbench.runner as synthbench_runner
from sloforge.cli.main import app
from sloforge.genesis.frontend import inspect_reference_package
from sloforge.genesis.ir import canonical_json
from sloforge.synthbench import (
    BaselineKind,
    BaselineStatus,
    BlockKind,
    CpuRunConfiguration,
    GrammarConfiguration,
    RawCpuSample,
    SynthBenchReport,
    audit_raw_samples,
    audit_special_casing,
    generate_tasks,
    load_hidden_cases,
    run_cpu_benchmark,
    run_synthbench_demo,
    validate_cpu_benchmark_report,
)

runner = CliRunner()


def _configuration(*, seed: int = 73129, count: int = 2) -> GrammarConfiguration:
    return GrammarConfiguration(
        seed=seed,
        count=count,
        minimum_blocks=4,
        maximum_blocks=5,
        public_cases_per_task=3,
        hidden_cases_per_task=6,
    )


def _tree_bytes(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_seeded_task_generation_is_byte_deterministic_and_hidden_is_separate(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    left = generate_tasks(_configuration(), first)
    right = generate_tasks(_configuration(), second)

    assert left == right
    assert _tree_bytes(first) == _tree_bytes(second)
    for descriptor in left:
        task = first / descriptor.task_id
        public_files = {path.name for path in (task / "public").iterdir()}
        assert "hidden_cases.jsonl" not in public_files
        cases = load_hidden_cases(task, descriptor)
        assert len(cases) == 6
        commitment = json.loads((task / "public" / "final_holdout_commitment.json").read_text())
        assert commitment["value"] == descriptor.hidden_commitment
        inspection = inspect_reference_package(task / "public")
        assert not inspection.has_unsupported_behavior
        assert inspection.package_hash


def test_task_grammar_exposes_every_declared_architecture_block(tmp_path: Path) -> None:
    descriptors = generate_tasks(
        GrammarConfiguration(
            seed=17,
            count=len(BlockKind),
            minimum_blocks=4,
            maximum_blocks=4,
            public_cases_per_task=2,
            hidden_cases_per_task=3,
        ),
        tmp_path / "tasks",
    )

    observed = {
        block.kind for descriptor in descriptors for block in descriptor.architecture.blocks
    }
    assert observed == set(BlockKind)


def test_hidden_commitment_detects_evaluator_case_tampering(tmp_path: Path) -> None:
    descriptor = generate_tasks(_configuration(count=1), tmp_path / "tasks")[0]
    task = tmp_path / "tasks" / descriptor.task_id
    hidden = task / descriptor.hidden_cases_path
    hidden.write_bytes(hidden.read_bytes() + b"\n")

    with pytest.raises(ValueError, match="hidden evaluation cases"):
        load_hidden_cases(task, descriptor)


def test_runner_rejects_public_package_tampering(tmp_path: Path) -> None:
    descriptor = generate_tasks(_configuration(count=1), tmp_path / "tasks")[0]
    task = tmp_path / "tasks" / descriptor.task_id
    reference = task / "public/reference.py"
    reference.write_text(reference.read_text(encoding="utf-8") + "\n# changed\n", encoding="utf-8")

    with pytest.raises(ValueError, match="public reference package"):
        run_cpu_benchmark(
            (task,),
            CpuRunConfiguration(seeds=(1, 2), repetitions=2, maximum_tasks=1),
            tmp_path / "run",
        )


def test_special_case_detector_rejects_task_seed_commitment_and_hidden_shapes(
    tmp_path: Path,
) -> None:
    descriptor = generate_tasks(_configuration(count=1), tmp_path / "tasks")[0]
    task = tmp_path / "tasks" / descriptor.task_id
    hidden = load_hidden_cases(task, descriptor)
    clean = tmp_path / "clean.py"
    clean.write_text("def schedule(length):\n    return length % 4\n", encoding="utf-8")
    assert audit_special_casing(clean, descriptor, hidden).passed

    cheating = tmp_path / "cheating.py"
    cheating.write_text(
        "TASK = "
        + repr(descriptor.task_id)
        + "\nSEED = "
        + repr(descriptor.seed)
        + "\nCOMMITMENT = "
        + repr(descriptor.hidden_commitment)
        + "\ndef schedule(length):\n"
        + "    if length == 7 or length == 11:\n"
        + "        return 1\n"
        + "    return 0\n",
        encoding="utf-8",
    )
    audit = audit_special_casing(cheating, descriptor, hidden)

    assert not audit.passed
    assert {finding.category for finding in audit.findings} >= {
        "task_identity",
        "seed_identity",
        "hidden_commitment",
        "hidden_shape",
    }


def test_cpu_runner_executes_all_local_baselines_and_derives_report_from_raw_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    tasks_root = tmp_path / "tasks"
    descriptors = generate_tasks(_configuration(count=2), tasks_root)
    task_directories = tuple(tasks_root / descriptor.task_id for descriptor in descriptors)
    output = Path("run")
    configuration = CpuRunConfiguration(
        seeds=(101, 202),
        warmup_count=1,
        repetitions=2,
        maximum_tasks=2,
        maximum_runtime_seconds=30.0,
    )

    report = run_cpu_benchmark(task_directories, configuration, output)

    assert report.report_source == "derived_from_raw_samples"
    assert len(report.tasks) == 4
    assert report.metrics.task_count == 2
    assert report.metrics.distinct_task_seeds == 2
    assert report.metrics.valid_system_rate == 1.0
    assert report.metrics.exact_request_rate == 1.0
    assert report.metrics.observed_request_wall_seconds > 0
    for task_report in report.tasks:
        assert task_report.integrity.passed
        assert {summary.baseline for summary in task_report.baselines} == set(BaselineKind)
        assert len(task_report.hidden_evaluations) == len(BaselineKind)
        measured = [
            summary
            for summary in task_report.baselines
            if summary.status is BaselineStatus.MEASURED
        ]
        surrogates = [
            summary
            for summary in task_report.baselines
            if summary.status is BaselineStatus.SURROGATE
        ]
        unavailable = [
            summary
            for summary in task_report.baselines
            if summary.status is BaselineStatus.NOT_APPLICABLE
        ]
        assert measured and surrogates and unavailable
        assert next(
            item for item in measured if item.baseline is BaselineKind.PYTHON_EAGER_REFERENCE
        ).reason.startswith("direct dependency-free eager reference")
        assert all(
            "request-order surrogate" in item.reason
            for item in surrogates
            if item.baseline not in {BaselineKind.PYTHON_EAGER_REFERENCE, BaselineKind.GENESIS_FULL}
        )
        genesis = next(item for item in measured if item.baseline is BaselineKind.GENESIS_FULL)
        assert genesis.execution_surface == "genesis_generated_runtime"
        assert "actual generated bounded Genesis runtime" in genesis.reason
        assert Path(task_report.genesis_runtime_manifest_path).is_file()
        assert all(summary.reason for summary in unavailable)
        assert all(summary.valid_request_rate == 1.0 for summary in (*measured, *surrogates))
        assert all(
            summary.human_authored_model_specific_lines == 0 for summary in (*measured, *surrogates)
        )
        assert all(
            hidden.exact_case_rate == 1.0
            for hidden in task_report.hidden_evaluations
            if hidden.status in {BaselineStatus.MEASURED, BaselineStatus.SURROGATE}
        )
        sample_summary = measured[0]
        assert sample_summary.raw_samples_path is not None
        raw_path = Path(sample_summary.raw_samples_path)
        samples = [
            RawCpuSample.model_validate_json(line, strict=True)
            for line in raw_path.read_bytes().splitlines()
        ]
        assert all(sample.source == "measured_cpu_monotonic_clock" for sample in samples)
        assert sample_summary.median_latency_ns == statistics.median(
            sample.latency_ns for sample in samples
        )
    reloaded = SynthBenchReport.model_validate_json(
        (output / "report.json").read_bytes(), strict=True
    )
    assert reloaded == report
    sandbox_requests = tuple((tmp_path / "run/sandbox").rglob("request-*.json"))
    assert sandbox_requests
    for request_path in sandbox_requests:
        request_document = json.loads(request_path.read_text(encoding="utf-8"))
        assert set(request_document) == {
            "request_id",
            "prompt_tokens",
            "maximum_new_tokens",
            "seed",
        }
    genesis_summary = next(
        item for item in report.tasks[0].baselines if item.baseline is BaselineKind.GENESIS_FULL
    )
    assert genesis_summary.raw_samples_path is not None
    generated_sample = RawCpuSample.model_validate_json(
        Path(genesis_summary.raw_samples_path).read_bytes().splitlines()[0], strict=True
    )
    assert generated_sample.execution_evidence_path is not None
    raw_path = Path(genesis_summary.raw_samples_path)
    original_raw = raw_path.read_bytes()
    raw_samples = [
        RawCpuSample.model_validate_json(line, strict=True)
        for line in original_raw.splitlines()
        if line
    ]
    raw_samples[0] = raw_samples[0].model_copy(
        update={
            "observed_tokens": tuple(value + 1 for value in raw_samples[0].observed_tokens),
            "expected_tokens": tuple(value + 1 for value in raw_samples[0].expected_tokens),
        }
    )
    raw_path.write_bytes(b"".join(canonical_json(sample) + b"\n" for sample in raw_samples))
    forged_summary = synthbench_runner._summary_from_artifact(
        genesis_summary.baseline,
        raw_path,
        candidate_count=genesis_summary.candidate_count,
    )
    first_task = report.tasks[0]
    forged_task = first_task.model_copy(
        update={
            "baselines": tuple(
                forged_summary if item.baseline is BaselineKind.GENESIS_FULL else item
                for item in first_task.baselines
            )
        }
    )
    forged_report = report.model_copy(update={"tasks": (forged_task, *report.tasks[1:])})
    with pytest.raises(ValueError, match=r"workload oracle|sandbox runner response"):
        validate_cpu_benchmark_report(forged_report)
    raw_path.write_bytes(original_raw)
    execution_evidence = Path(generated_sample.execution_evidence_path)
    execution_evidence.write_bytes(execution_evidence.read_bytes() + b" ")
    with pytest.raises(ValueError, match="execution evidence"):
        validate_cpu_benchmark_report(report)


def test_integrity_auditor_detects_discarded_sample(tmp_path: Path) -> None:
    descriptor = generate_tasks(_configuration(count=1), tmp_path / "tasks")[0]
    task = tmp_path / "tasks" / descriptor.task_id
    report = run_cpu_benchmark(
        (task,),
        CpuRunConfiguration(
            seeds=(1, 2),
            warmup_count=0,
            repetitions=2,
            maximum_tasks=1,
            maximum_runtime_seconds=10.0,
        ),
        tmp_path / "run",
    )
    task_report = report.tasks[0]
    summary = next(
        item for item in task_report.baselines if item.baseline is BaselineKind.GENESIS_FULL
    )
    assert summary.raw_samples_path is not None
    samples = tuple(
        RawCpuSample.model_validate_json(line, strict=True)
        for line in Path(summary.raw_samples_path).read_bytes().splitlines()
    )
    damaged = audit_raw_samples(
        samples[:-1],
        measured_baselines=(BaselineKind.GENESIS_FULL,),
        workload_fingerprint=task_report.integrity.workload_fingerprint,
        environment_fingerprint=task_report.integrity.environment_fingerprint,
        task_hash=task_report.task_hash,
        run_seed=task_report.run_seed,
        repetitions=task_report.repetitions,
        request_ids=task_report.integrity.expected_request_ids,
        measurement_run_order=tuple(
            (int(value.split(":", 1)[0]), BaselineKind(value.split(":", 1)[1]))
            for value in task_report.measurement_run_order
        ),
    )

    assert not damaged.passed
    assert any("input_distribution_mismatch" in violation for violation in damaged.violations)

    dishonest_first = samples[0].model_copy(
        update={
            "observed_tokens": tuple(value + 1 for value in samples[0].expected_tokens),
            "exact_match": True,
            "run_seed": samples[0].run_seed + 1,
            "measurement_order_ordinal": len(task_report.measurement_run_order) + 1,
            "ttft_ns": samples[0].latency_ns + 1,
        }
    )
    dishonest = audit_raw_samples(
        (dishonest_first, *samples[1:]),
        measured_baselines=(BaselineKind.GENESIS_FULL,),
        workload_fingerprint=task_report.integrity.workload_fingerprint,
        environment_fingerprint=task_report.integrity.environment_fingerprint,
        task_hash=task_report.task_hash,
        run_seed=task_report.run_seed,
        repetitions=task_report.repetitions,
        request_ids=task_report.integrity.expected_request_ids,
        measurement_run_order=tuple(
            (int(value.split(":", 1)[0]), BaselineKind(value.split(":", 1)[1]))
            for value in task_report.measurement_run_order
        ),
    )
    assert not dishonest.passed
    assert {violation.split(":", 1)[1] for violation in dishonest.violations} >= {
        "false_exact_match",
        "ordinal_set_mismatch",
        "run_seed_mismatch",
        "ttft_exceeds_latency",
    }


def test_cpu_runner_requires_multiple_distinct_seeds() -> None:
    with pytest.raises(ValueError, match="at least two distinct seeds"):
        CpuRunConfiguration(
            seeds=(1,),
            warmup_count=0,
            repetitions=2,
            maximum_tasks=1,
        )


def test_synthbench_cli_generates_runs_and_compares_cpu_artifacts(tmp_path: Path) -> None:
    tasks = tmp_path / "tasks"
    generated = runner.invoke(
        app,
        [
            "synthbench",
            "generate",
            "--seed",
            "73129",
            "--count",
            "1",
            "--output",
            str(tasks),
        ],
    )
    assert generated.exit_code == 0, generated.output
    run = tmp_path / "runs/genesis"
    executed = runner.invoke(
        app,
        [
            "synthbench",
            "run",
            "--tasks",
            str(tasks),
            "--system",
            "genesis",
            "--seeds",
            "1,2",
            "--repetitions",
            "2",
            "--output",
            str(run),
        ],
    )
    assert executed.exit_code == 0, executed.output
    compared = runner.invoke(
        app,
        [
            "synthbench",
            "compare",
            "--runs",
            str(tmp_path / "runs"),
            "--output",
            str(tmp_path / "comparison"),
        ],
    )
    assert compared.exit_code == 0, compared.output
    comparison = json.loads((tmp_path / "comparison/comparison.json").read_text(encoding="utf-8"))
    assert comparison["source"] == "validated_synthbench_reports"
    assert comparison["hardware_comparison"] == "not_measured_in_cpu_profile"

    report = SynthBenchReport.model_validate_json((run / "report.json").read_bytes(), strict=True)
    genesis = next(
        baseline
        for baseline in report.tasks[0].baselines
        if baseline.baseline is BaselineKind.GENESIS_FULL
    )
    assert genesis.raw_samples_path is not None
    Path(genesis.raw_samples_path).write_bytes(Path(genesis.raw_samples_path).read_bytes() + b"\n")
    rejected = runner.invoke(
        app,
        [
            "synthbench",
            "compare",
            "--runs",
            str(tmp_path / "runs"),
            "--output",
            str(tmp_path / "comparison-tampered"),
        ],
    )
    assert rejected.exit_code != 0


def test_synthbench_demo_produces_artifact_derived_cpu_summary(tmp_path: Path) -> None:
    result = run_synthbench_demo(tmp_path / "demo", seed=73129, count=2)
    assert result.task_count == 2
    assert result.valid_system_rate == 1.0
    assert result.exact_request_rate == 1.0
    assert result.observed_request_wall_seconds > 0
    assert result.hardware_backed is False
    assert Path(result.report_path).is_file()
