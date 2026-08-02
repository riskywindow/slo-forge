from __future__ import annotations

import json
import statistics
from pathlib import Path

import pytest

from sloforge.genesis.frontend import inspect_reference_package
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
)


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
    tmp_path: Path,
) -> None:
    tasks_root = tmp_path / "tasks"
    descriptors = generate_tasks(_configuration(count=2), tasks_root)
    task_directories = tuple(tasks_root / descriptor.task_id for descriptor in descriptors)
    output = tmp_path / "run"
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
    assert report.metrics.measured_cpu_seconds > 0
    for task_report in report.tasks:
        assert task_report.integrity.passed
        assert {summary.baseline for summary in task_report.baselines} == set(BaselineKind)
        assert len(task_report.hidden_evaluations) == len(BaselineKind)
        measured = [
            summary
            for summary in task_report.baselines
            if summary.status is BaselineStatus.MEASURED
        ]
        unavailable = [
            summary
            for summary in task_report.baselines
            if summary.status is BaselineStatus.NOT_APPLICABLE
        ]
        assert measured and unavailable
        assert all(summary.reason for summary in unavailable)
        assert all(summary.valid_request_rate == 1.0 for summary in measured)
        assert all(summary.human_authored_model_specific_lines == 0 for summary in measured)
        assert all(
            hidden.exact_case_rate == 1.0
            for hidden in task_report.hidden_evaluations
            if hidden.status is BaselineStatus.MEASURED
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
        repetitions=task_report.repetitions,
        request_ids=task_report.integrity.expected_request_ids,
    )

    assert not damaged.passed
    assert any("input_distribution_mismatch" in violation for violation in damaged.violations)


def test_cpu_runner_requires_multiple_distinct_seeds() -> None:
    with pytest.raises(ValueError, match="at least two distinct seeds"):
        CpuRunConfiguration(
            seeds=(1,),
            warmup_count=0,
            repetitions=2,
            maximum_tasks=1,
        )
