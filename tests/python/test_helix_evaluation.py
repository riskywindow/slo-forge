from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from pydantic import ValidationError

from sloforge.helix.evaluation import EvaluationRun, run_reference_evaluation
from sloforge.helix.scheduler import SchedulerPolicy


def test_reference_evaluation_rejects_invalid_seed_matrix(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="two to 32 unique seeds"):
        run_reference_evaluation(
            tmp_path / "evaluation",
            reports=tmp_path / "reports",
            workload=Path("scenarios/helix/resource/cpu-learning-aware.json"),
            seeds=(41,),
        )


def test_reference_evaluation_refuses_nonempty_output(tmp_path: Path) -> None:
    output = tmp_path / "evaluation"
    output.mkdir()
    (output / "unrelated.txt").write_text("preserve me")
    with pytest.raises(FileExistsError, match="absent or empty"):
        run_reference_evaluation(
            output,
            reports=tmp_path / "reports",
            workload=Path("scenarios/helix/resource/cpu-learning-aware.json"),
            seeds=(41, 73),
        )


def test_reference_evaluation_is_multi_seed_provenance_sealed_and_honest(
    tmp_path: Path,
) -> None:
    output = tmp_path / "evaluation"
    reports = tmp_path / "reports"
    run = run_reference_evaluation(
        output,
        reports=reports,
        workload=Path("scenarios/helix/resource/cpu-learning-aware.json"),
        seeds=(41, 73),
    )

    assert len(run.scheduler_comparisons) == 2 * len(SchedulerPolicy)
    assert {(item.seed, item.policy) for item in run.scheduler_comparisons} == {
        (seed, policy) for seed in run.seeds for policy in SchedulerPolicy
    }
    assert all(item.serving_predicted_slo_feasible for item in run.scheduler_comparisons)
    assert run.success_rate_delta.method == "two_sided_student_t_mean"
    assert run.success_rate_delta.sample_count == 2
    assert len(run.seed_observations) == 2
    assert all((output / item.path).is_file() for item in run.raw_artifacts)
    assert all(
        hashlib.sha256((output / item.path).read_bytes()).hexdigest() == item.sha256
        for item in run.raw_artifacts
    )
    assert EvaluationRun.model_validate_json((output / "evaluation.json").read_bytes()) == run

    hypotheses = {item.hypothesis_id: item for item in run.hypotheses}
    assert hypotheses["H4"].status == "not_exercised"
    assert hypotheses["H9"].status == "not_exercised"
    report = (reports / "helix-evaluation.md").read_text()
    assert "Student-t 95% interval" in report
    assert "modeled predictions, not measurements" in report
    assert "Raw provenance" in report

    payload = run.model_dump(mode="python")
    payload["limitations"] = (*run.limitations, "tampered claim")
    with pytest.raises(ValidationError, match="evaluation identifier is invalid"):
        EvaluationRun.model_validate(payload)
