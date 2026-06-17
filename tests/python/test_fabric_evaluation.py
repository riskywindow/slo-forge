from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from sloforge.fabric.evaluation import (
    EvaluationConfig,
    EvaluationMethod,
    WorkloadRegime,
    _bootstrap_median,
    _spearman,
    _workload,
    run_fabric_evaluation,
    validate_evaluation_artifacts,
)


def test_bootstrap_interval_is_seeded_and_contains_median() -> None:
    values = (1.0, 2.0, 4.0, 8.0, 16.0)
    first = _bootstrap_median(values, seed=9, repetitions=200)
    second = _bootstrap_median(values, seed=9, repetitions=200)
    assert first == second
    assert first.confidence_low <= first.median <= first.confidence_high
    assert first.sample_count == len(values)


def test_spearman_handles_order_reversal_and_ties() -> None:
    assert _spearman((1.0, 2.0, 3.0), (3.0, 2.0, 1.0)) == pytest.approx(-1.0)
    assert _spearman((1.0, 1.0, 1.0), (1.0, 2.0, 3.0)) == 0.0


def test_evaluation_dimensions_are_unique() -> None:
    with pytest.raises(ValidationError, match="seeds must be non-empty and unique"):
        EvaluationConfig(seeds=(3, 3))


def test_expert_skew_evaluation_workload_sets_typed_factor() -> None:
    balanced = _workload(7, 2, WorkloadRegime.MIXED_BURSTY)
    skewed = _workload(7, 2, WorkloadRegime.EXPERT_SKEWED)
    assert all(request.expert_skew_factor == 1.0 for request in balanced.requests)
    assert all(request.expert_skew_factor == 4.0 for request in skewed.requests)


def test_checked_in_evaluation_reports_match_raw_artifacts() -> None:
    repository_root = Path(__file__).resolve().parents[2]
    result = validate_evaluation_artifacts(
        artifact_root=repository_root / "artifacts" / "fabric" / "evaluation",
        report_dir=repository_root / "reports",
    )
    assert result.validation_mode == "synthetic_cpu"
    assert len(result.plan_trials) == 180
    assert result.diagnosis_summary.top_three_accuracy >= result.diagnosis_summary.top_one_accuracy


@pytest.mark.expensive
def test_artifact_derived_evaluation_round_trip(tmp_path: Path) -> None:
    repository_root = Path(__file__).resolve().parents[2]
    artifact_root = tmp_path / "artifacts"
    report_dir = tmp_path / "reports"
    result = run_fabric_evaluation(
        repository_root=repository_root,
        artifact_root=artifact_root,
        report_dir=report_dir,
        config=EvaluationConfig(
            seeds=(7,),
            topology_fixtures=("two_node_infiniband",),
            workload_regimes=(WorkloadRegime.MIXED_BURSTY,),
            methods=(EvaluationMethod.RANDOM, EvaluationMethod.HIERARCHICAL),
            request_count=2,
            bootstrap_repetitions=100,
        ),
        reset=True,
    )

    assert len(result.plan_trials) == 2
    assert len(result.diagnosis_trials) == 2
    assert len(result.recovery_trials) == 10
    assert result.hardware_backed_validation == "not_exercised_no_compatible_hardware"
    assert all(item.simulator_calls >= 1 for item in result.plan_trials)
    random_trial = next(
        item for item in result.plan_trials if item.method is EvaluationMethod.RANDOM
    )
    # Evaluation must retain a baseline even when it misses the declared SLO;
    # hard production compiler constraints are tested in the compiler suite.
    assert random_trial.observed_p95_ttft_ms >= 0.0
    assert 0.0 <= random_trial.slo_attainment <= 1.0
    validated = validate_evaluation_artifacts(
        artifact_root=artifact_root,
        report_dir=report_dir,
    )
    assert validated == result
    assert "Synthetic" in (report_dir / "fabric-evaluation.md").read_text(encoding="utf-8")

    report_path = report_dir / "fabric-evaluation.md"
    report_text = report_path.read_text(encoding="utf-8")
    report_path.write_text("tampered report\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="evaluation report hash mismatch"):
        validate_evaluation_artifacts(artifact_root=artifact_root, report_dir=report_dir)
    report_path.write_text(report_text, encoding="utf-8")

    target = artifact_root / result.artifacts[-1].path
    target.write_text("tampered\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="artifact hash mismatch"):
        validate_evaluation_artifacts(artifact_root=artifact_root, report_dir=report_dir)
