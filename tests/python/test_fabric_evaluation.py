from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from sloforge.fabric.evaluation import (
    EvaluationConfig,
    EvaluationMethod,
    _bootstrap_median,
    _spearman,
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
    assert all(item.simulator_calls == 0 for item in result.plan_trials)
    validated = validate_evaluation_artifacts(
        artifact_root=artifact_root,
        report_dir=report_dir,
    )
    assert validated == result
    assert "Synthetic" in (report_dir / "fabric-evaluation.md").read_text(encoding="utf-8")

    target = artifact_root / result.artifacts[-1].path
    target.write_text("tampered\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="artifact hash mismatch"):
        validate_evaluation_artifacts(artifact_root=artifact_root, report_dir=report_dir)
