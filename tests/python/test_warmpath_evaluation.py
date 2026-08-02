from __future__ import annotations

import hashlib
import json
from pathlib import Path

from sloforge.warmpath.evaluation import (
    EvaluationScenario,
    WarmPathEvaluationManifest,
    WarmPathEvaluationResult,
    WarmPathStrategy,
    run_warmpath_evaluation,
)


def test_checked_in_h6_evaluation_is_artifact_derived() -> None:
    root = Path(__file__).parents[2]
    output = root / "artifacts" / "warmpath" / "evaluation"
    result = WarmPathEvaluationResult.model_validate_json(
        (output / "result.json").read_text(encoding="utf-8")
    )
    manifest = WarmPathEvaluationManifest.model_validate_json(
        (output / "manifest.json").read_text(encoding="utf-8")
    )
    assert result.hypothesis == "H6"
    assert result.all_timings_hardware_measured is False
    assert {item.strategy for item in result.strategies} == set(WarmPathStrategy)
    assert manifest.raw_trial_count == 6 * 2 * 11 * 101
    assert len(result.limitations) >= 4
    by_strategy = {item.strategy: item for item in result.strategies}
    assert by_strategy[WarmPathStrategy.WARMPATH].eviction_absolute_penalty_ms > 0.0
    assert by_strategy[WarmPathStrategy.WARM_REPLICA].storage_bytes > 0
    for artifact in manifest.artifacts:
        path = root / artifact.path
        assert path.is_file()
        assert hashlib.sha256(path.read_bytes()).hexdigest() == artifact.sha256


def test_h6_evaluation_is_deterministic_and_preserves_raw_trials(tmp_path: Path) -> None:
    root = Path(__file__).parents[2]
    config = root / "benchmarks" / "warmpath" / "h6-evaluation.yaml"
    first_output = tmp_path / "first"
    second_output = tmp_path / "second"
    first = run_warmpath_evaluation(
        config_path=config,
        output_directory=first_output,
        report_path=tmp_path / "first.md",
        repository_root=root,
    )
    second = run_warmpath_evaluation(
        config_path=config,
        output_directory=second_output,
        report_path=tmp_path / "second.md",
        repository_root=root,
    )
    assert first.model_dump(mode="json") == second.model_dump(mode="json")
    assert (first_output / "raw" / "trials.jsonl").read_bytes() == (
        second_output / "raw" / "trials.jsonl"
    ).read_bytes()

    rows = [
        json.loads(line)
        for line in (first_output / "raw" / "trials.jsonl").read_text().splitlines()
    ]
    assert {row["strategy"] for row in rows} == {item.value for item in WarmPathStrategy}
    assert {row["scenario"] for row in rows} == {item.value for item in EvaluationScenario}
    assert any(row["restore_failed"] for row in rows)
    assert any(row["eviction_occurred"] for row in rows)


def test_h6_report_contains_values_loaded_from_result() -> None:
    root = Path(__file__).parents[2]
    result = WarmPathEvaluationResult.model_validate_json(
        (root / "artifacts" / "warmpath" / "evaluation" / "result.json").read_text()
    )
    report = (root / "reports" / "warmpath-evaluation.md").read_text()
    assert f"{result.warmpath_vs_local_disk_p95_improvement_percent:+.2f}%" in report
    for strategy in result.strategies:
        assert f"{strategy.p95_ready_time_ms:.4f}" in report
