from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

import pytest
from pydantic import ValidationError

from sloforge.helix.evaluation.experience_selection import (
    DownstreamEvaluationTrace,
    ExperienceSelectionCampaignRun,
    ExperienceSelectionCampaignValidationError,
    load_experience_selection_campaign_scenario,
    run_experience_selection_campaign,
    validate_experience_selection_campaign,
)
from sloforge.helix.experience import SelectionStrategy

SCENARIO = Path("scenarios/helix/experience/h9-reference-campaign.json")
SEEDS = (41, 73, 101)


def test_campaign_measures_complete_strategy_seed_matrix_with_raw_provenance(
    tmp_path: Path,
) -> None:
    output = tmp_path / "campaign"
    run = run_experience_selection_campaign(SCENARIO, output, seeds=SEEDS)

    assert run.seeds == SEEDS
    assert run.strategies == tuple(SelectionStrategy)
    assert len(run.observations) == len(SEEDS) * len(SelectionStrategy)
    assert len(run.raw_artifacts) == 1 + 10 + len(run.observations) * 5
    assert run.measurement_definition.endswith("selector scores are not gains")

    ledger = {item.path: item for item in run.raw_artifacts}
    for relative, artifact in ledger.items():
        payload = (output / relative).read_bytes()
        assert sha256(payload).hexdigest() == artifact.sha256
    assert (output / run.scenario_artifact.path).read_bytes() == SCENARIO.read_bytes()

    candidate_hashes = {item.artifact.sha256 for item in run.candidate_evidence}
    for observation in run.observations:
        request = json.loads((output / observation.provenance.selection_request.path).read_text())
        plan = json.loads((output / observation.provenance.selection_plan.path).read_text())
        samples = json.loads((output / observation.provenance.training_samples.path).read_text())
        training = json.loads((output / observation.provenance.training_result.path).read_text())
        trace = DownstreamEvaluationTrace.model_validate_json(
            (output / observation.provenance.evaluation_trace.path).read_bytes(), strict=True
        )
        assert request["seed"] == observation.campaign_seed
        assert request["strategy"] == observation.strategy.value
        assert {
            artifact["artifact_sha256"]
            for candidate in request["candidates"]
            for artifact in candidate["artifacts"]
        } == candidate_hashes
        assert tuple(plan["selected_candidate_ids"]) == observation.selected_candidate_ids
        assert len(samples) == observation.selected_count
        assert training["checkpoint_hash"] == trace.training_checkpoint_sha256
        assert len(trace.decisions) == observation.evaluation_decision_count
        assert trace.paired_measured_success_change == observation.paired_measured_success_change
        assert trace.failures_repaired == observation.failures_repaired
        assert trace.failures_introduced == observation.failures_introduced

    report = (output / "report.md").read_text()
    assert "Selector scores and expected learning values are not treated as measured gain" in report
    assert "Cost mean" in report
    assert "Diversity" in report
    assert "Redundancy" in report
    assert "Repaired" in report
    assert "Introduced" in report
    assert "Paired change" in report
    assert "95% Student-t sensitivity interval" in report
    assert validate_experience_selection_campaign(output) == run


def test_campaign_is_byte_deterministic_and_uses_paired_holdout_decisions(
    tmp_path: Path,
) -> None:
    first_output = tmp_path / "first"
    second_output = tmp_path / "second"
    first = run_experience_selection_campaign(SCENARIO, first_output, seeds=(101, 41, 73))
    second = run_experience_selection_campaign(SCENARIO, second_output, seeds=SEEDS)

    assert first == second
    first_files = {
        path.relative_to(first_output): path.read_bytes()
        for path in first_output.rglob("*")
        if path.is_file()
    }
    second_files = {
        path.relative_to(second_output): path.read_bytes()
        for path in second_output.rglob("*")
        if path.is_file()
    }
    assert first_files == second_files

    for seed in SEEDS:
        traces = []
        for strategy in SelectionStrategy:
            observation = next(
                item
                for item in first.observations
                if item.campaign_seed == seed and item.strategy is strategy
            )
            traces.append(
                DownstreamEvaluationTrace.model_validate_json(
                    (first_output / observation.provenance.evaluation_trace.path).read_bytes(),
                    strict=True,
                )
            )
        champion_records = tuple(
            (item.case_id, item.replicate, item.decision_seed, item.champion_action)
            for item in traces[0].decisions
        )
        assert all(
            tuple(
                (item.case_id, item.replicate, item.decision_seed, item.champion_action)
                for item in trace.decisions
            )
            == champion_records
            for trace in traces[1:]
        )


def test_h9_result_is_honest_when_small_seed_intervals_do_not_separate(
    tmp_path: Path,
) -> None:
    run = run_experience_selection_campaign(SCENARIO, tmp_path / "campaign", seeds=SEEDS)
    summaries = {item.strategy: item for item in run.strategy_summaries}
    comparisons = {item.baseline_strategy: item for item in run.helix_baseline_comparisons}

    helix = summaries[SelectionStrategy.HELIX_VALUE_AWARE]
    assert helix.paired_measured_success_change.mean > max(
        item.paired_measured_success_change.mean
        for strategy, item in summaries.items()
        if strategy is not SelectionStrategy.HELIX_VALUE_AWARE
    )
    assert comparisons[SelectionStrategy.RANDOM].sensitivity.lower_95 < 0.0
    assert comparisons[SelectionStrategy.FAILURE_ONLY].sensitivity.lower_95 < 0.0
    assert run.h9_result.status == "inconclusive"
    assert all(
        item.sensitivity.method == "two_sided_student_t_mean"
        and item.sensitivity.sample_count == len(SEEDS)
        for item in run.helix_baseline_comparisons
    )
    assert any("not multiplicity-adjusted" in item for item in run.limitations)
    assert any("not claimed to be independent" in item for item in run.limitations)


def test_campaign_relational_seal_rejects_metric_tampering(tmp_path: Path) -> None:
    run = run_experience_selection_campaign(SCENARIO, tmp_path / "campaign", seeds=SEEDS)
    payload = run.model_dump(mode="json")
    payload["observations"][0]["failures_repaired"] += 1
    with pytest.raises(ValidationError, match="repair accounting"):
        ExperienceSelectionCampaignRun.model_validate_json(json.dumps(payload), strict=True)

    payload = run.model_dump(mode="json")
    payload["campaign_id"] = "0" * 64
    with pytest.raises(ValidationError, match="identifier is invalid"):
        ExperienceSelectionCampaignRun.model_validate_json(json.dumps(payload), strict=True)


def test_persisted_campaign_validation_rejects_tampered_raw_artifact(
    tmp_path: Path,
) -> None:
    output = tmp_path / "campaign"
    run = run_experience_selection_campaign(SCENARIO, output, seeds=(41, 73))
    evaluation = run.observations[0].provenance.evaluation_trace
    path = output / evaluation.path
    path.write_bytes(path.read_bytes() + b" ")

    with pytest.raises(ExperienceSelectionCampaignValidationError, match="SHA-256 mismatch"):
        validate_experience_selection_campaign(output)


def test_persisted_campaign_validation_rejects_missing_raw_artifact(
    tmp_path: Path,
) -> None:
    output = tmp_path / "campaign"
    run = run_experience_selection_campaign(SCENARIO, output, seeds=(41, 73))
    missing = run.observations[-1].provenance.selection_plan
    (output / missing.path).unlink()

    with pytest.raises(ExperienceSelectionCampaignValidationError, match="artifact is missing"):
        validate_experience_selection_campaign(output)


def test_campaign_rejects_invalid_seeds_scenario_actions_and_nonempty_output(
    tmp_path: Path,
) -> None:
    duplicate_output = tmp_path / "duplicate"
    with pytest.raises(ValueError, match="unique"):
        run_experience_selection_campaign(SCENARIO, duplicate_output, seeds=(41, 41))
    assert not duplicate_output.exists()

    payload = json.loads(SCENARIO.read_text())
    payload["candidates"][0]["training_signal"]["action"] = "unknown_action"
    invalid = tmp_path / "invalid.json"
    invalid.write_text(json.dumps(payload))
    with pytest.raises(ValidationError, match="unknown action"):
        load_experience_selection_campaign_scenario(invalid)

    nonempty = tmp_path / "nonempty"
    nonempty.mkdir()
    (nonempty / "owned-by-user").write_text("preserve")
    with pytest.raises(FileExistsError, match="absent or empty"):
        run_experience_selection_campaign(SCENARIO, nonempty, seeds=(41, 73))
    assert (nonempty / "owned-by-user").read_text() == "preserve"
