from __future__ import annotations

from pathlib import Path

import pytest

from sloforge.genesis.evaluation_campaigns.redteam import (
    RedTeamCampaignValidationError,
    run_redteam_campaign,
    validate_redteam_campaign,
)
from sloforge.redteam import RedTeamReport, RedTeamSurface, ScheduleAdversarialCase


def test_h8_campaign_finds_minimizes_converts_and_replays_new_violations(
    tmp_path: Path,
) -> None:
    root = tmp_path / "h8"
    report = run_redteam_campaign(root, seeds=(73129, 73130, 73131))

    assert report.conclusion == "supported_in_declared_fixture_scope"
    assert report.scope.hardware_backed_runs == 0
    assert report.scope.gpu_hours == 0
    assert not report.scope.wall_clock_performance_recorded
    assert report.normal_unique_violation_contracts == ()
    assert report.redteam_unique_violation_family_count == 19
    assert report.redteam_only_violation_contracts == report.redteam_unique_violation_contracts
    assert report.normal_evaluation_steps == 15
    assert report.redteam_evaluation_steps == 117
    assert report.converted_regressions == 57
    assert report.reproduced_regressions == report.converted_regressions

    for result in report.results:
        assert result.redteam_evaluation_steps == 39
        assert result.time_to_first_violation_evaluation_units == 1
        assert not result.timed_out
        assert all(item.violated_contract is None for item in result.normal_observations)
        assert {item.surface for item in result.findings} == set(RedTeamSurface)
        protocol = next(item for item in result.findings if item.surface is RedTeamSurface.PROTOCOL)
        assert protocol.original_size_units == 8
        assert protocol.minimized_size_units == 4
        assert protocol.reduced
        assert protocol.minimization_evaluations <= 128

        raw = RedTeamReport.model_validate_json(
            Path(result.redteam_report_path).read_bytes(), strict=True
        )
        raw_protocol = next(
            item for item in raw.findings if item.surface is RedTeamSurface.PROTOCOL
        )
        assert isinstance(raw_protocol.original_case, ScheduleAdversarialCase)
        assert any(event.action == "fail" for event in raw_protocol.original_case.events)
        assert raw_protocol.counterexample.minimized

    assert all(item.later_recurrence_count == 2 for item in report.recurrence)
    assert all(item.regression_reproduction_count == 3 for item in report.recurrence)
    validate_redteam_campaign(root / "report.json")


def test_h8_campaign_validator_rejects_summary_and_artifact_tampering(tmp_path: Path) -> None:
    root = tmp_path / "h8"
    report = run_redteam_campaign(root)

    altered_summary = report.model_copy(update={"redteam_unique_violation_family_count": 18})
    with pytest.raises(RedTeamCampaignValidationError, match="aggregate"):
        validate_redteam_campaign(altered_summary)

    corpus_path = Path(report.results[0].regression_corpus_path)
    corpus_path.write_bytes(corpus_path.read_bytes() + b" ")
    with pytest.raises(RedTeamCampaignValidationError, match="supporting artifact digest"):
        validate_redteam_campaign(report)


def test_h8_campaign_is_bounded_and_refuses_overwrite(tmp_path: Path) -> None:
    root = tmp_path / "h8"
    first = run_redteam_campaign(root, seeds=(7, 11, 13))
    assert tuple(result.configuration.maximum_findings for result in first.results) == (32, 32, 32)
    assert tuple(result.configuration.run_timeout_seconds for result in first.results) == (
        10.0,
        10.0,
        10.0,
    )
    with pytest.raises(FileExistsError, match="must be empty"):
        run_redteam_campaign(root, seeds=(7, 11, 13))
    with pytest.raises(ValueError, match="at least three unique"):
        run_redteam_campaign(tmp_path / "bad", seeds=(7, 7, 11))
