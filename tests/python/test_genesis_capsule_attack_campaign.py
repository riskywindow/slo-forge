from __future__ import annotations

from pathlib import Path

import pytest

from sloforge.genesis.evaluation_campaigns.capsule_attacks import (
    AttackKind,
    CapsuleAttackCampaignValidationError,
    run_capsule_attack_campaign,
    validate_capsule_attack_campaign,
)


def test_h6_campaign_rejects_every_attack_and_independently_replays(tmp_path: Path) -> None:
    report = run_capsule_attack_campaign(tmp_path / "h6", seeds=(17, 29, 73129))

    assert report.conclusion == "supported_in_declared_fixture_scope"
    assert report.scope.hardware_backed_runs == 0
    assert report.scope.gpu_hours == 0
    assert report.scope.performance_claims is False
    assert report.attack_count == 30
    assert report.rejected_attack_count == report.attack_count
    assert all(result.baseline_promotion_eligible for result in report.results)
    for result in report.results:
        assert tuple(item.attack for item in result.attacks) == tuple(AttackKind)
        assert all(item.rejected and not item.promotion_eligible for item in result.attacks)
        assert all(
            set(item.expected_issue_codes).issubset(item.observed_issue_codes)
            for item in result.attacks
        )

    reopened = validate_capsule_attack_campaign(tmp_path / "h6" / "report.json")
    assert reopened == report


def test_h6_campaign_rejects_changed_supporting_evidence(tmp_path: Path) -> None:
    report = run_capsule_attack_campaign(tmp_path / "h6", seeds=(3, 5, 7))
    validation_path = Path(report.results[0].attacks[0].validation_report_path)
    validation_path.write_bytes(validation_path.read_bytes() + b"tampered")

    with pytest.raises(
        CapsuleAttackCampaignValidationError,
        match="artifact digest mismatch",
    ):
        validate_capsule_attack_campaign(tmp_path / "h6" / "report.json")


def test_h6_campaign_requires_an_empty_output_and_three_unique_seeds(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="at least three unique"):
        run_capsule_attack_campaign(tmp_path / "too-few", seeds=(1, 2))

    output = tmp_path / "not-empty"
    output.mkdir()
    (output / "owned.txt").write_text("user data", encoding="utf-8")
    with pytest.raises(FileExistsError, match="must be empty"):
        run_capsule_attack_campaign(output, seeds=(1, 2, 3))
    assert (output / "owned.txt").read_text(encoding="utf-8") == "user data"
