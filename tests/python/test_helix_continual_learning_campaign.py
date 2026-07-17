from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

import pytest

from sloforge.helix.evaluation.continual_learning import (
    ContinualLearningCampaign,
    run_continual_learning_campaign,
    validate_continual_learning_campaign,
)


def _digest(value: object) -> str:
    return sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def test_continual_campaign_executes_training_gates_and_rollback(tmp_path: Path) -> None:
    campaign = run_continual_learning_campaign(tmp_path / "h8", seeds=(7, 11))
    assert all(len(run.phases) == 3 for run in campaign.runs)
    assert all(run.accepted_update_count >= 1 for run in campaign.runs)
    assert all(run.candidate_rejection_count >= 1 for run in campaign.runs)
    assert all(phase.evidence_sha256 for run in campaign.runs for phase in run.phases)
    assert all(
        (tmp_path / "h8" / phase.evidence_path).is_file()
        for run in campaign.runs
        for phase in run.phases
    )
    parsed = ContinualLearningCampaign.model_validate_json(
        (tmp_path / "h8" / "campaign.json").read_text(encoding="utf-8"), strict=True
    )
    assert parsed == campaign
    assert validate_continual_learning_campaign(tmp_path / "h8") == campaign


def test_continual_campaign_is_deterministic(tmp_path: Path) -> None:
    first = run_continual_learning_campaign(tmp_path / "first", seeds=(13, 17))
    second = run_continual_learning_campaign(tmp_path / "second", seeds=(13, 17))
    assert first == second
    assert (tmp_path / "first" / "campaign.json").read_bytes() == (
        tmp_path / "second" / "campaign.json"
    ).read_bytes()


def test_continual_campaign_rejects_tampering_and_invalid_gate(tmp_path: Path) -> None:
    output = tmp_path / "h8"
    run_continual_learning_campaign(output, seeds=(19, 23))
    document = json.loads((output / "campaign.json").read_text(encoding="utf-8"))
    document["runs"][0]["candidate_rejection_count"] = 0
    with pytest.raises(ValueError, match="candidate rejection count"):
        ContinualLearningCampaign.model_validate_json(json.dumps(document), strict=True)
    with pytest.raises(ValueError, match="target improvement"):
        run_continual_learning_campaign(
            tmp_path / "invalid", seeds=(1, 2), target_improvement_gate=0
        )
    evidence_path = output / document["runs"][0]["phases"][0]["evidence_path"]
    evidence_path.write_bytes(evidence_path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="digest mismatch"):
        validate_continual_learning_campaign(output)


def test_continual_campaign_rejects_resigned_metric_forgery(tmp_path: Path) -> None:
    output = tmp_path / "h8"
    run_continual_learning_campaign(output, seeds=(29, 31))
    path = output / "campaign.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    document["runs"][0]["phases"][0]["policy_kl"] = 9.0
    document["campaign_id"] = _digest(
        {key: value for key, value in document.items() if key != "campaign_id"}
    )
    path.write_text(json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n")
    with pytest.raises(ValueError, match="summary disagrees"):
        validate_continual_learning_campaign(output)


def test_continual_campaign_rejects_orphaned_raw_artifact(tmp_path: Path) -> None:
    output = tmp_path / "h8"
    run_continual_learning_campaign(output, seeds=(37, 41))
    orphan = output / "raw" / "orphan.json"
    orphan.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="inventory"):
        validate_continual_learning_campaign(output)
