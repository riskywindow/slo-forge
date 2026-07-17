from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

import pytest

from sloforge.helix.evaluation.training_algorithms import (
    TrainingAlgorithmCampaign,
    run_training_algorithm_campaign,
    validate_training_algorithm_campaign,
)
from sloforge.helix.trainers import TrainingAlgorithm


def _digest(value: object) -> str:
    return sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def test_training_algorithm_campaign_executes_full_matrix(tmp_path: Path) -> None:
    campaign = run_training_algorithm_campaign(
        tmp_path / "h3",
        seeds=(7, 11),
        example_budgets=(1, 2),
    )
    assert len(campaign.runs) == 2 * 4 * 2
    assert {run.algorithm for run in campaign.runs} == set(TrainingAlgorithm)
    assert all(run.evidence_sha256 for run in campaign.runs)
    assert all((tmp_path / "h3" / run.evidence_path).is_file() for run in campaign.runs)
    parsed = TrainingAlgorithmCampaign.model_validate_json(
        (tmp_path / "h3" / "campaign.json").read_text(encoding="utf-8"), strict=True
    )
    assert parsed == campaign
    assert validate_training_algorithm_campaign(tmp_path / "h3") == campaign
    assert any(run.task_success_rate > campaign.base_task_success_rate for run in campaign.runs)
    assert campaign.h3_result.status == "not_supported"
    assert len(campaign.h3_result.comparisons) == 3


def test_training_algorithm_campaign_is_deterministic(tmp_path: Path) -> None:
    first = run_training_algorithm_campaign(
        tmp_path / "first", seeds=(3, 5), example_budgets=(1, 2)
    )
    second = run_training_algorithm_campaign(
        tmp_path / "second", seeds=(3, 5), example_budgets=(1, 2)
    )
    assert first == second
    assert (tmp_path / "first" / "campaign.json").read_bytes() == (
        tmp_path / "second" / "campaign.json"
    ).read_bytes()


def test_training_algorithm_campaign_rejects_tampering_and_reuse(tmp_path: Path) -> None:
    output = tmp_path / "h3"
    run_training_algorithm_campaign(output, seeds=(2, 4), example_budgets=(1, 2))
    document = json.loads((output / "campaign.json").read_text(encoding="utf-8"))
    document["runs"][0]["task_success_rate"] = 1.0
    with pytest.raises(ValueError, match=r"H3 result|identity"):
        TrainingAlgorithmCampaign.model_validate_json(json.dumps(document), strict=True)
    with pytest.raises(FileExistsError, match="absent or empty"):
        run_training_algorithm_campaign(output, seeds=(2, 4), example_budgets=(1, 2))
    result_path = output / document["runs"][0]["evidence_path"]
    result_path.write_bytes(result_path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="digest mismatch"):
        validate_training_algorithm_campaign(output)


def test_training_algorithm_campaign_rejects_resigned_metric_forgery(tmp_path: Path) -> None:
    output = tmp_path / "h3"
    run_training_algorithm_campaign(output, seeds=(31, 37), example_budgets=(1, 2))
    path = output / "campaign.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    document["runs"][0]["policy_kl"] = 9.0
    document["campaign_id"] = _digest(
        {key: value for key, value in document.items() if key != "campaign_id"}
    )
    path.write_text(json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n")
    with pytest.raises(ValueError, match="summary disagrees"):
        validate_training_algorithm_campaign(output)


def test_training_algorithm_campaign_rejects_symlinked_evidence(tmp_path: Path) -> None:
    output = tmp_path / "h3"
    campaign = run_training_algorithm_campaign(output, seeds=(41, 43), example_budgets=(1, 2))
    evidence = output / campaign.runs[0].evidence_path
    external = tmp_path / "external-evidence.json"
    external.write_bytes(evidence.read_bytes())
    evidence.unlink()
    evidence.symlink_to(external)
    with pytest.raises(ValueError, match="symbolic link"):
        validate_training_algorithm_campaign(output)
