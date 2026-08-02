from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import pytest
from _pytest.tmpdir import TempPathFactory

from sloforge.genesis.evaluation_campaigns.cegis import (
    CampaignValidationError,
    FaultDisposition,
    H3CampaignReport,
    VerificationStrategy,
    run_cegis_campaign,
    validate_cegis_campaign,
)
from sloforge.genesis.ir import canonical_json


@pytest.fixture(scope="module")
def campaign_directory(tmp_path_factory: TempPathFactory) -> Path:
    output = tmp_path_factory.mktemp("h3-campaign") / "run"
    run_cegis_campaign(
        output,
        base_seed=73129,
        seed_count=3,
        fuzz_cases_per_candidate=12,
    )
    return output


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_bytes(canonical_json(value) + b"\n")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_h3_campaign_runs_real_ablation_and_replays(
    campaign_directory: Path,
) -> None:
    report = validate_cegis_campaign(campaign_directory)
    aggregates = {item.strategy: item for item in report.aggregates}

    assert report.scope.hardware_backed is False
    assert report.scope.hardware_performance_claims is False
    assert report.scope.universal_proof is False
    assert report.scope.oracle == "bounded_policy_domain_enumeration"
    assert report.run_seeds == tuple(dict.fromkeys(report.run_seeds))
    assert len(report.run_seeds) == 3

    tests_only = aggregates[VerificationStrategy.TESTS_ONLY]
    fuzzing = aggregates[VerificationStrategy.FUZZING_ONLY]
    model_check = aggregates[VerificationStrategy.MODEL_CHECK_ONLY]
    full = aggregates[VerificationStrategy.FULL_CEGIS]
    assert tests_only.escaped_faults == tests_only.fault_instances
    assert 0 < fuzzing.detected_faults < fuzzing.fault_instances
    assert model_check.detected_faults == model_check.fault_instances
    assert model_check.escaped_faults == 0
    assert full.detected_faults == len(report.run_seeds)
    assert full.prevented_repeat_faults == len(report.run_seeds)
    assert full.escaped_faults == 0
    assert full.learned_constraint_reuses == len(report.run_seeds)
    assert full.repeated_fault_family_reevaluations == 0
    assert model_check.repeated_fault_family_reevaluations == len(report.run_seeds)
    assert full.median_initial_counterexample_events == 6.0
    assert full.median_minimized_counterexample_events == 3.0
    assert full.containment_interval.estimate == 1.0
    assert full.containment_interval.sample_unit == "candidate_fault_instance"


def test_h3_raw_records_preserve_detection_prevention_and_minimization(
    campaign_directory: Path,
) -> None:
    report = H3CampaignReport.model_validate_json(
        (campaign_directory / "report.json").read_bytes(), strict=True
    )
    records = [
        json.loads(line)
        for line in (campaign_directory / report.raw_records.path).read_text().splitlines()
    ]
    full_records = [
        item for item in records if item["strategy"] == VerificationStrategy.FULL_CEGIS.value
    ]

    for run_seed in report.run_seeds:
        seed_records = [item for item in full_records if item["run_seed"] == run_seed]
        assert [item["disposition"] for item in seed_records] == [
            FaultDisposition.DETECTED.value,
            FaultDisposition.PREVENTED_BY_LEARNED_CONSTRAINT.value,
            FaultDisposition.VALID_CONFIRMED.value,
        ]
        assert seed_records[0]["initial_counterexample_events"] == 6
        assert seed_records[0]["minimized_counterexample_events"] == 3
        assert seed_records[0]["minimization_evaluations"] > 0
        assert seed_records[1]["learned_constraint_id"].startswith("constraint-")
        assert seed_records[1]["verifier_invocations"] == 0


def test_h3_validator_rejects_unmodified_hash_tampering(
    campaign_directory: Path, tmp_path: Path
) -> None:
    altered = tmp_path / "altered"
    shutil.copytree(campaign_directory, altered)
    raw_path = altered / "raw_fault_records.jsonl"
    raw_path.write_bytes(raw_path.read_bytes() + b"{}\n")

    with pytest.raises(CampaignValidationError, match="changed"):
        validate_cegis_campaign(altered)


def test_h3_validator_replays_evidence_after_forged_hashes(
    campaign_directory: Path, tmp_path: Path
) -> None:
    altered = tmp_path / "forged"
    shutil.copytree(campaign_directory, altered)
    report_path = altered / "report.json"
    report = json.loads(report_path.read_bytes())
    raw_path = altered / report["raw_records"]["path"]
    records = [json.loads(line) for line in raw_path.read_text().splitlines()]
    record = next(item for item in records if item["strategy"] == "tests_only")
    evidence_path = altered / record["evidence_path"]
    evidence = json.loads(evidence_path.read_bytes())
    evidence["cases"][0]["outcome"]["evidence_id"] = "forged-evidence-id"
    _write_json(evidence_path, evidence)
    record["evidence_sha256"] = _sha256(evidence_path)
    raw_path.write_bytes(b"".join(canonical_json(item) + b"\n" for item in records))
    for artifact in report["artifacts"]:
        path = altered / artifact["path"]
        artifact["sha256"] = _sha256(path)
        artifact["size_bytes"] = path.stat().st_size
    report["raw_records"]["sha256"] = _sha256(raw_path)
    report["raw_records"]["size_bytes"] = raw_path.stat().st_size
    _write_json(report_path, report)

    with pytest.raises(CampaignValidationError, match="does not replay"):
        validate_cegis_campaign(altered)


@pytest.mark.parametrize(
    ("keyword", "value"),
    (("seed_count", 0), ("seed_count", 65), ("fuzz_cases_per_candidate", 0)),
)
def test_h3_campaign_enforces_bounded_configuration(
    tmp_path: Path, keyword: str, value: int
) -> None:
    arguments: dict[str, int] = {keyword: value}
    with pytest.raises(ValueError, match="must be in"):
        run_cegis_campaign(tmp_path / keyword, **arguments)
