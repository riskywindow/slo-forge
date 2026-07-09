from __future__ import annotations

import json
from pathlib import Path

import pytest

import sloforge.genesis.evaluation_suite as suite
from sloforge.genesis.evaluation_suite import (
    CampaignRecord,
    EvaluationStatus,
    EvaluationSuiteConfiguration,
    EvaluationSuiteValidationError,
    GenesisEvaluationSuiteReport,
    HypothesisOutcome,
    HypothesisResult,
    InputSnapshot,
    _build_manifest,
    _campaign_seeds,
    _h1_configuration,
    _reference,
    _tree_digest,
    _tree_files,
    _write_once,
    validate_genesis_evaluation_suite,
)
from sloforge.genesis.ir import canonical_json

CAMPAIGNS = ("core", "h1", "h2_h9", "h3", "h4", "h5", "h6", "h7", "h8")


def _fixture_suite(
    root: Path,
) -> tuple[GenesisEvaluationSuiteReport, tuple[CampaignRecord, ...], tuple[HypothesisResult, ...]]:
    root.mkdir()
    configuration = EvaluationSuiteConfiguration(
        seed=17,
        core_run_count=2,
        campaign_seed_count=3,
        h1_task_count=1,
    )
    _write_once(root / "configuration.json", configuration)
    inputs = (
        ("reference_package", root / "inputs/reference-package/model.py"),
        ("autopsy_diagnosis", root / "inputs/autopsy-diagnosis.json"),
        ("fabric_plan", root / "inputs/physical-execution-plan-v1.json"),
    )
    for _identifier, path in inputs:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"artifact:{path.name}\n", encoding="utf-8")
    snapshots = tuple(
        InputSnapshot(
            input_id=identifier,  # type: ignore[arg-type]
            root=(path.parent if identifier == "reference_package" else path)
            .relative_to(root)
            .as_posix(),
            tree_sha256=_tree_digest(path.parent if identifier == "reference_package" else path),
            file_count=len(_tree_files(path.parent if identifier == "reference_package" else path)),
        )
        for identifier, path in inputs
    )
    campaign_records: list[CampaignRecord] = []
    for campaign_id in CAMPAIGNS:
        report_path = root / "campaigns" / campaign_id / "report.json"
        _write_once(report_path, {"campaign_id": campaign_id})
        campaign_records.append(
            CampaignRecord(
                campaign_id=campaign_id,  # type: ignore[arg-type]
                hypothesis_ids=(),
                status=EvaluationStatus.COMPLETED,
                evidence_scope="focused suite validator fixture",
                hardware_backed_runs=0,
                report=_reference(root, report_path),
                native_validator="focused_test_validator",
                limitations=("fixture exercises root composition only",),
            )
        )
    hypotheses = tuple(
        HypothesisResult(
            hypothesis_id=f"H{index}",  # type: ignore[arg-type]
            statement=f"H{index} focused fixture statement",
            status=EvaluationStatus.COMPLETED,
            outcome=HypothesisOutcome.DESCRIPTIVE_ONLY,
            source_campaign_id="focused-fixture",
            evidence_scope="focused suite validator fixture",
            hardware_backed_runs=0,
            metrics=(),
            limitations=("fixture exercises root composition only",),
        )
        for index in range(1, 10)
    )
    manifest = _build_manifest(root)
    manifest_path = root / "artifact-manifest.json"
    _write_once(manifest_path, manifest)
    report_path = root / "GENESIS_EVALUATION_SUITE.json"
    report = GenesisEvaluationSuiteReport(
        configuration=configuration,
        input_snapshots=snapshots,
        artifact_manifest=_reference(root, manifest_path),
        campaigns=tuple(campaign_records),
        hypotheses=hypotheses,
        hardware_backed_campaigns=0,
        synthetic_or_cpu_only_campaigns=9,
        report_path=str(report_path.resolve()),
    )
    _write_once(report_path, report)
    return report, tuple(campaign_records), hypotheses


def test_suite_configuration_derives_distinct_bounded_seeds() -> None:
    configuration = EvaluationSuiteConfiguration(
        seed=73129,
        core_run_count=2,
        campaign_seed_count=3,
        h1_task_count=2,
        h1_synthesis_seed_count=2,
    )

    assert _campaign_seeds(configuration) == (74129, 74130, 74131)
    h1 = _h1_configuration(configuration)
    assert h1.grammar.seed == 76129
    assert h1.grammar.count == 2
    assert h1.synthesis_seeds == (75129, 75130)
    assert len(set((*_campaign_seeds(configuration), *h1.synthesis_seeds))) == 5


def test_suite_validator_reopens_manifest_and_derived_claims(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report, campaigns, hypotheses = _fixture_suite(tmp_path / "suite")
    monkeypatch.setattr(
        suite, "_load_native_records", lambda _root, _config: (campaigns, hypotheses)
    )

    assert validate_genesis_evaluation_suite(Path(report.report_path)) == report


def test_suite_validator_rejects_raw_artifact_tampering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report, campaigns, hypotheses = _fixture_suite(tmp_path / "suite")
    monkeypatch.setattr(
        suite, "_load_native_records", lambda _root, _config: (campaigns, hypotheses)
    )
    evidence = tmp_path / "suite/campaigns/h3/report.json"
    evidence.write_bytes(evidence.read_bytes() + b"tampered")

    with pytest.raises(EvaluationSuiteValidationError, match="manifest does not match"):
        validate_genesis_evaluation_suite(Path(report.report_path))


def test_suite_validator_rejects_report_claim_rewrite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report, campaigns, hypotheses = _fixture_suite(tmp_path / "suite")
    monkeypatch.setattr(
        suite, "_load_native_records", lambda _root, _config: (campaigns, hypotheses)
    )
    report_path = Path(report.report_path)
    document = json.loads(report_path.read_bytes())
    document["hypotheses"][0]["outcome"] = "supported_in_declared_scope"
    report_path.write_bytes(canonical_json(document) + b"\n")

    with pytest.raises(EvaluationSuiteValidationError, match="claims are not derived"):
        validate_genesis_evaluation_suite(report_path)


def test_suite_report_rejects_incomplete_campaign_matrix(tmp_path: Path) -> None:
    report, _campaigns, _hypotheses = _fixture_suite(tmp_path / "suite")

    with pytest.raises(ValueError, match="campaign order"):
        report.model_copy(update={"campaigns": report.campaigns[:-1]}).__class__.model_validate(
            report.model_copy(update={"campaigns": report.campaigns[:-1]}).model_dump(),
            strict=True,
        )
