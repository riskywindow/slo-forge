from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

import pytest

from sloforge.genesis.evaluation_campaigns.lineage import (
    H5CampaignValidationError,
    H5CaseResult,
    H5LineageCampaignReport,
    LineageScenario,
    RawCandidateRecord,
    run_h5_lineage_campaign,
    validate_h5_lineage_campaign,
)
from sloforge.genesis.ir import canonical_json
from sloforge.lineage import EvidenceFreshness, LineageStore, TransferOutcome


def _case(
    report: H5LineageCampaignReport,
    seed: int,
    scenario: LineageScenario,
) -> H5CaseResult:
    return next(item for item in report.cases if item.seed == seed and item.scenario is scenario)


def _raw(root: Path, relative_path: str) -> tuple[RawCandidateRecord, ...]:
    return tuple(
        RawCandidateRecord.model_validate_json(line, strict=True)
        for line in (root / relative_path).read_bytes().splitlines()
    )


def test_h5_campaign_exercises_transfer_reverification_and_invalidation(
    tmp_path: Path,
) -> None:
    root = tmp_path / "campaign"
    report = run_h5_lineage_campaign(root, seed=73129, count=3)

    assert validate_h5_lineage_campaign(root) == report
    assert report.hardware_backed_runs == 0
    assert len(report.cases) == 3 * 5
    assert report.conclusion == "supported_in_deterministic_synthetic_scope"
    assert report.effect_summary.related_faster_to_first_correct_every_seed
    assert report.effect_summary.related_faster_to_first_improved_every_seed
    assert report.effect_summary.related_better_final_objective_every_seed
    assert report.effect_summary.stale_seed_suppressed_after_invalidation_every_seed

    for seed in report.seeds:
        empty = _case(report, seed, LineageScenario.EMPTY)
        unrelated = _case(report, seed, LineageScenario.UNRELATED)
        related = _case(report, seed, LineageScenario.RELATED)
        stale_before = _case(report, seed, LineageScenario.STALE_BEFORE_INVALIDATION)
        stale_after = _case(report, seed, LineageScenario.STALE_AFTER_INVALIDATION)

        assert related.metrics.candidate_units_to_first_improved == 1
        assert related.metrics.time_units_to_first_improved < (
            empty.metrics.time_units_to_first_improved
        )
        assert related.metrics.final_objective < empty.metrics.final_objective
        assert unrelated.metrics.negative_transfers == 1
        assert stale_before.metrics.negative_transfers == 1
        assert stale_after.metrics.negative_transfers == 0
        assert stale_after.metrics == empty.metrics
        assert stale_after.invalidated_evidence_count == 1
        assert not stale_after.initialization.lineage_seeds

        related_raw = _raw(root, related.raw_candidates_path)
        unrelated_raw = _raw(root, unrelated.raw_candidates_path)
        assert len(related_raw) == report.population_size
        assert related_raw[0].proposal_kind == "lineage"
        assert related_raw[0].preconditions_satisfied is True
        assert related_raw[0].reverification_required is True
        assert related_raw[0].reverification_passed is True
        assert related_raw[0].transfer_outcome is TransferOutcome.IMPROVED
        assert unrelated_raw[0].preconditions_satisfied is True
        assert unrelated_raw[0].reverification_passed is False
        assert unrelated_raw[0].transfer_outcome is TransferOutcome.NEGATIVE_TRANSFER
        assert all(not item.hardware_backed for item in (*related_raw, *unrelated_raw))

        with LineageStore(root / stale_after.initial_lineage_store_path) as store:
            transformation_evidence = tuple(
                item for item in store.list_evidence() if item.target_id.startswith("transferable-")
            )
            assert len(store.list_invalidations()) == 1
            assert len(transformation_evidence) == 1
            assert transformation_evidence[0].freshness is EvidenceFreshness.STALE
        with LineageStore(root / unrelated.evaluated_lineage_store_path) as store:
            transfers = store.list_transfers()
            assert len(transfers) == 1
            assert transfers[0].outcome is TransferOutcome.NEGATIVE_TRANSFER


def test_h5_campaign_is_byte_deterministic_across_output_directories(
    tmp_path: Path,
) -> None:
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    first = run_h5_lineage_campaign(first_root, seed=37, count=2)
    second = run_h5_lineage_campaign(second_root, seed=37, count=2)

    assert first == second
    assert (first_root / "report.json").read_bytes() == (second_root / "report.json").read_bytes()
    for case in first.cases:
        assert (first_root / case.raw_candidates_path).read_bytes() == (
            second_root / case.raw_candidates_path
        ).read_bytes()


def test_h5_validator_rejects_raw_candidate_and_metric_tampering(
    tmp_path: Path,
) -> None:
    raw_root = tmp_path / "raw-tamper"
    raw_report = run_h5_lineage_campaign(raw_root, seed=101, count=2)
    raw_path = raw_root / raw_report.cases[0].raw_candidates_path
    raw_path.write_bytes(raw_path.read_bytes() + b"{}\n")
    with pytest.raises(H5CampaignValidationError, match="raw candidate digest changed"):
        validate_h5_lineage_campaign(raw_root)

    metric_root = tmp_path / "metric-tamper"
    metric_report = run_h5_lineage_campaign(metric_root, seed=101, count=2)
    first_case = metric_report.cases[0]
    tampered_case = first_case.model_copy(
        update={
            "metrics": first_case.metrics.model_copy(
                update={"final_objective": first_case.metrics.final_objective - 10.0}
            )
        }
    )
    tampered_report = metric_report.model_copy(
        update={"cases": (tampered_case, *metric_report.cases[1:])}
    )
    (metric_root / "report.json").write_bytes(canonical_json(tampered_report) + b"\n")
    with pytest.raises(H5CampaignValidationError, match="case metrics"):
        validate_h5_lineage_campaign(metric_root)


def test_h5_validator_rejects_lineage_store_tampering(tmp_path: Path) -> None:
    root = tmp_path / "store-tamper"
    report = run_h5_lineage_campaign(root, seed=501, count=2)
    store_path = root / report.cases[0].initial_lineage_store_path
    store_path.write_bytes(store_path.read_bytes() + b"tamper")

    with pytest.raises(H5CampaignValidationError, match="initial lineage store digest"):
        validate_h5_lineage_campaign(root)


def test_h5_validator_rejects_semantic_store_tampering_with_updated_digest(
    tmp_path: Path,
) -> None:
    root = tmp_path / "semantic-store-tamper"
    report = run_h5_lineage_campaign(root, seed=701, count=2)
    case_index = next(
        index for index, item in enumerate(report.cases) if item.scenario is LineageScenario.RELATED
    )
    case = report.cases[case_index]
    store_path = root / case.initial_lineage_store_path
    with sqlite3.connect(store_path) as connection:
        transformation_id, document = connection.execute(
            "SELECT transformation_id, document FROM transformations"
        ).fetchone()
        payload = json.loads(document)
        payload["expected_benefit"] = 0.99
        connection.execute(
            "UPDATE transformations SET document = ? WHERE transformation_id = ?",
            (json.dumps(payload, sort_keys=True), transformation_id),
        )
        connection.commit()
    tampered_case = case.model_copy(
        update={"initial_lineage_store_sha256": hashlib.sha256(store_path.read_bytes()).hexdigest()}
    )
    cases = list(report.cases)
    cases[case_index] = tampered_case
    tampered_report = report.model_copy(update={"cases": tuple(cases)})
    (root / "report.json").write_bytes(canonical_json(tampered_report) + b"\n")

    with pytest.raises(H5CampaignValidationError, match="scenario fixture"):
        validate_h5_lineage_campaign(root)


def test_h5_campaign_rejects_invalid_budget_and_existing_output(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=r"at least|between 2 and 100"):
        run_h5_lineage_campaign(tmp_path / "too-small", seed=1, count=1)
    with pytest.raises(ValueError, match="non-negative"):
        run_h5_lineage_campaign(tmp_path / "negative", seed=-1, count=2)

    output = tmp_path / "existing"
    output.mkdir()
    with pytest.raises(FileExistsError, match="already exists"):
        run_h5_lineage_campaign(output, seed=1, count=2)


def test_h5_report_is_strict_and_has_no_hardware_claim(tmp_path: Path) -> None:
    root = tmp_path / "strict"
    report = run_h5_lineage_campaign(root, seed=900, count=2)
    payload = json.loads((root / "report.json").read_text(encoding="utf-8"))
    payload["hardware_backed_runs"] = 1
    (root / "report.json").write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(H5CampaignValidationError, match="strict schema"):
        validate_h5_lineage_campaign(root)
    assert report.hardware_backed_runs == 0
