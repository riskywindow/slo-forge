from __future__ import annotations

from pathlib import Path

import pytest

from sloforge.genesis.evaluation_campaigns.autopsy import (
    CampaignValidationError,
    SearchStrategy,
    run_autopsy_guided_campaign,
    validate_autopsy_guided_campaign,
)

ROOT = Path(__file__).resolve().parents[2]
DIAGNOSIS = ROOT / "tests/fixtures/autopsy/diagnosis-v1.json"
SEED = 73129


def _run(path: Path):
    return run_autopsy_guided_campaign(
        path,
        diagnosis_path=DIAGNOSIS,
        seed=SEED,
        count=3,
        maximum_candidates=8,
    )


def test_h4_campaign_compares_three_strategies_from_raw_candidates(tmp_path: Path) -> None:
    report = _run(tmp_path / "campaign")

    validate_autopsy_guided_campaign(report)
    assert tuple(item.strategy for item in report.aggregates) == tuple(SearchStrategy)
    assert len(report.run_seeds) == 3
    assert report.scope.evidence_scope == "synthetic_cpu_only"
    assert not report.scope.hardware_backed
    assert all(item.actual_hardware_experiments == 0 for item in report.aggregates)
    assert all(item.candidates_evaluated > 0 for item in report.aggregates)
    assert all(item.synthetic_high_fidelity_experiments > 0 for item in report.aggregates)
    assert report.aggregates[0].improvement_run_count > 0
    assert tuple(item.baseline for item in report.guided_deltas) == (
        SearchStrategy.RANDOM_REGION,
        SearchStrategy.UNRESTRICTED,
    )
    assert Path(report.raw_candidates_path).read_text(encoding="utf-8").count("\n") == sum(
        item.candidates_evaluated for item in report.aggregates
    )


def test_h4_campaign_is_deterministic_across_output_roots(tmp_path: Path) -> None:
    first = _run(tmp_path / "first")
    second = _run(tmp_path / "second")

    assert first.run_seeds == second.run_seeds
    assert first.aggregates == second.aggregates
    assert first.guided_deltas == second.guided_deltas
    assert first.raw_candidates_sha256 == second.raw_candidates_sha256
    assert (
        Path(first.raw_candidates_path).read_bytes()
        == Path(second.raw_candidates_path).read_bytes()
    )


def test_h4_validator_rejects_changed_raw_records(tmp_path: Path) -> None:
    report = _run(tmp_path / "campaign")
    raw = Path(report.raw_candidates_path)
    raw.write_bytes(raw.read_bytes() + b"\n")

    with pytest.raises(CampaignValidationError, match="raw candidates artifact"):
        validate_autopsy_guided_campaign(report)


def test_h4_validator_recomputes_summary_and_rejects_forgery(tmp_path: Path) -> None:
    report = _run(tmp_path / "campaign")
    guided = report.aggregates[0]
    forged = report.model_copy(
        update={
            "aggregates": (
                guided.model_copy(update={"invalid_candidates": guided.invalid_candidates + 1}),
                *report.aggregates[1:],
            )
        }
    )

    with pytest.raises(CampaignValidationError, match="aggregates are not derived"):
        validate_autopsy_guided_campaign(forged)


def test_h4_random_surface_is_seeded_and_equal_sized_to_guided_surface(tmp_path: Path) -> None:
    report = _run(tmp_path / "campaign")
    guided = report.aggregates[0]
    random_region = report.aggregates[1]

    assert all(
        len(random_item.mutation_surface) == len(guided_item.mutation_surface)
        for guided_item, random_item in zip(guided.per_seed, random_region.per_seed, strict=True)
    )
    assert len({item.mutation_surface for item in random_region.per_seed}) > 1


def test_h4_campaign_bounds_seed_and_candidate_counts(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="campaign count"):
        run_autopsy_guided_campaign(
            tmp_path / "too-many",
            diagnosis_path=DIAGNOSIS,
            seed=SEED,
            count=65,
        )
    with pytest.raises(ValueError, match="maximum candidates"):
        run_autopsy_guided_campaign(
            tmp_path / "too-many-candidates",
            diagnosis_path=DIAGNOSIS,
            seed=SEED,
            maximum_candidates=65,
        )
