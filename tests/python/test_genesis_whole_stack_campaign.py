from __future__ import annotations

from pathlib import Path

import pytest

from sloforge.genesis.evaluation_campaigns.whole_stack import (
    WholeStackValidationError,
    run_whole_stack_campaign,
    validate_whole_stack_campaign,
)
from sloforge.genesis.ir import TransformationFamily

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "models/reference_tasks/hybrid_decoder"
FABRIC = ROOT / "tests/fixtures/fabric/physical-execution-plan-v1.json"


def test_whole_stack_campaign_replays_real_cross_layer_candidate(tmp_path: Path) -> None:
    report = run_whole_stack_campaign(
        tmp_path / "campaign",
        seeds=(73129, 73130, 73131),
        reference_package=PACKAGE,
        fabric_fixture=FABRIC,
    )

    validate_whole_stack_campaign(report)
    assert report.hardware_backed_runs == 0
    assert report.h2_conclusion == "supported_in_declared_synthetic_scope"
    assert report.h2_effect.confidence_low > 0
    assert report.h2_effect.positive_difference_favors == "genesis"
    assert report.h2_effect.resampling_unit == "paired_workload_seed"
    assert all(value > 0 for value in report.h2_effect.per_seed_differences)
    assert report.h9_conclusion == "not_supported"
    assert report.h9_effect.confidence_high < 0
    assert all(value < 0 for value in report.h9_effect.per_seed_differences)
    for result in report.results:
        assert result.transformation_families == (
            TransformationFamily.BATCHING,
            TransformationFamily.STATE_LAYOUT,
        )
        assert result.affected_genome_regions == ("request", "serving", "state")
        assert result.category_evidence.distributed_performance_comparison_eligible is False
        assert result.category_evidence.tensor_source_nodes == 2
        assert result.category_evidence.tensor_target_nodes == 1


def test_whole_stack_validator_rejects_raw_and_summary_tampering(tmp_path: Path) -> None:
    report = run_whole_stack_campaign(
        tmp_path / "campaign",
        seeds=(73129, 73130, 73131),
        reference_package=PACKAGE,
        fabric_fixture=FABRIC,
    )
    raw_path = Path(report.raw_results_path)
    original = raw_path.read_bytes()
    raw_path.write_bytes(original.replace(b'"objective_units":', b'"objective_units":9'))
    with pytest.raises(WholeStackValidationError, match="digest"):
        validate_whole_stack_campaign(report)
    raw_path.write_bytes(original)

    forged = report.model_copy(
        update={
            "h2_effect": report.h2_effect.model_copy(
                update={"mean_difference": report.h2_effect.mean_difference + 1.0}
            )
        }
    )
    with pytest.raises(WholeStackValidationError, match="paired effects"):
        validate_whole_stack_campaign(forged)
