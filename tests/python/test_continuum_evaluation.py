from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from sloforge.continuum.benchmarking import (
    EvaluationBundle,
    EvaluationRequest,
    load_evaluation,
    run_evaluation,
    run_evaluation_campaign,
)


def _request(output: Path) -> EvaluationRequest:
    return EvaluationRequest(
        output_dir=output,
        seeds=(101, 202, 303),
        git_commit="7e51ea7f7338755d23f889820558a4e046d6c42e",
        initial_output_tokens=8,
        delta_rounds=(1, 1),
        resumed_tokens=2,
        converter_repetitions=3,
    )


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_multiseed_campaign_preserves_raw_results_manifests_statistics_and_commands(
    tmp_path: Path,
) -> None:
    root = tmp_path / "evaluation"
    campaign = run_evaluation_campaign(_request(root))
    restored = load_evaluation(root / campaign.summary_artifact.path)

    assert isinstance(restored, EvaluationBundle)
    assert restored == campaign.evaluation
    assert restored.seeds == (101, 202, 303)
    assert "--seeds 101,202,303" in restored.exact_command
    assert restored.exact_command.endswith("--reset")
    assert restored.hardware.mode == "cpu_only"
    assert not restored.hardware.gpu_exercised
    assert not restored.hardware.rdma_exercised
    assert restored.hardware.hardware_result_claims == ()
    assert {item.metric_class for item in restored.confidence_intervals} == {
        "artifact_derived",
        "observed_host",
        "synthetic_protocol",
    }
    assert all(item.sample_count == 3 for item in restored.confidence_intervals)
    assert all(item.lower <= item.mean <= item.upper for item in restored.confidence_intervals)

    for item in restored.per_seed:
        flagship = root / item.flagship_artifact.path
        conversion = root / item.conversion_artifact.path
        stop_copy = root / item.stop_and_copy_artifact.path
        planner = root / item.planner_artifact.path
        measurements = root / item.measurement_artifact.path
        assert flagship.is_file() and conversion.is_file() and stop_copy.is_file()
        assert planner.is_file() and measurements.is_file()
        assert _digest(flagship) == item.flagship_artifact.sha256
        assert _digest(conversion) == item.conversion_artifact.sha256
        assert item.conversion_exact
        assert item.conversion_maximum_absolute_error == 0.0
        assert item.gateway_duplicate_count == 0
        assert item.gateway_gap_count == 0
        assert item.failed_transaction_final_phase == "ROLLED_BACK"
        assert item.successful_transaction_final_phase == "COMPLETED"
        assert item.direct_reuse_class == "incompatible"
        assert item.recomputation_class == "recomputation_assisted"
        assert item.observed_precopy_interruption_ns > 0
        assert item.observed_stop_and_copy_interruption_ns > 0
        assert item.planner_regret == 0.0
        assert item.planner_objective <= item.fixed_stop_objective
        assert item.planner_predicted_interruption_ms >= 0
        assert item.planner_observed_interruption_ms > 0
        assert item.planner_interruption_absolute_error_ms == abs(
            item.planner_predicted_interruption_ms - item.planner_observed_interruption_ms
        )

    statuses = {item.hypothesis: item.status for item in restored.hypotheses}
    assert statuses["H1"] == "pass"
    assert statuses["H3"] in {"pass", "mixed", "negative"}
    assert statuses["H4"] == "pass"
    assert statuses["H5"] == "pass"
    assert any("No GPU" in item for item in restored.negative_results)
    migrated = {item.runtime for item in restored.adapters if item.migration_exercised}
    assert migrated == {
        "continuum-reference-token-major",
        "continuum-reference-head-major",
    }


def test_protocol_artifacts_are_seed_deterministic_despite_observed_timing_variance(
    tmp_path: Path,
) -> None:
    first = run_evaluation(_request(tmp_path / "first"))
    second = run_evaluation(_request(tmp_path / "second"))

    assert first.evaluation_id == second.evaluation_id
    assert tuple(item.flagship_artifact.sha256 for item in first.per_seed) == tuple(
        item.flagship_artifact.sha256 for item in second.per_seed
    )
    assert tuple(item.synthetic_transport_bytes_on_wire for item in first.per_seed) == tuple(
        item.synthetic_transport_bytes_on_wire for item in second.per_seed
    )
    assert tuple(item.checkpoint_bytes_deduplicated for item in first.per_seed) == tuple(
        item.checkpoint_bytes_deduplicated for item in second.per_seed
    )


def test_loader_rejects_resealed_aggregate_not_derived_from_raw_seeds(
    tmp_path: Path,
) -> None:
    root = tmp_path / "resealed-summary"
    campaign = run_evaluation_campaign(_request(root))
    summary = root / campaign.summary_artifact.path
    document = json.loads(summary.read_text(encoding="utf-8"))
    interval = document["confidence_intervals"][0]
    interval["mean"] += 1000.0
    interval["upper"] = interval["mean"] + 1.0
    interval["lower"] = interval["mean"] - 1.0
    summary.write_text(json.dumps(document, sort_keys=True), encoding="utf-8")
    resealed_digest = _digest(summary)
    assert resealed_digest != campaign.summary_artifact.sha256

    with pytest.raises(ValueError, match="confidence interval summary differs"):
        load_evaluation(summary)
