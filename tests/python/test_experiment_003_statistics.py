from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_MODULE_PATH = _ROOT / "benchmarks/branchfabric/experiment_003_statistics.py"
_SPEC = importlib.util.spec_from_file_location(
    "sloforge_experiment_003_statistics_tests",
    _MODULE_PATH,
)
assert _SPEC is not None and _SPEC.loader is not None
_STATS = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _STATS
_SPEC.loader.exec_module(_STATS)


def _provenance(label: str):
    return _STATS.ArtifactProvenance(
        artifact_reference=f"raw/{label}.json",
        artifact_sha256=(label[0] if label[0] in "abcdef" else "a") * 64,
        sample_selector=f"$.samples[{label!r}]",
    )


def _key(seed: int, *, repetition: int = 0):
    return _STATS.PairKey(
        pair_id=f"seed-{seed}",
        seed=seed,
        repetition=repetition,
        workload_digest="workload-sha256",
        function_call_id="fc-same-container",
        gpu_uuid="GPU-same-a100",
    )


def _paired_observations():
    values = ((41, 100.0, 90.0), (73, 100.0, 95.0), (113, 100.0, 105.0))
    return tuple(
        _STATS.PairedObservation(
            key=_key(seed),
            baseline_value=baseline,
            candidate_value=candidate,
            baseline_provenance=_provenance(f"baseline-{seed}"),
            candidate_provenance=_provenance(f"candidate-{seed}"),
        )
        for seed, baseline, candidate in values
    )


def test_three_pair_effect_preserves_raw_pairs_and_has_no_tail_estimates() -> None:
    observations = _paired_observations()
    first = _STATS.analyze_three_pair_effect(
        observations,
        metric="post_root_ready",
        unit="ms",
        baseline_label="independent",
        candidate_label="shared_root",
        bootstrap_seed=149,
        bootstrap_repetitions=2000,
    )
    second = _STATS.analyze_three_pair_effect(
        observations,
        metric="post_root_ready",
        unit="ms",
        baseline_label="independent",
        candidate_label="shared_root",
        bootstrap_seed=149,
        bootstrap_repetitions=2000,
    )

    assert first == second
    assert first["raw_differences"] == [-10.0, -5.0, 5.0]
    assert first["summary"]["median_difference"] == -5.0
    assert first["summary"]["minimum_difference"] == -10.0
    assert first["summary"]["maximum_difference"] == 5.0
    assert first["summary"]["range_difference"] == 15.0
    assert first["effect_sizes"]["paired_cohens_dz"] == pytest.approx(-0.4364357804719847)
    assert first["effect_sizes"]["matched_pairs_rank_biserial"] == -0.5
    assert first["tail_statistics"] == {
        "p95": None,
        "p99": None,
        "status": "TAIL_INSUFFICIENT",
        "reason": "three paired trials do not support tail-percentile inference",
    }
    assert [item["pair_key"]["seed"] for item in first["raw_pairs"]] == [41, 73, 113]
    assert first["raw_pairs"][0]["baseline_provenance"]["artifact_reference"] == (
        "raw/baseline-41.json"
    )
    assert {interval["statistic"] for interval in first["bootstrap_confidence_intervals"]} == {
        "mean",
        "median",
    }
    assert all(
        interval["resampling_unit"] == "whole matched pair difference"
        for interval in first["bootstrap_confidence_intervals"]
    )
    json.dumps(first, allow_nan=False)


def test_three_pair_effect_rejects_unmatched_cardinality_and_duplicate_keys() -> None:
    observations = _paired_observations()
    kwargs: dict[str, Any] = {
        "metric": "post_root_ready",
        "unit": "ms",
        "baseline_label": "before",
        "candidate_label": "after",
        "bootstrap_seed": 41,
        "bootstrap_repetitions": 100,
    }
    with pytest.raises(ValueError, match="exactly three"):
        _STATS.analyze_three_pair_effect(observations[:2], **kwargs)
    with pytest.raises(ValueError, match="duplicate typed pair key"):
        _STATS.analyze_three_pair_effect(
            (observations[0], observations[0], observations[2]),
            **kwargs,
        )


def test_zero_variance_pair_effect_is_reported_without_infinity() -> None:
    observations = tuple(
        _STATS.PairedObservation(
            key=_key(seed),
            baseline_value=100.0,
            candidate_value=90.0,
            baseline_provenance=_provenance(f"before-{seed}"),
            candidate_provenance=_provenance(f"after-{seed}"),
        )
        for seed in (41, 73, 113)
    )
    result = _STATS.analyze_three_pair_effect(
        observations,
        metric="metadata",
        unit="ms",
        baseline_label="before",
        candidate_label="after",
        bootstrap_seed=73,
        bootstrap_repetitions=100,
    )
    assert result["effect_sizes"]["paired_cohens_dz"] is None
    assert result["effect_sizes"]["matched_pairs_rank_biserial"] == -1.0
    json.dumps(result, allow_nan=False)


def test_trace_overhead_uses_disabled_control_ratios_and_raw_provenance() -> None:
    observations = (
        _STATS.TraceOverheadObservation(
            key=_key(41),
            disabled_value=100.0,
            minimal_value=102.0,
            full_value=110.0,
            disabled_provenance=_provenance("disabled-41"),
            minimal_provenance=_provenance("minimal-41"),
            full_provenance=_provenance("full-41"),
        ),
        _STATS.TraceOverheadObservation(
            key=_key(73),
            disabled_value=200.0,
            minimal_value=210.0,
            full_value=240.0,
            disabled_provenance=_provenance("disabled-73"),
            minimal_provenance=_provenance("minimal-73"),
            full_provenance=_provenance("full-73"),
        ),
    )
    result = _STATS.analyze_trace_overhead(
        observations,
        metric="post_root_ready",
        unit="ms",
    )

    assert result["raw_controls"][0]["minimal_to_disabled_ratio"] == 1.02
    assert result["raw_controls"][0]["full_to_disabled_ratio"] == 1.1
    assert result["minimal_trace_overhead"]["median"] == pytest.approx(0.035)
    assert result["full_trace_overhead"]["median"] == pytest.approx(0.15)
    assert result["tail_statistics"]["p95"] is None
    assert result["raw_controls"][1]["provenance"]["full"]["artifact_reference"] == (
        "raw/full-73.json"
    )


def _scaling_observations(
    values: tuple[tuple[int, int, float], ...],
):
    return tuple(
        _STATS.ScalingObservation(
            fanout=fanout,
            prefix_blocks=prefix_blocks,
            value=value,
            repetition=0,
            provenance=_provenance(f"scale-{fanout}"),
        )
        for fanout, prefix_blocks, value in values
    )


def test_scaling_selects_linear_and_marks_constant_prefix_alias() -> None:
    observations = _scaling_observations(
        ((1, 1024, 7.0), (8, 1024, 21.0), (16, 1024, 37.0), (32, 1024, 69.0))
    )
    result = _STATS.select_scaling_model(
        observations,
        metric="metadata_cpu_time",
        unit="ms",
        relative_uncertainty=0.0,
    )

    assert result["selected_model"] == "O(N)"
    candidates = {candidate["model"]: candidate for candidate in result["candidate_models"]}
    assert candidates["O(N)"]["rmse"] == pytest.approx(0.0, abs=1e-12)
    assert candidates["O(N*prefix_blocks)"]["identifiable"] is False
    assert candidates["O(N*prefix_blocks)"]["aliased_with"] == "O(N)"
    assert result["raw_observations"][0]["provenance"]["sample_selector"]
    assert result["tail_statistics"]["p99"] is None


def test_scaling_selects_constant_or_n_times_prefix_blocks_when_justified() -> None:
    constant = _STATS.select_scaling_model(
        _scaling_observations(((1, 1024, 5.0), (8, 1024, 5.0), (16, 1024, 5.0), (32, 1024, 5.0))),
        metric="fixed_cost",
        unit="ms",
        relative_uncertainty=0.0,
    )
    assert constant["selected_model"] == "O(1)"

    product = _STATS.select_scaling_model(
        _scaling_observations(
            tuple(
                (fanout, prefix_blocks, 3.0 + 0.01 * fanout * prefix_blocks)
                for fanout, prefix_blocks in ((1, 64), (8, 128), (16, 256), (32, 512))
            )
        ),
        metric="block_operations",
        unit="count",
        relative_uncertainty=0.0,
    )
    assert product["selected_model"] == "O(N*prefix_blocks)"


def test_scaling_rejects_duplicate_observation_keys() -> None:
    observation = _STATS.ScalingObservation(
        fanout=8,
        prefix_blocks=1024,
        value=10.0,
        repetition=0,
        provenance=_provenance("scale-duplicate"),
    )
    with pytest.raises(ValueError, match="duplicate typed observation key"):
        _STATS.select_scaling_model(
            (observation, observation, observation),
            metric="metadata",
            unit="ms",
        )


def test_amdahl_projections_cover_2x_5x_10x_and_free() -> None:
    result = _STATS.amdahl_metadata_projections(
        post_root_ready_ms=100.0,
        metadata_ms=20.0,
        fanout=8,
        root_inclusive_ready_ms=200.0,
        post_root_process_cpu_ms=50.0,
        metadata_process_cpu_ms=10.0,
        provenance=_provenance("amdahl"),
    )
    projections = {item["metadata_speedup"]: item for item in result["projections"]}

    assert set(projections) == {"2x", "5x", "10x", "free"}
    assert projections["2x"]["projected_post_root_ready_ms"] == 90.0
    assert projections["5x"]["projected_post_root_ready_ms"] == 84.0
    assert projections["10x"]["projected_post_root_ready_ms"] == 82.0
    assert projections["free"]["projected_post_root_ready_ms"] == 80.0
    assert projections["free"]["post_root_ready_speedup"] == 1.25
    assert projections["free"]["projected_valid_branches_per_second"] == 100.0
    assert projections["free"]["projected_root_inclusive_ready_ms"] == 180.0
    assert projections["free"]["projected_post_root_process_cpu_ms"] == 40.0
    assert projections["free"]["post_root_process_cpu_reduction_fraction"] == 0.2
    assert projections["free"]["projected_process_cpu_to_wall_ratio"] == 0.5
    assert result["baseline_valid_branches_per_second"] == 80.0
    assert result["baseline_process_cpu_to_wall_ratio"] == 0.5
    assert result["ideal_metadata_free_speedup"] == 1.25
    assert result["provenance"]["artifact_reference"] == "raw/amdahl.json"
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize(
    ("total", "metadata", "root"),
    [(0.0, 0.0, None), (10.0, 11.0, None), (10.0, 1.0, 9.0)],
)
def test_amdahl_rejects_invalid_readiness_inputs(
    total: float,
    metadata: float,
    root: float | None,
) -> None:
    with pytest.raises(ValueError):
        _STATS.amdahl_metadata_projections(
            post_root_ready_ms=total,
            metadata_ms=metadata,
            fanout=8,
            root_inclusive_ready_ms=root,
            provenance=_provenance("amdahl-invalid"),
        )
