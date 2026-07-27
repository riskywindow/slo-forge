from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from sloforge.helix.characterization.fanout_vertical import (
    FanoutImplementation,
    FanoutStudyConfig,
    FanoutStudyReport,
    WorkloadClass,
    WorkloadSpec,
    run_fanout_study,
    write_fanout_study,
)


def _config() -> FanoutStudyConfig:
    return FanoutStudyConfig(
        workloads=(
            WorkloadSpec(
                workload_class=WorkloadClass.CODING_AGENT,
                context_tokens=32,
                suffix_tokens=2,
                common_suffix_tokens=0,
                divergence_pattern="immediate_rng_divergence",
            ),
            WorkloadSpec(
                workload_class=WorkloadClass.REASONING_VERIFICATION,
                context_tokens=48,
                suffix_tokens=3,
                common_suffix_tokens=1,
                divergence_pattern="late_rng_divergence",
            ),
        ),
        bootstrap_repetitions=100,
    )


@pytest.fixture(scope="module")
def study():
    config = _config()
    fixtures, samples = run_fanout_study(config)
    return config, fixtures, samples


def test_cpu_reference_fixtures_use_exact_continuum_state_bytes(study) -> None:
    config, fixtures, _samples = study

    assert len(fixtures) == len(config.seeds) * 2
    assert {item.model_state_class for item in fixtures} == {"CPU_REFERENCE_MODEL_STATE"}
    assert all(item.reference_adapter_max_context_tokens == 4096 for item in fixtures)
    assert all(
        item.physical_byte_accounting
        == "DETERMINISTIC_SERIALIZED_STATE_BYTES_NOT_RSS_OS_PAGES_OR_GPU_ALLOCATION"
        for item in fixtures
    )
    assert all(item.serialized_model_state_bytes > item.attention_kv_bytes > 0 for item in fixtures)
    assert all(
        item.runtime_metadata_bytes > 0
        and item.page_metadata_bytes > 0
        and item.segment_count > 0
        and item.page_count > 0
        for item in fixtures
    )
    assert all(
        len(item.suffix_private_bytes_per_branch) == max(config.fanouts) for item in fixtures
    )
    assert all(set(item.suffix_all_branches_ready_ns) == set(config.fanouts) for item in fixtures)
    assert all(not item.real_transformer_measured and not item.gpu_measured for item in fixtures)


def test_randomized_raw_samples_cover_fanout_workloads_seeds_and_repetitions(study) -> None:
    config, _fixtures, samples = study
    expected = (
        len(config.seeds)
        * len(config.fanouts)
        * len(config.workloads)
        * len(FanoutImplementation)
        * (config.warmup_repetitions + config.measurement_repetitions)
    )

    assert len(samples) == expected
    assert [item.randomized_order for item in samples] == list(range(expected))
    assert {item.fanout for item in samples} == {8, 16, 32}
    assert {item.workload_class for item in samples} == set(WorkloadClass)
    assert {item.seed for item in samples} == {41, 73, 113}
    assert {item.implementation for item in samples} == set(FanoutImplementation)
    assert sum(not item.warmup for item in samples) == expected * 3 // 4
    assert all(item.observed_max_queue_concurrency == 1 for item in samples)
    assert all(len(item.per_branch_readiness_ns) == item.fanout for item in samples)

    phase_size = len(samples) // 4
    for phase in range(4):
        order = [
            (item.workload_class, item.seed, item.fanout, item.implementation)
            for item in samples[phase * phase_size : (phase + 1) * phase_size]
        ]
        assert order != sorted(order, key=lambda item: (item[0], item[1], item[2], item[3]))


def test_shared_root_is_semantically_equal_and_reduces_ready_allocation(study) -> None:
    _config_value, _fixtures, samples = study
    measured = [item for item in samples if not item.warmup]
    by_key = {
        (item.workload_class, item.seed, item.fanout, item.repetition, item.implementation): item
        for item in measured
    }

    for workload in WorkloadClass:
        for seed in (41, 73, 113):
            for fanout in (8, 16, 32):
                for repetition in range(3):
                    prefix = (workload, seed, fanout, repetition)
                    naive = by_key[(*prefix, FanoutImplementation.NAIVE_PRIVATE)]
                    shared = by_key[(*prefix, FanoutImplementation.SHARED_ROOT_COW_LAZY)]
                    assert naive.semantic_state_hash == shared.semantic_state_hash
                    assert naive.logical_branch_state_bytes == shared.logical_branch_state_bytes
                    assert shared.branch_private_physical_bytes_at_ready == 0
                    assert shared.total_physical_state_bytes_at_ready < (
                        naive.total_physical_state_bytes_at_ready
                    )
                    assert shared.allocation_count_lower_bound < naive.allocation_count_lower_bound
                    assert shared.total_physical_state_bytes_after_suffix > (
                        shared.total_physical_state_bytes_at_ready
                    )


def test_writer_binds_aggregate_statistics_and_headroom_to_raw_samples(
    study, tmp_path: Path
) -> None:
    config, fixtures, samples = study
    report = write_fanout_study(config, fixtures, samples, tmp_path, repository=Path.cwd())
    raw = tmp_path / "raw-samples.jsonl"
    fixture_evidence = tmp_path / "fixture-evidence.jsonl"
    summary = tmp_path / "summary.json"
    manifest = json.loads((tmp_path / "manifest.json").read_text())

    assert report.raw_samples_sha256 == hashlib.sha256(raw.read_bytes()).hexdigest()
    assert (
        report.fixture_evidence_sha256 == hashlib.sha256(fixture_evidence.read_bytes()).hexdigest()
    )
    assert report.raw_sample_count == len(raw.read_text().splitlines())
    assert len(report.summaries) == 2 * 3 * 2
    assert len(report.baseline_effects) == 2 * 3
    assert len(report.headroom) == 2 * 3
    assert all(item.sample_count == 9 and item.seed_count == 3 for item in report.summaries)
    assert all(item.p50_all_ready_ns <= item.p99_all_ready_ns for item in report.summaries)
    assert all(
        item.median_confidence_interval.lower
        <= item.median_confidence_interval.observed
        <= item.median_confidence_interval.upper
        for item in report.summaries
    )
    assert all(item.median_logical_branch_state_bytes > 0 for item in report.summaries)
    assert all(item.matched_pair_count == 9 for item in report.baseline_effects)
    assert all(
        item.local_branch_lifecycle_median_ns
        == item.optimized_readiness_median_ns + item.measured_suffix_path_median_ns
        and item.ideal_zero_cost_readiness_speedup > 1.0
        and not item.hardware_gate_eligible
        for item in report.headroom
    )
    assert not report.real_transformer_measured and not report.cuda_gpu_measured
    assert manifest["hardware_gate_eligible"] is False
    assert FanoutStudyReport.model_validate_json(summary.read_bytes(), strict=True) == report
    with pytest.raises(FileExistsError, match="replace=True"):
        write_fanout_study(config, fixtures, samples, tmp_path, repository=Path.cwd())


def test_config_rejects_incomplete_or_unbounded_sweeps() -> None:
    assert tuple(item.context_tokens for item in FanoutStudyConfig().workloads) == (2048, 3968)
    with pytest.raises(ValidationError, match="three unique seeds"):
        FanoutStudyConfig(seeds=(41, 73))
    with pytest.raises(ValidationError, match="include 8, 16, and 32"):
        FanoutStudyConfig(fanouts=(8, 16))
    with pytest.raises(ValidationError, match="common suffix"):
        WorkloadSpec(
            workload_class=WorkloadClass.CODING_AGENT,
            context_tokens=32,
            suffix_tokens=2,
            common_suffix_tokens=3,
            divergence_pattern="late_rng_divergence",
        )
