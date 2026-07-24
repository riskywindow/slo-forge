from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from pydantic import ValidationError

from sloforge.helix.characterization.software_baselines import (
    FANOUTS,
    BaselineFamily,
    SoftwareBaselineConfig,
    SoftwareBaselineReport,
    run_software_baseline_study,
    write_software_baseline_report,
)


def _config(*, seed: int = 41, warmups: int = 1, repetitions: int = 2) -> SoftwareBaselineConfig:
    return SoftwareBaselineConfig(
        seed=seed,
        payload_bytes=8 * 1024,
        transform_output_tokens=4,
        chunk_sizes=(4096, 16 * 1024),
        transfer_concurrency=(1, 2),
        warmup_repetitions=warmups,
        measurement_repetitions=repetitions,
        sample_timeout_seconds=10.0,
    )


def test_study_preserves_randomized_raw_samples_and_real_cpu_provenance() -> None:
    config = _config()
    report = run_software_baseline_study(config)
    case_count = (
        2
        + 1
        + len(config.chunk_sizes)
        + (len(config.chunk_sizes) * len(config.transfer_concurrency))
        + 2 * len(FANOUTS)
    )

    assert len(report.raw_samples) == case_count * 3
    assert [sample.randomized_order for sample in report.raw_samples] == list(
        range(len(report.raw_samples))
    )
    assert all(sample.duration_ns > 0 and sample.cpu_time_ns > 0 for sample in report.raw_samples)
    assert all(sample.workload_evidence_class == "SYNTHETIC" for sample in report.raw_samples)
    assert all(
        sample.timing_measurement_class == "HARDWARE_BACKED_REAL" for sample in report.raw_samples
    )
    assert report.no_gpu_measurements
    assert report.no_network_hardware_measurements
    assert not report.outliers_removed
    for summary in report.summaries:
        assert summary.warmup_count == 1
        assert summary.sample_count == 2
        assert summary.samples_removed == 0

    phase_orders = []
    for warmup, repetition in ((True, 0), (False, 0), (False, 1)):
        phase_orders.append(
            tuple(
                sample.case_id
                for sample in report.raw_samples
                if sample.warmup is warmup and sample.repetition == repetition
            )
        )
    assert all(order != tuple(sorted(order)) for order in phase_orders)
    assert len(set(phase_orders)) == len(phase_orders)


def test_transform_and_hash_comparators_require_exact_equivalence() -> None:
    report = run_software_baseline_study(_config(warmups=0, repetitions=1))

    assert len(report.equivalence_proofs) == 2
    assert all(proof.exact_byte_or_semantic_equality for proof in report.equivalence_proofs)
    assert all(not proof.timing_sample for proof in report.equivalence_proofs)
    transform = [
        sample for sample in report.raw_samples if sample.family is BaselineFamily.TRANSFORM
    ]
    assert {sample.implementation for sample in transform} == {
        "continuum_direct_convert_capture",
        "trusted_canonical_staged",
    }
    assert {sample.result_sha256 for sample in transform} == {report.transform_expected_sha256}
    fused = next(
        sample
        for sample in transform
        if sample.implementation == "continuum_direct_convert_capture"
    )
    staged = next(
        sample for sample in transform if sample.implementation == "trusted_canonical_staged"
    )
    assert fused.component_temporary_bound_bytes == 512
    assert staged.component_temporary_bound_bytes is None
    assert fused.actual_allocator_peak_bytes is None
    assert staged.actual_allocator_peak_bytes is None
    hashes = [sample for sample in report.raw_samples if sample.family is BaselineFamily.HASH]
    assert all(sample.result_sha256 == report.payload_sha256 for sample in hashes)


def test_transfer_and_tree_sweeps_move_identical_host_bytes_without_hardware_claims() -> None:
    report = run_software_baseline_study(_config(warmups=0, repetitions=1))
    transfers = [
        sample
        for sample in report.raw_samples
        if sample.family is BaselineFamily.IN_PROCESS_TRANSFER
    ]
    assert {(sample.chunk_size_bytes, sample.requested_concurrency) for sample in transfers} == {
        (chunk_size, concurrency)
        for chunk_size in report.config.chunk_sizes
        for concurrency in report.config.transfer_concurrency
    }
    assert all(sample.result_sha256 == report.payload_sha256 for sample in transfers)
    assert all(not sample.network_hardware_measured for sample in transfers)

    fanout = [
        sample for sample in report.raw_samples if sample.family is BaselineFamily.SOFTWARE_FANOUT
    ]
    assert {sample.fanout for sample in fanout} == set(FANOUTS)
    assert {sample.implementation for sample in fanout} == {"repeated_unicast", "binary_tree"}
    for branch_count in FANOUTS:
        repeated = next(
            sample
            for sample in fanout
            if sample.fanout == branch_count and sample.implementation == "repeated_unicast"
        )
        tree = next(
            sample
            for sample in fanout
            if sample.fanout == branch_count and sample.implementation == "binary_tree"
        )
        assert repeated.destination_logical_bytes == tree.destination_logical_bytes
        assert repeated.source_read_bytes == report.config.payload_bytes * branch_count
        assert tree.source_read_bytes == report.config.payload_bytes * min(2, branch_count)
        assert repeated.result_sha256 == tree.result_sha256 == report.payload_sha256
        assert not repeated.network_hardware_measured and not tree.network_hardware_measured


def test_seed_reproduces_fixture_hashes_and_randomized_case_order_not_timing() -> None:
    config = SoftwareBaselineConfig(
        seed=73,
        payload_bytes=4096,
        transform_output_tokens=2,
        chunk_sizes=(4096,),
        transfer_concurrency=(1,),
        warmup_repetitions=0,
        measurement_repetitions=1,
        sample_timeout_seconds=10.0,
    )
    first = run_software_baseline_study(config)
    second = run_software_baseline_study(config)

    assert first.payload_sha256 == second.payload_sha256
    assert first.transform_source_sha256 == second.transform_source_sha256
    assert first.transform_expected_sha256 == second.transform_expected_sha256
    assert [(sample.case_id, sample.warmup, sample.repetition) for sample in first.raw_samples] == [
        (sample.case_id, sample.warmup, sample.repetition) for sample in second.raw_samples
    ]


def test_report_writer_is_hashed_and_round_trips_strictly(tmp_path: Path) -> None:
    report = run_software_baseline_study(_config(warmups=0, repetitions=1))
    output = tmp_path / "software-baselines.json"
    artifact_sha256 = write_software_baseline_report(report, output)

    assert artifact_sha256 == hashlib.sha256(output.read_bytes()).hexdigest()
    restored = SoftwareBaselineReport.model_validate_json(output.read_bytes(), strict=True)
    assert restored.payload_sha256 == report.payload_sha256
    assert restored.raw_samples == report.raw_samples
    with pytest.raises(FileExistsError, match="replace=True"):
        write_software_baseline_report(report, output)
    assert write_software_baseline_report(report, output, replace=True) == artifact_sha256


def test_config_rejects_unbounded_or_incomplete_sweeps() -> None:
    with pytest.raises(ValidationError, match="required software fanout sweep"):
        SoftwareBaselineConfig(seed=1, fanouts=(2, 4))
    with pytest.raises(ValidationError, match="strictly increasing"):
        SoftwareBaselineConfig(seed=1, chunk_sizes=(16 * 1024, 4096))
    with pytest.raises(ValidationError, match="begin at 1"):
        SoftwareBaselineConfig(seed=1, transfer_concurrency=(2, 4))
