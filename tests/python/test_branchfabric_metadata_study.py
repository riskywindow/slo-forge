from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from sloforge.helix.characterization.matrix import EvidenceClass
from sloforge.helix.characterization.metadata_study import (
    MAX_THREADS,
    MetadataImplementation,
    MetadataOperation,
    MetadataStudyConfig,
    build_metadata_run_order,
    measured_operations,
    measurement_samples,
    run_metadata_study,
    write_metadata_study,
)


def _small_config(*, operations: tuple[MetadataOperation, ...]) -> MetadataStudyConfig:
    return MetadataStudyConfig(
        seed=41,
        operations=operations,
        operations_per_thread=2,
        warmup_repetitions=1,
        measurement_repetitions=2,
        thread_counts=(1, 2),
        shard_count=4,
        working_set_entries=4,
        state_payload_bytes=32,
        include_software_baselines=True,
        sample_timeout_seconds=10.0,
    )


def test_run_order_is_seeded_randomized_and_warmup_first() -> None:
    config = _small_config(
        operations=(MetadataOperation.PAGE_LOOKUP, MetadataOperation.DIRTY_UPDATE)
    )
    first = build_metadata_run_order(config)
    second = build_metadata_run_order(config)

    assert first == second
    assert [item.randomized_sequence for item in first] == list(range(len(first)))
    first_measurement = next(index for index, item in enumerate(first) if not item.warmup)
    assert all(item.warmup for item in first[:first_measurement])
    assert all(not item.warmup for item in first[first_measurement:])
    assert len({item.trial_seed for item in first}) == len(first)


def test_every_required_hot_path_has_a_raw_current_software_sample() -> None:
    config = MetadataStudyConfig(
        seed=73,
        operations_per_thread=1,
        warmup_repetitions=0,
        measurement_repetitions=1,
        thread_counts=(1,),
        working_set_entries=2,
        state_payload_bytes=16,
        include_software_baselines=False,
        sample_timeout_seconds=10.0,
    )
    report = run_metadata_study(config)

    assert measured_operations(report) == frozenset(MetadataOperation)
    assert len(report.raw_samples) == len(MetadataOperation)
    assert all(
        sample.descriptor.implementation is MetadataImplementation.CURRENT_SOFTWARE
        for sample in report.raw_samples
    )
    assert all(sample.duration_ns > 0 for sample in report.raw_samples)
    assert all(sample.operation_count == 1 for sample in report.raw_samples)
    assert all(len(sample.result_sha256) == 64 for sample in report.raw_samples)
    assert (
        next(
            sample
            for sample in report.raw_samples
            if sample.descriptor.operation is MetadataOperation.REFCOUNT_UPDATE
        ).semantic_updates_per_operation
        == 2
    )


def test_evidence_axes_raw_samples_and_software_baselines_remain_separate(
    tmp_path: Path,
) -> None:
    operations = (
        MetadataOperation.BRANCH_CREATE_BOOKKEEPING,
        MetadataOperation.PAGE_LOOKUP,
        MetadataOperation.REFCOUNT_UPDATE,
        MetadataOperation.DIRTY_UPDATE,
        MetadataOperation.STATE_HASH_LOOKUP,
        MetadataOperation.LINEAGE_BOOKKEEPING,
    )
    report = run_metadata_study(_small_config(operations=operations))

    assert report.workload_evidence_class is EvidenceClass.SYNTHETIC
    assert report.timing_evidence_class is EvidenceClass.HARDWARE_BACKED_REAL
    assert report.counters.cpu_cycles == "unavailable"
    assert report.counters.cache_misses == "unavailable"
    assert report.counters.lock_wait_time == "unavailable"
    assert len(report.run_order) == len(report.raw_samples)
    assert any(sample.descriptor.warmup for sample in report.raw_samples)
    assert len(measurement_samples(report)) > 0
    assert all(not sample.descriptor.warmup for sample in measurement_samples(report))
    assert report.software_baseline_comparisons
    assert all(
        comparison.matched_semantic_scope == "faithful isolated metadata operation"
        and not comparison.hardware_claim_permitted
        for comparison in report.software_baseline_comparisons
    )
    actual = measurement_samples(
        report,
        operation=MetadataOperation.BRANCH_CREATE_BOOKKEEPING,
        implementation=MetadataImplementation.CURRENT_SOFTWARE,
    )
    isolated = measurement_samples(
        report,
        operation=MetadataOperation.BRANCH_CREATE_BOOKKEEPING,
        implementation=MetadataImplementation.ISOLATED_GLOBAL_LOCK,
    )
    assert actual and isolated
    assert actual[0].timing_scope != isolated[0].timing_scope

    output = tmp_path / "metadata-study.json"
    artifact_hash = write_metadata_study(report, output)
    payload = output.read_bytes()
    decoded = json.loads(payload)
    assert len(artifact_hash) == 64
    assert decoded["schema_version"] == "sloforge.branchfabric.metadata-study/v1"
    assert len(decoded["raw_samples"]) == len(report.raw_samples)


def test_concurrent_semantic_result_hash_is_deterministic_despite_scheduling() -> None:
    config = MetadataStudyConfig(
        seed=113,
        operations=(MetadataOperation.RECLAIM,),
        operations_per_thread=8,
        warmup_repetitions=0,
        measurement_repetitions=2,
        thread_counts=(1, 4),
        working_set_entries=16,
        state_payload_bytes=16,
        include_software_baselines=False,
    )

    first = run_metadata_study(config)
    second = run_metadata_study(config)
    assert [item.descriptor for item in first.raw_samples] == [
        item.descriptor for item in second.raw_samples
    ]
    assert [item.result_sha256 for item in first.raw_samples] == [
        item.result_sha256 for item in second.raw_samples
    ]


def test_bounds_reject_duplicate_unordered_or_excessive_configuration() -> None:
    with pytest.raises(ValidationError, match="operations must be non-empty and unique"):
        MetadataStudyConfig(
            seed=1,
            operations=(MetadataOperation.PAGE_LOOKUP, MetadataOperation.PAGE_LOOKUP),
        )
    with pytest.raises(ValidationError, match="thread_counts must be strictly increasing"):
        MetadataStudyConfig(seed=1, thread_counts=(1, 4, 2))
    with pytest.raises(ValidationError, match="not exceed"):
        MetadataStudyConfig(seed=1, thread_counts=(1, MAX_THREADS + 1))
    with pytest.raises(ValidationError, match="per-sample bound"):
        MetadataStudyConfig(
            seed=1,
            operations=(MetadataOperation.PAGE_LOOKUP,),
            operations_per_thread=257,
            thread_counts=(1, MAX_THREADS),
            include_software_baselines=False,
        )
