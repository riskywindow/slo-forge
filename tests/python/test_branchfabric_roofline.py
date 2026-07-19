import pytest

from sloforge.helix.characterization.analysis.roofline import (
    CeilingSourceClass,
    RooflineCeilings,
    RooflineClassification,
    StateOperationRooflineSample,
    analyze_roofline,
)
from sloforge.helix.characterization.trace.models import (
    StateOperationType,
    TimingMeasurementClass,
    WorkloadProvenance,
)


def _ceilings() -> RooflineCeilings:
    return RooflineCeilings(
        ceiling_id="host-a",
        source_class=CeilingSourceClass.MEASURED,
        artifact_reference="artifacts/raw/ceilings.json",
        latency_floor_ns=100,
        hbm_bandwidth_bytes_per_second=1_000_000_000.0,
        host_memory_bandwidth_bytes_per_second=1_000_000_000.0,
        pcie_bandwidth_bytes_per_second=1_000_000_000.0,
        network_bandwidth_bytes_per_second=1_000.0,
        metadata_operations_per_second=1_000_000.0,
        transform_operations_per_second=1_000_000.0,
    )


def _sample(
    sample_id: str,
    *,
    host_memory_bytes: int | None = 0,
    network_bytes: int = 1_000,
    ceiling_id: str = "host-a",
) -> StateOperationRooflineSample:
    return StateOperationRooflineSample(
        sample_id=sample_id,
        experiment_id="experiment-network",
        artifact_reference=f"artifacts/raw/{sample_id}.jsonl",
        provenance=WorkloadProvenance.SYNTHETIC,
        timing_measurement_class=TimingMeasurementClass.HARDWARE_BACKED_REAL,
        ceiling_id=ceiling_id,
        operation_type=StateOperationType.STATE_SEND,
        operation_count=1,
        duration_ns=1_000_000_000,
        concurrency=1,
        hbm_bytes=0,
        host_memory_bytes=host_memory_bytes,
        pcie_bytes=0,
        network_bytes=network_bytes,
        metadata_operations=0,
        transform_operations=0,
        synchronization_wait_ns=0,
    )


def test_roofline_classifies_only_with_complete_counters_and_ceilings() -> None:
    result = analyze_roofline((_sample("network"),), (_ceilings(),)).results[0]

    assert result.counter_complete
    assert result.classification is RooflineClassification.NETWORK_BOUND
    assert result.dominant_lower_bound_ns == pytest.approx(1_000_000_000)
    assert result.dominant_observed_fraction == pytest.approx(1.0)
    assert result.ceiling_source_class is CeilingSourceClass.MEASURED
    assert not result.ceiling_exceeds_observed_duration
    assert result.provenance is WorkloadProvenance.SYNTHETIC
    assert result.timing_measurement_class is TimingMeasurementClass.HARDWARE_BACKED_REAL


def test_roofline_reports_unknown_for_missing_counter_or_ceiling() -> None:
    missing_counter = analyze_roofline(
        (_sample("missing-counter", host_memory_bytes=None),), (_ceilings(),)
    ).results[0]
    assert missing_counter.classification is RooflineClassification.UNKNOWN
    assert missing_counter.missing_inputs == ("host_memory_bytes",)
    assert missing_counter.lower_bounds == ()

    missing_ceiling = analyze_roofline(
        (_sample("missing-ceiling", ceiling_id="absent"),), (_ceilings(),)
    ).results[0]
    assert missing_ceiling.classification is RooflineClassification.UNKNOWN
    assert missing_ceiling.missing_inputs == ("ceiling:absent",)

    incomplete_ceiling = _ceilings().model_copy(update={"network_bandwidth_bytes_per_second": None})
    missing_rate = analyze_roofline((_sample("missing-rate"),), (incomplete_ceiling,)).results[0]
    assert missing_rate.classification is RooflineClassification.UNKNOWN
    assert missing_rate.missing_inputs == ("network_bandwidth_bytes_per_second",)


def test_roofline_rejects_duplicate_sample_and_ceiling_identifiers() -> None:
    sample = _sample("same")
    ceiling = _ceilings()
    with pytest.raises(ValueError, match="sample identifiers"):
        analyze_roofline((sample, sample), (ceiling,))
    with pytest.raises(ValueError, match="ceiling identifiers"):
        analyze_roofline((sample,), (ceiling, ceiling))
