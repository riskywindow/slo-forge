import pytest
from pydantic import ValidationError

from sloforge.helix.characterization.analysis.amdahl import (
    AccelerationScenario,
    AmdahlTimingSample,
    CandidatePrimitive,
    EndToEndObjective,
    analyze_amdahl,
)
from sloforge.helix.characterization.trace.models import (
    TimingMeasurementClass,
    WorkloadProvenance,
)


def _sample(
    sample_id: str,
    *,
    total_ns: int,
    primitive_ns: int,
    objective: EndToEndObjective = EndToEndObjective.BRANCH_READINESS,
    provenance: WorkloadProvenance = WorkloadProvenance.SYNTHETIC,
    primitive: CandidatePrimitive = CandidatePrimitive.COW_HANDLING,
) -> AmdahlTimingSample:
    return AmdahlTimingSample(
        sample_id=sample_id,
        experiment_id=f"experiment-{sample_id}",
        artifact_reference=f"artifacts/raw/{sample_id}.json",
        provenance=provenance,
        timing_measurement_class=TimingMeasurementClass.HARDWARE_BACKED_REAL,
        objective=objective,
        primitive=primitive,
        total_duration_ns=total_ns,
        primitive_exclusive_duration_ns=primitive_ns,
        operation_count=1 if primitive_ns else 0,
    )


def test_amdahl_bounds_use_duration_weighted_exclusive_samples() -> None:
    report = analyze_amdahl(
        (
            _sample("long", total_ns=800, primitive_ns=200),
            _sample("short", total_ns=200, primitive_ns=0),
        )
    )

    result = report.results[0]
    assert report.sample_count == 2
    assert result.sample_count == 2
    assert result.total_duration_ns == 1_000
    assert result.primitive_exclusive_duration_ns == 200
    assert result.critical_path_fraction == pytest.approx(0.2)
    assert result.sample_fraction_p50 == pytest.approx(0.125)
    assert result.sample_fraction_p95 == pytest.approx(0.2375)
    bounds = {bound.scenario: bound for bound in result.bounds}
    assert bounds[AccelerationScenario.TWO_X].projected_speedup == pytest.approx(1 / 0.9)
    assert bounds[AccelerationScenario.FIVE_X].projected_speedup == pytest.approx(1 / 0.84)
    assert bounds[AccelerationScenario.TEN_X].projected_speedup == pytest.approx(1 / 0.82)
    assert bounds[AccelerationScenario.FREE].projected_speedup == pytest.approx(1.25)
    assert result.artifact_references == (
        "artifacts/raw/long.json",
        "artifacts/raw/short.json",
    )


def test_amdahl_preserves_evidence_classes_and_supports_every_objective() -> None:
    samples = (
        *(
            _sample(
                f"objective-{objective.value}",
                total_ns=100,
                primitive_ns=10,
                objective=objective,
            )
            for objective in EndToEndObjective
        ),
        _sample(
            "real-workload",
            total_ns=100,
            primitive_ns=10,
            provenance=WorkloadProvenance.HARDWARE_BACKED_REAL,
        ),
    )

    report = analyze_amdahl(samples)

    assert len(report.results) == len(EndToEndObjective) + 1
    assert {result.objective for result in report.results} == set(EndToEndObjective)
    branch_results = [
        result
        for result in report.results
        if result.objective is EndToEndObjective.BRANCH_READINESS
    ]
    assert {result.provenance for result in branch_results} == {
        WorkloadProvenance.SYNTHETIC,
        WorkloadProvenance.HARDWARE_BACKED_REAL,
    }


def test_amdahl_rejects_invalid_or_duplicate_samples_and_marks_unbounded() -> None:
    with pytest.raises(ValidationError, match="cannot exceed"):
        _sample("bad", total_ns=100, primitive_ns=101)
    sample = _sample("same", total_ns=100, primitive_ns=100)
    with pytest.raises(ValueError, match="unique"):
        analyze_amdahl((sample, sample))

    free = {bound.scenario: bound for bound in analyze_amdahl((sample,)).results[0].bounds}[
        AccelerationScenario.FREE
    ]
    assert free.unbounded
    assert free.projected_speedup is None
    assert free.projected_duration_ns == 0


def test_amdahl_ranks_measured_candidates_by_end_to_end_leverage() -> None:
    report = analyze_amdahl(
        (
            _sample(
                "cow",
                total_ns=100,
                primitive_ns=20,
                primitive=CandidatePrimitive.COW_HANDLING,
            ),
            _sample(
                "allocation",
                total_ns=100,
                primitive_ns=40,
                primitive=CandidatePrimitive.ALLOCATION,
            ),
        )
    )

    assert [(result.primitive, result.leverage_rank) for result in report.results] == [
        (CandidatePrimitive.ALLOCATION, 1),
        (CandidatePrimitive.COW_HANDLING, 2),
    ]
