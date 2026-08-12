import pytest
from pydantic import ValidationError

from sloforge.helix.characterization.analysis.placement import (
    CandidatePlacementAssessment,
    CriterionSuitability,
    PlacementCriterion,
    PlacementLocation,
    PlacementStudy,
    PlacementWeight,
    PlacementWeights,
    score_placements,
)
from sloforge.helix.characterization.trace.models import (
    TimingMeasurementClass,
    WorkloadProvenance,
)


def _weights() -> PlacementWeights:
    return PlacementWeights(
        schema_version="sloforge.branchfabric.placement-weights/v1",
        values=tuple(
            PlacementWeight(
                criterion=criterion,
                weight=1.0,
                rationale=f"Equal exploratory weight for {criterion.value}",
            )
            for criterion in PlacementCriterion
        ),
    )


def _candidate(location: PlacementLocation, score: float) -> CandidatePlacementAssessment:
    return CandidatePlacementAssessment(
        location=location,
        criteria=tuple(
            CriterionSuitability(
                criterion=criterion,
                suitability_score=score,
                rationale=f"Assessment of {criterion.value} for {location.value}",
                evidence_references=(
                    f"artifacts/placement/{location.value}/{criterion.value}.json",
                ),
            )
            for criterion in PlacementCriterion
        ),
    )


def _study() -> PlacementStudy:
    scores = {
        PlacementLocation.HOST_CPU: 0.9,
        PlacementLocation.GPU: 0.6,
        PlacementLocation.FPGA_HBM: 0.7,
        PlacementLocation.FPGA_DDR: 0.5,
        PlacementLocation.SMARTNIC_DPU: 0.8,
        PlacementLocation.CXL_ACCELERATOR: 0.4,
        PlacementLocation.STORAGE_SIDE: 0.2,
    }
    return PlacementStudy(
        schema_version="sloforge.branchfabric.placement-study/v1",
        operation_id="state-fork",
        operation="STATE_FORK",
        provenance=WorkloadProvenance.SYNTHETIC,
        timing_measurement_class=TimingMeasurementClass.HARDWARE_BACKED_REAL,
        sample_count=30,
        experiment_ids=("fork-fanout-sweep",),
        artifact_references=("artifacts/raw/fork-fanout-sweep.jsonl",),
        candidates=tuple(_candidate(location, scores[location]) for location in reversed(scores)),
    )


def test_placement_scoring_is_transparent_evidence_linked_and_deterministic() -> None:
    recommendation = score_placements(_study(), _weights())

    assert recommendation.ranked[0].location is PlacementLocation.HOST_CPU
    assert recommendation.ranked[0].normalized_score == pytest.approx(0.9)
    assert recommendation.ranked[1].location is PlacementLocation.SMARTNIC_DPU
    assert [item.rank for item in recommendation.ranked] == list(range(1, 8))
    assert sum(
        item.weighted_contribution for item in recommendation.ranked[0].contributions
    ) == pytest.approx(recommendation.ranked[0].normalized_score)
    assert all(
        contribution.evidence_references
        for candidate in recommendation.ranked
        for contribution in candidate.contributions
    )
    assert recommendation.provenance is WorkloadProvenance.SYNTHETIC
    assert recommendation.artifact_references == ("artifacts/raw/fork-fanout-sweep.jsonl",)


def test_placement_requires_all_candidates_criteria_and_explicit_positive_weights() -> None:
    with pytest.raises(ValidationError):
        CandidatePlacementAssessment(
            location=PlacementLocation.HOST_CPU,
            criteria=_candidate(PlacementLocation.HOST_CPU, 0.5).criteria[:-1],
        )
    weights = _weights()
    with pytest.raises(ValidationError, match="positive"):
        PlacementWeights(
            schema_version="sloforge.branchfabric.placement-weights/v1",
            values=tuple(value.model_copy(update={"weight": 0.0}) for value in weights.values),
        )
    with pytest.raises(ValidationError):
        PlacementStudy(
            **_study().model_dump(exclude={"candidates"}),
            candidates=_study().candidates[:-1],
        )
