"""Transparent, evidence-linked placement scoring for state operations.

This module intentionally has no built-in view of which device is best. The
caller supplies every criterion score, rationale, evidence reference, and
weight. The output exposes each weighted contribution so a recommendation can
be reproduced or challenged without reverse engineering a composite score.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from sloforge.helix.characterization.trace.models import (
    TimingMeasurementClass,
    WorkloadProvenance,
)

Identifier = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=256)]
ArtifactReference = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1, max_length=4096)
]
Explanation = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1, max_length=4096)
]


class PlacementModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, validate_default=True)


class PlacementLocation(StrEnum):
    HOST_CPU = "host_cpu"
    GPU = "gpu"
    FPGA_HBM = "fpga_hbm"
    FPGA_DDR = "fpga_ddr"
    SMARTNIC_DPU = "smartnic_dpu"
    CXL_ACCELERATOR = "cxl_accelerator"
    STORAGE_SIDE = "storage_side"


class PlacementCriterion(StrEnum):
    SOURCE_DATA_LOCALITY = "source_data_locality"
    DESTINATION_DATA_LOCALITY = "destination_data_locality"
    LATENCY_SENSITIVITY = "latency_sensitivity"
    BANDWIDTH_REQUIREMENT = "bandwidth_requirement"
    COMPUTE_REQUIREMENT = "compute_requirement"
    METADATA_REQUIREMENT = "metadata_requirement"
    BRANCH_FANOUT = "branch_fanout"
    PROGRAMMABILITY = "programmability"
    FAULT_DOMAIN = "fault_domain"
    SECURITY = "security"
    IMPLEMENTATION_COMPLEXITY = "implementation_complexity"
    EXPECTED_UTILIZATION = "expected_utilization"


class PlacementWeight(PlacementModel):
    criterion: PlacementCriterion
    weight: float = Field(ge=0.0, allow_inf_nan=False)
    rationale: Explanation


class PlacementWeights(PlacementModel):
    schema_version: Literal["sloforge.branchfabric.placement-weights/v1"]
    values: tuple[PlacementWeight, ...] = Field(min_length=12, max_length=12)

    @model_validator(mode="after")
    def covers_each_criterion(self) -> PlacementWeights:
        observed = [value.criterion for value in self.values]
        if len(observed) != len(set(observed)):
            raise ValueError("placement weights must not repeat a criterion")
        if set(observed) != set(PlacementCriterion):
            raise ValueError("placement weights must cover every criterion")
        if sum(value.weight for value in self.values) <= 0.0:
            raise ValueError("placement weights must contain at least one positive weight")
        return self


class CriterionSuitability(PlacementModel):
    criterion: PlacementCriterion
    suitability_score: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    rationale: Explanation
    evidence_references: tuple[ArtifactReference, ...] = Field(min_length=1, max_length=64)

    @model_validator(mode="after")
    def evidence_is_unique(self) -> CriterionSuitability:
        if len(self.evidence_references) != len(set(self.evidence_references)):
            raise ValueError("criterion evidence references must be unique")
        return self


class CandidatePlacementAssessment(PlacementModel):
    location: PlacementLocation
    criteria: tuple[CriterionSuitability, ...] = Field(min_length=12, max_length=12)

    @model_validator(mode="after")
    def covers_each_criterion(self) -> CandidatePlacementAssessment:
        observed = [score.criterion for score in self.criteria]
        if len(observed) != len(set(observed)):
            raise ValueError("candidate assessment must not repeat a criterion")
        if set(observed) != set(PlacementCriterion):
            raise ValueError("candidate assessment must cover every criterion")
        return self


class PlacementStudy(PlacementModel):
    schema_version: Literal["sloforge.branchfabric.placement-study/v1"]
    operation_id: Identifier
    operation: Identifier
    provenance: WorkloadProvenance
    timing_measurement_class: TimingMeasurementClass
    sample_count: int = Field(ge=1)
    experiment_ids: tuple[Identifier, ...] = Field(min_length=1, max_length=100_000)
    artifact_references: tuple[ArtifactReference, ...] = Field(min_length=1, max_length=100_000)
    candidates: tuple[CandidatePlacementAssessment, ...] = Field(min_length=7, max_length=7)

    @model_validator(mode="after")
    def evidence_and_candidates_are_complete(self) -> PlacementStudy:
        if len(self.experiment_ids) != len(set(self.experiment_ids)):
            raise ValueError("placement experiment identifiers must be unique")
        if len(self.artifact_references) != len(set(self.artifact_references)):
            raise ValueError("placement artifact references must be unique")
        locations = [candidate.location for candidate in self.candidates]
        if len(locations) != len(set(locations)):
            raise ValueError("placement study must not repeat a candidate location")
        if set(locations) != set(PlacementLocation):
            raise ValueError("placement study must assess every candidate location")
        return self


class CriterionContribution(PlacementModel):
    criterion: PlacementCriterion
    suitability_score: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    weight: float = Field(ge=0.0, allow_inf_nan=False)
    normalized_weight: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    weighted_contribution: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    rationale: Explanation
    evidence_references: tuple[ArtifactReference, ...] = Field(min_length=1, max_length=64)


class RankedPlacement(PlacementModel):
    rank: int = Field(ge=1, le=7)
    location: PlacementLocation
    normalized_score: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    contributions: tuple[CriterionContribution, ...] = Field(min_length=12, max_length=12)


class PlacementRecommendation(PlacementModel):
    schema_version: Literal["sloforge.branchfabric.placement-recommendation/v1"]
    operation_id: Identifier
    operation: Identifier
    provenance: WorkloadProvenance
    timing_measurement_class: TimingMeasurementClass
    sample_count: int = Field(ge=1)
    experiment_ids: tuple[Identifier, ...] = Field(min_length=1)
    artifact_references: tuple[ArtifactReference, ...] = Field(min_length=1)
    weights: PlacementWeights
    ranked: tuple[RankedPlacement, ...] = Field(min_length=7, max_length=7)

    @model_validator(mode="after")
    def ranks_are_dense(self) -> Self:
        if tuple(item.rank for item in self.ranked) != tuple(range(1, 8)):
            raise ValueError("placement ranks must be dense and ordered")
        return self


def score_placements(
    study: PlacementStudy,
    weights: PlacementWeights,
) -> PlacementRecommendation:
    """Rank all placement candidates using caller-supplied transparent inputs."""

    weight_by_criterion = {item.criterion: item.weight for item in weights.values}
    total_weight = sum(weight_by_criterion.values())
    scored: list[tuple[float, PlacementLocation, tuple[CriterionContribution, ...]]] = []
    for candidate in study.candidates:
        suitability_by_criterion = {item.criterion: item for item in candidate.criteria}
        contributions = tuple(
            CriterionContribution(
                criterion=criterion,
                suitability_score=suitability_by_criterion[criterion].suitability_score,
                weight=weight_by_criterion[criterion],
                normalized_weight=weight_by_criterion[criterion] / total_weight,
                weighted_contribution=(
                    suitability_by_criterion[criterion].suitability_score
                    * weight_by_criterion[criterion]
                    / total_weight
                ),
                rationale=suitability_by_criterion[criterion].rationale,
                evidence_references=suitability_by_criterion[criterion].evidence_references,
            )
            for criterion in PlacementCriterion
        )
        normalized_score = sum(item.weighted_contribution for item in contributions)
        scored.append((normalized_score, candidate.location, contributions))
    scored.sort(key=lambda item: (-item[0], item[1].value))
    ranked = tuple(
        RankedPlacement(
            rank=rank,
            location=location,
            normalized_score=score,
            contributions=contributions,
        )
        for rank, (score, location, contributions) in enumerate(scored, start=1)
    )
    return PlacementRecommendation(
        schema_version="sloforge.branchfabric.placement-recommendation/v1",
        operation_id=study.operation_id,
        operation=study.operation,
        provenance=study.provenance,
        timing_measurement_class=study.timing_measurement_class,
        sample_count=study.sample_count,
        experiment_ids=study.experiment_ids,
        artifact_references=study.artifact_references,
        weights=weights,
        ranked=ranked,
    )


__all__ = [
    "CandidatePlacementAssessment",
    "CriterionContribution",
    "CriterionSuitability",
    "PlacementCriterion",
    "PlacementLocation",
    "PlacementRecommendation",
    "PlacementStudy",
    "PlacementWeight",
    "PlacementWeights",
    "RankedPlacement",
    "score_placements",
]
