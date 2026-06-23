"""Typed evidence products for Autopsy-guided synthesis."""

from __future__ import annotations

import math
from typing import Annotated, Literal, Self, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from sloforge.autopsy.models import BottleneckKind
from sloforge.genesis.ir import TransformationFamily

NonEmpty = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
Region: TypeAlias = Literal[
    "workflow", "request", "serving", "state", "distributed", "tensor", "kernel", "recovery"
]
ALL_REGIONS: tuple[Region, ...] = (
    "workflow",
    "request",
    "serving",
    "state",
    "distributed",
    "tensor",
    "kernel",
    "recovery",
)


class GuidanceModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, validate_default=True)


class UpsideEstimate(GuidanceModel):
    lower_ms: float
    expected_ms: float
    upper_ms: float
    source: NonEmpty

    @model_validator(mode="after")
    def ordered(self) -> Self:
        values = (self.lower_ms, self.expected_ms, self.upper_ms)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("upside interval must be finite")
        if not self.lower_ms <= self.expected_ms <= self.upper_ms:
            raise ValueError("upside interval must be ordered")
        return self


class MutationBudget(GuidanceModel):
    schema_version: Literal["1.0.0"] = "1.0.0"
    diagnosis_id: NonEmpty
    bottleneck: BottleneckKind
    confidence: Annotated[float, Field(ge=0.0, le=1.0)]
    mutable_regions: tuple[Region, ...]
    frozen_regions: tuple[Region, ...]
    allowed_families: tuple[TransformationFamily, ...]
    expected_upside: UpsideEstimate | None
    expected_verification_cost: Literal["low", "medium", "high"]
    next_bottleneck: BottleneckKind | None
    evidence_ids: tuple[NonEmpty, ...]

    @model_validator(mode="after")
    def regions_partition_genome(self) -> Self:
        if not self.mutable_regions:
            raise ValueError("mutation budget must permit at least one region")
        if set(self.mutable_regions) & set(self.frozen_regions):
            raise ValueError("mutable and frozen regions overlap")
        if set(self.mutable_regions) | set(self.frozen_regions) != set(ALL_REGIONS):
            raise ValueError("mutation budget must partition all genome regions")
        if not self.allowed_families:
            raise ValueError("mutation budget must declare allowed transformation families")
        return self


class SearchEfficiencySummary(GuidanceModel):
    label: NonEmpty
    candidates_evaluated: Annotated[int, Field(ge=0)]
    invalid_candidates: Annotated[int, Field(ge=0)]
    hardware_experiments: Annotated[int, Field(ge=0)]
    time_to_improvement_seconds: Annotated[float, Field(ge=0.0)] | None
    final_objective: float | None
    distinct_transformation_families: Annotated[int, Field(ge=0)]

    @model_validator(mode="after")
    def finite_metrics(self) -> Self:
        values = (self.time_to_improvement_seconds, self.final_objective)
        if any(value is not None and not math.isfinite(value) for value in values):
            raise ValueError("search efficiency metrics must be finite")
        if self.invalid_candidates > self.candidates_evaluated:
            raise ValueError("invalid candidates cannot exceed evaluated candidates")
        return self


class SearchEfficiencyComparison(GuidanceModel):
    guided: SearchEfficiencySummary
    unguided: SearchEfficiencySummary
    candidate_reduction: int
    invalid_candidate_reduction: int
    hardware_experiment_reduction: int
    seconds_to_improvement_reduction: float | None
    objective_delta: float | None
    diversity_delta: int


__all__ = [
    "ALL_REGIONS",
    "MutationBudget",
    "Region",
    "SearchEfficiencyComparison",
    "SearchEfficiencySummary",
    "UpsideEstimate",
]
