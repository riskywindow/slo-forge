"""Evidence-preserving Amdahl bounds for Helix state operations.

The module performs no workload modeling. Callers supply exclusive operation
timings and the corresponding end-to-end critical-path timing from raw
experiments. Results remain partitioned by workload provenance and timing
measurement class so synthetic and hardware-backed observations cannot be
silently combined.
"""

from __future__ import annotations

import math
from collections import defaultdict
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from sloforge.helix.characterization.trace.models import (
    TimingMeasurementClass,
    WorkloadProvenance,
)

MAX_AMDAHL_SAMPLES = 1_000_000

Identifier = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=256)]
ArtifactReference = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1, max_length=4096)
]


class AmdahlModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, validate_default=True)


class EndToEndObjective(StrEnum):
    BRANCH_READINESS = "branch_readiness"
    ROLLOUT_THROUGHPUT = "rollout_throughput"
    MIGRATION_LATENCY = "migration_latency"
    CAPACITY_RECLAMATION = "capacity_reclamation"
    FULL_HELIX_TRANSACTION = "full_helix_transaction"
    HELIX_LIFECYCLE_WINDOW = "helix_lifecycle_window"
    CONTINUUM_LIFECYCLE_WINDOW = "continuum_lifecycle_window"


class CandidatePrimitive(StrEnum):
    BRANCH_METADATA_FORK = "branch_metadata_fork"
    COW_HANDLING = "cow_handling"
    ALLOCATION = "allocation"
    DIRTY_TRACKING = "dirty_tracking"
    DELTA_EXTRACTION = "delta_extraction"
    CHECKSUM = "checksum"
    HASH = "hash"
    RESHARD = "reshard"
    REPACK = "repack"
    QUANTIZATION = "quantization"
    COMPRESSION = "compression"
    TRANSFER = "transfer"
    MULTICAST = "multicast"
    TRANSACTION_COMMIT = "transaction_commit"
    RECLAMATION = "reclamation"


class AmdahlTimingSample(AmdahlModel):
    """One exclusive primitive timing paired with its end-to-end sample."""

    sample_id: Identifier
    experiment_id: Identifier
    artifact_reference: ArtifactReference
    provenance: WorkloadProvenance
    timing_measurement_class: TimingMeasurementClass
    objective: EndToEndObjective
    primitive: CandidatePrimitive
    total_duration_ns: int = Field(gt=0)
    primitive_exclusive_duration_ns: int = Field(ge=0)
    operation_count: int = Field(ge=0)

    @model_validator(mode="after")
    def exclusive_time_fits_critical_path(self) -> AmdahlTimingSample:
        if self.primitive_exclusive_duration_ns > self.total_duration_ns:
            raise ValueError("exclusive primitive duration cannot exceed total duration")
        if self.operation_count == 0 and self.primitive_exclusive_duration_ns != 0:
            raise ValueError("a zero operation count requires zero primitive duration")
        return self


class AccelerationScenario(StrEnum):
    TWO_X = "2x"
    FIVE_X = "5x"
    TEN_X = "10x"
    FREE = "free"


class AmdahlBound(AmdahlModel):
    scenario: AccelerationScenario
    primitive_acceleration: float | None = Field(default=None, gt=1.0, allow_inf_nan=False)
    projected_duration_ns: float = Field(ge=0.0, allow_inf_nan=False)
    projected_speedup: float | None = Field(default=None, ge=1.0, allow_inf_nan=False)
    unbounded: bool

    @model_validator(mode="after")
    def scenario_fields_are_consistent(self) -> AmdahlBound:
        if self.scenario is AccelerationScenario.FREE:
            if self.primitive_acceleration is not None:
                raise ValueError("free acceleration must not carry a finite factor")
        elif self.primitive_acceleration is None:
            raise ValueError("finite acceleration scenarios require a factor")
        if self.unbounded != (self.projected_speedup is None):
            raise ValueError("only unbounded results omit projected_speedup")
        if self.unbounded and self.projected_duration_ns != 0.0:
            raise ValueError("an unbounded speedup requires zero projected duration")
        return self


class AmdahlResult(AmdahlModel):
    leverage_rank: int = Field(ge=1)
    objective: EndToEndObjective
    primitive: CandidatePrimitive
    provenance: WorkloadProvenance
    timing_measurement_class: TimingMeasurementClass
    sample_count: int = Field(ge=1)
    operation_count: int = Field(ge=0)
    total_duration_ns: int = Field(gt=0)
    primitive_exclusive_duration_ns: int = Field(ge=0)
    critical_path_fraction: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    sample_fraction_p50: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    sample_fraction_p95: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    sample_fraction_p99: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    experiment_ids: tuple[Identifier, ...] = Field(min_length=1)
    artifact_references: tuple[ArtifactReference, ...] = Field(min_length=1)
    bounds: tuple[AmdahlBound, ...] = Field(min_length=4, max_length=4)


class AmdahlReport(AmdahlModel):
    schema_version: Literal["sloforge.branchfabric.amdahl-report/v1"]
    sample_count: int = Field(ge=1)
    results: tuple[AmdahlResult, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def leverage_ranks_are_dense(self) -> AmdahlReport:
        groups: dict[
            tuple[EndToEndObjective, WorkloadProvenance, TimingMeasurementClass], list[int]
        ] = defaultdict(list)
        for result in self.results:
            groups[(result.objective, result.provenance, result.timing_measurement_class)].append(
                result.leverage_rank
            )
        if any(sorted(ranks) != list(range(1, len(ranks) + 1)) for ranks in groups.values()):
            raise ValueError("Amdahl leverage ranks must be dense within each evidence group")
        return self


def _percentile(values: tuple[float, ...], percentile: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _bound(total_ns: int, primitive_ns: int, scenario: AccelerationScenario) -> AmdahlBound:
    if scenario is AccelerationScenario.FREE:
        factor: float | None = None
        projected_ns = float(total_ns - primitive_ns)
    else:
        factor = {
            AccelerationScenario.TWO_X: 2.0,
            AccelerationScenario.FIVE_X: 5.0,
            AccelerationScenario.TEN_X: 10.0,
        }[scenario]
        projected_ns = total_ns - primitive_ns + primitive_ns / factor
    unbounded = projected_ns == 0.0
    speedup = None if unbounded else total_ns / projected_ns
    return AmdahlBound(
        scenario=scenario,
        primitive_acceleration=factor,
        projected_duration_ns=projected_ns,
        projected_speedup=speedup,
        unbounded=unbounded,
    )


def analyze_amdahl(samples: tuple[AmdahlTimingSample, ...]) -> AmdahlReport:
    """Calculate aggregate Amdahl bounds without mixing evidence classes.

    Durations are summed within a provenance-preserving group. This is
    equivalent to duration-weighting each raw sample and avoids treating a
    one-microsecond sample as equally influential as a one-second sample.
    """

    if not samples:
        raise ValueError("at least one Amdahl timing sample is required")
    if len(samples) > MAX_AMDAHL_SAMPLES:
        raise ValueError(f"Amdahl analysis is bounded to {MAX_AMDAHL_SAMPLES} samples")
    sample_ids = [sample.sample_id for sample in samples]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("Amdahl sample identifiers must be unique")

    grouped: dict[
        tuple[
            EndToEndObjective,
            CandidatePrimitive,
            WorkloadProvenance,
            TimingMeasurementClass,
        ],
        list[AmdahlTimingSample],
    ] = defaultdict(list)
    for sample in samples:
        grouped[
            (
                sample.objective,
                sample.primitive,
                sample.provenance,
                sample.timing_measurement_class,
            )
        ].append(sample)

    results: list[AmdahlResult] = []
    for key in sorted(grouped, key=lambda item: tuple(value.value for value in item)):
        objective, primitive, provenance, timing_class = key
        group = grouped[key]
        total_ns = sum(sample.total_duration_ns for sample in group)
        primitive_ns = sum(sample.primitive_exclusive_duration_ns for sample in group)
        fractions = tuple(
            sample.primitive_exclusive_duration_ns / sample.total_duration_ns for sample in group
        )
        results.append(
            AmdahlResult(
                leverage_rank=1,
                objective=objective,
                primitive=primitive,
                provenance=provenance,
                timing_measurement_class=timing_class,
                sample_count=len(group),
                operation_count=sum(sample.operation_count for sample in group),
                total_duration_ns=total_ns,
                primitive_exclusive_duration_ns=primitive_ns,
                critical_path_fraction=primitive_ns / total_ns,
                sample_fraction_p50=_percentile(fractions, 0.50),
                sample_fraction_p95=_percentile(fractions, 0.95),
                sample_fraction_p99=_percentile(fractions, 0.99),
                experiment_ids=tuple(sorted({sample.experiment_id for sample in group})),
                artifact_references=tuple(sorted({sample.artifact_reference for sample in group})),
                bounds=tuple(
                    _bound(total_ns, primitive_ns, scenario) for scenario in AccelerationScenario
                ),
            )
        )
    rank_groups: dict[
        tuple[EndToEndObjective, WorkloadProvenance, TimingMeasurementClass],
        list[AmdahlResult],
    ] = defaultdict(list)
    for result in results:
        rank_groups[(result.objective, result.provenance, result.timing_measurement_class)].append(
            result
        )
    ranked_results: list[AmdahlResult] = []
    for group_key in sorted(rank_groups, key=lambda item: tuple(value.value for value in item)):
        result_group = sorted(
            rank_groups[group_key],
            key=lambda item: (-item.critical_path_fraction, item.primitive.value),
        )
        ranked_results.extend(
            result.model_copy(update={"leverage_rank": rank})
            for rank, result in enumerate(result_group, start=1)
        )
    return AmdahlReport(
        schema_version="sloforge.branchfabric.amdahl-report/v1",
        sample_count=len(samples),
        results=tuple(ranked_results),
    )


__all__ = [
    "AccelerationScenario",
    "AmdahlBound",
    "AmdahlReport",
    "AmdahlResult",
    "AmdahlTimingSample",
    "CandidatePrimitive",
    "EndToEndObjective",
    "analyze_amdahl",
]
