"""A counter-complete roofline model for state movement operations.

Every demand counter is optional because not every profiler exposes every
resource. ``None`` means unmeasured while zero means measured and absent. The
classifier returns ``unknown`` whenever a missing counter or required resource
ceiling could conceal the dominant bottleneck.
"""

from __future__ import annotations

import math
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from sloforge.helix.characterization.trace.models import (
    StateOperationType,
    TimingMeasurementClass,
    WorkloadProvenance,
)

MAX_ROOFLINE_SAMPLES = 1_000_000

Identifier = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=256)]
ArtifactReference = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1, max_length=4096)
]


class RooflineModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, validate_default=True)


class CeilingSourceClass(StrEnum):
    MEASURED = "measured"
    VENDOR_SPECIFICATION = "vendor_specification"
    SIMULATED = "simulated"
    USER_SUPPLIED = "user_supplied"


class RooflineClassification(StrEnum):
    LATENCY_BOUND = "latency_bound"
    METADATA_BOUND = "metadata_bound"
    HBM_BANDWIDTH_BOUND = "hbm_bandwidth_bound"
    HOST_MEMORY_BOUND = "host_memory_bound"
    PCIE_BOUND = "pcie_bound"
    NETWORK_BOUND = "network_bound"
    COMPUTE_BOUND = "compute_bound"
    SYNCHRONIZATION_BOUND = "synchronization_bound"
    UNKNOWN = "unknown"


class RooflineCeilings(RooflineModel):
    ceiling_id: Identifier
    source_class: CeilingSourceClass
    artifact_reference: ArtifactReference
    latency_floor_ns: int | None = Field(default=None, gt=0)
    hbm_bandwidth_bytes_per_second: float | None = Field(default=None, gt=0.0, allow_inf_nan=False)
    host_memory_bandwidth_bytes_per_second: float | None = Field(
        default=None, gt=0.0, allow_inf_nan=False
    )
    pcie_bandwidth_bytes_per_second: float | None = Field(default=None, gt=0.0, allow_inf_nan=False)
    network_bandwidth_bytes_per_second: float | None = Field(
        default=None, gt=0.0, allow_inf_nan=False
    )
    metadata_operations_per_second: float | None = Field(default=None, gt=0.0, allow_inf_nan=False)
    transform_operations_per_second: float | None = Field(default=None, gt=0.0, allow_inf_nan=False)


class StateOperationRooflineSample(RooflineModel):
    sample_id: Identifier
    experiment_id: Identifier
    artifact_reference: ArtifactReference
    provenance: WorkloadProvenance
    timing_measurement_class: TimingMeasurementClass
    ceiling_id: Identifier
    operation_type: StateOperationType
    operation_count: int = Field(ge=1)
    duration_ns: int = Field(gt=0)
    concurrency: int = Field(ge=1)
    hbm_bytes: int | None = Field(default=None, ge=0)
    host_memory_bytes: int | None = Field(default=None, ge=0)
    pcie_bytes: int | None = Field(default=None, ge=0)
    network_bytes: int | None = Field(default=None, ge=0)
    metadata_operations: int | None = Field(default=None, ge=0)
    transform_operations: int | None = Field(default=None, ge=0)
    synchronization_wait_ns: int | None = Field(default=None, ge=0)


class ResourceLowerBound(RooflineModel):
    resource: RooflineClassification
    demand_value: float = Field(ge=0.0, allow_inf_nan=False)
    ceiling_per_second: float | None = Field(default=None, gt=0.0, allow_inf_nan=False)
    lower_bound_ns: float = Field(ge=0.0, allow_inf_nan=False)
    observed_duration_fraction: float = Field(ge=0.0, allow_inf_nan=False)


class RooflineResult(RooflineModel):
    sample_id: Identifier
    experiment_id: Identifier
    artifact_reference: ArtifactReference
    provenance: WorkloadProvenance
    timing_measurement_class: TimingMeasurementClass
    operation_type: StateOperationType
    operation_count: int = Field(ge=1)
    duration_ns: int = Field(gt=0)
    concurrency: int = Field(ge=1)
    classification: RooflineClassification
    counter_complete: bool
    missing_inputs: tuple[str, ...]
    ceiling_id: Identifier
    ceiling_source_class: CeilingSourceClass | None
    ceiling_artifact_reference: ArtifactReference | None
    lower_bounds: tuple[ResourceLowerBound, ...]
    dominant_lower_bound_ns: float | None = Field(default=None, ge=0.0, allow_inf_nan=False)
    dominant_observed_fraction: float | None = Field(default=None, ge=0.0, allow_inf_nan=False)
    ceiling_exceeds_observed_duration: bool | None


class RooflineReport(RooflineModel):
    schema_version: Literal["sloforge.branchfabric.roofline-report/v1"]
    sample_count: int = Field(ge=1)
    results: tuple[RooflineResult, ...] = Field(min_length=1)


def _unknown_result(
    sample: StateOperationRooflineSample,
    *,
    missing_inputs: tuple[str, ...],
    ceiling: RooflineCeilings | None,
) -> RooflineResult:
    return RooflineResult(
        sample_id=sample.sample_id,
        experiment_id=sample.experiment_id,
        artifact_reference=sample.artifact_reference,
        provenance=sample.provenance,
        timing_measurement_class=sample.timing_measurement_class,
        operation_type=sample.operation_type,
        operation_count=sample.operation_count,
        duration_ns=sample.duration_ns,
        concurrency=sample.concurrency,
        classification=RooflineClassification.UNKNOWN,
        counter_complete=False,
        missing_inputs=missing_inputs,
        ceiling_id=sample.ceiling_id,
        ceiling_source_class=None if ceiling is None else ceiling.source_class,
        ceiling_artifact_reference=None if ceiling is None else ceiling.artifact_reference,
        lower_bounds=(),
        dominant_lower_bound_ns=None,
        dominant_observed_fraction=None,
        ceiling_exceeds_observed_duration=None,
    )


def _bandwidth_bound_ns(demand: int, ceiling_per_second: float) -> float:
    return demand * 1_000_000_000.0 / ceiling_per_second


def _classify_sample(
    sample: StateOperationRooflineSample, ceiling: RooflineCeilings | None
) -> RooflineResult:
    if ceiling is None:
        return _unknown_result(
            sample,
            missing_inputs=(f"ceiling:{sample.ceiling_id}",),
            ceiling=None,
        )

    counter_values = {
        "hbm_bytes": sample.hbm_bytes,
        "host_memory_bytes": sample.host_memory_bytes,
        "pcie_bytes": sample.pcie_bytes,
        "network_bytes": sample.network_bytes,
        "metadata_operations": sample.metadata_operations,
        "transform_operations": sample.transform_operations,
        "synchronization_wait_ns": sample.synchronization_wait_ns,
    }
    missing = [name for name, value in counter_values.items() if value is None]
    if ceiling.latency_floor_ns is None:
        missing.append("latency_floor_ns")

    demand_and_ceilings = (
        (
            "hbm_bandwidth_bytes_per_second",
            sample.hbm_bytes,
            ceiling.hbm_bandwidth_bytes_per_second,
        ),
        (
            "host_memory_bandwidth_bytes_per_second",
            sample.host_memory_bytes,
            ceiling.host_memory_bandwidth_bytes_per_second,
        ),
        (
            "pcie_bandwidth_bytes_per_second",
            sample.pcie_bytes,
            ceiling.pcie_bandwidth_bytes_per_second,
        ),
        (
            "network_bandwidth_bytes_per_second",
            sample.network_bytes,
            ceiling.network_bandwidth_bytes_per_second,
        ),
        (
            "metadata_operations_per_second",
            sample.metadata_operations,
            ceiling.metadata_operations_per_second,
        ),
        (
            "transform_operations_per_second",
            sample.transform_operations,
            ceiling.transform_operations_per_second,
        ),
    )
    for ceiling_name, demand, rate in demand_and_ceilings:
        if demand is not None and demand > 0 and rate is None:
            missing.append(ceiling_name)
    if missing:
        return _unknown_result(
            sample,
            missing_inputs=tuple(sorted(set(missing))),
            ceiling=ceiling,
        )

    assert sample.hbm_bytes is not None
    assert sample.host_memory_bytes is not None
    assert sample.pcie_bytes is not None
    assert sample.network_bytes is not None
    assert sample.metadata_operations is not None
    assert sample.transform_operations is not None
    assert sample.synchronization_wait_ns is not None
    assert ceiling.latency_floor_ns is not None

    raw_bounds: list[tuple[RooflineClassification, float, float | None, float]] = [
        (
            RooflineClassification.LATENCY_BOUND,
            float(sample.operation_count),
            None,
            float(
                ceiling.latency_floor_ns * math.ceil(sample.operation_count / sample.concurrency)
            ),
        ),
        (
            RooflineClassification.SYNCHRONIZATION_BOUND,
            float(sample.synchronization_wait_ns),
            None,
            float(sample.synchronization_wait_ns),
        ),
    ]
    resource_demands = (
        (
            RooflineClassification.HBM_BANDWIDTH_BOUND,
            sample.hbm_bytes,
            ceiling.hbm_bandwidth_bytes_per_second,
        ),
        (
            RooflineClassification.HOST_MEMORY_BOUND,
            sample.host_memory_bytes,
            ceiling.host_memory_bandwidth_bytes_per_second,
        ),
        (
            RooflineClassification.PCIE_BOUND,
            sample.pcie_bytes,
            ceiling.pcie_bandwidth_bytes_per_second,
        ),
        (
            RooflineClassification.NETWORK_BOUND,
            sample.network_bytes,
            ceiling.network_bandwidth_bytes_per_second,
        ),
        (
            RooflineClassification.METADATA_BOUND,
            sample.metadata_operations,
            ceiling.metadata_operations_per_second,
        ),
        (
            RooflineClassification.COMPUTE_BOUND,
            sample.transform_operations,
            ceiling.transform_operations_per_second,
        ),
    )
    for resource, demand, rate in resource_demands:
        if demand == 0:
            lower_bound = 0.0
        else:
            assert rate is not None
            lower_bound = _bandwidth_bound_ns(demand, rate)
        raw_bounds.append((resource, float(demand), rate, lower_bound))

    lower_bounds = tuple(
        ResourceLowerBound(
            resource=resource,
            demand_value=demand,
            ceiling_per_second=rate,
            lower_bound_ns=lower_bound,
            observed_duration_fraction=lower_bound / sample.duration_ns,
        )
        for resource, demand, rate, lower_bound in raw_bounds
    )
    dominant = min(
        lower_bounds,
        key=lambda item: (-item.lower_bound_ns, item.resource.value),
    )
    return RooflineResult(
        sample_id=sample.sample_id,
        experiment_id=sample.experiment_id,
        artifact_reference=sample.artifact_reference,
        provenance=sample.provenance,
        timing_measurement_class=sample.timing_measurement_class,
        operation_type=sample.operation_type,
        operation_count=sample.operation_count,
        duration_ns=sample.duration_ns,
        concurrency=sample.concurrency,
        classification=dominant.resource,
        counter_complete=True,
        missing_inputs=(),
        ceiling_id=ceiling.ceiling_id,
        ceiling_source_class=ceiling.source_class,
        ceiling_artifact_reference=ceiling.artifact_reference,
        lower_bounds=lower_bounds,
        dominant_lower_bound_ns=dominant.lower_bound_ns,
        dominant_observed_fraction=dominant.observed_duration_fraction,
        ceiling_exceeds_observed_duration=dominant.lower_bound_ns > sample.duration_ns,
    )


def analyze_roofline(
    samples: tuple[StateOperationRooflineSample, ...],
    ceilings: tuple[RooflineCeilings, ...],
) -> RooflineReport:
    """Classify state-operation samples against explicitly supplied ceilings."""

    if not samples:
        raise ValueError("at least one roofline sample is required")
    if len(samples) > MAX_ROOFLINE_SAMPLES:
        raise ValueError(f"roofline analysis is bounded to {MAX_ROOFLINE_SAMPLES} samples")
    sample_ids = [sample.sample_id for sample in samples]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("roofline sample identifiers must be unique")
    ceiling_ids = [ceiling.ceiling_id for ceiling in ceilings]
    if len(ceiling_ids) != len(set(ceiling_ids)):
        raise ValueError("roofline ceiling identifiers must be unique")
    ceiling_by_id = {ceiling.ceiling_id: ceiling for ceiling in ceilings}
    results = tuple(
        _classify_sample(sample, ceiling_by_id.get(sample.ceiling_id))
        for sample in sorted(samples, key=lambda item: item.sample_id)
    )
    return RooflineReport(
        schema_version="sloforge.branchfabric.roofline-report/v1",
        sample_count=len(samples),
        results=results,
    )


__all__ = [
    "CeilingSourceClass",
    "ResourceLowerBound",
    "RooflineCeilings",
    "RooflineClassification",
    "RooflineReport",
    "RooflineResult",
    "StateOperationRooflineSample",
    "analyze_roofline",
]
