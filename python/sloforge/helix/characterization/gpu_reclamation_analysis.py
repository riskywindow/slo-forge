"""Evidence-gated Experiment 004 Amdahl and hardware-interest analysis.

This module is deliberately pure: it consumes validated, non-overlapping raw
measurements and never substitutes fixture or modelled values for hardware
observations.  Report/plot generation can therefore be exercised locally while
the final classification remains impossible until the required GPU trials exist.
"""

from __future__ import annotations

import math
from enum import StrEnum
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, validate_default=True)


class Experiment004Outcome(StrEnum):
    MOVEMENT_CLOSED = "MOVEMENT_CLOSED"
    GPU_SOFTWARE_TARGET = "GPU_SOFTWARE_TARGET"
    HOST_PIPELINE_HARDWARE_INTEREST = "HOST_PIPELINE_HARDWARE_INTEREST"
    FABRIC_HARDWARE_INTEREST = "FABRIC_HARDWARE_INTEREST"
    PRESERVATION_NOT_ECONOMIC = "PRESERVATION_NOT_ECONOMIC"


class CriticalPathKind(StrEnum):
    RECLAMATION = "reclamation"
    RESTORE = "restore"
    FULL_TRANSACTION = "full_transaction"
    SLO_RESTORATION = "slo_restoration"


class PlacementClass(StrEnum):
    GPU = "gpu"
    HOST = "host"
    FABRIC = "fabric"
    GENERAL_SOFTWARE = "general_software"


class MeasuredInterval(_StrictModel):
    """One exclusive critical-path interval from a causal hardware trial."""

    name: str = Field(min_length=1, max_length=256)
    start_ns: int = Field(ge=0)
    end_ns: int = Field(ge=0)
    gpu_time_ns: int | None = Field(default=None, ge=0)
    cpu_time_ns: int | None = Field(default=None, ge=0)
    logical_bytes: int = Field(default=0, ge=0)
    physical_bytes: int = Field(default=0, ge=0)
    temporary_bytes: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def valid_interval(self) -> Self:
        if self.end_ns < self.start_ns:
            raise ValueError("measured interval ends before it starts")
        if self.gpu_time_ns is not None and self.gpu_time_ns > self.duration_ns:
            raise ValueError("exclusive GPU time cannot exceed interval wall time")
        return self

    @property
    def duration_ns(self) -> int:
        return self.end_ns - self.start_ns


class CriticalPath(_StrictModel):
    """A strict, gap-free and overlap-free end-to-end decomposition."""

    kind: CriticalPathKind
    intervals: tuple[MeasuredInterval, ...]

    @model_validator(mode="after")
    def exact_decomposition(self) -> Self:
        if not self.intervals:
            raise ValueError("critical path cannot be empty")
        if len({item.name for item in self.intervals}) != len(self.intervals):
            raise ValueError("critical path contains duplicate stage names")
        if any(
            left.end_ns != right.start_ns
            for left, right in zip(self.intervals, self.intervals[1:], strict=False)
        ):
            raise ValueError("critical path contains a gap or overlap")
        if sum(item.duration_ns for item in self.intervals) != self.duration_ns:
            raise ValueError("critical-path intervals do not conserve elapsed time")
        return self

    @property
    def duration_ns(self) -> int:
        return self.intervals[-1].end_ns - self.intervals[0].start_ns


class AmdahlPoint(_StrictModel):
    acceleration: Literal["2x", "5x", "10x", "free"]
    projected_total_ns: int = Field(ge=0)
    projected_speedup: float | None = Field(default=None, ge=1.0, allow_inf_nan=False)
    projected_reduction_fraction: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)


class AmdahlProjection(_StrictModel):
    path_kind: CriticalPathKind
    target_names: tuple[str, ...]
    baseline_total_ns: int = Field(gt=0)
    target_total_ns: int = Field(ge=0)
    target_fraction: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    points: tuple[AmdahlPoint, ...]


def project_amdahl(path: CriticalPath, *, target_names: tuple[str, ...]) -> AmdahlProjection:
    """Accelerate an exact set of non-overlapping stages without double counting."""

    if not target_names or len(set(target_names)) != len(target_names):
        raise ValueError("Amdahl targets must be nonempty and unique")
    by_name = {item.name: item for item in path.intervals}
    missing = set(target_names) - set(by_name)
    if missing:
        raise ValueError(f"Amdahl targets are absent from the critical path: {sorted(missing)}")
    target_ns = sum(by_name[name].duration_ns for name in target_names)
    total_ns = path.duration_ns
    points: list[AmdahlPoint] = []
    accelerations: tuple[tuple[Literal["2x", "5x", "10x", "free"], float | None], ...] = (
        ("2x", 2.0),
        ("5x", 5.0),
        ("10x", 10.0),
        ("free", None),
    )
    for label, factor in accelerations:
        accelerated = 0 if factor is None else round(target_ns / factor)
        projected = total_ns - target_ns + accelerated
        speedup = total_ns / projected if projected else None
        points.append(
            AmdahlPoint(
                acceleration=label,
                projected_total_ns=projected,
                projected_speedup=speedup,
                projected_reduction_fraction=(total_ns - projected) / total_ns,
            )
        )
    return AmdahlProjection(
        path_kind=path.kind,
        target_names=target_names,
        baseline_total_ns=total_ns,
        target_total_ns=target_ns,
        target_fraction=target_ns / total_ns,
        points=tuple(points),
    )


class FusedChainEvidence(_StrictModel):
    """Measured chain evidence after the optimized software implementation."""

    chain_id: str = Field(min_length=1, max_length=256)
    operations: tuple[str, ...] = Field(min_length=2)
    occurrence_count: int = Field(gt=0)
    logical_bytes: int = Field(gt=0)
    physical_bytes: int = Field(gt=0)
    state_passes: int = Field(gt=0)
    wall_time_ns: int = Field(gt=0)
    gpu_time_ns: int | None = Field(default=None, ge=0)
    cpu_time_ns: int | None = Field(default=None, ge=0)
    temporary_bytes: int = Field(ge=0)
    dependencies_permit_streaming: bool
    materialized_intermediate_bytes: int = Field(ge=0)
    placement_class: PlacementClass
    measured_fabric_transfer_bytes: int = Field(default=0, ge=0)
    measured_fabric_endpoint_count: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def times_fit(self) -> Self:
        if self.gpu_time_ns is not None and self.gpu_time_ns > self.wall_time_ns:
            raise ValueError("chain GPU time exceeds measured wall time")
        if self.placement_class is PlacementClass.FABRIC and (
            self.measured_fabric_transfer_bytes <= 0 or self.measured_fabric_endpoint_count < 2
        ):
            raise ValueError("fabric placement requires measured fabric bytes and endpoints")
        if self.placement_class is not PlacementClass.FABRIC and (
            self.measured_fabric_transfer_bytes or self.measured_fabric_endpoint_count
        ):
            raise ValueError("non-fabric placement cannot claim fabric measurements")
        return self


class HardwareInterestEvidence(_StrictModel):
    """All mandatory system and realizability gates for one optimized chain."""

    chain: FusedChainEvidence
    fraction_of_reclamation: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    fraction_of_resume: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    fraction_of_full_transaction: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    fraction_of_slo_restoration: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    fraction_of_movement_time: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    serving_degradation_fraction: float = Field(ge=0.0, allow_inf_nan=False)
    avoidable_physical_byte_fraction: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    ideal_free_end_to_end_speedup: float = Field(ge=1.0, allow_inf_nan=False)
    realistic_end_to_end_speedup: float = Field(ge=1.0, allow_inf_nan=False)
    regular_dataflow: bool
    measured_byte_rate: bool
    measured_concurrency: bool
    measured_latency_target: bool
    plausible_off_critical_path_placement: bool

    @model_validator(mode="after")
    def physically_possible_speedup(self) -> Self:
        if self.realistic_end_to_end_speedup > self.ideal_free_end_to_end_speedup:
            raise ValueError("realistic acceleration cannot exceed ideal-free speedup")
        return self

    @property
    def system_gate(self) -> bool:
        return any(
            (
                self.fraction_of_reclamation >= 0.15,
                self.fraction_of_movement_time >= 0.20,
                self.serving_degradation_fraction >= 0.20,
                self.avoidable_physical_byte_fraction >= 0.25,
            )
        )

    @property
    def realizability_gate(self) -> bool:
        return all(
            (
                self.chain.dependencies_permit_streaming,
                self.regular_dataflow,
                self.measured_byte_rate,
                self.measured_concurrency,
                self.measured_latency_target,
                self.plausible_off_critical_path_placement,
            )
        )

    @property
    def hardware_interest(self) -> bool:
        return (
            self.chain.placement_class is not PlacementClass.GENERAL_SOFTWARE
            and self.system_gate
            and self.realizability_gate
            and self.ideal_free_end_to_end_speedup >= 1.15
            and self.realistic_end_to_end_speedup >= 1.15
        )


class OutcomeEvidence(_StrictModel):
    """Complete evidence required to select exactly one Experiment 004 outcome."""

    valid_pilot: Literal[True]
    kill_trials: int = Field(ge=1)
    naive_trials: int = Field(ge=1)
    optimized_trials: int = Field(ge=1)
    optimized_semantics_valid: Literal[True]
    preservation_economic_for_measured_workload: bool
    optimized_removed_most_naive_headroom: bool
    profiling_hardware_backed: Literal[True]
    trace_overhead_gate_passed: Literal[True]
    optimized_path_measured_after_naive: Literal[True]
    optimized_path_semantics_match_naive: Literal[True]
    optimized_movement_fraction: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    chain_gates: tuple[HardwareInterestEvidence, ...]

    @model_validator(mode="after")
    def optimized_path_is_a_real_comparator(self) -> Self:
        if min(self.kill_trials, self.naive_trials, self.optimized_trials) <= 0:
            raise ValueError("every reclamation mode requires a measured trial")
        return self


class OutcomeDecision(_StrictModel):
    outcome: Experiment004Outcome
    rationale: str = Field(min_length=1, max_length=4096)
    strong_hardware_result: bool
    hardware_interest_chain_ids: tuple[str, ...]


def select_outcome(evidence: OutcomeEvidence) -> OutcomeDecision:
    """Apply the ordered decision gate only to complete optimized evidence."""

    interested = tuple(item for item in evidence.chain_gates if item.hardware_interest)
    interested_ids = tuple(sorted(item.chain.chain_id for item in interested))
    if not evidence.preservation_economic_for_measured_workload:
        return OutcomeDecision(
            outcome=Experiment004Outcome.PRESERVATION_NOT_ECONOMIC,
            rationale="kill/recompute dominated complete preservation cost for the measured workload",
            strong_hardware_result=False,
            hardware_interest_chain_ids=(),
        )
    fabric = tuple(
        item for item in interested if item.chain.placement_class is PlacementClass.FABRIC
    )
    if fabric:
        return OutcomeDecision(
            outcome=Experiment004Outcome.FABRIC_HARDWARE_INTEREST,
            rationale="an optimized fabric-adjacent chain passed every system and realizability gate",
            strong_hardware_result=any(
                item.realistic_end_to_end_speedup >= 1.20 for item in fabric
            ),
            hardware_interest_chain_ids=tuple(sorted(item.chain.chain_id for item in fabric)),
        )
    host = tuple(item for item in interested if item.chain.placement_class is PlacementClass.HOST)
    if host:
        return OutcomeDecision(
            outcome=Experiment004Outcome.HOST_PIPELINE_HARDWARE_INTEREST,
            rationale="an optimized GPU-host state chain passed every system and realizability gate",
            strong_hardware_result=any(item.realistic_end_to_end_speedup >= 1.20 for item in host),
            hardware_interest_chain_ids=tuple(sorted(item.chain.chain_id for item in host)),
        )
    gpu = tuple(item for item in interested if item.chain.placement_class is PlacementClass.GPU)
    if gpu or evidence.optimized_removed_most_naive_headroom:
        return OutcomeDecision(
            outcome=Experiment004Outcome.GPU_SOFTWARE_TARGET,
            rationale=(
                "the remaining regular path is GPU-local and belongs in CUDA/Triton software"
                if gpu
                else "movement mattered before optimization but measured GPU software removed most headroom"
            ),
            strong_hardware_result=False,
            hardware_interest_chain_ids=tuple(sorted(item.chain.chain_id for item in gpu)),
        )
    if evidence.optimized_movement_fraction < 0.15 and not interested:
        rationale = (
            "optimized preservation movement was below the end-to-end hardware-interest floor"
        )
    else:
        rationale = "no optimized chain satisfied both the system and realizability hardware gates"
    return OutcomeDecision(
        outcome=Experiment004Outcome.MOVEMENT_CLOSED,
        rationale=rationale,
        strong_hardware_result=False,
        hardware_interest_chain_ids=interested_ids,
    )


def finite_fraction(numerator: int | float, denominator: int | float) -> float:
    """Shared fail-closed fraction helper for report generation."""

    values = (float(numerator), float(denominator))
    if not all(math.isfinite(value) and value >= 0 for value in values):
        raise ValueError("fraction inputs must be finite and nonnegative")
    if values[1] == 0:
        raise ValueError("fraction denominator must be positive")
    return values[0] / values[1]


__all__ = [
    "AmdahlPoint",
    "AmdahlProjection",
    "CriticalPath",
    "CriticalPathKind",
    "Experiment004Outcome",
    "FusedChainEvidence",
    "HardwareInterestEvidence",
    "MeasuredInterval",
    "OutcomeDecision",
    "OutcomeEvidence",
    "PlacementClass",
    "finite_fraction",
    "project_amdahl",
    "select_outcome",
]
