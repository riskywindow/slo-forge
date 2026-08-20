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
    SOFTWARE_WINS = "SOFTWARE_WINS"
    GPU_SOFTWARE_WINS = "GPU_SOFTWARE_WINS"
    BRANCHFABRIC_HARDWARE_INTEREST = "BRANCHFABRIC_HARDWARE_INTEREST"
    PRESERVATION_NOT_ECONOMIC = "PRESERVATION_NOT_ECONOMIC"
    HARDWARE_GATE_NOT_REACHED = "HARDWARE_GATE_NOT_REACHED"


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
        """Require a material share of at least one integrated critical path."""

        return any(
            (
                self.fraction_of_reclamation >= 0.15,
                self.fraction_of_resume >= 0.15,
                self.fraction_of_full_transaction >= 0.15,
                self.fraction_of_slo_restoration >= 0.15,
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
            and self.realistic_end_to_end_speedup >= 1.20
        )


class OutcomeEvidence(_StrictModel):
    """Evidence available to the final Experiment 004 classification gate.

    Incomplete evidence is intentionally representable.  The classifier must
    turn it into ``HARDWARE_GATE_NOT_REACHED`` instead of making callers catch a
    validation error and choose a misleading fallback outcome.
    """

    valid_pilot: bool | None = None
    integrated_v11_scientifically_valid: bool | None = None
    kill_trials: int = Field(default=0, ge=0)
    naive_trials: int = Field(default=0, ge=0)
    optimized_trials: int = Field(default=0, ge=0)
    optimized_semantics_valid: bool | None = None
    preservation_economic_for_measured_workload: bool | None = None
    optimized_removed_most_naive_headroom: bool | None = None
    profiling_hardware_backed: bool | None = None
    trace_overhead_gate_passed: bool | None = None
    optimized_path_measured_after_naive: bool | None = None
    optimized_path_semantics_match_naive: bool | None = None
    optimized_movement_fraction: float | None = Field(
        default=None, ge=0.0, le=1.0, allow_inf_nan=False
    )
    integrated_residual_critical_path_measured: bool = False
    amdahl_analysis_calculated: bool = False
    economic_comparison_established: bool = False
    gpu_methodology_valid: bool | None = None
    chain_gates: tuple[HardwareInterestEvidence, ...] = ()

    @property
    def missing_mandatory_evidence(self) -> tuple[str, ...]:
        missing: list[str] = []
        required_true = (
            ("pilot_scientific_validity", self.valid_pilot),
            ("integrated_v11_scientific_validity", self.integrated_v11_scientifically_valid),
            ("optimized_semantics", self.optimized_semantics_valid),
            ("hardware_backed_profiling", self.profiling_hardware_backed),
            ("trace_overhead_gate", self.trace_overhead_gate_passed),
            ("optimized_after_naive", self.optimized_path_measured_after_naive),
            ("optimized_naive_semantic_equivalence", self.optimized_path_semantics_match_naive),
            ("integrated_residual_critical_path", self.integrated_residual_critical_path_measured),
            ("amdahl_analysis", self.amdahl_analysis_calculated),
            ("economic_comparison", self.economic_comparison_established),
            ("gpu_methodology", self.gpu_methodology_valid),
        )
        missing.extend(name for name, value in required_true if value is not True)
        for name, count in (
            ("kill_and_recompute_trial", self.kill_trials),
            ("naive_preservation_baseline", self.naive_trials),
            ("integrated_optimized_v11_trial", self.optimized_trials),
        ):
            if count < 1:
                missing.append(name)
        if self.preservation_economic_for_measured_workload is None:
            missing.append("preservation_economic_result")
        if self.optimized_removed_most_naive_headroom is None:
            missing.append("software_headroom_result")
        if self.optimized_movement_fraction is None:
            missing.append("integrated_movement_fraction")
        return tuple(missing)


class OutcomeDecision(_StrictModel):
    outcome: Experiment004Outcome
    rationale: str = Field(min_length=1, max_length=4096)
    strong_hardware_result: bool
    hardware_interest_chain_ids: tuple[str, ...]
    missing_mandatory_evidence: tuple[str, ...] = ()


def select_outcome(evidence: OutcomeEvidence) -> OutcomeDecision:
    """Apply the ordered final gate, failing closed when evidence is unavailable."""

    missing = evidence.missing_mandatory_evidence
    if missing:
        return OutcomeDecision(
            outcome=Experiment004Outcome.HARDWARE_GATE_NOT_REACHED,
            rationale=(
                "mandatory evidence for the Experiment 004 hardware decision is unavailable or "
                f"invalid: {', '.join(missing)}"
            ),
            strong_hardware_result=False,
            hardware_interest_chain_ids=(),
            missing_mandatory_evidence=missing,
        )

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
            outcome=Experiment004Outcome.BRANCHFABRIC_HARDWARE_INTEREST,
            rationale="an optimized fabric-adjacent chain passed every system and realizability gate",
            strong_hardware_result=any(
                item.realistic_end_to_end_speedup >= 1.20 for item in fabric
            ),
            hardware_interest_chain_ids=tuple(sorted(item.chain.chain_id for item in fabric)),
        )
    host = tuple(item for item in interested if item.chain.placement_class is PlacementClass.HOST)
    if host:
        return OutcomeDecision(
            outcome=Experiment004Outcome.BRANCHFABRIC_HARDWARE_INTEREST,
            rationale="an optimized GPU-host state chain passed every system and realizability gate",
            strong_hardware_result=any(item.realistic_end_to_end_speedup >= 1.20 for item in host),
            hardware_interest_chain_ids=tuple(sorted(item.chain.chain_id for item in host)),
        )
    gpu = tuple(item for item in interested if item.chain.placement_class is PlacementClass.GPU)
    if gpu or evidence.optimized_removed_most_naive_headroom:
        return OutcomeDecision(
            outcome=Experiment004Outcome.GPU_SOFTWARE_WINS,
            rationale=(
                "the remaining regular path is GPU-local and belongs in CUDA/Triton software"
                if gpu
                else "movement mattered before optimization but measured GPU software removed most headroom"
            ),
            strong_hardware_result=False,
            hardware_interest_chain_ids=tuple(sorted(item.chain.chain_id for item in gpu)),
        )
    assert evidence.optimized_movement_fraction is not None
    realistic_speedups = tuple(item.realistic_end_to_end_speedup for item in evidence.chain_gates)
    software_gate_passed = evidence.optimized_movement_fraction < 0.15 or (
        bool(realistic_speedups) and max(realistic_speedups) < 1.15
    )
    if software_gate_passed:
        rationale = (
            "optimized preservation is below the integrated headroom threshold"
            if evidence.optimized_movement_fraction < 0.15
            else "every realistic acceleration projection is below 1.15x end to end"
        )
        return OutcomeDecision(
            outcome=Experiment004Outcome.SOFTWARE_WINS,
            rationale=rationale,
            strong_hardware_result=False,
            hardware_interest_chain_ids=interested_ids,
        )
    return OutcomeDecision(
        outcome=Experiment004Outcome.HARDWARE_GATE_NOT_REACHED,
        rationale=(
            "complete measurements did not establish any allowed numeric terminal gate; "
            "additional decision evidence is required"
        ),
        strong_hardware_result=False,
        hardware_interest_chain_ids=interested_ids,
        missing_mandatory_evidence=("terminal_gate_result",),
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
