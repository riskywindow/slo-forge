from __future__ import annotations

import pytest
from pydantic import ValidationError

from sloforge.helix.characterization.gpu_reclamation_analysis import (
    CriticalPath,
    CriticalPathKind,
    Experiment004Outcome,
    FusedChainEvidence,
    HardwareInterestEvidence,
    MeasuredInterval,
    OutcomeEvidence,
    PlacementClass,
    project_amdahl,
    select_outcome,
)


def _path() -> CriticalPath:
    return CriticalPath(
        kind=CriticalPathKind.RECLAMATION,
        intervals=(
            MeasuredInterval(name="quiesce", start_ns=0, end_ns=10),
            MeasuredInterval(name="transform", start_ns=10, end_ns=40),
            MeasuredInterval(name="d2h", start_ns=40, end_ns=100),
        ),
    )


def _chain(*, placement: PlacementClass = PlacementClass.HOST) -> FusedChainEvidence:
    return FusedChainEvidence(
        chain_id="transform-checksum-d2h",
        operations=("transform", "checksum", "d2h"),
        occurrence_count=1,
        logical_bytes=1_000,
        physical_bytes=3_000,
        state_passes=3,
        wall_time_ns=30,
        gpu_time_ns=20,
        cpu_time_ns=4,
        temporary_bytes=1_000,
        dependencies_permit_streaming=True,
        materialized_intermediate_bytes=1_000,
        placement_class=placement,
        measured_fabric_transfer_bytes=(1_000 if placement is PlacementClass.FABRIC else 0),
        measured_fabric_endpoint_count=(2 if placement is PlacementClass.FABRIC else 0),
    )


def _gate(
    *, placement: PlacementClass = PlacementClass.HOST, realistic_speedup: float = 1.2
) -> HardwareInterestEvidence:
    return HardwareInterestEvidence(
        chain=_chain(placement=placement),
        fraction_of_reclamation=0.3,
        fraction_of_resume=0.0,
        fraction_of_full_transaction=0.2,
        fraction_of_slo_restoration=0.1,
        fraction_of_movement_time=0.5,
        serving_degradation_fraction=0.0,
        avoidable_physical_byte_fraction=0.3,
        ideal_free_end_to_end_speedup=1.3,
        realistic_end_to_end_speedup=realistic_speedup,
        regular_dataflow=True,
        measured_byte_rate=True,
        measured_concurrency=True,
        measured_latency_target=True,
        plausible_off_critical_path_placement=True,
    )


def test_critical_path_rejects_gap_and_overlap() -> None:
    with pytest.raises(ValidationError, match="gap or overlap"):
        CriticalPath(
            kind=CriticalPathKind.RESTORE,
            intervals=(
                MeasuredInterval(name="h2d", start_ns=0, end_ns=5),
                MeasuredInterval(name="write", start_ns=6, end_ns=9),
            ),
        )


def test_amdahl_combined_chain_uses_nonoverlapping_stages() -> None:
    projection = project_amdahl(_path(), target_names=("transform", "d2h"))
    assert projection.target_total_ns == 90
    assert projection.target_fraction == pytest.approx(0.9)
    assert projection.points[0].projected_total_ns == 55
    assert projection.points[-1].projected_total_ns == 10
    with pytest.raises(ValueError, match="unique"):
        project_amdahl(_path(), target_names=("d2h", "d2h"))


def test_hardware_gate_requires_system_headroom_and_realizability() -> None:
    assert _gate().hardware_interest
    blocked = _gate().model_copy(update={"measured_concurrency": False})
    assert not blocked.hardware_interest
    too_small = _gate(realistic_speedup=1.1)
    assert not too_small.hardware_interest


def test_outcome_ordering_is_exact_and_economic_gate_is_first() -> None:
    base = OutcomeEvidence(
        valid_pilot=True,
        kill_trials=1,
        naive_trials=1,
        optimized_trials=1,
        optimized_semantics_valid=True,
        preservation_economic_for_measured_workload=True,
        optimized_removed_most_naive_headroom=False,
        profiling_hardware_backed=True,
        trace_overhead_gate_passed=True,
        optimized_path_measured_after_naive=True,
        optimized_path_semantics_match_naive=True,
        optimized_movement_fraction=0.3,
        chain_gates=(_gate(),),
    )
    assert select_outcome(base).outcome is Experiment004Outcome.HOST_PIPELINE_HARDWARE_INTEREST
    fabric = base.model_copy(update={"chain_gates": (_gate(placement=PlacementClass.FABRIC),)})
    assert select_outcome(fabric).outcome is Experiment004Outcome.FABRIC_HARDWARE_INTEREST
    uneconomic = fabric.model_copy(update={"preservation_economic_for_measured_workload": False})
    assert select_outcome(uneconomic).outcome is Experiment004Outcome.PRESERVATION_NOT_ECONOMIC


def test_software_and_closed_outcomes_require_no_passing_chain() -> None:
    blocked = _gate().model_copy(update={"regular_dataflow": False})
    software = OutcomeEvidence(
        valid_pilot=True,
        kill_trials=1,
        naive_trials=1,
        optimized_trials=1,
        optimized_semantics_valid=True,
        preservation_economic_for_measured_workload=True,
        optimized_removed_most_naive_headroom=True,
        profiling_hardware_backed=True,
        trace_overhead_gate_passed=True,
        optimized_path_measured_after_naive=True,
        optimized_path_semantics_match_naive=True,
        optimized_movement_fraction=0.1,
        chain_gates=(blocked,),
    )
    assert select_outcome(software).outcome is Experiment004Outcome.GPU_SOFTWARE_TARGET
    closed = software.model_copy(update={"optimized_removed_most_naive_headroom": False})
    assert select_outcome(closed).outcome is Experiment004Outcome.MOVEMENT_CLOSED


def test_fabric_and_speedup_claims_require_physical_evidence() -> None:
    with pytest.raises(ValidationError, match="measured fabric"):
        _chain().model_copy(
            update={"placement_class": PlacementClass.FABRIC},
        ).model_validate(
            _chain().model_copy(update={"placement_class": PlacementClass.FABRIC}).model_dump(),
            strict=True,
        )
    with pytest.raises(ValidationError, match="ideal-free"):
        HardwareInterestEvidence.model_validate(
            _gate().model_copy(update={"realistic_end_to_end_speedup": 1.4}).model_dump(),
            strict=True,
        )
    general = _gate().model_copy(
        update={"chain": _chain(placement=PlacementClass.GENERAL_SOFTWARE)}
    )
    assert not general.hardware_interest
