from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from sloforge.helix.characterization.gates import (
    CandidateEvidence,
    CandidateKind,
    CriticalPathEvidence,
    EvidenceClass,
    EvidenceReference,
    GateInput,
    GateStatus,
    HeadroomEvidence,
    PlatformFeasibility,
    compile_gate,
    evaluate_gate,
)


def _reference(path: Path, evidence_class: EvidenceClass) -> EvidenceReference:
    return EvidenceReference(
        path=path.name,
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        evidence_class=evidence_class,
        sample_count=3,
        raw_samples=True,
    )


def _signals(kind: CandidateKind, value: bool) -> dict[str, bool]:
    names = {
        CandidateKind.DATA_PATH: (
            "large_state_operations",
            "repeated_frequency",
            "bandwidth_pressure",
            "temporary_memory_or_interference",
            "full_system_leverage",
        ),
        CandidateKind.FANOUT: (
            "repeated_physical_unicast",
            "destination_count_gt_one",
            "identical_large_source",
            "unicast_material_to_bytes_or_latency",
            "transforms_preserve_benefit",
        ),
        CandidateKind.METADATA: (
            "optimized_software_metadata_is_critical",
            "operation_rate_and_concurrency_measured",
            "cache_lock_allocation_pressure_measured",
            "hardware_working_set_and_queue_derived",
            "full_system_leverage",
        ),
        CandidateKind.STATE_SHARING: (
            "real_physical_state_measured",
            "representative_fanout",
            "shared_root_lifetime_material",
            "allocation_savings_affect_capacity_or_throughput",
            "software_cow_insufficient",
        ),
    }[kind]
    return dict.fromkeys(names, value)


def _candidate(reference: EvidenceReference, *, passing: bool) -> CandidateEvidence:
    return CandidateEvidence(
        candidate="reshard-transform-send",
        kind=CandidateKind.DATA_PATH,
        workload_classes=("coding_agent", "reasoning_verification"),
        seeds=(41, 73, 113),
        software_baseline_optimized=True,
        complete_software_manifest=True,
        complete_hardware_manifest=True,
        tracing_overhead_controlled=True,
        hidden_fallback=False,
        hidden_cpu_substitution=False,
        fabricated_hardware_result=False,
        evidence=(reference,),
        critical_path=CriticalPathEvidence(target_critical_path_fraction=0.11 if passing else 0.01),
        headroom=HeadroomEvidence(end_to_end_speedup_lower_bound=1.16 if passing else 1.01),
        platform=PlatformFeasibility(
            byte_rate_bytes_per_second=1.0,
            operation_rate_per_second=1.0,
            concurrency=1,
            queue_depth=1,
            working_set_bytes=1,
            latency_target_ns=1,
            bandwidth_target_bytes_per_second=1.0,
            fault_behavior="fail closed",
            interface_requirement="versioned command queue",
            credible_resource_estimate=passing,
            selected_target_fits=passing,
        ),
        workload_value_signals=_signals(CandidateKind.DATA_PATH, passing),
    )


def _input(candidate: CandidateEvidence) -> GateInput:
    return GateInput(
        schema_version="sloforge.branchfabric.gate-input/v1",
        phase="preliminary",
        baseline_commit="a" * 40,
        generated_at="2026-08-09T00:00:00Z",
        thresholds_changed=False,
        candidates=(candidate,),
    )


def test_cpu_only_evidence_cannot_pass_hardware_gate(tmp_path: Path) -> None:
    raw = tmp_path / "raw.json"
    raw.write_text("{}\n", encoding="utf-8")
    candidate = _candidate(_reference(raw, EvidenceClass.CPU_REFERENCE_MODEL_STATE), passing=True)
    result = evaluate_gate(_input(candidate), repository_root=tmp_path)
    assert result.outcome == "FAIL_NO_BUILD"
    assert result.candidates[0].mandatory_real_evidence.status is GateStatus.FAIL
    assert not result.hardware_implementation_allowed
    assert not result.functional_model_or_cycle_simulator_allowed


def test_all_fixed_gates_must_pass(tmp_path: Path) -> None:
    raw = tmp_path / "raw.json"
    raw.write_text("{}\n", encoding="utf-8")
    reference = _reference(raw, EvidenceClass.TARGET_HARDWARE_REAL)
    result = evaluate_gate(_input(_candidate(reference, passing=True)), repository_root=tmp_path)
    assert result.outcome == "PASS"
    assert result.passing_candidates == ("reshard-transform-send",)


def test_isolated_speedup_does_not_override_system_headroom(tmp_path: Path) -> None:
    raw = tmp_path / "raw.json"
    raw.write_text("{}\n", encoding="utf-8")
    reference = _reference(raw, EvidenceClass.TARGET_HARDWARE_REAL)
    result = evaluate_gate(_input(_candidate(reference, passing=False)), repository_root=tmp_path)
    candidate = result.candidates[0]
    assert candidate.mandatory_system_level_headroom.status is GateStatus.FAIL
    assert candidate.mandatory_end_to_end_relevance.status is GateStatus.FAIL
    assert candidate.disposition == "NOT_JUSTIFIED"


def test_artifact_digest_mismatch_fails_before_gate_evaluation(tmp_path: Path) -> None:
    raw = tmp_path / "raw.json"
    raw.write_text("{}\n", encoding="utf-8")
    reference = _reference(raw, EvidenceClass.TARGET_HARDWARE_REAL)
    raw.write_text('{"changed": true}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="evidence digest mismatch"):
        evaluate_gate(_input(_candidate(reference, passing=True)), repository_root=tmp_path)


def test_compiler_report_preserves_calculations_and_evidence_bindings(tmp_path: Path) -> None:
    raw = tmp_path / "raw.json"
    raw.write_text("{}\n", encoding="utf-8")
    gate_input = _input(
        _candidate(_reference(raw, EvidenceClass.CPU_REFERENCE_MODEL_STATE), passing=True)
    )
    input_path = tmp_path / "input.json"
    output_path = tmp_path / "result.json"
    report_path = tmp_path / "report.md"
    input_path.write_text(gate_input.model_dump_json(indent=2), encoding="utf-8")

    compile_gate(
        input_path=input_path,
        output_path=output_path,
        report_path=report_path,
        repository_root=tmp_path,
        replace=False,
    )

    report = report_path.read_text(encoding="utf-8")
    assert "no target-relevant real hardware raw samples" in report
    assert raw.name in report
    assert hashlib.sha256(raw.read_bytes()).hexdigest() in report
    assert "cycle-simulator work is allowed only after a hardware gate pass: **false**" in report
