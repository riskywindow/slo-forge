"""Fail-closed BranchFabric hardware evidence gates.

The compiler consumes measured evidence summaries whose raw artifact bindings
are verified before any gate is evaluated.  Missing values stay missing: a
candidate cannot pass by projection, by a CPU-only substitute for target
hardware, or by an isolated-operation speedup.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

GATE_INPUT_SCHEMA_VERSION: Literal["sloforge.branchfabric.gate-input/v1"] = (
    "sloforge.branchfabric.gate-input/v1"
)
GATE_RESULT_SCHEMA_VERSION: Literal["sloforge.branchfabric.gate-result/v1"] = (
    "sloforge.branchfabric.gate-result/v1"
)
MAX_CANDIDATES = 64
MAX_EVIDENCE_REFERENCES = 512

NonEmpty = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=4096)]
Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
GitCommit = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{40}$")]
Fraction = Annotated[float, Field(ge=0.0, le=1.0, allow_inf_nan=False)]
NonNegative = Annotated[float, Field(ge=0.0, allow_inf_nan=False)]


class GateModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, validate_default=True)


class EvidenceClass(StrEnum):
    TARGET_HARDWARE_REAL = "TARGET_HARDWARE_REAL"
    LOCAL_CPU_REAL = "LOCAL_CPU_REAL"
    CPU_REFERENCE_MODEL_STATE = "CPU_REFERENCE_MODEL_STATE"
    CPU_REFERENCE_LOCAL_TRANSACTION = "CPU_REFERENCE_LOCAL_TRANSACTION"
    SYNTHETIC = "SYNTHETIC"
    SIMULATED_HARDWARE = "SIMULATED_HARDWARE"
    ARTIFACT_REPLAY = "ARTIFACT_REPLAY"


class CandidateKind(StrEnum):
    DATA_PATH = "DATA_PATH"
    FANOUT = "FANOUT"
    METADATA = "METADATA"
    STATE_SHARING = "STATE_SHARING"


class GateStatus(StrEnum):
    PASS = "PASS"
    FAIL = "FAIL"


class EvidenceReference(GateModel):
    path: NonEmpty
    sha256: Sha256
    evidence_class: EvidenceClass
    sample_count: int = Field(ge=0, le=2**63 - 1)
    raw_samples: bool


class CriticalPathEvidence(GateModel):
    target_critical_path_fraction: Fraction | None = None
    gpu_interference_fraction: Fraction | None = None
    state_movement_time_fraction: Fraction | None = None
    state_movement_bytes_fraction: Fraction | None = None
    capacity_reclamation_fraction: Fraction | None = None
    branch_readiness_fraction: Fraction | None = None


class HeadroomEvidence(GateModel):
    end_to_end_speedup_lower_bound: NonNegative | None = None
    p99_branch_readiness_reduction_lower_bound: Fraction | None = None
    reclamation_interruption_reduction_lower_bound: Fraction | None = None
    gpu_interference_reduction_lower_bound: Fraction | None = None
    physical_state_movement_reduction_lower_bound: Fraction | None = None
    valid_trajectories_per_gpu_hour_increase_lower_bound: Fraction | None = None


class PlatformFeasibility(GateModel):
    byte_rate_bytes_per_second: NonNegative | None = None
    operation_rate_per_second: NonNegative | None = None
    concurrency: int | None = Field(default=None, ge=1)
    queue_depth: int | None = Field(default=None, ge=1)
    working_set_bytes: int | None = Field(default=None, ge=1)
    latency_target_ns: int | None = Field(default=None, ge=1)
    bandwidth_target_bytes_per_second: NonNegative | None = None
    fault_behavior: NonEmpty | None = None
    interface_requirement: NonEmpty | None = None
    credible_resource_estimate: bool = False
    selected_target_fits: bool = False


class MetricEvidenceBinding(GateModel):
    """Bind one gate metric to its derivation and immutable raw samples."""

    summary_artifact_path: NonEmpty
    raw_artifact_paths: tuple[NonEmpty, ...] = Field(min_length=1, max_length=64)
    workload_classes: tuple[NonEmpty, ...] = Field(min_length=1, max_length=16)
    seeds: tuple[int, ...] = Field(min_length=1, max_length=64)
    derivation_method: NonEmpty
    confidence_level: Fraction | None = None
    target_hardware_timing: bool

    @model_validator(mode="after")
    def distinct_inputs(self) -> Self:
        if len(self.raw_artifact_paths) != len(set(self.raw_artifact_paths)):
            raise ValueError("metric raw artifact paths must be unique")
        if len(self.workload_classes) != len(set(self.workload_classes)):
            raise ValueError("metric workload classes must be unique")
        if len(self.seeds) != len(set(self.seeds)):
            raise ValueError("metric seeds must be unique")
        return self


class CandidateEvidence(GateModel):
    candidate: NonEmpty
    kind: CandidateKind
    workload_classes: tuple[NonEmpty, ...] = Field(min_length=1, max_length=16)
    seeds: tuple[int, ...] = Field(min_length=1, max_length=64)
    software_baseline_optimized: bool
    complete_software_manifest: bool
    complete_hardware_manifest: bool
    tracing_overhead_controlled: bool
    hidden_fallback: bool
    hidden_cpu_substitution: bool
    fabricated_hardware_result: bool
    evidence: tuple[EvidenceReference, ...] = Field(
        min_length=1, max_length=MAX_EVIDENCE_REFERENCES
    )
    critical_path: CriticalPathEvidence
    headroom: HeadroomEvidence
    critical_path_bindings: dict[NonEmpty, MetricEvidenceBinding] = Field(default_factory=dict)
    headroom_bindings: dict[NonEmpty, MetricEvidenceBinding] = Field(default_factory=dict)
    platform: PlatformFeasibility
    workload_value_signals: dict[NonEmpty, bool] = Field(default_factory=dict)

    @model_validator(mode="after")
    def distinct_workloads_and_seeds(self) -> Self:
        if len(self.workload_classes) != len(set(self.workload_classes)):
            raise ValueError("workload classes must be unique")
        if len(self.seeds) != len(set(self.seeds)):
            raise ValueError("seeds must be unique")
        evidence_paths = {reference.path for reference in self.evidence}
        for bindings in (self.critical_path_bindings, self.headroom_bindings):
            for metric, binding in bindings.items():
                bound_paths = {binding.summary_artifact_path, *binding.raw_artifact_paths}
                missing = bound_paths - evidence_paths
                if missing:
                    raise ValueError(
                        f"metric binding {metric!r} cites evidence absent from the candidate: "
                        + ", ".join(sorted(missing))
                    )
        return self


class GateInput(GateModel):
    schema_version: Literal["sloforge.branchfabric.gate-input/v1"]
    phase: Literal["preliminary", "final"]
    baseline_commit: GitCommit
    generated_at: NonEmpty
    thresholds_changed: bool
    candidates: tuple[CandidateEvidence, ...] = Field(min_length=1, max_length=MAX_CANDIDATES)

    @model_validator(mode="after")
    def candidate_names_are_unique(self) -> Self:
        names = tuple(candidate.candidate for candidate in self.candidates)
        if len(names) != len(set(names)):
            raise ValueError("candidate names must be unique")
        if self.thresholds_changed:
            raise ValueError("this compiler implements the fixed conservative gates")
        return self


class GateCheck(GateModel):
    gate: NonEmpty
    status: GateStatus
    rationale: NonEmpty
    thresholds: tuple[NonEmpty, ...]


class CandidateGateResult(GateModel):
    candidate: NonEmpty
    kind: CandidateKind
    mandatory_real_evidence: GateCheck
    mandatory_end_to_end_relevance: GateCheck
    mandatory_system_level_headroom: GateCheck
    mandatory_platform_feasibility: GateCheck
    workload_value_gate: GateCheck
    passes_all_mandatory_and_one_value_gate: bool
    disposition: Literal["HARDWARE_HIGH_VALUE", "NOT_JUSTIFIED"]
    evidence: tuple[EvidenceReference, ...]


class GateResult(GateModel):
    schema_version: Literal["sloforge.branchfabric.gate-result/v1"]
    phase: Literal["preliminary", "final"]
    baseline_commit: GitCommit
    generated_at: NonEmpty
    thresholds_changed: Literal[False]
    outcome: Literal["PASS", "FAIL_NO_BUILD"]
    hardware_implementation_allowed: bool
    functional_model_or_cycle_simulator_allowed: bool
    passing_candidates: tuple[NonEmpty, ...]
    candidates: tuple[CandidateGateResult, ...]
    required_action: Literal["IMPLEMENT_MINIMAL_PASSING_SUBSET", "TERMINATE_HARDWARE_PATH"]
    negative_result_preserved: bool


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _verify_evidence(root: Path, references: tuple[EvidenceReference, ...]) -> None:
    for reference in references:
        path = root / reference.path
        try:
            resolved = path.resolve(strict=True)
        except FileNotFoundError as error:
            raise ValueError(f"evidence artifact does not exist: {reference.path}") from error
        resolved.relative_to(root.resolve())
        if not resolved.is_file():
            raise ValueError(f"evidence artifact is not a file: {reference.path}")
        actual = _sha256(resolved)
        if actual != reference.sha256:
            raise ValueError(
                f"evidence digest mismatch for {reference.path}: {actual} != {reference.sha256}"
            )


def _check_real_evidence(candidate: CandidateEvidence) -> GateCheck:
    has_target_hardware = any(
        reference.evidence_class is EvidenceClass.TARGET_HARDWARE_REAL
        and reference.raw_samples
        and reference.sample_count > 0
        for reference in candidate.evidence
    )
    passed = all(
        (
            has_target_hardware,
            len(candidate.workload_classes) >= 2,
            len(candidate.seeds) >= 3,
            candidate.software_baseline_optimized,
            candidate.complete_software_manifest,
            candidate.complete_hardware_manifest,
            candidate.tracing_overhead_controlled,
            not candidate.hidden_fallback,
            not candidate.hidden_cpu_substitution,
            not candidate.fabricated_hardware_result,
        )
    )
    missing: list[str] = []
    if not has_target_hardware:
        missing.append("no target-relevant real hardware raw samples")
    if len(candidate.workload_classes) < 2:
        missing.append("fewer than two workload classes")
    if len(candidate.seeds) < 3:
        missing.append("fewer than three seeds")
    if not candidate.software_baseline_optimized:
        missing.append("optimized software baseline absent")
    if not candidate.complete_software_manifest or not candidate.complete_hardware_manifest:
        missing.append("software/hardware manifest incomplete")
    if not candidate.tracing_overhead_controlled:
        missing.append("tracing overhead not controlled")
    if candidate.hidden_fallback or candidate.hidden_cpu_substitution:
        missing.append("fallback or CPU substitution detected")
    if candidate.fabricated_hardware_result:
        missing.append("fabricated hardware result detected")
    rationale = "all real-evidence requirements passed" if passed else "; ".join(missing)
    return GateCheck(
        gate="mandatory_real_evidence",
        status=GateStatus.PASS if passed else GateStatus.FAIL,
        rationale=rationale,
        thresholds=(
            "target-relevant hardware-backed raw measurements",
            ">=2 Helix workload classes",
            ">=3 seeds",
            "optimized software baseline and complete manifests",
            "controlled tracing with no fallback, substitution, or fabrication",
        ),
    )


def _binding_has_raw_samples(candidate: CandidateEvidence, binding: MetricEvidenceBinding) -> bool:
    references = {reference.path: reference for reference in candidate.evidence}
    return all(references[path].raw_samples for path in binding.raw_artifact_paths)


def _binding_has_target_hardware_samples(
    candidate: CandidateEvidence, binding: MetricEvidenceBinding
) -> bool:
    references = {reference.path: reference for reference in candidate.evidence}
    return binding.target_hardware_timing and any(
        references[path].raw_samples
        and references[path].sample_count > 0
        and references[path].evidence_class is EvidenceClass.TARGET_HARDWARE_REAL
        for path in binding.raw_artifact_paths
    )


def _check_relevance(candidate: CandidateEvidence) -> GateCheck:
    metrics = {
        "target critical path": (
            "target_critical_path_fraction",
            candidate.critical_path.target_critical_path_fraction,
            0.10,
        ),
        "GPU interference": (
            "gpu_interference_fraction",
            candidate.critical_path.gpu_interference_fraction,
            0.15,
        ),
        "state-movement time": (
            "state_movement_time_fraction",
            candidate.critical_path.state_movement_time_fraction,
            0.15,
        ),
        "state-movement bytes": (
            "state_movement_bytes_fraction",
            candidate.critical_path.state_movement_bytes_fraction,
            0.15,
        ),
        "capacity reclamation": (
            "capacity_reclamation_fraction",
            candidate.critical_path.capacity_reclamation_fraction,
            0.10,
        ),
        "BranchPoint-to-readiness": (
            "branch_readiness_fraction",
            candidate.critical_path.branch_readiness_fraction,
            0.10,
        ),
    }
    passing = [
        name
        for name, (field, value, threshold) in metrics.items()
        if value is not None
        and value >= threshold
        and (binding := candidate.critical_path_bindings.get(field)) is not None
        and _binding_has_raw_samples(candidate, binding)
    ]
    observed = [
        f"{name}={value:.6f}"
        + (
            " (raw-bound)"
            if (binding := candidate.critical_path_bindings.get(field)) is not None
            and _binding_has_raw_samples(candidate, binding)
            else " (unbound)"
        )
        for name, (field, value, _) in metrics.items()
        if value is not None
    ]
    return GateCheck(
        gate="mandatory_end_to_end_relevance",
        status=GateStatus.PASS if passing else GateStatus.FAIL,
        rationale=(
            f"passing measured fractions: {', '.join(passing)}"
            if passing
            else "no measured fraction reached a threshold; "
            + (", ".join(observed) or "all are unknown")
        ),
        thresholds=(
            ">=10% target critical path or BranchPoint-to-readiness",
            ">=15% GPU interference, state-movement time, or state-movement bytes",
            ">=10% capacity-reclamation critical path",
        ),
    )


def _check_headroom(candidate: CandidateEvidence) -> GateCheck:
    metrics = {
        "end-to-end speedup": (
            "end_to_end_speedup_lower_bound",
            candidate.headroom.end_to_end_speedup_lower_bound,
            1.15,
        ),
        "p99 readiness reduction": (
            "p99_branch_readiness_reduction_lower_bound",
            candidate.headroom.p99_branch_readiness_reduction_lower_bound,
            0.20,
        ),
        "reclamation interruption reduction": (
            "reclamation_interruption_reduction_lower_bound",
            candidate.headroom.reclamation_interruption_reduction_lower_bound,
            0.20,
        ),
        "GPU interference reduction": (
            "gpu_interference_reduction_lower_bound",
            candidate.headroom.gpu_interference_reduction_lower_bound,
            0.20,
        ),
        "physical movement reduction": (
            "physical_state_movement_reduction_lower_bound",
            candidate.headroom.physical_state_movement_reduction_lower_bound,
            0.25,
        ),
        "valid trajectories/GPU-hour increase": (
            "valid_trajectories_per_gpu_hour_increase_lower_bound",
            candidate.headroom.valid_trajectories_per_gpu_hour_increase_lower_bound,
            0.20,
        ),
    }
    passing = [
        name
        for name, (field, value, threshold) in metrics.items()
        if value is not None
        and value >= threshold
        and (binding := candidate.headroom_bindings.get(field)) is not None
        and binding.confidence_level is not None
        and binding.confidence_level >= 0.95
        and len(binding.workload_classes) >= 2
        and len(binding.seeds) >= 3
        and _binding_has_raw_samples(candidate, binding)
        and _binding_has_target_hardware_samples(candidate, binding)
    ]
    observed = [
        f"{name}={value:.6f}"
        + (
            " (95%+ target-hardware raw-bound)"
            if (binding := candidate.headroom_bindings.get(field)) is not None
            and binding.confidence_level is not None
            and binding.confidence_level >= 0.95
            and len(binding.workload_classes) >= 2
            and len(binding.seeds) >= 3
            and _binding_has_raw_samples(candidate, binding)
            and _binding_has_target_hardware_samples(candidate, binding)
            else " (not qualifying evidence-bound LCB)"
        )
        for name, (field, value, _) in metrics.items()
        if value is not None
    ]
    return GateCheck(
        gate="mandatory_system_level_headroom",
        status=GateStatus.PASS if passing else GateStatus.FAIL,
        rationale=(
            f"passing lower confidence bounds: {', '.join(passing)}"
            if passing
            else "no lower confidence bound reached a threshold; "
            + (", ".join(observed) or "all are unknown")
        ),
        thresholds=(
            ">=1.15x end-to-end speedup",
            ">=20% p99 readiness, reclamation interruption, or GPU-interference reduction",
            ">=25% physical state-movement reduction",
            ">=20% valid trajectories per GPU-hour increase",
        ),
    )


def _check_platform(candidate: CandidateEvidence) -> GateCheck:
    platform = candidate.platform
    required = {
        "byte rate": platform.byte_rate_bytes_per_second,
        "operation rate": platform.operation_rate_per_second,
        "concurrency": platform.concurrency,
        "queue depth": platform.queue_depth,
        "working set": platform.working_set_bytes,
        "latency target": platform.latency_target_ns,
        "bandwidth target": platform.bandwidth_target_bytes_per_second,
        "fault behavior": platform.fault_behavior,
        "interface requirement": platform.interface_requirement,
    }
    missing = [name for name, value in required.items() if value is None]
    passed = not missing and platform.credible_resource_estimate and platform.selected_target_fits
    if not platform.credible_resource_estimate:
        missing.append("credible resource estimate")
    if not platform.selected_target_fits:
        missing.append("selected-target fit")
    return GateCheck(
        gate="mandatory_platform_feasibility",
        status=GateStatus.PASS if passed else GateStatus.FAIL,
        rationale="all placement requirements measured"
        if passed
        else "missing: " + ", ".join(missing),
        thresholds=(
            "measured byte/operation rates, concurrency, queue, working set, latency, bandwidth",
            "evidence-backed fault/interface requirements",
            "credible resource estimate fitting selected target",
        ),
    )


_VALUE_SIGNALS: dict[CandidateKind, tuple[str, ...]] = {
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
}


def _check_workload_value(candidate: CandidateEvidence) -> GateCheck:
    required = _VALUE_SIGNALS[candidate.kind]
    missing = [name for name in required if not candidate.workload_value_signals.get(name, False)]
    return GateCheck(
        gate=f"workload_value_{candidate.kind.value.lower()}",
        status=GateStatus.PASS if not missing else GateStatus.FAIL,
        rationale="all workload-value signals measured"
        if not missing
        else "missing: " + ", ".join(missing),
        thresholds=required,
    )


def evaluate_gate(gate_input: GateInput, *, repository_root: Path) -> GateResult:
    """Verify evidence and apply the fixed gate thresholds."""

    candidate_results: list[CandidateGateResult] = []
    for candidate in gate_input.candidates:
        _verify_evidence(repository_root, candidate.evidence)
        real = _check_real_evidence(candidate)
        relevance = _check_relevance(candidate)
        headroom = _check_headroom(candidate)
        platform = _check_platform(candidate)
        value = _check_workload_value(candidate)
        passed = all(
            check.status is GateStatus.PASS
            for check in (real, relevance, headroom, platform, value)
        )
        candidate_results.append(
            CandidateGateResult(
                candidate=candidate.candidate,
                kind=candidate.kind,
                mandatory_real_evidence=real,
                mandatory_end_to_end_relevance=relevance,
                mandatory_system_level_headroom=headroom,
                mandatory_platform_feasibility=platform,
                workload_value_gate=value,
                passes_all_mandatory_and_one_value_gate=passed,
                disposition="HARDWARE_HIGH_VALUE" if passed else "NOT_JUSTIFIED",
                evidence=candidate.evidence,
            )
        )
    passing = tuple(
        result.candidate
        for result in candidate_results
        if result.passes_all_mandatory_and_one_value_gate
    )
    allowed = bool(passing)
    return GateResult(
        schema_version=GATE_RESULT_SCHEMA_VERSION,
        phase=gate_input.phase,
        baseline_commit=gate_input.baseline_commit,
        generated_at=gate_input.generated_at,
        thresholds_changed=False,
        outcome="PASS" if allowed else "FAIL_NO_BUILD",
        hardware_implementation_allowed=allowed,
        functional_model_or_cycle_simulator_allowed=allowed,
        passing_candidates=passing,
        candidates=tuple(candidate_results),
        required_action="IMPLEMENT_MINIMAL_PASSING_SUBSET"
        if allowed
        else "TERMINATE_HARDWARE_PATH",
        negative_result_preserved=not allowed,
    )


def _write_json(path: Path, value: BaseModel, *, replace: bool) -> None:
    if path.exists() and not replace:
        raise FileExistsError(f"output exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value.model_dump(mode="json"), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_report(path: Path, result: GateResult, *, replace: bool) -> None:
    if path.exists() and not replace:
        raise FileExistsError(f"output exists: {path}")
    lines = [
        "# BranchFabric hardware gate report",
        "",
        f"Phase: **{result.phase}**.",
        f"Outcome: **{result.outcome}**.",
        f"Hardware implementation allowed: **{str(result.hardware_implementation_allowed).lower()}**.",
        "",
        "No threshold was loosened. CPU-reference, synthetic, simulated-hardware, and artifact-replay evidence cannot satisfy the target-hardware requirement.",
        "",
        "## Candidate results",
        "",
        "| Candidate | Real evidence | End-to-end relevance | Headroom | Platform | Workload value | Disposition |",
        "|---|---|---|---|---|---|---|",
    ]
    for candidate in result.candidates:
        lines.append(
            "| "
            + " | ".join(
                (
                    candidate.candidate,
                    candidate.mandatory_real_evidence.status.value,
                    candidate.mandatory_end_to_end_relevance.status.value,
                    candidate.mandatory_system_level_headroom.status.value,
                    candidate.mandatory_platform_feasibility.status.value,
                    candidate.workload_value_gate.status.value,
                    candidate.disposition,
                )
            )
            + " |"
        )
    lines.extend(
        (
            "",
            "## Candidate calculations and evidence",
            "",
        )
    )
    for candidate in result.candidates:
        lines.extend((f"### `{candidate.candidate}`", ""))
        for check in (
            candidate.mandatory_real_evidence,
            candidate.mandatory_end_to_end_relevance,
            candidate.mandatory_system_level_headroom,
            candidate.mandatory_platform_feasibility,
            candidate.workload_value_gate,
        ):
            lines.append(f"- `{check.gate}`: **{check.status.value}** — {check.rationale}.")
        lines.append("")
        lines.append("Evidence bindings:")
        lines.append("")
        for reference in candidate.evidence:
            raw = "raw samples" if reference.raw_samples else "derived/replay"
            lines.append(
                f"- `{reference.path}` — SHA-256 `{reference.sha256}`; "
                f"{reference.evidence_class.value}; n={reference.sample_count}; {raw}."
            )
        lines.append("")
    lines.extend(
        (
            "## Required action",
            "",
            result.required_action.replace("_", " ").title() + ".",
            "",
            "Functional-model and cycle-simulator work is allowed only after a hardware gate pass: "
            f"**{str(result.functional_model_or_cycle_simulator_allowed).lower()}**.",
        )
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def compile_gate(
    *, input_path: Path, output_path: Path, report_path: Path, repository_root: Path, replace: bool
) -> GateResult:
    gate_input = GateInput.model_validate_json(input_path.read_text(encoding="utf-8"))
    result = evaluate_gate(gate_input, repository_root=repository_root)
    _write_json(output_path, result, replace=replace)
    _write_report(report_path, result, replace=replace)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Compile fail-closed BranchFabric gates")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--replace", action="store_true")
    arguments = parser.parse_args()
    result = compile_gate(
        input_path=arguments.input,
        output_path=arguments.output,
        report_path=arguments.report,
        repository_root=arguments.repository_root,
        replace=arguments.replace,
    )
    print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
