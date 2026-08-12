"""Evidence-gated report and plot publication for Experiment 004.

Synthetic fixtures may exercise deterministic plot construction, but their
outputs live under ``fixture-preview`` and contain no outcome decision.  The
final report/document paths are reachable only from complete, hash-verified,
hardware-backed evidence.
"""

from __future__ import annotations

import hashlib
import html
import itertools
import json
import math
import statistics
from collections import defaultdict
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from sloforge.helix.characterization.analysis.placement import PlacementRecommendation
from sloforge.helix.characterization.gpu_reclamation import (
    PilotValidityEvidence,
    ReclamationMode,
)
from sloforge.helix.characterization.gpu_reclamation_accounting import (
    MemoryDomain,
    StateMovementReport,
    StatePassOperation,
    StatePassRecord,
)
from sloforge.helix.characterization.gpu_reclamation_analysis import (
    AmdahlPoint,
    AmdahlProjection,
    CriticalPath,
    CriticalPathKind,
    Experiment004Outcome,
    FusedChainEvidence,
    HardwareInterestEvidence,
    OutcomeDecision,
    OutcomeEvidence,
    PlacementClass,
    project_amdahl,
    select_outcome,
)
from sloforge.helix.characterization.gpu_reclamation_methodology import (
    ArtifactSampleRef,
    Experiment004GpuHourLedger,
    RawTrial,
    TraceControlGate,
    TrialPlan,
    evaluate_trial_validity,
)
from sloforge.helix.characterization.gpu_reclamation_policy import (
    ReclamationAction,
    ReclamationDecision,
)
from sloforge.helix.characterization.gpu_reclamation_serving import ServingSLO
from sloforge.helix.characterization.matrix import EvidenceClass
from sloforge.helix.characterization.trace.models import (
    TimingMeasurementClass,
    WorkloadProvenance,
)

NonEmpty = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=4096)]
GpuUuid = Annotated[str, StringConstraints(pattern=r"^GPU-[0-9A-Za-z-]+$")]
PlotId = Literal[
    "01-capacity-reclamation-waterfall",
    "02-rollout-restore-waterfall",
    "03-physical-byte-movement-by-stage",
    "04-logical-vs-total-physical-bytes",
    "05-gpu0-serving-over-time",
    "06-gpu-hbm-over-time",
    "07-baseline-comparison",
    "08-amdahl-projection",
    "09-state-movement-sankey",
]


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, validate_default=True)


class ReportEmissionMode(StrEnum):
    FIXTURE_PREVIEW = "fixture_preview"
    FINAL_HARDWARE_REPORT = "final_hardware_report"


class RuntimeEnvironmentSummary(_StrictModel):
    gpu_models: tuple[NonEmpty, NonEmpty]
    gpu_uuids: tuple[GpuUuid, GpuUuid]
    topology_summary: NonEmpty
    cuda_version: NonEmpty
    driver_version: NonEmpty
    vllm_version: Literal["0.23.0"] = "0.23.0"
    torch_version: Literal["2.11.0"] = "2.11.0"
    model: Literal["Qwen/Qwen2.5-7B-Instruct"] = "Qwen/Qwen2.5-7B-Instruct"
    model_revision: Literal["a09a35458c702b33eeacc393d103063234e8bc28"] = (
        "a09a35458c702b33eeacc393d103063234e8bc28"
    )
    p2p_capability: NonEmpty

    @model_validator(mode="after")
    def exact_pair(self) -> Self:
        if len(set(self.gpu_uuids)) != 2:
            raise ValueError("report environment requires two distinct GPU UUIDs")
        if any(
            "A100" not in model or "80GB" not in model.replace(" ", "") for model in self.gpu_models
        ):
            raise ValueError("report environment requires two A100-80GB GPUs")
        return self


class ServingPlotPoint(_StrictModel):
    timestamp_ns: int = Field(ge=0)
    gpu0_ttft_ns: int | None = Field(default=None, ge=0)
    gpu0_throughput_tokens_per_second: float = Field(ge=0.0, allow_inf_nan=False)
    queue_depth: int = Field(ge=0)
    phase: NonEmpty


class HbmPlotPoint(_StrictModel):
    timestamp_ns: int = Field(ge=0)
    gpu0_used_bytes: int = Field(ge=0)
    gpu1_used_bytes: int = Field(ge=0)
    phase: NonEmpty


class SoftwareOptimizationEvidence(_StrictModel):
    description: NonEmpty
    bottleneck_stage: NonEmpty
    naive_trial_ids: tuple[NonEmpty, ...] = Field(min_length=1)
    optimized_trial_ids: tuple[NonEmpty, ...] = Field(min_length=1)
    measured_reclamation_improvement_fraction: float = Field(ge=-10.0, le=1.0, allow_inf_nan=False)
    profile_hardware_backed: bool
    semantics_unchanged: bool
    trace_overhead_gate_passed: bool


class ReportTrialSummary(_StrictModel):
    trial_id: NonEmpty
    seed: int = Field(ge=0, lt=1 << 63)
    mode: ReclamationMode
    evidence_class: EvidenceClass
    raw_provenance: ArtifactSampleRef | None = None
    provenance: ArtifactSampleRef
    gpu_uuids: tuple[GpuUuid, GpuUuid]
    semantic_validation_passed: Literal[True]
    reclamation_path: CriticalPath
    restore_path: CriticalPath
    full_transaction_path: CriticalPath
    slo_restoration_path: CriticalPath
    movement: StateMovementReport | None
    logical_state_bytes: int = Field(gt=0)
    kv_logical_state_bytes: int = Field(gt=0)
    logical_state_accounting_scope: NonEmpty
    host_resident_transport_metadata_bytes: int = Field(gt=0)
    helix_environment_state_bytes: int = Field(ge=0)
    helix_environment_state_scope: NonEmpty
    shared_bytes: int = Field(gt=0)
    private_bytes: int = Field(gt=0)
    physical_state_bytes: int = Field(gt=0)
    physical_block_count: int = Field(gt=0)
    temporary_memory_bytes: int = Field(ge=0)
    time_to_useful_reclaimed_capacity_ns: int = Field(gt=0)
    time_to_serving_slo_restoration_ns: int | None = Field(default=None, gt=0)
    serving_interference_fraction: float = Field(ge=-10.0, le=10.0, allow_inf_nan=False)
    lost_gpu_work_ns: int = Field(ge=0)
    serving_series: tuple[ServingPlotPoint, ...] = Field(min_length=2)
    hbm_series: tuple[HbmPlotPoint, ...] = Field(min_length=2)

    @model_validator(mode="after")
    def complete_trial(self) -> Self:
        if len(set(self.gpu_uuids)) != 2:
            raise ValueError("report trial requires two distinct GPU UUIDs")
        if self.reclamation_path.kind is not CriticalPathKind.RECLAMATION:
            raise ValueError("trial reclamation path has the wrong critical-path kind")
        if self.restore_path.kind is not CriticalPathKind.RESTORE:
            raise ValueError("trial restore path has the wrong critical-path kind")
        if self.full_transaction_path.kind is not CriticalPathKind.FULL_TRANSACTION:
            raise ValueError("trial full-transaction path has the wrong critical-path kind")
        if self.slo_restoration_path.kind is not CriticalPathKind.SLO_RESTORATION:
            raise ValueError("trial SLO-restoration path has the wrong critical-path kind")
        if self.time_to_useful_reclaimed_capacity_ns != self.reclamation_path.duration_ns:
            raise ValueError("useful-capacity scalar and reclamation critical path disagree")
        if (
            self.time_to_serving_slo_restoration_ns is not None
            and self.slo_restoration_path.duration_ns != self.time_to_serving_slo_restoration_ns
        ):
            raise ValueError("SLO restoration scalar and critical path disagree")
        if self.logical_state_bytes != self.shared_bytes + self.private_bytes:
            raise ValueError("report trial shared/private bytes do not conserve logical state")
        if self.logical_state_bytes != self.kv_logical_state_bytes:
            raise ValueError("trial logical denominator must be explicitly scoped to KV state")
        if "KV payload" not in self.logical_state_accounting_scope:
            raise ValueError("trial does not declare its KV-only amplification scope")
        if self.helix_environment_state_bytes != 0 or "model-only" not in (
            self.helix_environment_state_scope
        ):
            raise ValueError("trial hides or mis-scopes required environment/trajectory state")
        if self.mode is ReclamationMode.KILL_AND_RECOMPUTE:
            if self.movement is not None:
                raise ValueError("kill/recompute cannot claim a preservation movement ledger")
            if self.lost_gpu_work_ns == 0:
                raise ValueError("kill/recompute must report lost GPU work")
        else:
            if self.movement is None:
                raise ValueError("preservation trial requires exact movement accounting")
            if self.movement.accounting.logical_state_bytes != self.logical_state_bytes:
                raise ValueError("trial and movement logical bytes disagree")
            if (
                self.movement.accounting.total_temporary_allocation_bytes
                != self.temporary_memory_bytes
            ):
                raise ValueError("trial and movement temporary bytes disagree")
            if self.lost_gpu_work_ns:
                raise ValueError("preservation trial cannot report killed rollout work")
        for name, timestamps in (
            ("serving", tuple(item.timestamp_ns for item in self.serving_series)),
            ("HBM", tuple(item.timestamp_ns for item in self.hbm_series)),
        ):
            if tuple(sorted(timestamps)) != timestamps or len(set(timestamps)) != len(timestamps):
                raise ValueError(f"{name} plot timestamps must be strictly increasing")
        return self


class Experiment004ReportEvidence(_StrictModel):
    schema_version: Literal["sloforge.branchfabric.experiment-004-report-evidence/v1"] = (
        "sloforge.branchfabric.experiment-004-report-evidence/v1"
    )
    report_id: NonEmpty
    evidence_class: EvidenceClass
    environment: RuntimeEnvironmentSummary
    pilot: PilotValidityEvidence
    pilot_provenance: ArtifactSampleRef
    trials: tuple[ReportTrialSummary, ...] = Field(min_length=3, max_length=64)
    outcome_evidence: OutcomeEvidence | None = None
    outcome_provenance: ArtifactSampleRef | None = None
    policy_decision: ReclamationDecision | None = None
    policy_provenance: ArtifactSampleRef | None = None
    placement_recommendation: PlacementRecommendation | None = None
    placement_provenance: ArtifactSampleRef | None = None
    serving_slo: ServingSLO
    gpu_hours: Experiment004GpuHourLedger
    trace_gate: TraceControlGate | None = None
    trace_gate_provenance: ArtifactSampleRef | None = None
    software_optimization: SoftwareOptimizationEvidence
    software_optimization_provenance: ArtifactSampleRef | None = None

    @model_validator(mode="after")
    def complete_evidence(self) -> Self:
        if len({trial.trial_id for trial in self.trials}) != len(self.trials):
            raise ValueError("report evidence contains duplicate trial IDs")
        if {trial.mode for trial in self.trials} != set(ReclamationMode):
            raise ValueError("report requires kill, naive-preserve, and optimized-preserve trials")
        if any(trial.evidence_class is not self.evidence_class for trial in self.trials):
            raise ValueError("trial evidence classes differ from the report evidence class")
        if any(trial.gpu_uuids != self.environment.gpu_uuids for trial in self.trials):
            raise ValueError("report trials were not run on the declared GPU pair")
        if self.pilot.gpu_uuids != self.environment.gpu_uuids:
            raise ValueError("pilot was not run on the declared GPU pair")
        if self.pilot.gpu_models != self.environment.gpu_models:
            raise ValueError("pilot GPU models differ from the report environment")
        pilot_state = (self.pilot.logical_bytes, self.pilot.shared_bytes, self.pilot.private_bytes)
        if any(
            (trial.logical_state_bytes, trial.shared_bytes, trial.private_bytes) != pilot_state
            for trial in self.trials
        ):
            raise ValueError("paired trials do not use the pilot state configuration")
        if (
            len({(trial.physical_state_bytes, trial.physical_block_count) for trial in self.trials})
            != 1
        ):
            raise ValueError("paired trials do not use the same physical state object")
        seed_sets = {
            mode: {trial.seed for trial in self.trials if trial.mode is mode}
            for mode in ReclamationMode
        }
        if len({frozenset(seeds) for seeds in seed_sets.values()}) != 1:
            raise ValueError("report baselines are not paired on the same seeds")
        counts = {
            mode: sum(trial.mode is mode for trial in self.trials) for mode in ReclamationMode
        }
        if self.evidence_class is EvidenceClass.HARDWARE_BACKED_REAL:
            if self.outcome_evidence is None or self.policy_decision is None:
                raise ValueError("final hardware report requires outcome and policy evidence")
            if any(
                item is None
                for item in (
                    self.outcome_provenance,
                    self.policy_provenance,
                    self.software_optimization_provenance,
                    self.trace_gate,
                    self.trace_gate_provenance,
                )
            ):
                raise ValueError("final hardware report requires derived-analysis provenance")
            if self.policy_decision.evidence_class is not EvidenceClass.HARDWARE_BACKED_REAL:
                raise ValueError("final policy decision is not hardware-backed")
            if self.gpu_hours.consumed_additional_gpu_seconds == 0.0:
                raise ValueError("hardware-backed report has no settled GPU-time evidence")
            if self.gpu_hours.reservations:
                raise ValueError("final hardware report cannot retain unsettled GPU reservations")
            if any(trial.time_to_serving_slo_restoration_ns is None for trial in self.trials):
                raise ValueError("hardware-backed report lacks serving SLO restoration timing")
            if not all(
                (
                    self.software_optimization.profile_hardware_backed,
                    self.software_optimization.semantics_unchanged,
                    self.software_optimization.trace_overhead_gate_passed,
                )
            ):
                raise ValueError(
                    "optimized software path lacks measured profile/semantic/trace proof"
                )
            trial_ids = {trial.trial_id for trial in self.trials}
            if not set(self.software_optimization.naive_trial_ids).issubset(trial_ids) or not set(
                self.software_optimization.optimized_trial_ids
            ).issubset(trial_ids):
                raise ValueError("software optimization references unknown raw trials")
            trial_by_id = {trial.trial_id: trial for trial in self.trials}
            if any(
                trial_by_id[trial_id].mode is not ReclamationMode.PRESERVE_NAIVE
                for trial_id in self.software_optimization.naive_trial_ids
            ) or any(
                trial_by_id[trial_id].mode is not ReclamationMode.PRESERVE_OPTIMIZED
                for trial_id in self.software_optimization.optimized_trial_ids
            ):
                raise ValueError("software optimization trial IDs have the wrong modes")
            naive_median = statistics.median(
                trial_by_id[trial_id].reclamation_path.duration_ns
                for trial_id in self.software_optimization.naive_trial_ids
            )
            optimized_median = statistics.median(
                trial_by_id[trial_id].reclamation_path.duration_ns
                for trial_id in self.software_optimization.optimized_trial_ids
            )
            observed_improvement = (naive_median - optimized_median) / naive_median
            if not math.isclose(
                self.software_optimization.measured_reclamation_improvement_fraction,
                observed_improvement,
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                raise ValueError("software optimization improvement differs from raw trials")
            if (
                len(self.software_optimization.naive_trial_ids)
                != counts[ReclamationMode.PRESERVE_NAIVE]
                or len(self.software_optimization.optimized_trial_ids)
                != counts[ReclamationMode.PRESERVE_OPTIMIZED]
            ):
                raise ValueError("software optimization must reference every paired comparator")
            if {
                trial_by_id[trial_id].seed
                for trial_id in self.software_optimization.naive_trial_ids
            } != {
                trial_by_id[trial_id].seed
                for trial_id in self.software_optimization.optimized_trial_ids
            }:
                raise ValueError("software optimization trial references are not seed-paired")
            expected = {
                ReclamationMode.KILL_AND_RECOMPUTE: self.outcome_evidence.kill_trials,
                ReclamationMode.PRESERVE_NAIVE: self.outcome_evidence.naive_trials,
                ReclamationMode.PRESERVE_OPTIMIZED: self.outcome_evidence.optimized_trials,
            }
            if any(counts[mode] != expected[mode] for mode in ReclamationMode):
                raise ValueError("outcome evidence trial counts differ from the report")
            decision = select_outcome(self.outcome_evidence)
            placement_required = decision.outcome in {
                Experiment004Outcome.HOST_PIPELINE_HARDWARE_INTEREST,
                Experiment004Outcome.FABRIC_HARDWARE_INTEREST,
            }
            if placement_required != (
                self.placement_recommendation is not None and self.placement_provenance is not None
            ):
                raise ValueError(
                    "hardware-interest outcomes require exactly one placement recommendation"
                )
        elif any(
            item is not None
            for item in (
                self.outcome_evidence,
                self.outcome_provenance,
                self.policy_decision,
                self.policy_provenance,
                self.placement_recommendation,
                self.placement_provenance,
                self.software_optimization_provenance,
                self.trace_gate,
                self.trace_gate_provenance,
            )
        ):
            raise ValueError("non-hardware report evidence cannot carry a final decision")
        return self


class PlotPayload(_StrictModel):
    schema_version: Literal["sloforge.branchfabric.experiment-004-plot/v1"] = (
        "sloforge.branchfabric.experiment-004-plot/v1"
    )
    plot_id: PlotId
    title: NonEmpty
    kind: Literal["waterfall", "bar", "line", "amdahl", "sankey-equivalent"]
    source_trial_ids: tuple[NonEmpty, ...]
    data: dict[str, Any]


class ReportPublication(_StrictModel):
    emission_mode: ReportEmissionMode
    publishable_hardware_report: bool
    report_json: str
    report_markdown: str
    decision_document: str | None
    characterization_document: str | None
    experiment_005_plan: str | None
    artifact_manifest: str | None
    plot_json: tuple[str, ...] = Field(min_length=9, max_length=9)
    plot_svg: tuple[str, ...] = Field(min_length=9, max_length=9)
    outcome: Experiment004Outcome | None


_CAPTURE_CHAIN_ID = "optimized-native-to-host"
_RESTORE_CHAIN_ID = "optimized-host-to-native"
_GPU_INTERMEDIATE_DOMAINS = frozenset(
    {MemoryDomain.GPU_TRANSFORM_BUFFER, MemoryDomain.GPU_TRANSPORT_BUFFER}
)
_HOST_INTERMEDIATE_DOMAINS = frozenset(
    {MemoryDomain.PINNED_HOST_TRANSPORT, MemoryDomain.PAGEABLE_HOST_BUFFER}
)


def _union_duration_ns(intervals: tuple[tuple[int, int], ...]) -> int:
    ordered = sorted((start, end) for start, end in intervals if end > start)
    if not ordered:
        return 0
    total = 0
    start, end = ordered[0]
    for next_start, next_end in ordered[1:]:
        if next_start <= end:
            end = max(end, next_end)
        else:
            total += end - start
            start, end = next_start, next_end
    return total + end - start


def _intersection_duration_ns(records: tuple[StatePassRecord, ...], path: CriticalPath) -> int:
    start = path.intervals[0].start_ns
    end = path.intervals[-1].end_ns
    return _union_duration_ns(
        tuple((max(item.start_ns, start), min(item.end_ns, end)) for item in records)
    )


def _logical_coverage_bytes(records: tuple[StatePassRecord, ...]) -> int:
    ranges: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for item in records:
        ranges[item.state_segment].append(
            (item.logical_offset_bytes, item.logical_offset_bytes + item.logical_bytes)
        )
    return sum(_union_duration_ns(tuple(items)) for items in ranges.values())


def _operational_records(movement: StateMovementReport) -> tuple[StatePassRecord, ...]:
    return tuple(
        item
        for item in movement.passes
        if any((item.bytes_read, item.bytes_written, item.transfer_bytes, item.checksum_bytes))
    )


def _capture_chain_records(movement: StateMovementReport) -> tuple[StatePassRecord, ...]:
    operational = _operational_records(movement)
    h2d_starts = [item.start_ns for item in operational if item.operation is StatePassOperation.H2D]
    if not h2d_starts:
        raise ValueError("optimized movement has no measured H2D restore boundary")
    restore_start = min(h2d_starts)
    candidates = tuple(item for item in operational if item.start_ns < restore_start)
    selected = tuple(
        item
        for item in candidates
        if (
            item.operation is StatePassOperation.FUSED_NATIVE_TO_HOST
            or (
                item.operation is StatePassOperation.READ
                and item.source_memory is MemoryDomain.SOURCE_GPU_NATIVE_PAGED
            )
            or item.operation
            in {
                StatePassOperation.UNPAGE,
                StatePassOperation.REPACK,
                StatePassOperation.TRANSFORM,
                StatePassOperation.RESHAPE,
                StatePassOperation.CHECKSUM,
            }
            or (
                item.operation is StatePassOperation.D2H
                and item.destination_memory is MemoryDomain.PINNED_HOST_TRANSPORT
            )
        )
    )
    operations = {item.operation for item in selected}
    if not (
        StatePassOperation.FUSED_NATIVE_TO_HOST in operations
        or (
            StatePassOperation.READ in operations
            and StatePassOperation.D2H in operations
            and StatePassOperation.CHECKSUM in operations
            and operations
            & {
                StatePassOperation.UNPAGE,
                StatePassOperation.REPACK,
                StatePassOperation.TRANSFORM,
                StatePassOperation.RESHAPE,
            }
        )
    ):
        raise ValueError("optimized movement lacks a complete native-to-host fused-chain candidate")
    return selected


def _restore_chain_records(movement: StateMovementReport) -> tuple[StatePassRecord, ...]:
    operational = _operational_records(movement)
    h2d = tuple(item for item in operational if item.operation is StatePassOperation.H2D)
    if not h2d:
        raise ValueError("optimized movement has no measured H2D restore")
    restore_start = min(item.start_ns for item in h2d)
    candidates = tuple(item for item in operational if item.start_ns >= restore_start)
    conversion_keys = {
        (item.start_ns, item.end_ns)
        for item in candidates
        if item.operation
        in {
            StatePassOperation.UNPACK,
            StatePassOperation.REPAGE,
            StatePassOperation.FUSED_HOST_TO_NATIVE,
        }
    }
    selected = tuple(
        item
        for item in candidates
        if (
            item.operation
            in {
                StatePassOperation.H2D,
                StatePassOperation.UNPACK,
                StatePassOperation.REPAGE,
                StatePassOperation.WRITE,
                StatePassOperation.FUSED_HOST_TO_NATIVE,
            }
            or (
                item.operation is StatePassOperation.ALLOCATE
                and item.destination_memory is MemoryDomain.GPU_TRANSFORM_BUFFER
                and item.bytes_written > 0
            )
            or (
                item.operation in {StatePassOperation.REPACK, StatePassOperation.RESHAPE}
                and (item.start_ns, item.end_ns) in conversion_keys
            )
        )
    )
    operations = {item.operation for item in selected}
    if not (
        StatePassOperation.FUSED_HOST_TO_NATIVE in operations
        or (
            StatePassOperation.H2D in operations
            and StatePassOperation.WRITE in operations
            and operations & {StatePassOperation.UNPACK, StatePassOperation.REPAGE}
        )
    ):
        raise ValueError("optimized movement lacks a complete host-to-native fused-chain candidate")
    return selected


def _distinct_operations_overlap(records: tuple[StatePassRecord, ...]) -> bool:
    events = {
        (
            item.operation,
            item.start_ns,
            item.end_ns,
            item.source_memory,
            item.destination_memory,
        )
        for item in records
    }
    ordered = tuple(events)
    if any(
        item[0]
        in {StatePassOperation.FUSED_NATIVE_TO_HOST, StatePassOperation.FUSED_HOST_TO_NATIVE}
        for item in ordered
    ):
        return True
    for index, left in enumerate(ordered):
        for right in ordered[index + 1 :]:
            if (
                left[0] is not right[0]
                and (left[1], left[2]) != (right[1], right[2])
                and left[1] < right[2]
                and right[1] < left[2]
            ):
                return True
    return False


def _chain_cpu_gpu_time(
    records: tuple[StatePassRecord, ...], path: CriticalPath
) -> tuple[int | None, int | None]:
    overlapping = tuple(
        interval
        for interval in path.intervals
        if any(
            record.start_ns < interval.end_ns and interval.start_ns < record.end_ns
            for record in records
        )
    )
    gpu_time = (
        sum(item.gpu_time_ns or 0 for item in overlapping)
        if overlapping and all(item.gpu_time_ns is not None for item in overlapping)
        else None
    )
    cpu_time = (
        sum(item.cpu_time_ns or 0 for item in overlapping)
        if overlapping and all(item.cpu_time_ns is not None for item in overlapping)
        else None
    )
    return gpu_time, cpu_time


def _dependencies_permit_streaming(records: tuple[StatePassRecord, ...]) -> bool:
    by_segment: dict[str, list[StatePassRecord]] = defaultdict(list)
    for item in records:
        by_segment[item.state_segment].append(item)
    if not by_segment:
        return False
    shapes: set[tuple[StatePassOperation, ...]] = set()
    for segment_records in by_segment.values():
        ordered = sorted(
            segment_records, key=lambda item: (item.start_ns, item.end_ns, item.record_id)
        )
        if any(left.end_ns > right.start_ns for left, right in itertools.pairwise(ordered)):
            return False
        shapes.add(tuple(item.operation for item in ordered))
    return len(shapes) == 1


def _derived_chain_gate(
    *,
    chain_id: str,
    optimized_trials: tuple[ReportTrialSummary, ...],
) -> HardwareInterestEvidence:
    capture = chain_id == _CAPTURE_CHAIN_ID
    records_by_trial: list[tuple[StatePassRecord, ...]] = []
    total_physical_bytes = 0
    total_state_passes = 0
    total_temporary_bytes = 0
    total_materialized_bytes = 0
    total_chain_wall_ns = 0
    total_movement_wall_ns = 0
    total_avoidable_bytes = 0
    total_movement_bytes = 0
    reclamation_overlap_ns = 0
    restore_overlap_ns = 0
    transaction_overlap_ns = 0
    slo_overlap_ns = 0
    total_reclamation_ns = 0
    total_restore_ns = 0
    total_transaction_ns = 0
    total_slo_ns = 0
    gpu_times: list[int] = []
    cpu_times: list[int] = []
    all_gpu_times = True
    all_cpu_times = True
    concurrency = False
    for trial in optimized_trials:
        assert trial.movement is not None
        records = (
            _capture_chain_records(trial.movement)
            if capture
            else _restore_chain_records(trial.movement)
        )
        records_by_trial.append(records)
        path = trial.reclamation_path if capture else trial.restore_path
        if any(
            item.start_ns < path.intervals[0].start_ns or item.end_ns > path.intervals[-1].end_ns
            for item in records
        ):
            raise ValueError(f"{chain_id} state passes escape their non-overlapping critical path")
        wall = _union_duration_ns(tuple((item.start_ns, item.end_ns) for item in records))
        if wall <= 0:
            raise ValueError(f"{chain_id} has no positive measured wall time")
        total_chain_wall_ns += wall
        operational = _operational_records(trial.movement)
        total_movement_wall_ns += _union_duration_ns(
            tuple((item.start_ns, item.end_ns) for item in operational)
        )
        total_physical_bytes += sum(
            item.bytes_read + item.bytes_written + item.transfer_bytes for item in records
        )
        total_state_passes += len(records)
        total_temporary_bytes += sum(item.temporary_allocation_bytes for item in records)
        total_materialized_bytes += sum(
            item.bytes_written
            for item in records
            if item.destination_memory in _GPU_INTERMEDIATE_DOMAINS | _HOST_INTERMEDIATE_DOMAINS
        )
        total_avoidable_bytes += sum(
            item.bytes_read + item.bytes_written + item.transfer_bytes
            for item in records
            if not item.required_unavoidable
        )
        total_movement_bytes += trial.movement.accounting.amplification_numerator_bytes
        reclamation_overlap_ns += _intersection_duration_ns(records, trial.reclamation_path)
        restore_overlap_ns += _intersection_duration_ns(records, trial.restore_path)
        transaction_overlap_ns += _intersection_duration_ns(records, trial.full_transaction_path)
        slo_overlap_ns += _intersection_duration_ns(records, trial.slo_restoration_path)
        total_reclamation_ns += trial.reclamation_path.duration_ns
        total_restore_ns += trial.restore_path.duration_ns
        total_transaction_ns += trial.full_transaction_path.duration_ns
        total_slo_ns += trial.slo_restoration_path.duration_ns
        gpu_time, cpu_time = _chain_cpu_gpu_time(records, path)
        if gpu_time is None:
            all_gpu_times = False
        else:
            gpu_times.append(gpu_time)
        if cpu_time is None:
            all_cpu_times = False
        else:
            cpu_times.append(cpu_time)
        concurrency = concurrency or _distinct_operations_overlap(records)
    logical_bytes = sum(_logical_coverage_bytes(records) for records in records_by_trial)
    expected_logical_bytes = sum(trial.logical_state_bytes for trial in optimized_trials)
    if logical_bytes != expected_logical_bytes:
        raise ValueError(f"{chain_id} logical coverage differs from the optimized state payload")
    operations = (
        ("read", "unpage/repack/transform", "d2h", "checksum")
        if capture
        else ("h2d", "unpack/repack/repage", "write")
    )
    full_fraction = transaction_overlap_ns / total_transaction_ns
    ideal_projected_ns = total_transaction_ns - transaction_overlap_ns
    chain = FusedChainEvidence(
        chain_id=chain_id,
        operations=operations,
        occurrence_count=len(optimized_trials),
        logical_bytes=logical_bytes,
        physical_bytes=total_physical_bytes,
        state_passes=total_state_passes,
        wall_time_ns=total_chain_wall_ns,
        gpu_time_ns=sum(gpu_times) if all_gpu_times else None,
        cpu_time_ns=sum(cpu_times) if all_cpu_times else None,
        temporary_bytes=total_temporary_bytes,
        dependencies_permit_streaming=all(
            _dependencies_permit_streaming(records) for records in records_by_trial
        ),
        materialized_intermediate_bytes=total_materialized_bytes,
        placement_class=PlacementClass.HOST,
    )
    return HardwareInterestEvidence(
        chain=chain,
        fraction_of_reclamation=reclamation_overlap_ns / total_reclamation_ns,
        fraction_of_resume=restore_overlap_ns / total_restore_ns,
        fraction_of_full_transaction=full_fraction,
        fraction_of_slo_restoration=slo_overlap_ns / total_slo_ns,
        fraction_of_movement_time=total_chain_wall_ns / total_movement_wall_ns,
        serving_degradation_fraction=0.0,
        avoidable_physical_byte_fraction=(
            total_avoidable_bytes / total_movement_bytes if total_movement_bytes else 0.0
        ),
        ideal_free_end_to_end_speedup=(
            total_transaction_ns / ideal_projected_ns if ideal_projected_ns else float("inf")
        ),
        # No independent hardware model has been measured in Experiment 004;
        # realistic hardware speedup therefore remains the fail-closed 1.0.
        realistic_end_to_end_speedup=1.0,
        regular_dataflow=(
            len({tuple(item.operation for item in records) for records in records_by_trial}) == 1
        ),
        measured_byte_rate=total_physical_bytes > 0 and total_chain_wall_ns > 0,
        measured_concurrency=concurrency,
        measured_latency_target=total_chain_wall_ns > 0,
        plausible_off_critical_path_placement=True,
    )


def derive_software_optimization_evidence(
    evidence: Experiment004ReportEvidence,
) -> SoftwareOptimizationEvidence:
    """Derive the software comparator from every exact paired trial cell."""

    naive = tuple(
        sorted(
            (item for item in evidence.trials if item.mode is ReclamationMode.PRESERVE_NAIVE),
            key=lambda item: (item.seed, item.trial_id),
        )
    )
    optimized = tuple(
        sorted(
            (item for item in evidence.trials if item.mode is ReclamationMode.PRESERVE_OPTIMIZED),
            key=lambda item: (item.seed, item.trial_id),
        )
    )
    if len(naive) != 3 or len(optimized) != 3:
        raise ValueError(
            "final software comparison requires exactly three naive and optimized trials"
        )
    if tuple(item.seed for item in naive) != tuple(item.seed for item in optimized):
        raise ValueError("naive and optimized software trials are not seed-paired")
    naive_median = statistics.median(item.reclamation_path.duration_ns for item in naive)
    optimized_median = statistics.median(item.reclamation_path.duration_ns for item in optimized)
    if naive_median <= 0:
        raise ValueError("naive reclamation median must be positive")
    operation_wall: dict[str, int] = defaultdict(int)
    for trial in naive:
        assert trial.movement is not None
        by_operation: dict[StatePassOperation, list[tuple[int, int]]] = defaultdict(list)
        for item in _operational_records(trial.movement):
            by_operation[item.operation].append((item.start_ns, item.end_ns))
        for operation, intervals in by_operation.items():
            operation_wall[operation.value] += _union_duration_ns(tuple(intervals))
    if not operation_wall:
        raise ValueError("naive trials contain no measured movement operations")
    bottleneck = max(operation_wall, key=lambda name: (operation_wall[name], name))
    trace_passed = bool(
        evidence.trace_gate is not None and evidence.trace_gate.minimal_trace_causal_trials_allowed
    )
    return SoftwareOptimizationEvidence(
        description=(
            "Measured optimized Continuum pipeline compared against all seed-paired naive "
            "preservation trials"
        ),
        bottleneck_stage=bottleneck,
        naive_trial_ids=tuple(item.trial_id for item in naive),
        optimized_trial_ids=tuple(item.trial_id for item in optimized),
        measured_reclamation_improvement_fraction=(naive_median - optimized_median) / naive_median,
        profile_hardware_backed=(
            evidence.evidence_class is EvidenceClass.HARDWARE_BACKED_REAL
            and all(item.evidence_class is EvidenceClass.HARDWARE_BACKED_REAL for item in naive)
        ),
        semantics_unchanged=all(item.semantic_validation_passed for item in (*naive, *optimized)),
        trace_overhead_gate_passed=trace_passed,
    )


def derive_outcome_evidence(evidence: Experiment004ReportEvidence) -> OutcomeEvidence:
    """Derive every outcome input and fused-chain metric from accepted raw trials."""

    by_mode = {
        mode: tuple(
            sorted(
                (item for item in evidence.trials if item.mode is mode),
                key=lambda item: (item.seed, item.trial_id),
            )
        )
        for mode in ReclamationMode
    }
    if any(len(items) != 3 for items in by_mode.values()):
        raise ValueError("final outcome requires exactly three trials for every mode")
    seed_sets = {tuple(item.seed for item in items) for items in by_mode.values()}
    if len(seed_sets) != 1:
        raise ValueError("final outcome trials are not exactly seed-paired")
    optimized = by_mode[ReclamationMode.PRESERVE_OPTIMIZED]
    naive = by_mode[ReclamationMode.PRESERVE_NAIVE]
    kill = by_mode[ReclamationMode.KILL_AND_RECOMPUTE]
    optimization = derive_software_optimization_evidence(evidence)
    optimized_full = statistics.median(item.full_transaction_path.duration_ns for item in optimized)
    kill_full = statistics.median(item.full_transaction_path.duration_ns for item in kill)
    operational_wall = 0
    full_wall = sum(item.full_transaction_path.duration_ns for item in optimized)
    for trial in optimized:
        assert trial.movement is not None
        operational_wall += _union_duration_ns(
            tuple((item.start_ns, item.end_ns) for item in _operational_records(trial.movement))
        )
    if operational_wall > full_wall:
        raise ValueError("optimized movement wall time exceeds the full transaction critical path")
    if not optimization.profile_hardware_backed:
        raise ValueError("outcome derivation requires hardware-backed software profiling")
    if not optimization.trace_overhead_gate_passed:
        raise ValueError("outcome derivation requires a passing tracing-overhead control")
    if not optimization.semantics_unchanged:
        raise ValueError("outcome derivation requires optimized/naive semantic equivalence")
    chain_gates = tuple(
        _derived_chain_gate(chain_id=chain_id, optimized_trials=optimized)
        for chain_id in (_CAPTURE_CHAIN_ID, _RESTORE_CHAIN_ID)
    )
    return OutcomeEvidence(
        valid_pilot=True,
        kill_trials=len(kill),
        naive_trials=len(naive),
        optimized_trials=len(optimized),
        optimized_semantics_valid=True,
        preservation_economic_for_measured_workload=optimized_full <= kill_full,
        optimized_removed_most_naive_headroom=(
            optimization.measured_reclamation_improvement_fraction >= 0.5
        ),
        profiling_hardware_backed=True,
        trace_overhead_gate_passed=True,
        optimized_path_measured_after_naive=True,
        optimized_path_semantics_match_naive=True,
        optimized_movement_fraction=operational_wall / full_wall,
        chain_gates=chain_gates,
    )


def _canonical_bytes(value: object) -> bytes:
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n"
    ).encode("utf-8")


def _representative(
    evidence: Experiment004ReportEvidence, mode: ReclamationMode
) -> ReportTrialSummary:
    candidates = sorted(
        (trial for trial in evidence.trials if trial.mode is mode),
        key=lambda trial: (trial.reclamation_path.duration_ns, trial.seed, trial.trial_id),
    )
    return candidates[len(candidates) // 2]


def _waterfall_payload(
    *, plot_id: PlotId, title: str, trial: ReportTrialSummary, path: CriticalPath
) -> PlotPayload:
    return PlotPayload(
        plot_id=plot_id,
        title=title,
        kind="waterfall",
        source_trial_ids=(trial.trial_id,),
        data={
            "total_ns": path.duration_ns,
            "stages": [
                {
                    "name": item.name,
                    "start_offset_ns": item.start_ns - path.intervals[0].start_ns,
                    "duration_ns": item.duration_ns,
                }
                for item in path.intervals
            ],
        },
    )


def build_plot_payloads(
    evidence: Experiment004ReportEvidence,
) -> tuple[PlotPayload, ...]:
    """Build all nine plots strictly from validated trial evidence."""

    optimized = _representative(evidence, ReclamationMode.PRESERVE_OPTIMIZED)
    assert optimized.movement is not None
    plots: list[PlotPayload] = [
        _waterfall_payload(
            plot_id="01-capacity-reclamation-waterfall",
            title="Capacity reclamation waterfall",
            trial=optimized,
            path=optimized.reclamation_path,
        ),
        _waterfall_payload(
            plot_id="02-rollout-restore-waterfall",
            title="Rollout restore waterfall",
            trial=optimized,
            path=optimized.restore_path,
        ),
    ]

    by_operation: dict[str, int] = defaultdict(int)
    for item in optimized.movement.passes:
        by_operation[item.operation.value] += (
            item.bytes_read + item.bytes_written + item.transfer_bytes
        )
    plots.append(
        PlotPayload(
            plot_id="03-physical-byte-movement-by-stage",
            title="Physical byte movement by stage",
            kind="bar",
            source_trial_ids=(optimized.trial_id,),
            data={
                "bars": [{"label": key, "value": by_operation[key]} for key in sorted(by_operation)]
            },
        )
    )
    movement_trials = tuple(
        sorted(
            (trial for trial in evidence.trials if trial.movement is not None),
            key=lambda trial: trial.trial_id,
        )
    )
    plots.append(
        PlotPayload(
            plot_id="04-logical-vs-total-physical-bytes",
            title="Logical bytes versus total physical bytes touched",
            kind="bar",
            source_trial_ids=tuple(trial.trial_id for trial in movement_trials),
            data={
                "groups": [
                    {
                        "label": trial.trial_id,
                        "logical_bytes": trial.logical_state_bytes,
                        "physical_bytes_touched": trial.movement.accounting.amplification_numerator_bytes,
                    }
                    for trial in movement_trials
                    if trial.movement is not None
                ]
            },
        )
    )
    plots.append(
        PlotPayload(
            plot_id="05-gpu0-serving-over-time",
            title="GPU0 serving TTFT and throughput over time",
            kind="line",
            source_trial_ids=(optimized.trial_id,),
            data={"points": [item.model_dump(mode="json") for item in optimized.serving_series]},
        )
    )
    plots.append(
        PlotPayload(
            plot_id="06-gpu-hbm-over-time",
            title="GPU0 and GPU1 HBM over time",
            kind="line",
            source_trial_ids=(optimized.trial_id,),
            data={"points": [item.model_dump(mode="json") for item in optimized.hbm_series]},
        )
    )
    baseline_rows = []
    for mode in ReclamationMode:
        rows = tuple(trial for trial in evidence.trials if trial.mode is mode)
        baseline_rows.append(
            {
                "mode": mode.value,
                "median_reclamation_ns": int(
                    statistics.median(trial.reclamation_path.duration_ns for trial in rows)
                ),
                "median_resume_ns": int(
                    statistics.median(trial.restore_path.duration_ns for trial in rows)
                ),
                "median_lost_gpu_work_ns": int(
                    statistics.median(trial.lost_gpu_work_ns for trial in rows)
                ),
            }
        )
    plots.append(
        PlotPayload(
            plot_id="07-baseline-comparison",
            title="Kill/recompute versus naive and optimized preservation",
            kind="bar",
            source_trial_ids=tuple(sorted(trial.trial_id for trial in evidence.trials)),
            data={"groups": baseline_rows},
        )
    )
    amdahl: list[dict[str, object]] = []
    for path in (
        optimized.reclamation_path,
        optimized.restore_path,
        optimized.full_transaction_path,
        optimized.slo_restoration_path,
    ):
        for interval in path.intervals:
            projection = project_amdahl(path, target_names=(interval.name,))
            amdahl.append(projection.model_dump(mode="json"))
    if evidence.outcome_evidence is not None:
        for gate in evidence.outcome_evidence.chain_gates:
            for path, fraction in (
                (optimized.reclamation_path, gate.fraction_of_reclamation),
                (optimized.restore_path, gate.fraction_of_resume),
                (optimized.full_transaction_path, gate.fraction_of_full_transaction),
                (optimized.slo_restoration_path, gate.fraction_of_slo_restoration),
            ):
                if fraction == 0.0:
                    continue
                amdahl.append(
                    _project_fraction(
                        path,
                        target_name=f"fused:{gate.chain.chain_id}",
                        fraction=fraction,
                    ).model_dump(mode="json")
                )
    plots.append(
        PlotPayload(
            plot_id="08-amdahl-projection",
            title="Amdahl projection for major state stages",
            kind="amdahl",
            source_trial_ids=(optimized.trial_id,),
            data={"projections": amdahl},
        )
    )
    plots.append(
        PlotPayload(
            plot_id="09-state-movement-sankey",
            title="State movement Sankey-equivalent edge graph",
            kind="sankey-equivalent",
            source_trial_ids=(optimized.trial_id,),
            data={"edges": [item.model_dump(mode="json") for item in optimized.movement.edges]},
        )
    )
    return tuple(plots)


def _project_fraction(path: CriticalPath, *, target_name: str, fraction: float) -> AmdahlProjection:
    """Project one measured fused-chain fraction without double-counting stages."""

    target_ns = round(path.duration_ns * fraction)
    points: list[AmdahlPoint] = []
    accelerations: tuple[tuple[Literal["2x", "5x", "10x", "free"], float | None], ...] = (
        ("2x", 2.0),
        ("5x", 5.0),
        ("10x", 10.0),
        ("free", None),
    )
    for label, factor in accelerations:
        accelerated = 0 if factor is None else round(target_ns / factor)
        projected = path.duration_ns - target_ns + accelerated
        points.append(
            AmdahlPoint(
                acceleration=label,
                projected_total_ns=projected,
                projected_speedup=path.duration_ns / projected if projected else None,
                projected_reduction_fraction=(path.duration_ns - projected) / path.duration_ns,
            )
        )
    return AmdahlProjection(
        path_kind=path.kind,
        target_names=(target_name,),
        baseline_total_ns=path.duration_ns,
        target_total_ns=target_ns,
        target_fraction=target_ns / path.duration_ns,
        points=tuple(points),
    )


def _bars_from_plot(plot: PlotPayload) -> list[tuple[str, float]]:
    if "bars" in plot.data:
        return [(str(item["label"]), float(item["value"])) for item in plot.data["bars"]]
    if plot.plot_id == "04-logical-vs-total-physical-bytes":
        return [
            (f"{item['label']}:logical", float(item["logical_bytes"]))
            for item in plot.data["groups"]
        ] + [
            (f"{item['label']}:physical", float(item["physical_bytes_touched"]))
            for item in plot.data["groups"]
        ]
    if plot.plot_id == "07-baseline-comparison":
        return (
            [
                (f"{item['mode']}:reclaim", float(item["median_reclamation_ns"]))
                for item in plot.data["groups"]
            ]
            + [
                (f"{item['mode']}:resume", float(item["median_resume_ns"]))
                for item in plot.data["groups"]
            ]
            + [
                (f"{item['mode']}:lost-work", float(item["median_lost_gpu_work_ns"]))
                for item in plot.data["groups"]
            ]
        )
    return []


def _svg(plot: PlotPayload, *, fixture: bool) -> bytes:
    width, height = 1000, 500
    title = html.escape(plot.title)
    watermark = "FIXTURE PREVIEW — NOT HARDWARE EVIDENCE" if fixture else "HARDWARE-BACKED"
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="30" y="35" font-family="sans-serif" font-size="22">{title}</text>',
        f'<text x="30" y="62" font-family="sans-serif" font-size="13" fill="#8b0000">{watermark}</text>',
    ]
    bars = _bars_from_plot(plot)
    if plot.kind == "waterfall":
        bars = [(str(item["name"]), float(item["duration_ns"])) for item in plot.data["stages"]]
    if plot.kind == "amdahl":
        bars = [
            (
                f"{item['path_kind']}:{item['target_names'][0]}",
                float(item["points"][-1]["projected_reduction_fraction"]),
            )
            for item in plot.data["projections"]
        ]
    if plot.kind == "sankey-equivalent":
        bars = [
            (f"{item['source']}→{item['destination']}", float(item["bytes"]))
            for item in plot.data["edges"][:20]
        ]
    if bars:
        maximum = max((value for _, value in bars), default=1.0) or 1.0
        bar_height = min(28.0, 360.0 / len(bars))
        for index, (label, value) in enumerate(bars):
            y = 85 + index * bar_height
            bar_width = 650 * value / maximum
            elements.append(
                f'<rect x="300" y="{y:.1f}" width="{bar_width:.2f}" height="{max(2.0, bar_height - 4):.1f}" fill="#356aa0"/>'
            )
            elements.append(
                f'<text x="295" y="{y + bar_height - 8:.1f}" text-anchor="end" font-family="monospace" font-size="10">{html.escape(label[:80])}</text>'
            )
    else:
        points = plot.data.get("points", [])
        if points:
            timestamps = [int(item["timestamp_ns"]) for item in points]
            start, end = min(timestamps), max(timestamps)
            span = max(1, end - start)
            series = (
                (
                    "throughput",
                    "gpu0_throughput_tokens_per_second",
                    "#356aa0",
                ),
                ("TTFT", "gpu0_ttft_ns", "#b24a3b"),
            )
            if plot.plot_id.startswith("06"):
                series = (
                    ("GPU0 HBM", "gpu0_used_bytes", "#356aa0"),
                    ("GPU1 HBM", "gpu1_used_bytes", "#b24a3b"),
                )
            for series_index, (label, key, color) in enumerate(series):
                samples = tuple(
                    (timestamp, float(item[key]))
                    for timestamp, item in zip(timestamps, points, strict=True)
                    if item[key] is not None
                )
                if not samples:
                    continue
                maximum = max(value for _, value in samples) or 1.0
                coordinates = " ".join(
                    f"{50 + 900 * (timestamp - start) / span:.2f},{450 - 340 * value / maximum:.2f}"
                    for timestamp, value in samples
                )
                elements.append(
                    f'<polyline points="{coordinates}" fill="none" stroke="{color}" stroke-width="3"/>'
                )
                elements.append(
                    f'<text x="{760 + series_index * 110}" y="62" font-family="sans-serif" '
                    f'font-size="11" fill="{color}">{html.escape(label)}</text>'
                )
            for point in points:
                x = 50 + 900 * (int(point["timestamp_ns"]) - start) / span
                elements.append(
                    f'<line x1="{x:.2f}" y1="82" x2="{x:.2f}" y2="455" '
                    'stroke="#bbbbbb" stroke-width="1" stroke-dasharray="2,3"/>'
                )
                elements.append(
                    f'<text x="{x + 2:.2f}" y="96" font-family="sans-serif" '
                    f'font-size="9">{html.escape(str(point["phase"]))}</text>'
                )
    elements.append("</svg>\n")
    return "".join(elements).encode("utf-8")


def _validate_real_provenance(evidence: Experiment004ReportEvidence) -> None:
    def validate_file(reference: ArtifactSampleRef) -> Path:
        path = Path(reference.artifact_reference)
        if not path.is_file():
            raise FileNotFoundError(f"hardware report provenance is absent: {path}")
        if hashlib.sha256(path.read_bytes()).hexdigest() != reference.artifact_sha256:
            raise ValueError(f"hardware report provenance hash mismatch: {path}")
        return path

    def selected(reference: ArtifactSampleRef, *, selector: str) -> object:
        if reference.sample_selector != selector:
            raise ValueError(f"hardware provenance selector must be exactly {selector}")
        path = validate_file(reference)
        payload = json.loads(path.read_text())
        if not isinstance(payload, dict):
            raise ValueError("hardware provenance root must be a JSON object")
        key = selector.removeprefix("$.")
        if key not in payload:
            raise ValueError(f"hardware provenance selector {selector} is absent")
        return payload[key]

    def trial_projection(trial: ReportTrialSummary) -> dict[str, Any]:
        return trial.model_dump(mode="json", exclude={"provenance", "raw_provenance"})

    def require_raw_metric(raw_trial: RawTrial, trial: ReportTrialSummary) -> None:
        required = {
            ("reclamation_interruption", "nanoseconds"): trial.reclamation_path.duration_ns,
            ("rollout_resume_latency", "nanoseconds"): trial.restore_path.duration_ns,
            ("full_preservation_transaction", "nanoseconds"): (
                trial.full_transaction_path.duration_ns
            ),
            ("time_to_serving_slo_restoration", "nanoseconds"): (
                trial.slo_restoration_path.duration_ns
            ),
            ("logical_state_bytes", "bytes"): trial.logical_state_bytes,
            ("physical_state_bytes", "bytes"): trial.physical_state_bytes,
            ("temporary_memory_bytes", "bytes"): trial.temporary_memory_bytes,
            ("serving_interference_fraction", "fraction"): (trial.serving_interference_fraction),
            ("lost_gpu_work", "nanoseconds"): trial.lost_gpu_work_ns,
        }
        observed = {(item.metric, item.unit): item.value for item in raw_trial.metrics}
        if trial.movement is not None:
            accounting = trial.movement.accounting
            required.update(
                {
                    ("movement_physical_bytes_read", "bytes"): accounting.physical_bytes_read,
                    ("movement_physical_bytes_written", "bytes"): (
                        accounting.physical_bytes_written
                    ),
                    ("movement_d2h_bytes", "bytes"): accounting.d2h_bytes,
                    ("movement_h2d_bytes", "bytes"): accounting.h2d_bytes,
                    ("movement_host_intermediate_bytes", "bytes"): (
                        accounting.host_intermediate_bytes
                    ),
                    ("movement_gpu_intermediate_bytes", "bytes"): (
                        accounting.gpu_intermediate_bytes
                    ),
                    ("movement_checksum_bytes", "bytes"): accounting.checksum_bytes,
                    ("movement_pass_count", "count"): accounting.pass_count,
                }
            )
        missing = set(required) - set(observed)
        if missing:
            raise ValueError(
                f"trial {trial.trial_id} raw metrics omit report inputs: {sorted(missing)}"
            )
        for key, expected in required.items():
            if not math.isclose(observed[key], float(expected), rel_tol=0.0, abs_tol=1e-9):
                raise ValueError(f"trial {trial.trial_id} summary differs from raw metric {key}")

    pilot = PilotValidityEvidence.model_validate(
        selected(evidence.pilot_provenance, selector="$.pilot_validity"), strict=True
    )
    if pilot != evidence.pilot:
        raise ValueError("pilot summary differs from its hash-bound raw analysis")
    settled_by_manifest = {
        (
            interval.raw_manifest.artifact_reference,
            interval.raw_manifest.artifact_sha256,
        ): interval
        for interval in evidence.gpu_hours.intervals
    }
    for trial in evidence.trials:
        observed = selected(trial.provenance, selector="$.report_trial_summary")
        if observed != trial_projection(trial):
            raise ValueError(f"trial {trial.trial_id} differs from its hash-bound raw analysis")
        if trial.raw_provenance is None:
            raise ValueError(
                f"trial {trial.trial_id} lacks a distinct raw-trial provenance binding"
            )
        raw_trial = RawTrial.model_validate(
            selected(trial.raw_provenance, selector="$.raw_trial"), strict=True
        )
        if (
            raw_trial.trial_id != trial.trial_id
            or raw_trial.seed != trial.seed
            or raw_trial.mode is not trial.mode
            or raw_trial.gpu_uuids != trial.gpu_uuids
        ):
            raise ValueError(f"trial {trial.trial_id} identity differs from its raw trial")
        raw_manifest_key = (
            raw_trial.raw_manifest.artifact_reference,
            raw_trial.raw_manifest.artifact_sha256,
        )
        settled = settled_by_manifest.get(raw_manifest_key)
        if settled is None or settled.invocation_id != trial.trial_id:
            raise ValueError(f"trial {trial.trial_id} is not bound to a settled GPU invocation")
        if settled.gpu_uuids != trial.gpu_uuids:
            raise ValueError(f"trial {trial.trial_id} settled GPU pair differs from raw evidence")
        if raw_trial.raw_manifest.sample_selector != "$":
            raise ValueError(f"trial {trial.trial_id} raw manifest selector is not exact")
        validate_file(raw_trial.raw_manifest)
        for metric in raw_trial.metrics:
            validate_file(metric.provenance)
    assert evidence.trace_gate is not None and evidence.trace_gate_provenance is not None
    trace_gate = TraceControlGate.model_validate(
        selected(evidence.trace_gate_provenance, selector="$.trace_gate"), strict=True
    )
    if trace_gate != evidence.trace_gate or not trace_gate.minimal_trace_causal_trials_allowed:
        raise ValueError("trace-overhead gate is unbound or rejects causal minimal tracing")
    if trace_gate.gpu_uuids != evidence.environment.gpu_uuids:
        raise ValueError("trace-overhead controls use a different GPU pair")
    raw_trials: list[tuple[RawTrial, ReportTrialSummary]] = []
    for trial in evidence.trials:
        assert trial.raw_provenance is not None
        raw_trial = RawTrial.model_validate(
            selected(trial.raw_provenance, selector="$.raw_trial"), strict=True
        )
        validity = evaluate_trial_validity(
            raw_trial,
            planned=TrialPlan(
                trial_id=trial.trial_id,
                seed=trial.seed,
                mode=trial.mode,
                order_position=raw_trial.order_position,
            ),
            plan_pilot_digest=evidence.pilot_provenance.artifact_sha256,
            trace_gate=trace_gate,
        )
        if not validity.causal_sample_accepted:
            raise ValueError(
                f"trial {trial.trial_id} failed raw causal validation: {validity.reasons}"
            )
        require_raw_metric(raw_trial, trial)
        raw_trials.append((raw_trial, trial))
    derived_optimization = derive_software_optimization_evidence(evidence)
    if derived_optimization != evidence.software_optimization:
        raise ValueError("software optimization contains caller-authored unsupported values")
    derived_outcome = derive_outcome_evidence(evidence)
    if derived_outcome != evidence.outcome_evidence:
        raise ValueError("outcome/fused-chain evidence differs from raw mechanical derivation")
    assert evidence.outcome_provenance is not None
    outcome = OutcomeEvidence.model_validate(
        selected(evidence.outcome_provenance, selector="$.outcome_evidence"), strict=True
    )
    if outcome != derived_outcome:
        raise ValueError("hash-bound outcome analysis differs from raw mechanical derivation")
    assert evidence.policy_decision is not None and evidence.policy_provenance is not None
    policy = ReclamationDecision.model_validate(
        selected(evidence.policy_provenance, selector="$.policy_decision"), strict=True
    )
    if policy != evidence.policy_decision:
        raise ValueError("policy decision differs from its hash-bound derived analysis")
    assert evidence.software_optimization_provenance is not None
    optimization = SoftwareOptimizationEvidence.model_validate(
        selected(
            evidence.software_optimization_provenance,
            selector="$.software_optimization",
        ),
        strict=True,
    )
    if optimization != derived_optimization:
        raise ValueError("software optimization differs from its hash-bound profile analysis")
    if evidence.placement_recommendation is not None:
        assert evidence.placement_provenance is not None
        placement = PlacementRecommendation.model_validate(
            selected(
                evidence.placement_provenance,
                selector="$.placement_recommendation",
            ),
            strict=True,
        )
        if placement != evidence.placement_recommendation:
            raise ValueError("placement recommendation differs from its hash-bound study")
        decision = select_outcome(derived_outcome)
        optimized_ids = tuple(
            item.trial_id
            for item in sorted(
                (
                    trial
                    for trial in evidence.trials
                    if trial.mode is ReclamationMode.PRESERVE_OPTIMIZED
                ),
                key=lambda item: (item.seed, item.trial_id),
            )
        )
        optimized_refs = tuple(
            sorted(
                trial.raw_provenance.artifact_reference
                for trial in evidence.trials
                if trial.mode is ReclamationMode.PRESERVE_OPTIMIZED
                and trial.raw_provenance is not None
            )
        )
        if (
            placement.timing_measurement_class is not TimingMeasurementClass.HARDWARE_BACKED_REAL
            or placement.provenance is not WorkloadProvenance.HARDWARE_BACKED_REAL
            or placement.operation_id not in decision.hardware_interest_chain_ids
            or set(placement.experiment_ids) != set(optimized_ids)
            or tuple(sorted(placement.artifact_references)) != optimized_refs
        ):
            raise ValueError("placement recommendation is not bound to optimized hardware evidence")


def decision_document_path(outcome: Experiment004Outcome) -> str:
    if outcome is Experiment004Outcome.GPU_SOFTWARE_TARGET:
        return "docs/branchfabric/GPU_SOFTWARE_TARGET.md"
    if outcome in {
        Experiment004Outcome.HOST_PIPELINE_HARDWARE_INTEREST,
        Experiment004Outcome.FABRIC_HARDWARE_INTEREST,
    }:
        return "docs/branchfabric/STATE_PIPELINE_HARDWARE_INTEREST.md"
    return "docs/branchfabric/MOVEMENT_CLOSED.md"


def _report_payload(
    evidence: Experiment004ReportEvidence,
    plots: tuple[PlotPayload, ...],
    decision: OutcomeDecision | None,
) -> dict[str, object]:
    optimized = _representative(evidence, ReclamationMode.PRESERVE_OPTIMIZED)
    naive = _representative(evidence, ReclamationMode.PRESERVE_NAIVE)
    kill = _representative(evidence, ReclamationMode.KILL_AND_RECOMPUTE)
    assert optimized.movement is not None and naive.movement is not None
    break_even = None
    if evidence.policy_decision is not None:
        break_even = next(
            item
            for item in evidence.policy_decision.break_even
            if item.preservation_action is ReclamationAction.PRESERVE_OPTIMIZED
        ).model_dump(mode="json")
    return {
        "schema_version": "sloforge.branchfabric.experiment-004-report/v1",
        "report_id": evidence.report_id,
        "evidence_class": evidence.evidence_class.value,
        "fixture_preview": evidence.evidence_class is not EvidenceClass.HARDWARE_BACKED_REAL,
        "environment": evidence.environment.model_dump(mode="json"),
        "real_state_size": {
            "branch_count": evidence.pilot.branch_count,
            "prefix_tokens": evidence.pilot.prefix_tokens,
            "suffix_tokens": evidence.pilot.suffix_tokens,
            "logical_bytes": optimized.logical_state_bytes,
            "kv_logical_state_bytes": optimized.kv_logical_state_bytes,
            "logical_state_accounting_scope": optimized.logical_state_accounting_scope,
            "host_resident_transport_metadata_bytes": (
                optimized.host_resident_transport_metadata_bytes
            ),
            "helix_environment_state_bytes": optimized.helix_environment_state_bytes,
            "helix_environment_state_scope": optimized.helix_environment_state_scope,
            "shared_bytes": optimized.shared_bytes,
            "private_bytes": optimized.private_bytes,
            "physical_bytes": optimized.physical_state_bytes,
            "physical_block_count": optimized.physical_block_count,
        },
        "baselines": {
            "kill": {
                "reclamation_ns": kill.reclamation_path.duration_ns,
                "resume_ns": kill.restore_path.duration_ns,
                "lost_gpu_work_ns": kill.lost_gpu_work_ns,
            },
            "naive": {
                "reclamation_ns": naive.reclamation_path.duration_ns,
                "resume_ns": naive.restore_path.duration_ns,
            },
            "optimized": {
                "reclamation_ns": optimized.reclamation_path.duration_ns,
                "resume_ns": optimized.restore_path.duration_ns,
            },
        },
        "optimized_reclamation_critical_path": optimized.reclamation_path.model_dump(mode="json"),
        "optimized_restore_critical_path": optimized.restore_path.model_dump(mode="json"),
        "movement_accounting": optimized.movement.accounting.model_dump(mode="json"),
        "temporary_memory_bytes": optimized.temporary_memory_bytes,
        "serving_interference_fraction": optimized.serving_interference_fraction,
        "serving_slo": evidence.serving_slo.model_dump(mode="json"),
        "time_to_useful_reclaimed_capacity_ns": optimized.time_to_useful_reclaimed_capacity_ns,
        "time_to_serving_slo_restoration_ns": optimized.time_to_serving_slo_restoration_ns,
        "rollout_resume_latency_ns": optimized.restore_path.duration_ns,
        "preservation_break_even": break_even,
        "software_optimization": evidence.software_optimization.model_dump(mode="json"),
        "placement_recommendation": (
            None
            if evidence.placement_recommendation is None
            else evidence.placement_recommendation.model_dump(mode="json")
        ),
        "gpu_hours": evidence.gpu_hours.model_dump(mode="json"),
        "outcome": None if decision is None else decision.model_dump(mode="json"),
        "provenance": {
            "pilot": evidence.pilot_provenance.model_dump(mode="json"),
            "trials": [
                {
                    "trial_id": trial.trial_id,
                    "sample": trial.provenance.model_dump(mode="json"),
                }
                for trial in sorted(evidence.trials, key=lambda item: item.trial_id)
            ],
        },
        "plots": [plot.model_dump(mode="json") for plot in plots],
    }


def _markdown(payload: dict[str, object]) -> bytes:
    fixture = bool(payload["fixture_preview"])
    warning = (
        "FIXTURE PREVIEW — NOT HARDWARE EVIDENCE — NO CLASSIFICATION"
        if fixture
        else "HARDWARE-BACKED FINAL REPORT"
    )
    baseline = payload["baselines"]
    assert isinstance(baseline, dict)
    lines = [
        "# BranchFabric GPU Validation Experiment 004",
        "",
        f"**{warning}**",
        "",
        f"Evidence class: `{payload['evidence_class']}`",
        "",
        "## Baselines",
        "",
        f"- Kill/recompute: `{json.dumps(baseline['kill'], sort_keys=True)}`",
        f"- Naive preservation: `{json.dumps(baseline['naive'], sort_keys=True)}`",
        f"- Optimized preservation: `{json.dumps(baseline['optimized'], sort_keys=True)}`",
        "",
        "## State movement",
        "",
        f"`{json.dumps(payload['movement_accounting'], sort_keys=True)}`",
        "",
        "## Decision",
        "",
        (
            "No decision: fixture evidence cannot classify hardware."
            if fixture
            else f"`{json.dumps(payload['outcome'], sort_keys=True)}`"
        ),
        "",
    ]
    return "\n".join(lines).encode("utf-8")


def _write_bundle(files: dict[Path, bytes]) -> None:
    collisions = sorted(str(path) for path in files if path.exists())
    if collisions:
        raise FileExistsError(f"report publication targets already exist: {collisions}")
    for path in sorted(files, key=str):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as handle:
            handle.write(files[path])
            handle.flush()


def _characterization_markdown(payload: dict[str, object]) -> bytes:
    outcome = payload["outcome"]
    assert isinstance(outcome, dict)
    state = payload["real_state_size"]
    movement = payload["movement_accounting"]
    lines = [
        "# Capacity Reclamation Characterization",
        "",
        "This characterization is generated only from validated, hash-verified hardware evidence.",
        "",
        f"- State: `{json.dumps(state, sort_keys=True)}`",
        f"- Movement: `{json.dumps(movement, sort_keys=True)}`",
        f"- Useful capacity latency (ns): `{payload['time_to_useful_reclaimed_capacity_ns']}`",
        f"- Serving SLO restoration (ns): `{payload['time_to_serving_slo_restoration_ns']}`",
        f"- Rollout resume latency (ns): `{payload['rollout_resume_latency_ns']}`",
        f"- Classification: `{outcome['outcome']}`",
        "",
    ]
    return "\n".join(lines).encode("utf-8")


def _experiment_005_markdown(decision: OutcomeDecision) -> bytes:
    return (
        "# BranchFabric Experiment 005 Plan\n\n"
        f"Triggering Experiment 004 outcome: `{decision.outcome.value}`.\n\n"
        "Compare the same evidence-calibrated workload across GPU→host→GPU, "
        "direct GPU peer copy where topology permits it, GPU transform plus host "
        "transfer, the fused optimized software pipeline, and an evidence-calibrated "
        "BranchFabric model. Preserve exact byte accounting, semantic validation, "
        "and paired two-GPU methodology. This document is a placement study plan; "
        "it does not authorize or execute GPU workloads or hardware implementation.\n"
    ).encode()


def _manifest_bytes(
    *, destination_root: Path, evidence: Experiment004ReportEvidence, files: dict[Path, bytes]
) -> bytes:
    entries = []
    for path, contents in sorted(files.items(), key=lambda item: str(item[0])):
        entries.append(
            {
                "path": str(path.relative_to(destination_root)),
                "bytes": len(contents),
                "sha256": hashlib.sha256(contents).hexdigest(),
            }
        )
    return _canonical_bytes(
        {
            "schema_version": "sloforge.branchfabric.experiment-004-manifest/v1",
            "report_id": evidence.report_id,
            "evidence_class": evidence.evidence_class.value,
            "generated_files_excluding_manifest": entries,
            "raw_provenance": {
                "pilot": evidence.pilot_provenance.model_dump(mode="json"),
                "trials": [
                    trial.provenance.model_dump(mode="json")
                    for trial in sorted(evidence.trials, key=lambda item: item.trial_id)
                ],
            },
        }
    )


def publish_experiment_004_report(
    evidence: Experiment004ReportEvidence,
    destination_root: Path,
    *,
    mode: ReportEmissionMode,
) -> ReportPublication:
    """Publish a fixture preview or the complete real hardware report."""

    is_real = evidence.evidence_class is EvidenceClass.HARDWARE_BACKED_REAL
    if mode is ReportEmissionMode.FINAL_HARDWARE_REPORT and not is_real:
        raise ValueError("fixture/non-hardware evidence cannot emit final Experiment 004 reports")
    if mode is ReportEmissionMode.FIXTURE_PREVIEW and is_real:
        raise ValueError("hardware-backed evidence must use final report emission")
    decision: OutcomeDecision | None = None
    if is_real:
        assert evidence.outcome_evidence is not None
        _validate_real_provenance(evidence)
        derived_outcome = derive_outcome_evidence(evidence)
        decision = select_outcome(derived_outcome)
    plots = build_plot_payloads(evidence)
    if len(plots) != 9 or len({plot.plot_id for plot in plots}) != 9:
        raise RuntimeError("Experiment 004 report did not produce exactly nine plots")
    payload = _report_payload(evidence, plots, decision)
    fixture = not is_real
    if mode is ReportEmissionMode.FIXTURE_PREVIEW:
        prefix = destination_root / "fixture-preview"
        plot_root = prefix / "plots"
        report_json = prefix / "branchfabric-gpu-validation-experiment-004.preview.json"
        report_markdown = prefix / "branchfabric-gpu-validation-experiment-004.preview.md"
        decision_path = None
        characterization_path = None
        experiment_005_path = None
        manifest_path = None
    else:
        plot_root = destination_root / "artifacts/branchfabric/gpu-validation/experiment-004/plots"
        report_json = destination_root / "reports/branchfabric-gpu-validation-experiment-004.json"
        report_markdown = destination_root / "reports/branchfabric-gpu-validation-experiment-004.md"
        assert decision is not None
        decision_path = destination_root / decision_document_path(decision.outcome)
        characterization_path = (
            destination_root / "docs/branchfabric/CAPACITY_RECLAMATION_CHARACTERIZATION.md"
        )
        experiment_005_path = (
            destination_root / "docs/branchfabric/EXPERIMENT_005_PLAN.md"
            if decision.outcome
            in {
                Experiment004Outcome.HOST_PIPELINE_HARDWARE_INTEREST,
                Experiment004Outcome.FABRIC_HARDWARE_INTEREST,
            }
            else None
        )
        manifest_path = (
            destination_root / "artifacts/branchfabric/gpu-validation/experiment-004/manifest.json"
        )
    files: dict[Path, bytes] = {
        report_json: _canonical_bytes(payload),
        report_markdown: _markdown(payload),
    }
    plot_json_paths: list[str] = []
    plot_svg_paths: list[str] = []
    for plot in plots:
        json_path = plot_root / f"{plot.plot_id}.json"
        svg_path = plot_root / f"{plot.plot_id}.svg"
        files[json_path] = _canonical_bytes(plot)
        files[svg_path] = _svg(plot, fixture=fixture)
        plot_json_paths.append(str(json_path))
        plot_svg_paths.append(str(svg_path))
    if decision_path is not None:
        assert decision is not None
        contract_note = (
            "The contract requires this outcome to use the MOVEMENT_CLOSED document "
            "filename; the measured conclusion remains PRESERVATION_NOT_ECONOMIC.\n\n"
            if decision.outcome is Experiment004Outcome.PRESERVATION_NOT_ECONOMIC
            else ""
        )
        files[decision_path] = (
            f"# {decision.outcome.value}\n\n{contract_note}{decision.rationale}\n\n"
            f"Hardware-interest chains: {', '.join(decision.hardware_interest_chain_ids) or 'none'}\n"
        ).encode()
    if characterization_path is not None:
        files[characterization_path] = _characterization_markdown(payload)
    if experiment_005_path is not None:
        assert decision is not None
        files[experiment_005_path] = _experiment_005_markdown(decision)
    if manifest_path is not None:
        files[manifest_path] = _manifest_bytes(
            destination_root=destination_root,
            evidence=evidence,
            files=files,
        )
    _write_bundle(files)
    return ReportPublication(
        emission_mode=mode,
        publishable_hardware_report=is_real,
        report_json=str(report_json),
        report_markdown=str(report_markdown),
        decision_document=None if decision_path is None else str(decision_path),
        characterization_document=(
            None if characterization_path is None else str(characterization_path)
        ),
        experiment_005_plan=(None if experiment_005_path is None else str(experiment_005_path)),
        artifact_manifest=None if manifest_path is None else str(manifest_path),
        plot_json=tuple(plot_json_paths),
        plot_svg=tuple(plot_svg_paths),
        outcome=None if decision is None else decision.outcome,
    )


__all__ = [
    "Experiment004ReportEvidence",
    "HbmPlotPoint",
    "PlotPayload",
    "ReportEmissionMode",
    "ReportPublication",
    "ReportTrialSummary",
    "RuntimeEnvironmentSummary",
    "ServingPlotPoint",
    "SoftwareOptimizationEvidence",
    "build_plot_payloads",
    "decision_document_path",
    "derive_outcome_evidence",
    "derive_software_optimization_evidence",
    "publish_experiment_004_report",
]
