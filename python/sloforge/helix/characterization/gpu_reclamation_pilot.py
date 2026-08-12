"""Fail-closed raw-pilot assessment for BranchFabric Experiment 004.

This module is deliberately CPU-only.  It consumes immutable worker and
controller observations after both GPU children have exited.  In particular,
``SERVING_SLO_RESTORED`` is derived only when the existing serving analyzer
finds a measured stable window; a missing restoration is an invalid pilot, not
an event synthesized from orchestration intent.
"""

from __future__ import annotations

import hashlib
import itertools
import json
from collections import Counter
from pathlib import Path
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from sloforge.continuum.adapters.vllm_reclamation import CanonicalKvTransportManifest
from sloforge.helix.characterization.gpu_reclamation import (
    REQUIRED_PRESERVATION_PHASES,
    BranchGroupSemanticEvidence,
    BranchSemanticRecord,
    ExperimentPhase,
    ExperimentPhaseMarker,
    PilotValidityEvidence,
    RuntimeAllocationIdentity,
    RuntimeIncarnation,
)
from sloforge.helix.characterization.gpu_reclamation_accounting import (
    MemoryDomain,
    StateMovementReport,
    TransferDirection,
)
from sloforge.helix.characterization.gpu_reclamation_serving import (
    ArrivalPhase,
    IntervalMetrics,
    IntervalSLOEvaluation,
    ObservationOutcome,
    PhaseInterval,
    ServingMeasurement,
    ServingMeasurementPlan,
    ServingObservation,
    ServingRequest,
    ServingSLO,
    ServingSpikeConfig,
    ServingWorkload,
    SLORestoration,
    SLOStabilityConfig,
    WeightedTokenDistribution,
    evaluate_serving_slo,
    find_serving_slo_restoration,
    measure_serving_intervals,
)
from sloforge.helix.characterization.gpu_reclamation_trace import (
    Experiment004TraceIdentity,
    phase_markers_to_trace_events,
)

_NS_PER_SECOND = 1_000_000_000
_RESTORATION_EVALUATION_WINDOW_NS = 250_000_000


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, validate_default=True)


class MeaningfulSpikeEvidence(_StrictModel):
    control: IntervalMetrics
    before_secondary_capacity: IntervalMetrics
    control_slo: IntervalSLOEvaluation
    spike_slo: IntervalSLOEvaluation
    queue_growth_confirmed: bool
    measured_slo_departure_confirmed: bool


class RestoreInterferenceOverlap(_StrictModel):
    interval: PhaseInterval | None = None
    request_count: int = Field(ge=0)
    emitted_tokens: int = Field(ge=0)
    confirmed: bool


class PilotAssessment(_StrictModel):
    """Structured outcome which preserves diagnostic evidence when invalid."""

    schema_version: Literal["sloforge.branchfabric.experiment-004-pilot-assessment/v1"] = (
        "sloforge.branchfabric.experiment-004-pilot-assessment/v1"
    )
    pilot_valid: bool
    invalid_reasons: tuple[str, ...]
    evidence: PilotValidityEvidence | None = None
    serving_measurement: ServingMeasurement | None = None
    meaningful_spike: MeaningfulSpikeEvidence | None = None
    slo_restoration: SLORestoration | None = None
    restoration_latency_from_reclaim_trigger_ns: int | None = Field(default=None, ge=0)
    restore_interference_overlap: RestoreInterferenceOverlap | None = None
    derived_phase_events: tuple[ExperimentPhaseMarker, ...] = ()
    derived_trace_events: tuple[dict[str, Any], ...] = ()

    @model_validator(mode="after")
    def valid_status(self) -> Self:
        if self.pilot_valid != (self.evidence is not None):
            raise ValueError("pilot validity and evidence presence disagree")
        if self.pilot_valid and self.invalid_reasons:
            raise ValueError("a valid pilot cannot retain invalidity reasons")
        if not self.pilot_valid and not self.invalid_reasons:
            raise ValueError("an invalid pilot requires at least one reason")
        if self.derived_phase_events and (
            self.slo_restoration is None or self.slo_restoration.status != "restored"
        ):
            raise ValueError("derived phase events require measured SLO restoration")
        return self


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _phase_times(rollout: dict[str, Any]) -> dict[ExperimentPhase, int]:
    result: dict[ExperimentPhase, int] = {}
    previous = -1
    for raw in rollout.get("phase_events", ()):
        phase = ExperimentPhase(str(raw["phase"]))
        timestamp = int(raw["monotonic_timestamp_ns"])
        if timestamp < previous:
            raise ValueError("rollout phase events are not monotonic")
        if phase in result:
            if phase not in {
                ExperimentPhase.STATE_VALIDATE_BEGIN,
                ExperimentPhase.STATE_VALIDATE_END,
            }:
                raise ValueError(f"rollout phase event is duplicated: {phase.value}")
            # Validation is invoked once per disjoint restore subset.  Its
            # lifecycle envelope is the first BEGIN through the last END.
            if phase is ExperimentPhase.STATE_VALIDATE_END:
                result[phase] = timestamp
        else:
            result[phase] = timestamp
        previous = timestamp
    expected_order = (
        ExperimentPhase.HELIX_RECLAIM_TRIGGER,
        ExperimentPhase.ROLLOUT_ADMISSION_STOP,
        ExperimentPhase.BRANCH_QUIESCE,
        ExperimentPhase.STATE_CAPTURE_BEGIN,
        ExperimentPhase.STATE_CAPTURE_END,
        ExperimentPhase.STATE_TRANSFORM_BEGIN,
        ExperimentPhase.STATE_TRANSFORM_END,
        ExperimentPhase.D2H_BEGIN,
        ExperimentPhase.D2H_END,
        ExperimentPhase.INTEGRITY_BEGIN,
        ExperimentPhase.INTEGRITY_END,
        ExperimentPhase.STATE_PUBLISH,
        ExperimentPhase.GPU_STATE_RELEASE_BEGIN,
        ExperimentPhase.GPU_STATE_RELEASE_END,
        ExperimentPhase.HBM_RECLAIM_CONFIRMED,
        ExperimentPhase.SERVING_SECONDARY_ENABLE,
        ExperimentPhase.GPU1_FIRST_SERVING_REQUEST,
        ExperimentPhase.ROLLOUT_RESTORE_TRIGGER,
        ExperimentPhase.H2D_BEGIN,
        ExperimentPhase.H2D_END,
        ExperimentPhase.STATE_IMPORT_BEGIN,
        ExperimentPhase.STATE_VALIDATE_BEGIN,
        ExperimentPhase.STATE_VALIDATE_END,
        ExperimentPhase.STATE_IMPORT_END,
        ExperimentPhase.BRANCH_RESUME_BEGIN,
        ExperimentPhase.FIRST_RESUMED_TOKEN,
        ExperimentPhase.ROLLOUT_RESUME_COMPLETE,
    )
    observed_order = tuple(
        phase for phase in result if phase is not ExperimentPhase.SERVING_SLO_RESTORED
    )
    if observed_order != expected_order:
        raise ValueError("rollout phase events do not follow the exact reclamation transaction")
    return result


def _serving_workload(
    config: dict[str, Any], serving: dict[str, Any], rollout: dict[str, Any]
) -> tuple[ServingWorkload, tuple[ServingObservation, ...]]:
    raw_rows = tuple(serving.get("requests", ())) + tuple(
        rollout.get("temporary_serving", {}).get("requests", ())
    )
    if not raw_rows:
        raise ValueError("workers emitted no production-like serving observations")
    ordered = tuple(
        sorted(raw_rows, key=lambda row: (int(row["scheduled_arrival_ns"]), row["request_id"]))
    )
    ids = [str(row["request_id"]) for row in ordered]
    if len(ids) != len(set(ids)):
        raise ValueError("GPU0/GPU1 serving request IDs are not globally unique")
    requests = tuple(
        ServingRequest(
            sequence=index,
            request_id=str(row["request_id"]),
            phase=str(row["phase"]),
            arrival_ns=int(row["scheduled_arrival_ns"]),
            prompt_tokens=int(config["serving_prompt_tokens"]),
            requested_output_tokens=int(config["serving_output_tokens"]),
        )
        for index, row in enumerate(ordered)
    )
    observations = tuple(
        ServingObservation(
            request_id=str(row["request_id"]),
            arrival_ns=int(row["scheduled_arrival_ns"]),
            service_start_ns=int(row["admitted_ns"]),
            token_timestamps_ns=tuple(int(item) for item in row["token_timestamps_ns"]),
            completion_ns=int(row["completed_ns"]),
            outcome=ObservationOutcome.COMPLETED,
            device=(
                str(serving["physical_gpu_uuid"])
                if str(row["request_id"]).startswith("gpu0-serving")
                else str(rollout["physical_gpu_uuid"])
            ),
        )
        for row in ordered
    )
    start_ns = min(request.arrival_ns for request in requests)
    end_ns = max(
        max(request.arrival_ns for request in requests) + 1,
        int(serving["end_ns"]),
        int(rollout["temporary_serving"]["end_ns"]),
    )
    workload = ServingWorkload(
        workload_id=_canonical_sha256(
            {
                "attempt_id": config["attempt_id"],
                "requests": [request.model_dump(mode="json") for request in requests],
            }
        ),
        start_ns=start_ns,
        end_ns=end_ns,
        config=ServingSpikeConfig(
            seed=int(config["seed"]),
            control_phase="control",
            spike_phase="spike",
            phases=(
                ArrivalPhase(name="control", start_offset_ns=0, end_offset_ns=1, interarrival_ns=1),
                ArrivalPhase(name="spike", start_offset_ns=1, end_offset_ns=2, interarrival_ns=1),
            ),
            prompt_tokens=WeightedTokenDistribution(
                values=(int(config["serving_prompt_tokens"]),), weights=(1,)
            ),
            output_tokens=WeightedTokenDistribution(
                values=(int(config["serving_output_tokens"]),), weights=(1,)
            ),
            request_id_prefix="exp004-observed",
        ),
        requests=requests,
    )
    return workload, observations


def _slo(config: dict[str, Any]) -> ServingSLO:
    return ServingSLO(
        maximum_p95_ttft_ns=int(float(config["serving_slo_maximum_ttft_seconds"]) * _NS_PER_SECOND),
        maximum_p95_inter_token_latency_ns=int(
            float(config["serving_slo_maximum_inter_token_latency_seconds"]) * _NS_PER_SECOND
        ),
    )


def _measure_serving(
    *,
    config: dict[str, Any],
    serving: dict[str, Any],
    rollout: dict[str, Any],
    phases: dict[ExperimentPhase, int],
) -> tuple[
    ServingMeasurement,
    MeaningfulSpikeEvidence,
    SLORestoration,
    RestoreInterferenceOverlap,
    ServingWorkload,
    tuple[ServingObservation, ...],
]:
    workload, observations = _serving_workload(config, serving, rollout)
    trigger_ns = phases[ExperimentPhase.HELIX_RECLAIM_TRIGGER]
    spike_arrival_start_ns = int(rollout["trigger_ns"])
    first_serving_ns = phases[ExperimentPhase.GPU1_FIRST_SERVING_REQUEST]
    restore_trigger_ns = phases[ExperimentPhase.ROLLOUT_RESTORE_TRIGGER]
    resume_complete_ns = phases[ExperimentPhase.ROLLOUT_RESUME_COMPLETE]
    if not trigger_ns < first_serving_ns < restore_trigger_ns < resume_complete_ns:
        raise ValueError("serving/reclamation/restore causal event order is invalid")
    control_start_ns = int(serving["start_ns"])
    if control_start_ns >= spike_arrival_start_ns or spike_arrival_start_ns > trigger_ns:
        raise ValueError("GPU0 control interval is empty")
    measurement = measure_serving_intervals(
        workload,
        observations,
        ServingMeasurementPlan(
            intervals=(
                PhaseInterval(
                    name="control", start_ns=control_start_ns, end_ns=spike_arrival_start_ns
                ),
                PhaseInterval(
                    name="spike-before-secondary-capacity",
                    start_ns=spike_arrival_start_ns,
                    end_ns=first_serving_ns,
                ),
            )
        ),
    )
    control, affected = measurement.intervals
    slo = _slo(config)
    control_slo = evaluate_serving_slo(control, slo)
    spike_slo = evaluate_serving_slo(affected, slo)
    queue_growth = (
        affected.max_queue_depth >= control.max_queue_depth + 1
        and affected.mean_queue_depth > control.mean_queue_depth
    )
    measured_failed_checks = tuple(
        check
        for check in spike_slo.checks
        if check.availability == "available" and not check.passed
    )
    meaningful = MeaningfulSpikeEvidence(
        control=control,
        before_secondary_capacity=affected,
        control_slo=control_slo,
        spike_slo=spike_slo,
        queue_growth_confirmed=queue_growth,
        measured_slo_departure_confirmed=bool(measured_failed_checks),
    )
    stability_ns = int(float(config["serving_slo_stability_window_seconds"]) * _NS_PER_SECOND)
    if stability_ns % _RESTORATION_EVALUATION_WINDOW_NS:
        raise ValueError("configured SLO stability window is not divisible into 250ms windows")
    restoration = find_serving_slo_restoration(
        workload,
        observations,
        trigger_ns=first_serving_ns,
        measurement_end_ns=restore_trigger_ns,
        slo=slo,
        stability=SLOStabilityConfig(
            evaluation_window_ns=_RESTORATION_EVALUATION_WINDOW_NS,
            stability_window_ns=stability_ns,
        ),
    )
    overlap_start = restore_trigger_ns
    overlap_end = min(resume_complete_ns, int(serving["end_ns"]))
    if overlap_end <= overlap_start:
        overlap = RestoreInterferenceOverlap(request_count=0, emitted_tokens=0, confirmed=False)
    else:
        overlap_interval = PhaseInterval(
            name="gpu0-restore-interference-overlap",
            start_ns=overlap_start,
            end_ns=overlap_end,
        )
        overlap_metrics = measure_serving_intervals(
            workload,
            observations,
            ServingMeasurementPlan(intervals=(overlap_interval,)),
        ).intervals[0]
        gpu0_rows = tuple(
            row
            for row in serving["requests"]
            if row["phase"] == "restore-interference"
            and int(row["completed_ns"]) > overlap_start
            and int(row["admitted_ns"]) < overlap_end
        )
        emitted = sum(
            overlap_start <= int(timestamp) < overlap_end
            for row in gpu0_rows
            for timestamp in row["token_timestamps_ns"]
        )
        overlap = RestoreInterferenceOverlap(
            interval=overlap_interval,
            request_count=len(gpu0_rows),
            emitted_tokens=emitted,
            confirmed=(
                bool(gpu0_rows) and emitted > 0 and overlap_metrics.emitted_tokens >= emitted
            ),
        )
    return measurement, meaningful, restoration, overlap, workload, observations


def _validate_timeline(rows: Any, *, name: str, expected_groups: tuple[str, ...]) -> None:
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{name} critical path is absent")
    seen = tuple(str(row["stage"]) for row in rows)
    cursor = int(rows[0]["start_ns"])
    total = 0
    for row in rows:
        start = int(row["start_ns"])
        end = int(row["end_ns"])
        duration = int(row["duration_ns"])
        if start != cursor or end < start or duration != end - start:
            raise ValueError(f"{name} critical path contains a gap, overlap, or bad duration")
        total += duration
        cursor = end
    if total != int(rows[-1]["end_ns"]) - int(rows[0]["start_ns"]):
        raise ValueError(f"{name} critical path does not conserve wall time")
    for group in expected_groups:
        if not any(stage == group or stage.startswith(group + "_") for stage in seen):
            raise ValueError(f"{name} critical path omits {group}")
    group_positions: list[int] = []
    for group in expected_groups:
        positions = tuple(
            index
            for index, stage in enumerate(seen)
            if stage == group or stage.startswith(group + "_")
        )
        group_positions.append(positions[0])
    # Restore allocates and validates the first branch before publishing its
    # prefix, then allocates the remaining branches and validates the second
    # disjoint subset.  First occurrence order is therefore the exact causal
    # invariant; requiring all occurrences of one group to precede the next
    # would reject the correct fail-closed import protocol.
    if any(left >= right for left, right in itertools.pairwise(group_positions)):
        raise ValueError(f"{name} critical-path stages do not follow the required order")


def _validate_independent_movement_telemetry(
    movement: StateMovementReport, rollout: dict[str, Any]
) -> None:
    """Reconcile the pass ledger with independently aggregated instrumentation."""

    telemetry = rollout.get("operation_telemetry")
    if not isinstance(telemetry, dict) or telemetry.get("status") != "succeeded":
        raise ValueError("operation telemetry is absent or failed")
    cuda = telemetry.get("cuda_summary")
    host = telemetry.get("host_allocation_summary")
    if not isinstance(cuda, dict) or not isinstance(host, dict):
        raise ValueError("operation telemetry summaries are malformed")
    bytes_by_kind = cuda.get("bytes_by_kind")
    counts_by_kind = cuda.get("copy_count_by_kind")
    sizes_by_kind = cuda.get("copy_sizes_bytes_by_kind")
    if not all(isinstance(item, dict) for item in (bytes_by_kind, counts_by_kind, sizes_by_kind)):
        raise ValueError("CUDA copy summary is malformed")
    assert isinstance(bytes_by_kind, dict)
    assert isinstance(counts_by_kind, dict)
    assert isinstance(sizes_by_kind, dict)
    accounting = movement.accounting
    if int(bytes_by_kind.get("d2h", -1)) != accounting.d2h_bytes:
        raise ValueError("D2H bytes disagree between movement and CUDA instrumentation")
    if int(bytes_by_kind.get("h2d", -1)) != accounting.h2d_bytes:
        raise ValueError("H2D bytes disagree between movement and CUDA instrumentation")
    expected_sizes: dict[TransferDirection, list[int]] = {
        TransferDirection.D2H: [],
        TransferDirection.H2D: [],
    }
    grouped_transfers: dict[tuple[TransferDirection, int, int], int] = {}
    for state_pass in movement.passes:
        if state_pass.transfer_direction is TransferDirection.NONE:
            continue
        key = (state_pass.transfer_direction, state_pass.start_ns, state_pass.end_ns)
        grouped_transfers[key] = grouped_transfers.get(key, 0) + state_pass.transfer_bytes
    for (direction, _start, _end), size in grouped_transfers.items():
        expected_sizes[direction].append(size)
    for direction in (TransferDirection.D2H, TransferDirection.H2D):
        observed = tuple(sorted(int(value) for value in sizes_by_kind.get(direction.value, ())))
        expected = tuple(sorted(expected_sizes[direction]))
        if observed != expected:
            raise ValueError(
                f"{direction.value.upper()} copy sizes disagree between movement and CUDA instrumentation"
            )
        if int(counts_by_kind.get(direction.value, -1)) != len(expected):
            raise ValueError(f"{direction.value.upper()} copy count is inconsistent")
    if int(cuda.get("copy_count", -1)) != sum(int(value) for value in counts_by_kind.values()):
        raise ValueError("CUDA aggregate copy count is inconsistent")

    allocated = host.get("allocated_bytes_by_kind")
    peak = host.get("peak_live_bytes_by_kind")
    active = host.get("active_bytes_by_kind")
    if not all(isinstance(item, dict) for item in (allocated, peak, active)):
        raise ValueError("host allocation summary is malformed")
    assert isinstance(allocated, dict) and isinstance(peak, dict) and isinstance(active, dict)
    if (
        sum(int(value) for value in allocated.values())
        != accounting.host_temporary_allocation_bytes
    ):
        raise ValueError("host temporary bytes disagree with the allocation ledger")
    if (
        sum(int(value) for value in peak.values())
        != accounting.peak_host_temporary_allocation_bytes
    ):
        raise ValueError("host peak temporary bytes disagree with the allocation ledger")
    if any(int(value) != 0 for value in active.values()):
        raise ValueError("host allocation telemetry reports live checkpoint buffers")


def _branch_semantics(
    *,
    config: dict[str, Any],
    rollout: dict[str, Any],
    manifest: CanonicalKvTransportManifest,
) -> BranchGroupSemanticEvidence:
    source_by_page = {
        str(item["logical_page_id"]): item["source"]
        for item in rollout["source_capture_evidence"]["bindings"]
    }
    destination_by_page = {
        str(page): int(block) for page, block in rollout["destination_page_map"].items()
    }
    destination_epochs = {
        int(block): int(epoch) for block, epoch in rollout["destination_allocations"]
    }
    gpu_uuid = str(rollout["physical_gpu_uuid"])
    sampling_by_branch = rollout["sampling_semantics_by_branch"]
    source: list[BranchSemanticRecord] = []
    restored: list[BranchSemanticRecord] = []
    source_incarnations: list[RuntimeIncarnation] = []
    restored_incarnations: list[RuntimeIncarnation] = []
    for branch in manifest.branches:
        sampling_hash = _canonical_sha256(sampling_by_branch[branch.logical_branch_id])
        source_runtime_ids = rollout.get("source_runtime_request_ids")
        restored_runtime_ids = rollout.get("restored_internal_runtime_request_ids")
        if isinstance(source_runtime_ids, dict) and isinstance(restored_runtime_ids, dict):
            source_id = str(source_runtime_ids[branch.logical_branch_id])
            restored_id = str(restored_runtime_ids[branch.logical_branch_id])
        elif config.get("serving_methodology") == "v10-global-capacity":
            raise ValueError("v10 branch evidence omits actual runtime request identities")
        else:
            # Historical v9/fixture compatibility. This fallback is never
            # eligible for the v10 scientific-validity object.
            source_id = branch.logical_branch_id
            restored_id = f"{branch.logical_branch_id}@restore-1"
        common = dict(
            logical_branch_id=branch.logical_branch_id,
            parent_logical_branch_id=branch.parent_logical_branch_id,
            policy_epoch=manifest.policy_epoch,
            model_id=manifest.model_id,
            model_revision=manifest.model_revision,
            tokenizer_id=manifest.tokenizer_id,
            tokenizer_revision=manifest.tokenizer_revision,
            token_count=len(branch.token_ids),
            computed_tokens=branch.computed_tokens,
            token_history_sha256=branch.token_history_sha256,
            sampling_params_sha256=sampling_hash,
        )
        source.append(
            BranchSemanticRecord.model_validate({"runtime_request_id": source_id, **common})
        )
        restored.append(
            BranchSemanticRecord.model_validate({"runtime_request_id": restored_id, **common})
        )
        source_incarnations.append(
            RuntimeIncarnation(
                logical_branch_id=branch.logical_branch_id,
                runtime_request_id=source_id,
                allocations=tuple(
                    RuntimeAllocationIdentity(
                        gpu_uuid=gpu_uuid,
                        block_index=int(source_by_page[page]["block_index"]),
                        allocation_epoch=int(source_by_page[page]["allocation_epoch"]),
                    )
                    for page in branch.logical_page_ids
                ),
            )
        )
        restored_incarnations.append(
            RuntimeIncarnation(
                logical_branch_id=branch.logical_branch_id,
                runtime_request_id=restored_id,
                allocations=tuple(
                    RuntimeAllocationIdentity(
                        gpu_uuid=gpu_uuid,
                        block_index=destination_by_page[page],
                        allocation_epoch=destination_epochs[destination_by_page[page]],
                    )
                    for page in branch.logical_page_ids
                ),
            )
        )
    continuation_counts: dict[str, int] = {}
    for runtime_id, value in rollout["continuation_token_counts"].items():
        runtime_id = str(runtime_id)
        suffix = "@restore-1"
        logical_id = runtime_id[: -len(suffix)] if runtime_id.endswith(suffix) else runtime_id
        if logical_id in continuation_counts:
            raise ValueError("continuation counts contain duplicate logical branch IDs")
        continuation_counts[logical_id] = int(value)
    return BranchGroupSemanticEvidence(
        source=tuple(source),
        restored=tuple(restored),
        source_incarnations=tuple(source_incarnations),
        restored_incarnations=tuple(restored_incarnations),
        expected_first_token_ids={
            str(key): int(value) for key, value in rollout["expected_first_tokens"].items()
        },
        observed_first_token_ids={
            str(key): int(value) for key, value in rollout["first_resumed_tokens"].items()
        },
        continuation_token_counts=continuation_counts,
    )


def _movement_complete(report: StateMovementReport, logical_bytes: int) -> bool:
    domains = {(item.source_memory, item.destination_memory) for item in report.passes}
    directions = {item.transfer_direction for item in report.passes}
    accounting = report.accounting
    return (
        accounting.logical_state_bytes == logical_bytes
        # The naive pilot necessarily has the checkpoint D2H payload and may
        # have additional destination-validation recapture D2H traffic.
        and accounting.d2h_bytes >= logical_bytes
        and accounting.h2d_bytes == logical_bytes
        and (MemoryDomain.SOURCE_GPU_NATIVE_PAGED, MemoryDomain.GPU_TRANSFORM_BUFFER) in domains
        and any(
            source in {MemoryDomain.GPU_TRANSFORM_BUFFER, MemoryDomain.GPU_TRANSPORT_BUFFER}
            and destination is MemoryDomain.PINNED_HOST_TRANSPORT
            for source, destination in domains
        )
        and any(
            source is MemoryDomain.PINNED_HOST_TRANSPORT
            and destination
            in {
                MemoryDomain.GPU_TRANSFORM_BUFFER,
                MemoryDomain.GPU_TRANSPORT_BUFFER,
            }
            for source, destination in domains
        )
        and any(
            destination is MemoryDomain.DESTINATION_GPU_NATIVE_PAGED for _, destination in domains
        )
        and TransferDirection.D2H in directions
        and TransferDirection.H2D in directions
    )


def _controller_cleanup_valid(controller: dict[str, Any]) -> bool:
    audits = tuple(controller.get("cuda_clean_import_audits", ()))
    returncodes = controller.get("worker_returncodes", {})
    return (
        controller.get("status") == "succeeded"
        and set(returncodes) == {"serving", "rollout"}
        and set(returncodes.values()) == {0}
        and not controller.get("compute_processes_after")
        and len(audits) == 3
        and all(item.get("cuda_clean") is True for item in audits)
    )


def _restoration_latency_from_reclaim(
    restoration: SLORestoration | None, phases: dict[ExperimentPhase, int]
) -> int | None:
    if (
        restoration is None
        or restoration.restored_at_ns is None
        or ExperimentPhase.HELIX_RECLAIM_TRIGGER not in phases
    ):
        return None
    return restoration.restored_at_ns - phases[ExperimentPhase.HELIX_RECLAIM_TRIGGER]


def assess_pilot_payloads(
    *,
    config: dict[str, Any],
    controller: dict[str, Any],
    serving: dict[str, Any],
    rollout: dict[str, Any],
    readiness: tuple[dict[str, Any], dict[str, Any]],
) -> PilotAssessment:
    """Assess one completed raw pilot without changing any source artifact."""

    reasons: list[str] = []
    measurement: ServingMeasurement | None = None
    meaningful: MeaningfulSpikeEvidence | None = None
    restoration: SLORestoration | None = None
    overlap: RestoreInterferenceOverlap | None = None
    phases: dict[ExperimentPhase, int] = {}
    manifest: CanonicalKvTransportManifest | None = None
    movement: StateMovementReport | None = None
    semantics: BranchGroupSemanticEvidence | None = None

    def reject(reason: str) -> None:
        if reason not in reasons:
            reasons.append(reason)

    for role, payload in (("serving", serving), ("rollout", rollout)):
        if payload.get("status") != "succeeded" or payload.get("role") != role:
            reject(f"{role} worker did not publish a successful matching result")
    try:
        phases = _phase_times(rollout)
    except (KeyError, TypeError, ValueError) as error:
        reject(f"invalid phase evidence: {error}")
    try:
        manifest = CanonicalKvTransportManifest.model_validate_json(
            json.dumps(rollout["transport_manifest"]), strict=True
        )
    except (KeyError, TypeError, ValueError) as error:
        reject(f"invalid canonical transport manifest: {error}")
    try:
        movement = StateMovementReport.model_validate_json(
            json.dumps(rollout["movement_report"]), strict=True
        )
    except (KeyError, TypeError, ValueError) as error:
        reject(f"invalid state movement report: {error}")
    try:
        _validate_timeline(
            rollout["reclamation_stages"],
            name="reclamation",
            expected_groups=(
                "admission_stop",
                "branch_quiesce",
                "final_state_capture",
                "delta_extraction",
                "source_layout_read",
                "state_transform",
                "device_to_host",
                "integrity_generation",
                "transport_publish",
                "runtime_state_release",
                "capacity_reclaim_confirmation",
                "serving_secondary_enable",
                "first_useful_serving_request",
            ),
        )
        _validate_timeline(
            rollout["restore_stages"],
            name="restore",
            expected_groups=(
                "destination_request_construction",
                "transport_layout_read_validation",
                "host_to_device",
                "destination_allocation",
                "destination_native_write",
                "destination_validation",
                "scheduler_first_forward_and_token",
            ),
        )
        critical_valid = True
    except (KeyError, TypeError, ValueError) as error:
        critical_valid = False
        reject(f"invalid critical-path decomposition: {error}")
    if phases:
        try:
            measurement, meaningful, restoration, overlap, _workload, _observations = (
                _measure_serving(
                    config=config,
                    serving=serving,
                    rollout=rollout,
                    phases=phases,
                )
            )
        except (KeyError, TypeError, ValueError) as error:
            reject(f"invalid serving measurement: {error}")
    if meaningful is not None:
        if not meaningful.control_slo.satisfied or meaningful.control.completed_requests == 0:
            reject("GPU0 control interval is not a stable serving baseline")
        if not meaningful.queue_growth_confirmed:
            reject("serving spike did not produce measured queue growth")
        if not meaningful.measured_slo_departure_confirmed:
            reject("serving spike did not produce an available measured SLO departure")
    if restoration is None or restoration.status != "restored":
        reject("serving SLO was not restored for the configured stability window")
    if overlap is None or not overlap.confirmed:
        reject("GPU0 emitted no serving tokens during the claimed restore-interference interval")
    if movement is not None and not _movement_complete(
        movement, int(rollout.get("logical_state_bytes", -1))
    ):
        reject("movement report does not cover exact native/D2H/H2D/destination domains")
    if (
        rollout.get("logical_state_accounting_scope")
        != "gpu-resident KV payload; required host-resident transport metadata is reported "
        "separately and excluded from movement amplification"
        or int(rollout.get("kv_logical_state_bytes", -1))
        != int(rollout.get("logical_state_bytes", -2))
        or int(rollout.get("host_resident_transport_metadata_bytes", 0)) <= 0
        or int(rollout.get("helix_environment_state_bytes", -1)) != 0
        or rollout.get("helix_environment_state_scope")
        != "bounded model-only greedy rollout has no external environment or trajectory payload"
    ):
        reject("KV denominator or required non-KV state scope is incomplete")
    if movement is not None:
        try:
            _validate_independent_movement_telemetry(movement, rollout)
        except (KeyError, TypeError, ValueError) as error:
            reject(f"movement/telemetry reconciliation failed: {error}")
    if manifest is not None:
        try:
            semantics = _branch_semantics(config=config, rollout=rollout, manifest=manifest)
        except (KeyError, TypeError, ValueError) as error:
            reject(f"invalid branch semantic continuity evidence: {error}")

    raw_phases = frozenset(phases)
    derived_markers: tuple[ExperimentPhaseMarker, ...] = ()
    derived_trace: tuple[dict[str, Any], ...] = ()
    if restoration is not None and restoration.restored_at_ns is not None:
        marker = ExperimentPhaseMarker(
            phase=ExperimentPhase.SERVING_SLO_RESTORED,
            monotonic_timestamp_ns=restoration.restored_at_ns,
            attributes={
                "derivation": "find_serving_slo_restoration",
                "stability_window_ns": restoration.stability_window_ns,
            },
        )
        derived_markers = (marker,)
        try:
            generated = phase_markers_to_trace_events(
                derived_markers,
                identity=Experiment004TraceIdentity(
                    trace_id=f"exp004:{config['attempt_id']}:derived-serving",
                    session_id="gpu0-serving",
                    branch_group_id="production-like-serving",
                    logical_state_id=f"exp004-state:{config['attempt_id']}",
                    tenant_id="branchfabric-exp004",
                    security_domain="branchfabric-exp004",
                    device=str(serving["physical_gpu_uuid"]),
                ),
            )
            derived_trace = tuple(item.model_dump(mode="json") for item in generated)
        except (KeyError, TypeError, ValueError) as error:
            reject(f"failed to project measured SLO restoration into trace-v1: {error}")
    effective_phases = raw_phases | {item.phase for item in derived_markers}
    if not REQUIRED_PRESERVATION_PHASES.issubset(effective_phases):
        missing = sorted(phase.value for phase in REQUIRED_PRESERVATION_PHASES - effective_phases)
        reject(f"required phase markers are missing: {missing}")

    trace_counts = Counter(
        str(event.get("attributes", {}).get("experiment_phase", ""))
        for event in rollout.get("minimal_trace_events", ())
    )
    trace_counts.update(
        str(event.get("attributes", {}).get("experiment_phase", "")) for event in derived_trace
    )
    missing_trace = tuple(
        phase.value for phase in REQUIRED_PRESERVATION_PHASES if trace_counts[phase.value] < 2
    )
    if missing_trace:
        reject(
            f"paired BranchWorkloadTrace/StateOperationTrace events are missing: {missing_trace}"
        )

    cleanup_passed = _controller_cleanup_valid(controller)
    if not cleanup_passed:
        reject("controller postflight cleanup evidence is invalid")
    inventory = tuple(controller.get("inventory_before", ()))
    if len(inventory) != 2:
        reject("controller did not record exactly two GPU inventory rows")
    if len(readiness) != 2 or any(
        int(item.get("model_ready_ns", 1 << 63))
        >= phases.get(ExperimentPhase.HELIX_RECLAIM_TRIGGER, -1)
        for item in readiness
    ):
        reject("both models were not warm before the reclamation trigger")
    if not any(item.get("role") == "rollout" and item.get("rollouts_ready") for item in readiness):
        reject("rollout branches were not live at the warm-worker barrier")
    if any(
        payload.get("resource_telemetry", {}).get("status") != "succeeded"
        for payload in (serving, rollout)
    ):
        reject("low-overhead resource telemetry is absent or failed")

    release = rollout.get("source_block_release_summary", {})
    memory_before = rollout.get("source_memory_before_release", {})
    memory_after = rollout.get("source_memory_after_release", {})
    if not all(
        release.get(key) is True
        for key in (
            "all_native_refcounts_zero",
            "all_allocator_available",
            "all_hashes_cleared",
            "allocation_epochs_match_capture",
            "full_free_pool_recovered",
        )
    ):
        reject("exact captured source blocks were not proven released")
    first_serving = rollout.get("first_serving_ns")
    if first_serving is None or not any(
        int(row.get("first_token_ns", -1)) == int(first_serving)
        for row in rollout.get("temporary_serving", {}).get("requests", ())
    ):
        reject("GPU1 first useful serving token lacks a raw request observation")
    if not rollout.get("transport_integrity_valid"):
        reject("transport integrity validation failed")
    if not rollout.get("source_destination_fresh"):
        reject("destination physical allocation generations are not fresh")
    if not rollout.get("all_branches_resumed"):
        reject("not all rollout branches resumed")
    if rollout.get("warm_pool_driver_hbm_released") is not False:
        reject("warm vLLM pool was falsely reported as driver-visible HBM release")

    if reasons or manifest is None or movement is None or semantics is None or meaningful is None:
        return PilotAssessment(
            pilot_valid=False,
            invalid_reasons=tuple(reasons or ("pilot evidence is incomplete",)),
            serving_measurement=measurement,
            meaningful_spike=meaningful,
            slo_restoration=restoration,
            restoration_latency_from_reclaim_trigger_ns=_restoration_latency_from_reclaim(
                restoration, phases
            ),
            restore_interference_overlap=overlap,
            derived_phase_events=derived_markers,
            derived_trace_events=derived_trace,
        )

    source_page_ids = {
        str(item["logical_page_id"]) for item in rollout["source_capture_evidence"]["bindings"]
    }
    destination_page_ids = {str(item) for item in rollout["destination_page_map"]}
    layouts_distinct = (
        manifest.canonical_layout == "page,layer,token,kv,head,dim"
        and source_page_ids == destination_page_ids
        and set(tuple(item) for item in rollout["source_allocations"]).isdisjoint(
            set(tuple(item) for item in rollout["destination_allocations"])
        )
    )
    gpu_uuids = tuple(str(item["uuid"]) for item in inventory)
    gpu_models = tuple(str(item["name"]) for item in inventory)
    if len(gpu_uuids) != 2 or len(gpu_models) != 2:
        return PilotAssessment(
            pilot_valid=False,
            invalid_reasons=("exact two-GPU identity evidence is incomplete",),
            serving_measurement=measurement,
            meaningful_spike=meaningful,
            slo_restoration=restoration,
            restore_interference_overlap=overlap,
            derived_phase_events=derived_markers,
            derived_trace_events=derived_trace,
        )
    try:
        evidence = PilotValidityEvidence(
            gpu_uuids=(gpu_uuids[0], gpu_uuids[1]),
            gpu_models=(gpu_models[0], gpu_models[1]),
            both_models_warm_before_trigger=True,
            gpu0_baseline_valid=True,
            branch_count=int(rollout["branch_count"]),
            prefix_tokens=int(config["prefix_length"]),
            suffix_tokens=int(config["suffix_length"]),
            shared_bytes=int(rollout["shared_bytes"]),
            private_bytes=int(rollout["private_bytes"]),
            logical_bytes=int(rollout["logical_state_bytes"]),
            source_assigned_bytes_before=int(memory_before["kv_assigned_bytes"]),
            source_assigned_bytes_after=int(memory_after["kv_assigned_bytes"]),
            source_pool_reserved_bytes_before=int(memory_before["kv_pool_reserved_bytes"]),
            source_pool_reserved_bytes_after=int(memory_after["kv_pool_reserved_bytes"]),
            source_blocks_allocator_available=bool(release["all_allocator_available"]),
            source_hashes_cleared=bool(release["all_hashes_cleared"]),
            transport_integrity_valid=True,
            source_transport_destination_layouts_distinct=layouts_distinct,
            movement_domains_complete=True,
            required_phase_events=effective_phases,
            critical_timelines_valid=critical_valid,
            gpu1_served_real_request=True,
            serving_metrics_recorded=True,
            restored_branch_count=len(rollout["first_resumed_tokens"]),
            all_restored_allocations_fresh=True,
            branch_semantics=semantics,
            resumed_continuation_valid=True,
            required_trace_events_dropped=0,
            cleanup_passed=True,
            nvml_hbm_release_claimed=False,
        )
    except (KeyError, TypeError, ValueError) as error:
        return PilotAssessment(
            pilot_valid=False,
            invalid_reasons=(f"pilot validity contract rejected evidence: {error}",),
            serving_measurement=measurement,
            meaningful_spike=meaningful,
            slo_restoration=restoration,
            restoration_latency_from_reclaim_trigger_ns=_restoration_latency_from_reclaim(
                restoration, phases
            ),
            restore_interference_overlap=overlap,
            derived_phase_events=derived_markers,
            derived_trace_events=derived_trace,
        )
    return PilotAssessment(
        pilot_valid=True,
        invalid_reasons=(),
        evidence=evidence,
        serving_measurement=measurement,
        meaningful_spike=meaningful,
        slo_restoration=restoration,
        restoration_latency_from_reclaim_trigger_ns=_restoration_latency_from_reclaim(
            restoration, phases
        ),
        restore_interference_overlap=overlap,
        derived_phase_events=derived_markers,
        derived_trace_events=derived_trace,
    )


def assess_pilot_directory(
    *, work_root: Path, config: dict[str, Any], controller: dict[str, Any]
) -> PilotAssessment:
    """Load the five immutable raw inputs and return a diagnostic assessment."""

    try:
        serving = json.loads((work_root / "serving/result.json").read_text())
        rollout = json.loads((work_root / "rollout/result.json").read_text())
        readiness = (
            json.loads((work_root / "barriers/serving.ready.json").read_text()),
            json.loads((work_root / "barriers/rollout.ready.json").read_text()),
        )
        if not isinstance(serving, dict) or not isinstance(rollout, dict):
            raise TypeError("worker results must be JSON objects")
        if any(not isinstance(item, dict) for item in readiness):
            raise TypeError("worker readiness records must be JSON objects")
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as error:
        return PilotAssessment(
            pilot_valid=False,
            invalid_reasons=(f"raw pilot evidence could not be loaded: {error}",),
        )
    try:
        return assess_pilot_payloads(
            config=config,
            controller=controller,
            serving=serving,
            rollout=rollout,
            readiness=readiness,
        )
    except Exception as error:  # preserve raw GPU evidence even on an unforeseen gate defect
        return PilotAssessment(
            pilot_valid=False,
            invalid_reasons=(
                f"pilot postprocessor failed closed: {type(error).__name__}: {error}",
            ),
        )


__all__ = [
    "MeaningfulSpikeEvidence",
    "PilotAssessment",
    "RestoreInterferenceOverlap",
    "assess_pilot_directory",
    "assess_pilot_payloads",
]
