"""Raw, fail-closed postprocessing for Experiment 004 v10.

This module is CPU-only. It reconstructs the serving workload, causal v10
timeline, state semantics, and corrected movement accounting from immutable
worker/controller artifacts after both GPU workers have exited.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import statistics
from itertools import pairwise
from pathlib import Path
from typing import Any

from sloforge.continuum.adapters.vllm_reclamation import CanonicalKvTransportManifest
from sloforge.helix.characterization.gpu_reclamation_accounting import (
    MemoryDomain,
    StateMovementReport,
    build_state_movement_report,
)
from sloforge.helix.characterization.gpu_reclamation_pilot import (
    _branch_semantics,
    _validate_independent_movement_telemetry,
)
from sloforge.helix.characterization.gpu_reclamation_serving import (
    ArrivalPhase,
    ObservationOutcome,
    PhaseInterval,
    ServingObservation,
    ServingRequest,
    ServingSpikeConfig,
    ServingWorkload,
    WeightedTokenDistribution,
)
from sloforge.helix.characterization.gpu_reclamation_v10_validity import (
    BoundedBacklogEvidence,
    BranchResumeEvidence,
    BudgetEvidence,
    CleanupEvidence,
    MovementAccountingEvidence,
    StateCorrectnessEvidence,
    TimelineEvidenceKind,
    V10AssessmentWindows,
    V10Phase,
    V10PrerequisiteEvidence,
    V10ScientificValidity,
    V10Timeline,
    V10TimelineEvent,
    V10ValidityConfig,
    assess_v10_scientific_validity,
    outstanding_queue_depth_at,
)

NS_PER_SECOND = 1_000_000_000
_MAXIMUM_ARRIVAL_LATENESS_NS = 100_000_000
_ARRIVAL_RATE_TOLERANCE_FRACTION = 0.05
_ARRIVAL_BURST_WINDOW_NS = 50_000_000
_V10_MOVEMENT_SEGMENT_COUNT = 9
_V10_MOVEMENT_RECORD_COUNT = 270
_V10_MOVEMENT_EDGE_COUNT = 522
_V10_MOVEMENT_LABELS = frozenset(
    {
        "capture-source-native-read",
        "capture-native-axis-contiguous",
        "capture-unpage-valid-tokens",
        "capture-stack-layers",
        "capture-concatenate-pages",
        "capture-d2h",
        "capture-integrity-manifest",
        "capture-integrity-hash-reads",
        "transport-publish-validation",
        "transport-publish-hash-reads",
        "restore-transport-validation",
        "restore-transport-hash-reads",
        "restore-h2d",
        "restore-zero-native-pages",
        "restore-overlay-valid-tokens",
        "restore-native-axis-contiguous",
        "restore-stack-native-pages",
        "restore-destination-native-write",
        "validation-destination-native-read",
        "validation-native-axis-contiguous",
        "validation-unpage-valid-tokens",
        "validation-stack-layers",
        "validation-concatenate-pages",
        "validation-d2h",
        "validation-expected-page-concatenation",
        "validation-host-tensor-compare",
        "validation-recapture-host-lifetime",
        "restore-import-validation",
        "restore-import-hash-reads",
        "capture-pinned-transport-lifetime",
    }
)


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _load_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"v10 raw artifact is not an object: {path}")
    return payload


def _phase_rows(rollout: dict[str, Any], name: str) -> tuple[dict[str, Any], ...]:
    rows = tuple(row for row in rollout.get("phase_events", ()) if str(row.get("phase")) == name)
    if not rows:
        raise ValueError(f"v10 rollout omitted raw phase {name}")
    return rows


def _phase_timestamp(rollout: dict[str, Any], name: str, *, last: bool = False) -> int:
    rows = _phase_rows(rollout, name)
    return int(rows[-1 if last else 0]["monotonic_timestamp_ns"])


def _validate_measured_transaction_compilation(
    *, serving: dict[str, Any], rollout: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    """Revalidate transaction-wide no-compilation evidence from both workers."""

    expected_fields = {
        "schema_version",
        "source",
        "role",
        "interval_start_ns",
        "interval_end_ns",
        "capture_buffer_valid",
        "events",
        "no_deferred_compilation_event",
        "passed",
    }
    evidence: dict[str, dict[str, Any]] = {}
    for role, payload in (("serving", serving), ("rollout", rollout)):
        observation = payload.get("measured_transaction_compilation_observation")
        if not isinstance(observation, dict):
            raise ValueError(f"v10 {role} omitted measured transaction compilation evidence")
        start_ns = observation.get("interval_start_ns")
        end_ns = observation.get("interval_end_ns")
        if (
            payload.get("role") != role
            or set(observation) != expected_fields
            or observation.get("schema_version")
            != "sloforge.branchfabric.measured-transaction-compilation-observation/v1"
            or observation.get("source") != "bounded-python-logging-handler"
            or observation.get("role") != role
            or isinstance(start_ns, bool)
            or not isinstance(start_ns, int)
            or isinstance(end_ns, bool)
            or not isinstance(end_ns, int)
            or end_ns <= start_ns
            or observation.get("capture_buffer_valid") is not True
            or observation.get("events") != []
            or observation.get("no_deferred_compilation_event") is not True
            or observation.get("passed") is not True
        ):
            raise ValueError(f"v10 {role} measured transaction compilation gate failed")
        if role == "serving":
            payload_start_ns = int(payload["start_ns"])
            payload_end_ns = int(payload["end_ns"])
        else:
            payload_start_ns = _phase_timestamp(payload, "RECLAIM_TRIGGER")
            payload_end_ns = int(payload["rollout_continuation_complete_ns"])
        if start_ns > payload_start_ns or end_ns < payload_end_ns:
            raise ValueError(f"v10 {role} compilation observation does not span the transaction")
        evidence[role] = observation
    return evidence


def _event(phase: V10Phase, timestamp: int, reference: str) -> V10TimelineEvent:
    return V10TimelineEvent(
        phase=phase,
        monotonic_timestamp_ns=timestamp,
        evidence_kind=(
            TimelineEvidenceKind.DERIVED_METRIC
            if reference.startswith("analysis:")
            else (
                TimelineEvidenceKind.RAW_REQUEST
                if "serving" in reference
                else TimelineEvidenceKind.RAW_RUNTIME_STATE
            )
        ),
        evidence_reference=reference,
    )


def _validate_raw_recovery_trend(
    recovery: dict[str, Any], *, config: dict[str, Any]
) -> tuple[int, int]:
    trend = recovery.get("queue_trend")
    if not isinstance(trend, dict) or set(trend) != {
        "window_start_ns",
        "window_end_ns",
        "sample_interval_ns",
        "samples",
        "initial_depth",
        "final_depth",
        "first_half_mean_depth",
        "second_half_mean_depth",
        "slope_requests_per_second",
        "offered_requests",
        "completed_requests",
        "offered_rate_per_second",
        "completed_rate_per_second",
        "sustained_negative",
        "completed_rate_exceeds_offered",
        "passed",
    }:
        raise ValueError("v10 raw recovery omitted its exact sustained queue-trend evidence")
    start = int(trend["window_start_ns"])
    end = int(trend["window_end_ns"])
    interval = int(trend["sample_interval_ns"])
    expected_window = round(float(config["serving_slo_stability_window_seconds"]) * NS_PER_SECOND)
    expected_interval = round(float(config["serving_recovery_evaluation_seconds"]) * NS_PER_SECOND)
    if end - start != expected_window or interval != expected_interval:
        raise ValueError("v10 raw queue trend does not span the declared stability window")
    samples = trend["samples"]
    if not isinstance(samples, list) or any(
        not isinstance(item, dict) or set(item) != {"timestamp_ns", "queue_depth"}
        for item in samples
    ):
        raise ValueError("v10 raw queue-trend samples are malformed")
    timestamps = tuple(int(item["timestamp_ns"]) for item in samples)
    depths = tuple(int(item["queue_depth"]) for item in samples)
    expected_timestamps = tuple(range(start, end + 1, interval))
    if timestamps != expected_timestamps or len(depths) < 3 or any(depth < 0 for depth in depths):
        raise ValueError("v10 raw queue trend lacks exact complete boundary samples")
    seconds = tuple((timestamp - start) / NS_PER_SECOND for timestamp in timestamps)
    mean_x = statistics.fmean(seconds)
    mean_y = statistics.fmean(depths)
    denominator = sum((value - mean_x) ** 2 for value in seconds)
    slope = (
        sum(
            (x_value - mean_x) * (y_value - mean_y)
            for x_value, y_value in zip(seconds, depths, strict=True)
        )
        / denominator
    )
    midpoint = len(depths) // 2
    first_mean = statistics.fmean(depths[:midpoint])
    second_mean = statistics.fmean(depths[midpoint:])
    sustained = slope < 0.0 and depths[-1] < depths[0] and second_mean < first_mean
    duration = (end - start) / NS_PER_SECOND
    offered_rate = int(trend["offered_requests"]) / duration
    completed_rate = int(trend["completed_requests"]) / duration
    excess = completed_rate > offered_rate
    if not (
        int(trend["initial_depth"]) == depths[0]
        and int(trend["final_depth"]) == depths[-1]
        and math.isclose(
            float(trend["first_half_mean_depth"]), first_mean, rel_tol=0.0, abs_tol=1e-12
        )
        and math.isclose(
            float(trend["second_half_mean_depth"]), second_mean, rel_tol=0.0, abs_tol=1e-12
        )
        and math.isclose(
            float(trend["slope_requests_per_second"]), slope, rel_tol=0.0, abs_tol=1e-12
        )
        and math.isclose(
            float(trend["offered_rate_per_second"]), offered_rate, rel_tol=0.0, abs_tol=1e-12
        )
        and math.isclose(
            float(trend["completed_rate_per_second"]),
            completed_rate,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        and trend["sustained_negative"] is sustained
        and trend["completed_rate_exceeds_offered"] is excess
        and trend["passed"] is (sustained and excess)
        and recovery.get("queue_drain_pass") is sustained
        and recovery.get("two_gpu_excess_capacity_pass") is excess
    ):
        raise ValueError("v10 raw recovery flags differ from sustained trend/rate recomputation")
    return start, end


def _build_timeline(
    *, serving: dict[str, Any], rollout: dict[str, Any], config: dict[str, Any]
) -> tuple[V10Timeline, V10AssessmentWindows]:
    trigger = serving.get("reclamation_trigger_evidence")
    recovery = serving.get("serving_recovery_evidence")
    enable = serving.get("serving_enable_ack")
    restore_cutover = serving.get("restore_route_cutover")
    gpu1_drained = serving.get("gpu1_drained")
    restore = serving.get("restore_start")
    if not all(
        isinstance(item, dict)
        for item in (trigger, recovery, enable, restore_cutover, gpu1_drained, restore)
    ):
        raise ValueError("v10 serving barriers are incomplete")
    assert isinstance(trigger, dict)
    assert isinstance(recovery, dict)
    assert isinstance(enable, dict)
    assert isinstance(restore_cutover, dict)
    assert isinstance(gpu1_drained, dict)
    assert isinstance(restore, dict)
    if (
        int(gpu1_drained.get("running_requests", -1)) != 0
        or int(gpu1_drained.get("waiting_requests", -1)) != 0
        or int(gpu1_drained.get("skipped_waiting_requests", -1)) != 0
        or int(gpu1_drained.get("scheduler_request_count", -1)) != 0
        or int(gpu1_drained.get("queue_depth", -1)) != 0
        or gpu1_drained.get("runtime_state_source") != "live-vllm-0.23-scheduler"
        or int(gpu1_drained.get("restore_cutover_sequence", -1))
        != int(restore_cutover["restore_cutover_sequence"])
        or int(gpu1_drained.get("last_admitted_sequence", -1))
        >= int(restore_cutover["restore_cutover_sequence"])
    ):
        raise ValueError("v10 GPU1 drain evidence is incomplete or crosses the route cutover")
    stability_rows = recovery.get("stability_windows")
    if not isinstance(stability_rows, list) or not stability_rows:
        raise ValueError("v10 recovery omitted stability windows")
    first_useful = int(recovery["gpu1_first_useful_ns"])
    stability_start = int(stability_rows[0]["start_ns"])
    stability_end = int(stability_rows[-1]["end_ns"])
    queue_trend_start, queue_trend_end = _validate_raw_recovery_trend(recovery, config=config)
    if queue_trend_end != stability_start:
        raise ValueError("v10 SLO stability began before the sustained queue-drain gate completed")
    evaluated = int(recovery["evaluated_ns"])
    restore_trigger = int(restore["observed_ns"])
    continuation_end = int(rollout["rollout_continuation_complete_ns"])

    timestamps: dict[V10Phase, tuple[int, str]] = {
        V10Phase.CONTROL_STABLE: (
            int(serving["spike_start_ns"]),
            "analysis:control-window-p95-and-throughput-confirmed",
        ),
        V10Phase.LOAD_SPIKE_BEGIN: (
            int(serving["spike_start_ns"]),
            "serving/result.json#spike_start_ns",
        ),
        V10Phase.GPU0_OVERLOAD_CONFIRMED: (
            int(trigger["window_end_ns"]),
            "serving/result.json#reclamation_trigger_evidence.window_end_ns",
        ),
        V10Phase.RECLAIM_TRIGGER: (
            int(trigger["triggered_ns"]),
            "barriers/v10-reclaim-trigger.json#triggered_ns",
        ),
        V10Phase.ROLLOUT_ADMISSION_STOP: (
            _phase_timestamp(rollout, "ROLLOUT_ADMISSION_STOP"),
            "rollout/result.json#phase_events/ROLLOUT_ADMISSION_STOP",
        ),
        V10Phase.BRANCH_QUIESCE_BEGIN: (
            _phase_timestamp(rollout, "BRANCH_QUIESCE_BEGIN"),
            "rollout/result.json#phase_events/BRANCH_QUIESCE_BEGIN",
        ),
        V10Phase.BRANCH_QUIESCE_END: (
            _phase_timestamp(rollout, "BRANCH_QUIESCE_END"),
            "rollout/result.json#phase_events/BRANCH_QUIESCE_END",
        ),
        V10Phase.STATE_CAPTURE_BEGIN: (
            _phase_timestamp(rollout, "STATE_CAPTURE_BEGIN"),
            "rollout/result.json#phase_events/STATE_CAPTURE_BEGIN",
        ),
        V10Phase.STATE_CAPTURE_END: (
            _phase_timestamp(rollout, "STATE_CAPTURE_END"),
            "rollout/result.json#phase_events/STATE_CAPTURE_END",
        ),
        V10Phase.STATE_TRANSFORM_BEGIN: (
            _phase_timestamp(rollout, "STATE_TRANSFORM_BEGIN"),
            "rollout/result.json#phase_events/STATE_TRANSFORM_BEGIN",
        ),
        V10Phase.STATE_TRANSFORM_END: (
            _phase_timestamp(rollout, "STATE_TRANSFORM_END"),
            "rollout/result.json#phase_events/STATE_TRANSFORM_END",
        ),
        V10Phase.D2H_BEGIN: (
            _phase_timestamp(rollout, "D2H_BEGIN"),
            "rollout/result.json#phase_events/D2H_BEGIN",
        ),
        V10Phase.D2H_END: (
            _phase_timestamp(rollout, "D2H_END"),
            "rollout/result.json#phase_events/D2H_END",
        ),
        V10Phase.INTEGRITY_BEGIN: (
            _phase_timestamp(rollout, "INTEGRITY_BEGIN"),
            "rollout/result.json#phase_events/INTEGRITY_BEGIN",
        ),
        V10Phase.INTEGRITY_END: (
            _phase_timestamp(rollout, "INTEGRITY_END"),
            "rollout/result.json#phase_events/INTEGRITY_END",
        ),
        V10Phase.STATE_PUBLISH: (
            _phase_timestamp(rollout, "STATE_PUBLISH"),
            "rollout/result.json#phase_events/STATE_PUBLISH",
        ),
        V10Phase.GPU1_STATE_RELEASE_BEGIN: (
            _phase_timestamp(rollout, "GPU1_STATE_RELEASE_BEGIN"),
            "rollout/result.json#phase_events/GPU1_STATE_RELEASE_BEGIN",
        ),
        V10Phase.GPU1_STATE_RELEASE_END: (
            _phase_timestamp(rollout, "GPU1_STATE_RELEASE_END"),
            "rollout/result.json#phase_events/GPU1_STATE_RELEASE_END",
        ),
        V10Phase.GPU1_HBM_RECLAIM_CONFIRMED: (
            _phase_timestamp(rollout, "GPU1_HBM_RECLAIM_CONFIRMED"),
            "rollout/result.json#phase_events/GPU1_HBM_RECLAIM_CONFIRMED",
        ),
        V10Phase.GPU1_SERVING_ENABLE: (
            int(enable["observed_ns"]),
            "barriers/v10-serving-enable-ack.json#observed_ns",
        ),
        V10Phase.GPU1_FIRST_USEFUL_SERVING_REQUEST: (
            first_useful,
            "barriers/v10-gpu1-first-useful.json#first_token_ns",
        ),
        V10Phase.SERVING_QUEUE_DRAIN_BEGIN: (
            queue_trend_start,
            "serving/result.json#serving_recovery_evidence.queue_trend.window_start_ns",
        ),
        V10Phase.SERVING_SLO_RECOVERY_BEGIN: (
            first_useful,
            "serving/result.json#serving_recovery_evidence.gpu1_first_useful_ns",
        ),
        V10Phase.SERVING_QUEUE_DRAIN_END: (
            queue_trend_end,
            "serving/result.json#serving_recovery_evidence.queue_trend.window_end_ns",
        ),
        V10Phase.SERVING_SLO_RESTORED: (
            stability_start,
            "serving/result.json#serving_recovery_evidence.stability_windows[0].start_ns",
        ),
        V10Phase.SERVING_SLO_STABILITY_BEGIN: (
            stability_start,
            "serving/result.json#serving_recovery_evidence.stability_windows[0].start_ns",
        ),
        V10Phase.SERVING_SLO_STABILITY_END: (
            stability_end,
            "serving/result.json#serving_recovery_evidence.stability_windows[-1].end_ns",
        ),
        V10Phase.TWO_GPU_SERVICE_STABLE: (
            evaluated,
            "serving/result.json#serving_recovery_evidence.evaluated_ns",
        ),
        V10Phase.RESTORE_LOAD_REDUCED: (
            int(restore_cutover["restore_start_ns"]),
            "barriers/v10-restore-route-cutover.json#restore_start_ns",
        ),
        V10Phase.GPU1_SERVING_DRAINED: (
            int(gpu1_drained["observed_ns"]),
            "barriers/v10-gpu1-drained.json#observed_ns",
        ),
        V10Phase.RESTORE_ELIGIBLE: (
            restore_trigger,
            "barriers/v10-restore-start.json#observed_ns",
        ),
        V10Phase.RESTORE_TRIGGER: (
            restore_trigger,
            "barriers/v10-restore-start.json#observed_ns",
        ),
        V10Phase.H2D_BEGIN: (
            _phase_timestamp(rollout, "H2D_BEGIN"),
            "rollout/result.json#phase_events/H2D_BEGIN",
        ),
        V10Phase.H2D_END: (
            _phase_timestamp(rollout, "H2D_END"),
            "rollout/result.json#phase_events/H2D_END",
        ),
        V10Phase.STATE_IMPORT_BEGIN: (
            _phase_timestamp(rollout, "STATE_IMPORT_BEGIN"),
            "rollout/result.json#phase_events/STATE_IMPORT_BEGIN",
        ),
        V10Phase.DESTINATION_NATIVE_WRITE_BEGIN: (
            _phase_timestamp(rollout, "DESTINATION_NATIVE_WRITE_BEGIN"),
            "rollout/result.json#phase_events/DESTINATION_NATIVE_WRITE_BEGIN",
        ),
        V10Phase.STATE_VALIDATE_BEGIN: (
            _phase_timestamp(rollout, "STATE_VALIDATE_BEGIN"),
            "rollout/result.json#phase_events/STATE_VALIDATE_BEGIN:first",
        ),
        V10Phase.STATE_VALIDATE_END: (
            _phase_timestamp(rollout, "STATE_VALIDATE_END", last=True),
            "rollout/result.json#phase_events/STATE_VALIDATE_END:last",
        ),
        V10Phase.DESTINATION_NATIVE_WRITE_END: (
            _phase_timestamp(rollout, "DESTINATION_NATIVE_WRITE_END"),
            "rollout/result.json#phase_events/DESTINATION_NATIVE_WRITE_END",
        ),
        V10Phase.STATE_IMPORT_END: (
            _phase_timestamp(rollout, "STATE_IMPORT_END"),
            "rollout/result.json#phase_events/STATE_IMPORT_END",
        ),
        V10Phase.BRANCH_RESUME_BEGIN: (
            _phase_timestamp(rollout, "BRANCH_RESUME_BEGIN"),
            "rollout/result.json#phase_events/BRANCH_RESUME_BEGIN",
        ),
        V10Phase.FIRST_RESUMED_TOKEN: (
            _phase_timestamp(rollout, "FIRST_RESUMED_TOKEN"),
            "rollout/result.json#phase_events/FIRST_RESUMED_TOKEN",
        ),
        V10Phase.ALL_BRANCHES_RESUMED: (
            int(rollout["all_branches_resumed_ns"]),
            "rollout/result.json#all_branches_resumed_ns",
        ),
        V10Phase.ROLLOUT_CONTINUATION_COMPLETE: (
            continuation_end,
            "rollout/result.json#rollout_continuation_complete_ns",
        ),
    }
    timeline = V10Timeline(events=tuple(_event(phase, *timestamps[phase]) for phase in V10Phase))
    windows = V10AssessmentWindows(
        control=_control_assessment_interval(serving=serving, config=config),
        overload=PhaseInterval(
            name="v10-gpu0-overload",
            start_ns=int(serving["spike_start_ns"]),
            end_ns=int(trigger["window_end_ns"]),
        ),
        two_gpu_excess_capacity=PhaseInterval(
            name="v10-two-gpu-capacity", start_ns=first_useful, end_ns=evaluated
        ),
        queue_drain=PhaseInterval(
            name="v10-queue-drain", start_ns=queue_trend_start, end_ns=queue_trend_end
        ),
        slo_stability=PhaseInterval(
            name="v10-slo-stability", start_ns=stability_start, end_ns=stability_end
        ),
        restore_activity=PhaseInterval(
            name="v10-restore-activity", start_ns=restore_trigger, end_ns=continuation_end
        ),
    )
    return timeline, windows


def _control_assessment_interval(
    *, serving: dict[str, Any], config: dict[str, Any]
) -> PhaseInterval:
    """Exclude only the predeclared empty-queue fill from control assessment."""

    raw_warmup = config.get("warmup_seconds")
    if isinstance(raw_warmup, bool) or not isinstance(raw_warmup, (int, float)):
        raise ValueError("v10 control assessment requires a numeric warmup_seconds")
    warmup_seconds = float(raw_warmup)
    if not math.isfinite(warmup_seconds) or not math.isclose(
        warmup_seconds, 1.0, rel_tol=0.0, abs_tol=1e-12
    ):
        raise ValueError("integrated v10 control assessment requires its predeclared 1s warmup")
    start_ns = int(serving["start_ns"])
    spike_start_ns = int(serving["spike_start_ns"])
    assessment_start_ns = start_ns + round(warmup_seconds * NS_PER_SECOND)
    if assessment_start_ns >= spike_start_ns:
        raise ValueError("v10 control warmup leaves no assessment interval")
    return PhaseInterval(
        name="v10-control",
        start_ns=assessment_start_ns,
        end_ns=spike_start_ns,
    )


def _build_serving(
    *, config: dict[str, Any], serving: dict[str, Any], rollout: dict[str, Any]
) -> tuple[ServingWorkload, tuple[ServingObservation, ...]]:
    offered = tuple(serving.get("global_offered_requests", ()))
    raw_rows = tuple(serving.get("requests", ())) + tuple(
        rollout.get("temporary_serving", {}).get("requests", ())
    )
    offered_by_id = {str(row["request_id"]): row for row in offered}
    observed_by_id = {str(row["request_id"]): row for row in raw_rows}
    if (
        not offered
        or len(offered_by_id) != len(offered)
        or len(observed_by_id) != len(raw_rows)
        or set(offered_by_id) != set(observed_by_id)
    ):
        raise ValueError("v10 global offered plan and observations are not exact bijections")
    actual_offered_ns = tuple(int(row["offered_ns"]) for row in offered)
    scheduled_ns = tuple(int(row["scheduled_arrival_ns"]) for row in offered)
    if any(
        actual < scheduled
        for actual, scheduled in zip(actual_offered_ns, scheduled_ns, strict=True)
    ):
        raise ValueError("v10 producer reported an arrival before its deterministic schedule")
    maximum_lateness_ns = max(
        actual - scheduled
        for actual, scheduled in zip(actual_offered_ns, scheduled_ns, strict=True)
    )
    # The producer is permitted one scheduler quantum of jitter, but not
    # catch-up bursts. This is independent of the deterministic plan and is
    # checked before offered-rate evidence is accepted.
    if maximum_lateness_ns > _MAXIMUM_ARRIVAL_LATENESS_NS:
        raise ValueError("v10 producer arrival lateness exceeded the 100ms bound")
    ordered = tuple(
        sorted(offered, key=lambda row: (int(row["scheduled_arrival_ns"]), row["request_id"]))
    )
    output_tokens = int(config["serving_output_tokens"])
    if output_tokens != 64:
        raise ValueError("v10 raw workload did not request exactly 64 output tokens")
    requests: list[ServingRequest] = []
    observations: list[ServingObservation] = []
    for sequence, offered_row in enumerate(ordered):
        request_id = str(offered_row["request_id"])
        row = observed_by_id[request_id]
        if str(row["device"]) != str(offered_row["device"]):
            raise ValueError(f"v10 request routing differs from plan: {request_id}")
        token_ids = tuple(int(item) for item in row["output_token_ids"])
        token_times = tuple(int(item) for item in row["token_timestamps_ns"])
        if len(token_ids) != output_tokens or len(token_times) != output_tokens:
            raise ValueError(f"v10 request did not emit exactly 64 tokens: {request_id}")
        arrival = int(offered_row["scheduled_arrival_ns"])
        requests.append(
            ServingRequest(
                sequence=sequence,
                request_id=request_id,
                phase=str(offered_row["phase"]),
                arrival_ns=arrival,
                prompt_tokens=int(config["serving_prompt_tokens"]),
                requested_output_tokens=output_tokens,
            )
        )
        observations.append(
            ServingObservation(
                request_id=request_id,
                arrival_ns=arrival,
                service_start_ns=int(row["service_start_ns"]),
                token_timestamps_ns=token_times,
                completion_ns=int(row["completed_ns"]),
                outcome=ObservationOutcome.COMPLETED,
                device=str(row["device"]),
            )
        )
    start_ns = min(row.arrival_ns for row in requests)
    end_ns = max(row.completion_ns for row in observations) + 1
    phase_order = ("control", "gpu0-overload", "two-gpu-recovery", "restore-interference")
    arrivals_by_phase = {
        phase: tuple(
            int(row["scheduled_arrival_ns"]) for row in ordered if str(row["phase"]) == phase
        )
        for phase in phase_order
    }
    if any(not arrivals_by_phase[phase] for phase in phase_order):
        raise ValueError("v10 raw workload omitted a required serving phase")
    phase_starts = tuple(min(arrivals_by_phase[phase]) for phase in phase_order)
    if phase_starts != tuple(sorted(phase_starts)):
        raise ValueError("v10 serving phases do not follow the global plan")
    declared_phases: list[ArrivalPhase] = []
    for index, phase in enumerate(phase_order):
        phase_arrivals = arrivals_by_phase[phase]
        start = phase_starts[index]
        phase_end = phase_starts[index + 1] if index + 1 < len(phase_order) else end_ns
        deltas = tuple(right - left for left, right in pairwise(phase_arrivals))
        declared_phases.append(
            ArrivalPhase(
                name=phase,
                start_offset_ns=start - start_ns,
                end_offset_ns=phase_end - start_ns,
                interarrival_ns=min(deltas) if deltas else max(1, phase_end - start),
                first_arrival_offset_ns=min(phase_arrivals) - start,
            )
        )
    workload = ServingWorkload(
        workload_id=_canonical_sha256(
            {"attempt_id": config["attempt_id"], "offered": list(offered)}
        ),
        start_ns=start_ns,
        end_ns=end_ns,
        config=ServingSpikeConfig(
            seed=int(config["seed"]),
            control_phase="control",
            spike_phase="gpu0-overload",
            phases=tuple(declared_phases),
            prompt_tokens=WeightedTokenDistribution(
                values=(int(config["serving_prompt_tokens"]),), weights=(1,)
            ),
            output_tokens=WeightedTokenDistribution(values=(64,), weights=(1,)),
            request_id_prefix="exp004-v10-observed",
        ),
        requests=tuple(requests),
    )
    return workload, tuple(observations)


def _bounded_backlog_evidence(
    *,
    config: dict[str, Any],
    serving: dict[str, Any],
    timeline: V10Timeline,
    workload: ServingWorkload,
    observations: tuple[ServingObservation, ...],
) -> BoundedBacklogEvidence:
    trigger = serving.get("reclamation_trigger_evidence")
    recovery = serving.get("serving_recovery_evidence")
    if not isinstance(trigger, dict) or not isinstance(recovery, dict):
        raise ValueError("v10 bounded-backlog evidence is absent")
    trigger_observation_ns = int(trigger["window_end_ns"])
    interval = PhaseInterval(
        name="v10-bounded-backlog",
        start_ns=timeline.timestamp(V10Phase.LOAD_SPIKE_BEGIN),
        end_ns=timeline.timestamp(V10Phase.GPU1_FIRST_USEFUL_SERVING_REQUEST),
    )
    by_request = {row.request_id: row for row in observations}
    if len(by_request) != len(observations) or set(by_request) != {
        row.request_id for row in workload.requests
    }:
        raise ValueError("v10 bounded-backlog calculation requires complete observations")
    event_times = {
        interval.start_ns,
        interval.end_ns,
        trigger_observation_ns,
        *(
            row.arrival_ns
            for row in workload.requests
            if interval.start_ns <= row.arrival_ns <= interval.end_ns
        ),
        *(
            timestamp
            for row in observations
            for timestamp in (row.completion_ns, max(interval.start_ns, row.completion_ns - 1))
            if interval.start_ns <= timestamp <= interval.end_ns
        ),
    }
    maximum_depth = max(
        outstanding_queue_depth_at(workload, by_request, timestamp) for timestamp in event_times
    )
    trigger_depth = outstanding_queue_depth_at(workload, by_request, trigger_observation_ns)
    first_useful_depth = outstanding_queue_depth_at(workload, by_request, interval.end_ns)
    configured_trigger = int(config["serving_overload_queue_trigger"])
    configured_abort = int(config["serving_overload_queue_abort"])
    if (
        trigger.get("overload_confirmed") is not True
        or int(trigger.get("queue_trigger", -1)) != configured_trigger
        or int(trigger.get("queue_abort", -1)) != configured_abort
        or int(trigger.get("queue_depth_end", -1)) != trigger_depth
        or int(recovery.get("queue_depth_at_gpu1_first_useful", -1)) != first_useful_depth
    ):
        raise ValueError("v10 raw and recomputed bounded-backlog evidence disagree")
    preferred_minimum = 10
    preferred_maximum = 25
    passed = (
        preferred_minimum <= configured_trigger <= preferred_maximum
        and configured_trigger <= trigger_depth <= preferred_maximum
        and trigger_depth <= maximum_depth <= configured_abort
        and first_useful_depth <= maximum_depth
    )
    return BoundedBacklogEvidence(
        interval=interval,
        trigger_observation_ns=trigger_observation_ns,
        trigger_queue_depth=trigger_depth,
        maximum_queue_depth=maximum_depth,
        queue_depth_at_gpu1_first_useful=first_useful_depth,
        configured_trigger_depth=configured_trigger,
        configured_abort_depth=configured_abort,
        preferred_minimum_depth=preferred_minimum,
        preferred_maximum_depth=preferred_maximum,
        passed=passed,
    )


def _actual_arrival_evidence(*, config: dict[str, Any], serving: dict[str, Any]) -> dict[str, Any]:
    offered = tuple(serving["global_offered_requests"])
    configured_rates = {
        "control": float(config["gpu0_control_request_rate_per_second"]),
        "gpu0-overload": float(config["serving_spike_request_rate_per_second"]),
        "two-gpu-recovery": float(config["serving_spike_request_rate_per_second"]),
        "restore-interference": float(config["gpu0_restore_request_rate_per_second"]),
    }
    phases: list[dict[str, Any]] = []
    for phase, configured_rate in configured_rates.items():
        rows = tuple(row for row in offered if str(row["phase"]) == phase)
        actual = tuple(int(row["offered_ns"]) for row in rows)
        scheduled = tuple(int(row["scheduled_arrival_ns"]) for row in rows)
        if len(actual) < 2 or actual != tuple(sorted(actual)):
            raise ValueError(f"v10 actual arrival evidence is incomplete for {phase}")
        deltas = tuple(right - left for left, right in pairwise(actual))
        expected_interval = NS_PER_SECOND / configured_rate
        observed_duration_seconds = (actual[-1] - actual[0]) / NS_PER_SECOND
        observed_rate = (len(actual) - 1) / observed_duration_seconds
        rate_error = abs(observed_rate - configured_rate) / configured_rate
        finite_sample_tolerance = max(
            _ARRIVAL_RATE_TOLERANCE_FRACTION,
            1.0 / (configured_rate * observed_duration_seconds),
        )
        minimum_delta = min(deltas)
        maximum_burst = 0
        right = 0
        for left, timestamp in enumerate(actual):
            while right < len(actual) and actual[right] < timestamp + _ARRIVAL_BURST_WINDOW_NS:
                right += 1
            maximum_burst = max(maximum_burst, right - left)
        maximum_allowed_burst = (
            math.ceil(configured_rate * _ARRIVAL_BURST_WINDOW_NS / NS_PER_SECOND) + 2
        )
        maximum_lateness = max(
            observed - planned for observed, planned in zip(actual, scheduled, strict=True)
        )
        passed = (
            rate_error <= finite_sample_tolerance
            and maximum_burst <= maximum_allowed_burst
            and maximum_lateness <= _MAXIMUM_ARRIVAL_LATENESS_NS
        )
        phases.append(
            {
                "phase": phase,
                "configured_rate_per_second": configured_rate,
                "observed_rate_per_second": observed_rate,
                "relative_rate_error": rate_error,
                "effective_rate_tolerance_fraction": finite_sample_tolerance,
                "sample_count": len(actual),
                "minimum_interarrival_ns": minimum_delta,
                "expected_interarrival_ns": expected_interval,
                "burst_window_ns": _ARRIVAL_BURST_WINDOW_NS,
                "maximum_arrivals_in_burst_window": maximum_burst,
                "maximum_allowed_arrivals_in_burst_window": maximum_allowed_burst,
                "maximum_schedule_lateness_ns": maximum_lateness,
                "passed": passed,
            }
        )
    if not all(row["passed"] for row in phases):
        failed = tuple(row for row in phases if not row["passed"])
        raise ValueError(f"v10 actual arrival cadence/rate evidence failed: {failed!r}")
    return {
        "schema_version": "sloforge.branchfabric.v10-actual-arrival-evidence/v1",
        "maximum_allowed_lateness_ns": _MAXIMUM_ARRIVAL_LATENESS_NS,
        "rate_tolerance_fraction": _ARRIVAL_RATE_TOLERANCE_FRACTION,
        "phases": phases,
        "passed": True,
    }


def _movement_record_label(record_id: str) -> str:
    parts = record_id.split(":", 2)
    if len(parts) != 3 or not parts[0].isdigit() or not parts[1]:
        raise ValueError(f"v10 movement record ID has an invalid worker format: {record_id!r}")
    return parts[1]


def _validate_fixed_v10_movement_shape(report: StateMovementReport) -> dict[str, Any]:
    """Prove the complete fixed 8-branch naive pass graph is present once.

    v10 has one shared-root segment plus eight private segments.  Every one of
    the 30 measured operations must cover each segment exactly once.  Checking
    only aggregate bytes would let a missing pass hide behind a duplicated
    pass with the same byte count, so this gate also validates range coverage
    and every raw/derived identity.
    """

    segments = report.logical_segments
    passes = report.passes
    edges = report.edges
    if len(segments) != _V10_MOVEMENT_SEGMENT_COUNT:
        raise ValueError(
            "v10 movement ledger must contain one shared and eight private logical segments"
        )
    shared = tuple(item for item in segments if item.branch_id is None)
    private = tuple(item for item in segments if item.branch_id is not None)
    if len(shared) != 1 or len(private) != 8:
        raise ValueError("v10 movement segment ownership is not the fixed 1+8 shape")
    if len({item.branch_id for item in private}) != 8:
        raise ValueError("v10 movement private segment branch identities are not injective")
    if len(passes) != _V10_MOVEMENT_RECORD_COUNT:
        raise ValueError(
            f"v10 movement ledger must contain exactly {_V10_MOVEMENT_RECORD_COUNT} records"
        )

    record_ids = tuple(item.record_id for item in passes)
    if len(set(record_ids)) != len(record_ids):
        raise ValueError("v10 movement ledger contains duplicate record IDs")
    explicit_event_ids = tuple(
        event_id
        for item in passes
        for event_id in (item.read_event_id, item.write_event_id, item.transfer_event_id)
        if event_id is not None
    )
    if len(set(explicit_event_ids)) != len(explicit_event_ids):
        raise ValueError("v10 movement ledger contains duplicate physical event IDs")

    records_by_label: dict[str, list[Any]] = {}
    for item in passes:
        records_by_label.setdefault(_movement_record_label(item.record_id), []).append(item)
    if frozenset(records_by_label) != _V10_MOVEMENT_LABELS:
        missing = sorted(_V10_MOVEMENT_LABELS - records_by_label.keys())
        unexpected = sorted(records_by_label.keys() - _V10_MOVEMENT_LABELS)
        raise ValueError(
            f"v10 movement operation labels differ from the fixed naive path; "
            f"missing={missing!r}, unexpected={unexpected!r}"
        )

    segment_bytes = {item.segment_id: item.logical_bytes for item in segments}
    logical_state_bytes = sum(segment_bytes.values())
    label_coverage: dict[str, int] = {}
    for label, records in records_by_label.items():
        if len(records) != _V10_MOVEMENT_SEGMENT_COUNT:
            raise ValueError(f"v10 movement label {label!r} must have exactly nine records")
        covered = 0
        for segment_id, expected_bytes in segment_bytes.items():
            ranges = sorted(
                (
                    item.logical_offset_bytes,
                    item.logical_offset_bytes + item.logical_bytes,
                )
                for item in records
                if item.state_segment == segment_id
            )
            cursor = 0
            for start, end in ranges:
                if start != cursor or end <= start:
                    raise ValueError(
                        f"v10 movement label {label!r} has missing/overlapping ranges "
                        f"for segment {segment_id!r}"
                    )
                cursor = end
            if cursor != expected_bytes:
                raise ValueError(
                    f"v10 movement label {label!r} does not cover segment {segment_id!r}"
                )
            covered += cursor
        if covered != logical_state_bytes:
            raise ValueError(f"v10 movement label {label!r} does not cover logical state once")
        label_coverage[label] = covered

    edge_ids = tuple(item.edge_id for item in edges)
    if len(set(edge_ids)) != len(edge_ids):
        raise ValueError("v10 movement graph contains duplicate edge IDs")
    derived_edge_count = len(segments) + sum(
        int(item.bytes_read > 0)
        + int(item.bytes_written > 0)
        + int(item.transfer_bytes > 0)
        + int(item.checksum_bytes > 0)
        for item in passes
    )
    if len(edges) != derived_edge_count:
        raise ValueError("v10 movement graph edge count does not match raw pass events")
    if len(edges) != _V10_MOVEMENT_EDGE_COUNT:
        raise ValueError(
            f"v10 movement graph must contain exactly {_V10_MOVEMENT_EDGE_COUNT} edges"
        )
    if report.accounting.pass_count != len(passes):
        raise ValueError("v10 movement accounting pass count differs from raw records")
    if report.accounting.movement_edge_count != len(edges):
        raise ValueError("v10 movement accounting edge count differs from the graph")
    if report.accounting.logical_state_bytes != logical_state_bytes:
        raise ValueError("v10 movement denominator differs from the nine raw segments")
    return {
        "logical_segment_count": len(segments),
        "shared_segment_count": len(shared),
        "private_segment_count": len(private),
        "operation_label_count": len(records_by_label),
        "operation_labels": sorted(records_by_label),
        "record_count": len(passes),
        "physical_event_id_count": len(explicit_event_ids),
        "graph_edge_count": len(edges),
        "derived_graph_edge_count": derived_edge_count,
        "logical_bytes_covered_per_label": label_coverage,
        "passed": True,
    }


def _enriched_state_passes(
    *,
    report: StateMovementReport,
    operation_telemetry: dict[str, Any],
) -> list[dict[str, Any]]:
    """Attach processor/timing/semantic classifications to every raw pass.

    A single measured CUDA operation can implement several logical segment
    records. Its event time is therefore repeated with an explicit shared-time
    semantic and must not be summed across records.
    """

    operations = operation_telemetry.get("cuda_operations")
    if not isinstance(operations, list):
        raise ValueError("v10 state-pass enrichment lacks CUDA operation records")
    result: list[dict[str, Any]] = []
    for row in report.passes:
        gpu_memory = "gpu" in row.source_memory.value or "gpu" in row.destination_memory.value
        processor = "GPU" if gpu_memory or row.transfer_direction.value != "none" else "CPU"
        matching = [
            operation
            for operation in operations
            if int(operation["cpu_start_monotonic_ns"]) <= row.end_ns
            and int(operation["cpu_end_monotonic_ns"]) >= row.start_ns
        ]
        cuda_time_ns = (
            sum(int(operation["cuda_event_elapsed_ns"]) for operation in matching)
            if processor == "GPU" and matching
            else None
        )
        if processor == "GPU" and cuda_time_ns is None:
            # Allocation-lifetime records and host-side scheduler metadata may
            # name a GPU/host buffer without launching a CUDA operation.
            cuda_time_semantics = "not-applicable-no-cuda-kernel-or-copy-in-pass"
        elif cuda_time_ns is None:
            cuda_time_semantics = "not-applicable-cpu-pass"
        else:
            cuda_time_semantics = (
                "shared-underlying-cuda-operation-time-do-not-sum-across-segment-records"
            )
        if row.required_unavoidable:
            avoidability = "required-unavoidable"
            semantic_requirement = (
                "required by the frozen transport, correctness, or fresh-destination contract"
            )
        else:
            avoidability = "avoidable-naive-baseline"
            semantic_requirement = (
                "current naive materialization or validation implementation; not a semantic minimum"
            )
        if not row.required_unavoidable:
            fusibility = "fusible-candidate"
        elif row.operation.value in {"d2h", "h2d", "write"}:
            fusibility = "boundary-operation-not-removable"
        elif row.operation.value in {"checksum", "validate"}:
            fusibility = "fusible-only-with-equivalent-integrity-proof"
        else:
            fusibility = "fusible-with-adjacent-pass-if-semantics-preserved"
        result.append(
            {
                **row.model_dump(mode="json"),
                "stage": row.record_id.split(":", 2)[1],
                "processor": processor,
                "cpu_process": "rollout-worker",
                "gpu": row.device if processor == "GPU" else None,
                "wall_time_ns": row.end_ns - row.start_ns,
                "cuda_time_ns": cuda_time_ns,
                "cuda_operation_ids": [str(operation["operation_id"]) for operation in matching],
                "cuda_time_semantics": cuda_time_semantics,
                "temporary_buffer": {
                    "bytes": row.temporary_allocation_bytes,
                    "memory": (
                        None
                        if row.temporary_allocation_memory is None
                        else row.temporary_allocation_memory.value
                    ),
                    "allocation_id": row.temporary_allocation_id,
                },
                "semantic_requirement": semantic_requirement,
                "avoidability_classification": avoidability,
                "fusibility_classification": fusibility,
            }
        )
    if len(result) != len(report.passes):
        raise AssertionError("state-pass enrichment lost records")
    return result


def _movement_artifact(
    rollout: dict[str, Any],
    *,
    operation_telemetry: dict[str, Any],
) -> tuple[dict[str, Any], MovementAccountingEvidence]:
    parsed = StateMovementReport.model_validate(rollout["movement_report"])
    report = build_state_movement_report(
        logical_segments=parsed.logical_segments, passes=parsed.passes
    )
    if report != parsed:
        raise ValueError("v10 movement report differs from raw-pass recomputation")
    _validate_independent_movement_telemetry(report, rollout)
    shape_evidence = _validate_fixed_v10_movement_shape(report)
    accounting = report.accounting
    avoidable = sum(
        row.bytes_read + row.bytes_written + row.transfer_bytes
        for row in report.passes
        if not row.required_unavoidable
    )
    external = accounting.d2h_bytes + accounting.h2d_bytes
    evidence = MovementAccountingEvidence(
        logical_state_bytes=accounting.logical_state_bytes,
        full_physical_touch_bytes=accounting.amplification_numerator_bytes,
        external_movement_bytes=external,
        avoidable_movement_bytes=avoidable,
        critical_path_movement_bytes=accounting.amplification_numerator_bytes,
        accounting_duplicate_bytes=0,
        all_physical_passes_recorded=True,
        formulas_recomputed_from_raw_artifacts=True,
    )
    source_gpu_reads = sum(
        row.bytes_read
        for row in report.passes
        if row.source_memory is MemoryDomain.SOURCE_GPU_NATIVE_PAGED
    )
    source_gpu_writes = sum(
        row.bytes_written
        for row in report.passes
        if row.destination_memory is MemoryDomain.SOURCE_GPU_NATIVE_PAGED
    )
    destination_gpu_reads = sum(
        row.bytes_read
        for row in report.passes
        if row.source_memory is MemoryDomain.DESTINATION_GPU_NATIVE_PAGED
    )
    destination_gpu_writes = sum(
        row.bytes_written
        for row in report.passes
        if row.destination_memory is MemoryDomain.DESTINATION_GPU_NATIVE_PAGED
    )
    host_domains = {
        MemoryDomain.PINNED_HOST_TRANSPORT,
        MemoryDomain.PAGEABLE_HOST_BUFFER,
        MemoryDomain.HOST_TRANSPORT_STORE,
        MemoryDomain.HOST_INTEGRITY_METADATA,
        MemoryDomain.HOST_RUNTIME_METADATA,
    }
    host_intermediate_reads = sum(
        row.bytes_read for row in report.passes if row.source_memory in host_domains
    )
    host_intermediate_writes = sum(
        row.bytes_written for row in report.passes if row.destination_memory in host_domains
    )
    if host_intermediate_reads + host_intermediate_writes != accounting.host_intermediate_bytes:
        raise ValueError("v10 host intermediate read/write split differs from accounting total")
    artifact = {
        "schema_version": "sloforge.branchfabric.experiment-004-v10-movement-accounting/v1",
        "logical_state_bytes": accounting.logical_state_bytes,
        "unique_real_d2h_bytes": accounting.d2h_bytes,
        "unique_real_h2d_bytes": accounting.h2d_bytes,
        "source_gpu_reads": source_gpu_reads,
        "source_gpu_writes": source_gpu_writes,
        "host_intermediate_reads": host_intermediate_reads,
        "host_intermediate_writes": host_intermediate_writes,
        "host_intermediate_reads_writes": accounting.host_intermediate_bytes,
        "destination_gpu_reads": destination_gpu_reads,
        "destination_gpu_writes": destination_gpu_writes,
        "gpu_intermediate_reads_writes": accounting.gpu_intermediate_bytes,
        "checksum_integrity_bytes": accounting.checksum_bytes,
        "integrity_read_bytes": accounting.checksum_bytes,
        "temporary_allocation_bytes": {
            "host": accounting.host_temporary_allocation_bytes,
            "gpu": accounting.gpu_temporary_allocation_bytes,
            "peak_host": accounting.peak_host_temporary_allocation_bytes,
        },
        "accounting_duplicate_bytes": 0,
        "full_physical_touch_bytes": evidence.full_physical_touch_bytes,
        "external_movement_bytes": external,
        "avoidable_movement_bytes": avoidable,
        "avoidable_movement_semantics": (
            "conservative lower bound: sum of endpoint/link bytes for raw passes marked "
            "not required_unavoidable by the frozen naive implementation"
        ),
        "required_movement_bytes": evidence.full_physical_touch_bytes - avoidable,
        "critical_path_movement_bytes": evidence.critical_path_movement_bytes,
        "critical_path_movement_semantics": (
            "all fixed naive state-pass bytes: the GPU1 worker synchronously awaits every "
            "capture, transform, transfer, integrity, import, native-write, and validation "
            "operation before the dependent state transition can complete"
        ),
        "amplification": {
            "full_physical_touch": (
                evidence.full_physical_touch_bytes / evidence.logical_state_bytes
            ),
            "external_movement": external / evidence.logical_state_bytes,
            "avoidable_movement": avoidable / evidence.logical_state_bytes,
            "critical_path_movement": (
                evidence.critical_path_movement_bytes / evidence.logical_state_bytes
            ),
        },
        "all_physical_passes_recorded": True,
        "formulas_recomputed_from_raw_artifacts": True,
        "fixed_naive_pass_graph_evidence": shape_evidence,
        "state_pass_records": _enriched_state_passes(
            report=report, operation_telemetry=operation_telemetry
        ),
        "raw_state_movement_report": report.model_dump(mode="json"),
    }
    return artifact, evidence


def assess_v10_directory(
    *,
    work_root: Path,
    config: dict[str, Any],
    controller: dict[str, Any],
    budget: BudgetEvidence,
    cleanup: CleanupEvidence,
    prerequisites: V10PrerequisiteEvidence | None = None,
) -> tuple[V10ScientificValidity, dict[str, Any], dict[str, Any]]:
    """Build all three required v10 analysis artifacts from raw files."""

    serving = _load_object(work_root / "serving/result.json")
    rollout = _load_object(work_root / "rollout/result.json")
    if serving.get("status") != "succeeded" or rollout.get("status") != "succeeded":
        raise ValueError("v10 workers did not both succeed")
    if serving.get("methodology") != "v10-global-capacity":
        raise ValueError("serving worker did not execute the explicit v10 methodology")
    compilation_evidence = _validate_measured_transaction_compilation(
        serving=serving, rollout=rollout
    )
    timeline, windows = _build_timeline(serving=serving, rollout=rollout, config=config)
    workload, observations = _build_serving(config=config, serving=serving, rollout=rollout)
    bounded_backlog = _bounded_backlog_evidence(
        config=config,
        serving=serving,
        timeline=timeline,
        workload=workload,
        observations=observations,
    )
    arrival_evidence = _actual_arrival_evidence(config=config, serving=serving)
    operation_telemetry = _load_object(
        work_root / "rollout/telemetry/cuda-and-host-operations.json"
    )
    movement_artifact, movement_evidence = _movement_artifact(
        rollout, operation_telemetry=operation_telemetry
    )
    state_pass_path = work_root / "state-passes/state-pass-records.json"
    state_pass_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = state_pass_path.with_name(f".{state_pass_path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(
        (
            json.dumps(
                {
                    "schema_version": "sloforge.branchfabric.experiment-004-v10-state-passes/v1",
                    "attempt_id": config["attempt_id"],
                    "records": movement_artifact["state_pass_records"],
                },
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode()
    )
    os.replace(temporary, state_pass_path)
    manifest = CanonicalKvTransportManifest.model_validate(rollout["transport_manifest"])
    semantics = _branch_semantics(config=config, rollout=rollout, manifest=manifest)
    continuation = semantics.continuation_token_counts
    branch_resume = BranchResumeEvidence(
        expected_first_tokens={
            str(key): int(value) for key, value in rollout["expected_first_tokens"].items()
        },
        observed_first_tokens={
            str(key): int(value) for key, value in rollout["first_resumed_tokens"].items()
        },
        continuation_token_counts=continuation,
        restored_into_fresh_allocations=bool(
            rollout.get("source_destination_fresh")
            and rollout.get("runtime_incarnations_fresh")
            and rollout.get("restored_runtime_identity_injective")
        ),
        integrity_valid=bool(rollout.get("transport_integrity_valid")),
        semantic_continuity=semantics,
    )
    released = rollout.get("source_block_release_summary", {})
    memory_before = rollout.get("source_memory_before_release", {})
    memory_after = rollout.get("source_memory_after_release", {})
    logical_bytes = int(rollout["logical_state_bytes"])
    state = StateCorrectnessEvidence(
        logical_state_bytes=logical_bytes,
        preserved_state_bytes=int(rollout["kv_logical_state_bytes"]),
        source_blocks_expected=int(rollout["block_count"]),
        source_blocks_released=int(released.get("exact_source_block_count", -1)),
        source_kv_assigned_bytes_before=int(memory_before.get("kv_assigned_bytes", -1)),
        source_kv_assigned_bytes_after=int(memory_after.get("kv_assigned_bytes", -1)),
        source_kv_pool_reserved_bytes_before=int(memory_before.get("kv_pool_reserved_bytes", -1)),
        source_kv_pool_reserved_bytes_after=int(memory_after.get("kv_pool_reserved_bytes", -1)),
        source_kv_unassigned_bytes_before=int(memory_before.get("kv_unassigned_bytes", -1)),
        source_kv_unassigned_bytes_after=int(memory_after.get("kv_unassigned_bytes", -1)),
        transport_integrity_valid=bool(rollout.get("transport_integrity_valid")),
        fresh_destination_allocations=bool(
            rollout.get("source_destination_fresh")
            and rollout.get("runtime_incarnations_fresh")
            and rollout.get("restored_runtime_identity_injective")
        ),
    )
    returncodes = controller.get("worker_returncodes", {})
    controller_cleanup = (
        controller.get("status") == "succeeded"
        and set(returncodes) == {"serving", "rollout"}
        and set(returncodes.values()) == {0}
        and not controller.get("compute_processes_after")
        and config.get("tracing_level") != "full"
    )
    if cleanup.passed and not controller_cleanup:
        raise ValueError("provider cleanup evidence conflicts with failed controller cleanup")
    validity_config = V10ValidityConfig(
        control_offered_rate_per_second=float(config["gpu0_control_request_rate_per_second"]),
        spike_offered_rate_per_second=float(config["serving_spike_request_rate_per_second"]),
        restore_offered_rate_per_second=float(config["gpu0_restore_request_rate_per_second"]),
        lambda_1_rps=float(config["lambda_1_rps"]),
        overload_queue_trigger_depth=int(config["serving_overload_queue_trigger"]),
        overload_queue_abort_depth=int(config["serving_overload_queue_abort"]),
        maximum_p95_ttft_ns=int(float(config["serving_slo_maximum_ttft_seconds"]) * NS_PER_SECOND),
        recovery_queue_depth_threshold=int(config["serving_recovery_queue_threshold"]),
        evaluation_window_ns=int(
            float(config["serving_recovery_evaluation_seconds"]) * NS_PER_SECOND
        ),
        stability_window_ns=int(
            float(config["serving_slo_stability_window_seconds"]) * NS_PER_SECOND
        ),
    )
    scientific = assess_v10_scientific_validity(
        config=validity_config,
        timeline=timeline,
        windows=windows,
        workload=workload,
        observations=observations,
        gpu0_device="gpu0",
        gpu1_device="gpu1",
        bounded_backlog=bounded_backlog,
        state_correctness=state,
        branch_resume=branch_resume,
        movement_accounting=movement_evidence,
        budget=budget,
        cleanup=cleanup,
        prerequisites=prerequisites,
    )
    serving_recovery = {
        "schema_version": "sloforge.branchfabric.experiment-004-v10-serving-recovery/v1",
        "timeline": timeline.model_dump(mode="json"),
        "assessment_windows": windows.model_dump(mode="json"),
        "reclamation_trigger_evidence": serving["reclamation_trigger_evidence"],
        "bounded_backlog_evidence": bounded_backlog.model_dump(mode="json"),
        "serving_recovery_evidence": serving["serving_recovery_evidence"],
        "global_offered_request_count": len(workload.requests),
        "complete_observation_count": len(observations),
        "output_tokens_per_request": 64,
        "routing_plan_observation_bijection": True,
        "actual_arrival_evidence": arrival_evidence,
        "measured_transaction_compilation": {
            "both_engines_compilation_free": True,
            "by_role": compilation_evidence,
        },
    }
    return scientific, movement_artifact, serving_recovery


__all__ = ["assess_v10_directory"]
