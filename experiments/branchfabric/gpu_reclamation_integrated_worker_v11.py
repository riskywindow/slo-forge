"""Two-role production worker for Experiment 004 optimized-preserve v11.

The state path in this module is isolated from the frozen v10 implementation.
The serving clock, sanity probes, routing, bounded overload trigger, recovery
gate, and restore handoff are imported unchanged from the frozen v10 runtime.
GPU libraries are imported only after :func:`run_worker` validates the sealed
configuration and the assigned physical device.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
import time
import traceback
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

EXPECTED_LOGICAL_STATE_BYTES = 1_056_964_608
EXPECTED_TOTAL_BLOCKS = 1_152
EXPECTED_SHARED_BLOCKS = 1_024
EXPECTED_PRIVATE_BLOCKS = 128
FROZEN_V10_EXECUTION_COMMIT = "c5fa2f169e8a343ef7a86ebc814b3a23a7de6b39"
FROZEN_V10_ANALYSIS_COMMIT = "1c51853e10809686d4368037153927f25e834117"
FROZEN_V10_TAG = "branchfabric-exp004-naive-baseline-v10"
_V11_POST_J_SANITY_PROBE_COUNT = 3


def _canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n"
    ).encode()


def _write_new(path: Path, value: Any) -> None:
    """Create one immutable barrier or artifact without replacement."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(_canonical_bytes(value))
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


_V11_TRANSACTION_COMMAND_COMMON_FIELDS = frozenset(
    {
        "schema_version",
        "effective_config",
        "selection_sha256",
        "authorization_artifact_hash",
        "sanity_result_sha256",
        "issued_at_monotonic_ns",
    }
)
_V11_TARGETED_TRANSACTION_SAFETY = {
    "terminal_phase": "SOURCE_ALLOCATION_IDENTITY_GATE",
    "full_export_authorized": False,
    "source_release_authorized": False,
}
_V11_TRANSACTION_ENVELOPES = {
    "integrated-reclamation-v11": (
        "sloforge.branchfabric.experiment-004-v11-integrated-config/v1",
        "sloforge.branchfabric.integrated-transaction-command/v1",
        False,
    ),
    "targeted-source-identity-v11": (
        "sloforge.branchfabric.experiment-004-v11-targeted-source-identity-config/v1",
        "sloforge.branchfabric.targeted-source-identity-command/v1",
        True,
    ),
}


def _validate_v11_transaction_command(
    *,
    base_config: dict[str, Any],
    command: dict[str, Any],
    selected_load_sha256: str,
    frozen_validator: Any,
) -> tuple[dict[str, Any], dict[str, str]]:
    """Validate the typed v11 envelope before delegating frozen common checks."""

    mode = base_config.get("execution_mode")
    envelope = _V11_TRANSACTION_ENVELOPES.get(mode)
    if envelope is None:
        raise ValueError("v11 transaction execution mode is missing or unsupported")
    expected_base_schema, expected_command_schema, targeted = envelope
    if base_config.get("schema_version") != expected_base_schema:
        raise ValueError("v11 transaction base schema differs from its execution mode")
    expected_fields = set(_V11_TRANSACTION_COMMAND_COMMON_FIELDS)
    if targeted:
        expected_fields.update(_V11_TARGETED_TRANSACTION_SAFETY)
    if set(command) != expected_fields:
        raise ValueError("v11 transaction command fields differ from its typed envelope")
    if command.get("schema_version") != expected_command_schema:
        raise ValueError("v11 transaction command schema differs from its execution mode")
    if targeted:
        effective_config = command.get("effective_config")
        if not isinstance(effective_config, dict):
            raise ValueError("targeted transaction command lacks effective config")
        for field, expected in _V11_TARGETED_TRANSACTION_SAFETY.items():
            command_value = command.get(field)
            base_value = base_config.get(field)
            effective_value = effective_config.get(field)
            if (
                type(command_value) is not type(expected)
                or command_value != expected
                or type(base_value) is not type(expected)
                or base_value != expected
                or type(effective_value) is not type(expected)
                or effective_value != expected
            ):
                raise ValueError(f"targeted transaction command changed safety field {field}")

    delegated = dict(command)
    if targeted:
        delegated["schema_version"] = "sloforge.branchfabric.integrated-transaction-command/v1"
        for field in _V11_TARGETED_TRANSACTION_SAFETY:
            del delegated[field]
    return frozen_validator(
        base_config=base_config,
        command=delegated,
        selected_load_sha256=selected_load_sha256,
    )


def _run_v11_integrated_calibration_phase(
    *,
    frozen_worker: Any,
    adapter: Any,
    config: dict[str, Any],
    inputs: dict[str, Any],
    role: str,
    physical_gpu_uuid: str,
    barrier_root: Path,
    model_load_started_ns: int,
    model_ready_ns: int,
) -> tuple[dict[str, Any], Any, dict[str, Any]]:
    """Run frozen calibration with one process-local, typed-v11 validator binding."""

    original = frozen_worker._validate_integrated_transaction_command
    original_probe_count = frozen_worker._V10_SANITY_GUARD_COUNT
    if original_probe_count != 2:
        raise RuntimeError("frozen v10 sanity probe count drifted from two")

    def validate_bound_command(**kwargs: Any) -> tuple[dict[str, Any], dict[str, str]]:
        return _validate_v11_transaction_command(
            **kwargs,
            frozen_validator=original,
        )

    frozen_worker._validate_integrated_transaction_command = validate_bound_command
    frozen_worker._V10_SANITY_GUARD_COUNT = _V11_POST_J_SANITY_PROBE_COUNT
    try:
        return frozen_worker._run_integrated_calibration_phase(
            adapter=adapter,
            config=config,
            inputs=inputs,
            role=role,
            physical_gpu_uuid=physical_gpu_uuid,
            barrier_root=barrier_root,
            model_load_started_ns=model_load_started_ns,
            model_ready_ns=model_ready_ns,
        )
    finally:
        frozen_worker._validate_integrated_transaction_command = original
        frozen_worker._V10_SANITY_GUARD_COUNT = original_probe_count


def _source_allocator_lifecycle_evidence(
    adapter: Any,
    source_commit: Any,
    *,
    start_sequence: int,
    rollout_ready_sequence: int,
) -> dict[str, Any]:
    """Bind the bounded native allocator journal to committed logical pages."""

    events = adapter._allocator_epoch_source.lifecycle_events()
    if not 0 <= start_sequence <= rollout_ready_sequence <= len(events):
        raise RuntimeError("allocator lifecycle phase boundaries are invalid")
    logical_page_by_lifetime = {
        (item.physical_block_id, item.allocation_epoch): item.logical_page_id
        for item in source_commit.allocations
    }
    rows = []
    for event in events[start_sequence:]:
        rows.append(
            {
                "sequence": event.sequence,
                "observed_at_monotonic_ns": event.observed_at_monotonic_ns,
                "event": event.event,
                "block_id": event.block_id,
                "allocation_epoch": event.allocation_epoch,
                "logical_page_id": logical_page_by_lifetime.get(
                    (event.block_id, event.allocation_epoch)
                ),
                "runtime_request_id": event.runtime_request_id,
                "old_state": event.old_state,
                "new_state": event.new_state,
                "caller_label": event.caller_label,
                "process_id": event.process_id,
                "thread_id": event.thread_id,
                "phase": (
                    "ROLLOUT_CREATE_TO_READY"
                    if event.sequence < rollout_ready_sequence
                    else "ROLLOUT_READY_TO_SOURCE_CAPTURE_COMMIT"
                ),
            }
        )
    payload = {
        "schema_version": "sloforge.branchfabric.v11-allocator-lifecycle-journal/v1",
        "bounded_maximum_event_count": 262_144,
        "start_sequence": start_sequence,
        "rollout_ready_sequence": rollout_ready_sequence,
        "capture_commit_sequence": len(events),
        "event_count": len(rows),
        "source_lifetime_event_count": sum(row["logical_page_id"] is not None for row in rows),
        "events": rows,
    }
    payload["sha256"] = hashlib.sha256(_canonical_bytes(payload)).hexdigest()
    return payload


def frozen_v10_methodology_identity() -> dict[str, Any]:
    """Return provenance for the exact imported v10 global-serving source."""

    source = Path(__file__).with_name("gpu_reclamation_v10_serving.py").resolve(strict=True)
    return {
        "schema_version": "sloforge.branchfabric.frozen-v10-methodology-binding/v1",
        "module": "gpu_reclamation_v10_serving",
        "source_path": str(source),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "execution_commit": FROZEN_V10_EXECUTION_COMMIT,
        "analysis_commit": FROZEN_V10_ANALYSIS_COMMIT,
        "annotated_tag": FROZEN_V10_TAG,
        "mutation_performed": False,
    }


def integrated_v11_trigger_overlay_identity() -> dict[str, Any]:
    """Bind the isolated controller fix without relabeling frozen v10 source."""

    source = (
        Path(__file__).with_name("gpu_reclamation_integrated_trigger_v11.py").resolve(strict=True)
    )
    return {
        "schema_version": "sloforge.branchfabric.integrated-v11-trigger-binding/v1",
        "module": "gpu_reclamation_integrated_trigger_v11",
        "source_path": str(source),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "frozen_v10_source_mutated": False,
        "normal_trigger_overlay_applied": True,
    }


def expanded_runtime_config(config: Any) -> dict[str, Any]:
    """Translate the strict v11 config into the frozen v10 worker vocabulary."""

    payload = config.model_dump(mode="json")
    payload.update(
        {
            "gpu0_control_request_rate_per_second": config.control_request_rate_rps,
            "serving_spike_request_rate_per_second": config.lambda_spike_rps,
            "gpu0_restore_request_rate_per_second": config.restore_request_rate_rps,
            "gpu0_overload_probe_seconds": config.overload_probe_seconds,
            "serving_slo_stability_window_seconds": config.serving_stability_seconds,
            "serving_recovery_evaluation_seconds": config.recovery_evaluation_seconds,
            "serving_recovery_queue_threshold": config.recovery_queue_threshold,
            "serving_maximum_pending_requests": config.maximum_pending_requests,
            "serving_restore_handoff_lead_requests": config.restore_handoff_lead_requests,
            "serving_overload_queue_trigger": config.overload_queue_trigger,
            "serving_overload_queue_abort": config.overload_queue_abort,
            "temporary_serving_seconds": config.temporary_serving_seconds,
            "authorization_artifact_hash": config.budget_authorization_sha256,
            "disable_log_stats": False,
        }
    )
    return payload


def build_live_v10_config(config: Any) -> Any:
    """Construct the immutable v10 global-clock contract from v11 config."""

    from gpu_reclamation_v10_serving import LiveV10Config

    result = LiveV10Config.from_mapping(expanded_runtime_config(config))
    exact = (
        result.control_rate_per_second == 9.0
        and result.spike_rate_per_second == 15.0
        and result.restore_rate_per_second == 9.0
        and result.baseline_seconds == 5.0
        and result.overload_probe_seconds == 3.0
        and result.recovery_stability_seconds == 5.0
        and result.overload_queue_trigger == 20
        and result.overload_queue_abort == 64
    )
    if not exact:
        raise RuntimeError("integrated v11 drifted from the frozen v10 final methodology")
    return result


def validate_sanity_guard_pair(
    assessments: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Bind exactly one three-second 12-rps and 15-rps guard, in order."""

    rows = tuple(dict(item) for item in assessments)
    expected = (("sanity_12rps_stable", 12.0), ("sanity_15rps_overload", 15.0))
    if len(rows) != 2:
        raise ValueError("integrated v11 requires exactly two short sanity guards")
    for row, (guard, rate) in zip(rows, expected, strict=True):
        signals = row.get("signals")
        if (
            row.get("schema_version") != "sloforge.branchfabric.v10-sanity-guard-assessment/v1"
            or row.get("guard") != guard
            or row.get("expected_rate_rps") != rate
            or row.get("measurement_seconds") != 3.0
            or row.get("passed") is not True
            or not isinstance(signals, dict)
        ):
            raise RuntimeError(f"integrated v11 {guard} failed closed")
        if rate == 12.0 and any(
            signals.get(name) is not True
            for name in (
                "completion_tracks_offer",
                "no_persistent_positive_drift",
                "p95_ttft_below_two_seconds",
                "bounded_backlog",
            )
        ):
            raise RuntimeError("integrated v11 sanity_12rps_stable lacks its stable-load signals")
        if rate == 15.0 and (
            signals.get("bounded_backlog") is not True
            or not (
                signals.get("positive_queue_slope") is True
                or signals.get("completed_rate_below_offered") is True
            )
        ):
            raise RuntimeError(
                "integrated v11 sanity_15rps_overload lacks queue growth or a service deficit"
            )
    return {
        "schema_version": "sloforge.branchfabric.v11-sanity-guard-pair/v1",
        "broad_capacity_calibration_performed": False,
        "assessments": rows,
        "sha256": hashlib.sha256(_canonical_bytes(rows)).hexdigest(),
        "passed": True,
    }


def validate_reclaim_trigger_timeline(
    trigger: Mapping[str, Any],
    *,
    rollout_admission_stop_ns: int,
    state_quiescence_ns: int,
) -> dict[str, Any]:
    """Require the emitted trigger to causally precede rollout quiescence."""

    events = trigger.get("events")
    if not isinstance(events, dict):
        raise RuntimeError("integrated v11 trigger lacks controller event timestamps")
    overload_ns = events.get("OVERLOAD_DETECTED")
    predicate_ns = events.get("TRIGGER_PREDICATE_SATISFIED")
    emitted_ns = events.get("RECLAIM_TRIGGER_EMITTED")
    stamps = (overload_ns, predicate_ns, emitted_ns)
    if any(isinstance(value, bool) or not isinstance(value, int) for value in stamps):
        raise RuntimeError("integrated v11 trigger timestamps are malformed")
    assert isinstance(overload_ns, int)
    assert isinstance(predicate_ns, int)
    assert isinstance(emitted_ns, int)
    if not (
        overload_ns
        <= predicate_ns
        <= emitted_ns
        <= rollout_admission_stop_ns
        <= state_quiescence_ns
    ):
        raise RuntimeError("integrated v11 trigger/quiescence timeline is not causal")
    return {
        "schema_version": "sloforge.branchfabric.v11-trigger-timeline/v1",
        "events": {
            **events,
            "ROLLOUT_ADMISSION_STOP": rollout_admission_stop_ns,
            "STATE_QUIESCENCE": state_quiescence_ns,
        },
        "controller_reaction_latency_ns": emitted_ns - predicate_ns,
        "rollout_stop_reaction_latency_ns": rollout_admission_stop_ns - emitted_ns,
        "trigger_precedes_state_quiescence": True,
        "passed": True,
    }


def _p95_ns(values: Sequence[int]) -> float | None:
    ordered = sorted(values)
    if not ordered:
        return None
    position = (len(ordered) - 1) * 0.95
    low = int(position)
    high = min(len(ordered) - 1, low + 1)
    fraction = position - low
    return ordered[low] * (1.0 - fraction) + ordered[high] * fraction


def _control_interval_evidence(
    result: Mapping[str, Any],
    *,
    expected_rate_rps: float,
    expected_duration_seconds: float,
    slo_ttft_seconds: float,
    warmup_seconds: float = 1.0,
    evaluation_seconds: float = 1.0,
    minimum_completion_fraction: float = 0.90,
) -> dict[str, Any]:
    full_start_ns = result.get("start_ns")
    end_ns = result.get("spike_start_ns")
    requests = result.get("requests")
    if (
        isinstance(full_start_ns, bool)
        or not isinstance(full_start_ns, int)
        or isinstance(end_ns, bool)
        or not isinstance(end_ns, int)
        or end_ns <= full_start_ns
        or not isinstance(requests, (tuple, list))
    ):
        raise RuntimeError("integrated v11 lacks a bounded control interval")
    full_duration_seconds = (end_ns - full_start_ns) / 1e9
    if abs(full_duration_seconds - expected_duration_seconds) > 1e-9:
        raise RuntimeError("integrated v11 control interval duration drifted")
    if (
        not 0.0 < warmup_seconds < expected_duration_seconds
        or evaluation_seconds <= 0.0
        or not 0.0 < minimum_completion_fraction <= 1.0
    ):
        raise RuntimeError("integrated v11 control assessment parameters are invalid")

    start_ns = full_start_ns + round(warmup_seconds * 1e9)
    duration_seconds = (end_ns - start_ns) / 1e9
    if abs(duration_seconds - (expected_duration_seconds - warmup_seconds)) > 1e-9:
        raise RuntimeError("integrated v11 control assessment duration drifted")
    timestamped_rows = tuple(
        row
        for row in requests
        if isinstance(row, dict)
        and isinstance(row.get("scheduled_arrival_ns"), int)
        and not isinstance(row.get("scheduled_arrival_ns"), bool)
    )
    full_control_rows = tuple(
        row
        for row in timestamped_rows
        if row.get("phase") == "control"
        and full_start_ns <= int(row["scheduled_arrival_ns"]) < end_ns
    )
    rows = tuple(
        row for row in full_control_rows if start_ns <= int(row["scheduled_arrival_ns"]) < end_ns
    )
    expected_full_arrivals = round(expected_rate_rps * expected_duration_seconds)
    expected_arrivals = round(expected_rate_rps * duration_seconds)
    completions_in_interval = sum(
        isinstance(row.get("completed_ns"), int)
        and not isinstance(row.get("completed_ns"), bool)
        and start_ns <= int(row["completed_ns"]) < end_ns
        for row in full_control_rows
    )
    completed_full_cohort = sum(
        isinstance(row.get("completed_ns"), int) and not isinstance(row.get("completed_ns"), bool)
        for row in full_control_rows
    )
    completed_cohort = sum(
        isinstance(row.get("completed_ns"), int) and not isinstance(row.get("completed_ns"), bool)
        for row in rows
    )
    request_ids = tuple(row.get("request_id") for row in full_control_rows)
    unique_control_request_ids = bool(
        all(isinstance(request_id, str) and request_id for request_id in request_ids)
        and len(set(request_ids)) == len(request_ids)
    )
    ordered_control_rows = tuple(
        sorted(full_control_rows, key=lambda row: int(row["scheduled_arrival_ns"]))
    )
    scheduled_arrivals = tuple(int(row["scheduled_arrival_ns"]) for row in ordered_control_rows)
    unique_scheduled_arrivals = len(set(scheduled_arrivals)) == len(scheduled_arrivals)
    expected_scheduled_arrivals = tuple(
        full_start_ns + int(index * 1e9 / expected_rate_rps)
        for index in range(expected_full_arrivals)
    )
    scheduled_arrival_cadence_exact = scheduled_arrivals == expected_scheduled_arrivals
    monotonic_control_timestamps = all(
        isinstance(row.get("service_start_ns"), int)
        and not isinstance(row.get("service_start_ns"), bool)
        and isinstance(row.get("first_token_ns"), int)
        and not isinstance(row.get("first_token_ns"), bool)
        and isinstance(row.get("completed_ns"), int)
        and not isinstance(row.get("completed_ns"), bool)
        and int(row["scheduled_arrival_ns"])
        <= int(row["service_start_ns"])
        <= int(row["first_token_ns"])
        <= int(row["completed_ns"])
        for row in full_control_rows
    )
    full_output_tokens_exact = all(
        isinstance(row.get("output_token_ids"), (tuple, list))
        and len(row["output_token_ids"]) == 64
        for row in full_control_rows
    )
    ttfts = tuple(
        int(row["first_token_ns"]) - int(row["scheduled_arrival_ns"])
        for row in rows
        if isinstance(row.get("first_token_ns"), int)
        and not isinstance(row.get("first_token_ns"), bool)
    )
    evaluation_ns = round(evaluation_seconds * 1e9)
    timestamps = list(range(start_ns, end_ns, evaluation_ns))
    if not timestamps or timestamps[-1] != end_ns:
        timestamps.append(end_ns)
    if len(timestamps) < 3:
        raise RuntimeError("integrated v11 control queue trend lacks boundary samples")
    depths = tuple(
        sum(
            int(row["scheduled_arrival_ns"]) <= timestamp
            and (
                not isinstance(row.get("completed_ns"), int)
                or isinstance(row.get("completed_ns"), bool)
                or int(row["completed_ns"]) > timestamp
            )
            for row in full_control_rows
        )
        for timestamp in timestamps
    )
    seconds = tuple((timestamp - timestamps[0]) / 1e9 for timestamp in timestamps)
    mean_x = sum(seconds) / len(seconds)
    mean_y = sum(depths) / len(depths)
    denominator = sum((value - mean_x) ** 2 for value in seconds)
    slope = (
        sum(
            (x_value - mean_x) * (y_value - mean_y)
            for x_value, y_value in zip(seconds, depths, strict=True)
        )
        / denominator
    )
    midpoint = len(depths) // 2
    first_half_mean = sum(depths[:midpoint]) / len(depths[:midpoint])
    second_half_mean = sum(depths[midpoint:]) / len(depths[midpoint:])
    sustained_positive = bool(
        slope > 0.0 and depths[-1] > depths[0] and second_half_mean > first_half_mean
    )

    queue_changes: dict[int, int] = {}
    for row in full_control_rows:
        arrival_ns = int(row["scheduled_arrival_ns"])
        leaves_queue_ns = row.get("service_start_ns")
        if isinstance(leaves_queue_ns, bool) or not isinstance(leaves_queue_ns, int):
            leaves_queue_ns = row.get("completed_ns")
        if isinstance(leaves_queue_ns, bool) or not isinstance(leaves_queue_ns, int):
            continue
        queue_changes[arrival_ns] = queue_changes.get(arrival_ns, 0) + 1
        queue_changes[leaves_queue_ns] = queue_changes.get(leaves_queue_ns, 0) - 1
    waiting_depth = sum(delta for timestamp, delta in queue_changes.items() if timestamp < start_ns)
    maximum_waiting_depth = waiting_depth
    for timestamp in sorted(
        timestamp for timestamp in queue_changes if start_ns <= timestamp < end_ns
    ):
        waiting_depth += queue_changes[timestamp]
        if waiting_depth < 0:
            raise RuntimeError("integrated v11 control waiting-queue accounting is negative")
        maximum_waiting_depth = max(maximum_waiting_depth, waiting_depth)

    offered_rate = len(rows) / duration_seconds
    completed_rate = completions_in_interval / duration_seconds
    p95 = _p95_ns(ttfts)
    terminal_depth = depths[-1] if depths else expected_arrivals
    maximum_depth = max(depths, default=expected_arrivals)
    offered_rate_matches = abs(offered_rate - expected_rate_rps) <= 1e-12
    completion_tracks_offer = completed_rate >= offered_rate * minimum_completion_fraction
    complete_full_request_accounting = completed_full_cohort == len(full_control_rows)
    complete_request_accounting = len(ttfts) == len(rows) and completed_cohort == len(rows)
    queue_stable = not sustained_positive
    waiting_queue_bounded = maximum_waiting_depth < 20
    total_outstanding_bounded = maximum_depth < 20
    passed = bool(
        len(full_control_rows) == expected_full_arrivals
        and len(rows) == expected_arrivals
        and unique_control_request_ids
        and unique_scheduled_arrivals
        and scheduled_arrival_cadence_exact
        and monotonic_control_timestamps
        and full_output_tokens_exact
        and complete_full_request_accounting
        and complete_request_accounting
        and offered_rate_matches
        and completion_tracks_offer
        and p95 is not None
        and p95 <= slo_ttft_seconds * 1e9
        and queue_stable
        and waiting_queue_bounded
        and total_outstanding_bounded
    )
    evidence = {
        "schema_version": "sloforge.branchfabric.v11-control-interval/v2",
        "full_start_ns": full_start_ns,
        "start_ns": start_ns,
        "end_ns": end_ns,
        "full_duration_seconds": full_duration_seconds,
        "warmup_seconds": warmup_seconds,
        "duration_seconds": duration_seconds,
        "expected_full_arrivals": expected_full_arrivals,
        "full_arrivals": len(full_control_rows),
        "expected_arrivals": expected_arrivals,
        "arrivals": len(rows),
        "eventual_full_cohort_completions": completed_full_cohort,
        "eventual_cohort_completions": completed_cohort,
        "completions_in_interval": completions_in_interval,
        "offered_rate_per_second": offered_rate,
        "completed_rate_per_second": completed_rate,
        "minimum_completion_fraction": minimum_completion_fraction,
        "p95_ttft_ns": p95,
        "unique_control_request_ids": unique_control_request_ids,
        "unique_scheduled_arrivals": unique_scheduled_arrivals,
        "scheduled_arrival_cadence_exact": scheduled_arrival_cadence_exact,
        "monotonic_control_timestamps": monotonic_control_timestamps,
        "full_cohort_output_tokens_exact": full_output_tokens_exact,
        "complete_full_request_accounting": complete_full_request_accounting,
        "complete_request_accounting": complete_request_accounting,
        "offered_rate_matches_config": offered_rate_matches,
        "completion_tracks_offer": completion_tracks_offer,
        "p95_ttft_below_slo": p95 is not None and p95 <= slo_ttft_seconds * 1e9,
        "waiting_queue_maximum_depth": maximum_waiting_depth,
        "waiting_queue_bounded_below_normal_trigger": waiting_queue_bounded,
        "total_outstanding_diagnostic": {
            "sample_interval_ns": evaluation_ns,
            "samples": [
                {"timestamp_ns": timestamp, "total_outstanding": depth}
                for timestamp, depth in zip(timestamps, depths, strict=True)
            ],
            "initial_depth": depths[0],
            "final_depth": terminal_depth,
            "maximum_depth": maximum_depth,
            "first_half_mean_depth": first_half_mean,
            "second_half_mean_depth": second_half_mean,
            "slope_requests_per_second": slope,
            "sustained_positive": sustained_positive,
        },
        "total_outstanding_bounded_below_normal_trigger": total_outstanding_bounded,
        "outstanding_queue_non_positive_trend": queue_stable,
        "in_service_requests_not_treated_as_waiting_queue": True,
        "passed": passed,
    }
    return evidence


def _non_overlapping_partition(
    *, chain: str, boundaries: Sequence[tuple[str, int, int]]
) -> dict[str, Any]:
    if not boundaries:
        raise ValueError("critical-path partition cannot be empty")
    stages: list[dict[str, Any]] = []
    prior_end: int | None = None
    for name, start_ns, end_ns in boundaries:
        if (
            not name
            or isinstance(start_ns, bool)
            or not isinstance(start_ns, int)
            or isinstance(end_ns, bool)
            or not isinstance(end_ns, int)
            or end_ns < start_ns
            or (prior_end is not None and start_ns != prior_end)
        ):
            raise RuntimeError(f"integrated v11 {chain} stage partition is not contiguous")
        stages.append(
            {
                "stage": name,
                "start_ns": start_ns,
                "end_ns": end_ns,
                "wall_time_ns": end_ns - start_ns,
            }
        )
        prior_end = end_ns
    total = stages[-1]["end_ns"] - stages[0]["start_ns"]
    if sum(int(item["wall_time_ns"]) for item in stages) != total:
        raise RuntimeError(f"integrated v11 {chain} stage partition double counted time")
    return {
        "schema_version": "sloforge.branchfabric.v11-non-overlapping-critical-path/v1",
        "chain": chain,
        "start_ns": stages[0]["start_ns"],
        "end_ns": stages[-1]["end_ns"],
        "wall_time_ns": total,
        "stages": stages,
        "overlap_double_count_ns": 0,
        "passed": True,
    }


def _validate_gpu0_result(
    result: dict[str, Any],
    *,
    runtime_queue_samples: Sequence[Mapping[str, int]] = (),
    restore_complete: Mapping[str, Any] | None = None,
    expected_control_rate_rps: float = 9.0,
    expected_control_seconds: float = 5.0,
    expected_restore_rate_rps: float = 9.0,
    slo_ttft_seconds: float = 2.0,
    warmup_seconds: float = 1.0,
    control_interval: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    recovery = result.get("serving_recovery_evidence")
    trigger = result.get("reclamation_trigger_evidence")
    requests = result.get("requests")
    if (
        not isinstance(recovery, dict)
        or not isinstance(trigger, dict)
        or not isinstance(requests, (tuple, list))
    ):
        raise RuntimeError("integrated v11 GPU0 lacks frozen serving evidence")
    required_recovery = (
        "two_gpu_excess_capacity_pass",
        "queue_drain_pass",
        "slo_restoration_pass",
        "pre_restore_stability_pass",
        "restore_eligible",
    )
    if trigger.get("overload_confirmed") is not True or any(
        recovery.get(field) is not True for field in required_recovery
    ):
        raise RuntimeError("integrated v11 failed the frozen overload/recovery gates")
    trigger_depth = trigger.get("queue_depth_at_trigger")
    trigger_threshold = trigger.get("queue_trigger")
    emergency_ceiling = trigger.get("queue_abort")
    if (
        isinstance(trigger_depth, bool)
        or not isinstance(trigger_depth, int)
        or trigger_threshold != 20
        or emergency_ceiling != 64
        or not 20 <= trigger_depth <= 30
        or trigger.get("positive_queue_slope") is not True
        or trigger.get("offered_rate_exceeds_completed_rate") is not True
    ):
        raise RuntimeError("integrated v11 normal trigger was not emitted at backlog 20..30")
    control = dict(
        control_interval
        if control_interval is not None
        else _control_interval_evidence(
            result,
            expected_rate_rps=expected_control_rate_rps,
            expected_duration_seconds=expected_control_seconds,
            slo_ttft_seconds=slo_ttft_seconds,
            warmup_seconds=warmup_seconds,
        )
    )
    if control.get("passed") is not True:
        raise RuntimeError("integrated v11 9-rps control interval was not stable")
    restore_rows = tuple(row for row in requests if row.get("phase") == "restore-interference")
    if not restore_rows or any(
        row.get("completed_ns") is None
        or row.get("first_token_ns") is None
        or len(row.get("output_token_ids", ())) != 64
        for row in restore_rows
    ):
        raise RuntimeError("GPU0 was not actively and completely serving during v11 restore")
    restore_start = result.get("restore_start")
    restore_interval_start_ns = (
        restore_start.get("observed_ns") if isinstance(restore_start, dict) else None
    )
    restore_interval_end_ns = (
        restore_complete.get("first_resumed_token_ns")
        if isinstance(restore_complete, Mapping)
        else None
    )
    if (
        isinstance(restore_interval_start_ns, bool)
        or not isinstance(restore_interval_start_ns, int)
        or isinstance(restore_interval_end_ns, bool)
        or not isinstance(restore_interval_end_ns, int)
        or restore_interval_end_ns <= restore_interval_start_ns
    ):
        raise RuntimeError("integrated v11 lacks the actual GPU1 restore interval")
    all_rows = tuple(row for row in requests if isinstance(row, dict))
    interval_rows = tuple(
        row
        for row in restore_rows
        if restore_interval_start_ns <= int(row["scheduled_arrival_ns"]) < restore_interval_end_ns
    )
    interval_completions = sum(
        isinstance(row.get("completed_ns"), int)
        and not isinstance(row.get("completed_ns"), bool)
        and restore_interval_start_ns <= int(row["completed_ns"]) < restore_interval_end_ns
        for row in all_rows
    )
    interval_tokens = sum(
        restore_interval_start_ns <= int(timestamp) < restore_interval_end_ns
        for row in all_rows
        for timestamp in row.get("token_timestamps_ns", ())
        if isinstance(timestamp, int) and not isinstance(timestamp, bool)
    )
    interval_queue_samples = tuple(
        dict(item)
        for item in runtime_queue_samples
        if isinstance(item.get("observed_ns"), int)
        and not isinstance(item.get("observed_ns"), bool)
        and restore_interval_start_ns <= int(item["observed_ns"]) <= restore_interval_end_ns
    )
    interval_ttfts = tuple(
        int(row["first_token_ns"]) - int(row["scheduled_arrival_ns"]) for row in interval_rows
    )
    interval_seconds = (restore_interval_end_ns - restore_interval_start_ns) / 1e9
    interval = {
        "schema_version": "sloforge.branchfabric.v11-gpu0-restore-interference/v1",
        "start_ns": restore_interval_start_ns,
        "end_ns": restore_interval_end_ns,
        "duration_seconds": interval_seconds,
        "arrivals": len(interval_rows),
        "completions": interval_completions,
        "emitted_tokens": interval_tokens,
        "observed_arrival_rate_per_second": len(interval_rows) / interval_seconds,
        "completion_rate_per_second": interval_completions / interval_seconds,
        "token_rate_per_second": interval_tokens / interval_seconds,
        "ttft_sample_count": len(interval_ttfts),
        "p95_ttft_ns": _p95_ns(interval_ttfts),
        "live_vllm_queue_depth_samples": interval_queue_samples,
        "methodology_stall_absent": bool(
            interval_rows
            and interval_completions > 0
            and interval_tokens > 0
            and interval_queue_samples
        ),
    }
    interval["passed"] = interval["methodology_stall_absent"]
    if interval["passed"] is not True:
        raise RuntimeError("GPU0 was idle or stalled during the actual GPU1 restore interval")
    first = min(int(row["scheduled_arrival_ns"]) for row in restore_rows)
    last = max(int(row["completed_ns"]) for row in restore_rows)
    ttfts = sorted(
        int(row["first_token_ns"]) - int(row["scheduled_arrival_ns"]) for row in restore_rows
    )
    queue_samples: list[dict[str, int]] = []
    for timestamp in sorted(
        {
            *(int(row["scheduled_arrival_ns"]) for row in restore_rows),
            *(int(row["completed_ns"]) for row in restore_rows),
        }
    ):
        queue_samples.append(
            {
                "observed_ns": timestamp,
                "outstanding_requests": sum(
                    int(row["scheduled_arrival_ns"]) <= timestamp < int(row["completed_ns"])
                    for row in restore_rows
                ),
            }
        )
    p95_index = max(0, min(len(ttfts) - 1, (95 * len(ttfts) + 99) // 100 - 1))
    return {
        "active_during_restore_pass": True,
        "control_interval": control,
        "measured_restore_interval": interval,
        "arrivals": len(restore_rows),
        "completions": len(restore_rows),
        "emitted_tokens": sum(len(row["output_token_ids"]) for row in restore_rows),
        "offered_rps": expected_restore_rate_rps,
        "completion_rps": len(restore_rows) / max((last - first) / 1e9, 1e-9),
        "token_rps": sum(len(row["output_token_ids"]) for row in restore_rows)
        / max((last - first) / 1e9, 1e-9),
        "p95_ttft_ns": ttfts[p95_index],
        "maximum_queue_depth": max(row["outstanding_requests"] for row in queue_samples),
        "terminal_queue_depth": queue_samples[-1]["outstanding_requests"],
        "offered_outstanding_samples": queue_samples,
        "live_vllm_queue_depth_samples": tuple(dict(item) for item in runtime_queue_samples),
        "live_vllm_maximum_queue_depth": max(
            (int(item["queue_depth"]) for item in runtime_queue_samples), default=None
        ),
        "live_vllm_terminal_queue_depth": (
            int(runtime_queue_samples[-1]["queue_depth"]) if runtime_queue_samples else None
        ),
        "request_ids": tuple(str(row["request_id"]) for row in restore_rows),
    }


def run_integrated_v11_gpu0(
    engine: Any,
    *,
    adapter: Any,
    inputs: Mapping[str, Any],
    config: Any,
    start_ns: int,
    barriers: Path,
) -> dict[str, Any]:
    """Run the frozen 9->15-rps transaction and prove GPU0 restore activity."""

    from gpu_capacity_calibration_worker import _runtime_queue_state
    from gpu_reclamation_integrated_trigger_v11 import run_v11_gpu0_with_early_trigger
    from gpu_reclamation_worker import _sampling_params

    prefix = tuple(int(item) for item in inputs["prefix_token_ids"][: config.serving_prompt_tokens])
    runtime_queue_samples: list[dict[str, int]] = []

    def runtime_queue_state() -> dict[str, int]:
        state = _runtime_queue_state(adapter)
        if (barriers / "v10-restore-start.json").is_file():
            runtime_queue_samples.append({**state, "observed_ns": time.monotonic_ns()})
        return state

    class _PostStepRestoreSampler:
        """Sample the live scheduler after steps during the actual restore."""

        def __init__(self, wrapped: Any) -> None:
            self.wrapped = wrapped

        def __getattr__(self, name: str) -> Any:
            return getattr(self.wrapped, name)

        def step(self) -> Any:
            outputs = self.wrapped.step()
            if (barriers / "v10-restore-start.json").is_file():
                runtime_queue_state()
            return outputs

    serving = run_v11_gpu0_with_early_trigger(
        _PostStepRestoreSampler(engine),
        prefix=prefix,
        params=_sampling_params(max_tokens=config.serving_output_tokens, seed=config.seed),
        config=build_live_v10_config(config),
        start_ns=start_ns,
        barriers=barriers,
        write_new=_write_new,
        runtime_queue_state=runtime_queue_state,
    )
    _write_new(barriers / "v11-gpu0-serving-prevalidation.json", serving)
    control = _control_interval_evidence(
        serving,
        expected_rate_rps=config.control_request_rate_rps,
        expected_duration_seconds=config.baseline_seconds,
        slo_ttft_seconds=config.serving_slo_ttft_seconds,
        warmup_seconds=config.warmup_seconds,
    )
    _write_new(barriers / "v11-gpu0-control-interval.json", control)
    restore_complete_path = barriers / "rollout-restore-complete.json"
    if not restore_complete_path.is_file():
        raise RuntimeError("integrated v11 GPU0 lacks rollout restore completion evidence")
    restore_complete = json.loads(restore_complete_path.read_text())
    if not isinstance(restore_complete, dict):
        raise RuntimeError("integrated v11 rollout restore completion is malformed")
    interference = _validate_gpu0_result(
        serving,
        runtime_queue_samples=runtime_queue_samples,
        restore_complete=restore_complete,
        expected_control_rate_rps=config.control_request_rate_rps,
        expected_control_seconds=config.baseline_seconds,
        expected_restore_rate_rps=config.restore_request_rate_rps,
        slo_ttft_seconds=config.serving_slo_ttft_seconds,
        warmup_seconds=config.warmup_seconds,
        control_interval=control,
    )
    if not runtime_queue_samples:
        raise RuntimeError("GPU0 restore lacks live vLLM scheduler queue samples")
    return {
        "schema_version": "sloforge.branchfabric.experiment-004-v11-gpu0-result/v1",
        "status": "succeeded",
        "attempt_id": config.attempt_id,
        "methodology_binding": frozen_v10_methodology_identity(),
        "trigger_overlay_binding": integrated_v11_trigger_overlay_identity(),
        "serving": serving,
        "gpu0_control_interval": interference["control_interval"],
        "gpu0_restore_interference": interference,
        "scientific_gates": {
            "control_stability_pass": True,
            "gpu0_overload_pass": True,
            "bounded_backlog_pass": True,
            "two_gpu_service_gt_offered_pass": True,
            "queue_drain_pass": True,
            "slo_restoration_pass": True,
            "slo_stability_pass": True,
            "gpu0_active_during_restore_pass": True,
        },
    }


def run_targeted_source_identity_v11_gpu0(
    engine: Any,
    *,
    adapter: Any,
    inputs: Mapping[str, Any],
    config: Any,
    start_ns: int,
    barriers: Path,
) -> dict[str, Any]:
    """Run real control/overload traffic and stop at GPU1's identity barrier."""

    from gpu_capacity_calibration_worker import _runtime_queue_state
    from gpu_reclamation_integrated_trigger_v11 import (
        run_v11_gpu0_until_source_identity_gate,
    )
    from gpu_reclamation_worker import _sampling_params

    raw = run_v11_gpu0_until_source_identity_gate(
        engine,
        prefix=tuple(
            int(item) for item in inputs["prefix_token_ids"][: config.serving_prompt_tokens]
        ),
        params=_sampling_params(max_tokens=config.serving_output_tokens, seed=config.seed),
        config=build_live_v10_config(config),
        start_ns=start_ns,
        barriers=barriers,
        write_new=_write_new,
        runtime_queue_state=lambda: _runtime_queue_state(adapter),
        expected_source_allocation_count=config.expected_total_blocks,
    )
    queue_drain = raw.get("queue_drain")
    runtime_state = (
        queue_drain.get("runtime_queue_state") if isinstance(queue_drain, dict) else None
    )
    if (
        raw.get("passed") is not True
        or not isinstance(runtime_state, dict)
        or any(int(value) != 0 for value in runtime_state.values())
    ):
        raise RuntimeError("targeted GPU0 did not reach a real queue-zero terminal state")
    return {
        "schema_version": "sloforge.branchfabric.experiment-004-v11-targeted-gpu0-result/v1",
        "status": "succeeded",
        "attempt_id": config.attempt_id,
        "terminal_phase": "SOURCE_ALLOCATION_IDENTITY_GATE",
        "reclamation_trigger_evidence": raw["reclamation_trigger_evidence"],
        "source_identity_terminal": raw["source_identity_gate"],
        "trigger_backlog_requests": raw["trigger_backlog_requests"],
        "maximum_outstanding_requests": raw["maximum_outstanding_requests"],
        "trigger_emergency_ceiling_headroom_requests": raw[
            "trigger_emergency_ceiling_headroom_requests"
        ],
        "minimum_emergency_ceiling_headroom_requests": raw[
            "minimum_emergency_ceiling_headroom_requests"
        ],
        "producer_stopped": raw["arrival_stopped_ns"] >= raw["arrival_stop_requested_ns"],
        "final_runtime_queue_state": runtime_state,
        "targeted_serving_trace": raw,
        "optimized_export_started": False,
        "transaction_source_release_started": False,
    }


def _drain_released_serving_allocation_queue(
    adapter: Any, *, admission_gate: Any
) -> dict[str, Any]:
    """Zero the allocator event queue after all temporary serving requests drain."""

    from sloforge.continuum.adapters.vllm_reclamation_v11_sync import require_production_gate_v11

    require_production_gate_v11(
        admission_gate,
        adapter._view.scheduler,
        adapter._view.manager,
        operation="IMPORT_ADMISSION",
    )
    manager = adapter._view.manager
    take = getattr(manager, "take_new_block_ids", None)
    if not callable(take):
        raise RuntimeError("production allocator exposes no serving-allocation zero queue")
    observed = tuple(take())
    if any(isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in observed):
        raise RuntimeError("serving-allocation zero queue returned an invalid block ID")
    if not observed:
        raise RuntimeError("GPU1 serving produced no allocator allocation events")
    scheduler = adapter._view.scheduler
    requests = getattr(scheduler, "requests", None)
    if not isinstance(requests, dict) or requests:
        raise RuntimeError("temporary GPU1 serving requests remain live at import")
    unique_ids = tuple(sorted(set(observed)))
    release = adapter.inspect_block_release_evidence(
        tuple(f"vllm:kv-group:0:block:{block_id}" for block_id in unique_ids),
        timeout_s=30.0,
    )
    if any(
        int(block.native_refcount) != 0
        or block.block_hash_present
        or not block.allocator_available
        or block.is_null
        for block in release.blocks
    ):
        raise RuntimeError("a temporary serving allocation is not allocator-available")
    require_production_gate_v11(
        admission_gate,
        adapter._view.scheduler,
        adapter._view.manager,
        operation="IMPORT_ADMISSION",
    )
    ordered = tuple(sorted(observed))
    return {
        "schema_version": "sloforge.branchfabric.v11-serving-allocation-zero-queue/v1",
        "gate_id": admission_gate.gate_id,
        "observed_count": len(ordered),
        "unique_physical_block_count": len(unique_ids),
        "physical_reuse_event_count": len(ordered) - len(unique_ids),
        "observed_block_ids_sha256": hashlib.sha256(_canonical_bytes(ordered)).hexdigest(),
        "scheduler_request_table_empty": True,
        "all_native_refcounts_zero": True,
        "all_allocator_available": True,
        "all_block_hashes_cleared": True,
        "drained_after_gpu1_serving_and_before_destination_allocation": True,
        "passed": True,
    }


def run_targeted_source_identity_v11_gpu1(
    *,
    adapter: Any,
    prepared: Mapping[str, Any],
    config: Any,
    physical_gpu_uuid: str,
    barriers: Path,
) -> dict[str, Any]:
    """Exercise the integrated allocator lifecycle and stop before state export."""

    from gpu_reclamation_worker import _geometry, _runtime_capture_inputs, _wait_for
    from gpu_reclamation_worker_v11 import (
        _public_value,
        _source_epoch_map,
        require_allocator_notification_queue_empty_v11,
        retire_source_allocation_history_v11,
        validate_exact_micro_topology,
    )

    from sloforge.continuum.adapters.vllm_reclamation import build_canonical_capture_plan
    from sloforge.continuum.adapters.vllm_reclamation_v11 import Vllm0230EngineStepBinding
    from sloforge.continuum.adapters.vllm_reclamation_v11_ownership import (
        capture_source_ownership_v11,
        create_source_capture_commit_v11,
        validate_source_capture_commit_v11,
    )

    if (
        getattr(config, "execution_mode", None) != "targeted-source-identity-v11"
        or getattr(config, "full_export_authorized", None) is not False
        or getattr(config, "source_release_authorized", None) is not False
    ):
        raise RuntimeError("targeted source identity received an export-capable config")
    trigger = _wait_for(
        barriers / "v10-reclaim-trigger.json", timeout_s=config.maximum_wall_seconds
    )
    trigger_ns = int(trigger["triggered_ns"])
    rollout_admission_stop_ns = time.monotonic_ns()
    _write_new(
        barriers / "v11-rollout-admission-stop.json",
        {
            "schema_version": "sloforge.branchfabric.v11-rollout-admission-stop/v1",
            "event": "ROLLOUT_ADMISSION_STOP",
            "reclaim_trigger_emitted_ns": trigger_ns,
            "observed_ns": rollout_admission_stop_ns,
        },
    )
    branches = tuple(str(item) for item in prepared["branches"])
    _tensors, geometry = _geometry(adapter)
    binding = Vllm0230EngineStepBinding(adapter._view, acquire_timeout_seconds=30.0)
    try:
        with binding.critical_section(
            "EXPORT_CAPTURE", gate_id=f"{config.attempt_id}:targeted-source-identity"
        ) as export_gate:
            state_quiescence_ns = time.monotonic_ns()
            runtime_inputs = _runtime_capture_inputs(
                adapter,
                branches,
                parent_logical_branch_id=str(prepared["root_id"]),
            )
            plan = build_canonical_capture_plan(
                branches=runtime_inputs,
                block_size_tokens=geometry.block_size_tokens,
                logical_token_bytes=geometry.logical_token_bytes,
                physical_page_bytes=geometry.physical_page_bytes,
                gpu_uuid=physical_gpu_uuid,
                allocation_epoch_by_block=dict(adapter._observer.block_epochs),
            )
            topology = validate_exact_micro_topology(
                page_order=plan.page_order,
                branch_count=len(plan.branch_tables),
                logical_state_bytes=plan.logical_state_bytes,
            )
            source_epochs = _source_epoch_map(plan)
            ownership = capture_source_ownership_v11(
                adapter,
                root_session_id=str(prepared["root_id"]),
                branch_session_ids=branches,
                expected_allocation_epochs=source_epochs,
                expected_source_block_count=config.expected_total_blocks,
                admission_gate=export_gate,
                timeout_s=60.0,
            )
            source_lifetime_sha = hashlib.sha256(
                _canonical_bytes(sorted(source_epochs.items()))
            ).hexdigest()
            ownership_sha = hashlib.sha256(_canonical_bytes(_public_value(ownership))).hexdigest()
            allocation_history = retire_source_allocation_history_v11(
                adapter._view.scheduler,
                adapter._view.manager,
                export_gate=export_gate,
                source_allocation_lifetime_sha256=source_lifetime_sha,
                ownership_snapshot_sha256=ownership_sha,
            )
            commit_started_ns = time.monotonic_ns()
            source_commit = create_source_capture_commit_v11(
                adapter,
                plan,
                ownership,
                block_size_tokens=geometry.block_size_tokens,
                admission_gate=export_gate,
            )
            commit_ended_ns = time.monotonic_ns()
            validation = validate_source_capture_commit_v11(
                adapter,
                source_commit,
                plan,
                block_size_tokens=geometry.block_size_tokens,
                admission_gate=export_gate,
                timeout_s=60.0,
            )
            post_commit_notifications = require_allocator_notification_queue_empty_v11(
                adapter._view.scheduler,
                adapter._view.manager,
                export_gate=export_gate,
                source_capture_commit_sha256=source_commit.semantic_sha256,
                phase="TARGETED_IDENTITY_GATE",
            )
            validation_public = _public_value(validation)
            validation_sha = hashlib.sha256(_canonical_bytes(validation_public)).hexdigest()
            commit_public = _public_value(source_commit)
            allocator_lifecycle = _source_allocator_lifecycle_evidence(
                adapter,
                source_commit,
                start_sequence=int(prepared["allocator_lifecycle_start_sequence"]),
                rollout_ready_sequence=int(prepared["allocator_rollout_ready_sequence"]),
            )
            _write_new(
                barriers / "v11-source-capture-commit.json",
                {
                    "schema_version": "sloforge.branchfabric.v11-source-capture-commit/v1",
                    "event": "CAPTURE_COMMIT_END",
                    "commit": commit_public,
                    "identity_validation": validation_public,
                    "pre_commit_allocation_history": allocation_history,
                    "post_commit_allocation_notifications": post_commit_notifications,
                    "allocator_lifecycle": allocator_lifecycle,
                    "passed": True,
                },
            )
            identity_gate = {
                "schema_version": "sloforge.branchfabric.v11-source-identity-gate/v1",
                "event": "SOURCE_ALLOCATION_IDENTITY_GATE",
                "attempt_id": config.attempt_id,
                "observed_at_monotonic_ns": time.monotonic_ns(),
                "source_capture_commit_sha256": source_commit.semantic_sha256,
                "source_identity_validation_sha256": validation_sha,
                "allocation_history_sha256": allocation_history["history_block_ids_sha256"],
                "expected_allocation_count": config.expected_total_blocks,
                "observed_allocation_count": validation.allocation_count,
                "exact_logical_mapping": validation.exact_logical_mapping,
                "exact_block_epoch_identity": validation.exact_block_epoch_identity,
                "exact_owner_sets": validation.exact_owner_sets,
                "exact_refcounts": validation.exact_refcounts,
                "all_allocations_live": validation.all_allocations_live,
                "device_identity_pass": source_commit.device == ownership.device,
                "post_commit_allocation_event_count": post_commit_notifications[
                    "observed_event_count"
                ],
                "no_post_commit_mutation": validation.no_post_commit_mutation,
                "optimized_export_started": False,
                "transaction_source_release_started": False,
                "passed": validation.passed
                and validation.allocation_count == config.expected_total_blocks
                and source_commit.device == ownership.device
                and post_commit_notifications["queue_empty"] is True,
            }
            if identity_gate["passed"] is not True:
                raise RuntimeError("targeted source identity gate failed closed")
            trigger_timeline = validate_reclaim_trigger_timeline(
                trigger,
                rollout_admission_stop_ns=rollout_admission_stop_ns,
                state_quiescence_ns=state_quiescence_ns,
            )
            _write_new(barriers / "v11-source-identity-pre-export-pass.json", identity_gate)
        return {
            "schema_version": ("sloforge.branchfabric.experiment-004-v11-targeted-gpu1-result/v1"),
            "status": "succeeded",
            "attempt_id": config.attempt_id,
            "physical_gpu_uuid": physical_gpu_uuid,
            "terminal_phase": "SOURCE_ALLOCATION_IDENTITY_GATE",
            "topology": topology,
            "trigger_timeline": trigger_timeline,
            "source_capture_commit": commit_public,
            "source_identity_validation": validation_public,
            "source_identity_gate": identity_gate,
            "allocation_history_retirement": allocation_history,
            "post_commit_allocation_notifications": post_commit_notifications,
            "allocator_lifecycle": allocator_lifecycle,
            "engine_step_binding": _public_value(binding.evidence()),
            "optimized_export_started": False,
            "transaction_source_release_started": False,
            "cleanup_release_only": True,
            "timings": {
                "reclaim_trigger_ns": trigger_ns,
                "rollout_admission_stop_ns": rollout_admission_stop_ns,
                "allocator_quiescent_ns": state_quiescence_ns,
                "capture_commit_started_ns": commit_started_ns,
                "capture_commit_ended_ns": commit_ended_ns,
                "identity_gate_passed_ns": identity_gate["observed_at_monotonic_ns"],
            },
        }
    finally:
        binding.close()


def run_integrated_v11_gpu1(
    serving_engine: Any,
    *,
    adapter: Any,
    prepared: Mapping[str, Any],
    inputs: Mapping[str, Any],
    config: Any,
    physical_gpu_uuid: str,
    barriers: Path,
) -> dict[str, Any]:
    """Export/release, serve, then import and resume the exact eight branches."""

    import torch  # type: ignore[import-not-found]
    from gpu_capacity_calibration_worker import _runtime_queue_state
    from gpu_reclamation_v10_serving import run_v10_gpu1
    from gpu_reclamation_worker import (
        _add_restore_requests,
        _geometry,
        _identity,
        _runtime_capture_inputs,
        _sampling_params,
        _wait_for,
    )
    from gpu_reclamation_worker_v11 import (
        _public_value,
        _RssPeakSampler,
        _run_continuation,
        _run_independent_oracle,
        _source_epoch_map,
        require_allocator_notification_queue_empty_v11,
        retire_source_allocation_history_v11,
        summarize_state_passes,
        validate_exact_micro_topology,
    )

    from sloforge.continuum.adapters.vllm_reclamation import build_canonical_capture_plan
    from sloforge.continuum.adapters.vllm_reclamation_v11 import (
        V11PipelineConfig,
        V11RuntimeTeardownRequired,
        Vllm0230EngineStepBinding,
        Vllm0230StreamingRestoreStager,
        capture_native_to_transport_v11,
        destination_target_identity_v11,
        scrub_native_pages_v11,
        verify_transport_streaming_v11,
        write_and_validate_native_subset_v11,
    )
    from sloforge.continuum.adapters.vllm_reclamation_v11_accounting import (
        V11StatePassRecorder,
        validate_authoritative_pass_records,
    )
    from sloforge.continuum.adapters.vllm_reclamation_v11_ownership import (
        capture_source_ownership_v11,
        create_source_capture_commit_v11,
        release_source_and_prove_v11,
        validate_source_capture_commit_v11,
    )

    trigger = _wait_for(
        barriers / "v10-reclaim-trigger.json", timeout_s=config.maximum_wall_seconds
    )
    trigger_ns = int(trigger["triggered_ns"])
    rollout_admission_stop_ns = time.monotonic_ns()
    _write_new(
        barriers / "v11-rollout-admission-stop.json",
        {
            "schema_version": "sloforge.branchfabric.v11-rollout-admission-stop/v1",
            "event": "ROLLOUT_ADMISSION_STOP",
            "reclaim_trigger_emitted_ns": trigger_ns,
            "observed_ns": rollout_admission_stop_ns,
        },
    )
    branches = tuple(str(item) for item in prepared["branches"])
    tensors, geometry = _geometry(adapter)
    pipeline = V11PipelineConfig(
        seed=config.seed,
        expected_device_type="cuda",
        expected_cuda_device_index=0,
        expected_gpu_uuid=physical_gpu_uuid,
        maximum_chunk_bytes=config.maximum_chunk_bytes,
        buffer_count=config.buffer_count,
        pin_memory=True,
        asynchronous_copy=True,
    )
    binding = Vllm0230EngineStepBinding(adapter._view, acquire_timeout_seconds=30.0)
    source_sampling_by_branch = {
        branch: {
            "effective_seed": int(adapter._effective_seed(config.seed, adapter._sessions[branch]))
        }
        for branch in branches
    }
    try:
        memory_before = adapter.inspect_gpu_memory_state(timeout_s=10.0)
        export_base_gpu = int(torch.cuda.memory_allocated(0))
        torch.cuda.reset_peak_memory_stats(0)
        export_rss_sampler = _RssPeakSampler()
        export_rss_sampler.start()
        export_started_ns = time.monotonic_ns()
        with binding.critical_section(
            "EXPORT_CAPTURE", gate_id=f"{config.attempt_id}:integrated-export"
        ) as export_gate:
            state_quiescence_ns = time.monotonic_ns()
            runtime_inputs = _runtime_capture_inputs(
                adapter,
                branches,
                parent_logical_branch_id=str(prepared["root_id"]),
            )
            plan = build_canonical_capture_plan(
                branches=runtime_inputs,
                block_size_tokens=geometry.block_size_tokens,
                logical_token_bytes=geometry.logical_token_bytes,
                physical_page_bytes=geometry.physical_page_bytes,
                gpu_uuid=physical_gpu_uuid,
                allocation_epoch_by_block=dict(adapter._observer.block_epochs),
            )
            topology = validate_exact_micro_topology(
                page_order=plan.page_order,
                branch_count=len(plan.branch_tables),
                logical_state_bytes=plan.logical_state_bytes,
            )
            source_epochs = _source_epoch_map(plan)
            ownership_before = capture_source_ownership_v11(
                adapter,
                root_session_id=str(prepared["root_id"]),
                branch_session_ids=branches,
                expected_allocation_epochs=source_epochs,
                expected_source_block_count=config.expected_total_blocks,
                admission_gate=export_gate,
                timeout_s=60.0,
            )
            source_authentication_ended_ns = time.monotonic_ns()
            source_lifetime_sha = hashlib.sha256(
                _canonical_bytes(sorted(source_epochs.items()))
            ).hexdigest()
            ownership_snapshot_sha = hashlib.sha256(
                _canonical_bytes(_public_value(ownership_before))
            ).hexdigest()
            source_queue = retire_source_allocation_history_v11(
                adapter._view.scheduler,
                adapter._view.manager,
                export_gate=export_gate,
                source_allocation_lifetime_sha256=source_lifetime_sha,
                ownership_snapshot_sha256=ownership_snapshot_sha,
            )
            allocation_history_retired_ns = time.monotonic_ns()
            capture_commit_started_ns = allocation_history_retired_ns
            source_commit = create_source_capture_commit_v11(
                adapter,
                plan,
                ownership_before,
                block_size_tokens=geometry.block_size_tokens,
                admission_gate=export_gate,
            )
            capture_commit_ended_ns = time.monotonic_ns()
            identity_before_export = validate_source_capture_commit_v11(
                adapter,
                source_commit,
                plan,
                block_size_tokens=geometry.block_size_tokens,
                admission_gate=export_gate,
                timeout_s=60.0,
            )
            post_commit_notifications = require_allocator_notification_queue_empty_v11(
                adapter._view.scheduler,
                adapter._view.manager,
                export_gate=export_gate,
                source_capture_commit_sha256=source_commit.semantic_sha256,
                phase="PRE_EXPORT_READ",
            )
            allocator_lifecycle = _source_allocator_lifecycle_evidence(
                adapter,
                source_commit,
                start_sequence=int(prepared["allocator_lifecycle_start_sequence"]),
                rollout_ready_sequence=int(prepared["allocator_rollout_ready_sequence"]),
            )
            _write_new(
                barriers / "v11-source-capture-commit.json",
                {
                    "schema_version": "sloforge.branchfabric.v11-source-capture-commit/v1",
                    "event": "CAPTURE_COMMIT_END",
                    "commit": _public_value(source_commit),
                    "pre_export_identity": _public_value(identity_before_export),
                    "pre_commit_allocation_history": source_queue,
                    "post_commit_allocation_notifications": post_commit_notifications,
                    "allocator_lifecycle": allocator_lifecycle,
                    "passed": True,
                },
            )
            _write_new(
                barriers / "v11-source-identity-pre-export-pass.json",
                {
                    "schema_version": "sloforge.branchfabric.v11-source-identity-gate/v1",
                    "event": "SOURCE_ALLOCATION_IDENTITY_GATE",
                    "attempt_id": config.attempt_id,
                    "observed_at_monotonic_ns": time.monotonic_ns(),
                    "source_capture_commit_sha256": source_commit.semantic_sha256,
                    "expected_allocation_count": config.expected_total_blocks,
                    "observed_allocation_count": identity_before_export.allocation_count,
                    "exact_logical_mapping": identity_before_export.exact_logical_mapping,
                    "exact_block_epoch_identity": (
                        identity_before_export.exact_block_epoch_identity
                    ),
                    "exact_owner_sets": identity_before_export.exact_owner_sets,
                    "exact_refcounts": identity_before_export.exact_refcounts,
                    "all_allocations_live": identity_before_export.all_allocations_live,
                    "post_commit_allocation_event_count": (
                        post_commit_notifications["observed_event_count"]
                    ),
                    "no_post_commit_mutation": True,
                    "optimized_export_started": False,
                    "transaction_source_release_started": False,
                    "passed": True,
                },
            )
            pre_export_identity_ended_ns = time.monotonic_ns()
            captured = capture_native_to_transport_v11(
                tensors,
                geometry,
                page_order=plan.page_order,
                branch_tables=plan.branch_tables,
                identity=_identity(),
                config=pipeline,
                engine_step_gate=export_gate,
            )
            source_pipeline_ended_ns = time.monotonic_ns()
            post_runtime_inputs = _runtime_capture_inputs(
                adapter,
                branches,
                parent_logical_branch_id=str(prepared["root_id"]),
            )
            post_capture_plan = build_canonical_capture_plan(
                branches=post_runtime_inputs,
                block_size_tokens=geometry.block_size_tokens,
                logical_token_bytes=geometry.logical_token_bytes,
                physical_page_bytes=geometry.physical_page_bytes,
                gpu_uuid=physical_gpu_uuid,
                allocation_epoch_by_block=dict(adapter._observer.block_epochs),
            )
            identity_after_export_read = validate_source_capture_commit_v11(
                adapter,
                source_commit,
                post_capture_plan,
                block_size_tokens=geometry.block_size_tokens,
                admission_gate=export_gate,
                timeout_s=60.0,
            )
            post_export_identity_ended_ns = time.monotonic_ns()
            manifest_sha = hashlib.sha256(captured.state.manifest.canonical_bytes()).hexdigest()
            post_export_notifications = require_allocator_notification_queue_empty_v11(
                adapter._view.scheduler,
                adapter._view.manager,
                export_gate=export_gate,
                source_capture_commit_sha256=source_commit.semantic_sha256,
                phase="POST_EXPORT_READ",
            )
            state_publish_ended_ns = time.monotonic_ns()
            _write_new(
                barriers / "v11-source-identity-post-export-pass.json",
                {
                    "schema_version": "sloforge.branchfabric.v11-source-identity-gate/v1",
                    "event": "POST_EXPORT_READ_IDENTITY_GATE",
                    "source_capture_commit_sha256": source_commit.semantic_sha256,
                    "allocation_count": len(source_commit.allocations),
                    "identity_before_export": _public_value(identity_before_export),
                    "identity_after_export_read": _public_value(identity_after_export_read),
                    "allocation_history_retirement": source_queue,
                    "post_commit_allocation_notifications": post_commit_notifications,
                    "post_export_allocation_notifications": post_export_notifications,
                    "zero_post_commit_semantic_mutations": True,
                    "passed": True,
                },
            )
            ownership_after = release_source_and_prove_v11(
                adapter, ownership_before, admission_gate=export_gate, timeout_s=60.0
            )
            ownership_after.require_passed()
            source_release_ended_ns = time.monotonic_ns()
        export_ended_ns = time.monotonic_ns()
        export_rss_delta = export_rss_sampler.stop()
        export_peak_gpu = int(torch.cuda.max_memory_allocated(0))
        export_temporary_gpu = max(0, export_peak_gpu - export_base_gpu)
        memory_after = adapter.inspect_gpu_memory_state(timeout_s=10.0)
        if int(memory_after.kv_assigned_bytes) != 0:
            raise RuntimeError("v11 source release left assigned KV bytes")
        hbm_reclaim_ns = time.monotonic_ns()

        serving = run_v10_gpu1(
            serving_engine,
            prefix=tuple(int(item) for item in inputs["prefix_token_ids"][:256]),
            params=_sampling_params(max_tokens=config.serving_output_tokens, seed=config.seed),
            config=build_live_v10_config(config),
            barriers=barriers,
            write_new=_write_new,
            runtime_queue_state=lambda: _runtime_queue_state(adapter),
        )
        gpu1_serving_ready = json.loads((barriers / "v10-gpu1-serving-ready.json").read_text())
        gpu1_serving_ready_ns = gpu1_serving_ready.get("observed_ns")
        if (
            isinstance(gpu1_serving_ready_ns, bool)
            or not isinstance(gpu1_serving_ready_ns, int)
            or gpu1_serving_ready_ns < hbm_reclaim_ns
        ):
            raise RuntimeError("integrated v11 GPU1 serving-ready timestamp is invalid")
        if not adapter._view.manager.reset_prefix_cache():
            raise V11RuntimeTeardownRequired(
                "GPU1 serving prefix cache did not drain before v11 import",
                affected_block_ids=(),
                teardown_evidence={
                    "schema_version": ("sloforge.branchfabric.v11-serving-to-import-teardown/v1"),
                    "stage": "post-serving-pre-import",
                    "prefix_cache_reset": False,
                    "passed": False,
                },
            )
        adapter._allocator_epoch_source.observe_prefix_cache_reset()
        restore_trigger = _wait_for(
            barriers / "v10-restore-start.json", timeout_s=config.maximum_wall_seconds
        )
        restore_trigger_ns = int(restore_trigger["observed_ns"])
        restore_base_gpu = int(torch.cuda.memory_allocated(0))
        torch.cuda.reset_peak_memory_stats(0)
        restore_rss_sampler = _RssPeakSampler()
        restore_rss_sampler.start()
        restore_started_ns = time.monotonic_ns()
        restore_stats: list[Any] = []
        validation_evidence: list[Any] = []
        branch_group = captured.state_passes[0].branch_group
        recorder = V11StatePassRecorder(
            device=f"cuda:0@{physical_gpu_uuid}", branch_group=branch_group
        )
        with binding.critical_section(
            "IMPORT_ADMISSION", gate_id=f"{config.attempt_id}:integrated-import"
        ) as import_gate:
            serving_queue = _drain_released_serving_allocation_queue(
                adapter, admission_gate=import_gate
            )
            checkpoint_authentication_started_ns = time.monotonic_ns()
            verified = verify_transport_streaming_v11(
                captured.state,
                maximum_chunk_bytes=config.maximum_chunk_bytes,
                pass_recorder=recorder,
            )
            checkpoint_authentication_ended_ns = time.monotonic_ns()
            runtime_ids, output_ids = _add_restore_requests(
                adapter._view.llm_engine,
                captured.state.manifest.branches,
                source_sampling_by_branch=source_sampling_by_branch,
            )
            destination_target = destination_target_identity_v11(tensors, geometry, pipeline)
            destination_staging_ended_ns = time.monotonic_ns()

            def write_and_validate(mapping: dict[str, int], epoch_proof: Any) -> Any:
                evidence, stats = write_and_validate_native_subset_v11(
                    captured.state,
                    tensors,
                    geometry,
                    destination_block_indices=mapping,
                    expected_identity=_identity(),
                    verified_transport=verified,
                    config=pipeline,
                    allocation_epoch_by_block=epoch_proof.allocation_epoch_by_block,
                    engine_step_gate=import_gate,
                    pass_recorder=recorder,
                    caller_guarantees_outer_scrub=True,
                )
                validation_evidence.append(evidence)
                restore_stats.append(stats)
                return evidence

            def scrub(mapping: dict[str, int], epoch_proof: Any) -> Any:
                return scrub_native_pages_v11(
                    tensors,
                    geometry,
                    state=captured.state,
                    verified_transport=verified,
                    config=pipeline,
                    allocation_epoch_by_block=epoch_proof.allocation_epoch_by_block,
                    destination_block_indices=mapping,
                    engine_step_gate=import_gate,
                    pass_recorder=recorder,
                )

            imported = Vllm0230StreamingRestoreStager(
                adapter._view.scheduler, adapter._view.manager
            ).import_group_v11(
                captured.state,
                verified_transport=verified,
                runtime_request_ids=runtime_ids,
                expected_identity=_identity(),
                destination_target_sha256=destination_target,
                admission_gate=import_gate,
                write_and_validate_pages=write_and_validate,
                scrub_pages_on_failure=scrub,
            )
            native_restore_pipeline_ended_ns = time.monotonic_ns()
        restore_passes = recorder.resolve()
        validate_authoritative_pass_records(
            restore_passes,
            expected_device=f"cuda:0@{physical_gpu_uuid}",
            expected_branch_group=branch_group,
        )
        restore_ended_ns = time.monotonic_ns()
        restore_rss_delta = restore_rss_sampler.stop()
        restore_peak_gpu = int(torch.cuda.max_memory_allocated(0))
        restore_temporary_gpu = max(0, restore_peak_gpu - restore_base_gpu)
        source_lifetimes = {(block, epoch) for block, epoch in source_epochs.items()}
        destination_lifetimes = {
            (int(block), int(epoch))
            for evidence in validation_evidence
            for block, epoch in evidence.allocation_epochs
        }
        destination_blocks = set(imported.logical_page_destinations.values())
        if (
            len(destination_blocks) != EXPECTED_TOTAL_BLOCKS
            or len(destination_lifetimes) != EXPECTED_TOTAL_BLOCKS
            or source_lifetimes & destination_lifetimes
        ):
            raise RuntimeError("integrated v11 destination lifetimes are not fresh and complete")

        continuation_started_ns = time.monotonic_ns()
        observed_first, counts, first_ns, all_first_ns, complete_ns = _run_continuation(
            adapter._view.llm_engine, output_ids=output_ids, timeout_seconds=120.0
        )
        _write_new(
            barriers / "rollout-restore-complete.json",
            {
                "schema_version": "sloforge.branchfabric.v11-rollout-restore-complete/v1",
                "restore_trigger_ns": restore_trigger_ns,
                "native_restore_complete_ns": restore_ended_ns,
                "first_resumed_token_ns": first_ns,
                "all_branches_resumed_ns": all_first_ns,
                "continuation_complete_ns": complete_ns,
                "minimum_tokens_per_branch": min(counts.values()),
                "branch_count": len(counts),
            },
        )
        if not adapter._view.manager.reset_prefix_cache():
            raise V11RuntimeTeardownRequired(
                "restored requests did not drain before independent oracle",
                affected_block_ids=tuple(destination_blocks),
                teardown_evidence={
                    "schema_version": "sloforge.branchfabric.v11-post-restore-teardown/v1",
                    "stage": "post-continuation-pre-oracle",
                    "prefix_cache_reset": False,
                    "passed": False,
                },
            )
        adapter._allocator_epoch_source.observe_prefix_cache_reset()
        expected_first = _run_independent_oracle(
            adapter._view.llm_engine,
            branch_tables=captured.state.manifest.branches,
            source_sampling_by_branch=source_sampling_by_branch,
            sampling_params=_sampling_params,
            timeout_seconds=180.0,
        )
        exact = observed_first == expected_first and len(observed_first) == 8
        all_passes = tuple(captured.state_passes) + tuple(restore_passes)
        movement = summarize_state_passes(
            all_passes, logical_state_bytes=captured.state.manifest.logical_state_bytes
        )
        export_movement = summarize_state_passes(
            captured.state_passes,
            logical_state_bytes=captured.state.manifest.logical_state_bytes,
        )
        restore_movement = summarize_state_passes(
            restore_passes,
            logical_state_bytes=captured.state.manifest.logical_state_bytes,
        )
        reclaim_critical_path = _non_overlapping_partition(
            chain="RECLAIM_TRIGGER_TO_GPU1_SERVING_READY",
            boundaries=(
                ("branch_quiesce", trigger_ns, state_quiescence_ns),
                (
                    "source_authentication",
                    state_quiescence_ns,
                    source_authentication_ended_ns,
                ),
                (
                    "pre_commit_allocation_history_retirement",
                    source_authentication_ended_ns,
                    allocation_history_retired_ns,
                ),
                (
                    "source_capture_commit",
                    allocation_history_retired_ns,
                    capture_commit_ended_ns,
                ),
                (
                    "pre_export_identity_validation_and_commit_publish",
                    capture_commit_ended_ns,
                    pre_export_identity_ended_ns,
                ),
                (
                    "fused_gather_repack_d2h",
                    pre_export_identity_ended_ns,
                    source_pipeline_ended_ns,
                ),
                (
                    "post_export_read_identity_validation",
                    source_pipeline_ended_ns,
                    post_export_identity_ended_ns,
                ),
                (
                    "state_publish_and_allocation_history_retirement",
                    post_export_identity_ended_ns,
                    state_publish_ended_ns,
                ),
                ("source_release", state_publish_ended_ns, source_release_ended_ns),
                ("hbm_reclaim_confirmation", source_release_ended_ns, hbm_reclaim_ns),
                ("gpu1_serving_ready", hbm_reclaim_ns, gpu1_serving_ready_ns),
            ),
        )
        restore_critical_path = _non_overlapping_partition(
            chain="RESTORE_TRIGGER_TO_FIRST_RESUMED_TOKEN",
            boundaries=(
                (
                    "restore_preconditions",
                    restore_trigger_ns,
                    checkpoint_authentication_started_ns,
                ),
                (
                    "checkpoint_authentication",
                    checkpoint_authentication_started_ns,
                    checkpoint_authentication_ended_ns,
                ),
                (
                    "destination_allocation_staging",
                    checkpoint_authentication_ended_ns,
                    destination_staging_ended_ns,
                ),
                (
                    "fused_h2d_direct_scatter_raw_validation_and_admission",
                    destination_staging_ended_ns,
                    native_restore_pipeline_ended_ns,
                ),
                (
                    "canonical_pass_validation_and_commit",
                    native_restore_pipeline_ended_ns,
                    restore_ended_ns,
                ),
                ("first_resumed_token", restore_ended_ns, first_ns),
            ),
        )
        peak_pinned_temporary = max(
            [captured.stats.peak_host_temporary_bytes]
            + [item.peak_host_temporary_bytes for item in restore_stats]
        )
        export_pageable_delta = max(
            0,
            export_rss_delta
            - captured.stats.checkpoint_resident_host_bytes
            - captured.stats.peak_host_temporary_bytes,
        )
        restore_pageable_delta = max(
            0,
            restore_rss_delta - max(item.peak_host_temporary_bytes for item in restore_stats),
        )
        correctness = {
            "allocator_epoch_pass": all(item.allocator_issued for item in validation_evidence),
            "source_capture_commit_pass": source_commit.semantic_sha256
            == identity_before_export.semantic_sha256,
            "source_identity_1152_of_1152_pass": (
                identity_before_export.passed
                and identity_after_export_read.passed
                and identity_after_export_read.allocation_count == config.expected_total_blocks
            ),
            "zero_post_commit_source_mutations_pass": (
                identity_after_export_read.no_post_commit_mutation
                and source_queue["queue_empty_after_retirement"] is True
                and post_commit_notifications["queue_empty"] is True
                and post_export_notifications["queue_empty"] is True
            ),
            "ownership_release_pass": ownership_after.passed,
            "engine_step_binding_pass": binding.evidence().passed,
            "integrity_pass": all(item.passed for item in validation_evidence),
            "destination_mapping_commitment_pass": all(
                bool(item.destination_mapping_sha256) for item in validation_evidence
            ),
            "fresh_destination_allocations_pass": not bool(
                source_lifetimes & destination_lifetimes
            ),
            "all_branches_resumed": min(counts.values()) >= config.continuation_tokens,
            "first_token_exact_8_of_8": exact,
            "movement_accounting_pass": movement["movement_accounting_complete"] is True,
            "gpu0_active_during_restore_protocol_pass": (
                barriers / "v10-restore-start.json"
            ).is_file(),
        }
        if any(value is not True for value in correctness.values()):
            raise RuntimeError("integrated v11 rollout correctness failed closed")
        trigger_timeline = validate_reclaim_trigger_timeline(
            trigger,
            rollout_admission_stop_ns=rollout_admission_stop_ns,
            state_quiescence_ns=state_quiescence_ns,
        )
        return {
            "schema_version": "sloforge.branchfabric.experiment-004-v11-gpu1-result/v1",
            "status": "succeeded",
            "attempt_id": config.attempt_id,
            "physical_gpu_uuid": physical_gpu_uuid,
            "topology": topology,
            "correctness": correctness,
            "trigger_timeline": trigger_timeline,
            "serving": serving,
            "timings": {
                "reclaim_trigger_ns": trigger_ns,
                "rollout_admission_stop_ns": rollout_admission_stop_ns,
                "state_quiescence_ns": state_quiescence_ns,
                "source_authentication_ended_ns": source_authentication_ended_ns,
                "allocation_history_retired_ns": allocation_history_retired_ns,
                "capture_commit_started_ns": capture_commit_started_ns,
                "capture_commit_ended_ns": capture_commit_ended_ns,
                "pre_export_identity_ended_ns": pre_export_identity_ended_ns,
                "source_pipeline_ended_ns": source_pipeline_ended_ns,
                "post_export_identity_ended_ns": post_export_identity_ended_ns,
                "state_publish_ended_ns": state_publish_ended_ns,
                "source_release_ended_ns": source_release_ended_ns,
                "export_started_ns": export_started_ns,
                "export_ended_ns": export_ended_ns,
                "hbm_reclaim_confirmed_ns": hbm_reclaim_ns,
                "gpu1_serving_ready_ns": gpu1_serving_ready_ns,
                "restore_trigger_ns": restore_trigger_ns,
                "restore_started_ns": restore_started_ns,
                "checkpoint_authentication_started_ns": (checkpoint_authentication_started_ns),
                "checkpoint_authentication_ended_ns": checkpoint_authentication_ended_ns,
                "destination_staging_ended_ns": destination_staging_ended_ns,
                "native_restore_pipeline_ended_ns": native_restore_pipeline_ended_ns,
                "restore_native_complete_ns": restore_ended_ns,
                "continuation_started_ns": continuation_started_ns,
                "first_resumed_token_ns": first_ns,
                "all_branches_resumed_ns": all_first_ns,
                "continuation_complete_ns": complete_ns,
            },
            "memory": {
                "kv_pool_reserved_bytes_before": int(memory_before.kv_pool_reserved_bytes),
                "kv_assigned_bytes_before": int(memory_before.kv_assigned_bytes),
                "kv_pool_reserved_bytes_after": int(memory_after.kv_pool_reserved_bytes),
                "kv_assigned_bytes_after": int(memory_after.kv_assigned_bytes),
            },
            "movement": movement,
            "movement_by_phase": {
                "source_export": export_movement,
                "rollout_restore": restore_movement,
            },
            "critical_paths": {
                "reclamation": reclaim_critical_path,
                "restore": restore_critical_path,
                "residual_source_chain_ns": (source_pipeline_ended_ns - state_quiescence_ns),
                "residual_restore_chain_ns": restore_ended_ns - restore_started_ns,
                "stage_overlap_policy": (
                    "streaming gather/repack/D2H and H2D/scatter/validation remain fused "
                    "because the frozen v11 pipeline overlaps them; canonical StatePassRecords "
                    "provide operation attribution without adding overlapping wall intervals"
                ),
            },
            "temporary_memory": {
                "peak_pinned_host_temporary_bytes": peak_pinned_temporary,
                "peak_pageable_host_temporary_bytes": max(
                    export_pageable_delta, restore_pageable_delta
                ),
                "checkpoint_resident_host_bytes": (captured.stats.checkpoint_resident_host_bytes),
                "peak_gpu_temporary_bytes": max(
                    export_temporary_gpu,
                    restore_temporary_gpu,
                    captured.stats.peak_transform_temporary_bytes,
                    *[item.peak_transform_temporary_bytes for item in restore_stats],
                ),
                "export_process_rss_baseline_bytes": export_rss_sampler.baseline_bytes,
                "export_process_rss_peak_bytes": export_rss_sampler.peak_bytes,
                "export_process_rss_delta_bytes": export_rss_delta,
                "restore_process_rss_baseline_bytes": restore_rss_sampler.baseline_bytes,
                "restore_process_rss_peak_bytes": restore_rss_sampler.peak_bytes,
                "restore_process_rss_delta_bytes": restore_rss_delta,
                "export_torch_allocated_baseline_bytes": export_base_gpu,
                "export_peak_torch_allocated_bytes": export_peak_gpu,
                "restore_torch_allocated_baseline_bytes": restore_base_gpu,
                "restore_peak_torch_allocated_bytes": restore_peak_gpu,
                "method": (
                    "10ms process RSS deltas plus torch CUDA allocator peaks; resident "
                    "checkpoint and explicit pinned staging are separated"
                ),
            },
            "state_passes": [item.as_dict() for item in all_passes],
            "source_allocations": {
                "count": len(source_lifetimes),
                "allocation_lifetime_sha256": source_lifetime_sha,
                "pre_release_ownership_snapshot_sha256": ownership_snapshot_sha,
                "source_capture_commit": _public_value(source_commit),
                "identity_before_export": _public_value(identity_before_export),
                "identity_after_export_read": _public_value(identity_after_export_read),
                "allocation_history_retirement": source_queue,
                "post_commit_allocation_notifications": post_commit_notifications,
                "post_export_allocation_notifications": post_export_notifications,
                "allocator_lifecycle": allocator_lifecycle,
            },
            "temporary_serving_allocation_zero_queue": serving_queue,
            "destination_allocations": {
                "count": len(destination_lifetimes),
                "physical_block_id_overlap_count": len(set(source_epochs) & destination_blocks),
                "allocation_lifetime_overlap_count": len(source_lifetimes & destination_lifetimes),
                "allocation_lifetime_sha256": hashlib.sha256(
                    _canonical_bytes(sorted(destination_lifetimes))
                ).hexdigest(),
            },
            "engine_step_binding": _public_value(binding.evidence()),
            "post_free_ownership": _public_value(ownership_after),
            "integrity": _public_value(captured.integrity),
            "state_manifest_sha256": manifest_sha,
            "continuation": {
                "observed_first_tokens": observed_first,
                "independent_recompute_first_tokens": expected_first,
                "exact_matches": sum(
                    observed_first.get(key) == expected_first.get(key) for key in expected_first
                ),
                "minimum_tokens_per_branch": min(counts.values()),
            },
            "methodology_binding": frozen_v10_methodology_identity(),
        }
    finally:
        for sampler_name in ("export_rss_sampler", "restore_rss_sampler"):
            sampler = locals().get(sampler_name)
            if sampler is not None:
                sampler.stop()
        binding.close()


def _validate_runtime(config: Any, *, physical_gpu_uuid: str) -> None:
    import torch  # type: ignore[import-not-found]

    visible = os.environ.get("SLOFORGE_EXP004_VISIBLE_GPU_INDEX")
    if (
        visible is None
        or os.environ.get("CUDA_VISIBLE_DEVICES") != visible
        or os.environ.get("SLOFORGE_EXP004_PHYSICAL_GPU_UUID") != physical_gpu_uuid
        or importlib.metadata.version("vllm") != config.runtime_version
        or importlib.metadata.version("torch") != config.torch_version
        or not torch.cuda.is_available()
        or torch.cuda.device_count() != 1
        or torch.cuda.get_device_capability(0) != (8, 0)
        or "A100" not in torch.cuda.get_device_name(0)
    ):
        raise RuntimeError("integrated v11 runtime pins or physical A100 binding are invalid")


def run_worker(
    role: Literal["serving", "rollout"],
    config_payload: dict[str, Any],
    *,
    model_snapshot: Path,
    physical_gpu_uuid: str,
    work_root: Path,
    barrier_root: Path,
) -> dict[str, Any]:
    """Load one retained engine and execute one side of the integrated run."""

    from sloforge.helix.characterization.gpu_reclamation_v11_methodology import (
        Experiment004V11IntegratedConfig,
        Experiment004V11TargetedSourceIdentityConfig,
        validate_bound_artifact,
    )

    if (
        config_payload.get("schema_version")
        == "sloforge.branchfabric.experiment-004-v11-targeted-source-identity-config/v1"
    ):
        config = Experiment004V11TargetedSourceIdentityConfig.model_validate(config_payload)
    else:
        config = Experiment004V11IntegratedConfig.model_validate(config_payload)
    repository_root = Path(__file__).resolve().parents[2]
    for reference, expected in (
        (config.offline_gate_manifest, config.offline_gate_manifest_sha256),
        (config.micro_validation_artifact, config.micro_validation_sha256),
        (config.post_micro_review_manifest, config.post_micro_review_manifest_sha256),
        (config.budget_authorization, config.budget_authorization_sha256),
    ):
        validate_bound_artifact(repository_root, reference=reference, expected_sha256=expected)
    os.environ.update(
        {
            "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
            "VLLM_USE_FLASHINFER_SAMPLER": "0",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "TRITON_CACHE_DIR": str(work_root / "triton"),
        }
    )
    (work_root / "triton").mkdir(parents=True, exist_ok=True)
    _validate_runtime(config, physical_gpu_uuid=physical_gpu_uuid)
    inputs = json.loads((model_snapshot / "BRANCHFABRIC_INPUTS.json").read_text())
    import gpu_reclamation_worker as frozen_worker
    from gpu_capacity_calibration_worker import _CompilationLogCapture
    from gpu_reclamation_worker import (
        _create_adapter,
        _prepare_rollouts,
        _run_measured_v10_transaction,
        _transaction_ready_path,
        _wait_for,
    )

    adapter: Any = None
    try:
        started_ns = time.monotonic_ns()
        _write_new(
            barrier_root / f"{role}.engine-started.json",
            {
                "schema_version": "sloforge.branchfabric.retained-engine-start/v1",
                "role": role,
                "device": "gpu0" if role == "serving" else "gpu1",
                "pid": os.getpid(),
                "physical_gpu_uuid": physical_gpu_uuid,
                "engine_started_ns": started_ns,
            },
        )
        adapter = _create_adapter(
            expanded_runtime_config(config), model_snapshot, physical_gpu_uuid
        )
        model_ready_ns = time.monotonic_ns()
        effective, serving_engine, handoff = _run_v11_integrated_calibration_phase(
            frozen_worker=frozen_worker,
            adapter=adapter,
            config=expanded_runtime_config(config),
            inputs=inputs,
            role=role,
            physical_gpu_uuid=physical_gpu_uuid,
            barrier_root=barrier_root,
            model_load_started_ns=started_ns,
            model_ready_ns=model_ready_ns,
        )
        if effective != expanded_runtime_config(config):
            raise RuntimeError("integrated v11 handoff changed its sealed effective config")
        allocator_lifecycle_start_sequence = (
            len(adapter._allocator_epoch_source.lifecycle_events()) if role == "rollout" else None
        )
        prepared = _prepare_rollouts(adapter, effective, inputs) if role == "rollout" else None
        if prepared is not None:
            prepared["allocator_lifecycle_start_sequence"] = allocator_lifecycle_start_sequence
            prepared["allocator_rollout_ready_sequence"] = len(
                adapter._allocator_epoch_source.lifecycle_events()
            )
        _write_new(
            _transaction_ready_path(barrier_root, role),
            {
                **handoff,
                "schema_version": "sloforge.branchfabric.v11-transaction-ready/v1",
                "rollouts_ready": prepared is not None,
                "branch_count": 0 if prepared is None else len(prepared["branches"]),
                "observed_at_monotonic_ns": time.monotonic_ns(),
            },
        )
        start = _wait_for(barrier_root / "start.json", timeout_s=config.maximum_wall_seconds)
        common_start_ns = int(start["start_monotonic_ns"])

        def operation() -> dict[str, Any]:
            targeted = config.execution_mode == "targeted-source-identity-v11"
            if role == "serving":
                if targeted:
                    return run_targeted_source_identity_v11_gpu0(
                        serving_engine,
                        adapter=adapter,
                        inputs=inputs,
                        config=config,
                        start_ns=common_start_ns,
                        barriers=barrier_root,
                    )
                return run_integrated_v11_gpu0(
                    serving_engine,
                    adapter=adapter,
                    inputs=inputs,
                    config=config,
                    start_ns=common_start_ns,
                    barriers=barrier_root,
                )
            assert prepared is not None
            if targeted:
                return run_targeted_source_identity_v11_gpu1(
                    adapter=adapter,
                    prepared=prepared,
                    config=config,
                    physical_gpu_uuid=physical_gpu_uuid,
                    barriers=barrier_root,
                )
            return run_integrated_v11_gpu1(
                serving_engine,
                adapter=adapter,
                prepared=prepared,
                inputs=inputs,
                config=config,
                physical_gpu_uuid=physical_gpu_uuid,
                barriers=barrier_root,
            )

        result = _run_measured_v10_transaction(
            role=role,
            work_root=work_root,
            operation=operation,
            compilation_capture_factory=_CompilationLogCapture,
        )
        result["retained_engine_handoff"] = handoff
        result.update(
            {
                "role": role,
                "pid": os.getpid(),
                "physical_gpu_uuid": physical_gpu_uuid,
                "attempt_id": config.attempt_id,
                "status": "succeeded",
            }
        )
        cache_evidence = serving_engine.evidence()
        if cache_evidence["passed"] is not True:
            raise RuntimeError("integrated v11 serving cache-salt evidence failed")
        result["cache_salt_evidence"] = cache_evidence
        return result
    finally:
        if adapter is not None:
            adapter.cleanup_runtime(timeout_s=float(config.cleanup_timeout_seconds))
        try:
            import torch  # type: ignore[import-not-found]

            torch.cuda.synchronize()
            gc.collect()
            torch.cuda.empty_cache()
        except (ImportError, RuntimeError):
            pass


def _arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", choices=("serving", "rollout"), required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model-snapshot", type=Path, required=True)
    parser.add_argument("--physical-gpu-uuid", required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--barrier-root", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _arguments(argv)
    args.work_root.mkdir(parents=True, exist_ok=True)
    try:
        payload = json.loads(args.config.read_text())
        result = run_worker(
            args.role,
            payload,
            model_snapshot=args.model_snapshot.resolve(strict=True),
            physical_gpu_uuid=args.physical_gpu_uuid,
            work_root=args.work_root,
            barrier_root=args.barrier_root,
        )
        _write_new(args.work_root / "result.json", result)
        return 0
    except BaseException as error:
        failure = {
            "schema_version": "sloforge.branchfabric.experiment-004-v11-worker-failure/v1",
            "status": "failed",
            "role": args.role,
            "error": {"type": type(error).__name__, "message": str(error)},
            "traceback": traceback.format_exc(),
            "observed_ns": time.monotonic_ns(),
        }
        affected_block_ids = getattr(error, "affected_block_ids", None)
        if affected_block_ids is not None:
            failure["affected_block_ids"] = [int(item) for item in affected_block_ids]
        teardown_evidence = getattr(error, "teardown_evidence", None)
        if teardown_evidence is not None:
            from gpu_reclamation_worker_v11 import _public_value

            failure["teardown_evidence"] = _public_value(teardown_evidence)
        try:
            _write_new(args.work_root / "failure.json", failure)
            _write_new(args.barrier_root / "abort.json", failure)
        except FileExistsError:
            pass
        return 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "build_live_v10_config",
    "expanded_runtime_config",
    "frozen_v10_methodology_identity",
    "run_integrated_v11_gpu0",
    "run_integrated_v11_gpu1",
    "run_worker",
    "validate_sanity_guard_pair",
]
