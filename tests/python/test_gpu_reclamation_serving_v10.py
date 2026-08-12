from __future__ import annotations

import pytest
from pydantic import ValidationError

from sloforge.helix.characterization.gpu_reclamation_serving_v10 import (
    ExecutedServingRequest,
    assess_serving_recovery,
    audit_serving_execution,
    build_global_serving_plan,
    build_reclamation_trigger_evidence,
    device_phase_counts,
    outstanding_depth,
)

NS = 1_000_000_000


def _plan():  # type: ignore[no-untyped-def]
    return build_global_serving_plan(
        attempt_id="exp004-v10-fixture",
        seed=41,
        control_start_ns=0,
        spike_start_ns=2 * NS,
        gpu1_route_start_ns=4 * NS,
        restore_start_ns=8 * NS,
        end_ns=10 * NS,
        control_rate_per_second=4.0,
        spike_rate_per_second=6.0,
        restore_rate_per_second=3.0,
    )


def _observation(request, *, duration_ns: int = NS // 2):  # type: ignore[no-untyped-def]
    admitted = request.scheduled_arrival_ns + 1
    service = admitted + 1
    first = service + 1
    token_times = tuple(first + index for index in range(64))
    return ExecutedServingRequest(
        request_id=request.request_id,
        sequence=request.sequence,
        phase=request.phase,
        device=request.planned_device,
        scheduled_arrival_ns=request.scheduled_arrival_ns,
        admitted_ns=admitted,
        service_start_ns=service,
        first_token_ns=first,
        completed_ns=max(token_times[-1], request.scheduled_arrival_ns + duration_ns),
        token_timestamps_ns=token_times,
        output_token_ids=tuple(range(64)),
    )


def test_global_plan_keeps_one_spike_clock_and_changes_only_routing() -> None:
    first = _plan()
    second = _plan()
    assert first == second
    assert first.plan_sha256 == second.plan_sha256
    assert first.epochs[1].rate_per_second == first.epochs[2].rate_per_second == 6.0

    counts = device_phase_counts(first)
    assert counts == {
        "control": {"gpu0": 8, "gpu1": 0},
        "gpu0-overload": {"gpu0": 12, "gpu1": 0},
        "two-gpu-recovery": {"gpu0": 12, "gpu1": 12},
        "restore-interference": {"gpu0": 6, "gpu1": 0},
    }
    overload = [item for item in first.requests if item.phase == "gpu0-overload"]
    recovery = [item for item in first.requests if item.phase == "two-gpu-recovery"]
    # Six total arrivals/s exist on either side of GPU1 enable.  Recovery is
    # not a second independent 6 rps stream.
    assert len(overload) == 2 * 6
    assert len(recovery) == 4 * 6
    assert {item.planned_device for item in overload} == {"gpu0"}
    assert {item.planned_device for item in recovery} == {"gpu0", "gpu1"}
    assert all(item.requested_output_tokens == 64 for item in first.requests)


def test_gpu1_enable_does_not_restart_an_unaligned_spike_clock() -> None:
    plan = build_global_serving_plan(
        attempt_id="exp004-v10-unaligned",
        seed=41,
        control_start_ns=0,
        spike_start_ns=2 * NS,
        gpu1_route_start_ns=4 * NS + 50_000_000,
        restore_start_ns=8 * NS,
        end_ns=10 * NS,
        control_rate_per_second=4.0,
        spike_rate_per_second=6.0,
        restore_rate_per_second=3.0,
    )
    spike = [item for item in plan.requests if item.phase in {"gpu0-overload", "two-gpu-recovery"}]
    # The last GPU0-only arrival remains on the clock rooted at spike_start;
    # GPU1 enable at 4.05s does not inject a new boundary arrival.
    assert spike[12].scheduled_arrival_ns == 4 * NS
    assert spike[13].scheduled_arrival_ns == 4 * NS + 166_666_666
    assert all(
        item.planned_device == "gpu0"
        for item in spike
        if item.scheduled_arrival_ns < 4 * NS + 50_000_000
    )


def test_execution_audit_requires_exact_route_and_active_gpu0_restore() -> None:
    plan = _plan()
    observations = tuple(_observation(request) for request in plan.requests)
    audit = audit_serving_execution(plan, observations)
    assert audit.passed
    assert audit.gpu0_restore_completions > 0
    assert audit.gpu0_restore_emitted_tokens > 0
    assert audit.gpu0_restore_ttft_samples > 0

    gpu1_recovery = next(row for row in observations if row.device == "gpu1")
    bad = gpu1_recovery.model_copy(update={"device": "gpu0"})
    replaced = tuple(bad if row.request_id == bad.request_id else row for row in observations)
    invalid = audit_serving_execution(plan, replaced)
    assert not invalid.passed
    assert not invalid.routing_matches_plan


def test_execution_records_fail_closed_on_token_count_and_causal_timestamps() -> None:
    request = _plan().requests[0]
    good = _observation(request)
    with pytest.raises(ValidationError, match="at least 64 items"):
        ExecutedServingRequest.model_validate(
            {**good.model_dump(mode="python"), "output_token_ids": (1,)}, strict=True
        )
    with pytest.raises(ValidationError, match="timestamps are not causal"):
        ExecutedServingRequest.model_validate(
            {**good.model_dump(mode="python"), "service_start_ns": good.admitted_ns - 1},
            strict=True,
        )


def test_trigger_uses_outstanding_work_not_host_admission_delay() -> None:
    plan = _plan()
    # During the two-second GPU0 overload, requests complete slowly enough for
    # outstanding work to grow even though every request is admitted promptly.
    observations = tuple(
        _observation(
            request,
            duration_ns=(3 * NS if request.phase == "gpu0-overload" else NS // 2),
        )
        for request in plan.requests
    )
    depth_at_start = outstanding_depth(observations, timestamp_ns=2 * NS)
    assert depth_at_start > 0
    assert outstanding_depth(observations, timestamp_ns=4 * NS) > depth_at_start
    evidence = build_reclamation_trigger_evidence(
        observations,
        window_start_ns=2 * NS,
        window_end_ns=4 * NS,
        offered_rate_per_second=6.0,
    )
    assert evidence.overload_confirmed
    assert evidence.queue_depth_slope_per_second > 0
    assert "queue_positive_drift" in evidence.trigger_reason
    assert "offered_above_completed" in evidence.trigger_reason

    sustainable = tuple(_observation(request, duration_ns=1) for request in plan.requests)
    no_trigger = build_reclamation_trigger_evidence(
        sustainable,
        window_start_ns=2 * NS,
        window_end_ns=4 * NS,
        offered_rate_per_second=6.0,
    )
    assert not no_trigger.overload_confirmed
    assert no_trigger.trigger_reason == ()


def test_recovery_gate_requires_excess_capacity_drain_and_full_stability() -> None:
    plan = _plan()
    observations = tuple(
        _observation(
            request,
            duration_ns=(3 * NS if request.phase == "gpu0-overload" else NS // 2),
        )
        for request in plan.requests
    )
    recovered = assess_serving_recovery(
        plan,
        observations,
        gpu1_first_useful_ns=4 * NS,
        recovery_queue_threshold=3,
        stability_window_ns=NS,
    )
    assert recovered.two_gpu_excess_capacity_pass
    assert recovered.queue_drain_pass
    assert recovered.slo_restoration_pass
    assert recovered.pre_restore_stability_pass
    assert recovered.restore_eligible

    too_strict = assess_serving_recovery(
        plan,
        observations,
        gpu1_first_useful_ns=4 * NS,
        recovery_queue_threshold=2,
        stability_window_ns=NS,
    )
    assert too_strict.two_gpu_excess_capacity_pass
    assert too_strict.queue_drain_pass
    assert not too_strict.pre_restore_stability_pass
    assert not too_strict.restore_eligible


def test_plan_rejects_noncausal_boundaries() -> None:
    with pytest.raises(ValueError, match="strictly ordered"):
        build_global_serving_plan(
            attempt_id="exp004-v10-fixture",
            seed=41,
            control_start_ns=0,
            spike_start_ns=2 * NS,
            gpu1_route_start_ns=2 * NS,
            restore_start_ns=8 * NS,
            end_ns=10 * NS,
            control_rate_per_second=4.0,
            spike_rate_per_second=6.0,
            restore_rate_per_second=3.0,
        )
