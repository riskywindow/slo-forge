from __future__ import annotations

import math

import pytest
from pydantic import ValidationError

from sloforge.helix.characterization.gpu_capacity_calibration import (
    NS_PER_SECOND,
    CapacityCalibrationBounds,
    CapacityProbeRaw,
    CapacityRequestObservation,
    ProbeTopology,
    ProbeVerdict,
    RequestTerminalState,
    build_probe_plan,
    choose_spike,
    early_stop_reason,
    evaluate_probe,
    recommend_next_probe,
)


def _observations(
    *,
    topology: ProbeTopology,
    rate_rps: float,
    service_capacity_rps: float,
    probe_id: str,
) -> CapacityProbeRaw:
    plan = build_probe_plan(
        probe_id=probe_id,
        seed=41,
        topology=topology,
        configured_rate_rps=rate_rps,
        start_ns=1_000_000_000,
    )
    next_available = {"gpu0": 0, "gpu1": 0}
    observations: list[CapacityRequestObservation] = []
    per_gpu_capacity = (
        service_capacity_rps if topology == ProbeTopology.GPU0_ONLY else service_capacity_rps / 2.0
    )
    service_ns = round(NS_PER_SECOND / per_gpu_capacity)
    for arrival in plan.arrivals:
        completed_ns = max(
            arrival.scheduled_arrival_ns + service_ns,
            next_available[arrival.assigned_device] + service_ns,
        )
        next_available[arrival.assigned_device] = completed_ns
        observations.append(
            CapacityRequestObservation(
                request_id=arrival.request_id,
                global_sequence=arrival.global_sequence,
                device=arrival.assigned_device,
                scheduled_arrival_ns=arrival.scheduled_arrival_ns,
                offered_ns=arrival.scheduled_arrival_ns,
                enqueued_ns=arrival.scheduled_arrival_ns,
                admitted_ns=arrival.scheduled_arrival_ns,
                first_token_ns=completed_ns - service_ns // 2,
                completed_ns=completed_ns,
                terminated_ns=completed_ns,
                requested_output_tokens=64,
                emitted_tokens=64,
                terminal_state=RequestTerminalState.COMPLETED,
            )
        )
    return CapacityProbeRaw(
        plan=plan,
        observations=tuple(observations),
        tail_drain_end_ns=plan.measurement_end_ns + 5 * NS_PER_SECOND,
        probe_end_ns=plan.measurement_end_ns + 5 * NS_PER_SECOND,
        tail_drain_seconds=5.0,
    )


def test_calibration_reservation_uses_measured_noise_allowance_and_preserves_v10() -> None:
    bounds = CapacityCalibrationBounds(seed=41, remaining_gpu_seconds_before=1616.2042090820032)

    assert bounds.maximum_gpu_seconds == 424.0
    assert bounds.maximum_gpu_seconds / bounds.remaining_gpu_seconds_before == pytest.approx(
        0.26234308611337553
    )
    assert 1616.2042090820032 - bounds.maximum_gpu_seconds - 680.0 == pytest.approx(
        512.2042090820032
    )
    with pytest.raises(ValidationError, match="fraction of remaining budget"):
        CapacityCalibrationBounds(
            seed=41,
            remaining_gpu_seconds_before=1616.2042090820032,
            maximum_allocation_wall_seconds=212.0,
            maximum_calibration_fraction=0.26,
        )


def test_probe_plan_is_one_global_clock_with_phase_appropriate_routing() -> None:
    one = build_probe_plan(
        probe_id="capacity-one-20",
        seed=41,
        topology=ProbeTopology.GPU0_ONLY,
        configured_rate_rps=20.0,
        start_ns=17,
    )
    two = build_probe_plan(
        probe_id="capacity-two-30",
        seed=41,
        topology=ProbeTopology.TWO_GPU_ROUND_ROBIN,
        configured_rate_rps=30.0,
        start_ns=17,
    )

    assert len(one.arrivals) == 220
    assert len(two.arrivals) == 330
    assert {item.assigned_device for item in one.arrivals} == {"gpu0"}
    assert tuple(item.assigned_device for item in two.arrivals[:4]) == (
        "gpu0",
        "gpu1",
        "gpu0",
        "gpu1",
    )
    assert [item.scheduled_arrival_ns for item in two.arrivals] == sorted(
        item.scheduled_arrival_ns for item in two.arrivals
    )
    assert len({item.request_id for item in two.arrivals}) == len(two.arrivals)


def test_probe_evaluation_distinguishes_stable_capacity_from_overload() -> None:
    stable = evaluate_probe(
        _observations(
            topology=ProbeTopology.GPU0_ONLY,
            rate_rps=20.0,
            service_capacity_rps=20.0,
            probe_id="capacity-one-20",
        )
    )
    overloaded = evaluate_probe(
        _observations(
            topology=ProbeTopology.GPU0_ONLY,
            rate_rps=25.0,
            service_capacity_rps=20.0,
            probe_id="capacity-one-25",
        )
    )
    two_gpu = evaluate_probe(
        _observations(
            topology=ProbeTopology.TWO_GPU_ROUND_ROBIN,
            rate_rps=30.0,
            service_capacity_rps=40.0,
            probe_id="capacity-two-30",
        )
    )

    assert stable.verdict == ProbeVerdict.SUSTAINABLE
    assert stable.completed_rate_rps >= stable.observed_offered_rate_rps
    assert not stable.queue_persistent_positive_drift
    assert stable.p95_ttft_seconds is not None and stable.p95_ttft_seconds <= 2.0
    assert overloaded.verdict == ProbeVerdict.UNSUSTAINABLE
    assert overloaded.queue_persistent_positive_drift
    assert not overloaded.throughput_pass
    assert two_gpu.verdict == ProbeVerdict.SUSTAINABLE


def test_probe_fails_closed_on_missing_request_or_misrouting() -> None:
    raw = _observations(
        topology=ProbeTopology.TWO_GPU_ROUND_ROBIN,
        rate_rps=30.0,
        service_capacity_rps=40.0,
        probe_id="capacity-two-30",
    )
    missing = raw.model_copy(update={"observations": raw.observations[:-1]})
    wrong = raw.observations[1].model_copy(update={"device": "gpu0"})
    misrouted = raw.model_copy(
        update={"observations": (raw.observations[0], wrong, *raw.observations[2:])}
    )

    assert evaluate_probe(missing).verdict == ProbeVerdict.INCONCLUSIVE
    routing = evaluate_probe(misrouted)
    assert routing.verdict == ProbeVerdict.INCONCLUSIVE
    assert not routing.routing_verified


def test_adaptive_search_uses_three_probes_and_selects_measured_overload() -> None:
    stable_one = evaluate_probe(
        _observations(
            topology=ProbeTopology.GPU0_ONLY,
            rate_rps=20.0,
            service_capacity_rps=20.0,
            probe_id="capacity-one-20",
        )
    )
    unstable_one = evaluate_probe(
        _observations(
            topology=ProbeTopology.GPU0_ONLY,
            rate_rps=25.0,
            service_capacity_rps=20.0,
            probe_id="capacity-one-25",
        )
    )
    stable_two = evaluate_probe(
        _observations(
            topology=ProbeTopology.TWO_GPU_ROUND_ROBIN,
            rate_rps=30.0,
            service_capacity_rps=40.0,
            probe_id="capacity-two-30",
        )
    )

    first = recommend_next_probe(())
    second = recommend_next_probe((stable_one,))
    third = recommend_next_probe((stable_one, unstable_one))
    ready = recommend_next_probe((stable_one, unstable_one, stable_two))

    assert (first.topology, first.rate_rps) == (ProbeTopology.GPU0_ONLY, 20.0)
    assert (second.topology, second.rate_rps) == (ProbeTopology.GPU0_ONLY, 25.0)
    assert (third.topology, third.rate_rps) == (
        ProbeTopology.TWO_GPU_ROUND_ROBIN,
        30.0,
    )
    assert ready.status == "ready"
    assert ready.selection == choose_spike((stable_one, unstable_one, stable_two))
    assert ready.selection is not None
    assert ready.selection.lambda_1_rps == 20.0
    assert ready.selection.lambda_2_rps == 30.0
    assert ready.selection.lambda_spike_rps == 25.0
    assert ready.selection.one_gpu_unsustainable_probe_id == "capacity-one-25"
    assert math.isclose(ready.selection.spike_over_lambda_1, 1.25)
    assert math.isclose(ready.selection.spike_over_lambda_2, 5 / 6)


def test_adaptive_search_uses_v9_lower_capacity_prior_without_a_fourth_probe() -> None:
    overloaded_20 = evaluate_probe(
        _observations(
            topology=ProbeTopology.GPU0_ONLY,
            rate_rps=20.0,
            service_capacity_rps=15.0,
            probe_id="capacity-one-20",
        )
    )
    stable_1725 = evaluate_probe(
        _observations(
            topology=ProbeTopology.GPU0_ONLY,
            rate_rps=17.25,
            service_capacity_rps=17.25,
            probe_id="capacity-one-1725",
        )
    )

    lower_probe = recommend_next_probe((overloaded_20,))
    assert (lower_probe.topology, lower_probe.rate_rps) == (ProbeTopology.GPU0_ONLY, 17.25)
    two_probe = recommend_next_probe((overloaded_20, stable_1725))
    assert (two_probe.topology, two_probe.rate_rps) == (
        ProbeTopology.TWO_GPU_ROUND_ROBIN,
        23.5,
    )


def test_inconclusive_probe_and_probe_limit_block_v10() -> None:
    raw = _observations(
        topology=ProbeTopology.GPU0_ONLY,
        rate_rps=20.0,
        service_capacity_rps=20.0,
        probe_id="capacity-one-20",
    )
    inconclusive = evaluate_probe(raw.model_copy(update={"observations": raw.observations[:-1]}))
    assert recommend_next_probe((inconclusive,)).status == "blocked"

    sustainable = evaluate_probe(raw)
    repeated = tuple(
        sustainable.model_copy(update={"probe_id": f"capacity-one-{index}"}) for index in range(3)
    )
    bounded = recommend_next_probe(repeated, maximum_probes=3)
    assert bounded.status == "blocked"
    assert "bounded search" in bounded.reason


def test_early_stop_requires_ten_seconds_and_strong_evidence() -> None:
    overloaded = evaluate_probe(
        _observations(
            topology=ProbeTopology.GPU0_ONLY,
            rate_rps=25.0,
            service_capacity_rps=20.0,
            probe_id="capacity-one-25",
        )
    )
    stable = evaluate_probe(
        _observations(
            topology=ProbeTopology.TWO_GPU_ROUND_ROBIN,
            rate_rps=30.0,
            service_capacity_rps=40.0,
            probe_id="capacity-two-30",
        )
    )

    assert early_stop_reason(overloaded, elapsed_seconds=4.99) is None
    assert early_stop_reason(stable, elapsed_seconds=9.99) is None
    assert early_stop_reason(stable, elapsed_seconds=10.0) == (
        "all empirical sustainability gates passed after the minimum window"
    )


def test_two_gpu_overload_with_aborted_tail_is_preserved_as_conclusive() -> None:
    complete = _observations(
        topology=ProbeTopology.TWO_GPU_ROUND_ROBIN,
        rate_rps=30.0,
        service_capacity_rps=40.0,
        probe_id="capacity-two-30-overload",
    )
    measurement_ids = {
        item.request_id
        for item in complete.plan.arrivals
        if item.scheduled_arrival_ns >= complete.plan.measurement_start_ns
    }
    abort_ids = set(sorted(measurement_ids)[-120:])
    observations = tuple(
        item.model_copy(
            update={
                "first_token_ns": None,
                "completed_ns": None,
                "terminated_ns": complete.tail_drain_end_ns,
                "emitted_tokens": 0,
                "terminal_state": RequestTerminalState.ABORTED,
            }
        )
        if item.request_id in abort_ids
        else item
        for item in complete.observations
    )
    raw = complete.model_copy(update={"observations": observations})

    result = evaluate_probe(raw)
    assert result.verdict == ProbeVerdict.UNSUSTAINABLE
    assert result.complete_request_accounting
    assert result.planned_requests == len(complete.plan.arrivals)
    assert result.observed_requests == len(complete.observations)
    assert result.measurement_requests == 300
    assert result.aborted_requests == 120
    assert not result.all_measurement_requests_completed
    assert not result.throughput_pass


def test_half_second_admission_bursts_are_methodologically_rejected() -> None:
    raw = _observations(
        topology=ProbeTopology.GPU0_ONLY,
        rate_rps=20.0,
        service_capacity_rps=20.0,
        probe_id="capacity-one-bursted",
    )
    observations = []
    for item in raw.observations:
        admitted_ns = (item.scheduled_arrival_ns // 500_000_000 + 1) * 500_000_000
        observations.append(
            item.model_copy(
                update={
                    "admitted_ns": admitted_ns,
                    "first_token_ns": admitted_ns + 10_000_000,
                    "completed_ns": admitted_ns + 20_000_000,
                    "terminated_ns": admitted_ns + 20_000_000,
                }
            )
        )
    result = evaluate_probe(raw.model_copy(update={"observations": tuple(observations)}))

    assert result.verdict == ProbeVerdict.INCONCLUSIVE
    assert not result.admission_cadence_verified
    assert result.maximum_admission_burst_50ms > result.admission_burst_limit_50ms


def test_smooth_admission_lag_is_overload_evidence_not_loadgen_invalidity() -> None:
    raw = _observations(
        topology=ProbeTopology.GPU0_ONLY,
        rate_rps=25.0,
        service_capacity_rps=20.0,
        probe_id="capacity-one-smooth-lag",
    )
    lag_ns = 500_000_000
    observations = tuple(
        item.model_copy(
            update={
                "admitted_ns": item.admitted_ns + lag_ns,
                "first_token_ns": item.first_token_ns + lag_ns,
                "completed_ns": item.completed_ns + lag_ns,
                "terminated_ns": item.terminated_ns + lag_ns,
            }
        )
        for item in raw.observations
        if item.admitted_ns is not None
        and item.first_token_ns is not None
        and item.completed_ns is not None
    )
    result = evaluate_probe(raw.model_copy(update={"observations": observations}))

    assert result.verdict == ProbeVerdict.UNSUSTAINABLE
    assert result.admission_cadence_verified
    assert not result.admission_lag_pass
    assert result.maximum_admission_burst_50ms <= result.admission_burst_limit_50ms
