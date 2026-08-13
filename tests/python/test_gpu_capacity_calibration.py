from __future__ import annotations

import math
from pathlib import Path

import pytest
from pydantic import ValidationError

from sloforge.helix.characterization.gpu_capacity_calibration import (
    NS_PER_SECOND,
    CapacityCalibrationBounds,
    CapacityProbeRaw,
    CapacityProbeResult,
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

ROOT = Path(__file__).resolve().parents[2]
V2_ONE_GPU_RAW_ROOT = (
    ROOT / "artifacts/branchfabric/gpu-validation/experiment-004/raw/modal/"
    "exp004-v10-integrated-s41-v2/calibration/one-gpu"
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
    assert stable.completed_rate_rps >= stable.observed_offered_rate_rps * 0.95
    assert not stable.queue_persistent_positive_drift
    assert stable.p95_ttft_seconds is not None and stable.p95_ttft_seconds <= 2.0
    assert overloaded.verdict == ProbeVerdict.UNSUSTAINABLE
    assert overloaded.queue_persistent_positive_drift
    assert not overloaded.throughput_pass
    assert two_gpu.verdict == ProbeVerdict.SUSTAINABLE


def test_completion_rate_counts_interval_events_but_ttft_uses_measurement_cohort() -> None:
    raw = _observations(
        topology=ProbeTopology.GPU0_ONLY,
        rate_rps=20.0,
        service_capacity_rps=20.0,
        probe_id="capacity-warmup-exclusion",
    )
    result = evaluate_probe(raw)
    all_completions = sum(
        item.completed_ns is not None
        and raw.plan.measurement_start_ns <= item.completed_ns < raw.plan.measurement_end_ns
        for item in raw.observations
    )
    cohort_ids = {
        item.request_id
        for item in raw.plan.arrivals
        if raw.plan.measurement_start_ns <= item.scheduled_arrival_ns < raw.plan.measurement_end_ns
    }
    assert all_completions > 0
    assert result.completed_rate_estimator == "measurement-interval-completion-events"
    assert result.completed_rate_sample_count == all_completions
    assert result.interval_completed_requests == all_completions
    assert result.completed_rate_rps == all_completions / 10.0
    cohort_ttfts = tuple(
        (item.first_token_ns - item.scheduled_arrival_ns) / NS_PER_SECOND
        for item in raw.observations
        if item.request_id in cohort_ids and item.first_token_ns is not None
    )
    assert result.p95_ttft_seconds == pytest.approx(max(cohort_ttfts))
    assert result.queue_flow_conservation_pass
    assert result.queue_flow_conservation_error_requests == 0


def test_event_window_and_queue_flow_avoid_cohort_completion_boundary_bias() -> None:
    raw = _observations(
        topology=ProbeTopology.GPU0_ONLY,
        rate_rps=10.0,
        service_capacity_rps=10.0,
        probe_id="capacity-boundary-displacement",
    )
    measurement_ids = {
        item.request_id
        for item in raw.plan.arrivals
        if raw.plan.measurement_start_ns <= item.scheduled_arrival_ns < raw.plan.measurement_end_ns
    }
    completion_displacement_ns = 900_000_000
    delayed = raw.model_copy(
        update={
            "observations": tuple(
                item.model_copy(
                    update={
                        "completed_ns": item.completed_ns + completion_displacement_ns,
                        "terminated_ns": item.terminated_ns + completion_displacement_ns,
                    }
                )
                for item in raw.observations
            )
        }
    )
    fixed_window_completions = sum(
        item.request_id in measurement_ids
        and item.completed_ns is not None
        and raw.plan.measurement_start_ns <= item.completed_ns < raw.plan.measurement_end_ns
        for item in delayed.observations
    )

    result = evaluate_probe(delayed)
    assert fixed_window_completions == 90
    assert result.completed_rate_sample_count == 100
    assert result.completed_rate_rps == pytest.approx(10.0)
    assert result.interval_offered_requests == 100
    assert result.queue_flow_expected_end_depth == result.queue_flow_observed_end_depth
    assert result.queue_flow_conservation_error_requests == 0
    assert result.queue_flow_conservation_pass
    assert abs(result.queue_depth_start - result.queue_depth_end) <= 1
    assert not result.queue_persistent_positive_drift
    assert result.throughput_pass
    assert result.verdict is ProbeVerdict.SUSTAINABLE


def test_v2_raw_probes_recompute_with_conserved_event_window_capacity() -> None:
    expected = {
        10.0: (10.0, 9, 9, ProbeVerdict.SUSTAINABLE),
        15.0: (14.2, 15, 23, ProbeVerdict.UNSUSTAINABLE),
        17.25: (14.0, 18, 50, ProbeVerdict.UNSUSTAINABLE),
        20.0: (14.4, 20, 76, ProbeVerdict.UNSUSTAINABLE),
    }
    paths = tuple(sorted(V2_ONE_GPU_RAW_ROOT.glob("*/raw.json")))
    assert len(paths) == len(expected)
    for path in paths:
        raw = CapacityProbeRaw.model_validate_json(path.read_text(), strict=True)
        result = evaluate_probe(raw)
        completed, queue_start, queue_end, verdict = expected[result.configured_rate_rps]
        assert result.completed_rate_rps == pytest.approx(completed)
        assert result.queue_flow_start_depth == queue_start
        assert result.queue_flow_expected_end_depth == queue_end
        assert result.queue_flow_observed_end_depth == queue_end
        assert result.queue_flow_conservation_pass
        assert result.queue_flow_conservation_error_requests == 0
        assert result.verdict is verdict
    stable = evaluate_probe(
        CapacityProbeRaw.model_validate_json(
            next(V2_ONE_GPU_RAW_ROOT.glob("capacity-03-*/raw.json")).read_text(), strict=True
        )
    )
    assert stable.interval_offered_requests == stable.interval_completed_requests == 100
    assert stable.p95_ttft_seconds == pytest.approx(0.0424315944)


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
    assert not evaluate_probe(missing).queue_flow_conservation_pass
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
            rate_rps=35.0,
            service_capacity_rps=40.0,
            probe_id="capacity-two-35",
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
        35.0,
    )
    assert ready.status == "ready"
    assert ready.selection == choose_spike((stable_one, unstable_one, stable_two))
    assert ready.selection is not None
    assert ready.selection.lambda_1_rps == 20.0
    assert ready.selection.lambda_2_rps == 35.0
    assert ready.selection.lambda_spike_rps == 25.0
    assert ready.selection.one_gpu_unsustainable_probe_id == "capacity-one-25"
    assert math.isclose(ready.selection.spike_over_lambda_1, 1.25)
    assert math.isclose(ready.selection.spike_over_lambda_2, 5 / 7)


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
        25.0,
    )


def test_selected_load_fails_closed_without_preferred_two_gpu_headroom() -> None:
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
    adr_only_two_gpu_point = evaluate_probe(
        _observations(
            topology=ProbeTopology.TWO_GPU_ROUND_ROBIN,
            rate_rps=30.0,
            service_capacity_rps=40.0,
            probe_id="capacity-two-30",
        )
    )
    results = (stable_one, unstable_one, adr_only_two_gpu_point)

    assert choose_spike(results) is None
    action = recommend_next_probe(results)
    assert action.status == "probe"
    assert (action.topology, action.rate_rps) == (
        ProbeTopology.TWO_GPU_ROUND_ROBIN,
        35.0,
    )
    assert action.selection is None


def test_adaptive_search_descends_past_two_overloads_and_finishes_within_eight() -> None:
    def point(rate: float, capacity: float, probe_id: str) -> CapacityProbeResult:
        return evaluate_probe(
            _observations(
                topology=ProbeTopology.GPU0_ONLY,
                rate_rps=rate,
                service_capacity_rps=capacity,
                probe_id=probe_id,
            )
        )

    overloaded_20 = point(20.0, 15.0, "capacity-one-20")
    overloaded_1725 = point(17.25, 15.0, "capacity-one-1725")
    stable_15 = point(15.0, 15.0, "capacity-one-15")
    stable_two = evaluate_probe(
        _observations(
            topology=ProbeTopology.TWO_GPU_ROUND_ROBIN,
            rate_rps=23.5,
            service_capacity_rps=35.0,
            probe_id="capacity-two-235",
        )
    )

    assert recommend_next_probe((overloaded_20,)).rate_rps == 17.25
    assert recommend_next_probe((overloaded_20, overloaded_1725)).rate_rps == 15.0
    next_overload = recommend_next_probe((overloaded_20, overloaded_1725, stable_15))
    assert (next_overload.topology, next_overload.rate_rps) == (
        ProbeTopology.TWO_GPU_ROUND_ROBIN,
        23.5,
    )
    ready = recommend_next_probe(
        (overloaded_20, overloaded_1725, stable_15, stable_two),
        maximum_probes=8,
    )
    assert ready.status == "ready"
    assert ready.selection is not None
    assert ready.selection.lambda_1_rps == 15.0
    assert ready.selection.lambda_spike_rps == 17.25


def test_adaptive_search_synthesizes_fractional_spike_for_low_lambda1() -> None:
    overloaded_15 = evaluate_probe(
        _observations(
            topology=ProbeTopology.GPU0_ONLY,
            rate_rps=15.0,
            service_capacity_rps=14.0,
            probe_id="capacity-one-15",
        )
    )
    stable_10 = evaluate_probe(
        _observations(
            topology=ProbeTopology.GPU0_ONLY,
            rate_rps=10.0,
            service_capacity_rps=10.0,
            probe_id="capacity-one-10",
        )
    )
    fractional = recommend_next_probe((overloaded_15, stable_10), maximum_probes=8)
    assert (fractional.topology, fractional.rate_rps) == (ProbeTopology.GPU0_ONLY, 12.0)
    overloaded_12 = evaluate_probe(
        _observations(
            topology=ProbeTopology.GPU0_ONLY,
            rate_rps=12.0,
            service_capacity_rps=10.0,
            probe_id="capacity-one-12",
        )
    )
    two_gpu = recommend_next_probe((overloaded_15, stable_10, overloaded_12), maximum_probes=8)
    assert two_gpu.topology == ProbeTopology.TWO_GPU_ROUND_ROBIN
    assert two_gpu.rate_rps is not None and 12.0 / two_gpu.rate_rps <= 0.80


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


def test_adaptive_search_continues_above_sustainable_grid_peak_but_stays_bounded() -> None:
    stable = evaluate_probe(
        _observations(
            topology=ProbeTopology.GPU0_ONLY,
            rate_rps=35.0,
            service_capacity_rps=35.0,
            probe_id="capacity-one-35",
        )
    )
    above_grid = recommend_next_probe((stable,), maximum_probes=8)
    assert (above_grid.topology, above_grid.rate_rps) == (
        ProbeTopology.GPU0_ONLY,
        43.75,
    )
    assert "96-rps prior cap" in above_grid.reason

    rates = (20.0, 25.0, 30.0, 35.0, 43.75, 54.688, 68.36, 85.45)
    all_stable = tuple(
        stable.model_copy(
            update={
                "probe_id": f"capacity-one-{index}",
                "configured_rate_rps": rate,
            }
        )
        for index, rate in enumerate(rates)
    )
    before_limit = recommend_next_probe(all_stable[:-1], maximum_probes=8)
    assert (before_limit.topology, before_limit.rate_rps) == (
        ProbeTopology.GPU0_ONLY,
        85.45,
    )
    at_limit = recommend_next_probe(all_stable, maximum_probes=8)
    assert at_limit.status == "blocked"
    assert "bounded search" in at_limit.reason


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
