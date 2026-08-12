from __future__ import annotations

from collections.abc import Callable

import pytest
from pydantic import ValidationError

from sloforge.helix.characterization.gpu_reclamation_serving import (
    NS_PER_SECOND,
    ArrivalPhase,
    ObservationOutcome,
    PhaseInterval,
    ServingMeasurementPlan,
    ServingObservation,
    ServingSLO,
    ServingSpikeConfig,
    ServingWorkload,
    SLOStabilityConfig,
    WeightedTokenDistribution,
    calculate_serving_interference,
    evaluate_serving_slo,
    experiment_004_serving_spike_config,
    find_serving_slo_restoration,
    generate_serving_spike,
    measure_serving_intervals,
    summarize_latency,
    workload_phase_intervals,
)

MS = 1_000_000


def _observations(
    workload: ServingWorkload,
    *,
    ttft_for_arrival: Callable[[int], int] = lambda _arrival: 10 * MS,
    inter_token_ns: int = 5 * MS,
) -> tuple[ServingObservation, ...]:
    observations = []
    for request in workload.requests:
        ttft_ns = ttft_for_arrival(request.arrival_ns)
        service_start_ns = request.arrival_ns + max(0, ttft_ns - MS)
        first_token_ns = request.arrival_ns + ttft_ns
        token_timestamps = tuple(
            first_token_ns + index * inter_token_ns
            for index in range(request.requested_output_tokens)
        )
        observations.append(
            ServingObservation(
                request_id=request.request_id,
                arrival_ns=request.arrival_ns,
                service_start_ns=service_start_ns,
                token_timestamps_ns=token_timestamps,
                completion_ns=token_timestamps[-1] + MS,
                outcome=ObservationOutcome.COMPLETED,
                device="gpu-0",
            )
        )
    return tuple(observations)


def test_serving_spike_is_seeded_stable_and_has_exact_cadence() -> None:
    config = experiment_004_serving_spike_config(
        seed=41,
        control_duration_ns=NS_PER_SECOND,
        spike_duration_ns=NS_PER_SECOND,
        recovery_duration_ns=NS_PER_SECOND,
        control_interarrival_ns=250 * MS,
        spike_interarrival_ns=100 * MS,
    )

    first = generate_serving_spike(config, start_ns=17 * NS_PER_SECOND)
    second = generate_serving_spike(config, start_ns=17 * NS_PER_SECOND)
    other_seed = generate_serving_spike(
        config.model_copy(update={"seed": 73}), start_ns=17 * NS_PER_SECOND
    )

    assert first == second
    assert first.workload_id == second.workload_id
    assert len(first.requests) == 18
    assert [request.phase for request in first.requests].count("control") == 4
    assert [request.phase for request in first.requests].count("spike") == 10
    assert [request.phase for request in first.requests].count("recovery") == 4
    assert [request.arrival_ns for request in first.requests[:4]] == [
        17 * NS_PER_SECOND + offset * 250 * MS for offset in range(4)
    ]
    assert first.workload_id != other_seed.workload_id
    assert [
        (request.prompt_tokens, request.requested_output_tokens) for request in first.requests
    ] != [
        (request.prompt_tokens, request.requested_output_tokens) for request in other_seed.requests
    ]


def test_arrival_and_measurement_intervals_reject_overlap() -> None:
    with pytest.raises(ValidationError, match="ordered and non-overlapping"):
        ServingSpikeConfig(
            seed=1,
            control_phase="control",
            spike_phase="spike",
            phases=(
                ArrivalPhase(
                    name="control",
                    start_offset_ns=0,
                    end_offset_ns=100,
                    interarrival_ns=10,
                ),
                ArrivalPhase(
                    name="spike",
                    start_offset_ns=99,
                    end_offset_ns=200,
                    interarrival_ns=5,
                ),
            ),
        )

    with pytest.raises(ValidationError, match="ordered and non-overlapping"):
        ServingMeasurementPlan(
            intervals=(
                PhaseInterval(name="a", start_ns=0, end_ns=100),
                PhaseInterval(name="b", start_ns=99, end_ns=200),
            )
        )


def test_interval_measurement_separates_cohort_latency_from_event_throughput() -> None:
    config = ServingSpikeConfig(
        seed=41,
        control_phase="control",
        spike_phase="checkpoint",
        phases=(
            ArrivalPhase(
                name="control", start_offset_ns=0, end_offset_ns=100 * MS, interarrival_ns=25 * MS
            ),
            ArrivalPhase(
                name="checkpoint",
                start_offset_ns=100 * MS,
                end_offset_ns=200 * MS,
                interarrival_ns=25 * MS,
            ),
        ),
        prompt_tokens=WeightedTokenDistribution(values=(16,), weights=(1,)),
        output_tokens=WeightedTokenDistribution(values=(4,), weights=(1,)),
    )
    workload = generate_serving_spike(config)
    observations = _observations(
        workload,
        ttft_for_arrival=lambda arrival: 10 * MS if arrival < 100 * MS else 20 * MS,
        inter_token_ns=5 * MS,
    )
    measurement = measure_serving_intervals(
        workload,
        observations,
        ServingMeasurementPlan(intervals=workload_phase_intervals(workload)),
    )
    control, checkpoint = measurement.intervals

    assert measurement.complete_request_accounting
    assert control.offered_requests == checkpoint.offered_requests == 4
    assert control.completed_requests == checkpoint.completed_requests == 4
    assert control.ttft.median == 10 * MS
    assert checkpoint.ttft.median == 20 * MS
    assert checkpoint.waiting_time.median == 19 * MS
    assert checkpoint.inter_token_latency.median == 5 * MS
    assert control.ttft.p99 is None
    assert control.ttft.p99_status == "insufficient-samples"
    assert control.output_token_throughput_per_second > 0.0
    assert checkpoint.output_token_throughput_per_second > 0.0
    assert checkpoint.max_queue_depth >= 1

    slo = ServingSLO(
        maximum_p95_ttft_ns=15 * MS,
        maximum_p95_inter_token_latency_ns=6 * MS,
        minimum_completion_fraction=1.0,
    )
    assert evaluate_serving_slo(control, slo).satisfied
    assert not evaluate_serving_slo(checkpoint, slo).satisfied

    interference = calculate_serving_interference(control, checkpoint)
    ttft = next(metric for metric in interference.metrics if metric.metric == "p95_ttft_ns")
    assert ttft.interference_fraction == pytest.approx(1.0)
    assert interference.maximum_positive_interference_fraction >= 1.0


def test_p99_is_only_reported_with_at_least_100_samples() -> None:
    small = summarize_latency(tuple(range(99)))
    sufficient = summarize_latency(tuple(range(100)))

    assert small.p99 is None
    assert small.p99_status == "insufficient-samples"
    assert sufficient.p99 == pytest.approx(98.01)
    assert sufficient.p99_status == "reported"


def test_slo_restoration_requires_a_full_stability_window() -> None:
    config = ServingSpikeConfig(
        seed=73,
        control_phase="control",
        spike_phase="spike",
        phases=(
            ArrivalPhase(
                name="control",
                start_offset_ns=0,
                end_offset_ns=NS_PER_SECOND,
                interarrival_ns=100 * MS,
            ),
            ArrivalPhase(
                name="spike",
                start_offset_ns=NS_PER_SECOND,
                end_offset_ns=4 * NS_PER_SECOND,
                interarrival_ns=100 * MS,
            ),
        ),
        prompt_tokens=WeightedTokenDistribution(values=(16,), weights=(1,)),
        output_tokens=WeightedTokenDistribution(values=(2,), weights=(1,)),
    )
    workload = generate_serving_spike(config)
    observations = _observations(
        workload,
        ttft_for_arrival=lambda arrival: 100 * MS if arrival < 2 * NS_PER_SECOND else 10 * MS,
    )

    restoration = find_serving_slo_restoration(
        workload,
        observations,
        trigger_ns=NS_PER_SECOND,
        measurement_end_ns=4 * NS_PER_SECOND,
        slo=ServingSLO(maximum_p95_ttft_ns=20 * MS),
        stability=SLOStabilityConfig(
            evaluation_window_ns=500 * MS,
            stability_window_ns=NS_PER_SECOND,
        ),
    )

    assert restoration.stable_window_confirmed
    assert restoration.slo_reentered_at_ns == 2 * NS_PER_SECOND
    assert restoration.restored_at_ns == 3 * NS_PER_SECOND
    assert restoration.restoration_latency_ns == 2 * NS_PER_SECOND
    assert [evaluation.satisfied for evaluation in restoration.evaluations] == [
        False,
        False,
        True,
        True,
        True,
        True,
    ]


def test_measurement_rejects_unmatched_or_incomplete_request_accounting() -> None:
    config = experiment_004_serving_spike_config(
        seed=113,
        control_duration_ns=100 * MS,
        spike_duration_ns=100 * MS,
        recovery_duration_ns=100 * MS,
        control_interarrival_ns=100 * MS,
        spike_interarrival_ns=100 * MS,
    )
    workload = generate_serving_spike(config)
    observations = _observations(workload)
    plan = ServingMeasurementPlan(intervals=workload_phase_intervals(workload))

    with pytest.raises(ValueError, match="missing planned request IDs"):
        measure_serving_intervals(workload, observations[:-1], plan)

    unknown = observations[0].model_copy(update={"request_id": "unknown.request"})
    with pytest.raises(ValueError, match="unknown request IDs"):
        measure_serving_intervals(workload, (*observations[1:], unknown), plan)


def test_observation_timeline_fails_closed() -> None:
    with pytest.raises(ValidationError, match="tokens cannot precede service start"):
        ServingObservation(
            request_id="request.0",
            arrival_ns=10,
            service_start_ns=20,
            token_timestamps_ns=(19,),
            completion_ns=21,
            outcome=ObservationOutcome.COMPLETED,
            device="gpu-0",
        )
