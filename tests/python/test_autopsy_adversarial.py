from __future__ import annotations

import hashlib

import pytest

from sloforge.autopsy import (
    AlignmentEstimate,
    AlignmentQuality,
    AutopsyEvent,
    AutopsyRun,
    BottleneckKind,
    ClockSample,
    CounterValue,
    EventType,
    EvidenceRef,
    FaultInterval,
    ResourceRef,
    SourceClock,
    compare_runs,
    diagnose,
    estimate_alignment,
    minimize_run,
)
from sloforge.autopsy.counterfactual import (
    CounterfactualScenario,
    RemoveFault,
    ScenarioEvaluation,
    SimulationObservation,
)
from sloforge.autopsy.models import CounterDelta, DifferentialComparison, StageDelta
from sloforge.util import canonical_json, sha256_bytes

DIGEST = hashlib.sha256(b"autopsy-adversarial").hexdigest()


def _evidence() -> EvidenceRef:
    return EvidenceRef(source="fixture", artifact_uri="fixture://adversarial", sha256=DIGEST)


def _alignment(host: str, *, uncertainty_ns: int = 10) -> AlignmentEstimate:
    return AlignmentEstimate(
        host=host,
        reference_host="host-a",
        offset_ns=0.0,
        drift_ppm=0.0,
        reference_local_ns=0,
        uncertainty_ns=uncertainty_ns,
        sample_count=8,
        confidence=0.9,
        quality=AlignmentQuality.GOOD,
        residual_p95_ns=1,
    )


def _event(
    identifier: str,
    *,
    duration_ns: int,
    start_ns: int = 1_000,
    host: str = "host-a",
    rank: int = 0,
    event_type: EventType = EventType.NETWORK_TRANSFER,
    request_id: str = "request-0",
    counter: CounterValue | None = None,
    uncertainty_ns: int = 10,
) -> AutopsyEvent:
    return AutopsyEvent(
        event_id=identifier,
        event_type=event_type,
        host=host,
        rank=rank,
        request_id=request_id,
        operation="operation",
        start_ns=start_ns,
        end_ns=start_ns + duration_ns,
        source_clock=SourceClock.SYNTHETIC,
        normalized_start_ns=start_ns,
        normalized_end_ns=start_ns + duration_ns,
        alignment_confidence=0.9,
        alignment_uncertainty_ns=uncertainty_ns,
        resource=ResourceRef(
            resource_id="rail-0",
            resource_type="network_rail",
            contention_domain="rail-0",
        ),
        counters=() if counter is None else (counter,),
        evidence=_evidence(),
    )


def _run(
    run_id: str,
    events: tuple[AutopsyEvent, ...],
    *,
    topology: str = DIGEST,
    plan: str = DIGEST,
    alignments: tuple[AlignmentEstimate, ...] = (),
    faults: tuple[FaultInterval, ...] = (),
) -> AutopsyRun:
    return AutopsyRun(
        run_id=run_id,
        source="synthetic_fixture",
        topology_fingerprint=topology,
        physical_plan_hash=plan,
        workload_fingerprint=DIGEST,
        reference_host="host-a",
        events=events,
        alignments=alignments,
        fault_intervals=faults,
        artifacts=(_evidence(),),
    )


def test_alignment_fit_resists_one_delayed_exchange() -> None:
    samples: list[ClockSample] = []
    for index in range(8):
        local = 1_000_000_000 + index * 1_000_000_000
        drift = int((local - 4_000_000_000) * 5 / 1_000_000)
        samples.append(
            ClockSample(
                host="host-a",
                local_monotonic_ns=local,
                reference_monotonic_ns=local + 2_000_000 + drift + 50_000,
                round_trip_ns=100_000,
                captured_wall_ns=20_000_000_000 + local,
            )
        )
    samples.append(
        ClockSample(
            host="host-a",
            local_monotonic_ns=9_000_000_000,
            reference_monotonic_ns=19_000_000_000,
            round_trip_ns=10_000_000_000,
            captured_wall_ns=29_000_000_000,
        )
    )
    estimate = estimate_alignment(samples, reference_host="host-reference")
    assert estimate.drift_ppm == pytest.approx(5.0, abs=0.01)
    assert estimate.offset_ns == pytest.approx(2_005_000.0, abs=2.0)


@pytest.mark.parametrize("field", ["topology", "plan"])
def test_comparison_rejects_unmatched_physical_context(field: str) -> None:
    healthy = _run("healthy", (_event("healthy", duration_ns=1_000),))
    changed = hashlib.sha256(field.encode()).hexdigest()
    degraded = _run(
        "degraded",
        (_event("degraded", duration_ns=2_000),),
        topology=changed if field == "topology" else DIGEST,
        plan=changed if field == "plan" else DIGEST,
    )
    with pytest.raises(ValueError, match=r"same topology|same physical plan"):
        compare_runs(healthy, degraded)


def test_first_divergence_fails_closed_when_alignment_intervals_overlap() -> None:
    alignments = (
        _alignment("host-a", uncertainty_ns=100),
        _alignment("host-b", uncertainty_ns=100),
    )
    healthy = _run(
        "healthy",
        (
            _event(
                "healthy-a",
                duration_ns=100_000,
                start_ns=1_000,
                host="host-a",
                rank=0,
                uncertainty_ns=100,
            ),
            _event(
                "healthy-b",
                duration_ns=100_000,
                start_ns=1_100,
                host="host-b",
                rank=1,
                uncertainty_ns=100,
            ),
        ),
        alignments=alignments,
    )
    degraded = _run(
        "degraded",
        (
            _event(
                "degraded-a",
                duration_ns=300_000,
                start_ns=1_000,
                host="host-a",
                rank=0,
                uncertainty_ns=100,
            ),
            _event(
                "degraded-b",
                duration_ns=300_000,
                start_ns=1_100,
                host="host-b",
                rank=1,
                uncertainty_ns=100,
            ),
        ),
        alignments=alignments,
    )
    comparison = compare_runs(healthy, degraded)
    assert comparison.first_divergence_event_id is None
    assert any("ambiguous" in warning for warning in comparison.warnings)


def test_same_host_divergences_keep_monotonic_order_despite_alignment_error() -> None:
    healthy = _run(
        "healthy",
        (
            _event(
                "healthy-a",
                duration_ns=100_000,
                start_ns=1_000,
                rank=0,
                uncertainty_ns=100,
            ),
            _event(
                "healthy-b",
                duration_ns=100_000,
                start_ns=1_050,
                rank=1,
                uncertainty_ns=100,
            ),
        ),
    )
    degraded = _run(
        "degraded",
        (
            _event(
                "degraded-a",
                duration_ns=300_000,
                start_ns=1_000,
                rank=0,
                uncertainty_ns=100,
            ),
            _event(
                "degraded-b",
                duration_ns=300_000,
                start_ns=1_050,
                rank=1,
                uncertainty_ns=100,
            ),
        ),
    )
    comparison = compare_runs(healthy, degraded)
    assert comparison.first_divergence_event_id == "degraded-a"
    assert not any("ambiguous" in warning for warning in comparison.warnings)


def test_run_rejects_alignment_with_a_different_reference_clock() -> None:
    wrong = _alignment("host-a").model_copy(update={"reference_host": "host-other"})
    with pytest.raises(ValueError, match="different reference host"):
        _run("run", (_event("event", duration_ns=1_000),), alignments=(wrong,))


def test_multi_host_diagnosis_requires_alignment_for_each_run() -> None:
    healthy = _run(
        "healthy",
        (
            _event("healthy-a", duration_ns=100_000, host="host-a", rank=0),
            _event("healthy-b", duration_ns=100_000, host="host-b", rank=1),
        ),
    )
    degraded = _run(
        "degraded",
        (
            _event("degraded-a", duration_ns=300_000, host="host-a", rank=0),
            _event("degraded-b", duration_ns=300_000, host="host-b", rank=1),
        ),
        alignments=(_alignment("host-a"), _alignment("host-b")),
    )
    comparison = compare_runs(healthy, degraded)
    result = diagnose(degraded, comparison=comparison, baseline=healthy)
    assert not result.sufficient_alignment
    assert comparison.first_divergence_event_id is None
    assert any("missing" in warning for warning in comparison.warnings)


def test_fault_ground_truth_labels_do_not_change_diagnosis() -> None:
    healthy = _run(
        "healthy",
        (
            _event(
                "healthy",
                duration_ns=1_000,
                counter=CounterValue(name="network_bandwidth_gbps", value=100.0, unit="Gbps"),
            ),
        ),
    )
    event = _event(
        "degraded",
        duration_ns=3_000,
        counter=CounterValue(name="network_bandwidth_gbps", value=50.0, unit="Gbps"),
    )
    network_label = FaultInterval(
        fault_id="fault",
        fault_type="network_bandwidth_degradation",
        target="rail-0",
        start_ns=0,
        end_ns=10_000,
    )
    false_label = network_label.model_copy(update={"fault_type": "worker_crash"})
    degraded = _run("degraded", (event,), faults=(network_label,))
    mislabeled = _run("degraded", (event,), faults=(false_label,))
    comparison = compare_runs(healthy, degraded)
    first = diagnose(degraded, comparison=comparison, baseline=healthy)
    second = diagnose(mislabeled, comparison=comparison, baseline=healthy)
    assert first.top_hypothesis is BottleneckKind.NETWORK_BANDWIDTH_DEGRADATION
    assert first.top_hypothesis is second.top_hypothesis
    assert first.confidence == second.confidence
    assert first.hypotheses == second.hypotheses


_STAGE_RULE_CASES: tuple[tuple[BottleneckKind, EventType], ...] = (
    (BottleneckKind.GATEWAY_QUEUEING, EventType.GATEWAY_QUEUE),
    (BottleneckKind.BACKEND_QUEUEING, EventType.BACKEND_QUEUE),
    (BottleneckKind.COLD_START_REGRESSION, EventType.STARTUP),
    (BottleneckKind.CPU_LAUNCH_BOTTLENECK, EventType.CPU_LAUNCH),
    (BottleneckKind.GPU_COMPUTE_REGRESSION, EventType.GPU_COMPUTE),
    (BottleneckKind.GPU_MEMORY_BANDWIDTH_REGRESSION, EventType.GPU_MEMORY),
    (BottleneckKind.PCIE_BOTTLENECK, EventType.PCIE_TRANSFER),
    (BottleneckKind.NVLINK_DEGRADATION, EventType.NVLINK_TRANSFER),
    (BottleneckKind.NETWORK_LATENCY_DEGRADATION, EventType.NETWORK_TRANSFER),
    (BottleneckKind.COLLECTIVE_IMBALANCE, EventType.COLLECTIVE_WAIT),
    (BottleneckKind.COLLECTIVE_ALGORITHM_REGRESSION, EventType.COLLECTIVE),
    (BottleneckKind.PREFILL_POOL_SATURATION, EventType.PREFILL),
    (BottleneckKind.DECODE_POOL_SATURATION, EventType.DECODE),
    (BottleneckKind.KV_TRANSFER_BOTTLENECK, EventType.KV_TRANSFER),
)

_COUNTER_RULE_CASES: tuple[tuple[BottleneckKind, str, float], ...] = (
    (BottleneckKind.ARRIVAL_OVERLOAD, "arrival_capacity_ratio", 0.5),
    (BottleneckKind.INSUFFICIENT_WARM_CAPACITY, "warm_fraction", -0.5),
    (BottleneckKind.MODEL_LOADING_REGRESSION, "model_load_time_ms", 0.5),
    (BottleneckKind.EXCESSIVE_KERNEL_LAUNCHES, "kernel_count", 0.5),
    (BottleneckKind.GPU_CLOCK_THROTTLING, "gpu_clock_mhz", -0.5),
    (BottleneckKind.NUMA_MISPLACEMENT, "numa_remote_fraction", 0.5),
    (BottleneckKind.NETWORK_BANDWIDTH_DEGRADATION, "network_bandwidth_gbps", -0.5),
    (BottleneckKind.EXPERT_LOAD_IMBALANCE, "expert_token_cv", 0.5),
    (BottleneckKind.UNHEALTHY_WORKER, "worker_unhealthy", 1.0),
    (BottleneckKind.WORKER_CRASH, "worker_crash", 1.0),
    (BottleneckKind.TOPOLOGY_MISMATCH, "topology_mismatch", 1.0),
    (BottleneckKind.INVALID_PHYSICAL_PLAN, "plan_assumption_error", 1.0),
)


@pytest.mark.parametrize(
    ("expected", "event_type"),
    _STAGE_RULE_CASES,
    ids=lambda value: value.value if isinstance(value, (BottleneckKind, EventType)) else str(value),
)
def test_each_stage_diagnosis_rule_can_win(expected: BottleneckKind, event_type: EventType) -> None:
    comparison = DifferentialComparison(
        comparison_id="comparison",
        healthy_run_id="healthy",
        degraded_run_id="degraded",
        matched_event_count=1,
        unmatched_healthy_count=0,
        unmatched_degraded_count=0,
        stage_deltas=(
            StageDelta(
                signature=f"{event_type.value}:operation",
                event_type=event_type,
                operation="operation",
                rank=0,
                healthy_count=1,
                degraded_count=1,
                matched_count=1,
                healthy_median_ms=1.0,
                degraded_median_ms=1.5,
                healthy_p95_ms=1.0,
                degraded_p95_ms=1.5,
                absolute_delta_ms=0.5,
                relative_delta=0.5,
            ),
        ),
        counter_deltas=(),
        first_divergence_event_id=None,
        first_divergence_ns=None,
        maximum_rank_skew=1.0,
    )
    healthy = _run("healthy", ())
    degraded = _run("degraded", ())
    assert diagnose(degraded, comparison=comparison, baseline=healthy).top_hypothesis is expected


@pytest.mark.parametrize(
    ("expected", "counter_name", "relative_delta"),
    _COUNTER_RULE_CASES,
    ids=lambda value: value.value if isinstance(value, BottleneckKind) else str(value),
)
def test_each_counter_diagnosis_rule_can_win(
    expected: BottleneckKind, counter_name: str, relative_delta: float
) -> None:
    comparison = DifferentialComparison(
        comparison_id="comparison",
        healthy_run_id="healthy",
        degraded_run_id="degraded",
        matched_event_count=1,
        unmatched_healthy_count=0,
        unmatched_degraded_count=0,
        stage_deltas=(),
        counter_deltas=(
            CounterDelta(
                name=counter_name,
                unit="ratio",
                healthy_median=1.0,
                degraded_median=1.0 + relative_delta,
                absolute_delta=relative_delta,
                relative_delta=relative_delta,
            ),
        ),
        first_divergence_event_id=None,
        first_divergence_ns=None,
        maximum_rank_skew=1.0,
    )
    healthy = _run("healthy", ())
    degraded = _run("degraded", ())
    assert diagnose(degraded, comparison=comparison, baseline=healthy).top_hypothesis is expected


def test_rank_skew_rule_completes_bottleneck_kind_coverage() -> None:
    covered = {
        *(kind for kind, _ in _STAGE_RULE_CASES),
        *(kind for kind, _, _ in _COUNTER_RULE_CASES),
        BottleneckKind.RANK_STRAGGLER,
    }
    assert covered == set(BottleneckKind)
    comparison = DifferentialComparison(
        comparison_id="comparison",
        healthy_run_id="healthy",
        degraded_run_id="degraded",
        matched_event_count=1,
        unmatched_healthy_count=0,
        unmatched_degraded_count=0,
        stage_deltas=(),
        counter_deltas=(),
        first_divergence_event_id=None,
        first_divergence_ns=None,
        maximum_rank_skew=1.5,
    )
    healthy = _run("healthy", ())
    degraded = _run("degraded", ())
    assert (
        diagnose(degraded, comparison=comparison, baseline=healthy).top_hypothesis
        is BottleneckKind.RANK_STRAGGLER
    )


def test_irrelevant_matched_events_do_not_inflate_network_confidence() -> None:
    bandwidth_healthy = CounterValue(name="network_bandwidth_gbps", value=100.0, unit="Gbps")
    bandwidth_slow = CounterValue(name="network_bandwidth_gbps", value=50.0, unit="Gbps")
    healthy_event = _event("healthy-network", duration_ns=1_000, counter=bandwidth_healthy)
    degraded_event = _event("degraded-network", duration_ns=3_000, counter=bandwidth_slow)

    def confidence(noise_count: int) -> float:
        healthy_noise = tuple(
            _event(
                f"healthy-noise-{index}",
                duration_ns=1_000,
                event_type=EventType.CPU_LAUNCH,
                request_id=f"noise-{index}",
            )
            for index in range(noise_count)
        )
        degraded_noise = tuple(
            _event(
                f"degraded-noise-{index}",
                duration_ns=1_000,
                event_type=EventType.CPU_LAUNCH,
                request_id=f"noise-{index}",
            )
            for index in range(noise_count)
        )
        healthy = _run("healthy", (healthy_event, *healthy_noise))
        degraded = _run("degraded", (degraded_event, *degraded_noise))
        return diagnose(
            degraded,
            comparison=compare_runs(healthy, degraded),
            baseline=healthy,
        ).confidence

    assert confidence(0) == confidence(100)


def test_no_signal_diagnosis_is_explicitly_rejected_and_low_confidence() -> None:
    healthy = _run("healthy", (_event("healthy", duration_ns=100_000),))
    degraded = _run("degraded", (_event("degraded", duration_ns=100_000),))
    result = diagnose(degraded, comparison=compare_runs(healthy, degraded), baseline=healthy)
    assert result.confidence <= 0.20
    assert all(hypothesis.rejected_reason is not None for hypothesis in result.hypotheses)
    assert all(
        next(
            hypothesis for hypothesis in result.hypotheses if hypothesis.kind is kind
        ).supporting_evidence
        == ()
        for kind in result.top_three
    )
    assert any("no causal hypothesis" in warning for warning in result.warnings)
    assert any("not a calibrated probability" in warning for warning in result.warnings)


def test_conflicting_counter_units_fail_closed() -> None:
    healthy = _run(
        "healthy",
        (
            _event(
                "healthy-gbps",
                duration_ns=1_000,
                request_id="a",
                counter=CounterValue(name="network_bandwidth_gbps", value=100.0, unit="Gbps"),
            ),
            _event(
                "healthy-ratio",
                duration_ns=1_000,
                request_id="b",
                counter=CounterValue(name="network_bandwidth_gbps", value=1.0, unit="ratio"),
            ),
        ),
    )
    degraded = _run(
        "degraded",
        (
            _event(
                "degraded-gbps",
                duration_ns=1_000,
                request_id="a",
                counter=CounterValue(name="network_bandwidth_gbps", value=50.0, unit="Gbps"),
            ),
            _event(
                "degraded-ratio",
                duration_ns=1_000,
                request_id="b",
                counter=CounterValue(name="network_bandwidth_gbps", value=0.5, unit="ratio"),
            ),
        ),
    )
    comparison = compare_runs(healthy, degraded)
    with pytest.raises(ValueError, match="conflicting units"):
        diagnose(degraded, comparison=comparison, baseline=healthy)


def test_counterfactual_status_must_match_interval() -> None:
    scenario = CounterfactualScenario(
        scenario_id="scenario",
        hypothesis_id="hypothesis",
        hypothesis_kind=BottleneckKind.NETWORK_BANDWIDTH_DEGRADATION,
        rationale="repair rail",
        modifications=(RemoveFault(fault_id="fault"),),
    )
    observation = SimulationObservation(
        makespan_us=8_000.0,
        predicted_lower_us=7_900.0,
        predicted_upper_us=8_100.0,
        operation_count=1,
        processed_events=1,
        output_sha256=DIGEST,
    )
    with pytest.raises(ValueError, match="status disagrees"):
        ScenarioEvaluation(
            scenario=scenario,
            status="contradicted",
            expected_improvement_ms=2.0,
            lower_improvement_ms=1.8,
            upper_improvement_ms=2.2,
            healthy_reference_residual_ms=0.1,
            confidence=0.8,
            observation=observation,
            rejected_reason="inconsistent",
        )


def test_minimization_hashes_emitted_bundle_and_reduces_fault_scope() -> None:
    required = FaultInterval(
        fault_id="required",
        fault_type="network_degradation",
        target="rail-0",
        start_ns=0,
        end_ns=1_000,
    )
    noise = FaultInterval(
        fault_id="noise",
        fault_type="worker_crash",
        target="rank-7",
        start_ns=0,
        end_ns=1_000,
    )
    run = _run("degraded", (_event("event", duration_ns=2_000),), faults=(required, noise))
    result = minimize_run(
        run,
        lambda candidate: any(fault.fault_id == "required" for fault in candidate.fault_intervals),
    )
    assert result.minimized_fault_count == 1
    assert result.removed_fault_ids == ("noise",)
    assert result.bundle_sha256 == sha256_bytes(
        canonical_json(result.minimized_run.model_dump(mode="json")).encode()
    )
