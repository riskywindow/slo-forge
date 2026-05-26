from __future__ import annotations

import hashlib
from collections.abc import Mapping

import pytest

from sloforge.autopsy.counterfactual import (
    CounterfactualScenario,
    RemoveFault,
    ScaleRank,
    SimulationObservation,
    attach_counterfactuals,
    replay_counterfactuals,
)
from sloforge.autopsy.minimize import minimize_run
from sloforge.autopsy.models import (
    AutopsyEvent,
    AutopsyRun,
    BottleneckKind,
    CausalHypothesis,
    CounterValue,
    DiagnosisRecord,
    EventType,
    EvidenceRef,
    EvidenceStatement,
    ResourceRef,
    SourceClock,
)

SHA = hashlib.sha256(b"fixture").hexdigest()


def _evidence() -> EvidenceRef:
    return EvidenceRef(source="fixture", artifact_uri="fixture://events", sha256=SHA)


def _event(
    identifier: str,
    *,
    rank: int | None,
    duration_ns: int,
    counter_name: str,
    counter_value: float,
    dependencies: tuple[str, ...] = (),
) -> AutopsyEvent:
    return AutopsyEvent(
        event_id=identifier,
        event_type=EventType.NETWORK_TRANSFER,
        host="host-0",
        rank=rank,
        request_id="request-0",
        operation="expert-dispatch",
        start_ns=0,
        end_ns=duration_ns,
        source_clock=SourceClock.SYNTHETIC,
        dependency_event_ids=dependencies,
        resource=ResourceRef(
            resource_id="rail-0", resource_type="network_rail", contention_domain="rail-0"
        ),
        counters=(CounterValue(name=counter_name, value=counter_value, unit="count"),),
        evidence=_evidence(),
    )


def _run() -> AutopsyRun:
    events = (
        _event("noise-0", rank=0, duration_ns=10, counter_name="noise", counter_value=1.0),
        _event("noise-1", rank=1, duration_ns=10, counter_name="noise", counter_value=2.0),
        _event(
            "cause",
            rank=6,
            duration_ns=1_000,
            counter_name="network_bandwidth_drop",
            counter_value=1.0,
            dependencies=("noise-0",),
        ),
    )
    return AutopsyRun(
        run_id="degraded",
        source="synthetic_fixture",
        topology_fingerprint=SHA,
        physical_plan_hash=SHA,
        workload_fingerprint=SHA,
        reference_host="host-0",
        events=events,
        artifacts=(_evidence(),),
    )


def _diagnosis() -> DiagnosisRecord:
    supporting = EvidenceStatement(
        metric="network_bandwidth_drop",
        observed=0.5,
        threshold=0.15,
        relation="greater_than",
        supports_hypothesis=True,
        explanation="bandwidth fell",
    )
    top = CausalHypothesis(
        hypothesis_id="h-network",
        kind=BottleneckKind.NETWORK_BANDWIDTH_DEGRADATION,
        target="rail-0",
        supporting_evidence=(supporting,),
        contradicting_evidence=(),
        confidence=0.9,
    )
    alternative = CausalHypothesis(
        hypothesis_id="h-rank",
        kind=BottleneckKind.RANK_STRAGGLER,
        target="rank-6",
        supporting_evidence=(supporting,),
        contradicting_evidence=(),
        confidence=0.7,
    )
    return DiagnosisRecord(
        diagnosis_id="diagnosis-0",
        degraded_run_id="degraded",
        baseline_run_id="healthy",
        comparison_id="comparison-0",
        top_hypothesis=BottleneckKind.NETWORK_BANDWIDTH_DEGRADATION,
        top_three=(
            BottleneckKind.NETWORK_BANDWIDTH_DEGRADATION,
            BottleneckKind.RANK_STRAGGLER,
        ),
        hypotheses=(top, alternative),
        first_divergence_event_id="cause",
        first_divergence_ns=0,
        confidence=0.9,
        sufficient_alignment=True,
        evidence=(_evidence(),),
    )


def _observation(makespan_us: float, width_us: float) -> SimulationObservation:
    return SimulationObservation(
        makespan_us=makespan_us,
        predicted_lower_us=makespan_us - width_us,
        predicted_upper_us=makespan_us + width_us,
        operation_count=3,
        processed_events=8,
        output_sha256=SHA,
    )


def test_replay_selects_supported_repair_and_rejects_alternative() -> None:
    calls: list[Mapping[str, object]] = []

    def runner(request: Mapping[str, object]) -> SimulationObservation:
        calls.append(request)
        modifiers = request.get("counterfactuals", [])
        if modifiers == []:
            return _observation(10_000.0, 100.0)
        assert isinstance(modifiers, list)
        if modifiers[0]["type"] == "remove_fault":  # type: ignore[index]
            return _observation(6_000.0, 100.0)
        return _observation(10_500.0, 100.0)

    scenarios = (
        CounterfactualScenario(
            scenario_id="repair-rail",
            hypothesis_id="h-network",
            hypothesis_kind=BottleneckKind.NETWORK_BANDWIDTH_DEGRADATION,
            rationale="remove the calibrated rail fault",
            modifications=(RemoveFault(fault_id="rail-degraded"),),
        ),
        CounterfactualScenario(
            scenario_id="repair-rank",
            hypothesis_id="h-rank",
            hypothesis_kind=BottleneckKind.RANK_STRAGGLER,
            rationale="restore rank service time",
            modifications=(ScaleRank(rank_id="rank-6", duration_multiplier=0.5),),
        ),
    )
    replay = replay_counterfactuals(
        _diagnosis(),
        simulation_request={"seed": 7, "counterfactuals": []},
        scenarios=scenarios,
        healthy_reference_us=5_900.0,
        runner=runner,
    )
    assert replay.selected_scenario_id == "repair-rail"
    assert replay.rejected_scenario_ids == ("repair-rank",)
    assert replay.evaluations[0].status == "supported"
    assert replay.evaluations[0].expected_improvement_ms == pytest.approx(4.0)
    assert replay.evaluations[0].healthy_reference_residual_ms == pytest.approx(0.1)
    assert replay.evaluations[1].status == "contradicted"
    assert replay.evaluations[1].rejected_reason is not None
    assert len(calls) == 3

    augmented = attach_counterfactuals(_diagnosis(), replay)
    assert augmented.hypotheses[0].counterfactual is not None
    assert augmented.hypotheses[0].counterfactual.scenario_id == "repair-rail"
    assert augmented.hypotheses[1].rejected_reason is not None


def test_replay_records_a_failed_alternative() -> None:
    call_count = 0

    def runner(request: Mapping[str, object]) -> SimulationObservation:
        nonlocal call_count
        call_count += 1
        if call_count > 1:
            raise RuntimeError("invalid replacement path")
        return _observation(10_000.0, 100.0)

    scenario = CounterfactualScenario(
        scenario_id="failed",
        hypothesis_id="h-network",
        hypothesis_kind=BottleneckKind.NETWORK_BANDWIDTH_DEGRADATION,
        rationale="test an invalid path",
        modifications=(RemoveFault(fault_id="unknown"),),
    )
    replay = replay_counterfactuals(
        _diagnosis(),
        simulation_request={"seed": 7},
        scenarios=(scenario,),
        healthy_reference_us=5_900.0,
        runner=runner,
    )
    assert replay.selected_scenario_id is None
    assert replay.evaluations[0].status == "simulation_failed"
    assert replay.evaluations[0].rejected_reason == "invalid replacement path"


def test_minimizer_reduces_events_ranks_and_counters_deterministically() -> None:
    run = _run()

    def preserves_diagnosis(candidate: AutopsyRun) -> bool:
        return any(
            event.rank == 6
            and any(counter.name == "network_bandwidth_drop" for counter in event.counters)
            for event in candidate.events
        )

    first = minimize_run(run, preserves_diagnosis)
    second = minimize_run(run, preserves_diagnosis)
    assert first.bundle_sha256 == second.bundle_sha256
    assert first.minimized_run.model_dump(mode="json") == second.minimized_run.model_dump(
        mode="json"
    )
    assert first.minimized_event_count == 1
    assert first.minimized_rank_count == 1
    assert first.minimized_counter_count == 1
    assert first.minimized_run.events[0].event_id == "cause"
    assert first.minimized_run.events[0].dependency_event_ids == ()
    assert first.removed_ranks == (0, 1)


def test_minimizer_rejects_a_nonreproducing_source() -> None:
    with pytest.raises(ValueError, match="does not hold"):
        minimize_run(_run(), lambda _: False)
