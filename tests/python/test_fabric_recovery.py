from __future__ import annotations

import json
import os
import stat
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from sloforge.autopsy import (
    BottleneckKind,
    CausalHypothesis,
    DiagnosisRecord,
    EvidenceRef,
)
from sloforge.autopsy.models import CounterfactualEstimate, EvidenceStatement
from sloforge.fabric.ir import (
    RecoveryAction,
    RecoveryActionKind,
    RecoveryPlan,
    load_physical_execution_plan,
)
from sloforge.recovery import (
    ActionAttempt,
    DeterministicRecoveryExecutor,
    ExecutionTarget,
    MetricObservation,
    RecoveryMachineConfig,
    RecoveryObservation,
    RecoveryPolicy,
    RecoveryState,
    SimulatedActionDriver,
    plan_recovery,
)

FIXTURES = Path(__file__).parents[1] / "fixtures" / "fabric"
SHA = "a" * 64


def _diagnosis(*, confidence: float = 0.92) -> DiagnosisRecord:
    evidence = EvidenceRef(
        source="fabric-simulator",
        artifact_uri="artifacts/autopsy/degraded/events.json",
        sha256=SHA,
        record_index=3,
    )
    network = CausalHypothesis(
        hypothesis_id="hypothesis-network",
        kind=BottleneckKind.NETWORK_BANDWIDTH_DEGRADATION,
        target="rail-0",
        supporting_evidence=(
            EvidenceStatement(
                metric="network_bandwidth_drop",
                observed=0.51,
                threshold=0.15,
                relation="greater_than",
                supports_hypothesis=True,
                event_ids=("collective-3",),
                explanation="measured bandwidth dropped on the rank-local rail",
            ),
        ),
        contradicting_evidence=(),
        counterfactual=CounterfactualEstimate(
            scenario_id="restore-rail-bandwidth",
            expected_improvement_ms=120.0,
            lower_improvement_ms=90.0,
            upper_improvement_ms=145.0,
            healthy_gap_remaining_ms=8.0,
            simulation_artifact=evidence,
        ),
        confidence=confidence,
    )
    straggler = CausalHypothesis(
        hypothesis_id="hypothesis-rank",
        kind=BottleneckKind.RANK_STRAGGLER,
        target="rank-6",
        supporting_evidence=(
            EvidenceStatement(
                metric="rank_skew",
                observed=2.4,
                threshold=1.2,
                relation="greater_than",
                supports_hypothesis=True,
                event_ids=("collective-3",),
                explanation="one collective participant completed after its peers",
            ),
        ),
        contradicting_evidence=(),
        confidence=0.73,
    )
    return DiagnosisRecord(
        diagnosis_id="diagnosis-degraded",
        degraded_run_id="degraded",
        baseline_run_id="healthy",
        comparison_id="comparison-1",
        top_hypothesis=BottleneckKind.NETWORK_BANDWIDTH_DEGRADATION,
        top_three=(
            BottleneckKind.NETWORK_BANDWIDTH_DEGRADATION,
            BottleneckKind.RANK_STRAGGLER,
        ),
        hypotheses=(network, straggler),
        first_divergence_event_id="collective-3",
        first_divergence_ns=1_000_000,
        confidence=confidence,
        sufficient_alignment=True,
        evidence=(evidence,),
    )


def _plan() -> RecoveryPlan:
    physical = load_physical_execution_plan(FIXTURES / "physical-execution-plan-v1.json")
    return plan_recovery(
        _diagnosis(),
        physical,
        policy=RecoveryPolicy(
            minimum_shadow_samples=3,
            minimum_canary_samples=4,
        ),
    )


def _metrics(*, error_rate: float = 0.0, tpot_ms: float = 30.0) -> tuple[MetricObservation, ...]:
    return (
        MetricObservation(name="p99_tpot_ms", value=tpot_ms, window_seconds=30.0),
        MetricObservation(name="p95_ttft_ms", value=190.0, window_seconds=30.0),
        MetricObservation(name="error_rate", value=error_rate, window_seconds=30.0),
    )


def _observation(
    sequence: int,
    *,
    simulation: bool = False,
    replacement: bool = False,
    shadow: int = 0,
    canary: int = 0,
    streams: int = 0,
    migrated: bool = False,
    metrics: tuple[MetricObservation, ...] = (),
) -> RecoveryObservation:
    return RecoveryObservation(
        observed_at_ms=sequence * 10,
        idempotency_key=f"observation-{sequence}",
        simulation_validated=simulation,
        replacement_ready=replacement,
        shadow_samples=shadow,
        canary_samples=canary,
        active_started_streams=streams,
        traffic_migration_complete=migrated,
        metrics=metrics,
    )


def test_planner_maps_diagnosis_to_evidence_linked_safe_actions() -> None:
    recovery = _plan()
    kinds = tuple(action.kind for action in recovery.actions)
    assert kinds == (
        RecoveryActionKind.QUARANTINE_RAIL,
        RecoveryActionKind.CHANGE_NIC_AFFINITY,
        RecoveryActionKind.CHANGE_RANK_PLACEMENT,
    )
    assert all(not action.requires_external_mutation for action in recovery.actions)
    assert recovery.external_mutation_authorized is False
    assert recovery.diagnosis.digest.value != SHA
    assert recovery.evidence[0].digest.value == SHA
    assert recovery.traffic_migration.preserve_started_streams is True
    assert recovery.expected_slo_improvement.p95_end_to_end_ms.lower > 0.0


def test_planner_rejects_implicit_external_mutation_and_low_confidence() -> None:
    physical = load_physical_execution_plan(FIXTURES / "physical-execution-plan-v1.json")
    with pytest.raises(ValueError, match="explicit mutation authorization"):
        RecoveryPolicy(execution_target=ExecutionTarget.EXTERNAL)
    with pytest.raises(ValueError, match="confidence"):
        plan_recovery(_diagnosis(confidence=0.30), physical)


def test_executor_rejects_low_confidence_and_requires_runtime_external_opt_in() -> None:
    plan = _plan()
    low_confidence = RecoveryPlan.model_validate(
        {**plan.model_dump(mode="python"), "confidence": 0.40}
    )
    rejected = DeterministicRecoveryExecutor(low_confidence, now_ms=0).tick(
        _observation(1, simulation=True)
    )
    assert rejected.state is RecoveryState.REJECTED

    physical = load_physical_execution_plan(FIXTURES / "physical-execution-plan-v1.json")
    external = plan_recovery(
        _diagnosis(),
        physical,
        policy=RecoveryPolicy(
            execution_target=ExecutionTarget.EXTERNAL,
            external_mutation_authorized=True,
        ),
    )
    operator = DeterministicRecoveryExecutor(external, now_ms=0).tick(
        _observation(1, simulation=True)
    )
    assert operator.state is RecoveryState.OPERATOR_REQUIRED


def test_state_machine_shadows_canaries_preserves_streams_and_recovers_after_restart() -> None:
    plan = _plan()
    driver = SimulatedActionDriver()
    config = RecoveryMachineConfig(promotion_cooldown_ms=0)
    executor = DeterministicRecoveryExecutor(plan, now_ms=0, config=config, driver=driver)

    assert (
        executor.tick(_observation(1, simulation=True)).state
        is RecoveryState.VALIDATED_IN_SIMULATION
    )
    building = executor.tick(_observation(2))
    assert building.state is RecoveryState.BUILDING_REPLACEMENT
    assert set(building.applied_action_ids) == {action.action_id for action in plan.actions}
    assert len(driver.attempted_action_ids) == len(plan.actions)
    assert executor.tick(_observation(3, replacement=True)).state is RecoveryState.SHADOWING
    assert (
        executor.tick(_observation(4, shadow=3, metrics=_metrics())).state
        is RecoveryState.CANARYING
    )
    assert (
        executor.tick(_observation(5, canary=4, metrics=_metrics())).state
        is RecoveryState.PROMOTING
    )
    assert executor.tick(_observation(6, migrated=True)).state is RecoveryState.DRAINING_OLD
    draining = executor.tick(_observation(7, streams=2))
    assert draining.state is RecoveryState.DRAINING_OLD
    assert "preserving started streaming requests" in draining.audit[-1].reason

    payload = executor.dump_state()
    restored = DeterministicRecoveryExecutor.restore(plan, payload, config=config)
    duplicate = restored.tick(_observation(7, streams=2))
    assert duplicate == draining
    completed = restored.tick(_observation(8, streams=0))
    assert completed.state is RecoveryState.COMPLETED
    transitions = [item for item in completed.audit if item.event == "transition"]
    assert [item.state_after for item in transitions][-3:] == [
        RecoveryState.PROMOTING,
        RecoveryState.DRAINING_OLD,
        RecoveryState.COMPLETED,
    ]


def test_canary_violation_rolls_back() -> None:
    executor = DeterministicRecoveryExecutor(
        _plan(),
        now_ms=0,
        config=RecoveryMachineConfig(promotion_cooldown_ms=0),
    )
    executor.tick(_observation(1, simulation=True))
    executor.tick(_observation(2))
    executor.tick(_observation(3, replacement=True))
    executor.tick(_observation(4, shadow=3, metrics=_metrics()))
    rolled_back = executor.tick(_observation(5, canary=4, metrics=_metrics(error_rate=0.2)))
    assert rolled_back.state is RecoveryState.ROLLED_BACK
    assert "error_rate" in rolled_back.audit[-1].reason


def test_action_failure_aborts_before_shadow_traffic() -> None:
    plan = _plan()
    driver = SimulatedActionDriver(fail_action_ids={plan.actions[1].action_id})
    executor = DeterministicRecoveryExecutor(plan, now_ms=0, driver=driver)
    executor.tick(_observation(1, simulation=True))
    building = executor.tick(_observation(2))
    assert building.state is RecoveryState.BUILDING_REPLACEMENT
    aborted = executor.tick(_observation(3, replacement=True))
    assert aborted.state is RecoveryState.ABORTED
    assert all(item.state_after is not RecoveryState.SHADOWING for item in aborted.audit)


def test_drain_timeout_escalates_without_interrupting_started_stream() -> None:
    plan = _plan()
    config = RecoveryMachineConfig(promotion_cooldown_ms=0, drain_timeout_ms=5)
    executor = DeterministicRecoveryExecutor(plan, now_ms=0, config=config)
    executor.tick(_observation(1, simulation=True))
    executor.tick(_observation(2))
    executor.tick(_observation(3, replacement=True))
    executor.tick(_observation(4, shadow=3, metrics=_metrics()))
    executor.tick(_observation(5, canary=4, metrics=_metrics()))
    executor.tick(_observation(6, migrated=True))
    escalated = executor.tick(_observation(7, streams=1))
    assert escalated.state is RecoveryState.OPERATOR_REQUIRED
    assert "deadline" in escalated.audit[-1].reason


def test_concurrent_duplicate_observation_is_applied_once() -> None:
    executor = DeterministicRecoveryExecutor(_plan(), now_ms=0)
    observation = _observation(1, simulation=True)
    with ThreadPoolExecutor(max_workers=16) as pool:
        snapshots = tuple(pool.map(lambda _: executor.tick(observation), range(64)))
    assert all(item.state is RecoveryState.VALIDATED_IN_SIMULATION for item in snapshots)
    assert executor.snapshot.processed_idempotency_keys.count(observation.idempotency_key) == 1
    assert (
        sum(item.idempotency_key == observation.idempotency_key for item in executor.snapshot.audit)
        == 1
    )


def test_driver_cannot_substitute_another_action_identity() -> None:
    plan = _plan()

    class MismatchedDriver:
        allow_external_mutation = False

        def apply(self, action: RecoveryAction, *, now_ms: int) -> ActionAttempt:
            substituted = plan.actions[-1]
            return ActionAttempt(
                action_id=substituted.action_id,
                idempotency_key=substituted.idempotency_key,
                attempted_at_ms=now_ms,
                succeeded=True,
                detail=f"claimed {action.action_id}",
            )

    executor = DeterministicRecoveryExecutor(plan, now_ms=0, driver=MismatchedDriver())
    executor.tick(_observation(1, simulation=True))
    building = executor.tick(_observation(2))
    assert building.action_attempts[0].action_id == plan.actions[0].action_id
    assert not building.action_attempts[0].succeeded
    assert "mismatched" in building.action_attempts[0].detail
    assert executor.tick(_observation(3)).state is RecoveryState.ABORTED


def test_restore_rejects_tampered_action_evidence_and_oversized_snapshot() -> None:
    plan = _plan()
    executor = DeterministicRecoveryExecutor(plan, now_ms=0)
    executor.tick(_observation(1, simulation=True))
    executor.tick(_observation(2))
    payload = json.loads(executor.dump_state())
    payload["action_attempts"] = []
    with pytest.raises(ValueError, match="do not match successful attempt evidence"):
        DeterministicRecoveryExecutor.restore(plan, json.dumps(payload))
    with pytest.raises(ValueError, match="bounded input size"):
        DeterministicRecoveryExecutor.restore(plan, b" " * (8 * 1024 * 1024 + 1))


@pytest.mark.skipif(os.name == "nt", reason="symlink replacement mode is POSIX-specific")
def test_persist_state_is_atomic_private_and_does_not_follow_final_symlink(
    tmp_path: Path,
) -> None:
    executor = DeterministicRecoveryExecutor(_plan(), now_ms=0)
    victim = tmp_path / "victim.json"
    victim.write_text("do-not-overwrite", encoding="utf-8")
    output = tmp_path / "snapshot.json"
    output.symlink_to(victim)
    executor.persist_state(output)
    assert victim.read_text(encoding="utf-8") == "do-not-overwrite"
    assert not output.is_symlink()
    assert stat.S_IMODE(output.stat().st_mode) == 0o600
    restored = DeterministicRecoveryExecutor.restore(_plan(), output.read_bytes())
    assert restored.snapshot == executor.snapshot


def test_restored_executor_never_replays_persisted_action_attempts() -> None:
    plan = _plan()
    executor = DeterministicRecoveryExecutor(plan, now_ms=0)
    executor.tick(_observation(1, simulation=True))
    executor.tick(_observation(2))

    class RecordingDriver:
        allow_external_mutation = False

        def __init__(self) -> None:
            self.calls = 0

        def apply(self, action: RecoveryAction, *, now_ms: int) -> ActionAttempt:
            self.calls += 1
            return ActionAttempt(
                action_id=action.action_id,
                idempotency_key=action.idempotency_key,
                attempted_at_ms=now_ms,
                succeeded=True,
                detail="unexpected replay",
            )

    driver = RecordingDriver()
    restored = DeterministicRecoveryExecutor.restore(plan, executor.dump_state(), driver=driver)
    restored.tick(_observation(3))
    assert driver.calls == 0
