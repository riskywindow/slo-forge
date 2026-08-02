from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from sloforge.genesis.capsule import CapsuleValidationReport, Digest
from sloforge.genesis.evolution import (
    CapsuleReference,
    ChallengerSpec,
    ChallengerStatus,
    EvolutionConfig,
    EvolutionController,
    EvolutionError,
    EvolutionPersistenceError,
    EvolutionPhase,
    EvolutionStore,
    EvolutionTrigger,
    ExecutionTarget,
    GateObservation,
    GateStage,
    IsolationContract,
    IsolationMode,
    TransitionCategory,
    TriggerObservation,
    classify_trigger,
    run_local_evolution_fixture,
)


class RecordingValidator:
    def __init__(self, digest: str, *, eligible: bool = True) -> None:
        self.digest = digest
        self.eligible = eligible
        self.paths: list[Path] = []

    def __call__(self, path: Path) -> CapsuleValidationReport:
        self.paths.append(path)
        return CapsuleValidationReport(
            capsule_digest=Digest(value=self.digest),
            integrity_valid=self.eligible,
            contract_compatible=self.eligible,
            evidence_complete=self.eligible,
            promotion_eligible=self.eligible,
            checked_at=datetime(2026, 8, 2, tzinfo=UTC),
            issues=(),
        )


def _capsule(tmp_path: Path, name: str, digest_character: str) -> CapsuleReference:
    root = tmp_path / name
    root.mkdir()
    return CapsuleReference(
        capsule_id=name,
        capsule_digest=digest_character * 64,
        genome_hash=("f" if digest_character != "f" else "e") * 64,
        path=str(root),
    )


def _challenger(
    tmp_path: Path,
    capsule: CapsuleReference,
    *,
    category: TransitionCategory = TransitionCategory.REQUEST_BOUNDARY_SWAP,
    active_stream_compatible: bool = False,
) -> ChallengerSpec:
    isolation = tmp_path / f"isolation-{capsule.capsule_id}"
    isolation.mkdir(exist_ok=True)
    return ChallengerSpec(
        candidate_id=f"candidate-{capsule.capsule_id}",
        capsule=capsule,
        transition_category=category,
        isolation=IsolationContract(
            mode=IsolationMode.SANDBOX,
            writable_artifact_root=str(isolation),
        ),
        active_stream_compatible=active_stream_compatible,
    )


def _config(**updates: object) -> EvolutionConfig:
    return EvolutionConfig.model_validate(
        {
            "execution_target": ExecutionTarget.LOCAL,
            "minimum_shadow_samples": 2,
            "minimum_canary_samples": 3,
            **updates,
        }
    )


def _controller(
    tmp_path: Path,
    *,
    config: EvolutionConfig | None = None,
    validator: RecordingValidator | None = None,
) -> tuple[EvolutionController, CapsuleReference, RecordingValidator, EvolutionStore]:
    champion = _capsule(tmp_path, "champion", "a")
    validator = validator or RecordingValidator("b" * 64)
    store = EvolutionStore(tmp_path / "controller-state.json")
    controller = EvolutionController.initialize(
        store=store,
        capsule_validator=validator,
        config=config or _config(),
        deployment_id="genesis-test",
        champion=champion,
        seed=73129,
        now_ms=0,
    )
    return controller, champion, validator, store


def _trigger(controller: EvolutionController, *, now_ms: int = 10) -> None:
    snapshot = controller.observe_trigger(
        TriggerObservation(
            event_id=f"trigger-{now_ms}",
            observed_at_ms=now_ms,
            workload_js_divergence=0.4,
            detail="bimodal prompt distribution detected",
        )
    )
    assert snapshot.active_trigger is EvolutionTrigger.WORKLOAD_DRIFT


def _gate(
    controller: EvolutionController,
    stage: GateStage,
    *,
    event_id: str,
    observed_at_ms: int,
    passes: bool = True,
) -> GateObservation:
    minimum = (
        controller.config.minimum_shadow_samples
        if stage is GateStage.SHADOW
        else controller.config.minimum_canary_samples
    )
    return GateObservation(
        event_id=event_id,
        stage=stage,
        observed_at_ms=observed_at_ms,
        deterministic_seed=controller.snapshot.seed,
        sample_count=minimum,
        error_rate=0.0 if passes else 0.5,
        p95_ttft_ratio=1.0,
        p99_tpot_ratio=1.0,
        quality_regression=0.0,
        interrupted_streams=0,
    )


def _ready_to_promote(
    controller: EvolutionController, challenger: ChallengerSpec, *, start_ms: int = 10
) -> None:
    _trigger(controller, now_ms=start_ms)
    controller.register_challenger(
        challenger, event_id=f"register-{start_ms}", observed_at_ms=start_ms + 10
    )
    controller.begin_shadow(event_id=f"shadow-{start_ms}", observed_at_ms=start_ms + 20)
    controller.record_gate(
        _gate(
            controller,
            GateStage.SHADOW,
            event_id=f"shadow-gate-{start_ms}",
            observed_at_ms=start_ms + 30,
        )
    )
    controller.begin_canary(event_id=f"canary-{start_ms}", observed_at_ms=start_ms + 40)
    controller.record_gate(
        _gate(
            controller,
            GateStage.CANARY,
            event_id=f"canary-gate-{start_ms}",
            observed_at_ms=start_ms + 50,
        )
    )
    assert controller.snapshot.phase is EvolutionPhase.READY_TO_PROMOTE


def test_trigger_classification_covers_workload_drift_and_fabric_degradation() -> None:
    config = _config()
    drift = TriggerObservation(
        event_id="drift",
        observed_at_ms=1,
        workload_js_divergence=0.21,
        detail="drift",
    )
    degraded = TriggerObservation(
        event_id="degraded",
        observed_at_ms=2,
        workload_js_divergence=0.9,
        fabric_health_ratio=0.5,
        detail="rail degradation",
    )
    assert classify_trigger(drift, config) is EvolutionTrigger.WORKLOAD_DRIFT
    assert classify_trigger(degraded, config) is EvolutionTrigger.FABRIC_DEGRADATION
    assert classify_trigger(degraded, config) is classify_trigger(degraded, config)


def test_capsule_validator_rejects_challenger_before_shadow(tmp_path: Path) -> None:
    validator = RecordingValidator("b" * 64, eligible=False)
    controller, _, _, _ = _controller(tmp_path, validator=validator)
    challenger = _challenger(tmp_path, _capsule(tmp_path, "challenger", "b"))
    _trigger(controller)
    snapshot = controller.register_challenger(challenger, event_id="register", observed_at_ms=20)
    assert snapshot.phase is EvolutionPhase.EVOLVING
    assert snapshot.challengers[0].status is ChallengerStatus.CAPSULE_REJECTED
    with pytest.raises(EvolutionError, match="capsule-validated"):
        controller.begin_shadow(event_id="shadow", observed_at_ms=30)


@pytest.mark.parametrize(
    ("category", "active_compatible"),
    [
        (TransitionCategory.POLICY_ONLY_HOT_SWAP, True),
        (TransitionCategory.REQUEST_BOUNDARY_SWAP, False),
    ],
)
def test_promotion_preserves_active_streams_and_retains_rollback(
    tmp_path: Path,
    category: TransitionCategory,
    active_compatible: bool,
) -> None:
    controller, champion, _, _ = _controller(tmp_path)
    challenger = _challenger(
        tmp_path,
        _capsule(tmp_path, "challenger", "b"),
        category=category,
        active_stream_compatible=active_compatible,
    )
    controller.open_stream(
        "stream-before-promotion",
        event_id="open-old",
        observed_at_ms=1,
        externally_visible_output=True,
    )
    _ready_to_promote(controller, challenger, start_ms=10)
    promoted = controller.promote(event_id="promote", observed_at_ms=70)
    assert promoted.champion == challenger.capsule
    assert promoted.previous_champion == champion
    assert promoted.active_streams[0].capsule_id == champion.capsule_id
    assert champion.capsule_id in promoted.retained_capsule_ids
    assert controller.route_request("new-request") == challenger.capsule
    controller.close_stream("stream-before-promotion", event_id="close-old", observed_at_ms=80)
    assert champion.capsule_id in controller.snapshot.retained_capsule_ids


def test_canary_failure_never_interrupts_challenger_stream(tmp_path: Path) -> None:
    controller, champion, _, _ = _controller(tmp_path, config=_config(canary_fraction=1.0))
    challenger = _challenger(tmp_path, _capsule(tmp_path, "challenger", "b"))
    _trigger(controller)
    controller.register_challenger(challenger, event_id="register", observed_at_ms=20)
    controller.begin_shadow(event_id="shadow", observed_at_ms=30)
    controller.record_gate(
        _gate(controller, GateStage.SHADOW, event_id="shadow-pass", observed_at_ms=40)
    )
    controller.begin_canary(event_id="canary", observed_at_ms=50)
    controller.open_stream("canary-stream", event_id="open-canary", observed_at_ms=51)
    rejected = controller.record_gate(
        _gate(
            controller,
            GateStage.CANARY,
            event_id="canary-fail",
            observed_at_ms=60,
            passes=False,
        )
    )
    assert rejected.champion == champion
    assert rejected.active_streams[0].capsule_id == challenger.capsule.capsule_id
    assert challenger.capsule.capsule_id in rejected.retained_capsule_ids
    assert rejected.challengers[0].status is ChallengerStatus.CANARY_REJECTED


def test_rollback_restores_champion_and_preserves_new_stream(tmp_path: Path) -> None:
    controller, champion, _, _ = _controller(tmp_path)
    challenger = _challenger(tmp_path, _capsule(tmp_path, "challenger", "b"))
    _ready_to_promote(controller, challenger)
    controller.promote(event_id="promote", observed_at_ms=70)
    controller.open_stream("new-stream", event_id="open-new", observed_at_ms=80)
    rolled_back = controller.rollback(
        event_id="rollback", observed_at_ms=90, reason="post-promotion SLO regression"
    )
    assert rolled_back.phase is EvolutionPhase.ROLLED_BACK
    assert rolled_back.champion == champion
    assert rolled_back.active_streams[0].capsule_id == challenger.capsule.capsule_id
    assert challenger.capsule.capsule_id in rolled_back.retained_capsule_ids
    assert rolled_back.challengers[0].status is ChallengerStatus.ROLLED_BACK


def test_restart_transition_requires_active_stream_drain(tmp_path: Path) -> None:
    controller, _, _, _ = _controller(tmp_path)
    challenger = _challenger(
        tmp_path,
        _capsule(tmp_path, "challenger", "b"),
        category=TransitionCategory.WORKER_RESTART,
    )
    controller.open_stream("active", event_id="open", observed_at_ms=1)
    _ready_to_promote(controller, challenger)
    with pytest.raises(EvolutionError, match="drain is required"):
        controller.promote(event_id="promote", observed_at_ms=70)
    assert controller.snapshot.phase is EvolutionPhase.READY_TO_PROMOTE


def test_controller_crash_recovery_and_event_idempotency(tmp_path: Path) -> None:
    controller, _, validator, store = _controller(tmp_path)
    challenger = _challenger(tmp_path, _capsule(tmp_path, "challenger", "b"))
    _trigger(controller)
    controller.register_challenger(challenger, event_id="register", observed_at_ms=20)
    before_crash = controller.begin_shadow(event_id="shadow", observed_at_ms=30)

    restored = EvolutionController.restore(
        store=store, capsule_validator=validator, config=controller.config
    )
    duplicate = restored.begin_shadow(event_id="shadow", observed_at_ms=30)
    assert duplicate == before_crash
    advanced = restored.record_gate(
        _gate(restored, GateStage.SHADOW, event_id="shadow-pass", observed_at_ms=40)
    )
    assert advanced.phase is EvolutionPhase.SHADOW_VALIDATED
    assert EvolutionStore(store.path).load() == advanced


def test_capsule_is_revalidated_immediately_before_promotion(tmp_path: Path) -> None:
    controller, champion, validator, _ = _controller(tmp_path)
    challenger = _challenger(tmp_path, _capsule(tmp_path, "challenger", "b"))
    _ready_to_promote(controller, challenger)
    validator.eligible = False
    rejected = controller.promote(event_id="promote", observed_at_ms=70)
    assert rejected.phase is EvolutionPhase.EVOLVING
    assert rejected.champion == champion
    assert rejected.challengers[0].status is ChallengerStatus.CAPSULE_REJECTED
    assert len(validator.paths) == 2


def test_external_live_canary_requires_both_opt_ins(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(
        execution_target=ExecutionTarget.EXTERNAL,
        live_traffic=True,
        live_promotion_authorized=True,
    )
    controller, _, _, _ = _controller(tmp_path, config=config)
    challenger = _challenger(tmp_path, _capsule(tmp_path, "challenger", "b"))
    _trigger(controller)
    controller.register_challenger(challenger, event_id="register", observed_at_ms=20)
    controller.begin_shadow(event_id="shadow", observed_at_ms=30)
    controller.record_gate(
        _gate(controller, GateStage.SHADOW, event_id="shadow-pass", observed_at_ms=40)
    )
    monkeypatch.delenv("SLOFORGE_GENESIS_ALLOW_LIVE_PROMOTION", raising=False)
    with pytest.raises(EvolutionError, match="SLOFORGE_GENESIS_ALLOW_LIVE_PROMOTION"):
        controller.begin_canary(event_id="canary", observed_at_ms=50)
    monkeypatch.setenv("SLOFORGE_GENESIS_ALLOW_LIVE_PROMOTION", "1")
    assert (
        controller.begin_canary(event_id="canary", observed_at_ms=50).phase
        is EvolutionPhase.CANARYING
    )


def test_persisted_state_tampering_is_detected(tmp_path: Path) -> None:
    _, _, _, store = _controller(tmp_path)
    document = json.loads(store.path.read_text())
    document["payload"]["phase"] = "promoted"
    store.path.write_text(json.dumps(document))
    with pytest.raises(EvolutionPersistenceError, match="digest mismatch"):
        store.load()


def test_deterministic_local_fixture_produces_artifact_backed_promotion(tmp_path: Path) -> None:
    controller, champion, validator, store = _controller(tmp_path)
    challenger = _challenger(tmp_path, _capsule(tmp_path, "challenger", "b"))
    snapshot = run_local_evolution_fixture(controller, challenger)
    assert snapshot.phase is EvolutionPhase.PROMOTED
    assert snapshot.champion == challenger.capsule
    assert snapshot.previous_champion == champion
    assert len(validator.paths) == 2
    assert store.load() == snapshot
    assert [record.sequence for record in snapshot.audit] == list(range(len(snapshot.audit)))
