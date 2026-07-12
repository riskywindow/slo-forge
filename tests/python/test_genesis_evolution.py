from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from threading import Barrier

import pytest

from sloforge.genesis.capsule import CapsuleValidationReport, Digest, VerificationLevel
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
    TransitionCompatibility,
    TriggerObservation,
    classify_trigger,
    run_local_evolution_fixture,
)


class RecordingValidator:
    def __init__(
        self,
        digest: str,
        *,
        eligible: bool = True,
        champion_eligible: bool = True,
        verification_level: VerificationLevel = VerificationLevel.PROPERTY,
    ) -> None:
        self.digest = digest
        self.eligible = eligible
        self.champion_eligible = champion_eligible
        self.verification_level = verification_level
        self.paths: list[Path] = []

    def __call__(self, path: Path) -> CapsuleValidationReport:
        self.paths.append(path)
        champion = path.name == "champion"
        eligible = self.champion_eligible if champion else self.eligible
        digest = "a" * 64 if champion else self.digest
        return CapsuleValidationReport(
            capsule_digest=Digest(value=digest),
            candidate_genome_hash=Digest(value="f" * 64),
            promotion_verification_level=(self.verification_level if eligible else None),
            integrity_valid=eligible,
            contract_compatible=eligible,
            evidence_complete=eligible,
            promotion_eligible=eligible,
            checked_at=datetime(2026, 8, 2, tzinfo=UTC),
            issues=(),
            local_evolution_eligible=eligible,
            external_production_eligible=(
                eligible and self.verification_level is VerificationLevel.HARDWARE_OPERATIONAL
            ),
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
    gate_validator: Callable[[GateObservation, ChallengerSpec, CapsuleReference], bool]
    | None = None,
) -> tuple[EvolutionController, CapsuleReference, RecordingValidator, EvolutionStore]:
    champion = _capsule(tmp_path, "champion", "a")
    resolved_config = config or _config()
    validator = validator or RecordingValidator(
        "b" * 64,
        verification_level=(
            VerificationLevel.HARDWARE_OPERATIONAL
            if resolved_config.execution_target is ExecutionTarget.EXTERNAL
            else VerificationLevel.PROPERTY
        ),
    )
    store = EvolutionStore(tmp_path / "controller-state.json")
    controller = EvolutionController.initialize(
        store=store,
        capsule_validator=validator,
        gate_evidence_validator=gate_validator
        or (lambda _observation, _challenger, _champion: True),
        transition_compatibility_validator=lambda _champion, _challenger, category: (
            TransitionCompatibility(
                compatible=category
                in {
                    TransitionCategory.POLICY_ONLY_HOT_SWAP,
                    TransitionCategory.REQUEST_BOUNDARY_SWAP,
                    TransitionCategory.NEW_REPLICA,
                },
                behavior="pin_existing_streams",
                reason="trusted test fixture permits only request-pinning categories",
                champion_recovery_digest="a" * 64,
                challenger_recovery_digest="b" * 64,
            )
        ),
        config=resolved_config,
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
    selected = next(
        item
        for item in controller.snapshot.challengers
        if item.spec.candidate_id == controller.snapshot.selected_candidate_id
    )
    evidence_digest = hashlib.sha256(
        (
            f"{controller.snapshot.seed}:{selected.spec.candidate_id}:"
            f"{selected.spec.capsule.capsule_digest}:{stage.value}:{event_id}"
        ).encode()
    ).hexdigest()
    return GateObservation(
        event_id=event_id,
        candidate_id=selected.spec.candidate_id,
        capsule_digest=selected.spec.capsule.capsule_digest,
        evidence_digest=evidence_digest,
        stage=stage,
        verification_level=(
            VerificationLevel.HARDWARE_OPERATIONAL
            if controller.config.execution_target is ExecutionTarget.EXTERNAL
            else VerificationLevel.PROPERTY
        ),
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


def test_gate_observation_is_bound_to_candidate_capsule_seed_and_evidence(tmp_path: Path) -> None:
    controller, _, _, _ = _controller(tmp_path)
    challenger = _challenger(tmp_path, _capsule(tmp_path, "challenger", "b"))
    _trigger(controller)
    controller.register_challenger(challenger, event_id="register", observed_at_ms=20)
    controller.begin_shadow(event_id="shadow", observed_at_ms=30)
    observation = _gate(
        controller,
        GateStage.SHADOW,
        event_id="shadow-pass",
        observed_at_ms=40,
    )
    for update, message in (
        ({"candidate_id": "candidate-other"}, "another candidate"),
        ({"capsule_digest": "c" * 64}, "another capsule"),
        ({"deterministic_seed": 7}, "controller seed"),
    ):
        with pytest.raises(EvolutionError, match=message):
            controller.record_gate(observation.model_copy(update=update))
    assert controller.record_gate(observation).phase is EvolutionPhase.SHADOW_VALIDATED


def test_external_evolution_requires_level_five_and_trusted_gate_evidence(
    tmp_path: Path,
) -> None:
    config = _config(
        execution_target=ExecutionTarget.EXTERNAL,
        live_traffic=True,
        live_promotion_authorized=True,
    )
    champion = _capsule(tmp_path, "champion", "a")
    validator = RecordingValidator(
        "b" * 64, verification_level=VerificationLevel.HARDWARE_OPERATIONAL
    )
    with pytest.raises(EvolutionError, match="gate-evidence validator"):
        EvolutionController.initialize(
            store=EvolutionStore(tmp_path / "missing-gate-validator.json"),
            capsule_validator=validator,
            config=config,
            deployment_id="external-test",
            champion=champion,
            seed=73129,
            now_ms=0,
        )

    controller = EvolutionController.initialize(
        store=EvolutionStore(tmp_path / "external-controller.json"),
        capsule_validator=validator,
        gate_evidence_validator=lambda _observation, _challenger, _champion: False,
        config=config,
        deployment_id="external-test",
        champion=champion,
        seed=73129,
        now_ms=0,
    )
    challenger = _challenger(tmp_path, _capsule(tmp_path, "challenger", "b"))
    _trigger(controller)
    validator.verification_level = VerificationLevel.PROPERTY
    rejected = controller.register_challenger(
        challenger, event_id="register-low-level", observed_at_ms=20
    )
    assert rejected.challengers[-1].status is ChallengerStatus.CAPSULE_REJECTED

    validator.verification_level = VerificationLevel.HARDWARE_OPERATIONAL
    accepted = _challenger(tmp_path, _capsule(tmp_path, "challenger-level-five", "b"))
    controller.register_challenger(accepted, event_id="register-level-five", observed_at_ms=30)
    controller.begin_shadow(event_id="shadow-level-five", observed_at_ms=40)
    observation = _gate(
        controller,
        GateStage.SHADOW,
        event_id="untrusted-shadow-evidence",
        observed_at_ms=50,
    )
    with pytest.raises(EvolutionError, match="trusted validation"):
        controller.record_gate(observation)


def test_local_evolution_rejects_untrusted_and_changed_gate_evidence(tmp_path: Path) -> None:
    accepted_digests: set[str] = set()
    controller, _, _, _ = _controller(
        tmp_path,
        gate_validator=lambda observation, _challenger, _champion: (
            observation.evidence_digest in accepted_digests
        ),
    )
    challenger = _challenger(tmp_path, _capsule(tmp_path, "challenger", "b"))
    _trigger(controller)
    controller.register_challenger(challenger, event_id="register", observed_at_ms=20)
    controller.begin_shadow(event_id="shadow", observed_at_ms=30)
    shadow = _gate(controller, GateStage.SHADOW, event_id="shadow-evidence", observed_at_ms=40)
    with pytest.raises(EvolutionError, match="local gate evidence"):
        controller.record_gate(shadow)
    accepted_digests.add(shadow.evidence_digest)
    controller.record_gate(shadow)
    controller.begin_canary(event_id="canary", observed_at_ms=50)
    canary = _gate(controller, GateStage.CANARY, event_id="canary-evidence", observed_at_ms=60)
    accepted_digests.add(canary.evidence_digest)
    controller.record_gate(canary)
    accepted_digests.remove(shadow.evidence_digest)
    with pytest.raises(EvolutionError, match="shadow evidence failed trusted revalidation"):
        controller.promote(event_id="promote", observed_at_ms=70)


def test_promotion_revalidates_gates_against_current_champion(tmp_path: Path) -> None:
    champion_digests: list[str] = []

    def validate_gate(
        _observation: GateObservation,
        _challenger: ChallengerSpec,
        champion: CapsuleReference,
    ) -> bool:
        champion_digests.append(champion.capsule_digest)
        return True

    controller, champion, _, _ = _controller(tmp_path, gate_validator=validate_gate)
    challenger = _challenger(tmp_path, _capsule(tmp_path, "challenger", "b"))
    _ready_to_promote(controller, challenger)
    controller.promote(event_id="promote", observed_at_ms=70)

    assert champion_digests == [champion.capsule_digest] * 4


def test_controller_refuses_an_unvalidated_bootstrap_champion(tmp_path: Path) -> None:
    champion = _capsule(tmp_path, "champion", "a")
    validator = RecordingValidator("b" * 64, champion_eligible=False)
    store = EvolutionStore(tmp_path / "controller-state.json")

    with pytest.raises(EvolutionError, match="initial champion"):
        EvolutionController.initialize(
            store=store,
            capsule_validator=validator,
            config=_config(),
            deployment_id="genesis-test",
            champion=champion,
            seed=73129,
            now_ms=0,
        )

    assert not store.exists


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


def test_stream_route_and_lease_are_atomic_across_promotion(tmp_path: Path) -> None:
    controller, champion, _, _ = _controller(tmp_path)
    challenger = _challenger(
        tmp_path,
        _capsule(tmp_path, "challenger", "b"),
        category=TransitionCategory.REQUEST_BOUNDARY_SWAP,
    )
    snapshot, assigned = controller.open_stream_and_route(
        "atomic-stream",
        event_id="open-atomic",
        observed_at_ms=1,
        externally_visible_output=True,
    )
    assert assigned == champion
    assert snapshot.active_streams[0].capsule_id == champion.capsule_id
    _ready_to_promote(controller, challenger, start_ms=10)
    promoted = controller.promote(event_id="promote", observed_at_ms=70)
    assert controller.route_request("atomic-stream") == champion
    assert champion in promoted.retained_capsules


def test_restore_revalidates_every_active_stream_capsule(tmp_path: Path) -> None:
    controller, _champion, validator, store = _controller(
        tmp_path, config=_config(canary_fraction=1.0)
    )
    challenger = _challenger(tmp_path, _capsule(tmp_path, "challenger", "b"))
    _trigger(controller)
    controller.register_challenger(challenger, event_id="register", observed_at_ms=20)
    controller.begin_shadow(event_id="shadow", observed_at_ms=30)
    controller.record_gate(
        _gate(controller, GateStage.SHADOW, event_id="shadow-pass", observed_at_ms=40)
    )
    controller.begin_canary(event_id="canary", observed_at_ms=50)
    _snapshot, assigned = controller.open_stream_and_route(
        "active", event_id="open-active", observed_at_ms=51
    )
    assert assigned == challenger.capsule
    controller.record_gate(
        _gate(
            controller,
            GateStage.CANARY,
            event_id="canary-fail",
            observed_at_ms=60,
            passes=False,
        )
    )
    validator.eligible = False
    with pytest.raises(EvolutionError, match="active-stream capsule"):
        EvolutionController.restore(
            store=store,
            capsule_validator=validator,
            gate_evidence_validator=lambda _observation, _challenger, _champion: True,
            config=controller.config,
        )


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


def test_rollback_revalidates_the_previous_champion(tmp_path: Path) -> None:
    controller, _, validator, _ = _controller(tmp_path)
    challenger = _challenger(tmp_path, _capsule(tmp_path, "challenger", "b"))
    _ready_to_promote(controller, challenger)
    controller.promote(event_id="promote", observed_at_ms=70)
    validator.champion_eligible = False

    with pytest.raises(EvolutionError, match="rollback champion"):
        controller.rollback(
            event_id="rollback", observed_at_ms=80, reason="post-promotion regression"
        )

    assert controller.snapshot.phase is EvolutionPhase.PROMOTED


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


def test_proposal_boolean_cannot_authorize_state_migration_with_active_stream(
    tmp_path: Path,
) -> None:
    controller, _, _, _ = _controller(tmp_path)
    challenger = _challenger(
        tmp_path,
        _capsule(tmp_path, "challenger", "b"),
        category=TransitionCategory.STATE_COMPATIBLE_MIGRATION,
        active_stream_compatible=True,
    )
    controller.open_stream("active", event_id="open", observed_at_ms=1)
    _ready_to_promote(controller, challenger)

    with pytest.raises(EvolutionError, match="drain is required"):
        controller.promote(event_id="promote", observed_at_ms=70)


def test_challenger_is_revalidated_at_shadow_boundary(tmp_path: Path) -> None:
    controller, _, validator, _ = _controller(tmp_path)
    challenger = _challenger(tmp_path, _capsule(tmp_path, "challenger", "b"))
    _trigger(controller)
    controller.register_challenger(challenger, event_id="register", observed_at_ms=20)
    validator.eligible = False

    with pytest.raises(EvolutionError, match="before shadowing"):
        controller.begin_shadow(event_id="shadow", observed_at_ms=30)


def test_challenger_is_revalidated_at_canary_boundary(tmp_path: Path) -> None:
    controller, _, validator, _ = _controller(tmp_path)
    challenger = _challenger(tmp_path, _capsule(tmp_path, "challenger", "b"))
    _trigger(controller)
    controller.register_challenger(challenger, event_id="register", observed_at_ms=20)
    controller.begin_shadow(event_id="shadow", observed_at_ms=30)
    controller.record_gate(
        _gate(controller, GateStage.SHADOW, event_id="shadow-pass", observed_at_ms=40)
    )
    validator.eligible = False

    with pytest.raises(EvolutionError, match="before canarying"):
        controller.begin_canary(event_id="canary", observed_at_ms=50)


def test_controller_crash_recovery_and_event_idempotency(tmp_path: Path) -> None:
    controller, _, validator, store = _controller(tmp_path)
    challenger = _challenger(tmp_path, _capsule(tmp_path, "challenger", "b"))
    _trigger(controller)
    controller.register_challenger(challenger, event_id="register", observed_at_ms=20)
    before_crash = controller.begin_shadow(event_id="shadow", observed_at_ms=30)

    restored = EvolutionController.restore(
        store=store,
        capsule_validator=validator,
        gate_evidence_validator=lambda _observation, _challenger, _champion: True,
        config=controller.config,
    )
    duplicate = restored.begin_shadow(event_id="shadow", observed_at_ms=30)
    assert duplicate == before_crash
    advanced = restored.record_gate(
        _gate(restored, GateStage.SHADOW, event_id="shadow-pass", observed_at_ms=40)
    )
    assert advanced.phase is EvolutionPhase.SHADOW_VALIDATED
    assert EvolutionStore(store.path).load() == advanced


def test_restore_fails_closed_when_champion_cannot_be_revalidated(tmp_path: Path) -> None:
    controller, _, validator, store = _controller(tmp_path)
    validator.champion_eligible = False

    with pytest.raises(EvolutionError, match="restored champion"):
        EvolutionController.restore(
            store=store,
            capsule_validator=validator,
            config=controller.config,
        )


def test_idempotency_key_is_bound_to_the_exact_event_payload(tmp_path: Path) -> None:
    controller, _, _, _ = _controller(tmp_path)
    original = TriggerObservation(
        event_id="observation",
        observed_at_ms=1,
        detail="below threshold",
    )
    snapshot = controller.observe_trigger(original)
    assert controller.observe_trigger(original) == snapshot

    with pytest.raises(EvolutionError, match="different event payload"):
        controller.observe_trigger(
            original.model_copy(update={"detail": "same key, altered payload"})
        )


def test_concurrent_stream_opens_are_serialized_without_lost_updates(tmp_path: Path) -> None:
    controller, _, _, store = _controller(tmp_path)
    worker_count = 8
    barrier = Barrier(worker_count + 1)

    def open_stream(index: int) -> None:
        barrier.wait()
        controller.open_stream(
            f"stream-{index}",
            event_id=f"open-{index}",
            observed_at_ms=1,
        )

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = [executor.submit(open_stream, index) for index in range(worker_count)]
        barrier.wait()
        for future in futures:
            future.result(timeout=5.0)

    assert len(controller.snapshot.active_streams) == worker_count
    assert controller.snapshot.sequence == worker_count
    assert store.load() == controller.snapshot


def test_two_restored_controllers_cannot_overwrite_the_same_sequence(tmp_path: Path) -> None:
    controller, _, validator, store = _controller(tmp_path)
    first = EvolutionController.restore(
        store=store, capsule_validator=validator, config=controller.config
    )
    stale = EvolutionController.restore(
        store=store, capsule_validator=validator, config=controller.config
    )
    first.observe_trigger(
        TriggerObservation(event_id="first", observed_at_ms=1, detail="first writer")
    )

    with pytest.raises(EvolutionPersistenceError, match="sequence mismatch"):
        stale.observe_trigger(
            TriggerObservation(event_id="stale", observed_at_ms=1, detail="stale writer")
        )

    assert store.load() == first.snapshot
    assert stale.snapshot.sequence == 0


def test_idempotency_registry_fails_closed_instead_of_forgetting_old_keys(
    tmp_path: Path,
) -> None:
    controller, _, _, _ = _controller(
        tmp_path,
        config=_config(maximum_processed_events=32),
    )
    for index in range(31):
        controller.open_stream(f"stream-{index}", event_id=f"open-{index}", observed_at_ms=1)
    snapshot = controller.snapshot

    assert controller.open_stream("stream-0", event_id="open-0", observed_at_ms=1) == snapshot
    with pytest.raises(EvolutionError, match="idempotency registry is full"):
        controller.open_stream("overflow", event_id="overflow", observed_at_ms=1)


def test_capsule_is_revalidated_immediately_before_promotion(tmp_path: Path) -> None:
    controller, champion, validator, _ = _controller(tmp_path)
    challenger = _challenger(tmp_path, _capsule(tmp_path, "challenger", "b"))
    _ready_to_promote(controller, challenger)
    validator.eligible = False
    rejected = controller.promote(event_id="promote", observed_at_ms=70)
    assert rejected.phase is EvolutionPhase.EVOLVING
    assert rejected.champion == champion
    assert rejected.challengers[0].status is ChallengerStatus.CAPSULE_REJECTED
    assert len(validator.paths) == 5


def test_external_live_canary_requires_config_and_both_environment_opt_ins(
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
    monkeypatch.delenv("SLOFORGE_GENESIS_ALLOW_EXTERNAL_DEPLOYMENT", raising=False)
    with pytest.raises(EvolutionError, match="SLOFORGE_GENESIS_ALLOW_LIVE_PROMOTION"):
        controller.begin_canary(event_id="canary", observed_at_ms=50)
    monkeypatch.setenv("SLOFORGE_GENESIS_ALLOW_LIVE_PROMOTION", "1")
    with pytest.raises(EvolutionError, match="SLOFORGE_GENESIS_ALLOW_EXTERNAL_DEPLOYMENT"):
        controller.begin_canary(event_id="canary", observed_at_ms=50)
    monkeypatch.setenv("SLOFORGE_GENESIS_ALLOW_EXTERNAL_DEPLOYMENT", "1")
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


def test_evolution_store_rejects_symlink_and_predictable_temp_attacks(tmp_path: Path) -> None:
    controller, _, _, store = _controller(tmp_path)
    victim = tmp_path / "victim.json"
    victim.write_text("preserve", encoding="utf-8")
    predictable = store.path.with_name(f".{store.path.name}.tmp")
    predictable.symlink_to(victim)

    controller.observe_trigger(
        TriggerObservation(
            event_id="safe-random-temp",
            observed_at_ms=1,
            detail="no material change",
        )
    )
    assert victim.read_text(encoding="utf-8") == "preserve"

    snapshot = controller.snapshot
    store.path.unlink()
    store.path.symlink_to(victim)
    with pytest.raises(EvolutionPersistenceError, match="must not be a symlink"):
        store.save(snapshot, expected_sequence=snapshot.sequence)
    with pytest.raises(EvolutionPersistenceError, match="must not be a symlink"):
        store.load()
    assert victim.read_text(encoding="utf-8") == "preserve"


def test_local_fixture_refuses_synthetic_gate_observations(tmp_path: Path) -> None:
    controller, _champion, _validator, _store = _controller(tmp_path)
    challenger = _challenger(tmp_path, _capsule(tmp_path, "challenger", "b"))
    with pytest.raises(ValueError, match="artifact-backed gate evidence"):
        run_local_evolution_fixture(controller, challenger)
