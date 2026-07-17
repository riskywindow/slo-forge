from __future__ import annotations

import time
from hashlib import sha256
from pathlib import Path

import pytest
from pydantic import ValidationError

from sloforge.helix.effects import (
    Effect,
    EffectClass,
    IllegalEffectError,
    require_effect_legal,
)
from sloforge.helix.faults import (
    ActivationInterval,
    DeterministicFaultInjector,
    FaultCampaignFailed,
    FaultCampaignResult,
    FaultKind,
    FaultObservation,
    FaultPlan,
    FaultPlanRequest,
    FaultResponse,
    FaultRunner,
    FaultSpec,
    FaultStage,
    InjectedFault,
    compile_fault_plan,
    load_fault_plan_request,
)
from sloforge.helix.policy import DeterministicPolicy
from sloforge.helix.promotion import GateEvidence, PolicyRegistry, PromotionState
from sloforge.helix.rollouts import TokenRecord

ROOT = Path(__file__).resolve().parents[2]


def _digest(value: str) -> str:
    return sha256(value.encode()).hexdigest()


def _observation(injection: InjectedFault) -> FaultObservation:
    return FaultObservation(
        actual_response=injection.expected_response,
        detail=f"observed fail-closed response for {injection.ground_truth_label}",
        evidence_sha256=_digest(f"observation:{injection.injection_id}"),
    )


def _illegal_effect_observation(injection: InjectedFault) -> FaultObservation:
    effect = Effect.build(
        EffectClass.IRREVERSIBLE_WRITE,
        "send-email",
        real_external=True,
    )
    try:
        require_effect_legal(
            effect,
            speculative=True,
            external_side_effects_enabled=True,
        )
    except IllegalEffectError as error:
        return FaultObservation(
            actual_response=FaultResponse.REJECTED,
            detail=str(error),
            evidence_sha256=_digest(effect.effect_id),
        )
    raise AssertionError("real effect legality API accepted an irreversible speculative effect")


def _missing_logprob_observation(injection: InjectedFault) -> FaultObservation:
    raw = {
        "token_index": 0,
        "token": "x",
        "policy_epoch_id": "policy.a",
        "sampler_seed": injection.seed,
        "rng_counter": 0,
        "event_index": 1,
    }
    try:
        TokenRecord.model_validate(raw, strict=True)
    except ValidationError as error:
        return FaultObservation(
            actual_response=FaultResponse.QUARANTINED,
            detail="rollout TokenRecord rejected missing behavior log probability",
            evidence_sha256=_digest(str(error.errors(include_url=False))),
        )
    raise AssertionError("real rollout record API accepted missing log-probability provenance")


def _gate(name: str, *, seed: int) -> GateEvidence:
    return GateEvidence.model_validate(
        {
            "gate": name,
            "evidence_id": f"evidence.{name}",
            "artifact_hash": _digest(name),
            "passed": True,
            "sample_count": 10,
            "measured_value": 0.0,
            "threshold": 0.0,
            "comparator": "le",
            "deterministic_seed": seed,
            "detail": f"{name} passed",
        }
    )


def _partial_pointer_observation(injection: InjectedFault, database: Path) -> FaultObservation:
    champion = DeterministicPolicy(
        policy_epoch_id="champion",
        actions=("bad", "good"),
        logits=(1.0, 0.0),
    )
    challenger = DeterministicPolicy(
        policy_epoch_id="challenger",
        actions=("bad", "good"),
        logits=(0.0, 1.0),
    )
    with PolicyRegistry(database) as registry:
        registry.register_policy(
            champion,
            parent_policy_epoch_id=None,
            status="champion",
            created_at_ms=1,
        )
        registry.register_policy(
            challenger,
            parent_policy_epoch_id="champion",
            status="challenger",
            created_at_ms=2,
        )
        registry.create_deployment("prod", "champion")
        gates = tuple(
            _gate(name, seed=injection.seed)
            for name in (
                "lineage",
                "reward_integrity",
                "quality",
                "safety",
                "serving",
                "compatibility",
            )
        )
        registry.create_promotion(
            transaction_id="tx",
            deployment="prod",
            candidate_policy_epoch_id="challenger",
            evidence=gates,
            observed_at_ms=3,
        )
        registry.start_shadow("tx", observed_at_ms=4)
        registry.finish_shadow("tx", _gate("shadow", seed=injection.seed), observed_at_ms=5)
        registry.start_canary("tx", observed_at_ms=6)
        registry.finish_canary("tx", _gate("canary", seed=injection.seed), observed_at_ms=7)
        with pytest.raises(RuntimeError, match="injected fault"):
            registry.promote("tx", observed_at_ms=8, fault_after_pointer_update=True)
        assert registry.champion("prod").policy_epoch_id == "champion"
        assert registry.promotion("tx").state is PromotionState.CANARY_PASSED
    return FaultObservation(
        actual_response=FaultResponse.ROLLED_BACK,
        detail="registry transaction retained the prior champion after partial pointer fault",
        evidence_sha256=_digest(f"promotion:{injection.injection_id}"),
    )


def _single_request(*, timeout_ms: int = 100) -> FaultPlanRequest:
    return FaultPlanRequest(
        request_id="fault.single",
        seed=7,
        horizon_steps=1,
        callback_timeout_ms=timeout_ms,
        max_callbacks=1,
        faults=(
            FaultSpec(
                fault_id="fault.effect",
                ground_truth_label="rollout.illegal_effect",
                kind=FaultKind.ILLEGAL_EFFECT,
                stage=FaultStage.ROLLOUT,
                activation=ActivationInterval(start_step=0, end_step=1),
                expected_response=FaultResponse.REJECTED,
                evidence_sha256=_digest("effect"),
            ),
        ),
    )


def test_machine_matrix_covers_every_stage_and_fault_and_is_order_independent() -> None:
    request = load_fault_plan_request(ROOT / "scenarios" / "helix" / "faults" / "cpu-matrix.json")
    assert {fault.kind for fault in request.faults} == set(FaultKind)
    assert {fault.stage for fault in request.faults} == set(FaultStage)
    plan = compile_fault_plan(request)
    reversed_plan = compile_fault_plan(request.model_copy(update={"faults": request.faults[::-1]}))
    assert plan == reversed_plan
    assert plan.seed == 20260803
    assert len(plan.faults) == 16


def test_full_cpu_campaign_is_deterministic_and_records_required_ground_truth(
    tmp_path: Path,
) -> None:
    request = load_fault_plan_request(ROOT / "scenarios" / "helix" / "faults" / "cpu-matrix.json")
    plan = compile_fault_plan(request)
    callbacks = {kind: _observation for kind in FaultKind}
    callbacks[FaultKind.ILLEGAL_EFFECT] = _illegal_effect_observation
    callbacks[FaultKind.MISSING_LOGPROB] = _missing_logprob_observation
    callbacks[FaultKind.PARTIAL_CHAMPION_POINTER] = lambda injection: _partial_pointer_observation(
        injection, tmp_path / "promotion.sqlite"
    )
    first = FaultRunner().run(plan, callbacks)

    # The promotion callback persists a real SQLite artifact, so use the generic deterministic
    # callback for the exact replay assertion while retaining the real API result above.
    deterministic_callbacks = {kind: _observation for kind in FaultKind}
    replay = FaultRunner().run(plan, deterministic_callbacks)
    repeated = FaultRunner().run(plan, deterministic_callbacks)
    assert replay == repeated
    assert first.passed and first.callback_count == 16
    assert not first.failed_fault_ids
    assert all(result.passed for result in first.results)
    assert all(result.ground_truth_label for result in first.results)
    assert all(
        result.activation.end_step > result.activation.start_step for result in first.results
    )
    assert all(
        result.evidence_sha256 and result.observation_evidence_sha256 for result in first.results
    )
    assert all(result.injection_id for result in first.results)
    first.require_passed()


def test_injector_is_seeded_and_mutations_are_kind_specific() -> None:
    request = load_fault_plan_request(ROOT / "scenarios" / "helix" / "faults" / "cpu-matrix.json")
    injector = DeterministicFaultInjector()
    traffic = next(fault for fault in request.faults if fault.kind is FaultKind.TRAFFIC_SPIKE)
    first = injector.inject(traffic, seed=request.seed)
    assert first == injector.inject(traffic, seed=request.seed)
    assert first != injector.inject(traffic, seed=request.seed + 1)
    assert first.mutations[0].field == "resource.traffic"
    assert first.mutations[0].encoded_value == "2.5"


def test_missing_error_and_timeout_callbacks_fail_closed() -> None:
    plan = compile_fault_plan(_single_request(timeout_ms=1))
    missing = FaultRunner().run(plan, {})
    assert missing.results[0].actual_response is FaultResponse.CALLBACK_MISSING
    with pytest.raises(FaultCampaignFailed, match="failed closed"):
        missing.require_passed()

    def error_callback(_injection: InjectedFault) -> FaultObservation:
        raise RuntimeError("bounded failure")

    error = FaultRunner().run(plan, {FaultKind.ILLEGAL_EFFECT: error_callback})
    assert error.results[0].actual_response is FaultResponse.CALLBACK_ERROR

    def slow_callback(injection: InjectedFault) -> FaultObservation:
        time.sleep(0.02)
        return _observation(injection)

    timeout = FaultRunner().run(plan, {FaultKind.ILLEGAL_EFFECT: slow_callback})
    assert timeout.results[0].actual_response is FaultResponse.CALLBACK_TIMEOUT


def test_duplicate_wrong_stage_bounds_and_unknown_fields_are_rejected() -> None:
    request = _single_request()
    fault = request.faults[0]
    with pytest.raises(ValidationError, match="fault identifiers must be unique"):
        FaultPlanRequest.model_validate(
            {**request.model_dump(), "faults": (fault, fault.model_copy())}
        )
    with pytest.raises(ValidationError, match="wrong Helix stage"):
        FaultSpec.model_validate({**fault.model_dump(), "stage": FaultStage.REWARD})
    with pytest.raises(ValidationError, match="activation exceeds"):
        FaultPlanRequest.model_validate(
            {
                **request.model_dump(),
                "faults": (
                    fault.model_copy(
                        update={"activation": ActivationInterval(start_step=0, end_step=2)}
                    ),
                ),
            }
        )
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        FaultSpec.model_validate({**fault.model_dump(), "silent_success": True})


def test_plan_injection_and_result_tampering_are_detected() -> None:
    plan = compile_fault_plan(_single_request())
    result = FaultRunner().run(plan, {FaultKind.ILLEGAL_EFFECT: _observation})
    assert FaultPlan.model_validate_json(plan.model_dump_json()) == plan
    assert FaultCampaignResult.model_validate_json(result.model_dump_json()) == result

    tampered_plan = plan.model_dump()
    tampered_plan["seed"] = 8
    with pytest.raises(ValidationError, match="plan identifier is invalid"):
        FaultPlan.model_validate(tampered_plan)
    injection = DeterministicFaultInjector().inject(plan.faults[0], seed=plan.seed)
    tampered_injection = injection.model_dump()
    tampered_injection["seed"] = injection.seed + 1
    with pytest.raises(ValidationError, match="injected fault identifier is invalid"):
        InjectedFault.model_validate(tampered_injection)
    tampered_result = result.model_dump()
    tampered_result["seed"] = 8
    with pytest.raises(ValidationError, match="result identifier is invalid"):
        FaultCampaignResult.model_validate(tampered_result)
