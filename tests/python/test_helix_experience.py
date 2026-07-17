from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from sloforge.helix.experience import (
    ArtifactRef,
    EvidenceSource,
    ExclusionReason,
    ExperienceCandidate,
    ExperienceFeatures,
    ExperienceSelectionConstraints,
    ExperienceSelectionPlan,
    ExperienceSelectionRequest,
    PrivacyClass,
    SelectionStrategy,
    SideEffectRisk,
    load_experience_selection_request,
    select_experiences,
)

ROOT = Path(__file__).resolve().parents[2]


def _digest(character: str) -> str:
    return character * 64


def _artifact(candidate_id: str, character: str) -> ArtifactRef:
    return ArtifactRef(
        artifact_id=f"artifact.{candidate_id}",
        artifact_uri=f"evidence://experience/{candidate_id}",
        artifact_sha256=_digest(character),
        sample_ids=(f"sample.{candidate_id}",),
    )


def _features(**updates: object) -> ExperienceFeatures:
    values: dict[str, object] = {
        "failure": False,
        "verifier_disagreement": 0.0,
        "policy_uncertainty": 0.0,
        "value_uncertainty": 0.0,
        "novelty": 0.0,
        "rarity": 0.0,
        "safety": 0.0,
        "recurrence": 0.0,
        "autopsy_issue": 0.0,
        "reward_disagreement": 0.0,
        "capability_regression": 0.0,
        "branchability": 0.0,
        "expected_learning_value": 1.0,
    }
    values.update(updates)
    return ExperienceFeatures.model_validate(values)


def _candidate(
    candidate_id: str,
    *,
    character: str,
    features: ExperienceFeatures | None = None,
    tenant_id: str = "tenant.a",
    source: EvidenceSource = EvidenceSource.SYNTHETIC,
    privacy: PrivacyClass = PrivacyClass.TENANT_PRIVATE,
    risk: SideEffectRisk = SideEffectRisk.PURE,
    consent: bool = False,
    redacted: bool = False,
    authorization_hash: str | None = None,
    redaction_hash: str | None = None,
    requires_live_side_effects: bool = False,
    fingerprint: str | None = None,
    cost: int = 1,
    capacity: int = 1,
) -> ExperienceCandidate:
    return ExperienceCandidate(
        candidate_id=candidate_id,
        tenant_id=tenant_id,
        source=source,
        privacy=privacy,
        side_effect_risk=risk,
        consent_granted=consent,
        redaction_applied=redacted,
        requires_live_side_effects=requires_live_side_effects,
        content_fingerprint=fingerprint or _digest(character),
        artifacts=(_artifact(candidate_id, character),),
        authorization_artifact_sha256=authorization_hash,
        redaction_artifact_sha256=redaction_hash,
        features=features or _features(),
        training_cost_microunits=cost,
        capacity_units=capacity,
    )


def _request(
    candidates: tuple[ExperienceCandidate, ...],
    *,
    strategy: SelectionStrategy = SelectionStrategy.HELIX_VALUE_AWARE,
    seed: int = 73,
    budget: int = 100,
    capacity: int = 100,
    count: int = 1,
    production: bool = False,
    maximum_privacy: PrivacyClass = PrivacyClass.TENANT_PRIVATE,
    risks: tuple[SideEffectRisk, ...] = (SideEffectRisk.PURE, SideEffectRisk.READ_ONLY),
) -> ExperienceSelectionRequest:
    return ExperienceSelectionRequest(
        request_id="experience.test",
        tenant_id="tenant.a",
        seed=seed,
        strategy=strategy,
        constraints=ExperienceSelectionConstraints(
            budget_microunits=budget,
            capacity_units=capacity,
            max_selected_experiences=count,
            maximum_privacy=maximum_privacy,
            allowed_side_effect_risks=risks,
            allow_production_evidence=production,
        ),
        candidates=candidates,
        assumptions=("Feature calibration comes from the referenced evaluation run.",),
    )


def test_value_aware_selection_is_seeded_deterministic_and_order_independent() -> None:
    candidates = (
        _candidate("a", character="1", features=_features(novelty=0.8, branchability=0.7)),
        _candidate(
            "b",
            character="2",
            features=_features(failure=True, safety=0.9, autopsy_issue=0.8),
        ),
        _candidate("c", character="3", features=_features(policy_uncertainty=0.9)),
    )
    forward = select_experiences(_request(candidates, count=2))
    repeated = select_experiences(_request(candidates, count=2))
    reversed_plan = select_experiences(_request(tuple(reversed(candidates)), count=2))
    assert forward == repeated == reversed_plan
    assert forward.seed == 73
    assert tuple(decision.candidate_id for decision in forward.decisions) == ("a", "b", "c")
    assert all(decision.candidate_digest for decision in forward.decisions)
    assert all(decision.artifact_hashes for decision in forward.decisions)
    assert all(
        decision.prediction_uncertainty == decision.score.value_uncertainty
        for decision in forward.decisions
    )
    assert forward.assumptions[-1].startswith("Feature calibration")


@pytest.mark.parametrize(
    ("strategy", "expected"),
    (
        (SelectionStrategy.FAILURE_ONLY, "failure"),
        (SelectionStrategy.UNCERTAINTY_ONLY, "uncertain"),
        (SelectionStrategy.NOVELTY_ONLY, "novel"),
    ),
)
def test_required_signal_baselines_select_only_their_signal(
    strategy: SelectionStrategy, expected: str
) -> None:
    candidates = (
        _candidate("failure", character="1", features=_features(failure=True)),
        _candidate(
            "uncertain",
            character="2",
            features=_features(policy_uncertainty=0.85, value_uncertainty=0.95),
        ),
        _candidate("novel", character="3", features=_features(novelty=0.9)),
        _candidate("plain", character="4"),
    )
    plan = select_experiences(_request(candidates, strategy=strategy, count=1))
    assert plan.selected_candidate_ids == (expected,)
    for decision in plan.decisions:
        if decision.candidate_id not in {expected, "plain"}:
            assert ExclusionReason.BASELINE_FILTERED in decision.exclusion_reasons


def test_random_baseline_is_repeatable_and_seed_sensitive() -> None:
    candidates = tuple(
        _candidate(f"candidate.{index}", character=str(index)) for index in range(1, 7)
    )
    first = select_experiences(
        _request(candidates, strategy=SelectionStrategy.RANDOM, seed=11, count=2)
    )
    again = select_experiences(
        _request(candidates, strategy=SelectionStrategy.RANDOM, seed=11, count=2)
    )
    changed = select_experiences(
        _request(candidates, strategy=SelectionStrategy.RANDOM, seed=12, count=2)
    )
    assert first == again
    assert first.selected_candidate_ids != changed.selected_candidate_ids


def test_budget_capacity_and_no_train_all_are_hard_limits() -> None:
    candidates = (
        _candidate(
            "expensive",
            character="1",
            features=_features(failure=True, safety=1.0),
            cost=9,
            capacity=1,
        ),
        _candidate(
            "large",
            character="2",
            features=_features(failure=True, safety=0.9),
            cost=2,
            capacity=9,
        ),
        _candidate(
            "fits",
            character="3",
            features=_features(failure=True, safety=0.8),
            cost=2,
            capacity=2,
        ),
    )
    plan = select_experiences(_request(candidates, budget=3, capacity=3, count=3))
    assert plan.selected_candidate_ids == ("fits",)
    decisions = {decision.candidate_id: decision for decision in plan.decisions}
    assert ExclusionReason.BUDGET_EXHAUSTED in decisions["expensive"].exclusion_reasons
    assert ExclusionReason.CAPACITY_EXHAUSTED in decisions["large"].exclusion_reasons
    assert plan.accounting.budget_used_microunits == 2
    assert plan.accounting.capacity_used_units == 2
    assert plan.accounting.effective_max_count == 2

    no_train_all = select_experiences(_request(candidates[:2], count=10))
    assert no_train_all.accounting.selected_count == 1
    assert ExclusionReason.TRAIN_ALL_GUARD in {
        reason for decision in no_train_all.decisions for reason in decision.exclusion_reasons
    }


def test_production_consent_redaction_privacy_and_tenant_are_fail_closed() -> None:
    unauthorized = _candidate(
        "production.bad",
        character="1",
        tenant_id="tenant.b",
        source=EvidenceSource.AUTHORIZED_PRODUCTION,
        privacy=PrivacyClass.RESTRICTED,
    )
    authorized = _candidate(
        "production.good",
        character="2",
        source=EvidenceSource.AUTHORIZED_PRODUCTION,
        consent=True,
        redacted=True,
        authorization_hash=_digest("a"),
        redaction_hash=_digest("b"),
        features=_features(novelty=1.0),
    )
    plan = select_experiences(
        _request(
            (unauthorized, authorized),
            production=True,
            count=1,
            maximum_privacy=PrivacyClass.TENANT_PRIVATE,
        )
    )
    assert plan.selected_candidate_ids == ("production.good",)
    bad = {decision.candidate_id: decision for decision in plan.decisions}["production.bad"]
    assert set(bad.exclusion_reasons) >= {
        ExclusionReason.TENANT_MISMATCH,
        ExclusionReason.CONSENT_REQUIRED,
        ExclusionReason.AUTHORIZATION_ARTIFACT_REQUIRED,
        ExclusionReason.REDACTION_REQUIRED,
        ExclusionReason.REDACTION_ARTIFACT_REQUIRED,
        ExclusionReason.PRIVACY_NOT_ALLOWED,
    }


def test_illegal_or_live_effects_are_never_selected() -> None:
    candidates = (
        _candidate(
            "external",
            character="1",
            risk=SideEffectRisk.EXTERNAL,
            features=_features(failure=True),
        ),
        _candidate(
            "live",
            character="2",
            risk=SideEffectRisk.REVERSIBLE,
            requires_live_side_effects=True,
            features=_features(failure=True),
        ),
        _candidate("pure", character="3", features=_features(failure=True)),
    )
    plan = select_experiences(
        _request(
            candidates,
            count=2,
            risks=(SideEffectRisk.PURE, SideEffectRisk.REVERSIBLE),
        )
    )
    decisions = {decision.candidate_id: decision for decision in plan.decisions}
    assert plan.selected_candidate_ids == ("pure",)
    assert ExclusionReason.SIDE_EFFECT_RISK_NOT_ALLOWED in decisions["external"].exclusion_reasons
    assert ExclusionReason.LIVE_SIDE_EFFECT_REQUIRED in decisions["live"].exclusion_reasons
    with pytest.raises(ValidationError, match="cannot be authorized"):
        ExperienceSelectionConstraints(
            budget_microunits=1,
            capacity_units=1,
            max_selected_experiences=1,
            maximum_privacy=PrivacyClass.PUBLIC,
            allowed_side_effect_risks=(SideEffectRisk.EXTERNAL,),
            allow_production_evidence=False,
        )


def test_redundant_evidence_is_excluded_after_the_best_representative() -> None:
    fingerprint = _digest("f")
    candidates = (
        _candidate(
            "duplicate.low",
            character="1",
            fingerprint=fingerprint,
            features=_features(novelty=0.2),
        ),
        _candidate(
            "duplicate.high",
            character="2",
            fingerprint=fingerprint,
            features=_features(novelty=0.9),
        ),
        _candidate("independent", character="3", features=_features(novelty=0.8)),
    )
    plan = select_experiences(_request(candidates, count=2))
    assert set(plan.selected_candidate_ids) == {"duplicate.high", "independent"}
    low = {decision.candidate_id: decision for decision in plan.decisions}["duplicate.low"]
    assert ExclusionReason.REDUNDANT in low.exclusion_reasons


def test_governance_blocked_outlier_cannot_calibrate_eligible_value_scores() -> None:
    eligible = _candidate(
        "eligible",
        character="1",
        features=_features(expected_learning_value=1.0),
    )
    blocked = _candidate(
        "blocked",
        character="2",
        tenant_id="tenant.other",
        features=_features(expected_learning_value=10**12),
    )
    request = _request((eligible, blocked), count=1)
    request = request.model_copy(
        update={"constraints": request.constraints.model_copy(update={"minimum_score": 0.05})}
    )
    plan = select_experiences(request)
    decisions = {item.candidate_id: item for item in plan.decisions}
    assert plan.selected_candidate_ids == ("eligible",)
    assert decisions["eligible"].score.normalized_value_cost == 1.0
    assert ExclusionReason.TENANT_MISMATCH in decisions["blocked"].exclusion_reasons


def test_strict_bounds_unknown_fields_and_plan_tampering_fail_validation() -> None:
    candidate = _candidate("candidate", character="1", features=_features(novelty=1.0))
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        ExperienceCandidate.model_validate({**candidate.model_dump(), "payload": "secret"})
    with pytest.raises(ValidationError):
        ExperienceFeatures.model_validate(
            {**candidate.features.model_dump(), "novelty": float("nan")}
        )

    plan = select_experiences(_request((candidate,), count=1))
    restored = ExperienceSelectionPlan.model_validate_json(plan.model_dump_json())
    assert restored == plan
    tampered = plan.model_dump()
    tampered["seed"] = 74
    with pytest.raises(ValidationError, match="plan identifier is invalid"):
        ExperienceSelectionPlan.model_validate(tampered)
    unknown = plan.model_dump()
    unknown["raw_production_payload"] = "forbidden"
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        ExperienceSelectionPlan.model_validate(unknown)


def test_value_aware_scenario_loads_through_the_bounded_json_boundary() -> None:
    request = load_experience_selection_request(
        ROOT / "scenarios" / "helix" / "experience" / "value-aware.json"
    )
    plan = select_experiences(request)
    assert plan.accounting.selected_count == 2
    blocked = {decision.candidate_id: decision for decision in plan.decisions}[
        "experience.production.blocked"
    ]
    assert ExclusionReason.TENANT_MISMATCH in blocked.exclusion_reasons
    assert ExclusionReason.LIVE_SIDE_EFFECT_REQUIRED in blocked.exclusion_reasons
