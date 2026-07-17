"""Deterministic reference selector for bounded Helix experience curation."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from .models import (
    CandidateDecision,
    EvidenceSource,
    ExclusionReason,
    ExperienceCandidate,
    ExperienceSelectionPlan,
    ExperienceSelectionRequest,
    PrivacyClass,
    SelectionAccounting,
    SelectionScore,
    SelectionStrategy,
    SideEffectRisk,
    canonical_digest,
)

_PRIVACY_RANK = {
    PrivacyClass.PUBLIC: 0,
    PrivacyClass.TENANT_PRIVATE: 1,
    PrivacyClass.RESTRICTED: 2,
}

_SYSTEM_ASSUMPTIONS = (
    "Supplied features, uncertainty, costs, and artifact hashes are treated as input claims.",
    "Capacity units and microunit costs are planning estimates, not observed measurements.",
    "Selection grants no authority to replay external effects or reveal evidence payloads.",
    "Exact content fingerprints define redundancy; semantic near-duplicates need upstream clustering.",
)

_SYSTEM_LIMITATIONS = (
    "The reference selector does not fetch artifacts or independently verify their payload hashes.",
    "Expected learning value is predictive and remains uncertain until separately evaluated.",
    "Value-per-cost normalization uses only governance-eligible candidates; blocked evidence cannot calibrate eligible scores.",
)


@dataclass(frozen=True, slots=True)
class _ScoredCandidate:
    candidate: ExperienceCandidate
    score: SelectionScore
    hard_reasons: tuple[ExclusionReason, ...]


def _tie_break(seed: int, candidate_id: str) -> str:
    payload = f"sloforge.helix.experience-selection/v1\0{seed}\0{candidate_id}".encode()
    return hashlib.sha256(payload).hexdigest()


def _random_score(tie_break: str) -> float:
    numerator = int(tie_break[:16], 16)
    return round(numerator / (2**64 - 1), 12)


def _request_digest(request: ExperienceSelectionRequest) -> str:
    payload = request.model_dump(mode="json")
    candidates = payload["candidates"]
    if not isinstance(candidates, list):
        raise TypeError("serialized experience candidates must be a list")
    payload["candidates"] = sorted(candidates, key=lambda item: str(item["candidate_id"]))
    return canonical_digest(payload)


def _candidate_digest(candidate: ExperienceCandidate) -> str:
    return canonical_digest(candidate.model_dump(mode="json"))


def _artifact_hashes(candidate: ExperienceCandidate) -> tuple[str, ...]:
    hashes = {artifact.artifact_sha256 for artifact in candidate.artifacts}
    if candidate.authorization_artifact_sha256 is not None:
        hashes.add(candidate.authorization_artifact_sha256)
    if candidate.redaction_artifact_sha256 is not None:
        hashes.add(candidate.redaction_artifact_sha256)
    return tuple(sorted(hashes))


def _governance_reasons(
    request: ExperienceSelectionRequest, candidate: ExperienceCandidate
) -> tuple[ExclusionReason, ...]:
    reasons: list[ExclusionReason] = []
    constraints = request.constraints
    if candidate.tenant_id != request.tenant_id:
        reasons.append(ExclusionReason.TENANT_MISMATCH)
    if candidate.source is EvidenceSource.AUTHORIZED_PRODUCTION:
        if not constraints.allow_production_evidence:
            reasons.append(ExclusionReason.PRODUCTION_DISABLED)
        if not candidate.consent_granted:
            reasons.append(ExclusionReason.CONSENT_REQUIRED)
        if candidate.authorization_artifact_sha256 is None:
            reasons.append(ExclusionReason.AUTHORIZATION_ARTIFACT_REQUIRED)
        if not candidate.redaction_applied:
            reasons.append(ExclusionReason.REDACTION_REQUIRED)
        if candidate.redaction_artifact_sha256 is None:
            reasons.append(ExclusionReason.REDACTION_ARTIFACT_REQUIRED)
    if _PRIVACY_RANK[candidate.privacy] > _PRIVACY_RANK[constraints.maximum_privacy]:
        reasons.append(ExclusionReason.PRIVACY_NOT_ALLOWED)
    if candidate.side_effect_risk not in constraints.allowed_side_effect_risks:
        reasons.append(ExclusionReason.SIDE_EFFECT_RISK_NOT_ALLOWED)
    if (
        candidate.side_effect_risk
        in {
            SideEffectRisk.EXTERNAL,
            SideEffectRisk.IRREVERSIBLE,
        }
        and ExclusionReason.SIDE_EFFECT_RISK_NOT_ALLOWED not in reasons
    ):
        reasons.append(ExclusionReason.SIDE_EFFECT_RISK_NOT_ALLOWED)
    if candidate.requires_live_side_effects:
        reasons.append(ExclusionReason.LIVE_SIDE_EFFECT_REQUIRED)
    return tuple(reasons)


def _baseline_reason(
    strategy: SelectionStrategy, candidate: ExperienceCandidate
) -> ExclusionReason | None:
    features = candidate.features
    if strategy is SelectionStrategy.FAILURE_ONLY and not features.failure:
        return ExclusionReason.BASELINE_FILTERED
    if (
        strategy is SelectionStrategy.UNCERTAINTY_ONLY
        and max(features.policy_uncertainty, features.value_uncertainty) <= 0.0
    ):
        return ExclusionReason.BASELINE_FILTERED
    if strategy is SelectionStrategy.NOVELTY_ONLY and features.novelty <= 0.0:
        return ExclusionReason.BASELINE_FILTERED
    if strategy is SelectionStrategy.HELIX_VALUE_AWARE and features.expected_learning_value <= 0.0:
        return ExclusionReason.NON_POSITIVE_LEARNING_VALUE
    return None


def _score_candidate(
    request: ExperienceSelectionRequest,
    candidate: ExperienceCandidate,
    *,
    maximum_value_cost_ratio: float,
) -> SelectionScore:
    features = candidate.features
    value_cost_ratio = features.expected_learning_value / candidate.training_cost_microunits
    normalized_value_cost = (
        value_cost_ratio / maximum_value_cost_ratio if maximum_value_cost_ratio > 0.0 else 0.0
    )
    tie_break = _tie_break(request.seed, candidate.candidate_id)
    if request.strategy is SelectionStrategy.RANDOM:
        strategy_score = _random_score(tie_break)
    elif request.strategy is SelectionStrategy.FAILURE_ONLY:
        strategy_score = 1.0 if features.failure else 0.0
    elif request.strategy is SelectionStrategy.UNCERTAINTY_ONLY:
        strategy_score = max(features.policy_uncertainty, features.value_uncertainty)
    elif request.strategy is SelectionStrategy.NOVELTY_ONLY:
        strategy_score = features.novelty
    else:
        weights = request.weights
        weighted = (
            weights.failure * float(features.failure)
            + weights.verifier_disagreement * features.verifier_disagreement
            + weights.policy_uncertainty * features.policy_uncertainty
            + weights.value_uncertainty * features.value_uncertainty
            + weights.novelty * features.novelty
            + weights.rarity * features.rarity
            + weights.safety * features.safety
            + weights.recurrence * features.recurrence
            + weights.autopsy_issue * features.autopsy_issue
            + weights.reward_disagreement * features.reward_disagreement
            + weights.capability_regression * features.capability_regression
            + weights.branchability * features.branchability
            + weights.expected_value_per_cost * normalized_value_cost
        )
        strategy_score = weighted / weights.total
    return SelectionScore(
        failure=float(features.failure),
        verifier_disagreement=features.verifier_disagreement,
        policy_uncertainty=features.policy_uncertainty,
        value_uncertainty=features.value_uncertainty,
        novelty=features.novelty,
        rarity=features.rarity,
        safety=features.safety,
        recurrence=features.recurrence,
        autopsy_issue=features.autopsy_issue,
        reward_disagreement=features.reward_disagreement,
        capability_regression=features.capability_regression,
        branchability=features.branchability,
        expected_learning_value=features.expected_learning_value,
        value_cost_ratio=round(value_cost_ratio, 15),
        normalized_value_cost=round(min(1.0, normalized_value_cost), 12),
        strategy_score=round(min(1.0, strategy_score), 12),
        deterministic_tie_break=tie_break,
    )


def _rank_key(item: _ScoredCandidate) -> tuple[float | str, ...]:
    return (
        -item.score.strategy_score,
        -item.score.normalized_value_cost,
        item.score.deterministic_tie_break,
        item.candidate.candidate_id,
    )


def select_experiences(request: ExperienceSelectionRequest) -> ExperienceSelectionPlan:
    """Compile a complete, deterministic experience-selection plan.

    Hard governance checks run before ranking.  Remaining candidates are greedily admitted
    in score order under all three explicit limits, with exact-fingerprint deduplication.
    """

    ordered_candidates = tuple(sorted(request.candidates, key=lambda item: item.candidate_id))
    governance = {
        candidate.candidate_id: _governance_reasons(request, candidate)
        for candidate in ordered_candidates
    }
    maximum_ratio = max(
        (
            candidate.features.expected_learning_value / candidate.training_cost_microunits
            for candidate in ordered_candidates
            if not governance[candidate.candidate_id]
        ),
        default=0.0,
    )
    scored = tuple(
        _ScoredCandidate(
            candidate=candidate,
            score=_score_candidate(
                request,
                candidate,
                maximum_value_cost_ratio=maximum_ratio,
            ),
            hard_reasons=governance[candidate.candidate_id],
        )
        for candidate in ordered_candidates
    )
    eligible: list[_ScoredCandidate] = []
    exclusions: dict[str, tuple[ExclusionReason, ...]] = {}
    for item in scored:
        eligibility_reasons = list(item.hard_reasons)
        baseline_reason = _baseline_reason(request.strategy, item.candidate)
        if baseline_reason is not None:
            eligibility_reasons.append(baseline_reason)
        if item.score.strategy_score < request.constraints.minimum_score:
            eligibility_reasons.append(ExclusionReason.BELOW_MINIMUM_SCORE)
        if eligibility_reasons:
            exclusions[item.candidate.candidate_id] = tuple(dict.fromkeys(eligibility_reasons))
        else:
            eligible.append(item)

    ranked = sorted(eligible, key=_rank_key)
    configured_max = request.constraints.max_selected_experiences
    train_all_limit = len(request.candidates) - 1 if len(request.candidates) > 1 else 1
    effective_max = min(configured_max, train_all_limit)
    selected: list[_ScoredCandidate] = []
    selected_fingerprints: set[str] = set()
    budget_used = 0
    capacity_used = 0
    for item in ranked:
        candidate = item.candidate
        admission_reasons: list[ExclusionReason] = []
        if candidate.content_fingerprint in selected_fingerprints:
            admission_reasons.append(ExclusionReason.REDUNDANT)
        if len(selected) >= effective_max:
            if configured_max > effective_max:
                admission_reasons.append(ExclusionReason.TRAIN_ALL_GUARD)
            else:
                admission_reasons.append(ExclusionReason.MAXIMUM_COUNT_REACHED)
        if budget_used + candidate.training_cost_microunits > request.constraints.budget_microunits:
            admission_reasons.append(ExclusionReason.BUDGET_EXHAUSTED)
        if capacity_used + candidate.capacity_units > request.constraints.capacity_units:
            admission_reasons.append(ExclusionReason.CAPACITY_EXHAUSTED)
        if admission_reasons:
            exclusions[candidate.candidate_id] = tuple(dict.fromkeys(admission_reasons))
            continue
        selected.append(item)
        selected_fingerprints.add(candidate.content_fingerprint)
        budget_used += candidate.training_cost_microunits
        capacity_used += candidate.capacity_units

    selection_ranks = {
        item.candidate.candidate_id: rank for rank, item in enumerate(selected, start=1)
    }
    decisions = tuple(
        CandidateDecision(
            candidate_id=item.candidate.candidate_id,
            candidate_digest=_candidate_digest(item.candidate),
            selected=item.candidate.candidate_id in selection_ranks,
            selection_rank=selection_ranks.get(item.candidate.candidate_id),
            score=item.score,
            prediction_uncertainty=item.candidate.features.value_uncertainty,
            training_cost_microunits=item.candidate.training_cost_microunits,
            capacity_units=item.candidate.capacity_units,
            artifact_hashes=_artifact_hashes(item.candidate),
            exclusion_reasons=exclusions.get(item.candidate.candidate_id, ()),
        )
        for item in scored
    )
    artifact_hashes = tuple(
        sorted({digest for decision in decisions for digest in decision.artifact_hashes})
    )
    accounting = SelectionAccounting(
        budget_limit_microunits=request.constraints.budget_microunits,
        budget_used_microunits=budget_used,
        budget_remaining_microunits=request.constraints.budget_microunits - budget_used,
        capacity_limit_units=request.constraints.capacity_units,
        capacity_used_units=capacity_used,
        capacity_remaining_units=request.constraints.capacity_units - capacity_used,
        configured_max_count=configured_max,
        effective_max_count=effective_max,
        selected_count=len(selected),
        excluded_count=len(decisions) - len(selected),
    )
    assumptions = tuple(dict.fromkeys((*_SYSTEM_ASSUMPTIONS, *request.assumptions)))
    request_digest = _request_digest(request)
    selected_candidate_ids = tuple(item.candidate.candidate_id for item in selected)
    unsealed = ExperienceSelectionPlan.model_construct(
        plan_id="0" * 64,
        request_digest=request_digest,
        request_id=request.request_id,
        tenant_id=request.tenant_id,
        seed=request.seed,
        strategy=request.strategy,
        selected_candidate_ids=selected_candidate_ids,
        decisions=decisions,
        accounting=accounting,
        input_artifact_hashes=artifact_hashes,
        assumptions=assumptions,
        limitations=_SYSTEM_LIMITATIONS,
    )
    plan_id = canonical_digest(unsealed.model_dump(mode="json", exclude={"plan_id"}))
    return ExperienceSelectionPlan(
        plan_id=plan_id,
        request_digest=request_digest,
        request_id=request.request_id,
        tenant_id=request.tenant_id,
        seed=request.seed,
        strategy=request.strategy,
        selected_candidate_ids=selected_candidate_ids,
        decisions=decisions,
        accounting=accounting,
        input_artifact_hashes=artifact_hashes,
        assumptions=assumptions,
        limitations=_SYSTEM_LIMITATIONS,
    )


def compile_experience_selection_plan(
    request: ExperienceSelectionRequest,
) -> ExperienceSelectionPlan:
    """Compatibility spelling matching the other Helix plan compilers."""

    return select_experiences(request)


class ExperienceSelector:
    """Stateless reference-selector facade for dependency injection."""

    def select(self, request: ExperienceSelectionRequest) -> ExperienceSelectionPlan:
        return select_experiences(request)


__all__ = [
    "ExperienceSelector",
    "compile_experience_selection_plan",
    "select_experiences",
]
