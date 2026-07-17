"""Counterfactual credit for controlled sibling branches."""

from __future__ import annotations

import math
from collections.abc import Mapping
from statistics import fmean, pstdev
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class _CreditModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class BranchOutcome(_CreditModel):
    branch_id: Annotated[str, Field(min_length=1, max_length=160)]
    trajectory_id: Annotated[str, Field(min_length=1, max_length=160)]
    policy_epoch_id: Annotated[str, Field(min_length=1, max_length=160)]
    action: Annotated[str, Field(min_length=1, max_length=160)]
    behavior_log_probability: float
    reward_components: Mapping[str, float]
    first_divergent_action_index: Annotated[int, Field(ge=0)]
    suffix_action_count: Annotated[int, Field(ge=1, le=1_000_000)]
    process_score: Annotated[float | None, Field(ge=-1.0, le=1.0)] = None
    valid: bool = True
    intervention: Literal["controlled_action", "controlled_rng", "controlled_tool", "observational"]

    @model_validator(mode="after")
    def validate_finite_evidence(self) -> Self:
        values = (self.behavior_log_probability, *self.reward_components.values())
        if any(not math.isfinite(value) for value in values):
            raise ValueError("credit inputs must be finite")
        if self.behavior_log_probability > 0.0:
            raise ValueError("behavior log probability must be non-positive")
        if not self.reward_components:
            raise ValueError("credit assignment requires preserved reward components")
        return self

    @property
    def total_reward(self) -> float:
        return float(math.fsum(self.reward_components.values()))


class BranchCredit(_CreditModel):
    branch_id: str
    trajectory_id: str
    action: str
    behavior_log_probability: float
    reward_components: Mapping[str, float]
    total_reward: float
    group_relative_advantage: float
    decision_local_weight: Annotated[float, Field(ge=0.0, le=1.0)]
    weighted_advantage: float
    confidence: Annotated[float, Field(ge=0.0, le=1.0)]
    excluded_reason: str | None = None

    @model_validator(mode="after")
    def validate_finite_credit(self) -> Self:
        values = (
            self.behavior_log_probability,
            *self.reward_components.values(),
            self.total_reward,
            self.group_relative_advantage,
            self.weighted_advantage,
        )
        if any(not math.isfinite(value) for value in values):
            raise ValueError("branch credit values must be finite")
        if self.behavior_log_probability > 0.0:
            raise ValueError("behavior log probability must be non-positive")
        if not self.reward_components:
            raise ValueError("branch credit must preserve reward components")
        if not math.isclose(
            math.fsum(self.reward_components.values()),
            self.total_reward,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError("branch credit total must equal its reward components")
        expected_weighted = self.group_relative_advantage * self.decision_local_weight
        if not math.isclose(
            self.weighted_advantage, expected_weighted, rel_tol=1e-12, abs_tol=1e-12
        ):
            raise ValueError("weighted advantage must equal advantage times local weight")
        if self.excluded_reason is not None and any(
            value != 0.0
            for value in (
                self.group_relative_advantage,
                self.decision_local_weight,
                self.weighted_advantage,
                self.confidence,
            )
        ):
            raise ValueError("excluded branch credit cannot carry optimization weight")
        return self


class PairwisePreference(_CreditModel):
    chosen_branch_id: str
    rejected_branch_id: str
    reward_margin: Annotated[float, Field(gt=0.0)]
    confidence: Annotated[float, Field(gt=0.0, le=1.0)]
    controlled_difference: bool


class BranchRelativeCredit(_CreditModel):
    schema_version: str = "sloforge.helix.branch-credit/v1"
    branch_group_id: Annotated[str, Field(min_length=1, max_length=160)]
    branch_point_id: Annotated[str, Field(min_length=1, max_length=160)]
    policy_epoch_id: Annotated[str, Field(min_length=1, max_length=160)]
    method: Literal["branch_relative"] = "branch_relative"
    divergence_action_index: Annotated[int, Field(ge=0)]
    credits: Annotated[tuple[BranchCredit, ...], Field(min_length=2, max_length=4_096)]
    preferences: Annotated[tuple[PairwisePreference, ...], Field(max_length=65_536)]
    assumptions: Annotated[tuple[str, ...], Field(min_length=1, max_length=32)]

    @model_validator(mode="after")
    def validate_credit_graph(self) -> Self:
        by_branch = {item.branch_id: item for item in self.credits}
        if len(by_branch) != len(self.credits):
            raise ValueError("branch credit identifiers must be unique")
        trajectory_ids = [item.trajectory_id for item in self.credits]
        if len(trajectory_ids) != len(set(trajectory_ids)):
            raise ValueError("branch credit trajectory identifiers must be unique")
        preference_pairs: set[frozenset[str]] = set()
        for preference in self.preferences:
            if preference.chosen_branch_id == preference.rejected_branch_id:
                raise ValueError("preference branches must be distinct")
            if (
                preference.chosen_branch_id not in by_branch
                or preference.rejected_branch_id not in by_branch
            ):
                raise ValueError("preference references unknown branch credit")
            pair = frozenset((preference.chosen_branch_id, preference.rejected_branch_id))
            if pair in preference_pairs:
                raise ValueError("branch pair can have only one preference direction")
            preference_pairs.add(pair)
            chosen = by_branch[preference.chosen_branch_id]
            rejected = by_branch[preference.rejected_branch_id]
            if chosen.excluded_reason is not None or rejected.excluded_reason is not None:
                raise ValueError("preference cannot reference excluded branch evidence")
            if not preference.controlled_difference:
                raise ValueError("branch-relative preference requires a controlled difference")
            expected_margin = chosen.total_reward - rejected.total_reward
            if not math.isclose(
                preference.reward_margin, expected_margin, rel_tol=1e-12, abs_tol=1e-12
            ):
                raise ValueError("preference margin disagrees with branch rewards")
        return self


def _confidence(outcome: BranchOutcome) -> float:
    intervention_confidence = {
        "controlled_action": 0.95,
        "controlled_rng": 0.75,
        "controlled_tool": 0.85,
        "observational": 0.35,
    }[outcome.intervention]
    if outcome.process_score is None:
        return intervention_confidence * 0.85
    return min(1.0, intervention_confidence * (0.85 + 0.15 * abs(outcome.process_score)))


def assign_branch_relative_credit(
    *,
    branch_group_id: str,
    branch_point_id: str,
    outcomes: tuple[BranchOutcome, ...],
    decay: float = 0.85,
    preference_margin: float = 0.1,
) -> BranchRelativeCredit:
    """Assign scoped counterfactual credit without claiming perfect causality."""

    if len(outcomes) < 2:
        raise ValueError("branch-relative credit requires at least two sibling outcomes")
    if not 0.0 < decay <= 1.0:
        raise ValueError("credit decay must be in (0, 1]")
    if not math.isfinite(preference_margin) or preference_margin < 0.0:
        raise ValueError("preference margin must be non-negative")
    ids = [item.branch_id for item in outcomes]
    if len(set(ids)) != len(ids):
        raise ValueError("sibling branch identifiers must be unique")
    trajectory_ids = [item.trajectory_id for item in outcomes]
    if len(set(trajectory_ids)) != len(trajectory_ids):
        raise ValueError("sibling trajectory identifiers must be unique")
    policy_epochs = {item.policy_epoch_id for item in outcomes}
    if len(policy_epochs) != 1:
        raise ValueError("branch-relative on-policy credit requires one policy epoch")
    valid = tuple(item for item in outcomes if item.valid and item.intervention != "observational")
    if len(valid) < 2:
        raise ValueError("fewer than two valid controlled sibling interventions remain")
    component_sets = {frozenset(item.reward_components) for item in valid}
    if len(component_sets) != 1:
        raise ValueError("valid sibling rewards must preserve the same component set")
    divergence_indices = {item.first_divergent_action_index for item in valid}
    if len(divergence_indices) != 1:
        raise ValueError("valid siblings must share one first divergent action index")
    rewards = tuple(item.total_reward for item in valid)
    mean = fmean(rewards)
    spread = pstdev(rewards)
    scale = spread if spread > 1e-12 else 1.0
    divergence = next(iter(divergence_indices))
    credits: list[BranchCredit] = []
    for outcome in outcomes:
        if not outcome.valid or outcome.intervention == "observational":
            credits.append(
                BranchCredit(
                    branch_id=outcome.branch_id,
                    trajectory_id=outcome.trajectory_id,
                    action=outcome.action,
                    behavior_log_probability=outcome.behavior_log_probability,
                    reward_components=outcome.reward_components,
                    total_reward=outcome.total_reward,
                    group_relative_advantage=0.0,
                    decision_local_weight=0.0,
                    weighted_advantage=0.0,
                    confidence=0.0,
                    excluded_reason=(
                        "invalid branch evidence"
                        if not outcome.valid
                        else "observational branch lacks controlled counterfactual support"
                    ),
                )
            )
            continue
        advantage = (outcome.total_reward - mean) / scale
        distance = outcome.suffix_action_count - 1
        locality = decay**distance
        if outcome.process_score is not None:
            locality *= 0.5 + 0.5 * max(0.0, outcome.process_score)
        locality = min(1.0, max(0.0, locality))
        credits.append(
            BranchCredit(
                branch_id=outcome.branch_id,
                trajectory_id=outcome.trajectory_id,
                action=outcome.action,
                behavior_log_probability=outcome.behavior_log_probability,
                reward_components=outcome.reward_components,
                total_reward=outcome.total_reward,
                group_relative_advantage=advantage,
                decision_local_weight=locality,
                weighted_advantage=advantage * locality,
                confidence=_confidence(outcome),
            )
        )
    preferences: list[PairwisePreference] = []
    ordered = sorted(valid, key=lambda item: (item.total_reward, item.branch_id), reverse=True)
    for chosen_index, chosen in enumerate(ordered):
        for rejected in ordered[chosen_index + 1 :]:
            margin = chosen.total_reward - rejected.total_reward
            if margin <= preference_margin:
                continue
            controlled = (
                chosen.intervention != "observational" and rejected.intervention != "observational"
            )
            preferences.append(
                PairwisePreference(
                    chosen_branch_id=chosen.branch_id,
                    rejected_branch_id=rejected.branch_id,
                    reward_margin=margin,
                    confidence=min(_confidence(chosen), _confidence(rejected)),
                    controlled_difference=controlled,
                )
            )
    return BranchRelativeCredit(
        branch_group_id=branch_group_id,
        branch_point_id=branch_point_id,
        policy_epoch_id=next(iter(policy_epochs)),
        divergence_action_index=divergence,
        credits=tuple(credits),
        preferences=tuple(preferences),
        assumptions=(
            "siblings share one validated BranchPoint",
            "reward components are independently validated",
            "observational branches are retained for accounting but excluded from optimization",
            "process scores must come from evidence independent of terminal reward",
            "controlled sibling differences support local counterfactual comparison, not universal causal identification",
            "unmodeled environment interactions may still confound later suffix credit",
        ),
    )
