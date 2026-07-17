from __future__ import annotations

import math

import pytest

from sloforge.helix.credit import BranchOutcome, assign_branch_relative_credit
from sloforge.helix.policy import DeterministicPolicy
from sloforge.helix.trainers import (
    ReferenceTrainer,
    ReferenceTrainingSample,
    TrainingAlgorithm,
)


def _policy() -> DeterministicPolicy:
    return DeterministicPolicy(
        policy_epoch_id="policy-champion-1",
        actions=("naive-total-credit", "merchandise-only-credit", "reject-credit"),
        logits=(2.0, 0.0, -1.0),
    )


def _outcomes(policy: DeterministicPolicy) -> tuple[BranchOutcome, ...]:
    return (
        BranchOutcome(
            branch_id="branch-naive",
            trajectory_id="trajectory-naive",
            policy_epoch_id=policy.policy_epoch_id,
            action="naive-total-credit",
            behavior_log_probability=policy.log_probability("naive-total-credit"),
            reward_components={"visible_tests": 1.0, "hidden_tests": -1.0, "minimality": 0.2},
            first_divergent_action_index=3,
            suffix_action_count=2,
            process_score=-0.5,
            intervention="controlled_rng",
        ),
        BranchOutcome(
            branch_id="branch-correct",
            trajectory_id="trajectory-correct",
            policy_epoch_id=policy.policy_epoch_id,
            action="merchandise-only-credit",
            behavior_log_probability=policy.log_probability("merchandise-only-credit"),
            reward_components={"visible_tests": 1.0, "hidden_tests": 1.0, "minimality": 0.2},
            first_divergent_action_index=3,
            suffix_action_count=2,
            process_score=1.0,
            intervention="controlled_action",
        ),
        BranchOutcome(
            branch_id="branch-reject",
            trajectory_id="trajectory-reject",
            policy_epoch_id=policy.policy_epoch_id,
            action="reject-credit",
            behavior_log_probability=policy.log_probability("reject-credit"),
            reward_components={"visible_tests": -1.0, "hidden_tests": -1.0, "minimality": 0.1},
            first_divergent_action_index=3,
            suffix_action_count=1,
            process_score=-1.0,
            intervention="controlled_tool",
        ),
    )


def _samples(policy: DeterministicPolicy) -> tuple[ReferenceTrainingSample, ...]:
    credit = assign_branch_relative_credit(
        branch_group_id="group-1",
        branch_point_id="point-1",
        outcomes=_outcomes(policy),
    )
    return tuple(
        ReferenceTrainingSample(
            sample_id=f"sample-{item.branch_id}",
            trajectory_id=item.trajectory_id,
            branch_point_id="point-1",
            branch_group_id="group-1",
            policy_epoch_id=policy.policy_epoch_id,
            action=item.action,
            behavior_log_probability=policy.log_probability(item.action),
            advantage=item.group_relative_advantage,
            token_weight=item.decision_local_weight,
            reward_margin=max(0.0, item.total_reward),
        )
        for item in credit.credits
    )


def test_policy_decisions_preserve_exact_behavior_probability_and_seed() -> None:
    policy = _policy()
    first = policy.decide("repository observation", seed=41, rng_counter=7)
    second = policy.decide("repository observation", seed=41, rng_counter=7)
    assert first == second
    assert first.policy_epoch_id == policy.policy_epoch_id
    assert first.behavior_log_probability == pytest.approx(math.log(first.probability))
    assert policy.weights_hash == _policy().weights_hash
    with pytest.raises(ValueError, match="does not match"):
        first.model_copy(update={"behavior_log_probability": 0.0}).model_validate(
            {**first.model_dump(), "behavior_log_probability": 0.0}
        )


def test_branch_credit_is_group_relative_localized_and_explicitly_scoped() -> None:
    policy = _policy()
    evidence = assign_branch_relative_credit(
        branch_group_id="group-1", branch_point_id="point-1", outcomes=_outcomes(policy)
    )
    credits = {item.branch_id: item for item in evidence.credits}
    assert credits["branch-correct"].weighted_advantage > 0.0
    assert credits["branch-reject"].weighted_advantage < 0.0
    assert evidence.divergence_action_index == 3
    assert evidence.preferences
    assert all(
        "causal" in assumption
        or "BranchPoint" in assumption
        or "reward" in assumption
        or "environment" in assumption
        or "observational" in assumption
        for assumption in evidence.assumptions
    )
    with pytest.raises(ValueError, match="one policy epoch"):
        assign_branch_relative_credit(
            branch_group_id="group-1",
            branch_point_id="point-1",
            outcomes=(
                _outcomes(policy)[0],
                _outcomes(policy)[1].model_copy(update={"policy_epoch_id": "policy-2"}),
            ),
        )
    with pytest.raises(ValueError, match="same component set"):
        assign_branch_relative_credit(
            branch_group_id="group-1",
            branch_point_id="point-1",
            outcomes=(
                _outcomes(policy)[0],
                _outcomes(policy)[1].model_copy(
                    update={"reward_components": {"visible_tests": 1.0}}
                ),
            ),
        )
    with pytest.raises(ValueError, match="one first divergent"):
        assign_branch_relative_credit(
            branch_group_id="group-1",
            branch_point_id="point-1",
            outcomes=(
                _outcomes(policy)[0],
                _outcomes(policy)[1].model_copy(update={"first_divergent_action_index": 4}),
            ),
        )
    localized = assign_branch_relative_credit(
        branch_group_id="group-1",
        branch_point_id="point-1",
        outcomes=(
            _outcomes(policy)[0].model_copy(
                update={"suffix_action_count": 1, "process_score": None}
            ),
            _outcomes(policy)[1].model_copy(
                update={"suffix_action_count": 3, "process_score": None}
            ),
        ),
        decay=0.5,
    )
    localized_by_branch = {item.branch_id: item for item in localized.credits}
    assert localized_by_branch["branch-naive"].decision_local_weight == 1.0
    assert localized_by_branch["branch-correct"].decision_local_weight == 0.25
    with_observation = assign_branch_relative_credit(
        branch_group_id="group-1",
        branch_point_id="point-1",
        outcomes=(
            _outcomes(policy)[0],
            _outcomes(policy)[1],
            _outcomes(policy)[2].model_copy(update={"intervention": "observational"}),
        ),
    )
    observed_credit = next(
        item for item in with_observation.credits if item.branch_id == "branch-reject"
    )
    assert observed_credit.excluded_reason is not None
    assert observed_credit.weighted_advantage == 0.0
    assert all(
        preference.chosen_branch_id != "branch-reject"
        and preference.rejected_branch_id != "branch-reject"
        for preference in with_observation.preferences
    )


@pytest.mark.parametrize(
    "algorithm",
    [
        TrainingAlgorithm.SUCCESSFUL_BRANCH_DISTILLATION,
        TrainingAlgorithm.GROUP_RELATIVE,
        TrainingAlgorithm.BRANCH_RELATIVE,
    ],
)
def test_reference_training_increases_successful_branch_probability(
    algorithm: TrainingAlgorithm,
) -> None:
    policy = _policy()
    result = ReferenceTrainer(learning_rate=0.6).train(
        base=policy,
        samples=_samples(policy),
        algorithm=algorithm,
        candidate_policy_epoch_id=f"candidate-{algorithm.value}",
        seed=73,
        steps=12,
    )
    before = policy.probabilities()[1]
    after = result.candidate.probabilities()[1]
    assert after > before
    assert result.data_hash
    assert result.checkpoint_hash
    assert all(metric.policy_kl >= 0.0 for metric in result.metrics)


def test_pairwise_training_and_staleness_rejection() -> None:
    policy = _policy()
    pair = ReferenceTrainingSample(
        sample_id="pair-1",
        trajectory_id="trajectory-correct",
        branch_point_id="point-1",
        branch_group_id="group-1",
        policy_epoch_id=policy.policy_epoch_id,
        action="merchandise-only-credit",
        behavior_log_probability=policy.log_probability("merchandise-only-credit"),
        advantage=1.0,
        token_weight=1.0,
        reward_margin=2.0,
        chosen_action="merchandise-only-credit",
        rejected_action="naive-total-credit",
    )
    result = ReferenceTrainer(learning_rate=0.5).train(
        base=policy,
        samples=(pair,),
        algorithm=TrainingAlgorithm.PAIRWISE_PREFERENCE,
        candidate_policy_epoch_id="candidate-pair",
        seed=79,
        steps=8,
    )
    assert (
        result.candidate.logits[1] - result.candidate.logits[0]
        > policy.logits[1] - policy.logits[0]
    )
    stale = pair.model_copy(update={"staleness_updates": 99})
    with pytest.raises(ValueError, match="all training samples"):
        ReferenceTrainer(maximum_staleness_updates=4).train(
            base=policy,
            samples=(stale,),
            algorithm=TrainingAlgorithm.PAIRWISE_PREFERENCE,
            candidate_policy_epoch_id="candidate-stale",
            seed=83,
        )


def test_reference_trainer_rejects_silent_policy_mixing() -> None:
    policy = _policy()
    mixed = _samples(policy)[0].model_copy(update={"policy_epoch_id": "other-policy"})
    with pytest.raises(ValueError, match="mixed behavior policy"):
        ReferenceTrainer().train(
            base=policy,
            samples=(mixed,),
            algorithm=TrainingAlgorithm.BRANCH_RELATIVE,
            candidate_policy_epoch_id="candidate-invalid",
            seed=89,
        )


@pytest.mark.parametrize(
    ("advantage", "log_ratio", "expected_surrogate"),
    [
        (1.0, math.log(2.0), 1.2),
        (-1.0, math.log(0.5), -0.8),
    ],
)
def test_reference_ppo_has_zero_gradient_on_clipped_plateau(
    advantage: float, log_ratio: float, expected_surrogate: float
) -> None:
    surrogate, signal = ReferenceTrainer()._ppo_surrogate(
        log_ratio=log_ratio,
        advantage=advantage,
    )
    assert surrogate == pytest.approx(expected_surrogate)
    assert signal == 0.0


def test_reference_objectives_apply_temperature_and_ignore_masked_samples() -> None:
    policy = DeterministicPolicy(
        policy_epoch_id="policy-temperature",
        actions=("chosen", "rejected"),
        logits=(0.0, 0.0),
        temperature=2.0,
    )
    pair = ReferenceTrainingSample(
        sample_id="effective-pair",
        trajectory_id="trajectory",
        branch_point_id="point",
        branch_group_id="group",
        policy_epoch_id=policy.policy_epoch_id,
        action="chosen",
        behavior_log_probability=math.log(0.5),
        advantage=1.0,
        token_weight=1.0,
        reward_margin=1.0,
        chosen_action="chosen",
        rejected_action="rejected",
    )
    masked = pair.model_copy(update={"sample_id": "masked-pair", "token_weight": 0.0})
    trainer = ReferenceTrainer(learning_rate=0.4, kl_coefficient=0.0)
    alone = trainer.train(
        base=policy,
        samples=(pair,),
        algorithm=TrainingAlgorithm.PAIRWISE_PREFERENCE,
        candidate_policy_epoch_id="candidate-alone",
        seed=93,
        steps=1,
    )
    together = trainer.train(
        base=policy,
        samples=(pair, masked),
        algorithm=TrainingAlgorithm.PAIRWISE_PREFERENCE,
        candidate_policy_epoch_id="candidate-together",
        seed=93,
        steps=1,
    )
    assert alone.candidate.logits == pytest.approx((0.1, -0.1))
    assert together.candidate.logits == alone.candidate.logits
    assert together.metrics[0].accepted_samples == 1
    assert together.metrics[0].rejected_samples == 1


def test_group_relative_does_not_apply_branch_locality_twice() -> None:
    policy = DeterministicPolicy(
        policy_epoch_id="policy-group-weight",
        actions=("chosen", "other"),
        logits=(0.0, 0.0),
    )
    sample = ReferenceTrainingSample(
        sample_id="group-weight",
        trajectory_id="trajectory",
        branch_point_id="point",
        branch_group_id="group",
        policy_epoch_id=policy.policy_epoch_id,
        action="chosen",
        behavior_log_probability=math.log(0.5),
        advantage=1.0,
        token_weight=0.0,
    )
    result = ReferenceTrainer(learning_rate=0.4, kl_coefficient=0.0).train(
        base=policy,
        samples=(sample,),
        algorithm=TrainingAlgorithm.GROUP_RELATIVE,
        candidate_policy_epoch_id="candidate-group-weight",
        seed=94,
        steps=1,
    )
    assert result.candidate.probabilities()[0] > policy.probabilities()[0]
    with pytest.raises(ValueError, match="all training samples"):
        ReferenceTrainer().train(
            base=policy,
            samples=(sample,),
            algorithm=TrainingAlgorithm.BRANCH_RELATIVE,
            candidate_policy_epoch_id="candidate-branch-weight",
            seed=94,
            steps=1,
        )


def test_checkpoint_identity_binds_algorithm_and_hyperparameters() -> None:
    policy = DeterministicPolicy(
        policy_epoch_id="policy-checkpoint",
        actions=("chosen", "other"),
        logits=(0.0, 0.0),
    )
    positive = ReferenceTrainingSample(
        sample_id="checkpoint-positive",
        trajectory_id="trajectory",
        branch_point_id="point",
        branch_group_id="group",
        policy_epoch_id=policy.policy_epoch_id,
        action="chosen",
        behavior_log_probability=math.log(0.5),
        advantage=1.0,
        token_weight=1.0,
    )
    negative = positive.model_copy(
        update={
            "sample_id": "checkpoint-negative",
            "trajectory_id": "trajectory-negative",
            "advantage": -1.0,
        }
    )
    group = ReferenceTrainer(learning_rate=0.2, kl_coefficient=0.0).train(
        base=policy,
        samples=(positive, negative),
        algorithm=TrainingAlgorithm.GROUP_RELATIVE,
        candidate_policy_epoch_id="candidate-checkpoint",
        seed=95,
        steps=1,
    )
    changed_rate = ReferenceTrainer(learning_rate=0.3, kl_coefficient=0.0).train(
        base=policy,
        samples=(positive, negative),
        algorithm=TrainingAlgorithm.GROUP_RELATIVE,
        candidate_policy_epoch_id="candidate-checkpoint",
        seed=95,
        steps=1,
    )
    branch = ReferenceTrainer(learning_rate=0.2, kl_coefficient=0.0).train(
        base=policy,
        samples=(positive, negative),
        algorithm=TrainingAlgorithm.BRANCH_RELATIVE,
        candidate_policy_epoch_id="candidate-checkpoint",
        seed=95,
        steps=1,
    )
    assert group.candidate == changed_rate.candidate == branch.candidate
    assert group.checkpoint_hash != changed_rate.checkpoint_hash
    assert group.checkpoint_hash != branch.checkpoint_hash


def test_training_sample_rejects_impossible_log_probability_and_preference() -> None:
    policy = _policy()
    raw = _samples(policy)[0].model_dump()
    with pytest.raises(ValueError, match="non-positive"):
        ReferenceTrainingSample.model_validate(
            {**raw, "behavior_log_probability": 0.1}, strict=True
        )
    with pytest.raises(ValueError, match="distinct"):
        ReferenceTrainingSample.model_validate(
            {**raw, "chosen_action": raw["action"], "rejected_action": raw["action"]},
            strict=True,
        )
    forged = _samples(policy)[0].model_copy(update={"behavior_log_probability": 0.0})
    with pytest.raises(ValueError, match="immutable policy epoch"):
        ReferenceTrainer().train(
            base=policy,
            samples=(forged,),
            algorithm=TrainingAlgorithm.GROUP_RELATIVE,
            candidate_policy_epoch_id="candidate-forged-probability",
            seed=97,
            steps=1,
        )
