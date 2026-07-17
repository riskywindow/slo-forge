from __future__ import annotations

from pathlib import Path

import pytest

from sloforge.helix.credit import (
    BranchOutcome,
    BranchRelativeCredit,
    assign_branch_relative_credit,
)
from sloforge.helix.datasets import build_reference_training_batch
from sloforge.helix.effects import EffectClass
from sloforge.helix.environments import EnvironmentBackend
from sloforge.helix.environments.models import content_digest
from sloforge.helix.policy import DeterministicPolicy
from sloforge.helix.rewards import DeterministicRewardWorker, HiddenCase, RewardRun
from sloforge.helix.rollouts import (
    ActionMutation,
    CandidateAction,
    ReferenceRolloutWorker,
    ReferenceTrajectory,
)
from sloforge.helix.trainers import ReferenceTrainer, TrainingAlgorithm
from sloforge.helix.trainers.optional import PeftTrainingExample, PeftTrainingRequest


def _siblings(
    tmp_path: Path,
) -> tuple[tuple[ReferenceTrajectory, ...], tuple[RewardRun, ...], BranchRelativeCredit]:
    source = tmp_path / "source"
    source.mkdir()
    original = b"print('base')\n"
    (source / "answer.py").write_bytes(original)
    backend = EnvironmentBackend(tmp_path / "store", tenant_id="tenant")
    capsule = backend.capture(source, seed=1)
    policy = DeterministicPolicy(
        policy_epoch_id="policy-0", actions=("bad", "good"), logits=(1.0, 0.0)
    )
    candidates = tuple(
        CandidateAction(
            action=name,
            tool_id="edit",
            effect_class=EffectClass.IDEMPOTENT_WRITE,
            mutations=(
                ActionMutation(
                    path="answer.py",
                    content=f"print('{output}')\n",
                    expected_before_hash=content_digest(original),
                ),
            ),
        )
        for name, output in (("bad", "bad"), ("good", "good"))
    )
    worker = ReferenceRolloutWorker(tenant_id="tenant")
    trajectories = tuple(
        worker.run(
            branch=backend.fork(capsule, branch_id=name),
            initial_environment_capsule_id=capsule.capsule_id,
            branch_group_id="group",
            branch_point_id="point",
            branch_point_hash="a" * 64,
            source_model_capsule_id="model-state",
            state_reuse_report_hash=("b" if name == "bad" else "c") * 64,
            policy=policy,
            observation="print good",
            candidates=candidates,
            seed=3,
            forced_action=name,
        )
        for name in ("bad", "good")
    )
    reward_worker = DeterministicRewardWorker()
    rewards = tuple(
        reward_worker.verify(
            reward_id=f"reward-{trajectory.branch_id}",
            tenant_id=trajectory.tenant_id,
            trajectory_id=trajectory.trajectory_id,
            policy_epoch_id=trajectory.policy_epoch_id,
            source=Path(backend.branch(trajectory.branch_id).workspace),
            evidence_directory=tmp_path / f"reward-{trajectory.branch_id}",
            commands=(),
            hidden_cases=(
                HiddenCase(
                    case_id="hidden-expected-output",
                    runner="answer.py",
                    arguments=(),
                    expected_stdout="good",
                ),
            ),
            seed=7,
        )
        for trajectory in trajectories
    )
    # One immutable black-box verifier is applied independently to both branches.
    outcomes = tuple(
        BranchOutcome(
            branch_id=trajectory.branch_id,
            trajectory_id=trajectory.trajectory_id,
            policy_epoch_id=trajectory.policy_epoch_id,
            action=trajectory.actions[0].action,
            behavior_log_probability=trajectory.actions[0].behavior_log_probability,
            reward_components={
                component.component_id: component.score for component in reward.components
            },
            first_divergent_action_index=0,
            suffix_action_count=1,
            process_score=1.0 if trajectory.branch_id == "good" else -1.0,
            intervention="controlled_action",
        )
        for trajectory, reward in zip(trajectories, rewards, strict=True)
    )
    return (
        trajectories,
        rewards,
        assign_branch_relative_credit(
            branch_group_id="group", branch_point_id="point", outcomes=outcomes
        ),
    )


def test_batch_preserves_complete_lineage_and_trains(tmp_path: Path) -> None:
    trajectories, rewards, credit = _siblings(tmp_path)
    manifest = build_reference_training_batch(
        trajectories=trajectories,
        rewards=rewards,
        credit=credit,
        algorithm=TrainingAlgorithm.BRANCH_RELATIVE,
        learner_policy_epoch_id="policy-0",
        staleness_updates={item.trajectory_id: 0 for item in trajectories},
        maximum_staleness_updates=2,
        holdout_trajectory_ids=("holdout-1",),
        creation_code_version="test-commit",
        seed=11,
    )
    assert len(manifest.samples) == 2
    assert manifest.tenant_id == "tenant"
    assert manifest.training_sample_ids == tuple(
        item.sample.sample_id for item in manifest.samples if item.sample.eligible
    )
    assert all(
        item.behavior_log_probability_source == "rollout_worker" for item in manifest.samples
    )
    result = ReferenceTrainer().train(
        base=DeterministicPolicy(
            policy_epoch_id="policy-0", actions=("bad", "good"), logits=(1.0, 0.0)
        ),
        samples=manifest.trainer_samples(),
        algorithm=manifest.algorithm,
        candidate_policy_epoch_id="policy-1",
        seed=11,
    )
    assert result.data_hash == ReferenceTrainer._data_hash(manifest.trainer_samples())


def test_batch_rejects_missing_reward_and_implicit_staleness(tmp_path: Path) -> None:
    trajectories, rewards, credit = _siblings(tmp_path)
    with pytest.raises(ValueError, match="no reward evidence"):
        build_reference_training_batch(
            trajectories=trajectories,
            rewards=rewards[:1],
            credit=credit,
            algorithm=TrainingAlgorithm.BRANCH_RELATIVE,
            learner_policy_epoch_id="policy-0",
            staleness_updates={item.trajectory_id: 0 for item in trajectories},
            maximum_staleness_updates=2,
            holdout_trajectory_ids=(),
            creation_code_version="test-commit",
            seed=11,
        )
    with pytest.raises(ValueError, match="explicit staleness"):
        build_reference_training_batch(
            trajectories=trajectories,
            rewards=rewards,
            credit=credit,
            algorithm=TrainingAlgorithm.BRANCH_RELATIVE,
            learner_policy_epoch_id="policy-0",
            staleness_updates={},
            maximum_staleness_updates=2,
            holdout_trajectory_ids=(),
            creation_code_version="test-commit",
            seed=11,
        )


def test_batch_excludes_stale_samples_from_the_training_split(tmp_path: Path) -> None:
    trajectories, rewards, credit = _siblings(tmp_path)
    manifest = build_reference_training_batch(
        trajectories=trajectories,
        rewards=rewards,
        credit=credit,
        algorithm=TrainingAlgorithm.GROUP_RELATIVE,
        learner_policy_epoch_id="policy-0",
        staleness_updates={
            trajectories[0].trajectory_id: 9,
            trajectories[1].trajectory_id: 0,
        },
        maximum_staleness_updates=2,
        holdout_trajectory_ids=(),
        creation_code_version="test-commit",
        seed=13,
    )
    assert len(manifest.samples) == 2
    assert len(manifest.training_sample_ids) == 1
    assert len(manifest.trainer_samples()) == 1
    assert trajectories[0].trajectory_id in manifest.excluded_trajectory_ids
    assert all(sample.eligible for sample in manifest.trainer_samples())


def test_pairwise_batch_emits_one_sample_with_both_sources(tmp_path: Path) -> None:
    trajectories, rewards, credit = _siblings(tmp_path)
    manifest = build_reference_training_batch(
        trajectories=trajectories,
        rewards=rewards,
        credit=credit,
        algorithm=TrainingAlgorithm.PAIRWISE_PREFERENCE,
        learner_policy_epoch_id="policy-0",
        staleness_updates={item.trajectory_id: 0 for item in trajectories},
        maximum_staleness_updates=2,
        holdout_trajectory_ids=(),
        creation_code_version="test-commit",
        seed=17,
    )
    assert len(credit.preferences) == 1
    assert len(manifest.samples) == 1
    provenance = manifest.samples[0]
    assert provenance.comparison_branch_id is not None
    assert provenance.comparison_trajectory_hash in {item.trajectory_id for item in trajectories}
    assert provenance.comparison_reward_hash is not None
    assert provenance.sample.chosen_action != provenance.sample.rejected_action


def test_batch_rejects_credit_that_disagrees_with_raw_probability_or_reward(
    tmp_path: Path,
) -> None:
    trajectories, rewards, credit = _siblings(tmp_path)
    altered_credit = credit.model_copy(
        update={
            "credits": (
                credit.credits[0].model_copy(
                    update={
                        "behavior_log_probability": credit.credits[0].behavior_log_probability
                        - 0.25
                    }
                ),
                *credit.credits[1:],
            )
        }
    )
    with pytest.raises(ValueError, match="behavior probability"):
        build_reference_training_batch(
            trajectories=trajectories,
            rewards=rewards,
            credit=altered_credit,
            algorithm=TrainingAlgorithm.BRANCH_RELATIVE,
            learner_policy_epoch_id="policy-0",
            staleness_updates={item.trajectory_id: 0 for item in trajectories},
            maximum_staleness_updates=2,
            holdout_trajectory_ids=(),
            creation_code_version="test-commit",
            seed=19,
        )
    component_id = next(iter(credit.credits[0].reward_components))
    altered_component_credit = credit.model_copy(
        update={
            "credits": (
                credit.credits[0].model_copy(update={"reward_components": {component_id: 123.0}}),
                *credit.credits[1:],
            )
        }
    )
    with pytest.raises(ValueError, match="credit components"):
        build_reference_training_batch(
            trajectories=trajectories,
            rewards=rewards,
            credit=altered_component_credit,
            algorithm=TrainingAlgorithm.BRANCH_RELATIVE,
            learner_policy_epoch_id="policy-0",
            staleness_updates={item.trajectory_id: 0 for item in trajectories},
            maximum_staleness_updates=2,
            holdout_trajectory_ids=(),
            creation_code_version="test-commit",
            seed=19,
        )
    altered_rewards = (
        rewards[0].model_copy(update={"total_score": rewards[0].total_score + 0.5}),
        *rewards[1:],
    )
    with pytest.raises(ValueError, match="independently verified reward"):
        build_reference_training_batch(
            trajectories=trajectories,
            rewards=altered_rewards,
            credit=credit,
            algorithm=TrainingAlgorithm.BRANCH_RELATIVE,
            learner_policy_epoch_id="policy-0",
            staleness_updates={item.trajectory_id: 0 for item in trajectories},
            maximum_staleness_updates=2,
            holdout_trajectory_ids=(),
            creation_code_version="test-commit",
            seed=19,
        )


@pytest.mark.parametrize("evidence_kind", ["trajectory", "reward"])
def test_batch_rejects_cross_tenant_composition(tmp_path: Path, evidence_kind: str) -> None:
    trajectories, rewards, credit = _siblings(tmp_path)
    if evidence_kind == "trajectory":
        trajectories = (
            trajectories[0].model_copy(update={"tenant_id": "other-tenant"}),
            *trajectories[1:],
        )
    else:
        rewards = (
            rewards[0].model_copy(update={"tenant_id": "other-tenant"}),
            *rewards[1:],
        )
    with pytest.raises(ValueError, match=r"cannot mix.*tenants"):
        build_reference_training_batch(
            trajectories=trajectories,
            rewards=rewards,
            credit=credit,
            algorithm=TrainingAlgorithm.BRANCH_RELATIVE,
            learner_policy_epoch_id="policy-0",
            staleness_updates={item.trajectory_id: 0 for item in trajectories},
            maximum_staleness_updates=2,
            holdout_trajectory_ids=(),
            creation_code_version="test-commit",
            seed=23,
        )


def test_peft_request_cannot_train_excluded_samples_or_change_batch_algorithm(
    tmp_path: Path,
) -> None:
    trajectories, rewards, credit = _siblings(tmp_path)
    batch = build_reference_training_batch(
        trajectories=trajectories,
        rewards=rewards,
        credit=credit,
        algorithm=TrainingAlgorithm.SUCCESSFUL_BRANCH_DISTILLATION,
        learner_policy_epoch_id="policy-0",
        staleness_updates={item.trajectory_id: 0 for item in trajectories},
        maximum_staleness_updates=2,
        holdout_trajectory_ids=(),
        creation_code_version="test-commit",
        seed=29,
    )
    all_examples = tuple(
        PeftTrainingExample(
            sample_id=item.sample.sample_id,
            prompt="prompt",
            completion=item.sample.action,
        )
        for item in batch.samples
    )
    assert len(all_examples) > len(batch.training_sample_ids)
    with pytest.raises(ValueError, match="each training sample exactly once"):
        PeftTrainingRequest(
            model_directory=(tmp_path / "source").resolve(),
            base_policy_epoch_id="policy-0",
            candidate_policy_epoch_id="policy-1",
            batch=batch,
            examples=all_examples,
            seed=29,
        )
    mismatched_batch = build_reference_training_batch(
        trajectories=trajectories,
        rewards=rewards,
        credit=credit,
        algorithm=TrainingAlgorithm.BRANCH_RELATIVE,
        learner_policy_epoch_id="policy-0",
        staleness_updates={item.trajectory_id: 0 for item in trajectories},
        maximum_staleness_updates=2,
        holdout_trajectory_ids=(),
        creation_code_version="test-commit",
        seed=29,
    )
    mismatched_examples = tuple(
        PeftTrainingExample(
            sample_id=sample_id,
            prompt="prompt",
            completion="completion",
        )
        for sample_id in mismatched_batch.training_sample_ids
    )
    with pytest.raises(ValueError, match="algorithm differs"):
        PeftTrainingRequest(
            model_directory=(tmp_path / "source").resolve(),
            base_policy_epoch_id="policy-0",
            candidate_policy_epoch_id="policy-1",
            batch=mismatched_batch,
            examples=mismatched_examples,
            seed=29,
        )
