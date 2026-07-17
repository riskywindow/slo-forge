"""Training manifest bridge for exact-state reference trajectories."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from hashlib import sha256
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from sloforge.helix.credit import BranchCredit, BranchRelativeCredit
from sloforge.helix.rewards import RewardRun
from sloforge.helix.rollouts import ActionRecord, ReferenceTrajectory
from sloforge.helix.trainers.reference import ReferenceTrainingSample, TrainingAlgorithm


class _DatasetModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class BatchSampleProvenance(_DatasetModel):
    sample: ReferenceTrainingSample
    tenant_id: Annotated[str, Field(min_length=1, max_length=160)]
    branch_id: Annotated[str, Field(min_length=1, max_length=160)]
    trajectory_hash: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    reward_id: Annotated[str, Field(min_length=1, max_length=160)]
    reward_hash: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    comparison_branch_id: Annotated[str, Field(min_length=1, max_length=160)] | None = None
    comparison_trajectory_hash: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")] | None = None
    comparison_reward_id: Annotated[str, Field(min_length=1, max_length=160)] | None = None
    comparison_reward_hash: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")] | None = None
    credit_hash: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    environment_capsule_id: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    source_model_capsule_id: Annotated[str, Field(min_length=1, max_length=256)]
    state_reuse_report_hash: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    behavior_log_probability_source: Literal["rollout_worker"] = "rollout_worker"
    staleness_disposition: Literal["accepted", "rejected"]

    @model_validator(mode="after")
    def consistent_eligibility(self) -> Self:
        if self.sample.eligible != (self.staleness_disposition == "accepted"):
            raise ValueError("sample eligibility disagrees with staleness disposition")
        if self.trajectory_hash != self.sample.trajectory_id:
            raise ValueError("sample trajectory hash disagrees with its source trajectory")
        comparison = (
            self.comparison_branch_id,
            self.comparison_trajectory_hash,
            self.comparison_reward_id,
            self.comparison_reward_hash,
        )
        if any(item is not None for item in comparison) != all(
            item is not None for item in comparison
        ):
            raise ValueError("pairwise comparison provenance must be complete")
        paired = self.sample.chosen_action is not None
        if paired != all(item is not None for item in comparison):
            raise ValueError("pairwise actions and comparison provenance must agree")
        return self


class ReferenceTrainingBatchManifest(_DatasetModel):
    schema_version: Literal["sloforge.helix.reference-training-batch/v1"] = (
        "sloforge.helix.reference-training-batch/v1"
    )
    batch_id: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    tenant_id: Annotated[str, Field(min_length=1, max_length=160)]
    branch_group_id: Annotated[str, Field(min_length=1, max_length=160)]
    branch_point_id: Annotated[str, Field(min_length=1, max_length=160)]
    behavior_policy_epoch_id: Annotated[str, Field(min_length=1, max_length=160)]
    learner_policy_epoch_id: Annotated[str, Field(min_length=1, max_length=160)]
    policy_mode: Literal["strict"] = "strict"
    algorithm: TrainingAlgorithm
    samples: Annotated[tuple[BatchSampleProvenance, ...], Field(min_length=1, max_length=65_536)]
    training_sample_ids: Annotated[tuple[str, ...], Field(min_length=1, max_length=65_536)]
    holdout_trajectory_ids: Annotated[tuple[str, ...], Field(max_length=65_536)]
    excluded_trajectory_ids: Annotated[tuple[str, ...], Field(max_length=65_536)]
    credit_hash: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    data_hash: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    creation_code_version: Annotated[str, Field(min_length=1, max_length=160)]
    seed: Annotated[int, Field(ge=0, le=2**64 - 1)]

    @model_validator(mode="after")
    def validate_complete_lineage(self) -> Self:
        ids = [item.sample.sample_id for item in self.samples]
        if len(ids) != len(set(ids)):
            raise ValueError("training batch sample IDs must be unique")
        eligible_ids = tuple(item.sample.sample_id for item in self.samples if item.sample.eligible)
        if eligible_ids != self.training_sample_ids:
            raise ValueError(
                "training split must enumerate only eligible samples in manifest order"
            )
        if len(self.holdout_trajectory_ids) != len(set(self.holdout_trajectory_ids)):
            raise ValueError("holdout trajectory identifiers must be unique")
        if len(self.excluded_trajectory_ids) != len(set(self.excluded_trajectory_ids)):
            raise ValueError("excluded trajectory identifiers must be unique")
        trajectories = {item.trajectory_hash for item in self.samples}
        trajectories.update(
            item.comparison_trajectory_hash
            for item in self.samples
            if item.comparison_trajectory_hash is not None
        )
        if trajectories & set(self.holdout_trajectory_ids):
            raise ValueError("holdout trajectories cannot enter the training split")
        accepted_trajectories = {
            item.trajectory_hash for item in self.samples if item.sample.eligible
        }
        accepted_trajectories.update(
            item.comparison_trajectory_hash
            for item in self.samples
            if item.sample.eligible and item.comparison_trajectory_hash is not None
        )
        if accepted_trajectories & set(self.excluded_trajectory_ids):
            raise ValueError("accepted trajectories cannot be marked excluded")
        if any(
            item.sample.policy_epoch_id != self.behavior_policy_epoch_id for item in self.samples
        ):
            raise ValueError("strict training batch silently mixed behavior policy epochs")
        if any(
            item.sample.branch_group_id != self.branch_group_id
            or item.sample.branch_point_id != self.branch_point_id
            for item in self.samples
        ):
            raise ValueError("training sample escaped the manifest branch scope")
        if any(item.tenant_id != self.tenant_id for item in self.samples):
            raise ValueError("sample tenant disagrees with the manifest tenant")
        if any(item.credit_hash != self.credit_hash for item in self.samples):
            raise ValueError("sample credit hash disagrees with the manifest")
        expected_data_hash = _canonical_hash(
            [item.model_dump(mode="json") for item in self.samples]
        )
        if expected_data_hash != self.data_hash:
            raise ValueError("training data hash does not match sample provenance")
        expected_batch_id = _canonical_hash(self.model_dump(mode="json", exclude={"batch_id"}))
        if expected_batch_id != self.batch_id:
            raise ValueError("training batch content identity mismatch")
        return self

    def trainer_samples(self) -> tuple[ReferenceTrainingSample, ...]:
        by_id = {item.sample.sample_id: item.sample for item in self.samples}
        return tuple(by_id[sample_id] for sample_id in self.training_sample_ids)


def _canonical_hash(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return sha256(payload.encode("utf-8")).hexdigest()


def _reward_hash(reward: RewardRun) -> str:
    return _canonical_hash(reward.model_dump(mode="json"))


def build_reference_training_batch(
    *,
    trajectories: tuple[ReferenceTrajectory, ...],
    rewards: tuple[RewardRun, ...],
    credit: BranchRelativeCredit,
    algorithm: TrainingAlgorithm,
    learner_policy_epoch_id: str,
    staleness_updates: Mapping[str, int],
    maximum_staleness_updates: int,
    holdout_trajectory_ids: tuple[str, ...],
    creation_code_version: str,
    seed: int,
) -> ReferenceTrainingBatchManifest:
    """Fail closed unless every training sample has trajectory/reward/credit/state lineage."""

    if not trajectories:
        raise ValueError("training batch requires trajectories")
    if maximum_staleness_updates < 0:
        raise ValueError("maximum staleness must be non-negative")
    if seed < 0 or seed > 2**64 - 1:
        raise ValueError("seed must fit an unsigned 64-bit value")
    if len(holdout_trajectory_ids) != len(set(holdout_trajectory_ids)):
        raise ValueError("holdout trajectory identifiers must be unique")
    trajectory_by_id = {item.trajectory_id: item for item in trajectories}
    if len(trajectory_by_id) != len(trajectories):
        raise ValueError("training input contains duplicate trajectories")
    reward_by_trajectory = {item.trajectory_id: item for item in rewards}
    if len(reward_by_trajectory) != len(rewards):
        raise ValueError("training input contains duplicate reward submissions")
    reward_ids = [item.reward_id for item in rewards]
    if len(reward_ids) != len(set(reward_ids)):
        raise ValueError("training input contains duplicate reward identifiers")
    credited_trajectory_ids = [item.trajectory_id for item in credit.credits]
    if len(credited_trajectory_ids) != len(set(credited_trajectory_ids)):
        raise ValueError("credit contains duplicate trajectory identifiers")
    credited_branch_ids = [item.branch_id for item in credit.credits]
    if len(credited_branch_ids) != len(set(credited_branch_ids)):
        raise ValueError("credit contains duplicate branch identifiers")
    required_ids = set(credited_trajectory_ids)
    if required_ids - set(trajectory_by_id):
        raise ValueError("credit references a trajectory absent from the batch")
    if required_ids - set(reward_by_trajectory):
        raise ValueError("credit references a trajectory with no reward evidence")
    if set(trajectory_by_id) - required_ids:
        raise ValueError("training input contains a trajectory with no credit assignment")
    if set(reward_by_trajectory) - required_ids:
        raise ValueError("training input contains reward evidence outside the credited sibling set")
    if set(staleness_updates) != required_ids:
        raise ValueError("every credited trajectory requires an explicit staleness report")
    if required_ids & set(holdout_trajectory_ids):
        raise ValueError("credited training trajectory was also designated as holdout")
    trajectory_tenants = {item.tenant_id for item in trajectories}
    reward_tenants = {item.tenant_id for item in rewards}
    if len(trajectory_tenants) != 1 or reward_tenants != trajectory_tenants:
        raise ValueError("training batches cannot mix trajectory or reward tenants")
    tenant_id = next(iter(trajectory_tenants))
    branch_group_ids = {item.branch_group_id for item in trajectories}
    branch_point_ids = {item.branch_point_id for item in trajectories}
    policy_ids = {item.policy_epoch_id for item in trajectories}
    if branch_group_ids != {credit.branch_group_id} or branch_point_ids != {credit.branch_point_id}:
        raise ValueError("trajectory siblings do not match the credit branch group")
    if policy_ids != {credit.policy_epoch_id}:
        raise ValueError("strict branch group has mixed behavior policy epochs")
    credit_hash = _canonical_hash(credit.model_dump(mode="json"))
    credit_by_branch = {item.branch_id: item for item in credit.credits}
    samples: list[BatchSampleProvenance] = []
    accepted_trajectory_ids: set[str] = set()

    def source_evidence(
        branch_id: str,
    ) -> tuple[BranchCredit, ReferenceTrajectory, RewardRun, ActionRecord, int, bool]:
        try:
            branch_credit = credit_by_branch[branch_id]
        except KeyError as exc:
            raise ValueError("preference references a branch absent from credit evidence") from exc
        trajectory = trajectory_by_id[branch_credit.trajectory_id]
        reward = reward_by_trajectory[trajectory.trajectory_id]
        action = trajectory.actions[0]
        if trajectory.branch_id != branch_credit.branch_id:
            raise ValueError("credit branch differs from trajectory branch provenance")
        if action.action != branch_credit.action:
            raise ValueError("credit action differs from trajectory action provenance")
        if not math.isclose(
            action.behavior_log_probability,
            branch_credit.behavior_log_probability,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError("credit behavior probability differs from trajectory provenance")
        if reward.trajectory_id != trajectory.trajectory_id:
            raise ValueError("reward trajectory differs from rollout provenance")
        if reward.policy_epoch_id != trajectory.policy_epoch_id:
            raise ValueError("reward policy epoch differs from behavior policy provenance")
        reward_components = {item.component_id: item.score for item in reward.components}
        if len(reward_components) != len(reward.components):
            raise ValueError("reward evidence contains duplicate component identifiers")
        if set(reward_components) != set(branch_credit.reward_components) or any(
            not math.isclose(
                reward_components[component_id],
                component_score,
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
            for component_id, component_score in branch_credit.reward_components.items()
        ):
            raise ValueError("credit components differ from independently verified reward evidence")
        if not math.isclose(
            reward.total_score,
            branch_credit.total_reward,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError("credit reward differs from independently verified reward evidence")
        lag = staleness_updates[trajectory.trajectory_id]
        if lag < 0:
            raise ValueError("staleness update distance cannot be negative")
        eligible = branch_credit.excluded_reason is None and lag <= maximum_staleness_updates
        return branch_credit, trajectory, reward, action, lag, eligible

    if algorithm is TrainingAlgorithm.PAIRWISE_PREFERENCE:
        seen_preferences: set[tuple[str, str]] = set()
        for preference in credit.preferences:
            preference_key = (preference.chosen_branch_id, preference.rejected_branch_id)
            if preference_key in seen_preferences:
                raise ValueError("credit contains a duplicate pairwise preference")
            seen_preferences.add(preference_key)
            (
                chosen_credit,
                chosen_trajectory,
                chosen_reward,
                chosen_action,
                chosen_lag,
                chosen_eligible,
            ) = source_evidence(preference.chosen_branch_id)
            (
                _rejected_credit,
                rejected_trajectory,
                rejected_reward,
                rejected_action,
                rejected_lag,
                rejected_eligible,
            ) = source_evidence(preference.rejected_branch_id)
            eligible = bool(chosen_eligible and rejected_eligible and preference.confidence > 0.0)
            if eligible:
                accepted_trajectory_ids.update(
                    (chosen_trajectory.trajectory_id, rejected_trajectory.trajectory_id)
                )
            sample_id = _canonical_hash(
                {
                    "chosen_trajectory_id": chosen_trajectory.trajectory_id,
                    "rejected_trajectory_id": rejected_trajectory.trajectory_id,
                    "credit_hash": credit_hash,
                    "algorithm": algorithm.value,
                }
            )
            sample = ReferenceTrainingSample(
                sample_id=sample_id,
                trajectory_id=chosen_trajectory.trajectory_id,
                branch_point_id=chosen_trajectory.branch_point_id,
                branch_group_id=chosen_trajectory.branch_group_id,
                policy_epoch_id=chosen_trajectory.policy_epoch_id,
                action=chosen_action.action,
                behavior_log_probability=chosen_action.behavior_log_probability,
                advantage=chosen_credit.group_relative_advantage,
                token_weight=preference.confidence,
                reward_margin=preference.reward_margin,
                chosen_action=chosen_action.action,
                rejected_action=rejected_action.action,
                staleness_updates=max(chosen_lag, rejected_lag),
                eligible=eligible,
            )
            samples.append(
                BatchSampleProvenance(
                    sample=sample,
                    tenant_id=tenant_id,
                    branch_id=chosen_trajectory.branch_id,
                    trajectory_hash=chosen_trajectory.trajectory_id,
                    reward_id=chosen_reward.reward_id,
                    reward_hash=_reward_hash(chosen_reward),
                    comparison_branch_id=rejected_trajectory.branch_id,
                    comparison_trajectory_hash=rejected_trajectory.trajectory_id,
                    comparison_reward_id=rejected_reward.reward_id,
                    comparison_reward_hash=_reward_hash(rejected_reward),
                    credit_hash=credit_hash,
                    environment_capsule_id=chosen_trajectory.final_environment_capsule_id,
                    source_model_capsule_id=chosen_trajectory.source_model_capsule_id,
                    state_reuse_report_hash=chosen_trajectory.state_reuse_report_hash,
                    staleness_disposition="accepted" if eligible else "rejected",
                )
            )
    else:
        for branch_credit in credit.credits:
            (
                _validated_credit,
                trajectory,
                reward,
                action,
                lag,
                provenance_eligible,
            ) = source_evidence(branch_credit.branch_id)
            objective_eligible = (
                branch_credit.group_relative_advantage > 0.0
                and branch_credit.decision_local_weight > 0.0
                if algorithm is TrainingAlgorithm.SUCCESSFUL_BRANCH_DISTILLATION
                else branch_credit.group_relative_advantage != 0.0
                and (
                    algorithm is TrainingAlgorithm.GROUP_RELATIVE
                    or branch_credit.decision_local_weight > 0.0
                )
            )
            eligible = bool(provenance_eligible and objective_eligible)
            if eligible:
                accepted_trajectory_ids.add(trajectory.trajectory_id)
            sample_id = _canonical_hash(
                {
                    "trajectory_id": trajectory.trajectory_id,
                    "credit_hash": credit_hash,
                    "reward_id": reward.reward_id,
                    "algorithm": algorithm.value,
                }
            )
            sample = ReferenceTrainingSample(
                sample_id=sample_id,
                trajectory_id=trajectory.trajectory_id,
                branch_point_id=trajectory.branch_point_id,
                branch_group_id=trajectory.branch_group_id,
                policy_epoch_id=trajectory.policy_epoch_id,
                action=action.action,
                behavior_log_probability=action.behavior_log_probability,
                advantage=branch_credit.group_relative_advantage,
                token_weight=branch_credit.decision_local_weight,
                staleness_updates=lag,
                eligible=eligible,
            )
            samples.append(
                BatchSampleProvenance(
                    sample=sample,
                    tenant_id=tenant_id,
                    branch_id=trajectory.branch_id,
                    trajectory_hash=trajectory.trajectory_id,
                    reward_id=reward.reward_id,
                    reward_hash=_reward_hash(reward),
                    credit_hash=credit_hash,
                    environment_capsule_id=trajectory.final_environment_capsule_id,
                    source_model_capsule_id=trajectory.source_model_capsule_id,
                    state_reuse_report_hash=trajectory.state_reuse_report_hash,
                    staleness_disposition="accepted" if eligible else "rejected",
                )
            )
    if not any(item.sample.eligible for item in samples):
        raise ValueError("all training samples were rejected")
    data_hash = _canonical_hash([item.model_dump(mode="json") for item in samples])
    excluded = required_ids - accepted_trajectory_ids
    draft: dict[str, object] = {
        "schema_version": "sloforge.helix.reference-training-batch/v1",
        "tenant_id": tenant_id,
        "branch_group_id": credit.branch_group_id,
        "branch_point_id": credit.branch_point_id,
        "behavior_policy_epoch_id": credit.policy_epoch_id,
        "learner_policy_epoch_id": learner_policy_epoch_id,
        "policy_mode": "strict",
        "algorithm": algorithm,
        "samples": tuple(samples),
        "training_sample_ids": tuple(
            item.sample.sample_id for item in samples if item.sample.eligible
        ),
        "holdout_trajectory_ids": tuple(sorted(holdout_trajectory_ids)),
        "excluded_trajectory_ids": tuple(sorted(excluded)),
        "credit_hash": credit_hash,
        "data_hash": data_hash,
        "creation_code_version": creation_code_version,
        "seed": seed,
    }
    identity = {
        key: (
            value.value
            if isinstance(value, TrainingAlgorithm)
            else [item.model_dump(mode="json") for item in value]
            if isinstance(value, tuple) and value and isinstance(value[0], BaseModel)
            else value
        )
        for key, value in draft.items()
    }
    return ReferenceTrainingBatchManifest.model_validate(
        {"batch_id": _canonical_hash(identity), **draft}, strict=True
    )
