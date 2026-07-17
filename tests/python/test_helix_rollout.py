from __future__ import annotations

from pathlib import Path

import pytest

from sloforge.helix.effects import EffectClass
from sloforge.helix.environments import EnvironmentBackend
from sloforge.helix.environments.models import content_digest
from sloforge.helix.policy import DeterministicPolicy
from sloforge.helix.rollouts import (
    ActionMutation,
    CandidateAction,
    ReferenceRolloutWorker,
    ReferenceTrajectory,
)


def _actions(original: bytes) -> tuple[CandidateAction, ...]:
    before = content_digest(original)
    return (
        CandidateAction(
            action="wrong",
            tool_id="workspace-edit",
            effect_class=EffectClass.IDEMPOTENT_WRITE,
            mutations=(
                ActionMutation(
                    path="solution.py",
                    content="def solve(value: int) -> int:\n    return value + 1\n",
                    expected_before_hash=before,
                ),
            ),
        ),
        CandidateAction(
            action="correct",
            tool_id="workspace-edit",
            effect_class=EffectClass.IDEMPOTENT_WRITE,
            mutations=(
                ActionMutation(
                    path="solution.py",
                    content="def solve(value: int) -> int:\n    return value * 2\n",
                    expected_before_hash=before,
                ),
            ),
        ),
    )


def _run(tmp_path: Path, *, branch_id: str, forced_action: str) -> ReferenceTrajectory:
    source = tmp_path / "source"
    source.mkdir(parents=True, exist_ok=True)
    original = b"def solve(value: int) -> int:\n    return value\n"
    (source / "solution.py").write_bytes(original)
    backend = EnvironmentBackend(tmp_path / "store", tenant_id="tenant-a")
    capsule = backend.capture(source, seed=17, event_watermark=4)
    branch = backend.fork(capsule, branch_id=branch_id)
    policy = DeterministicPolicy(
        policy_epoch_id="policy-0",
        actions=("wrong", "correct"),
        logits=(2.0, 0.0),
    )
    return ReferenceRolloutWorker(tenant_id="tenant-a").run(
        branch=branch,
        initial_environment_capsule_id=capsule.capsule_id,
        branch_group_id="group-1",
        branch_point_id="point-1",
        branch_point_hash="a" * 64,
        source_model_capsule_id="continuum-capsule-1",
        state_reuse_report_hash="b" * 64,
        policy=policy,
        observation="repair solve without changing its public contract",
        candidates=_actions(original),
        seed=19,
        forced_action=forced_action,
    )


def test_reference_rollout_is_strict_hash_chained_and_isolated(tmp_path: Path) -> None:
    wrong = _run(tmp_path / "wrong", branch_id="wrong", forced_action="wrong")
    correct = _run(tmp_path / "correct", branch_id="correct", forced_action="correct")
    assert wrong.policy_mode == correct.policy_mode == "strict"
    assert wrong.initial_environment_capsule_id == correct.initial_environment_capsule_id
    assert wrong.final_environment_capsule_id != correct.final_environment_capsule_id
    assert wrong.actions[0].behavior_log_probability > correct.actions[0].behavior_log_probability
    assert wrong.event_chain_hash == wrong.events[-1].event_hash
    assert wrong.trajectory_id != correct.trajectory_id


def test_trajectory_detects_policy_mixing_and_event_tampering(tmp_path: Path) -> None:
    trajectory = _run(tmp_path, branch_id="branch", forced_action="correct")
    mixed = trajectory.actions[0].model_copy(update={"policy_epoch_id": "other-policy"})
    payload = trajectory.model_dump(mode="python")
    with pytest.raises(ValueError, match="exactly one policy"):
        ReferenceTrajectory.model_validate(
            {**payload, "actions": (mixed.model_dump(mode="python"),)},
            strict=True,
        )
    event = trajectory.events[1].model_copy(update={"payload_hash": "0" * 64})
    with pytest.raises(ValueError, match="event chain"):
        ReferenceTrajectory.model_validate(
            {
                **payload,
                "events": tuple(
                    item.model_dump(mode="python")
                    for item in (trajectory.events[0], event, *trajectory.events[2:])
                ),
            },
            strict=True,
        )


def test_reference_worker_rejects_unsafe_or_incomplete_action_contract(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unsafe external effects"):
        CandidateAction(
            action="send",
            tool_id="email",
            effect_class=EffectClass.IRREVERSIBLE_WRITE,
        )
    with pytest.raises(ValueError, match="write effect"):
        CandidateAction(
            action="bad",
            tool_id="workspace-edit",
            effect_class=EffectClass.PURE,
            mutations=(ActionMutation(path="file.txt", content="bad"),),
        )
