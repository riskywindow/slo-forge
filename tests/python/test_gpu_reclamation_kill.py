from __future__ import annotations

from dataclasses import dataclass

import pytest

from sloforge.helix.characterization.gpu_reclamation import (
    CriticalStage,
    ReclamationTransactionState,
)
from sloforge.helix.characterization.gpu_reclamation_kill import (
    KILL_RECLAMATION_STAGES,
    KILL_RESTORE_STAGES,
    KillRecomputeConfig,
    KillRecomputeExecutionError,
    KillRecomputeHooks,
    KillStageObservation,
    run_kill_and_recompute_transaction,
)
from sloforge.helix.characterization.gpu_reclamation_naive import (
    ResumeContinuationObservation,
    SecondaryServingObservation,
)


@dataclass
class _Clock:
    now: int = 0

    def __call__(self) -> int:
        self.now += 10
        return self.now


def _config() -> KillRecomputeConfig:
    return KillRecomputeConfig(
        seed=41,
        branch_group_id="group-1",
        branch_count=8,
        prefix_tokens=16_384,
        private_tokens_per_branch=256,
        expected_recompute_tokens=18_432,
        expected_lost_rollout_work_tokens=2_048,
        stage_timeout_ns=1_000,
    )


def _hooks(clock: _Clock, *, fail_stage: CriticalStage | None = None) -> KillRecomputeHooks:
    actions = {}
    for stage in (*KILL_RECLAMATION_STAGES, *KILL_RESTORE_STAGES):

        def action(stage: CriticalStage = stage) -> KillStageObservation:
            if stage is fail_stage:
                raise RuntimeError("injected")
            if stage is CriticalStage.STATE_DISCARD:
                return KillStageObservation(lost_rollout_work_tokens=2_048)
            if stage is CriticalStage.RECOMPUTE_PREFILL:
                return KillStageObservation(
                    recompute_tokens=18_432,
                    gpu_time_ns=7,
                    environment_reconstruction_ns=3,
                )
            return KillStageObservation()

        actions[stage] = action
    return KillRecomputeHooks(
        stage_actions=actions,
        run_secondary_serving_spike=lambda: SecondaryServingObservation(
            gpu1_first_serving_request_ns=clock()
        ),
        drain_secondary_serving=lambda: None,
        confirm_rollout_continuation=lambda: ResumeContinuationObservation(
            completed_ns=clock(),
            resumed_branch_count=8,
            continuation_tokens_per_branch=8,
            semantics_valid=True,
        ),
        rollback_before_release=lambda: None,
        fail_closed_after_release=lambda: None,
    )


def test_kill_baseline_is_one_causal_transaction() -> None:
    clock = _Clock()
    result = run_kill_and_recompute_transaction(_config(), _hooks(clock), clock_ns=clock)
    assert result.transaction.state is ReclamationTransactionState.COMPLETED
    assert result.recompute_tokens == 18_432
    assert result.lost_rollout_work_tokens == 2_048
    assert result.recompute_gpu_time_ns == 7
    assert result.environment_reconstruction_ns == 3
    assert result.reclamation_timeline.duration_ns > 0
    assert result.restore_timeline.duration_ns > 0
    assert result.time_to_useful_reclaimed_capacity_ns > 0


def test_kill_baseline_rejects_missing_stage() -> None:
    clock = _Clock()
    hooks = _hooks(clock)
    actions = dict(hooks.stage_actions)
    del actions[CriticalStage.STATE_DISCARD]
    invalid = KillRecomputeHooks(
        stage_actions=actions,
        run_secondary_serving_spike=hooks.run_secondary_serving_spike,
        drain_secondary_serving=hooks.drain_secondary_serving,
        confirm_rollout_continuation=hooks.confirm_rollout_continuation,
        rollback_before_release=hooks.rollback_before_release,
        fail_closed_after_release=hooks.fail_closed_after_release,
    )
    with pytest.raises(ValueError, match="exact causal contract"):
        run_kill_and_recompute_transaction(_config(), invalid, clock_ns=clock)


def test_kill_baseline_fails_closed_after_discard() -> None:
    clock = _Clock()
    calls: list[str] = []
    hooks = _hooks(clock, fail_stage=CriticalStage.RECOMPUTE_PREFILL)
    guarded = KillRecomputeHooks(
        stage_actions=hooks.stage_actions,
        run_secondary_serving_spike=hooks.run_secondary_serving_spike,
        drain_secondary_serving=hooks.drain_secondary_serving,
        confirm_rollout_continuation=hooks.confirm_rollout_continuation,
        rollback_before_release=lambda: calls.append("rollback"),
        fail_closed_after_release=lambda: calls.append("closed"),
    )
    with pytest.raises(KillRecomputeExecutionError, match="failed closed"):
        run_kill_and_recompute_transaction(_config(), guarded, clock_ns=clock)
    assert calls == ["closed"]


def test_kill_baseline_validates_measured_recompute_tokens() -> None:
    clock = _Clock()
    config = _config().model_copy(update={"expected_recompute_tokens": 1})
    with pytest.raises(KillRecomputeExecutionError, match="recompute token count"):
        run_kill_and_recompute_transaction(config, _hooks(clock), clock_ns=clock)
