"""Causal kill-and-recompute baseline for Experiment 004.

The runtime remains owned by worker callbacks.  This controller enforces the
same warm role-switch ordering as preservation while making discarded work and
re-prefill cost explicit.  It contains no simulated GPU measurements.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict, Field

from sloforge.helix.characterization.gpu_reclamation import (
    CriticalStage,
    CriticalStageRecord,
    CriticalTimeline,
    ExperimentPhase,
    ExperimentPhaseMarker,
    ReclamationMode,
    ReclamationTransaction,
    ReclamationTransactionState,
    TimelineKind,
)
from sloforge.helix.characterization.gpu_reclamation_naive import (
    ResumeContinuationObservation,
    SecondaryServingObservation,
)

KILL_RECLAMATION_STAGES = (
    CriticalStage.ADMISSION_STOP,
    CriticalStage.BRANCH_QUIESCE,
    CriticalStage.STATE_DISCARD,
    CriticalStage.RUNTIME_STATE_RELEASE,
    CriticalStage.CAPACITY_RECLAIM_CONFIRMATION,
    CriticalStage.SERVING_SECONDARY_ENABLE,
)
KILL_RESTORE_STAGES = (
    CriticalStage.DESTINATION_ALLOCATION,
    CriticalStage.RECOMPUTE_PREFILL,
    CriticalStage.STATE_VALIDATION,
    CriticalStage.SCHEDULER_ADMISSION,
    CriticalStage.FIRST_FORWARD,
    CriticalStage.FIRST_TOKEN_OBSERVATION,
)
_ALL_STAGES = frozenset((*KILL_RECLAMATION_STAGES, *KILL_RESTORE_STAGES))


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, validate_default=True)


class KillRecomputeConfig(_StrictModel):
    seed: int = Field(ge=0, lt=1 << 63)
    branch_group_id: str = Field(min_length=1, max_length=512)
    branch_count: int = Field(gt=0, le=64)
    prefix_tokens: int = Field(gt=0)
    private_tokens_per_branch: int = Field(gt=0)
    expected_recompute_tokens: int = Field(gt=0)
    expected_lost_rollout_work_tokens: int = Field(gt=0)
    stage_timeout_ns: int = Field(gt=0)


class KillStageObservation(_StrictModel):
    recompute_tokens: int = Field(default=0, ge=0)
    lost_rollout_work_tokens: int = Field(default=0, ge=0)
    gpu_time_ns: int = Field(default=0, ge=0)
    environment_reconstruction_ns: int = Field(default=0, ge=0)


StageAction = Callable[[], KillStageObservation]


@dataclass(frozen=True, slots=True)
class KillRecomputeHooks:
    stage_actions: Mapping[CriticalStage, StageAction]
    run_secondary_serving_spike: Callable[[], SecondaryServingObservation]
    drain_secondary_serving: Callable[[], None]
    confirm_rollout_continuation: Callable[[], ResumeContinuationObservation]
    rollback_before_release: Callable[[], None]
    fail_closed_after_release: Callable[[], None]


@dataclass(frozen=True, slots=True)
class KillRecomputeResult:
    transaction: ReclamationTransaction
    reclamation_timeline: CriticalTimeline
    restore_timeline: CriticalTimeline
    serving: SecondaryServingObservation
    resume: ResumeContinuationObservation
    trigger_ns: int
    first_resumed_token_ns: int
    recompute_tokens: int
    lost_rollout_work_tokens: int
    recompute_gpu_time_ns: int
    environment_reconstruction_ns: int

    @property
    def reclamation_interruption_ns(self) -> int:
        return self.reclamation_timeline.duration_ns

    @property
    def rollout_resume_latency_ns(self) -> int:
        return self.first_resumed_token_ns - self.restore_timeline.stages[0].start_ns

    @property
    def time_to_useful_reclaimed_capacity_ns(self) -> int:
        return self.serving.gpu1_first_serving_request_ns - self.trigger_ns


class KillRecomputeExecutionError(RuntimeError):
    """Raised only after bounded rollback or fail-closed cleanup."""


def _run_stage(
    stage: CriticalStage,
    action: StageAction,
    *,
    cursor_ns: int,
    clock_ns: Callable[[], int],
    timeout_ns: int,
) -> tuple[CriticalStageRecord, KillStageObservation]:
    observation = action()
    if not isinstance(observation, KillStageObservation):
        raise TypeError(f"kill stage {stage.value!r} returned invalid evidence")
    end_ns = clock_ns()
    if end_ns < cursor_ns:
        raise ValueError("kill baseline monotonic clock moved backwards")
    if end_ns - cursor_ns > timeout_ns:
        raise TimeoutError(f"kill stage {stage.value!r} exceeded its measured timeout")
    return CriticalStageRecord(stage=stage, start_ns=cursor_ns, end_ns=end_ns), observation


def run_kill_and_recompute_transaction(
    config: KillRecomputeConfig,
    hooks: KillRecomputeHooks,
    *,
    clock_ns: Callable[[], int],
) -> KillRecomputeResult:
    """Execute discard -> serve -> history replay -> resume as one transaction."""

    if set(hooks.stage_actions) != _ALL_STAGES:
        raise ValueError("kill baseline stage hooks differ from the exact causal contract")
    transaction = ReclamationTransaction(
        ReclamationMode.KILL_AND_RECOMPUTE, seed=config.seed, clock_ns=clock_ns
    )
    reclamation: list[CriticalStageRecord] = []
    restore: list[CriticalStageRecord] = []
    trigger_ns = clock_ns()
    transaction.mark(
        ExperimentPhase.HELIX_RECLAIM_TRIGGER,
        attributes={"baseline": ReclamationMode.KILL_AND_RECOMPUTE.value},
    )
    transaction.transition(ReclamationTransactionState.RECLAIM_TRIGGERED)
    serving: SecondaryServingObservation | None = None
    resume: ResumeContinuationObservation | None = None
    recompute_tokens = 0
    lost_tokens = 0
    recompute_gpu_ns = 0
    environment_ns = 0
    cursor_ns = clock_ns()
    try:
        for stage in KILL_RECLAMATION_STAGES:
            if stage is CriticalStage.RUNTIME_STATE_RELEASE:
                transaction.transition(ReclamationTransactionState.SOURCE_RELEASING)
                transaction.mark(ExperimentPhase.GPU_STATE_RELEASE_BEGIN)
            record, observation = _run_stage(
                stage,
                hooks.stage_actions[stage],
                cursor_ns=cursor_ns,
                clock_ns=clock_ns,
                timeout_ns=config.stage_timeout_ns,
            )
            reclamation.append(record)
            cursor_ns = record.end_ns
            lost_tokens += observation.lost_rollout_work_tokens
            if stage is CriticalStage.ADMISSION_STOP:
                transaction.transition(ReclamationTransactionState.ADMISSIONS_STOPPED)
                transaction.mark(ExperimentPhase.ROLLOUT_ADMISSION_STOP)
            elif stage is CriticalStage.BRANCH_QUIESCE:
                transaction.transition(ReclamationTransactionState.SOURCE_FROZEN)
                transaction.mark(ExperimentPhase.BRANCH_QUIESCE)
            elif stage is CriticalStage.STATE_DISCARD:
                transaction.transition(ReclamationTransactionState.STATE_DISCARDED)
            elif stage is CriticalStage.RUNTIME_STATE_RELEASE:
                transaction.mark(ExperimentPhase.GPU_STATE_RELEASE_END)
            elif stage is CriticalStage.CAPACITY_RECLAIM_CONFIRMATION:
                transaction.transition(ReclamationTransactionState.CAPACITY_CONFIRMED)
                transaction.mark(ExperimentPhase.HBM_RECLAIM_CONFIRMED)
            elif stage is CriticalStage.SERVING_SECONDARY_ENABLE:
                transaction.transition(ReclamationTransactionState.SECONDARY_SERVING_ENABLED)
                transaction.mark(ExperimentPhase.SERVING_SECONDARY_ENABLE)

        transaction.transition(ReclamationTransactionState.SECONDARY_SERVING_ACTIVE)
        serving = hooks.run_secondary_serving_spike()
        observed_ns = clock_ns()
        if not cursor_ns <= serving.gpu1_first_serving_request_ns <= observed_ns:
            raise ValueError("GPU1 serving event is outside the measured role-switch interval")
        transaction.phase_markers.append(
            ExperimentPhaseMarker(
                phase=ExperimentPhase.GPU1_FIRST_SERVING_REQUEST,
                monotonic_timestamp_ns=serving.gpu1_first_serving_request_ns,
                attributes={"baseline": ReclamationMode.KILL_AND_RECOMPUTE.value},
            )
        )
        hooks.drain_secondary_serving()
        transaction.transition(ReclamationTransactionState.SECONDARY_SERVING_DRAINED)
        cursor_ns = clock_ns()
        transaction.mark(ExperimentPhase.ROLLOUT_RESTORE_TRIGGER)
        transaction.transition(ReclamationTransactionState.RESTORE_TRIGGERED)

        for stage in KILL_RESTORE_STAGES:
            record, observation = _run_stage(
                stage,
                hooks.stage_actions[stage],
                cursor_ns=cursor_ns,
                clock_ns=clock_ns,
                timeout_ns=config.stage_timeout_ns,
            )
            restore.append(record)
            cursor_ns = record.end_ns
            recompute_tokens += observation.recompute_tokens
            recompute_gpu_ns += observation.gpu_time_ns
            environment_ns += observation.environment_reconstruction_ns
            if stage is CriticalStage.DESTINATION_ALLOCATION:
                transaction.transition(ReclamationTransactionState.RECOMPUTING)
            elif stage is CriticalStage.STATE_VALIDATION:
                transaction.transition(ReclamationTransactionState.STATE_VALIDATED)
                transaction.mark(ExperimentPhase.STATE_VALIDATE_END)
            elif stage is CriticalStage.SCHEDULER_ADMISSION:
                transaction.transition(ReclamationTransactionState.ROLLOUT_ADMITTED)
                transaction.mark(ExperimentPhase.BRANCH_RESUME_BEGIN)
            elif stage is CriticalStage.FIRST_TOKEN_OBSERVATION:
                transaction.transition(ReclamationTransactionState.FIRST_RESUMED_TOKEN)
                transaction.mark(ExperimentPhase.FIRST_RESUMED_TOKEN)

        if recompute_tokens != config.expected_recompute_tokens:
            raise ValueError("measured recompute token count differs from the configured history")
        if lost_tokens != config.expected_lost_rollout_work_tokens:
            raise ValueError("measured discarded rollout work differs from the configured baseline")
        resume = hooks.confirm_rollout_continuation()
        if (
            not resume.semantics_valid
            or resume.resumed_branch_count != config.branch_count
            or resume.completed_ns < cursor_ns
        ):
            raise ValueError("kill/recompute continuation evidence is invalid")
        transaction.transition(ReclamationTransactionState.ROLLOUT_RESUMED)
        transaction.mark(ExperimentPhase.ROLLOUT_RESUME_COMPLETE)
        transaction.transition(ReclamationTransactionState.COMPLETED)
    except BaseException as error:
        try:
            if transaction.source_released:
                hooks.fail_closed_after_release()
            else:
                hooks.rollback_before_release()
            transaction.fail()
        except BaseException as cleanup_error:
            raise KillRecomputeExecutionError(
                f"kill baseline failed ({error!r}) and cleanup failed ({cleanup_error!r})"
            ) from cleanup_error
        raise KillRecomputeExecutionError(f"kill baseline failed closed: {error!r}") from error

    assert serving is not None and resume is not None
    reclamation_timeline = CriticalTimeline(
        kind=TimelineKind.RECLAMATION,
        mode=ReclamationMode.KILL_AND_RECOMPUTE,
        stages=tuple(reclamation),
    )
    restore_timeline = CriticalTimeline(
        kind=TimelineKind.RESTORE,
        mode=ReclamationMode.KILL_AND_RECOMPUTE,
        stages=tuple(restore),
    )
    first_resumed = next(
        item.monotonic_timestamp_ns
        for item in transaction.phase_markers
        if item.phase is ExperimentPhase.FIRST_RESUMED_TOKEN
    )
    return KillRecomputeResult(
        transaction=transaction,
        reclamation_timeline=reclamation_timeline,
        restore_timeline=restore_timeline,
        serving=serving,
        resume=resume,
        trigger_ns=trigger_ns,
        first_resumed_token_ns=first_resumed,
        recompute_tokens=recompute_tokens,
        lost_rollout_work_tokens=lost_tokens,
        recompute_gpu_time_ns=recompute_gpu_ns,
        environment_reconstruction_ns=environment_ns,
    )


__all__ = [
    "KILL_RECLAMATION_STAGES",
    "KILL_RESTORE_STAGES",
    "KillRecomputeConfig",
    "KillRecomputeExecutionError",
    "KillRecomputeHooks",
    "KillRecomputeResult",
    "KillStageObservation",
    "run_kill_and_recompute_transaction",
]
