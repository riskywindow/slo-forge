"""Causal orchestration for Experiment 004's naive preservation baseline.

The GPU worker owns all runtime objects.  This module owns only ordering,
non-overlapping wall-clock accounting, fail-closed transitions, and the exact
byte ledger returned by worker callbacks.  The deliberately separate hooks
prevent a naive trial from being silently reported as a fused/overlapped path.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

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
from sloforge.helix.characterization.gpu_reclamation_accounting import (
    LogicalStateSegment,
    StateMovementReport,
    StatePassOperation,
    StatePassRecord,
    TransferDirection,
    build_state_movement_report,
)

NonEmpty = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=512)]


NAIVE_RECLAMATION_STAGES = (
    CriticalStage.ADMISSION_STOP,
    CriticalStage.BRANCH_QUIESCE,
    CriticalStage.FINAL_STATE_CAPTURE,
    CriticalStage.DELTA_EXTRACTION,
    CriticalStage.SOURCE_LAYOUT_READ,
    CriticalStage.STATE_TRANSFORM,
    CriticalStage.DEVICE_TO_HOST,
    CriticalStage.INTEGRITY_GENERATION,
    CriticalStage.TRANSPORT_PUBLISH,
    CriticalStage.RUNTIME_STATE_RELEASE,
    CriticalStage.CAPACITY_RECLAIM_CONFIRMATION,
    CriticalStage.SERVING_SECONDARY_ENABLE,
)

NAIVE_RESTORE_STAGES = (
    CriticalStage.DESTINATION_ALLOCATION,
    CriticalStage.TRANSPORT_LAYOUT_READ,
    CriticalStage.H2D,
    CriticalStage.DESTINATION_CONVERSION,
    CriticalStage.BLOCK_PAGE_RECONSTRUCTION,
    CriticalStage.RUNTIME_IMPORT,
    CriticalStage.STATE_VALIDATION,
    CriticalStage.SCHEDULER_ADMISSION,
    CriticalStage.FIRST_FORWARD,
    CriticalStage.FIRST_TOKEN_OBSERVATION,
)

_ALL_NAIVE_STAGES = frozenset((*NAIVE_RECLAMATION_STAGES, *NAIVE_RESTORE_STAGES))
_REQUIRED_MOVEMENT_STAGES = frozenset(
    {
        CriticalStage.SOURCE_LAYOUT_READ,
        CriticalStage.STATE_TRANSFORM,
        CriticalStage.DEVICE_TO_HOST,
        CriticalStage.INTEGRITY_GENERATION,
        CriticalStage.TRANSPORT_LAYOUT_READ,
        CriticalStage.H2D,
        CriticalStage.DESTINATION_CONVERSION,
    }
)
_ALLOWED_PASS_OPERATIONS: Mapping[CriticalStage, frozenset[StatePassOperation]] = {
    CriticalStage.SOURCE_LAYOUT_READ: frozenset({StatePassOperation.READ}),
    CriticalStage.STATE_TRANSFORM: frozenset(
        {
            StatePassOperation.UNPAGE,
            StatePassOperation.REPACK,
            StatePassOperation.RESHAPE,
            StatePassOperation.TRANSFORM,
        }
    ),
    CriticalStage.DEVICE_TO_HOST: frozenset({StatePassOperation.D2H}),
    CriticalStage.INTEGRITY_GENERATION: frozenset({StatePassOperation.CHECKSUM}),
    CriticalStage.TRANSPORT_PUBLISH: frozenset({StatePassOperation.PUBLISH}),
    CriticalStage.DESTINATION_ALLOCATION: frozenset({StatePassOperation.ALLOCATE}),
    CriticalStage.TRANSPORT_LAYOUT_READ: frozenset({StatePassOperation.READ}),
    CriticalStage.H2D: frozenset({StatePassOperation.H2D}),
    CriticalStage.DESTINATION_CONVERSION: frozenset(
        {
            StatePassOperation.UNPACK,
            StatePassOperation.REPAGE,
            StatePassOperation.RESHAPE,
            StatePassOperation.TRANSFORM,
        }
    ),
    CriticalStage.BLOCK_PAGE_RECONSTRUCTION: frozenset(
        {StatePassOperation.REPAGE, StatePassOperation.WRITE}
    ),
    CriticalStage.RUNTIME_IMPORT: frozenset({StatePassOperation.IMPORT}),
    CriticalStage.STATE_VALIDATION: frozenset(
        {StatePassOperation.CHECKSUM, StatePassOperation.VALIDATE}
    ),
}


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, validate_default=True)


class NaivePreservationConfig(_StrictModel):
    """Immutable measured-run identity and byte boundaries."""

    seed: int = Field(ge=0, lt=1 << 63)
    branch_group_id: NonEmpty
    logical_state_bytes: int = Field(gt=0)
    physical_source_bytes: int = Field(gt=0)
    physical_destination_bytes: int = Field(gt=0)
    branch_count: int = Field(gt=0, le=64)
    stage_timeout_ns: int = Field(gt=0)
    logical_segments: tuple[LogicalStateSegment, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def exact_logical_payload(self) -> NaivePreservationConfig:
        if {item.branch_group_id for item in self.logical_segments} != {self.branch_group_id}:
            raise ValueError("naive baseline logical segments belong to another branch group")
        if sum(item.logical_bytes for item in self.logical_segments) != self.logical_state_bytes:
            raise ValueError("naive baseline logical segments do not conserve logical bytes")
        if min(self.physical_source_bytes, self.physical_destination_bytes) < (
            self.logical_state_bytes
        ):
            raise ValueError("native physical state cannot be smaller than logical state")
        return self


class SecondaryServingObservation(_StrictModel):
    """Observed causal serving events while rollout state is held on the host."""

    gpu1_first_serving_request_ns: int = Field(ge=0)
    serving_slo_restored_ns: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def chronological(self) -> SecondaryServingObservation:
        if (
            self.serving_slo_restored_ns is not None
            and self.serving_slo_restored_ns < self.gpu1_first_serving_request_ns
        ):
            raise ValueError("serving SLO restoration precedes GPU1's first serving request")
        return self


class ResumeContinuationObservation(_StrictModel):
    """Post-first-token proof that every branch continued from a valid boundary."""

    completed_ns: int = Field(ge=0)
    resumed_branch_count: int = Field(gt=0, le=64)
    continuation_tokens_per_branch: int = Field(gt=0)
    semantics_valid: bool


@dataclass(frozen=True, slots=True)
class NaiveStageObservation:
    """Low-overhead evidence returned by one synchronous worker hook."""

    passes: tuple[StatePassRecord, ...] = ()


class NaivePreservationExecutionError(RuntimeError):
    """Raised after rollback/fail-closed cleanup for an invalid transaction."""


class TransportIntegrityError(ValueError):
    """Signals transport corruption so restore is classified fail-closed."""


StageAction = Callable[[], NaiveStageObservation]


@dataclass(frozen=True, slots=True)
class NaivePreservationHooks:
    """Runtime-owned callbacks invoked by the CPU-only causal controller."""

    stage_actions: Mapping[CriticalStage, StageAction]
    run_secondary_serving_spike: Callable[[], SecondaryServingObservation]
    drain_secondary_serving: Callable[[], None]
    confirm_rollout_continuation: Callable[[], ResumeContinuationObservation]
    rollback_before_release: Callable[[], None]
    fail_closed_after_release: Callable[[], None]


@dataclass(frozen=True, slots=True)
class NaivePreservationResult:
    transaction: ReclamationTransaction
    reclamation_timeline: CriticalTimeline
    restore_timeline: CriticalTimeline
    movement: StateMovementReport
    serving: SecondaryServingObservation
    resume: ResumeContinuationObservation
    trigger_ns: int
    first_resumed_token_ns: int
    phase_markers: tuple[ExperimentPhaseMarker, ...] = field(default_factory=tuple)

    @property
    def reclamation_interruption_ns(self) -> int:
        return self.reclamation_timeline.duration_ns

    @property
    def rollout_resume_latency_ns(self) -> int:
        return self.first_resumed_token_ns - self.restore_timeline.stages[0].start_ns

    @property
    def time_to_useful_reclaimed_capacity_ns(self) -> int:
        return self.serving.gpu1_first_serving_request_ns - self.trigger_ns

    @property
    def time_to_serving_slo_restoration_ns(self) -> int | None:
        if self.serving.serving_slo_restored_ns is None:
            return None
        return self.serving.serving_slo_restored_ns - self.trigger_ns


def _phase_marker(
    transaction: ReclamationTransaction,
    phase: ExperimentPhase,
    timestamp_ns: int,
    config: NaivePreservationConfig,
    *,
    physical_bytes: int = 0,
) -> None:
    transaction.phase_markers.append(
        ExperimentPhaseMarker(
            phase=phase,
            monotonic_timestamp_ns=timestamp_ns,
            logical_bytes=config.logical_state_bytes,
            physical_bytes=physical_bytes,
            attributes={"baseline": ReclamationMode.PRESERVE_NAIVE.value},
        )
    )


def _validate_hooks(hooks: NaivePreservationHooks) -> None:
    supplied = set(hooks.stage_actions)
    if supplied != _ALL_NAIVE_STAGES:
        missing = sorted(stage.value for stage in _ALL_NAIVE_STAGES - supplied)
        extra = sorted(stage.value for stage in supplied - _ALL_NAIVE_STAGES)
        raise ValueError(
            f"naive stage hooks differ from the exact contract: missing={missing}, extra={extra}"
        )


def _validate_stage_passes(
    stage: CriticalStage,
    observation: NaiveStageObservation,
    *,
    start_ns: int,
    end_ns: int,
) -> None:
    if stage in _REQUIRED_MOVEMENT_STAGES and not observation.passes:
        raise ValueError(f"naive movement stage {stage.value!r} returned no byte evidence")
    allowed = _ALLOWED_PASS_OPERATIONS.get(stage, frozenset())
    for item in observation.passes:
        if item.operation not in allowed:
            raise ValueError(
                f"state pass {item.record_id!r} operation is invalid for naive stage {stage.value!r}"
            )
        if not start_ns <= item.start_ns <= item.end_ns <= end_ns:
            raise ValueError(f"state pass {item.record_id!r} escapes its causal stage interval")
        if item.operation in {
            StatePassOperation.FUSED_NATIVE_TO_HOST,
            StatePassOperation.FUSED_HOST_TO_NATIVE,
        }:
            raise ValueError("a fused state operation cannot appear in the naive baseline")
    if stage is CriticalStage.DEVICE_TO_HOST and any(
        item.transfer_direction is not TransferDirection.D2H for item in observation.passes
    ):
        raise ValueError("naive D2H stage contains a non-D2H byte pass")
    if stage is CriticalStage.H2D and any(
        item.transfer_direction is not TransferDirection.H2D for item in observation.passes
    ):
        raise ValueError("naive H2D stage contains a non-H2D byte pass")


def _run_stage(
    *,
    stage: CriticalStage,
    action: StageAction,
    clock_ns: Callable[[], int],
    cursor_ns: int,
    timeout_ns: int,
    before: Callable[[int], None] | None = None,
    after: Callable[[int], None] | None = None,
) -> tuple[CriticalStageRecord, tuple[StatePassRecord, ...]]:
    if before is not None:
        before(cursor_ns)
    observation = action()
    if not isinstance(observation, NaiveStageObservation):
        raise TypeError(f"naive stage {stage.value!r} did not return NaiveStageObservation")
    end_ns = clock_ns()
    if end_ns < cursor_ns:
        raise ValueError("monotonic stage clock moved backwards")
    if end_ns - cursor_ns > timeout_ns:
        raise TimeoutError(f"naive stage {stage.value!r} exceeded its measured timeout")
    _validate_stage_passes(stage, observation, start_ns=cursor_ns, end_ns=end_ns)
    if after is not None:
        after(end_ns)
    return CriticalStageRecord(stage=stage, start_ns=cursor_ns, end_ns=end_ns), observation.passes


def run_naive_preservation_transaction(
    config: NaivePreservationConfig,
    hooks: NaivePreservationHooks,
    *,
    clock_ns: Callable[[], int],
) -> NaivePreservationResult:
    """Execute one synchronous preserve -> serve -> restore transaction.

    Every measured state operation is a separate callback and therefore a
    separate synchronization boundary.  Worker callbacks must emit raw byte
    passes; this function derives movement totals and never invents them.
    """

    _validate_hooks(hooks)
    transaction = ReclamationTransaction(
        ReclamationMode.PRESERVE_NAIVE,
        seed=config.seed,
        clock_ns=clock_ns,
    )
    records: dict[TimelineKind, list[CriticalStageRecord]] = {
        TimelineKind.RECLAMATION: [],
        TimelineKind.RESTORE: [],
    }
    passes: list[StatePassRecord] = []
    cursor_ns = clock_ns()
    trigger_ns = cursor_ns
    _phase_marker(transaction, ExperimentPhase.HELIX_RECLAIM_TRIGGER, cursor_ns, config)
    transaction.transition(ReclamationTransactionState.RECLAIM_TRIGGERED)

    begin_phases = {
        CriticalStage.FINAL_STATE_CAPTURE: ExperimentPhase.STATE_CAPTURE_BEGIN,
        CriticalStage.STATE_TRANSFORM: ExperimentPhase.STATE_TRANSFORM_BEGIN,
        CriticalStage.DEVICE_TO_HOST: ExperimentPhase.D2H_BEGIN,
        CriticalStage.INTEGRITY_GENERATION: ExperimentPhase.INTEGRITY_BEGIN,
        CriticalStage.RUNTIME_STATE_RELEASE: ExperimentPhase.GPU_STATE_RELEASE_BEGIN,
        CriticalStage.H2D: ExperimentPhase.H2D_BEGIN,
        CriticalStage.DESTINATION_CONVERSION: ExperimentPhase.STATE_IMPORT_BEGIN,
        CriticalStage.STATE_VALIDATION: ExperimentPhase.STATE_VALIDATE_BEGIN,
        CriticalStage.SCHEDULER_ADMISSION: ExperimentPhase.BRANCH_RESUME_BEGIN,
    }
    end_phases = {
        CriticalStage.ADMISSION_STOP: ExperimentPhase.ROLLOUT_ADMISSION_STOP,
        CriticalStage.BRANCH_QUIESCE: ExperimentPhase.BRANCH_QUIESCE,
        CriticalStage.SOURCE_LAYOUT_READ: ExperimentPhase.STATE_CAPTURE_END,
        CriticalStage.STATE_TRANSFORM: ExperimentPhase.STATE_TRANSFORM_END,
        CriticalStage.DEVICE_TO_HOST: ExperimentPhase.D2H_END,
        CriticalStage.INTEGRITY_GENERATION: ExperimentPhase.INTEGRITY_END,
        CriticalStage.TRANSPORT_PUBLISH: ExperimentPhase.STATE_PUBLISH,
        CriticalStage.RUNTIME_STATE_RELEASE: ExperimentPhase.GPU_STATE_RELEASE_END,
        CriticalStage.CAPACITY_RECLAIM_CONFIRMATION: ExperimentPhase.HBM_RECLAIM_CONFIRMED,
        CriticalStage.SERVING_SECONDARY_ENABLE: ExperimentPhase.SERVING_SECONDARY_ENABLE,
        CriticalStage.H2D: ExperimentPhase.H2D_END,
        CriticalStage.RUNTIME_IMPORT: ExperimentPhase.STATE_IMPORT_END,
        CriticalStage.STATE_VALIDATION: ExperimentPhase.STATE_VALIDATE_END,
        CriticalStage.FIRST_TOKEN_OBSERVATION: ExperimentPhase.FIRST_RESUMED_TOKEN,
    }

    after_transitions: Mapping[CriticalStage, ReclamationTransactionState] = {
        CriticalStage.ADMISSION_STOP: ReclamationTransactionState.ADMISSIONS_STOPPED,
        CriticalStage.BRANCH_QUIESCE: ReclamationTransactionState.SOURCE_FROZEN,
        CriticalStage.SOURCE_LAYOUT_READ: ReclamationTransactionState.CAPTURED,
        CriticalStage.INTEGRITY_GENERATION: ReclamationTransactionState.TRANSPORT_VALIDATED,
        CriticalStage.TRANSPORT_PUBLISH: ReclamationTransactionState.TRANSPORT_PUBLISHED,
        CriticalStage.CAPACITY_RECLAIM_CONFIRMATION: (
            ReclamationTransactionState.CAPACITY_CONFIRMED
        ),
        CriticalStage.SERVING_SECONDARY_ENABLE: (
            ReclamationTransactionState.SECONDARY_SERVING_ENABLED
        ),
        CriticalStage.DESTINATION_ALLOCATION: ReclamationTransactionState.DESTINATION_PREPARED,
        CriticalStage.RUNTIME_IMPORT: ReclamationTransactionState.STATE_IMPORTED,
        CriticalStage.STATE_VALIDATION: ReclamationTransactionState.STATE_VALIDATED,
        CriticalStage.SCHEDULER_ADMISSION: ReclamationTransactionState.ROLLOUT_ADMITTED,
        CriticalStage.FIRST_TOKEN_OBSERVATION: (ReclamationTransactionState.FIRST_RESUMED_TOKEN),
    }

    try:
        for kind, sequence in (
            (TimelineKind.RECLAMATION, NAIVE_RECLAMATION_STAGES),
            (TimelineKind.RESTORE, NAIVE_RESTORE_STAGES),
        ):
            if kind is TimelineKind.RESTORE:
                transaction.transition(ReclamationTransactionState.SECONDARY_SERVING_ACTIVE)
                serving = hooks.run_secondary_serving_spike()
                serving_observation_end_ns = clock_ns()
                if serving.gpu1_first_serving_request_ns < cursor_ns:
                    raise ValueError("GPU1 serving observation precedes secondary admission")
                if serving.gpu1_first_serving_request_ns > serving_observation_end_ns or (
                    serving.serving_slo_restored_ns is not None
                    and serving.serving_slo_restored_ns > serving_observation_end_ns
                ):
                    raise ValueError("secondary serving observation lies in the future")
                _phase_marker(
                    transaction,
                    ExperimentPhase.GPU1_FIRST_SERVING_REQUEST,
                    serving.gpu1_first_serving_request_ns,
                    config,
                )
                if serving.serving_slo_restored_ns is not None:
                    _phase_marker(
                        transaction,
                        ExperimentPhase.SERVING_SLO_RESTORED,
                        serving.serving_slo_restored_ns,
                        config,
                    )
                hooks.drain_secondary_serving()
                cursor_ns = clock_ns()
                transaction.transition(ReclamationTransactionState.SECONDARY_SERVING_DRAINED)
                _phase_marker(
                    transaction, ExperimentPhase.ROLLOUT_RESTORE_TRIGGER, cursor_ns, config
                )
                transaction.transition(ReclamationTransactionState.RESTORE_TRIGGERED)

            for stage in sequence:
                if stage is CriticalStage.RUNTIME_STATE_RELEASE:
                    transaction.transition(ReclamationTransactionState.SOURCE_RELEASING)

                begin_phase = begin_phases.get(stage)
                end_phase = end_phases.get(stage)
                after_transition = after_transitions.get(stage)
                physical_bytes = (
                    config.physical_source_bytes
                    if kind is TimelineKind.RECLAMATION
                    else config.physical_destination_bytes
                )

                def before(
                    timestamp_ns: int,
                    phase: ExperimentPhase | None = begin_phase,
                    phase_physical_bytes: int = physical_bytes,
                ) -> None:
                    if phase is not None:
                        _phase_marker(
                            transaction,
                            phase,
                            timestamp_ns,
                            config,
                            physical_bytes=phase_physical_bytes,
                        )

                def after(
                    timestamp_ns: int,
                    phase: ExperimentPhase | None = end_phase,
                    transition: ReclamationTransactionState | None = after_transition,
                    phase_physical_bytes: int = physical_bytes,
                ) -> None:
                    if phase is not None:
                        _phase_marker(
                            transaction,
                            phase,
                            timestamp_ns,
                            config,
                            physical_bytes=phase_physical_bytes,
                        )
                    if transition is not None:
                        transaction.transition(transition)

                record, stage_passes = _run_stage(
                    stage=stage,
                    action=hooks.stage_actions[stage],
                    clock_ns=clock_ns,
                    cursor_ns=cursor_ns,
                    timeout_ns=config.stage_timeout_ns,
                    before=before,
                    after=after,
                )
                records[kind].append(record)
                passes.extend(stage_passes)
                cursor_ns = record.end_ns

        first_resumed_token_ns = records[TimelineKind.RESTORE][-1].end_ns
        resume = hooks.confirm_rollout_continuation()
        resume_observation_end_ns = clock_ns()
        if resume.completed_ns < first_resumed_token_ns:
            raise ValueError("rollout continuation completion precedes the first resumed token")
        if resume.completed_ns > resume_observation_end_ns:
            raise ValueError("rollout continuation observation lies in the future")
        if resume.resumed_branch_count != config.branch_count or not resume.semantics_valid:
            raise ValueError("rollout continuation did not validate every branch")
        transaction.transition(ReclamationTransactionState.ROLLOUT_RESUMED)
        _phase_marker(
            transaction,
            ExperimentPhase.ROLLOUT_RESUME_COMPLETE,
            resume.completed_ns,
            config,
            physical_bytes=config.physical_destination_bytes,
        )
        transaction.transition(ReclamationTransactionState.COMPLETED)

        reclamation = CriticalTimeline(
            kind=TimelineKind.RECLAMATION,
            mode=ReclamationMode.PRESERVE_NAIVE,
            stages=tuple(records[TimelineKind.RECLAMATION]),
        )
        restore = CriticalTimeline(
            kind=TimelineKind.RESTORE,
            mode=ReclamationMode.PRESERVE_NAIVE,
            stages=tuple(records[TimelineKind.RESTORE]),
        )
        movement = build_state_movement_report(
            logical_segments=config.logical_segments,
            passes=passes,
        )
        return NaivePreservationResult(
            transaction=transaction,
            reclamation_timeline=reclamation,
            restore_timeline=restore,
            movement=movement,
            serving=serving,
            resume=resume,
            trigger_ns=trigger_ns,
            first_resumed_token_ns=first_resumed_token_ns,
            phase_markers=tuple(transaction.phase_markers),
        )
    except Exception as error:
        was_released = transaction.source_released
        transaction.fail(corruption=was_released and isinstance(error, TransportIntegrityError))
        try:
            if was_released:
                hooks.fail_closed_after_release()
            else:
                hooks.rollback_before_release()
        except Exception as cleanup_error:
            raise NaivePreservationExecutionError(
                f"naive preservation failed and cleanup also failed: {cleanup_error}"
            ) from error
        raise NaivePreservationExecutionError(
            f"naive preservation transaction failed in state {transaction.state.value}: {error}"
        ) from error


__all__ = [
    "NAIVE_RECLAMATION_STAGES",
    "NAIVE_RESTORE_STAGES",
    "NaivePreservationConfig",
    "NaivePreservationExecutionError",
    "NaivePreservationHooks",
    "NaivePreservationResult",
    "NaiveStageObservation",
    "ResumeContinuationObservation",
    "SecondaryServingObservation",
    "TransportIntegrityError",
    "run_naive_preservation_transaction",
]
