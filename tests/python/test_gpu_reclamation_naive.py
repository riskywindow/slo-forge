from __future__ import annotations

from dataclasses import dataclass

import pytest

from sloforge.helix.characterization.gpu_reclamation import (
    CriticalStage,
    ExperimentPhase,
    ReclamationTransactionState,
)
from sloforge.helix.characterization.gpu_reclamation_accounting import (
    LogicalStateSegment,
    MemoryDomain,
    StatePassOperation,
    StatePassRecord,
    StateSegmentKind,
    TransferDirection,
)
from sloforge.helix.characterization.gpu_reclamation_naive import (
    NAIVE_RECLAMATION_STAGES,
    NAIVE_RESTORE_STAGES,
    NaivePreservationConfig,
    NaivePreservationExecutionError,
    NaivePreservationHooks,
    NaiveStageObservation,
    ResumeContinuationObservation,
    SecondaryServingObservation,
    TransportIntegrityError,
    run_naive_preservation_transaction,
)


@dataclass
class _Clock:
    now_ns: int = 1_000

    def __call__(self) -> int:
        return self.now_ns

    def advance(self, duration_ns: int) -> None:
        self.now_ns += duration_ns


def _config(*, timeout_ns: int = 50) -> NaivePreservationConfig:
    return NaivePreservationConfig(
        seed=41,
        branch_group_id="group-004",
        logical_state_bytes=100,
        physical_source_bytes=128,
        physical_destination_bytes=128,
        branch_count=8,
        stage_timeout_ns=timeout_ns,
        logical_segments=(
            LogicalStateSegment(
                segment_id="shared-root",
                branch_group_id="group-004",
                kind=StateSegmentKind.SHARED_KV,
                logical_bytes=100,
            ),
        ),
    )


def _state_pass(stage: CriticalStage, start_ns: int, end_ns: int) -> StatePassRecord | None:
    common: dict[str, object] = {
        "record_id": f"pass-{stage.value}",
        "state_segment": "shared-root",
        "branch_group": "group-004",
        "logical_offset_bytes": 0,
        "logical_bytes": 100,
        "start_ns": start_ns,
        "end_ns": end_ns,
        "device": "GPU-fixture",
        "required_unavoidable": False,
    }
    if stage is CriticalStage.SOURCE_LAYOUT_READ:
        return StatePassRecord(
            **common,
            operation=StatePassOperation.READ,
            source_memory=MemoryDomain.SOURCE_GPU_NATIVE_PAGED,
            destination_memory=MemoryDomain.GPU_TRANSFORM_BUFFER,
            bytes_read=128,
            bytes_written=128,
            temporary_allocation_bytes=128,
            temporary_allocation_memory=MemoryDomain.GPU_TRANSFORM_BUFFER,
            temporary_allocation_id="source-materialization",
        )
    if stage is CriticalStage.STATE_TRANSFORM:
        return StatePassRecord(
            **common,
            operation=StatePassOperation.TRANSFORM,
            source_memory=MemoryDomain.GPU_TRANSFORM_BUFFER,
            destination_memory=MemoryDomain.GPU_TRANSPORT_BUFFER,
            bytes_read=128,
            bytes_written=100,
            temporary_allocation_bytes=100,
            temporary_allocation_memory=MemoryDomain.GPU_TRANSPORT_BUFFER,
            temporary_allocation_id="canonical-device",
        )
    if stage is CriticalStage.DEVICE_TO_HOST:
        return StatePassRecord(
            **common,
            operation=StatePassOperation.D2H,
            source_memory=MemoryDomain.GPU_TRANSPORT_BUFFER,
            destination_memory=MemoryDomain.PINNED_HOST_TRANSPORT,
            bytes_read=100,
            bytes_written=100,
            transfer_direction=TransferDirection.D2H,
            transfer_bytes=100,
            temporary_allocation_bytes=100,
            temporary_allocation_memory=MemoryDomain.PINNED_HOST_TRANSPORT,
            temporary_allocation_id="canonical-host",
        )
    if stage is CriticalStage.INTEGRITY_GENERATION:
        return StatePassRecord(
            **common,
            operation=StatePassOperation.CHECKSUM,
            source_memory=MemoryDomain.PINNED_HOST_TRANSPORT,
            destination_memory=MemoryDomain.NONE,
            bytes_read=100,
            checksum_bytes=100,
        )
    if stage is CriticalStage.TRANSPORT_PUBLISH:
        return StatePassRecord(
            **common,
            operation=StatePassOperation.PUBLISH,
            source_memory=MemoryDomain.PINNED_HOST_TRANSPORT,
            destination_memory=MemoryDomain.HOST_TRANSPORT_STORE,
            bytes_read=100,
            bytes_written=100,
        )
    if stage is CriticalStage.TRANSPORT_LAYOUT_READ:
        return StatePassRecord(
            **common,
            operation=StatePassOperation.READ,
            source_memory=MemoryDomain.HOST_TRANSPORT_STORE,
            destination_memory=MemoryDomain.PINNED_HOST_TRANSPORT,
            bytes_read=100,
            bytes_written=100,
        )
    if stage is CriticalStage.H2D:
        return StatePassRecord(
            **common,
            operation=StatePassOperation.H2D,
            source_memory=MemoryDomain.PINNED_HOST_TRANSPORT,
            destination_memory=MemoryDomain.GPU_TRANSPORT_BUFFER,
            bytes_read=100,
            bytes_written=100,
            transfer_direction=TransferDirection.H2D,
            transfer_bytes=100,
            temporary_allocation_bytes=100,
            temporary_allocation_memory=MemoryDomain.GPU_TRANSPORT_BUFFER,
            temporary_allocation_id="restore-device",
        )
    if stage is CriticalStage.DESTINATION_CONVERSION:
        common["required_unavoidable"] = True
        return StatePassRecord(
            **common,
            operation=StatePassOperation.UNPACK,
            source_memory=MemoryDomain.GPU_TRANSPORT_BUFFER,
            destination_memory=MemoryDomain.DESTINATION_GPU_NATIVE_PAGED,
            bytes_read=100,
            bytes_written=128,
        )
    return None


@dataclass
class _HookState:
    clock: _Clock
    order: list[str]
    rollback_count: int = 0
    fail_closed_count: int = 0


def _hooks(
    state: _HookState,
    *,
    replacement: dict[CriticalStage, object] | None = None,
) -> NaivePreservationHooks:
    actions = {}
    for stage in (*NAIVE_RECLAMATION_STAGES, *NAIVE_RESTORE_STAGES):

        def action(selected: CriticalStage = stage) -> NaiveStageObservation:
            state.order.append(selected.value)
            start_ns = state.clock()
            state.clock.advance(10)
            item = _state_pass(selected, start_ns, state.clock())
            return NaiveStageObservation(passes=() if item is None else (item,))

        actions[stage] = action
    if replacement:
        actions.update(replacement)  # type: ignore[arg-type]

    def serve() -> SecondaryServingObservation:
        state.order.append("serving-spike")
        start_ns = state.clock()
        state.clock.advance(30)
        return SecondaryServingObservation(
            gpu1_first_serving_request_ns=start_ns + 5,
            serving_slo_restored_ns=start_ns + 20,
        )

    def drain() -> None:
        state.order.append("serving-drain")
        state.clock.advance(10)

    def resume() -> ResumeContinuationObservation:
        state.order.append("resume-continuation")
        state.clock.advance(20)
        return ResumeContinuationObservation(
            completed_ns=state.clock(),
            resumed_branch_count=8,
            continuation_tokens_per_branch=4,
            semantics_valid=True,
        )

    def rollback() -> None:
        state.rollback_count += 1

    def fail_closed() -> None:
        state.fail_closed_count += 1

    return NaivePreservationHooks(
        stage_actions=actions,
        run_secondary_serving_spike=serve,
        drain_secondary_serving=drain,
        confirm_rollout_continuation=resume,
        rollback_before_release=rollback,
        fail_closed_after_release=fail_closed,
    )


def test_naive_transaction_is_causal_separate_and_byte_accounted() -> None:
    state = _HookState(clock=_Clock(), order=[])
    result = run_naive_preservation_transaction(
        _config(),
        _hooks(state),
        clock_ns=state.clock,
    )

    assert result.transaction.state is ReclamationTransactionState.COMPLETED
    assert result.reclamation_interruption_ns == 120
    assert result.rollout_resume_latency_ns == 100
    assert result.time_to_useful_reclaimed_capacity_ns == 125
    assert result.time_to_serving_slo_restoration_ns == 140
    assert [item.stage for item in result.reclamation_timeline.stages] == list(
        NAIVE_RECLAMATION_STAGES
    )
    assert [item.stage for item in result.restore_timeline.stages] == list(NAIVE_RESTORE_STAGES)
    assert state.order == [
        *(stage.value for stage in NAIVE_RECLAMATION_STAGES),
        "serving-spike",
        "serving-drain",
        *(stage.value for stage in NAIVE_RESTORE_STAGES),
        "resume-continuation",
    ]
    assert {item.phase for item in result.phase_markers} == set(ExperimentPhase)
    assert len(result.phase_markers) == len(ExperimentPhase)
    assert result.movement.accounting.logical_state_bytes == 100
    assert result.movement.accounting.physical_bytes_read == 128
    assert result.movement.accounting.physical_bytes_written == 128
    assert result.movement.accounting.d2h_bytes == 100
    assert result.movement.accounting.h2d_bytes == 100
    assert result.movement.accounting.checksum_bytes == 100
    assert result.movement.accounting.state_movement_amplification == pytest.approx(18.12)
    assert (
        result.movement.accounting.state_movement_amplification_excluding_unavoidable_final_writes
        == pytest.approx(16.84)
    )
    assert state.rollback_count == 0
    assert state.fail_closed_count == 0


def test_integrity_failure_before_release_rolls_back_and_never_releases() -> None:
    state = _HookState(clock=_Clock(), order=[])

    def corrupt() -> NaiveStageObservation:
        state.order.append(CriticalStage.INTEGRITY_GENERATION.value)
        state.clock.advance(10)
        raise TransportIntegrityError("fixture checksum mismatch")

    with pytest.raises(NaivePreservationExecutionError, match="FAILED_ROLLED_BACK_PRE_RELEASE"):
        run_naive_preservation_transaction(
            _config(),
            _hooks(state, replacement={CriticalStage.INTEGRITY_GENERATION: corrupt}),
            clock_ns=state.clock,
        )
    assert CriticalStage.RUNTIME_STATE_RELEASE.value not in state.order
    assert state.rollback_count == 1
    assert state.fail_closed_count == 0


def test_restore_corruption_fails_closed_and_never_admits_branches() -> None:
    state = _HookState(clock=_Clock(), order=[])

    def corrupt() -> NaiveStageObservation:
        state.order.append(CriticalStage.STATE_VALIDATION.value)
        state.clock.advance(10)
        raise TransportIntegrityError("fixture restore digest mismatch")

    with pytest.raises(NaivePreservationExecutionError, match="FAILED_CLOSED_CORRUPT"):
        run_naive_preservation_transaction(
            _config(),
            _hooks(state, replacement={CriticalStage.STATE_VALIDATION: corrupt}),
            clock_ns=state.clock,
        )
    assert CriticalStage.SCHEDULER_ADMISSION.value not in state.order
    assert state.rollback_count == 0
    assert state.fail_closed_count == 1


def test_naive_runner_rejects_missing_stage_hook_before_side_effects() -> None:
    state = _HookState(clock=_Clock(), order=[])
    hooks = _hooks(state)
    actions = dict(hooks.stage_actions)
    actions.pop(CriticalStage.H2D)
    invalid = NaivePreservationHooks(
        stage_actions=actions,
        run_secondary_serving_spike=hooks.run_secondary_serving_spike,
        drain_secondary_serving=hooks.drain_secondary_serving,
        confirm_rollout_continuation=hooks.confirm_rollout_continuation,
        rollback_before_release=hooks.rollback_before_release,
        fail_closed_after_release=hooks.fail_closed_after_release,
    )
    with pytest.raises(ValueError, match=r"missing=.*h2d"):
        run_naive_preservation_transaction(_config(), invalid, clock_ns=state.clock)
    assert state.order == []


def test_stage_timeout_is_invalid_and_uses_pre_release_rollback() -> None:
    state = _HookState(clock=_Clock(), order=[])

    def slow_capture() -> NaiveStageObservation:
        state.order.append(CriticalStage.FINAL_STATE_CAPTURE.value)
        state.clock.advance(51)
        return NaiveStageObservation()

    with pytest.raises(NaivePreservationExecutionError, match="exceeded its measured timeout"):
        run_naive_preservation_transaction(
            _config(timeout_ns=50),
            _hooks(state, replacement={CriticalStage.FINAL_STATE_CAPTURE: slow_capture}),
            clock_ns=state.clock,
        )
    assert state.rollback_count == 1
    assert state.fail_closed_count == 0
