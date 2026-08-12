from __future__ import annotations

from collections.abc import Iterator

import pytest

from sloforge.helix.characterization.gpu_reclamation import (
    REQUIRED_PRESERVATION_PHASES,
    BranchGroupSemanticEvidence,
    BranchSemanticRecord,
    CriticalStage,
    CriticalStageRecord,
    CriticalTimeline,
    ExperimentPhase,
    PilotValidityEvidence,
    ReclamationMode,
    ReclamationTransaction,
    ReclamationTransactionState,
    RuntimeAllocationIdentity,
    RuntimeIncarnation,
    TimelineKind,
    allocation_is_fresh,
    phase_trace_binding,
)


def _clock() -> Iterator[int]:
    value = 100
    while True:
        yield value
        value += 10


def _run_preserve(mode: ReclamationMode) -> ReclamationTransaction:
    ticks = _clock()
    transaction = ReclamationTransaction(mode, seed=41, clock_ns=lambda: next(ticks))
    for state in (
        ReclamationTransactionState.RECLAIM_TRIGGERED,
        ReclamationTransactionState.ADMISSIONS_STOPPED,
        ReclamationTransactionState.SOURCE_FROZEN,
        ReclamationTransactionState.CAPTURED,
        ReclamationTransactionState.TRANSPORT_VALIDATED,
        ReclamationTransactionState.TRANSPORT_PUBLISHED,
        ReclamationTransactionState.SOURCE_RELEASING,
        ReclamationTransactionState.CAPACITY_CONFIRMED,
        ReclamationTransactionState.SECONDARY_SERVING_ENABLED,
        ReclamationTransactionState.SECONDARY_SERVING_ACTIVE,
        ReclamationTransactionState.SECONDARY_SERVING_DRAINED,
        ReclamationTransactionState.RESTORE_TRIGGERED,
        ReclamationTransactionState.DESTINATION_PREPARED,
        ReclamationTransactionState.STATE_IMPORTED,
        ReclamationTransactionState.STATE_VALIDATED,
        ReclamationTransactionState.ROLLOUT_ADMITTED,
        ReclamationTransactionState.FIRST_RESUMED_TOKEN,
        ReclamationTransactionState.ROLLOUT_RESUMED,
        ReclamationTransactionState.COMPLETED,
    ):
        transaction.transition(state)
    return transaction


@pytest.mark.parametrize(
    "mode", (ReclamationMode.PRESERVE_NAIVE, ReclamationMode.PRESERVE_OPTIMIZED)
)
def test_preservation_transaction_reaches_completion(mode: ReclamationMode) -> None:
    transaction = _run_preserve(mode)
    assert transaction.state is ReclamationTransactionState.COMPLETED
    assert transaction.source_released
    assert [stamp for _, stamp in transaction.history] == sorted(
        stamp for _, stamp in transaction.history
    )


def test_kill_and_recompute_uses_discard_and_recompute_states() -> None:
    ticks = _clock()
    transaction = ReclamationTransaction(
        ReclamationMode.KILL_AND_RECOMPUTE, seed=73, clock_ns=lambda: next(ticks)
    )
    for state in (
        ReclamationTransactionState.RECLAIM_TRIGGERED,
        ReclamationTransactionState.ADMISSIONS_STOPPED,
        ReclamationTransactionState.SOURCE_FROZEN,
        ReclamationTransactionState.STATE_DISCARDED,
        ReclamationTransactionState.SOURCE_RELEASING,
        ReclamationTransactionState.CAPACITY_CONFIRMED,
        ReclamationTransactionState.SECONDARY_SERVING_ENABLED,
        ReclamationTransactionState.SECONDARY_SERVING_ACTIVE,
        ReclamationTransactionState.SECONDARY_SERVING_DRAINED,
        ReclamationTransactionState.RESTORE_TRIGGERED,
        ReclamationTransactionState.RECOMPUTING,
        ReclamationTransactionState.STATE_VALIDATED,
        ReclamationTransactionState.ROLLOUT_ADMITTED,
        ReclamationTransactionState.FIRST_RESUMED_TOKEN,
        ReclamationTransactionState.ROLLOUT_RESUMED,
        ReclamationTransactionState.COMPLETED,
    ):
        transaction.transition(state)
    assert transaction.state is ReclamationTransactionState.COMPLETED


def test_release_before_publish_and_resume_before_validation_are_rejected() -> None:
    ticks = _clock()
    transaction = ReclamationTransaction(
        ReclamationMode.PRESERVE_NAIVE, seed=41, clock_ns=lambda: next(ticks)
    )
    transaction.transition(ReclamationTransactionState.RECLAIM_TRIGGERED)
    transaction.transition(ReclamationTransactionState.ADMISSIONS_STOPPED)
    transaction.transition(ReclamationTransactionState.SOURCE_FROZEN)
    with pytest.raises(RuntimeError, match="illegal reclamation transition"):
        transaction.transition(ReclamationTransactionState.SOURCE_RELEASING)
    transaction.transition(ReclamationTransactionState.CAPTURED)
    with pytest.raises(RuntimeError, match="illegal reclamation transition"):
        transaction.transition(ReclamationTransactionState.ROLLOUT_ADMITTED)


def test_failure_is_rollback_before_release_and_closed_after_release() -> None:
    ticks = _clock()
    early = ReclamationTransaction(
        ReclamationMode.PRESERVE_NAIVE, seed=41, clock_ns=lambda: next(ticks)
    )
    early.fail()
    assert early.state is ReclamationTransactionState.FAILED_ROLLED_BACK_PRE_RELEASE

    late = ReclamationTransaction(
        ReclamationMode.PRESERVE_NAIVE, seed=41, clock_ns=lambda: next(ticks)
    )
    for state in (
        ReclamationTransactionState.RECLAIM_TRIGGERED,
        ReclamationTransactionState.ADMISSIONS_STOPPED,
        ReclamationTransactionState.SOURCE_FROZEN,
        ReclamationTransactionState.CAPTURED,
        ReclamationTransactionState.TRANSPORT_VALIDATED,
        ReclamationTransactionState.TRANSPORT_PUBLISHED,
        ReclamationTransactionState.SOURCE_RELEASING,
    ):
        late.transition(state)
    late.fail(corruption=True)
    assert late.state is ReclamationTransactionState.FAILED_CLOSED_CORRUPT


def test_transaction_rejects_regressing_clock_and_terminal_failure() -> None:
    times = iter((100, 90))
    transaction = ReclamationTransaction(
        ReclamationMode.PRESERVE_NAIVE, seed=41, clock_ns=lambda: next(times)
    )
    with pytest.raises(RuntimeError, match="clock moved backwards"):
        transaction.transition(ReclamationTransactionState.RECLAIM_TRIGGERED)
    completed = _run_preserve(ReclamationMode.PRESERVE_NAIVE)
    with pytest.raises(RuntimeError, match="already-terminal"):
        completed.fail()


def _timeline(mode: ReclamationMode, kind: TimelineKind) -> CriticalTimeline:
    stages = {
        (ReclamationMode.PRESERVE_NAIVE, TimelineKind.RECLAMATION): (
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
        ),
        (ReclamationMode.PRESERVE_OPTIMIZED, TimelineKind.RESTORE): (
            CriticalStage.DESTINATION_ALLOCATION,
            CriticalStage.H2D_UNPACK_REPAGE_PIPELINE,
            CriticalStage.RUNTIME_IMPORT,
            CriticalStage.STATE_VALIDATION,
            CriticalStage.SCHEDULER_ADMISSION,
            CriticalStage.FIRST_FORWARD,
            CriticalStage.FIRST_TOKEN_OBSERVATION,
        ),
    }[(mode, kind)]
    records = tuple(
        CriticalStageRecord(stage=stage, start_ns=index * 7, end_ns=(index + 1) * 7)
        for index, stage in enumerate(stages)
    )
    return CriticalTimeline(kind=kind, mode=mode, stages=records)


def test_critical_timeline_is_non_overlapping_and_conserves_wall_time() -> None:
    timeline = _timeline(ReclamationMode.PRESERVE_NAIVE, TimelineKind.RECLAMATION)
    assert timeline.duration_ns == sum(stage.duration_ns for stage in timeline.stages)
    optimized = _timeline(ReclamationMode.PRESERVE_OPTIMIZED, TimelineKind.RESTORE)
    assert CriticalStage.H2D_UNPACK_REPAGE_PIPELINE in {item.stage for item in optimized.stages}
    broken = list(timeline.stages)
    broken[1] = broken[1].model_copy(update={"start_ns": broken[1].start_ns + 1})
    with pytest.raises(ValueError, match="gap or overlap"):
        CriticalTimeline(
            kind=timeline.kind,
            mode=timeline.mode,
            stages=tuple(broken),
        )


def test_all_required_phase_names_bind_to_existing_trace_v1_operations() -> None:
    assert len(REQUIRED_PRESERVATION_PHASES) == len(ExperimentPhase)
    for phase in ExperimentPhase:
        branch_operation, state_operation = phase_trace_binding(phase)
        assert branch_operation
        assert state_operation.startswith("STATE_")


def test_reused_pool_index_is_fresh_only_with_new_allocation_epoch() -> None:
    source = RuntimeAllocationIdentity(gpu_uuid="GPU-source-1", block_index=11, allocation_epoch=4)
    reused = RuntimeAllocationIdentity(gpu_uuid="GPU-source-1", block_index=11, allocation_epoch=5)
    stale = RuntimeAllocationIdentity(gpu_uuid="GPU-source-1", block_index=11, allocation_epoch=4)
    assert allocation_is_fresh(source, reused)
    assert not allocation_is_fresh(source, stale)


def _valid_pilot() -> dict[str, object]:
    return {
        "gpu_uuids": ("GPU-first-1", "GPU-second-2"),
        "gpu_models": ("NVIDIA A100 80GB PCIe", "NVIDIA A100 80GB PCIe"),
        "both_models_warm_before_trigger": True,
        "gpu0_baseline_valid": True,
        "branch_count": 8,
        "prefix_tokens": 16_384,
        "suffix_tokens": 256,
        "shared_bytes": 939_524_096,
        "private_bytes": 117_440_512,
        "logical_bytes": 1_056_964_608,
        "source_assigned_bytes_before": 1_056_964_608,
        "source_assigned_bytes_after": 0,
        "source_pool_reserved_bytes_before": 50_000_000_000,
        "source_pool_reserved_bytes_after": 50_000_000_000,
        "source_blocks_allocator_available": True,
        "source_hashes_cleared": True,
        "transport_integrity_valid": True,
        "source_transport_destination_layouts_distinct": True,
        "movement_domains_complete": True,
        "required_phase_events": REQUIRED_PRESERVATION_PHASES,
        "critical_timelines_valid": True,
        "gpu1_served_real_request": True,
        "serving_metrics_recorded": True,
        "restored_branch_count": 8,
        "all_restored_allocations_fresh": True,
        "branch_semantics": _semantic_evidence(),
        "resumed_continuation_valid": True,
        "required_trace_events_dropped": 0,
        "cleanup_passed": True,
        "nvml_hbm_release_claimed": False,
    }


def _semantic_evidence() -> BranchGroupSemanticEvidence:
    source: list[BranchSemanticRecord] = []
    restored: list[BranchSemanticRecord] = []
    source_incarnations: list[RuntimeIncarnation] = []
    restored_incarnations: list[RuntimeIncarnation] = []
    expected: dict[str, int] = {}
    for index in range(8):
        logical_id = f"branch.{index}"
        common = {
            "logical_branch_id": logical_id,
            "parent_logical_branch_id": "root",
            "policy_epoch": "policy-1",
            "model_id": "Qwen/Qwen2.5-7B-Instruct",
            "model_revision": "a" * 40,
            "tokenizer_id": "Qwen/Qwen2.5-7B-Instruct",
            "tokenizer_revision": "a" * 40,
            "token_count": 16_641,
            "computed_tokens": 16_640,
            "token_history_sha256": f"{index + 1:064x}",
            "sampling_params_sha256": "f" * 64,
            "token_boundary_valid": True,
        }
        source_request = f"{logical_id}@source"
        restored_request = f"{logical_id}@restore-1"
        source.append(BranchSemanticRecord(runtime_request_id=source_request, **common))
        restored.append(BranchSemanticRecord(runtime_request_id=restored_request, **common))
        source_incarnations.append(
            RuntimeIncarnation(
                logical_branch_id=logical_id,
                runtime_request_id=source_request,
                allocations=(
                    RuntimeAllocationIdentity(
                        gpu_uuid="GPU-second-2", block_index=index, allocation_epoch=1
                    ),
                ),
            )
        )
        restored_incarnations.append(
            RuntimeIncarnation(
                logical_branch_id=logical_id,
                runtime_request_id=restored_request,
                allocations=(
                    RuntimeAllocationIdentity(
                        gpu_uuid="GPU-second-2", block_index=index, allocation_epoch=2
                    ),
                ),
            )
        )
        expected[logical_id] = 100 + index
    return BranchGroupSemanticEvidence(
        source=tuple(source),
        restored=tuple(restored),
        source_incarnations=tuple(source_incarnations),
        restored_incarnations=tuple(restored_incarnations),
        expected_first_token_ids=expected,
        observed_first_token_ids=dict(expected),
        continuation_token_counts={logical_id: 8 for logical_id in expected},
    )


def test_pilot_accepts_allocator_reclaim_with_fixed_driver_pool() -> None:
    pilot = PilotValidityEvidence.model_validate(_valid_pilot(), strict=True)
    assert pilot.source_pool_reserved_bytes_before == pilot.source_pool_reserved_bytes_after


def test_semantic_evidence_rejects_changed_policy_and_stale_allocation_generation() -> None:
    evidence = _semantic_evidence()
    changed = evidence.model_dump(mode="python")
    changed["restored"][0]["policy_epoch"] = "different-policy"
    with pytest.raises(ValueError, match="identity or token boundary"):
        BranchGroupSemanticEvidence.model_validate(changed, strict=True)

    stale = evidence.model_dump(mode="python")
    stale["restored_incarnations"][0]["allocations"] = stale["source_incarnations"][0][
        "allocations"
    ]
    with pytest.raises(ValueError, match="stale physical allocation"):
        BranchGroupSemanticEvidence.model_validate(stale, strict=True)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("restored_branch_count", 7),
        ("transport_integrity_valid", False),
        ("source_blocks_allocator_available", False),
        ("gpu1_served_real_request", False),
        ("required_trace_events_dropped", 1),
        ("nvml_hbm_release_claimed", True),
    ),
)
def test_pilot_rejects_missing_causal_evidence(field: str, value: object) -> None:
    data = _valid_pilot()
    data[field] = value
    with pytest.raises(ValueError, match="invalid Experiment 004 pilot"):
        PilotValidityEvidence.model_validate(data, strict=True)
