from __future__ import annotations

import pytest
from pydantic import ValidationError

from sloforge.helix.characterization.gpu_reclamation import RuntimeAllocationIdentity
from sloforge.helix.characterization.gpu_reclamation_serving import (
    ArrivalPhase,
    ObservationOutcome,
    PhaseInterval,
    ServingObservation,
    ServingRequest,
    ServingSpikeConfig,
    ServingWorkload,
    WeightedTokenDistribution,
)
from sloforge.helix.characterization.gpu_reclamation_v10_validity import (
    BoundedBacklogEvidence,
    BranchResumeEvidence,
    BranchSemanticSnapshot,
    BudgetEvidence,
    CleanupEvidence,
    MovementAccountingEvidence,
    RawBranchGroupRestoreEvidence,
    StateCorrectnessEvidence,
    TimelineEvidenceKind,
    V10AssessmentWindows,
    V10Phase,
    V10PrerequisiteEvidence,
    V10ScientificValidity,
    V10Timeline,
    V10TimelineEvent,
    V10ValidityConfig,
    assess_v10_scientific_validity,
    build_branch_group_semantic_evidence,
    fail_closed_v10_scientific_validity,
    outstanding_queue_depth_at,
)

NS = 1_000_000_000
GPU0 = "GPU-serving-0"
GPU1 = "GPU-rollout-1"


def _serving_evidence() -> tuple[ServingWorkload, tuple[ServingObservation, ...]]:
    raw: list[tuple[int, str, int, str]] = []
    for second in range(5):
        raw.append((second * NS, "control", second * NS + 50_000_000, GPU0))
    for index in range(25):
        arrival = 5 * NS + index * 200_000_000
        service = arrival + 50_000_000 if index < 5 else 12 * NS + (index - 5) * 400_000_000
        raw.append((arrival, "spike", service, GPU0))
    for index in range(110):
        arrival = 10 * NS + index * 200_000_000
        if arrival >= 29 * NS and index % 5 == 4:
            # Restore uses the measured lower GPU0-only rate (4 rps), not the
            # 5 rps recovery spike.
            continue
        if arrival < 12 * NS:
            device = GPU0
        elif arrival < 29 * NS:
            device = GPU0 if index % 2 == 0 else GPU1
        else:
            device = GPU0
        phase = "restore" if arrival >= 29 * NS else "spike"
        raw.append((arrival, phase, arrival + 50_000_000, device))
    raw.sort(key=lambda item: item[0])
    requests: list[ServingRequest] = []
    observations: list[ServingObservation] = []
    for sequence, (arrival, phase, service, device) in enumerate(raw):
        request_id = f"request-{sequence:04d}"
        requests.append(
            ServingRequest(
                sequence=sequence,
                request_id=request_id,
                phase=phase,
                arrival_ns=arrival,
                prompt_tokens=256,
                requested_output_tokens=2,
            )
        )
        observations.append(
            ServingObservation(
                request_id=request_id,
                arrival_ns=arrival,
                service_start_ns=service,
                token_timestamps_ns=(service + 50_000_000, service + 100_000_000),
                completion_ns=service + 150_000_000,
                outcome=ObservationOutcome.COMPLETED,
                device=device,
            )
        )
    workload = ServingWorkload(
        workload_id="a" * 64,
        start_ns=0,
        end_ns=33 * NS,
        config=ServingSpikeConfig(
            seed=41,
            control_phase="control",
            spike_phase="spike",
            phases=(
                ArrivalPhase(
                    name="control",
                    start_offset_ns=0,
                    end_offset_ns=5 * NS,
                    interarrival_ns=NS,
                ),
                ArrivalPhase(
                    name="spike",
                    start_offset_ns=5 * NS,
                    end_offset_ns=33 * NS,
                    interarrival_ns=200_000_000,
                ),
            ),
            prompt_tokens=WeightedTokenDistribution(values=(256,), weights=(1,)),
            output_tokens=WeightedTokenDistribution(values=(2,), weights=(1,)),
        ),
        requests=tuple(requests),
    )
    return workload, tuple(observations)


def _timeline() -> V10Timeline:
    timestamps = {
        V10Phase.CONTROL_STABLE: 5 * NS,
        V10Phase.LOAD_SPIKE_BEGIN: 5 * NS,
        V10Phase.GPU0_OVERLOAD_CONFIRMED: 10 * NS,
        V10Phase.RECLAIM_TRIGGER: 10 * NS,
        V10Phase.ROLLOUT_ADMISSION_STOP: 10 * NS + 10_000_000,
        V10Phase.BRANCH_QUIESCE_BEGIN: 10 * NS + 20_000_000,
        V10Phase.BRANCH_QUIESCE_END: 10 * NS + 30_000_000,
        V10Phase.STATE_CAPTURE_BEGIN: 10 * NS + 40_000_000,
        V10Phase.STATE_CAPTURE_END: 10 * NS + 100_000_000,
        V10Phase.STATE_TRANSFORM_BEGIN: 10 * NS + 110_000_000,
        V10Phase.STATE_TRANSFORM_END: 10 * NS + 300_000_000,
        V10Phase.D2H_BEGIN: 10 * NS + 310_000_000,
        V10Phase.D2H_END: 10 * NS + 400_000_000,
        V10Phase.INTEGRITY_BEGIN: 10 * NS + 410_000_000,
        V10Phase.INTEGRITY_END: 10 * NS + 600_000_000,
        V10Phase.STATE_PUBLISH: 10 * NS + 700_000_000,
        V10Phase.GPU1_STATE_RELEASE_BEGIN: 10 * NS + 710_000_000,
        V10Phase.GPU1_STATE_RELEASE_END: 10 * NS + 720_000_000,
        V10Phase.GPU1_HBM_RECLAIM_CONFIRMED: 10 * NS + 800_000_000,
        V10Phase.GPU1_SERVING_ENABLE: 11 * NS,
        V10Phase.GPU1_FIRST_USEFUL_SERVING_REQUEST: 12 * NS,
        V10Phase.SERVING_QUEUE_DRAIN_BEGIN: 12 * NS + 500_000_000,
        V10Phase.SERVING_QUEUE_DRAIN_END: 20 * NS,
        V10Phase.SERVING_SLO_RECOVERY_BEGIN: 12 * NS,
        V10Phase.SERVING_SLO_RESTORED: 21 * NS,
        V10Phase.SERVING_SLO_STABILITY_BEGIN: 21 * NS,
        V10Phase.SERVING_SLO_STABILITY_END: 26 * NS,
        V10Phase.TWO_GPU_SERVICE_STABLE: 26 * NS,
        V10Phase.RESTORE_LOAD_REDUCED: 27 * NS,
        V10Phase.GPU1_SERVING_DRAINED: 28 * NS,
        V10Phase.RESTORE_ELIGIBLE: 29 * NS,
        V10Phase.RESTORE_TRIGGER: 29 * NS,
        V10Phase.H2D_BEGIN: 29 * NS,
        V10Phase.H2D_END: 29 * NS + 40_000_000,
        V10Phase.STATE_IMPORT_BEGIN: 29 * NS + 40_000_000,
        V10Phase.STATE_IMPORT_END: 30 * NS + 500_000_000,
        V10Phase.DESTINATION_NATIVE_WRITE_BEGIN: 29 * NS + 100_000_000,
        V10Phase.DESTINATION_NATIVE_WRITE_END: 30 * NS + 400_000_000,
        V10Phase.STATE_VALIDATE_BEGIN: 29 * NS + 550_000_000,
        V10Phase.STATE_VALIDATE_END: 30 * NS + 450_000_000,
        V10Phase.BRANCH_RESUME_BEGIN: 30 * NS + 500_000_000,
        V10Phase.FIRST_RESUMED_TOKEN: 30 * NS + 600_000_000,
        V10Phase.ALL_BRANCHES_RESUMED: 30 * NS + 700_000_000,
        V10Phase.ROLLOUT_CONTINUATION_COMPLETE: 32 * NS,
    }
    return V10Timeline(
        events=tuple(
            V10TimelineEvent(
                phase=phase,
                monotonic_timestamp_ns=timestamps[phase],
                evidence_kind=TimelineEvidenceKind.RAW_RUNTIME_STATE,
                evidence_reference=f"fixture:{phase.value}",
            )
            for phase in V10Phase
        )
    )


def _raw_branch_evidence(
    expected: dict[str, int],
) -> RawBranchGroupRestoreEvidence:
    snapshots = tuple(
        BranchSemanticSnapshot(
            logical_branch_id=branch_id,
            parent_logical_branch_id="rollout-root",
            policy_epoch="policy-1",
            model_id="Qwen/Qwen2.5-7B-Instruct",
            model_revision="a" * 40,
            tokenizer_id="Qwen/Qwen2.5-7B-Instruct",
            tokenizer_revision="a" * 40,
            token_count=16_641,
            computed_tokens=16_640,
            token_history_sha256=f"{index + 1:064x}",
            sampling_params_sha256=f"{index + 101:064x}",
        )
        for index, branch_id in enumerate(sorted(expected))
    )
    pages = {branch_id: (f"page-{index}",) for index, branch_id in enumerate(sorted(expected))}
    return RawBranchGroupRestoreEvidence(
        source=snapshots,
        restored=snapshots,
        source_runtime_request_ids={
            branch_id: f"source-runtime-{index}" for index, branch_id in enumerate(sorted(expected))
        },
        restored_runtime_request_ids={
            branch_id: f"restore-runtime-{index}"
            for index, branch_id in enumerate(sorted(expected))
        },
        logical_page_ids_by_branch=pages,
        source_allocations_by_page={
            page_ids[0]: RuntimeAllocationIdentity(
                gpu_uuid=GPU1,
                block_index=index,
                allocation_epoch=1,
            )
            for index, page_ids in enumerate(pages.values())
        },
        restored_allocations_by_page={
            page_ids[0]: RuntimeAllocationIdentity(
                gpu_uuid=GPU1,
                block_index=index,
                allocation_epoch=2,
            )
            for index, page_ids in enumerate(pages.values())
        },
        expected_first_token_ids=expected,
        observed_first_token_ids=dict(expected),
        continuation_token_counts={branch_id: 8 for branch_id in expected},
    )


def _assessment(
    *,
    gpu1_device: str = GPU1,
    observations_override: tuple[ServingObservation, ...] | None = None,
    scheduler_waiting_queue_depths: tuple[int, ...] | None = None,
) -> V10ScientificValidity:
    workload, observations = _serving_evidence()
    if observations_override is not None:
        observations = observations_override
    branches = {f"branch.{index}": 100 + index for index in range(8)}
    semantic_continuity = build_branch_group_semantic_evidence(_raw_branch_evidence(branches))
    return assess_v10_scientific_validity(
        config=V10ValidityConfig(
            control_offered_rate_per_second=1.0,
            spike_offered_rate_per_second=5.0,
            restore_offered_rate_per_second=4.0,
            lambda_1_rps=4.5,
            recovery_queue_depth_threshold=1,
        ),
        timeline=_timeline(),
        windows=V10AssessmentWindows(
            control=PhaseInterval(name="control", start_ns=0, end_ns=5 * NS),
            overload=PhaseInterval(name="overload", start_ns=5 * NS, end_ns=10 * NS),
            two_gpu_excess_capacity=PhaseInterval(
                name="two-gpu-excess", start_ns=12 * NS, end_ns=26 * NS
            ),
            queue_drain=PhaseInterval(
                name="queue-drain", start_ns=12 * NS + 500_000_000, end_ns=20 * NS
            ),
            slo_stability=PhaseInterval(name="slo-stability", start_ns=21 * NS, end_ns=26 * NS),
            restore_activity=PhaseInterval(
                name="restore-activity", start_ns=29 * NS, end_ns=32 * NS
            ),
        ),
        workload=workload,
        observations=observations,
        gpu0_device=GPU0,
        gpu1_device=gpu1_device,
        bounded_backlog=BoundedBacklogEvidence(
            interval=PhaseInterval(name="bounded-backlog", start_ns=5 * NS, end_ns=12 * NS),
            trigger_observation_ns=10 * NS,
            trigger_queue_depth=21,
            maximum_queue_depth=21,
            queue_depth_at_gpu1_first_useful=21,
            configured_trigger_depth=20,
            configured_abort_depth=64,
            preferred_minimum_depth=10,
            preferred_maximum_depth=25,
            passed=True,
        ),
        state_correctness=StateCorrectnessEvidence(
            logical_state_bytes=1_056_964_608,
            preserved_state_bytes=1_056_964_608,
            source_blocks_expected=1152,
            source_blocks_released=1152,
            source_kv_assigned_bytes_before=1_056_964_608,
            source_kv_assigned_bytes_after=0,
            source_kv_pool_reserved_bytes_before=2_113_929_216,
            source_kv_pool_reserved_bytes_after=2_113_929_216,
            source_kv_unassigned_bytes_before=1_056_964_608,
            source_kv_unassigned_bytes_after=2_113_929_216,
            transport_integrity_valid=True,
            fresh_destination_allocations=True,
        ),
        branch_resume=BranchResumeEvidence(
            expected_first_tokens=branches,
            observed_first_tokens=branches,
            continuation_token_counts={branch: 8 for branch in branches},
            restored_into_fresh_allocations=True,
            integrity_valid=True,
            semantic_continuity=semantic_continuity,
        ),
        movement_accounting=MovementAccountingEvidence(
            logical_state_bytes=1_056_964_608,
            full_physical_touch_bytes=58_133_053_440,
            external_movement_bytes=3_170_893_824,
            avoidable_movement_bytes=32_765_902_848,
            critical_path_movement_bytes=58_133_053_440,
            accounting_duplicate_bytes=0,
            all_physical_passes_recorded=True,
            formulas_recomputed_from_raw_artifacts=True,
        ),
        budget=BudgetEvidence(
            conservative_gpu_seconds_before=100.0,
            invocation_gpu_seconds=10.0,
            conservative_gpu_seconds_after=110.0,
            hard_ceiling_gpu_seconds=200.0,
            ledger_updated=True,
        ),
        cleanup=CleanupEvidence(
            zero_active_tasks=True,
            zero_running_containers=True,
            zero_endpoints=True,
            zero_reservations=True,
            zero_owned_child_processes=True,
            zero_profilers=True,
            authorized_persistent_volumes_only=True,
        ),
        prerequisites=V10PrerequisiteEvidence(
            authorization_valid=True,
            phase_budget_valid=True,
            two_engine_ready=True,
            sanity_12rps_stable=True,
            sanity_15rps_overload=True,
            budget_pass=True,
            cleanup_pass=True,
            evidence_references=tuple(
                f"{kind}:{index:064x}:fixture/{kind}.json"
                for index, kind in enumerate(
                    (
                        "authorization",
                        "phase-budget",
                        "engine-readiness",
                        "sanity-12rps",
                        "sanity-15rps",
                        "budget-settlement",
                        "modal-cleanup",
                    ),
                    start=1,
                )
            ),
        ),
        scheduler_waiting_queue_depths=scheduler_waiting_queue_depths,
    )


def test_valid_v10_requires_every_serving_and_transaction_gate() -> None:
    assessment = _assessment()

    assert assessment.scientifically_valid
    assert assessment.gpu0_overload.overload_condition_count == 3
    assert assessment.bounded_backlog_pass


def test_slo_stability_accepts_exact_scheduler_waiting_samples() -> None:
    assessment = _assessment(scheduler_waiting_queue_depths=(0,) * 20)

    assert assessment.slo_stability is not None
    assert assessment.slo_stability.maximum_queue_depth == 0
    assert assessment.slo_stability_pass

    with pytest.raises(ValueError, match="do not match evaluation windows"):
        _assessment(scheduler_waiting_queue_depths=(0,) * 19)
    assert assessment.gpu1_hbm_reclaim_pass
    assert assessment.two_gpu_excess_capacity.completed_rate_per_second > 5.0
    assert assessment.queue_drain.trend.direction == "negative"
    assert assessment.slo_stability.duration_ns == 5 * NS
    assert assessment.slo_stability.all_windows_have_ttft_samples
    assert assessment.gpu0_restore_activity.completion_count > 0
    assert assessment.gpu0_restore_activity.emitted_tokens > 0
    assert assessment.gpu0_restore_activity.ttft_sample_count > 0
    assert [stage.stage for stage in assessment.gpu0_restore_activity.stage_activity] == [
        "h2d",
        "import",
        "destination-native-write",
        "destination-validation",
        "resume",
    ]
    h2d, import_stage, native_write, destination_validation, resume = (
        assessment.gpu0_restore_activity.stage_activity
    )
    assert not h2d.eligible_for_interference
    assert not h2d.sufficient_sample
    assert h2d.passed is None
    assert import_stage.passed is True
    assert native_write.passed is True
    assert destination_validation.passed is True
    assert resume.passed is True
    assert assessment.gpu0_restore_activity.early_half_completion_count > 0
    assert assessment.gpu0_restore_activity.late_half_completion_count > 0
    assert assessment.gpu0_restore_activity.activity_spans_restore


def test_scientific_validity_cannot_be_true_with_a_false_gate() -> None:
    source = _assessment().model_dump(mode="python")
    source["gpu0_active_during_restore_pass"] = False

    with pytest.raises(ValidationError, match="booleans disagree"):
        V10ScientificValidity.model_validate(source, strict=True)


def test_scientific_validity_rejects_missing_or_null_boolean() -> None:
    missing = _assessment().model_dump(mode="python")
    del missing["queue_drain_pass"]
    with pytest.raises(ValidationError):
        V10ScientificValidity.model_validate(missing, strict=True)

    null = _assessment().model_dump(mode="python")
    null["queue_drain_pass"] = None
    with pytest.raises(ValidationError):
        V10ScientificValidity.model_validate(null, strict=True)


def test_absent_prerequisite_evidence_forces_every_preflight_gate_false() -> None:
    source = _assessment().model_dump(mode="python")
    source["prerequisites"] = None
    for field in (
        "authorization_alid",
        "phase_budget_valid",
        "two_engine_ready",
        "sanity_12rps_stable",
        "sanity_15rps_overload",
    ):
        source[field] = False
    source["scientifically_valid"] = False
    source["invalid_reasons"] = ("verified readiness/calibration evidence is absent",)

    assessment = V10ScientificValidity.model_validate(source, strict=True)

    assert not assessment.scientifically_valid
    assert not assessment.authorization_alid


def test_movement_duplicate_bytes_fail_the_accounting_gate() -> None:
    source = _assessment().model_dump(mode="python")
    source["movement_accounting"]["accounting_duplicate_bytes"] = 1
    source["movement_accounting_pass"] = False
    source["scientifically_valid"] = False
    source["invalid_reasons"] = ("required gate failed: movement accounting",)

    assessment = V10ScientificValidity.model_validate(source, strict=True)

    assert assessment.movement_accounting is not None
    assert not assessment.movement_accounting.passed
    assert not assessment.movement_accounting_pass


def test_hbm_reclamation_is_distinct_from_source_block_release() -> None:
    source = _assessment().model_dump(mode="python")
    source["state_correctness"]["source_kv_pool_reserved_bytes_after"] += 1
    source["gpu1_hbm_reclaim_pass"] = False
    source["scientifically_valid"] = False
    source["invalid_reasons"] = ("required gate failed: GPU1 HBM reclamation",)

    assessment = V10ScientificValidity.model_validate(source, strict=True)

    assert assessment.gpu1_state_release_pass
    assert not assessment.gpu1_hbm_reclaim_pass


def test_backlog_must_remain_below_the_precommitted_abort_bound() -> None:
    source = _assessment().model_dump(mode="python")
    source["bounded_backlog"]["maximum_queue_depth"] = 65
    source["bounded_backlog"]["passed"] = False
    source["bounded_backlog_pass"] = False
    source["scientifically_valid"] = False
    source["invalid_reasons"] = ("required gate failed: bounded backlog",)

    assessment = V10ScientificValidity.model_validate(source, strict=True)

    assert not assessment.bounded_backlog_pass


def test_backlog_may_exceed_preferred_trigger_range_but_not_hard_abort() -> None:
    source = _assessment().model_dump(mode="python")
    source["bounded_backlog"]["maximum_queue_depth"] = 26

    assessment = V10ScientificValidity.model_validate(source, strict=True)

    assert assessment.bounded_backlog_pass
    assert assessment.scientifically_valid


def test_missing_gpu1_completions_fails_capacity_gate_without_aborting_assessment() -> None:
    assessment = _assessment(gpu1_device="GPU-unused-2")

    assert not assessment.scientifically_valid
    assert not assessment.two_gpu_service_gt_offered_pass
    assert assessment.two_gpu_excess_capacity is not None
    assert assessment.two_gpu_excess_capacity.gpu1_completions == 0
    assert "required gate failed: GPU1 useful capacity" in assessment.invalid_reasons
    assert (
        "required gate failed: two-GPU service greater than offered" in assessment.invalid_reasons
    )


def test_gpu0_restore_activity_requires_global_gpu0_routing_even_when_h2d_is_short() -> None:
    _workload, observations = _serving_evidence()
    observations = tuple(
        row.model_copy(update={"device": GPU1})
        if 29 * NS <= row.arrival_ns < 29 * NS + 40_000_000
        else row
        for row in observations
    )

    assessment = _assessment(observations_override=observations)

    assert not assessment.gpu0_active_during_restore_pass
    assert assessment.gpu0_restore_activity is not None
    h2d = assessment.gpu0_restore_activity.stage_activity[0]
    assert not h2d.eligible_for_interference
    assert h2d.passed is None
    assert not h2d.all_scheduled_arrivals_routed_gpu0
    assert not assessment.gpu0_restore_activity.all_scheduled_arrivals_routed_gpu0


def test_raw_branch_semantics_use_actual_runtime_ids_and_fresh_allocations() -> None:
    expected = {f"branch.{index}": 100 + index for index in range(8)}
    raw = _raw_branch_evidence(expected)

    evidence = build_branch_group_semantic_evidence(raw)

    assert {row.runtime_request_id for row in evidence.source} == set(
        raw.source_runtime_request_ids.values()
    )
    assert {row.runtime_request_id for row in evidence.restored} == set(
        raw.restored_runtime_request_ids.values()
    )
    assert evidence.expected_first_token_ids == evidence.observed_first_token_ids


def test_raw_branch_semantics_reject_reused_runtime_or_allocation_incarnation() -> None:
    expected = {f"branch.{index}": 100 + index for index in range(8)}
    raw = _raw_branch_evidence(expected)
    reused_runtime = dict(raw.restored_runtime_request_ids)
    reused_runtime["branch.0"] = raw.source_runtime_request_ids["branch.0"]
    with pytest.raises(ValueError, match="runtime request incarnations"):
        build_branch_group_semantic_evidence(
            raw.model_copy(update={"restored_runtime_request_ids": reused_runtime})
        )

    with pytest.raises(ValueError, match="stale physical allocation"):
        build_branch_group_semantic_evidence(
            raw.model_copy(update={"restored_allocations_by_page": raw.source_allocations_by_page})
        )

    aliased = dict(raw.restored_allocations_by_page)
    pages = tuple(aliased)
    aliased[pages[1]] = aliased[pages[0]]
    with pytest.raises(ValueError, match="distinct logical pages alias"):
        build_branch_group_semantic_evidence(
            raw.model_copy(update={"restored_allocations_by_page": aliased})
        )


def test_timeline_requires_every_event_exactly_once() -> None:
    source = _timeline().model_dump(mode="python")
    source["events"] = source["events"][:-1]

    with pytest.raises(ValidationError, match="every required event exactly once"):
        V10Timeline.model_validate(source, strict=True)


def test_timeline_accepts_validation_nested_inside_import() -> None:
    timeline = _timeline()

    assert timeline.timestamp(V10Phase.SERVING_SLO_RECOVERY_BEGIN) < timeline.timestamp(
        V10Phase.SERVING_QUEUE_DRAIN_BEGIN
    )
    assert timeline.timestamp(V10Phase.STATE_IMPORT_BEGIN) < timeline.timestamp(
        V10Phase.STATE_VALIDATE_BEGIN
    )
    assert timeline.timestamp(V10Phase.STATE_VALIDATE_END) < timeline.timestamp(
        V10Phase.STATE_IMPORT_END
    )


def test_timeline_rejects_validation_after_import_or_other_causal_reordering() -> None:
    source = _timeline().model_dump(mode="python")
    by_phase = {row["phase"]: row for row in source["events"]}
    by_phase[V10Phase.STATE_VALIDATE_END]["monotonic_timestamp_ns"] = 31 * NS + 1
    with pytest.raises(ValidationError, match="causal event order"):
        V10Timeline.model_validate(source, strict=True)

    source = _timeline().model_dump(mode="python")
    by_phase = {row["phase"]: row for row in source["events"]}
    by_phase[V10Phase.RESTORE_TRIGGER]["monotonic_timestamp_ns"] = 25 * NS
    with pytest.raises(ValidationError, match="causal event order"):
        V10Timeline.model_validate(source, strict=True)


def test_stability_window_cannot_be_shorter_than_five_seconds() -> None:
    with pytest.raises(ValidationError, match="at least five seconds"):
        V10ValidityConfig(
            control_offered_rate_per_second=1.0,
            spike_offered_rate_per_second=5.0,
            restore_offered_rate_per_second=4.0,
            lambda_1_rps=4.5,
            recovery_queue_depth_threshold=1,
            stability_window_ns=4 * NS,
        )


def test_outstanding_queue_does_not_treat_admission_as_completion() -> None:
    workload, observations = _serving_evidence()
    by_request = {row.request_id: row for row in observations}
    request = workload.requests[5]
    observation = by_request[request.request_id]
    assert observation.service_start_ns is not None
    timestamp = observation.service_start_ns + 1
    assert timestamp < observation.completion_ns

    assert outstanding_queue_depth_at(workload, by_request, timestamp) >= 1


def test_postprocessor_failure_still_emits_all_required_false_booleans() -> None:
    assessment = fail_closed_v10_scientific_validity("raw serving observations are absent")

    assert not assessment.scientifically_valid
    assert assessment.invalid_reasons == ("raw serving observations are absent",)
    assert assessment.timeline is None
    boolean_fields = (
        "authorization_alid",
        "phase_budget_valid",
        "two_engine_ready",
        "sanity_12rps_stable",
        "sanity_15rps_overload",
        "control_stable",
        "gpu0_overload_pass",
        "bounded_backlog_pass",
        "state_correctness_pass",
        "gpu1_state_release_pass",
        "gpu1_hbm_reclaim_pass",
        "gpu1_useful_capacity_pass",
        "two_gpu_service_gt_offered_pass",
        "queue_drain_pass",
        "slo_restoration_pass",
        "slo_stability_pass",
        "restore_load_reduced_pass",
        "gpu1_serving_drained_pass",
        "gpu0_active_during_restore_pass",
        "branch_resume_pass",
        "movement_accounting_pass",
        "budget_pass",
        "cleanup_pass",
    )
    payload = assessment.model_dump(mode="json")
    assert all(payload[name] is False for name in boolean_fields)
