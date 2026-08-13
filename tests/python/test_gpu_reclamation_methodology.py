from __future__ import annotations

import json

import pytest

from sloforge.helix.characterization.gpu_reclamation import (
    REQUIRED_PRESERVATION_PHASES,
    BranchGroupSemanticEvidence,
    BranchSemanticRecord,
    PilotValidityEvidence,
    ReclamationMode,
    RuntimeAllocationIdentity,
    RuntimeIncarnation,
)
from sloforge.helix.characterization.gpu_reclamation_methodology import (
    ArtifactSampleRef,
    Experiment004GpuHourLedger,
    MetricSample,
    RawTrial,
    TraceMetricControl,
    TrialSemanticEvidence,
    analyze_final_campaign,
    charge_failed_gpu_reservation,
    evaluate_trace_controls,
    evaluate_trial_validity,
    plan_final_campaign,
    reserve_gpu_invocation,
    settle_gpu_invocation,
)


def _artifact(label: str, selector: str = "$") -> ArtifactSampleRef:
    digit = "a" if not label or label[0] not in "abcdef0123456789" else label[0]
    return ArtifactSampleRef(
        artifact_reference=f"raw/{label}.json",
        artifact_sha256=digit * 64,
        sample_selector=selector,
    )


def _pilot() -> PilotValidityEvidence:
    return PilotValidityEvidence(
        gpu_uuids=("GPU-first-1", "GPU-second-2"),
        gpu_models=("NVIDIA A100 80GB PCIe", "NVIDIA A100 80GB PCIe"),
        both_models_warm_before_trigger=True,
        gpu0_baseline_valid=True,
        branch_count=8,
        prefix_tokens=16_384,
        suffix_tokens=256,
        shared_bytes=939_524_096,
        private_bytes=117_440_512,
        logical_bytes=1_056_964_608,
        source_assigned_bytes_before=1_056_964_608,
        source_assigned_bytes_after=0,
        source_pool_reserved_bytes_before=50_000_000_000,
        source_pool_reserved_bytes_after=50_000_000_000,
        source_blocks_allocator_available=True,
        source_hashes_cleared=True,
        transport_integrity_valid=True,
        source_transport_destination_layouts_distinct=True,
        movement_domains_complete=True,
        required_phase_events=REQUIRED_PRESERVATION_PHASES,
        critical_timelines_valid=True,
        gpu1_served_real_request=True,
        serving_metrics_recorded=True,
        restored_branch_count=8,
        all_restored_allocations_fresh=True,
        branch_semantics=_branch_semantics(),
        resumed_continuation_valid=True,
        required_trace_events_dropped=0,
        cleanup_passed=True,
        nvml_hbm_release_claimed=False,
    )


def _branch_semantics() -> BranchGroupSemanticEvidence:
    source = []
    restored = []
    source_incarnations = []
    restored_incarnations = []
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
        }
        source_id = f"{logical_id}@source"
        restored_id = f"{logical_id}@restore-1"
        source.append(BranchSemanticRecord(runtime_request_id=source_id, **common))
        restored.append(BranchSemanticRecord(runtime_request_id=restored_id, **common))
        source_incarnations.append(
            RuntimeIncarnation(
                logical_branch_id=logical_id,
                runtime_request_id=source_id,
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
                runtime_request_id=restored_id,
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


def _trace_gate(*, material: bool = False):
    return evaluate_trace_controls(
        control_id="trace-control-s191",
        seed=191,
        workload_digest="b" * 64,
        gpu_uuids=("GPU-first-1", "GPU-second-2"),
        controls=(
            TraceMetricControl(
                metric="reclamation_interruption",
                unit="ms",
                disabled_value=100.0,
                minimal_value=106.0 if material else 103.0,
                full_value=118.0,
                materiality_threshold_fraction=0.05,
                disabled_provenance=_artifact("disabled"),
                minimal_provenance=_artifact("minimal"),
                full_provenance=_artifact("full"),
            ),
            TraceMetricControl(
                metric="logical_state_bytes",
                unit="bytes",
                disabled_value=1_056_964_608.0,
                minimal_value=1_056_964_608.0,
                full_value=1_056_964_608.0,
                materiality_threshold_fraction=0.0,
                exact_behavior_metric=True,
                disabled_provenance=_artifact("disabled-bytes"),
                minimal_provenance=_artifact("minimal-bytes"),
                full_provenance=_artifact("full-bytes"),
            ),
        ),
    )


def _semantic(mode: ReclamationMode) -> TrialSemanticEvidence:
    preserved = mode is not ReclamationMode.KILL_AND_RECOMPUTE
    return TrialSemanticEvidence(
        transaction_completed=True,
        serving_scenario_valid=True,
        hardware_identity_valid=True,
        workload_replay_valid=True,
        critical_timelines_nonoverlapping=True,
        movement_accounting_valid=True,
        allocator_reclaim_valid=True,
        gpu1_useful_serving_capacity_observed=True,
        cleanup_passed=True,
        budget_accounted=True,
        preservation_integrity_valid=preserved,
        fresh_restore_allocations=preserved,
        all_branches_resumed=preserved,
        recompute_valid=not preserved,
    )


def _trials(plan, values) -> tuple[RawTrial, ...]:
    trials = []
    for planned in plan.trials:
        value = values[(planned.seed, planned.mode)]
        trials.append(
            RawTrial(
                trial_id=planned.trial_id,
                seed=planned.seed,
                mode=planned.mode,
                order_position=planned.order_position,
                tracing_level="minimal",
                pilot_digest=plan.pilot_digest,
                workload_digest="b" * 64,
                gpu_uuids=("GPU-first-1", "GPU-second-2"),
                semantic_evidence=_semantic(planned.mode),
                metrics=(
                    MetricSample(
                        metric="reclamation_interruption",
                        unit="ms",
                        value=value,
                        provenance=_artifact(
                            f"{planned.seed}-{planned.mode.value}",
                            "$.reclamation_interruption_ms",
                        ),
                    ),
                ),
                raw_manifest=_artifact(f"manifest-{planned.trial_id}"),
            )
        )
    return tuple(trials)


def test_campaign_order_is_deterministic_and_position_balanced() -> None:
    first = plan_final_campaign(
        pilot=_pilot(),
        pilot_provenance=_artifact("pilot"),
        order_seed=20_260_811,
    )
    second = plan_final_campaign(
        pilot=_pilot(),
        pilot_provenance=_artifact("pilot"),
        order_seed=20_260_811,
    )
    different = plan_final_campaign(
        pilot=_pilot(),
        pilot_provenance=_artifact("pilot"),
        order_seed=20_260_812,
    )

    assert first == second
    assert first.trials != different.trials
    for position in (1, 2, 3):
        assert {trial.mode for trial in first.trials if trial.order_position == position} == set(
            ReclamationMode
        )
    json.dumps(first.model_dump(mode="json"), allow_nan=False)


def test_campaign_requires_exact_three_unique_seeds() -> None:
    with pytest.raises(ValueError, match="three unique"):
        plan_final_campaign(
            pilot=_pilot(),
            pilot_provenance=_artifact("pilot"),
            order_seed=41,
            seeds=(41, 41, 113),
        )


def test_trace_gate_compares_minimal_and_full_to_disabled_controls() -> None:
    gate = _trace_gate()
    assert gate.minimal_trace_causal_trials_allowed
    assert gate.full_trace_diagnostic_only
    evaluation = gate.evaluations[0]
    assert evaluation.minimal_relative_change == pytest.approx(0.03)
    assert evaluation.full_relative_change == pytest.approx(0.18)
    assert not evaluation.minimal_material_change
    assert evaluation.full_material_change


def test_trace_gate_rejects_material_minimal_overhead_or_behavior_change() -> None:
    material = _trace_gate(material=True)
    assert not material.minimal_trace_causal_trials_allowed
    assert material.rejection_reasons == (
        "minimal tracing materially changed reclamation_interruption",
    )

    control = TraceMetricControl(
        metric="d2h_bytes",
        unit="bytes",
        disabled_value=10.0,
        minimal_value=11.0,
        full_value=10.0,
        materiality_threshold_fraction=1.0,
        exact_behavior_metric=True,
        disabled_provenance=_artifact("disabled"),
        minimal_provenance=_artifact("minimal"),
        full_provenance=_artifact("full"),
    )
    exact = evaluate_trace_controls(
        control_id="exact",
        seed=1,
        workload_digest="b" * 64,
        gpu_uuids=("GPU-first-1", "GPU-second-2"),
        controls=(control,),
    )
    assert not exact.minimal_trace_causal_trials_allowed


def test_full_trace_and_invalid_restore_are_not_causal_samples() -> None:
    plan = plan_final_campaign(
        pilot=_pilot(),
        pilot_provenance=_artifact("pilot"),
        order_seed=20_260_811,
    )
    planned = next(trial for trial in plan.trials if trial.mode is ReclamationMode.PRESERVE_NAIVE)
    trial = RawTrial(
        trial_id=planned.trial_id,
        seed=planned.seed,
        mode=planned.mode,
        order_position=planned.order_position,
        tracing_level="full",
        pilot_digest=plan.pilot_digest,
        workload_digest="b" * 64,
        gpu_uuids=("GPU-first-1", "GPU-second-2"),
        semantic_evidence=_semantic(planned.mode).model_copy(
            update={"all_branches_resumed": False}
        ),
        metrics=(
            MetricSample(
                metric="reclamation_interruption",
                unit="ms",
                value=1.0,
                provenance=_artifact("metric"),
            ),
        ),
        raw_manifest=_artifact("manifest"),
    )
    validity = evaluate_trial_validity(
        trial,
        planned=planned,
        plan_pilot_digest=plan.pilot_digest,
        trace_gate=_trace_gate(),
    )
    assert not validity.causal_sample_accepted
    assert "full tracing is diagnostic-only" in validity.reasons
    assert "not all branches resumed" in validity.reasons


def test_final_analysis_keeps_raw_pairs_medians_and_declines_unsupported_ci() -> None:
    plan = plan_final_campaign(
        pilot=_pilot(),
        pilot_provenance=_artifact("pilot"),
        order_seed=20_260_811,
    )
    values = {}
    for offset, seed in enumerate((41, 73, 113)):
        values[(seed, ReclamationMode.KILL_AND_RECOMPUTE)] = 20.0 + offset
        values[(seed, ReclamationMode.PRESERVE_NAIVE)] = 100.0 + 2 * offset
        values[(seed, ReclamationMode.PRESERVE_OPTIMIZED)] = 60.0 + offset
    result = analyze_final_campaign(
        plan=plan,
        trials=_trials(plan, values),
        trace_gate=_trace_gate(),
        metric="reclamation_interruption",
        unit="ms",
    )

    assert result["accepted_trial_count"] == 9
    modes = result["modes"]
    assert modes["KILL_AND_RECOMPUTE"]["median"] == 21.0
    assert modes["PRESERVE_NAIVE"]["median"] == 102.0
    comparison = next(
        item
        for item in result["pairwise"]
        if item["baseline_mode"] == "PRESERVE_NAIVE"
        and item["candidate_mode"] == "PRESERVE_OPTIMIZED"
    )
    assert comparison["differences"] == [-40.0, -41.0, -42.0]
    assert comparison["summary"]["median_difference"] == -41.0
    assert comparison["confidence_interval"]["status"] == "UNAVAILABLE"
    assert comparison["tail_statistics"]["p99"] is None
    assert comparison["raw_pairs"][0]["baseline_provenance"]
    json.dumps(result, allow_nan=False)


def test_final_analysis_rejects_one_semantically_invalid_trial() -> None:
    plan = plan_final_campaign(
        pilot=_pilot(),
        pilot_provenance=_artifact("pilot"),
        order_seed=20_260_811,
    )
    values = {(trial.seed, trial.mode): 10.0 for trial in plan.trials}
    trials = list(_trials(plan, values))
    trials[0] = trials[0].model_copy(
        update={
            "semantic_evidence": trials[0].semantic_evidence.model_copy(
                update={"cleanup_passed": False}
            )
        }
    )
    with pytest.raises(ValueError, match="postflight cleanup failed"):
        analyze_final_campaign(
            plan=plan,
            trials=trials,
            trace_gate=_trace_gate(),
            metric="reclamation_interruption",
            unit="ms",
        )


def test_gpu_budget_preflight_counts_two_gpus_and_active_reservation() -> None:
    ledger = Experiment004GpuHourLedger(consumed_additional_gpu_seconds=0.0)
    reserved, preflight = reserve_gpu_invocation(
        ledger,
        reservation_id="reservation-pilot",
        invocation_id="pilot",
        maximum_wall_seconds=900.0,
        config_sha256="c" * 64,
    )

    assert preflight.proposed_maximum_gpu_seconds == 1_800.0
    assert preflight.remaining_hard_budget_after_gpu_seconds == 9_000.0
    assert not preflight.target_exceeded_if_fully_consumed
    assert reserved.committed_additional_gpu_seconds == 1_800.0
    with pytest.raises(ValueError, match="already active"):
        reserve_gpu_invocation(
            reserved,
            reservation_id="second",
            invocation_id="second",
            maximum_wall_seconds=10.0,
            config_sha256="d" * 64,
        )


def test_gpu_budget_hard_stop_includes_consumed_hours() -> None:
    ledger = Experiment004GpuHourLedger(consumed_additional_gpu_seconds=0.0)
    reserved, _ = reserve_gpu_invocation(
        ledger,
        reservation_id="first",
        invocation_id="first",
        maximum_wall_seconds=1_801.0,
        config_sha256="c" * 64,
    )
    settled = settle_gpu_invocation(
        reserved,
        reservation_id="first",
        function_call_id="fc-first",
        actual_gpu_models=("NVIDIA A100 80GB PCIe", "NVIDIA A100 80GB PCIe"),
        gpu_uuids=("GPU-first-1", "GPU-second-2"),
        client_elapsed_seconds=1_801.0,
        remote_observed_allocation_seconds=1_801.0,
        raw_manifest=_artifact("first"),
        gpu_price_per_hour_usd=3.0,
    )
    assert settled.consumed_additional_gpu_hours == pytest.approx(3_602.0 / 3_600.0)
    assert settled.intervals[0].gpu_cost_usd == pytest.approx(3_602.0 / 3_600.0 * 3.0)
    with pytest.raises(ValueError, match="configured hard ceiling"):
        reserve_gpu_invocation(
            settled,
            reservation_id="too-large",
            invocation_id="too-large",
            maximum_wall_seconds=3_600.0,
            config_sha256="d" * 64,
        )


def test_gpu_budget_honors_authoritative_configured_larger_ceiling() -> None:
    ledger = Experiment004GpuHourLedger(
        hard_additional_gpu_seconds=12_600.0,
        consumed_additional_gpu_seconds=0.0,
    )
    reserved, preflight = reserve_gpu_invocation(
        ledger,
        reservation_id="explicit-larger-ceiling",
        invocation_id="explicit-larger-ceiling",
        maximum_wall_seconds=3_600.0,
        config_sha256="c" * 64,
    )

    assert reserved.hard_additional_gpu_seconds == 12_600.0
    assert preflight.projected_committed_gpu_seconds == 7_200.0
    assert preflight.remaining_hard_budget_after_gpu_seconds == 5_400.0
    settled = settle_gpu_invocation(
        reserved,
        reservation_id="explicit-larger-ceiling",
        function_call_id="fc-explicit-larger-ceiling",
        actual_gpu_models=("NVIDIA A100 80GB PCIe", "NVIDIA A100 80GB PCIe"),
        gpu_uuids=("GPU-larger-first", "GPU-larger-second"),
        client_elapsed_seconds=1.0,
        remote_observed_allocation_seconds=1.0,
        raw_manifest=_artifact("explicit-larger-ceiling"),
    )
    assert settled.hard_additional_gpu_seconds == 12_600.0


def test_gpu_settlement_uses_conservative_elapsed_bound_and_rejects_overrun() -> None:
    ledger, _ = reserve_gpu_invocation(
        Experiment004GpuHourLedger(consumed_additional_gpu_seconds=0.0),
        reservation_id="pilot",
        invocation_id="pilot",
        maximum_wall_seconds=100.0,
        config_sha256="c" * 64,
    )
    with pytest.raises(ValueError, match="exceeded its preflight"):
        settle_gpu_invocation(
            ledger,
            reservation_id="pilot",
            function_call_id="fc-pilot",
            actual_gpu_models=("NVIDIA A100 80GB PCIe", "NVIDIA A100 80GB PCIe"),
            gpu_uuids=("GPU-first-1", "GPU-second-2"),
            client_elapsed_seconds=101.0,
            remote_observed_allocation_seconds=99.0,
            raw_manifest=_artifact("pilot"),
        )


def test_failed_gpu_reservation_is_cleared_and_charged_without_fabricated_interval() -> None:
    ledger, _ = reserve_gpu_invocation(
        Experiment004GpuHourLedger(consumed_additional_gpu_seconds=0.0),
        reservation_id="failed-reservation",
        invocation_id="failed-invocation",
        maximum_wall_seconds=456.3125,
        config_sha256="c" * 64,
    )
    charged = charge_failed_gpu_reservation(
        ledger,
        reservation_id="failed-reservation",
        failure_stage="modal-process",
        failure_evidence=_artifact("failed-charge"),
        gpu_price_per_hour_usd=2.4984,
    )

    assert not charged.reservations
    assert not charged.intervals
    assert charged.consumed_additional_gpu_seconds == 912.625
    charge = charged.conservative_failure_charges[0]
    assert charge.charged_wall_seconds == 456.3125
    assert charge.conservative_gpu_seconds == 912.625
    assert charge.conservative_gpu_cost_usd == pytest.approx(912.625 / 3_600.0 * 2.4984)


def test_gpu_ledger_rejects_fabricated_consumption() -> None:
    with pytest.raises(ValueError, match="must equal settled intervals and failure charges"):
        Experiment004GpuHourLedger(consumed_additional_gpu_seconds=1.0)
