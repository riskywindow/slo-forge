from __future__ import annotations

import math

import pytest
from pydantic import ValidationError

from sloforge.helix.characterization.gpu_reclamation_policy import (
    BreakEvenDirection,
    ReclamationAction,
    ReclamationMeasurements,
    ReclamationWorkload,
    choose_reclamation_policy,
    estimate_kill_and_recompute,
    estimate_reclamation_paths,
    preservation_break_even,
)
from sloforge.helix.characterization.matrix import EvidenceClass
from sloforge.helix.scheduler.models import EvidenceRef


def _evidence() -> EvidenceRef:
    return EvidenceRef(
        artifact_uri="raw://branchfabric/experiment-004/trial-001",
        artifact_sha256="1" * 64,
        sample_ids=("trial-001.checkpoint", "trial-001.restore", "trial-001.recompute"),
    )


def _workload(**updates: object) -> ReclamationWorkload:
    values: dict[str, object] = {
        "seed": 41,
        "logical_state_bytes": 2_000,
        "prefix_tokens": 1_000,
        "branch_count": 2,
        "private_tokens_per_branch": 500,
        "expected_remaining_tokens_per_branch": 100,
        "serving_urgency_seconds": 2.0,
        "gpu_hour_price_usd": 3.6,
    }
    values.update(updates)
    return ReclamationWorkload(**values)  # type: ignore[arg-type]


def _measurements(**updates: object) -> ReclamationMeasurements:
    values: dict[str, object] = {
        "kill_release_seconds": 0.0,
        "first_resumed_token_seconds": 0.1,
        "recompute_tokens_per_second": 100.0,
        "naive_checkpoint_bytes_per_second": 500.0,
        "naive_restore_bytes_per_second": 500.0,
        "optimized_checkpoint_bytes_per_second": 1_000.0,
        "optimized_restore_bytes_per_second": 1_000.0,
        "naive_checkpoint_fixed_seconds": 0.2,
        "naive_restore_fixed_seconds": 0.2,
        "optimized_checkpoint_fixed_seconds": 0.1,
        "optimized_restore_fixed_seconds": 0.1,
        "evidence_class": EvidenceClass.SYNTHETIC,
        "evidence": (_evidence(),),
    }
    values.update(updates)
    return ReclamationMeasurements(**values)  # type: ignore[arg-type]


def test_estimates_real_kill_recompute_cost_and_exact_action_set() -> None:
    workload = _workload()
    measurements = _measurements()

    estimates = estimate_reclamation_paths(workload, measurements)

    assert tuple(estimate.action for estimate in estimates) == tuple(ReclamationAction)
    kill, naive, optimized = estimates
    assert kill.recompute_tokens == 2_000
    assert kill.lost_rollout_work_tokens == 1_000
    assert kill.recompute_gpu_seconds == 20.0
    assert kill.rollout_resume_seconds == 20.1
    assert kill.transition_gpu_cost_usd == pytest.approx(0.0201)
    assert naive.full_transition_seconds == pytest.approx(8.5)
    assert optimized.reclamation_seconds == pytest.approx(2.1)
    assert optimized.full_transition_seconds == pytest.approx(4.3)
    assert optimized.expected_remaining_tokens == 200


def test_explicit_kill_baseline_matches_canonical_estimate() -> None:
    workload = _workload()
    measurements = _measurements()

    assert (
        estimate_kill_and_recompute(workload, measurements)
        == estimate_reclamation_paths(workload, measurements)[0]
    )


def test_policy_selects_lowest_full_cost_that_meets_urgency() -> None:
    decision = choose_reclamation_policy(
        _workload(serving_urgency_seconds=3.0),
        _measurements(),
    )

    assert decision.selected_action is ReclamationAction.PRESERVE_OPTIMIZED
    assert decision.evidence == (_evidence(),)
    assert decision.evidence_class is EvidenceClass.SYNTHETIC
    assert decision.common_future_work_cancels is True
    assert len(decision.request_sha256) == 64


def test_urgent_reclamation_chooses_kill_even_when_preservation_saves_total_time() -> None:
    decision = choose_reclamation_policy(
        _workload(serving_urgency_seconds=0.05),
        _measurements(kill_release_seconds=0.01),
    )

    assert decision.selected_action is ReclamationAction.KILL_AND_RECOMPUTE
    assert decision.estimates[0].meets_serving_urgency is True
    assert all(not estimate.meets_serving_urgency for estimate in decision.estimates[1:])


def test_measured_naive_path_can_beat_path_named_optimized() -> None:
    decision = choose_reclamation_policy(
        _workload(serving_urgency_seconds=10.0),
        _measurements(
            naive_checkpoint_bytes_per_second=2_000.0,
            naive_restore_bytes_per_second=2_000.0,
            naive_checkpoint_fixed_seconds=0.0,
            naive_restore_fixed_seconds=0.0,
            optimized_checkpoint_bytes_per_second=1_000.0,
            optimized_restore_bytes_per_second=1_000.0,
        ),
    )

    assert decision.selected_action is ReclamationAction.PRESERVE_NAIVE


def test_completed_rollout_group_is_never_preserved() -> None:
    decision = choose_reclamation_policy(
        _workload(expected_remaining_tokens_per_branch=0, serving_urgency_seconds=10.0),
        _measurements(),
    )

    assert decision.selected_action is ReclamationAction.KILL_AND_RECOMPUTE
    assert "no rollout work remains" in decision.selection_reason
    assert all(
        item.minimum_remaining_tokens_per_branch_for_preservation == 1
        for item in decision.break_even
    )


def test_break_even_reports_tokens_age_state_gpu_seconds_and_dollars() -> None:
    result = preservation_break_even(
        _workload(),
        _measurements(),
        ReclamationAction.PRESERVE_OPTIMIZED,
    )

    assert result.direction is BreakEvenDirection.AT_OR_ABOVE
    assert result.minimum_history_tokens_at_parity == 25
    assert result.minimum_prefix_tokens_if_no_private_state == 25
    assert result.minimum_rollout_age_tokens_per_branch_at_current_prefix == 0
    assert result.equivalent_logical_state_bytes_at_parity == 25
    assert result.recompute_gpu_seconds_at_parity == 0.25
    assert result.recompute_gpu_cost_usd_at_parity == pytest.approx(0.00025)
    assert result.preservation_is_no_more_expensive_for_current_workload is True
    assert result.minimum_remaining_tokens_per_branch_for_preservation == 1


def test_break_even_fails_closed_when_transfer_scales_worse_than_recompute() -> None:
    result = preservation_break_even(
        _workload(),
        _measurements(
            optimized_checkpoint_bytes_per_second=100.0,
            optimized_restore_bytes_per_second=100.0,
        ),
        ReclamationAction.PRESERVE_OPTIMIZED,
    )

    assert result.direction is BreakEvenDirection.NO_HISTORY_SIZE
    assert result.minimum_history_tokens_at_parity is None
    assert result.maximum_history_tokens_at_parity is None
    assert result.equivalent_logical_state_bytes_at_parity is None
    assert result.preservation_is_no_more_expensive_for_current_workload is False
    assert result.minimum_remaining_tokens_per_branch_for_preservation is None


def test_decision_is_deterministic_and_seed_is_in_input_digest() -> None:
    first = choose_reclamation_policy(_workload(seed=41), _measurements())
    repeated = choose_reclamation_policy(_workload(seed=41), _measurements())
    different_seed = choose_reclamation_policy(_workload(seed=73), _measurements())

    assert first == repeated
    assert first.request_sha256 == repeated.request_sha256
    assert first.request_sha256 != different_seed.request_sha256


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("logical_state_bytes", 0),
        ("branch_count", 0),
        ("serving_urgency_seconds", math.inf),
        ("serving_urgency_seconds", 0.0),
    ],
)
def test_workload_validation_is_bounded(field: str, value: object) -> None:
    with pytest.raises(ValidationError):
        _workload(**{field: value})


def test_measurement_validation_rejects_nonpositive_rates_and_duplicate_evidence() -> None:
    with pytest.raises(ValidationError):
        _measurements(recompute_tokens_per_second=0.0)
    with pytest.raises(ValidationError, match="must be unique"):
        _measurements(evidence=(_evidence(), _evidence()))


def test_aggregate_token_bound_rejects_pathological_policy_input() -> None:
    with pytest.raises(ValidationError, match="aggregate reconstructable history"):
        _workload(
            prefix_tokens=1 << 30,
            branch_count=4096,
            private_tokens_per_branch=1 << 30,
        )


def test_break_even_rejects_kill_action() -> None:
    with pytest.raises(ValueError, match="does not have preservation transfer rates"):
        preservation_break_even(
            _workload(),
            _measurements(),
            ReclamationAction.KILL_AND_RECOMPUTE,
        )
