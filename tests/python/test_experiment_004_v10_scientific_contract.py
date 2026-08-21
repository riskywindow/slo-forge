from __future__ import annotations

from sloforge.helix.characterization.gpu_reclamation_v10_validity import (
    V10ScientificValidity,
    fail_closed_v10_scientific_validity,
)

REQUIRED_V10_GATE_FIELDS = (
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


def test_v10_canonical_boolean_gate_names_exactly_match_the_completion_contract() -> None:
    boolean_fields = tuple(
        name
        for name, field in V10ScientificValidity.model_fields.items()
        if field.annotation is bool and name != "scientifically_valid"
    )

    assert boolean_fields == REQUIRED_V10_GATE_FIELDS


def test_v10_fail_closed_artifact_contains_every_required_gate_as_false() -> None:
    payload = fail_closed_v10_scientific_validity("deliberate contract fixture").model_dump(
        mode="json"
    )

    assert tuple(name for name in payload if name in REQUIRED_V10_GATE_FIELDS) == (
        REQUIRED_V10_GATE_FIELDS
    )
    assert all(payload[name] is False for name in REQUIRED_V10_GATE_FIELDS)
    assert payload["scientifically_valid"] is False
