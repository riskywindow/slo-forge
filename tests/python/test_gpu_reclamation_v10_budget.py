from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from sloforge.helix.characterization.gpu_reclamation_v10_budget import (
    PHASE_ORDER,
    V10PhaseBudget,
    evaluate_budget_gate,
    load_v10_phase_budget,
)

ROOT = Path(__file__).resolve().parents[2]
PHASE_BUDGET = ROOT / "artifacts/branchfabric/gpu-validation/experiment-004/v10-phase-budget.json"


def test_checked_v10_phase_budget_is_complete_hash_bound_and_evidence_derived() -> None:
    budget = load_v10_phase_budget(PHASE_BUDGET, repository_root=ROOT)

    assert tuple(phase.phase for phase in budget.phases) == PHASE_ORDER
    assert budget.phase_upper_bound_subtotal_wall_seconds == 470.05
    assert budget.safety_margin.selected_margin_fraction == 0.25
    assert budget.safety_margin.sdk_timeout_rounding_margin_seconds == 0.4375
    assert budget.safety_margin.selected_upper_bound_seconds == 117.95
    assert budget.predicted_wall_clock_seconds == 588.0
    assert budget.required_A100_seconds == 1176.0
    assert budget.required_A100_seconds == 2 * budget.predicted_wall_clock_seconds
    assert budget.required_A100_seconds != 1800.0
    assert budget.cleanup_upper_bound_seconds == 10.0
    assert all("review" not in phase.phase for phase in budget.phases)
    assert all("calibration" not in phase.phase for phase in budget.phases)


def test_live_budget_gate_uses_required_phase_budget_without_weakening_ceiling() -> None:
    budget = load_v10_phase_budget(PHASE_BUDGET, repository_root=ROOT)
    accepted = evaluate_budget_gate(
        budget,
        current_conservative_usage_A100_seconds=10298.167208919993,
        configured_hard_ceiling_A100_seconds=12600.0,
    )
    assert accepted == {
        "status": "AUTHORIZE_V10",
        "budget_pass": True,
        "current_conservative_usage_A100_seconds": 10298.167208919993,
        "required_A100_seconds": 1176.0,
        "projected_conservative_usage_A100_seconds": 11474.167208919993,
        "configured_hard_ceiling_A100_seconds": 12600.0,
        "remaining_A100_seconds_after_reservation": pytest.approx(1125.8327910800071),
    }

    blocked = evaluate_budget_gate(
        budget,
        current_conservative_usage_A100_seconds=10000.0,
        configured_hard_ceiling_A100_seconds=10800.0,
    )
    assert blocked["status"] == "BUDGET_BLOCKER"
    assert blocked["budget_pass"] is False
    assert blocked["projected_conservative_usage_A100_seconds"] == 11176.0
    assert blocked["configured_hard_ceiling_A100_seconds"] == 10800.0


def test_phase_budget_rejects_arithmetic_or_source_hash_substitution(tmp_path: Path) -> None:
    payload = json.loads(PHASE_BUDGET.read_text())
    payload["required_A100_seconds"] = 900.0
    with pytest.raises(ValidationError, match="required A100-seconds"):
        V10PhaseBudget.model_validate_json(json.dumps(payload), strict=True)

    payload = json.loads(PHASE_BUDGET.read_text())
    payload["phases"][0]["source_measurement"]["artifacts"][0]["artifact_sha256"] = "0" * 64
    substituted = tmp_path / "substituted-phase-budget.json"
    substituted.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="source hash mismatch"):
        load_v10_phase_budget(substituted, repository_root=ROOT)
