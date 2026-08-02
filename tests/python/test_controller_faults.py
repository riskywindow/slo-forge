from __future__ import annotations

from pathlib import Path

import pytest

from sloforge.controller import ControllerConfig, PredictiveController, evaluate_controllers
from sloforge.controller.core import ObservedState
from sloforge.faults import execute_scenario, load_scenario
from sloforge.trace.format import generate_bursty_trace


def test_controller_actions_stay_inside_guards_and_emit_outcomes() -> None:
    trace = generate_bursty_trace(seed=41, count=120)
    config = ControllerConfig(min_replicas=1, max_replicas=4, max_concurrency=8)
    result = evaluate_controllers(trace, config)
    assert result.predictive_decisions
    for decision in result.predictive_decisions:
        assert 1 <= decision.chosen.target_replicas <= 4
        assert 1 <= decision.chosen.target_concurrency <= 8
        assert decision.actual_outcome_p95_ttft_ms is not None
        assert decision.safety_checks
    assert result.predictive.action_count > 0
    assert result.reactive.action_count > 0


def test_all_required_fault_labels_are_diagnosed() -> None:
    root = Path(__file__).resolve().parents[2]
    result = execute_scenario(load_scenario(root / "scenarios" / "cpu-demo.yaml"))
    assert len(result.executions) == 8
    assert result.diagnosis_accuracy == 1.0
    assert all(item.diagnosis.correct for item in result.executions)
    assert all(item.diagnosis.evidence for item in result.executions)
    assert all(item.diagnosis.counterfactual_improvement_ms > 0 for item in result.executions)
    assert result.negative_window_count >= 8
    assert result.false_positive_count == 0
    assert result.false_positive_rate == 0
    assert sum(sum(row.values()) for row in result.confusion_matrix.values()) == 8


def test_fault_loader_rejects_yaml_references(tmp_path: Path) -> None:
    scenario = tmp_path / "alias.yaml"
    scenario.write_text(
        "schema_version: &version sloforge.chaos/v1\nname: fixture\nseed: 1\nevents: *version\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="anchors or aliases"):
        load_scenario(scenario)


def test_material_routing_change_canaries_and_rolls_back() -> None:
    config = ControllerConfig(
        min_replicas=1,
        max_replicas=1,
        initial_replicas=1,
        min_concurrency=2,
        max_concurrency=2,
        initial_concurrency=2,
        capacity_per_replica_rps=20,
        min_samples=8,
        cooldown_windows=0,
        calibrated_base_ttft_ms=20,
    )
    controller = PredictiveController(config)

    def observed(index: int, rate: float) -> ObservedState:
        return ObservedState(
            window_index=index,
            window_start_ms=index * 1000,
            window_end_ms=(index + 1) * 1000,
            sample_count=8,
            arrival_rate_rps=rate,
            interactive_fraction=0.75,
            long_context_fraction=0.25,
            prompt_p95=512,
            output_p95=64,
            observed_p95_ttft_ms=30,
            backend_error_rate=0,
            replicas=1,
            concurrency=2,
        )

    controller.record_outcome(controller.decide(observed(0, 2)), 30)
    canary = controller.decide(observed(1, 8))
    assert canary.canary
    assert canary.chosen.action_type == "change_routing"
    outcome = controller.record_outcome(canary, 1000)
    assert outcome.rolled_back
    assert outcome.rollback_reason
    assert controller.routing_policy == "round_robin"
