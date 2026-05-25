from __future__ import annotations

from collections.abc import Mapping

import pytest

from sloforge.models.service_curve import CalibratedModels
from sloforge.optimizer.core import OptimizationResult, pareto_frontier
from sloforge.profiler.core import ProfileBundle, ProfilingBudget


def test_profile_is_reproducible_and_preserves_raw_samples(
    compiled_inputs: Mapping[str, object],
) -> None:
    profile = compiled_inputs["profile"]
    assert isinstance(profile, ProfileBundle)
    assert profile.raw_measurements
    assert all(item.seed == 7 for item in profile.raw_measurements)
    assert any(item.warmup for item in profile.raw_measurements)
    assert all(item.raw_measurement_ids for item in profile.candidates if item.feasible)
    assert profile.budget.spent_cost_usd <= profile.budget.max_cost_usd
    assert profile.budget.measured_seconds <= profile.budget.max_duration_s


def test_budget_rejects_projected_overrun_without_mutating() -> None:
    budget = ProfilingBudget(max_duration_s=1, max_cost_usd=0.001)
    budget.reserve(duration_s=0.5, hourly_price_usd=1)
    before = budget.model_copy(deep=True)
    with pytest.raises(RuntimeError, match="duration budget exhausted"):
        budget.reserve(duration_s=0.6, hourly_price_usd=1)
    assert budget.measured_seconds == before.measured_seconds
    assert budget.spent_cost_usd == before.spent_cost_usd


def test_monotonic_curves_and_intervals_cover_the_point(
    compiled_inputs: Mapping[str, object],
) -> None:
    models = compiled_inputs["models"]
    assert isinstance(models, CalibratedModels)
    for candidate in models.candidates:
        assert 0 <= candidate.interval_coverage <= 1
        assert 0 <= candidate.decode_interval_coverage <= 1
        assert candidate.decode_held_out_mape >= 0
        for curve in (candidate.prefill, candidate.decode):
            assert curve.calibration_sample_count > 0
            assert curve.nominal_coverage == 0.95
            medians = [point.median_ms for point in curve.points]
            assert medians == sorted(medians)
            for point in curve.points:
                center, lower, upper = curve.predict(point.x)
                assert lower <= center <= upper


def test_optimizer_respects_constraints_and_pareto_invariant(
    compiled_inputs: Mapping[str, object],
) -> None:
    result = compiled_inputs["optimization"]
    assert isinstance(result, OptimizationResult)
    assert result.selected.feasible
    assert all(margin >= 0 for margin in result.selected.constraint_margins.values())
    for constraint in result.request.constraints:
        center = float(getattr(result.selected.predicted, constraint.metric))
        radius = (
            float(getattr(result.selected.uncertainty, constraint.metric))
            * result.request.uncertainty_safety
        )
        expected_margin = (
            constraint.value - (center + radius)
            if constraint.operator == "<="
            else (center - radius) - constraint.value
        )
        assert result.selected.constraint_margins[constraint.metric] == pytest.approx(
            expected_margin
        )
    assert result.selected.configuration.config_id in {
        item.configuration.config_id for item in result.pareto_frontier
    }
    assert pareto_frontier(result.pareto_frontier) == result.pareto_frontier
    assert {item.strategy for item in result.baseline_outcomes} == {
        "exhaustive",
        "random",
        "successive_halving",
        "uncertainty_aware",
    }
    measured = [item for item in result.evaluated_candidates if item.fidelity == "measured"]
    assert measured
    assert all(
        item.configuration.replicas == 1
        and item.configuration.concurrency == 1
        and item.configuration.max_batched_tokens == 2048
        and not item.configuration.chunked_prefill
        and item.configuration.routing_policy == "round_robin"
        for item in measured
    )
    promoted_ids = {step.config_id for step in result.optimizer_history}
    assert result.selected.configuration.config_id in promoted_ids
    assert (
        next(
            item for item in result.baseline_outcomes if item.strategy == "uncertainty_aware"
        ).best_config_id
        is not None
    )
