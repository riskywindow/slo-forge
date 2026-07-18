import pytest

from sloforge.helix.characterization.prioritizer import (
    ExperimentCandidate,
    prioritize_experiments,
)


def _candidate(
    experiment_id: str,
    *,
    uncertainty: float,
    leverage: float,
    cost: float = 0.0,
    accessible: bool = True,
) -> ExperimentCandidate:
    return ExperimentCandidate(
        experiment_id=experiment_id,
        question="page_size",
        uncertainty=uncertainty,
        sensitivity=1.0,
        amdahl_leverage=leverage,
        measurement_cost_usd=cost,
        wall_time_minutes=10.0,
        accessible_hardware=accessible,
        evidence_gap="controlled test gap",
        expected_artifacts=(f"artifacts/{experiment_id}.json",),
    )


def test_prioritizer_is_deterministic_transparent_and_budget_bounded() -> None:
    candidates = (
        _candidate("cpu-page", uncertainty=0.8, leverage=0.8),
        _candidate("gpu-page", uncertainty=1.0, leverage=1.0, accessible=False),
        _candidate("paid-page", uncertainty=1.0, leverage=1.0, cost=90.0),
        _candidate("cpu-low", uncertainty=0.2, leverage=0.2),
    )
    first = prioritize_experiments(candidates, seed=41, gpu_budget_usd=100.0)
    second = prioritize_experiments(candidates, seed=41, gpu_budget_usd=100.0)

    assert first == second
    assert first.spendable_budget_usd == pytest.approx(85.0)
    assert [item.experiment_id for item in first.ranked] == [
        "cpu-page",
        "cpu-low",
        "gpu-page",
        "paid-page",
    ]
    assert [item.rank for item in first.ranked] == [1, 2, 3, 4]
    assert first.ranked[0].reason.startswith("ELIGIBLE")
    assert all(not item.eligible and item.score == 0 for item in first.ranked[2:])


def test_prioritizer_rejects_duplicates_and_keeps_rerun_reserve() -> None:
    candidate = _candidate("same", uncertainty=0.5, leverage=0.5)
    with pytest.raises(ValueError, match="unique"):
        prioritize_experiments((candidate, candidate), seed=7, gpu_budget_usd=0.0)
    with pytest.raises(ValueError, match=r"0\.15"):
        prioritize_experiments(
            (candidate,), seed=7, gpu_budget_usd=0.0, reserved_rerun_fraction=0.1
        )
