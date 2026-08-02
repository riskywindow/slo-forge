"""Deterministic distance-weighted surrogate experiment selection."""

from __future__ import annotations

import math
from dataclasses import dataclass

from .models import CandidateDesign


@dataclass(frozen=True)
class SurrogateObservation:
    feature_vector: tuple[float, ...]
    utility: float

    def __post_init__(self) -> None:
        if not self.feature_vector or not math.isfinite(self.utility):
            raise ValueError("surrogate observation must be finite and non-empty")
        if not all(math.isfinite(value) for value in self.feature_vector):
            raise ValueError("surrogate observation features must be finite")


@dataclass(frozen=True)
class ExperimentScore:
    candidate: CandidateDesign
    predicted_utility: float
    uncertainty: float
    acquisition: float


def _distance(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    if len(left) != len(right):
        raise ValueError("surrogate feature dimensions must match")
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(left, right, strict=True)))


class DeterministicSurrogateRanker:
    def __init__(self, *, exploration_weight: float = 0.25) -> None:
        if exploration_weight < 0 or not math.isfinite(exploration_weight):
            raise ValueError("exploration_weight must be finite and non-negative")
        self.exploration_weight = exploration_weight

    def rank(
        self,
        candidates: tuple[CandidateDesign, ...],
        observations: tuple[SurrogateObservation, ...],
    ) -> tuple[ExperimentScore, ...]:
        scores: list[ExperimentScore] = []
        for candidate in candidates:
            if observations:
                distances = [
                    _distance(candidate.feature_vector, observation.feature_vector)
                    for observation in observations
                ]
                weights = [1.0 / (distance + 1e-9) for distance in distances]
                predicted = sum(
                    weight * observation.utility
                    for weight, observation in zip(weights, observations, strict=True)
                ) / sum(weights)
                uncertainty = min(distances)
            else:
                predicted = sum(
                    mutation.expected_upside - mutation.invalidity_risk
                    for mutation in candidate.mutations
                )
                uncertainty = math.sqrt(sum(value * value for value in candidate.feature_vector))
            acquisition = predicted + self.exploration_weight * uncertainty
            scores.append(
                ExperimentScore(
                    candidate=candidate,
                    predicted_utility=predicted,
                    uncertainty=uncertainty,
                    acquisition=acquisition,
                )
            )
        scores.sort(key=lambda item: (-item.acquisition, item.candidate.candidate_id))
        return tuple(scores)
