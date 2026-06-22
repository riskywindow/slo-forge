"""Bounded Pareto archive over the full Genesis serving objective vector."""

from __future__ import annotations

from dataclasses import dataclass

from .models import CandidateDesign, ObjectiveVector


@dataclass(frozen=True)
class ArchiveEntry:
    candidate: CandidateDesign
    objective: ObjectiveVector


def dominates(left: ObjectiveVector, right: ObjectiveVector) -> bool:
    left_values = left.desirability()
    right_values = right.desirability()
    return all(a >= b for a, b in zip(left_values, right_values, strict=True)) and any(
        a > b for a, b in zip(left_values, right_values, strict=True)
    )


def _crowding(entries: list[ArchiveEntry]) -> dict[str, float]:
    distances = {entry.candidate.candidate_id: 0.0 for entry in entries}
    if len(entries) <= 2:
        return {identifier: float("inf") for identifier in distances}
    width = len(entries[0].objective.desirability())
    for dimension in range(width):
        ordered = sorted(
            entries,
            key=lambda entry: (
                entry.objective.desirability()[dimension],
                entry.candidate.candidate_id,
            ),
        )
        low = ordered[0].objective.desirability()[dimension]
        high = ordered[-1].objective.desirability()[dimension]
        distances[ordered[0].candidate.candidate_id] = float("inf")
        distances[ordered[-1].candidate.candidate_id] = float("inf")
        if high == low:
            continue
        for index in range(1, len(ordered) - 1):
            previous = ordered[index - 1].objective.desirability()[dimension]
            following = ordered[index + 1].objective.desirability()[dimension]
            identifier = ordered[index].candidate.candidate_id
            distances[identifier] += (following - previous) / (high - low)
    return distances


class ParetoArchive:
    def __init__(self, maximum_size: int) -> None:
        if maximum_size <= 0:
            raise ValueError("maximum Pareto archive size must be positive")
        self.maximum_size = maximum_size
        self._entries: list[ArchiveEntry] = []

    @property
    def entries(self) -> tuple[ArchiveEntry, ...]:
        return tuple(sorted(self._entries, key=lambda item: item.candidate.candidate_id))

    def add(self, candidate: CandidateDesign, objective: ObjectiveVector) -> bool:
        if any(entry.candidate.candidate_id == candidate.candidate_id for entry in self._entries):
            return False
        if any(dominates(entry.objective, objective) for entry in self._entries):
            return False
        self._entries = [
            entry for entry in self._entries if not dominates(objective, entry.objective)
        ]
        self._entries.append(ArchiveEntry(candidate=candidate, objective=objective))
        if len(self._entries) > self.maximum_size:
            crowding = _crowding(self._entries)
            remove = min(
                self._entries,
                key=lambda entry: (
                    crowding[entry.candidate.candidate_id],
                    entry.candidate.candidate_id,
                ),
            )
            self._entries.remove(remove)
        return any(
            entry.candidate.candidate_id == candidate.candidate_id for entry in self._entries
        )
