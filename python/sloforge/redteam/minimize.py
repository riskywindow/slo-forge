"""Bounded deterministic delta minimization."""

from __future__ import annotations

import random
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Generic, TypeVar

Item = TypeVar("Item")


@dataclass(frozen=True)
class DeltaMinimizationResult(Generic[Item]):
    items: tuple[Item, ...]
    evaluations: int
    timed_out: bool


def minimize_sequence(
    items: Sequence[Item],
    predicate: Callable[[tuple[Item, ...]], bool],
    *,
    seed: int,
    maximum_evaluations: int,
    timeout_seconds: float,
) -> DeltaMinimizationResult[Item]:
    """Minimize an ordered sequence while preserving ``predicate``.

    The seed controls equally valid chunk visitation orders. The original
    sequence is evaluated first; an input that does not reproduce is rejected.
    """

    if seed < 0:
        raise ValueError("seed must be non-negative")
    if maximum_evaluations < 1:
        raise ValueError("maximum_evaluations must be positive")
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    current = tuple(items)
    evaluations = 1
    deadline = time.monotonic() + timeout_seconds
    if not predicate(current):
        raise ValueError("initial sequence does not reproduce the violation")
    randomizer = random.Random(seed)
    granularity = 2
    timed_out = False
    while len(current) >= 2 and evaluations < maximum_evaluations:
        if time.monotonic() >= deadline:
            timed_out = True
            break
        chunk_size = (len(current) + granularity - 1) // granularity
        starts = list(range(0, len(current), chunk_size))
        randomizer.shuffle(starts)
        reduced = False
        for start in starts:
            if evaluations >= maximum_evaluations or time.monotonic() >= deadline:
                timed_out = time.monotonic() >= deadline
                break
            candidate = current[:start] + current[start + chunk_size :]
            if not candidate:
                continue
            evaluations += 1
            if predicate(candidate):
                current = candidate
                granularity = max(2, granularity - 1)
                reduced = True
                break
        if timed_out or evaluations >= maximum_evaluations:
            break
        if not reduced:
            if granularity >= len(current):
                break
            granularity = min(len(current), granularity * 2)
    return DeltaMinimizationResult(
        items=current,
        evaluations=evaluations,
        timed_out=timed_out,
    )


__all__ = ["DeltaMinimizationResult", "minimize_sequence"]
