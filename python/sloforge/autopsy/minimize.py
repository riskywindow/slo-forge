"""Deterministic delta debugging for compact Autopsy reproducers."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TypeVar

from pydantic import Field

from sloforge.util import canonical_json, sha256_bytes

from .models import AutopsyEvent, AutopsyRun, NonEmpty, StrictModel

DiagnosisPredicate = Callable[[AutopsyRun], bool]
Item = TypeVar("Item")


class MinimizationResult(StrictModel):
    schema_version: str = "sloforge.autopsy.minimization/v1"
    original_run_id: NonEmpty
    minimized_run: AutopsyRun
    original_event_count: int = Field(ge=0)
    minimized_event_count: int = Field(ge=0)
    original_rank_count: int = Field(ge=0)
    minimized_rank_count: int = Field(ge=0)
    original_counter_count: int = Field(ge=0)
    minimized_counter_count: int = Field(ge=0)
    removed_event_ids: tuple[str, ...]
    removed_ranks: tuple[int, ...]
    removed_counters: tuple[str, ...]
    predicate_evaluations: int = Field(ge=1)
    bundle_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


def _with_events(run: AutopsyRun, events: Sequence[AutopsyEvent]) -> AutopsyRun:
    retained = {event.event_id for event in events}
    repaired = tuple(
        AutopsyEvent.model_validate(
            {
                **event.model_dump(mode="python"),
                "parent_event_id": (
                    event.parent_event_id if event.parent_event_id in retained else None
                ),
                "dependency_event_ids": tuple(
                    item for item in event.dependency_event_ids if item in retained
                ),
            }
        )
        for event in events
    )
    return AutopsyRun.model_validate({**run.model_dump(mode="python"), "events": repaired})


def _chunks(items: Sequence[Item], count: int) -> tuple[tuple[Item, ...], ...]:
    size, extra = divmod(len(items), count)
    result: list[tuple[Item, ...]] = []
    start = 0
    for index in range(count):
        width = size + (1 if index < extra else 0)
        if width:
            result.append(tuple(items[start : start + width]))
        start += width
    return tuple(result)


def _ddmin(
    items: Sequence[Item],
    accepts: Callable[[tuple[Item, ...]], bool],
) -> tuple[Item, ...]:
    current = tuple(items)
    granularity = 2
    while current:
        pieces = _chunks(current, min(granularity, len(current)))
        reduced = False
        for piece in pieces:
            removed = set(map(id, piece))
            candidate = tuple(item for item in current if id(item) not in removed)
            if accepts(candidate):
                current = candidate
                granularity = max(2, granularity - 1)
                reduced = True
                break
        if reduced:
            continue
        if granularity >= len(current):
            break
        granularity = min(len(current), granularity * 2)
    return current


def _ranks(run: AutopsyRun) -> tuple[int, ...]:
    return tuple(sorted({event.rank for event in run.events if event.rank is not None}))


def _counter_keys(run: AutopsyRun) -> tuple[tuple[str, str], ...]:
    return tuple(
        (event.event_id, counter.name) for event in run.events for counter in event.counters
    )


def minimize_run(run: AutopsyRun, predicate: DiagnosisPredicate) -> MinimizationResult:
    """Reduce ranks, events, and counters while preserving ``predicate``.

    The predicate is evaluated on a fully validated event graph at every step.
    Dependency edges to removed events are pruned rather than left dangling.
    """
    evaluations = 1
    if not predicate(run):
        raise ValueError("the diagnosis predicate does not hold for the source run")
    original_events = tuple(run.events)
    original_ranks = _ranks(run)
    original_counters = _counter_keys(run)
    current = run

    for rank in _ranks(current):
        candidate = _with_events(
            current, tuple(event for event in current.events if event.rank != rank)
        )
        evaluations += 1
        if predicate(candidate):
            current = candidate

    source_events = tuple(current.events)

    def accepts_events(events: tuple[AutopsyEvent, ...]) -> bool:
        nonlocal evaluations
        evaluations += 1
        return predicate(_with_events(current, events))

    minimized_events = _ddmin(source_events, accepts_events)
    current = _with_events(current, minimized_events)

    counter_keys = _counter_keys(current)

    def accepts_counters(keys: tuple[tuple[str, str], ...]) -> bool:
        nonlocal evaluations
        keep = set(keys)
        events = tuple(
            AutopsyEvent.model_validate(
                {
                    **event.model_dump(mode="python"),
                    "counters": tuple(
                        counter
                        for counter in event.counters
                        if (event.event_id, counter.name) in keep
                    ),
                }
            )
            for event in current.events
        )
        candidate = _with_events(current, events)
        evaluations += 1
        return predicate(candidate)

    retained_counters = _ddmin(counter_keys, accepts_counters)
    keep_counters = set(retained_counters)
    final_events = tuple(
        AutopsyEvent.model_validate(
            {
                **event.model_dump(mode="python"),
                "counters": tuple(
                    counter
                    for counter in event.counters
                    if (event.event_id, counter.name) in keep_counters
                ),
            }
        )
        for event in current.events
    )
    current = _with_events(current, final_events)
    digest = sha256_bytes(canonical_json(current.model_dump(mode="json")).encode())
    minimized = AutopsyRun.model_validate(
        {
            **current.model_dump(mode="python"),
            "run_id": f"{run.run_id}-min-{digest[:12]}",
            "warnings": (*current.warnings, "deterministically minimized Autopsy reproducer"),
        }
    )
    retained_ids = {event.event_id for event in minimized.events}
    minimized_counter_keys = set(_counter_keys(minimized))
    return MinimizationResult(
        original_run_id=run.run_id,
        minimized_run=minimized,
        original_event_count=len(original_events),
        minimized_event_count=len(minimized.events),
        original_rank_count=len(original_ranks),
        minimized_rank_count=len(_ranks(minimized)),
        original_counter_count=len(original_counters),
        minimized_counter_count=len(minimized_counter_keys),
        removed_event_ids=tuple(
            event.event_id for event in original_events if event.event_id not in retained_ids
        ),
        removed_ranks=tuple(rank for rank in original_ranks if rank not in _ranks(minimized)),
        removed_counters=tuple(
            f"{event_id}:{counter_name}"
            for event_id, counter_name in original_counters
            if (event_id, counter_name) not in minimized_counter_keys
        ),
        predicate_evaluations=evaluations,
        bundle_sha256=digest,
    )
