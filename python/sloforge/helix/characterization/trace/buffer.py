"""Bounded, thread-safe, non-blocking trace collection buffers."""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from threading import Lock
from typing import Final

from .canonical import seal_event
from .models import (
    BranchOperationType,
    BranchWorkloadEventV1,
    SamplingConfigurationV1,
    StateOperationType,
    TraceEventV1,
    TraceLevel,
)

_MINIMAL_BRANCH_OPERATIONS: Final = frozenset(
    {
        BranchOperationType.BRANCH_POINT,
        BranchOperationType.BRANCH_FORK,
        BranchOperationType.ENVIRONMENT_FORK,
        BranchOperationType.BRANCH_READY,
        BranchOperationType.BRANCH_PRUNE,
        BranchOperationType.BRANCH_ABORT,
        BranchOperationType.BRANCH_COMMIT,
        BranchOperationType.BRANCH_COMPLETE,
        BranchOperationType.BRANCH_MIGRATION,
        BranchOperationType.CHECKPOINT,
        BranchOperationType.STATE_COW,
        BranchOperationType.REWARD,
        BranchOperationType.TRAIN,
        BranchOperationType.PROMOTE,
        BranchOperationType.ROLLBACK,
        BranchOperationType.LEARNING_TRANSACTION_STAGE,
    }
)
_MINIMAL_STATE_OPERATIONS: Final = frozenset(
    {
        StateOperationType.STATE_FORK,
        StateOperationType.STATE_COW,
        StateOperationType.STATE_SNAPSHOT,
        StateOperationType.STATE_DELTA,
        StateOperationType.STATE_SEND,
        StateOperationType.STATE_RECEIVE,
        StateOperationType.STATE_COMMIT,
        StateOperationType.STATE_ABORT,
    }
)


@dataclass(frozen=True)
class TraceBufferStats:
    capacity_events: int
    attempted_events: int
    accepted_events: int
    dropped_events: int
    filtered_events: int
    buffered_events: int
    highest_event_sequence: int | None
    event_counts: dict[str, int]


class BoundedTraceBuffer:
    """A bounded hot-path buffer that never blocks on storage.

    Full buffers reject new events and increment ``dropped_events``.  Sequence
    numbers are assigned to attempts before filtering or overflow, so gaps in a
    persisted stream are direct evidence of omitted records.
    """

    def __init__(
        self,
        *,
        trace_id: str,
        capacity_events: int,
        level: TraceLevel,
        sampling: SamplingConfigurationV1 | None = None,
    ) -> None:
        if not trace_id.strip():
            raise ValueError("trace_id must be non-empty")
        if capacity_events < 1:
            raise ValueError("capacity_events must be positive")
        self.trace_id = trace_id
        self.capacity_events = capacity_events
        self.level = level
        self.sampling = sampling or SamplingConfigurationV1()
        self._events: deque[TraceEventV1] = deque()
        self._lock = Lock()
        self._attempted = 0
        self._accepted = 0
        self._dropped = 0
        self._filtered = 0
        self._event_counts: Counter[str] = Counter()

    def _included_at_level(self, event: TraceEventV1) -> bool:
        if self.level is TraceLevel.DISABLED:
            return False
        if self.level is TraceLevel.FULL:
            return True
        if isinstance(event, BranchWorkloadEventV1):
            return event.operation_type in _MINIMAL_BRANCH_OPERATIONS
        return event.operation_type in _MINIMAL_STATE_OPERATIONS

    def record(self, event: TraceEventV1) -> bool:
        """Attempt to enqueue one event; return whether it was accepted."""

        if event.trace_id != self.trace_id:
            raise ValueError(
                f"event trace_id {event.trace_id!r} does not match buffer {self.trace_id!r}"
            )
        with self._lock:
            sequence = self._attempted
            self._attempted += 1
            sampled = (sequence + self.sampling.seed) % self.sampling.stride == 0
            if not self._included_at_level(event) or not sampled:
                self._filtered += 1
                return False
            if len(self._events) >= self.capacity_events:
                self._dropped += 1
                return False
            sealed = seal_event(
                event,
                event_sequence=sequence,
                collection_level=self.level,
            )
            self._events.append(sealed)
            self._accepted += 1
            self._event_counts[sealed.operation_type.value] += 1
            return True

    def drain(self, *, max_events: int | None = None) -> tuple[TraceEventV1, ...]:
        """Remove a bounded batch in original event order."""

        if max_events is not None and max_events < 1:
            raise ValueError("max_events must be positive when supplied")
        with self._lock:
            count = len(self._events) if max_events is None else min(max_events, len(self._events))
            return tuple(self._events.popleft() for _ in range(count))

    def stats(self) -> TraceBufferStats:
        with self._lock:
            return TraceBufferStats(
                capacity_events=self.capacity_events,
                attempted_events=self._attempted,
                accepted_events=self._accepted,
                dropped_events=self._dropped,
                filtered_events=self._filtered,
                buffered_events=len(self._events),
                highest_event_sequence=self._attempted - 1 if self._attempted else None,
                event_counts=dict(sorted(self._event_counts.items())),
            )


__all__ = ["BoundedTraceBuffer", "TraceBufferStats"]
