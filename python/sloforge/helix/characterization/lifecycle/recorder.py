"""A deliberately small boundary between lifecycle measurement and trace storage.

The characterization harness emits plain JSON-compatible mappings.  The canonical
BranchWorkloadTrace and StateOperationTrace writers can therefore validate and
persist these records without this hot-path package depending on a storage format.
"""

from __future__ import annotations

import threading
from collections.abc import Mapping
from enum import StrEnum
from typing import Protocol, TypeAlias, runtime_checkable

JsonValue: TypeAlias = (
    bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"] | None
)


class TraceStream(StrEnum):
    BRANCH_WORKLOAD = "branch_workload"
    STATE_OPERATION = "state_operation"


@runtime_checkable
class LifecycleRecorder(Protocol):
    """Consumer for one already measured lifecycle event.

    Implementations must not retain and mutate the input mapping.  Production
    recorders are expected to enqueue quickly and perform serialization elsewhere.
    """

    def record(self, stream: TraceStream, event: Mapping[str, JsonValue]) -> None: ...


class InMemoryLifecycleRecorder:
    """Thread-safe conformance/test recorder with snapshot semantics."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._events: list[tuple[TraceStream, dict[str, JsonValue]]] = []

    def record(self, stream: TraceStream, event: Mapping[str, JsonValue]) -> None:
        with self._lock:
            self._events.append((stream, dict(event)))

    def events(self, stream: TraceStream | None = None) -> tuple[dict[str, JsonValue], ...]:
        with self._lock:
            return tuple(
                dict(event)
                for event_stream, event in self._events
                if stream is None or event_stream is stream
            )

    @property
    def branch_events(self) -> tuple[dict[str, JsonValue], ...]:
        return self.events(TraceStream.BRANCH_WORKLOAD)

    @property
    def state_events(self) -> tuple[dict[str, JsonValue], ...]:
        return self.events(TraceStream.STATE_OPERATION)
