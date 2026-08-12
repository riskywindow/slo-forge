"""Streaming conversion to the Perfetto/Chrome trace-event JSON format."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any, TextIO

from .canonical import verify_event
from .models import BranchWorkloadEventV1, TraceEventV1


def _perfetto_event(event: TraceEventV1) -> dict[str, Any]:
    category = "helix.branch" if isinstance(event, BranchWorkloadEventV1) else "continuum.state"
    thread = event.device or (f"rank-{event.rank}" if event.rank is not None else "host")
    record: dict[str, Any] = {
        "name": event.operation_type.value,
        "cat": category,
        "ph": "X" if event.duration_ns else "i",
        "ts": event.normalized_timestamp_ns / 1_000,
        "pid": event.process_id,
        "tid": thread,
        "args": {
            "trace_id": event.trace_id,
            "branch_group_id": event.branch_group_id,
            "branch_id": event.branch_id,
            "event_sequence": event.event_sequence,
            "content_hash": event.content_hash,
            "clock_source": event.clock_source.value,
            "alignment_confidence": event.alignment_confidence,
            "provenance": event.provenance.value,
            "timing_measurement_class": event.timing_measurement_class.value,
        },
    }
    if event.duration_ns:
        record["dur"] = event.duration_ns / 1_000
    else:
        record["s"] = "t"
    if isinstance(event, BranchWorkloadEventV1):
        record["args"].update(
            {
                "logical_bytes": event.logical_bytes,
                "physical_bytes": event.physical_bytes,
                "transferred_bytes": event.transferred_bytes,
                "state_segment": event.state_segment.value,
            }
        )
    else:
        record["args"].update(
            {
                "logical_state_id": event.logical_state_id,
                "bytes": event.bytes,
                "state_segment": event.state_segment.value,
                "result": event.result.value,
            }
        )
    return record


def write_perfetto(path: Path, events: Iterable[TraceEventV1], *, overwrite: bool = False) -> int:
    """Write events incrementally and return the number converted."""

    handle: TextIO
    handle = path.open("w" if overwrite else "x", encoding="utf-8", newline="\n")
    count = 0
    try:
        handle.write('{"displayTimeUnit":"ns","traceEvents":[')
        first = True
        for event in events:
            verify_event(event)
            if not first:
                handle.write(",")
            json.dump(
                _perfetto_event(event),
                handle,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            )
            first = False
            count += 1
        handle.write("]}\n")
    finally:
        handle.close()
    return count


__all__ = ["write_perfetto"]
