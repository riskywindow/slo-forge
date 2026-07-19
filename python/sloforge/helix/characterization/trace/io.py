"""Streaming JSONL persistence for million-event characterization traces."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import BinaryIO

from pydantic import ValidationError

from .canonical import TraceIntegrityError, canonical_json, verify_event
from .models import BranchWorkloadEventV1, StateOperationEventV1, TraceArtifactV1, TraceEventV1

MAX_EVENT_LINE_BYTES = 16 * 1024 * 1024
DEFAULT_BATCH_EVENTS = 1024
DEFAULT_BATCH_BYTES = 8 * 1024 * 1024


class TraceFormatError(ValueError):
    """A trace stream has an unsupported or malformed record."""


def _parse_event(payload: bytes) -> TraceEventV1:
    try:
        discriminator = json.loads(payload).get("kind")
        if discriminator == "BranchWorkloadTraceEvent":
            return BranchWorkloadEventV1.model_validate_json(payload)
        if discriminator == "StateOperationTraceEvent":
            return StateOperationEventV1.model_validate_json(payload)
    except (AttributeError, json.JSONDecodeError, ValidationError, ValueError) as error:
        raise TraceFormatError(f"invalid trace event: {error}") from error
    raise TraceFormatError(f"unsupported trace event kind: {discriminator!r}")


class JsonlTraceWriter:
    """Batching writer with incremental integrity and artifact accounting."""

    def __init__(
        self,
        path: Path,
        *,
        batch_events: int = DEFAULT_BATCH_EVENTS,
        batch_bytes: int = DEFAULT_BATCH_BYTES,
        overwrite: bool = False,
    ) -> None:
        if batch_events < 1 or batch_bytes < 1:
            raise ValueError("batch limits must be positive")
        self.path = path
        self.batch_events = batch_events
        self.batch_bytes = batch_bytes
        self._handle: BinaryIO = path.open("wb" if overwrite else "xb")
        self._pending: list[bytes] = []
        self._pending_bytes = 0
        self._event_count = 0
        self._byte_length = 0
        self._digest = hashlib.sha256()
        self._closed = False

    def write(self, event: TraceEventV1) -> None:
        if self._closed:
            raise ValueError("trace writer is closed")
        verify_event(event)
        encoded = canonical_json(event) + b"\n"
        if len(encoded) > MAX_EVENT_LINE_BYTES:
            raise TraceFormatError("trace event exceeds bounded JSONL record size")
        self._pending.append(encoded)
        self._pending_bytes += len(encoded)
        self._event_count += 1
        if len(self._pending) >= self.batch_events or self._pending_bytes >= self.batch_bytes:
            self.flush()

    def write_many(self, events: Iterable[TraceEventV1]) -> None:
        for event in events:
            self.write(event)

    def flush(self) -> None:
        if self._closed:
            raise ValueError("trace writer is closed")
        if not self._pending:
            return
        block = b"".join(self._pending)
        self._handle.write(block)
        self._handle.flush()
        self._digest.update(block)
        self._byte_length += len(block)
        self._pending.clear()
        self._pending_bytes = 0

    def close(self) -> None:
        if self._closed:
            return
        self.flush()
        self._handle.close()
        self._closed = True

    def artifact(self) -> TraceArtifactV1:
        if not self._closed:
            raise ValueError("close trace writer before creating its artifact record")
        return TraceArtifactV1(
            format="jsonl",
            uri=str(self.path),
            byte_length=self._byte_length,
            sha256=self._digest.hexdigest(),
            event_count=self._event_count,
        )

    def __enter__(self) -> JsonlTraceWriter:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()


def iter_jsonl(
    path: Path,
    *,
    verify_integrity: bool = True,
    verify_order: bool = True,
) -> Iterator[TraceEventV1]:
    """Stream validated events without loading the trace corpus into memory."""

    last_sequences: dict[str, int] = {}
    with path.open("rb") as handle:
        for line_number, line in enumerate(handle, start=1):
            if len(line) > MAX_EVENT_LINE_BYTES:
                raise TraceFormatError(f"line {line_number} exceeds bounded record size")
            if not line.strip():
                raise TraceFormatError(f"line {line_number} is empty")
            try:
                event = _parse_event(line)
                if verify_integrity:
                    verify_event(event)
                if verify_order:
                    previous = last_sequences.get(event.trace_id)
                    if previous is not None and event.event_sequence <= previous:
                        raise TraceIntegrityError(
                            f"event sequence is not increasing at line {line_number}: "
                            f"{event.event_sequence} follows {previous}"
                        )
                    last_sequences[event.trace_id] = event.event_sequence
            except (TraceFormatError, TraceIntegrityError) as error:
                raise type(error)(f"{path}:{line_number}: {error}") from error
            yield event


def write_jsonl(
    path: Path,
    events: Iterable[TraceEventV1],
    *,
    overwrite: bool = False,
) -> TraceArtifactV1:
    writer = JsonlTraceWriter(path, overwrite=overwrite)
    try:
        writer.write_many(events)
    finally:
        writer.close()
    return writer.artifact()


__all__ = [
    "DEFAULT_BATCH_BYTES",
    "DEFAULT_BATCH_EVENTS",
    "MAX_EVENT_LINE_BYTES",
    "JsonlTraceWriter",
    "TraceFormatError",
    "iter_jsonl",
    "write_jsonl",
]
