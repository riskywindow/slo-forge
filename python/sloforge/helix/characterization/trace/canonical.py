"""Canonical encoding and event integrity for characterization traces."""

from __future__ import annotations

import hashlib
import json
from typing import Any, cast

from pydantic import BaseModel

from .models import (
    UNSEALED_CONTENT_HASH,
    BranchWorkloadEventV1,
    StateOperationEventV1,
    TraceEventV1,
    TraceLevel,
)


class TraceIntegrityError(ValueError):
    """A trace record failed its canonical integrity or sequence checks."""


def canonical_json(value: BaseModel | dict[str, Any]) -> bytes:
    """Return deterministic UTF-8 JSON with no insignificant whitespace."""

    document = value.model_dump(mode="json") if isinstance(value, BaseModel) else value
    return json.dumps(
        document,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def canonical_hash(value: BaseModel | dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def event_content_hash(event: TraceEventV1) -> str:
    """Hash an event with its content-hash field replaced by the zero sentinel."""

    payload = event.model_dump(mode="json")
    payload["content_hash"] = UNSEALED_CONTENT_HASH
    return canonical_hash(payload)


def seal_event(
    event: TraceEventV1,
    *,
    event_sequence: int | None = None,
    collection_level: TraceLevel | None = None,
) -> TraceEventV1:
    """Return a canonical immutable event with a deterministic content hash."""

    updates: dict[str, Any] = {"content_hash": UNSEALED_CONTENT_HASH}
    if event_sequence is not None:
        if event_sequence < 0:
            raise ValueError("event_sequence must be non-negative")
        updates["event_sequence"] = event_sequence
    if collection_level is not None:
        updates["collection_level"] = collection_level
    unsealed = event.model_copy(update=updates)
    sealed = unsealed.model_copy(update={"content_hash": event_content_hash(unsealed)})
    if isinstance(event, BranchWorkloadEventV1):
        return cast(BranchWorkloadEventV1, sealed)
    return cast(StateOperationEventV1, sealed)


def verify_event(event: TraceEventV1) -> None:
    expected = event_content_hash(event)
    if event.content_hash != expected:
        raise TraceIntegrityError(
            f"event {event.trace_id}:{event.event_sequence} content hash mismatch: "
            f"expected {expected}, observed {event.content_hash}"
        )


__all__ = [
    "TraceIntegrityError",
    "canonical_hash",
    "canonical_json",
    "event_content_hash",
    "seal_event",
    "verify_event",
]
