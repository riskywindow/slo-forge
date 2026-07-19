"""Optional streaming Parquet conversion.

The base SLOForge installation deliberately does not depend on PyArrow.  Callers
receive an explicit error rather than a silent JSONL fallback when it is absent.
"""

from __future__ import annotations

import importlib
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from .canonical import canonical_json, verify_event
from .io import _parse_event
from .models import TraceEventV1


class ParquetUnavailableError(RuntimeError):
    """PyArrow is not installed, so Parquet was not produced or consumed."""


def _pyarrow() -> tuple[Any, Any]:
    try:
        pa = importlib.import_module("pyarrow")
        pq = importlib.import_module("pyarrow.parquet")
    except ImportError as error:
        raise ParquetUnavailableError(
            "Parquet trace support requires the optional 'pyarrow' package; "
            "the JSONL trace remains authoritative"
        ) from error
    return pa, pq


def write_parquet(
    path: Path,
    events: Iterable[TraceEventV1],
    *,
    batch_events: int = 4096,
) -> int:
    """Write bounded batches with canonical JSON payloads and indexed columns."""

    if batch_events < 1:
        raise ValueError("batch_events must be positive")
    pa, pq = _pyarrow()
    schema = pa.schema(
        [
            ("kind", pa.string()),
            ("trace_id", pa.string()),
            ("event_sequence", pa.uint64()),
            ("monotonic_timestamp_ns", pa.uint64()),
            ("operation_type", pa.string()),
            ("event_json", pa.binary()),
        ]
    )
    writer = pq.ParquetWriter(path, schema, compression="zstd")
    rows: list[dict[str, Any]] = []
    count = 0
    try:
        for event in events:
            verify_event(event)
            rows.append(
                {
                    "kind": event.kind,
                    "trace_id": event.trace_id,
                    "event_sequence": event.event_sequence,
                    "monotonic_timestamp_ns": event.monotonic_timestamp_ns,
                    "operation_type": event.operation_type.value,
                    "event_json": canonical_json(event),
                }
            )
            count += 1
            if len(rows) >= batch_events:
                writer.write_table(pa.Table.from_pylist(rows, schema=schema))
                rows.clear()
        if rows:
            writer.write_table(pa.Table.from_pylist(rows, schema=schema))
    finally:
        writer.close()
    return count


def iter_parquet(path: Path, *, batch_events: int = 4096) -> Iterator[TraceEventV1]:
    if batch_events < 1:
        raise ValueError("batch_events must be positive")
    _, pq = _pyarrow()
    parquet_file = pq.ParquetFile(path)
    for batch in parquet_file.iter_batches(batch_size=batch_events, columns=["event_json"]):
        for payload in batch.column(0).to_pylist():
            event = _parse_event(payload)
            verify_event(event)
            yield event


__all__ = ["ParquetUnavailableError", "iter_parquet", "write_parquet"]
