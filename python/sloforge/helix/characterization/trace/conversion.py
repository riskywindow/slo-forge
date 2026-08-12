"""Integrity-checking conversions among authoritative and derived trace formats."""

from __future__ import annotations

from pathlib import Path

from .io import iter_jsonl, write_jsonl
from .models import TraceArtifactV1
from .parquet import iter_parquet, write_parquet
from .perfetto import write_perfetto


def jsonl_to_parquet(source: Path, destination: Path, *, batch_events: int = 4096) -> int:
    return write_parquet(
        destination,
        iter_jsonl(source),
        batch_events=batch_events,
    )


def parquet_to_jsonl(
    source: Path,
    destination: Path,
    *,
    batch_events: int = 4096,
    overwrite: bool = False,
) -> TraceArtifactV1:
    return write_jsonl(
        destination,
        iter_parquet(source, batch_events=batch_events),
        overwrite=overwrite,
    )


def jsonl_to_perfetto(source: Path, destination: Path, *, overwrite: bool = False) -> int:
    return write_perfetto(destination, iter_jsonl(source), overwrite=overwrite)


def parquet_to_perfetto(
    source: Path,
    destination: Path,
    *,
    batch_events: int = 4096,
    overwrite: bool = False,
) -> int:
    return write_perfetto(
        destination,
        iter_parquet(source, batch_events=batch_events),
        overwrite=overwrite,
    )


__all__ = [
    "jsonl_to_parquet",
    "jsonl_to_perfetto",
    "parquet_to_jsonl",
    "parquet_to_perfetto",
]
