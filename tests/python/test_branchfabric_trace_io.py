from __future__ import annotations

import builtins
import json
from pathlib import Path

import pytest
from test_branchfabric_trace_models import branch_event, state_event

from sloforge.helix.characterization.trace import (
    BoundedTraceBuffer,
    BranchOperationType,
    ParquetUnavailableError,
    SamplingConfigurationV1,
    StateOperationType,
    TraceIntegrityError,
    TraceLevel,
    iter_jsonl,
    jsonl_to_perfetto,
    seal_event,
    write_jsonl,
    write_parquet,
)


def test_buffer_is_bounded_and_overflow_is_observable() -> None:
    buffer = BoundedTraceBuffer(trace_id="trace-1", capacity_events=2, level=TraceLevel.FULL)
    assert buffer.record(branch_event(operation_type=BranchOperationType.BRANCH_FORK))
    assert buffer.record(state_event(operation_type=StateOperationType.STATE_COW))
    assert not buffer.record(branch_event(operation_type=BranchOperationType.BRANCH_COMPLETE))

    stats = buffer.stats()
    assert stats.attempted_events == 3
    assert stats.accepted_events == 2
    assert stats.dropped_events == 1
    assert stats.filtered_events == 0
    assert stats.highest_event_sequence == 2
    assert [event.event_sequence for event in buffer.drain()] == [0, 1]
    assert buffer.stats().buffered_events == 0


def test_disabled_minimal_and_sampling_accounting_preserve_sequence_gaps() -> None:
    disabled = BoundedTraceBuffer(trace_id="trace-1", capacity_events=4, level=TraceLevel.DISABLED)
    assert not disabled.record(branch_event())
    assert disabled.stats().filtered_events == 1
    assert disabled.drain() == ()

    minimal = BoundedTraceBuffer(
        trace_id="trace-1",
        capacity_events=4,
        level=TraceLevel.MINIMAL,
        sampling=SamplingConfigurationV1(mode="deterministic_stride", stride=2, seed=0),
    )
    assert not minimal.record(branch_event(operation_type=BranchOperationType.ROLLOUT))
    assert not minimal.record(branch_event(operation_type=BranchOperationType.BRANCH_FORK))
    assert minimal.record(branch_event(operation_type=BranchOperationType.BRANCH_COMPLETE))
    events = minimal.drain()
    assert len(events) == 1
    assert events[0].event_sequence == 2
    assert events[0].collection_level is TraceLevel.MINIMAL
    assert minimal.stats().filtered_events == 2


def test_jsonl_round_trip_is_streaming_ordered_and_integrity_checked(tmp_path: Path) -> None:
    events = (
        seal_event(branch_event(), event_sequence=index)
        if index % 2 == 0
        else seal_event(state_event(), event_sequence=index)
        for index in range(5_000)
    )
    path = tmp_path / "trace.jsonl"
    artifact = write_jsonl(path, events)
    observed = iter_jsonl(path)
    assert next(observed).event_sequence == 0
    tail = list(observed)
    assert len(tail) == 4_999
    assert tail[-1].event_sequence == 4_999
    assert artifact.event_count == 5_000
    assert artifact.byte_length == path.stat().st_size

    lines = path.read_bytes().splitlines()
    record = json.loads(lines[7])
    record["bytes"] += 1
    lines[7] = json.dumps(record, sort_keys=True, separators=(",", ":")).encode()
    corrupt = tmp_path / "corrupt.jsonl"
    corrupt.write_bytes(b"\n".join(lines) + b"\n")
    with pytest.raises(TraceIntegrityError, match="content hash mismatch"):
        list(iter_jsonl(corrupt))


def test_jsonl_reader_rejects_reordered_records(tmp_path: Path) -> None:
    path = tmp_path / "out-of-order.jsonl"
    write_jsonl(
        path,
        [
            seal_event(branch_event(), event_sequence=2),
            seal_event(branch_event(), event_sequence=1),
        ],
    )
    with pytest.raises(TraceIntegrityError, match="not increasing"):
        list(iter_jsonl(path))


def test_perfetto_conversion_preserves_measurement_provenance(tmp_path: Path) -> None:
    jsonl = tmp_path / "source.jsonl"
    write_jsonl(
        jsonl,
        [
            seal_event(branch_event(), event_sequence=0),
            seal_event(state_event(), event_sequence=1),
        ],
    )
    path = tmp_path / "trace.perfetto.json"
    count = jsonl_to_perfetto(jsonl, path)
    payload = json.loads(path.read_text())
    assert count == 2
    assert payload["displayTimeUnit"] == "ns"
    assert [event["cat"] for event in payload["traceEvents"]] == [
        "helix.branch",
        "continuum.state",
    ]
    assert payload["traceEvents"][0]["args"]["content_hash"]
    assert payload["traceEvents"][1]["args"]["logical_state_id"] == "logical-1"


def test_parquet_dependency_failure_is_explicit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    original_import = builtins.__import__

    def fail_pyarrow(name: str, *args: object, **kwargs: object) -> object:
        if name == "pyarrow" or name.startswith("pyarrow."):
            raise ImportError("intentional test fixture")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fail_pyarrow)
    with pytest.raises(ParquetUnavailableError, match="requires the optional 'pyarrow'"):
        write_parquet(
            tmp_path / "trace.parquet",
            [seal_event(branch_event(), event_sequence=0)],
        )
