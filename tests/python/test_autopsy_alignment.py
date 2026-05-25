from __future__ import annotations

import hashlib

import pytest

from sloforge.autopsy import (
    AlignmentQuality,
    AutopsyEvent,
    AutopsyRun,
    ClockSample,
    CounterValue,
    EventType,
    EvidenceRef,
    ResourceRef,
    SourceClock,
    align_run,
    estimate_alignment,
)

DIGEST = hashlib.sha256(b"fixture").hexdigest()


def _evidence() -> EvidenceRef:
    return EvidenceRef(source="fixture", artifact_uri="tests/fixture.json", sha256=DIGEST)


def _event(host: str = "host-a") -> AutopsyEvent:
    return AutopsyEvent(
        event_id="event-1",
        event_type=EventType.COLLECTIVE,
        host=host,
        rank=0,
        operation="all_reduce",
        start_ns=1_000_000_000,
        end_ns=1_001_000_000,
        source_clock=SourceClock.MONOTONIC_RAW,
        resource=ResourceRef(resource_id="rail-0", resource_type="network_rail"),
        counters=(CounterValue(name="bytes", value=4096.0, unit="bytes"),),
        evidence=_evidence(),
    )


def test_alignment_recovers_offset_and_drift() -> None:
    samples = []
    for index in range(8):
        local = 1_000_000_000 + index * 1_000_000_000
        # 2 ms offset and 5 ppm drift with a symmetric 100 us exchange.
        drift = int((local - 4_000_000_000) * 5 / 1_000_000)
        samples.append(
            ClockSample(
                host="host-a",
                local_monotonic_ns=local,
                reference_monotonic_ns=local + 2_000_000 + drift + 50_000,
                round_trip_ns=100_000,
                captured_wall_ns=10_000_000_000 + local,
            )
        )
    estimate = estimate_alignment(samples, reference_host="host-reference")
    assert estimate.quality is AlignmentQuality.GOOD
    # Offset is expressed at the estimate's reference_local_ns (the fifth
    # sample), which is 1 second beyond the synthetic drift origin.
    assert estimate.offset_ns == pytest.approx(2_005_000.0, abs=2.0)
    assert estimate.drift_ppm == pytest.approx(5.0, abs=0.01)

    run = AutopsyRun(
        run_id="healthy",
        source="synthetic_fixture",
        topology_fingerprint=DIGEST,
        physical_plan_hash=DIGEST,
        workload_fingerprint=DIGEST,
        reference_host="host-reference",
        events=(_event(),),
        artifacts=(_evidence(),),
    )
    aligned = align_run(run, {"host-a": estimate})
    event = aligned.events[0]
    assert event.normalized_start_ns is not None
    assert event.normalized_end_ns is not None
    assert event.normalized_end_ns - event.normalized_start_ns == 1_000_005


def test_alignment_rejects_insufficient_evidence() -> None:
    sample = ClockSample(
        host="host-a",
        local_monotonic_ns=1,
        reference_monotonic_ns=10_000_001,
        round_trip_ns=20_000_000,
        captured_wall_ns=1,
    )
    estimate = estimate_alignment([sample], reference_host="host-reference")
    assert estimate.quality is AlignmentQuality.INSUFFICIENT
    run = AutopsyRun(
        run_id="run",
        source="synthetic_fixture",
        topology_fingerprint=DIGEST,
        physical_plan_hash=DIGEST,
        workload_fingerprint=DIGEST,
        reference_host="host-reference",
        events=(_event(),),
        artifacts=(_evidence(),),
    )
    with pytest.raises(ValueError, match="insufficient"):
        align_run(run, {"host-a": estimate})


def test_event_rejects_invalid_dependencies_and_intervals() -> None:
    with pytest.raises(ValueError, match="end precedes start"):
        _event().model_copy(update={"end_ns": 0}).model_validate(
            {**_event().model_dump(), "end_ns": 0}
        )
    with pytest.raises(ValueError, match="depend on itself"):
        AutopsyEvent.model_validate({**_event().model_dump(), "dependency_event_ids": ("event-1",)})
