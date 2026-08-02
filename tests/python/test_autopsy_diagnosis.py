from __future__ import annotations

import hashlib

from sloforge.autopsy import (
    AlignmentEstimate,
    AlignmentQuality,
    AutopsyEvent,
    AutopsyRun,
    BottleneckKind,
    CounterValue,
    EventType,
    EvidenceRef,
    ResourceRef,
    SourceClock,
    compare_runs,
    diagnose,
)

DIGEST = hashlib.sha256(b"fabric-fixture").hexdigest()


def _evidence() -> EvidenceRef:
    return EvidenceRef(source="fabric-simulator", artifact_uri="artifacts/run.json", sha256=DIGEST)


def _alignment(host: str) -> AlignmentEstimate:
    return AlignmentEstimate(
        host=host,
        reference_host="host-a",
        offset_ns=0.0,
        drift_ppm=0.0,
        reference_local_ns=0,
        uncertainty_ns=20_000,
        sample_count=8,
        confidence=0.9,
        quality=AlignmentQuality.GOOD,
        residual_p95_ns=5_000,
    )


def _event(
    event_id: str,
    event_type: EventType,
    duration_ms: float,
    *,
    rank: int,
    counter_name: str | None = None,
    counter_value: float = 0.0,
) -> AutopsyEvent:
    counters = (
        ()
        if counter_name is None
        else (CounterValue(name=counter_name, value=counter_value, unit="ratio"),)
    )
    return AutopsyEvent(
        event_id=event_id,
        event_type=event_type,
        host="host-a" if rank < 4 else "host-b",
        rank=rank,
        request_id="request-1",
        operation="expert_all_to_all",
        start_ns=1_000_000_000 + rank * 10_000,
        end_ns=1_000_000_000 + rank * 10_000 + round(duration_ms * 1_000_000),
        source_clock=SourceClock.SYNTHETIC,
        normalized_start_ns=1_000_000_000 + rank * 10_000,
        normalized_end_ns=1_000_000_000 + rank * 10_000 + round(duration_ms * 1_000_000),
        alignment_confidence=0.9,
        alignment_uncertainty_ns=20_000,
        resource=ResourceRef(
            resource_id="rail-0", resource_type="network_rail", contention_domain="rail-0"
        ),
        counters=counters,
        evidence=_evidence(),
    )


def _run(run_id: str, durations: list[float], bandwidth: float) -> AutopsyRun:
    return AutopsyRun(
        run_id=run_id,
        source="synthetic_fixture",
        topology_fingerprint=DIGEST,
        physical_plan_hash=DIGEST,
        workload_fingerprint=DIGEST,
        reference_host="host-a",
        events=tuple(
            _event(
                f"{run_id}-{rank}",
                EventType.NETWORK_TRANSFER if rank == 0 else EventType.COLLECTIVE_WAIT,
                duration,
                rank=rank,
                counter_name="network_bandwidth_gbps" if rank == 0 else None,
                counter_value=bandwidth,
            )
            for rank, duration in enumerate(durations)
        ),
        alignments=(_alignment("host-a"), _alignment("host-b")),
        artifacts=(_evidence(),),
    )


def test_network_degradation_and_rank_skew_are_ranked() -> None:
    healthy = _run("healthy", [1.0, 1.0, 1.0, 1.0, 1.0], 100.0)
    degraded = _run("degraded", [3.0, 1.0, 1.0, 1.0, 3.0], 50.0)
    comparison = compare_runs(healthy, degraded)
    assert comparison.matched_event_count == 5
    assert comparison.maximum_rank_skew == 3.0
    assert comparison.first_divergence_event_id == "degraded-0"

    result = diagnose(degraded, comparison=comparison, baseline=healthy)
    assert result.top_hypothesis is BottleneckKind.NETWORK_BANDWIDTH_DEGRADATION
    assert BottleneckKind.RANK_STRAGGLER in result.top_three
    assert result.sufficient_alignment
    network = next(
        item
        for item in result.hypotheses
        if item.kind is BottleneckKind.NETWORK_BANDWIDTH_DEGRADATION
    )
    assert network.supporting_evidence
    assert network.rejected_reason is None


def test_alternative_hypotheses_record_contradictions() -> None:
    healthy = _run("healthy", [1.0] * 5, 100.0)
    degraded = _run("degraded", [1.0] * 5, 100.0)
    result = diagnose(degraded, comparison=compare_runs(healthy, degraded), baseline=healthy)
    gpu_clock = next(
        item for item in result.hypotheses if item.kind is BottleneckKind.GPU_CLOCK_THROTTLING
    )
    assert gpu_clock.contradicting_evidence
    assert gpu_clock.rejected_reason == "primary signal did not cross its threshold"
