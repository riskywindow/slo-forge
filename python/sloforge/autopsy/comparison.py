"""Matched healthy/degraded differential analysis."""

from __future__ import annotations

import hashlib
import math
import statistics
from collections import defaultdict

from .models import (
    AlignmentQuality,
    AutopsyEvent,
    AutopsyRun,
    CounterDelta,
    DifferentialComparison,
    EventType,
    StageDelta,
)


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    index = (len(ordered) - 1) * quantile
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    weight = index - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _signature(event: AutopsyEvent) -> tuple[EventType, str | None, int | None, str | None]:
    return (event.event_type, event.operation, event.rank, event.request_id)


def _signature_text(signature: tuple[EventType, str | None, int | None, str | None]) -> str:
    event_type, operation, rank, request_id = signature
    return f"{event_type.value}:{operation or '-'}:rank={rank}:request={request_id or '-'}"


def _relative_delta(healthy: float, degraded: float) -> float:
    if healthy > 0.0:
        return (degraded - healthy) / healthy
    return 0.0 if degraded == 0.0 else degraded


def _counter_groups(run: AutopsyRun) -> dict[tuple[str, str], list[float]]:
    values: dict[tuple[str, str], list[float]] = defaultdict(list)
    for event in run.events:
        for counter in event.counters:
            values[(counter.name, counter.unit)].append(counter.value)
    return values


def _rank_skew(run: AutopsyRun) -> float:
    groups: dict[tuple[EventType, str | None, str | None], dict[int, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for event in run.events:
        if event.rank is None or event.event_type not in {
            EventType.COLLECTIVE,
            EventType.COLLECTIVE_WAIT,
            EventType.DECODE,
            EventType.EXPERT_DISPATCH,
            EventType.EXPERT_COMBINE,
            EventType.GPU_COMPUTE,
            EventType.KV_TRANSFER,
            EventType.NETWORK_TRANSFER,
            EventType.NVLINK_TRANSFER,
            EventType.PCIE_TRANSFER,
            EventType.PREFILL,
        }:
            continue
        groups[(event.event_type, event.operation, event.request_id)][event.rank].append(
            event.duration_ns / 1_000_000.0
        )
    maximum = 1.0
    for ranks in groups.values():
        medians = [statistics.median(values) for values in ranks.values() if values]
        if len(medians) >= 2 and min(medians) > 0.0:
            # ``median_low`` keeps the two-rank case sensitive to one straggler;
            # the arithmetic midpoint would dilute a 2.5x skew to 1.43x.
            maximum = max(maximum, max(medians) / statistics.median_low(medians))
    return maximum


def compare_runs(healthy: AutopsyRun, degraded: AutopsyRun) -> DifferentialComparison:
    if healthy.workload_fingerprint != degraded.workload_fingerprint:
        raise ValueError("healthy and degraded runs must use the same workload fingerprint")
    healthy_groups: dict[
        tuple[EventType, str | None, int | None, str | None], list[AutopsyEvent]
    ] = defaultdict(list)
    degraded_groups: dict[
        tuple[EventType, str | None, int | None, str | None], list[AutopsyEvent]
    ] = defaultdict(list)
    for event in healthy.events:
        healthy_groups[_signature(event)].append(event)
    for event in degraded.events:
        degraded_groups[_signature(event)].append(event)

    stage_deltas: list[StageDelta] = []
    matched = 0
    unmatched_healthy = 0
    unmatched_degraded = 0
    first_divergence: tuple[int, str] | None = None
    for signature in sorted(
        set(healthy_groups) | set(degraded_groups), key=lambda item: _signature_text(item)
    ):
        healthy_events = sorted(healthy_groups[signature], key=lambda event: event.start_ns)
        degraded_events = sorted(degraded_groups[signature], key=lambda event: event.start_ns)
        pair_count = min(len(healthy_events), len(degraded_events))
        matched += pair_count
        unmatched_healthy += len(healthy_events) - pair_count
        unmatched_degraded += len(degraded_events) - pair_count
        healthy_ms = [event.duration_ns / 1_000_000.0 for event in healthy_events]
        degraded_ms = [event.duration_ns / 1_000_000.0 for event in degraded_events]
        healthy_median = statistics.median(healthy_ms) if healthy_ms else 0.0
        degraded_median = statistics.median(degraded_ms) if degraded_ms else 0.0
        delta = degraded_median - healthy_median
        stage_deltas.append(
            StageDelta(
                signature=_signature_text(signature),
                event_type=signature[0],
                operation=signature[1],
                rank=signature[2],
                healthy_count=len(healthy_events),
                degraded_count=len(degraded_events),
                matched_count=pair_count,
                healthy_median_ms=healthy_median,
                degraded_median_ms=degraded_median,
                healthy_p95_ms=_percentile(healthy_ms, 0.95),
                degraded_p95_ms=_percentile(degraded_ms, 0.95),
                absolute_delta_ms=delta,
                relative_delta=_relative_delta(healthy_median, degraded_median),
            )
        )
        for healthy_event, degraded_event in zip(
            healthy_events[:pair_count], degraded_events[:pair_count], strict=True
        ):
            threshold_ns = max(100_000, int(healthy_event.duration_ns * 0.15))
            if degraded_event.duration_ns - healthy_event.duration_ns <= threshold_ns:
                continue
            start = (
                degraded_event.normalized_start_ns
                if degraded_event.normalized_start_ns is not None
                else degraded_event.start_ns
            )
            if first_divergence is None or start < first_divergence[0]:
                first_divergence = (start, degraded_event.event_id)

    healthy_counters = _counter_groups(healthy)
    degraded_counters = _counter_groups(degraded)
    counter_deltas: list[CounterDelta] = []
    for name, unit in sorted(set(healthy_counters) | set(degraded_counters)):
        healthy_median = (
            statistics.median(healthy_counters[(name, unit)])
            if healthy_counters[(name, unit)]
            else 0.0
        )
        degraded_median = (
            statistics.median(degraded_counters[(name, unit)])
            if degraded_counters[(name, unit)]
            else 0.0
        )
        counter_deltas.append(
            CounterDelta(
                name=name,
                unit=unit,
                healthy_median=healthy_median,
                degraded_median=degraded_median,
                absolute_delta=degraded_median - healthy_median,
                relative_delta=_relative_delta(healthy_median, degraded_median),
            )
        )

    warnings: list[str] = []
    qualities = {estimate.quality for estimate in (*healthy.alignments, *degraded.alignments)}
    if AlignmentQuality.INSUFFICIENT in qualities:
        warnings.append(
            "cross-host alignment is insufficient; event ordering is not causal evidence"
        )
    elif AlignmentQuality.DEGRADED in qualities:
        warnings.append(
            "cross-host alignment is degraded; fine-grained ordering has reduced confidence"
        )
    identifier = hashlib.sha256(
        f"{healthy.run_id}\0{degraded.run_id}\0{matched}".encode()
    ).hexdigest()[:16]
    return DifferentialComparison(
        comparison_id=f"comparison-{identifier}",
        healthy_run_id=healthy.run_id,
        degraded_run_id=degraded.run_id,
        matched_event_count=matched,
        unmatched_healthy_count=unmatched_healthy,
        unmatched_degraded_count=unmatched_degraded,
        stage_deltas=tuple(sorted(stage_deltas, key=lambda item: item.signature)),
        counter_deltas=tuple(counter_deltas),
        first_divergence_event_id=first_divergence[1] if first_divergence else None,
        first_divergence_ns=first_divergence[0] if first_divergence else None,
        maximum_rank_skew=_rank_skew(degraded),
        warnings=tuple(warnings),
    )
