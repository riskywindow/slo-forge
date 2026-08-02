"""Robust cross-host clock offset and drift estimation."""

from __future__ import annotations

import math
import statistics
from collections.abc import Iterable, Mapping

from .models import (
    AlignmentEstimate,
    AlignmentQuality,
    AutopsyEvent,
    AutopsyRun,
    ClockSample,
)


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        raise ValueError("cannot take percentile of no values")
    ordered = sorted(values)
    rank = (len(ordered) - 1) * percentile
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[lower]
    weight = rank - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def estimate_alignment(samples: Iterable[ClockSample], *, reference_host: str) -> AlignmentEstimate:
    records = sorted(samples, key=lambda item: item.local_monotonic_ns)
    if not records:
        raise ValueError("clock alignment requires at least one sample")
    hosts = {sample.host for sample in records}
    if len(hosts) != 1:
        raise ValueError("clock alignment samples must belong to one host")
    host = records[0].host
    reference_local = records[len(records) // 2].local_monotonic_ns

    # The reference timestamp is observed after a round trip. The midpoint is
    # the least-assumptive offset estimate; half RTT remains uncertainty.
    xs = [float(sample.local_monotonic_ns - reference_local) for sample in records]
    ys = [
        float(sample.reference_monotonic_ns - sample.round_trip_ns // 2 - sample.local_monotonic_ns)
        for sample in records
    ]
    # Use a bounded Theil-Sen fit over the lowest-RTT observations. Ordinary
    # least squares lets one delayed clock exchange arbitrarily move both the
    # drift and offset. Lowest-RTT selection follows the NTP minimum-delay
    # principle; the cap prevents quadratic work on untrusted trace input.
    selected_indices = sorted(
        range(len(records)),
        key=lambda index: (records[index].round_trip_ns, records[index].local_monotonic_ns),
    )[:256]
    selected_xs = [xs[index] for index in selected_indices]
    selected_ys = [ys[index] for index in selected_indices]
    slopes = [
        (selected_ys[right] - selected_ys[left]) / (selected_xs[right] - selected_xs[left])
        for left in range(len(selected_xs))
        for right in range(left + 1, len(selected_xs))
        if selected_xs[right] != selected_xs[left]
    ]
    slope = statistics.median(slopes) if slopes else 0.0
    intercept = statistics.median(
        y - slope * x for x, y in zip(selected_xs, selected_ys, strict=True)
    )
    residuals = [abs(y - (intercept + slope * x)) for x, y in zip(xs, ys, strict=True)]
    residual_p95 = math.ceil(_percentile(residuals, 0.95))
    half_rtt_p95 = math.ceil(_percentile([sample.round_trip_ns / 2.0 for sample in records], 0.95))
    uncertainty = residual_p95 + half_rtt_p95
    if len(records) >= 5 and uncertainty <= 250_000:
        quality = AlignmentQuality.GOOD
        confidence = min(0.99, 0.8 + len(records) / 100.0)
    elif len(records) >= 3 and uncertainty <= 5_000_000:
        quality = AlignmentQuality.DEGRADED
        confidence = min(0.79, 0.45 + len(records) / 100.0)
    else:
        quality = AlignmentQuality.INSUFFICIENT
        confidence = min(0.4, len(records) / 20.0)
    return AlignmentEstimate(
        host=host,
        reference_host=reference_host,
        offset_ns=intercept,
        drift_ppm=slope * 1_000_000.0,
        reference_local_ns=reference_local,
        uncertainty_ns=uncertainty,
        sample_count=len(records),
        confidence=confidence,
        quality=quality,
        residual_p95_ns=residual_p95,
    )


def _normalize(timestamp_ns: int, estimate: AlignmentEstimate) -> int:
    elapsed = timestamp_ns - estimate.reference_local_ns
    corrected = timestamp_ns + estimate.offset_ns + elapsed * estimate.drift_ppm / 1_000_000.0
    if corrected < 0.0 or corrected > float(2**63 - 1):
        raise ValueError("normalized timestamp is outside supported range")
    return round(corrected)


def align_event(event: AutopsyEvent, estimate: AlignmentEstimate) -> AutopsyEvent:
    if event.host != estimate.host:
        raise ValueError("alignment estimate host does not match event host")
    return event.model_copy(
        update={
            "normalized_start_ns": _normalize(event.start_ns, estimate),
            "normalized_end_ns": _normalize(event.end_ns, estimate),
            "alignment_confidence": estimate.confidence,
            "alignment_uncertainty_ns": estimate.uncertainty_ns,
        }
    )


def align_run(
    run: AutopsyRun, estimates: Mapping[str, AlignmentEstimate], *, require_usable: bool = True
) -> AutopsyRun:
    missing = {event.host for event in run.events} - estimates.keys()
    if missing:
        raise ValueError(f"missing alignment estimates for hosts: {sorted(missing)}")
    wrong_reference = {
        host
        for host, estimate in estimates.items()
        if estimate.reference_host != run.reference_host
    }
    if wrong_reference:
        raise ValueError(
            f"alignment estimates use a different reference host for {sorted(wrong_reference)}"
        )
    if require_usable:
        unusable = {
            host
            for host, estimate in estimates.items()
            if estimate.quality is AlignmentQuality.INSUFFICIENT
        }
        if unusable:
            raise ValueError(f"alignment quality is insufficient for hosts: {sorted(unusable)}")
    aligned = tuple(align_event(event, estimates[event.host]) for event in run.events)
    return AutopsyRun.model_validate(
        {
            **run.model_dump(mode="python"),
            "events": aligned,
            "alignments": tuple(estimates[key] for key in sorted(estimates)),
        }
    )
