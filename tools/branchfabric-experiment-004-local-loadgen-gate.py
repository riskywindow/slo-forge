#!/usr/bin/env python3
"""Run the Experiment 004 load-generator gate using deterministic CPU fixtures.

This command validates methodology and arithmetic only.  Its synthetic fixture
values are explicitly segregated from GPU measurements and must never be used
as serving-capacity evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "python"))


def _canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n"
    ).encode()


def _write_new(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("xb") as handle:
        handle.write(_canonical_bytes(value))
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _percentile(values: tuple[float, ...], fraction: float) -> float:
    ordered = tuple(sorted(values))
    if not ordered:
        raise ValueError("fixture percentile requires samples")
    position = (len(ordered) - 1) * fraction
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    return ordered[low] * (high - position) + ordered[high] * (position - low)


def _capacity_fixture(*, topology: Any, rate_rps: float, capacity_rps: float, seed: int) -> Any:
    from sloforge.helix.characterization.gpu_capacity_calibration import (
        NS_PER_SECOND,
        CapacityProbeRaw,
        CapacityRequestObservation,
        RequestTerminalState,
        build_probe_plan,
    )

    plan = build_probe_plan(
        probe_id=f"local-gate-{topology.value}-{int(rate_rps)}",
        seed=seed,
        topology=topology,
        configured_rate_rps=rate_rps,
        start_ns=1_000_000_000,
        warmup_seconds=1.0,
        measurement_seconds=10.0,
    )
    per_device_capacity = capacity_rps if topology.value == "gpu0-only" else capacity_rps / 2
    service_ns = round(NS_PER_SECOND / per_device_capacity)
    next_available = {"gpu0": 0, "gpu1": 0}
    rows = []
    for arrival in plan.arrivals:
        completed_ns = max(
            arrival.scheduled_arrival_ns + service_ns,
            next_available[arrival.assigned_device] + service_ns,
        )
        next_available[arrival.assigned_device] = completed_ns
        first_token_ns = completed_ns - service_ns // 2
        rows.append(
            CapacityRequestObservation(
                request_id=arrival.request_id,
                global_sequence=arrival.global_sequence,
                device=arrival.assigned_device,
                scheduled_arrival_ns=arrival.scheduled_arrival_ns,
                offered_ns=arrival.scheduled_arrival_ns,
                enqueued_ns=arrival.scheduled_arrival_ns,
                admitted_ns=arrival.scheduled_arrival_ns,
                first_token_ns=first_token_ns,
                completed_ns=completed_ns,
                terminated_ns=completed_ns,
                requested_output_tokens=64,
                emitted_tokens=64,
                terminal_state=RequestTerminalState.COMPLETED,
            )
        )
    return CapacityProbeRaw(
        plan=plan,
        observations=tuple(rows),
        tail_drain_end_ns=plan.measurement_end_ns + 5 * NS_PER_SECOND,
        probe_end_ns=plan.measurement_end_ns + 5 * NS_PER_SECOND,
        tail_drain_seconds=5.0,
    )


def _independent_metrics(raw: Any) -> dict[str, Any]:
    from sloforge.helix.characterization.gpu_capacity_calibration import NS_PER_SECOND

    plan = raw.plan
    rows = {row.request_id: row for row in raw.observations}
    cohort = tuple(
        item
        for item in plan.arrivals
        if plan.measurement_start_ns <= item.scheduled_arrival_ns < plan.measurement_end_ns
    )
    duration = (plan.measurement_end_ns - plan.measurement_start_ns) / NS_PER_SECOND
    offered = tuple(
        row
        for row in raw.observations
        if plan.measurement_start_ns <= row.offered_ns < plan.measurement_end_ns
    )
    ttft = tuple(
        (rows[item.request_id].first_token_ns - item.scheduled_arrival_ns) / NS_PER_SECOND
        for item in cohort
        if rows[item.request_id].first_token_ns is not None
    )
    cohort_interval_completions = sum(
        rows[item.request_id].completed_ns is not None
        and plan.measurement_start_ns
        <= rows[item.request_id].completed_ns
        < plan.measurement_end_ns
        for item in cohort
    )
    all_completions = sum(
        row.completed_ns is not None
        and plan.measurement_start_ns <= row.completed_ns < plan.measurement_end_ns
        for row in raw.observations
    )
    sample_count = math.floor(duration) + 1
    boundaries = tuple(
        plan.measurement_start_ns - 1
        if index == 0
        else (
            plan.measurement_end_ns - 1
            if index == sample_count - 1
            else plan.measurement_start_ns
            + round(duration * index / (sample_count - 1) * NS_PER_SECOND)
        )
        for index in range(sample_count)
    )
    depths = tuple(
        sum(
            rows[item.request_id].offered_ns <= boundary
            and (
                rows[item.request_id].completed_ns is None
                or rows[item.request_id].completed_ns > boundary
            )
            for item in plan.arrivals
        )
        for boundary in boundaries
    )
    return {
        "offered_rate_rps": len(offered) / duration,
        "completion_rate_rps": all_completions / duration,
        "measurement_cohort_interval_completions": cohort_interval_completions,
        "measurement_cohort_total_completions": sum(
            rows[item.request_id].completed_ns is not None for item in cohort
        ),
        "interval_completion_events": all_completions,
        "warmup_completions_in_measurement_window": (all_completions - cohort_interval_completions),
        "p95_ttft_seconds": _percentile(ttft, 0.95),
        "queue_depth_start": depths[0],
        "queue_depth_end": depths[-1],
        "maximum_queue_depth": max(depths),
        "queue_depth_samples": depths,
        "queue_flow_expected_end_depth": depths[0] + len(offered) - all_completions,
        "queue_flow_conservation_pass": (depths[-1] == depths[0] + len(offered) - all_completions),
    }


def build_gate(*, seed: int) -> dict[str, Any]:
    from sloforge.helix.characterization.gpu_capacity_calibration import (
        ProbeTopology,
        ProbeVerdict,
        evaluate_probe,
    )
    from sloforge.helix.characterization.gpu_reclamation_serving_v10 import (
        build_global_serving_plan,
    )

    raw_one = _capacity_fixture(
        topology=ProbeTopology.GPU0_ONLY,
        rate_rps=20.0,
        capacity_rps=22.0,
        seed=seed,
    )
    raw_two = _capacity_fixture(
        topology=ProbeTopology.TWO_GPU_ROUND_ROBIN,
        rate_rps=30.0,
        capacity_rps=40.0,
        seed=seed,
    )
    result_one = evaluate_probe(raw_one)
    result_two = evaluate_probe(raw_two)
    independent = {
        "gpu0_only": _independent_metrics(raw_one),
        "two_gpu": _independent_metrics(raw_two),
    }
    start = 20_000_000_000
    spike = start + 2_000_000_000
    enable = spike + 2_000_000_000
    restore = enable + 6_000_000_000
    end = restore + 2_000_000_000
    live_plan = build_global_serving_plan(
        attempt_id="exp004-local-loadgen-gate",
        seed=seed,
        control_start_ns=start,
        spike_start_ns=spike,
        gpu1_route_start_ns=enable,
        restore_start_ns=restore,
        end_ns=end,
        control_rate_per_second=15.0,
        spike_rate_per_second=25.0,
        restore_rate_per_second=15.0,
    )
    before_enable = tuple(item for item in live_plan.requests if item.scheduled_arrival_ns < enable)
    recovery = tuple(item for item in live_plan.requests if item.phase == "two-gpu-recovery")
    restore_rows = tuple(
        item for item in live_plan.requests if item.phase == "restore-interference"
    )

    assertions = {
        "configured_actual_arrival_rate_match": all(
            result.configured_arrival_rate_verified for result in (result_one, result_two)
        ),
        "timestamps_monotonic": all(
            result.monotonic_timestamps_verified for result in (result_one, result_two)
        ),
        "output_length_exactly_64": all(
            result.output_target_verified for result in (result_one, result_two)
        )
        and all(item.requested_output_tokens == 64 for item in live_plan.requests),
        "gpu0_only_route_exact": result_one.routing_verified
        and {item.device for item in raw_one.observations} == {"gpu0"},
        "two_gpu_route_exact": result_two.routing_verified
        and {item.device for item in raw_two.observations} == {"gpu0", "gpu1"},
        "queue_depth_independently_recomputed": all(
            result.queue_depth_start == metrics["queue_depth_start"]
            and result.queue_depth_end == metrics["queue_depth_end"]
            and result.maximum_queue_depth == metrics["maximum_queue_depth"]
            for result, metrics in (
                (result_one, independent["gpu0_only"]),
                (result_two, independent["two_gpu"]),
            )
        ),
        "ttft_independently_recomputed": all(
            math.isclose(
                result.p95_ttft_seconds or -1.0,
                metrics["p95_ttft_seconds"],
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            for result, metrics in (
                (result_one, independent["gpu0_only"]),
                (result_two, independent["two_gpu"]),
            )
        ),
        "completion_rate_independently_recomputed": all(
            math.isclose(
                result.completed_rate_rps,
                metrics["completion_rate_rps"],
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            for result, metrics in (
                (result_one, independent["gpu0_only"]),
                (result_two, independent["two_gpu"]),
            )
        ),
        "warmup_carry_in_accounted_without_cohort_contamination": any(
            metrics["warmup_completions_in_measurement_window"] > 0
            for metrics in independent.values()
        )
        and all(
            result.completed_rate_rps == metrics["interval_completion_events"] / 10.0
            and result.all_measurement_requests_completed
            and metrics["measurement_cohort_total_completions"] == result.measurement_requests
            for result, metrics in (
                (result_one, independent["gpu0_only"]),
                (result_two, independent["two_gpu"]),
            )
        ),
        "queue_flow_conservation_independently_recomputed": all(
            metrics["queue_flow_conservation_pass"]
            and result.queue_flow_conservation_pass
            and result.queue_flow_expected_end_depth == metrics["queue_flow_expected_end_depth"]
            for result, metrics in (
                (result_one, independent["gpu0_only"]),
                (result_two, independent["two_gpu"]),
            )
        ),
        "offered_rate_independently_recomputed": all(
            math.isclose(
                result.observed_offered_rate_rps,
                metrics["offered_rate_rps"],
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            for result, metrics in (
                (result_one, independent["gpu0_only"]),
                (result_two, independent["two_gpu"]),
            )
        ),
        "no_client_timer_burst": all(
            result.offer_cadence_verified
            and result.maximum_offer_burst_50ms <= result.offer_burst_limit_50ms
            for result in (result_one, result_two)
        ),
        "routing_cutovers_exact": bool(before_enable and recovery and restore_rows)
        and all(item.planned_device == "gpu0" for item in before_enable)
        and recovery[0].scheduled_arrival_ns >= enable
        and {item.planned_device for item in recovery} == {"gpu0", "gpu1"}
        and all(item.planned_device == "gpu0" for item in restore_rows),
        "load_reduced_before_restore": live_plan.epochs[3].rate_per_second
        < live_plan.epochs[1].rate_per_second
        and live_plan.epochs[3].rate_per_second == live_plan.epochs[0].rate_per_second,
        "fixture_operating_points_sustainable": result_one.verdict
        == result_two.verdict
        == ProbeVerdict.SUSTAINABLE,
    }
    fixture_inputs = {
        "seed": seed,
        "capacity": {
            "gpu0_only": {"offered_rps": 20.0, "fixture_capacity_rps": 22.0},
            "two_gpu": {"offered_rps": 30.0, "fixture_capacity_rps": 40.0},
        },
        "live_rates_rps": {"control": 15.0, "spike": 25.0, "restore": 15.0},
        "boundaries_ns": {
            "control_start": start,
            "spike_start": spike,
            "gpu1_route_start": enable,
            "restore_start": restore,
            "end": end,
        },
    }
    return {
        "schema_version": "sloforge.branchfabric.experiment-004-local-loadgen-gate/v1",
        "status": "passed" if all(assertions.values()) else "failed",
        "fixture_kind": "deterministic-cpu-methodology-validation-not-gpu-measurement",
        "seed": seed,
        "provenance": {
            "fixture_input_sha256": hashlib.sha256(_canonical_bytes(fixture_inputs)).hexdigest(),
            "capacity_module": "python/sloforge/helix/characterization/gpu_capacity_calibration.py",
            "routing_module": "python/sloforge/helix/characterization/gpu_reclamation_serving_v10.py",
            "generator": "tools/branchfabric-experiment-004-local-loadgen-gate.py",
        },
        "fixture_inputs": fixture_inputs,
        "derived_metrics": independent,
        "routing_plan_sha256": live_plan.plan_sha256,
        "assertions": assertions,
        "passed": all(assertions.values()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not 0 <= args.seed < 1 << 63:
        raise ValueError("fixture seed must fit signed 63-bit range")
    payload = build_gate(seed=args.seed)
    if args.output is not None:
        _write_new(args.output, payload)
    sys.stdout.buffer.write(_canonical_bytes(payload))
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
