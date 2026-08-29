#!/usr/bin/env python3
"""Exercise the actual Experiment 004 v10 serving driver on deterministic CPU fixtures.

This is a methodology gate, not a capacity measurement.  It imports and runs
the exact live GPU0/GPU1 orchestration module with bounded fake engines, records
the real monotonic arrival timestamps produced by its timer thread, and
independently recomputes the trigger arithmetic before emitting immutable JSON.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import sys
import tempfile
import threading
import time
from collections.abc import Iterable
from itertools import pairwise
from pathlib import Path
from types import SimpleNamespace
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "python"))
_LIVE_PATH = _ROOT / "experiments/branchfabric/gpu_reclamation_v10_serving.py"
_SPEC = importlib.util.spec_from_file_location("experiment_004_v10_live_driver_gate", _LIVE_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError("could not load the Experiment 004 v10 live driver")
_LIVE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _LIVE
_SPEC.loader.exec_module(_LIVE)


def _canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n"
    ).encode()


def _sha256_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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


class _DeterministicCpuEngine:
    """A FIFO, one-request-per-step engine with a fixed service delay."""

    def __init__(self, *, service_seconds: float, output_tokens: int) -> None:
        self.service_seconds = service_seconds
        self.output_tokens = output_tokens
        self.requests: list[str] = []

    def add_request(self, request_id: str, _prompt: object, _params: object) -> None:
        self.requests.append(request_id)

    def step(self) -> tuple[Any, ...]:
        time.sleep(self.service_seconds)
        if not self.requests:
            return ()
        request_id = self.requests.pop(0)
        return (
            SimpleNamespace(
                request_id=request_id,
                outputs=(SimpleNamespace(token_ids=tuple(range(self.output_tokens))),),
                finished=True,
            ),
        )


def _config(*, seed: int) -> Any:
    # Short intervals keep this preflight CPU-only. Production configuration
    # still enters through LiveV10Config.from_mapping, which separately
    # requires the full five-second recovery stability window.
    return _LIVE.LiveV10Config(
        attempt_id="exp004-v10-live-driver-local-gate",
        seed=seed,
        control_rate_per_second=20.0,
        spike_rate_per_second=40.0,
        restore_rate_per_second=20.0,
        baseline_seconds=0.12,
        overload_probe_seconds=0.12,
        recovery_stability_seconds=0.2,
        recovery_evaluation_seconds=0.1,
        recovery_queue_threshold=20,
        output_tokens=64,
        maximum_wall_seconds=5.0,
        restore_grace_seconds=0.12,
        producer_queue_capacity=256,
        maximum_pending_requests=100,
        restore_handoff_lead_requests=4,
        overload_queue_trigger=10,
        overload_queue_abort=64,
    )


def _percentile(values: Iterable[int], fraction: float) -> float | None:
    ordered = sorted(values)
    if not ordered:
        return None
    position = (len(ordered) - 1) * fraction
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return float(ordered[low])
    return ordered[low] * (high - position) + ordered[high] * (position - low)


def _outstanding(
    offered: tuple[dict[str, Any], ...],
    completed: tuple[dict[str, Any], ...],
    timestamp_ns: int,
) -> int:
    completed_ids = {
        str(row["request_id"])
        for row in completed
        if row.get("completed_ns") is not None and int(row["completed_ns"]) <= timestamp_ns
    }
    return sum(
        int(row["scheduled_arrival_ns"]) <= timestamp_ns
        and str(row["request_id"]) not in completed_ids
        for row in offered
    )


def _recompute_trigger(
    *, serving: dict[str, Any], trigger: dict[str, Any]
) -> dict[str, Any]:
    start = int(trigger["window_start_ns"])
    end = int(trigger["window_end_ns"])
    duration = (end - start) / _LIVE.NS_PER_SECOND
    all_offered = tuple(dict(row) for row in serving["global_offered_requests"])
    offered = all_offered[: int(trigger["offered_snapshot_count"])]
    completed = tuple(dict(row) for row in serving["requests"])
    cohort = tuple(
        row for row in offered if start <= int(row["scheduled_arrival_ns"]) < end
    )
    completed_count = sum(
        start <= int(row["completed_ns"]) < end
        for row in completed
        if row.get("completed_ns") is not None
    )
    p95 = _percentile(
        (
            int(row["first_token_ns"]) - int(row["scheduled_arrival_ns"])
            for row in completed
            if row.get("first_token_ns") is not None
            and row.get("completed_ns") is not None
            and int(row["completed_ns"]) <= end
            and start <= int(row["scheduled_arrival_ns"]) < end
        ),
        0.95,
    )
    queue_start = _outstanding(offered, completed, start)
    queue_end = _outstanding(offered, completed, end)
    return {
        "window_start_ns": start,
        "window_end_ns": end,
        "offered_snapshot_count": len(offered),
        "offered_requests": len(cohort),
        "offered_rate_per_second": len(cohort) / duration,
        "completed_requests": completed_count,
        "completed_rate_per_second": completed_count / duration,
        "queue_depth_start": queue_start,
        "queue_depth_end": queue_end,
        "queue_depth_slope_per_second": (queue_end - queue_start) / duration,
        "p95_ttft_ns": p95,
    }


def _matches_trigger(recomputed: dict[str, Any], trigger: dict[str, Any]) -> bool:
    integer_fields = (
        "window_start_ns",
        "window_end_ns",
        "offered_snapshot_count",
        "offered_requests",
        "completed_requests",
        "queue_depth_start",
        "queue_depth_end",
    )
    float_fields = (
        "offered_rate_per_second",
        "completed_rate_per_second",
        "queue_depth_slope_per_second",
        "p95_ttft_ns",
    )

    def float_matches(key: str) -> bool:
        left = recomputed[key]
        right = trigger[key]
        if left is None or right is None:
            return left is right
        return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-9)

    return all(recomputed[key] == trigger[key] for key in integer_fields) and all(
        float_matches(key) for key in float_fields
    )


def _run_actual_driver(*, config: Any, barriers: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    gpu0_result: dict[str, Any] = {}
    gpu1_result: dict[str, Any] = {}
    errors: list[BaseException] = []
    common_start = time.monotonic_ns() + 50_000_000

    def gpu0() -> None:
        try:
            gpu0_result.update(
                _LIVE.run_v10_gpu0(
                    _DeterministicCpuEngine(service_seconds=0.03, output_tokens=64),
                    prefix=(1,),
                    params=object(),
                    config=config,
                    start_ns=common_start,
                    barriers=barriers,
                    write_new=_write_new,
                    runtime_queue_state=lambda: {
                        "request_count": 0,
                        "running_requests": 0,
                        "waiting_requests": 0,
                        "skipped_waiting_requests": 0,
                        "queue_depth": 0,
                    },
                )
            )
        except BaseException as error:
            errors.append(error)

    def gpu1() -> None:
        try:
            trigger = barriers / "v10-reclaim-trigger.json"
            while not trigger.is_file():
                time.sleep(0.005)
            gpu1_result.update(
                _LIVE.run_v10_gpu1(
                    _DeterministicCpuEngine(service_seconds=0.03, output_tokens=64),
                    prefix=(1,),
                    params=object(),
                    config=config,
                    barriers=barriers,
                    write_new=_write_new,
                    runtime_queue_state=lambda: {
                        "request_count": 0,
                        "running_requests": 0,
                        "waiting_requests": 0,
                        "skipped_waiting_requests": 0,
                        "queue_depth": 0,
                    },
                )
            )
            _write_new(
                barriers / "rollout-restore-complete.json",
                {"observed_at_monotonic_ns": time.monotonic_ns()},
            )
        except BaseException as error:
            errors.append(error)

    threads = (
        threading.Thread(target=gpu0, name="live-gate-gpu0"),
        threading.Thread(target=gpu1, name="live-gate-gpu1"),
    )
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=7.0)
    if any(thread.is_alive() for thread in threads):
        raise TimeoutError("live-driver CPU fixture exceeded its bounded seven-second join")
    if errors:
        raise RuntimeError("live-driver CPU fixture failed") from errors[0]
    return gpu0_result, gpu1_result


def build_gate(*, seed: int) -> dict[str, Any]:
    from sloforge.helix.characterization.gpu_reclamation_v10_postprocess import (
        _actual_arrival_evidence,
    )

    config = _config(seed=seed)
    with tempfile.TemporaryDirectory(prefix="sloforge-exp004-live-driver-gate-") as scratch:
        serving, gpu1 = _run_actual_driver(config=config, barriers=Path(scratch))

    trigger = dict(serving["reclamation_trigger_evidence"])
    recovery = dict(serving["serving_recovery_evidence"])
    enable = dict(serving["serving_enable_ack"])
    restore = dict(serving["restore_route_cutover"])
    drained = dict(serving["gpu1_drained"])
    offered = tuple(dict(row) for row in serving["global_offered_requests"])
    gpu0_rows = tuple(dict(row) for row in serving["requests"])
    gpu1_rows = tuple(dict(row) for row in gpu1["requests"])
    combined = gpu0_rows + gpu1_rows
    by_id = {str(row["request_id"]): row for row in combined}
    recomputed_trigger = _recompute_trigger(serving=serving, trigger=trigger)
    arrival = _actual_arrival_evidence(
        config={
            "gpu0_control_request_rate_per_second": config.control_rate_per_second,
            "serving_spike_request_rate_per_second": config.spike_rate_per_second,
            "gpu0_restore_request_rate_per_second": config.restore_rate_per_second,
        },
        serving=serving,
    )
    enable_sequence = int(enable["enable_cutover_sequence"])
    restore_sequence = int(restore["restore_cutover_sequence"])
    spike_rows = tuple(row for row in offered if str(row["request_id"]).find(".spike.") >= 0)
    recovery_rows = tuple(
        row
        for row in spike_rows
        if enable_sequence <= int(row["sequence"]) < restore_sequence
    )
    restore_rows = tuple(row for row in offered if row["phase"] == "restore-interference")
    abort_guard_passed = False
    try:
        _LIVE._enforce_pre_gpu1_queue_abort(
            trigger_written=True,
            gpu1_first_useful=None,
            instantaneous_depth=config.overload_queue_abort + 1,
            maximum_depth=config.overload_queue_abort,
        )
    except RuntimeError as error:
        abort_guard_passed = "post-reclaim-pre-gpu1" in str(error)

    expected_enable_ns = _LIVE.spike_arrival_ns(
        spike_start_ns=int(enable["spike_start_ns"]),
        rate_per_second=config.spike_rate_per_second,
        sequence=enable_sequence,
    )
    expected_restore_ns = _LIVE.spike_arrival_ns(
        spike_start_ns=int(enable["spike_start_ns"]),
        rate_per_second=config.spike_rate_per_second,
        sequence=restore_sequence,
    )
    assertions = {
        "actual_driver_completed": bool(gpu0_rows and gpu1_rows),
        "actual_arrival_rate_and_timer_cadence_pass": bool(arrival["passed"]),
        "actual_offered_timestamps_monotonic": all(
            int(left["offered_ns"]) <= int(right["offered_ns"])
            for left, right in pairwise(offered)
        ),
        "every_offered_request_completed_once": len(by_id) == len(combined) == len(offered)
        and {str(row["request_id"]) for row in offered} == set(by_id),
        "output_length_exactly_64": all(
            len(row["output_token_ids"]) == 64 for row in combined
        ),
        "ttft_completion_and_queue_math_independently_recomputed": _matches_trigger(
            recomputed_trigger, trigger
        ),
        "overload_trigger_is_bounded_10_to_30": 10 <= config.overload_queue_trigger <= 30
        and config.overload_queue_trigger
        <= int(trigger["queue_depth_end"])
        <= config.overload_queue_abort,
        "overload_trigger_rule_passed": bool(trigger["overload_confirmed"])
        and float(trigger["queue_depth_slope_per_second"]) > 0,
        "post_trigger_abort_guard_passed": abort_guard_passed,
        "gpu0_only_before_enable": all(
            row["device"] == "gpu0" for row in spike_rows if int(row["sequence"]) < enable_sequence
        ),
        "route_to_both_after_enable": {row["device"] for row in recovery_rows}
        == {"gpu0", "gpu1"}
        and all(
            row["device"]
            == _LIVE.recovery_route(
                sequence=int(row["sequence"]), enable_cutover_sequence=enable_sequence
            )
            for row in recovery_rows
        ),
        "routing_cutover_timestamps_exact": int(enable["enable_cutover_scheduled_ns"])
        == expected_enable_ns
        and int(restore["restore_start_ns"]) == expected_restore_ns
        and bool(restore_rows)
        and int(restore_rows[0]["scheduled_arrival_ns"]) == expected_restore_ns,
        "load_reduced_before_restore": config.restore_rate_per_second
        < config.spike_rate_per_second
        and all(row["device"] == "gpu0" for row in restore_rows),
        "gpu1_drain_exact": int(drained["running_requests"]) == 0
        and int(drained["waiting_requests"]) == 0
        and int(drained["last_admitted_sequence"]) < restore_sequence
        and int(gpu1["last_admitted_sequence"]) < restore_sequence,
        "two_gpu_recovery_math_passed": float(recovery["completed_rate_per_second"])
        > float(recovery["offered_rate_per_second"])
        and float(recovery["queue_depth_slope_per_second"]) < 0
        and bool(recovery["restore_eligible"]),
        "gpu0_completed_restore_requests": any(
            row["phase"] == "restore-interference" and row["completed_ns"] is not None
            for row in gpu0_rows
        ),
    }
    config_inputs = {
        field: getattr(config, field)
        for field in config.__dataclass_fields__
    }
    raw_evidence = {
        "global_offered_requests": offered,
        "gpu0_requests": gpu0_rows,
        "gpu1_requests": gpu1_rows,
        "reclamation_trigger_evidence": trigger,
        "serving_recovery_evidence": recovery,
        "serving_enable_ack": enable,
        "restore_route_cutover": restore,
        "gpu1_drained": drained,
    }
    return {
        "schema_version": "sloforge.branchfabric.experiment-004-live-driver-local-gate/v1",
        "status": "passed" if all(assertions.values()) else "failed",
        "fixture_kind": (
            "deterministic-seeded-cpu-live-driver-validation-not-gpu-capacity-measurement"
        ),
        "seed": seed,
        "provenance": {
            "live_driver_path": str(_LIVE_PATH.relative_to(_ROOT)),
            "live_driver_sha256": _sha256_path(_LIVE_PATH),
            "generator_path": str(Path(__file__).resolve().relative_to(_ROOT)),
            "generator_sha256": _sha256_path(Path(__file__).resolve()),
            "fixture_config_sha256": hashlib.sha256(_canonical_bytes(config_inputs)).hexdigest(),
            "raw_evidence_sha256": hashlib.sha256(_canonical_bytes(raw_evidence)).hexdigest(),
        },
        "fixture_config": config_inputs,
        "actual_arrival_evidence": arrival,
        "actual_timer_cadence": {
            phase: [
                {
                    "sequence": row["sequence"],
                    "device": row["device"],
                    "scheduled_arrival_ns": row["scheduled_arrival_ns"],
                    "offered_ns": row["offered_ns"],
                    "schedule_lateness_ns": int(row["offered_ns"])
                    - int(row["scheduled_arrival_ns"]),
                }
                for row in offered
                if row["phase"] == phase
            ]
            for phase in (
                "control",
                "gpu0-overload",
                "two-gpu-recovery",
                "restore-interference",
            )
        },
        "independently_recomputed_trigger": recomputed_trigger,
        "routing_cutovers": {
            "enable": enable,
            "restore": restore,
            "gpu1_drained": drained,
        },
        "raw_evidence": raw_evidence,
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
