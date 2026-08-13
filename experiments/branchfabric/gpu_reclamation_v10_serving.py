"""Live, bounded serving protocol for Experiment 004 v10.

GPU0 owns the only offered-load clock.  GPU1 independently reconstructs only
its deterministic shard after an acknowledged sequence cutover.  Small
immutable JSON barriers coordinate the processes; no request payload crosses
the process boundary and no GPU library is imported by this module.
"""

from __future__ import annotations

import json
import math
import queue
import statistics
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from decimal import Decimal
from fractions import Fraction
from pathlib import Path
from typing import Any, Literal

NS_PER_SECOND = 1_000_000_000
_POLL_SECONDS = 0.005


@dataclass(frozen=True)
class LiveV10Config:
    attempt_id: str
    seed: int
    control_rate_per_second: float
    spike_rate_per_second: float
    restore_rate_per_second: float
    baseline_seconds: float
    overload_probe_seconds: float
    recovery_stability_seconds: float
    recovery_evaluation_seconds: float
    recovery_queue_threshold: int
    output_tokens: int
    maximum_wall_seconds: float
    restore_grace_seconds: float
    maximum_pending_requests: int
    restore_handoff_lead_requests: int
    overload_queue_trigger: int
    overload_queue_abort: int

    @classmethod
    def from_mapping(cls, config: dict[str, Any]) -> LiveV10Config:
        if config.get("serving_methodology") != "v10-global-capacity":
            raise ValueError("live v10 serving requires its explicit methodology selector")
        result = cls(
            attempt_id=str(config["attempt_id"]),
            seed=int(config["seed"]),
            control_rate_per_second=float(config["gpu0_control_request_rate_per_second"]),
            spike_rate_per_second=float(config["serving_spike_request_rate_per_second"]),
            restore_rate_per_second=float(config["gpu0_restore_request_rate_per_second"]),
            baseline_seconds=float(config["baseline_seconds"]),
            overload_probe_seconds=float(config["gpu0_overload_probe_seconds"]),
            recovery_stability_seconds=float(config["serving_slo_stability_window_seconds"]),
            recovery_evaluation_seconds=float(config["serving_recovery_evaluation_seconds"]),
            recovery_queue_threshold=int(config["serving_recovery_queue_threshold"]),
            output_tokens=int(config["serving_output_tokens"]),
            maximum_wall_seconds=float(config["maximum_wall_seconds"]),
            restore_grace_seconds=float(config["temporary_serving_seconds"]),
            maximum_pending_requests=int(config["serving_maximum_pending_requests"]),
            restore_handoff_lead_requests=int(config["serving_restore_handoff_lead_requests"]),
            overload_queue_trigger=int(config["serving_overload_queue_trigger"]),
            overload_queue_abort=int(config["serving_overload_queue_abort"]),
        )
        finite_positive = (
            result.control_rate_per_second,
            result.spike_rate_per_second,
            result.restore_rate_per_second,
            result.baseline_seconds,
            result.overload_probe_seconds,
            result.recovery_stability_seconds,
            result.recovery_evaluation_seconds,
            result.maximum_wall_seconds,
            result.restore_grace_seconds,
        )
        if any(not math.isfinite(value) or value <= 0 for value in finite_positive):
            raise ValueError("v10 serving durations and rates must be finite and positive")
        if result.output_tokens != 64:
            raise ValueError("v10 serving requires exactly 64 output tokens")
        if result.recovery_stability_seconds < 5.0:
            raise ValueError("v10 serving requires at least five seconds of SLO stability")
        windows = result.recovery_stability_seconds / result.recovery_evaluation_seconds
        if not math.isclose(windows, round(windows), rel_tol=0.0, abs_tol=1e-9):
            raise ValueError("v10 stability must contain whole recovery evaluation windows")
        if result.recovery_queue_threshold < 0:
            raise ValueError("v10 recovery queue threshold cannot be negative")
        if not 1 <= result.maximum_pending_requests <= 20_000:
            raise ValueError("v10 pending-request queue bound is outside 1..20000")
        if not 2 <= result.restore_handoff_lead_requests <= 64:
            raise ValueError("v10 restore handoff lead must be within 2..64 requests")
        if not 10 <= result.overload_queue_trigger <= 30:
            raise ValueError("v10 overload queue trigger must be within the 10..30 target")
        if not result.overload_queue_trigger < result.overload_queue_abort <= 64:
            raise ValueError("v10 overload queue abort must exceed trigger and be at most 64")
        return result


@dataclass(frozen=True)
class OfferedRequest:
    sequence: int
    request_id: str
    phase: Literal["control", "gpu0-overload", "two-gpu-recovery", "restore-interference"]
    scheduled_arrival_ns: int
    device: Literal["gpu0", "gpu1"]
    offered_ns: int | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "sequence": self.sequence,
            "request_id": self.request_id,
            "phase": self.phase,
            "scheduled_arrival_ns": self.scheduled_arrival_ns,
            "device": self.device,
            "offered_ns": self.offered_ns,
        }


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text())
    except FileNotFoundError:
        return None
    if not isinstance(value, dict):
        raise ValueError(f"v10 protocol artifact is not an object: {path}")
    return value


def _interval_ns(rate_per_second: float) -> Fraction:
    if not math.isfinite(rate_per_second) or rate_per_second <= 0:
        raise ValueError("serving rate must be finite and positive")
    rate = Fraction(Decimal(str(rate_per_second)))
    return Fraction(NS_PER_SECOND * rate.denominator, rate.numerator)


def spike_arrival_ns(*, spike_start_ns: int, rate_per_second: float, sequence: int) -> int:
    if spike_start_ns < 0 or sequence < 0:
        raise ValueError("spike clock inputs cannot be negative")
    interval = _interval_ns(rate_per_second)
    return spike_start_ns + (sequence * interval.numerator) // interval.denominator


def recovery_route(*, sequence: int, enable_cutover_sequence: int) -> Literal["gpu0", "gpu1"]:
    if sequence < enable_cutover_sequence:
        return "gpu0"
    return "gpu0" if (sequence - enable_cutover_sequence) % 2 == 0 else "gpu1"


def _p95(values: list[int]) -> float | None:
    if not values:
        return None
    values.sort()
    position = (len(values) - 1) * 0.95
    low = math.floor(position)
    high = math.ceil(position)
    return (
        float(values[low])
        if low == high
        else values[low] * (high - position) + values[high] * (position - low)
    )


class _EngineDriver:
    def __init__(
        self,
        *,
        engine: Any,
        prefix: tuple[int, ...],
        params: Any,
        output_tokens: int,
    ) -> None:
        self.engine = engine
        self.prefix = prefix
        self.params = params
        self.output_tokens = output_tokens
        self.active: set[str] = set()
        self.observations: dict[str, dict[str, Any]] = {}

    def admit(self, request: OfferedRequest) -> None:
        offset = request.sequence % len(self.prefix)
        request_prefix = self.prefix[offset:] + self.prefix[:offset]
        self.engine.add_request(
            request.request_id,
            {"prompt_token_ids": list(request_prefix)},
            self.params,
        )
        self.observations[request.request_id] = {
            **request.as_dict(),
            "admitted_ns": time.monotonic_ns(),
            "service_start_ns": None,
            "first_token_ns": None,
            "completed_ns": None,
            "token_timestamps_ns": [],
            "output_token_ids": [],
        }
        self.active.add(request.request_id)

    def step(self) -> None:
        if not self.active:
            return
        outputs = self.engine.step()
        observed_ns = time.monotonic_ns()
        for output in outputs:
            request_id = str(getattr(output, "request_id", ""))
            row = self.observations.get(request_id)
            if row is None:
                continue
            if row["service_start_ns"] is None:
                # First scheduler-visible output is a conservative service
                # start when vLLM does not expose a monotonic scheduler stamp.
                row["service_start_ns"] = observed_ns
            completions = getattr(output, "outputs", ())
            token_ids = (
                tuple(int(item) for item in getattr(completions[0], "token_ids", ()))
                if completions
                else ()
            )
            old_count = len(row["output_token_ids"])
            if len(token_ids) > old_count:
                row["token_timestamps_ns"].extend([observed_ns] * (len(token_ids) - old_count))
                row["output_token_ids"] = list(token_ids)
                if row["first_token_ns"] is None:
                    row["first_token_ns"] = observed_ns
            if bool(getattr(output, "finished", False)):
                row["completed_ns"] = observed_ns
                self.active.discard(request_id)

    def complete_rows(self) -> tuple[dict[str, Any], ...]:
        return tuple(
            row
            for row in self.observations.values()
            if row["completed_ns"] is not None
            and row["first_token_ns"] is not None
            and row["service_start_ns"] is not None
            and len(row["output_token_ids"]) == self.output_tokens
        )

    def validate_complete(self) -> None:
        if self.active or len(self.complete_rows()) != len(self.observations):
            raise RuntimeError("v10 serving driver stopped with incomplete requests")


class _GlobalGpu0Producer:
    def __init__(
        self,
        *,
        config: LiveV10Config,
        start_ns: int,
        barriers: Path,
        write_new: Callable[[Path, Any], None],
    ) -> None:
        self.config = config
        self.start_ns = start_ns
        self.spike_start_ns = start_ns + int(config.baseline_seconds * NS_PER_SECOND)
        self.barriers = barriers
        self.write_new = write_new
        self.gpu0_queue: queue.Queue[OfferedRequest] = queue.Queue(
            maxsize=config.maximum_pending_requests
        )
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._restore_requested = False
        self._next_spike_sequence = 0
        self._enable_cutover: int | None = None
        self._restore_cutover: int | None = None
        self._restore_start_ns: int | None = None
        self._offered: list[OfferedRequest] = []
        self.error: BaseException | None = None
        self.thread = threading.Thread(target=self._run, name="exp004-v10-arrivals", daemon=False)

    @property
    def offered(self) -> tuple[OfferedRequest, ...]:
        with self._lock:
            return tuple(self._offered)

    @property
    def enable_cutover(self) -> int | None:
        with self._lock:
            return self._enable_cutover

    def start(self) -> None:
        self.thread.start()

    def request_restore(self) -> None:
        with self._lock:
            self._restore_requested = True

    def stop(self) -> None:
        self._stop.set()
        self.thread.join(timeout=10.0)
        if self.thread.is_alive():
            raise TimeoutError("v10 arrival producer did not stop")
        if self.error is not None:
            raise RuntimeError("v10 arrival producer failed") from self.error

    def _offer(self, request: OfferedRequest) -> None:
        offered = OfferedRequest(
            sequence=request.sequence,
            request_id=request.request_id,
            phase=request.phase,
            scheduled_arrival_ns=request.scheduled_arrival_ns,
            device=request.device,
            offered_ns=time.monotonic_ns(),
        )
        with self._lock:
            self._offered.append(offered)
        if offered.device == "gpu0":
            try:
                self.gpu0_queue.put_nowait(offered)
            except queue.Full as exc:
                raise RuntimeError("bounded v10 GPU0 admission queue overflowed") from exc

    def _wait_until(self, target_ns: int) -> bool:
        while not self._stop.is_set():
            remaining = (target_ns - time.monotonic_ns()) / NS_PER_SECOND
            if remaining <= 0:
                return True
            self._stop.wait(min(remaining, 0.02))
        return False

    def _control(self) -> None:
        interval = _interval_ns(self.config.control_rate_per_second)
        sequence = 0
        while True:
            scheduled = self.start_ns + (sequence * interval.numerator) // interval.denominator
            if scheduled >= self.spike_start_ns or not self._wait_until(scheduled):
                return
            request = OfferedRequest(
                sequence=sequence,
                request_id=f"{self.config.attempt_id}.control.{sequence:06d}",
                phase="control",
                scheduled_arrival_ns=scheduled,
                device="gpu0",
            )
            self._offer(request)
            sequence += 1

    def _spike_and_restore(self) -> None:
        restore_sequence = 0
        restore_interval = _interval_ns(self.config.restore_rate_per_second)
        while not self._stop.is_set():
            with self._lock:
                sequence = self._next_spike_sequence
                restore_requested = self._restore_requested
                enable_cutover = self._enable_cutover
                restore_cutover = self._restore_cutover
                restore_start_ns = self._restore_start_ns
            if restore_cutover is not None and sequence >= restore_cutover:
                assert restore_start_ns is not None
                scheduled = (
                    restore_start_ns
                    + (restore_sequence * restore_interval.numerator)
                    // restore_interval.denominator
                )
                if not self._wait_until(scheduled):
                    return
                self._offer(
                    OfferedRequest(
                        sequence=sequence,
                        request_id=f"{self.config.attempt_id}.restore.{restore_sequence:06d}",
                        phase="restore-interference",
                        scheduled_arrival_ns=scheduled,
                        device="gpu0",
                    )
                )
                restore_sequence += 1
                with self._lock:
                    self._next_spike_sequence += 1
                continue
            scheduled = spike_arrival_ns(
                spike_start_ns=self.spike_start_ns,
                rate_per_second=self.config.spike_rate_per_second,
                sequence=sequence,
            )
            if not self._wait_until(scheduled):
                return
            if enable_cutover is None and (self.barriers / "v10-gpu1-serving-ready.json").is_file():
                enable_cutover = sequence
                with self._lock:
                    self._enable_cutover = enable_cutover
                self.write_new(
                    self.barriers / "v10-serving-enable-ack.json",
                    {
                        "schema_version": "sloforge.branchfabric.v10-serving-enable-ack/v1",
                        "spike_start_ns": self.spike_start_ns,
                        "spike_rate_per_second": self.config.spike_rate_per_second,
                        "enable_cutover_sequence": enable_cutover,
                        "enable_cutover_scheduled_ns": scheduled,
                        "observed_ns": time.monotonic_ns(),
                    },
                )
            if restore_requested and restore_cutover is None:
                restore_cutover = sequence + self.config.restore_handoff_lead_requests
                restore_start_ns = spike_arrival_ns(
                    spike_start_ns=self.spike_start_ns,
                    rate_per_second=self.config.spike_rate_per_second,
                    sequence=restore_cutover,
                )
                with self._lock:
                    self._restore_cutover = restore_cutover
                    self._restore_start_ns = restore_start_ns
                self.write_new(
                    self.barriers / "v10-restore-route-cutover.json",
                    {
                        "schema_version": "sloforge.branchfabric.v10-restore-route-cutover/v1",
                        "restore_cutover_sequence": restore_cutover,
                        "restore_start_ns": restore_start_ns,
                        "requested_ns": time.monotonic_ns(),
                    },
                )
            device = (
                "gpu0"
                if enable_cutover is None
                else recovery_route(sequence=sequence, enable_cutover_sequence=enable_cutover)
            )
            phase: Literal["gpu0-overload", "two-gpu-recovery"] = (
                "gpu0-overload" if enable_cutover is None else "two-gpu-recovery"
            )
            self._offer(
                OfferedRequest(
                    sequence=sequence,
                    request_id=f"{self.config.attempt_id}.spike.{sequence:06d}",
                    phase=phase,
                    scheduled_arrival_ns=scheduled,
                    device=device,
                )
            )
            with self._lock:
                self._next_spike_sequence += 1

    def _run(self) -> None:
        try:
            self._control()
            self._spike_and_restore()
        except BaseException as exc:
            self.error = exc
            self._stop.set()


def _drain_admission_queue(
    source: queue.Queue[OfferedRequest],
    driver: _EngineDriver,
    *,
    maximum_active: int,
) -> None:
    while len(driver.active) < maximum_active:
        try:
            driver.admit(source.get_nowait())
        except queue.Empty:
            return


def _load_gpu1_snapshot(
    barriers: Path,
) -> tuple[tuple[dict[str, Any], ...], int | None]:
    snapshots = sorted((barriers / "v10-gpu1-telemetry").glob("*.json"))
    if not snapshots:
        return (), None
    payload = _read_json(snapshots[-1])
    if payload is None or not isinstance(payload.get("requests"), list):
        raise ValueError("GPU1 telemetry snapshot is malformed")
    return tuple(payload["requests"]), int(payload["observed_ns"])


def _combined_rows(
    driver: _EngineDriver, barriers: Path
) -> tuple[tuple[dict[str, Any], ...], int | None]:
    gpu1_rows, gpu1_watermark_ns = _load_gpu1_snapshot(barriers)
    return tuple(driver.complete_rows()) + gpu1_rows, gpu1_watermark_ns


def _trigger_evidence(
    *,
    producer: _GlobalGpu0Producer,
    driver: _EngineDriver,
    window_start_ns: int,
    window_end_ns: int,
) -> dict[str, Any]:
    rows = driver.complete_rows()
    offered_snapshot = producer.offered
    offered = tuple(
        item
        for item in offered_snapshot
        if window_start_ns <= item.scheduled_arrival_ns < window_end_ns
    )
    completed = sum(window_start_ns <= int(row["completed_ns"]) < window_end_ns for row in rows)
    duration = (window_end_ns - window_start_ns) / NS_PER_SECOND
    queue_start = sum(
        item.scheduled_arrival_ns <= window_start_ns
        and not any(
            row["request_id"] == item.request_id and int(row["completed_ns"]) <= window_start_ns
            for row in rows
        )
        for item in offered_snapshot
    )
    queue_end = sum(
        item.scheduled_arrival_ns <= window_end_ns
        and not any(
            row["request_id"] == item.request_id and int(row["completed_ns"]) <= window_end_ns
            for row in rows
        )
        for item in offered_snapshot
    )
    p95 = _p95(
        [
            int(row["first_token_ns"]) - int(row["scheduled_arrival_ns"])
            for row in rows
            if window_start_ns <= int(row["scheduled_arrival_ns"]) < window_end_ns
        ]
    )
    completed_rate = completed / duration
    offered_rate = len(offered) / duration
    queue_slope = (queue_end - queue_start) / duration
    ttft_above_slo = p95 is not None and p95 > 2 * NS_PER_SECOND
    positive_slope = queue_slope > 0.10
    material_deficit = completed_rate < 0.90 * offered_rate
    bounded_backlog = (
        producer.config.overload_queue_trigger <= queue_end <= 25
    )
    reasons: list[str] = []
    if ttft_above_slo:
        reasons.append("p95_ttft_above_slo")
    if positive_slope:
        reasons.append("queue_sustained_positive_drift")
    if material_deficit:
        reasons.append("material_service_deficit")
    if bounded_backlog:
        reasons.append("bounded_backlog_trigger_reached")
    # The load selection already has independent approval.  Trigger at the
    # fixed depth on its approved overload signal instead of waiting for a
    # catastrophic TTFT/deficit threshold that would exceed the preferred
    # 10..25-request backlog range.
    overload_confirmed = bounded_backlog and positive_slope
    return {
        "schema_version": "sloforge.branchfabric.reclamation-trigger-evidence/v1",
        "window_start_ns": window_start_ns,
        "window_end_ns": window_end_ns,
        "offered_snapshot_count": len(offered_snapshot),
        "offered_requests": len(offered),
        "offered_rate_per_second": offered_rate,
        "completed_requests": completed,
        "completed_rate_per_second": completed_rate,
        "queue_depth_start": queue_start,
        "queue_depth_end": queue_end,
        "queue_depth_slope_per_second": queue_slope,
        "p95_ttft_ns": p95,
        "queue_trigger": producer.config.overload_queue_trigger,
        "queue_abort": producer.config.overload_queue_abort,
        "material_service_deficit": material_deficit,
        "trigger_rule": (
            f"fixed_depth_{producer.config.overload_queue_trigger}_with_positive_queue_slope_"
            "and_trigger_depth_at_most_25"
        ),
        "trigger_reason": reasons,
        "overload_confirmed": overload_confirmed,
        "triggered_ns": time.monotonic_ns(),
    }


def _outstanding_at(
    offered: tuple[OfferedRequest, ...], rows: tuple[dict[str, Any], ...], timestamp_ns: int
) -> int:
    completed = {
        str(row["request_id"])
        for row in rows
        if row.get("completed_ns") is not None and int(row["completed_ns"]) <= timestamp_ns
    }
    return sum(
        item.scheduled_arrival_ns <= timestamp_ns and item.request_id not in completed
        for item in offered
    )


def _enforce_pre_gpu1_queue_abort(
    *,
    trigger_written: bool,
    gpu1_first_useful: dict[str, Any] | None,
    instantaneous_depth: int,
    maximum_depth: int,
) -> None:
    """Keep the hard backlog ceiling live through the preservation interval."""

    if instantaneous_depth < 0 or maximum_depth < 0:
        raise ValueError("v10 queue-depth abort inputs cannot be negative")
    if gpu1_first_useful is None and instantaneous_depth > maximum_depth:
        phase = "post-reclaim-pre-gpu1" if trigger_written else "pre-reclaim"
        raise RuntimeError(
            f"calibrated v10 backlog exceeded its bounded safety ceiling during {phase}"
        )


def _validated_runtime_drain_state(state: dict[str, int]) -> dict[str, int]:
    """Fail closed unless the live vLLM scheduler is observably empty."""

    required = (
        "request_count",
        "running_requests",
        "waiting_requests",
        "skipped_waiting_requests",
        "queue_depth",
    )
    if set(state) != set(required) or any(
        isinstance(state[name], bool) or not isinstance(state[name], int) or state[name] < 0
        for name in required
    ):
        raise ValueError("GPU1 runtime drain state is incomplete or malformed")
    if state["queue_depth"] != state["waiting_requests"] + state["skipped_waiting_requests"]:
        raise ValueError("GPU1 runtime drain queue-depth accounting is inconsistent")
    if any(state[name] != 0 for name in required):
        raise RuntimeError("GPU1 vLLM scheduler was not fully drained before restore")
    return dict(state)


def _sustained_queue_trend(
    *,
    config: LiveV10Config,
    offered: tuple[OfferedRequest, ...],
    rows: tuple[dict[str, Any], ...],
    window_end_ns: int,
) -> dict[str, Any]:
    """Measure a complete declared window; an endpoint decrease is insufficient."""

    stability_ns = int(config.recovery_stability_seconds * NS_PER_SECOND)
    evaluation_ns = int(config.recovery_evaluation_seconds * NS_PER_SECOND)
    window_start_ns = window_end_ns - stability_ns
    timestamps = tuple(range(window_start_ns, window_end_ns + 1, evaluation_ns))
    if (
        window_start_ns < 0
        or not timestamps
        or timestamps[-1] != window_end_ns
        or len(timestamps) < 3
    ):
        raise ValueError("v10 queue-trend window cannot be sampled at exact boundaries")
    depths = tuple(_outstanding_at(offered, rows, timestamp) for timestamp in timestamps)
    seconds = tuple((timestamp - window_start_ns) / NS_PER_SECOND for timestamp in timestamps)
    mean_x = statistics.fmean(seconds)
    mean_y = statistics.fmean(depths)
    denominator = sum((value - mean_x) ** 2 for value in seconds)
    slope = (
        sum(
            (x_value - mean_x) * (y_value - mean_y)
            for x_value, y_value in zip(seconds, depths, strict=True)
        )
        / denominator
    )
    midpoint = len(depths) // 2
    first_half_mean = statistics.fmean(depths[:midpoint])
    second_half_mean = statistics.fmean(depths[midpoint:])
    duration = stability_ns / NS_PER_SECOND
    offered_count = sum(
        window_start_ns <= item.scheduled_arrival_ns < window_end_ns for item in offered
    )
    completed_count = sum(
        row.get("completed_ns") is not None
        and window_start_ns <= int(row["completed_ns"]) < window_end_ns
        for row in rows
    )
    offered_rate = offered_count / duration
    completed_rate = completed_count / duration
    sustained_negative = (
        slope < 0.0 and depths[-1] < depths[0] and second_half_mean < first_half_mean
    )
    return {
        "window_start_ns": window_start_ns,
        "window_end_ns": window_end_ns,
        "sample_interval_ns": evaluation_ns,
        "samples": [
            {"timestamp_ns": timestamp, "queue_depth": depth}
            for timestamp, depth in zip(timestamps, depths, strict=True)
        ],
        "initial_depth": depths[0],
        "final_depth": depths[-1],
        "first_half_mean_depth": first_half_mean,
        "second_half_mean_depth": second_half_mean,
        "slope_requests_per_second": slope,
        "offered_requests": offered_count,
        "completed_requests": completed_count,
        "offered_rate_per_second": offered_rate,
        "completed_rate_per_second": completed_rate,
        "sustained_negative": sustained_negative,
        "completed_rate_exceeds_offered": completed_rate > offered_rate,
        "passed": sustained_negative and completed_rate > offered_rate,
    }


def _recovery_evidence(
    *,
    config: LiveV10Config,
    producer: _GlobalGpu0Producer,
    rows: tuple[dict[str, Any], ...],
    gpu1_first_useful_ns: int,
    now_ns: int,
    candidate_start_ns: int | None,
) -> tuple[dict[str, Any], int | None]:
    offered = producer.offered
    queue_now = _outstanding_at(offered, rows, now_ns)
    queue_first = _outstanding_at(offered, rows, gpu1_first_useful_ns)
    if candidate_start_ns is None:
        minimum_trend_ns = int(config.recovery_stability_seconds * NS_PER_SECOND)
        if now_ns >= gpu1_first_useful_ns + minimum_trend_ns:
            trend = _sustained_queue_trend(
                config=config,
                offered=offered,
                rows=rows,
                window_end_ns=now_ns,
            )
            if queue_now <= config.recovery_queue_threshold and trend["passed"]:
                return {}, now_ns
        return {}, None
    stability_ns = int(config.recovery_stability_seconds * NS_PER_SECOND)
    evaluation_ns = int(config.recovery_evaluation_seconds * NS_PER_SECOND)
    if now_ns < candidate_start_ns + stability_ns + 2 * NS_PER_SECOND:
        if queue_now > config.recovery_queue_threshold:
            return {}, None
        return {}, candidate_start_ns
    windows: list[dict[str, Any]] = []
    cursor = candidate_start_ns
    while cursor < candidate_start_ns + stability_ns:
        end = cursor + evaluation_ns
        cohort = tuple(item for item in offered if cursor <= item.scheduled_arrival_ns < end)
        by_id = {str(row["request_id"]): row for row in rows}
        ttft = [
            int(by_id[item.request_id]["first_token_ns"]) - item.scheduled_arrival_ns
            for item in cohort
            if item.request_id in by_id and by_id[item.request_id].get("first_token_ns") is not None
        ]
        depth = _outstanding_at(offered, rows, end)
        p95 = _p95(ttft)
        passed = (
            bool(cohort)
            and len(ttft) == len(cohort)
            and p95 is not None
            and p95 <= 2 * NS_PER_SECOND
            and depth <= config.recovery_queue_threshold
        )
        windows.append(
            {
                "start_ns": cursor,
                "end_ns": end,
                "offered_requests": len(cohort),
                "ttft_sample_count": len(ttft),
                "p95_ttft_ns": p95,
                "queue_depth_end": depth,
                "passed": passed,
            }
        )
        cursor = end
    trend = _sustained_queue_trend(
        config=config,
        offered=offered,
        rows=rows,
        window_end_ns=candidate_start_ns,
    )
    completed_rate = float(trend["completed_rate_per_second"])
    offered_rate = float(trend["offered_rate_per_second"])
    queue_candidate = _outstanding_at(offered, rows, candidate_start_ns)
    slope = float(trend["slope_requests_per_second"])
    evidence = {
        "schema_version": "sloforge.branchfabric.serving-recovery-evidence/v1",
        "gpu1_first_useful_ns": gpu1_first_useful_ns,
        "evaluated_ns": now_ns,
        "offered_rate_per_second": offered_rate,
        "completed_rate_per_second": completed_rate,
        "queue_depth_at_gpu1_first_useful": queue_first,
        "queue_depth_at_stability_start": queue_candidate,
        "queue_depth_at_evaluation": queue_now,
        "queue_depth_slope_per_second": slope,
        "queue_trend": trend,
        "recovery_queue_threshold": config.recovery_queue_threshold,
        "stability_windows": windows,
        "two_gpu_excess_capacity_pass": bool(trend["completed_rate_exceeds_offered"]),
        "queue_drain_pass": bool(trend["sustained_negative"]),
        "slo_restoration_pass": all(row["passed"] for row in windows),
        "pre_restore_stability_pass": all(row["passed"] for row in windows),
    }
    evidence["restore_eligible"] = (
        all(
            evidence[key]
            for key in (
                "two_gpu_excess_capacity_pass",
                "queue_drain_pass",
                "slo_restoration_pass",
                "pre_restore_stability_pass",
            )
        )
        and queue_now <= config.recovery_queue_threshold
    )
    return evidence, candidate_start_ns


def run_v10_gpu0(
    engine: Any,
    *,
    prefix: tuple[int, ...],
    params: Any,
    config: LiveV10Config,
    start_ns: int,
    barriers: Path,
    write_new: Callable[[Path, Any], None],
) -> dict[str, Any]:
    """Run the global clock, GPU0 shard, trigger, recovery, and restore gates."""

    producer = _GlobalGpu0Producer(
        config=config,
        start_ns=start_ns,
        barriers=barriers,
        write_new=write_new,
    )
    driver = _EngineDriver(
        engine=engine,
        prefix=prefix,
        params=params,
        output_tokens=config.output_tokens,
    )
    producer.start()
    deadline_ns = start_ns + int(config.maximum_wall_seconds * NS_PER_SECOND)
    trigger_deadline_ns = producer.spike_start_ns + int(
        config.overload_probe_seconds * NS_PER_SECOND
    )
    trigger_written = False
    recovery_candidate: int | None = None
    recovery: dict[str, Any] | None = None
    restore_started_ns: int | None = None
    complete_observed_ns: int | None = None
    try:
        while True:
            now = time.monotonic_ns()
            if now >= deadline_ns:
                raise TimeoutError("v10 GPU0 serving protocol exceeded its wall bound")
            if producer.error is not None:
                raise RuntimeError("v10 arrival producer failed") from producer.error
            _drain_admission_queue(
                producer.gpu0_queue,
                driver,
                maximum_active=config.maximum_pending_requests,
            )
            driver.step()
            # Re-sample after ``engine.step`` so a request completed in this
            # iteration is not conservatively counted as still outstanding.
            now = time.monotonic_ns()
            first_useful = _read_json(barriers / "v10-gpu1-first-useful.json")
            instantaneous_depth = _outstanding_at(producer.offered, driver.complete_rows(), now)
            _enforce_pre_gpu1_queue_abort(
                trigger_written=trigger_written,
                gpu1_first_useful=first_useful,
                instantaneous_depth=instantaneous_depth,
                maximum_depth=config.overload_queue_abort,
            )
            if not trigger_written and now >= trigger_deadline_ns:
                trigger = _trigger_evidence(
                    producer=producer,
                    driver=driver,
                    window_start_ns=max(
                        producer.spike_start_ns,
                        now - int(config.overload_probe_seconds * NS_PER_SECOND),
                    ),
                    window_end_ns=now,
                )
                if not trigger["overload_confirmed"]:
                    time.sleep(_POLL_SECONDS)
                    continue
                write_new(barriers / "v10-reclaim-trigger.json", trigger)
                trigger_written = True
            # An empty evidence object means the drain/stability candidate has
            # not matured yet.  Keep evaluating until all recovery gates pass;
            # treating the first empty result as terminal would deadlock both
            # workers at the restore barrier.
            if first_useful is not None and not (
                recovery is not None and recovery.get("restore_eligible")
            ):
                rows, gpu1_watermark_ns = _combined_rows(driver, barriers)
                if gpu1_watermark_ns is None:
                    continue
                recovery, recovery_candidate = _recovery_evidence(
                    config=config,
                    producer=producer,
                    rows=rows,
                    gpu1_first_useful_ns=int(first_useful["first_token_ns"]),
                    # Queue depth must be evaluated only through the newest
                    # timestamp for which both workers have published complete
                    # observations.  Advancing on GPU0's clock while GPU1's
                    # snapshot is stale invents a growing queue.
                    now_ns=min(now, gpu1_watermark_ns),
                    candidate_start_ns=recovery_candidate,
                )
                if recovery and recovery["restore_eligible"]:
                    write_new(barriers / "v10-serving-recovery.json", recovery)
                    producer.request_restore()
            if recovery is not None and recovery.get("restore_eligible"):
                drained = _read_json(barriers / "v10-gpu1-drained.json")
                cutover = _read_json(barriers / "v10-restore-route-cutover.json")
                if drained is not None and cutover is not None and restore_started_ns is None:
                    if int(drained["last_admitted_sequence"]) >= int(
                        cutover["restore_cutover_sequence"]
                    ):
                        raise RuntimeError("GPU1 admitted work at or beyond restore cutover")
                    if (
                        int(drained.get("running_requests", -1)) != 0
                        or int(drained.get("waiting_requests", -1)) != 0
                    ):
                        raise RuntimeError("GPU1 restore handoff was not fully drained")
                    if now < int(cutover["restore_start_ns"]):
                        continue
                    if not any(
                        item.phase == "restore-interference" and item.offered_ns is not None
                        for item in producer.offered
                    ):
                        continue
                    restore_started_ns = time.monotonic_ns()
                    write_new(
                        barriers / "v10-restore-start.json",
                        {
                            "schema_version": "sloforge.branchfabric.v10-restore-start/v1",
                            "observed_ns": restore_started_ns,
                            "restore_cutover_sequence": cutover["restore_cutover_sequence"],
                            "gpu1_drained_ns": drained["observed_ns"],
                        },
                    )
            if restore_started_ns is not None:
                completed = _read_json(barriers / "rollout-restore-complete.json")
                if completed is not None and complete_observed_ns is None:
                    complete_observed_ns = time.monotonic_ns()
                if complete_observed_ns is not None and now >= complete_observed_ns + int(
                    config.restore_grace_seconds * NS_PER_SECOND
                ):
                    producer.stop()
                    _drain_admission_queue(
                        producer.gpu0_queue,
                        driver,
                        maximum_active=config.maximum_pending_requests,
                    )
                    if not driver.active:
                        break
            if not driver.active and producer.gpu0_queue.empty():
                time.sleep(_POLL_SECONDS)
        driver.validate_complete()
    finally:
        if producer.thread.is_alive():
            producer.stop()
    return {
        "schema_version": "sloforge.branchfabric.experiment-004-v10-serving-raw/v1",
        "methodology": "v10-global-capacity",
        "start_ns": start_ns,
        "spike_start_ns": producer.spike_start_ns,
        "end_ns": time.monotonic_ns(),
        "requests": tuple(driver.observations.values()),
        "global_offered_requests": tuple(item.as_dict() for item in producer.offered),
        "reclamation_trigger_evidence": _read_json(barriers / "v10-reclaim-trigger.json"),
        "serving_recovery_evidence": _read_json(barriers / "v10-serving-recovery.json"),
        "serving_enable_ack": _read_json(barriers / "v10-serving-enable-ack.json"),
        "restore_route_cutover": _read_json(barriers / "v10-restore-route-cutover.json"),
        "gpu1_drained": _read_json(barriers / "v10-gpu1-drained.json"),
        "restore_start": _read_json(barriers / "v10-restore-start.json"),
        "prefix_cache_policy": {
            "enabled_for_rollout_semantics": True,
            "serving_reuse_prevented": True,
            "method": "request-unique-first-block-rotation-transitive-hash-chain",
        },
    }


def run_v10_gpu1(
    engine: Any,
    *,
    prefix: tuple[int, ...],
    params: Any,
    config: LiveV10Config,
    barriers: Path,
    write_new: Callable[[Path, Any], None],
    runtime_queue_state: Callable[[], dict[str, int]],
) -> dict[str, Any]:
    """Serve only GPU1's acknowledged shard, publish telemetry, then drain."""

    write_new(
        barriers / "v10-gpu1-serving-ready.json",
        {
            "schema_version": "sloforge.branchfabric.v10-gpu1-serving-ready/v1",
            "observed_ns": time.monotonic_ns(),
        },
    )
    deadline_ns = time.monotonic_ns() + int(config.maximum_wall_seconds * NS_PER_SECOND)
    while (ack := _read_json(barriers / "v10-serving-enable-ack.json")) is None:
        if time.monotonic_ns() >= deadline_ns:
            raise TimeoutError("GPU1 timed out waiting for global route acknowledgement")
        time.sleep(_POLL_SECONDS)
    cutover = int(ack["enable_cutover_sequence"])
    spike_start_ns = int(ack["spike_start_ns"])
    driver = _EngineDriver(
        engine=engine,
        prefix=prefix,
        params=params,
        output_tokens=config.output_tokens,
    )
    sequence = cutover
    telemetry_sequence = 0
    last_snapshot_ns = 0
    restore_cutover: int | None = None
    restore_start_ns: int | None = None
    last_admitted = -1
    while True:
        now = time.monotonic_ns()
        if now >= deadline_ns:
            raise TimeoutError("v10 GPU1 serving protocol exceeded its wall bound")
        restore = _read_json(barriers / "v10-restore-route-cutover.json")
        if restore is not None:
            restore_cutover = int(restore["restore_cutover_sequence"])
            restore_start_ns = int(restore["restore_start_ns"])
        scheduled = spike_arrival_ns(
            spike_start_ns=spike_start_ns,
            rate_per_second=config.spike_rate_per_second,
            sequence=sequence,
        )
        if (
            (restore_cutover is None or sequence < restore_cutover)
            and now >= scheduled
            and len(driver.active) < config.maximum_pending_requests
        ):
            if recovery_route(sequence=sequence, enable_cutover_sequence=cutover) == "gpu1":
                request = OfferedRequest(
                    sequence=sequence,
                    request_id=f"{config.attempt_id}.spike.{sequence:06d}",
                    phase="two-gpu-recovery",
                    scheduled_arrival_ns=scheduled,
                    device="gpu1",
                )
                driver.admit(request)
                last_admitted = sequence
            sequence += 1
        driver.step()
        complete = driver.complete_rows()
        if complete and not (barriers / "v10-gpu1-first-useful.json").is_file():
            first = min(complete, key=lambda row: int(row["first_token_ns"]))
            write_new(
                barriers / "v10-gpu1-first-useful.json",
                {
                    "schema_version": "sloforge.branchfabric.v10-gpu1-first-useful/v1",
                    "request_id": first["request_id"],
                    "sequence": first["sequence"],
                    "first_token_ns": first["first_token_ns"],
                },
            )
        if complete and now - last_snapshot_ns >= int(config.recovery_evaluation_seconds * 1e9):
            snapshot_ns = time.monotonic_ns()
            write_new(
                barriers / "v10-gpu1-telemetry" / f"{telemetry_sequence:06d}.json",
                {
                    "schema_version": "sloforge.branchfabric.v10-gpu1-telemetry/v1",
                    "sequence": telemetry_sequence,
                    "observed_ns": snapshot_ns,
                    "requests": complete,
                },
            )
            telemetry_sequence += 1
            last_snapshot_ns = snapshot_ns
        if (
            restore_cutover is not None
            and restore_start_ns is not None
            and sequence >= restore_cutover
            and now >= restore_start_ns
            and not driver.active
        ):
            scheduler_state = _validated_runtime_drain_state(runtime_queue_state())
            final = driver.complete_rows()
            write_new(
                barriers / "v10-gpu1-telemetry" / f"{telemetry_sequence:06d}.json",
                {
                    "schema_version": "sloforge.branchfabric.v10-gpu1-telemetry/v1",
                    "sequence": telemetry_sequence,
                    "observed_ns": time.monotonic_ns(),
                    "requests": final,
                },
            )
            write_new(
                barriers / "v10-gpu1-drained.json",
                {
                    "schema_version": "sloforge.branchfabric.v10-gpu1-drained/v1",
                    "observed_ns": time.monotonic_ns(),
                    "restore_cutover_sequence": restore_cutover,
                    "last_admitted_sequence": last_admitted,
                    "request_count": len(final),
                    "running_requests": scheduler_state["running_requests"],
                    "waiting_requests": scheduler_state["waiting_requests"],
                    "skipped_waiting_requests": scheduler_state[
                        "skipped_waiting_requests"
                    ],
                    "scheduler_request_count": scheduler_state["request_count"],
                    "queue_depth": scheduler_state["queue_depth"],
                    "runtime_state_source": "live-vllm-0.23-scheduler",
                },
            )
            break
        if not driver.active and now < scheduled:
            time.sleep(min(_POLL_SECONDS, max(0.0, (scheduled - now) / NS_PER_SECOND)))
    while _read_json(barriers / "v10-restore-start.json") is None:
        if time.monotonic_ns() >= deadline_ns:
            raise TimeoutError("GPU1 timed out waiting for restore-start acknowledgement")
        time.sleep(_POLL_SECONDS)
    driver.validate_complete()
    return {
        "schema_version": "sloforge.branchfabric.experiment-004-v10-gpu1-serving-raw/v1",
        "methodology": "v10-global-capacity",
        "start_ns": spike_start_ns,
        "end_ns": time.monotonic_ns(),
        "requests": tuple(driver.observations.values()),
        "enable_ack": ack,
        "restore_cutover_sequence": restore_cutover,
        "last_admitted_sequence": last_admitted,
    }


__all__ = [
    "LiveV10Config",
    "OfferedRequest",
    "recovery_route",
    "run_v10_gpu0",
    "run_v10_gpu1",
    "spike_arrival_ns",
]
