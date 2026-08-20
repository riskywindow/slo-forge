"""Early, fail-closed overload trigger for the integrated v11 transaction.

The frozen v10 controller remains the owner of the global arrival clock,
routing, recovery, and restore handoff.  This module applies one narrowly
scoped v11 control-plane overlay: the normal reclaim predicate is evaluated
over a short confirmation window as soon as backlog reaches its configured
threshold, while the emergency ceiling remains a separate abort condition.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Any

import gpu_reclamation_v10_serving as _v10

NORMAL_RECLAIM_CONFIRMATION_SECONDS = 0.25
_OVERLAY_LOCK = threading.Lock()
_ACTIVE_CONTEXT: _TriggerOverlayContext | None = None


class _TriggerOverlayContext:
    """Single-transaction state shared by the scoped producer/driver types."""

    def __init__(self, write_new: Callable[[Path, Any], None]) -> None:
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.published = threading.Event()
        self.producer: Any | None = None
        self.driver: Any | None = None
        self.thread: threading.Thread | None = None
        self.error: BaseException | None = None
        self.write_new = write_new

    def register_producer(self, producer: Any) -> None:
        self.producer = producer

    def register_driver(self, driver: Any) -> None:
        self.driver = driver

    def start(self) -> None:
        if self.producer is None or self.driver is None or self.thread is not None:
            raise RuntimeError("v11 trigger overlay lifecycle is incomplete")
        self.thread = threading.Thread(
            target=self._monitor,
            name="exp004-v11-early-reclaim-trigger",
            daemon=False,
        )
        self.thread.start()

    def publish(self, path: Path, value: Any) -> None:
        producer = self.producer
        if producer is None:
            raise RuntimeError("v11 trigger publisher has no producer")
        trigger_path = producer.barriers / "v10-reclaim-trigger.json"
        if path != trigger_path:
            self.write_new(path, value)
            return
        with self.lock:
            if self.published.is_set():
                return
            if not isinstance(value, dict) or not isinstance(value.get("events"), dict):
                raise RuntimeError("v11 trigger publisher received malformed evidence")
            payload = dict(value)
            events = dict(value["events"])
            predicate_ns = events.get("TRIGGER_PREDICATE_SATISFIED")
            if isinstance(predicate_ns, bool) or not isinstance(predicate_ns, int):
                raise RuntimeError("v11 trigger predicate timestamp is malformed")
            # Stamp the actual control-plane emission while holding the
            # one-shot lock, immediately before immutable fsync/publication.
            emitted_ns = time.monotonic_ns()
            if emitted_ns < predicate_ns:
                raise RuntimeError("v11 trigger emission preceded its predicate")
            events["RECLAIM_TRIGGER_EMITTED"] = emitted_ns
            payload["events"] = events
            payload["reclaim_trigger_emitted_ns"] = emitted_ns
            payload["controller_reaction_latency_ns"] = emitted_ns - predicate_ns
            # Compatibility with the frozen GPU1 barrier consumer.
            payload["triggered_ns"] = emitted_ns
            self.write_new(path, payload)
            self.published.set()

    def _monitor(self) -> None:
        assert self.producer is not None and self.driver is not None
        producer = self.producer
        driver = self.driver
        try:
            while not self.stop.is_set():
                now_ns = time.monotonic_ns()
                if now_ns < producer.spike_start_ns:
                    self.stop.wait(
                        min(
                            (producer.spike_start_ns - now_ns) / _v10.NS_PER_SECOND,
                            0.02,
                        )
                    )
                    continue
                offered = producer.offered
                rows = driver.observation_rows()
                outstanding = _v10._outstanding_at(offered, rows, now_ns)
                if outstanding >= producer.config.overload_queue_abort:
                    raise RuntimeError(
                        "integrated v11 backlog reached its emergency ceiling "
                        "during independent pre-reclaim monitoring"
                    )
                if self.published.is_set():
                    # Keep the independent emergency guard live until GPU1 has
                    # completed useful serving work; the main engine step may
                    # remain blocked during export/reclamation.
                    if (producer.barriers / "v10-gpu1-first-useful.json").is_file():
                        return
                    self.stop.wait(_v10._POLL_SECONDS)
                    continue
                confirmation_ns = int(NORMAL_RECLAIM_CONFIRMATION_SECONDS * _v10.NS_PER_SECOND)
                if now_ns >= producer.spike_start_ns + confirmation_ns:
                    trigger = build_early_trigger_evidence_v11(
                        producer=producer,
                        driver=driver,
                        window_start_ns=max(producer.spike_start_ns, now_ns - confirmation_ns),
                        window_end_ns=now_ns,
                    )
                    if trigger["overload_confirmed"]:
                        self.publish(producer.barriers / "v10-reclaim-trigger.json", trigger)
                        continue
                self.stop.wait(_v10._POLL_SECONDS)
        except BaseException as error:
            self.error = error
            producer.error = error
            producer._stop.set()

    def raise_if_failed(self) -> None:
        if self.error is not None:
            raise RuntimeError("integrated v11 independent trigger monitor failed") from self.error

    def close(self) -> None:
        self.stop.set()
        if self.thread is not None:
            self.thread.join(timeout=2.0)
            if self.thread.is_alive():
                raise TimeoutError("integrated v11 trigger monitor did not stop")


def _active_context() -> _TriggerOverlayContext:
    if _ACTIVE_CONTEXT is None:
        raise RuntimeError("v11 trigger overlay is not active")
    return _ACTIVE_CONTEXT


class _EarlyTriggerProducer(_v10._GlobalGpu0Producer):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        _active_context().register_producer(self)

    def start(self) -> None:
        super().start()
        _active_context().start()


class _EarlyTriggerDriver(_v10._EngineDriver):
    """Expose consistent snapshots while ``engine.step`` itself is blocked."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._snapshot_lock = threading.Lock()
        _active_context().register_driver(self)

    def admit(self, request: Any) -> None:
        _active_context().raise_if_failed()
        offset = request.sequence % len(self.prefix)
        request_prefix = self.prefix[offset:] + self.prefix[:offset]
        self.engine.add_request(
            request.request_id,
            {"prompt_token_ids": list(request_prefix)},
            self.params,
        )
        with self._snapshot_lock:
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
        _active_context().raise_if_failed()
        if not self.active:
            return
        outputs = self.engine.step()
        observed_ns = time.monotonic_ns()
        with self._snapshot_lock:
            for output in outputs:
                request_id = str(getattr(output, "request_id", ""))
                row = self.observations.get(request_id)
                if row is None:
                    continue
                if row["service_start_ns"] is None:
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
        _active_context().raise_if_failed()

    def complete_rows(self) -> tuple[dict[str, Any], ...]:
        with self._snapshot_lock:
            return tuple(
                dict(row)
                for row in self.observations.values()
                if row["completed_ns"] is not None
                and row["first_token_ns"] is not None
                and row["service_start_ns"] is not None
                and len(row["output_token_ids"]) == self.output_tokens
            )

    def observation_rows(self) -> tuple[dict[str, Any], ...]:
        with self._snapshot_lock:
            return tuple(dict(row) for row in self.observations.values())

    def active_telemetry_rows(self) -> tuple[dict[str, Any], ...]:
        fields = (
            "sequence",
            "request_id",
            "phase",
            "scheduled_arrival_ns",
            "device",
            "admitted_ns",
            "service_start_ns",
            "first_token_ns",
        )
        with self._snapshot_lock:
            return tuple(
                {field_name: row[field_name] for field_name in fields}
                for request_id, row in self.observations.items()
                if request_id in self.active
            )

    def validate_complete(self) -> None:
        with self._snapshot_lock:
            active = bool(self.active)
            observations = len(self.observations)
        if active or len(self.complete_rows()) != observations:
            raise RuntimeError("v11 serving driver stopped with incomplete requests")


def _completed_by_id(rows: tuple[dict[str, Any], ...]) -> dict[str, int]:
    return {str(row["request_id"]): int(row["completed_ns"]) for row in rows}


def build_early_trigger_evidence_v11(
    *,
    producer: Any,
    driver: Any,
    window_start_ns: int,
    window_end_ns: int,
) -> dict[str, Any]:
    """Evaluate the exact normal trigger over one completed evidence window."""

    if window_end_ns <= window_start_ns:
        raise ValueError("v11 trigger evidence window must have positive duration")
    completed_rows = driver.complete_rows()
    completed_by_id = _completed_by_id(completed_rows)
    offered_snapshot = producer.offered
    offered = tuple(
        item
        for item in offered_snapshot
        if window_start_ns <= item.scheduled_arrival_ns < window_end_ns
    )
    completed = sum(
        window_start_ns <= completed_ns < window_end_ns for completed_ns in completed_by_id.values()
    )
    duration_seconds = (window_end_ns - window_start_ns) / _v10.NS_PER_SECOND

    def outstanding_at(timestamp_ns: int) -> int:
        return sum(
            item.scheduled_arrival_ns <= timestamp_ns
            and completed_by_id.get(item.request_id, timestamp_ns + 1) > timestamp_ns
            for item in offered_snapshot
        )

    queue_start = outstanding_at(window_start_ns)
    queue_end = outstanding_at(window_end_ns)
    offered_rate = len(offered) / duration_seconds
    completed_rate = completed / duration_seconds
    queue_slope = (queue_end - queue_start) / duration_seconds
    positive_slope = queue_slope > 0.0
    offered_exceeds_completed = offered_rate > completed_rate
    normal_depth_reached = queue_end >= producer.config.overload_queue_trigger
    normal_depth_confirmed = (
        queue_start >= producer.config.overload_queue_trigger and normal_depth_reached
    )
    below_emergency_ceiling = queue_end < producer.config.overload_queue_abort
    overload_confirmed = (
        normal_depth_confirmed
        and positive_slope
        and offered_exceeds_completed
        and below_emergency_ceiling
    )
    predicate_satisfied_ns = window_end_ns
    emitted_ns = time.monotonic_ns()
    p95 = _v10._p95(
        [
            int(row["first_token_ns"]) - int(row["scheduled_arrival_ns"])
            for row in completed_rows
            if window_start_ns <= int(row["scheduled_arrival_ns"]) < window_end_ns
        ]
    )
    reasons = []
    if normal_depth_reached:
        reasons.append("normal_backlog_threshold_reached")
    if positive_slope:
        reasons.append("queue_slope_positive")
    if offered_exceeds_completed:
        reasons.append("offered_rate_exceeds_completed_rate")
    return {
        "schema_version": "sloforge.branchfabric.reclamation-trigger-evidence/v2",
        "controller": "integrated-v11-early-trigger-overlay",
        "window_start_ns": window_start_ns,
        "window_end_ns": window_end_ns,
        "confirmation_window_seconds": duration_seconds,
        "required_confirmation_window_seconds": NORMAL_RECLAIM_CONFIRMATION_SECONDS,
        "offered_snapshot_count": len(offered_snapshot),
        "offered_requests": len(offered),
        "offered_rate_per_second": offered_rate,
        "completed_requests": completed,
        "completed_rate_per_second": completed_rate,
        "offered_rate_exceeds_completed_rate": offered_exceeds_completed,
        "queue_depth_start": queue_start,
        "queue_depth_end": queue_end,
        "queue_depth_at_overload_detection": queue_start,
        "queue_depth_at_trigger": queue_end,
        "queue_depth_slope_per_second": queue_slope,
        "positive_queue_slope": positive_slope,
        "p95_ttft_ns": p95,
        "queue_trigger": producer.config.overload_queue_trigger,
        "queue_abort": producer.config.overload_queue_abort,
        "emergency_ceiling_headroom_requests": (producer.config.overload_queue_abort - queue_end),
        "material_service_deficit": offered_exceeds_completed,
        "trigger_rule": (
            "outstanding_requests>=normal_trigger AND queue_slope>0 AND "
            "offered_rate>completed_rate over confirmation_window"
        ),
        "trigger_reason": reasons,
        "overload_confirmed": overload_confirmed,
        "events": {
            "OVERLOAD_DETECTED": window_start_ns,
            "TRIGGER_PREDICATE_SATISFIED": predicate_satisfied_ns,
            "RECLAIM_TRIGGER_EMITTED": emitted_ns,
        },
        "overload_detected_ns": window_start_ns,
        "trigger_predicate_satisfied_ns": predicate_satisfied_ns,
        "reclaim_trigger_emitted_ns": emitted_ns,
        "controller_reaction_latency_ns": emitted_ns - predicate_satisfied_ns,
        # Compatibility with the frozen GPU1 barrier consumer.
        "triggered_ns": emitted_ns,
    }


def enforce_emergency_queue_abort_v11(
    *, total_outstanding: int, maximum_depth: int, phase: str = "two-gpu-recovery"
) -> None:
    """Abort at the ceiling itself; the normal trigger is not this guard."""

    if total_outstanding < 0 or maximum_depth < 0:
        raise ValueError("v11 queue-depth abort inputs cannot be negative")
    if total_outstanding >= maximum_depth:
        raise RuntimeError(f"integrated v11 backlog reached its emergency ceiling during {phase}")


@contextmanager
def _scoped_trigger_overlay(write_new: Callable[[Path, Any], None]) -> Any:
    """Install the v11 trigger only for one serialized GPU0 transaction."""

    global _ACTIVE_CONTEXT
    with _OVERLAY_LOCK:
        if _ACTIVE_CONTEXT is not None:
            raise RuntimeError("nested integrated v11 trigger overlays are forbidden")
        context = _TriggerOverlayContext(write_new)
        _ACTIVE_CONTEXT = context
        original_trigger = _v10._trigger_evidence
        original_abort = _v10._enforce_recovery_queue_bound
        original_producer = _v10._GlobalGpu0Producer
        original_driver = _v10._EngineDriver
        _v10._trigger_evidence = build_early_trigger_evidence_v11
        _v10._enforce_recovery_queue_bound = enforce_emergency_queue_abort_v11
        _v10._GlobalGpu0Producer = _EarlyTriggerProducer
        _v10._EngineDriver = _EarlyTriggerDriver
        try:
            yield context
        finally:
            try:
                context.close()
            finally:
                _v10._trigger_evidence = original_trigger
                _v10._enforce_recovery_queue_bound = original_abort
                _v10._GlobalGpu0Producer = original_producer
                _v10._EngineDriver = original_driver
                _ACTIVE_CONTEXT = None


def _validate_emitted_trigger(trigger: Any) -> dict[str, Any]:
    if not isinstance(trigger, dict):
        raise RuntimeError("integrated v11 did not emit normal reclaim evidence")
    events = trigger.get("events")
    required = (
        "OVERLOAD_DETECTED",
        "TRIGGER_PREDICATE_SATISFIED",
        "RECLAIM_TRIGGER_EMITTED",
    )
    if (
        trigger.get("overload_confirmed") is not True
        or trigger.get("positive_queue_slope") is not True
        or trigger.get("offered_rate_exceeds_completed_rate") is not True
        or not isinstance(events, dict)
        or any(
            isinstance(events.get(name), bool) or not isinstance(events.get(name), int)
            for name in required
        )
        or not events["OVERLOAD_DETECTED"]
        <= events["TRIGGER_PREDICATE_SATISFIED"]
        <= events["RECLAIM_TRIGGER_EMITTED"]
        or trigger.get("triggered_ns") != events["RECLAIM_TRIGGER_EMITTED"]
        or int(trigger.get("queue_depth_at_trigger", -1)) < int(trigger.get("queue_trigger", 0))
        or int(trigger.get("queue_depth_at_trigger", -1)) >= int(trigger.get("queue_abort", 0))
    ):
        raise RuntimeError("integrated v11 early reclaim trigger evidence is invalid")
    return dict(trigger)


def run_v11_gpu0_with_early_trigger(
    engine: Any,
    *,
    prefix: tuple[int, ...],
    params: Any,
    config: Any,
    start_ns: int,
    barriers: Path,
    write_new: Callable[[Path, Any], None],
    runtime_queue_state: Callable[[], dict[str, int]],
) -> dict[str, Any]:
    """Run frozen v10 traffic/recovery with only the v11 trigger overlaid."""

    if config.overload_queue_trigger != 20 or config.overload_queue_abort != 64:
        raise RuntimeError("integrated v11 trigger requires normal=20 and emergency=64")
    effective = replace(
        config,
        overload_probe_seconds=NORMAL_RECLAIM_CONFIRMATION_SECONDS,
    )
    with _scoped_trigger_overlay(write_new) as context:
        result = _v10.run_v10_gpu0(
            engine,
            prefix=prefix,
            params=params,
            config=effective,
            start_ns=start_ns,
            barriers=barriers,
            write_new=context.publish,
            runtime_queue_state=runtime_queue_state,
        )
    trigger = _validate_emitted_trigger(result.get("reclamation_trigger_evidence"))
    result["reclamation_trigger_evidence"] = trigger
    result["integrated_v11_trigger_overlay"] = {
        "schema_version": "sloforge.branchfabric.integrated-v11-trigger-overlay/v1",
        "normal_reclaim_trigger": 20,
        "emergency_abort_ceiling": 64,
        "emergency_abort_comparison": "outstanding_requests>=64",
        "confirmation_window_seconds": NORMAL_RECLAIM_CONFIRMATION_SECONDS,
        "frozen_v10_default_trigger_semantics_modified": False,
        "passed": True,
    }
    return result
