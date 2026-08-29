from __future__ import annotations

import importlib.util
import json
import os
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ROOT = ROOT / "experiments/branchfabric"
sys.path.insert(0, str(EXPERIMENT_ROOT))
PATH = EXPERIMENT_ROOT / "gpu_reclamation_integrated_trigger_v11.py"
SPEC = importlib.util.spec_from_file_location("integrated_trigger_v11_test", PATH)
assert SPEC is not None and SPEC.loader is not None
TRIGGER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TRIGGER)
V10 = TRIGGER._v10


class _Rows:
    def __init__(self, rows: tuple[dict[str, Any], ...]) -> None:
        self.rows = rows

    def complete_rows(self) -> tuple[dict[str, Any], ...]:
        return self.rows


def _offered(sequence: int, scheduled_ns: int) -> Any:
    return V10.OfferedRequest(
        sequence=sequence,
        request_id=f"request-{sequence:03d}",
        phase="gpu0-overload",
        scheduled_arrival_ns=scheduled_ns,
        device="gpu0",
        offered_ns=scheduled_ns,
    )


def _complete(request: Any, completed_ns: int) -> dict[str, Any]:
    return {
        "request_id": request.request_id,
        "scheduled_arrival_ns": request.scheduled_arrival_ns,
        "first_token_ns": completed_ns - 1,
        "completed_ns": completed_ns,
    }


def _producer(offered: tuple[Any, ...]) -> Any:
    return SimpleNamespace(
        offered=offered,
        config=SimpleNamespace(overload_queue_trigger=20, overload_queue_abort=64),
    )


def test_early_trigger_fires_at_expected_backlog_with_all_three_signals() -> None:
    start = 1_000_000_000
    end = start + 250_000_000
    old = tuple(_offered(sequence, start - 1) for sequence in range(20))
    new = tuple(
        _offered(20 + offset, start + 10_000_000 + offset * 50_000_000) for offset in range(4)
    )
    rows = tuple(_complete(request, end - 1) for request in new[:2])
    evidence = TRIGGER.build_early_trigger_evidence_v11(
        producer=_producer(old + new),
        driver=_Rows(rows),
        window_start_ns=start,
        window_end_ns=end,
    )
    assert evidence["queue_depth_start"] == 20
    assert evidence["queue_depth_at_trigger"] == 22
    assert evidence["positive_queue_slope"] is True
    assert evidence["offered_rate_exceeds_completed_rate"] is True
    assert evidence["overload_confirmed"] is True
    assert evidence["emergency_ceiling_headroom_requests"] == 42


def test_early_trigger_does_not_fire_under_stable_load() -> None:
    start = 1_000_000_000
    end = start + 250_000_000
    old = tuple(_offered(sequence, start - 1) for sequence in range(20))
    new = tuple(_offered(20 + offset, start + offset + 1) for offset in range(2))
    rows = tuple(_complete(request, end - 1) for request in old[:2])
    evidence = TRIGGER.build_early_trigger_evidence_v11(
        producer=_producer(old + new),
        driver=_Rows(rows),
        window_start_ns=start,
        window_end_ns=end,
    )
    assert evidence["queue_depth_start"] == evidence["queue_depth_end"] == 20
    assert evidence["positive_queue_slope"] is False
    assert evidence["offered_rate_exceeds_completed_rate"] is False
    assert evidence["overload_confirmed"] is False


def test_offered_completed_comparison_is_an_independent_required_signal() -> None:
    start = 1_000_000_000
    end = start + 250_000_000
    old = tuple(_offered(sequence, start - 1) for sequence in range(20))
    new = (_offered(20, start + 1),)
    unrelated = _offered(999, start - 1)
    evidence = TRIGGER.build_early_trigger_evidence_v11(
        producer=_producer(old + new),
        driver=_Rows((_complete(unrelated, end - 1),)),
        window_start_ns=start,
        window_end_ns=end,
    )
    assert evidence["queue_depth_slope_per_second"] > 0
    assert evidence["offered_rate_per_second"] == evidence["completed_rate_per_second"]
    assert evidence["offered_rate_exceeds_completed_rate"] is False
    assert evidence["overload_confirmed"] is False


def test_trigger_eligibility_does_not_expire_after_backlog_25() -> None:
    start = 1_000_000_000
    end = start + 250_000_000
    old = tuple(_offered(sequence, start - 1) for sequence in range(20))
    new = tuple(_offered(20 + sequence, start + sequence + 1) for sequence in range(10))
    evidence = TRIGGER.build_early_trigger_evidence_v11(
        producer=_producer(old + new),
        driver=_Rows(()),
        window_start_ns=start,
        window_end_ns=end,
    )
    assert evidence["queue_depth_at_trigger"] == 30
    assert evidence["overload_confirmed"] is True


def test_emergency_abort_is_independent_and_applies_at_64() -> None:
    TRIGGER.enforce_emergency_queue_abort_v11(total_outstanding=63, maximum_depth=64)
    with pytest.raises(RuntimeError, match="emergency ceiling"):
        TRIGGER.enforce_emergency_queue_abort_v11(
            total_outstanding=64, maximum_depth=64, phase="pre-reclaim"
        )


def _live_config() -> Any:
    return V10.LiveV10Config(
        attempt_id="exp004-v11-trigger-race",
        seed=41,
        control_rate_per_second=100.0,
        spike_rate_per_second=100.0,
        restore_rate_per_second=20.0,
        baseline_seconds=0.01,
        overload_probe_seconds=0.25,
        recovery_stability_seconds=0.2,
        recovery_evaluation_seconds=0.1,
        recovery_queue_threshold=4,
        output_tokens=64,
        maximum_wall_seconds=2.0,
        restore_grace_seconds=0.1,
        producer_queue_capacity=256,
        maximum_pending_requests=64,
        restore_handoff_lead_requests=4,
        overload_queue_trigger=20,
        overload_queue_abort=64,
    )


class _ObservationRows:
    def __init__(self, rows: tuple[dict[str, Any], ...]) -> None:
        self.rows = rows

    def observation_rows(self) -> tuple[dict[str, Any], ...]:
        return self.rows


def _queue_sync_producer(
    *, barriers: Path, offered: tuple[Any, ...], evaluation_seconds: float = 1.0
) -> Any:
    return SimpleNamespace(
        barriers=barriers,
        offered=offered,
        config=SimpleNamespace(
            recovery_evaluation_seconds=evaluation_seconds,
            maximum_wall_seconds=588.0,
        ),
    )


def _write_gpu1_telemetry(
    barriers: Path,
    *,
    sequence: int,
    observed_ns: int,
    completed: list[dict[str, Any]],
    active: list[dict[str, Any]],
    cumulative_completed: int,
) -> None:
    active_ids = sorted(str(row["request_id"]) for row in active)
    payload = {
        "schema_version": "sloforge.branchfabric.v10-gpu1-telemetry-incremental/v1",
        "sequence": sequence,
        "observed_ns": observed_ns,
        "completed_requests": completed,
        "active_requests": active,
        "runtime_queue_state": {
            "request_count": len(active),
            "running_requests": len(active),
            "waiting_requests": 0,
            "skipped_waiting_requests": 0,
            "queue_depth": 0,
        },
        "cumulative_completed_requests": cumulative_completed,
        "provenance": {
            "completed_batch_sha256": V10._canonical_sha256(completed),
            "active_request_ids_sha256": V10._canonical_sha256(active_ids),
        },
    }
    path = barriers / "v10-gpu1-telemetry" / f"{sequence:06d}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def _enable_gpu1_queue_sync(barriers: Path) -> None:
    (barriers / "v10-gpu1-first-useful.json").write_text(
        json.dumps(
            {
                "schema_version": "sloforge.branchfabric.v10-gpu1-first-useful/v1",
                "request_id": "gpu1-first",
                "sequence": 103,
                "first_token_ns": 197_940_803_432,
            }
        )
    )


def test_attempt_f_stale_64_synchronizes_to_52_and_passes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    watermark_ns = 199_864_068_274
    failure_ns = 200_395_090_782
    offered = tuple(
        V10.OfferedRequest(
            sequence=sequence,
            request_id=f"request-{sequence:03d}",
            phase="two-gpu-recovery",
            scheduled_arrival_ns=watermark_ns - 1 if sequence < 179 else failure_ns - 1,
            device="gpu0" if sequence < 164 else "gpu1",
            offered_ns=failure_ns - 1,
        )
        for sequence in range(187)
    )
    gpu0_rows = tuple(
        {
            "request_id": f"request-{sequence:03d}",
            "completed_ns": watermark_ns - 1 if sequence < 119 else failure_ns - 1,
        }
        for sequence in range(123)
    )
    first_gpu1_ids = tuple(f"request-{sequence:03d}" for sequence in range(164, 172))
    next_gpu1_ids = tuple(f"request-{sequence:03d}" for sequence in range(172, 179))
    _write_gpu1_telemetry(
        tmp_path,
        sequence=0,
        observed_ns=197_804_728_104,
        completed=[],
        active=[],
        cumulative_completed=0,
    )
    _write_gpu1_telemetry(
        tmp_path,
        sequence=1,
        observed_ns=198_840_857_403,
        completed=[],
        active=[{"request_id": request_id} for request_id in first_gpu1_ids],
        cumulative_completed=0,
    )
    _write_gpu1_telemetry(
        tmp_path,
        sequence=2,
        observed_ns=watermark_ns,
        completed=[
            {"request_id": request_id, "completed_ns": watermark_ns - 1}
            for request_id in first_gpu1_ids
        ],
        active=[{"request_id": request_id} for request_id in next_gpu1_ids],
        cumulative_completed=8,
    )
    _enable_gpu1_queue_sync(tmp_path)
    context = TRIGGER._TriggerOverlayContext(lambda _path, _value: None)
    context.register_producer(_queue_sync_producer(barriers=tmp_path, offered=offered))
    context.register_driver(_ObservationRows(gpu0_rows))
    monkeypatch.setattr(TRIGGER.time, "monotonic_ns", lambda: failure_ns)

    context.enforce_recovery_queue_bound(total_outstanding=64, maximum_depth=64)

    assert context.last_synchronized_queue_evidence == {
        "schema_version": "sloforge.branchfabric.v11-synchronized-recovery-queue/v1",
        "observed_ns": failure_ns,
        "gpu1_telemetry_watermark_ns": watermark_ns,
        "gpu1_telemetry_next_sequence": 3,
        "supplied_unsynchronized_upper_bound": 64,
        "synchronized_total_outstanding": 52,
        "maximum_depth": 64,
        "phase": "two-gpu-recovery",
    }


def test_coherent_post_gpu1_depth_64_still_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    now_ns = 2_000_000_000
    offered = tuple(_offered(sequence, now_ns - 1) for sequence in range(64))
    _write_gpu1_telemetry(
        tmp_path,
        sequence=0,
        observed_ns=now_ns,
        completed=[],
        active=[],
        cumulative_completed=0,
    )
    _enable_gpu1_queue_sync(tmp_path)
    context = TRIGGER._TriggerOverlayContext(lambda _path, _value: None)
    context.register_producer(_queue_sync_producer(barriers=tmp_path, offered=offered))
    context.register_driver(_ObservationRows(()))
    monkeypatch.setattr(TRIGGER.time, "monotonic_ns", lambda: now_ns)

    with pytest.raises(RuntimeError, match="emergency ceiling"):
        context.enforce_recovery_queue_bound(total_outstanding=64, maximum_depth=64)


def test_post_gpu1_telemetry_gap_fails_typed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    now_ns = 2_000_000_000
    _write_gpu1_telemetry(
        tmp_path,
        sequence=1,
        observed_ns=now_ns,
        completed=[],
        active=[],
        cumulative_completed=0,
    )
    _enable_gpu1_queue_sync(tmp_path)
    context = TRIGGER._TriggerOverlayContext(lambda _path, _value: None)
    context.register_producer(_queue_sync_producer(barriers=tmp_path, offered=()))
    context.register_driver(_ObservationRows(()))
    monkeypatch.setattr(TRIGGER.time, "monotonic_ns", lambda: now_ns)

    with pytest.raises(TRIGGER.V11Gpu1TelemetryGapError):
        context.enforce_recovery_queue_bound(total_outstanding=0, maximum_depth=64)


def test_post_gpu1_stale_telemetry_fails_typed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    now_ns = 3_000_000_000
    _write_gpu1_telemetry(
        tmp_path,
        sequence=0,
        observed_ns=1_000_000_000,
        completed=[],
        active=[],
        cumulative_completed=0,
    )
    _enable_gpu1_queue_sync(tmp_path)
    context = TRIGGER._TriggerOverlayContext(lambda _path, _value: None)
    context.register_producer(_queue_sync_producer(barriers=tmp_path, offered=()))
    context.register_driver(_ObservationRows(()))
    monkeypatch.setattr(TRIGGER.time, "monotonic_ns", lambda: now_ns)

    with pytest.raises(TRIGGER.V11Gpu1TelemetryStaleError):
        context.enforce_recovery_queue_bound(total_outstanding=0, maximum_depth=64)


def test_pre_gpu1_depth_64_uses_unchanged_hard_guard(tmp_path: Path) -> None:
    context = TRIGGER._TriggerOverlayContext(lambda _path, _value: None)
    context.register_producer(_queue_sync_producer(barriers=tmp_path, offered=()))
    context.register_driver(_ObservationRows(()))

    with pytest.raises(RuntimeError, match="emergency ceiling"):
        context.enforce_recovery_queue_bound(
            total_outstanding=64,
            maximum_depth=64,
            phase="post-reclaim-pre-gpu1",
        )


def test_independent_monitor_emits_once_when_engine_bookkeeping_is_starved(
    tmp_path: Path,
) -> None:
    writes: list[Path] = []
    writes_lock = threading.Lock()

    def write_new(path: Path, value: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{threading.get_ident()}.tmp")
        temporary.write_text(json.dumps(value))
        os.link(temporary, path)
        temporary.unlink()
        with writes_lock:
            writes.append(path)

    config = _live_config()
    start_ns = time.monotonic_ns() + 20_000_000
    with TRIGGER._scoped_trigger_overlay(write_new) as context:
        producer = V10._GlobalGpu0Producer(
            config=config,
            start_ns=start_ns,
            barriers=tmp_path,
            write_new=context.publish,
        )
        # Construct the driver but deliberately perform no engine steps. The
        # trigger monitor must remain live while all engine bookkeeping stalls.
        V10._EngineDriver(engine=object(), prefix=(1,), params=object(), output_tokens=64)
        producer.start()
        deadline = time.monotonic() + 1.0
        barrier = tmp_path / "v10-reclaim-trigger.json"
        while not barrier.is_file() and time.monotonic() < deadline:
            time.sleep(0.005)
        assert barrier.is_file()
        evidence = json.loads(barrier.read_text())
        assert 20 <= evidence["queue_depth_at_trigger"] < 64
        assert evidence["overload_confirmed"] is True
        assert evidence["controller_reaction_latency_ns"] >= 0
        time.sleep(0.05)
        assert writes.count(barrier) == 1
        producer.stop()


def test_independent_emergency_guard_remains_live_after_trigger(tmp_path: Path) -> None:
    def write_new(path: Path, value: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{threading.get_ident()}.tmp")
        temporary.write_text(json.dumps(value))
        os.link(temporary, path)
        temporary.unlink()

    config = _live_config()
    start_ns = time.monotonic_ns() + 20_000_000
    with TRIGGER._scoped_trigger_overlay(write_new) as context:
        producer = V10._GlobalGpu0Producer(
            config=config,
            start_ns=start_ns,
            barriers=tmp_path,
            write_new=context.publish,
        )
        V10._EngineDriver(engine=object(), prefix=(1,), params=object(), output_tokens=64)
        producer.start()
        deadline = time.monotonic() + 1.2
        while context.error is None and time.monotonic() < deadline:
            time.sleep(0.005)
        assert (tmp_path / "v10-reclaim-trigger.json").is_file()
        assert isinstance(context.error, RuntimeError)
        assert "emergency ceiling" in str(context.error)
        with pytest.raises(RuntimeError, match="arrival producer failed"):
            producer.stop()


class _TargetedEngine:
    def __init__(self, identity_path: Path) -> None:
        self.identity_path = identity_path
        self.pending: dict[str, tuple[int, ...]] = {}

    def add_request(self, request_id: str, _prompt: Any, _params: Any) -> None:
        self.pending[request_id] = tuple(range(64))

    def step(self) -> list[Any]:
        if not self.identity_path.is_file():
            return []
        completed = tuple(self.pending.items())
        self.pending.clear()
        return [
            SimpleNamespace(
                request_id=request_id,
                outputs=(SimpleNamespace(token_ids=token_ids),),
                finished=True,
            )
            for request_id, token_ids in completed
        ]


def _atomic_write_new(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{threading.get_ident()}.tmp")
    temporary.write_text(json.dumps(value))
    os.link(temporary, path)
    temporary.unlink()


def _source_identity_barrier(*, attempt_id: str, allocation_count: int = 1_152) -> dict[str, Any]:
    return {
        "schema_version": "sloforge.branchfabric.v11-source-identity-gate/v1",
        "event": "SOURCE_ALLOCATION_IDENTITY_GATE",
        "attempt_id": attempt_id,
        "observed_at_monotonic_ns": time.monotonic_ns(),
        "source_capture_commit_sha256": "a" * 64,
        "expected_allocation_count": allocation_count,
        "observed_allocation_count": allocation_count,
        "exact_logical_mapping": True,
        "exact_block_epoch_identity": True,
        "exact_owner_sets": True,
        "exact_refcounts": True,
        "all_allocations_live": True,
        "device_identity_pass": True,
        "post_commit_allocation_event_count": 0,
        "no_post_commit_mutation": True,
        "optimized_export_started": False,
        "transaction_source_release_started": False,
        "passed": True,
    }


def test_targeted_gpu0_stops_on_real_identity_gate_and_drains_without_gpu1_barriers(
    tmp_path: Path,
) -> None:
    config = _live_config()
    identity_path = tmp_path / "v11-source-identity-pre-export-pass.json"
    engine = _TargetedEngine(identity_path)
    writer_error: list[BaseException] = []

    def publish_identity_after_trigger() -> None:
        try:
            trigger_path = tmp_path / "v10-reclaim-trigger.json"
            deadline = time.monotonic() + 1.5
            while not trigger_path.is_file():
                if time.monotonic() >= deadline:
                    raise TimeoutError("test did not observe real reclaim trigger")
                time.sleep(0.002)
            _atomic_write_new(
                identity_path,
                _source_identity_barrier(attempt_id=config.attempt_id),
            )
        except BaseException as error:
            writer_error.append(error)

    publisher = threading.Thread(target=publish_identity_after_trigger, daemon=False)
    publisher.start()
    result = TRIGGER.run_v11_gpu0_until_source_identity_gate(
        engine,
        prefix=(11, 12, 13),
        params=object(),
        config=config,
        start_ns=time.monotonic_ns() + 20_000_000,
        barriers=tmp_path,
        write_new=_atomic_write_new,
        runtime_queue_state=lambda: {
            "request_count": len(engine.pending),
            "running_requests": len(engine.pending),
            "waiting_requests": 0,
            "skipped_waiting_requests": 0,
            "queue_depth": 0,
        },
    )
    publisher.join(timeout=1.0)

    assert not publisher.is_alive()
    assert writer_error == []
    assert result["passed"] is True
    assert result["source_identity_gate"]["observed_allocation_count"] == 1_152
    assert 20 <= result["trigger_backlog_requests"] < 64
    assert result["maximum_outstanding_requests"] < 64
    assert result["minimum_emergency_ceiling_headroom_requests"] > 0
    assert result["queue_drain"] == {
        "external_gpu0_queue_size": 0,
        "driver_active_requests": 0,
        "runtime_queue_state": {
            "request_count": 0,
            "running_requests": 0,
            "waiting_requests": 0,
            "skipped_waiting_requests": 0,
            "queue_depth": 0,
        },
        "passed": True,
    }
    assert result["gpu1_recovery_barriers_synthesized"] is False
    assert not (tmp_path / "v10-gpu1-first-useful.json").exists()
    assert not (tmp_path / "v10-serving-recovery.json").exists()
    assert not (tmp_path / "v10-gpu1-drained.json").exists()
    assert not (tmp_path / "v10-restore-start.json").exists()


def test_targeted_source_identity_barrier_requires_exact_1152_pass() -> None:
    with pytest.raises(RuntimeError, match="exact contract"):
        TRIGGER._validate_targeted_source_identity_gate(
            _source_identity_barrier(attempt_id="attempt", allocation_count=1_151),
            attempt_id="attempt",
            expected_source_allocation_count=1_152,
        )


def test_targeted_runtime_drain_requires_every_live_counter_zero() -> None:
    with pytest.raises(RuntimeError, match="scheduler-zero"):
        TRIGGER._targeted_runtime_drain_state(
            lambda: {
                "request_count": 1,
                "running_requests": 1,
                "waiting_requests": 0,
                "skipped_waiting_requests": 0,
                "queue_depth": 0,
            }
        )
