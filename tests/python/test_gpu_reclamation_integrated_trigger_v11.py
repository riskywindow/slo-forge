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
