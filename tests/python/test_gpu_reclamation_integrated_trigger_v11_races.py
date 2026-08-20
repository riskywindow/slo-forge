from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ROOT = ROOT / "experiments/branchfabric"
sys.path.insert(0, str(EXPERIMENT_ROOT))
TRIGGER_PATH = EXPERIMENT_ROOT / "gpu_reclamation_integrated_trigger_v11.py"
SPEC = importlib.util.spec_from_file_location("integrated_trigger_v11_race_test", TRIGGER_PATH)
assert SPEC is not None and SPEC.loader is not None
TRIGGER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TRIGGER)


def _candidate_trigger() -> dict[str, Any]:
    return {
        "overload_confirmed": True,
        "positive_queue_slope": True,
        "offered_rate_exceeds_completed_rate": True,
        "queue_depth_at_trigger": 20,
        "queue_trigger": 20,
        "queue_abort": 64,
        "events": {
            "OVERLOAD_DETECTED": 100,
            "TRIGGER_PREDICATE_SATISFIED": 120,
            # The builder's candidate stamp must be replaced at publication.
            "RECLAIM_TRIGGER_EMITTED": 121,
        },
        "reclaim_trigger_emitted_ns": 121,
        "controller_reaction_latency_ns": 1,
        "triggered_ns": 121,
    }


def test_one_shot_publisher_stamps_actual_emission_under_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    writes: list[tuple[Path, dict[str, Any]]] = []
    context = TRIGGER._TriggerOverlayContext(lambda path, value: writes.append((path, dict(value))))
    context.register_producer(SimpleNamespace(barriers=tmp_path))
    barrier = tmp_path / "v10-reclaim-trigger.json"
    monkeypatch.setattr(TRIGGER.time, "monotonic_ns", lambda: 150)

    candidate = _candidate_trigger()
    context.publish(barrier, candidate)
    context.publish(barrier, candidate)

    assert len(writes) == 1
    emitted = writes[0][1]
    assert emitted["events"]["RECLAIM_TRIGGER_EMITTED"] == 150
    assert emitted["reclaim_trigger_emitted_ns"] == 150
    assert emitted["triggered_ns"] == 150
    assert emitted["controller_reaction_latency_ns"] == 30
    assert TRIGGER._validate_emitted_trigger(emitted) == emitted
    # Publication creates a fresh payload instead of racing with readers of
    # the predicate candidate.
    assert candidate["events"]["RECLAIM_TRIGGER_EMITTED"] == 121


def test_publisher_fails_closed_on_regressed_emission_clock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    context = TRIGGER._TriggerOverlayContext(lambda _path, _value: None)
    context.register_producer(SimpleNamespace(barriers=tmp_path))
    monkeypatch.setattr(TRIGGER.time, "monotonic_ns", lambda: 119)

    with pytest.raises(RuntimeError, match="preceded its predicate"):
        context.publish(tmp_path / "v10-reclaim-trigger.json", _candidate_trigger())
    assert context.published.is_set() is False
