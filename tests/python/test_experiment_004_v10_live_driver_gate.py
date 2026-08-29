from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_PATH = _ROOT / "tools/branchfabric-experiment-004-live-driver-gate.py"
_SPEC = importlib.util.spec_from_file_location("experiment_004_v10_live_driver_gate", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
_GATE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _GATE
_SPEC.loader.exec_module(_GATE)


def test_actual_live_driver_cpu_gate_is_source_bound_and_fail_closed() -> None:
    result = _GATE.build_gate(seed=41)
    assert result["passed"], {
        key: passed for key, passed in result["assertions"].items() if not passed
    }
    assert result["status"] == "passed"
    assert result["seed"] == 41
    assert len(result["provenance"]["live_driver_sha256"]) == 64
    assert len(result["provenance"]["raw_evidence_sha256"]) == 64
    assert all(result["assertions"].values())
    assert set(result["actual_timer_cadence"]) == {
        "control",
        "gpu0-overload",
        "two-gpu-recovery",
        "restore-interference",
    }
    assert all(result["actual_timer_cadence"].values())
    assert result["fixture_config"]["output_tokens"] == 64
    assert result["fixture_config"]["producer_queue_capacity"] == 256
    assert (
        result["fixture_config"]["producer_queue_capacity"]
        > result["fixture_config"]["maximum_pending_requests"]
    )
    assert 10 <= result["fixture_config"]["overload_queue_trigger"] <= 30
    assert result["fixture_config"]["overload_queue_abort"] <= 64

    trigger = dict(result["raw_evidence"]["reclamation_trigger_evidence"])
    assert _GATE._matches_trigger(result["independently_recomputed_trigger"], trigger)
    trigger["queue_depth_end"] += 1
    assert not _GATE._matches_trigger(result["independently_recomputed_trigger"], trigger)
