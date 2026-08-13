from __future__ import annotations

import importlib.util
import inspect
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
CONTROLLER_PATH = ROOT / "experiments/branchfabric/gpu_reclamation_controller.py"
SPEC = importlib.util.spec_from_file_location(
    "gpu_reclamation_controller_v10_sanity_path", CONTROLLER_PATH
)
assert SPEC is not None and SPEC.loader is not None
CONTROLLER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = CONTROLLER
SPEC.loader.exec_module(CONTROLLER)


def _result(**overrides: object) -> SimpleNamespace:
    values: dict[str, object] = {
        "topology": SimpleNamespace(value="gpu0-only"),
        "configured_rate_rps": 12.0,
        "configured_arrival_rate_verified": True,
        "admission_cadence_verified": True,
        "routing_verified": True,
        "monotonic_timestamps_verified": True,
        "output_target_verified": True,
        "complete_request_accounting": True,
        "queue_flow_conservation_pass": True,
        "p95_ttft_seconds": 0.050,
        "completed_rate_rps": 11.9,
        "observed_offered_rate_rps": 12.0,
        "queue_persistent_positive_drift": False,
        "queue_slope_requests_per_second": 0.04,
        "maximum_queue_depth": 12,
        "queue_depth_start": 11,
        "queue_depth_end": 12,
        "completed_rate_span_seconds": 3.0,
        "verdict": SimpleNamespace(value="inconclusive"),
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_integrated_controller_has_exact_nonadaptive_sanity_protocol() -> None:
    signature = inspect.signature(CONTROLLER.run_integrated_two_gpu_controller)
    assert "external_review" not in signature.parameters

    source = inspect.getsource(CONTROLLER.run_integrated_two_gpu_controller)
    assert "recommend_next_probe" not in source
    assert "external_review" not in source
    assert "for probe_index, rate in enumerate((12.0, 15.0)):" in source
    assert "topology=ProbeTopology.GPU0_ONLY" in source
    assert 'adaptive_capacity_search_performed": False' in source
    assert 'two_gpu_capacity_probe_performed": False' in source


def test_integrated_readiness_clock_uses_worker_model_load_start() -> None:
    source = inspect.getsource(CONTROLLER.run_integrated_two_gpu_controller)

    peer_probe_end = source.index("peer_access = _run_peer_probe(")
    first_worker_launch = source.index(
        'for role, gpu in zip(("serving", "rollout"), inventory_before, strict=True):'
    )
    engine_start_wait = source.index('phase="retained-engine startup signal"')
    readiness_clock_start = source.index("readiness_window_started_ns = min(")
    readiness_wait = source.index('phase="integrated strict readiness"')

    assert (
        peer_probe_end
        < first_worker_launch
        < engine_start_wait
        < readiness_clock_start
        < readiness_wait
    )
    assert "readiness_window_started_ns + round(readiness_timeout_seconds * 1e9)" in source
    assert not any(
        line.strip().startswith("started_ns + round(readiness_timeout_seconds * 1e9)")
        for line in source.splitlines()
    )
    assert '"origin": "first-worker-emitted-model-load-start"' in source


def test_12_rps_guard_is_short_strict_and_backlog_bounded() -> None:
    assessment = CONTROLLER._assess_sanity_guard(_result(), expected_rate_rps=12.0)

    assert assessment["guard"] == "sanity_12rps_stable"
    assert assessment["measurement_seconds"] == 3.0
    assert assessment["result_verdict_ignored"] == "inconclusive"
    assert assessment["passed"] is True

    for invalid in (
        {"admission_cadence_verified": False},
        {"completed_rate_span_seconds": 4.0},
        {"completed_rate_rps": 10.0},
        {"queue_persistent_positive_drift": True},
        {"p95_ttft_seconds": 2.0},
        {"maximum_queue_depth": 26},
    ):
        assert (
            CONTROLLER._assess_sanity_guard(_result(**invalid), expected_rate_rps=12.0)["passed"]
            is False
        )


def test_15_rps_guard_accepts_declared_disjunction_but_never_large_backlog() -> None:
    base = {
        "configured_rate_rps": 15.0,
        "observed_offered_rate_rps": 15.0,
        "completed_rate_rps": 14.7,
        "p95_ttft_seconds": 0.080,
        "queue_slope_requests_per_second": 0.30,
        "maximum_queue_depth": 20,
        "queue_depth_start": 15,
        "queue_depth_end": 17,
    }
    assessment = CONTROLLER._assess_sanity_guard(_result(**base), expected_rate_rps=15.0)
    assert assessment["guard"] == "sanity_15rps_overload"
    assert assessment["signals"]["positive_queue_slope"] is True
    assert assessment["signals"]["bounded_backlog"] is True
    assert assessment["passed"] is True

    no_signal = {
        **base,
        "completed_rate_rps": 15.0,
        "queue_slope_requests_per_second": 0.0,
        "p95_ttft_seconds": 0.080,
    }
    assert (
        CONTROLLER._assess_sanity_guard(_result(**no_signal), expected_rate_rps=15.0)["passed"]
        is False
    )
    assert (
        CONTROLLER._assess_sanity_guard(
            _result(**{**base, "maximum_queue_depth": 26}), expected_rate_rps=15.0
        )["passed"]
        is False
    )


def test_sanity_evaluator_rejects_any_unsettled_rate() -> None:
    with pytest.raises(ValueError, match="exactly 12 or 15"):
        CONTROLLER._assess_sanity_guard(_result(), expected_rate_rps=13.0)
