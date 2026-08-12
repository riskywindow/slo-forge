from __future__ import annotations

import importlib.util
import json
import os
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from sloforge.helix.characterization.gpu_reclamation_serving_v10 import (
    build_global_serving_plan,
)
from sloforge.helix.characterization.gpu_reclamation_v10_postprocess import (
    _actual_arrival_evidence,
    _build_serving,
    _validate_raw_recovery_trend,
)

_ROOT = Path(__file__).resolve().parents[2]
_PATH = _ROOT / "experiments/branchfabric/gpu_reclamation_v10_serving.py"
_SPEC = importlib.util.spec_from_file_location("experiment_004_v10_live_protocol", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
_LIVE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _LIVE
_SPEC.loader.exec_module(_LIVE)


def _write_new(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{threading.get_ident()}.tmp")
    with temporary.open("x") as handle:
        json.dump(value, handle)
    os.link(temporary, path)
    temporary.unlink()


class _FakeEngine:
    """One-request-at-a-time engine with ~33 requests/s capacity."""

    def __init__(self) -> None:
        self.requests: list[str] = []

    def add_request(self, request_id: str, _prompt: object, _params: object) -> None:
        self.requests.append(request_id)

    def step(self):  # type: ignore[no-untyped-def]
        time.sleep(0.03)
        if not self.requests:
            return ()
        request_id = self.requests.pop(0)
        return (
            SimpleNamespace(
                request_id=request_id,
                outputs=(SimpleNamespace(token_ids=tuple(range(64))),),
                finished=True,
            ),
        )


def _config():  # type: ignore[no-untyped-def]
    # Direct construction permits a short synthetic stability interval. The
    # production from_mapping gate separately requires at least five seconds.
    return _LIVE.LiveV10Config(
        attempt_id="exp004-v10-protocol",
        seed=41,
        control_rate_per_second=20.0,
        spike_rate_per_second=40.0,
        restore_rate_per_second=20.0,
        baseline_seconds=0.12,
        overload_probe_seconds=0.12,
        recovery_stability_seconds=0.2,
        recovery_evaluation_seconds=0.1,
        # Full cumulative JSON snapshots deliberately stress the cross-worker
        # telemetry path in this CPU fixture. Keep the threshold above that
        # bounded serialization backlog; the fixture still begins with a
        # strictly larger overload backlog and requires a negative drain.
        recovery_queue_threshold=20,
        output_tokens=64,
        maximum_wall_seconds=5.0,
        restore_grace_seconds=0.12,
        maximum_pending_requests=100,
        restore_handoff_lead_requests=4,
    )


def test_live_config_requires_explicit_v10_and_production_stability() -> None:
    payload = {
        "serving_methodology": "v10-global-capacity",
        "attempt_id": "exp004-v10-protocol",
        "seed": 41,
        "gpu0_control_request_rate_per_second": 10.0,
        "serving_spike_request_rate_per_second": 15.0,
        "gpu0_restore_request_rate_per_second": 8.0,
        "baseline_seconds": 2.0,
        "gpu0_overload_probe_seconds": 2.0,
        "serving_slo_stability_window_seconds": 5.0,
        "serving_recovery_evaluation_seconds": 1.0,
        "serving_recovery_queue_threshold": 4,
        "serving_output_tokens": 64,
        "maximum_wall_seconds": 90,
        "temporary_serving_seconds": 2.0,
        "serving_maximum_pending_requests": 1024,
        "serving_restore_handoff_lead_requests": 4,
    }
    config = _LIVE.LiveV10Config.from_mapping(payload)
    assert config.spike_rate_per_second == 15.0
    payload["serving_slo_stability_window_seconds"] = 4.0
    try:
        _LIVE.LiveV10Config.from_mapping(payload)
    except ValueError as error:
        assert "at least five seconds" in str(error)
    else:
        raise AssertionError("short production stability window was accepted")


def test_sequence_clock_and_route_are_reconstructible_by_both_workers() -> None:
    start = 10_000_000_000
    assert (
        _LIVE.spike_arrival_ns(spike_start_ns=start, rate_per_second=15.0, sequence=3)
        == start + 200_000_000
    )
    assert [
        _LIVE.recovery_route(sequence=sequence, enable_cutover_sequence=10)
        for sequence in range(8, 15)
    ] == ["gpu0", "gpu0", "gpu0", "gpu1", "gpu0", "gpu1", "gpu0"]


def test_restore_candidate_rejects_endpoint_only_queue_decrease(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config()
    first_useful = 1_000_000_000
    now = first_useful + int(config.recovery_stability_seconds * 1e9)
    producer = SimpleNamespace(offered=())
    monkeypatch.setattr(
        _LIVE,
        "_outstanding_at",
        lambda _offered, _rows, timestamp: 5 if timestamp == first_useful else 4,
    )
    monkeypatch.setattr(
        _LIVE,
        "_sustained_queue_trend",
        lambda **_kwargs: {
            "initial_depth": 5,
            "final_depth": 4,
            "sustained_negative": False,
            "completed_rate_exceeds_offered": True,
            "passed": False,
        },
    )
    evidence, candidate = _LIVE._recovery_evidence(
        config=config,
        producer=producer,
        rows=(),
        gpu1_first_useful_ns=first_useful,
        now_ns=now,
        candidate_start_ns=None,
    )
    assert evidence == {}
    assert candidate is None


def test_raw_final_gate_rejects_endpoint_decrease_without_sustained_trend() -> None:
    start = 1_000_000_000
    interval = 100_000_000
    end = start + 2 * interval
    # 5 -> 7 -> 4 has a lower endpoint but its second half is not lower than
    # its first half; it is not a sustained negative trend.
    recovery = {
        "queue_trend": {
            "window_start_ns": start,
            "window_end_ns": end,
            "sample_interval_ns": interval,
            "samples": [
                {"timestamp_ns": start, "queue_depth": 5},
                {"timestamp_ns": start + interval, "queue_depth": 7},
                {"timestamp_ns": end, "queue_depth": 4},
            ],
            "initial_depth": 5,
            "final_depth": 4,
            "first_half_mean_depth": 5.0,
            "second_half_mean_depth": 5.5,
            "slope_requests_per_second": -5.0,
            "offered_requests": 1,
            "completed_requests": 2,
            "offered_rate_per_second": 5.0,
            "completed_rate_per_second": 10.0,
            "sustained_negative": False,
            "completed_rate_exceeds_offered": True,
            "passed": False,
        },
        "queue_drain_pass": True,
        "two_gpu_excess_capacity_pass": True,
    }
    with pytest.raises(ValueError, match="flags differ"):
        _validate_raw_recovery_trend(
            recovery,
            config={
                "serving_slo_stability_window_seconds": 0.2,
                "serving_recovery_evaluation_seconds": 0.1,
            },
        )


def test_synthetic_live_protocol_overloads_recovers_drains_and_restores(tmp_path: Path) -> None:
    config = _config()
    gpu0_result: dict[str, object] = {}
    gpu1_result: dict[str, object] = {}
    errors: list[BaseException] = []
    common_start = time.monotonic_ns() + 50_000_000

    def gpu0() -> None:
        try:
            gpu0_result.update(
                _LIVE.run_v10_gpu0(
                    _FakeEngine(),
                    prefix=(1,),
                    params=object(),
                    config=config,
                    start_ns=common_start,
                    barriers=tmp_path,
                    write_new=_write_new,
                )
            )
        except BaseException as error:
            errors.append(error)

    def gpu1() -> None:
        try:
            while not (tmp_path / "v10-reclaim-trigger.json").is_file():
                time.sleep(0.005)
            gpu1_result.update(
                _LIVE.run_v10_gpu1(
                    _FakeEngine(),
                    prefix=(1,),
                    params=object(),
                    config=config,
                    barriers=tmp_path,
                    write_new=_write_new,
                )
            )
            _write_new(
                tmp_path / "rollout-restore-complete.json",
                {"observed_at_monotonic_ns": time.monotonic_ns()},
            )
        except BaseException as error:
            errors.append(error)

    threads = (threading.Thread(target=gpu0), threading.Thread(target=gpu1))
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=7.0)
    assert not any(thread.is_alive() for thread in threads)
    assert not errors

    trigger = gpu0_result["reclamation_trigger_evidence"]
    recovery = gpu0_result["serving_recovery_evidence"]
    assert isinstance(trigger, dict) and trigger["overload_confirmed"]
    assert isinstance(recovery, dict) and recovery["restore_eligible"]
    assert recovery["completed_rate_per_second"] > recovery["offered_rate_per_second"]
    assert recovery["queue_depth_slope_per_second"] < 0
    assert all(window["passed"] for window in recovery["stability_windows"])

    offered = gpu0_result["global_offered_requests"]
    assert isinstance(offered, tuple)
    enable = gpu0_result["serving_enable_ack"]
    restore = gpu0_result["restore_route_cutover"]
    assert isinstance(enable, dict) and isinstance(restore, dict)
    reconstructed = build_global_serving_plan(
        attempt_id=config.attempt_id,
        seed=config.seed,
        control_start_ns=common_start,
        spike_start_ns=common_start + int(config.baseline_seconds * 1e9),
        gpu1_route_start_ns=int(enable["enable_cutover_scheduled_ns"]),
        restore_start_ns=int(restore["restore_start_ns"]),
        end_ns=int(gpu0_result["end_ns"]),
        control_rate_per_second=config.control_rate_per_second,
        spike_rate_per_second=config.spike_rate_per_second,
        restore_rate_per_second=config.restore_rate_per_second,
    )
    # The post-run strict plan has the same routing semantics even though its
    # diagnostic IDs are normalized independently from live request IDs.
    assert all(
        request.planned_device == "gpu0"
        for request in reconstructed.requests
        if request.phase in {"control", "gpu0-overload", "restore-interference"}
    )
    assert {row["device"] for row in gpu1_result["requests"]} == {"gpu1"}
    assert all(len(row["output_token_ids"]) == 64 for row in gpu0_result["requests"])
    assert all(len(row["output_token_ids"]) == 64 for row in gpu1_result["requests"])
    assert int(gpu1_result["last_admitted_sequence"]) < int(restore["restore_cutover_sequence"])
    workload, observations = _build_serving(
        config={
            "attempt_id": config.attempt_id,
            "seed": config.seed,
            "serving_prompt_tokens": 256,
            "serving_output_tokens": 64,
        },
        serving=gpu0_result,
        rollout={"temporary_serving": gpu1_result},
    )
    assert tuple(phase.name for phase in workload.config.phases) == (
        "control",
        "gpu0-overload",
        "two-gpu-recovery",
        "restore-interference",
    )
    assert len(workload.requests) == len(observations)
    arrival_evidence = _actual_arrival_evidence(
        config={
            "gpu0_control_request_rate_per_second": config.control_rate_per_second,
            "serving_spike_request_rate_per_second": config.spike_rate_per_second,
            "gpu0_restore_request_rate_per_second": config.restore_rate_per_second,
        },
        serving=gpu0_result,
    )
    assert arrival_evidence["passed"]
    corrupted = dict(gpu0_result)
    corrupted_rows = [dict(row) for row in offered]
    corrupted_rows[0]["offered_ns"] = int(corrupted_rows[0]["scheduled_arrival_ns"]) + 200_000_000
    corrupted["global_offered_requests"] = corrupted_rows
    with pytest.raises(ValueError, match="arrival"):
        _actual_arrival_evidence(
            config={
                "gpu0_control_request_rate_per_second": config.control_rate_per_second,
                "serving_spike_request_rate_per_second": config.spike_rate_per_second,
                "gpu0_restore_request_rate_per_second": config.restore_rate_per_second,
            },
            serving=corrupted,
        )
