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

from sloforge.helix.characterization.gpu_reclamation_serving import (
    ArrivalPhase,
    ObservationOutcome,
    ServingMeasurementPlan,
    ServingObservation,
    ServingRequest,
    ServingSpikeConfig,
    ServingWorkload,
    WeightedTokenDistribution,
    measure_serving_intervals,
)
from sloforge.helix.characterization.gpu_reclamation_serving_v10 import (
    build_global_serving_plan,
)
from sloforge.helix.characterization.gpu_reclamation_v10_postprocess import (
    _actual_arrival_evidence,
    _build_serving,
    _control_assessment_interval,
    _validate_measured_transaction_compilation,
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
        overload_queue_trigger=10,
        overload_queue_abort=64,
    )


def test_control_assessment_excludes_only_predeclared_empty_queue_fill() -> None:
    start_ns = 10_000_000_000
    spike_start_ns = start_ns + 5_000_000_000
    interval = _control_assessment_interval(
        serving={"start_ns": start_ns, "spike_start_ns": spike_start_ns},
        config={"warmup_seconds": 1.0},
    )
    assert interval.start_ns == start_ns + 1_000_000_000
    assert interval.end_ns == spike_start_ns

    # The independently measured request completion latency is approximately
    # 1.05 seconds. From an empty queue, assessing the entire five-second
    # traffic interval therefore undercounts completions even at a stable
    # deterministic 9 rps. The predeclared one-second settling boundary keeps
    # every request in the raw workload while assessing steady-state service.
    requests = tuple(
        ServingRequest(
            sequence=sequence,
            request_id=f"control-latency-{sequence:02d}",
            phase="control",
            arrival_ns=start_ns + (sequence * 1_000_000_000) // 9,
            prompt_tokens=256,
            requested_output_tokens=1,
        )
        for sequence in range(45)
    )
    observations = tuple(
        ServingObservation(
            request_id=request.request_id,
            arrival_ns=request.arrival_ns,
            service_start_ns=request.arrival_ns + 30_000_000,
            token_timestamps_ns=(request.arrival_ns + 40_000_000,),
            completion_ns=request.arrival_ns + 1_050_000_000,
            outcome=ObservationOutcome.COMPLETED,
            device="gpu0",
        )
        for request in requests
    )
    workload = ServingWorkload(
        workload_id="a" * 64,
        start_ns=start_ns,
        end_ns=start_ns + 7_000_000_000,
        config=ServingSpikeConfig(
            seed=41,
            control_phase="control",
            spike_phase="tail",
            phases=(
                ArrivalPhase(
                    name="control",
                    start_offset_ns=0,
                    end_offset_ns=5_000_000_000,
                    interarrival_ns=111_111_111,
                ),
                ArrivalPhase(
                    name="tail",
                    start_offset_ns=5_000_000_000,
                    end_offset_ns=7_000_000_000,
                    interarrival_ns=1_000_000_000,
                ),
            ),
            prompt_tokens=WeightedTokenDistribution(values=(256,), weights=(1,)),
            output_tokens=WeightedTokenDistribution(values=(1,), weights=(1,)),
        ),
        requests=requests,
    )
    full = measure_serving_intervals(
        workload,
        observations,
        ServingMeasurementPlan(
            intervals=(interval.model_copy(update={"start_ns": start_ns, "name": "full-control"}),)
        ),
    ).intervals[0]
    settled = measure_serving_intervals(
        workload,
        observations,
        ServingMeasurementPlan(intervals=(interval,)),
    ).intervals[0]

    assert len(workload.requests) == 45
    assert full.offered_requests == 45
    assert full.request_throughput_per_second == pytest.approx(7.2)
    assert settled.offered_requests == 36
    assert settled.request_throughput_per_second == pytest.approx(9.0)


def test_control_assessment_rejects_posthoc_warmup_changes() -> None:
    with pytest.raises(ValueError, match="predeclared 1s warmup"):
        _control_assessment_interval(
            serving={"start_ns": 0, "spike_start_ns": 5_000_000_000},
            config={"warmup_seconds": 1.5},
        )


def test_postprocessor_rejects_transaction_compilation_or_partial_coverage() -> None:
    def observation(role: str, *, start_ns: int, end_ns: int) -> dict[str, object]:
        return {
            "schema_version": (
                "sloforge.branchfabric.measured-transaction-compilation-observation/v1"
            ),
            "source": "bounded-python-logging-handler",
            "role": role,
            "interval_start_ns": start_ns,
            "interval_end_ns": end_ns,
            "capture_buffer_valid": True,
            "events": [],
            "no_deferred_compilation_event": True,
            "passed": True,
        }

    serving = {
        "role": "serving",
        "start_ns": 100,
        "end_ns": 900,
        "measured_transaction_compilation_observation": observation(
            "serving", start_ns=90, end_ns=910
        ),
    }
    rollout = {
        "role": "rollout",
        "phase_events": [
            {"phase": "RECLAIM_TRIGGER", "monotonic_timestamp_ns": 300},
        ],
        "rollout_continuation_complete_ns": 800,
        "measured_transaction_compilation_observation": observation(
            "rollout", start_ns=90, end_ns=910
        ),
    }
    assert set(_validate_measured_transaction_compilation(serving=serving, rollout=rollout)) == {
        "serving",
        "rollout",
    }

    compiled = json.loads(json.dumps(rollout))
    compiled["measured_transaction_compilation_observation"]["events"] = [
        "jit compilation during inference"
    ]
    compiled["measured_transaction_compilation_observation"]["passed"] = False
    with pytest.raises(ValueError, match="rollout measured transaction compilation gate"):
        _validate_measured_transaction_compilation(serving=serving, rollout=compiled)

    partial = json.loads(json.dumps(serving))
    partial["measured_transaction_compilation_observation"]["interval_start_ns"] = 101
    with pytest.raises(ValueError, match="does not span"):
        _validate_measured_transaction_compilation(serving=partial, rollout=rollout)


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
        "serving_overload_queue_trigger": 10,
        "serving_overload_queue_abort": 64,
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


def test_hard_queue_abort_remains_live_after_trigger_until_gpu1_is_useful() -> None:
    with pytest.raises(RuntimeError, match="post-reclaim-pre-gpu1"):
        _LIVE._enforce_pre_gpu1_queue_abort(
            trigger_written=True,
            gpu1_first_useful=None,
            instantaneous_depth=65,
            maximum_depth=64,
        )

    # The bounded pre-GPU1 guard deliberately stops applying only after the
    # second engine has completed a useful serving request.
    _LIVE._enforce_pre_gpu1_queue_abort(
        trigger_written=True,
        gpu1_first_useful={"request_id": "gpu1-useful"},
        instantaneous_depth=65,
        maximum_depth=64,
    )


def test_restore_drain_uses_live_scheduler_counts_and_rejects_hidden_work() -> None:
    empty = {
        "request_count": 0,
        "running_requests": 0,
        "waiting_requests": 0,
        "skipped_waiting_requests": 0,
        "queue_depth": 0,
    }
    assert _LIVE._validated_runtime_drain_state(empty) == empty

    hidden_waiting = {
        **empty,
        "request_count": 1,
        "waiting_requests": 1,
        "queue_depth": 1,
    }
    with pytest.raises(RuntimeError, match="not fully drained"):
        _LIVE._validated_runtime_drain_state(hidden_waiting)

    inconsistent = {**empty, "waiting_requests": 1}
    with pytest.raises(ValueError, match="queue-depth accounting"):
        _LIVE._validated_runtime_drain_state(inconsistent)


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
