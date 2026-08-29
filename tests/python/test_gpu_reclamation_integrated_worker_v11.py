from __future__ import annotations

import ast
import importlib.util
import json
import sys
import types
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from sloforge.helix.characterization.gpu_reclamation_v11_methodology import (
    Experiment004V11IntegratedConfig,
)

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ROOT = ROOT / "experiments/branchfabric"
WORKER_PATH = EXPERIMENT_ROOT / "gpu_reclamation_integrated_worker_v11.py"
sys.path.insert(0, str(EXPERIMENT_ROOT))
SPEC = importlib.util.spec_from_file_location("gpu_reclamation_integrated_worker_v11", WORKER_PATH)
assert SPEC is not None and SPEC.loader is not None
WORKER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(WORKER)


def _sha(character: str = "a") -> str:
    return character * 64


def _config(**changes: Any) -> Experiment004V11IntegratedConfig:
    payload: dict[str, Any] = {
        "attempt_id": "exp004-v11-integrated-s41-v1",
        "seed": 41,
        "offline_gate_manifest": "artifacts/offline.json",
        "offline_gate_manifest_sha256": _sha("1"),
        "micro_validation_artifact": "artifacts/micro.json",
        "micro_validation_sha256": _sha("2"),
        "post_micro_review_manifest": "artifacts/reviews.json",
        "post_micro_review_manifest_sha256": _sha("3"),
        "budget_authorization": "artifacts/budget.json",
        "budget_authorization_sha256": _sha("4"),
        "ledger_sha256_before_reservation": _sha("5"),
    }
    payload.update(changes)
    return Experiment004V11IntegratedConfig.model_validate(payload)


def _guard(name: str, rate: float) -> dict[str, Any]:
    signals = (
        {
            "completion_tracks_offer": True,
            "no_persistent_positive_drift": True,
            "p95_ttft_below_two_seconds": True,
            "bounded_backlog": True,
        }
        if rate == 12.0
        else {
            "positive_queue_slope": True,
            "completed_rate_below_offered": True,
            "ttft_degradation": False,
            "bounded_backlog": True,
        }
    )
    return {
        "schema_version": "sloforge.branchfabric.v10-sanity-guard-assessment/v1",
        "guard": name,
        "expected_rate_rps": rate,
        "measurement_seconds": 3.0,
        "signals": signals,
        "passed": True,
    }


def test_authoritative_config_is_exact_frozen_transaction_and_state() -> None:
    config = _config()
    assert (
        config.control_request_rate_rps,
        config.lambda_1_rps,
        config.lambda_spike_rps,
        config.lambda_2_rps,
        config.restore_request_rate_rps,
    ) == (9.0, 12.0, 15.0, 20.0, 9.0)
    assert (
        config.baseline_seconds,
        config.sanity_guard_measurement_seconds,
        config.overload_probe_seconds,
        config.serving_stability_seconds,
    ) == (5.0, 3.0, 3.0, 5.0)
    assert (
        config.producer_queue_capacity,
        config.maximum_pending_requests,
        config.overload_queue_trigger,
        config.overload_queue_abort,
    ) == (256, 64, 20, 64)
    assert (
        config.prefix_length,
        config.fanout,
        config.suffix_length,
        config.expected_total_blocks,
    ) == (16_384, 8, 256, 1_152)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("control_request_rate_rps", 8.0),
        ("restore_request_rate_rps", 10.0),
        ("baseline_seconds", 6.0),
        ("sanity_guard_measurement_seconds", 10.0),
        ("overload_probe_seconds", 4.0),
        ("serving_stability_seconds", 4.0),
        ("producer_queue_capacity", 64),
        ("maximum_pending_requests", 256),
        ("overload_queue_trigger", 30),
        ("overload_queue_abort", 65),
        ("prefix_length", 8_192),
        ("fanout", 4),
    ),
)
def test_authoritative_config_rejects_methodology_drift(field: str, value: object) -> None:
    with pytest.raises(ValidationError):
        _config(**{field: value})


def test_frozen_v10_config_translation_has_no_adaptive_calibration() -> None:
    config = _config()
    mapping = WORKER.expanded_runtime_config(config)
    live = WORKER.build_live_v10_config(config)
    assert mapping["serving_methodology"] == "v10-global-capacity"
    assert mapping["gpu0_control_request_rate_per_second"] == 9.0
    assert mapping["serving_spike_request_rate_per_second"] == 15.0
    assert mapping["gpu0_restore_request_rate_per_second"] == 9.0
    assert mapping["authorization_artifact_hash"] == config.budget_authorization_sha256
    assert "rate_grid_rps" not in mapping
    assert "maximum_probes" not in mapping
    assert live.producer_queue_capacity == 256
    assert live.maximum_pending_requests == 64


def test_sanity_pair_accepts_only_ordered_exact_short_guards() -> None:
    pair = WORKER.validate_sanity_guard_pair(
        (
            _guard("sanity_12rps_stable", 12.0),
            _guard("sanity_15rps_overload", 15.0),
        )
    )
    assert pair["passed"] is True
    assert pair["broad_capacity_calibration_performed"] is False
    assert len(pair["sha256"]) == 64

    with pytest.raises(RuntimeError, match="sanity_12rps_stable"):
        WORKER.validate_sanity_guard_pair(
            (
                _guard("sanity_15rps_overload", 15.0),
                _guard("sanity_12rps_stable", 12.0),
            )
        )
    failed = _guard("sanity_15rps_overload", 15.0) | {"passed": False}
    with pytest.raises(RuntimeError, match="sanity_15rps_overload"):
        WORKER.validate_sanity_guard_pair((_guard("sanity_12rps_stable", 12.0), failed))

    ttft_only = _guard("sanity_15rps_overload", 15.0)
    ttft_only["signals"] = {
        "positive_queue_slope": False,
        "completed_rate_below_offered": False,
        "ttft_degradation": True,
        "bounded_backlog": True,
    }
    with pytest.raises(RuntimeError, match="queue growth or a service deficit"):
        WORKER.validate_sanity_guard_pair((_guard("sanity_12rps_stable", 12.0), ttft_only))


def test_methodology_identity_hashes_only_the_imported_frozen_module() -> None:
    identity = WORKER.frozen_v10_methodology_identity()
    source = EXPERIMENT_ROOT / "gpu_reclamation_v10_serving.py"
    assert (
        identity["source_sha256"] == __import__("hashlib").sha256(source.read_bytes()).hexdigest()
    )
    assert identity["execution_commit"] == "c5fa2f169e8a343ef7a86ebc814b3a23a7de6b39"
    assert identity["mutation_performed"] is False


def _complete_gpu0_result() -> dict[str, Any]:
    rows = []
    start_ns = 1_000_000_000
    spike_start_ns = start_ns + 5_000_000_000
    for index in range(45):
        scheduled = start_ns + int(index * 1_000_000_000 / 9)
        rows.append(
            {
                "request_id": f"control-{index}",
                "phase": "control",
                "scheduled_arrival_ns": scheduled,
                "service_start_ns": scheduled + 10_000_000,
                "first_token_ns": scheduled + 10_000_000,
                "completed_ns": scheduled + 80_000_000,
                "token_timestamps_ns": [scheduled + 20_000_000] * 64,
                "output_token_ids": list(range(64)),
            }
        )
    for index in range(3):
        scheduled = 6_100_000_000 + index * 100_000_000
        rows.append(
            {
                "request_id": f"restore-{index}",
                "phase": "restore-interference",
                "scheduled_arrival_ns": scheduled,
                "first_token_ns": scheduled + 10_000_000,
                "completed_ns": scheduled + 80_000_000,
                "token_timestamps_ns": [scheduled + 20_000_000] * 64,
                "output_token_ids": list(range(64)),
            }
        )
    return {
        "start_ns": start_ns,
        "spike_start_ns": spike_start_ns,
        "reclamation_trigger_evidence": {
            "overload_confirmed": True,
            "queue_depth_at_trigger": 22,
            "queue_trigger": 20,
            "queue_abort": 64,
            "positive_queue_slope": True,
            "offered_rate_exceeds_completed_rate": True,
        },
        "serving_recovery_evidence": {
            "two_gpu_excess_capacity_pass": True,
            "queue_drain_pass": True,
            "slo_restoration_pass": True,
            "pre_restore_stability_pass": True,
            "restore_eligible": True,
        },
        "restore_start": {"observed_ns": 6_000_000_000},
        "requests": rows,
    }


def _frozen_v10_control_replay() -> dict[str, Any]:
    """Compact exact replay of the frozen valid s41-v7 control cohort."""

    scheduled = (
        217086831552,
        217197942663,
        217309053774,
        217420164885,
        217531275996,
        217642387107,
        217753498218,
        217864609329,
        217975720440,
        218086831552,
        218197942663,
        218309053774,
        218420164885,
        218531275996,
        218642387107,
        218753498218,
        218864609329,
        218975720440,
        219086831552,
        219197942663,
        219309053774,
        219420164885,
        219531275996,
        219642387107,
        219753498218,
        219864609329,
        219975720440,
        220086831552,
        220197942663,
        220309053774,
        220420164885,
        220531275996,
        220642387107,
        220753498218,
        220864609329,
        220975720440,
        221086831552,
        221197942663,
        221309053774,
        221420164885,
        221531275996,
        221642387107,
        221753498218,
        221864609329,
        221975720440,
    )
    first_tokens = (
        217116203705,
        217242316440,
        217346314931,
        217450906678,
        217563044695,
        217677018266,
        217792533045,
        217896628281,
        218005980825,
        218122966805,
        218231992237,
        218352361851,
        218459579261,
        218568983515,
        218674182792,
        218794191303,
        218905924825,
        219013251263,
        219117898413,
        219229793752,
        219340992895,
        219462190110,
        219572184580,
        219680222934,
        219785035978,
        219905900377,
        220016453253,
        220122930073,
        220241102545,
        220347357927,
        220452746819,
        220573028820,
        220678134598,
        220787914011,
        220906041871,
        221012085503,
        221116861303,
        221236373146,
        221339816666,
        221458825512,
        221564713421,
        221683523726,
        221787746450,
        221894164237,
        222016361772,
    )
    completions = (
        218043128862,
        218163451068,
        218270173450,
        218377534694,
        218499235937,
        218618848845,
        218735128598,
        218849944532,
        218955214782,
        219075849499,
        219185485826,
        219311139465,
        219417658656,
        219528795702,
        219636874080,
        219755538728,
        219863561366,
        219973488692,
        220078845714,
        220185971499,
        220292912589,
        220410526305,
        220517685568,
        220623259268,
        220728594815,
        220851780476,
        220957242729,
        221063258655,
        221179835581,
        221286234743,
        221392775810,
        221509444361,
        221615147261,
        221721029023,
        221837426088,
        221945617969,
        222058884660,
        222193647143,
        222301470235,
        222454352638,
        222563240105,
        222720272061,
        222830974006,
        222958538787,
        223101552850,
    )
    rows = [
        {
            "request_id": f"exp004-v10-naive-s41-v7.control.{index:06d}",
            "phase": "control",
            "scheduled_arrival_ns": arrival,
            "service_start_ns": first_token,
            "first_token_ns": first_token,
            "completed_ns": completion,
            "output_token_ids": list(range(64)),
        }
        for index, (arrival, first_token, completion) in enumerate(
            zip(scheduled, first_tokens, completions, strict=True)
        )
    ]
    rows.append(
        {
            "request_id": "exp004-v10-naive-s41-v7.spike.000000",
            "phase": "gpu0-overload",
            "scheduled_arrival_ns": 222086831552,
            "service_start_ns": 222124593265,
            "first_token_ns": 222124593265,
            "completed_ns": 223210256469,
        }
    )
    return {
        "start_ns": 217086831552,
        "spike_start_ns": 222086831552,
        "requests": rows,
    }


def test_control_interval_exact_frozen_v10_replay_excludes_declared_warmup() -> None:
    result = _frozen_v10_control_replay()
    old_full_rows = result["requests"][:45]
    assert sum(row["completed_ns"] <= result["spike_start_ns"] for row in old_full_rows) == 37

    evidence = WORKER._control_interval_evidence(
        result,
        expected_rate_rps=9.0,
        expected_duration_seconds=5.0,
        slo_ttft_seconds=2.0,
    )
    assert evidence["schema_version"] == "sloforge.branchfabric.v11-control-interval/v2"
    assert evidence["full_arrivals"] == evidence["expected_full_arrivals"] == 45
    assert evidence["warmup_seconds"] == 1.0
    assert evidence["duration_seconds"] == 4.0
    assert evidence["arrivals"] == evidence["expected_arrivals"] == 36
    assert evidence["eventual_full_cohort_completions"] == 45
    assert evidence["eventual_cohort_completions"] == 36
    assert evidence["completions_in_interval"] == 36
    assert evidence["offered_rate_per_second"] == 9.0
    assert evidence["completed_rate_per_second"] == 9.0
    assert evidence["p95_ttft_ns"] == pytest.approx(42_308_889.25)
    assert evidence["waiting_queue_maximum_depth"] == 1
    assert evidence["total_outstanding_diagnostic"]["maximum_depth"] == 9
    assert evidence["total_outstanding_diagnostic"]["final_depth"] == 8
    assert evidence["total_outstanding_diagnostic"]["slope_requests_per_second"] == -0.2
    assert evidence["outstanding_queue_non_positive_trend"] is True
    assert evidence["passed"] is True


def test_control_interval_rejects_real_steady_state_service_deficit() -> None:
    within_threshold = _frozen_v10_control_replay()
    for index in (35, 36):
        within_threshold["requests"][index]["completed_ns"] += 2_000_000_000
    within_threshold_evidence = WORKER._control_interval_evidence(
        within_threshold,
        expected_rate_rps=9.0,
        expected_duration_seconds=5.0,
        slo_ttft_seconds=2.0,
    )
    assert within_threshold_evidence["completions_in_interval"] == 34
    assert within_threshold_evidence["minimum_completion_fraction"] == 0.90
    assert within_threshold_evidence["completion_tracks_offer"] is True

    result = _frozen_v10_control_replay()
    for index in (33, 34, 35, 36):
        result["requests"][index]["completed_ns"] += 2_000_000_000
    evidence = WORKER._control_interval_evidence(
        result,
        expected_rate_rps=9.0,
        expected_duration_seconds=5.0,
        slo_ttft_seconds=2.0,
    )
    assert evidence["completions_in_interval"] == 32
    assert evidence["completion_tracks_offer"] is False
    assert evidence["passed"] is False


def test_control_interval_rejects_growing_queues_ttft_and_missing_output() -> None:
    growing_total = _frozen_v10_control_replay()
    for row in growing_total["requests"][9:45]:
        row["completed_ns"] += 3_000_000_000
    total_evidence = WORKER._control_interval_evidence(
        growing_total,
        expected_rate_rps=9.0,
        expected_duration_seconds=5.0,
        slo_ttft_seconds=2.0,
    )
    assert total_evidence["total_outstanding_bounded_below_normal_trigger"] is False
    assert total_evidence["passed"] is False

    growing_wait = _frozen_v10_control_replay()
    for row in growing_wait["requests"][9:45]:
        row["service_start_ns"] = growing_wait["spike_start_ns"] + 1
    wait_evidence = WORKER._control_interval_evidence(
        growing_wait,
        expected_rate_rps=9.0,
        expected_duration_seconds=5.0,
        slo_ttft_seconds=2.0,
    )
    assert wait_evidence["waiting_queue_bounded_below_normal_trigger"] is False
    assert wait_evidence["monotonic_control_timestamps"] is False
    assert wait_evidence["passed"] is False

    slow_ttft = _frozen_v10_control_replay()
    for row in slow_ttft["requests"][9:45]:
        row["first_token_ns"] = row["scheduled_arrival_ns"] + 2_100_000_000
        row["service_start_ns"] = row["first_token_ns"]
        row["completed_ns"] = max(row["completed_ns"], row["first_token_ns"])
    ttft_evidence = WORKER._control_interval_evidence(
        slow_ttft,
        expected_rate_rps=9.0,
        expected_duration_seconds=5.0,
        slo_ttft_seconds=2.0,
    )
    assert ttft_evidence["p95_ttft_below_slo"] is False
    assert ttft_evidence["passed"] is False

    missing_output = _frozen_v10_control_replay()
    missing_output["requests"][9]["output_token_ids"] = list(range(63))
    output_evidence = WORKER._control_interval_evidence(
        missing_output,
        expected_rate_rps=9.0,
        expected_duration_seconds=5.0,
        slo_ttft_seconds=2.0,
    )
    assert output_evidence["full_cohort_output_tokens_exact"] is False
    assert output_evidence["passed"] is False

    malformed_warmup = _frozen_v10_control_replay()
    malformed_warmup["requests"][0]["output_token_ids"] = list(range(63))
    warmup_output_evidence = WORKER._control_interval_evidence(
        malformed_warmup,
        expected_rate_rps=9.0,
        expected_duration_seconds=5.0,
        slo_ttft_seconds=2.0,
    )
    assert warmup_output_evidence["full_cohort_output_tokens_exact"] is False
    assert warmup_output_evidence["passed"] is False

    incomplete_warmup = _frozen_v10_control_replay()
    incomplete_warmup["requests"][0]["completed_ns"] = None
    warmup_completion_evidence = WORKER._control_interval_evidence(
        incomplete_warmup,
        expected_rate_rps=9.0,
        expected_duration_seconds=5.0,
        slo_ttft_seconds=2.0,
    )
    assert warmup_completion_evidence["complete_full_request_accounting"] is False
    assert warmup_completion_evidence["passed"] is False


def test_control_interval_rejects_deficit_despite_cross_phase_completion_contamination() -> None:
    result = _frozen_v10_control_replay()
    for index in (33, 34, 35, 36):
        result["requests"][index]["completed_ns"] += 2_000_000_000
    uncontaminated = WORKER._control_interval_evidence(
        result,
        expected_rate_rps=9.0,
        expected_duration_seconds=5.0,
        slo_ttft_seconds=2.0,
    )
    for index in range(2):
        result["requests"].append(
            {
                "request_id": f"unrelated.{index}",
                "phase": "restore-interference",
                "scheduled_arrival_ns": result["start_ns"] + 2_000_000_000 + index,
                "service_start_ns": result["start_ns"] + 2_100_000_000 + index,
                "first_token_ns": result["start_ns"] + 2_200_000_000 + index,
                "completed_ns": result["start_ns"] + 2_300_000_000 + index,
                "output_token_ids": list(range(64)),
            }
        )
    evidence = WORKER._control_interval_evidence(
        result,
        expected_rate_rps=9.0,
        expected_duration_seconds=5.0,
        slo_ttft_seconds=2.0,
    )
    assert evidence["completions_in_interval"] == 32
    assert evidence["completion_tracks_offer"] is False
    assert (
        evidence["total_outstanding_diagnostic"] == uncontaminated["total_outstanding_diagnostic"]
    )
    assert evidence["waiting_queue_maximum_depth"] == uncontaminated["waiting_queue_maximum_depth"]
    assert evidence["passed"] is False


def test_control_interval_rejects_duplicate_cadence_and_misordered_timestamps() -> None:
    duplicate = _frozen_v10_control_replay()
    duplicate["requests"][10]["request_id"] = duplicate["requests"][9]["request_id"]
    duplicate_evidence = WORKER._control_interval_evidence(
        duplicate,
        expected_rate_rps=9.0,
        expected_duration_seconds=5.0,
        slo_ttft_seconds=2.0,
    )
    assert duplicate_evidence["unique_control_request_ids"] is False
    assert duplicate_evidence["passed"] is False

    cadence = _frozen_v10_control_replay()
    cadence["requests"][10]["scheduled_arrival_ns"] += 1
    cadence_evidence = WORKER._control_interval_evidence(
        cadence,
        expected_rate_rps=9.0,
        expected_duration_seconds=5.0,
        slo_ttft_seconds=2.0,
    )
    assert cadence_evidence["unique_scheduled_arrivals"] is True
    assert cadence_evidence["scheduled_arrival_cadence_exact"] is False
    assert cadence_evidence["passed"] is False

    misordered = _frozen_v10_control_replay()
    row = misordered["requests"][10]
    row["first_token_ns"] = row["service_start_ns"] - 1
    timing_evidence = WORKER._control_interval_evidence(
        misordered,
        expected_rate_rps=9.0,
        expected_duration_seconds=5.0,
        slo_ttft_seconds=2.0,
    )
    assert timing_evidence["monotonic_control_timestamps"] is False
    assert timing_evidence["passed"] is False


def test_gpu0_requires_complete_active_restore_interference() -> None:
    samples = (
        {"observed_ns": 6_050_000_000, "queue_depth": 1},
        {"observed_ns": 6_250_000_000, "queue_depth": 0},
    )
    restore_complete = {"first_resumed_token_ns": 6_300_000_000}
    evidence = WORKER._validate_gpu0_result(
        _complete_gpu0_result(),
        runtime_queue_samples=samples,
        restore_complete=restore_complete,
    )
    assert evidence["active_during_restore_pass"] is True
    assert evidence["arrivals"] == evidence["completions"] == 3
    assert evidence["emitted_tokens"] == 192
    assert evidence["control_interval"]["passed"] is True
    assert evidence["measured_restore_interval"]["passed"] is True
    source = WORKER_PATH.read_text()
    assert "class _PostStepRestoreSampler:" in source
    assert 'barriers / "v10-restore-start.json"' in source
    assert "runtime_queue_state()" in source

    incomplete = _complete_gpu0_result()
    incomplete["requests"][-1]["completed_ns"] = None
    with pytest.raises(RuntimeError, match="actively and completely"):
        WORKER._validate_gpu0_result(
            incomplete,
            runtime_queue_samples=samples,
            restore_complete=restore_complete,
        )


def test_gpu0_persists_raw_and_control_evidence_before_posthoc_rejection(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    serving = _frozen_v10_control_replay()
    serving.update(
        {
            "reclamation_trigger_evidence": {
                "overload_confirmed": True,
                "queue_depth_at_trigger": 21,
                "queue_trigger": 20,
                "queue_abort": 64,
                "positive_queue_slope": True,
                "offered_rate_exceeds_completed_rate": True,
            },
            "serving_recovery_evidence": {
                "two_gpu_excess_capacity_pass": True,
                "queue_drain_pass": True,
                "slo_restoration_pass": True,
                "pre_restore_stability_pass": True,
                "restore_eligible": True,
            },
        }
    )
    for index in (33, 34, 35, 36):
        serving["requests"][index]["completed_ns"] += 2_000_000_000
    monkeypatch.setitem(
        sys.modules,
        "gpu_capacity_calibration_worker",
        types.SimpleNamespace(_runtime_queue_state=lambda _adapter: {"queue_depth": 0}),
    )
    monkeypatch.setitem(
        sys.modules,
        "gpu_reclamation_integrated_trigger_v11",
        types.SimpleNamespace(run_v11_gpu0_with_early_trigger=lambda *_args, **_kwargs: serving),
    )
    monkeypatch.setitem(
        sys.modules,
        "gpu_reclamation_worker",
        types.SimpleNamespace(_sampling_params=lambda **_kwargs: object()),
    )
    (tmp_path / "rollout-restore-complete.json").write_text(
        json.dumps({"first_resumed_token_ns": 10_000_000_000})
    )
    config = types.SimpleNamespace(
        serving_prompt_tokens=1,
        serving_output_tokens=64,
        seed=41,
        control_request_rate_rps=9.0,
        baseline_seconds=5.0,
        serving_slo_ttft_seconds=2.0,
        warmup_seconds=1.0,
        restore_request_rate_rps=9.0,
    )
    monkeypatch.setattr(WORKER, "build_live_v10_config", lambda _config: object())
    with pytest.raises(RuntimeError, match="9-rps control interval was not stable"):
        WORKER.run_integrated_v11_gpu0(
            object(),
            adapter=object(),
            inputs={"prefix_token_ids": [1]},
            config=config,
            start_ns=1,
            barriers=tmp_path,
        )
    raw_path = tmp_path / "v11-gpu0-serving-prevalidation.json"
    control_path = tmp_path / "v11-gpu0-control-interval.json"
    assert raw_path.is_file()
    assert control_path.is_file()
    assert json.loads(raw_path.read_text())["start_ns"] == serving["start_ns"]
    persisted = json.loads(control_path.read_text())
    assert persisted["completions_in_interval"] == 32
    assert persisted["passed"] is False


def test_gpu0_rejects_late_normal_trigger_and_restore_grace_only_activity() -> None:
    result = _complete_gpu0_result()
    result["reclamation_trigger_evidence"]["queue_depth_at_trigger"] = 31
    with pytest.raises(RuntimeError, match=r"backlog 20\.\.30"):
        WORKER._validate_gpu0_result(
            result,
            runtime_queue_samples=({"observed_ns": 6_050_000_000, "queue_depth": 1},),
            restore_complete={"first_resumed_token_ns": 6_300_000_000},
        )

    result = _complete_gpu0_result()
    for row in result["requests"]:
        if row["phase"] == "restore-interference":
            row["completed_ns"] += 1_000_000_000
            row["token_timestamps_ns"] = [
                timestamp + 1_000_000_000 for timestamp in row["token_timestamps_ns"]
            ]
    with pytest.raises(RuntimeError, match="actual GPU1 restore interval"):
        WORKER._validate_gpu0_result(
            result,
            runtime_queue_samples=({"observed_ns": 6_050_000_000, "queue_depth": 1},),
            restore_complete={"first_resumed_token_ns": 6_300_000_000},
        )


def test_non_overlapping_critical_path_partition_is_exact() -> None:
    evidence = WORKER._non_overlapping_partition(
        chain="test",
        boundaries=(("one", 10, 20), ("two", 20, 35)),
    )
    assert evidence["wall_time_ns"] == 25
    assert sum(stage["wall_time_ns"] for stage in evidence["stages"]) == 25
    assert evidence["overlap_double_count_ns"] == 0
    with pytest.raises(RuntimeError, match="not contiguous"):
        WORKER._non_overlapping_partition(
            chain="test",
            boundaries=(("one", 10, 20), ("two", 19, 35)),
        )


def test_gpu1_source_release_precedes_serving_and_restore_is_after_drain() -> None:
    source = WORKER_PATH.read_text()
    tree = ast.parse(source)
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "run_integrated_v11_gpu1"
    )
    body = ast.get_source_segment(source, function)
    assert body is not None
    gate = body.index('critical_section(\n            "EXPORT_CAPTURE"')
    ownership = body.index("capture_source_ownership_v11(", gate)
    retirement = body.index("retire_source_allocation_history_v11(", ownership)
    commit = body.index("create_source_capture_commit_v11(", retirement)
    pre_export_identity = body.index("validate_source_capture_commit_v11(", commit)
    capture = body.index("capture_native_to_transport_v11(", pre_export_identity)
    post_export_identity = body.index("validate_source_capture_commit_v11(", capture)
    release = body.index("release_source_and_prove_v11(", post_export_identity)
    assert (
        gate
        < ownership
        < retirement
        < commit
        < pre_export_identity
        < capture
        < post_export_identity
        < release
    )
    assert ".step(" not in body[commit:release]
    assert release < body.index("run_v10_gpu1(")
    assert body.index("run_v10_gpu1(") < body.index(
        'critical_section(\n            "IMPORT_ADMISSION"'
    )
    assert body.index("_drain_released_serving_allocation_queue(") < body.index(
        "Vllm0230StreamingRestoreStager("
    )
    assert body.index("write_and_validate_native_subset_v11(") < body.index(
        'barriers / "rollout-restore-complete.json"'
    )
    assert body.index("_run_continuation(") < body.index(
        'barriers / "rollout-restore-complete.json"'
    )
    assert 'chain="RECLAIM_TRIGGER_TO_GPU1_SERVING_READY"' in body
    assert 'chain="RESTORE_TRIGGER_TO_FIRST_RESUMED_TOKEN"' in body
    assert '"movement_by_phase"' in body
    assert '"fused_gather_repack_d2h"' in body
    assert '"fused_h2d_direct_scatter_raw_validation_and_admission"' in body


def test_targeted_gpu1_has_no_export_release_serving_or_restore_path() -> None:
    source = WORKER_PATH.read_text()
    tree = ast.parse(source)
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "run_targeted_source_identity_v11_gpu1"
    )
    body = ast.get_source_segment(source, function)
    assert body is not None
    for forbidden in (
        "capture_native_to_transport_v11",
        "release_source_and_prove_v11",
        "run_v10_gpu1",
        "Vllm0230StreamingRestoreStager",
        "write_and_validate_native_subset_v11",
        "_run_continuation",
    ):
        assert forbidden not in body
    assert body.index("create_source_capture_commit_v11(") < body.index(
        "validate_source_capture_commit_v11("
    )
    assert body.index("validate_source_capture_commit_v11(") < body.index(
        'barriers / "v11-source-identity-pre-export-pass.json"'
    )
    assert body.index("validate_reclaim_trigger_timeline(") < body.index(
        'barriers / "v11-source-identity-pre-export-pass.json"'
    )
    assert '"optimized_export_started": False' in body
    assert '"transaction_source_release_started": False' in body
    assert '"cleanup_release_only": True' in body


def test_trigger_timeline_requires_emission_before_quiescence() -> None:
    evidence = WORKER.validate_reclaim_trigger_timeline(
        {
            "events": {
                "OVERLOAD_DETECTED": 10,
                "TRIGGER_PREDICATE_SATISFIED": 20,
                "RECLAIM_TRIGGER_EMITTED": 21,
            }
        },
        rollout_admission_stop_ns=22,
        state_quiescence_ns=30,
    )
    assert evidence["events"]["ROLLOUT_ADMISSION_STOP"] == 22
    assert evidence["trigger_precedes_state_quiescence"] is True

    with pytest.raises(RuntimeError, match="not causal"):
        WORKER.validate_reclaim_trigger_timeline(
            {
                "events": {
                    "OVERLOAD_DETECTED": 10,
                    "TRIGGER_PREDICATE_SATISFIED": 20,
                    "RECLAIM_TRIGGER_EMITTED": 31,
                }
            },
            rollout_admission_stop_ns=22,
            state_quiescence_ns=30,
        )


def test_gpu1_binds_all_gate_evidence_and_exact_oracle() -> None:
    source = WORKER_PATH.read_text()
    required = (
        "capture_source_ownership_v11",
        "retire_source_allocation_history_v11",
        "create_source_capture_commit_v11",
        "validate_source_capture_commit_v11",
        "require_allocator_notification_queue_empty_v11",
        "release_source_and_prove_v11",
        "V11StatePassRecorder",
        "validate_authoritative_pass_records",
        "destination_target_identity_v11",
        "fresh_destination_allocations_pass",
        "destination_mapping_commitment_pass",
        "engine_step_binding_pass",
        "first_token_exact_8_of_8",
        "independent_recompute_first_tokens",
    )
    assert all(marker in source for marker in required)


def test_integrated_typed_source_failure_serializes_teardown_evidence(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from sloforge.continuum.adapters.vllm_reclamation_v11 import (
        V11RuntimeTeardownRequired,
    )

    config = tmp_path / "config.json"
    config.write_text("{}\n")
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    work_root = tmp_path / "work"
    barriers = tmp_path / "barriers"

    def fail(*_args: object, **_kwargs: object) -> dict[str, object]:
        raise V11RuntimeTeardownRequired(
            "typed integrated post-commit allocation mutation",
            affected_block_ids=(11, 7, 11),
            teardown_evidence={
                "post_commit_event_count": 1,
                "post_commit_block_ids": (19,),
                "queue_empty_after_retirement": False,
                "passed": False,
            },
        )

    monkeypatch.setattr(WORKER, "run_worker", fail)
    status = WORKER.main(
        [
            "--role",
            "rollout",
            "--config",
            str(config),
            "--model-snapshot",
            str(snapshot),
            "--physical-gpu-uuid",
            "GPU-00000000-0000-0000-0000-000000000000",
            "--work-root",
            str(work_root),
            "--barrier-root",
            str(barriers),
        ]
    )
    assert status == 1
    failure = json.loads((work_root / "failure.json").read_text())
    abort = json.loads((barriers / "abort.json").read_text())
    assert failure == abort
    assert failure["status"] == "failed"
    assert failure["role"] == "rollout"
    assert failure["error"] == {
        "type": "V11RuntimeTeardownRequired",
        "message": "typed integrated post-commit allocation mutation",
    }
    assert failure["affected_block_ids"] == [7, 11]
    assert failure["teardown_evidence"] == {
        "post_commit_event_count": 1,
        "post_commit_block_ids": [19],
        "queue_empty_after_retirement": False,
        "passed": False,
    }


def test_worker_import_is_cuda_clean_and_v10_files_are_never_write_targets() -> None:
    source = WORKER_PATH.read_text()
    top_level_imports = tuple(
        node for node in ast.parse(source).body if isinstance(node, (ast.Import, ast.ImportFrom))
    )
    assert all(
        "torch" not in ast.unparse(node) and "vllm" not in ast.unparse(node)
        for node in top_level_imports
    )
    assert 'gpu_reclamation_worker.py").write' not in source
    assert 'gpu_reclamation_v10_serving.py").write' not in source


def test_worker_cli_is_role_bounded_and_has_no_modal_surface() -> None:
    source = WORKER_PATH.read_text()
    assert 'choices=("serving", "rollout")' in source
    assert "modal" not in source.lower()
    assert "--role" in source
    assert "--physical-gpu-uuid" in source
    assert "--barrier-root" in source
