from __future__ import annotations

import ast
import importlib.util
import sys
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
        scheduled = start_ns + index * 111_111_111
        rows.append(
            {
                "request_id": f"control-{index}",
                "phase": "control",
                "scheduled_arrival_ns": scheduled,
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
    assert body.index('critical_section(\n            "EXPORT_CAPTURE"') < body.index(
        "release_source_and_prove_v11("
    )
    assert body.index("release_source_and_prove_v11(") < body.index("run_v10_gpu1(")
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
        "consume_exact_source_allocation_queue_v11",
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
