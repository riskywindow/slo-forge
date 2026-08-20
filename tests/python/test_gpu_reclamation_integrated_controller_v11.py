from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import signal
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from sloforge.helix.characterization.gpu_reclamation_v11_methodology import (
    Experiment004V11IntegratedConfig,
)

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ROOT = ROOT / "experiments/branchfabric"
CONTROLLER_PATH = EXPERIMENT_ROOT / "gpu_reclamation_integrated_controller_v11.py"
sys.path.insert(0, str(EXPERIMENT_ROOT))
SPEC = importlib.util.spec_from_file_location(
    "gpu_reclamation_integrated_controller_v11_test", CONTROLLER_PATH
)
assert SPEC is not None and SPEC.loader is not None
CONTROLLER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CONTROLLER)


def _write_json(path: Path, value: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sealed_fixture(root: Path) -> Experiment004V11IntegratedConfig:
    review_entries: list[dict[str, Any]] = []
    for agent in range(9, 13):
        reference = f"reviews/agent{agent}.json"
        digest = _write_json(root / reference, {"status": "PASS", "agent": agent})
        review_entries.append(
            {"agent": agent, "status": "PASS", "artifact": reference, "sha256": digest}
        )
    prelaunch_bindings: list[dict[str, str]] = []
    for label in (
        "methodology",
        "worker",
        "controller",
        "launcher",
        "trigger",
        "worker_test",
        "controller_test",
        "launcher_test",
        "trigger_test",
        "trigger_race_test",
        "cross_runtime_review",
    ):
        reference = CONTROLLER.INTEGRATED_PRELAUNCH_BINDINGS[label]
        path = root / reference
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(label + "\n")
        prelaunch_bindings.append(
            {
                "label": label,
                "artifact": reference,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    make_check_reference = CONTROLLER.INTEGRATED_PRELAUNCH_BINDINGS["make_check"]
    make_check_sha = _write_json(
        root / make_check_reference,
        {"status": "PASS", "command": "make check"},
    )
    prelaunch_bindings.append(
        {
            "label": "make_check",
            "artifact": make_check_reference,
            "sha256": make_check_sha,
        }
    )
    post_sha = _write_json(
        root / "post.json",
        {
            "status": "PASS",
            "integrated_v11_admission": "APPROVED_EXACTLY_ONCE",
            "attempt_id": "exp004-v11-micro-s41-d",
            "reviews": review_entries,
            "integrated_prelaunch": {"status": "PASS", "bindings": prelaunch_bindings},
        },
    )
    gates = []
    for gate in ("A", "B", "C", "D"):
        reference = f"gates/{gate}.json"
        gate_sha = _write_json(root / reference, {"gate": gate, "status": "PASS"})
        gates.append(
            {
                "gate": gate,
                "status": "PASS_OFFLINE_POST_FIX",
                "artifact": reference,
                "sha256": gate_sha,
            }
        )
    offline_sha = _write_json(
        root / "offline.json",
        {
            "status": "PASS",
            "production_integration_gates": gates,
        },
    )
    micro_sha = _write_json(
        root / "micro.json",
        {
            "status": "MICRO_VALIDATION_PASS",
            "attempt_id": "exp004-v11-micro-s41-d",
            "scientifically_admissible_measurement": True,
            "integrated_v11_allowed": True,
            "correctness": {
                "allocator_epoch_pass": True,
                "post_free_ownership_pass": True,
                "fresh_destination_allocations_pass": True,
                "destination_mapping_commitment_pass": True,
                "integrity_pass": True,
                "movement_accounting_pass": True,
                "all_branches_resumed": True,
                "first_resumed_token_exact_matches": 8,
                "first_resumed_token_total": 8,
            },
            "movement": {"full_physical_amplification": 21.000762818351625},
        },
    )
    budget_sha = _write_json(
        root / "budget.json",
        {
            "schema_version": "sloforge.branchfabric.experiment-004-budget-authorization/v1",
            "authorized_cumulative_gpu_seconds": 21_600.0,
            "authorized_scope": ["future v11 integrated reclamation"],
        },
    )
    return Experiment004V11IntegratedConfig(
        attempt_id="exp004-v11-integrated-s41-a",
        seed=41,
        offline_gate_manifest="offline.json",
        offline_gate_manifest_sha256=offline_sha,
        micro_validation_artifact="micro.json",
        micro_validation_sha256=micro_sha,
        post_micro_review_manifest="post.json",
        post_micro_review_manifest_sha256=post_sha,
        budget_authorization="budget.json",
        budget_authorization_sha256=budget_sha,
        ledger_sha256_before_reservation="5" * 64,
    )


def _continuity() -> dict[str, dict[str, Any]]:
    return {
        "gpu0": {"pid": 101, "engine_nonce": "a" * 64, "physical_gpu_uuid": "GPU-a"},
        "gpu1": {"pid": 202, "engine_nonce": "b" * 64, "physical_gpu_uuid": "GPU-b"},
    }


def _handoffs() -> tuple[dict[str, Any], dict[str, Any]]:
    rows = []
    for device, identity in _continuity().items():
        rows.append(
            {
                "schema_version": "sloforge.branchfabric.v11-transaction-ready/v1",
                "device": device,
                **identity,
                "passed": True,
                "selected_load_sha256": "1" * 64,
                "authorization_artifact_hash": "3" * 64,
                "pre_gpu_evidence_hashes": {
                    "sanity_result_sha256": "2" * 64,
                    "authorization_artifact_hash": "3" * 64,
                },
            }
        )
    return rows[0], rows[1]


def _worker_results() -> tuple[dict[str, Any], dict[str, Any]]:
    logical = 1_056_964_608
    full = 22_197_063_040
    trigger = {
        "schema_version": "sloforge.branchfabric.reclamation-trigger-evidence/v2",
        "overload_confirmed": True,
        "positive_queue_slope": True,
        "offered_rate_exceeds_completed_rate": True,
        "queue_trigger": 20,
        "queue_abort": 64,
        "queue_depth_at_trigger": 22,
        "emergency_ceiling_headroom_requests": 42,
        "controller_reaction_latency_ns": 50,
        "triggered_ns": 1_000_000_000,
        "events": {
            "OVERLOAD_DETECTED": 900_000_000,
            "TRIGGER_PREDICATE_SATISFIED": 999_999_950,
            "RECLAIM_TRIGGER_EMITTED": 1_000_000_000,
        },
    }
    recovery = {
        "schema_version": "sloforge.branchfabric.serving-recovery-evidence/v1",
        "completed_rate_per_second": 16.0,
        "offered_rate_per_second": 15.0,
        "queue_depth_slope_per_second": -1.0,
        "gpu1_first_useful_ns": 3_000_000_000,
        "stability_windows": [
            {
                "start_ns": 4_000_000_000 + index * 1_000_000_000,
                "end_ns": 5_000_000_000 + index * 1_000_000_000,
                "passed": True,
                "p95_ttft_ns": 100_000_000,
                "queue_depth_end": 0,
                "ttft_sample_count": 9,
            }
            for index in range(5)
        ],
    }
    timings = {
        "reclaim_trigger_ns": 1_000_000_000,
        "rollout_admission_stop_ns": 1_100_000_000,
        "export_started_ns": 1_200_000_000,
        "state_quiescence_ns": 1_300_000_000,
        "source_authentication_ended_ns": 1_400_000_000,
        "source_pipeline_ended_ns": 1_500_000_000,
        "state_publish_ended_ns": 1_600_000_000,
        "source_release_ended_ns": 1_700_000_000,
        "export_ended_ns": 1_800_000_000,
        "hbm_reclaim_confirmed_ns": 2_000_000_000,
        "gpu1_serving_ready_ns": 2_100_000_000,
        "restore_trigger_ns": 10_000_000_000,
        "restore_started_ns": 10_100_000_000,
        "checkpoint_authentication_started_ns": 10_200_000_000,
        "checkpoint_authentication_ended_ns": 10_300_000_000,
        "destination_staging_ended_ns": 10_400_000_000,
        "native_restore_pipeline_ended_ns": 10_900_000_000,
        "restore_native_complete_ns": 11_000_000_000,
        "continuation_started_ns": 11_100_000_000,
        "first_resumed_token_ns": 12_000_000_000,
        "all_branches_resumed_ns": 13_000_000_000,
        "continuation_complete_ns": 14_000_000_000,
    }

    def partition(chain: str, stages: list[tuple[str, int, int]]) -> dict[str, Any]:
        rows = [
            {"stage": name, "start_ns": start, "end_ns": end, "wall_time_ns": end - start}
            for name, start, end in stages
        ]
        return {
            "schema_version": "sloforge.branchfabric.v11-non-overlapping-critical-path/v1",
            "chain": chain,
            "start_ns": rows[0]["start_ns"],
            "end_ns": rows[-1]["end_ns"],
            "wall_time_ns": rows[-1]["end_ns"] - rows[0]["start_ns"],
            "stages": rows,
            "overlap_double_count_ns": 0,
            "passed": True,
        }

    critical_paths = {
        "reclamation": partition(
            "RECLAIM_TRIGGER_TO_GPU1_SERVING_READY",
            [
                ("branch_quiesce", 1_000_000_000, 1_300_000_000),
                ("source_authentication", 1_300_000_000, 1_400_000_000),
                ("fused_gather_repack_d2h", 1_400_000_000, 1_500_000_000),
                ("publish_and_release_preconditions", 1_500_000_000, 1_600_000_000),
                ("source_release", 1_600_000_000, 1_700_000_000),
                ("hbm_reclaim_confirmation", 1_700_000_000, 2_000_000_000),
                ("gpu1_serving_ready", 2_000_000_000, 2_100_000_000),
            ],
        ),
        "restore": partition(
            "RESTORE_TRIGGER_TO_FIRST_RESUMED_TOKEN",
            [
                ("restore_preconditions", 10_000_000_000, 10_200_000_000),
                ("checkpoint_authentication", 10_200_000_000, 10_300_000_000),
                ("destination_allocation_staging", 10_300_000_000, 10_400_000_000),
                (
                    "fused_h2d_direct_scatter_raw_validation_and_admission",
                    10_400_000_000,
                    10_900_000_000,
                ),
                ("canonical_pass_validation_and_commit", 10_900_000_000, 11_000_000_000),
                ("first_resumed_token", 11_000_000_000, 12_000_000_000),
            ],
        ),
        "residual_source_chain_ns": 200_000_000,
        "residual_restore_chain_ns": 900_000_000,
        "stage_overlap_policy": "fused streaming stages avoid overlapping wall attribution",
    }
    rows = []
    for role, pid, uuid in (("serving", 101, "GPU-a"), ("rollout", 202, "GPU-b")):
        common = {
            "schema_version": (
                "sloforge.branchfabric.experiment-004-v11-gpu0-result/v1"
                if role == "serving"
                else "sloforge.branchfabric.experiment-004-v11-gpu1-result/v1"
            ),
            "status": "succeeded",
            "role": role,
            "pid": pid,
            "physical_gpu_uuid": uuid,
            "attempt_id": "exp004-v11-integrated-s41-a",
            "measured_transaction_compilation_observation": {
                "schema_version": (
                    "sloforge.branchfabric.measured-transaction-compilation-observation/v1"
                ),
                "source": "bounded-python-logging-handler",
                "role": role,
                "interval_start_ns": 100,
                "interval_end_ns": 200,
                "capture_buffer_valid": True,
                "passed": True,
                "events": [],
                "no_deferred_compilation_event": True,
            },
        }
        if role == "serving":
            common.update(
                {
                    "scientific_gates": {
                        "control_stability_pass": True,
                        "gpu0_overload_pass": True,
                        "bounded_backlog_pass": True,
                        "two_gpu_service_gt_offered_pass": True,
                        "queue_drain_pass": True,
                        "slo_restoration_pass": True,
                        "slo_stability_pass": True,
                        "gpu0_active_during_restore_pass": True,
                    },
                    "gpu0_restore_interference": {
                        "active_during_restore_pass": True,
                        "control_interval": {
                            "schema_version": "sloforge.branchfabric.v11-control-interval/v1",
                            "passed": True,
                            "duration_seconds": 5.0,
                            "expected_arrivals": 45,
                            "arrivals": 45,
                            "offered_rate_per_second": 9.0,
                            "completed_rate_per_second": 9.0,
                            "p95_ttft_ns": 100_000_000.0,
                            "maximum_total_outstanding": 2,
                            "terminal_total_outstanding": 0,
                            "completion_tracks_offer": True,
                            "p95_ttft_below_slo": True,
                            "queue_bounded_below_normal_trigger": True,
                            "terminal_queue_at_recovery_threshold": True,
                        },
                        "measured_restore_interval": {
                            "schema_version": (
                                "sloforge.branchfabric.v11-gpu0-restore-interference/v1"
                            ),
                            "passed": True,
                            "methodology_stall_absent": True,
                            "start_ns": 10_000_000_000,
                            "end_ns": 12_000_000_000,
                            "duration_seconds": 2.0,
                            "arrivals": 18,
                            "completions": 17,
                            "emitted_tokens": 1024,
                            "observed_arrival_rate_per_second": 9.0,
                            "completion_rate_per_second": 8.5,
                            "token_rate_per_second": 512.0,
                            "ttft_sample_count": 18,
                            "p95_ttft_ns": 100_000_000.0,
                            "live_vllm_queue_depth_samples": [
                                {"observed_ns": 11_000_000_000, "queue_depth": 1}
                            ],
                        },
                        "arrivals": 2,
                        "completions": 2,
                        "emitted_tokens": 128,
                        "offered_rps": 9.0,
                        "live_vllm_queue_depth_samples": [{"queue_depth": 1}],
                        "offered_outstanding_samples": [{"outstanding_requests": 1}],
                    },
                    "gpu0_control_interval": {
                        "schema_version": "sloforge.branchfabric.v11-control-interval/v1",
                        "passed": True,
                        "duration_seconds": 5.0,
                        "expected_arrivals": 45,
                        "arrivals": 45,
                        "offered_rate_per_second": 9.0,
                        "completed_rate_per_second": 9.0,
                        "p95_ttft_ns": 100_000_000.0,
                        "maximum_total_outstanding": 2,
                        "terminal_total_outstanding": 0,
                        "completion_tracks_offer": True,
                        "p95_ttft_below_slo": True,
                        "queue_bounded_below_normal_trigger": True,
                        "terminal_queue_at_recovery_threshold": True,
                    },
                    "serving": {
                        "reclamation_trigger_evidence": trigger,
                        "serving_recovery_evidence": recovery,
                        "restore_start": {"observed_ns": 10_000_000_000},
                    },
                }
            )
        else:
            common.update(
                {
                    "topology": {
                        "branch_count": 8,
                        "shared_blocks": 1024,
                        "private_blocks": 128,
                        "total_blocks": 1152,
                        "logical_state_bytes": 1_056_964_608,
                    },
                    "correctness": {
                        "allocator_epoch_pass": True,
                        "ownership_release_pass": True,
                        "engine_step_binding_pass": True,
                        "integrity_pass": True,
                        "destination_mapping_commitment_pass": True,
                        "fresh_destination_allocations_pass": True,
                        "all_branches_resumed": True,
                        "first_token_exact_8_of_8": True,
                        "movement_accounting_pass": True,
                        "gpu0_active_during_restore_protocol_pass": True,
                    },
                    "continuation": {"exact_matches": 8, "minimum_tokens_per_branch": 8},
                    "movement": {
                        "movement_accounting_complete": True,
                        "logical_state_bytes": logical,
                        "full_physical_bytes": full,
                        "external_movement_bytes": 2_114_200_960,
                        "avoidable_physical_bytes": 12_684_381_568,
                        "required_physical_bytes": full - 12_684_381_568,
                        "diagnostic_physical_bytes": 0,
                        "full_physical_amplification": full / logical,
                        "external_movement_amplification": 2_114_200_960 / logical,
                        "avoidable_amplification": 12_684_381_568 / logical,
                        "record_count": 2,
                    },
                    "movement_by_phase": {
                        "source_export": {
                            "movement_accounting_complete": True,
                            "logical_state_bytes": logical,
                            "full_physical_bytes": full // 2,
                            "external_movement_bytes": 1_057_100_480,
                            "avoidable_physical_bytes": 6_342_190_784,
                            "required_physical_bytes": full // 2 - 6_342_190_784,
                            "diagnostic_physical_bytes": 0,
                            "record_count": 1,
                            "full_physical_amplification": (full // 2) / logical,
                            "external_movement_amplification": 1_057_100_480 / logical,
                            "avoidable_amplification": 6_342_190_784 / logical,
                        },
                        "rollout_restore": {
                            "movement_accounting_complete": True,
                            "logical_state_bytes": logical,
                            "full_physical_bytes": full - full // 2,
                            "external_movement_bytes": 1_057_100_480,
                            "avoidable_physical_bytes": 6_342_190_784,
                            "required_physical_bytes": (full - full // 2 - 6_342_190_784),
                            "diagnostic_physical_bytes": 0,
                            "record_count": 1,
                            "full_physical_amplification": (full - full // 2) / logical,
                            "external_movement_amplification": 1_057_100_480 / logical,
                            "avoidable_amplification": 6_342_190_784 / logical,
                        },
                    },
                    "timings": timings,
                    "critical_paths": critical_paths,
                    "trigger_timeline": {
                        "passed": True,
                        "trigger_precedes_state_quiescence": True,
                    },
                    "temporary_memory": {
                        "peak_pinned_host_temporary_bytes": 1,
                        "peak_pageable_host_temporary_bytes": 2,
                        "peak_gpu_temporary_bytes": 3,
                    },
                    "serving": {
                        "requests": [{"request_id": "useful-gpu1", "first_token_ns": 3_000_000_000}]
                    },
                    "state_passes": [
                        {
                            "schema_version": "sloforge.branchfabric.state-pass-record/v11",
                            "pass_id": "capture:measured-pass",
                            "wall_start_ns": 1_400_000_000,
                            "wall_end_ns": 1_500_000_000,
                            "physical_touch_bytes": full // 2,
                        },
                        {
                            "schema_version": "sloforge.branchfabric.state-pass-record/v11",
                            "pass_id": "restore:measured-pass",
                            "wall_start_ns": 10_200_000_000,
                            "wall_end_ns": 10_300_000_000,
                            "physical_touch_bytes": full - full // 2,
                        },
                    ],
                }
            )
        rows.append(common)
    return rows[0], rows[1]


def test_sealed_evidence_accepts_exact_four_gates_micro_and_reviews(tmp_path: Path) -> None:
    result = CONTROLLER.verify_sealed_evidence(_sealed_fixture(tmp_path), tmp_path)
    assert result["status"] == "PASS"
    assert result["review_agents"] == [9, 10, 11, 12]
    assert result["micro_exact_first_token_matches"] == 8
    assert result["micro_full_physical_amplification"] == pytest.approx(21.000762818351625)


def test_sealed_evidence_rejects_hash_tamper(tmp_path: Path) -> None:
    config = _sealed_fixture(tmp_path)
    (tmp_path / "micro.json").write_text("{}\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        CONTROLLER.verify_sealed_evidence(config, tmp_path)


@pytest.mark.parametrize(
    ("artifact", "mutate", "message"),
    (
        ("offline.json", lambda value: value.update(status="FAIL"), "four production gates"),
        (
            "micro.json",
            lambda value: value["movement"].update(full_physical_amplification=25.0001),
            "does not admit",
        ),
        ("post.json", lambda value: value["reviews"].pop(), "reviews 9-12"),
        (
            "post.json",
            lambda value: value["integrated_prelaunch"].update(status="FAIL"),
            "prelaunch PASS",
        ),
        (
            "budget.json",
            lambda value: value.update(authorized_scope=["micro only"]),
            "does not cover",
        ),
    ),
)
def test_sealed_evidence_fails_closed_on_semantic_drift(
    tmp_path: Path,
    artifact: str,
    mutate: Any,
    message: str,
) -> None:
    config = _sealed_fixture(tmp_path)
    path = tmp_path / artifact
    value = json.loads(path.read_text())
    mutate(value)
    digest = _write_json(path, value)
    update = {
        "offline.json": {"offline_gate_manifest_sha256": digest},
        "micro.json": {"micro_validation_sha256": digest},
        "post.json": {"post_micro_review_manifest_sha256": digest},
        "budget.json": {"budget_authorization_sha256": digest},
    }[artifact]
    changed = config.model_copy(update=update)
    with pytest.raises((RuntimeError, ValueError), match=message):
        CONTROLLER.verify_sealed_evidence(changed, tmp_path)


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("missing", "incomplete"),
        ("extra", "incomplete"),
        ("duplicate", "not unique"),
        ("path", "path substitution"),
        ("hash", "hash mismatch"),
    ),
)
def test_integrated_prelaunch_bindings_fail_closed(
    tmp_path: Path, mutation: str, message: str
) -> None:
    config = _sealed_fixture(tmp_path)
    path = tmp_path / "post.json"
    value = json.loads(path.read_text())
    bindings = value["integrated_prelaunch"]["bindings"]
    if mutation == "missing":
        bindings.pop()
    elif mutation == "extra":
        bindings.append({"label": "extra", "artifact": "extra.txt", "sha256": "0" * 64})
    elif mutation == "duplicate":
        bindings.append(dict(bindings[0]))
    elif mutation == "path":
        bindings[0]["artifact"] = bindings[1]["artifact"]
    elif mutation == "hash":
        bindings[0]["sha256"] = "0" * 64
    digest = _write_json(path, value)
    changed = config.model_copy(update={"post_micro_review_manifest_sha256": digest})
    with pytest.raises((RuntimeError, ValueError), match=message):
        CONTROLLER.verify_sealed_evidence(changed, tmp_path)


def test_integrated_prelaunch_make_check_semantics_fail_closed(tmp_path: Path) -> None:
    config = _sealed_fixture(tmp_path)
    post_path = tmp_path / "post.json"
    post = json.loads(post_path.read_text())
    binding = next(
        item for item in post["integrated_prelaunch"]["bindings"] if item["label"] == "make_check"
    )
    make_path = tmp_path / binding["artifact"]
    binding["sha256"] = _write_json(make_path, {"status": "FAIL", "command": "make check"})
    post_sha = _write_json(post_path, post)
    changed = config.model_copy(update={"post_micro_review_manifest_sha256": post_sha})
    with pytest.raises(RuntimeError, match="make check is not PASS"):
        CONTROLLER.verify_sealed_evidence(changed, tmp_path)


def _replacement_fixture(root: Path) -> Experiment004V11IntegratedConfig:
    config = _sealed_fixture(root)
    post_path = root / "post.json"
    post = json.loads(post_path.read_text())
    reviews: list[dict[str, Any]] = []
    for agent, role in CONTROLLER.FINAL_PRE_GPU_REVIEW_ROLES.items():
        reference = f"final-reviews/agent{agent:02d}.json"
        digest = _write_json(
            root / reference,
            {
                "schema_version": "test-review/v1",
                "agent": agent,
                "role": role,
                "status": "PASS",
                "gpu_resources_consumed": False,
            },
        )
        reviews.append(
            {
                "agent": agent,
                "role": role,
                "status": "PASS",
                "artifact": reference,
                "sha256": digest,
            }
        )
    make_check_reference = CONTROLLER.INTEGRATED_PRELAUNCH_BINDINGS["make_check"]
    make_check_sha = hashlib.sha256((root / make_check_reference).read_bytes()).hexdigest()
    post["final_integrated_pre_gpu_gate"] = {
        "status": "PASS",
        "attempt_id": "exp004-v11-integrated-s41-c",
        "reviews": reviews,
        "make_check": {
            "status": "PASS",
            "command": "make check",
            "artifact": make_check_reference,
            "sha256": make_check_sha,
        },
    }
    payloads = {
        "prior_attempt_cleanup": {
            "schema_version": ("sloforge.branchfabric.experiment-004-v11-integrated-postflight/v1"),
            "status": "PASS_CLEANUP_SCIENTIFICALLY_INVALID_BUDGET_CHARGED",
            "attempt_id": "exp004-v11-integrated-s41-b",
            "scientific_status": "INVALID_PRE_RECLAIM_BOUNDED_BACKLOG_ABORT",
            "scientifically_valid_integrated_transaction": False,
            "failure_stage": "PRE_RECLAIM_GPU0_OVERLOAD_BACKLOG_SAFETY_ABORT",
            "reclaim_trigger_reached": False,
            "v11_export_started": False,
            "v11_release_started": False,
            "v11_import_started": False,
            "cleanup_pass": True,
            "provider_cleanup": {
                "pass": True,
                "running_tasks": 0,
                "running_containers": 0,
                "active_apps": 0,
                "endpoints": 0,
                "provider_reservations": 0,
                "owned_child_processes": 0,
                "profilers": 0,
            },
            "settlement": {
                "active_ledger_reservations_after": 0,
                "ledger_sha256_after_settlement": config.ledger_sha256_before_reservation,
            },
            "remote_evidence": {"manifest_verification_pass": True},
        }
    }
    bindings: dict[str, dict[str, str]] = {}
    for label in ("prior_attempt_cleanup",):
        reference = CONTROLLER.REPLACEMENT_EVIDENCE_BINDINGS[label]
        digest = _write_json(root / reference, payloads[label])
        bindings[label] = {
            "status": "PASS",
            "artifact": reference,
            "sha256": digest,
        }
    post["integrated_replacement_authorization"] = {
        "status": "APPROVED_EXACTLY_ONCE",
        "replacement_attempt_id": "exp004-v11-integrated-s41-c",
        "prior_attempt_id": "exp004-v11-integrated-s41-b",
        "prior_attempt_scientific_status": "INVALID_PRE_RECLAIM_BOUNDED_BACKLOG_ABORT",
        "ledger_sha256_before_reservation": config.ledger_sha256_before_reservation,
        **bindings,
    }
    post_sha = _write_json(post_path, post)
    return config.model_copy(
        update={
            "attempt_id": "exp004-v11-integrated-s41-c",
            "post_micro_review_manifest_sha256": post_sha,
        }
    )


def test_replacement_authorization_is_bound_to_exact_config_and_evidence(
    tmp_path: Path,
) -> None:
    config = _replacement_fixture(tmp_path)
    result = CONTROLLER.verify_sealed_evidence(config, tmp_path)
    assert result["replacement_authorization_verified"] is True
    assert [item["agent"] for item in result["final_pre_gpu_gate"]["reviews"]] == list(range(1, 9))


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("missing_review", "all eight"),
        ("duplicate_agent", "invalid or duplicated"),
        ("wrong_role", "Agent 8 binding is not PASS"),
        ("failed_evidence", "Agent 8 evidence is not PASS offline"),
        ("impersonated_evidence", "Agent 8 evidence is not PASS offline"),
        ("make_check_hash", "hash mismatch"),
    ),
)
def test_fresh_attempt_requires_all_eight_reviews_and_fresh_make_check(
    tmp_path: Path, mutation: str, message: str
) -> None:
    config = _replacement_fixture(tmp_path)
    post_path = tmp_path / "post.json"
    post = json.loads(post_path.read_text())
    gate = post["final_integrated_pre_gpu_gate"]
    if mutation == "missing_review":
        gate["reviews"].pop()
    elif mutation == "duplicate_agent":
        gate["reviews"][-1]["agent"] = 7
    elif mutation == "wrong_role":
        gate["reviews"][-1]["role"] = "not-integrated-methodology"
    elif mutation == "failed_evidence":
        binding = gate["reviews"][-1]
        evidence_path = tmp_path / binding["artifact"]
        evidence = json.loads(evidence_path.read_text())
        evidence["status"] = "FAIL"
        binding["sha256"] = _write_json(evidence_path, evidence)
    elif mutation == "impersonated_evidence":
        binding = gate["reviews"][-1]
        evidence_path = tmp_path / binding["artifact"]
        evidence = json.loads(evidence_path.read_text())
        evidence["agent"] = 7
        evidence["role"] = "gpu-budget-review"
        binding["sha256"] = _write_json(evidence_path, evidence)
    elif mutation == "make_check_hash":
        gate["make_check"]["sha256"] = "0" * 64
    post_sha = _write_json(post_path, post)
    changed = config.model_copy(update={"post_micro_review_manifest_sha256": post_sha})
    with pytest.raises((RuntimeError, ValueError), match=message):
        CONTROLLER.verify_sealed_evidence(changed, tmp_path)


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("attempt", "inconsistent"),
        ("prior", "inconsistent"),
        ("ledger", "inconsistent"),
        ("cleanup_status", "cleanup is not PASS"),
        ("cleanup_path", "path substitution"),
        ("review_hash", "hash mismatch"),
    ),
)
def test_replacement_authorization_fails_closed_on_tamper(
    tmp_path: Path, mutation: str, message: str
) -> None:
    config = _replacement_fixture(tmp_path)
    post_path = tmp_path / "post.json"
    post = json.loads(post_path.read_text())
    replacement = post["integrated_replacement_authorization"]
    if mutation == "attempt":
        replacement["replacement_attempt_id"] = "exp004-v11-integrated-s41-d"
    elif mutation == "prior":
        replacement["prior_attempt_id"] = config.attempt_id
    elif mutation == "ledger":
        replacement["ledger_sha256_before_reservation"] = "0" * 64
    elif mutation == "cleanup_status":
        replacement["prior_attempt_cleanup"]["status"] = "FAIL"
    elif mutation == "cleanup_path":
        replacement["prior_attempt_cleanup"]["artifact"] = "wrong-cleanup.json"
    elif mutation == "review_hash":
        post["final_integrated_pre_gpu_gate"]["reviews"][0]["sha256"] = "0" * 64
    post_sha = _write_json(post_path, post)
    changed = config.model_copy(update={"post_micro_review_manifest_sha256": post_sha})
    with pytest.raises((RuntimeError, ValueError), match=message):
        CONTROLLER.verify_sealed_evidence(changed, tmp_path)


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("embedded_fail", "cleanup payload is inconsistent"),
        ("wrong_attempt", "cleanup payload is inconsistent"),
        ("wrong_ledger", "cleanup payload is inconsistent"),
        ("zero_provider_not_clean", "cleanup payload is inconsistent"),
    ),
)
def test_replacement_authorization_rejects_rehashed_invalid_payload(
    tmp_path: Path, mutation: str, message: str
) -> None:
    config = _replacement_fixture(tmp_path)
    post_path = tmp_path / "post.json"
    post = json.loads(post_path.read_text())
    binding = post["integrated_replacement_authorization"]["prior_attempt_cleanup"]
    cleanup_path = tmp_path / binding["artifact"]
    cleanup = json.loads(cleanup_path.read_text())
    if mutation == "embedded_fail":
        cleanup["status"] = "FAIL"
        cleanup["cleanup_pass"] = False
    elif mutation == "wrong_attempt":
        cleanup["attempt_id"] = config.attempt_id
    elif mutation == "wrong_ledger":
        cleanup["settlement"]["ledger_sha256_after_settlement"] = "0" * 64
    elif mutation == "zero_provider_not_clean":
        cleanup["provider_cleanup"]["running_tasks"] = 1
    binding["sha256"] = _write_json(cleanup_path, cleanup)
    post_sha = _write_json(post_path, post)
    changed = config.model_copy(update={"post_micro_review_manifest_sha256": post_sha})
    with pytest.raises(RuntimeError, match=message):
        CONTROLLER.verify_sealed_evidence(changed, tmp_path)


def test_replacement_authorization_rejects_rehashed_pass_prefix_spoof(
    tmp_path: Path,
) -> None:
    config = _replacement_fixture(tmp_path)
    post_path = tmp_path / "post.json"
    post = json.loads(post_path.read_text())
    binding = post["final_integrated_pre_gpu_gate"]["reviews"][0]
    evidence_path = tmp_path / binding["artifact"]
    evidence = json.loads(evidence_path.read_text())
    evidence["status"] = "PASS_REVOKED"
    binding["sha256"] = _write_json(evidence_path, evidence)
    post_sha = _write_json(post_path, post)
    changed = config.model_copy(update={"post_micro_review_manifest_sha256": post_sha})
    with pytest.raises(RuntimeError, match="evidence is not PASS offline"):
        CONTROLLER.verify_sealed_evidence(changed, tmp_path)


def test_replacement_authorization_rejects_coordinated_third_attempt_reseal(
    tmp_path: Path,
) -> None:
    config = _replacement_fixture(tmp_path)
    post_path = tmp_path / "post.json"
    post = json.loads(post_path.read_text())
    replacement = post["integrated_replacement_authorization"]
    replacement["replacement_attempt_id"] = "exp004-v11-integrated-s41-d"
    post["final_integrated_pre_gpu_gate"]["attempt_id"] = "exp004-v11-integrated-s41-d"
    post_sha = _write_json(post_path, post)
    changed = config.model_copy(
        update={
            "attempt_id": "exp004-v11-integrated-s41-d",
            "post_micro_review_manifest_sha256": post_sha,
        }
    )
    with pytest.raises(RuntimeError, match="inconsistent"):
        CONTROLLER.verify_sealed_evidence(changed, tmp_path)


def test_nonoriginal_attempt_requires_replacement_authorization(tmp_path: Path) -> None:
    config = _sealed_fixture(tmp_path).model_copy(
        update={"attempt_id": "exp004-v11-integrated-s41-b"}
    )
    with pytest.raises(RuntimeError, match="lacks replacement-specific"):
        CONTROLLER.verify_sealed_evidence(config, tmp_path)


def test_inventory_requires_exact_distinct_a100_80gb_pair() -> None:
    row = "0, GPU-a, NVIDIA A100-SXM4-80GB, 580.1, 81920, 0, 0"
    second = "1, GPU-b, NVIDIA A100 80GB PCIe, 580.1, 81920, 0, 0"
    result = CONTROLLER._parse_inventory(f"{row}\n{second}\n")
    assert [item["uuid"] for item in result] == ["GPU-a", "GPU-b"]
    with pytest.raises(RuntimeError, match="exactly two"):
        CONTROLLER._parse_inventory(row)
    with pytest.raises(RuntimeError, match="duplicate physical"):
        CONTROLLER._parse_inventory(f"{row}\n{second.replace('GPU-b', 'GPU-a')}\n")
    with pytest.raises(RuntimeError, match="A100-80GB"):
        CONTROLLER._parse_inventory(f"{row}\n{second.replace('A100 80GB', 'H100 80GB')}\n")


def test_compute_process_parser_and_zero_process_gate() -> None:
    assert CONTROLLER._parse_compute_processes("No running processes found\n") == ()
    processes = CONTROLLER._parse_compute_processes("GPU-a, 123, python, 4096\n")
    with pytest.raises(RuntimeError, match="postflight"):
        CONTROLLER._require_no_compute_processes(processes, phase="postflight")


def test_immutable_barrier_rejects_overwrite(tmp_path: Path) -> None:
    path = tmp_path / "barrier.json"
    CONTROLLER._write_new(path, {"value": 1})
    with pytest.raises(FileExistsError):
        CONTROLLER._write_new(path, {"value": 2})
    assert json.loads(path.read_text()) == {"value": 1}


def test_handoff_validator_binds_selection_sanity_and_engine_identity() -> None:
    rows = _handoffs()
    validated = CONTROLLER._validate_handoffs(
        rows,
        continuity=_continuity(),
        selection_sha256="1" * 64,
        sanity_result_sha256="2" * 64,
        authorization_sha256="3" * 64,
    )
    assert tuple(row["device"] for row in validated) == ("gpu0", "gpu1")
    tampered = (rows[0], rows[1] | {"selected_load_sha256": "3" * 64})
    with pytest.raises(RuntimeError, match="identity/commitment"):
        CONTROLLER._validate_handoffs(
            tampered,
            continuity=_continuity(),
            selection_sha256="1" * 64,
            sanity_result_sha256="2" * 64,
            authorization_sha256="3" * 64,
        )


def test_worker_results_bind_role_pid_uuid_attempt_and_compilation() -> None:
    rows = _worker_results()
    expected = {"serving": (101, "GPU-a"), "rollout": (202, "GPU-b")}
    assert (
        CONTROLLER._validate_worker_results(
            rows,
            expected=expected,
            attempt_id="exp004-v11-integrated-s41-a",
        )
        == rows
    )
    for change in (
        {"pid": 999},
        {"physical_gpu_uuid": "GPU-wrong"},
        {"role": "serving"},
        {"measured_transaction_compilation_observation": {"passed": False}},
    ):
        with pytest.raises(RuntimeError, match="result failed"):
            CONTROLLER._validate_worker_results(
                (rows[0], rows[1] | change),
                expected=expected,
                attempt_id="exp004-v11-integrated-s41-a",
            )


def test_worker_results_reject_missing_or_false_role_scientific_evidence() -> None:
    rows = _worker_results()
    expected = {"serving": (101, "GPU-a"), "rollout": (202, "GPU-b")}
    serving_missing = dict(rows[0])
    serving_missing.pop("gpu0_restore_interference")
    with pytest.raises(RuntimeError, match="interference evidence is absent"):
        CONTROLLER._validate_worker_results(
            (serving_missing, rows[1]),
            expected=expected,
            attempt_id="exp004-v11-integrated-s41-a",
        )
    serving_false = dict(rows[0])
    serving_false["scientific_gates"] = dict(rows[0]["scientific_gates"]) | {
        "slo_restoration_pass": False
    }
    with pytest.raises(RuntimeError, match="GPU0 scientific gate"):
        CONTROLLER._validate_worker_results(
            (serving_false, rows[1]),
            expected=expected,
            attempt_id="exp004-v11-integrated-s41-a",
        )
    rollout_false = dict(rows[1])
    rollout_false["correctness"] = dict(rows[1]["correctness"]) | {"integrity_pass": False}
    with pytest.raises(RuntimeError, match="GPU1 correctness gate"):
        CONTROLLER._validate_worker_results(
            (rows[0], rollout_false),
            expected=expected,
            attempt_id="exp004-v11-integrated-s41-a",
        )
    rollout_short = dict(rows[1])
    rollout_short["continuation"] = {"exact_matches": 8, "minimum_tokens_per_branch": 7}
    with pytest.raises(RuntimeError, match="continuation/movement"):
        CONTROLLER._validate_worker_results(
            (rows[0], rollout_short),
            expected=expected,
            attempt_id="exp004-v11-integrated-s41-a",
        )


@pytest.mark.parametrize(
    ("target", "message"),
    (
        ("control", "control-interval evidence is invalid"),
        ("restore_interference", "actual-restore interference is invalid"),
        ("critical_path", "critical path overlaps or has gaps"),
        ("movement_phase", "movement totals are incomplete or inconsistent"),
    ),
)
def test_worker_results_reject_inconsistent_integrated_methodology_evidence(
    target: str, message: str
) -> None:
    rows = json.loads(json.dumps(_worker_results()))
    expected = {"serving": (101, "GPU-a"), "rollout": (202, "GPU-b")}
    if target == "control":
        rows[0]["gpu0_control_interval"]["completed_rate_per_second"] = 1.0
        rows[0]["gpu0_restore_interference"]["control_interval"]["completed_rate_per_second"] = 1.0
    elif target == "restore_interference":
        rows[0]["gpu0_restore_interference"]["measured_restore_interval"]["emitted_tokens"] = 0
    elif target == "critical_path":
        rows[1]["critical_paths"]["restore"]["stages"][1]["start_ns"] -= 1
    else:
        rows[1]["movement_by_phase"]["source_export"]["full_physical_bytes"] += 1
        rows[1]["movement_by_phase"]["source_export"]["full_physical_amplification"] = (
            rows[1]["movement_by_phase"]["source_export"]["full_physical_bytes"] / 1_056_964_608
        )
    with pytest.raises(RuntimeError, match=message):
        CONTROLLER._validate_worker_results(
            tuple(rows),
            expected=expected,
            attempt_id="exp004-v11-integrated-s41-a",
        )


def test_worker_command_is_exact_single_role_single_gpu_cli(tmp_path: Path) -> None:
    command = CONTROLLER._worker_command(
        worker_path=tmp_path / "worker.py",
        role="serving",
        config_path=tmp_path / "config.json",
        model_snapshot=tmp_path / "model",
        role_root=tmp_path / "serving",
        barrier_root=tmp_path / "barriers",
        physical_gpu_uuid="GPU-a",
    )
    assert command[1:] == [
        str(tmp_path / "worker.py"),
        "--role",
        "serving",
        "--config",
        str(tmp_path / "config.json"),
        "--model-snapshot",
        str(tmp_path / "model"),
        "--work-root",
        str(tmp_path / "serving"),
        "--barrier-root",
        str(tmp_path / "barriers"),
        "--physical-gpu-uuid",
        "GPU-a",
    ]
    with pytest.raises(ValueError, match="role"):
        CONTROLLER._worker_command(
            worker_path=tmp_path / "worker.py",
            role="gpu0",
            config_path=tmp_path / "config.json",
            model_snapshot=tmp_path / "model",
            role_root=tmp_path / "serving",
            barrier_root=tmp_path / "barriers",
            physical_gpu_uuid="GPU-a",
        )


def test_transaction_command_contains_exact_expanded_json_mapping(tmp_path: Path) -> None:
    config = _sealed_fixture(tmp_path)
    expanded = CONTROLLER.expanded_runtime_config(config)
    command = CONTROLLER._transaction_command(
        config=config,
        effective_config=expanded,
        selection_sha256="1" * 64,
        sanity_result_sha256="2" * 64,
    )
    assert command["effective_config"] == expanded
    assert isinstance(command["effective_config"], dict)
    assert command["authorization_artifact_hash"] == config.budget_authorization_sha256
    with pytest.raises(RuntimeError, match="differs from expanded"):
        CONTROLLER._transaction_command(
            config=config,
            effective_config=expanded | {"lambda_spike_rps": 99.0},
            selection_sha256="1" * 64,
            sanity_result_sha256="2" * 64,
        )


def test_engine_start_evidence_is_child_uuid_and_clock_bound() -> None:
    children = (
        ("serving", SimpleNamespace(pid=101), None, None),
        ("rollout", SimpleNamespace(pid=202), None, None),
    )
    inventory = ({"uuid": "GPU-a"}, {"uuid": "GPU-b"})
    rows = tuple(
        {
            "schema_version": "sloforge.branchfabric.retained-engine-start/v1",
            "role": role,
            "pid": pid,
            "physical_gpu_uuid": uuid,
            "engine_started_ns": 150,
        }
        for role, pid, uuid in (("serving", 101, "GPU-a"), ("rollout", 202, "GPU-b"))
    )
    CONTROLLER._validate_engine_start(
        rows,
        children=children,
        inventory=inventory,
        launch_started_ns=100,
        observed_ns=200,
    )
    with pytest.raises(RuntimeError, match="rollout"):
        CONTROLLER._validate_engine_start(
            (rows[0], rows[1] | {"engine_started_ns": 99}),
            children=children,
            inventory=inventory,
            launch_started_ns=100,
            observed_ns=200,
        )


def test_controller_source_has_exact_cli_effective_mapping_and_absolute_bound() -> None:
    source = CONTROLLER_PATH.read_text()
    tree = ast.parse(source)
    imports = {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert not imports & {"torch", "vllm", "triton", "cupy", "pynvml", "cuda"}
    assert source.count('"--model-snapshot"') == 1
    assert '"--role"' in source and '"--physical-gpu-uuid"' in source
    assert "build_live_v10_config(config)" in source
    assert "effective_config = expanded_runtime_config(config)" in source
    assert "_partition_controller_deadlines(" in source
    assert "ABSOLUTE_WALL_SECONDS = 588.0" in source
    assert "start_new_session=True" in source
    assert 'phase="integrated v11 postflight"' in source


def test_operation_cleanup_and_publication_windows_do_not_overlap() -> None:
    second = 1_000_000_000
    entry = 1_000 * second
    # Launcher gives the controller 578s, retaining 10s for immutable result
    # publication/volume commit before Modal's absolute 588s function timeout.
    operation, cleanup = CONTROLLER._partition_controller_deadlines(
        started_ns=entry,
        caller_deadline_ns=entry + 578 * second,
        cleanup_reserve_seconds=10,
    )
    publication_deadline = entry + 588 * second
    assert operation == entry + 568 * second
    assert cleanup == entry + 578 * second
    assert operation < cleanup < publication_deadline
    assert (cleanup - operation) / second == 10
    assert (publication_deadline - cleanup) / second == 10


def test_postflight_zero_compute_wait_uses_caller_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    samples = iter(
        (
            ({"gpu_uuid": "GPU-a", "pid": 1},),
            (),
        )
    )
    monkeypatch.setattr(
        CONTROLLER,
        "_compute_processes",
        lambda *, deadline_ns=None: next(samples),
    )
    monkeypatch.setattr(CONTROLLER.time, "sleep", lambda _seconds: None)
    assert (
        CONTROLLER._wait_for_postflight_zero_compute(
            deadline_ns=time.monotonic_ns() + 1_000_000_000
        )
        == ()
    )

    monkeypatch.setattr(
        CONTROLLER,
        "_compute_processes",
        lambda *, deadline_ns=None: ({"gpu_uuid": "GPU-a", "pid": 1},),
    )
    with pytest.raises(RuntimeError, match="postflight"):
        CONTROLLER._wait_for_postflight_zero_compute(deadline_ns=time.monotonic_ns() - 1)


def test_cleanup_window_terminates_group_after_operation_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    alive = True
    signals: list[int] = []

    class Process:
        pid = 4321

        @staticmethod
        def poll() -> None:
            return None

        @staticmethod
        def wait(*, timeout: float) -> int:
            assert timeout > 0.0
            return 0

    def group_exists(_pid: int) -> bool:
        return alive

    def kill_group(_pid: int, sent: int) -> None:
        nonlocal alive
        signals.append(sent)
        alive = False

    monkeypatch.setattr(CONTROLLER, "_process_group_exists", group_exists)
    monkeypatch.setattr(CONTROLLER.os, "killpg", kill_group)
    actions: list[dict[str, Any]] = []
    # The operation deadline may already have elapsed. Teardown receives the
    # separately reserved cleanup deadline and must still terminate the group.
    CONTROLLER._terminate_group(
        Process(),
        actions,
        deadline_ns=time.monotonic_ns() + 10_000_000_000,
    )
    assert signals == [signal.SIGTERM]
    assert actions[0]["signal"] == "SIGTERM"


def test_two_worker_cleanup_signals_both_groups_within_one_shared_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    alive = {101: True, 202: True}
    sent: list[tuple[int, int]] = []

    class Process:
        def __init__(self, pid: int) -> None:
            self.pid = pid

        @staticmethod
        def poll() -> None:
            return None

        @staticmethod
        def wait(*, timeout: float) -> int:
            assert timeout > 0.0
            return 0

    monkeypatch.setattr(CONTROLLER, "_process_group_exists", lambda pid: alive[pid])

    def kill_group(pid: int, sent_signal: int) -> None:
        sent.append((pid, sent_signal))
        alive[pid] = False

    monkeypatch.setattr(CONTROLLER.os, "killpg", kill_group)
    actions: list[dict[str, Any]] = []
    CONTROLLER._terminate_groups(
        (Process(101), Process(202)),
        actions,
        deadline_ns=time.monotonic_ns() + 10_000_000_000,
    )
    assert sent == [(101, signal.SIGTERM), (202, signal.SIGTERM)]
    assert [item["process_group"] for item in actions] == [101, 202]


def test_group_cleanup_reaps_dead_leader_before_declaring_pgid_alive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    killed = False
    reaped = False

    class Process:
        pid = 909
        returncode: int | None = None

        def poll(self) -> int | None:
            nonlocal reaped
            if killed:
                self.returncode = -signal.SIGTERM
                reaped = True
            return self.returncode

        def wait(self, *, timeout: float) -> int:
            assert timeout > 0.0
            return self.poll() or 0

    process = Process()

    def group_exists(_pid: int) -> bool:
        # Linux reports a PGID containing an unreaped zombie as present.
        return not reaped

    def kill_group(_pid: int, sent_signal: int) -> None:
        nonlocal killed
        assert sent_signal == signal.SIGTERM
        killed = True

    monkeypatch.setattr(CONTROLLER, "_process_group_exists", group_exists)
    monkeypatch.setattr(CONTROLLER.os, "killpg", kill_group)
    actions: list[dict[str, Any]] = []
    CONTROLLER._terminate_groups(
        (process,),
        actions,
        deadline_ns=time.monotonic_ns() + 1_000_000_000,
    )
    assert reaped is True
    assert process.returncode == -signal.SIGTERM
    assert [item["signal"] for item in actions] == ["SIGTERM"]
    assert actions[0]["forced"] is False


def test_group_cleanup_records_bounded_sigkill_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    alive = True

    class Process:
        pid = 910
        returncode: int | None = None

        def poll(self) -> int | None:
            return self.returncode

        def wait(self, *, timeout: float) -> int:
            assert timeout >= 0.0
            return self.returncode or 0

    process = Process()
    clock_ns = 0

    def monotonic_ns() -> int:
        nonlocal clock_ns
        clock_ns += 1_000_000_000
        return clock_ns

    def kill_group(_pid: int, sent_signal: int) -> None:
        nonlocal alive
        if sent_signal == signal.SIGKILL:
            alive = False
            process.returncode = -signal.SIGKILL

    monkeypatch.setattr(CONTROLLER.time, "monotonic_ns", monotonic_ns)
    monkeypatch.setattr(CONTROLLER.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(CONTROLLER, "_process_group_exists", lambda _pid: alive)
    monkeypatch.setattr(CONTROLLER.os, "killpg", kill_group)
    actions: list[dict[str, Any]] = []
    CONTROLLER._terminate_groups((process,), actions, deadline_ns=20_000_000_000)
    assert [item["signal"] for item in actions] == ["SIGTERM", "SIGKILL"]
    assert [item["forced"] for item in actions] == [False, True]
    assert process.returncode == -signal.SIGKILL


def test_lifecycle_tracks_reused_pid_as_new_owned_descendant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = CONTROLLER.os.getpid()

    def row(pid: int, ppid: int, *, start_token: str) -> dict[str, Any]:
        return {
            "pid": pid,
            "ppid": ppid,
            "pgid": pid,
            "sid": pid,
            "state": "S",
            "start_token": start_token,
            "command": f"worker-{pid}",
        }

    snapshots = iter(
        (
            (row(500, parent, start_token="old"),),
            (row(600, parent, start_token="worker"),),
            (
                row(600, parent, start_token="worker"),
                row(500, parent, start_token="new"),
            ),
        )
    )
    monkeypatch.setattr(CONTROLLER, "_process_tree", lambda: next(snapshots))
    monkeypatch.setattr(CONTROLLER, "_thread_snapshot", lambda: ())
    monkeypatch.setattr(CONTROLLER, "_ipc_snapshot", lambda: ())
    monkeypatch.setattr(CONTROLLER.os, "getpgid", lambda pid: pid if pid else parent)
    monkeypatch.setattr(CONTROLLER.os, "getsid", lambda pid: pid if pid else parent)
    lifecycle = CONTROLLER._ProcessLifecycleAudit()
    process = SimpleNamespace(pid=600)
    lifecycle.register_worker("serving", process)
    lifecycle.observe("pid-reuse", force=True)
    assert set(lifecycle.owned) == {500, 600}
    assert lifecycle.owned[500]["start_token"] == "new"
    lifecycle.subreaper.restore()


def test_in_function_cleanup_schema_requires_complete_ordered_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    threads = (
        {
            "name": "MainThread",
            "ident": 1,
            "native_id": 1,
            "daemon": False,
            "alive": True,
        },
    )
    monkeypatch.setattr(CONTROLLER, "_thread_snapshot", lambda: threads)
    monkeypatch.setattr(CONTROLLER, "_ipc_snapshot", lambda: ())
    lifecycle = SimpleNamespace(
        parent_pid=10,
        parent_pgid=10,
        parent_sid=10,
        transitions=[
            {"phase": phase, "observed_at_monotonic_ns": index}
            for index, phase in enumerate(CONTROLLER.PROCESS_LIFECYCLE_PHASES)
        ],
        initial_threads=threads,
        initial_ipc=(),
        owned={
            101: {
                "pid": 101,
                "ppid": 10,
                "pgid": 101,
                "sid": 101,
                "role": "serving",
                "process_kind": "worker",
                "command": "python serving-worker.py",
                "cuda_owning_candidate": True,
            }
        },
        reaped={
            101: {
                "exit_status": 0,
                "reap_method": "popen-waitpid",
                "reap_timestamp_utc": "2026-08-19T00:00:00+00:00",
                "reap_timestamp_monotonic_ns": 9,
            }
        },
    )
    process = SimpleNamespace(pid=101, returncode=0)
    cleanup = CONTROLLER._build_in_function_cleanup(
        lifecycle=lifecycle,
        actions=(),
        processes=(process,),
        surviving_children=(),
        surviving_groups=(),
        compute_processes_after=(),
        cuda_release_verified=True,
        pipes_closed=True,
        cleanup_errors=(),
    )
    assert cleanup["schema_version"] == "sloforge.branchfabric.in-function-cleanup/v1"
    assert cleanup["status"] == "PASS"
    assert cleanup["pass"] is True
    assert cleanup["parent_pid"] == 10
    assert cleanup["parent_pgid"] == 10
    assert cleanup["parent_sid"] == 10
    assert cleanup["parent_reaped_all_owned_children"] is True
    assert cleanup["owned_ipc_resources_released"] is True
    assert cleanup["cuda_released"] is True
    assert cleanup["owned_children"][0]["exit_status"] == 0
    assert cleanup["owned_children"][0]["reap_timestamp_monotonic_ns"] == 9


@pytest.mark.parametrize(
    "survivor",
    (
        {
            "pid": 301,
            "ppid": 10,
            "pgid": 301,
            "sid": 301,
            "state": "S",
            "role": "serving",
            "command": "python serving-worker.py",
        },
        {
            "pid": 302,
            "ppid": 10,
            "pgid": 302,
            "sid": 302,
            "state": "S",
            "role": "helper",
            "command": "python -m multiprocessing.resource_tracker",
        },
        {
            "pid": 303,
            "ppid": 10,
            "pgid": 303,
            "sid": 303,
            "state": "S",
            "role": "helper",
            "command": "/usr/local/bin/nsys profile worker",
        },
        {
            "pid": 304,
            "ppid": 10,
            "pgid": 304,
            "sid": 304,
            "state": "Z",
            "role": "rollout",
            "command": "python rollout-worker.py",
        },
    ),
)
def test_in_function_cleanup_fails_closed_on_any_owned_survivor(
    monkeypatch: pytest.MonkeyPatch,
    survivor: dict[str, Any],
) -> None:
    monkeypatch.setattr(CONTROLLER, "_thread_snapshot", lambda: ())
    monkeypatch.setattr(CONTROLLER, "_ipc_snapshot", lambda: ())
    lifecycle = SimpleNamespace(
        parent_pid=10,
        parent_pgid=10,
        parent_sid=10,
        transitions=[{"phase": phase} for phase in CONTROLLER.PROCESS_LIFECYCLE_PHASES],
        initial_threads=(),
        initial_ipc=(),
        owned={},
        reaped={},
    )
    cleanup = CONTROLLER._build_in_function_cleanup(
        lifecycle=lifecycle,
        actions=(),
        processes=(),
        surviving_children=(survivor,),
        surviving_groups=(int(survivor["pgid"]),),
        compute_processes_after=(),
        cuda_release_verified=True,
        pipes_closed=True,
        cleanup_errors=(),
    )
    assert cleanup["status"] == "FAIL"
    assert cleanup["pass"] is False


def test_forced_kill_is_explicit_but_successful_cleanup_can_pass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(CONTROLLER, "_thread_snapshot", lambda: ())
    monkeypatch.setattr(CONTROLLER, "_ipc_snapshot", lambda: ())
    lifecycle = SimpleNamespace(
        parent_pid=10,
        parent_pgid=10,
        parent_sid=10,
        transitions=[{"phase": phase} for phase in CONTROLLER.PROCESS_LIFECYCLE_PHASES],
        initial_threads=(),
        initial_ipc=(),
        owned={},
        reaped={},
    )
    action = {
        "pid": 101,
        "process_group": 101,
        "signal": "SIGKILL",
        "forced": True,
        "target": "worker-process-group",
    }
    cleanup = CONTROLLER._build_in_function_cleanup(
        lifecycle=lifecycle,
        actions=(action,),
        processes=(),
        surviving_children=(),
        surviving_groups=(),
        compute_processes_after=(),
        cuda_release_verified=True,
        pipes_closed=True,
        cleanup_errors=(),
    )
    assert cleanup["pass"] is True
    assert cleanup["forced_kill_required"] is True
    assert cleanup["forced_kills"] == [action]


def test_controller_rejects_expired_deadline_before_any_gpu_or_file_work(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="future absolute deadline"):
        CONTROLLER.run_integrated_v11_controller(
            config_path=tmp_path / "missing-config",
            work_root=tmp_path / "work",
            worker_path=tmp_path / "missing-worker",
            model_snapshot=tmp_path / "missing-model",
            repository_root=tmp_path,
            absolute_deadline_ns=time.monotonic_ns() - 1,
        )


def test_frozen_v10_sources_are_unchanged() -> None:
    expected = {
        "gpu_reclamation_controller.py": (
            "c91ccd8b3aa4b4c5a9034a5137ccc33257e3471528a96dcc2ca7920c8b7eefc9"
        ),
        "gpu_reclamation_worker.py": (
            "e6858e445ddbca912877fc67155dd9422c82d6f6fd4dcbf71871831dee15008a"
        ),
        "gpu_reclamation_v10_serving.py": (
            "d53b007be4fc72d760ae7a83223773a206e0a0eed52137a834ce5b601bcad3ba"
        ),
    }
    observed = {
        name: hashlib.sha256((EXPERIMENT_ROOT / name).read_bytes()).hexdigest() for name in expected
    }
    assert observed == expected
