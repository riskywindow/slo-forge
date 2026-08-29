from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import ValidationError

from sloforge.helix.characterization.gpu_reclamation_v11_methodology import (
    Experiment004V11KillRecomputeConfig,
)

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS = ROOT / "experiments/branchfabric"
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))


def _load_experiment_module(name: str) -> Any:
    path = EXPERIMENTS / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


controller = _load_experiment_module("gpu_reclamation_kill_recompute_controller_v11")
worker = _load_experiment_module("gpu_reclamation_kill_recompute_worker_v11")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _config_payload() -> dict[str, Any]:
    path = (
        ROOT / "artifacts/branchfabric/gpu-validation/experiment-004/v11-final/integrated/"
        "exp004-v11-integrated-s41-k-config.json"
    )
    payload = json.loads(path.read_text())
    payload.update(
        {
            "schema_version": ("sloforge.branchfabric.experiment-004-kill-recompute-config/v1"),
            "mode": "KILL_AND_RECOMPUTE",
            "execution_mode": "integrated-kill-recompute-v11",
            "attempt_id": "exp004-v11-kill-recompute-s41-g",
            "maximum_wall_seconds": 660.0,
            "budget_authorization": (
                "artifacts/branchfabric/gpu-validation/experiment-004/"
                "budget-authorization-v11-continuation.json"
            ),
            "budget_authorization_sha256": (
                "cf71d63d6427a58003f25dd87922cc19406a1c720e4b03dbb250094d2d9e02d6"
            ),
            "ledger_sha256_before_reservation": (
                "aa81df8d3a26e370af1e715f86cb13a1b670cdccd4bbf6d4707dae1802a4ca26"
            ),
            "integrated_k_status_artifact": (
                "artifacts/branchfabric/gpu-validation/experiment-004/v11-final/"
                "integrated/status-attempt-k.json"
            ),
            "integrated_k_status_sha256": (
                "6036621e56cea813278307b4957212509dffc38aa4e5fd679430896e289d7b30"
            ),
            "integrated_k_remote_manifest": (
                "artifacts/branchfabric/gpu-validation/experiment-004/v11-final/"
                "integrated/raw/exp004-v11-integrated-s41-k/REMOTE_MANIFEST.json"
            ),
            "integrated_k_remote_manifest_sha256": (
                "e4571c830bf695cca56e2e9880852204a045be237ae76e668368268ec28a025d"
            ),
            "integrated_k_provider_cleanup_artifact": (
                "artifacts/branchfabric/gpu-validation/experiment-004/v11-final/"
                "integrated/provider-cleanup-attempt-k.json"
            ),
            "integrated_k_provider_cleanup_sha256": (
                "5a946827a594010051a39bbd1feb0becf02e0b027325cdd2da7d8dd175499d33"
            ),
            "optimized_v11_freeze_artifact": (
                "artifacts/branchfabric/gpu-validation/experiment-004/v11-final/"
                "integrated/v11-freeze-record.json"
            ),
            "optimized_v11_freeze_sha256": (
                "76bddc6fcd35c304ced2202e044babdb8bc56d108e3d7d63ac36c448b08ddbf3"
            ),
            "optimized_v11_freeze_tag_binding_artifact": (
                "artifacts/branchfabric/gpu-validation/experiment-004/v11-final/"
                "integrated/v11-freeze-tag-binding.json"
            ),
            "optimized_v11_freeze_tag_binding_sha256": (
                "9bc0120698a387229ddd3e646cb17064b8eacab2ccac5cd5369135e7a81c1a0a"
            ),
            "optimized_v11_freeze_tag_object": ("14c6a7203b1b0811661126901ef78097091f86fa"),
            "ledger_snapshot_artifact": (
                "artifacts/branchfabric/gpu-validation/experiment-004/v11-final/"
                "kill-recompute/authorization/ledger-snapshot-before-attempt-g.json"
            ),
            "ledger_snapshot_sha256": (
                "aa81df8d3a26e370af1e715f86cb13a1b670cdccd4bbf6d4707dae1802a4ca26"
            ),
        }
    )
    return payload


def _tables() -> list[SimpleNamespace]:
    prefix = tuple(range(16_384))
    return [
        SimpleNamespace(
            logical_branch_id=f"branch.{index}",
            token_ids=prefix + tuple([20_000 + index] * 256),
            computed_tokens=16_640,
        )
        for index in range(8)
    ]


def _live_boundary_tables() -> list[SimpleNamespace]:
    """Mirror vLLM 0.23's quiesced decode boundary from Attempt C.

    The final sampled token is committed to ``all_token_ids`` but has not yet
    been consumed by the model, so the authoritative KV boundary trails the
    live history by exactly one token.
    """

    prefix = tuple(range(16_384))
    return [
        SimpleNamespace(
            logical_branch_id=f"branch.{index}",
            token_ids=(prefix + tuple([20_000 + index] * 256) + (30_000 + index,)),
            computed_tokens=16_640,
        )
        for index in range(8)
    ]


def _attempt_c_bundle_root() -> Path:
    return (
        ROOT / "artifacts/branchfabric/gpu-validation/experiment-004/v11-final/"
        "kill-recompute/raw/exp004-v11-kill-recompute-s41-c"
    )


def _attempt_g_prelaunch_ledger_path() -> Path:
    return (
        ROOT / "artifacts/branchfabric/gpu-validation/experiment-004/v11-final/"
        "kill-recompute/authorization/ledger-snapshot-before-attempt-g.json"
    )


def _postworker_success_controller_payload(coordinator: Any) -> dict[str, Any]:
    payload = json.loads(
        (
            ROOT / "artifacts/branchfabric/gpu-validation/experiment-004/v11-final/integrated/"
            "raw/exp004-v11-integrated-s41-k/controller-result.json"
        ).read_text()
    )
    payload.update(
        {
            "attempt_id": coordinator.ATTEMPT_ID,
            "absolute_wall_seconds": 660.0,
            "execution_mode": "integrated-kill-recompute-v11",
        }
    )
    payload["no_preservation_output_artifacts"] = {
        "schema_version": "sloforge.branchfabric.kill-recompute-no-preservation-outputs/v1",
        "scan_scope": "controller work root after both workers and before function return",
        "scanned_file_count": 1,
        "scanned_relative_paths_sha256": "a" * 64,
        "forbidden_path_tokens": [
            "checkpoint",
            "source-capture-commit",
            "pre-export",
            "post-export",
            "source-release",
            "state-pass",
            "movement",
            "optimized-export",
        ],
        "forbidden_matches": [],
        "checkpoint_materialized": False,
        "optimized_export_executed": False,
        "passed": True,
    }
    for result in payload["worker_results"]:
        result["attempt_id"] = coordinator.ATTEMPT_ID
        result["mode"] = "KILL_AND_RECOMPUTE"
    return payload


def _postworker_failure_controller_payload(coordinator: Any) -> dict[str, Any]:
    payload = json.loads((_attempt_c_bundle_root() / "kill-controller-result.json").read_text())
    payload["attempt_id"] = coordinator.ATTEMPT_ID
    return payload


def _bound_k_evidence(config: Experiment004V11KillRecomputeConfig) -> dict[str, str]:
    return {
        "integrated_k_status": config.integrated_k_status_artifact,
        "integrated_k_status_sha256": config.integrated_k_status_sha256,
        "integrated_k_remote_manifest": config.integrated_k_remote_manifest,
        "integrated_k_remote_manifest_sha256": config.integrated_k_remote_manifest_sha256,
        "integrated_k_provider_cleanup": config.integrated_k_provider_cleanup_artifact,
        "integrated_k_provider_cleanup_sha256": config.integrated_k_provider_cleanup_sha256,
        "optimized_v11_freeze": config.optimized_v11_freeze_artifact,
        "optimized_v11_freeze_sha256": config.optimized_v11_freeze_sha256,
        "optimized_v11_freeze_tag_binding": config.optimized_v11_freeze_tag_binding_artifact,
        "optimized_v11_freeze_tag_binding_sha256": (config.optimized_v11_freeze_tag_binding_sha256),
        "ledger_snapshot": config.ledger_snapshot_artifact,
        "ledger_snapshot_sha256": config.ledger_snapshot_sha256,
    }


def _postworker_failure_envelope(
    coordinator: Any,
    *,
    config: Experiment004V11KillRecomputeConfig,
    reservation_id: str,
    reservation_commitment: str,
) -> dict[str, Any]:
    controller_payload = _postworker_failure_controller_payload(coordinator)
    cleanup = controller_payload["in_function_cleanup"]
    entry = 1_000_000_000
    remote = {
        "schema_version": "sloforge.branchfabric.kill-recompute-completion/v1",
        "status": "failed",
        "scientific_status": "invalid",
        "attempt_id": coordinator.ATTEMPT_ID,
        "reservation_id": reservation_id,
        "reservation_commitment_sha256": reservation_commitment,
        "config_sha256": hashlib.sha256(
            coordinator._BASE.canonical_json_bytes(
                Experiment004V11KillRecomputeConfig.model_validate(config, strict=True)
            )
        ).hexdigest(),
        "function_call_id": "fc-postworker-fixture",
        "requested_gpu": "A100-80GB:2",
        "gpu_count": 2,
        "controller_and_analysis_interval_seconds": 0.9,
        "gpu_allocation_seconds_status": "pending-final-function-return",
        "hardware_comparability": {
            "modal_request": "A100-80GB:2",
            "observed_gpu_names": [],
            "observed_gpu_uuids": [],
            "direct_k_comparison_ready": False,
        },
        "bound_k_evidence": _bound_k_evidence(config),
        "absolute_deadlines": {
            "function_entry_monotonic_ns": entry,
            "controller_deadline_monotonic_ns": entry + 650_000_000_000,
            "function_deadline_monotonic_ns": entry + 660_000_000_000,
            "post_controller_reserve_seconds": 10.0,
        },
        "controller": controller_payload,
        "in_function_cleanup": cleanup,
        "run_error": {
            "type": "RuntimeError",
            "message": "kill/recompute controller failed closed",
            "traceback": (
                "Traceback: raise RuntimeError('kill/recompute controller failed closed')"
            ),
        },
        "completed_at_utc": "2026-08-27T00:00:00+00:00",
        "gpu_allocation_seconds": 1.0,
        "gpu_seconds": 2.0,
        "gpu_hours": 2.0 / 3600.0,
        "remote_prefix": f"{coordinator.REMOTE_PREFIX}/{coordinator.ATTEMPT_ID}",
        "remote_manifest_sha256": "d" * 64,
    }
    return {
        "result": remote,
        "materialized": {
            "remote_path": remote["remote_prefix"],
            "volume_name": coordinator.RESULTS_VOLUME,
        },
    }


def _preworker_failure_envelope(
    coordinator: Any,
    *,
    config: Experiment004V11KillRecomputeConfig,
    reservation_id: str,
    reservation_commitment: str,
) -> dict[str, Any]:
    envelope = _postworker_failure_envelope(
        coordinator,
        config=config,
        reservation_id=reservation_id,
        reservation_commitment=reservation_commitment,
    )
    controller_payload = json.loads(
        (
            ROOT / "artifacts/branchfabric/gpu-validation/experiment-004/v11-final/integrated/"
            "raw/exp004-v11-integrated-s41-h/controller-result.json"
        ).read_text()
    )
    controller_payload.update(
        {
            "attempt_id": coordinator.ATTEMPT_ID,
            "absolute_wall_seconds": 660.0,
            "execution_mode": "integrated-kill-recompute-v11",
            "no_preservation_output_artifacts": copy.deepcopy(
                envelope["result"]["controller"]["no_preservation_output_artifacts"]
            ),
        }
    )
    envelope["result"]["controller"] = controller_payload
    envelope["result"]["in_function_cleanup"] = controller_payload["in_function_cleanup"]
    return envelope


def _coordinator() -> Any:
    path = ROOT / "tools/branchfabric-experiment-004-v11-kill-recompute.py"
    spec = importlib.util.spec_from_file_location("_kill_coordinator_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _attempt_c_coordinator() -> Any:
    module = _coordinator()
    module.ATTEMPT_ID = module.ATTEMPT_C_ID
    return module


def _run_g_loader_fixture(*, tamper: str | None) -> None:
    coordinator = _coordinator()
    temporary_root = Path(tempfile.mkdtemp(prefix=".kill-g-loader-", dir=ROOT))
    try:
        coordinator.FUNCTION_CALL_EVIDENCE_PARENT = temporary_root / "function-calls"
        ledger_path = temporary_root / "ledger-before-attempt-g.json"
        ledger_path.write_bytes(_attempt_g_prelaunch_ledger_path().read_bytes())
        ledger = json.loads(ledger_path.read_text())
        ledger_sha = _sha(ledger_path)
        remaining = (
            ledger["hard_additional_gpu_seconds"] - ledger["consumed_additional_gpu_seconds"]
        )
        continuation = json.loads(
            (ROOT / coordinator.CONTINUATION_AUTHORIZATION_REFERENCE).read_text()
        )
        if tamper == "status-lifecycle":
            payload = json.loads((ROOT / coordinator.ATTEMPT_C_STATUS_REFERENCE).read_text())
            payload["lifecycle"]["source_state_discarded"] = True
            path = temporary_root / "attempt-c-status.json"
            path.write_text(json.dumps(payload, indent=2) + "\n")
            coordinator.ATTEMPT_C_STATUS_REFERENCE = str(path.relative_to(ROOT))
            coordinator.ATTEMPT_C_STATUS_SHA256 = _sha(path)
        elif tamper == "provider-active":
            payload = json.loads((ROOT / coordinator.ATTEMPT_C_PROVIDER_REFERENCE).read_text())
            payload["active_apps"] = ["ap-tamper"]
            path = temporary_root / "attempt-c-provider.json"
            path.write_text(json.dumps(payload, indent=2) + "\n")
            coordinator.ATTEMPT_C_PROVIDER_REFERENCE = str(path.relative_to(ROOT))
            coordinator.ATTEMPT_C_PROVIDER_SHA256 = _sha(path)
        elif tamper == "provider-inventory":
            payload = json.loads(
                (ROOT / coordinator.ATTEMPT_C_PROVIDER_INVENTORY_REFERENCE).read_text()
            )
            payload["commands"][1]["rows"] = [{"container_id": "ct-tamper"}]
            path = temporary_root / "attempt-c-provider-inventory.json"
            path.write_text(json.dumps(payload, indent=2) + "\n")
            coordinator.ATTEMPT_C_PROVIDER_INVENTORY_REFERENCE = str(path.relative_to(ROOT))
            coordinator.ATTEMPT_C_PROVIDER_INVENTORY_SHA256 = _sha(path)
        elif tamper == "charge":
            payload = json.loads((ROOT / coordinator.ATTEMPT_C_CHARGE_REFERENCE).read_text())
            payload["charged_gpu_seconds"] = 1.0
            path = temporary_root / "attempt-c-charge.json"
            path.write_text(json.dumps(payload, indent=2) + "\n")
            coordinator.ATTEMPT_C_CHARGE_REFERENCE = str(path.relative_to(ROOT))
            coordinator.ATTEMPT_C_CHARGE_SHA256 = _sha(path)
        elif tamper == "review":
            payload = json.loads((ROOT / coordinator.ATTEMPT_C_REVIEW_REFERENCE).read_text())
            payload["root_cause"]["root_cause_class"] = "LIFECYCLE_BUG"
            path = temporary_root / "attempt-c-review.json"
            path.write_text(json.dumps(payload, indent=2) + "\n")
            coordinator.ATTEMPT_C_REVIEW_REFERENCE = str(path.relative_to(ROOT))
            coordinator.ATTEMPT_C_REVIEW_SHA256 = _sha(path)
        elif tamper == "manifest-hash":
            coordinator.ATTEMPT_C_MANIFEST_SHA256 = "0" * 64
        elif tamper == "d-status":
            payload = json.loads((ROOT / coordinator.ATTEMPT_D_STATUS_REFERENCE).read_text())
            payload["prebuild_started"] = True
            path = temporary_root / "attempt-d-status.json"
            path.write_text(json.dumps(payload, indent=2) + "\n")
            coordinator.ATTEMPT_D_STATUS_REFERENCE = str(path.relative_to(ROOT))
            coordinator.ATTEMPT_D_STATUS_SHA256 = _sha(path)
        elif tamper == "d-capture":
            payload = json.loads((ROOT / coordinator.ATTEMPT_D_CAPTURE_REFERENCE).read_text())
            payload["original_cli_inventory_rows_available"] = True
            payload["original_cli_inventory_rows"] = []
            path = temporary_root / "attempt-d-capture.json"
            path.write_text(json.dumps(payload, indent=2) + "\n")
            coordinator.ATTEMPT_D_CAPTURE_REFERENCE = str(path.relative_to(ROOT))
            coordinator.ATTEMPT_D_CAPTURE_SHA256 = _sha(path)
        elif tamper == "f-status":
            payload = json.loads((ROOT / coordinator.ATTEMPT_F_STATUS_REFERENCE).read_text())
            payload["lifecycle"]["strict_nine_rps_control_interval_passed"] = True
            path = temporary_root / "attempt-f-status.json"
            path.write_text(json.dumps(payload, indent=2) + "\n")
            coordinator.ATTEMPT_F_STATUS_REFERENCE = str(path.relative_to(ROOT))
            coordinator.ATTEMPT_F_STATUS_SHA256 = _sha(path)
        elif tamper == "f-provider":
            payload = json.loads((ROOT / coordinator.ATTEMPT_F_PROVIDER_REFERENCE).read_text())
            payload["active_apps"] = ["ap-tamper"]
            path = temporary_root / "attempt-f-provider.json"
            path.write_text(json.dumps(payload, indent=2) + "\n")
            coordinator.ATTEMPT_F_PROVIDER_REFERENCE = str(path.relative_to(ROOT))
            coordinator.ATTEMPT_F_PROVIDER_SHA256 = _sha(path)
        elif tamper == "f-provider-inventory":
            payload = json.loads(
                (ROOT / coordinator.ATTEMPT_F_PROVIDER_INVENTORY_REFERENCE).read_text()
            )
            payload["commands"][1]["rows"] = [{"container_id": "ct-tamper"}]
            path = temporary_root / "attempt-f-provider-inventory.json"
            path.write_text(json.dumps(payload, indent=2) + "\n")
            coordinator.ATTEMPT_F_PROVIDER_INVENTORY_REFERENCE = str(path.relative_to(ROOT))
            coordinator.ATTEMPT_F_PROVIDER_INVENTORY_SHA256 = _sha(path)
        elif tamper == "f-charge":
            payload = json.loads((ROOT / coordinator.ATTEMPT_F_CHARGE_REFERENCE).read_text())
            payload["charged_gpu_seconds"] = 1.0
            path = temporary_root / "attempt-f-charge.json"
            path.write_text(json.dumps(payload, indent=2) + "\n")
            coordinator.ATTEMPT_F_CHARGE_REFERENCE = str(path.relative_to(ROOT))
            coordinator.ATTEMPT_F_CHARGE_SHA256 = _sha(path)
        elif tamper == "f-manifest-hash":
            coordinator.ATTEMPT_F_MANIFEST_SHA256 = "0" * 64
        elif tamper == "continuation-binding":
            continuation["evidence_bindings"]["attempt_c_status"]["sha256"] = "0" * 64
        elif tamper == "continuation-d-binding":
            continuation["evidence_bindings"]["attempt_d_status"]["sha256"] = "0" * 64
        elif tamper == "continuation-f-binding":
            continuation["evidence_bindings"]["attempt_f_status"]["sha256"] = "0" * 64
        elif tamper == "continuation-budget":
            continuation["budget"]["remaining_gpu_seconds"] += 1.0
        elif tamper == "continuation-scope":
            continuation["restrictions"] = []
        elif tamper == "continuation-policy":
            continuation["control_gate_fix_contract"]["scope"] = "INTEGRATED"
        if tamper in {
            "continuation-binding",
            "continuation-d-binding",
            "continuation-f-binding",
            "continuation-budget",
            "continuation-scope",
            "continuation-policy",
        }:
            continuation_path = temporary_root / "continuation.json"
            continuation_path.write_text(json.dumps(continuation, indent=2) + "\n")
            coordinator.CONTINUATION_AUTHORIZATION_REFERENCE = str(
                continuation_path.relative_to(ROOT)
            )
            coordinator.CONTINUATION_AUTHORIZATION_SHA256 = _sha(continuation_path)

        config_payload = _config_payload()
        ledger_snapshot = temporary_root / "ledger-snapshot-before-attempt-g.json"
        ledger_snapshot.write_bytes(ledger_path.read_bytes())
        config_payload.update(
            {
                "ledger_sha256_before_reservation": ledger_sha,
                "ledger_snapshot_artifact": str(ledger_snapshot.relative_to(ROOT)),
                "ledger_snapshot_sha256": ledger_sha,
            }
        )
        config_path = temporary_root / "config.json"
        config_path.write_text(json.dumps(config_payload, indent=2) + "\n")

        make_check = {
            "schema_version": (
                "sloforge.branchfabric.experiment-004-v11-kill-recompute-make-check/v1"
            ),
            "status": "PASS",
            "attempt_id": coordinator.ATTEMPT_ID,
            "command": "make check",
            "gpu_or_cloud_invoked": False,
            "completed_at_utc": "2026-08-28T04:00:00Z",
            "exit_code": 0,
            "python": {
                "ruff_format_check": "PASS; 714 files",
                "ruff_check": "PASS",
                "mypy": "PASS; 484 source files",
                "pytest": "PASS; 1 passed, 0 skipped, 0 warnings",
            },
            "rust": {
                "cargo_fmt": "PASS",
                "cargo_clippy_all_targets_all_features": "PASS",
                "cargo_test_workspace_all_features": ("PASS; unit, integration, and doc tests"),
            },
            "ui": {
                "typecheck": "PASS",
                "lint": "PASS",
                "tests": "PASS; 37 passed, 1 skipped",
                "build": "PASS",
            },
            "provenance": "CPU-only loader fixture",
        }
        make_path = temporary_root / "make-check.json"
        make_path.write_text(json.dumps(make_check, indent=2) + "\n")
        coordinator.MAKE_CHECK_REFERENCE = str(make_path.relative_to(ROOT))

        seal = {
            "schema_version": ("sloforge.branchfabric.experiment-004-v11-kill-recompute-seal/v1"),
            "status": "PASS",
            "attempt_id": coordinator.ATTEMPT_ID,
            "maximum_invocations": 1,
            "maximum_wall_seconds": 660.0,
            "maximum_gpu_seconds": 1320.0,
            "config": {
                "artifact": str(config_path.relative_to(ROOT)),
                "sha256": _sha(config_path),
            },
            "current_ledger": {
                "artifact": str(ledger_path.relative_to(ROOT)),
                "sha256": ledger_sha,
                "reservations_empty": True,
            },
            "optimized_v11_freeze": {
                "artifact": config_payload["optimized_v11_freeze_artifact"],
                "sha256": config_payload["optimized_v11_freeze_sha256"],
                "tag": "branchfabric-exp004-optimized-preserve-v11",
                "tag_object": config_payload["optimized_v11_freeze_tag_object"],
            },
            "make_check": {
                "artifact": coordinator.MAKE_CHECK_REFERENCE,
                "sha256": _sha(make_path),
                "status": "PASS",
            },
            "prior_attempt_a": {
                "attempt_id": coordinator.ATTEMPT_A_ID,
                "failure_classification": (
                    "INVALID_PRE_RESERVATION_MISSING_PREFIX_PARSER_ASSUMPTION"
                ),
                "status": {
                    "artifact": coordinator.PRIOR_ATTEMPT_STATUS_REFERENCE,
                    "sha256": coordinator.PRIOR_ATTEMPT_STATUS_SHA256,
                },
                "cli_capture": {
                    "artifact": coordinator.PRIOR_ATTEMPT_CAPTURE_REFERENCE,
                    "sha256": coordinator.PRIOR_ATTEMPT_CAPTURE_SHA256,
                },
                "ledger_unchanged": True,
                "consumed_authorized_arm": False,
            },
            "prior_attempt_b": {
                "attempt_id": coordinator.ATTEMPT_B_ID,
                "failure_classification": (
                    "INVALID_PAID_INFRASTRUCTURE_REMOTE_MODULE_IMPORT_ROOT_RESOLUTION"
                ),
                "status": {
                    "artifact": coordinator.ATTEMPT_B_STATUS_REFERENCE,
                    "sha256": coordinator.ATTEMPT_B_STATUS_SHA256,
                },
                "provider_cleanup": {
                    "artifact": coordinator.ATTEMPT_B_PROVIDER_REFERENCE,
                    "sha256": coordinator.ATTEMPT_B_PROVIDER_SHA256,
                },
                "conservative_charge": {
                    "artifact": coordinator.ATTEMPT_B_CHARGE_REFERENCE,
                    "sha256": coordinator.ATTEMPT_B_CHARGE_SHA256,
                    "charged_gpu_seconds": 1320.0,
                },
                "scientific_arm_executed": False,
            },
            "prior_attempt_c": {
                "attempt_id": coordinator.ATTEMPT_C_ID,
                "failure_classification": (
                    "INVALID_KILL_BRANCH_HISTORY_LIVE_KV_BOUNDARY_PRE_DISCARD"
                ),
                "status": {
                    "artifact": coordinator.ATTEMPT_C_STATUS_REFERENCE,
                    "sha256": coordinator.ATTEMPT_C_STATUS_SHA256,
                },
                "provider_cleanup": {
                    "artifact": coordinator.ATTEMPT_C_PROVIDER_REFERENCE,
                    "sha256": coordinator.ATTEMPT_C_PROVIDER_SHA256,
                },
                "provider_inventory_capture": {
                    "artifact": coordinator.ATTEMPT_C_PROVIDER_INVENTORY_REFERENCE,
                    "sha256": coordinator.ATTEMPT_C_PROVIDER_INVENTORY_SHA256,
                },
                "remote_manifest": {
                    "artifact": coordinator.ATTEMPT_C_MANIFEST_REFERENCE,
                    "sha256": coordinator.ATTEMPT_C_MANIFEST_SHA256,
                    "verified_artifact_count": 478,
                },
                "conservative_charge": {
                    "artifact": coordinator.ATTEMPT_C_CHARGE_REFERENCE,
                    "sha256": coordinator.ATTEMPT_C_CHARGE_SHA256,
                    "charged_gpu_seconds": 1320.0,
                },
                "forensic_review": {
                    "artifact": coordinator.ATTEMPT_C_REVIEW_REFERENCE,
                    "sha256": coordinator.ATTEMPT_C_REVIEW_SHA256,
                    "classification": "KILL_HISTORY_GATE_FIELD_CONFLATION_BUG",
                },
                "destructive_discard_reached": False,
                "valid_measurement_materialized": False,
            },
            "prior_attempt_d": {
                "attempt_id": coordinator.ATTEMPT_D_ID,
                "failure_classification": (
                    "INVALID_PRE_RESERVATION_HISTORICAL_C_REMOTE_PREFIX_POLICY_BUG"
                ),
                "status": {
                    "artifact": coordinator.ATTEMPT_D_STATUS_REFERENCE,
                    "sha256": coordinator.ATTEMPT_D_STATUS_SHA256,
                },
                "terminal_capture": {
                    "artifact": coordinator.ATTEMPT_D_CAPTURE_REFERENCE,
                    "sha256": coordinator.ATTEMPT_D_CAPTURE_SHA256,
                },
                "ledger_unchanged": True,
                "consumed_authorized_arm": False,
            },
            "prior_attempt_e": {
                "attempt_id": coordinator.ATTEMPT_E_ID,
                "failure_classification": (
                    "INVALID_CLIENT_FUNCTION_CALL_POLL_TIMEOUT_CANCELED_REMOTE_STARTUP"
                ),
                "status": {
                    "artifact": coordinator.ATTEMPT_E_STATUS_REFERENCE,
                    "sha256": coordinator.ATTEMPT_E_STATUS_SHA256,
                },
                "provider_cleanup": {
                    "artifact": coordinator.ATTEMPT_E_PROVIDER_REFERENCE,
                    "sha256": coordinator.ATTEMPT_E_PROVIDER_SHA256,
                },
                "provider_inventory_capture": {
                    "artifact": coordinator.ATTEMPT_E_PROVIDER_INVENTORY_REFERENCE,
                    "sha256": coordinator.ATTEMPT_E_PROVIDER_INVENTORY_SHA256,
                },
                "remote_manifest": {
                    "artifact": coordinator.ATTEMPT_E_MANIFEST_REFERENCE,
                    "sha256": coordinator.ATTEMPT_E_MANIFEST_SHA256,
                    "verified_artifact_count": 16,
                },
                "conservative_charge": {
                    "artifact": coordinator.ATTEMPT_E_CHARGE_REFERENCE,
                    "sha256": coordinator.ATTEMPT_E_CHARGE_SHA256,
                    "charged_gpu_seconds": 1320.0,
                },
                "function_call_reconciliation": {
                    "artifact": coordinator.ATTEMPT_E_RECONCILIATION_REFERENCE,
                    "sha256": coordinator.ATTEMPT_E_RECONCILIATION_SHA256,
                    "function_call_id": "fc-01M152JAGCBMQSBQ1TD55Y9HXV",
                    "reconciliation_result": "REMOTE_ERROR_EMPTY_MESSAGE",
                },
                "remote_function_entered": True,
                "scientific_arm_executed": False,
                "destructive_discard_reached": False,
                "valid_measurement_materialized": False,
            },
            "prior_attempt_f": {
                "attempt_id": coordinator.ATTEMPT_F_ID,
                "failure_classification": "INVALID_SCIENTIFIC_CONTROL_STABILITY_GATE",
                "status": {
                    "artifact": coordinator.ATTEMPT_F_STATUS_REFERENCE,
                    "sha256": coordinator.ATTEMPT_F_STATUS_SHA256,
                },
                "provider_cleanup": {
                    "artifact": coordinator.ATTEMPT_F_PROVIDER_REFERENCE,
                    "sha256": coordinator.ATTEMPT_F_PROVIDER_SHA256,
                },
                "provider_inventory_capture": {
                    "artifact": coordinator.ATTEMPT_F_PROVIDER_INVENTORY_REFERENCE,
                    "sha256": coordinator.ATTEMPT_F_PROVIDER_INVENTORY_SHA256,
                },
                "remote_manifest": {
                    "artifact": coordinator.ATTEMPT_F_MANIFEST_REFERENCE,
                    "sha256": coordinator.ATTEMPT_F_MANIFEST_SHA256,
                    "verified_artifact_count": 508,
                },
                "conservative_charge": {
                    "artifact": coordinator.ATTEMPT_F_CHARGE_REFERENCE,
                    "sha256": coordinator.ATTEMPT_F_CHARGE_SHA256,
                    "charged_gpu_seconds": 1320.0,
                },
                "strict_nine_rps_control_interval_passed": False,
                "partial_diagnostic_recompute_evidence_materialized": True,
                "valid_measurement_materialized": False,
            },
            "continuation_authorization": {
                "artifact": coordinator.CONTINUATION_AUTHORIZATION_REFERENCE,
                "sha256": coordinator.CONTINUATION_AUTHORIZATION_SHA256,
                "authorized_attempt_id": coordinator.ATTEMPT_ID,
                "maximum_fresh_invocations": 1,
            },
            "control_gate_policy_review_requirement": {
                "artifact": coordinator.CONTROL_GATE_POLICY_REVIEW_REFERENCE,
                "required_schema": (
                    "sloforge.branchfabric.experiment-004-v11-attempt-f-"
                    "kill-control-gate-methodology-review/v1"
                ),
                "required_status": "PASS",
                "required_before": "MAKE_CHECK_AND_SEAL",
                "content_hash_must_be_bound_by_final_seal": True,
            },
            "source_bindings": {
                label: {"artifact": reference, "sha256": _sha(ROOT / reference)}
                for label, reference in coordinator.SOURCE_BINDINGS.items()
            },
            "preimport_bindings": coordinator._validate_preimport_closure(ROOT),
            "freshness_contract": {
                "local_final_absent": True,
                "local_function_call_evidence_absent": True,
                "remote_final_staging_inflight_absent_before_reservation": True,
                "maximum_active_reservations": 1,
                "exactly_once_spawn": True,
                "minimum_remaining_gpu_seconds_with_15_percent_margin": 1518.0,
                "actual_remaining_gpu_seconds": remaining,
            },
        }
        seal_path = temporary_root / "seal.json"
        seal_path.write_text(json.dumps(seal, indent=2) + "\n")
        coordinator.SEAL_PATH = seal_path
        coordinator._load_kill_config(config_path, repository_root=ROOT, ledger_path=ledger_path)
    finally:
        shutil.rmtree(temporary_root)


def test_attempt_g_real_loader_accepts_exact_c_d_e_f_evidence_and_continuation() -> None:
    _run_g_loader_fixture(tamper=None)


@pytest.mark.parametrize(
    "tamper",
    (
        "status-lifecycle",
        "provider-active",
        "provider-inventory",
        "charge",
        "review",
        "manifest-hash",
        "d-status",
        "d-capture",
        "f-status",
        "f-provider",
        "f-provider-inventory",
        "f-charge",
        "f-manifest-hash",
        "continuation-binding",
        "continuation-d-binding",
        "continuation-f-binding",
        "continuation-budget",
        "continuation-scope",
        "continuation-policy",
    ),
)
def test_attempt_g_real_loader_rejects_resealed_history_or_continuation_tamper(
    tamper: str,
) -> None:
    with pytest.raises(RuntimeError):
        _run_g_loader_fixture(tamper=tamper)


@pytest.mark.parametrize(
    "tamper",
    (
        "duplicate-b-charge",
        "duplicate-c-charge",
        "duplicate-e-charge",
        "duplicate-f-charge",
        "b-interval",
        "c-interval",
        "d-interval",
        "e-interval",
        "f-interval",
        "g-interval",
        "a-charge",
        "d-charge",
        "g-charge",
        "reservation",
        "c-config",
        "c-failure-binding",
        "f-config",
        "f-failure-binding",
    ),
)
def test_attempt_g_ledger_history_rejects_prior_or_current_reuse(tamper: str) -> None:
    coordinator = _coordinator()
    ledger = json.loads(_attempt_g_prelaunch_ledger_path().read_text())
    assert coordinator._kill_attempt_history_is_valid(ledger)
    charges = ledger["conservative_failure_charges"]
    b_row = next(row for row in charges if row.get("invocation_id") == coordinator.ATTEMPT_B_ID)
    c_row = next(row for row in charges if row.get("invocation_id") == coordinator.ATTEMPT_C_ID)
    e_row = next(row for row in charges if row.get("invocation_id") == coordinator.ATTEMPT_E_ID)
    f_row = next(row for row in charges if row.get("invocation_id") == coordinator.ATTEMPT_F_ID)
    if tamper == "duplicate-b-charge":
        charges.append(copy.deepcopy(b_row))
    elif tamper == "duplicate-c-charge":
        charges.append(copy.deepcopy(c_row))
    elif tamper == "duplicate-e-charge":
        charges.append(copy.deepcopy(e_row))
    elif tamper == "duplicate-f-charge":
        charges.append(copy.deepcopy(f_row))
    elif tamper in {
        "b-interval",
        "c-interval",
        "d-interval",
        "e-interval",
        "f-interval",
        "g-interval",
    }:
        attempt = {
            "b-interval": coordinator.ATTEMPT_B_ID,
            "c-interval": coordinator.ATTEMPT_C_ID,
            "d-interval": coordinator.ATTEMPT_D_ID,
            "e-interval": coordinator.ATTEMPT_E_ID,
            "f-interval": coordinator.ATTEMPT_F_ID,
            "g-interval": coordinator.ATTEMPT_ID,
        }[tamper]
        ledger["intervals"].append({"invocation_id": attempt})
    elif tamper in {"a-charge", "d-charge", "g-charge"}:
        row = copy.deepcopy(c_row)
        row["invocation_id"] = {
            "a-charge": coordinator.ATTEMPT_A_ID,
            "d-charge": coordinator.ATTEMPT_D_ID,
            "g-charge": coordinator.ATTEMPT_ID,
        }[tamper]
        charges.append(row)
    elif tamper == "reservation":
        ledger["reservations"] = [
            {"invocation_id": coordinator.ATTEMPT_ID, "reservation_id": "duplicate"}
        ]
    elif tamper == "c-config":
        c_row["config_sha256"] = "0" * 64
    elif tamper == "c-failure-binding":
        c_row["failure_evidence"]["artifact_sha256"] = "0" * 64
    elif tamper == "f-config":
        f_row["config_sha256"] = "0" * 64
    else:
        f_row["failure_evidence"]["artifact_sha256"] = "0" * 64
    assert not coordinator._kill_attempt_history_is_valid(ledger)


def _volume_row(path: str, *, kind: str = "dir") -> dict[str, str]:
    return {
        "filename": path,
        "type": kind,
        "created_modified": "2026-08-27 21:00 CDT",
        "size": "0 B",
    }


def _inventory_runner(payloads: list[tuple[int, Any, str]], calls: list[list[str]]) -> Any:
    responses = iter(payloads)

    def run(command: list[str], **kwargs: Any) -> SimpleNamespace:
        assert kwargs == {
            "check": False,
            "capture_output": True,
            "text": True,
            "timeout": 60.0,
        }
        calls.append(command)
        returncode, stdout, stderr = next(responses)
        return SimpleNamespace(
            returncode=returncode,
            stdout=stdout if isinstance(stdout, str) else json.dumps(stdout),
            stderr=stderr,
        )

    return run


def test_hierarchical_remote_inventory_rejects_absent_required_hierarchy() -> None:
    coordinator = _coordinator()
    calls: list[list[str]] = []
    runner = _inventory_runner([(0, [_volume_row("experiment-004/v11/integrated")], "")], calls)
    with pytest.raises(FileNotFoundError, match="historical kill/recompute hierarchy is absent"):
        coordinator._require_hierarchical_remote_attempt_prefix_fresh(
            SimpleNamespace(attempt_id=coordinator.ATTEMPT_ID), runner=runner
        )
    assert calls == [
        [
            "modal",
            "volume",
            "ls",
            coordinator.RESULTS_VOLUME,
            "experiment-004/v11",
            "--json",
        ]
    ]


def test_hierarchical_remote_inventory_descends_only_through_exact_directories() -> None:
    coordinator = _coordinator()
    calls: list[list[str]] = []
    runner = _inventory_runner(
        [
            (0, [_volume_row("experiment-004/v11/kill-recompute")], ""),
            (0, [_volume_row("experiment-004/v11/kill-recompute/modal")], ""),
            (
                0,
                [
                    _volume_row(f"{coordinator.REMOTE_PREFIX}/{coordinator.ATTEMPT_C_ID}"),
                    _volume_row(f"{coordinator.REMOTE_PREFIX}/{coordinator.ATTEMPT_E_ID}"),
                    _volume_row(f"{coordinator.REMOTE_PREFIX}/{coordinator.ATTEMPT_F_ID}"),
                ],
                "",
            ),
        ],
        calls,
    )
    result = coordinator._require_hierarchical_remote_attempt_prefix_fresh(
        SimpleNamespace(attempt_id=coordinator.ATTEMPT_ID), runner=runner
    )
    assert result["schema_version"] == "sloforge.branchfabric.remote-prefix-freshness/v5"
    assert result["missing_expected_count"] == 0
    assert len(result["inventory_steps"]) == 3
    assert [call[4] for call in calls] == [
        "experiment-004/v11",
        "experiment-004/v11/kill-recompute",
        "experiment-004/v11/kill-recompute/modal",
    ]


@pytest.mark.parametrize(
    ("attempt_attribute", "suffix"),
    (
        ("ATTEMPT_ID", ""),
        ("ATTEMPT_ID", ".staging"),
        ("ATTEMPT_ID", ".inflight"),
        ("ATTEMPT_A_ID", ""),
        ("ATTEMPT_A_ID", ".staging"),
        ("ATTEMPT_A_ID", ".inflight"),
        ("ATTEMPT_B_ID", ""),
        ("ATTEMPT_B_ID", ".staging"),
        ("ATTEMPT_B_ID", ".inflight"),
        ("ATTEMPT_C_ID", ".staging"),
        ("ATTEMPT_C_ID", ".inflight"),
        ("ATTEMPT_D_ID", ""),
        ("ATTEMPT_D_ID", ".staging"),
        ("ATTEMPT_D_ID", ".inflight"),
        ("ATTEMPT_E_ID", ".staging"),
        ("ATTEMPT_E_ID", ".inflight"),
        ("ATTEMPT_F_ID", ".staging"),
        ("ATTEMPT_F_ID", ".inflight"),
    ),
)
def test_hierarchical_remote_inventory_rejects_attempt_collisions(
    attempt_attribute: str, suffix: str
) -> None:
    coordinator = _coordinator()
    attempt_path = f"{coordinator.REMOTE_PREFIX}/{getattr(coordinator, attempt_attribute)}{suffix}"
    runner = _inventory_runner(
        [
            (0, [_volume_row("experiment-004/v11/kill-recompute")], ""),
            (0, [_volume_row("experiment-004/v11/kill-recompute/modal")], ""),
            (0, [_volume_row(attempt_path)], ""),
        ],
        [],
    )
    with pytest.raises(FileExistsError, match="remote prefix exists"):
        coordinator._require_hierarchical_remote_attempt_prefix_fresh(
            SimpleNamespace(attempt_id=coordinator.ATTEMPT_ID), runner=runner
        )


def test_hierarchical_remote_inventory_allows_only_exact_bound_c_e_f_directories() -> None:
    coordinator = _coordinator()
    calls: list[list[str]] = []
    historical = f"{coordinator.REMOTE_PREFIX}/{coordinator.ATTEMPT_C_ID}"
    interrupted = f"{coordinator.REMOTE_PREFIX}/{coordinator.ATTEMPT_E_ID}"
    control_failed = f"{coordinator.REMOTE_PREFIX}/{coordinator.ATTEMPT_F_ID}"
    result = coordinator._require_hierarchical_remote_attempt_prefix_fresh(
        SimpleNamespace(attempt_id=coordinator.ATTEMPT_ID),
        runner=_inventory_runner(
            [
                (0, [_volume_row("experiment-004/v11/kill-recompute")], ""),
                (0, [_volume_row("experiment-004/v11/kill-recompute/modal")], ""),
                (
                    0,
                    [
                        _volume_row(historical),
                        _volume_row(interrupted),
                        _volume_row(control_failed),
                    ],
                    "",
                ),
            ],
            calls,
        ),
    )
    assert result["schema_version"] == "sloforge.branchfabric.remote-prefix-freshness/v5"
    assert result["collision_count"] == 0
    assert result["expected_present_paths"] == [historical, interrupted, control_failed]
    assert result["observed_expected_paths"] == [historical, interrupted, control_failed]
    assert result["missing_expected_count"] == 0
    assert result["allowed_historical_prefixes"] == [
        {
            "attempt_id": coordinator.ATTEMPT_C_ID,
            "path": historical,
            "type": "dir",
            "manifest": {
                "artifact": coordinator.ATTEMPT_C_MANIFEST_REFERENCE,
                "sha256": coordinator.ATTEMPT_C_MANIFEST_SHA256,
                "verified_artifact_count": 478,
            },
        },
        {
            "attempt_id": coordinator.ATTEMPT_E_ID,
            "path": interrupted,
            "type": "dir",
            "manifest": {
                "artifact": coordinator.ATTEMPT_E_MANIFEST_REFERENCE,
                "sha256": coordinator.ATTEMPT_E_MANIFEST_SHA256,
                "verified_artifact_count": 16,
            },
        },
        {
            "attempt_id": coordinator.ATTEMPT_F_ID,
            "path": control_failed,
            "type": "dir",
            "manifest": {
                "artifact": coordinator.ATTEMPT_F_MANIFEST_REFERENCE,
                "sha256": coordinator.ATTEMPT_F_MANIFEST_SHA256,
                "verified_artifact_count": 508,
            },
        },
    ]
    assert result["historical_contents_used_for_current_attempt"] is False
    assert historical not in result["forbidden_paths"]
    assert f"{historical}.staging" in result["forbidden_paths"]
    assert f"{historical}.inflight" in result["forbidden_paths"]
    assert f"{interrupted}.staging" in result["forbidden_paths"]
    assert f"{interrupted}.inflight" in result["forbidden_paths"]
    assert f"{control_failed}.staging" in result["forbidden_paths"]
    assert f"{control_failed}.inflight" in result["forbidden_paths"]
    # C's downloaded 478-file manifest is the immutable evidence. Freshness is
    # namespace-local, so the coordinator must not traverse or select on C's
    # historical descendants before authorizing a distinct attempt namespace.
    assert [call[4] for call in calls] == [
        "experiment-004/v11",
        "experiment-004/v11/kill-recompute",
        coordinator.REMOTE_PREFIX,
    ]


def test_hierarchical_remote_inventory_rejects_historical_c_file_substitution() -> None:
    coordinator = _coordinator()
    historical = f"{coordinator.REMOTE_PREFIX}/{coordinator.ATTEMPT_C_ID}"
    with pytest.raises(ValueError, match="historical Attempt-C prefix is not a directory"):
        coordinator._require_hierarchical_remote_attempt_prefix_fresh(
            SimpleNamespace(attempt_id=coordinator.ATTEMPT_ID),
            runner=_inventory_runner(
                [
                    (0, [_volume_row("experiment-004/v11/kill-recompute")], ""),
                    (0, [_volume_row("experiment-004/v11/kill-recompute/modal")], ""),
                    (0, [_volume_row(historical, kind="file")], ""),
                ],
                [],
            ),
        )


def test_hierarchical_remote_inventory_rejects_missing_historical_c_leaf() -> None:
    coordinator = _coordinator()
    with pytest.raises(FileNotFoundError, match="historical Attempt-C prefix is absent"):
        coordinator._require_hierarchical_remote_attempt_prefix_fresh(
            SimpleNamespace(attempt_id=coordinator.ATTEMPT_ID),
            runner=_inventory_runner(
                [
                    (0, [_volume_row("experiment-004/v11/kill-recompute")], ""),
                    (0, [_volume_row("experiment-004/v11/kill-recompute/modal")], ""),
                    (
                        0,
                        [_volume_row(f"{coordinator.REMOTE_PREFIX}/{coordinator.ATTEMPT_E_ID}")],
                        "",
                    ),
                ],
                [],
            ),
        )


def test_hierarchical_remote_inventory_rejects_missing_interrupted_e_leaf() -> None:
    coordinator = _coordinator()
    historical = f"{coordinator.REMOTE_PREFIX}/{coordinator.ATTEMPT_C_ID}"
    with pytest.raises(FileNotFoundError, match="historical Attempt-E prefix is absent"):
        coordinator._require_hierarchical_remote_attempt_prefix_fresh(
            SimpleNamespace(attempt_id=coordinator.ATTEMPT_ID),
            runner=_inventory_runner(
                [
                    (0, [_volume_row("experiment-004/v11/kill-recompute")], ""),
                    (0, [_volume_row("experiment-004/v11/kill-recompute/modal")], ""),
                    (0, [_volume_row(historical)], ""),
                ],
                [],
            ),
        )


def test_hierarchical_remote_inventory_rejects_missing_control_failed_f_leaf() -> None:
    coordinator = _coordinator()
    historical = f"{coordinator.REMOTE_PREFIX}/{coordinator.ATTEMPT_C_ID}"
    interrupted = f"{coordinator.REMOTE_PREFIX}/{coordinator.ATTEMPT_E_ID}"
    with pytest.raises(FileNotFoundError, match="historical Attempt-F prefix is absent"):
        coordinator._require_hierarchical_remote_attempt_prefix_fresh(
            SimpleNamespace(attempt_id=coordinator.ATTEMPT_ID),
            runner=_inventory_runner(
                [
                    (0, [_volume_row("experiment-004/v11/kill-recompute")], ""),
                    (0, [_volume_row("experiment-004/v11/kill-recompute/modal")], ""),
                    (0, [_volume_row(historical), _volume_row(interrupted)], ""),
                ],
                [],
            ),
        )


def test_hierarchical_remote_inventory_rejects_control_failed_f_file_substitution() -> None:
    coordinator = _coordinator()
    historical = f"{coordinator.REMOTE_PREFIX}/{coordinator.ATTEMPT_C_ID}"
    interrupted = f"{coordinator.REMOTE_PREFIX}/{coordinator.ATTEMPT_E_ID}"
    control_failed = f"{coordinator.REMOTE_PREFIX}/{coordinator.ATTEMPT_F_ID}"
    with pytest.raises(ValueError, match="historical Attempt-F prefix is not a directory"):
        coordinator._require_hierarchical_remote_attempt_prefix_fresh(
            SimpleNamespace(attempt_id=coordinator.ATTEMPT_ID),
            runner=_inventory_runner(
                [
                    (0, [_volume_row("experiment-004/v11/kill-recompute")], ""),
                    (0, [_volume_row("experiment-004/v11/kill-recompute/modal")], ""),
                    (
                        0,
                        [
                            _volume_row(historical),
                            _volume_row(interrupted),
                            _volume_row(control_failed, kind="file"),
                        ],
                        "",
                    ),
                ],
                [],
            ),
        )


def test_hierarchical_remote_inventory_rejects_noncanonical_c_descendant_row() -> None:
    coordinator = _coordinator()
    historical = f"{coordinator.REMOTE_PREFIX}/{coordinator.ATTEMPT_C_ID}"
    with pytest.raises(ValueError, match="paths are invalid"):
        coordinator._require_hierarchical_remote_attempt_prefix_fresh(
            SimpleNamespace(attempt_id=coordinator.ATTEMPT_ID),
            runner=_inventory_runner(
                [
                    (0, [_volume_row("experiment-004/v11/kill-recompute")], ""),
                    (0, [_volume_row("experiment-004/v11/kill-recompute/modal")], ""),
                    (0, [_volume_row(f"{historical}/unexpected")], ""),
                ],
                [],
            ),
        )


def test_hierarchical_remote_inventory_rejects_attempt_identity_substitution() -> None:
    coordinator = _coordinator()
    with pytest.raises(ValueError, match="freshness attempt identity changed"):
        coordinator._require_hierarchical_remote_attempt_prefix_fresh(
            SimpleNamespace(attempt_id=coordinator.ATTEMPT_D_ID),
            runner=_inventory_runner([], []),
        )


def test_hierarchical_remote_inventory_rejects_descendants_at_attempt_ancestor() -> None:
    coordinator = _coordinator()
    attempt_path = f"{coordinator.REMOTE_PREFIX}/{coordinator.ATTEMPT_ID}"
    runner = _inventory_runner(
        [
            (0, [_volume_row("experiment-004/v11/kill-recompute")], ""),
            (0, [_volume_row("experiment-004/v11/kill-recompute/modal")], ""),
            (0, [_volume_row(attempt_path, kind="dir")], ""),
        ],
        [],
    )
    with pytest.raises(FileExistsError, match=coordinator.ATTEMPT_ID):
        coordinator._require_hierarchical_remote_attempt_prefix_fresh(
            SimpleNamespace(attempt_id=coordinator.ATTEMPT_ID), runner=runner
        )


def test_hierarchical_remote_inventory_allows_prefix_neighbors() -> None:
    coordinator = _coordinator()
    neighbor = f"{coordinator.REMOTE_PREFIX}/{coordinator.ATTEMPT_ID}-neighbor"
    historical = f"{coordinator.REMOTE_PREFIX}/{coordinator.ATTEMPT_C_ID}"
    interrupted = f"{coordinator.REMOTE_PREFIX}/{coordinator.ATTEMPT_E_ID}"
    control_failed = f"{coordinator.REMOTE_PREFIX}/{coordinator.ATTEMPT_F_ID}"
    runner = _inventory_runner(
        [
            (0, [_volume_row("experiment-004/v11/kill-recompute")], ""),
            (0, [_volume_row("experiment-004/v11/kill-recompute/modal")], ""),
            (
                0,
                [
                    _volume_row(historical),
                    _volume_row(interrupted),
                    _volume_row(control_failed),
                    _volume_row(neighbor),
                ],
                "",
            ),
        ],
        [],
    )
    result = coordinator._require_hierarchical_remote_attempt_prefix_fresh(
        SimpleNamespace(attempt_id=coordinator.ATTEMPT_ID), runner=runner
    )
    assert result["collision_count"] == 0
    assert result["observed_expected_paths"] == [historical, interrupted, control_failed]


def test_hierarchical_remote_inventory_rejects_missing_second_component() -> None:
    coordinator = _coordinator()
    calls: list[list[str]] = []
    with pytest.raises(FileNotFoundError, match="historical kill/recompute hierarchy is absent"):
        coordinator._require_hierarchical_remote_attempt_prefix_fresh(
            SimpleNamespace(attempt_id=coordinator.ATTEMPT_ID),
            runner=_inventory_runner(
                [
                    (0, [_volume_row("experiment-004/v11/kill-recompute")], ""),
                    (0, [_volume_row("experiment-004/v11/kill-recompute/history")], ""),
                ],
                calls,
            ),
        )
    assert [call[4] for call in calls] == [
        "experiment-004/v11",
        "experiment-004/v11/kill-recompute",
    ]


@pytest.mark.parametrize("level", (0, 1, 2))
def test_hierarchical_remote_inventory_rejects_nonzero_at_every_listed_level(
    level: int,
) -> None:
    coordinator = _coordinator()
    success_rows = [
        (0, [_volume_row("experiment-004/v11/kill-recompute")], ""),
        (0, [_volume_row("experiment-004/v11/kill-recompute/modal")], ""),
    ]
    payloads = [*success_rows[:level], (1, "", "No such file or directory")]
    with pytest.raises(RuntimeError, match="existing-parent inventory failed"):
        coordinator._require_hierarchical_remote_attempt_prefix_fresh(
            SimpleNamespace(attempt_id=coordinator.ATTEMPT_ID),
            runner=_inventory_runner(payloads, []),
        )


@pytest.mark.parametrize(
    ("payloads", "message"),
    (
        ([(1, "", "unexpected provider error")], "existing-parent inventory failed"),
        ([(0, "not-json", "")], "inventory is not JSON"),
        ([(0, [{"Filename": "experiment-004/v11/kill-recompute"}], "")], "schema"),
        (
            [(0, [_volume_row("/experiment-004/v11/kill-recompute")], "")],
            "paths are invalid",
        ),
        (
            [(0, [_volume_row("experiment-004/v11/kill-recompute/descendant")], "")],
            "paths are invalid",
        ),
        (
            [
                (
                    0,
                    [
                        _volume_row("experiment-004/v11/kill-recompute"),
                        _volume_row("experiment-004/v11/kill-recompute"),
                    ],
                    "",
                )
            ],
            "duplicated",
        ),
        (
            [
                (
                    0,
                    [
                        {
                            **_volume_row("experiment-004/v11/kill-recompute"),
                            "size": 0,
                        }
                    ],
                    "",
                )
            ],
            "schema",
        ),
        (
            [
                (
                    0,
                    [
                        {
                            **_volume_row("experiment-004/v11/kill-recompute"),
                            "created_modified": "",
                        }
                    ],
                    "",
                )
            ],
            "schema",
        ),
    ),
)
def test_hierarchical_remote_inventory_fails_closed_on_errors_and_malformed_rows(
    payloads: list[tuple[int, Any, str]], message: str
) -> None:
    coordinator = _coordinator()
    with pytest.raises((RuntimeError, ValueError), match=message):
        coordinator._require_hierarchical_remote_attempt_prefix_fresh(
            SimpleNamespace(attempt_id=coordinator.ATTEMPT_ID),
            runner=_inventory_runner(payloads, []),
        )


@pytest.mark.parametrize("level", (0, 1))
def test_hierarchical_remote_inventory_rejects_file_where_directory_is_required(
    level: int,
) -> None:
    coordinator = _coordinator()
    payloads = (
        [(0, [_volume_row("experiment-004/v11/kill-recompute", kind="file")], "")]
        if level == 0
        else [
            (0, [_volume_row("experiment-004/v11/kill-recompute")], ""),
            (0, [_volume_row("experiment-004/v11/kill-recompute/modal", kind="file")], ""),
        ]
    )
    with pytest.raises(ValueError, match="not a directory"):
        coordinator._require_hierarchical_remote_attempt_prefix_fresh(
            SimpleNamespace(attempt_id=coordinator.ATTEMPT_ID),
            runner=_inventory_runner(payloads, []),
        )


def test_kill_install_uses_hierarchical_freshness_before_prebuild_and_reservation() -> None:
    coordinator = _coordinator()
    base = SimpleNamespace()
    coordinator._install(base)
    assert (
        base._require_remote_attempt_prefix_fresh
        is coordinator._require_hierarchical_remote_attempt_prefix_fresh
    )
    source = (ROOT / "tools/branchfabric-experiment-004-v11-final.py").read_text()
    freshness = source.index("remote_prefix_freshness = _require_remote_attempt_prefix_fresh")
    prebuild = source.index("prebuild = _prebuild_modal_image", freshness)
    second_freshness = source.index(
        "post_prebuild_remote_prefix_freshness = _require_remote_attempt_prefix_fresh",
        prebuild,
    )
    reservation = source.index("_ledger, preflight, commitment = _reserve_locked", prebuild)
    invocation = source.index("completed, elapsed = _invoke_modal", reservation)
    assert freshness < prebuild < second_freshness < reservation < invocation


def test_post_prebuild_collision_aborts_before_reservation_and_invoke(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    coordinator = _coordinator()
    base = coordinator._BASE
    coordinator._install(base)
    config_path = tmp_path / "config.json"
    config_path.write_text("{}\n")
    config = Experiment004V11KillRecomputeConfig.model_validate(_config_payload(), strict=True)
    seal = {"verified": True}
    calls = {"freshness": 0, "prebuild": 0, "reserve": 0, "invoke": 0}
    monkeypatch.setattr(base, "_ROOT", tmp_path)
    monkeypatch.setattr(base, "_EXPERIMENT_ROOT", tmp_path / "experiment")
    monkeypatch.setattr(base, "_LOCK", tmp_path / "gpu-hours.lock")
    monkeypatch.setattr(base, "_LEDGER", tmp_path / "gpu-hours.json")
    monkeypatch.setattr(base, "_LOCAL_RESULT_PARENT", tmp_path / "kill/raw")
    monkeypatch.setattr(base, "_FAILURE_ROOT", tmp_path / "kill/failures")
    monkeypatch.setattr(base, "_PROVIDER_INPUTS_PARENT", tmp_path / "kill")
    monkeypatch.setattr(base, "_PREBUILD_ROOT", tmp_path / "prebuild")
    monkeypatch.setattr(base, "_load_sealed_config", lambda *args, **kwargs: (config, seal))
    monkeypatch.setattr(base, "_require_launch_budget", lambda *args, **kwargs: 80.0)
    monkeypatch.setattr(base, "_require_local_result_output_fresh", lambda *args: None)
    monkeypatch.setattr(
        base, "_require_integrated_pre_reservation_outputs_fresh", lambda *args, **kwargs: None
    )

    historical = f"{coordinator.REMOTE_PREFIX}/{coordinator.ATTEMPT_C_ID}"
    interrupted = f"{coordinator.REMOTE_PREFIX}/{coordinator.ATTEMPT_E_ID}"
    control_failed = f"{coordinator.REMOTE_PREFIX}/{coordinator.ATTEMPT_F_ID}"
    fresh_attempt = f"{coordinator.REMOTE_PREFIX}/{coordinator.ATTEMPT_ID}"
    inventory_calls: list[list[str]] = []
    inventory_runner = _inventory_runner(
        [
            (0, [_volume_row("experiment-004/v11/kill-recompute")], ""),
            (0, [_volume_row("experiment-004/v11/kill-recompute/modal")], ""),
            (
                0,
                [_volume_row(historical), _volume_row(interrupted), _volume_row(control_failed)],
                "",
            ),
            (0, [_volume_row("experiment-004/v11/kill-recompute")], ""),
            (0, [_volume_row("experiment-004/v11/kill-recompute/modal")], ""),
            (
                0,
                [
                    _volume_row(historical),
                    _volume_row(interrupted),
                    _volume_row(control_failed),
                    _volume_row(fresh_attempt),
                ],
                "",
            ),
        ],
        inventory_calls,
    )

    def fresh(candidate: Any, **kwargs: Any) -> dict[str, Any]:
        assert not kwargs
        calls["freshness"] += 1
        return coordinator._require_hierarchical_remote_attempt_prefix_fresh(
            candidate, runner=inventory_runner
        )

    def prebuild(**kwargs: Any) -> dict[str, Any]:
        calls["prebuild"] += 1
        return {"schema_version": "prebuild-test/v1", "status": "PASS"}

    def reserve(*args: Any, **kwargs: Any) -> None:
        calls["reserve"] += 1
        raise AssertionError("reservation must not be reached")

    def invoke(*args: Any, **kwargs: Any) -> None:
        calls["invoke"] += 1
        raise AssertionError("Modal invocation must not be reached")

    monkeypatch.setattr(base, "_require_remote_attempt_prefix_fresh", fresh)
    monkeypatch.setattr(base, "_prebuild_modal_image", prebuild)
    monkeypatch.setattr(base, "_reserve_locked", reserve)
    monkeypatch.setattr(base, "_invoke_modal", invoke)
    monkeypatch.setattr(sys, "argv", ["kill", "--config-path", str(config_path)])
    with pytest.raises(FileExistsError, match=coordinator.ATTEMPT_ID):
        base.main()
    assert calls == {"freshness": 2, "prebuild": 1, "reserve": 0, "invoke": 0}
    assert len(inventory_calls) == 6


def test_kill_config_requires_exact_40_hex_tag_object() -> None:
    valid = _config_payload()
    Experiment004V11KillRecomputeConfig.model_validate(valid, strict=True)
    for invalid in ("a" * 39, "a" * 41, "a" * 64):
        candidate = {**valid, "optimized_v11_freeze_tag_object": invalid}
        with pytest.raises(ValidationError):
            Experiment004V11KillRecomputeConfig.model_validate(candidate, strict=True)


def test_history_distinguishes_submitted_unique_and_lost_private_work() -> None:
    evidence = worker._history_evidence(
        _live_boundary_tables(), prefix_length=16_384, suffix_length=256
    )
    assert evidence["submitted_replay_tokens"] == 133_128
    assert evidence["unique_submitted_replay_history_tokens"] == 18_440
    assert evidence["computed_state_topology_tokens"] == 18_432
    assert evidence["lost_private_rollout_work_tokens"] == 2_048
    altered = _live_boundary_tables()
    altered[7].token_ids = (
        tuple([9] * 16_384) + tuple([27] * 256) + tuple(altered[7].token_ids[-1:])
    )
    with pytest.raises(RuntimeError, match="share the exact prefix"):
        worker._history_evidence(altered, prefix_length=16_384, suffix_length=256)


def test_attempt_c_live_history_preserves_the_uncomputed_tail_for_replay() -> None:
    evidence = worker._history_evidence(
        _live_boundary_tables(), prefix_length=16_384, suffix_length=256
    )
    assert evidence["schema_version"] == "sloforge.branchfabric.kill-recompute-history/v2"
    assert evidence["computed_boundary_tokens"] == 133_120
    assert evidence["live_uncomputed_tail_tokens"] == 8
    assert evidence["submitted_prefix_tokens"] == 131_072
    assert evidence["submitted_private_tokens"] == 2_056
    assert evidence["submitted_replay_tokens"] == 133_128
    assert evidence["submitted_vs_analytical_delta_tokens"] == 8
    assert evidence["analytical_independent_branch_expectation_tokens"] == 133_120
    assert evidence["unique_submitted_replay_history_tokens"] == 18_440
    assert evidence["computed_state_topology_tokens"] == 18_432
    assert evidence["lost_private_rollout_work_tokens"] == 2_048
    for index, row in enumerate(evidence["branches"]):
        assert row["logical_branch_id"] == f"branch.{index}"
        assert row["computed_tokens"] == 16_640
        assert row["computed_private_tokens"] == 256
        assert row["uncomputed_tail_token_count"] == 1
        assert row["uncomputed_tail_token_ids"] == [30_000 + index]
        assert row["total_tokens_submitted"] == 16_641
        assert len(row["token_ids"]) == 16_641
        assert row["token_ids"][-1] == 30_000 + index


@pytest.mark.parametrize(
    ("computed_tokens", "extra_tail_tokens"),
    ((16_639, 1), (16_641, 0), (16_640, 0), (16_640, 2)),
)
def test_live_history_rejects_wrong_or_multi_token_kv_boundary(
    computed_tokens: int, extra_tail_tokens: int
) -> None:
    tables = _live_boundary_tables()
    table = tables[0]
    base = tuple(table.token_ids[:16_640])
    table.token_ids = base + tuple(range(40_000, 40_000 + extra_tail_tokens))
    table.computed_tokens = computed_tokens
    with pytest.raises(RuntimeError, match="live KV boundary"):
        worker._history_evidence(tables, prefix_length=16_384, suffix_length=256)


def test_attempt_c_exact_postworker_failure_and_cleanup_are_admissible() -> None:
    coordinator = _attempt_c_coordinator()
    bundle = _attempt_c_bundle_root()
    controller_payload = json.loads((bundle / "kill-controller-result.json").read_text())
    cleanup = json.loads((bundle / "in_function_cleanup.json").read_text())
    assert controller_payload["worker_results"] == []
    assert controller_payload["worker_returncodes"] == {"rollout": 1, "serving": -15}
    assert coordinator._post_worker_cleanup_ownership_is_exact(controller_payload, cleanup)
    assert coordinator._post_worker_controller_is_exact(
        controller_payload, cleanup, status="failed"
    )


def test_attempt_c_exact_postworker_failure_reaches_immutable_download_admission() -> None:
    coordinator = _attempt_c_coordinator()
    config = Experiment004V11KillRecomputeConfig.model_validate(_config_payload(), strict=True)
    envelope = _postworker_failure_envelope(
        coordinator,
        config=config,
        reservation_id="exp004-v11-kill-c-failure-replay",
        reservation_commitment="c" * 64,
    )
    controller_payload = json.loads(
        (_attempt_c_bundle_root() / "kill-controller-result.json").read_text()
    )
    envelope["result"]["controller"] = controller_payload
    envelope["result"]["in_function_cleanup"] = controller_payload["in_function_cleanup"]
    _validated, _materialized, inventory = coordinator._validate_kill_remote_result(
        envelope,
        config=config,
        reservation_id="exp004-v11-kill-c-failure-replay",
        reservation_commitment_sha256="c" * 64,
    )
    assert inventory is not None


@pytest.mark.parametrize(
    "mutation",
    (
        lambda row: row["worker_results"].append({"role": "foreign"}),
        lambda row: row["worker_returncodes"].update(rollout=0, serving=0),
        lambda row: row["worker_results"].append(
            {
                "role": "serving",
                "status": "succeeded",
                "attempt_id": "exp004-v11-kill-recompute-s41-c",
                "mode": "KILL_AND_RECOMPUTE",
            }
        ),
        lambda row: row.update(cleanup_actions=[]),
        lambda row: next(
            child
            for child in row["in_function_cleanup"]["owned_children"]
            if child["role"] == "serving" and child["process_kind"] == "worker"
        ).update(exit_status=0),
        lambda row: row["in_function_cleanup"].update(owned_children=[]),
    ),
)
def test_attempt_c_postworker_failure_rejects_partial_result_or_cleanup_tamper(
    mutation: Any,
) -> None:
    coordinator = _attempt_c_coordinator()
    bundle = _attempt_c_bundle_root()
    controller_payload = json.loads((bundle / "kill-controller-result.json").read_text())
    mutation(controller_payload)
    cleanup = controller_payload["in_function_cleanup"]
    assert not coordinator._post_worker_controller_is_exact(
        controller_payload, cleanup, status="failed"
    )


def test_postworker_failure_keeps_results_empty_when_one_worker_exited_cleanly() -> None:
    coordinator = _attempt_c_coordinator()
    controller_payload = json.loads(
        (_attempt_c_bundle_root() / "kill-controller-result.json").read_text()
    )
    controller_payload["worker_returncodes"] = {"rollout": 1, "serving": 0}
    cleanup = controller_payload["in_function_cleanup"]
    cleanup["termination_actions"] = []
    controller_payload["cleanup_actions"] = []
    for child in cleanup["owned_children"]:
        if child["role"] == "serving":
            child["termination_signal"] = None
            if child["process_kind"] == "worker":
                child["exit_status"] = 0
    assert coordinator._post_worker_controller_is_exact(
        controller_payload, cleanup, status="failed"
    )


@pytest.mark.parametrize(
    "mutation",
    (
        lambda row: row["engine_start_evidence"][0].update(pid=999_999),
        lambda row: row["engine_start_evidence"][0].update(physical_gpu_uuid="GPU-fake"),
        lambda row: row["engine_start_evidence"][1].update(role="serving"),
        lambda row: row["readiness_evidence"][0].update(pid=999_999),
        lambda row: row["readiness_evidence"][0].update(physical_gpu_uuid="GPU-fake"),
        lambda row: row["readiness_evidence"][1].update(role="serving"),
        lambda row: row["readiness_evidence"][0].update(queue_empty_pass=False),
        lambda row: row["readiness_evidence"][0]["runtime_state"].update(queue_depth=1),
        lambda row: row["readiness_evidence"][0]["compilation_observation"].update(
            no_active_compilation_event=False
        ),
        lambda row: row["readiness_evidence"][0].update(model_ready_ns=row["ended_ns"] + 1),
        lambda row: row["readiness_evidence"][0].update(junk=True),
    ),
)
def test_attempt_c_postworker_failure_rejects_engine_continuity_tamper(
    mutation: Any,
) -> None:
    coordinator = _attempt_c_coordinator()
    controller_payload = json.loads(
        (_attempt_c_bundle_root() / "kill-controller-result.json").read_text()
    )
    mutation(controller_payload)
    assert not coordinator._post_worker_controller_is_exact(
        controller_payload, controller_payload["in_function_cleanup"], status="failed"
    )


def test_scheduler_counter_accepts_vllm_0230_req_id_and_preserves_raw_rows() -> None:
    request_ids = {f"branch.{index}": f"req-{index}" for index in range(8)}

    class Scheduler:
        def schedule(self) -> Any:
            return SimpleNamespace(
                scheduled_new_reqs=[
                    SimpleNamespace(req_id=request_id, num_computed_tokens=0)
                    for request_id in request_ids.values()
                ],
                num_scheduled_tokens={request_id: 16_640 for request_id in request_ids.values()},
            )

    scheduler = Scheduler()
    counter = worker._SchedulerReplayCounter(scheduler, request_ids)
    with counter:
        scheduler.schedule()
        for logical in request_ids:
            counter.observe_first_token(logical)
    evidence = counter.evidence(submitted_by_branch={logical: 16_640 for logical in request_ids})
    assert evidence["runtime_uncached_scheduled_tokens"] == 133_120
    assert len(evidence["raw_cache_admission_rows"]) == 8
    assert len({row["runtime_request_id"] for row in evidence["raw_cache_admission_rows"]}) == 8


def test_scheduler_counter_rejects_unknown_and_duplicate_admission() -> None:
    class Scheduler:
        def __init__(self) -> None:
            self.rows = [SimpleNamespace(req_id="unknown", num_computed_tokens=0)]

        def schedule(self) -> Any:
            return SimpleNamespace(scheduled_new_reqs=self.rows, num_scheduled_tokens={})

    scheduler = Scheduler()
    counter = worker._SchedulerReplayCounter(scheduler, {"branch.0": "req-0"})
    with counter, pytest.raises(RuntimeError, match="unknown replay request"):
        scheduler.schedule()
    with pytest.raises(RuntimeError, match="identities conflict"):
        worker._SchedulerReplayCounter._request_id(SimpleNamespace(request_id="one", req_id="two"))


def test_kill_calibration_scopes_three_probes_and_restores_on_error() -> None:
    calls: list[Any] = []

    def original_validator(**kwargs: Any) -> tuple[dict[str, Any], dict[str, str]]:
        calls.append(kwargs)
        assert kwargs["command"]["schema_version"] == (
            "sloforge.branchfabric.integrated-transaction-command/v1"
        )
        return kwargs["command"]["effective_config"], {}

    def calibration(**_kwargs: Any) -> Any:
        assert frozen._V10_SANITY_GUARD_COUNT == 3
        command = {
            "schema_version": "sloforge.branchfabric.kill-recompute-transaction-command/v1",
            "effective_config": {"attempt_id": "a"},
            "selection_sha256": "a" * 64,
            "authorization_artifact_hash": "b" * 64,
            "sanity_result_sha256": "c" * 64,
            "issued_at_monotonic_ns": 1,
        }
        frozen._validate_integrated_transaction_command(
            base_config={
                "schema_version": ("sloforge.branchfabric.experiment-004-kill-recompute-config/v1"),
                "mode": "KILL_AND_RECOMPUTE",
                "execution_mode": "integrated-kill-recompute-v11",
            },
            command=command,
            selected_load_sha256="a" * 64,
        )
        raise RuntimeError("fixture-stop")

    frozen = SimpleNamespace(
        _V10_SANITY_GUARD_COUNT=2,
        _validate_integrated_transaction_command=original_validator,
        _run_integrated_calibration_phase=calibration,
    )
    with pytest.raises(RuntimeError, match="fixture-stop"):
        worker._run_kill_calibration_phase(
            frozen_worker=frozen,
            adapter=object(),
            config={},
            inputs={},
            role="serving",
            physical_gpu_uuid="GPU-test",
            barrier_root=Path("unused"),
            model_load_started_ns=1,
            model_ready_ns=2,
        )
    assert frozen._V10_SANITY_GUARD_COUNT == 2
    assert frozen._validate_integrated_transaction_command is original_validator
    assert len(calls) == 1


def test_real_k_bundle_freeze_and_current_empty_ledger_verify() -> None:
    config = Experiment004V11KillRecomputeConfig.model_validate(_config_payload(), strict=True)
    evidence = controller.verify_kill_recompute_evidence(config, ROOT)
    assert evidence["integrated_k_remote_manifest_entries"] == 507
    assert evidence["integrated_k_scientifically_valid"] is True
    assert evidence["optimized_v11_frozen"] is True
    assert evidence["passed"] is True


def test_coordinator_source_uses_kill_specific_reservation_prefix() -> None:
    source = (ROOT / "tools/branchfabric-experiment-004-v11-final.py").read_text()
    assert 'f"exp004-v11-kill-{attempt_suffix}-' in source


def test_preimport_closure_rejects_byte_drift_and_symlink(tmp_path: Path) -> None:
    coordinator = _coordinator()
    source = tmp_path / "source.py"
    source.write_text("value = 1\n")
    binding = {"source": ("source.py", _sha(source))}
    assert coordinator._validate_preimport_closure(tmp_path, binding)["source"]["sha256"]
    source.write_text("value = 2\n")
    with pytest.raises(RuntimeError, match="source changed"):
        coordinator._validate_preimport_closure(tmp_path, binding)
    source.unlink()
    target = tmp_path / "target.py"
    target.write_text("value = 1\n")
    source.symlink_to(target)
    with pytest.raises(RuntimeError, match="not regular"):
        coordinator._validate_preimport_closure(tmp_path, binding)


def test_prebuild_receipts_are_outside_modal_image_input_tree() -> None:
    coordinator = _coordinator()
    coordinator._install(coordinator._BASE)
    assert "v11-final" not in coordinator._BASE._PREBUILD_ROOT.parts
    assert coordinator._BASE._PREBUILD_ROOT.name == "v11-kill-recompute-image-prebuild"


def test_attempt_g_snapshot_is_immutable_and_current_ledger_records_exact_measurement() -> None:
    live = ROOT / "artifacts/branchfabric/gpu-validation/experiment-004/gpu-hours.json"
    attempt_e_snapshot = (
        ROOT / "artifacts/branchfabric/gpu-validation/experiment-004/v11-final/"
        "kill-recompute/authorization/ledger-snapshot-before-attempt-e.json"
    )
    snapshot = (
        ROOT / "artifacts/branchfabric/gpu-validation/experiment-004/v11-final/"
        "kill-recompute/authorization/ledger-snapshot-before-attempt-g.json"
    )
    assert snapshot != live
    assert _sha(attempt_e_snapshot) == (
        "0c4c600b64eb58307758ee782d41f122c90b7610976e208b20a4b4b81ac57d9c"
    )
    assert _sha(snapshot) == "aa81df8d3a26e370af1e715f86cb13a1b670cdccd4bbf6d4707dae1802a4ca26"
    assert _sha(live) == "d6b515388a7ffa9579b28fe957c56e854370497afd60d01742d8a2c94f6c5d24"
    live_payload = json.loads(live.read_text())
    attempt_e_payload = json.loads(attempt_e_snapshot.read_text())
    snapshot_payload = json.loads(snapshot.read_text())
    for attempt in (
        "exp004-v11-kill-recompute-s41-b",
        "exp004-v11-kill-recompute-s41-c",
    ):
        assert (
            sum(
                row.get("invocation_id") == attempt
                for row in attempt_e_payload["conservative_failure_charges"]
            )
            == 1
        )
    for attempt in (
        "exp004-v11-kill-recompute-s41-a",
        "exp004-v11-kill-recompute-s41-d",
        "exp004-v11-kill-recompute-s41-g",
    ):
        assert not any(
            row.get("invocation_id") == attempt
            for row in live_payload["conservative_failure_charges"]
        )
    assert not any(
        row.get("invocation_id") == "exp004-v11-kill-recompute-s41-e"
        for row in attempt_e_payload["conservative_failure_charges"]
    )
    attempt_f_charges = [
        row
        for row in live_payload["conservative_failure_charges"]
        if row.get("invocation_id") == "exp004-v11-kill-recompute-s41-f"
    ]
    assert len(attempt_f_charges) == 1
    assert attempt_f_charges[0]["charged_wall_seconds"] == 660.0
    assert attempt_f_charges[0]["gpu_count"] == 2
    assert attempt_f_charges[0]["failure_evidence"]["artifact_sha256"] == (
        "38905e7cbd92ced0ea523a001954135155c2451140beb861b41a31a417aec174"
    )
    assert not any(
        row.get("invocation_id") == "exp004-v11-kill-recompute-s41-g"
        for row in snapshot_payload["intervals"]
    )
    attempt_g_intervals = [
        row
        for row in live_payload["intervals"]
        if row.get("invocation_id") == "exp004-v11-kill-recompute-s41-g"
    ]
    assert attempt_g_intervals == [
        {
            "accounted_wall_seconds": 279.360269719,
            "actual_gpu_models": [
                "NVIDIA A100-SXM4-80GB",
                "NVIDIA A100-SXM4-80GB",
            ],
            "function_call_id": "fc-01M15EW2B2F21RZWFKMSAZ92FS",
            "gpu_count": 2,
            "gpu_price_per_hour_usd": 2.4984,
            "gpu_uuids": [
                "GPU-e7446667-04f6-a4f6-39e6-46aa0871cf3f",
                "GPU-72c41885-eaa2-00d0-a040-94e0baa07376",
            ],
            "invocation_id": "exp004-v11-kill-recompute-s41-g",
            "raw_manifest": {
                "artifact_reference": str(
                    ROOT / "artifacts/branchfabric/gpu-validation/experiment-004/v11-final/"
                    "kill-recompute/raw/exp004-v11-kill-recompute-s41-g/REMOTE_MANIFEST.json"
                ),
                "artifact_sha256": (
                    "db9d56c305faddca5f24cb411b45d8ec3b57496b1bd56f9f66db544505b43361"
                ),
                "sample_selector": "$",
            },
            "requested_gpu": "A100-80GB",
        }
    ]
    assert not any(
        row.get("invocation_id") == "exp004-v11-kill-recompute-s41-g"
        for row in live_payload["conservative_failure_charges"]
    )
    assert snapshot_payload["reservations"] == []
    assert live_payload["reservations"] == []


def test_remote_runtime_closure_matches_local_execution_projection_and_rejects_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    coordinator = _coordinator()
    modal_worker = _load_experiment_module("modal_gpu_reclamation_kill_recompute_v11")
    local_labels = {
        "kill_controller",
        "kill_worker",
        "frozen_integrated_controller",
        "frozen_integrated_worker",
        "frozen_trigger",
        "frozen_v11_worker",
        "capacity_worker",
        "frozen_worker",
        "frozen_gpu1_serving",
        "frozen_sanity",
        "allocator_epochs",
        "canonical_capture_plan",
        "v11_pipeline",
        "v11_ownership",
        "v11_sync",
        "capacity_model",
        "real_runtime_adapter",
        "vllm_live_adapter",
        "gpu_reclamation_accounting",
        "gpu_reclamation_instrumentation",
        "gpu_reclamation_methodology",
        "vllm_metadata_0230",
        "capacity_controller",
        "v11_accounting",
        "vllm_live_trace",
        "external_adapter",
        "sdk_adapter",
        "continuum_canonical",
        "gpu_reclamation_model",
        "gpu_reclamation_trace",
        "trace_init",
        "trace_adapters",
        "trace_buffer",
        "trace_canonical",
        "trace_conversion",
        "trace_io",
        "trace_manifest",
        "trace_models",
        "trace_parquet",
        "trace_perfetto",
        "v10_authorization",
    }
    expected = {
        coordinator.PREIMPORT_BINDINGS[label][0]: coordinator.PREIMPORT_BINDINGS[label][1]
        for label in local_labels
    }
    assert expected == modal_worker._REMOTE_RUNTIME_BINDINGS

    runtime = tmp_path / "runtime.py"
    runtime.write_text("value = 1\n")
    monkeypatch.setattr(modal_worker, "_REMOTE_RUNTIME_BINDINGS", {"runtime.py": _sha(runtime)})
    modal_worker._validate_remote_runtime_closure(tmp_path)
    runtime.write_text("value = 2\n")
    with pytest.raises(RuntimeError, match="remote runtime changed"):
        modal_worker._validate_remote_runtime_closure(tmp_path)


def test_modal_bootstrap_validates_shallow_remote_layout_before_import(
    tmp_path: Path,
) -> None:
    modal_worker = _load_experiment_module("modal_gpu_reclamation_kill_recompute_v11")
    shallow_root = tmp_path / "root"
    shallow_root.mkdir()
    source = shallow_root / "modal_gpu_reclamation_kill_recompute_v11.py"
    source.write_text("# mounted function module\n")
    remote_repository = tmp_path / "opt/sloforge"
    base_path = (
        remote_repository / "experiments/branchfabric/modal_gpu_reclamation_integrated_v11.py"
    )
    base_path.parent.mkdir(parents=True)
    base_path.write_bytes((EXPERIMENTS / "modal_gpu_reclamation_integrated_v11.py").read_bytes())
    methodology_path = (
        remote_repository
        / "python/sloforge/helix/characterization/gpu_reclamation_v11_methodology.py"
    )
    methodology_path.parent.mkdir(parents=True)
    methodology_path.write_bytes(
        (
            ROOT / "python/sloforge/helix/characterization/gpu_reclamation_v11_methodology.py"
        ).read_bytes()
    )
    repository_root, observed_base, observed_methodology = modal_worker._resolve_bootstrap_paths(
        source, remote_repository_root=remote_repository
    )
    assert repository_root == remote_repository
    assert observed_base == base_path
    assert observed_methodology == methodology_path
    modal_worker._validate_bootstrap_sources(observed_base, observed_methodology)


@pytest.mark.parametrize("target", ("base", "methodology"))
@pytest.mark.parametrize("mutation", ("wrong-hash", "missing", "symlink"))
def test_modal_bootstrap_fails_closed_for_remote_source_tamper(
    tmp_path: Path, target: str, mutation: str
) -> None:
    modal_worker = _load_experiment_module("modal_gpu_reclamation_kill_recompute_v11")
    base_path = (
        tmp_path / "opt/sloforge/experiments/branchfabric/modal_gpu_reclamation_integrated_v11.py"
    )
    base_path.parent.mkdir(parents=True)
    base_path.write_bytes((EXPERIMENTS / "modal_gpu_reclamation_integrated_v11.py").read_bytes())
    methodology_path = (
        tmp_path / "opt/sloforge/python/sloforge/helix/characterization/"
        "gpu_reclamation_v11_methodology.py"
    )
    methodology_path.parent.mkdir(parents=True, exist_ok=True)
    methodology_path.write_bytes(
        (
            ROOT / "python/sloforge/helix/characterization/gpu_reclamation_v11_methodology.py"
        ).read_bytes()
    )
    path = base_path if target == "base" else methodology_path
    if mutation == "wrong-hash":
        path.write_text("changed\n")
    elif mutation == "missing":
        path.unlink()
    else:
        path.unlink()
        symlink_target = tmp_path / f"{target}-target.py"
        symlink_target.write_text("changed\n")
        path.symlink_to(symlink_target)
    with pytest.raises(
        RuntimeError, match=f"{target.replace('base', 'base Modal')} source changed"
    ):
        modal_worker._validate_bootstrap_sources(base_path, methodology_path)


class _FunctionCallReplayFixture:
    def __init__(self, outcomes: list[Any], *, object_id: str = "fc-01M152JAGCBMQSBQ1TD55Y9HXF"):
        self.object_id = object_id
        self._outcomes = iter(outcomes)
        self.timeouts: list[float] = []

    def get(self, *, timeout: float) -> Any:
        self.timeouts.append(timeout)
        outcome = next(self._outcomes)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


def test_function_call_first_timeout_is_persisted_and_same_call_is_repolled(
    tmp_path: Path,
) -> None:
    modal_worker = _load_experiment_module("modal_gpu_reclamation_kill_recompute_v11")
    call = _FunctionCallReplayFixture([TimeoutError(), {"status": "remote-terminal"}])
    result = modal_worker._get_function_call_result_with_repoll(
        call,
        evidence_root=tmp_path,
        attempt_id="exp004-v11-kill-recompute-s41-f",
        reservation_id="exp004-v11-kill-f-fixture",
        reservation_commitment_sha256="a" * 64,
        config_sha256="b" * 64,
        observed_at_utc=lambda: "2026-08-28T21:30:00+00:00",
    )
    assert result == {"status": "remote-terminal"}
    assert call.timeouts == [
        modal_worker.FIRST_FUNCTION_CALL_POLL_TIMEOUT_SECONDS,
        modal_worker.FUNCTION_CALL_REPOLL_TIMEOUT_SECONDS,
    ]
    spawned = json.loads((tmp_path / "function-call.json").read_text())
    timed_out = json.loads((tmp_path / "first-poll-timeout.json").read_text())
    assert spawned["function_call_id"] == call.object_id
    assert spawned["status"] == "SPAWNED_AND_PERSISTED_BEFORE_FIRST_POLL"
    assert timed_out["function_call_id"] == call.object_id
    assert timed_out["status"] == "FIRST_POLL_TIMED_OUT_REPOLLING_SAME_CALL"
    assert timed_out["same_function_call_repolled"] is True
    assert timed_out["error"] == {"type": "TimeoutError", "message": ""}


def test_function_call_immediate_result_does_not_create_timeout_evidence(tmp_path: Path) -> None:
    modal_worker = _load_experiment_module("modal_gpu_reclamation_kill_recompute_v11")
    call = _FunctionCallReplayFixture([{"status": "remote-terminal"}])
    result = modal_worker._get_function_call_result_with_repoll(
        call,
        evidence_root=tmp_path,
        attempt_id="exp004-v11-kill-recompute-s41-f",
        reservation_id="exp004-v11-kill-f-fixture",
        reservation_commitment_sha256="a" * 64,
        config_sha256="b" * 64,
        observed_at_utc=lambda: "2026-08-28T21:30:00+00:00",
    )
    assert result == {"status": "remote-terminal"}
    assert call.timeouts == [modal_worker.FIRST_FUNCTION_CALL_POLL_TIMEOUT_SECONDS]
    assert (tmp_path / "function-call.json").is_file()
    assert not (tmp_path / "first-poll-timeout.json").exists()


@pytest.mark.parametrize(
    "object_id",
    ("", "fc-lowercase000000000000000000", "fc-01M152JAGCBMQSBQ1TD55Y9HXV-extra"),
)
def test_function_call_id_is_validated_before_poll_or_evidence(
    tmp_path: Path, object_id: str
) -> None:
    modal_worker = _load_experiment_module("modal_gpu_reclamation_kill_recompute_v11")
    call = _FunctionCallReplayFixture([{"status": "unused"}], object_id=object_id)
    with pytest.raises(ValueError, match="FunctionCall ID"):
        modal_worker._get_function_call_result_with_repoll(
            call,
            evidence_root=tmp_path,
            attempt_id="exp004-v11-kill-recompute-s41-f",
            reservation_id="exp004-v11-kill-f-fixture",
            reservation_commitment_sha256="a" * 64,
            config_sha256="b" * 64,
        )
    assert call.timeouts == []
    assert list(tmp_path.iterdir()) == []


def test_function_call_second_timeout_propagates_without_changing_call_identity(
    tmp_path: Path,
) -> None:
    modal_worker = _load_experiment_module("modal_gpu_reclamation_kill_recompute_v11")
    call = _FunctionCallReplayFixture([TimeoutError(), TimeoutError("repoll expired")])
    with pytest.raises(TimeoutError, match="repoll expired"):
        modal_worker._get_function_call_result_with_repoll(
            call,
            evidence_root=tmp_path,
            attempt_id="exp004-v11-kill-recompute-s41-f",
            reservation_id="exp004-v11-kill-f-fixture",
            reservation_commitment_sha256="a" * 64,
            config_sha256="b" * 64,
        )
    assert call.timeouts == [
        modal_worker.FIRST_FUNCTION_CALL_POLL_TIMEOUT_SECONDS,
        modal_worker.FUNCTION_CALL_REPOLL_TIMEOUT_SECONDS,
    ]
    assert (
        json.loads((tmp_path / "first-poll-timeout.json").read_text())["function_call_id"]
        == call.object_id
    )
    terminal = json.loads((tmp_path / "terminal-poll-timeout.json").read_text())
    assert terminal["function_call_id"] == call.object_id
    assert terminal["status"] == "SECOND_POLL_TIMED_OUT_FAILING_CLOSED"
    assert terminal["error"] == {"type": "TimeoutError", "message": "repoll expired"}


def test_kill_modal_outer_timeout_covers_both_polls_and_hydration(tmp_path: Path) -> None:
    coordinator = _coordinator()
    captured: dict[str, Any] = {}

    def runner(command: list[str], **kwargs: Any) -> SimpleNamespace:
        captured["command"] = command
        captured.update(kwargs)
        return SimpleNamespace(returncode=0, stdout="{}", stderr="")

    class Reservation:
        reservation_id = "exp004-v11-kill-f-fixture"

    class Ledger:
        def __init__(self) -> None:
            self.reservations = [Reservation()]

    original_loader = coordinator._BASE._load_ledger
    original_canonical = coordinator._BASE.canonical_json_bytes
    original_validate = coordinator._validate_completed_function_call_evidence
    config_path = tmp_path / "config.json"
    config_path.write_text("{}\n")
    try:
        coordinator._BASE._load_ledger = lambda _path: Ledger()
        coordinator._BASE.canonical_json_bytes = lambda _value: b"reservation\n"
        coordinator._validate_completed_function_call_evidence = lambda **_kwargs: {}
        coordinator._invoke_kill_modal(
            config_path,
            wall_seconds=660.0,
            reservation_id="exp004-v11-kill-f-fixture",
            budget_usd=80.0,
            runner=runner,
        )
    finally:
        coordinator._BASE._load_ledger = original_loader
        coordinator._BASE.canonical_json_bytes = original_canonical
        coordinator._validate_completed_function_call_evidence = original_validate
    assert captured["timeout"] == coordinator.KILL_MODAL_RETRIEVAL_TIMEOUT_SECONDS
    assert captured["timeout"] > 2 * 660.0


def _write_function_call_evidence(
    coordinator: Any,
    parent: Path,
    *,
    config: dict[str, Any],
    reservation_id: str,
    commitment: str,
    first_timeout: bool = False,
    terminal_timeout: bool = False,
) -> tuple[Path, str]:
    evidence_root = parent / coordinator.ATTEMPT_ID
    evidence_root.mkdir(parents=True)
    call_id = "fc-01M152JAGCBMQSBQ1TD55Y9HXF"
    common = {
        "attempt_id": coordinator.ATTEMPT_ID,
        "function_call_id": call_id,
        "reservation_id": reservation_id,
        "reservation_commitment_sha256": commitment,
        "config_sha256": hashlib.sha256(
            coordinator._BASE.canonical_json_bytes(
                Experiment004V11KillRecomputeConfig.model_validate(config, strict=True)
            )
        ).hexdigest(),
        "function_timeout_seconds": 660,
        "function_startup_timeout_seconds": 180,
        "first_poll_timeout_seconds": 840,
        "repoll_timeout_seconds": 720,
        "publication_grace_seconds": 60,
    }
    (evidence_root / "function-call.json").write_text(
        json.dumps(
            {
                "schema_version": ("sloforge.branchfabric.kill-recompute-function-call-capture/v1"),
                "status": "SPAWNED_AND_PERSISTED_BEFORE_FIRST_POLL",
                **common,
                "recorded_at_utc": "2026-08-28T21:30:00+00:00",
            }
        )
        + "\n"
    )
    if first_timeout:
        (evidence_root / "first-poll-timeout.json").write_text(
            json.dumps(
                {
                    "schema_version": (
                        "sloforge.branchfabric.kill-recompute-function-call-timeout/v1"
                    ),
                    "status": "FIRST_POLL_TIMED_OUT_REPOLLING_SAME_CALL",
                    **common,
                    "same_function_call_repolled": True,
                    "error": {"type": "TimeoutError", "message": "first timeout"},
                    "recorded_at_utc": "2026-08-28T21:44:00+00:00",
                }
            )
            + "\n"
        )
    if terminal_timeout:
        (evidence_root / "terminal-poll-timeout.json").write_text(
            json.dumps(
                {
                    "schema_version": (
                        "sloforge.branchfabric.kill-recompute-function-call-terminal-timeout/v1"
                    ),
                    "status": "SECOND_POLL_TIMED_OUT_FAILING_CLOSED",
                    **common,
                    "same_function_call_repolled": True,
                    "error": {"type": "TimeoutError", "message": "second timeout"},
                    "recorded_at_utc": "2026-08-28T21:56:00+00:00",
                }
            )
            + "\n"
        )
    return evidence_root, call_id


@pytest.mark.parametrize(
    "collision",
    (
        "root",
        "function-call.json",
        "first-poll-timeout.json",
        "terminal-poll-timeout.json",
    ),
)
def test_function_call_evidence_freshness_rejects_root_and_file_collisions(
    tmp_path: Path, collision: str
) -> None:
    coordinator = _coordinator()
    parent = (
        tmp_path / "artifacts/branchfabric/gpu-validation/experiment-004/v11-final/kill-recompute/"
        "function-calls"
    )
    attempt_root = parent / coordinator.ATTEMPT_ID
    if collision == "root":
        attempt_root.mkdir(parents=True)
    else:
        attempt_root.mkdir(parents=True)
        (attempt_root / collision).write_text("{}\n")
    with pytest.raises(FileExistsError, match="FunctionCall evidence"):
        coordinator._require_function_call_evidence_fresh(tmp_path)


def test_function_call_evidence_freshness_rejects_symlink_parent(tmp_path: Path) -> None:
    kill_root = (
        tmp_path / "artifacts/branchfabric/gpu-validation/experiment-004/v11-final/kill-recompute"
    )
    kill_root.mkdir(parents=True)
    target = tmp_path / "function-call-target"
    target.mkdir()
    (kill_root / "function-calls").symlink_to(target, target_is_directory=True)
    with pytest.raises(FileExistsError, match="not a safe directory"):
        coordinator = _coordinator()
        coordinator._require_function_call_evidence_fresh(tmp_path)


def test_completed_function_call_evidence_exactly_joins_remote_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    coordinator = _coordinator()
    config = _config_payload()
    reservation_id = "exp004-v11-kill-f-fixture"
    commitment = "a" * 64
    evidence_root, call_id = _write_function_call_evidence(
        coordinator,
        tmp_path,
        config=config,
        reservation_id=reservation_id,
        commitment=commitment,
        first_timeout=True,
    )
    monkeypatch.setattr(coordinator, "FUNCTION_CALL_EVIDENCE_PARENT", tmp_path)
    completed = subprocess.CompletedProcess(
        args=("modal",),
        returncode=0,
        stdout=json.dumps({"result": {"function_call_id": call_id}, "materialized": {}}) + "\n",
        stderr="",
    )
    verified = coordinator._validate_completed_function_call_evidence(
        config=config,
        reservation_id=reservation_id,
        reservation_commitment_sha256=commitment,
        completed=completed,
    )
    assert verified["function_call_id"] == call_id
    assert set(verified["timeout_evidence_sha256"]) == {"first-poll-timeout.json"}

    (evidence_root / "unexpected.json").write_text("{}\n")
    with pytest.raises(RuntimeError, match="file set changed"):
        coordinator._validate_completed_function_call_evidence(
            config=config,
            reservation_id=reservation_id,
            reservation_commitment_sha256=commitment,
            completed=completed,
        )


def test_attempt_f_real_function_call_capture_uses_validated_config_canonical_sha(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    coordinator = _coordinator()
    config_path = (
        ROOT / "artifacts/branchfabric/gpu-validation/experiment-004/v11-final/kill-recompute/"
        "exp004-v11-kill-recompute-s41-f-config.json"
    )
    evidence_parent = (
        ROOT / "artifacts/branchfabric/gpu-validation/experiment-004/v11-final/kill-recompute/"
        "function-calls"
    )
    capture = json.loads(
        (evidence_parent / coordinator.ATTEMPT_F_ID / "function-call.json").read_text()
    )
    monkeypatch.setattr(coordinator, "FUNCTION_CALL_EVIDENCE_PARENT", evidence_parent)
    monkeypatch.setattr(coordinator, "ATTEMPT_ID", coordinator.ATTEMPT_F_ID)
    verified = coordinator._validate_completed_function_call_evidence(
        config=json.loads(config_path.read_text()),
        reservation_id=capture["reservation_id"],
        reservation_commitment_sha256=capture["reservation_commitment_sha256"],
        completed=subprocess.CompletedProcess(
            args=("modal",),
            returncode=1,
            stdout="",
            stderr="remote scientific failure\n",
        ),
    )
    assert verified["function_call_id"] == "fc-01M159JSQJM88WHXC702WG8GNS"
    assert capture["config_sha256"] == (
        "0880d14f78a16108c53f72cd3f0669a9ffd6ccb15d90869568c3f669cff0e865"
    )


def _attempt_f_control_raw() -> dict[str, Any]:
    return json.loads(
        (
            ROOT / "artifacts/branchfabric/gpu-validation/experiment-004/v11-final/"
            "kill-recompute/raw/exp004-v11-kill-recompute-s41-f/barriers/"
            "v11-gpu0-serving-prevalidation.json"
        ).read_text()
    )


def _control_drift_diagnostic(depths: list[int]) -> dict[str, Any]:
    timestamps = [1_000_000_000 * (index + 1) for index in range(len(depths))]
    seconds = [float(index) for index in range(len(depths))]
    mean_x = sum(seconds) / len(seconds)
    mean_y = sum(depths) / len(depths)
    slope = sum(
        (x_value - mean_x) * (y_value - mean_y)
        for x_value, y_value in zip(seconds, depths, strict=True)
    ) / sum((value - mean_x) ** 2 for value in seconds)
    midpoint = len(depths) // 2
    first_half = sum(depths[:midpoint]) / len(depths[:midpoint])
    second_half = sum(depths[midpoint:]) / len(depths[midpoint:])
    return {
        "sample_interval_ns": 1_000_000_000,
        "samples": [
            {"timestamp_ns": timestamp, "total_outstanding": depth}
            for timestamp, depth in zip(timestamps, depths, strict=True)
        ],
        "initial_depth": depths[0],
        "final_depth": depths[-1],
        "maximum_depth": max(depths),
        "first_half_mean_depth": first_half,
        "second_half_mean_depth": second_half,
        "slope_requests_per_second": slope,
        "sustained_positive": bool(
            slope > 0.0 and depths[-1] > depths[0] and second_half > first_half
        ),
    }


def test_attempt_f_control_revalidates_under_preexisting_material_drift_invariant() -> None:
    integrated_worker = _load_experiment_module("gpu_reclamation_integrated_worker_v11")
    serving = _attempt_f_control_raw()
    evidence = worker._kill_control_interval_evidence(
        serving,
        expected_rate_rps=9.0,
        expected_duration_seconds=5.0,
        slo_ttft_seconds=2.0,
        warmup_seconds=1.0,
        evaluation_seconds=1.0,
        minimum_completion_fraction=0.90,
        _base_evaluator=integrated_worker._control_interval_evidence,
    )
    assert evidence["completions_in_interval"] == 34
    assert evidence["completion_tracks_offer"] is True
    assert evidence["p95_ttft_ns"] == pytest.approx(57_507_309.25)
    assert evidence["waiting_queue_maximum_depth"] == 1
    assert evidence["total_outstanding_diagnostic"]["maximum_depth"] == 12
    assert evidence["total_outstanding_diagnostic"]["slope_requests_per_second"] == 0.2
    persisted = json.loads(
        (
            ROOT / "artifacts/branchfabric/gpu-validation/experiment-004/v11-final/kill-recompute/"
            "raw/exp004-v11-kill-recompute-s41-f/barriers/v11-gpu0-control-interval.json"
        ).read_text()
    )
    assert persisted["outstanding_queue_non_positive_trend"] is False
    assert persisted["passed"] is False
    diagnostic = evidence["total_outstanding_diagnostic"]
    assert diagnostic["sustained_positive"] is False
    assert diagnostic["first_quarter_median_depth"] == 11.0
    assert diagnostic["last_quarter_median_depth"] == 11.5
    assert diagnostic["drift_policy"] == (
        "slope>0.10-and-last-quarter-median>first-quarter-median+2"
    )
    assert evidence["kill_recompute_drift_policy"] == {
        "schema_version": "sloforge.branchfabric.kill-recompute-control-drift-policy/v1",
        "policy": "slope>0.10-and-last-quarter-median>first-quarter-median+2",
        "quarter_sample_count": 2,
        "first_quarter_median_depth": 11.0,
        "last_quarter_median_depth": 11.5,
        "minimum_positive_slope_requests_per_second": 0.10,
        "minimum_material_median_increase": 2,
        "persistent_positive": False,
    }
    assert evidence["outstanding_queue_non_positive_trend"] is True
    assert evidence["passed"] is True

    full_start_ns = serving["start_ns"]
    start_ns = full_start_ns + 1_000_000_000
    end_ns = serving["spike_start_ns"]
    control_rows = [
        row
        for row in serving["requests"]
        if row.get("phase") == "control" and full_start_ns <= row["scheduled_arrival_ns"] < end_ns
    ]
    timestamps = [*range(start_ns, end_ns, 1_000_000_000), end_ns]
    waiting_depths = [
        sum(
            row["scheduled_arrival_ns"] <= timestamp < row["service_start_ns"]
            for row in control_rows
        )
        for timestamp in timestamps
    ]
    assert waiting_depths == [1, 1, 1, 1, 0]


def test_frozen_v10_control_replay_passes_kill_scoped_material_drift_policy() -> None:
    integrated_worker = _load_experiment_module("gpu_reclamation_integrated_worker_v11")
    result = json.loads(
        (
            ROOT / "artifacts/branchfabric/gpu-validation/experiment-004/raw/modal/"
            "exp004-v10-naive-s41-v7/serving/result.json"
        ).read_text()
    )
    evidence = worker._kill_control_interval_evidence(
        result,
        expected_rate_rps=9.0,
        expected_duration_seconds=5.0,
        slo_ttft_seconds=2.0,
        _base_evaluator=integrated_worker._control_interval_evidence,
    )
    assert evidence["passed"] is True
    assert evidence["outstanding_queue_non_positive_trend"] is True
    assert evidence["kill_recompute_drift_policy"]["persistent_positive"] is False


@pytest.mark.parametrize(
    ("depths", "persistent"),
    [
        ([10, 12, 15, 17, 19], True),
        ([10, 10, 11, 12, 12], False),
        ([10, 10, 11, 12, 13], True),
        ([10, 12, 12, 12, 11], False),
        ([10, 10, 100, 0, 0, 0, 13, 13], False),
    ],
)
def test_kill_control_material_drift_requires_slope_and_more_than_two_median_delta(
    depths: list[int], persistent: bool
) -> None:
    result = worker._material_persistent_control_drift(_control_drift_diagnostic(depths))
    assert result["persistent_positive"] is persistent
    assert result["minimum_positive_slope_requests_per_second"] == 0.10
    assert result["minimum_material_median_increase"] == 2


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value.update(sample_interval_ns=True),
        lambda value: value["samples"][1].update(timestamp_ns=2_000_000_001),
        lambda value: value["samples"][1].update(total_outstanding=True),
        lambda value: value["samples"][1].update(total_outstanding=-1),
        lambda value: value.update(first_half_mean_depth=999.0),
        lambda value: value.update(second_half_mean_depth=999.0),
        lambda value: value.update(slope_requests_per_second=999.0),
        lambda value: value.update(unexpected=True),
    ],
)
def test_kill_control_material_drift_rejects_malformed_or_tampered_diagnostic(
    mutation: Any,
) -> None:
    diagnostic = _control_drift_diagnostic([10, 12, 12, 12, 11])
    mutation(diagnostic)
    with pytest.raises(RuntimeError, match="kill/recompute control drift"):
        worker._material_persistent_control_drift(diagnostic)


def test_kill_control_material_drift_rejects_too_few_samples() -> None:
    with pytest.raises(RuntimeError, match="diagnostic is malformed"):
        worker._material_persistent_control_drift(_control_drift_diagnostic([10, 11, 12]))


@pytest.mark.parametrize(
    "failed_gate",
    [
        "completion_tracks_offer",
        "p95_ttft_below_slo",
        "waiting_queue_bounded_below_normal_trigger",
        "total_outstanding_bounded_below_normal_trigger",
        "full_cohort_output_tokens_exact",
    ],
)
def test_kill_control_override_cannot_mask_any_other_frozen_gate(failed_gate: str) -> None:
    integrated_worker = _load_experiment_module("gpu_reclamation_integrated_worker_v11")
    raw = _attempt_f_control_raw()
    base = integrated_worker._control_interval_evidence(
        raw,
        expected_rate_rps=9.0,
        expected_duration_seconds=5.0,
        slo_ttft_seconds=2.0,
    )
    base[failed_gate] = False
    evidence = worker._kill_control_interval_evidence(
        raw,
        expected_rate_rps=9.0,
        expected_duration_seconds=5.0,
        slo_ttft_seconds=2.0,
        _base_evaluator=lambda *_args, **_kwargs: copy.deepcopy(base),
    )
    assert evidence[failed_gate] is False
    assert evidence["passed"] is False


def test_kill_control_rejects_thirty_two_of_thirty_six_completions() -> None:
    integrated_worker = _load_experiment_module("gpu_reclamation_integrated_worker_v11")
    raw = _attempt_f_control_raw()
    evaluation_start_ns = raw["start_ns"] + 1_000_000_000
    rows = [
        row
        for row in raw["requests"]
        if row.get("phase") == "control"
        and raw["start_ns"] <= row["scheduled_arrival_ns"] < raw["spike_start_ns"]
        and evaluation_start_ns <= row["completed_ns"] < raw["spike_start_ns"]
    ]
    assert len(rows) == 34
    for row in rows[-2:]:
        row["completed_ns"] = raw["spike_start_ns"] + 1
    evidence = worker._kill_control_interval_evidence(
        raw,
        expected_rate_rps=9.0,
        expected_duration_seconds=5.0,
        slo_ttft_seconds=2.0,
        _base_evaluator=integrated_worker._control_interval_evidence,
    )
    assert evidence["completions_in_interval"] == 32
    assert evidence["completion_tracks_offer"] is False
    assert evidence["passed"] is False


@pytest.mark.parametrize("raises", [False, True])
def test_kill_worker_control_override_is_scoped_and_restored(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, raises: bool
) -> None:
    integrated_worker = _load_experiment_module("gpu_reclamation_integrated_worker_v11")
    frozen_evaluator = integrated_worker._control_interval_evidence

    def fake_run(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        assert integrated_worker._control_interval_evidence is not frozen_evaluator
        evidence = integrated_worker._control_interval_evidence(
            _attempt_f_control_raw(),
            expected_rate_rps=9.0,
            expected_duration_seconds=5.0,
            slo_ttft_seconds=2.0,
        )
        assert evidence["passed"] is True
        if raises:
            raise RuntimeError("fixture kill serving failure")
        return {}

    monkeypatch.setattr(integrated_worker, "run_integrated_v11_gpu0", fake_run)
    if raises:
        with pytest.raises(RuntimeError, match="fixture kill serving failure"):
            worker.run_kill_recompute_gpu0(
                object(),
                adapter=object(),
                inputs={},
                config=object(),
                start_ns=1,
                barriers=tmp_path,
            )
    else:
        result = worker.run_kill_recompute_gpu0(
            object(),
            adapter=object(),
            inputs={},
            config=object(),
            start_ns=1,
            barriers=tmp_path,
        )
        assert result["schema_version"] == "sloforge.branchfabric.kill-recompute-gpu0-result/v1"
    assert integrated_worker._control_interval_evidence is frozen_evaluator


@pytest.mark.parametrize("raises", [False, True])
def test_kill_controller_control_recomputation_is_scoped_and_restored(
    monkeypatch: pytest.MonkeyPatch, raises: bool
) -> None:
    import gpu_reclamation_integrated_controller_v11 as integrated

    frozen_evaluator = integrated._control_interval_evidence

    def fake_validate(_payload: Mapping[str, Any]) -> None:
        assert integrated._control_interval_evidence is not frozen_evaluator
        evidence = integrated._control_interval_evidence(
            _attempt_f_control_raw(),
            expected_rate_rps=9.0,
            expected_duration_seconds=5.0,
            slo_ttft_seconds=2.0,
        )
        assert evidence["passed"] is True
        if raises:
            raise RuntimeError("fixture kill control validation failure")

    monkeypatch.setattr(integrated, "_validate_gpu0_scientific_result", fake_validate)
    monkeypatch.setattr(
        integrated, "_validate_cross_role_scientific_consistency", lambda *_args: None
    )
    if raises:
        with pytest.raises(RuntimeError, match="fixture kill control validation failure"):
            controller._validate_kill_control_consistency({}, {})
    else:
        controller._validate_kill_control_consistency({}, {})
    assert integrated._control_interval_evidence is frozen_evaluator


def test_kill_controller_deep_recompute_rejects_control_policy_summary_tamper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import gpu_reclamation_integrated_controller_v11 as integrated

    payload = json.loads(
        (
            ROOT / "artifacts/branchfabric/gpu-validation/experiment-004/v11-final/integrated/"
            "raw/exp004-v11-integrated-s41-k/serving/result.json"
        ).read_text()
    )
    control = worker._kill_control_interval_evidence(
        payload["serving"],
        expected_rate_rps=9.0,
        expected_duration_seconds=5.0,
        slo_ttft_seconds=2.0,
        _base_evaluator=integrated._control_interval_evidence,
    )
    payload["gpu0_control_interval"] = control
    payload["gpu0_restore_interference"]["control_interval"] = copy.deepcopy(control)
    monkeypatch.setattr(
        integrated, "_validate_cross_role_scientific_consistency", lambda *_args: None
    )
    controller._validate_kill_control_consistency(payload, {})

    payload["gpu0_control_interval"]["kill_recompute_drift_policy"][
        "first_quarter_median_depth"
    ] += 1.0
    payload["gpu0_restore_interference"]["control_interval"] = copy.deepcopy(
        payload["gpu0_control_interval"]
    )
    with pytest.raises(RuntimeError, match="control-interval evidence is invalid"):
        controller._validate_kill_control_consistency(payload, {})


def test_terminal_function_call_timeout_requires_first_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    coordinator = _coordinator()
    config = _config_payload()
    reservation_id = "exp004-v11-kill-f-fixture"
    commitment = "a" * 64
    _write_function_call_evidence(
        coordinator,
        tmp_path,
        config=config,
        reservation_id=reservation_id,
        commitment=commitment,
        terminal_timeout=True,
    )
    monkeypatch.setattr(coordinator, "FUNCTION_CALL_EVIDENCE_PARENT", tmp_path)
    completed = subprocess.CompletedProcess(
        args=("modal",), returncode=1, stdout="", stderr="TimeoutError\n"
    )
    with pytest.raises(RuntimeError, match="lacks its first timeout evidence"):
        coordinator._validate_completed_function_call_evidence(
            config=config,
            reservation_id=reservation_id,
            reservation_commitment_sha256=commitment,
            completed=completed,
        )


def test_function_call_validation_failure_preserves_completed_process(
    tmp_path: Path,
) -> None:
    coordinator = _coordinator()
    config_path = tmp_path / "config.json"
    config_path.write_text("{}\n")

    class Reservation:
        reservation_id = "exp004-v11-kill-f-fixture"

    class Ledger:
        def __init__(self) -> None:
            self.reservations = [Reservation()]

    original_loader = coordinator._BASE._load_ledger
    original_canonical = coordinator._BASE.canonical_json_bytes
    original_validate = coordinator._validate_completed_function_call_evidence
    try:
        coordinator._BASE._load_ledger = lambda _path: Ledger()
        coordinator._BASE.canonical_json_bytes = lambda _value: b"reservation\n"

        def reject(**_kwargs: Any) -> None:
            raise RuntimeError("typed FunctionCall evidence mismatch")

        coordinator._validate_completed_function_call_evidence = reject
        completed, _elapsed = coordinator._invoke_kill_modal(
            config_path,
            wall_seconds=660.0,
            reservation_id="exp004-v11-kill-f-fixture",
            budget_usd=80.0,
            runner=lambda *_args, **_kwargs: subprocess.CompletedProcess(
                args=("modal",),
                returncode=0,
                stdout="original stdout\n",
                stderr="original stderr\n",
            ),
        )
    finally:
        coordinator._BASE._load_ledger = original_loader
        coordinator._BASE.canonical_json_bytes = original_canonical
        coordinator._validate_completed_function_call_evidence = original_validate
    assert completed.returncode == 86
    assert completed.stdout == "original stdout\n"
    assert "original stderr" in completed.stderr
    assert "function-call-evidence-validation-failure/v1" in completed.stderr
    assert "typed FunctionCall evidence mismatch" in completed.stderr


def test_post_controller_scan_rejects_preservation_artifact(tmp_path: Path) -> None:
    forbidden = tmp_path / "v11-source-capture-commit.json"
    forbidden.write_text("{}")
    assert any(
        token in forbidden.name
        for token in (
            "checkpoint",
            "source-capture-commit",
            "pre-export",
            "post-export",
            "source-release",
            "state-pass",
            "movement",
            "optimized-export",
        )
    )


@pytest.mark.parametrize("raises", [False, True])
def test_controller_scoped_bindings_restore_on_success_and_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, raises: bool
) -> None:
    import gpu_reclamation_integrated_controller_v11 as integrated

    originals = {
        "config": integrated.Experiment004V11IntegratedConfig,
        "reservation": integrated.V11_INTEGRATED_RESERVATION_WALL_SECONDS,
        "absolute": integrated.ABSOLUTE_WALL_SECONDS,
        "verify": integrated.verify_sealed_evidence,
        "command": integrated._transaction_command,
        "results": integrated._validate_worker_results,
    }

    def fake_run(**_kwargs: Any) -> dict[str, Any]:
        if raises:
            raise RuntimeError("fixture-controller-error")
        return {"status": "succeeded"}

    monkeypatch.setattr(integrated, "run_integrated_v11_controller", fake_run)
    if raises:
        with pytest.raises(RuntimeError, match="fixture-controller-error"):
            controller.run_kill_recompute_controller(
                config_path=tmp_path / "config.json",
                work_root=tmp_path,
                worker_path=tmp_path / "worker.py",
                model_snapshot=tmp_path / "model",
                repository_root=ROOT,
                absolute_deadline_ns=10,
            )
    else:
        result = controller.run_kill_recompute_controller(
            config_path=tmp_path / "config.json",
            work_root=tmp_path,
            worker_path=tmp_path / "worker.py",
            model_snapshot=tmp_path / "model",
            repository_root=ROOT,
            absolute_deadline_ns=10,
        )
        assert result["no_preservation_output_artifacts"]["passed"] is True
    assert integrated.Experiment004V11IntegratedConfig is originals["config"]
    assert originals["reservation"] == integrated.V11_INTEGRATED_RESERVATION_WALL_SECONDS
    assert originals["absolute"] == integrated.ABSOLUTE_WALL_SECONDS
    assert integrated.verify_sealed_evidence is originals["verify"]
    assert integrated._transaction_command is originals["command"]
    assert integrated._validate_worker_results is originals["results"]


def test_controller_scan_fails_on_checkpoint_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import gpu_reclamation_integrated_controller_v11 as integrated

    (tmp_path / "checkpoint.bin").write_bytes(b"forbidden")
    monkeypatch.setattr(
        integrated,
        "run_integrated_v11_controller",
        lambda **_kwargs: {"status": "succeeded"},
    )
    with pytest.raises(RuntimeError, match="preservation output artifacts"):
        controller.run_kill_recompute_controller(
            config_path=tmp_path / "config.json",
            work_root=tmp_path,
            worker_path=tmp_path / "worker.py",
            model_snapshot=tmp_path / "model",
            repository_root=ROOT,
            absolute_deadline_ns=10,
        )


def test_config_token_accounting_tamper_rejected() -> None:
    payload = _config_payload()
    payload["expected_submitted_replay_tokens"] = 18_432
    with pytest.raises(ValidationError):
        Experiment004V11KillRecomputeConfig.model_validate(payload, strict=True)


def test_history_row_hash_tamper_is_detectable() -> None:
    evidence = worker._history_evidence(
        _live_boundary_tables(), prefix_length=16_384, suffix_length=256
    )
    tampered = copy.deepcopy(evidence)
    tampered["branches"][0]["token_ids"][0] += 1
    with pytest.raises(RuntimeError, match="history row is invalid"):
        controller._validate_replay_history(tampered)


def test_postworker_failure_envelope_is_exact_and_rejects_substitution() -> None:
    coordinator = _coordinator()
    config = Experiment004V11KillRecomputeConfig.model_validate(_config_payload(), strict=True)
    envelope = _postworker_failure_envelope(
        coordinator,
        config=config,
        reservation_id="exp004-v11-kill-a-failure-fixture",
        reservation_commitment="c" * 64,
    )
    _validated, _materialized, inventory = coordinator._validate_kill_remote_result(
        envelope,
        config=config,
        reservation_id="exp004-v11-kill-a-failure-fixture",
        reservation_commitment_sha256="c" * 64,
    )
    assert inventory is not None

    extra_controller_field = copy.deepcopy(envelope)
    extra_controller_field["result"]["controller"]["unexpected"] = True
    with pytest.raises(ValueError, match="identity/resource/status"):
        coordinator._validate_kill_remote_result(
            extra_controller_field,
            config=config,
            reservation_id="exp004-v11-kill-a-failure-fixture",
            reservation_commitment_sha256="c" * 64,
        )
    launcher_substitution = copy.deepcopy(envelope)
    launcher_substitution["result"]["run_error"]["message"] = "different failure"
    with pytest.raises(ValueError, match="identity/resource/status"):
        coordinator._validate_kill_remote_result(
            launcher_substitution,
            config=config,
            reservation_id="exp004-v11-kill-a-failure-fixture",
            reservation_commitment_sha256="c" * 64,
        )


@pytest.mark.parametrize(
    "mutate",
    (
        lambda row: row.update(failure_stage="WORKERS_RUNNING"),
        lambda row: row.update(sealed_evidence={}),
        lambda row: row.update(stable_physical_gpu_identity=True),
        lambda row: row.update(readiness_deadline_ns=1),
        lambda row: row.update(cleanup_error={"type": "RuntimeError", "message": "tamper"}),
        lambda row: row.update(ended_ns=row["started_ns"]),
        lambda row: row.update(controller_error={}),
        lambda row: row["cuda_clean_import_audits"][0].update(cuda_clean=False),
        lambda row: row["no_preservation_output_artifacts"].update(passed=False),
        lambda row: row["in_function_cleanup"].update(forced_kills=[123]),
    ),
)
def test_preworker_failure_requires_exact_semantic_envelope(mutate: Any) -> None:
    coordinator = _coordinator()
    config = Experiment004V11KillRecomputeConfig.model_validate(_config_payload(), strict=True)
    envelope = _preworker_failure_envelope(
        coordinator,
        config=config,
        reservation_id="exp004-v11-kill-a-preworker-fixture",
        reservation_commitment="b" * 64,
    )
    _validated, _materialized, inventory = coordinator._validate_kill_remote_result(
        envelope,
        config=config,
        reservation_id="exp004-v11-kill-a-preworker-fixture",
        reservation_commitment_sha256="b" * 64,
    )
    assert inventory is None
    tampered = copy.deepcopy(envelope)
    mutate(tampered["result"]["controller"])
    with pytest.raises(ValueError, match="identity/resource/status"):
        coordinator._validate_kill_remote_result(
            tampered,
            config=config,
            reservation_id="exp004-v11-kill-a-preworker-fixture",
            reservation_commitment_sha256="b" * 64,
        )


def test_preworker_failure_rejects_hardware_substitution() -> None:
    coordinator = _coordinator()
    config = Experiment004V11KillRecomputeConfig.model_validate(_config_payload(), strict=True)
    envelope = _preworker_failure_envelope(
        coordinator,
        config=config,
        reservation_id="exp004-v11-kill-a-preworker-fixture",
        reservation_commitment="b" * 64,
    )
    envelope["result"]["hardware_comparability"] = {
        "modal_request": "A100-80GB:2",
        "observed_gpu_names": ["NVIDIA A100-SXM4-80GB"] * 2,
        "observed_gpu_uuids": ["GPU-a", "GPU-b"],
        "direct_k_comparison_ready": True,
    }
    with pytest.raises(ValueError, match="identity/resource/status"):
        coordinator._validate_kill_remote_result(
            envelope,
            config=config,
            reservation_id="exp004-v11-kill-a-preworker-fixture",
            reservation_commitment_sha256="b" * 64,
        )


@pytest.mark.parametrize(
    ("tamper", "message"),
    (
        ("function-failure.json", "failed controller bundle is incomplete"),
        ("function-finally-cleanup.json", "failed controller bundle is incomplete"),
        (
            "in_function_cleanup.json",
            "downloaded cleanup evidence differs from the returned result",
        ),
        ("kill-controller-result.json", "failed controller bundle is incomplete"),
        ("controller-result.json", "failed controller bundle is incomplete"),
        (
            "kill-no-preservation-output-artifacts.json",
            "failed no-preservation output proof is invalid",
        ),
    ),
)
def test_downloaded_postworker_failure_exactly_joins_all_failure_artifacts(
    tmp_path: Path, tamper: str, message: str
) -> None:
    coordinator = _coordinator()
    config = Experiment004V11KillRecomputeConfig.model_validate(_config_payload(), strict=True)
    envelope = _postworker_failure_envelope(
        coordinator,
        config=config,
        reservation_id="exp004-v11-kill-a-failure-fixture",
        reservation_commitment="c" * 64,
    )
    remote = envelope["result"]
    controller_payload = remote["controller"]
    cleanup = remote["in_function_cleanup"]
    completion = {
        key: value
        for key, value in remote.items()
        if key
        not in {
            "gpu_allocation_seconds",
            "gpu_seconds",
            "gpu_hours",
            "remote_prefix",
            "remote_manifest_sha256",
        }
    }
    inherited_controller = copy.deepcopy(controller_payload)
    inherited_controller.pop("no_preservation_output_artifacts")
    files = {
        "controller-result.json": inherited_controller,
        "function-completion.json": completion,
        "function-failure.json": remote["run_error"],
        "function-finally-cleanup.json": {
            "schema_version": "sloforge.branchfabric.kill-recompute-function-cleanup/v1",
            "attempt_id": coordinator.ATTEMPT_ID,
            "controller_status": "failed",
            "controller_cleanup_evidence_present": True,
            "in_function_cleanup_pass": True,
            "in_function_cleanup_artifact": "in_function_cleanup.json",
            "provider_cleanup_status": "pending-function-return-and-local-postflight",
            "recorded_at_utc": "2026-08-27T00:00:00+00:00",
        },
        "in_function_cleanup.json": cleanup,
        "kill-controller-result.json": controller_payload,
        "kill-no-preservation-output-artifacts.json": controller_payload[
            "no_preservation_output_artifacts"
        ],
    }
    for name, payload in files.items():
        (tmp_path / name).write_text(
            json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
        )

    def write_manifest() -> Path:
        manifest_payload = {
            "schema_version": (
                "sloforge.branchfabric.experiment-004-v11-integrated-remote-manifest/v1"
            ),
            "attempt_id": coordinator.ATTEMPT_ID,
            "remote_prefix": remote["remote_prefix"],
            "artifacts": [
                {
                    "relative_path": path.name,
                    "bytes": path.stat().st_size,
                    "sha256": _sha(path),
                }
                for path in sorted(tmp_path.iterdir())
                if path.name != "REMOTE_MANIFEST.json"
            ],
        }
        manifest = tmp_path / "REMOTE_MANIFEST.json"
        manifest.write_text(
            json.dumps(manifest_payload, sort_keys=True, separators=(",", ":")) + "\n"
        )
        remote["remote_manifest_sha256"] = _sha(manifest)
        return manifest

    manifest = write_manifest()
    assert (
        coordinator._verify_kill_manifest(
            tmp_path,
            remote=remote,
            materialized=envelope["materialized"],
        )
        == manifest
    )

    tampered = json.loads((tmp_path / tamper).read_text())
    if tamper == "function-failure.json":
        tampered["message"] = "substituted"
    elif tamper == "function-finally-cleanup.json":
        tampered["provider_cleanup_status"] = "PASS"
    elif tamper == "in_function_cleanup.json":
        tampered["forced_kills"] = [123]
    elif tamper in {"kill-controller-result.json", "controller-result.json"}:
        tampered["status"] = "succeeded"
    else:
        tampered["passed"] = False
    (tmp_path / tamper).write_text(
        json.dumps(tampered, sort_keys=True, separators=(",", ":")) + "\n"
    )
    write_manifest()
    with pytest.raises(ValueError, match=message):
        coordinator._verify_kill_manifest(
            tmp_path,
            remote=remote,
            materialized=envelope["materialized"],
        )


def test_cpu_config_remote_envelope_and_measured_settlement(tmp_path: Path) -> None:
    coordinator = _coordinator()
    config = Experiment004V11KillRecomputeConfig.model_validate(_config_payload(), strict=True)
    ledger = tmp_path / "gpu-hours.json"
    ledger.write_bytes(
        (
            ROOT / "artifacts/branchfabric/gpu-validation/experiment-004/v11-final/"
            "kill-recompute/authorization/ledger-snapshot-before-attempt-g.json"
        ).read_bytes()
    )
    reservation_id = "exp004-v11-kill-a-cpu-fixture"
    _reserved, _preflight, commitment = coordinator._BASE._reserve_locked(
        config, ledger_path=ledger, reservation_id=reservation_id
    )
    controller_payload = _postworker_success_controller_payload(coordinator)
    inventory = controller_payload["inventory_before"]
    cleanup = controller_payload["in_function_cleanup"]
    bound = _bound_k_evidence(config)
    entry = 1_000_000_000
    remote = {
        "schema_version": "sloforge.branchfabric.kill-recompute-completion/v1",
        "status": "provisional",
        "scientific_status": "pending-local-budget-settlement-and-provider-cleanup",
        "attempt_id": coordinator.ATTEMPT_ID,
        "reservation_id": reservation_id,
        "reservation_commitment_sha256": commitment,
        "config_sha256": hashlib.sha256(coordinator._BASE.canonical_json_bytes(config)).hexdigest(),
        "function_call_id": "fc-cpu-fixture",
        "requested_gpu": "A100-80GB:2",
        "gpu_count": 2,
        "controller_and_analysis_interval_seconds": 0.9,
        "gpu_allocation_seconds_status": "pending-final-function-return",
        "hardware_comparability": {
            "modal_request": "A100-80GB:2",
            "modal_interconnect_selection_available": False,
            "observed_gpu_names": [row["name"] for row in inventory],
            "observed_gpu_uuids": [row["uuid"] for row in inventory],
            "frozen_v10_gpu_name": "NVIDIA A100-SXM4-80GB",
            "direct_v10_timing_comparable": True,
            "timing_comparability_reason": (
                "both physical devices exactly match frozen v10 A100-SXM4-80GB"
            ),
            "byte_and_amplification_comparable": True,
            "direct_k_comparison_ready": True,
        },
        "bound_k_evidence": bound,
        "absolute_deadlines": {
            "function_entry_monotonic_ns": entry,
            "controller_deadline_monotonic_ns": entry + 650_000_000_000,
            "function_deadline_monotonic_ns": entry + 660_000_000_000,
            "post_controller_reserve_seconds": 10.0,
        },
        "controller": controller_payload,
        "in_function_cleanup": cleanup,
        "run_error": None,
        "completed_at_utc": "2026-08-27T00:00:00+00:00",
        "gpu_allocation_seconds": 1.0,
        "gpu_seconds": 2.0,
        "gpu_hours": 2.0 / 3600.0,
        "remote_prefix": f"{coordinator.REMOTE_PREFIX}/{coordinator.ATTEMPT_ID}",
        "remote_manifest_sha256": "d" * 64,
    }
    envelope = {
        "result": remote,
        "materialized": {
            "remote_path": remote["remote_prefix"],
            "volume_name": coordinator.RESULTS_VOLUME,
        },
    }
    validated, _materialized, validated_inventory = coordinator._validate_kill_remote_result(
        envelope,
        config=config,
        reservation_id=reservation_id,
        reservation_commitment_sha256=commitment,
    )
    assert validated_inventory is not None
    controller_tampers = (
        ("schema_version", "sloforge.branchfabric.wrong/v1"),
        ("attempt_id", "exp004-v11-kill-recompute-s41-a"),
        ("execution_mode", "integrated-reclamation-v11"),
        ("terminal_phase", "SOURCE_ALLOCATION_IDENTITY_GATE"),
    )
    for field, value in controller_tampers:
        tampered = copy.deepcopy(envelope)
        tampered["result"]["controller"][field] = value
        with pytest.raises(ValueError, match="identity/resource/status"):
            coordinator._validate_kill_remote_result(
                tampered,
                config=config,
                reservation_id=reservation_id,
                reservation_commitment_sha256=commitment,
            )
    proof_tamper = copy.deepcopy(envelope)
    proof_tamper["result"]["controller"]["no_preservation_output_artifacts"]["passed"] = False
    with pytest.raises(ValueError, match="identity/resource/status"):
        coordinator._validate_kill_remote_result(
            proof_tamper,
            config=config,
            reservation_id=reservation_id,
            reservation_commitment_sha256=commitment,
        )
    hardware_tamper = copy.deepcopy(envelope)
    hardware_tamper["result"]["hardware_comparability"]["direct_k_comparison_ready"] = False
    with pytest.raises(ValueError, match="identity/resource/status"):
        coordinator._validate_kill_remote_result(
            hardware_tamper,
            config=config,
            reservation_id=reservation_id,
            reservation_commitment_sha256=commitment,
        )
    cleanup_tamper = copy.deepcopy(envelope)
    cleanup_tamper["result"]["in_function_cleanup"]["forced_kills"] = [123]
    cleanup_tamper["result"]["controller"]["in_function_cleanup"] = cleanup_tamper["result"][
        "in_function_cleanup"
    ]
    with pytest.raises(ValueError, match="identity/resource/status"):
        coordinator._validate_kill_remote_result(
            cleanup_tamper,
            config=config,
            reservation_id=reservation_id,
            reservation_commitment_sha256=commitment,
        )
    for mutate_cleanup in (
        lambda value: value.update(owned_children=[]),
        lambda value: next(
            row
            for row in value["owned_children"]
            if row.get("role") == "serving" and row.get("process_kind") == "worker"
        ).update(pgid=999_999),
    ):
        ownership_tamper = copy.deepcopy(envelope)
        mutate_cleanup(ownership_tamper["result"]["in_function_cleanup"])
        ownership_tamper["result"]["controller"]["in_function_cleanup"] = ownership_tamper[
            "result"
        ]["in_function_cleanup"]
        with pytest.raises(ValueError, match="identity/resource/status"):
            coordinator._validate_kill_remote_result(
                ownership_tamper,
                config=config,
                reservation_id=reservation_id,
                reservation_commitment_sha256=commitment,
            )
    manifest = tmp_path / "REMOTE_MANIFEST.json"
    manifest.write_text("{}\n")
    settled, accounted = coordinator._BASE._settle_verified(
        ledger_path=ledger,
        reservation_id=reservation_id,
        remote=validated,
        inventory=validated_inventory,
        manifest_path=manifest,
        coordinator_elapsed_seconds=1.1,
    )
    assert accounted == 1.0
    assert settled.reservations == ()
    assert settled.intervals[-1].invocation_id == coordinator.ATTEMPT_ID


def test_downloaded_success_manifest_reaches_local_cleanup_and_deep_validator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    coordinator = _coordinator()
    cleanup = json.loads(
        (
            ROOT / "artifacts/branchfabric/gpu-validation/experiment-004/v11-final/integrated/"
            "raw/exp004-v11-integrated-s41-k/in_function_cleanup.json"
        ).read_text()
    )
    inventory = [
        {
            "index": index,
            "uuid": f"GPU-{index}",
            "name": "NVIDIA A100-SXM4-80GB",
            "driver_version": "580.95.05",
            "memory_total_mib": 81_920,
            "memory_used_mib": 4,
            "utilization_percent": 0,
        }
        for index in range(2)
    ]
    proof = {
        "schema_version": "sloforge.branchfabric.kill-recompute-no-preservation-outputs/v1",
        "scan_scope": "controller work root after both workers and before function return",
        "scanned_file_count": 1,
        "scanned_relative_paths_sha256": "a" * 64,
        "forbidden_path_tokens": [
            "checkpoint",
            "source-capture-commit",
            "pre-export",
            "post-export",
            "source-release",
            "state-pass",
            "movement",
            "optimized-export",
        ],
        "forbidden_matches": [],
        "checkpoint_materialized": False,
        "optimized_export_executed": False,
        "passed": True,
    }
    controller_payload = {
        "status": "succeeded",
        "in_function_cleanup": cleanup,
        "no_preservation_output_artifacts": proof,
        "worker_pids": {"serving": 10, "rollout": 11},
        "inventory_before": inventory,
        "worker_results": [],
    }
    completion = {
        "schema_version": "sloforge.branchfabric.kill-recompute-completion/v1",
        "status": "provisional",
        "scientific_status": "pending-local-budget-settlement-and-provider-cleanup",
        "attempt_id": coordinator.ATTEMPT_ID,
        "controller": controller_payload,
        "in_function_cleanup": cleanup,
    }
    files = {
        "function-completion.json": completion,
        "in_function_cleanup.json": cleanup,
        "kill-no-preservation-output-artifacts.json": proof,
        "kill-controller-result.json": controller_payload,
    }
    for name, payload in files.items():
        (tmp_path / name).write_text(
            json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
        )
    remote_prefix = f"{coordinator.REMOTE_PREFIX}/{coordinator.ATTEMPT_ID}"
    manifest_payload = {
        "schema_version": (
            "sloforge.branchfabric.experiment-004-v11-integrated-remote-manifest/v1"
        ),
        "attempt_id": coordinator.ATTEMPT_ID,
        "remote_prefix": remote_prefix,
        "artifacts": [
            {
                "relative_path": name,
                "bytes": (tmp_path / name).stat().st_size,
                "sha256": _sha(tmp_path / name),
            }
            for name in sorted(files)
        ],
    }
    manifest = tmp_path / "REMOTE_MANIFEST.json"
    manifest.write_text(json.dumps(manifest_payload, sort_keys=True, separators=(",", ":")) + "\n")
    remote = {
        **completion,
        "gpu_allocation_seconds": 1.0,
        "gpu_seconds": 2.0,
        "gpu_hours": 2.0 / 3600.0,
        "remote_prefix": remote_prefix,
        "remote_manifest_sha256": _sha(manifest),
    }
    monkeypatch.setattr(controller, "_validate_kill_worker_results", lambda *_args, **_kwargs: ())
    observed = coordinator._verify_kill_manifest(
        tmp_path,
        remote=remote,
        materialized={
            "remote_path": remote_prefix,
            "volume_name": coordinator.RESULTS_VOLUME,
        },
    )
    assert observed == manifest

    forbidden = tmp_path / "source-capture-commit.json"
    forbidden.write_text("{}\n")
    manifest_payload["artifacts"] = [
        {
            "relative_path": path.name,
            "bytes": path.stat().st_size,
            "sha256": _sha(path),
        }
        for path in sorted(tmp_path.iterdir())
        if path.name != "REMOTE_MANIFEST.json"
    ]
    manifest.write_text(json.dumps(manifest_payload, sort_keys=True, separators=(",", ":")) + "\n")
    remote["remote_manifest_sha256"] = _sha(manifest)
    with pytest.raises(ValueError, match="forbidden export artifact"):
        coordinator._verify_kill_manifest(
            tmp_path,
            remote=remote,
            materialized={
                "remote_path": remote_prefix,
                "volume_name": coordinator.RESULTS_VOLUME,
            },
        )
