from __future__ import annotations

import ast
import copy
import hashlib
import importlib.util
import json
import shutil
import signal
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import ValidationError

from sloforge.helix.characterization.gpu_reclamation_v11_methodology import (
    Experiment004V11IntegratedConfig,
    Experiment004V11TargetedSourceIdentityConfig,
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
INTEGRATED_WORKER = sys.modules["gpu_reclamation_integrated_worker_v11"]

SHARED_WORKER_PATH = EXPERIMENT_ROOT / "gpu_reclamation_worker.py"
SHARED_WORKER_SPEC = importlib.util.spec_from_file_location(
    "gpu_reclamation_worker_targeted_contract_test", SHARED_WORKER_PATH
)
assert SHARED_WORKER_SPEC is not None and SHARED_WORKER_SPEC.loader is not None
SHARED_WORKER = importlib.util.module_from_spec(SHARED_WORKER_SPEC)
sys.modules[SHARED_WORKER_SPEC.name] = SHARED_WORKER
SHARED_WORKER_SPEC.loader.exec_module(SHARED_WORKER)


def _write_json(path: Path, value: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _materialize_exact_historical_binding(
    *, reference: str, destination: Path, expected_sha256: str
) -> None:
    """Copy a bound historical source revision without weakening its digest assertion."""

    payload = (ROOT / reference).read_bytes()
    if (
        reference == "python/sloforge/helix/characterization/gpu_reclamation_v11_methodology.py"
        and expected_sha256 == "4d6c2f00d5f97b6d245bceee1942a30e312dd85cf7d5a26a07153316590ae3c3"
        and hashlib.sha256(payload).hexdigest() != expected_sha256
    ):
        # The successful K seal predates only the kill-arm-specific extension
        # of this model. Reconstruct those exact historical bytes inside the
        # isolated fixture while preserving K's immutable whole-file digest.
        text = payload.decode()
        text = text.replace(
            '_GitObject = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{40}$")]\n',
            "",
        )
        class_start = text.index("\n\nclass Experiment004V11KillRecomputeConfig(")
        class_end = text.index("\n\ndef canonical_json_bytes", class_start)
        historical_class = """

class Experiment004V11KillRecomputeConfig(_StrictModel):
    schema_version: Literal["sloforge.branchfabric.experiment-004-kill-recompute-config/v1"] = (
        "sloforge.branchfabric.experiment-004-kill-recompute-config/v1"
    )
    mode: Literal["KILL_AND_RECOMPUTE"] = "KILL_AND_RECOMPUTE"
    execution_mode: Literal["integrated-kill-recompute-v11"] = "integrated-kill-recompute-v11"
    attempt_id: _Attempt
    seed: int = Field(ge=0, lt=1 << 63)
    model: Literal["Qwen/Qwen2.5-7B-Instruct"] = "Qwen/Qwen2.5-7B-Instruct"
    model_revision: Literal["a09a35458c702b33eeacc393d103063234e8bc28"] = (
        "a09a35458c702b33eeacc393d103063234e8bc28"
    )
    tokenizer_revision: Literal["a09a35458c702b33eeacc393d103063234e8bc28"] = (
        "a09a35458c702b33eeacc393d103063234e8bc28"
    )
    runtime: Literal["vllm"] = "vllm"
    runtime_version: Literal["0.23.0"] = "0.23.0"
    requested_gpu: Literal["A100-80GB"] = "A100-80GB"
    gpu_count: Literal[2] = 2
    prefix_length: Literal[16384] = 16384
    fanout: Literal[8] = 8
    suffix_length: Literal[256] = 256
    continuation_tokens: Literal[8] = 8
    lambda_1_rps: float = Field(default=12.0, ge=12.0, le=12.0)
    lambda_spike_rps: float = Field(default=15.0, ge=15.0, le=15.0)
    lambda_2_rps: float = Field(default=20.0, ge=20.0, le=20.0)
    maximum_wall_seconds: float = Field(default=660.0, ge=660.0, le=660.0)
    v11_integrated_artifact: str = Field(min_length=1)
    v11_integrated_sha256: _Sha256
    budget_authorization: str = Field(min_length=1)
    budget_authorization_sha256: _Sha256
    ledger_sha256_before_reservation: _Sha256
"""
        payload = (text[:class_start] + historical_class + text[class_end:]).encode()
    if (
        reference == "experiments/branchfabric/gpu_reclamation_integrated_worker_v11.py"
        and expected_sha256 == "fbf9ae25a0f80bf3d108e936420c66c9f2a2762d1661ee34b68b7d89bd02f600"
        and hashlib.sha256(payload).hexdigest() != expected_sha256
    ):
        # Attempts G/J bind the exact pre-post-J-calibration wrapper. Recreate
        # those immutable bytes from the current successor solely inside the
        # temporary verifier fixture; the historical assertion stays exact.
        text = payload.decode()
        text = text.replace("_V11_POST_J_SANITY_PROBE_COUNT = 3\n", "")
        text = text.replace(
            "    original = frozen_worker._validate_integrated_transaction_command\n"
            "    original_probe_count = frozen_worker._V10_SANITY_GUARD_COUNT\n"
            "    if original_probe_count != 2:\n"
            '        raise RuntimeError("frozen v10 sanity probe count drifted from two")\n',
            "    original = frozen_worker._validate_integrated_transaction_command\n",
        )
        text = text.replace(
            "    frozen_worker._validate_integrated_transaction_command = "
            "validate_bound_command\n"
            "    frozen_worker._V10_SANITY_GUARD_COUNT = "
            "_V11_POST_J_SANITY_PROBE_COUNT\n",
            "    frozen_worker._validate_integrated_transaction_command = validate_bound_command\n",
        )
        text = text.replace(
            "    finally:\n"
            "        frozen_worker._validate_integrated_transaction_command = original\n"
            "        frozen_worker._V10_SANITY_GUARD_COUNT = original_probe_count\n",
            "    finally:\n"
            "        frozen_worker._validate_integrated_transaction_command = original\n",
            1,
        )
        payload = text.encode()
    if (
        reference == "tools/branchfabric-experiment-004-v11-final.py"
        and expected_sha256 == "35d54a05c64f7f9c82d107e045e421d6e3139c4a976787c924963c22d604449a"
        and hashlib.sha256(payload).hexdigest() != expected_sha256
    ):
        # Attempts I/J bind the immediately preceding coordinator. Attempt K
        # changes only its fresh attempt identity; reproduce the historical
        # content inside the isolated fixture so old evidence stays immutable.
        preimport_bindings_start = payload.index(b"_RETAINED_CAPACITY_PREIMPORT_BINDINGS = {")
        app_name_start = payload.index(b"\nAPP_NAME =", preimport_bindings_start)
        payload = payload[:preimport_bindings_start] + payload[app_name_start + 1 :]
        preimport_function_start = payload.index(
            b"\ndef _require_retained_capacity_preimport_closure("
        )
        verifier_start = payload.index(
            b"\ndef _default_seal_verifier(",
            preimport_function_start,
        )
        payload = payload[:preimport_function_start] + payload[verifier_start:]
        payload = payload.replace(
            b"    _require_retained_capacity_preimport_closure(config, repository_root)\n",
            b"",
            1,
        )
        payload = payload.replace(
            b"exp004-v11-integrated-s41-k",
            b"exp004-v11-integrated-s41-j",
        )
        payload = payload.replace(
            b"    if local_root.exists() or local_root.is_symlink():\n"
            b'        raise FileExistsError(f"immutable integrated result already exists: '
            b'{local_root}")\n',
            b"    if local_root.exists():\n"
            b'        raise FileExistsError(f"immutable integrated result already exists: '
            b'{local_root}")\n',
            1,
        )
        payload = payload.replace(
            b"    if local_root.is_symlink() or not local_root.is_dir():\n"
            b'        raise FileNotFoundError("Modal volume download did not materialize the '
            b'expected prefix")\n',
            b"    if not local_root.is_dir():\n"
            b'        raise FileNotFoundError("Modal volume download did not materialize the '
            b'expected prefix")\n',
            1,
        )
        payload = payload.replace(
            b"    if local_root.is_symlink() or not local_root.is_dir():\n"
            b'        raise ValueError("downloaded integrated artifact root is invalid")\n',
            b"",
            1,
        )
        payload = payload.replace(
            b"        post_prebuild_freshness_path = (\n"
            b'            _PREBUILD_ROOT / f"{config.attempt_id}-pre-reservation-'
            b'freshness.json"\n'
            b"        )\n",
            b"",
            1,
        )
        payload = payload.replace(
            b"        if (\n"
            b"            post_prebuild_freshness_path.exists()\n"
            b"            or post_prebuild_freshness_path.is_symlink()\n"
            b"        ):\n"
            b"            raise FileExistsError(\n"
            b'                "fresh integrated attempt already has post-prebuild freshness '
            b'evidence: "\n'
            b'                f"{post_prebuild_freshness_path}"\n'
            b"            )\n",
            b"",
            1,
        )
        payload = payload.replace(
            b"        if post_prebuild_freshness_path.exists() or "
            b"post_prebuild_freshness_path.is_symlink():\n"
            b"            raise FileExistsError(\n"
            b'                "fresh integrated attempt already has post-prebuild freshness '
            b'evidence: "\n'
            b'                f"{post_prebuild_freshness_path}"\n'
            b"            )\n",
            b"",
            1,
        )
        post_prebuild_start = payload.index(
            b"        _require_local_result_output_fresh(local_root)\n",
            payload.index(b"integrated v11 seal changed during Modal image prebuild"),
        )
        reservation_start = payload.index(
            b"        reservation_id = (\n",
            post_prebuild_start,
        )
        payload = payload[:post_prebuild_start] + payload[reservation_start:]
        payload = payload.replace(
            b"        reservation_id = (\n"
            b'            f"exp004-v11-target-{secrets.token_hex(8)}"\n'
            b"            if targeted\n"
            b"            else (\n"
            b'                f"exp004-v11-kill-{attempt_suffix}-{secrets.token_hex(10)}"\n'
            b'                if getattr(config, "mode", None) == "KILL_AND_RECOMPUTE"\n'
            b'                else f"exp004-v11-{attempt_suffix}-{secrets.token_hex(10)}"\n'
            b"            )\n"
            b"        )\n",
            b"        reservation_id = (\n"
            b'            f"exp004-v11-target-{secrets.token_hex(8)}"\n'
            b"            if targeted\n"
            b'            else f"exp004-v11-{attempt_suffix}-{secrets.token_hex(10)}"\n'
            b"        )\n",
            1,
        )
    if (
        reference == "tests/python/test_experiment_004_v11_final_coordinator.py"
        and expected_sha256 == "db0e6a8459c3830b6b66ea8d4348fb991434df885224f9e7a052e43fba0a2b8c"
        and hashlib.sha256(payload).hexdigest() != expected_sha256
    ):
        payload = (
            payload.replace(
                b"exp004-v11-integrated-s41-k",
                b"exp004-v11-integrated-s41-j",
            )
            .replace(b"status-attempt-k.json", b"status-attempt-j.json")
            .replace(
                b"provider-cleanup-attempt-k.json",
                b"provider-cleanup-attempt-j.json",
            )
        )
        preimport_test_start = payload.index(
            b"\n\ndef test_attempt_k_preimport_closure_is_exact_and_precedes_controller_import("
        )
        next_coordinator_test = payload.index(
            b'\n\n@pytest.mark.parametrize(\n    "collision",',
            preimport_test_start,
        )
        payload = payload[:preimport_test_start] + payload[next_coordinator_test:]
        download_hardening_start = payload.index(
            b'\n    dangling = tmp_path / "raw" / "dangling-attempt"\n'
        )
        next_download_test = payload.index(
            b"\n\ndef test_manifest_tamper_and_duplicate_result_are_rejected",
            download_hardening_start,
        )
        payload = payload[:download_hardening_start] + payload[next_download_test:]
        freshness_test_start = payload.index(
            b"\n\ndef test_attempt_k_rechecks_remote_and_local_freshness_after_prebuild("
        )
        next_freshness_test = payload.index(
            b"\n\ndef test_attempt_i_remote_prefix_freshness_is_fail_closed",
            freshness_test_start,
        )
        payload = payload[:freshness_test_start] + payload[next_freshness_test:]
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise AssertionError(f"historical binding bytes unavailable for {reference}")
    destination.write_bytes(payload)


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
        "frozen_calibration_worker",
        "capacity_worker",
        "optimized_state_worker",
        "allocator_epochs",
        "source_identity",
        "controller",
        "launcher",
        "paid_coordinator",
        "trigger",
        "worker_test",
        "frozen_calibration_worker_test",
        "capacity_worker_test",
        "optimized_state_worker_test",
        "allocator_epochs_test",
        "source_identity_test",
        "controller_test",
        "launcher_test",
        "paid_coordinator_test",
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


def _targeted_fixture(root: Path) -> Experiment004V11TargetedSourceIdentityConfig:
    base = _sealed_fixture(root)
    attempt_id = "exp004-v11-targeted-identity-s41-a"
    bindings: list[dict[str, str]] = []
    for index in range(4):
        reference = f"targeted/offline-binding-{index}.json"
        digest = _write_json(
            root / reference,
            {"status": "PASS", "binding": index, "attempt_id": attempt_id},
        )
        bindings.append({"artifact": reference, "sha256": digest})
    post_path = root / base.post_micro_review_manifest
    post = json.loads(post_path.read_text())
    post["targeted_source_identity_authorization"] = {
        "status": "APPROVED_EXACTLY_ONCE",
        "attempt_id": attempt_id,
        "prior_failed_attempt_id": CONTROLLER.REPLACEMENT_INTEGRATED_ATTEMPT_ID,
        "root_cause_class": "GATE_BUG",
        "terminal_phase": "SOURCE_ALLOCATION_IDENTITY_GATE",
        "full_export_authorized": False,
        "source_release_authorized": False,
        "ledger_sha256_before_reservation": base.ledger_sha256_before_reservation,
        "offline_bindings": bindings,
    }
    post_sha = _write_json(post_path, post)
    payload = base.model_dump(mode="python")
    payload.update(
        {
            "schema_version": (
                "sloforge.branchfabric.experiment-004-v11-targeted-source-identity-config/v1"
            ),
            "execution_mode": "targeted-source-identity-v11",
            "attempt_id": attempt_id,
            "terminal_phase": "SOURCE_ALLOCATION_IDENTITY_GATE",
            "full_export_authorized": False,
            "source_release_authorized": False,
            "maximum_wall_seconds": 300.0,
            "post_micro_review_manifest_sha256": post_sha,
        }
    )
    return Experiment004V11TargetedSourceIdentityConfig.model_validate(payload, strict=True)


def _targeted_retry_fixture(root: Path) -> Experiment004V11TargetedSourceIdentityConfig:
    initial = _targeted_fixture(root)
    attempt_id = CONTROLLER.RETRY_TARGETED_IDENTITY_ATTEMPT_ID
    function_call_id = "fc-targeted-attempt-a"
    status_reference = CONTROLLER.PRIOR_TARGETED_RETRY_STATUS
    cleanup_reference = CONTROLLER.PRIOR_TARGETED_RETRY_CLEANUP
    in_function_cleanup_sha = _write_json(
        root / CONTROLLER.PRIOR_TARGETED_RETRY_IN_FUNCTION_CLEANUP,
        {"status": "PASS", "pass": True},
    )
    status_sha = _write_json(
        root / status_reference,
        {
            "attempt_id": CONTROLLER.INITIAL_TARGETED_IDENTITY_ATTEMPT_ID,
            "status": "FAIL_PRE_IDENTITY_COMMAND_SCHEMA",
            "source_identity_gate_reached": False,
            "source_identity_gate_passed": None,
            "optimized_export_started": False,
            "transaction_source_release_started": False,
            "blocker": {"code": "TARGETED_COMMAND_SCHEMA_REJECTED_BY_SHARED_WORKER_VALIDATOR"},
            "cleanup": {"in_function": "PASS", "provider": "PASS"},
            "measured_invocation": {"function_call_id": function_call_id},
        },
    )
    cleanup_sha = _write_json(
        root / cleanup_reference,
        {
            "status": "PASS",
            "attempt_id": CONTROLLER.INITIAL_TARGETED_IDENTITY_ATTEMPT_ID,
            "function_call_id": function_call_id,
            "provider_observation": {
                "active_apps": 0,
                "running_tasks": 0,
                "running_containers": 0,
                "endpoints": 0,
                "provider_reservations": 0,
            },
            "ledger": {
                "active_reservations": 0,
                "sha256_after_settlement": initial.ledger_sha256_before_reservation,
            },
            "in_function_cleanup": {
                "status": "PASS",
                "artifact": ("raw/exp004-v11-targeted-identity-s41-a/in_function_cleanup.json"),
                "sha256": in_function_cleanup_sha,
            },
        },
    )
    post_path = root / initial.post_micro_review_manifest
    post = json.loads(post_path.read_text())
    authorization = post["targeted_source_identity_authorization"]
    authorization.update(
        {
            "attempt_id": attempt_id,
            "prior_failed_attempt_id": CONTROLLER.INITIAL_TARGETED_IDENTITY_ATTEMPT_ID,
            "prior_targeted_attempt_evidence": {
                "status": {"artifact": status_reference, "sha256": status_sha},
                "cleanup": {"artifact": cleanup_reference, "sha256": cleanup_sha},
            },
        }
    )
    post_sha = _write_json(post_path, post)
    return initial.model_copy(
        update={"attempt_id": attempt_id, "post_micro_review_manifest_sha256": post_sha}
    )


def _post_target_integrated_fixture(root: Path) -> Experiment004V11IntegratedConfig:
    base = _sealed_fixture(root)
    evidence_bindings: dict[str, dict[str, str]] = {}
    for label, reference in CONTROLLER.POST_TARGET_EVIDENCE_BINDINGS.items():
        destination = root / reference
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / reference, destination)
        evidence_bindings[label] = {
            "artifact": reference,
            "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
        }
    review_bindings: list[dict[str, Any]] = []
    for agent, (role, reference) in CONTROLLER.POST_TARGET_REVIEW_BINDINGS.items():
        destination = root / reference
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / reference, destination)
        review_bindings.append(
            {
                "agent": agent,
                "role": role,
                "status": "PASS",
                "artifact": reference,
                "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
            }
        )
    post_path = root / base.post_micro_review_manifest
    post = json.loads(post_path.read_text())
    post["post_target_integrated_authorization"] = {
        "schema_version": (
            "sloforge.branchfabric.exp004-v11-post-target-integrated-authorization/v1"
        ),
        "status": "APPROVED_EXACTLY_ONCE",
        "integrated_attempt_id": CONTROLLER.POST_TARGET_INTEGRATED_ATTEMPT_ID,
        "targeted_attempt_id": CONTROLLER.RETRY_TARGETED_IDENTITY_ATTEMPT_ID,
        "ledger_sha256_before_reservation": CONTROLLER.POST_TARGET_LEDGER_SHA256,
        "targeted_attempt_evidence": evidence_bindings,
        "target_reviews": review_bindings,
    }
    post_sha = _write_json(post_path, post)
    return base.model_copy(
        update={
            "attempt_id": CONTROLLER.POST_TARGET_INTEGRATED_ATTEMPT_ID,
            "ledger_sha256_before_reservation": CONTROLLER.POST_TARGET_LEDGER_SHA256,
            "post_micro_review_manifest_sha256": post_sha,
        }
    )


def _successful_integrated_contract() -> dict[str, Any]:
    return {
        "config_schema_version": ("sloforge.branchfabric.experiment-004-v11-integrated-config/v1"),
        "execution_mode": "integrated-reclamation-v11",
        "completion_schema_version": (
            "sloforge.branchfabric.experiment-004-v11-integrated-completion/v1"
        ),
        "remote_manifest_schema_version": (
            "sloforge.branchfabric.experiment-004-v11-integrated-remote-manifest/v1"
        ),
        "terminal_phase": "INTEGRATED_TRANSACTION",
        "remote_status": "provisional",
        "scientific_status": "pending-local-budget-settlement-and-provider-cleanup",
        "result_envelope_fields": ["materialized", "result"],
        "remote_completion_fields": sorted(CONTROLLER.SUCCESSFUL_INTEGRATED_COMPLETION_FIELDS),
        "materialized_fields": ["remote_path", "volume_name"],
        "run_error": None,
        "in_function_cleanup": "PASS",
        "full_export_authorized": True,
        "source_release_authorized": True,
    }


def _prefunction_retry_integrated_fixture(root: Path) -> Experiment004V11IntegratedConfig:
    base = _sealed_fixture(root)
    evidence_bindings: dict[str, dict[str, str]] = {}
    for label, reference in CONTROLLER.PREFUNCTION_RETRY_EVIDENCE_BINDINGS.items():
        destination = root / reference
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / reference, destination)
        evidence_bindings[label] = {
            "artifact": reference,
            "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
        }

    budget_reference = CONTROLLER.PREFUNCTION_RETRY_BUDGET_AUTHORIZATION
    budget_path = root / budget_reference
    budget_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / budget_reference, budget_path)
    ledger_reference = "artifacts/branchfabric/gpu-validation/experiment-004/gpu-hours.json"
    ledger_path = root / ledger_reference
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    ledger = json.loads((ROOT / ledger_reference).read_text())
    attempt_c_charges = [
        row
        for row in ledger["conservative_failure_charges"]
        if row.get("invocation_id") == "exp004-v11-kill-recompute-s41-c"
    ]
    assert len(attempt_c_charges) == 1
    assert attempt_c_charges[0] == {
        "charged_wall_seconds": 660.0,
        "config_sha256": "e139a62ae055c5c749193628e9bb881005d44c009744c1a446d67385a2952c0b",
        "failure_evidence": {
            "artifact_reference": str(
                ROOT / "artifacts/branchfabric/gpu-validation/experiment-004/v11-final/"
                "kill-recompute/failures/"
                "exp004-v11-kill-recompute-s41-c-conservative-charge.json"
            ),
            "artifact_sha256": ("08c9b0c3cd50b58b81b170d8de5ec47141b3c3d3b4c8e1d7315c862ba6e68bb0"),
            "sample_selector": "$",
        },
        "failure_stage": "result-validation",
        "gpu_count": 2,
        "gpu_price_per_hour_usd": 2.4984,
        "invocation_id": "exp004-v11-kill-recompute-s41-c",
        "requested_gpu": "A100-80GB",
        "reservation_id": "exp004-v11-kill-c-c580c2896a62212fb83f",
    }
    attempt_e_charges = [
        row
        for row in ledger["conservative_failure_charges"]
        if row.get("invocation_id") == "exp004-v11-kill-recompute-s41-e"
    ]
    assert len(attempt_e_charges) == 1
    assert attempt_e_charges[0] == {
        "charged_wall_seconds": 660.0,
        "config_sha256": "9622b798da11ccf5090680c7d10f26841946d881bc62dfa0583043ba8b6eac78",
        "failure_evidence": {
            "artifact_reference": str(
                ROOT / "artifacts/branchfabric/gpu-validation/experiment-004/v11-final/"
                "kill-recompute/failures/"
                "exp004-v11-kill-recompute-s41-e-conservative-charge.json"
            ),
            "artifact_sha256": ("5bc8f8798e81af9b06634358574147c5c8930b322c284fecba4bef6603602e47"),
            "sample_selector": "$",
        },
        "failure_stage": "modal-launch",
        "gpu_count": 2,
        "gpu_price_per_hour_usd": 2.4984,
        "invocation_id": "exp004-v11-kill-recompute-s41-e",
        "requested_gpu": "A100-80GB",
        "reservation_id": "exp004-v11-kill-e-a2132b54b248c90b8385",
    }
    attempt_f_charges = [
        row
        for row in ledger["conservative_failure_charges"]
        if row.get("invocation_id") == "exp004-v11-kill-recompute-s41-f"
    ]
    assert len(attempt_f_charges) == 1
    assert attempt_f_charges[0] == {
        "charged_wall_seconds": 660.0,
        "config_sha256": "0880d14f78a16108c53f72cd3f0669a9ffd6ccb15d90869568c3f669cff0e865",
        "failure_evidence": {
            "artifact_reference": str(
                ROOT / "artifacts/branchfabric/gpu-validation/experiment-004/v11-final/"
                "kill-recompute/failures/"
                "exp004-v11-kill-recompute-s41-f-conservative-charge.json"
            ),
            "artifact_sha256": ("38905e7cbd92ced0ea523a001954135155c2451140beb861b41a31a417aec174"),
            "sample_selector": "$",
        },
        "failure_stage": "modal-launch",
        "gpu_count": 2,
        "gpu_price_per_hour_usd": 2.4984,
        "invocation_id": "exp004-v11-kill-recompute-s41-f",
        "requested_gpu": "A100-80GB",
        "reservation_id": "exp004-v11-kill-f-21196ad66e1c8dfa16a0",
    }
    ledger["intervals"] = [
        row
        for row in ledger["intervals"]
        if row["invocation_id"]
        not in {
            CONTROLLER.BUNDLED_LEDGER_RETRY_INTEGRATED_ATTEMPT_ID,
            CONTROLLER.QUEUE_ACCOUNTING_RETRY_INTEGRATED_ATTEMPT_ID,
            CONTROLLER.RETAINED_CAPACITY_RETRY_INTEGRATED_ATTEMPT_ID,
        }
    ]
    ledger["conservative_failure_charges"] = [
        row
        for row in ledger["conservative_failure_charges"]
        if row["invocation_id"]
        not in {
            CONTROLLER.BUNDLED_LEDGER_FAILED_INTEGRATED_ATTEMPT_ID,
            CONTROLLER.CONTROL_GATE_RETRY_INTEGRATED_ATTEMPT_ID,
            CONTROLLER.REMOTE_PREFIX_RETRY_INTEGRATED_ATTEMPT_ID,
            "exp004-v11-kill-recompute-s41-b",
            "exp004-v11-kill-recompute-s41-c",
            "exp004-v11-kill-recompute-s41-e",
            "exp004-v11-kill-recompute-s41-f",
        }
    ]
    ledger["consumed_additional_gpu_seconds"] = 19397.975013231975
    ledger_path.write_text(json.dumps(ledger, indent=2) + "\n")
    assert hashlib.sha256(ledger_path.read_bytes()).hexdigest() == (
        CONTROLLER.PREFUNCTION_RETRY_LEDGER_SHA256
    )

    post_path = root / base.post_micro_review_manifest
    post = json.loads(post_path.read_text())
    post["post_prefunction_failure_integrated_authorization"] = {
        "schema_version": (
            "sloforge.branchfabric.exp004-v11-post-prefunction-failure-integrated-authorization/v1"
        ),
        "status": "APPROVED_EXACTLY_ONCE",
        "retry_attempt_id": CONTROLLER.PREFUNCTION_RETRY_INTEGRATED_ATTEMPT_ID,
        "prior_failed_attempt_id": CONTROLLER.PREFUNCTION_FAILED_INTEGRATED_ATTEMPT_ID,
        "prior_settled_ledger_sha256": CONTROLLER.PREFUNCTION_FAILED_LEDGER_SHA256,
        "ledger_sha256_before_reservation": CONTROLLER.PREFUNCTION_RETRY_LEDGER_SHA256,
        "prior_attempt_evidence": evidence_bindings,
        "expected_successful_integrated_contract": _successful_integrated_contract(),
    }
    post_sha = _write_json(post_path, post)
    return base.model_copy(
        update={
            "attempt_id": CONTROLLER.PREFUNCTION_RETRY_INTEGRATED_ATTEMPT_ID,
            "ledger_sha256_before_reservation": CONTROLLER.PREFUNCTION_RETRY_LEDGER_SHA256,
            "budget_authorization": budget_reference,
            "budget_authorization_sha256": hashlib.sha256(budget_path.read_bytes()).hexdigest(),
            "post_micro_review_manifest_sha256": post_sha,
        }
    )


def _pre_worker_cleanup_contract() -> dict[str, Any]:
    return {
        "cleanup_schema_version": "sloforge.branchfabric.in-function-cleanup/v1",
        "cleanup_scope": "PRE_WORKER_PREFLIGHT",
        "allowed_failure_stages": [
            "SEALED_EVIDENCE",
            "LIVE_CONTRACT",
            "CPU_CONTROL_IMPORTS",
            "PRE_WORKER_ARTIFACT_INITIALIZATION",
            "PRE_INVENTORY_CUDA_CLEAN_AUDIT",
            "GPU_INVENTORY_BEFORE_WORKERS",
            "ZERO_COMPUTE_BEFORE_WORKERS",
        ],
        "cleanup_pass_required": True,
        "empty_worker_ownership_required": True,
        "empty_runtime_evidence_required": True,
        "zero_compute_processes_required": True,
        "cuda_clean_parent_required": True,
        "controller_error_required": True,
        "cleanup_error": None,
        "cleanup_actions": [],
        "stable_physical_gpu_identity": False,
        "readiness_deadline_ns": None,
        "sanity_guard_pair": None,
        "cuda_clean_import_audit_count": 1,
    }


def _bundled_ledger_retry_integrated_fixture(root: Path) -> Experiment004V11IntegratedConfig:
    base = _post_target_integrated_fixture(root)
    evidence_bindings: dict[str, dict[str, str]] = {}
    for label, (
        reference,
        expected_sha256,
    ) in CONTROLLER.BUNDLED_LEDGER_RETRY_EVIDENCE_BINDINGS.items():
        destination = root / reference
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / reference, destination)
        assert hashlib.sha256(destination.read_bytes()).hexdigest() == expected_sha256
        evidence_bindings[label] = {
            "artifact": reference,
            "sha256": expected_sha256,
        }

    snapshot_reference, snapshot_sha256 = CONTROLLER.BUNDLED_LEDGER_SNAPSHOT_BINDING
    snapshot_path = root / snapshot_reference
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / snapshot_reference, snapshot_path)
    assert hashlib.sha256(snapshot_path.read_bytes()).hexdigest() == snapshot_sha256

    target_reference, target_sha256 = CONTROLLER.BUNDLED_LEDGER_TARGET_B_MANIFEST_BINDING
    target_path = root / target_reference
    target_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / target_reference, target_path)
    assert hashlib.sha256(target_path.read_bytes()).hexdigest() == target_sha256

    budget_reference = CONTROLLER.PREFUNCTION_RETRY_BUDGET_AUTHORIZATION
    budget_path = root / budget_reference
    budget_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / budget_reference, budget_path)

    post_path = root / base.post_micro_review_manifest
    post = json.loads(post_path.read_text())
    post["post_remote_precontroller_failure_integrated_authorization"] = {
        "schema_version": (
            "sloforge.branchfabric.exp004-v11-post-remote-precontroller-failure-"
            "integrated-authorization/v1"
        ),
        "status": "APPROVED_EXACTLY_ONCE",
        "retry_attempt_id": CONTROLLER.BUNDLED_LEDGER_RETRY_INTEGRATED_ATTEMPT_ID,
        "prior_failed_attempt_id": CONTROLLER.BUNDLED_LEDGER_FAILED_INTEGRATED_ATTEMPT_ID,
        "prior_settled_ledger_sha256": CONTROLLER.BUNDLED_LEDGER_RETRY_LEDGER_SHA256,
        "ledger_sha256_before_reservation": CONTROLLER.BUNDLED_LEDGER_RETRY_LEDGER_SHA256,
        "immutable_ledger_snapshot": {
            "artifact": snapshot_reference,
            "sha256": snapshot_sha256,
        },
        "prior_attempt_evidence": evidence_bindings,
        "unchanged_target_b_authorization": {
            "artifact": target_reference,
            "sha256": target_sha256,
        },
        "expected_successful_integrated_contract": _successful_integrated_contract(),
        "required_pre_worker_cleanup_contract": _pre_worker_cleanup_contract(),
    }
    post_sha = _write_json(post_path, post)
    return base.model_copy(
        update={
            "attempt_id": CONTROLLER.BUNDLED_LEDGER_RETRY_INTEGRATED_ATTEMPT_ID,
            "ledger_sha256_before_reservation": CONTROLLER.BUNDLED_LEDGER_RETRY_LEDGER_SHA256,
            "budget_authorization": budget_reference,
            "budget_authorization_sha256": hashlib.sha256(budget_path.read_bytes()).hexdigest(),
            "post_micro_review_manifest_sha256": post_sha,
        }
    )


def _queue_accounting_retry_integrated_fixture(root: Path) -> Experiment004V11IntegratedConfig:
    base = _post_target_integrated_fixture(root)

    evidence_bindings: dict[str, dict[str, str]] = {}
    for label, (
        reference,
        expected_sha256,
    ) in CONTROLLER.QUEUE_ACCOUNTING_RETRY_EVIDENCE_BINDINGS.items():
        destination = root / reference
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / reference, destination)
        assert hashlib.sha256(destination.read_bytes()).hexdigest() == expected_sha256
        evidence_bindings[label] = {"artifact": reference, "sha256": expected_sha256}

    snapshot_reference, snapshot_sha256 = CONTROLLER.QUEUE_ACCOUNTING_RETRY_LEDGER_SNAPSHOT_BINDING
    snapshot_path = root / snapshot_reference
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / snapshot_reference, snapshot_path)
    assert hashlib.sha256(snapshot_path.read_bytes()).hexdigest() == snapshot_sha256

    def copy_bindings(
        expected: dict[str, tuple[str, str]],
    ) -> dict[str, dict[str, str]]:
        copied: dict[str, dict[str, str]] = {}
        for label, (reference, digest) in expected.items():
            destination = root / reference
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / reference, destination)
            assert hashlib.sha256(destination.read_bytes()).hexdigest() == digest
            copied[label] = {"artifact": reference, "sha256": digest}
        return copied

    fix_bindings = copy_bindings(CONTROLLER.QUEUE_ACCOUNTING_FIX_BINDINGS)
    unchanged_bindings = copy_bindings(CONTROLLER.QUEUE_ACCOUNTING_UNCHANGED_RUNTIME_BINDINGS)
    review_bindings: list[dict[str, Any]] = []
    for agent, (
        role,
        reference,
        digest,
    ) in CONTROLLER.QUEUE_ACCOUNTING_RETRY_REVIEW_BINDINGS.items():
        destination = root / reference
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / reference, destination)
        assert hashlib.sha256(destination.read_bytes()).hexdigest() == digest
        review_bindings.append(
            {
                "agent": agent,
                "role": role,
                "status": "PASS",
                "artifact": reference,
                "sha256": digest,
            }
        )

    budget_reference = CONTROLLER.PREFUNCTION_RETRY_BUDGET_AUTHORIZATION
    budget_path = root / budget_reference
    budget_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / budget_reference, budget_path)

    post_path = root / base.post_micro_review_manifest
    post = json.loads(post_path.read_text())
    # Exact Attempt-G sources replace the lightweight fixture placeholders at
    # several prelaunch-bound paths. Recommit those final bytes before sealing.
    for binding in post["integrated_prelaunch"]["bindings"]:
        binding["sha256"] = hashlib.sha256((root / binding["artifact"]).read_bytes()).hexdigest()
    post["post_queue_accounting_failure_integrated_authorization"] = {
        "schema_version": (
            "sloforge.branchfabric.exp004-v11-post-queue-accounting-failure-"
            "integrated-authorization/v1"
        ),
        "status": "APPROVED_EXACTLY_ONCE",
        "retry_attempt_id": CONTROLLER.QUEUE_ACCOUNTING_RETRY_INTEGRATED_ATTEMPT_ID,
        "prior_failed_attempt_id": CONTROLLER.QUEUE_ACCOUNTING_FAILED_INTEGRATED_ATTEMPT_ID,
        "prior_settled_ledger_sha256": CONTROLLER.QUEUE_ACCOUNTING_RETRY_LEDGER_SHA256,
        "ledger_sha256_before_reservation": CONTROLLER.QUEUE_ACCOUNTING_RETRY_LEDGER_SHA256,
        "immutable_ledger_snapshot": {
            "artifact": snapshot_reference,
            "sha256": snapshot_sha256,
        },
        "prior_attempt_evidence": evidence_bindings,
        "queue_accounting_fix_bindings": fix_bindings,
        "unchanged_runtime_bindings": unchanged_bindings,
        "independent_reviews": review_bindings,
        "expected_successful_integrated_contract": _successful_integrated_contract(),
    }
    post_sha = _write_json(post_path, post)
    return base.model_copy(
        update={
            "attempt_id": CONTROLLER.QUEUE_ACCOUNTING_RETRY_INTEGRATED_ATTEMPT_ID,
            "ledger_sha256_before_reservation": CONTROLLER.QUEUE_ACCOUNTING_RETRY_LEDGER_SHA256,
            "budget_authorization": budget_reference,
            "budget_authorization_sha256": hashlib.sha256(budget_path.read_bytes()).hexdigest(),
            "post_micro_review_manifest_sha256": post_sha,
        }
    )


def _control_gate_retry_integrated_fixture(root: Path) -> Experiment004V11IntegratedConfig:
    base = _post_target_integrated_fixture(root)
    raw_reference_root = (
        ROOT
        / "artifacts/branchfabric/gpu-validation/experiment-004/v11-final/integrated/raw"
        / CONTROLLER.CONTROL_GATE_FAILED_INTEGRATED_ATTEMPT_ID
    )
    raw_destination_root = (
        root
        / "artifacts/branchfabric/gpu-validation/experiment-004/v11-final/integrated/raw"
        / CONTROLLER.CONTROL_GATE_FAILED_INTEGRATED_ATTEMPT_ID
    )
    shutil.copytree(raw_reference_root, raw_destination_root)

    def copy_bindings(
        expected: dict[str, tuple[str, str]],
    ) -> dict[str, dict[str, str]]:
        copied: dict[str, dict[str, str]] = {}
        for label, (reference, digest) in expected.items():
            destination = root / reference
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not reference.startswith(
                "artifacts/branchfabric/gpu-validation/experiment-004/v11-final/integrated/raw/"
            ):
                _materialize_exact_historical_binding(
                    reference=reference,
                    destination=destination,
                    expected_sha256=digest,
                )
            assert hashlib.sha256(destination.read_bytes()).hexdigest() == digest
            copied[label] = {"artifact": reference, "sha256": digest}
        return copied

    evidence_bindings = copy_bindings(CONTROLLER.CONTROL_GATE_RETRY_EVIDENCE_BINDINGS)
    baseline_bindings = copy_bindings(CONTROLLER.CONTROL_GATE_FROZEN_V10_BINDINGS)
    fix_bindings = copy_bindings(CONTROLLER.CONTROL_GATE_FIX_BINDINGS)
    unchanged_bindings = copy_bindings(CONTROLLER.CONTROL_GATE_UNCHANGED_RUNTIME_BINDINGS)
    snapshot_reference, snapshot_sha = CONTROLLER.CONTROL_GATE_RETRY_LEDGER_SNAPSHOT_BINDING
    snapshot_path = root / snapshot_reference
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / snapshot_reference, snapshot_path)
    assert hashlib.sha256(snapshot_path.read_bytes()).hexdigest() == snapshot_sha

    review_bindings: list[dict[str, Any]] = []
    for agent, (role, reference, digest) in CONTROLLER.CONTROL_GATE_RETRY_REVIEW_BINDINGS.items():
        destination = root / reference
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / reference, destination)
        assert hashlib.sha256(destination.read_bytes()).hexdigest() == digest
        review_bindings.append(
            {
                "agent": agent,
                "role": role,
                "status": "PASS",
                "artifact": reference,
                "sha256": digest,
            }
        )

    budget_reference = CONTROLLER.PREFUNCTION_RETRY_BUDGET_AUTHORIZATION
    budget_path = root / budget_reference
    budget_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / budget_reference, budget_path)
    post_path = root / base.post_micro_review_manifest
    post = json.loads(post_path.read_text())
    for binding in post["integrated_prelaunch"]["bindings"]:
        binding["sha256"] = hashlib.sha256((root / binding["artifact"]).read_bytes()).hexdigest()
    post["post_control_gate_failure_integrated_authorization"] = {
        "schema_version": (
            "sloforge.branchfabric.exp004-v11-post-control-gate-failure-integrated-authorization/v1"
        ),
        "status": "APPROVED_EXACTLY_ONCE",
        "retry_attempt_id": CONTROLLER.CONTROL_GATE_RETRY_INTEGRATED_ATTEMPT_ID,
        "prior_failed_attempt_id": CONTROLLER.CONTROL_GATE_FAILED_INTEGRATED_ATTEMPT_ID,
        "prior_settled_ledger_sha256": CONTROLLER.CONTROL_GATE_RETRY_LEDGER_SHA256,
        "ledger_sha256_before_reservation": CONTROLLER.CONTROL_GATE_RETRY_LEDGER_SHA256,
        "immutable_ledger_snapshot": {
            "artifact": snapshot_reference,
            "sha256": snapshot_sha,
        },
        "prior_attempt_evidence": evidence_bindings,
        "frozen_v10_control_bindings": baseline_bindings,
        "control_gate_fix_bindings": fix_bindings,
        "unchanged_runtime_bindings": unchanged_bindings,
        "independent_reviews": review_bindings,
        "expected_successful_integrated_contract": _successful_integrated_contract(),
    }
    post_sha = _write_json(post_path, post)
    return base.model_copy(
        update={
            "attempt_id": CONTROLLER.CONTROL_GATE_RETRY_INTEGRATED_ATTEMPT_ID,
            "ledger_sha256_before_reservation": CONTROLLER.CONTROL_GATE_RETRY_LEDGER_SHA256,
            "budget_authorization": budget_reference,
            "budget_authorization_sha256": hashlib.sha256(budget_path.read_bytes()).hexdigest(),
            "post_micro_review_manifest_sha256": post_sha,
        }
    )


def _image_closure_retry_integrated_fixture(root: Path) -> Experiment004V11IntegratedConfig:
    base = _control_gate_retry_integrated_fixture(root)
    raw_reference_root = (
        ROOT
        / "artifacts/branchfabric/gpu-validation/experiment-004/v11-final/integrated/raw"
        / CONTROLLER.IMAGE_CLOSURE_FAILED_INTEGRATED_ATTEMPT_ID
    )
    raw_destination_root = (
        root
        / "artifacts/branchfabric/gpu-validation/experiment-004/v11-final/integrated/raw"
        / CONTROLLER.IMAGE_CLOSURE_FAILED_INTEGRATED_ATTEMPT_ID
    )
    shutil.copytree(raw_reference_root, raw_destination_root)

    def copy_bindings(
        expected: dict[str, tuple[str, str]],
    ) -> dict[str, dict[str, str]]:
        copied: dict[str, dict[str, str]] = {}
        raw_prefix = (
            "artifacts/branchfabric/gpu-validation/experiment-004/v11-final/integrated/raw/"
        )
        for label, (reference, digest) in expected.items():
            destination = root / reference
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not reference.startswith(raw_prefix):
                _materialize_exact_historical_binding(
                    reference=reference,
                    destination=destination,
                    expected_sha256=digest,
                )
            assert hashlib.sha256(destination.read_bytes()).hexdigest() == digest
            copied[label] = {"artifact": reference, "sha256": digest}
        return copied

    evidence_bindings = copy_bindings(CONTROLLER.IMAGE_CLOSURE_RETRY_EVIDENCE_BINDINGS)
    fix_bindings = copy_bindings(CONTROLLER.IMAGE_CLOSURE_FIX_BINDINGS)
    unchanged_bindings = copy_bindings(
        CONTROLLER.IMAGE_CLOSURE_UNCHANGED_ALLOCATOR_MOVEMENT_BINDINGS
    )
    target_b_bindings = copy_bindings(CONTROLLER.IMAGE_CLOSURE_TARGET_B_EVIDENCE_BINDINGS)
    snapshot_reference, snapshot_sha = CONTROLLER.IMAGE_CLOSURE_RETRY_LEDGER_SNAPSHOT_BINDING
    snapshot_path = root / snapshot_reference
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / snapshot_reference, snapshot_path)
    assert hashlib.sha256(snapshot_path.read_bytes()).hexdigest() == snapshot_sha

    review_bindings: list[dict[str, Any]] = []
    for agent, (role, reference, digest) in CONTROLLER.IMAGE_CLOSURE_RETRY_REVIEW_BINDINGS.items():
        destination = root / reference
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / reference, destination)
        assert hashlib.sha256(destination.read_bytes()).hexdigest() == digest
        review_bindings.append(
            {
                "agent": agent,
                "role": role,
                "status": "PASS",
                "artifact": reference,
                "sha256": digest,
            }
        )

    budget_reference = CONTROLLER.PREFUNCTION_RETRY_BUDGET_AUTHORIZATION
    budget_path = root / budget_reference
    budget_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / budget_reference, budget_path)
    post_path = root / base.post_micro_review_manifest
    post = json.loads(post_path.read_text())
    for binding in post["integrated_prelaunch"]["bindings"]:
        binding["sha256"] = hashlib.sha256((root / binding["artifact"]).read_bytes()).hexdigest()
    post["post_image_closure_failure_integrated_authorization"] = {
        "schema_version": (
            "sloforge.branchfabric.exp004-v11-post-image-closure-failure-"
            "integrated-authorization/v1"
        ),
        "status": "APPROVED_EXACTLY_ONCE",
        "retry_attempt_id": CONTROLLER.IMAGE_CLOSURE_RETRY_INTEGRATED_ATTEMPT_ID,
        "prior_failed_attempt_id": CONTROLLER.IMAGE_CLOSURE_FAILED_INTEGRATED_ATTEMPT_ID,
        "prior_settled_ledger_sha256": CONTROLLER.IMAGE_CLOSURE_RETRY_LEDGER_SHA256,
        "ledger_sha256_before_reservation": CONTROLLER.IMAGE_CLOSURE_RETRY_LEDGER_SHA256,
        "immutable_ledger_snapshot": {
            "artifact": snapshot_reference,
            "sha256": snapshot_sha,
        },
        "prior_attempt_evidence": evidence_bindings,
        "image_closure_fix_bindings": fix_bindings,
        "unchanged_allocator_movement_bindings": unchanged_bindings,
        "target_b_identity_evidence": target_b_bindings,
        "independent_reviews": review_bindings,
        "expected_successful_integrated_contract": _successful_integrated_contract(),
    }
    post_sha = _write_json(post_path, post)
    return base.model_copy(
        update={
            "attempt_id": CONTROLLER.IMAGE_CLOSURE_RETRY_INTEGRATED_ATTEMPT_ID,
            "ledger_sha256_before_reservation": CONTROLLER.IMAGE_CLOSURE_RETRY_LEDGER_SHA256,
            "budget_authorization": budget_reference,
            "budget_authorization_sha256": hashlib.sha256(budget_path.read_bytes()).hexdigest(),
            "post_micro_review_manifest_sha256": post_sha,
        }
    )


def _remote_prefix_retry_integrated_fixture(root: Path) -> Experiment004V11IntegratedConfig:
    base = _control_gate_retry_integrated_fixture(root)

    def copy_bindings(
        expected: dict[str, tuple[str, str]],
    ) -> dict[str, dict[str, str]]:
        copied: dict[str, dict[str, str]] = {}
        for label, (reference, digest) in expected.items():
            destination = root / reference
            destination.parent.mkdir(parents=True, exist_ok=True)
            _materialize_exact_historical_binding(
                reference=reference,
                destination=destination,
                expected_sha256=digest,
            )
            assert hashlib.sha256(destination.read_bytes()).hexdigest() == digest
            copied[label] = {"artifact": reference, "sha256": digest}
        return copied

    evidence_bindings = copy_bindings(CONTROLLER.REMOTE_PREFIX_RETRY_EVIDENCE_BINDINGS)
    fix_bindings = copy_bindings(CONTROLLER.REMOTE_PREFIX_PARSER_FIX_BINDINGS)
    unchanged_bindings = copy_bindings(CONTROLLER.REMOTE_PREFIX_UNCHANGED_RUNTIME_BINDINGS)
    target_b_bindings = copy_bindings(CONTROLLER.IMAGE_CLOSURE_TARGET_B_EVIDENCE_BINDINGS)
    snapshot_reference, snapshot_sha = CONTROLLER.REMOTE_PREFIX_RETRY_LEDGER_SNAPSHOT_BINDING
    snapshot_path = root / snapshot_reference
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / snapshot_reference, snapshot_path)
    assert hashlib.sha256(snapshot_path.read_bytes()).hexdigest() == snapshot_sha

    budget_reference = CONTROLLER.PREFUNCTION_RETRY_BUDGET_AUTHORIZATION
    budget_path = root / budget_reference
    budget_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / budget_reference, budget_path)
    post_path = root / base.post_micro_review_manifest
    post = json.loads(post_path.read_text())
    for binding in post["integrated_prelaunch"]["bindings"]:
        binding["sha256"] = hashlib.sha256((root / binding["artifact"]).read_bytes()).hexdigest()
    post["post_remote_prefix_parser_failure_integrated_authorization"] = {
        "schema_version": (
            "sloforge.branchfabric.exp004-v11-post-remote-prefix-parser-failure-"
            "integrated-authorization/v1"
        ),
        "status": "APPROVED_EXACTLY_ONCE",
        "retry_attempt_id": CONTROLLER.REMOTE_PREFIX_RETRY_INTEGRATED_ATTEMPT_ID,
        "prior_failed_attempt_id": CONTROLLER.REMOTE_PREFIX_FAILED_INTEGRATED_ATTEMPT_ID,
        "prior_settled_ledger_sha256": CONTROLLER.REMOTE_PREFIX_RETRY_LEDGER_SHA256,
        "ledger_sha256_before_reservation": CONTROLLER.REMOTE_PREFIX_RETRY_LEDGER_SHA256,
        "immutable_ledger_snapshot": {
            "artifact": snapshot_reference,
            "sha256": snapshot_sha,
        },
        "prior_attempt_evidence": evidence_bindings,
        "remote_prefix_parser_fix_bindings": fix_bindings,
        "unchanged_scientific_runtime_bindings": unchanged_bindings,
        "target_b_identity_evidence": target_b_bindings,
        "expected_successful_integrated_contract": _successful_integrated_contract(),
    }
    post_sha = _write_json(post_path, post)
    return base.model_copy(
        update={
            "attempt_id": CONTROLLER.REMOTE_PREFIX_RETRY_INTEGRATED_ATTEMPT_ID,
            "ledger_sha256_before_reservation": CONTROLLER.REMOTE_PREFIX_RETRY_LEDGER_SHA256,
            "budget_authorization": budget_reference,
            "budget_authorization_sha256": hashlib.sha256(budget_path.read_bytes()).hexdigest(),
            "post_micro_review_manifest_sha256": post_sha,
        }
    )


def _retained_capacity_retry_integrated_fixture(
    root: Path,
) -> Experiment004V11IntegratedConfig:
    base = _sealed_fixture(root)
    raw_reference = (
        "artifacts/branchfabric/gpu-validation/experiment-004/v11-final/integrated/raw/"
        f"{CONTROLLER.RETAINED_CAPACITY_FAILED_INTEGRATED_ATTEMPT_ID}"
    )
    shutil.copytree(ROOT / raw_reference, root / raw_reference)

    def copy_exact_bindings(
        expected: dict[str, tuple[str, str]],
    ) -> dict[str, dict[str, str]]:
        copied: dict[str, dict[str, str]] = {}
        for label, (reference, digest) in expected.items():
            destination = root / reference
            destination.parent.mkdir(parents=True, exist_ok=True)
            _materialize_exact_historical_binding(
                reference=reference,
                destination=destination,
                expected_sha256=digest,
            )
            copied[label] = {"artifact": reference, "sha256": digest}
        return copied

    evidence_bindings = copy_exact_bindings(CONTROLLER.RETAINED_CAPACITY_RETRY_EVIDENCE_BINDINGS)
    runtime_bindings = copy_exact_bindings(CONTROLLER.RETAINED_CAPACITY_UNCHANGED_RUNTIME_BINDINGS)
    movement_bindings = copy_exact_bindings(
        CONTROLLER.RETAINED_CAPACITY_UNCHANGED_ALLOCATOR_MOVEMENT_BINDINGS
    )
    target_b_bindings = copy_exact_bindings(CONTROLLER.RETAINED_CAPACITY_TARGET_B_EVIDENCE_BINDINGS)

    fix_bindings: dict[str, dict[str, str]] = {}
    for label, reference in CONTROLLER.RETAINED_CAPACITY_FIX_BINDING_PATHS.items():
        source = ROOT / reference
        destination = root / reference
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        fix_bindings[label] = {
            "artifact": reference,
            "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
        }

    review_bindings: list[dict[str, Any]] = []
    for agent, (role, reference) in CONTROLLER.RETAINED_CAPACITY_RETRY_REVIEW_BINDINGS.items():
        destination = root / reference
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / reference, destination)
        digest = hashlib.sha256(destination.read_bytes()).hexdigest()
        review_bindings.append(
            {
                "agent": agent,
                "role": role,
                "artifact": reference,
                "sha256": digest,
                "status": "PASS",
            }
        )

    snapshot_reference, snapshot_sha = CONTROLLER.RETAINED_CAPACITY_RETRY_LEDGER_SNAPSHOT_BINDING
    snapshot_path = root / snapshot_reference
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot_path.write_bytes((ROOT / snapshot_reference).read_bytes())
    assert hashlib.sha256(snapshot_path.read_bytes()).hexdigest() == snapshot_sha

    budget_reference = CONTROLLER.PREFUNCTION_RETRY_BUDGET_AUTHORIZATION
    budget_path = root / budget_reference
    budget_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / budget_reference, budget_path)

    post_path = root / base.post_micro_review_manifest
    post = json.loads(post_path.read_text())
    for binding in post["integrated_prelaunch"]["bindings"]:
        path = root / binding["artifact"]
        binding["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    post["post_retained_capacity_failure_integrated_authorization"] = {
        "schema_version": (
            "sloforge.branchfabric.exp004-v11-post-retained-capacity-failure-"
            "integrated-authorization/v1"
        ),
        "status": "APPROVED_EXACTLY_ONCE",
        "retry_attempt_id": CONTROLLER.RETAINED_CAPACITY_RETRY_INTEGRATED_ATTEMPT_ID,
        "prior_failed_attempt_id": CONTROLLER.RETAINED_CAPACITY_FAILED_INTEGRATED_ATTEMPT_ID,
        "prior_settled_ledger_sha256": CONTROLLER.RETAINED_CAPACITY_RETRY_LEDGER_SHA256,
        "ledger_sha256_before_reservation": CONTROLLER.RETAINED_CAPACITY_RETRY_LEDGER_SHA256,
        "immutable_ledger_snapshot": {
            "artifact": snapshot_reference,
            "sha256": snapshot_sha,
        },
        "prior_attempt_evidence": evidence_bindings,
        "readiness_and_sanity_fix_bindings": fix_bindings,
        "unchanged_scientific_runtime_bindings": runtime_bindings,
        "unchanged_allocator_movement_bindings": movement_bindings,
        "target_b_identity_evidence": target_b_bindings,
        "independent_reviews": review_bindings,
        "expected_successful_integrated_contract": _successful_integrated_contract(),
    }
    post_sha = _write_json(post_path, post)
    return base.model_copy(
        update={
            "attempt_id": CONTROLLER.RETAINED_CAPACITY_RETRY_INTEGRATED_ATTEMPT_ID,
            "ledger_sha256_before_reservation": (CONTROLLER.RETAINED_CAPACITY_RETRY_LEDGER_SHA256),
            "budget_authorization": budget_reference,
            "budget_authorization_sha256": hashlib.sha256(budget_path.read_bytes()).hexdigest(),
            "post_micro_review_manifest_sha256": post_sha,
        }
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


def _control_serving_fixture(*, completions_in_interval: int = 36) -> dict[str, Any]:
    """Build an exact 9-rps cohort with a selectable integral boundary count."""

    if not 32 <= completions_in_interval <= 45:
        raise ValueError("unsupported control completion fixture")
    full_start_ns = 0
    end_ns = 5_000_000_000
    rows = []
    for index in range(45):
        scheduled_ns = int(index * 1e9 / 9.0)
        first_token_ns = scheduled_ns + 10_000_000
        rows.append(
            {
                "request_id": f"control.{index:06d}",
                "phase": "control",
                "scheduled_arrival_ns": scheduled_ns,
                "service_start_ns": first_token_ns,
                "first_token_ns": first_token_ns,
                "completed_ns": scheduled_ns + 50_000_000,
                "output_token_ids": [index] * 64,
            }
        )
    if completions_in_interval < 36:
        # An early, bounded in-service burst makes the diagnostic total-depth
        # trend non-positive even when a few right-edge requests are censored.
        for offset, row in enumerate(rows[9:18]):
            row["completed_ns"] = 2_010_000_000 + offset * 10_000_000
        for offset in range(36 - completions_in_interval):
            rows[-1 - offset]["completed_ns"] = end_ns + (offset + 1) * 10_000_000
    elif completions_in_interval > 36:
        for offset in range(completions_in_interval - 36):
            rows[8 - offset]["completed_ns"] = 1_010_000_000 + offset * 10_000_000
    return {
        "start_ns": full_start_ns,
        "spike_start_ns": end_ns,
        "requests": rows,
    }


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
        "rollout_admission_stop_ns": 1_050_000_000,
        "export_started_ns": 1_100_000_000,
        "state_quiescence_ns": 1_200_000_000,
        "source_authentication_ended_ns": 1_300_000_000,
        "allocation_history_retired_ns": 1_350_000_000,
        "capture_commit_started_ns": 1_350_000_000,
        "capture_commit_ended_ns": 1_400_000_000,
        "pre_export_identity_ended_ns": 1_450_000_000,
        "source_pipeline_ended_ns": 1_600_000_000,
        "post_export_identity_ended_ns": 1_650_000_000,
        "state_publish_ended_ns": 1_700_000_000,
        "source_release_ended_ns": 1_800_000_000,
        "export_ended_ns": 1_900_000_000,
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
                ("branch_quiesce", 1_000_000_000, 1_200_000_000),
                ("source_authentication", 1_200_000_000, 1_300_000_000),
                (
                    "pre_commit_allocation_history_retirement",
                    1_300_000_000,
                    1_350_000_000,
                ),
                ("source_capture_commit", 1_350_000_000, 1_400_000_000),
                (
                    "pre_export_identity_validation_and_commit_publish",
                    1_400_000_000,
                    1_450_000_000,
                ),
                ("fused_gather_repack_d2h", 1_450_000_000, 1_600_000_000),
                (
                    "post_export_read_identity_validation",
                    1_600_000_000,
                    1_650_000_000,
                ),
                (
                    "state_publish_and_allocation_history_retirement",
                    1_650_000_000,
                    1_700_000_000,
                ),
                ("source_release", 1_700_000_000, 1_800_000_000),
                ("hbm_reclaim_confirmation", 1_800_000_000, 2_000_000_000),
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
        "residual_source_chain_ns": 400_000_000,
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
                            "schema_version": "sloforge.branchfabric.v11-control-interval/v2",
                            "passed": True,
                            "full_duration_seconds": 5.0,
                            "warmup_seconds": 1.0,
                            "duration_seconds": 4.0,
                            "expected_full_arrivals": 45,
                            "full_arrivals": 45,
                            "expected_arrivals": 36,
                            "arrivals": 36,
                            "eventual_cohort_completions": 36,
                            "completions_in_interval": 36,
                            "offered_rate_per_second": 9.0,
                            "completed_rate_per_second": 9.0,
                            "arrival_rate_tolerance_fraction": 0.05,
                            "p95_ttft_ns": 100_000_000.0,
                            "unique_control_request_ids": True,
                            "unique_scheduled_arrivals": True,
                            "scheduled_arrival_cadence_exact": True,
                            "monotonic_control_timestamps": True,
                            "cohort_output_tokens_exact": True,
                            "complete_request_accounting": True,
                            "offered_rate_matches_config": True,
                            "completion_tracks_offer": True,
                            "p95_ttft_below_slo": True,
                            "waiting_queue_maximum_depth": 1,
                            "waiting_queue_bounded_below_normal_trigger": True,
                            "total_outstanding_diagnostic": {
                                "sample_interval_ns": 1_000_000_000,
                                "samples": [
                                    {
                                        "timestamp_ns": index * 1_000_000_000,
                                        "total_outstanding": 9,
                                    }
                                    for index in range(5)
                                ],
                                "sustained_positive": False,
                            },
                            "total_outstanding_bounded_below_normal_trigger": True,
                            "outstanding_queue_non_positive_trend": True,
                            "in_service_requests_not_treated_as_waiting_queue": True,
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
                        "schema_version": "sloforge.branchfabric.v11-control-interval/v2",
                        "passed": True,
                        "full_duration_seconds": 5.0,
                        "warmup_seconds": 1.0,
                        "duration_seconds": 4.0,
                        "expected_full_arrivals": 45,
                        "full_arrivals": 45,
                        "expected_arrivals": 36,
                        "arrivals": 36,
                        "eventual_cohort_completions": 36,
                        "completions_in_interval": 36,
                        "offered_rate_per_second": 9.0,
                        "completed_rate_per_second": 9.0,
                        "arrival_rate_tolerance_fraction": 0.05,
                        "p95_ttft_ns": 100_000_000.0,
                        "unique_control_request_ids": True,
                        "unique_scheduled_arrivals": True,
                        "scheduled_arrival_cadence_exact": True,
                        "monotonic_control_timestamps": True,
                        "cohort_output_tokens_exact": True,
                        "complete_request_accounting": True,
                        "offered_rate_matches_config": True,
                        "completion_tracks_offer": True,
                        "p95_ttft_below_slo": True,
                        "waiting_queue_maximum_depth": 1,
                        "waiting_queue_bounded_below_normal_trigger": True,
                        "total_outstanding_diagnostic": {
                            "sample_interval_ns": 1_000_000_000,
                            "samples": [
                                {
                                    "timestamp_ns": index * 1_000_000_000,
                                    "total_outstanding": 9,
                                }
                                for index in range(5)
                            ],
                            "sustained_positive": False,
                        },
                        "total_outstanding_bounded_below_normal_trigger": True,
                        "outstanding_queue_non_positive_trend": True,
                        "in_service_requests_not_treated_as_waiting_queue": True,
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
    raw_control = _control_serving_fixture()
    rows[0]["serving"].update(raw_control)
    control = CONTROLLER._control_interval_evidence(
        rows[0]["serving"],
        expected_rate_rps=9.0,
        expected_duration_seconds=5.0,
        slo_ttft_seconds=2.0,
    )
    rows[0]["gpu0_control_interval"] = control
    rows[0]["gpu0_restore_interference"]["control_interval"] = control
    return rows[0], rows[1]


def _targeted_worker_results() -> tuple[dict[str, Any], dict[str, Any]]:
    attempt_id = "exp004-v11-targeted-identity-s41-a"
    identity_gate = {
        "schema_version": "sloforge.branchfabric.v11-source-identity-gate/v1",
        "event": "SOURCE_ALLOCATION_IDENTITY_GATE",
        "attempt_id": attempt_id,
        "expected_allocation_count": 1_152,
        "observed_allocation_count": 1_152,
        "exact_logical_mapping": True,
        "exact_block_epoch_identity": True,
        "exact_owner_sets": True,
        "exact_refcounts": True,
        "all_allocations_live": True,
        "device_identity_pass": True,
        "post_commit_allocation_event_count": 0,
        "no_post_commit_mutation": True,
        "optimized_export_started": False,
        "transaction_source_release_started": False,
        "passed": True,
    }
    trigger = {
        "schema_version": "sloforge.branchfabric.reclamation-trigger-evidence/v2",
        "overload_confirmed": True,
        "positive_queue_slope": True,
        "offered_rate_exceeds_completed_rate": True,
        "queue_depth_at_trigger": 21,
        "queue_trigger": 20,
        "queue_abort": 64,
        "emergency_ceiling_headroom_requests": 43,
        "events": {
            "OVERLOAD_DETECTED": 900,
            "TRIGGER_PREDICATE_SATISFIED": 999,
            "RECLAIM_TRIGGER_EMITTED": 1_000,
        },
        "overload_detected_ns": 900,
        "trigger_predicate_satisfied_ns": 999,
        "reclaim_trigger_emitted_ns": 1_000,
        "triggered_ns": 1_000,
        "controller_reaction_latency_ns": 1,
    }

    def common(role: str, pid: int, uuid: str, schema: str) -> dict[str, Any]:
        return {
            "schema_version": schema,
            "status": "succeeded",
            "role": role,
            "pid": pid,
            "physical_gpu_uuid": uuid,
            "attempt_id": attempt_id,
            "terminal_phase": "SOURCE_ALLOCATION_IDENTITY_GATE",
            "optimized_export_started": False,
            "transaction_source_release_started": False,
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

    serving = common(
        "serving",
        101,
        "GPU-a",
        "sloforge.branchfabric.experiment-004-v11-targeted-gpu0-result/v1",
    )
    serving.update(
        {
            "reclamation_trigger_evidence": trigger,
            "source_identity_terminal": dict(identity_gate),
            "producer_stopped": True,
            "final_runtime_queue_state": {
                "request_count": 0,
                "running_requests": 0,
                "waiting_requests": 0,
                "skipped_waiting_requests": 0,
                "queue_depth": 0,
            },
        }
    )
    rollout = common(
        "rollout",
        202,
        "GPU-b",
        "sloforge.branchfabric.experiment-004-v11-targeted-gpu1-result/v1",
    )
    rollout.update(
        {
            "source_identity_gate": dict(identity_gate),
            "source_identity_validation": {
                "schema_version": ("sloforge.continuum.vllm-v11-source-identity-validation/v1"),
                "passed": True,
            },
            "allocation_history_retirement": {
                "schema_version": ("sloforge.branchfabric.v11-allocation-history-retirement/v1"),
                "semantic_identity_claimed": False,
                "passed": True,
            },
            "post_commit_allocation_notifications": {
                "schema_version": (
                    "sloforge.branchfabric.v11-allocation-notification-zero-gate/v1"
                ),
                "queue_empty": True,
                "observed_event_count": 0,
                "passed": True,
            },
            "cleanup_release_only": True,
        }
    )
    return serving, rollout


def test_sealed_evidence_accepts_exact_four_gates_micro_and_reviews(tmp_path: Path) -> None:
    result = CONTROLLER.verify_sealed_evidence(_sealed_fixture(tmp_path), tmp_path)
    assert result["status"] == "PASS"
    assert result["review_agents"] == [9, 10, 11, 12]
    assert result["micro_exact_first_token_matches"] == 8
    assert result["micro_full_physical_amplification"] == pytest.approx(21.000762818351625)


def test_targeted_config_is_fail_closed_and_sealed_authorization_is_bound(
    tmp_path: Path,
) -> None:
    config = _targeted_fixture(tmp_path)
    assert config.execution_mode == "targeted-source-identity-v11"
    assert config.terminal_phase == "SOURCE_ALLOCATION_IDENTITY_GATE"
    assert config.full_export_authorized is False
    assert config.source_release_authorized is False
    assert config.maximum_wall_seconds == 300.0

    sealed = CONTROLLER.verify_sealed_evidence(config, tmp_path)
    authorization = sealed["targeted_source_identity_authorization"]
    assert sealed["status"] == "PASS"
    assert sealed["replacement_authorization_verified"] is False
    assert authorization["verified"] is True
    assert authorization["attempt_id"] == config.attempt_id
    assert len(authorization["offline_bindings"]) == 4

    payload = config.model_dump(mode="python")
    for field, unsafe in (
        ("terminal_phase", "INTEGRATED_TRANSACTION"),
        ("full_export_authorized", True),
        ("source_release_authorized", True),
        ("maximum_wall_seconds", 301.0),
    ):
        with pytest.raises(ValidationError):
            Experiment004V11TargetedSourceIdentityConfig.model_validate(
                payload | {field: unsafe}, strict=True
            )


def test_targeted_sealed_authorization_rejects_attempt_and_safety_tamper(
    tmp_path: Path,
) -> None:
    config = _targeted_fixture(tmp_path)
    post_path = tmp_path / config.post_micro_review_manifest
    post = json.loads(post_path.read_text())
    authorization = post["targeted_source_identity_authorization"]
    authorization["source_release_authorized"] = True
    post_sha = _write_json(post_path, post)
    changed = config.model_copy(update={"post_micro_review_manifest_sha256": post_sha})
    with pytest.raises(RuntimeError, match="authorization is inconsistent"):
        CONTROLLER.verify_sealed_evidence(changed, tmp_path)


def test_targeted_retry_b_binds_exact_attempt_a_failure_and_cleanup(
    tmp_path: Path,
) -> None:
    config = _targeted_retry_fixture(tmp_path)

    sealed = CONTROLLER.verify_sealed_evidence(config, tmp_path)
    authorization = sealed["targeted_source_identity_authorization"]

    assert config.attempt_id == CONTROLLER.RETRY_TARGETED_IDENTITY_ATTEMPT_ID
    assert authorization["verified"] is True
    assert authorization["prior_failed_attempt_id"] == (
        CONTROLLER.INITIAL_TARGETED_IDENTITY_ATTEMPT_ID
    )
    assert authorization["prior_targeted_attempt_evidence"]["verified"] is True


def test_targeted_retry_b_rejects_missing_or_substituted_prior_evidence(
    tmp_path: Path,
) -> None:
    config = _targeted_retry_fixture(tmp_path)
    post_path = tmp_path / config.post_micro_review_manifest
    original = json.loads(post_path.read_text())

    missing = json.loads(json.dumps(original))
    del missing["targeted_source_identity_authorization"]["prior_targeted_attempt_evidence"]
    missing_sha = _write_json(post_path, missing)
    with pytest.raises(RuntimeError, match="lacks prior targeted-attempt evidence"):
        CONTROLLER.verify_sealed_evidence(
            config.model_copy(update={"post_micro_review_manifest_sha256": missing_sha}),
            tmp_path,
        )

    substituted = json.loads(json.dumps(original))
    substituted["targeted_source_identity_authorization"]["prior_targeted_attempt_evidence"][
        "cleanup"
    ]["artifact"] = CONTROLLER.PRIOR_TARGETED_RETRY_STATUS
    substituted_sha = _write_json(post_path, substituted)
    with pytest.raises(RuntimeError, match="prior cleanup binding is malformed"):
        CONTROLLER.verify_sealed_evidence(
            config.model_copy(update={"post_micro_review_manifest_sha256": substituted_sha}),
            tmp_path,
        )


@pytest.mark.parametrize(
    ("binding_name", "mutate"),
    (
        (
            "status",
            lambda value: value["blocker"].update(code="UNRELATED_FAILURE"),
        ),
        (
            "status",
            lambda value: value.update(source_identity_gate_reached=True),
        ),
        (
            "status",
            lambda value: value.update(optimized_export_started=True),
        ),
        (
            "cleanup",
            lambda value: value["provider_observation"].update(active_apps=1),
        ),
        (
            "cleanup",
            lambda value: value["ledger"].update(sha256_after_settlement="0" * 64),
        ),
        (
            "cleanup",
            lambda value: value.update(function_call_id="fc-unrelated-attempt"),
        ),
        (
            "cleanup",
            lambda value: value["in_function_cleanup"].update(status="FAIL"),
        ),
    ),
)
def test_targeted_retry_b_rejects_semantically_invalid_prior_evidence(
    tmp_path: Path,
    binding_name: str,
    mutate: Any,
) -> None:
    config = _targeted_retry_fixture(tmp_path)
    post_path = tmp_path / config.post_micro_review_manifest
    post = json.loads(post_path.read_text())
    binding = post["targeted_source_identity_authorization"]["prior_targeted_attempt_evidence"][
        binding_name
    ]
    artifact = tmp_path / binding["artifact"]
    payload = json.loads(artifact.read_text())
    mutate(payload)
    binding["sha256"] = _write_json(artifact, payload)
    post_sha = _write_json(post_path, post)

    with pytest.raises(RuntimeError, match="prior targeted-attempt evidence is inconsistent"):
        CONTROLLER.verify_sealed_evidence(
            config.model_copy(update={"post_micro_review_manifest_sha256": post_sha}),
            tmp_path,
        )


def test_targeted_retry_b_rejects_prior_evidence_hash_tamper(tmp_path: Path) -> None:
    config = _targeted_retry_fixture(tmp_path)
    status_path = tmp_path / CONTROLLER.PRIOR_TARGETED_RETRY_STATUS
    status_path.write_bytes(status_path.read_bytes() + b"\n")

    with pytest.raises(ValueError, match="hash mismatch"):
        CONTROLLER.verify_sealed_evidence(config, tmp_path)


def test_targeted_retry_b_rejects_nested_cleanup_hash_tamper(tmp_path: Path) -> None:
    config = _targeted_retry_fixture(tmp_path)
    cleanup_path = tmp_path / CONTROLLER.PRIOR_TARGETED_RETRY_IN_FUNCTION_CLEANUP
    cleanup_path.write_bytes(cleanup_path.read_bytes() + b"\n")

    with pytest.raises(ValueError, match="hash mismatch"):
        CONTROLLER.verify_sealed_evidence(config, tmp_path)


def test_post_target_integrated_d_accepts_exact_target_b_evidence_and_reviews(
    tmp_path: Path,
) -> None:
    config = _post_target_integrated_fixture(tmp_path)

    sealed = CONTROLLER.verify_sealed_evidence(config, tmp_path)
    authorization = sealed["post_target_integrated_authorization"]

    assert config.attempt_id == CONTROLLER.POST_TARGET_INTEGRATED_ATTEMPT_ID
    assert sealed["status"] == "PASS"
    assert sealed["replacement_authorization_verified"] is False
    assert authorization["verified"] is True
    assert authorization["source_capture_commit"] == {
        "semantic_sha256": ("28168c11daf69cdc89e1ddf369a8c828fc274315d37c234c480ba63c5ab67086"),
        "allocation_count": 1152,
        "allocation_history_sha256": (
            "f30b50721d62a2936b8b110ce1633042314541637bff0f61fa8dc4551e81e241"
        ),
        "zero_post_commit_events": True,
    }
    assert len(authorization["target_reviews"]) == 3
    assert authorization["function_call_id"] == "fc-01M0RNWPEBZ2R9M4YWXSAGRPA4"


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("status", "PASS"),
        ("integrated_attempt_id", CONTROLLER.REPLACEMENT_INTEGRATED_ATTEMPT_ID),
        ("targeted_attempt_id", CONTROLLER.INITIAL_TARGETED_IDENTITY_ATTEMPT_ID),
        ("ledger_sha256_before_reservation", "0" * 64),
    ),
)
def test_post_target_integrated_d_rejects_authorization_tamper(
    tmp_path: Path, field: str, value: Any
) -> None:
    config = _post_target_integrated_fixture(tmp_path)
    post_path = tmp_path / config.post_micro_review_manifest
    post = json.loads(post_path.read_text())
    post["post_target_integrated_authorization"][field] = value
    post_sha = _write_json(post_path, post)

    with pytest.raises(RuntimeError, match="authorization is inconsistent"):
        CONTROLLER.verify_sealed_evidence(
            config.model_copy(update={"post_micro_review_manifest_sha256": post_sha}),
            tmp_path,
        )


@pytest.mark.parametrize("mutation", ("missing", "extra", "path", "hash"))
def test_post_target_integrated_d_rejects_evidence_binding_tamper(
    tmp_path: Path, mutation: str
) -> None:
    config = _post_target_integrated_fixture(tmp_path)
    post_path = tmp_path / config.post_micro_review_manifest
    post = json.loads(post_path.read_text())
    bindings = post["post_target_integrated_authorization"]["targeted_attempt_evidence"]
    if mutation == "missing":
        del bindings["identity_gate"]
    elif mutation == "extra":
        bindings["unrelated"] = dict(bindings["identity_gate"])
    elif mutation == "path":
        bindings["identity_gate"]["artifact"] = bindings["source_capture_commit"]["artifact"]
    else:
        bindings["identity_gate"]["sha256"] = "0" * 64
    post_sha = _write_json(post_path, post)
    message = (
        "bindings are incomplete"
        if mutation in {"missing", "extra"}
        else "binding is malformed|hash mismatch"
    )

    with pytest.raises((RuntimeError, ValueError), match=message):
        CONTROLLER.verify_sealed_evidence(
            config.model_copy(update={"post_micro_review_manifest_sha256": post_sha}),
            tmp_path,
        )


@pytest.mark.parametrize(
    ("label", "mutate", "message"),
    (
        (
            "status",
            lambda value: value.update(observed_allocation_count=1151),
            "1,152/1,152",
        ),
        (
            "status",
            lambda value: value.update(post_commit_allocation_event_count=True),
            "1,152/1,152",
        ),
        (
            "status",
            lambda value: value.update(optimized_export_started=True),
            "1,152/1,152",
        ),
        (
            "identity_gate",
            lambda value: value.update(exact_block_epoch_identity=False),
            "identity gate",
        ),
        (
            "identity_gate",
            lambda value: value.update(post_commit_allocation_event_count=True),
            "identity gate",
        ),
        (
            "source_capture_commit",
            lambda value: value["commit"]["allocations"][0].update(allocation_epoch=999999),
            "SourceCaptureCommit semantics",
        ),
        (
            "source_capture_commit",
            lambda value: value["post_commit_allocation_notifications"].update(
                observed_event_count=True
            ),
            "post-commit mutations",
        ),
        (
            "source_capture_commit",
            lambda value: (
                value["allocator_lifecycle"]["events"].append(
                    {
                        "sequence": value["allocator_lifecycle"]["capture_commit_sequence"],
                        "event": "ALLOC",
                    }
                ),
                value["allocator_lifecycle"].update(
                    event_count=value["allocator_lifecycle"]["event_count"] + 1
                ),
            ),
            "journal crosses",
        ),
        (
            "provider_cleanup",
            lambda value: value["provider_observation"].update(active_apps=False),
            "provider cleanup",
        ),
        (
            "provider_cleanup",
            lambda value: value.update(function_call_id="fc-wrong"),
            "provider cleanup",
        ),
        (
            "in_function_cleanup",
            lambda value: value.update(cuda_released=False),
            "in-function cleanup",
        ),
    ),
)
def test_post_target_integrated_d_rejects_semantic_evidence_tamper(
    tmp_path: Path,
    label: str,
    mutate: Any,
    message: str,
) -> None:
    config = _post_target_integrated_fixture(tmp_path)
    post_path = tmp_path / config.post_micro_review_manifest
    post = json.loads(post_path.read_text())
    authorization = post["post_target_integrated_authorization"]
    binding = authorization["targeted_attempt_evidence"][label]
    artifact = tmp_path / binding["artifact"]
    payload = json.loads(artifact.read_text())
    mutate(payload)
    binding["sha256"] = _write_json(artifact, payload)
    if label in {"identity_gate", "source_capture_commit", "in_function_cleanup"}:
        manifest_binding = authorization["targeted_attempt_evidence"]["remote_manifest"]
        manifest_path = tmp_path / manifest_binding["artifact"]
        manifest = json.loads(manifest_path.read_text())
        relative_path = str(binding["artifact"]).split(
            f"raw/{CONTROLLER.RETRY_TARGETED_IDENTITY_ATTEMPT_ID}/", 1
        )[1]
        manifest_row = next(
            row for row in manifest["artifacts"] if row["relative_path"] == relative_path
        )
        manifest_row["sha256"] = binding["sha256"]
        manifest_row["bytes"] = artifact.stat().st_size
        manifest_binding["sha256"] = _write_json(manifest_path, manifest)
        if label in {"identity_gate", "source_capture_commit"}:
            status_binding = authorization["targeted_attempt_evidence"]["status"]
            status_path = tmp_path / status_binding["artifact"]
            status = json.loads(status_path.read_text())
            status_key = {
                "identity_gate": "identity_gate",
                "source_capture_commit": "source_capture_commit",
            }[label]
            status["immutable_artifacts"][status_key]["sha256"] = binding["sha256"]
            status["immutable_artifacts"]["remote_manifest"]["sha256"] = manifest_binding["sha256"]
            status_binding["sha256"] = _write_json(status_path, status)
        elif label == "in_function_cleanup":
            provider_binding = authorization["targeted_attempt_evidence"]["provider_cleanup"]
            provider_path = tmp_path / provider_binding["artifact"]
            provider = json.loads(provider_path.read_text())
            provider["in_function_cleanup"]["sha256"] = binding["sha256"]
            provider_binding["sha256"] = _write_json(provider_path, provider)
            status_binding = authorization["targeted_attempt_evidence"]["status"]
            status_path = tmp_path / status_binding["artifact"]
            status = json.loads(status_path.read_text())
            status["immutable_artifacts"]["remote_manifest"]["sha256"] = manifest_binding["sha256"]
            status_binding["sha256"] = _write_json(status_path, status)
    post_sha = _write_json(post_path, post)

    with pytest.raises(RuntimeError, match=message):
        CONTROLLER.verify_sealed_evidence(
            config.model_copy(update={"post_micro_review_manifest_sha256": post_sha}),
            tmp_path,
        )


@pytest.mark.parametrize(
    ("agent", "mutate", "message"),
    (
        (11, lambda value: value["identity"].update(block_epoch_passed=1151), "Agent 11"),
        (
            12,
            lambda value: value["allocator_quiescence"].update(post_commit_allocation_events=True),
            "Agent 12",
        ),
        (13, lambda value: value["budget"].update(ledger_sha256="0" * 64), "Agent 13"),
    ),
)
def test_post_target_integrated_d_rejects_independent_review_tamper(
    tmp_path: Path, agent: int, mutate: Any, message: str
) -> None:
    config = _post_target_integrated_fixture(tmp_path)
    post_path = tmp_path / config.post_micro_review_manifest
    post = json.loads(post_path.read_text())
    binding = next(
        item
        for item in post["post_target_integrated_authorization"]["target_reviews"]
        if item["agent"] == agent
    )
    artifact = tmp_path / binding["artifact"]
    payload = json.loads(artifact.read_text())
    mutate(payload)
    binding["sha256"] = _write_json(artifact, payload)
    post_sha = _write_json(post_path, post)

    with pytest.raises(RuntimeError, match=message):
        CONTROLLER.verify_sealed_evidence(
            config.model_copy(update={"post_micro_review_manifest_sha256": post_sha}),
            tmp_path,
        )


@pytest.mark.parametrize("mutation", ("missing", "duplicate", "path", "hash"))
def test_post_target_integrated_d_rejects_review_binding_tamper(
    tmp_path: Path, mutation: str
) -> None:
    config = _post_target_integrated_fixture(tmp_path)
    post_path = tmp_path / config.post_micro_review_manifest
    post = json.loads(post_path.read_text())
    reviews = post["post_target_integrated_authorization"]["target_reviews"]
    if mutation == "missing":
        reviews.pop()
    elif mutation == "duplicate":
        reviews[1] = dict(reviews[0])
    elif mutation == "path":
        reviews[0]["artifact"] = reviews[1]["artifact"]
    else:
        reviews[0]["sha256"] = "0" * 64
    post_sha = _write_json(post_path, post)

    with pytest.raises((RuntimeError, ValueError)):
        CONTROLLER.verify_sealed_evidence(
            config.model_copy(update={"post_micro_review_manifest_sha256": post_sha}),
            tmp_path,
        )


def test_post_target_integrated_d_rejects_config_ledger_tamper(tmp_path: Path) -> None:
    config = _post_target_integrated_fixture(tmp_path)

    with pytest.raises(RuntimeError, match="authorization is inconsistent"):
        CONTROLLER.verify_sealed_evidence(
            config.model_copy(update={"ledger_sha256_before_reservation": "0" * 64}),
            tmp_path,
        )


def _post_target_copied_evidence(
    config: Experiment004V11IntegratedConfig, root: Path
) -> dict[str, dict[str, Any]]:
    post = json.loads((root / config.post_micro_review_manifest).read_text())
    bindings = post["post_target_integrated_authorization"]["targeted_attempt_evidence"]
    return {
        label: json.loads((root / binding["artifact"]).read_text())
        for label, binding in bindings.items()
    }


@pytest.mark.parametrize(
    "mutation",
    (
        "reduced-runtime-identity",
        "wrong-root-session",
        "wrong-parent",
        "wrong-computed-tokens",
        "wrong-private-topology",
        "duplicate-runtime-request",
    ),
)
def test_post_target_source_commit_rejects_runtime_and_branch_identity_tamper(
    tmp_path: Path, mutation: str
) -> None:
    config = _post_target_integrated_fixture(tmp_path)
    envelope = _post_target_copied_evidence(config, tmp_path)["source_capture_commit"]
    commit = envelope["commit"]
    if mutation == "reduced-runtime-identity":
        commit["runtime_model_identity"] = [["device", "cuda:0"]]
        commit["runtime_model_identity_sha256"] = hashlib.sha256(
            CONTROLLER._canonical_bytes(commit["runtime_model_identity"])
        ).hexdigest()
    elif mutation == "wrong-root-session":
        commit["root_session_id"] = "unrelated-root"
    elif mutation == "wrong-parent":
        commit["branches"][0]["parent_logical_branch_id"] = "unrelated-root"
    elif mutation == "wrong-computed-tokens":
        commit["branches"][0]["computed_tokens"] = 16384
    elif mutation == "wrong-private-topology":
        commit["branches"][0]["logical_page_ids"][-1] = "logical-page-001145"
    else:
        commit["branches"][1]["runtime_request_id"] = commit["branches"][0]["runtime_request_id"]
    commit["semantic_sha256"] = CONTROLLER._source_capture_semantic_sha256(commit)

    with pytest.raises(RuntimeError):
        CONTROLLER._verify_post_target_source_commit(
            envelope,
            expected_semantic_sha256=commit["semantic_sha256"],
            config=config,
        )


@pytest.mark.parametrize("mutation", ("empty", "truncated", "duplicate", "out-of-order"))
def test_post_target_source_commit_rejects_incomplete_or_nonmonotonic_journal(
    tmp_path: Path, mutation: str
) -> None:
    config = _post_target_integrated_fixture(tmp_path)
    envelope = _post_target_copied_evidence(config, tmp_path)["source_capture_commit"]
    lifecycle = envelope["allocator_lifecycle"]
    events = lifecycle["events"]
    if mutation == "empty":
        events.clear()
    elif mutation == "truncated":
        events.pop()
    elif mutation == "duplicate":
        events[1] = dict(events[0])
    else:
        events[0], events[1] = events[1], events[0]
    lifecycle["event_count"] = len(events)
    lifecycle["source_lifetime_event_count"] = sum(
        event.get("logical_page_id") is not None for event in events
    )
    lifecycle["sha256"] = hashlib.sha256(
        CONTROLLER._canonical_bytes(
            {key: value for key, value in lifecycle.items() if key != "sha256"}
        )
    ).hexdigest()

    with pytest.raises(RuntimeError, match="journal"):
        CONTROLLER._verify_post_target_source_commit(
            envelope,
            expected_semantic_sha256=envelope["commit"]["semantic_sha256"],
            config=config,
        )


@pytest.mark.parametrize("mutation", ("delete", "replace"))
def test_post_target_source_commit_requires_history_retirement_proof(
    tmp_path: Path, mutation: str
) -> None:
    config = _post_target_integrated_fixture(tmp_path)
    envelope = _post_target_copied_evidence(config, tmp_path)["source_capture_commit"]
    if mutation == "delete":
        del envelope["pre_commit_allocation_history"]
    else:
        envelope["pre_commit_allocation_history"] = {"passed": True}

    with pytest.raises(RuntimeError, match=r"envelope|history retirement"):
        CONTROLLER._verify_post_target_source_commit(
            envelope,
            expected_semantic_sha256=envelope["commit"]["semantic_sha256"],
            config=config,
        )


@pytest.mark.parametrize(
    ("label", "mutate", "message"),
    (
        ("trigger", lambda value: value.update(positive_queue_slope=False), "trigger"),
        (
            "admission_stop",
            lambda value: value.update(observed_ns=value["reclaim_trigger_emitted_ns"]),
            "admission-stop",
        ),
        (
            "controller_result",
            lambda value: value["sanity_guard_pair"].update(passed=False),
            "controller result",
        ),
        (
            "serving_result",
            lambda value: value["final_runtime_queue_state"].update(queue_depth=False),
            "serving result",
        ),
        (
            "rollout_result",
            lambda value: value.update(optimized_export_started=True),
            "rollout result",
        ),
    ),
)
def test_post_target_runtime_evidence_rejects_raw_lifecycle_tamper(
    tmp_path: Path, label: str, mutate: Any, message: str
) -> None:
    config = _post_target_integrated_fixture(tmp_path)
    evidence = _post_target_copied_evidence(config, tmp_path)
    mutate(evidence[label])

    with pytest.raises(RuntimeError, match=message):
        CONTROLLER._verify_post_target_runtime_evidence(
            evidence,
            commit_envelope=evidence["source_capture_commit"],
            gate=evidence["identity_gate"],
            in_function_cleanup=evidence["in_function_cleanup"],
            config=config,
        )


def _reseal_prefunction_retry_evidence(
    root: Path,
    config: Experiment004V11IntegratedConfig,
    *,
    label: str,
    mutate: Any,
) -> Experiment004V11IntegratedConfig:
    reference = CONTROLLER.PREFUNCTION_RETRY_EVIDENCE_BINDINGS[label]
    evidence_path = root / reference
    evidence = json.loads(evidence_path.read_text())
    mutate(evidence)
    evidence_sha = _write_json(evidence_path, evidence)
    post_path = root / config.post_micro_review_manifest
    post = json.loads(post_path.read_text())
    post["post_prefunction_failure_integrated_authorization"]["prior_attempt_evidence"][label][
        "sha256"
    ] = evidence_sha
    post_sha = _write_json(post_path, post)
    return config.model_copy(update={"post_micro_review_manifest_sha256": post_sha})


def test_prefunction_retry_e_accepts_exact_d_failure_budget_and_success_contract(
    tmp_path: Path,
) -> None:
    config = _prefunction_retry_integrated_fixture(tmp_path)

    sealed = CONTROLLER.verify_sealed_evidence(config, tmp_path)
    authorization = sealed["post_prefunction_failure_integrated_authorization"]

    assert config.attempt_id == "exp004-v11-integrated-s41-e"
    assert authorization["verified"] is True
    assert authorization["prior_function_entered"] is False
    assert authorization["prior_in_function_cleanup"] == ("NOT_APPLICABLE_FUNCTION_NOT_ENTERED")
    assert authorization["prior_provider_cleanup"] == "PASS"
    assert authorization["expanded_budget_verified"] is True
    assert authorization["successful_integrated_contract_verified"] is True
    assert authorization["expected_successful_integrated_contract"] == (
        _successful_integrated_contract()
    )


@pytest.mark.parametrize(
    ("mutate", "message"),
    (
        (
            lambda value: value.update(retry_attempt_id="exp004-v11-integrated-s41-d"),
            "authorization is inconsistent",
        ),
        (
            lambda value: value.pop("prior_settled_ledger_sha256"),
            "authorization is inconsistent",
        ),
        (
            lambda value: value.update(unexpected=True),
            "authorization is inconsistent",
        ),
        (
            lambda value: value["expected_successful_integrated_contract"].update(
                completion_schema_version=(
                    "sloforge.branchfabric.experiment-004-v11-targeted-source-identity-"
                    "completion/v1"
                )
            ),
            "result contract is inconsistent",
        ),
        (
            lambda value: value["expected_successful_integrated_contract"][
                "remote_completion_fields"
            ].pop(),
            "result contract is inconsistent",
        ),
        (
            lambda value: value["expected_successful_integrated_contract"].update(
                source_release_authorized=False
            ),
            "result contract is inconsistent",
        ),
    ),
)
def test_prefunction_retry_e_rejects_authorization_or_success_schema_tamper(
    tmp_path: Path, mutate: Any, message: str
) -> None:
    config = _prefunction_retry_integrated_fixture(tmp_path)
    post_path = tmp_path / config.post_micro_review_manifest
    post = json.loads(post_path.read_text())
    mutate(post["post_prefunction_failure_integrated_authorization"])
    post_sha = _write_json(post_path, post)

    with pytest.raises(RuntimeError, match=message):
        CONTROLLER.verify_sealed_evidence(
            config.model_copy(update={"post_micro_review_manifest_sha256": post_sha}),
            tmp_path,
        )


@pytest.mark.parametrize(
    ("label", "mutate", "message"),
    (
        (
            "status",
            lambda value: value.update(function_entered=True),
            "terminal status is inconsistent",
        ),
        (
            "status",
            lambda value: value.update(source_identity_gate_reached=True),
            "terminal status is inconsistent",
        ),
        (
            "status",
            lambda value: value["accounting"].update(actual_gpu_seconds=0),
            "terminal status is inconsistent",
        ),
        (
            "provider_cleanup",
            lambda value: value["provider_observation"].update(active_apps=1),
            "provider cleanup is inconsistent",
        ),
        (
            "provider_cleanup",
            lambda value: value.update(function_call_id="fc-unexpected"),
            "provider cleanup is inconsistent",
        ),
        (
            "provider_cleanup",
            lambda value: value["in_function_cleanup"].update(status="PASS"),
            "provider cleanup is inconsistent",
        ),
        (
            "failure",
            lambda value: value.update(actual_gpu_seconds=1.0),
            "provider cleanup is inconsistent",
        ),
        (
            "failure",
            lambda value: value.update(charged_gpu_seconds=1_175.0),
            "provider cleanup is inconsistent",
        ),
        (
            "failure",
            lambda value: value.update(stdout_tail='{"result":{}}'),
            "provider cleanup is inconsistent",
        ),
    ),
)
def test_prefunction_retry_e_rejects_self_resealed_d_semantic_tamper(
    tmp_path: Path, label: str, mutate: Any, message: str
) -> None:
    config = _prefunction_retry_integrated_fixture(tmp_path)
    changed = _reseal_prefunction_retry_evidence(
        tmp_path,
        config,
        label=label,
        mutate=mutate,
    )
    with pytest.raises(RuntimeError, match=message):
        CONTROLLER.verify_sealed_evidence(changed, tmp_path)


def test_prefunction_retry_e_rejects_cross_artifact_failure_hash_tamper(
    tmp_path: Path,
) -> None:
    config = _prefunction_retry_integrated_fixture(tmp_path)
    changed = _reseal_prefunction_retry_evidence(
        tmp_path,
        config,
        label="provider_cleanup",
        mutate=lambda value: value["failure_evidence"].update(sha256="0" * 64),
    )
    with pytest.raises(RuntimeError, match="provider cleanup is inconsistent"):
        CONTROLLER.verify_sealed_evidence(changed, tmp_path)


@pytest.mark.parametrize(
    "mutate",
    (
        lambda value: value.update(authorized_gpu_budget_usd=79.0),
        lambda value: value.update(previous_ledger_sha256="0" * 64),
        lambda value: value.update(active_reservations_at_authorization=False),
        lambda value: value["authorized_scope"].pop(0),
        lambda value: value["ledger_mutation"].update(conservative_failure_charges_changed=True),
    ),
)
def test_prefunction_retry_e_rejects_expanded_budget_tamper(tmp_path: Path, mutate: Any) -> None:
    config = _prefunction_retry_integrated_fixture(tmp_path)
    budget_path = tmp_path / config.budget_authorization
    budget = json.loads(budget_path.read_text())
    mutate(budget)
    budget_sha = _write_json(budget_path, budget)
    changed = config.model_copy(update={"budget_authorization_sha256": budget_sha})
    with pytest.raises(RuntimeError, match="expanded budget authorization is inconsistent"):
        CONTROLLER.verify_sealed_evidence(changed, tmp_path)


def test_prefunction_retry_e_rejects_any_attempt_d_raw_function_tree(tmp_path: Path) -> None:
    config = _prefunction_retry_integrated_fixture(tmp_path)
    raw_root = (
        tmp_path
        / "artifacts/branchfabric/gpu-validation/experiment-004/v11-final/integrated/raw"
        / CONTROLLER.PREFUNCTION_FAILED_INTEGRATED_ATTEMPT_ID
    )
    raw_root.mkdir(parents=True)
    _write_json(raw_root / "function-completion.json", {"status": "unexpected"})

    with pytest.raises(RuntimeError, match="unexpectedly has remote function"):
        CONTROLLER.verify_sealed_evidence(config, tmp_path)


def test_bundled_ledger_retry_f_accepts_exact_e_target_b_snapshot_and_contracts(
    tmp_path: Path,
) -> None:
    config = _bundled_ledger_retry_integrated_fixture(tmp_path)
    assert not (
        tmp_path / "artifacts/branchfabric/gpu-validation/experiment-004/gpu-hours.json"
    ).exists()

    sealed = CONTROLLER.verify_sealed_evidence(config, tmp_path)
    authorization = sealed["post_remote_precontroller_failure_integrated_authorization"]

    assert config.attempt_id == "exp004-v11-integrated-s41-f"
    assert authorization["verified"] is True
    assert authorization["immutable_ledger_snapshot_verified"] is True
    assert authorization["unchanged_target_b_verified"] is True
    assert authorization["target_b_identity_gate_passed"] == 1152
    assert authorization["prior_controller_transaction_started"] is False
    assert authorization["prior_in_function_cleanup"] == (
        "FAIL_MISSING_PRE_CONTROLLER_CLEANUP_EVIDENCE"
    )
    assert authorization["prior_provider_cleanup"] == "PASS"
    assert authorization["expanded_budget_verified"] is True
    assert authorization["successful_integrated_contract_verified"] is True
    assert authorization["pre_worker_cleanup_contract_verified"] is True


@pytest.mark.parametrize(
    ("mutate", "message"),
    (
        (
            lambda value: value.update(retry_attempt_id="exp004-v11-integrated-s41-e"),
            "authorization is inconsistent",
        ),
        (
            lambda value: value.pop("prior_settled_ledger_sha256"),
            "authorization is inconsistent",
        ),
        (
            lambda value: value.update(unexpected=True),
            "authorization is inconsistent",
        ),
        (
            lambda value: value["immutable_ledger_snapshot"].update(sha256="0" * 64),
            "snapshot binding is malformed",
        ),
        (
            lambda value: value["unchanged_target_b_authorization"].update(sha256="0" * 64),
            "Target-B binding is malformed",
        ),
        (
            lambda value: value["expected_successful_integrated_contract"].update(
                source_release_authorized=False
            ),
            "result contract is inconsistent",
        ),
        (
            lambda value: value["expected_successful_integrated_contract"][
                "remote_completion_fields"
            ].pop(),
            "result contract is inconsistent",
        ),
        (
            lambda value: value["required_pre_worker_cleanup_contract"].update(
                cleanup_pass_required=False
            ),
            "cleanup contract is inconsistent",
        ),
        (
            lambda value: value["required_pre_worker_cleanup_contract"][
                "allowed_failure_stages"
            ].pop(),
            "cleanup contract is inconsistent",
        ),
    ),
)
def test_bundled_ledger_retry_f_rejects_authorization_contract_or_binding_tamper(
    tmp_path: Path, mutate: Any, message: str
) -> None:
    config = _bundled_ledger_retry_integrated_fixture(tmp_path)
    post_path = tmp_path / config.post_micro_review_manifest
    post = json.loads(post_path.read_text())
    mutate(post["post_remote_precontroller_failure_integrated_authorization"])
    post_sha = _write_json(post_path, post)

    with pytest.raises(RuntimeError, match=message):
        CONTROLLER.verify_sealed_evidence(
            config.model_copy(update={"post_micro_review_manifest_sha256": post_sha}),
            tmp_path,
        )


@pytest.mark.parametrize(
    "label",
    (
        "status",
        "provider_cleanup",
        "failure",
        "remote_manifest",
        "function_completion",
        "function_failure",
        "function_finally_cleanup",
        "pre_gpu_manifest",
        "seal_verification",
        "resource_audit",
        "independent_resource_review",
    ),
)
def test_bundled_ledger_retry_f_rejects_any_attempt_e_evidence_hash_tamper(
    tmp_path: Path, label: str
) -> None:
    config = _bundled_ledger_retry_integrated_fixture(tmp_path)
    post_path = tmp_path / config.post_micro_review_manifest
    post = json.loads(post_path.read_text())
    post["post_remote_precontroller_failure_integrated_authorization"]["prior_attempt_evidence"][
        label
    ]["sha256"] = "0" * 64
    post_sha = _write_json(post_path, post)

    with pytest.raises(RuntimeError, match=f"prior {label} binding is malformed"):
        CONTROLLER.verify_sealed_evidence(
            config.model_copy(update={"post_micro_review_manifest_sha256": post_sha}),
            tmp_path,
        )


@pytest.mark.parametrize(
    "mutation",
    (
        "content",
        "missing",
        "symlink",
    ),
)
def test_bundled_ledger_retry_f_rejects_snapshot_drift_or_substitution(
    tmp_path: Path, mutation: str
) -> None:
    config = _bundled_ledger_retry_integrated_fixture(tmp_path)
    reference, _sha256 = CONTROLLER.BUNDLED_LEDGER_SNAPSHOT_BINDING
    snapshot_path = tmp_path / reference
    if mutation == "content":
        snapshot = json.loads(snapshot_path.read_text())
        snapshot["reservations"] = [{}]
        _write_json(snapshot_path, snapshot)
    elif mutation == "missing":
        snapshot_path.unlink()
    else:
        snapshot_path.unlink()
        snapshot_path.symlink_to(tmp_path / "missing-ledger.json")

    with pytest.raises((RuntimeError, ValueError, FileNotFoundError)):
        CONTROLLER.verify_sealed_evidence(config, tmp_path)


def test_bundled_ledger_retry_f_rejects_any_target_b_evidence_drift(tmp_path: Path) -> None:
    config = _bundled_ledger_retry_integrated_fixture(tmp_path)
    gate_path = tmp_path / CONTROLLER.POST_TARGET_EVIDENCE_BINDINGS["identity_gate"]
    gate = json.loads(gate_path.read_text())
    gate["passed"] = False
    _write_json(gate_path, gate)

    with pytest.raises(ValueError, match="hash mismatch"):
        CONTROLLER.verify_sealed_evidence(config, tmp_path)


def test_bundled_ledger_retry_f_rejects_preexisting_remote_output(tmp_path: Path) -> None:
    config = _bundled_ledger_retry_integrated_fixture(tmp_path)
    raw_root = (
        tmp_path
        / "artifacts/branchfabric/gpu-validation/experiment-004/v11-final/integrated/raw"
        / CONTROLLER.BUNDLED_LEDGER_RETRY_INTEGRATED_ATTEMPT_ID
    )
    raw_root.mkdir(parents=True)

    with pytest.raises(RuntimeError, match="pre-existing remote evidence"):
        CONTROLLER.verify_sealed_evidence(config, tmp_path)


def test_queue_accounting_retry_g_accepts_exact_f_identity_queue_cleanup_and_budget(
    tmp_path: Path,
) -> None:
    config = _queue_accounting_retry_integrated_fixture(tmp_path)

    sealed = CONTROLLER.verify_sealed_evidence(config, tmp_path)
    authorization = sealed["post_queue_accounting_failure_integrated_authorization"]

    assert config.attempt_id == "exp004-v11-integrated-s41-g"
    assert authorization["verified"] is True
    assert authorization["immutable_ledger_snapshot_verified"] is True
    assert authorization["attempt_f_identity"]["allocation_count"] == 1152
    assert authorization["attempt_f_identity"]["pre_export_identity_gate_passed"] == 1152
    assert authorization["attempt_f_identity"]["post_export_identity_gate_passed"] == 1152
    assert authorization["attempt_f_queue_accounting"] == {
        "raw_stale_outstanding": 64,
        "synchronized_outstanding": 52,
        "true_maximum_outstanding": 57,
        "true_maximum_observed_ns": 198858810179,
    }
    assert authorization["prior_in_function_cleanup"] == "PASS"
    assert authorization["prior_provider_cleanup"] == "PASS"
    assert authorization["expanded_budget_verified"] is True
    assert authorization["successful_integrated_contract_verified"] is True


@pytest.mark.parametrize(
    ("mutate", "message"),
    (
        (
            lambda value: value.update(retry_attempt_id="exp004-v11-integrated-s41-f"),
            "authorization is inconsistent",
        ),
        (
            lambda value: value.pop("prior_settled_ledger_sha256"),
            "authorization is inconsistent",
        ),
        (
            lambda value: value.update(unexpected=True),
            "authorization is inconsistent",
        ),
        (
            lambda value: value["immutable_ledger_snapshot"].update(sha256="0" * 64),
            "snapshot binding is malformed",
        ),
        (
            lambda value: value["expected_successful_integrated_contract"].update(
                source_release_authorized=False
            ),
            "result contract is inconsistent",
        ),
        (
            lambda value: value["expected_successful_integrated_contract"][
                "remote_completion_fields"
            ].pop(),
            "result contract is inconsistent",
        ),
    ),
)
def test_queue_accounting_retry_g_rejects_authorization_or_contract_tamper(
    tmp_path: Path, mutate: Any, message: str
) -> None:
    config = _queue_accounting_retry_integrated_fixture(tmp_path)
    post_path = tmp_path / config.post_micro_review_manifest
    post = json.loads(post_path.read_text())
    mutate(post["post_queue_accounting_failure_integrated_authorization"])
    post_sha = _write_json(post_path, post)

    with pytest.raises(RuntimeError, match=message):
        CONTROLLER.verify_sealed_evidence(
            config.model_copy(update={"post_micro_review_manifest_sha256": post_sha}),
            tmp_path,
        )


@pytest.mark.parametrize("label", tuple(CONTROLLER.QUEUE_ACCOUNTING_RETRY_EVIDENCE_BINDINGS))
def test_queue_accounting_retry_g_rejects_any_attempt_f_evidence_binding_tamper(
    tmp_path: Path, label: str
) -> None:
    config = _queue_accounting_retry_integrated_fixture(tmp_path)
    post_path = tmp_path / config.post_micro_review_manifest
    post = json.loads(post_path.read_text())
    post["post_queue_accounting_failure_integrated_authorization"]["prior_attempt_evidence"][label][
        "sha256"
    ] = "0" * 64
    post_sha = _write_json(post_path, post)

    with pytest.raises(RuntimeError, match=f"prior {label} binding is malformed"):
        CONTROLLER.verify_sealed_evidence(
            config.model_copy(update={"post_micro_review_manifest_sha256": post_sha}),
            tmp_path,
        )


@pytest.mark.parametrize(
    ("group", "label", "message"),
    (
        *(
            ("queue_accounting_fix_bindings", label, "queue-accounting fix")
            for label in CONTROLLER.QUEUE_ACCOUNTING_FIX_BINDINGS
        ),
        *(
            ("unchanged_runtime_bindings", label, "unchanged runtime")
            for label in CONTROLLER.QUEUE_ACCOUNTING_UNCHANGED_RUNTIME_BINDINGS
        ),
    ),
)
def test_queue_accounting_retry_g_rejects_fix_or_runtime_binding_tamper(
    tmp_path: Path, group: str, label: str, message: str
) -> None:
    config = _queue_accounting_retry_integrated_fixture(tmp_path)
    post_path = tmp_path / config.post_micro_review_manifest
    post = json.loads(post_path.read_text())
    post["post_queue_accounting_failure_integrated_authorization"][group][label]["sha256"] = (
        "0" * 64
    )
    post_sha = _write_json(post_path, post)

    with pytest.raises(RuntimeError, match=message):
        CONTROLLER.verify_sealed_evidence(
            config.model_copy(update={"post_micro_review_manifest_sha256": post_sha}),
            tmp_path,
        )


@pytest.mark.parametrize("agent", tuple(CONTROLLER.QUEUE_ACCOUNTING_RETRY_REVIEW_BINDINGS))
def test_queue_accounting_retry_g_rejects_review_binding_tamper(tmp_path: Path, agent: int) -> None:
    config = _queue_accounting_retry_integrated_fixture(tmp_path)
    post_path = tmp_path / config.post_micro_review_manifest
    post = json.loads(post_path.read_text())
    review = next(
        row
        for row in post["post_queue_accounting_failure_integrated_authorization"][
            "independent_reviews"
        ]
        if row["agent"] == agent
    )
    review["sha256"] = "0" * 64
    post_sha = _write_json(post_path, post)

    with pytest.raises(RuntimeError, match="review identity is inconsistent"):
        CONTROLLER.verify_sealed_evidence(
            config.model_copy(update={"post_micro_review_manifest_sha256": post_sha}),
            tmp_path,
        )


@pytest.mark.parametrize("mutation", ("content", "missing", "symlink"))
def test_queue_accounting_retry_g_rejects_snapshot_drift_or_substitution(
    tmp_path: Path, mutation: str
) -> None:
    config = _queue_accounting_retry_integrated_fixture(tmp_path)
    reference, _sha256 = CONTROLLER.QUEUE_ACCOUNTING_RETRY_LEDGER_SNAPSHOT_BINDING
    snapshot_path = tmp_path / reference
    if mutation == "content":
        snapshot = json.loads(snapshot_path.read_text())
        snapshot["reservations"] = [{}]
        _write_json(snapshot_path, snapshot)
    elif mutation == "missing":
        snapshot_path.unlink()
    else:
        snapshot_path.unlink()
        snapshot_path.symlink_to(tmp_path / "missing-ledger.json")

    with pytest.raises((RuntimeError, ValueError, FileNotFoundError)):
        CONTROLLER.verify_sealed_evidence(config, tmp_path)


def test_queue_accounting_retry_g_rejects_preexisting_remote_output(tmp_path: Path) -> None:
    config = _queue_accounting_retry_integrated_fixture(tmp_path)
    raw_root = (
        tmp_path
        / "artifacts/branchfabric/gpu-validation/experiment-004/v11-final/integrated/raw"
        / CONTROLLER.QUEUE_ACCOUNTING_RETRY_INTEGRATED_ATTEMPT_ID
    )
    raw_root.mkdir(parents=True)

    with pytest.raises(RuntimeError, match="pre-existing remote evidence"):
        CONTROLLER.verify_sealed_evidence(config, tmp_path)


def test_control_gate_retry_h_accepts_exact_g_transaction_baseline_fix_and_budget(
    tmp_path: Path,
) -> None:
    config = _control_gate_retry_integrated_fixture(tmp_path)

    sealed = CONTROLLER.verify_sealed_evidence(config, tmp_path)
    authorization = sealed["post_control_gate_failure_integrated_authorization"]

    assert config.attempt_id == "exp004-v11-integrated-s41-h"
    assert authorization["verified"] is True
    assert authorization["immutable_ledger_snapshot_verified"] is True
    assert authorization["attempt_g_identity"]["allocation_count"] == 1152
    assert authorization["attempt_g_identity"]["pre_export_identity_gate_passed"] == 1152
    assert authorization["attempt_g_identity"]["post_export_identity_gate_passed"] == 1152
    assert authorization["attempt_g_preservation_transaction_verified"] is True
    assert authorization["frozen_v10_control_replay"]["passed"] is True
    assert authorization["frozen_v10_control_replay"]["full_arrivals"] == 45
    assert authorization["frozen_v10_control_replay"]["completions_in_interval"] == 36
    assert authorization["prior_in_function_cleanup"] == "PASS"
    assert authorization["prior_provider_cleanup"] == "PASS"
    assert authorization["expanded_budget_verified"] is True


@pytest.mark.parametrize(
    ("mutate", "message"),
    (
        (
            lambda value: value.update(retry_attempt_id="exp004-v11-integrated-s41-g"),
            "authorization is inconsistent",
        ),
        (
            lambda value: value.update(unexpected=True),
            "authorization is inconsistent",
        ),
        (
            lambda value: value["immutable_ledger_snapshot"].update(sha256="0" * 64),
            "snapshot binding is malformed",
        ),
        (
            lambda value: value["expected_successful_integrated_contract"].update(
                full_export_authorized=False
            ),
            "result contract is inconsistent",
        ),
    ),
)
def test_control_gate_retry_h_rejects_authorization_or_contract_tamper(
    tmp_path: Path, mutate: Any, message: str
) -> None:
    config = _control_gate_retry_integrated_fixture(tmp_path)
    post_path = tmp_path / config.post_micro_review_manifest
    post = json.loads(post_path.read_text())
    mutate(post["post_control_gate_failure_integrated_authorization"])
    post_sha = _write_json(post_path, post)

    with pytest.raises(RuntimeError, match=message):
        CONTROLLER.verify_sealed_evidence(
            config.model_copy(update={"post_micro_review_manifest_sha256": post_sha}), tmp_path
        )


@pytest.mark.parametrize(
    ("group", "label"),
    (
        ("prior_attempt_evidence", "source_capture_commit"),
        ("frozen_v10_control_bindings", "serving_result"),
        ("control_gate_fix_bindings", "worker"),
        ("unchanged_runtime_bindings", "source_identity"),
    ),
)
def test_control_gate_retry_h_rejects_any_evidence_or_source_binding_tamper(
    tmp_path: Path, group: str, label: str
) -> None:
    config = _control_gate_retry_integrated_fixture(tmp_path)
    post_path = tmp_path / config.post_micro_review_manifest
    post = json.loads(post_path.read_text())
    post["post_control_gate_failure_integrated_authorization"][group][label]["sha256"] = "0" * 64
    post_sha = _write_json(post_path, post)

    with pytest.raises(RuntimeError, match="binding is malformed"):
        CONTROLLER.verify_sealed_evidence(
            config.model_copy(update={"post_micro_review_manifest_sha256": post_sha}), tmp_path
        )


@pytest.mark.parametrize("agent", (12, 13))
def test_control_gate_retry_h_rejects_review_tamper(tmp_path: Path, agent: int) -> None:
    config = _control_gate_retry_integrated_fixture(tmp_path)
    post_path = tmp_path / config.post_micro_review_manifest
    post = json.loads(post_path.read_text())
    review = next(
        row
        for row in post["post_control_gate_failure_integrated_authorization"]["independent_reviews"]
        if row["agent"] == agent
    )
    review["sha256"] = "0" * 64
    post_sha = _write_json(post_path, post)

    with pytest.raises(RuntimeError, match="review binding is inconsistent"):
        CONTROLLER.verify_sealed_evidence(
            config.model_copy(update={"post_micro_review_manifest_sha256": post_sha}), tmp_path
        )


def test_control_gate_retry_h_rejects_remote_manifest_child_drift(tmp_path: Path) -> None:
    config = _control_gate_retry_integrated_fixture(tmp_path)
    manifest_reference, _digest = CONTROLLER.CONTROL_GATE_RETRY_EVIDENCE_BINDINGS["remote_manifest"]
    manifest = json.loads((tmp_path / manifest_reference).read_text())
    directly_bound = {
        reference.rsplit(f"{CONTROLLER.CONTROL_GATE_FAILED_INTEGRATED_ATTEMPT_ID}/", 1)[-1]
        for reference, _digest in CONTROLLER.CONTROL_GATE_RETRY_EVIDENCE_BINDINGS.values()
        if f"{CONTROLLER.CONTROL_GATE_FAILED_INTEGRATED_ATTEMPT_ID}/" in reference
    }
    relative_path = next(
        row["relative_path"]
        for row in manifest["artifacts"]
        if row["relative_path"] not in directly_bound
    )
    raw_root = (tmp_path / manifest_reference).parent
    with (raw_root / relative_path).open("ab") as handle:
        handle.write(b"\n")

    with pytest.raises(RuntimeError, match="manifest child failed exact verification"):
        CONTROLLER.verify_sealed_evidence(config, tmp_path)


def test_control_gate_retry_h_rejects_preexisting_remote_output(tmp_path: Path) -> None:
    config = _control_gate_retry_integrated_fixture(tmp_path)
    raw_root = (
        tmp_path
        / "artifacts/branchfabric/gpu-validation/experiment-004/v11-final/integrated/raw"
        / CONTROLLER.CONTROL_GATE_RETRY_INTEGRATED_ATTEMPT_ID
    )
    raw_root.mkdir(parents=True)

    with pytest.raises(RuntimeError, match="pre-existing remote evidence"):
        CONTROLLER.verify_sealed_evidence(config, tmp_path)


def test_image_closure_retry_i_accepts_exact_h_failure_fix_target_b_and_budget(
    tmp_path: Path,
) -> None:
    config = _image_closure_retry_integrated_fixture(tmp_path)

    sealed = CONTROLLER.verify_sealed_evidence(config, tmp_path)
    authorization = sealed["post_image_closure_failure_integrated_authorization"]

    assert config.attempt_id == "exp004-v11-integrated-s41-i"
    assert authorization["verified"] is True
    assert authorization["immutable_ledger_snapshot_verified"] is True
    assert authorization["attempt_h_pre_worker_failure_verified"] is True
    assert authorization["target_b_identity"]["allocation_count"] == 1152
    assert authorization["target_b_identity"]["zero_post_commit_events"] is True
    assert authorization["prior_in_function_cleanup"] == "PASS"
    assert authorization["prior_provider_cleanup"] == "PASS"
    assert authorization["expanded_budget_verified"] is True
    assert authorization["successful_integrated_contract_verified"] is True


@pytest.mark.parametrize(
    ("mutate", "message"),
    (
        (
            lambda value: value.update(retry_attempt_id="exp004-v11-integrated-s41-h"),
            "authorization is inconsistent",
        ),
        (
            lambda value: value.update(unexpected=True),
            "authorization is inconsistent",
        ),
        (
            lambda value: value["immutable_ledger_snapshot"].update(sha256="0" * 64),
            "snapshot binding is malformed",
        ),
        (
            lambda value: value["expected_successful_integrated_contract"].update(
                full_export_authorized=False
            ),
            "result contract is inconsistent",
        ),
    ),
)
def test_image_closure_retry_i_rejects_authorization_or_contract_tamper(
    tmp_path: Path,
    mutate: Any,
    message: str,
) -> None:
    config = _image_closure_retry_integrated_fixture(tmp_path)
    post_path = tmp_path / config.post_micro_review_manifest
    post = json.loads(post_path.read_text())
    mutate(post["post_image_closure_failure_integrated_authorization"])
    post_sha = _write_json(post_path, post)

    with pytest.raises(RuntimeError, match=message):
        CONTROLLER.verify_sealed_evidence(
            config.model_copy(update={"post_micro_review_manifest_sha256": post_sha}),
            tmp_path,
        )


@pytest.mark.parametrize(
    ("group", "label"),
    (
        ("prior_attempt_evidence", "status"),
        ("image_closure_fix_bindings", "modal_launcher"),
        ("unchanged_allocator_movement_bindings", "source_identity"),
        ("target_b_identity_evidence", "identity_gate"),
    ),
)
def test_image_closure_retry_i_rejects_any_evidence_or_source_binding_tamper(
    tmp_path: Path,
    group: str,
    label: str,
) -> None:
    config = _image_closure_retry_integrated_fixture(tmp_path)
    post_path = tmp_path / config.post_micro_review_manifest
    post = json.loads(post_path.read_text())
    post["post_image_closure_failure_integrated_authorization"][group][label]["sha256"] = "0" * 64
    post_sha = _write_json(post_path, post)

    with pytest.raises(RuntimeError, match="binding is malformed"):
        CONTROLLER.verify_sealed_evidence(
            config.model_copy(update={"post_micro_review_manifest_sha256": post_sha}),
            tmp_path,
        )


@pytest.mark.parametrize("agent", (12, 13))
def test_image_closure_retry_i_rejects_review_tamper(tmp_path: Path, agent: int) -> None:
    config = _image_closure_retry_integrated_fixture(tmp_path)
    post_path = tmp_path / config.post_micro_review_manifest
    post = json.loads(post_path.read_text())
    review = next(
        row
        for row in post["post_image_closure_failure_integrated_authorization"][
            "independent_reviews"
        ]
        if row["agent"] == agent
    )
    review["sha256"] = "0" * 64
    post_sha = _write_json(post_path, post)

    with pytest.raises(RuntimeError, match="review binding is inconsistent"):
        CONTROLLER.verify_sealed_evidence(
            config.model_copy(update={"post_micro_review_manifest_sha256": post_sha}),
            tmp_path,
        )


def test_image_closure_retry_i_rejects_h_remote_manifest_extra_child(tmp_path: Path) -> None:
    config = _image_closure_retry_integrated_fixture(tmp_path)
    manifest_reference, _digest = CONTROLLER.IMAGE_CLOSURE_RETRY_EVIDENCE_BINDINGS[
        "remote_manifest"
    ]
    extra = (tmp_path / manifest_reference).parent / "unexpected.json"
    extra.write_text("{}\n")

    with pytest.raises(RuntimeError, match="does not exactly cover raw evidence"):
        CONTROLLER.verify_sealed_evidence(config, tmp_path)


def test_image_closure_retry_i_rejects_rehashed_invalid_h_status(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _image_closure_retry_integrated_fixture(tmp_path)
    reference, _digest = CONTROLLER.IMAGE_CLOSURE_RETRY_EVIDENCE_BINDINGS["status"]
    status_path = tmp_path / reference
    status = json.loads(status_path.read_text())
    status["scientifically_valid"] = True
    digest = _write_json(status_path, status)
    expected = dict(CONTROLLER.IMAGE_CLOSURE_RETRY_EVIDENCE_BINDINGS)
    expected["status"] = (reference, digest)
    monkeypatch.setattr(CONTROLLER, "IMAGE_CLOSURE_RETRY_EVIDENCE_BINDINGS", expected)
    post_path = tmp_path / config.post_micro_review_manifest
    post = json.loads(post_path.read_text())
    post["post_image_closure_failure_integrated_authorization"]["prior_attempt_evidence"]["status"][
        "sha256"
    ] = digest
    post_sha = _write_json(post_path, post)

    with pytest.raises(RuntimeError, match="terminal status is inconsistent"):
        CONTROLLER.verify_sealed_evidence(
            config.model_copy(update={"post_micro_review_manifest_sha256": post_sha}),
            tmp_path,
        )


def test_image_closure_retry_i_rejects_snapshot_drift_and_preexisting_output(
    tmp_path: Path,
) -> None:
    config = _image_closure_retry_integrated_fixture(tmp_path)
    snapshot_reference, _digest = CONTROLLER.IMAGE_CLOSURE_RETRY_LEDGER_SNAPSHOT_BINDING
    with (tmp_path / snapshot_reference).open("ab") as handle:
        handle.write(b"\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        CONTROLLER.verify_sealed_evidence(config, tmp_path)

    config = _image_closure_retry_integrated_fixture(tmp_path / "fresh")
    raw_root = (
        tmp_path
        / "fresh/artifacts/branchfabric/gpu-validation/experiment-004/v11-final/integrated/raw"
        / CONTROLLER.IMAGE_CLOSURE_RETRY_INTEGRATED_ATTEMPT_ID
    )
    raw_root.mkdir(parents=True)
    with pytest.raises(RuntimeError, match="pre-existing remote evidence"):
        CONTROLLER.verify_sealed_evidence(config, tmp_path / "fresh")


def test_remote_prefix_retry_j_accepts_exact_i_preflight_fix_target_b_and_budget(
    tmp_path: Path,
) -> None:
    config = _remote_prefix_retry_integrated_fixture(tmp_path)

    sealed = CONTROLLER.verify_sealed_evidence(config, tmp_path)
    authorization = sealed["post_remote_prefix_parser_failure_integrated_authorization"]

    assert config.attempt_id == "exp004-v11-integrated-s41-j"
    assert authorization["verified"] is True
    assert authorization["prior_attempt_pre_reservation_failure_verified"] is True
    assert authorization["immutable_ledger_snapshot_verified"] is True
    assert authorization["target_b_identity"]["allocation_count"] == 1152
    assert authorization["target_b_identity"]["zero_post_commit_events"] is True
    assert authorization["expanded_budget_verified"] is True
    assert authorization["successful_integrated_contract_verified"] is True


@pytest.mark.parametrize(
    ("mutate", "message"),
    (
        (
            lambda value: value.update(retry_attempt_id="exp004-v11-integrated-s41-i"),
            "authorization is inconsistent",
        ),
        (
            lambda value: value.update(unexpected=True),
            "authorization is inconsistent",
        ),
        (
            lambda value: value["immutable_ledger_snapshot"].update(sha256="0" * 64),
            "snapshot binding is malformed",
        ),
        (
            lambda value: value["expected_successful_integrated_contract"].update(
                source_release_authorized=False
            ),
            "result contract is inconsistent",
        ),
    ),
)
def test_remote_prefix_retry_j_rejects_authorization_or_contract_tamper(
    tmp_path: Path,
    mutate: Any,
    message: str,
) -> None:
    config = _remote_prefix_retry_integrated_fixture(tmp_path)
    post_path = tmp_path / config.post_micro_review_manifest
    post = json.loads(post_path.read_text())
    mutate(post["post_remote_prefix_parser_failure_integrated_authorization"])
    post_sha = _write_json(post_path, post)

    with pytest.raises(RuntimeError, match=message):
        CONTROLLER.verify_sealed_evidence(
            config.model_copy(update={"post_micro_review_manifest_sha256": post_sha}),
            tmp_path,
        )


@pytest.mark.parametrize(
    ("group", "label"),
    (
        ("prior_attempt_evidence", "status"),
        ("remote_prefix_parser_fix_bindings", "sole_coordinator"),
        ("unchanged_scientific_runtime_bindings", "integrated_worker"),
        ("target_b_identity_evidence", "identity_gate"),
    ),
)
def test_remote_prefix_retry_j_rejects_evidence_or_source_binding_tamper(
    tmp_path: Path,
    group: str,
    label: str,
) -> None:
    config = _remote_prefix_retry_integrated_fixture(tmp_path)
    post_path = tmp_path / config.post_micro_review_manifest
    post = json.loads(post_path.read_text())
    authorization = post["post_remote_prefix_parser_failure_integrated_authorization"]
    authorization[group][label]["sha256"] = "0" * 64
    post_sha = _write_json(post_path, post)

    with pytest.raises(RuntimeError, match="binding is malformed"):
        CONTROLLER.verify_sealed_evidence(
            config.model_copy(update={"post_micro_review_manifest_sha256": post_sha}),
            tmp_path,
        )


@pytest.mark.parametrize(
    ("label", "mutate", "message"),
    (
        (
            "status",
            lambda value: value["lifecycle"].update(gpu_reservation_created=True),
            "terminal status is inconsistent",
        ),
        (
            "remote_prefix_inventory",
            lambda value: value["stdout_rows"].pop(),
            "inventory is inconsistent",
        ),
    ),
)
def test_remote_prefix_retry_j_rejects_rehashed_semantic_evidence_tamper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    label: str,
    mutate: Any,
    message: str,
) -> None:
    config = _remote_prefix_retry_integrated_fixture(tmp_path)
    reference, _digest = CONTROLLER.REMOTE_PREFIX_RETRY_EVIDENCE_BINDINGS[label]
    evidence_path = tmp_path / reference
    evidence = json.loads(evidence_path.read_text())
    mutate(evidence)
    digest = _write_json(evidence_path, evidence)
    expected = dict(CONTROLLER.REMOTE_PREFIX_RETRY_EVIDENCE_BINDINGS)
    expected[label] = (reference, digest)
    if label == "remote_prefix_inventory":
        status_reference, _status_digest = expected["status"]
        status_path = tmp_path / status_reference
        status = json.loads(status_path.read_text())
        status["sealed_inputs"][label]["sha256"] = digest
        status_digest = _write_json(status_path, status)
        expected["status"] = (status_reference, status_digest)
    monkeypatch.setattr(CONTROLLER, "REMOTE_PREFIX_RETRY_EVIDENCE_BINDINGS", expected)
    post_path = tmp_path / config.post_micro_review_manifest
    post = json.loads(post_path.read_text())
    post["post_remote_prefix_parser_failure_integrated_authorization"]["prior_attempt_evidence"][
        label
    ]["sha256"] = digest
    if label == "remote_prefix_inventory":
        post["post_remote_prefix_parser_failure_integrated_authorization"][
            "prior_attempt_evidence"
        ]["status"]["sha256"] = expected["status"][1]
    post_sha = _write_json(post_path, post)

    with pytest.raises(RuntimeError, match=message):
        CONTROLLER.verify_sealed_evidence(
            config.model_copy(update={"post_micro_review_manifest_sha256": post_sha}),
            tmp_path,
        )


def test_remote_prefix_retry_j_rejects_snapshot_drift_and_preexisting_output(
    tmp_path: Path,
) -> None:
    config = _remote_prefix_retry_integrated_fixture(tmp_path)
    snapshot_reference, _digest = CONTROLLER.REMOTE_PREFIX_RETRY_LEDGER_SNAPSHOT_BINDING
    with (tmp_path / snapshot_reference).open("ab") as handle:
        handle.write(b"\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        CONTROLLER.verify_sealed_evidence(config, tmp_path)

    config = _remote_prefix_retry_integrated_fixture(tmp_path / "fresh")
    raw_root = (
        tmp_path
        / "fresh/artifacts/branchfabric/gpu-validation/experiment-004/v11-final/integrated/raw"
        / CONTROLLER.REMOTE_PREFIX_RETRY_INTEGRATED_ATTEMPT_ID
    )
    raw_root.mkdir(parents=True)
    with pytest.raises(RuntimeError, match="pre-existing remote evidence"):
        CONTROLLER.verify_sealed_evidence(config, tmp_path / "fresh")


def test_retained_capacity_retry_k_accepts_exact_j_failure_fix_reviews_and_budget(
    tmp_path: Path,
) -> None:
    config = _retained_capacity_retry_integrated_fixture(tmp_path)

    sealed = CONTROLLER.verify_sealed_evidence(config, tmp_path)
    authorization = sealed["post_retained_capacity_failure_integrated_authorization"]

    assert config.attempt_id == "exp004-v11-integrated-s41-k"
    assert authorization["verified"] is True
    assert authorization["attempt_j_readiness_assessment"]["status"] == "BELOW_FLOOR"
    assert authorization["attempt_j_measurement_window_completions"] == 20
    assert authorization["attempt_j_terminal_counts"] == {"completed": 31, "aborted": 17}
    assert authorization["prior_in_function_cleanup"] == "PASS"
    assert authorization["prior_provider_cleanup"] == "PASS"
    assert authorization["target_b_identity"]["allocation_count"] == 1152
    assert authorization["target_b_identity"]["zero_post_commit_events"] is True
    assert authorization["immutable_ledger_snapshot_verified"] is True
    assert authorization["expanded_budget_verified"] is True
    assert authorization["successful_integrated_contract_verified"] is True


@pytest.mark.parametrize(
    ("mutate", "message"),
    (
        (
            lambda value: value.update(retry_attempt_id="exp004-v11-integrated-s41-j"),
            "authorization is inconsistent",
        ),
        (
            lambda value: value.update(unexpected=True),
            "authorization is inconsistent",
        ),
        (
            lambda value: value["immutable_ledger_snapshot"].update(sha256="0" * 64),
            "snapshot binding is malformed",
        ),
        (
            lambda value: value["expected_successful_integrated_contract"].update(
                source_release_authorized=False
            ),
            "result contract is inconsistent",
        ),
    ),
)
def test_retained_capacity_retry_k_rejects_authorization_or_contract_tamper(
    tmp_path: Path,
    mutate: Any,
    message: str,
) -> None:
    config = _retained_capacity_retry_integrated_fixture(tmp_path)
    post_path = tmp_path / config.post_micro_review_manifest
    post = json.loads(post_path.read_text())
    mutate(post["post_retained_capacity_failure_integrated_authorization"])
    post_sha = _write_json(post_path, post)

    with pytest.raises(RuntimeError, match=message):
        CONTROLLER.verify_sealed_evidence(
            config.model_copy(update={"post_micro_review_manifest_sha256": post_sha}),
            tmp_path,
        )


@pytest.mark.parametrize(
    ("group", "label", "message"),
    (
        ("prior_attempt_evidence", "status", "binding is malformed"),
        (
            "readiness_and_sanity_fix_bindings",
            "three_probe_scoped_worker_wrapper",
            "binding is malformed",
        ),
        (
            "unchanged_scientific_runtime_bindings",
            "frozen_sanity_assessor",
            "binding is malformed",
        ),
        ("unchanged_scientific_runtime_bindings", "source_identity", "binding is malformed"),
        ("unchanged_allocator_movement_bindings", "allocator_epochs", "binding is malformed"),
        ("target_b_identity_evidence", "identity_gate", "binding is malformed"),
    ),
)
def test_retained_capacity_retry_k_rejects_bound_evidence_or_source_tamper(
    tmp_path: Path,
    group: str,
    label: str,
    message: str,
) -> None:
    config = _retained_capacity_retry_integrated_fixture(tmp_path)
    post_path = tmp_path / config.post_micro_review_manifest
    post = json.loads(post_path.read_text())
    authorization = post["post_retained_capacity_failure_integrated_authorization"]
    authorization[group][label]["sha256"] = "0" * (
        63 if group == "readiness_and_sanity_fix_bindings" else 64
    )
    post_sha = _write_json(post_path, post)

    with pytest.raises(RuntimeError, match=message):
        CONTROLLER.verify_sealed_evidence(
            config.model_copy(update={"post_micro_review_manifest_sha256": post_sha}),
            tmp_path,
        )


@pytest.mark.parametrize("agent", (12, 13))
def test_retained_capacity_retry_k_rejects_review_tamper(
    tmp_path: Path,
    agent: int,
) -> None:
    config = _retained_capacity_retry_integrated_fixture(tmp_path)
    post_path = tmp_path / config.post_micro_review_manifest
    post = json.loads(post_path.read_text())
    reviews = post["post_retained_capacity_failure_integrated_authorization"]["independent_reviews"]
    next(row for row in reviews if row["agent"] == agent)["sha256"] = "0" * 63
    post_sha = _write_json(post_path, post)

    with pytest.raises(RuntimeError, match=f"agent{agent} review binding is malformed"):
        CONTROLLER.verify_sealed_evidence(
            config.model_copy(update={"post_micro_review_manifest_sha256": post_sha}),
            tmp_path,
        )


def test_retained_capacity_retry_k_rejects_rehashed_invalid_j_status(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _retained_capacity_retry_integrated_fixture(tmp_path)
    reference, _digest = CONTROLLER.RETAINED_CAPACITY_RETRY_EVIDENCE_BINDINGS["status"]
    status_path = tmp_path / reference
    status = json.loads(status_path.read_text())
    status["scientifically_valid"] = True
    digest = _write_json(status_path, status)
    expected = dict(CONTROLLER.RETAINED_CAPACITY_RETRY_EVIDENCE_BINDINGS)
    expected["status"] = (reference, digest)
    monkeypatch.setattr(CONTROLLER, "RETAINED_CAPACITY_RETRY_EVIDENCE_BINDINGS", expected)
    post_path = tmp_path / config.post_micro_review_manifest
    post = json.loads(post_path.read_text())
    post["post_retained_capacity_failure_integrated_authorization"]["prior_attempt_evidence"][
        "status"
    ]["sha256"] = digest
    post_sha = _write_json(post_path, post)

    with pytest.raises(RuntimeError, match="terminal runtime chain is inconsistent"):
        CONTROLLER.verify_sealed_evidence(
            config.model_copy(update={"post_micro_review_manifest_sha256": post_sha}),
            tmp_path,
        )


def test_retained_capacity_retry_k_rejects_resealed_modified_runtime_source(
    tmp_path: Path,
) -> None:
    config = _retained_capacity_retry_integrated_fixture(tmp_path)
    worker_reference = CONTROLLER.RETAINED_CAPACITY_FIX_BINDING_PATHS[
        "three_probe_scoped_worker_wrapper"
    ]
    worker_path = tmp_path / worker_reference
    worker_path.write_bytes(worker_path.read_bytes() + b"\n# unreviewed runtime mutation\n")
    worker_sha = hashlib.sha256(worker_path.read_bytes()).hexdigest()
    post_path = tmp_path / config.post_micro_review_manifest
    post = json.loads(post_path.read_text())
    post["post_retained_capacity_failure_integrated_authorization"][
        "readiness_and_sanity_fix_bindings"
    ]["three_probe_scoped_worker_wrapper"]["sha256"] = worker_sha
    prelaunch_binding = next(
        row
        for row in post["integrated_prelaunch"]["bindings"]
        if row["artifact"] == worker_reference
    )
    prelaunch_binding["sha256"] = worker_sha
    post_sha = _write_json(post_path, post)

    with pytest.raises(RuntimeError, match="independent review conclusions"):
        CONTROLLER.verify_sealed_evidence(
            config.model_copy(update={"post_micro_review_manifest_sha256": post_sha}),
            tmp_path,
        )


def test_retained_capacity_retry_k_rejects_manifest_snapshot_budget_and_output_drift(
    tmp_path: Path,
) -> None:
    config = _retained_capacity_retry_integrated_fixture(tmp_path / "manifest")
    manifest_reference, _digest = CONTROLLER.RETAINED_CAPACITY_RETRY_EVIDENCE_BINDINGS[
        "remote_manifest"
    ]
    ((tmp_path / "manifest" / manifest_reference).parent / "unexpected.json").write_text("{}\n")
    with pytest.raises(RuntimeError, match="set equality failed"):
        CONTROLLER.verify_sealed_evidence(config, tmp_path / "manifest")

    config = _retained_capacity_retry_integrated_fixture(tmp_path / "symlink")
    raw_reference = (
        "artifacts/branchfabric/gpu-validation/experiment-004/v11-final/integrated/raw/"
        f"{CONTROLLER.RETAINED_CAPACITY_FAILED_INTEGRATED_ATTEMPT_ID}"
    )
    (tmp_path / "symlink" / raw_reference / "unmanifested-directory").symlink_to(
        tmp_path / "symlink",
        target_is_directory=True,
    )
    with pytest.raises(RuntimeError, match="tree contains a symlink"):
        CONTROLLER.verify_sealed_evidence(config, tmp_path / "symlink")

    config = _retained_capacity_retry_integrated_fixture(tmp_path / "root-symlink")
    raw_root = tmp_path / "root-symlink" / raw_reference
    backing_root = raw_root.with_name(f"{raw_root.name}-backing")
    raw_root.rename(backing_root)
    raw_root.symlink_to(backing_root, target_is_directory=True)
    with pytest.raises(RuntimeError, match="remote manifest root is invalid"):
        CONTROLLER.verify_sealed_evidence(config, tmp_path / "root-symlink")

    config = _retained_capacity_retry_integrated_fixture(tmp_path / "snapshot")
    snapshot_reference, _digest = CONTROLLER.RETAINED_CAPACITY_RETRY_LEDGER_SNAPSHOT_BINDING
    with (tmp_path / "snapshot" / snapshot_reference).open("ab") as handle:
        handle.write(b"\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        CONTROLLER.verify_sealed_evidence(config, tmp_path / "snapshot")

    config = _retained_capacity_retry_integrated_fixture(tmp_path / "budget")
    budget_path = tmp_path / "budget" / config.budget_authorization
    budget = json.loads(budget_path.read_text())
    budget["authorized_cumulative_gpu_seconds"] = 24_000.0
    budget_sha = _write_json(budget_path, budget)
    with pytest.raises(RuntimeError, match="snapshot or budget is inconsistent"):
        CONTROLLER.verify_sealed_evidence(
            config.model_copy(update={"budget_authorization_sha256": budget_sha}),
            tmp_path / "budget",
        )

    config = _retained_capacity_retry_integrated_fixture(tmp_path / "output")
    raw_root = (
        tmp_path / "output/artifacts/branchfabric/gpu-validation/experiment-004/v11-final/"
        "integrated/raw" / CONTROLLER.RETAINED_CAPACITY_RETRY_INTEGRATED_ATTEMPT_ID
    )
    raw_root.mkdir(parents=True)
    with pytest.raises(RuntimeError, match="pre-existing remote evidence"):
        CONTROLLER.verify_sealed_evidence(config, tmp_path / "output")


def _attempt_f_bound_evidence() -> dict[str, dict[str, Any]]:
    return {
        label: json.loads((ROOT / reference).read_text())
        for label, (
            reference,
            _digest,
        ) in CONTROLLER.QUEUE_ACCOUNTING_RETRY_EVIDENCE_BINDINGS.items()
    }


def _overlap_attempt_f_gpu0_gpu1_completion(
    partial: dict[str, Any], telemetry: list[dict[str, Any]]
) -> None:
    gpu1_ids = {
        row["request_id"] for evidence in telemetry for row in evidence["completed_requests"]
    }
    gpu0_request_id = next(
        row["request_id"]
        for row in partial["gpu0_requests"]
        if row.get("completed_ns") is not None and row["request_id"] not in gpu1_ids
    )
    completed = next(
        evidence["completed_requests"] for evidence in telemetry if evidence["completed_requests"]
    )
    completed[0]["request_id"] = gpu0_request_id


@pytest.mark.parametrize(
    ("mutate", "message"),
    (
        (
            lambda partial, _telemetry: partial["global_offered_requests"].pop(),
            "raw queue-accounting inputs",
        ),
        (
            lambda partial, _telemetry: partial["queue_state"].update(total_outstanding=63),
            "raw queue-accounting inputs",
        ),
        (
            lambda _partial, telemetry: telemetry[1].update(sequence=7),
            "telemetry replay is malformed",
        ),
        (
            lambda _partial, telemetry: telemetry[2].update(
                observed_ns=telemetry[1]["observed_ns"]
            ),
            "telemetry replay is malformed",
        ),
        (
            lambda _partial, telemetry: telemetry[2].update(cumulative_completed_requests=7),
            "cumulative completion count",
        ),
        (
            _overlap_attempt_f_gpu0_gpu1_completion,
            "completion ownership is not disjoint",
        ),
        (
            lambda partial, _telemetry: partial["global_offered_requests"][-1].update(
                offered_ns=partial["observed_ns"] + 1
            ),
            "offered-request timestamp",
        ),
    ),
)
def test_attempt_f_queue_replay_rejects_raw_stale_or_ordering_tamper(
    mutate: Any, message: str
) -> None:
    evidence = _attempt_f_bound_evidence()
    partial = copy.deepcopy(evidence["partial_failure"])
    telemetry = [copy.deepcopy(evidence[f"gpu1_telemetry_{sequence:06d}"]) for sequence in range(3)]
    mutate(partial, telemetry)

    with pytest.raises(RuntimeError, match=message):
        CONTROLLER._verify_attempt_f_queue_accounting(partial, telemetry)


@pytest.mark.parametrize(
    ("label", "mutate", "message"),
    (
        (
            "pre_export_identity",
            lambda value: value.update(exact_block_epoch_identity=False),
            "pre-export identity gate",
        ),
        (
            "post_export_identity",
            lambda value: value["post_export_allocation_notifications"].update(
                observed_event_count=1
            ),
            "post-export identity gate",
        ),
        (
            "source_capture_commit",
            lambda value: value["commit"]["allocations"][0].update(allocation_epoch=999999),
            "SourceCaptureCommit semantics",
        ),
    ),
)
def test_attempt_f_identity_recomputation_rejects_semantic_or_notification_tamper(
    tmp_path: Path, label: str, mutate: Any, message: str
) -> None:
    config = _queue_accounting_retry_integrated_fixture(tmp_path)
    evidence = _attempt_f_bound_evidence()
    evidence[label] = copy.deepcopy(evidence[label])
    mutate(evidence[label])

    with pytest.raises(RuntimeError, match=message):
        CONTROLLER._verify_attempt_f_identity_evidence(evidence, config=config)


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


def test_integrated_reclaim_partition_exposes_every_identity_boundary() -> None:
    stages = _worker_results()[1]["critical_paths"]["reclamation"]["stages"]
    assert [item["stage"] for item in stages] == [
        "branch_quiesce",
        "source_authentication",
        "pre_commit_allocation_history_retirement",
        "source_capture_commit",
        "pre_export_identity_validation_and_commit_publish",
        "fused_gather_repack_d2h",
        "post_export_read_identity_validation",
        "state_publish_and_allocation_history_retirement",
        "source_release",
        "hbm_reclaim_confirmation",
        "gpu1_serving_ready",
    ]
    assert sum(item["wall_time_ns"] for item in stages) == 1_100_000_000


def test_targeted_worker_results_stop_at_shared_1152_identity_gate() -> None:
    rows = _targeted_worker_results()
    expected = {"serving": (101, "GPU-a"), "rollout": (202, "GPU-b")}
    assert (
        CONTROLLER._validate_worker_results(
            rows,
            expected=expected,
            attempt_id="exp004-v11-targeted-identity-s41-a",
            targeted=True,
        )
        == rows
    )


@pytest.mark.parametrize(
    ("target", "message"),
    (
        ("schema", "result failed identity/status gates"),
        ("export", "exceeded its authorized terminal phase"),
        ("release", "exceeded its authorized terminal phase"),
        ("trigger", "early trigger evidence failed closed"),
        ("trigger_clock", "early trigger evidence failed closed"),
        ("queue", "serving result failed trigger/terminal/queue gates"),
        ("queue_missing", "serving result failed trigger/terminal/queue gates"),
        ("allocation_count", "source identity evidence failed closed"),
        ("device", "source identity evidence failed closed"),
        ("history", "source identity evidence failed closed"),
        ("notification", "source identity evidence failed closed"),
        ("cross_role", "source identity barrier differs across workers"),
    ),
)
def test_targeted_worker_results_reject_terminal_or_identity_drift(
    target: str, message: str
) -> None:
    rows = json.loads(json.dumps(_targeted_worker_results()))
    if target == "schema":
        rows[1]["schema_version"] = "sloforge.branchfabric.experiment-004-v11-gpu1-result/v1"
    elif target == "export":
        rows[0]["optimized_export_started"] = True
    elif target == "release":
        rows[1]["transaction_source_release_started"] = True
    elif target == "trigger":
        rows[0]["reclamation_trigger_evidence"]["queue_depth_at_trigger"] = 64
        rows[0]["reclamation_trigger_evidence"]["emergency_ceiling_headroom_requests"] = 0
    elif target == "trigger_clock":
        rows[0]["reclamation_trigger_evidence"]["events"]["RECLAIM_TRIGGER_EMITTED"] = 998
    elif target == "queue":
        rows[0]["final_runtime_queue_state"]["running_requests"] = 1
    elif target == "queue_missing":
        rows[0]["final_runtime_queue_state"].pop("request_count")
    elif target == "allocation_count":
        rows[1]["source_identity_gate"]["observed_allocation_count"] = 1_151
    elif target == "device":
        rows[1]["source_identity_gate"]["device_identity_pass"] = False
    elif target == "history":
        rows[1]["allocation_history_retirement"]["semantic_identity_claimed"] = True
    elif target == "notification":
        rows[1]["post_commit_allocation_notifications"]["queue_empty"] = False
    else:
        rows[0]["source_identity_terminal"]["source_capture_commit_sha256"] = "f" * 64
    with pytest.raises(RuntimeError, match=message):
        CONTROLLER._validate_worker_results(
            tuple(rows),
            expected={"serving": (101, "GPU-a"), "rollout": (202, "GPU-b")},
            attempt_id="exp004-v11-targeted-identity-s41-a",
            targeted=True,
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


@pytest.mark.parametrize("completions_in_interval", (33, 35, 37, 38, 45))
def test_gpu0_control_validator_accepts_one_sided_ninety_percent_boundary(
    completions_in_interval: int,
) -> None:
    rows = list(_worker_results())
    serving = rows[0]
    raw = _control_serving_fixture(completions_in_interval=completions_in_interval)
    serving["serving"].update(raw)
    control = CONTROLLER._control_interval_evidence(
        serving["serving"],
        expected_rate_rps=9.0,
        expected_duration_seconds=5.0,
        slo_ttft_seconds=2.0,
    )
    assert control["completions_in_interval"] == completions_in_interval
    assert control["completion_tracks_offer"] is True
    assert control["passed"] is True
    serving["gpu0_control_interval"] = control
    serving["gpu0_restore_interference"]["control_interval"] = control

    CONTROLLER._validate_worker_results(
        tuple(rows),
        expected={"serving": (101, "GPU-a"), "rollout": (202, "GPU-b")},
        attempt_id="exp004-v11-integrated-s41-a",
    )


def test_gpu0_control_validator_rejects_below_ninety_percent_boundary() -> None:
    completions_in_interval = 32
    rows = list(_worker_results())
    serving = rows[0]
    raw = _control_serving_fixture(completions_in_interval=completions_in_interval)
    serving["serving"].update(raw)
    control = CONTROLLER._control_interval_evidence(
        serving["serving"],
        expected_rate_rps=9.0,
        expected_duration_seconds=5.0,
        slo_ttft_seconds=2.0,
    )
    assert control["completions_in_interval"] == completions_in_interval
    assert control["completion_tracks_offer"] is False
    assert control["passed"] is False
    serving["gpu0_control_interval"] = control
    serving["gpu0_restore_interference"]["control_interval"] = control

    with pytest.raises(RuntimeError, match="control-interval evidence is invalid"):
        CONTROLLER._validate_worker_results(
            tuple(rows),
            expected={"serving": (101, "GPU-a"), "rollout": (202, "GPU-b")},
            attempt_id="exp004-v11-integrated-s41-a",
        )


def test_gpu0_control_validator_rejects_raw_and_emitted_evidence_mismatch() -> None:
    rows = list(_worker_results())
    rows[0]["serving"]["requests"][10]["output_token_ids"].pop()

    with pytest.raises(RuntimeError, match="control-interval evidence is invalid"):
        CONTROLLER._validate_worker_results(
            tuple(rows),
            expected={"serving": (101, "GPU-a"), "rollout": (202, "GPU-b")},
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


def test_targeted_transaction_command_forbids_export_and_release(tmp_path: Path) -> None:
    config = _targeted_fixture(tmp_path)
    expanded = CONTROLLER.expanded_runtime_config(config)
    command = CONTROLLER._transaction_command(
        config=config,
        effective_config=expanded,
        selection_sha256="1" * 64,
        sanity_result_sha256="2" * 64,
    )
    assert command["schema_version"] == (
        "sloforge.branchfabric.targeted-source-identity-command/v1"
    )
    assert command["terminal_phase"] == "SOURCE_ALLOCATION_IDENTITY_GATE"
    assert command["full_export_authorized"] is False
    assert command["source_release_authorized"] is False
    assert command["effective_config"] == expanded
    assert expanded["execution_mode"] == "targeted-source-identity-v11"
    with pytest.raises(RuntimeError, match="differs from expanded"):
        CONTROLLER._transaction_command(
            config=config,
            effective_config=expanded | {"source_release_authorized": True},
            selection_sha256="1" * 64,
            sanity_result_sha256="2" * 64,
        )


def test_targeted_transaction_command_round_trips_through_shared_worker_validator(
    tmp_path: Path,
) -> None:
    config = _targeted_fixture(tmp_path)
    expanded = CONTROLLER.expanded_runtime_config(config)
    command = CONTROLLER._transaction_command(
        config=config,
        effective_config=expanded,
        selection_sha256="1" * 64,
        sanity_result_sha256="2" * 64,
    )

    effective, evidence_hashes = INTEGRATED_WORKER._validate_v11_transaction_command(
        base_config=expanded,
        command=command,
        selected_load_sha256="1" * 64,
        frozen_validator=SHARED_WORKER._validate_integrated_transaction_command,
    )

    assert effective == expanded
    assert evidence_hashes == {
        "authorization_artifact_hash": config.budget_authorization_sha256,
        "sanity_result_sha256": "2" * 64,
    }


@pytest.mark.parametrize(
    ("field", "tampered", "error"),
    (
        (
            "schema_version",
            "sloforge.branchfabric.integrated-transaction-command/v1",
            "schema differs",
        ),
        ("terminal_phase", "INTEGRATED_TRANSACTION", "safety field terminal_phase"),
        ("full_export_authorized", True, "safety field full_export_authorized"),
        ("source_release_authorized", True, "safety field source_release_authorized"),
    ),
)
def test_targeted_transaction_command_consumer_rejects_safety_tampering(
    tmp_path: Path,
    field: str,
    tampered: object,
    error: str,
) -> None:
    config = _targeted_fixture(tmp_path)
    expanded = CONTROLLER.expanded_runtime_config(config)
    command = CONTROLLER._transaction_command(
        config=config,
        effective_config=expanded,
        selection_sha256="1" * 64,
        sanity_result_sha256="2" * 64,
    )
    command[field] = tampered

    with pytest.raises(ValueError, match=error):
        INTEGRATED_WORKER._validate_v11_transaction_command(
            base_config=expanded,
            command=command,
            selected_load_sha256="1" * 64,
            frozen_validator=SHARED_WORKER._validate_integrated_transaction_command,
        )


def test_integrated_transaction_command_round_trips_through_v11_validator(
    tmp_path: Path,
) -> None:
    config = _sealed_fixture(tmp_path)
    expanded = CONTROLLER.expanded_runtime_config(config)
    command = CONTROLLER._transaction_command(
        config=config,
        effective_config=expanded,
        selection_sha256="1" * 64,
        sanity_result_sha256="2" * 64,
    )

    effective, evidence_hashes = INTEGRATED_WORKER._validate_v11_transaction_command(
        base_config=expanded,
        command=command,
        selected_load_sha256="1" * 64,
        frozen_validator=SHARED_WORKER._validate_integrated_transaction_command,
    )

    assert effective == expanded
    assert evidence_hashes == {
        "authorization_artifact_hash": config.budget_authorization_sha256,
        "sanity_result_sha256": "2" * 64,
    }


@pytest.mark.parametrize("mutation", ("missing", "extra"))
def test_v11_transaction_command_rejects_inexact_envelope_fields(
    tmp_path: Path,
    mutation: str,
) -> None:
    config = _targeted_fixture(tmp_path)
    expanded = CONTROLLER.expanded_runtime_config(config)
    command = CONTROLLER._transaction_command(
        config=config,
        effective_config=expanded,
        selection_sha256="1" * 64,
        sanity_result_sha256="2" * 64,
    )
    if mutation == "missing":
        del command["sanity_result_sha256"]
    else:
        command["unexpected_field"] = "unexpected"

    with pytest.raises(ValueError, match="fields differ from its typed envelope"):
        INTEGRATED_WORKER._validate_v11_transaction_command(
            base_config=expanded,
            command=command,
            selected_load_sha256="1" * 64,
            frozen_validator=SHARED_WORKER._validate_integrated_transaction_command,
        )


@pytest.mark.parametrize("mode", (None, "unknown-v11-mode"))
def test_v11_transaction_command_rejects_missing_or_unknown_mode(
    tmp_path: Path,
    mode: str | None,
) -> None:
    config = _targeted_fixture(tmp_path)
    expanded = CONTROLLER.expanded_runtime_config(config)
    command = CONTROLLER._transaction_command(
        config=config,
        effective_config=expanded,
        selection_sha256="1" * 64,
        sanity_result_sha256="2" * 64,
    )
    if mode is None:
        del expanded["execution_mode"]
    else:
        expanded["execution_mode"] = mode

    with pytest.raises(ValueError, match="execution mode is missing or unsupported"):
        INTEGRATED_WORKER._validate_v11_transaction_command(
            base_config=expanded,
            command=command,
            selected_load_sha256="1" * 64,
            frozen_validator=SHARED_WORKER._validate_integrated_transaction_command,
        )


@pytest.mark.parametrize("targeted", (False, True))
def test_v11_transaction_command_rejects_mode_base_schema_mismatch(
    tmp_path: Path,
    targeted: bool,
) -> None:
    config = _targeted_fixture(tmp_path) if targeted else _sealed_fixture(tmp_path)
    expanded = CONTROLLER.expanded_runtime_config(config)
    command = CONTROLLER._transaction_command(
        config=config,
        effective_config=expanded,
        selection_sha256="1" * 64,
        sanity_result_sha256="2" * 64,
    )
    expanded["schema_version"] = (
        "sloforge.branchfabric.experiment-004-v11-integrated-config/v1"
        if targeted
        else "sloforge.branchfabric.experiment-004-v11-targeted-source-identity-config/v1"
    )

    with pytest.raises(ValueError, match="base schema differs from its execution mode"):
        INTEGRATED_WORKER._validate_v11_transaction_command(
            base_config=expanded,
            command=command,
            selected_load_sha256="1" * 64,
            frozen_validator=SHARED_WORKER._validate_integrated_transaction_command,
        )


def test_integrated_transaction_command_rejects_target_safety_field_injection(
    tmp_path: Path,
) -> None:
    config = _sealed_fixture(tmp_path)
    expanded = CONTROLLER.expanded_runtime_config(config)
    command = CONTROLLER._transaction_command(
        config=config,
        effective_config=expanded,
        selection_sha256="1" * 64,
        sanity_result_sha256="2" * 64,
    )
    command.update(
        {
            "terminal_phase": "SOURCE_ALLOCATION_IDENTITY_GATE",
            "full_export_authorized": False,
            "source_release_authorized": False,
        }
    )

    with pytest.raises(ValueError, match="fields differ from its typed envelope"):
        INTEGRATED_WORKER._validate_v11_transaction_command(
            base_config=expanded,
            command=command,
            selected_load_sha256="1" * 64,
            frozen_validator=SHARED_WORKER._validate_integrated_transaction_command,
        )


@pytest.mark.parametrize("targeted", (False, True))
def test_v11_transaction_command_validation_does_not_mutate_inputs(
    tmp_path: Path,
    targeted: bool,
) -> None:
    config = _targeted_fixture(tmp_path) if targeted else _sealed_fixture(tmp_path)
    expanded = CONTROLLER.expanded_runtime_config(config)
    command = CONTROLLER._transaction_command(
        config=config,
        effective_config=expanded,
        selection_sha256="1" * 64,
        sanity_result_sha256="2" * 64,
    )
    expanded_before = json.loads(json.dumps(expanded))
    command_before = json.loads(json.dumps(command))

    INTEGRATED_WORKER._validate_v11_transaction_command(
        base_config=expanded,
        command=command,
        selected_load_sha256="1" * 64,
        frozen_validator=SHARED_WORKER._validate_integrated_transaction_command,
    )

    assert expanded == expanded_before
    assert command == command_before


@pytest.mark.parametrize(
    ("field", "tampered"),
    (
        ("full_export_authorized", 0),
        ("full_export_authorized", 1),
        ("source_release_authorized", 0),
        ("source_release_authorized", 1),
    ),
)
def test_targeted_transaction_command_rejects_boolean_type_confusion(
    tmp_path: Path,
    field: str,
    tampered: int,
) -> None:
    config = _targeted_fixture(tmp_path)
    expanded = CONTROLLER.expanded_runtime_config(config)
    command = CONTROLLER._transaction_command(
        config=config,
        effective_config=expanded,
        selection_sha256="1" * 64,
        sanity_result_sha256="2" * 64,
    )
    command[field] = tampered

    with pytest.raises(ValueError, match=f"safety field {field}"):
        INTEGRATED_WORKER._validate_v11_transaction_command(
            base_config=expanded,
            command=command,
            selected_load_sha256="1" * 64,
            frozen_validator=SHARED_WORKER._validate_integrated_transaction_command,
        )


@pytest.mark.parametrize("location", ("base", "effective"))
@pytest.mark.parametrize(
    "field",
    ("full_export_authorized", "source_release_authorized"),
)
def test_targeted_transaction_command_rejects_nested_boolean_type_confusion(
    tmp_path: Path,
    location: str,
    field: str,
) -> None:
    config = _targeted_fixture(tmp_path)
    expanded = CONTROLLER.expanded_runtime_config(config)
    command = CONTROLLER._transaction_command(
        config=config,
        effective_config=expanded,
        selection_sha256="1" * 64,
        sanity_result_sha256="2" * 64,
    )
    if location == "base":
        expanded[field] = 0
    else:
        command["effective_config"][field] = 0

    with pytest.raises(ValueError, match=f"safety field {field}"):
        INTEGRATED_WORKER._validate_v11_transaction_command(
            base_config=expanded,
            command=command,
            selected_load_sha256="1" * 64,
            frozen_validator=SHARED_WORKER._validate_integrated_transaction_command,
        )


@pytest.mark.parametrize("phase_fails", (False, True))
def test_v11_calibration_validator_binding_is_scoped_and_restored(
    tmp_path: Path,
    phase_fails: bool,
) -> None:
    config = _targeted_fixture(tmp_path)
    expanded = CONTROLLER.expanded_runtime_config(config)
    command = CONTROLLER._transaction_command(
        config=config,
        effective_config=expanded,
        selection_sha256="1" * 64,
        sanity_result_sha256="2" * 64,
    )
    original = SHARED_WORKER._validate_integrated_transaction_command
    observed: list[object] = []
    frozen_worker = SimpleNamespace(
        _V10_SANITY_GUARD_COUNT=2,
        _validate_integrated_transaction_command=original,
    )

    def run_phase(**_kwargs: object) -> tuple[dict[str, Any], object, dict[str, bool]]:
        bound = frozen_worker._validate_integrated_transaction_command
        assert bound is not original
        observed.append(bound)
        effective, _evidence = bound(
            base_config=expanded,
            command=command,
            selected_load_sha256="1" * 64,
        )
        if phase_fails:
            raise RuntimeError("calibration failed")
        return effective, object(), {"passed": True}

    frozen_worker._run_integrated_calibration_phase = run_phase

    def call() -> tuple[dict[str, Any], object, dict[str, bool]]:
        return INTEGRATED_WORKER._run_v11_integrated_calibration_phase(
            frozen_worker=frozen_worker,
            adapter=object(),
            config=expanded,
            inputs={},
            role="rollout",
            physical_gpu_uuid="GPU-test",
            barrier_root=tmp_path / "barriers",
            model_load_started_ns=1,
            model_ready_ns=2,
        )

    if phase_fails:
        with pytest.raises(RuntimeError, match="calibration failed"):
            call()
    else:
        effective, _engine, handoff = call()
        assert effective == expanded
        assert handoff == {"passed": True}
    assert len(observed) == 1
    assert frozen_worker._validate_integrated_transaction_command is original
    assert frozen_worker._V10_SANITY_GUARD_COUNT == 2


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


def test_targeted_deadlines_honor_300_second_cap_and_cleanup_reserve(
    tmp_path: Path,
) -> None:
    config = _targeted_fixture(tmp_path)
    second = 1_000_000_000
    entry = 1_000 * second
    operation, cleanup = CONTROLLER._partition_controller_deadlines(
        started_ns=entry,
        caller_deadline_ns=entry + 400 * second,
        cleanup_reserve_seconds=config.cleanup_timeout_seconds,
        absolute_wall_seconds=config.maximum_wall_seconds,
    )
    assert operation == entry + 290 * second
    assert cleanup == entry + 300 * second

    # The paid launcher retains ten seconds after the controller's caller
    # deadline for immutable publication/provider settlement.
    operation, cleanup = CONTROLLER._partition_controller_deadlines(
        started_ns=entry,
        caller_deadline_ns=entry + 290 * second,
        cleanup_reserve_seconds=config.cleanup_timeout_seconds,
        absolute_wall_seconds=config.maximum_wall_seconds,
    )
    assert operation == entry + 280 * second
    assert cleanup == entry + 290 * second
    assert entry + 300 * second - cleanup == 10 * second


def test_targeted_deadline_rejects_absent_operation_window(tmp_path: Path) -> None:
    config = _targeted_fixture(tmp_path)
    entry = 1_000_000_000
    with pytest.raises(TimeoutError, match="no bounded operation window"):
        CONTROLLER._partition_controller_deadlines(
            started_ns=entry,
            caller_deadline_ns=entry + config.cleanup_timeout_seconds * 1_000_000_000,
            cleanup_reserve_seconds=config.cleanup_timeout_seconds,
            absolute_wall_seconds=config.maximum_wall_seconds,
        )


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


def test_pre_worker_verifier_failure_emits_complete_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _sealed_fixture(tmp_path)
    config_path = tmp_path / "pre-worker-config.json"
    config_path.write_text(json.dumps(config.model_dump(mode="json"), sort_keys=True) + "\n")
    worker_path = tmp_path / "worker.py"
    worker_path.write_text("# not launched\n")
    model_snapshot = tmp_path / "model"
    model_snapshot.mkdir()

    def reject_seal(_config: Any, _root: Path) -> dict[str, Any]:
        raise FileNotFoundError("immutable pre-reservation ledger snapshot is absent")

    monkeypatch.setattr(CONTROLLER, "verify_sealed_evidence", reject_seal)
    monkeypatch.setattr(CONTROLLER, "_process_tree", lambda: ())
    monkeypatch.setattr(CONTROLLER, "_thread_snapshot", lambda: ())
    monkeypatch.setattr(CONTROLLER, "_ipc_snapshot", lambda: ())
    monkeypatch.setattr(CONTROLLER, "_wait_for_postflight_zero_compute", lambda **_kw: ())
    monkeypatch.setattr(
        CONTROLLER,
        "cuda_clean_import_audit",
        lambda stage: {"stage": stage, "cuda_clean": True, "forbidden_modules": []},
    )
    work_root = tmp_path / "pre-worker-result"
    result = CONTROLLER.run_integrated_v11_controller(
        config_path=config_path,
        work_root=work_root,
        worker_path=worker_path,
        model_snapshot=model_snapshot,
        repository_root=tmp_path,
        absolute_deadline_ns=time.monotonic_ns() + 60_000_000_000,
    )

    cleanup = result["in_function_cleanup"]
    assert result["status"] == "failed"
    assert result["cleanup_scope"] == "PRE_WORKER_PREFLIGHT"
    assert result["failure_stage"] == "SEALED_EVIDENCE"
    assert result["worker_pids"] == {}
    assert result["worker_process_groups"] == {}
    assert result["worker_session_ids"] == {}
    assert result["inventory_before"] == []
    assert result["engine_start_evidence"] == []
    assert result["controller_error"] == {
        "type": "FileNotFoundError",
        "message": "immutable pre-reservation ledger snapshot is absent",
    }
    assert cleanup["pass"] is True
    assert cleanup["owned_children"] == []
    assert cleanup["compute_processes_after"] == []
    assert [item["phase"] for item in cleanup["lifecycle"]] == list(
        CONTROLLER.PROCESS_LIFECYCLE_PHASES
    )
    assert json.loads((work_root / "in_function_cleanup.json").read_text()) == cleanup
    assert json.loads((work_root / "controller-result.json").read_text()) == json.loads(
        json.dumps(result)
    )


RETAINED_READINESS_REPLAYS = {
    "v10-v7": (
        "artifacts/branchfabric/gpu-validation/experiment-004/raw/modal/"
        "exp004-v10-naive-s41-v7/readiness/both-engines-ready.json",
        (17.91996744300315, 18.400739982078232),
    ),
    "attempt-f": (
        "artifacts/branchfabric/gpu-validation/experiment-004/v11-final/integrated/raw/"
        "exp004-v11-integrated-s41-f/readiness/both-engines-ready.json",
        (17.66411065844632, 17.606956571926613),
    ),
    "attempt-g": (
        "artifacts/branchfabric/gpu-validation/experiment-004/v11-final/integrated/raw/"
        "exp004-v11-integrated-s41-g/readiness/both-engines-ready.json",
        (17.78071997326811, 17.753324821590514),
    ),
    "attempt-j": (
        "artifacts/branchfabric/gpu-validation/experiment-004/v11-final/integrated/raw/"
        "exp004-v11-integrated-s41-j/readiness/both-engines-ready.json",
        (13.837854042338916, 12.97720271683092),
    ),
}


def _retained_readiness_replay(label: str) -> list[dict[str, Any]]:
    reference, _expected_throughputs = RETAINED_READINESS_REPLAYS[label]
    payload = json.loads((ROOT / reference).read_text())
    assert payload["schema_version"] == "sloforge.branchfabric.both-engines-ready/v1"
    assert payload["BOTH_ENGINES_READY"] is True
    evidence = payload["engine_evidence"]
    assert isinstance(evidence, list)
    return copy.deepcopy(evidence)


def _batch16_verification(row: dict[str, Any]) -> dict[str, Any]:
    batches = [
        batch
        for batch in row["verification_batches"]
        if type(batch.get("batch_size")) is int and batch["batch_size"] == 16
    ]
    assert len(batches) == 1
    return batches[0]


@pytest.mark.parametrize("label", ("v10-v7", "attempt-f", "attempt-g"))
def test_retained_engine_readiness_comparability_accepts_measured_replays(label: str) -> None:
    evidence = _retained_readiness_replay(label)

    serving, rollout = CONTROLLER._validate_retained_engine_readiness_comparability(
        evidence,
        minimum_batch16_request_throughput_rps=15.0,
    )

    expected = RETAINED_READINESS_REPLAYS[label][1]
    assert (serving["role"], serving["device"]) == ("serving", "gpu0")
    assert (rollout["role"], rollout["device"]) == ("rollout", "gpu1")
    assert _batch16_verification(serving)["request_throughput_rps"] == expected[0]
    assert _batch16_verification(rollout)["request_throughput_rps"] == expected[1]
    assert serving["physical_gpu_uuid"] != rollout["physical_gpu_uuid"]


def test_retained_engine_readiness_comparability_rejects_attempt_j_replay() -> None:
    evidence = _retained_readiness_replay("attempt-j")

    with pytest.raises(RuntimeError, match=r"batch-16.*throughput"):
        CONTROLLER._validate_retained_engine_readiness_comparability(
            evidence,
            minimum_batch16_request_throughput_rps=15.0,
        )


def test_retained_engine_readiness_assessment_persists_recomputed_attempt_j_failure() -> None:
    evidence = _retained_readiness_replay("attempt-j")

    assessment = CONTROLLER._assess_retained_engine_readiness_comparability(evidence)

    assert assessment["status"] == "BELOW_FLOOR"
    assert assessment["readiness_comparable"] is False
    assert assessment["minimum_batch16_request_throughput_rps"] == 15.0
    assert [item["device"] for item in assessment["devices"]] == ["gpu0", "gpu1"]
    assert [item["passed"] for item in assessment["devices"]] == [False, False]
    for item, payload in zip(assessment["devices"], evidence, strict=True):
        batch16 = _batch16_verification(payload)
        expected = 16.0 / ((batch16["completed_ns"] - batch16["started_ns"]) / 1_000_000_000.0)
        assert item["reported_request_throughput_rps"] == batch16["request_throughput_rps"]
        assert item["recomputed_request_throughput_rps"] == pytest.approx(
            expected, rel=1e-15, abs=1e-15
        )
        assert item["engine_nonce"] == payload["engine_nonce"]
        assert item["engine_object_identity"] == payload["engine_object_identity"]


@pytest.mark.parametrize("floor", [True, 15, float("nan"), float("inf"), 0.0])
def test_retained_engine_readiness_comparability_rejects_invalid_floor(floor: Any) -> None:
    evidence = _retained_readiness_replay("attempt-g")

    with pytest.raises(ValueError, match="exact finite positive float"):
        CONTROLLER._validate_retained_engine_readiness_comparability(
            evidence,
            minimum_batch16_request_throughput_rps=floor,
        )


@pytest.mark.parametrize("value", [True, 15, float("nan"), float("inf"), -float("inf")])
def test_retained_engine_readiness_comparability_rejects_non_exact_finite_float(
    value: Any,
) -> None:
    evidence = _retained_readiness_replay("attempt-g")
    _batch16_verification(evidence[0])["request_throughput_rps"] = value

    with pytest.raises((TypeError, ValueError, RuntimeError), match="throughput"):
        CONTROLLER._validate_retained_engine_readiness_comparability(evidence)


@pytest.mark.parametrize("mutation", ["missing", "duplicate"])
def test_retained_engine_readiness_comparability_requires_exactly_one_batch16(
    mutation: str,
) -> None:
    evidence = _retained_readiness_replay("attempt-g")
    batches = evidence[0]["verification_batches"]
    batch16 = _batch16_verification(evidence[0])
    if mutation == "missing":
        batches.remove(batch16)
    else:
        batches.append(copy.deepcopy(batch16))

    with pytest.raises(RuntimeError, match="batch-1/2/4/8/16"):
        CONTROLLER._validate_retained_engine_readiness_comparability(evidence)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("missing-batch-1", "batch-1/2/4/8/16"),
        ("duplicate-batch-2", "duplicates batch-2"),
        ("unknown-batch", "batch-1/2/4/8/16"),
        ("bool-batch", "batch-1/2/4/8/16"),
        ("nonmapping-batch", "batch evidence"),
    ],
)
def test_retained_engine_readiness_comparability_requires_exact_batch_matrix(
    mutation: str,
    message: str,
) -> None:
    evidence = _retained_readiness_replay("attempt-g")
    batches = evidence[0]["verification_batches"]
    if mutation == "missing-batch-1":
        batches.pop(0)
    elif mutation == "duplicate-batch-2":
        batches[0] = copy.deepcopy(batches[1])
    elif mutation == "unknown-batch":
        batches[0]["batch_size"] = 32
    elif mutation == "bool-batch":
        batches[0]["batch_size"] = True
    else:
        batches[0] = None

    with pytest.raises(RuntimeError, match=message):
        CONTROLLER._validate_retained_engine_readiness_comparability(evidence)


@pytest.mark.parametrize("value", [True, 1, float("nan"), float("inf"), -1.0])
def test_retained_engine_readiness_comparability_requires_exact_float_for_every_batch(
    value: Any,
) -> None:
    evidence = _retained_readiness_replay("attempt-g")
    evidence[0]["verification_batches"][0]["request_throughput_rps"] = value

    with pytest.raises(RuntimeError, match=r"batch-1 throughput"):
        CONTROLLER._validate_retained_engine_readiness_comparability(evidence)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("started_ns", True),
        ("first_token_ns", 0),
        ("completed_ns", 1.5),
    ],
)
def test_retained_engine_readiness_comparability_rejects_non_exact_batch16_timing(
    field: str,
    value: Any,
) -> None:
    evidence = _retained_readiness_replay("attempt-g")
    _batch16_verification(evidence[0])[field] = value

    with pytest.raises(RuntimeError, match=r"batch-16 timing/order"):
        CONTROLLER._validate_retained_engine_readiness_comparability(evidence)


@pytest.mark.parametrize("mutation", ("first-before-start", "complete-before-first"))
def test_retained_engine_readiness_comparability_rejects_batch16_timestamp_order(
    mutation: str,
) -> None:
    evidence = _retained_readiness_replay("attempt-g")
    batch16 = _batch16_verification(evidence[0])
    if mutation == "first-before-start":
        batch16["first_token_ns"] = batch16["started_ns"]
    else:
        batch16["completed_ns"] = batch16["first_token_ns"] - 1

    with pytest.raises(RuntimeError, match=r"batch-16 timing/order"):
        CONTROLLER._validate_retained_engine_readiness_comparability(evidence)


@pytest.mark.parametrize("field", ("request_throughput_rps", "completed_ns"))
def test_retained_engine_readiness_comparability_binds_batch16_throughput_to_timing(
    field: str,
) -> None:
    evidence = _retained_readiness_replay("attempt-g")
    batch16 = _batch16_verification(evidence[0])
    if field == "request_throughput_rps":
        batch16[field] += 0.01
    else:
        batch16[field] += 1

    with pytest.raises(RuntimeError, match=r"batch-16 throughput/timing commitment mismatch"):
        CONTROLLER._validate_retained_engine_readiness_comparability(evidence)


@pytest.mark.parametrize(
    "mutation",
    ("batch-order", "request-count", "duplicate-request", "output-count", "output-length"),
)
def test_retained_engine_readiness_comparability_rejects_batch_accounting_tamper(
    mutation: str,
) -> None:
    evidence = _retained_readiness_replay("attempt-g")
    batches = evidence[0]["verification_batches"]
    if mutation == "batch-order":
        batches[0], batches[1] = batches[1], batches[0]
        message = "timing/order"
    else:
        batch16 = _batch16_verification(evidence[0])
        if mutation == "request-count":
            batch16["request_ids"].pop()
        elif mutation == "duplicate-request":
            batch16["request_ids"][1] = batch16["request_ids"][0]
        elif mutation == "output-count":
            batch16["output_lengths"].pop()
        else:
            batch16["output_lengths"][0] = 63
        message = "request/output accounting"

    with pytest.raises(RuntimeError, match=message):
        CONTROLLER._validate_retained_engine_readiness_comparability(evidence)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("missing-role", "role"),
        ("duplicate-role", "role"),
        ("duplicate-device", "device"),
        ("duplicate-uuid", "UUID|uuid"),
        ("cross-role-device", "role|device"),
    ],
)
def test_retained_engine_readiness_comparability_rejects_identity_tamper(
    mutation: str,
    message: str,
) -> None:
    evidence = _retained_readiness_replay("attempt-g")
    if mutation == "missing-role":
        evidence[0].pop("role")
    elif mutation == "duplicate-role":
        evidence[1]["role"] = evidence[0]["role"]
    elif mutation == "duplicate-device":
        evidence[1]["device"] = evidence[0]["device"]
    elif mutation == "duplicate-uuid":
        evidence[1]["physical_gpu_uuid"] = evidence[0]["physical_gpu_uuid"]
    else:
        evidence[0]["role"], evidence[1]["role"] = (
            evidence[1]["role"],
            evidence[0]["role"],
        )

    with pytest.raises(RuntimeError, match=message):
        CONTROLLER._validate_retained_engine_readiness_comparability(evidence)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("engine_reloaded", True),
        ("engine_nonce", ""),
        ("engine_object_identity", True),
        ("adapter_object_identity", False),
        ("engine_started_ns", 0),
        ("pid", True),
    ],
)
def test_retained_engine_readiness_comparability_rejects_continuity_tamper(
    field: str,
    value: Any,
) -> None:
    evidence = _retained_readiness_replay("attempt-g")
    evidence[0][field] = value

    with pytest.raises(RuntimeError, match="continuity"):
        CONTROLLER._validate_retained_engine_readiness_comparability(evidence)


@pytest.mark.parametrize(
    "field",
    ("engine_nonce", "pid", "engine_object_identity", "adapter_object_identity"),
)
def test_retained_engine_readiness_comparability_requires_disjoint_engine_identity(
    field: str,
) -> None:
    evidence = _retained_readiness_replay("attempt-g")
    evidence[1][field] = evidence[0][field]

    with pytest.raises(RuntimeError, match="continuity"):
        CONTROLLER._validate_retained_engine_readiness_comparability(evidence)


def test_retained_engine_readiness_comparability_requires_exact_pair() -> None:
    evidence = _retained_readiness_replay("attempt-g")
    with pytest.raises(RuntimeError, match="exactly two"):
        CONTROLLER._validate_retained_engine_readiness_comparability(evidence[:1])
    with pytest.raises(RuntimeError, match="exactly two"):
        CONTROLLER._validate_retained_engine_readiness_comparability(
            [*evidence, copy.deepcopy(evidence[0])]
        )


def test_retained_readiness_comparator_does_not_replace_frozen_sanity_assessor() -> None:
    frozen = EXPERIMENT_ROOT / "gpu_reclamation_controller.py"
    assert hashlib.sha256(frozen.read_bytes()).hexdigest() == (
        "c91ccd8b3aa4b4c5a9034a5137ccc33257e3471528a96dcc2ca7920c8b7eefc9"
    )
    source = CONTROLLER_PATH.read_text()
    assert source.count("_assess_sanity_guard(result, expected_rate_rps=rate)") == 1
    assert "for probe_index, rate in enumerate(SANITY_RATES_RPS[1:], start=2)" in source


def test_integrated_readiness_comparability_precedes_all_serving_probes() -> None:
    source = CONTROLLER_PATH.read_text()
    tree = ast.parse(source)
    controller = next(
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "run_integrated_v11_controller"
    )
    controller_source = ast.get_source_segment(source, controller)
    assert controller_source is not None
    assessment = controller_source.index(
        "readiness_comparability = _assess_retained_engine_readiness_comparability("
    )
    persisted = controller_source.index(
        "_write_new(readiness_comparability_path, readiness_comparability)"
    )
    comparator = controller_source.index(
        "_validate_retained_engine_readiness_comparability(ready_payloads)"
    )
    first_reproduction = controller_source.index("_run_targeted_serving_sanity_probe(")
    ordinary_sanity = controller_source.index(
        "for probe_index, rate in enumerate(SANITY_RATES_RPS[1:], start=2)"
    )
    transaction = controller_source.index("_transaction_command(")
    assert assessment < persisted < comparator < first_reproduction < ordinary_sanity < transaction


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
