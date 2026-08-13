#!/usr/bin/env python3
"""Budget-first sole Modal coordinator for BranchFabric Experiment 004."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import re
import secrets
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
_APP = _ROOT / "experiments/branchfabric/modal_gpu_reclamation.py"
_EXPERIMENT_ROOT = _ROOT / "artifacts/branchfabric/gpu-validation/experiment-004"
_LEDGER = _EXPERIMENT_ROOT / "gpu-hours.json"
_LOCK = _EXPERIMENT_ROOT / "gpu-hours.lock"
_FUNCTION_WALL_SECONDS = 469.0
_LEGACY_RESERVATION_WALL_SECONDS = 340.0
_INTEGRATED_POST_CONTROLLER_RESERVE_SECONDS = 10.0
_FUNCTION_STARTUP_TIMEOUT_SECONDS = 180.0
_GPU_COUNT = 2
_GPU_USD_PER_HOUR = 2.4984
_CPU_USD_PER_CORE_SECOND = 0.0000131
_MEMORY_USD_PER_GIB_SECOND = 0.00000222
_AUTHORIZATION = _EXPERIMENT_ROOT / "v10-authorization.json"
_PHASE_BUDGET = _EXPERIMENT_ROOT / "v10-phase-budget.json"
_CONFIG_FIELDS = frozenset(
    {
        "attempt_id",
        "baseline_seconds",
        "cleanup_timeout_seconds",
        "fanout",
        "gpu0_control_request_rate_per_second",
        "gpu0_restore_request_rate_per_second",
        "gpu0_spike_request_rate_per_second",
        "gpu1_spike_request_rate_per_second",
        "gpu_memory_utilization",
        "initialization_timeout_seconds",
        "maximum_wall_seconds",
        "mode",
        "model",
        "model_revision",
        "prefix_length",
        "runtime",
        "runtime_version",
        "schema_version",
        "seed",
        "serving_output_tokens",
        "serving_prompt_tokens",
        "serving_slo_maximum_inter_token_latency_seconds",
        "serving_slo_maximum_ttft_seconds",
        "serving_slo_stability_window_seconds",
        "serving_spike_seconds",
        "suffix_length",
        "temporary_serving_seconds",
        "tokenizer_revision",
        "tracing_level",
    }
)
_V10_CONFIG_FIELDS = _CONFIG_FIELDS | {
    "serving_methodology",
    "serving_spike_request_rate_per_second",
    "gpu0_overload_probe_seconds",
    "serving_recovery_evaluation_seconds",
    "serving_recovery_queue_threshold",
    "serving_maximum_pending_requests",
    "serving_restore_handoff_lead_requests",
    "calibration_attempt_id",
    "lambda_1_rps",
    "lambda_2_rps",
    "lambda_spike_rps",
    "calibration_selected_load_artifact",
    "calibration_selected_load_sha256",
    "agent9_approval_artifact",
    "agent9_approval_sha256",
    "two_gpu_headroom_adr_artifact",
    "two_gpu_headroom_adr_sha256",
}
_INTEGRATED_CONFIG_FIELDS = _CONFIG_FIELDS | {
    "execution_mode",
    "serving_methodology",
    "serving_spike_request_rate_per_second",
    "gpu0_overload_probe_seconds",
    "serving_recovery_evaluation_seconds",
    "serving_recovery_queue_threshold",
    "serving_maximum_pending_requests",
    "serving_restore_handoff_lead_requests",
    "serving_overload_queue_trigger",
    "serving_overload_queue_abort",
    "disable_log_stats",
    "warmup_seconds",
    "sanity_guard_measurement_seconds",
    "probe_timeout_seconds",
    "tail_drain_seconds",
    "producer_queue_capacity",
    "controller_timeout_seconds",
    "worker_shutdown_timeout_seconds",
    "calibration_attempt_id",
    "lambda_1_rps",
    "lambda_2_rps",
    "lambda_spike_rps",
    "calibration_selected_load_artifact",
    "calibration_selected_load_sha256",
    "authorization_artifact",
    "authorization_artifact_hash",
    "phase_budget_artifact",
    "phase_budget_sha256",
}
_SELECTED_LOAD_REFERENCE = (
    "artifacts/branchfabric/gpu-validation/experiment-004/calibration/selected-load.json"
)
_AGENT9_APPROVAL_REFERENCE = (
    "artifacts/branchfabric/gpu-validation/experiment-004/calibration/reviews/"
    "agent9-selected-load-approval.json"
)
_HEADROOM_ADR_REFERENCE = (
    "artifacts/branchfabric/gpu-validation/experiment-004/calibration/reviews/"
    "agent13-two-gpu-headroom-adr.json"
)


def _canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n"
    ).encode("utf-8")


def _atomic_write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("xb") as handle:
        handle.write(_canonical_bytes(value))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _write_new(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(_canonical_bytes(value))
        handle.flush()
        os.fsync(handle.fileno())


def _load_pre_gpu_evidence(config: dict[str, Any]) -> tuple[Any, Any]:
    """Verify both immutable pre-GPU artifacts and all historical source hashes."""

    from sloforge.helix.characterization.gpu_reclamation_v10_authorization import (
        verify_authorization_file,
    )
    from sloforge.helix.characterization.gpu_reclamation_v10_budget import (
        load_v10_phase_budget,
        phase_budget_sha256,
    )

    if config.get("execution_mode") != "integrated-calibration-v10":
        raise ValueError("pre-GPU evidence is defined only for integrated v10")
    expected_authorization_reference = (
        "artifacts/branchfabric/gpu-validation/experiment-004/v10-authorization.json"
    )
    expected_budget_reference = (
        "artifacts/branchfabric/gpu-validation/experiment-004/v10-phase-budget.json"
    )
    if (
        config.get("authorization_artifact") != expected_authorization_reference
        or config.get("phase_budget_artifact") != expected_budget_reference
    ):
        raise ValueError("v10 pre-GPU artifact references differ from the exact contract")
    authorization = verify_authorization_file(
        _AUTHORIZATION,
        repository_root=_ROOT,
        expected_artifact_hash=str(config.get("authorization_artifact_hash")),
    )
    _verify_authorized_code_commit(str(authorization["code_commit"]))
    budget = load_v10_phase_budget(_PHASE_BUDGET, repository_root=_ROOT)
    if phase_budget_sha256(_PHASE_BUDGET) != config.get("phase_budget_sha256"):
        raise ValueError("v10 phase-budget hash differs from the immutable config")
    if authorization["status"] != "APPROVED" or budget.status != "EVIDENCE_DERIVED":
        raise ValueError("v10 pre-GPU authorization or phase budget is not approved")
    return authorization, budget


def _verify_authorized_code_commit(expected_commit: str) -> None:
    """Bind the paid path to the reviewed commit and reject source drift."""

    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=10.0,
    )
    if head.returncode != 0 or head.stdout.strip() != expected_commit:
        raise ValueError("v10 authorization code_commit does not match repository HEAD")
    paid_paths = (
        "adapters/continuum/vllm",
        "experiments/branchfabric/gpu_capacity_calibration_controller.py",
        "experiments/branchfabric/gpu_capacity_calibration_worker.py",
        "experiments/branchfabric/gpu_reclamation_controller.py",
        "experiments/branchfabric/gpu_reclamation_v10_serving.py",
        "experiments/branchfabric/gpu_reclamation_worker.py",
        "experiments/branchfabric/modal_gpu_reclamation.py",
        "python/sloforge/continuum/adapters",
        "python/sloforge/helix/characterization/gpu_capacity_calibration.py",
        "python/sloforge/helix/characterization/gpu_reclamation_methodology.py",
        "python/sloforge/helix/characterization/gpu_reclamation_v10_authorization.py",
        "python/sloforge/helix/characterization/gpu_reclamation_v10_budget.py",
        "python/sloforge/helix/characterization/gpu_reclamation_v10_postprocess.py",
        "python/sloforge/helix/characterization/gpu_reclamation_v10_publish.py",
        "python/sloforge/helix/characterization/gpu_reclamation_v10_validity.py",
        "tools/branchfabric-experiment-004-authorize-v10.py",
        "tools/branchfabric-experiment-004-finalize-v10.py",
        "tools/branchfabric-experiment-004-launch.py",
        "tools/branchfabric-experiment-004-live-driver-gate.py",
        "tools/branchfabric-experiment-004-local-loadgen-gate.py",
        "tools/branchfabric-experiment-004-publish-v10.py",
    )
    status = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all", "--", *paid_paths],
        cwd=_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=10.0,
    )
    if status.returncode != 0 or status.stdout.strip():
        raise ValueError("v10 paid-path source differs from the authorized code commit")


def _reservation_wall_seconds(config: dict[str, Any]) -> float:
    if config.get("execution_mode") != "integrated-calibration-v10":
        return _LEGACY_RESERVATION_WALL_SECONDS
    _authorization, budget = _load_pre_gpu_evidence(config)
    return float(budget.predicted_wall_clock_seconds)


def _controller_wall_seconds(config: dict[str, Any]) -> float:
    return _reservation_wall_seconds(config) - _INTEGRATED_POST_CONTROLLER_RESERVE_SECONDS


def _function_wall_seconds(config: dict[str, Any]) -> float:
    return (
        _reservation_wall_seconds(config)
        if config.get("execution_mode") == "integrated-calibration-v10"
        else _FUNCTION_WALL_SECONDS
    )


def _maximum_cost_usd(config: dict[str, Any]) -> float:
    wall_seconds = _reservation_wall_seconds(config)
    return wall_seconds * _GPU_COUNT / 3600.0 * _GPU_USD_PER_HOUR + wall_seconds * (
        16.0 * _CPU_USD_PER_CORE_SECOND + 64.0 * _MEMORY_USD_PER_GIB_SECOND
    )


def _require_budget(config: dict[str, Any]) -> float:
    raw = os.environ.get("SLOFORGE_GPU_BUDGET_USD")
    if raw is None:
        raise RuntimeError("SLOFORGE_GPU_BUDGET_USD is required before invoking Modal")
    try:
        budget = float(raw)
    except ValueError as error:
        raise RuntimeError("SLOFORGE_GPU_BUDGET_USD must be finite and positive") from error
    if not math.isfinite(budget) or budget <= 0:
        raise RuntimeError("SLOFORGE_GPU_BUDGET_USD must be finite and positive")
    maximum_cost = _maximum_cost_usd(config)
    if budget < maximum_cost:
        raise RuntimeError(f"bounded two-A100 pilot may cost ${maximum_cost:.4f}, exceeding budget")
    return budget


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve_repo_artifact(reference: str) -> Path:
    relative = Path(reference)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("v10 provenance artifact reference is unsafe")
    path = (_ROOT / relative).resolve(strict=True)
    if path.is_symlink() or not path.is_file():
        raise ValueError("v10 provenance artifact must be a regular non-symlink file")
    return path


def _validate_artifact_ref(value: Any) -> tuple[Path, dict[str, Any]]:
    if not isinstance(value, dict) or set(value) != {
        "artifact_reference",
        "artifact_sha256",
        "sample_selector",
    }:
        raise ValueError("capacity evidence reference differs from its exact contract")
    if value["sample_selector"] != "$":
        raise ValueError("capacity evidence must reference its complete raw artifact")
    path = Path(str(value["artifact_reference"])).resolve(strict=True)
    experiment_root = _EXPERIMENT_ROOT.resolve(strict=True)
    if not path.is_relative_to(experiment_root) or path.is_symlink() or not path.is_file():
        raise ValueError("capacity evidence escapes the immutable Experiment 004 tree")
    if _sha256(path) != value["artifact_sha256"]:
        raise ValueError("capacity evidence hash differs from the referenced raw artifact")
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError("capacity evidence artifact must contain one JSON object")
    return path, payload


def _validate_probe_evidence(
    *,
    attempt_id: str,
    topology: str,
    probe_id: str,
    expected_rate: float,
    expected_verdict: str,
) -> None:
    from sloforge.helix.characterization.gpu_capacity_calibration import (
        CapacityProbePlan,
        CapacityProbeRaw,
        CapacityProbeResult,
        evaluate_probe,
    )

    topology_directory = "one-gpu" if topology == "gpu0-only" else "two-gpu"
    index_path = _EXPERIMENT_ROOT / f"calibration/{topology_directory}/{attempt_id}.index.json"
    if not index_path.is_file() or index_path.is_symlink():
        raise FileNotFoundError("v10 capacity probe index is absent before launch")
    index = json.loads(index_path.read_text())
    if (
        not isinstance(index, dict)
        or set(index) != {"schema_version", "attempt_id", "topology", "entries"}
        or index.get("schema_version") != "sloforge.branchfabric.capacity-artifact-index/v1"
        or index.get("attempt_id") != attempt_id
        or index.get("topology") != topology_directory
        or not isinstance(index.get("entries"), list)
    ):
        raise ValueError("v10 capacity probe index identity is invalid")
    matches = tuple(
        item
        for item in index["entries"]
        if isinstance(item, dict) and item.get("probe_id") == probe_id
    )
    if len(matches) != 1 or set(matches[0]) != {"probe_id", "plan", "raw", "result"}:
        raise ValueError("v10 selected probe is absent, duplicated, or malformed")
    plan_path, _plan_payload = _validate_artifact_ref(matches[0]["plan"])
    raw_path, _raw_payload = _validate_artifact_ref(matches[0]["raw"])
    result_path, _result_payload = _validate_artifact_ref(matches[0]["result"])
    expected_probe_root = (
        _EXPERIMENT_ROOT
        / f"calibration/raw/modal/{attempt_id}/calibration/{topology_directory}/{probe_id}"
    ).resolve(strict=True)
    if (
        plan_path != expected_probe_root / "plan.json"
        or raw_path != expected_probe_root / "raw.json"
        or result_path != expected_probe_root / "result.json"
    ):
        raise ValueError("v10 capacity index points outside its exact attempt/probe directory")
    plan = CapacityProbePlan.model_validate_json(plan_path.read_text(), strict=True)
    raw = CapacityProbeRaw.model_validate_json(raw_path.read_text(), strict=True)
    result = CapacityProbeResult.model_validate_json(result_path.read_text(), strict=True)
    if raw.plan != plan or evaluate_probe(raw) != result:
        raise ValueError("v10 selected probe result differs from raw recomputation")
    if (
        result.probe_id != probe_id
        or result.topology.value != topology
        or result.verdict.value != expected_verdict
        or not math.isclose(result.configured_rate_rps, expected_rate, rel_tol=0.0, abs_tol=1e-12)
    ):
        raise ValueError("v10 selected probe does not support its declared operating point")


def _validate_headroom_adr(
    *,
    payload: dict[str, Any],
    selection: Any,
    selection_path: Path,
) -> None:
    """Validate the explicit strong-stability exception for >80% two-GPU load."""

    from sloforge.helix.characterization.gpu_capacity_calibration import CapacityProbeResult

    reference = payload.get("two_gpu_headroom_adr_artifact")
    digest = payload.get("two_gpu_headroom_adr_sha256")
    if reference != _HEADROOM_ADR_REFERENCE or not isinstance(digest, str):
        raise ValueError("v10 >80% two-GPU load requires the pinned headroom ADR")
    adr_path = _resolve_repo_artifact(reference)
    if _sha256(adr_path) != digest:
        raise ValueError("v10 headroom ADR SHA256 differs from its immutable artifact")
    adr = json.loads(adr_path.read_text())
    expected_fields = {
        "schema_version",
        "reviewer",
        "verdict",
        "calibration_attempt_id",
        "selected_load_artifact",
        "selected_load_sha256",
        "spike_over_lambda_2",
        "two_gpu_sustainable_probe_id",
        "two_gpu_result_provenance",
        "strong_stability",
        "justification",
    }
    if not isinstance(adr, dict) or set(adr) != expected_fields:
        raise ValueError("v10 headroom ADR differs from its exact contract")
    justification = adr.get("justification")
    if not isinstance(justification, str) or not 32 <= len(justification.strip()) <= 1_000:
        raise ValueError("v10 headroom ADR lacks a bounded written justification")
    result_path, result_payload = _validate_artifact_ref(adr["two_gpu_result_provenance"])
    expected_result_path = (
        _EXPERIMENT_ROOT
        / (
            f"calibration/raw/modal/{payload['calibration_attempt_id']}/calibration/"
            f"two-gpu/{selection.two_gpu_sustainable_probe_id}/result.json"
        )
    ).resolve(strict=True)
    if result_path != expected_result_path:
        raise ValueError("v10 headroom ADR provenance differs from its selected two-GPU probe")
    result = CapacityProbeResult.model_validate_json(
        json.dumps(result_payload, sort_keys=True, separators=(",", ":")), strict=True
    )
    strong = adr.get("strong_stability")
    expected_strong_fields = {
        "measurement_seconds",
        "p95_ttft_seconds",
        "completed_rate_rps",
        "observed_offered_rate_rps",
        "queue_slope_requests_per_second",
        "maximum_queue_depth",
        "all_measurement_requests_completed",
    }
    if not isinstance(strong, dict) or set(strong) != expected_strong_fields:
        raise ValueError("v10 headroom ADR strong-stability evidence is malformed")
    expected_strong = {
        "measurement_seconds": 10.0,
        "p95_ttft_seconds": result.p95_ttft_seconds,
        "completed_rate_rps": result.completed_rate_rps,
        "observed_offered_rate_rps": result.observed_offered_rate_rps,
        "queue_slope_requests_per_second": result.queue_slope_requests_per_second,
        "maximum_queue_depth": result.maximum_queue_depth,
        "all_measurement_requests_completed": result.all_measurement_requests_completed,
    }
    selected_digest = _sha256(selection_path)
    ratio = selection.lambda_spike_rps / selection.lambda_2_rps
    identity_pass = (
        adr.get("schema_version") == "sloforge.branchfabric.capacity-headroom-adr/v1"
        and adr.get("reviewer") == "agent13"
        and adr.get("verdict") == "approved"
        and adr.get("calibration_attempt_id") == payload["calibration_attempt_id"]
        and adr.get("selected_load_artifact") == _SELECTED_LOAD_REFERENCE
        and adr.get("selected_load_sha256") == selected_digest
        and adr.get("two_gpu_sustainable_probe_id") == selection.two_gpu_sustainable_probe_id
        and math.isclose(
            float(adr.get("spike_over_lambda_2", math.nan)),
            ratio,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        and strong == expected_strong
    )
    stability_pass = (
        result.verdict.value == "sustainable"
        and result.p95_ttft_seconds is not None
        and result.p95_ttft_seconds <= 1.0
        and result.completed_rate_rps + 1e-12 >= result.observed_offered_rate_rps * 0.95
        and not result.queue_persistent_positive_drift
        and result.queue_slope_requests_per_second <= 0.10
        and result.all_measurement_requests_completed
    )
    if not identity_pass or not stability_pass:
        raise ValueError("v10 headroom ADR does not prove strong two-GPU stability")


def _validate_v10_calibration_binding(payload: dict[str, Any]) -> None:
    from sloforge.helix.characterization.gpu_capacity_calibration import SpikeSelection

    if payload["calibration_selected_load_artifact"] != _SELECTED_LOAD_REFERENCE:
        raise ValueError("v10 selected-load artifact path differs from the pinned reference")
    if payload["agent9_approval_artifact"] != _AGENT9_APPROVAL_REFERENCE:
        raise ValueError("v10 Agent 9 approval path differs from the pinned reference")
    selection_path = _resolve_repo_artifact(str(payload["calibration_selected_load_artifact"]))
    approval_path = _resolve_repo_artifact(str(payload["agent9_approval_artifact"]))
    if _sha256(selection_path) != payload["calibration_selected_load_sha256"]:
        raise ValueError("v10 selected-load SHA256 does not match its immutable artifact")
    if _sha256(approval_path) != payload["agent9_approval_sha256"]:
        raise ValueError("v10 Agent 9 approval SHA256 does not match its immutable artifact")

    selection_document = json.loads(selection_path.read_text())
    if not isinstance(selection_document, dict) or set(selection_document) != {
        "schema_version",
        "lambda_1_rps",
        "lambda_2_rps",
        "lambda_spike_rps",
        "spike_over_lambda_1",
        "spike_over_lambda_2",
        "one_gpu_unsustainable_probe_id",
        "one_gpu_sustainable_probe_id",
        "two_gpu_sustainable_probe_id",
        "reviewer_gate_required",
        "raw_provenance",
    }:
        raise ValueError("v10 selected-load artifact differs from its exact contract")
    raw_provenance = selection_document.pop("raw_provenance")
    selection = SpikeSelection.model_validate_json(
        json.dumps(selection_document, sort_keys=True, separators=(",", ":")), strict=True
    )
    raw_selection_path, raw_selection = _validate_artifact_ref(raw_provenance)
    expected_raw_selection_path = (
        _EXPERIMENT_ROOT
        / f"calibration/raw/modal/{payload['calibration_attempt_id']}/calibration/selected-load.json"
    ).resolve(strict=True)
    if raw_selection_path != expected_raw_selection_path:
        raise ValueError("v10 selected-load provenance differs from its calibration attempt")
    if raw_selection != selection.model_dump(mode="json"):
        raise ValueError("local selected-load index differs from downloaded raw selection")

    calibration_attempt_id = str(payload["calibration_attempt_id"])
    selected_numbers = {
        "lambda_1_rps": selection.lambda_1_rps,
        "lambda_2_rps": selection.lambda_2_rps,
        "lambda_spike_rps": selection.lambda_spike_rps,
    }
    if any(
        not math.isclose(float(payload[field]), value, rel_tol=0.0, abs_tol=1e-12)
        for field, value in selected_numbers.items()
    ) or not math.isclose(
        float(payload["serving_spike_request_rate_per_second"]),
        selection.lambda_spike_rps,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("v10 configured lambda values differ from selected-load evidence")
    if float(payload["gpu0_control_request_rate_per_second"]) > selection.lambda_1_rps:
        raise ValueError("v10 control load exceeds measured one-GPU stable capacity")
    if float(payload["gpu0_restore_request_rate_per_second"]) >= selection.lambda_1_rps:
        raise ValueError("v10 restore load must remain below one-GPU stable capacity")
    if not (
        1.15 <= selection.spike_over_lambda_1 <= 1.30
        and 0.0 < selection.spike_over_lambda_2 <= 0.90
        and math.isclose(
            selection.spike_over_lambda_1,
            selection.lambda_spike_rps / selection.lambda_1_rps,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        and math.isclose(
            selection.spike_over_lambda_2,
            selection.lambda_spike_rps / selection.lambda_2_rps,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ):
        raise ValueError("v10 selected-load capacity margins are invalid")
    if selection.spike_over_lambda_2 <= 0.80:
        if (
            payload.get("two_gpu_headroom_adr_artifact") is not None
            or payload.get("two_gpu_headroom_adr_sha256") is not None
        ):
            raise ValueError("preferred v10 headroom must not carry an exception ADR")
    else:
        _validate_headroom_adr(
            payload=payload,
            selection=selection,
            selection_path=selection_path,
        )

    _validate_probe_evidence(
        attempt_id=calibration_attempt_id,
        topology="gpu0-only",
        probe_id=selection.one_gpu_sustainable_probe_id,
        expected_rate=selection.lambda_1_rps,
        expected_verdict="sustainable",
    )
    _validate_probe_evidence(
        attempt_id=calibration_attempt_id,
        topology="gpu0-only",
        probe_id=selection.one_gpu_unsustainable_probe_id,
        expected_rate=selection.lambda_spike_rps,
        expected_verdict="unsustainable",
    )
    _validate_probe_evidence(
        attempt_id=calibration_attempt_id,
        topology="two-gpu-round-robin",
        probe_id=selection.two_gpu_sustainable_probe_id,
        expected_rate=selection.lambda_2_rps,
        expected_verdict="sustainable",
    )

    approval = json.loads(approval_path.read_text())
    expected_approval = {
        "schema_version": "sloforge.branchfabric.capacity-agent9-approval/v1",
        "reviewer": "agent9",
        "verdict": "approved",
        "calibration_attempt_id": calibration_attempt_id,
        "selected_load_artifact": _SELECTED_LOAD_REFERENCE,
        "selected_load_sha256": payload["calibration_selected_load_sha256"],
        **selected_numbers,
        "one_gpu_unsustainable_probe_id": selection.one_gpu_unsustainable_probe_id,
        "one_gpu_sustainable_probe_id": selection.one_gpu_sustainable_probe_id,
        "two_gpu_sustainable_probe_id": selection.two_gpu_sustainable_probe_id,
    }
    if approval != expected_approval:
        raise ValueError("Agent 9 approval does not exactly approve the selected raw evidence")


def _validate_config(path: Path) -> dict[str, Any]:
    if path.stat().st_size > 64 * 1024:
        raise ValueError("Experiment 004 config exceeds 64 KiB")
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError("Experiment 004 config must be a JSON object")
    methodology = payload.get("serving_methodology", "v9-independent-shards")
    integrated = payload.get("execution_mode") == "integrated-calibration-v10"
    expected_fields = (
        _INTEGRATED_CONFIG_FIELDS
        if integrated
        else (_V10_CONFIG_FIELDS if methodology == "v10-global-capacity" else _CONFIG_FIELDS)
    )
    if set(payload) != expected_fields:
        missing = sorted(expected_fields - set(payload))
        extra = sorted(set(payload) - expected_fields)
        raise ValueError(
            f"Experiment 004 config fields differ from the exact contract: missing={missing}, extra={extra}"
        )
    required = {
        "schema_version": "sloforge.branchfabric.experiment-004-modal-config/v1",
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "model_revision": "a09a35458c702b33eeacc393d103063234e8bc28",
        "tokenizer_revision": "a09a35458c702b33eeacc393d103063234e8bc28",
        "runtime": "vllm",
        "runtime_version": "0.23.0",
        "mode": "PRESERVE_NAIVE",
        "prefix_length": 16_384,
        "fanout": 8,
        "suffix_length": 256,
        "tracing_level": "minimal",
    }
    if methodology == "v9-independent-shards" and not integrated:
        required.update(
            {
                "gpu0_control_request_rate_per_second": 8.0,
                "gpu0_spike_request_rate_per_second": 64.0,
                "gpu1_spike_request_rate_per_second": 32.0,
                "gpu0_restore_request_rate_per_second": 8.0,
            }
        )
    if any(payload.get(field) != expected for field, expected in required.items()):
        raise ValueError("Experiment 004 config differs from the exact pilot contract")
    attempt_id = payload.get("attempt_id")
    if (
        not isinstance(attempt_id, str)
        or re.fullmatch(r"[a-z0-9][a-z0-9-]{7,95}", attempt_id) is None
    ):
        raise ValueError("Experiment 004 attempt_id is invalid")
    seed = payload.get("seed")
    if not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed < 1 << 63:
        raise ValueError("Experiment 004 seed is invalid")

    def bounded_number(
        field: str,
        *,
        lower: float,
        upper: float,
        integer: bool = False,
    ) -> float:
        value = payload[field]
        valid_type = isinstance(value, int) if integer else isinstance(value, (int, float))
        if (
            not valid_type
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            or not lower <= float(value) <= upper
        ):
            raise ValueError(f"Experiment 004 {field} is outside its bounded contract")
        return float(value)

    bounded_number("gpu_memory_utilization", lower=0.0000001, upper=0.85)
    bounded_number("baseline_seconds", lower=2.0, upper=15.0)
    bounded_number("serving_spike_seconds", lower=5.0, upper=30.0)
    bounded_number("temporary_serving_seconds", lower=1.0, upper=10.0)
    bounded_number("serving_prompt_tokens", lower=16, upper=2_048, integer=True)
    bounded_number("serving_output_tokens", lower=1, upper=128, integer=True)
    control_rate = bounded_number("gpu0_control_request_rate_per_second", lower=1.0, upper=64.0)
    gpu0_spike_rate = bounded_number("gpu0_spike_request_rate_per_second", lower=4.0, upper=128.0)
    gpu1_spike_rate = bounded_number("gpu1_spike_request_rate_per_second", lower=1.0, upper=128.0)
    bounded_number("gpu0_restore_request_rate_per_second", lower=1.0, upper=64.0)
    if (
        methodology == "v9-independent-shards"
        and not integrated
        and (gpu0_spike_rate + gpu1_spike_rate < 4.0 * control_rate)
    ):
        raise ValueError("Experiment 004 serving spike offers less than 4x control load")
    bounded_number("serving_slo_maximum_ttft_seconds", lower=0.0000001, upper=10.0)
    bounded_number(
        "serving_slo_maximum_inter_token_latency_seconds",
        lower=0.0000001,
        upper=5.0,
    )
    stability = bounded_number("serving_slo_stability_window_seconds", lower=0.5, upper=5.0)
    if (
        methodology == "v9-independent-shards"
        and not integrated
        and stability > float(payload["temporary_serving_seconds"])
    ):
        raise ValueError("Experiment 004 serving SLO stability window exceeds its measurement")
    if integrated:
        integrated_required = {
            "execution_mode": "integrated-calibration-v10",
            "serving_methodology": "v10-global-capacity",
            "serving_spike_request_rate_per_second": 15.0,
            "disable_log_stats": False,
            "serving_output_tokens": 64,
            "serving_prompt_tokens": 256,
            "calibration_attempt_id": "exp004-v10-integrated-s41-v3",
            "lambda_1_rps": 12.0,
            "lambda_spike_rps": 15.0,
            "lambda_2_rps": 20.0,
            "warmup_seconds": 1.0,
            "sanity_guard_measurement_seconds": 3.0,
            "initialization_timeout_seconds": 160,
        }
        if any(payload.get(field) != value for field, value in integrated_required.items()):
            raise ValueError("integrated config differs from the preauthorized v10 contract")
        if stability < 5.0:
            raise ValueError("integrated v10 requires its late-bound five-second SLO window")
        bounded_number("gpu0_overload_probe_seconds", lower=1.0, upper=10.0)
        evaluation = bounded_number("serving_recovery_evaluation_seconds", lower=0.25, upper=2.0)
        if not math.isclose(stability / evaluation, round(stability / evaluation)):
            raise ValueError("integrated stability must contain whole evaluation windows")
        bounded_number("serving_recovery_queue_threshold", lower=0, upper=128, integer=True)
        bounded_number("serving_maximum_pending_requests", lower=1, upper=20_000, integer=True)
        bounded_number("serving_restore_handoff_lead_requests", lower=2, upper=64, integer=True)
        queue_trigger = bounded_number(
            "serving_overload_queue_trigger", lower=10, upper=30, integer=True
        )
        queue_abort = bounded_number(
            "serving_overload_queue_abort", lower=11, upper=64, integer=True
        )
        if queue_abort <= queue_trigger:
            raise ValueError("integrated queue abort must exceed its overload trigger")
        if queue_trigger != 20 or queue_abort != 25:
            raise ValueError(
                "integrated v10 requires the preselected 20-request trigger and "
                "25-request scientific abort"
            )
        bounded_number("warmup_seconds", lower=1.0, upper=1.0)
        bounded_number("sanity_guard_measurement_seconds", lower=3.0, upper=3.0)
        bounded_number("probe_timeout_seconds", lower=15.0, upper=30.0)
        bounded_number("tail_drain_seconds", lower=1.0, upper=2.0)
        bounded_number("producer_queue_capacity", lower=64, upper=512, integer=True)
        bounded_number(
            "controller_timeout_seconds",
            lower=120.0,
            upper=_controller_wall_seconds(payload),
        )
        bounded_number("worker_shutdown_timeout_seconds", lower=5.0, upper=15.0)
        authorization, phase_budget = _load_pre_gpu_evidence(payload)
        if any(
            float(payload[field]) != float(authorization[field])
            for field in ("lambda_1_rps", "lambda_spike_rps", "lambda_2_rps")
        ):
            raise ValueError("integrated loads differ from immutable authorization")
        selected_source = next(
            row
            for row in authorization["source_calibration_artifact_hashes"]
            if row["role"] == "selected_load"
        )
        if (
            payload["calibration_selected_load_artifact"] != selected_source["artifact_reference"]
            or payload["calibration_selected_load_sha256"] != selected_source["artifact_sha256"]
        ):
            raise ValueError("integrated selected-load binding differs from authorization")
        if phase_budget.required_A100_seconds != 2.0 * phase_budget.predicted_wall_clock_seconds:
            raise ValueError("integrated phase budget does not bind two allocated A100s")
    elif methodology == "v10-global-capacity":
        spike_rate = bounded_number("serving_spike_request_rate_per_second", lower=1.0, upper=128.0)
        if spike_rate <= control_rate:
            raise ValueError("Experiment 004 v10 spike must exceed GPU0 control capacity")
        if bounded_number("serving_output_tokens", lower=64, upper=64, integer=True) != 64:
            raise AssertionError("unreachable exact output-token validation")
        if bounded_number("serving_prompt_tokens", lower=256, upper=256, integer=True) != 256:
            raise AssertionError("unreachable exact prompt-token validation")
        if stability < 5.0:
            raise ValueError("Experiment 004 v10 requires at least five seconds of stability")
        evaluation = bounded_number("serving_recovery_evaluation_seconds", lower=0.25, upper=2.0)
        if not math.isclose(stability / evaluation, round(stability / evaluation)):
            raise ValueError("Experiment 004 v10 stability must contain whole windows")
        bounded_number("gpu0_overload_probe_seconds", lower=1.0, upper=10.0)
        bounded_number("serving_recovery_queue_threshold", lower=0, upper=128, integer=True)
        bounded_number("serving_maximum_pending_requests", lower=1, upper=20_000, integer=True)
        bounded_number("serving_restore_handoff_lead_requests", lower=2, upper=64, integer=True)
        for field in (
            "lambda_1_rps",
            "lambda_2_rps",
            "lambda_spike_rps",
        ):
            bounded_number(field, lower=1.0, upper=256.0)
        calibration_attempt_id = payload["calibration_attempt_id"]
        if (
            not isinstance(calibration_attempt_id, str)
            or re.fullmatch(r"[a-z0-9][a-z0-9-]{7,95}", calibration_attempt_id) is None
        ):
            raise ValueError("Experiment 004 calibration_attempt_id is invalid")
        for field in ("calibration_selected_load_sha256", "agent9_approval_sha256"):
            if (
                not isinstance(payload[field], str)
                or re.fullmatch(r"[0-9a-f]{64}", payload[field]) is None
            ):
                raise ValueError(f"Experiment 004 {field} is invalid")
        for field in ("two_gpu_headroom_adr_artifact", "two_gpu_headroom_adr_sha256"):
            if payload[field] is not None and not isinstance(payload[field], str):
                raise ValueError(f"Experiment 004 {field} is invalid")
        if (
            payload["two_gpu_headroom_adr_sha256"] is not None
            and re.fullmatch(r"[0-9a-f]{64}", payload["two_gpu_headroom_adr_sha256"]) is None
        ):
            raise ValueError("Experiment 004 two_gpu_headroom_adr_sha256 is invalid")
        _validate_v10_calibration_binding(payload)
    elif methodology != "v9-independent-shards":
        raise ValueError("Experiment 004 serving methodology is unsupported")
    maximum_wall = bounded_number("maximum_wall_seconds", lower=60, upper=900)
    initialization = bounded_number(
        "initialization_timeout_seconds", lower=60, upper=900, integer=True
    )
    cleanup = bounded_number("cleanup_timeout_seconds", lower=10, upper=120, integer=True)
    if integrated:
        if (
            initialization != 160
            or not math.isclose(
                maximum_wall,
                _controller_wall_seconds(payload),
                rel_tol=0.0,
                abs_tol=1e-9,
            )
            or not math.isclose(
                float(payload["controller_timeout_seconds"]),
                _controller_wall_seconds(payload),
                rel_tol=0.0,
                abs_tol=1e-9,
            )
        ):
            raise ValueError("integrated readiness/transaction bounds are invalid")
    elif maximum_wall + initialization + cleanup + 40 > _LEGACY_RESERVATION_WALL_SECONDS:
        raise ValueError("Experiment 004 inner bounds exceed the reserved Modal Function lifetime")
    return payload


def _load_ledger() -> Any:
    from sloforge.helix.characterization.gpu_reclamation_methodology import (
        Experiment004GpuHourLedger,
    )

    if not _LEDGER.is_file():
        return Experiment004GpuHourLedger(consumed_additional_gpu_seconds=0.0)
    payload = json.loads(_LEDGER.read_text())
    # Migrate only a recognized prior ceiling after the explicit v10 operator
    # authorization. The next atomic reservation write persists the new hard
    # ceiling, so remote validation sees the same immutable ledger.
    if payload.get("hard_additional_gpu_seconds") in {5_400.0, 7_200.0}:
        if os.environ.get("SLOFORGE_EXP004_EXTENDED_GPU_HOURS") != "3.0":
            raise RuntimeError("v10 requires SLOFORGE_EXP004_EXTENDED_GPU_HOURS=3.0 authorization")
        payload["hard_additional_gpu_seconds"] = 10_800.0
    return Experiment004GpuHourLedger.model_validate_json(
        json.dumps(payload, sort_keys=True, separators=(",", ":")), strict=True
    )


def _reserve(config: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    from sloforge.helix.characterization.gpu_reclamation_methodology import (
        reserve_gpu_invocation,
    )

    reservation_id = f"exp004-{secrets.token_hex(10)}"
    digest = hashlib.sha256(_canonical_bytes(config)).hexdigest()
    ledger = _load_ledger()
    if config.get("execution_mode") == "integrated-calibration-v10":
        from sloforge.helix.characterization.gpu_reclamation_v10_budget import (
            evaluate_budget_gate,
        )

        _authorization, phase_budget = _load_pre_gpu_evidence(config)
        gate = evaluate_budget_gate(
            phase_budget,
            current_conservative_usage_A100_seconds=ledger.committed_additional_gpu_seconds,
            configured_hard_ceiling_A100_seconds=ledger.hard_additional_gpu_seconds,
        )
        if gate["status"] != "AUTHORIZE_V10" or gate["budget_pass"] is not True:
            raise RuntimeError(f"BUDGET_BLOCKER: {gate}")
        expected_required = _reservation_wall_seconds(config) * _GPU_COUNT
        if not math.isclose(
            float(gate["required_A100_seconds"]),
            expected_required,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise RuntimeError("phase-budget gate and reservation use different A100-seconds")
    updated, preflight = reserve_gpu_invocation(
        ledger,
        reservation_id=reservation_id,
        invocation_id=str(config["attempt_id"]),
        maximum_wall_seconds=_reservation_wall_seconds(config),
        config_sha256=digest,
    )
    _atomic_write(_LEDGER, updated.model_dump(mode="json"))
    return reservation_id, preflight.model_dump(mode="json")


def _validate_remote_identity(
    reservation_id: str,
    result: dict[str, Any],
    config: dict[str, Any],
) -> tuple[
    dict[str, Any],
    tuple[dict[str, Any], dict[str, Any]] | None,
    Any,
]:
    expected_result_fields = {
        "schema_version",
        "status",
        "attempt_id",
        "reservation_id",
        "config_sha256",
        "function_call_id",
        "requested_gpu",
        "gpu_count",
        "gpu_allocation_seconds",
        "gpu_seconds",
        "gpu_hours",
        "pilot_valid",
        "scientific_status",
        "measurable_gates_pass",
        "pilot_invalid_reasons",
        "run_error",
        "controller",
        "absolute_deadlines",
        "completed_at_utc",
        "remote_prefix",
        "remote_manifest_sha256",
    }
    if set(result) != {"result", "materialized"}:
        raise ValueError("pilot result envelope differs from its exact contract")
    remote = result["result"]
    materialized = result["materialized"]
    if (
        not isinstance(remote, dict)
        or set(remote) != expected_result_fields
        or not isinstance(materialized, dict)
        or set(materialized) != {"remote_path", "volume_name"}
    ):
        raise ValueError("pilot result/materialized payload differs from its exact contract")
    ledger = _load_ledger()
    matches = tuple(item for item in ledger.reservations if item.reservation_id == reservation_id)
    if len(matches) != 1:
        raise RuntimeError("pilot settlement reservation is absent or duplicated")
    reservation = matches[0]
    config_digest = hashlib.sha256(_canonical_bytes(config)).hexdigest()
    expected_remote_path = f"experiment-004/modal/{reservation.invocation_id}"
    methodology = config.get("serving_methodology", "v9-independent-shards")
    v10_run = (
        methodology == "v10-global-capacity"
        or config.get("execution_mode") == "integrated-calibration-v10"
    )
    deadlines = remote.get("absolute_deadlines")
    expected_deadline_fields = {
        "function_entry_monotonic_ns",
        "function_deadline_monotonic_ns",
        "controller_deadline_monotonic_ns",
        "post_controller_reserve_seconds",
        "remaining_function_seconds_at_completion",
    }
    if not isinstance(deadlines, dict) or set(deadlines) != expected_deadline_fields:
        raise ValueError("pilot absolute deadline evidence differs from its exact contract")
    entry_ns = deadlines["function_entry_monotonic_ns"]
    function_deadline_ns = deadlines["function_deadline_monotonic_ns"]
    controller_deadline_ns = deadlines["controller_deadline_monotonic_ns"]
    reserve_seconds = deadlines["post_controller_reserve_seconds"]
    remaining_seconds = deadlines["remaining_function_seconds_at_completion"]
    numeric_ns = (entry_ns, function_deadline_ns, controller_deadline_ns)
    integrated = config.get("execution_mode") == "integrated-calibration-v10"
    expected_reserve = _INTEGRATED_POST_CONTROLLER_RESERVE_SECONDS if integrated else 0.0
    expected_controller_seconds = (
        min(
            float(config["maximum_wall_seconds"]),
            _controller_wall_seconds(config),
        )
        if integrated
        else float(config["maximum_wall_seconds"])
    )
    if (
        any(isinstance(value, bool) or not isinstance(value, int) for value in numeric_ns)
        or entry_ns < 0
        or function_deadline_ns - entry_ns != round(_function_wall_seconds(config) * 1e9)
        or controller_deadline_ns - entry_ns != round(expected_controller_seconds * 1e9)
        or reserve_seconds != expected_reserve
        or not isinstance(remaining_seconds, (int, float))
        or isinstance(remaining_seconds, bool)
        or not math.isfinite(float(remaining_seconds))
        or not 0.0 <= float(remaining_seconds) <= _function_wall_seconds(config)
    ):
        raise ValueError("pilot absolute deadline evidence is invalid")
    successful_status = (
        remote.get("status") == "provisional"
        and remote.get("scientific_status") == "pending-local-finalization"
        and remote.get("measurable_gates_pass") is True
        and remote.get("pilot_valid") is False
        if v10_run
        else remote.get("status") == "succeeded"
        and remote.get("scientific_status") == "valid"
        and remote.get("pilot_valid") is True
    )
    invalid_reasons = remote.get("pilot_invalid_reasons")
    run_error = remote.get("run_error")
    failure_status = (
        remote.get("status") == "failed"
        and remote.get("scientific_status") == "invalid"
        and remote.get("measurable_gates_pass") is False
        and remote.get("pilot_valid") is False
        and isinstance(invalid_reasons, list)
        and bool(invalid_reasons)
        and all(isinstance(reason, str) and reason for reason in invalid_reasons)
        and (
            run_error is None
            or (
                isinstance(run_error, dict)
                and set(run_error) == {"type", "message", "traceback"}
                and all(
                    isinstance(run_error[field], str) and run_error[field] for field in run_error
                )
            )
        )
    )
    allocation_seconds = remote.get("gpu_allocation_seconds")
    gpu_seconds = remote.get("gpu_seconds")
    gpu_hours = remote.get("gpu_hours")

    def bounded_remote_number(value: Any, *, upper: float) -> float:
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            or not 0.0 < float(value) <= upper
        ):
            raise RuntimeError("pilot remote allocation timing is invalid")
        return float(value)

    allocation_seconds_value = bounded_remote_number(
        allocation_seconds, upper=_reservation_wall_seconds(config)
    )
    gpu_seconds_value = bounded_remote_number(
        gpu_seconds, upper=_reservation_wall_seconds(config) * _GPU_COUNT
    )
    gpu_hours_value = bounded_remote_number(
        gpu_hours, upper=_reservation_wall_seconds(config) * _GPU_COUNT / 3600.0
    )
    function_call_id = remote.get("function_call_id")
    if (
        not (successful_status or failure_status)
        or remote.get("schema_version")
        != "sloforge.branchfabric.experiment-004-function-completion/v1"
        or (successful_status and run_error is not None)
        or remote.get("attempt_id") != reservation.invocation_id
        or remote.get("reservation_id") != reservation_id
        or reservation.config_sha256 != config_digest
        or remote.get("config_sha256") != config_digest
        or not isinstance(function_call_id, str)
        or not function_call_id
        or len(function_call_id) > 256
        or remote.get("requested_gpu") != "A100-80GB:2"
        or remote.get("gpu_count") != 2
        or materialized.get("volume_name") != "sloforge-branchfabric-results"
        or materialized.get("remote_path") != expected_remote_path
        or remote.get("remote_prefix") != expected_remote_path
        or not isinstance(remote.get("remote_manifest_sha256"), str)
        or re.fullmatch(r"[0-9a-f]{64}", str(remote.get("remote_manifest_sha256"))) is None
        or not math.isclose(
            gpu_seconds_value,
            _GPU_COUNT * allocation_seconds_value,
            rel_tol=0.0,
            abs_tol=1e-6,
        )
        or not math.isclose(gpu_hours_value, gpu_seconds_value / 3600.0, rel_tol=0.0, abs_tol=1e-9)
        or allocation_seconds_value + float(remaining_seconds)
        > _function_wall_seconds(config) + 1e-6
    ):
        raise RuntimeError("pilot remote result failed status, resource, or identity validation")
    controller = remote.get("controller")
    if (
        not isinstance(controller, dict)
        or controller.get("status") not in {"succeeded", "failed"}
        or (successful_status and controller.get("status") != "succeeded")
    ):
        raise RuntimeError("pilot controller status contradicts the completion outcome")
    inventory_value = controller.get("inventory_before")
    if failure_status and inventory_value is None:
        # A controller may fail before it constructs its normal result. The
        # hash-bound topology bundle is validated after download and supplies
        # the exact physical identities for settlement in that case.
        return remote, None, reservation
    if not isinstance(inventory_value, list) or len(inventory_value) != 2:
        raise RuntimeError("pilot result lacks the exact two-GPU inventory")
    inventory = (inventory_value[0], inventory_value[1])
    if (
        any(not isinstance(item, dict) for item in inventory)
        or len({str(item.get("uuid")) for item in inventory}) != 2
        or any(
            "A100" not in str(item.get("name"))
            or "80GB" not in str(item.get("name"))
            or int(item.get("memory_total_mib", 0)) < 79_000
            for item in inventory
        )
    ):
        raise RuntimeError("pilot result did not use two distinct A100-80GB GPUs")
    return remote, inventory, reservation


def _validate_downloaded_manifest(
    local_root: Path,
    *,
    remote: dict[str, Any],
    materialized: dict[str, Any],
    expected_attempt_id: str,
) -> None:
    manifest_path = local_root / "REMOTE_MANIFEST.json"
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise FileNotFoundError("downloaded pilot manifest is absent or symlinked")
    if _sha256(manifest_path) != remote["remote_manifest_sha256"]:
        raise ValueError("returned and downloaded pilot manifest digests differ")
    manifest = json.loads(manifest_path.read_text())
    if (
        not isinstance(manifest, dict)
        or set(manifest) != {"schema_version", "attempt_id", "remote_prefix", "artifacts"}
        or manifest.get("schema_version")
        != "sloforge.branchfabric.experiment-004-remote-manifest/v1"
        or manifest.get("attempt_id") != expected_attempt_id
        or manifest.get("remote_prefix") != materialized.get("remote_path")
        or not isinstance(manifest.get("artifacts"), list)
    ):
        raise ValueError("downloaded pilot manifest identity is invalid")
    declared: dict[str, tuple[int, str]] = {}
    for item in manifest["artifacts"]:
        if not isinstance(item, dict) or set(item) != {"relative_path", "bytes", "sha256"}:
            raise ValueError("pilot manifest contains an invalid inventory row")
        relative = Path(str(item["relative_path"]))
        relative_key = relative.as_posix()
        size = item["bytes"]
        digest = item["sha256"]
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or relative_key in declared
            or not isinstance(size, int)
            or isinstance(size, bool)
            or size < 0
            or not isinstance(digest, str)
            or re.fullmatch(r"[0-9a-f]{64}", digest) is None
        ):
            raise ValueError("pilot manifest contains an unsafe or duplicate path")
        declared[relative_key] = (size, digest)
    actual_files = tuple(
        path for path in local_root.rglob("*") if path.is_file() and path != manifest_path
    )
    if any(path.is_symlink() for path in actual_files):
        raise ValueError("downloaded pilot artifact tree contains a symlink")
    actual = {path.relative_to(local_root).as_posix(): path for path in actual_files}
    if set(actual) != set(declared):
        raise ValueError("downloaded pilot artifact inventory differs from its manifest")
    for relative_key, path in actual.items():
        expected_bytes, expected_sha = declared[relative_key]
        if path.stat().st_size != expected_bytes or _sha256(path) != expected_sha:
            raise ValueError(f"downloaded pilot artifact failed integrity: {relative_key}")
    completion_path = local_root / "function-completion.json"
    completion = json.loads(completion_path.read_text())
    returned_completion = {
        key: value
        for key, value in remote.items()
        if key not in {"remote_prefix", "remote_manifest_sha256"}
    }
    if completion != returned_completion:
        raise ValueError("downloaded function completion differs from the returned result")


def _validate_exact_a100_inventory(
    inventory: tuple[dict[str, Any], dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    if (
        any(not isinstance(item, dict) for item in inventory)
        or len({str(item.get("uuid")) for item in inventory}) != 2
        or any(
            "A100" not in str(item.get("name"))
            or "80GB" not in str(item.get("name")).replace(" ", "")
            or int(item.get("memory_total_mib", 0)) < 79_000
            for item in inventory
        )
    ):
        raise RuntimeError("pilot result did not use two distinct A100-80GB GPUs")
    return inventory


def _inventory_from_downloaded_topology(
    local_root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Recover failed-controller hardware identity from hash-validated raw topology."""

    path = local_root / "topology/topology-commands.json"
    payload = json.loads(path.read_text())
    if not isinstance(payload, list):
        raise ValueError("failed pilot topology artifact is not a command list")
    matches = tuple(
        row
        for row in payload
        if isinstance(row, dict)
        and isinstance(row.get("command"), dict)
        and row["command"].get("name") == "gpu_inventory"
    )
    if len(matches) != 1:
        raise ValueError("failed pilot bundle lacks one exact GPU inventory command")
    command = matches[0]
    if command.get("status") != "success" or command.get("returncode") != 0:
        raise ValueError("failed pilot GPU inventory command did not succeed")
    rows = tuple(line for line in str(command.get("stdout", "")).splitlines() if line.strip())
    if len(rows) != 2:
        raise ValueError("failed pilot inventory does not contain exactly two GPUs")
    inventory: list[dict[str, Any]] = []
    for row in rows:
        fields = tuple(field.strip() for field in row.split(","))
        if len(fields) != 6 or not fields[0].isdigit():
            raise ValueError("failed pilot inventory contains an invalid nvidia-smi row")
        try:
            memory_total_mib = int(fields[4])
        except ValueError as error:
            raise ValueError("failed pilot inventory memory is not integral MiB") from error
        inventory.append(
            {
                "index": int(fields[0]),
                "uuid": fields[1],
                "name": fields[2],
                "pci_bus_id": fields[3],
                "memory_total_mib": memory_total_mib,
                "driver_version": fields[5],
            }
        )
    if {item["index"] for item in inventory} != {0, 1}:
        raise ValueError("failed pilot inventory indices are not the exact visible pair")
    return _validate_exact_a100_inventory((inventory[0], inventory[1]))


def _preserve_settled_invalid_outcome(
    *,
    local_root: Path,
    remote: dict[str, Any],
    ledger_before: Any,
    settled: Any,
    manifest_path: Path,
) -> None:
    if remote["status"] != "failed":
        return
    _validate_downloaded_invalid_outcome(local_root=local_root, remote=remote)
    v10_path = local_root / "analysis/scientific-validity.remote-provisional.json"
    if v10_path.is_file():
        from sloforge.helix.characterization.gpu_reclamation_v10_validity import (
            V10ScientificValidity,
        )

        validity = V10ScientificValidity.model_validate_json(v10_path.read_text(), strict=True)
        canonical_path = local_root / "analysis/scientific-validity.json"
        if canonical_path.exists():
            raise FileExistsError("failed v10 bundle already contains local canonical validity")
        _write_new(canonical_path, validity.model_dump(mode="json"))
    interval = next(
        item for item in settled.intervals if item.invocation_id == remote["attempt_id"]
    )
    _write_new(
        local_root / "analysis/failed-run-settlement.json",
        {
            "schema_version": "sloforge.branchfabric.experiment-004-failed-run-settlement/v1",
            "attempt_id": remote["attempt_id"],
            "function_call_id": remote["function_call_id"],
            "status": "settled-invalid",
            "scientific_status": "invalid",
            "classification": (
                "runtime-failure" if remote["run_error"] is not None else "scientific-blocker"
            ),
            "invalid_reasons": remote["pilot_invalid_reasons"],
            "remote_observed_allocation_seconds": remote["gpu_allocation_seconds"],
            "conservatively_accounted_wall_seconds": interval.accounted_wall_seconds,
            "accounted_gpu_seconds": interval.gpu_seconds,
            "consumed_gpu_seconds_before": ledger_before.consumed_additional_gpu_seconds,
            "consumed_gpu_seconds_after": settled.consumed_additional_gpu_seconds,
            "reservation_cleared": not settled.reservations,
            "remote_manifest_sha256": _sha256(manifest_path),
            "gpu_hours_sha256": _sha256(_LEDGER),
        },
    )


def _validate_downloaded_invalid_outcome(*, local_root: Path, remote: dict[str, Any]) -> None:
    """Require raw scientific evidence to agree with a returned failed status."""

    if remote["status"] != "failed":
        return
    v10_path = local_root / "analysis/scientific-validity.remote-provisional.json"
    if v10_path.is_file():
        from sloforge.helix.characterization.gpu_reclamation_v10_validity import (
            V10ScientificValidity,
        )

        validity = V10ScientificValidity.model_validate_json(v10_path.read_text(), strict=True)
        if (local_root / "analysis/scientific-validity.json").exists():
            raise ValueError("failed remote bundle improperly claims local canonical validity")
        if validity.scientifically_valid or tuple(remote["pilot_invalid_reasons"]) != (
            validity.invalid_reasons
        ):
            raise ValueError("failed v10 bundle does not preserve its scientific invalidity")
    else:
        from sloforge.helix.characterization.gpu_reclamation_pilot import PilotAssessment

        assessment_path = local_root / "analysis/pilot-assessment.json"
        assessment = PilotAssessment.model_validate_json(assessment_path.read_text(), strict=True)
        if assessment.pilot_valid is not False or assessment.invalid_reasons != tuple(
            remote["pilot_invalid_reasons"]
        ):
            raise ValueError("failed legacy bundle does not preserve its scientific invalidity")


def _settle(
    reservation_id: str,
    result: dict[str, Any],
    client_elapsed_s: float,
    config: dict[str, Any],
) -> Path:
    from sloforge.helix.characterization.gpu_reclamation_methodology import (
        ArtifactSampleRef,
        settle_gpu_invocation,
    )

    remote, inventory, reservation = _validate_remote_identity(reservation_id, result, config)
    materialized = result["materialized"]
    local_root = _EXPERIMENT_ROOT / "raw/modal" / str(remote["attempt_id"])
    if local_root.exists():
        raise FileExistsError(f"local immutable result already exists: {local_root}")
    local_root.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "modal",
            "volume",
            "get",
            str(materialized["volume_name"]),
            str(materialized["remote_path"]),
            str(local_root.parent),
        ],
        check=True,
        timeout=120.0,
    )
    _validate_downloaded_manifest(
        local_root,
        remote=remote,
        materialized=materialized,
        expected_attempt_id=reservation.invocation_id,
    )
    _validate_downloaded_invalid_outcome(local_root=local_root, remote=remote)
    if inventory is None:
        inventory = _inventory_from_downloaded_topology(local_root)
    manifest_path = local_root / "REMOTE_MANIFEST.json"
    raw_ref = ArtifactSampleRef(
        artifact_reference=str(manifest_path),
        artifact_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        sample_selector="$",
    )
    ledger_before = _load_ledger()
    settled = settle_gpu_invocation(
        ledger_before,
        reservation_id=reservation_id,
        function_call_id=str(remote["function_call_id"]),
        actual_gpu_models=(str(inventory[0]["name"]), str(inventory[1]["name"])),
        gpu_uuids=(str(inventory[0]["uuid"]), str(inventory[1]["uuid"])),
        client_elapsed_seconds=client_elapsed_s,
        remote_observed_allocation_seconds=float(remote["gpu_allocation_seconds"]),
        raw_manifest=raw_ref,
        gpu_price_per_hour_usd=_GPU_USD_PER_HOUR,
    )
    _atomic_write(_LEDGER, settled.model_dump(mode="json"))
    if (local_root / "analysis/scientific-validity.remote-provisional.json").is_file():
        from sloforge.helix.characterization.gpu_reclamation_v10_validity import BudgetEvidence

        invocation_gpu_seconds = (
            settled.consumed_additional_gpu_seconds - ledger_before.consumed_additional_gpu_seconds
        )
        actual_gpu_seconds = float(remote["gpu_seconds"])
        cumulative_conservative_cost_usd = sum(
            interval.gpu_cost_usd or 0.0 for interval in settled.intervals
        ) + sum(
            charge.conservative_gpu_cost_usd or 0.0
            for charge in settled.conservative_failure_charges
        )
        budget_evidence = BudgetEvidence(
            conservative_gpu_seconds_before=ledger_before.consumed_additional_gpu_seconds,
            invocation_gpu_seconds=invocation_gpu_seconds,
            conservative_gpu_seconds_after=settled.consumed_additional_gpu_seconds,
            hard_ceiling_gpu_seconds=settled.hard_additional_gpu_seconds,
            ledger_updated=True,
        )
        _atomic_write(
            local_root / "analysis/budget-settlement.json",
            {
                "schema_version": ("sloforge.branchfabric.experiment-004-v10-budget-settlement/v1"),
                "attempt_id": remote["attempt_id"],
                "budget": budget_evidence.model_dump(mode="json"),
                "client_elapsed_seconds": client_elapsed_s,
                "actual_gpu_seconds": actual_gpu_seconds,
                "actual_gpu_hours": float(remote["gpu_hours"]),
                "actual_gpu_cost_usd": (actual_gpu_seconds / 3_600.0 * _GPU_USD_PER_HOUR),
                "conservative_gpu_seconds": invocation_gpu_seconds,
                "conservative_gpu_hours": invocation_gpu_seconds / 3_600.0,
                "invocation_conservative_cost_usd": (
                    invocation_gpu_seconds / 3_600.0 * _GPU_USD_PER_HOUR
                ),
                "cumulative_conservative_gpu_seconds": (settled.consumed_additional_gpu_seconds),
                "cumulative_conservative_gpu_hours": (
                    settled.consumed_additional_gpu_seconds / 3_600.0
                ),
                "cumulative_conservative_gpu_cost_usd": (cumulative_conservative_cost_usd),
                "remaining_hard_budget_gpu_seconds": (
                    settled.hard_additional_gpu_seconds - settled.committed_additional_gpu_seconds
                ),
                "gpu_hours_sha256": hashlib.sha256(_LEDGER.read_bytes()).hexdigest(),
            },
        )
    _preserve_settled_invalid_outcome(
        local_root=local_root,
        remote=remote,
        ledger_before=ledger_before,
        settled=settled,
        manifest_path=manifest_path,
    )
    return local_root


def _record_terminal_failure(
    *,
    reservation_id: str,
    attempt_id: str,
    stage: str,
    error: BaseException,
    elapsed_seconds: float,
    completed: subprocess.CompletedProcess[str] | None = None,
    reservation_wall_seconds: float = _LEGACY_RESERVATION_WALL_SECONDS,
) -> dict[str, Any]:
    """Preserve a failure and clear its reservation with a conservative charge.

    If trustworthy remote timing cannot be validated, the complete preflight
    reservation is charged explicitly as a conservative failure charge.  It is
    never represented as an observed GPU-active interval.
    """

    from sloforge.helix.characterization.gpu_reclamation_methodology import (
        ArtifactSampleRef,
        charge_failed_gpu_reservation,
    )

    ledger_before = _load_ledger()
    active = tuple(
        item for item in ledger_before.reservations if item.reservation_id == reservation_id
    )
    settled_intervals = tuple(
        item for item in ledger_before.intervals if item.invocation_id == attempt_id
    )
    failure_charges = tuple(
        item
        for item in ledger_before.conservative_failure_charges
        if item.invocation_id == attempt_id
    )
    if len(active) > 1 or len(settled_intervals) > 1 or len(failure_charges) > 1:
        raise RuntimeError("terminal failure accounting identity is duplicated")

    charge_evidence_path: Path | None = None
    charge_evidence_sha256: str | None = None
    if active:
        reservation = active[0]
        if reservation.invocation_id != attempt_id or not math.isclose(
            reservation.maximum_wall_seconds,
            reservation_wall_seconds,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise RuntimeError("terminal failure reservation identity or bound differs")
        charge_evidence_path = _EXPERIMENT_ROOT / (
            f"logs/{attempt_id}-conservative-failure-charge-evidence.json"
        )
        _write_new(
            charge_evidence_path,
            {
                "schema_version": (
                    "sloforge.branchfabric.experiment-004-conservative-failure-charge-evidence/v1"
                ),
                "attempt_id": attempt_id,
                "reservation_id": reservation_id,
                "stage": stage,
                "error": {"type": type(error).__name__, "message": str(error)},
                "client_elapsed_seconds": elapsed_seconds,
                "actual_gpu_seconds": None,
                "actual_gpu_seconds_status": "unavailable-after-launcher-failure",
                "charged_wall_seconds": reservation.maximum_wall_seconds,
                "conservatively_charged_gpu_seconds": reservation.maximum_gpu_seconds,
                "accounting_policy": "charge-full-preflight-bound-without-fabricated-actuals",
                "stdout_tail": None if completed is None else completed.stdout[-65_536:],
                "stderr_tail": None if completed is None else completed.stderr[-65_536:],
            },
        )
        charge_evidence_sha256 = _sha256(charge_evidence_path)
        ledger_after = charge_failed_gpu_reservation(
            ledger_before,
            reservation_id=reservation_id,
            failure_stage=stage,
            failure_evidence=ArtifactSampleRef(
                artifact_reference=str(charge_evidence_path),
                artifact_sha256=charge_evidence_sha256,
                sample_selector="$",
            ),
            gpu_price_per_hour_usd=_GPU_USD_PER_HOUR,
        )
        _atomic_write(_LEDGER, ledger_after.model_dump(mode="json"))
        failure_charge = ledger_after.conservative_failure_charges[-1]
        invocation_conservative_gpu_seconds = failure_charge.conservative_gpu_seconds
        invocation_conservative_cost_usd = failure_charge.conservative_gpu_cost_usd
        charge_state = "full_reservation_settled_as_conservative_failure_charge"
    elif settled_intervals or failure_charges:
        ledger_after = ledger_before
        if settled_intervals:
            invocation_conservative_gpu_seconds = settled_intervals[0].gpu_seconds
            invocation_conservative_cost_usd = settled_intervals[0].gpu_cost_usd
            charge_state = "already_settled_as_verified_interval"
        else:
            invocation_conservative_gpu_seconds = failure_charges[0].conservative_gpu_seconds
            invocation_conservative_cost_usd = failure_charges[0].conservative_gpu_cost_usd
            charge_state = "already_settled_as_conservative_failure_charge"
    else:
        raise RuntimeError("terminal failure has neither an active nor settled ledger identity")

    path = _EXPERIMENT_ROOT / f"logs/{attempt_id}-terminal-failure.json"
    payload = {
        "schema_version": "sloforge.branchfabric.experiment-004-terminal-failure/v2",
        "attempt_id": attempt_id,
        "reservation_id": reservation_id,
        "stage": stage,
        "error": {"type": type(error).__name__, "message": str(error)},
        "client_elapsed_seconds": elapsed_seconds,
        "actual_gpu_seconds": None,
        "actual_gpu_seconds_status": "unavailable-or-not-trusted-after-launcher-failure",
        "conservative_charge_state": charge_state,
        "conservatively_accounted_gpu_seconds": invocation_conservative_gpu_seconds,
        "conservative_gpu_cost_usd": invocation_conservative_cost_usd,
        "conservative_gpu_seconds_before": ledger_before.consumed_additional_gpu_seconds,
        "conservative_gpu_seconds_after": ledger_after.consumed_additional_gpu_seconds,
        "remaining_hard_budget_gpu_seconds": (
            ledger_after.hard_additional_gpu_seconds - ledger_after.committed_additional_gpu_seconds
        ),
        "reservation_cleared": not ledger_after.reservations,
        "charge_evidence_reference": (
            None if charge_evidence_path is None else str(charge_evidence_path)
        ),
        "charge_evidence_sha256": charge_evidence_sha256,
        "gpu_hours_sha256": _sha256(_LEDGER),
        "stdout_tail": None if completed is None else completed.stdout[-65_536:],
        "stderr_tail": None if completed is None else completed.stderr[-65_536:],
    }
    _write_new(path, payload)
    return payload


def _parse_modal_result(stdout: str) -> dict[str, Any]:
    """Extract the local-entrypoint JSON without assuming Modal CLI log routing."""

    candidates: list[dict[str, Any]] = []
    for line in stdout.splitlines():
        if not line.lstrip().startswith("{"):
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and set(value) == {"result", "materialized"}:
            candidates.append(value)
    if len(candidates) != 1:
        raise ValueError(
            f"Modal pilot output contained {len(candidates)} exact local-entrypoint result objects"
        )
    return candidates[0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-path", required=True, type=Path)
    args = parser.parse_args()
    # Fail before even parsing a config when paid-resource authorization is
    # absent. The second call below applies the mode-specific price ceiling.
    _require_budget({})
    config_path = args.config_path.resolve(strict=True)
    config = _validate_config(config_path)
    _require_budget(config)
    _LOCK.parent.mkdir(parents=True, exist_ok=True)
    with _LOCK.open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        reservation_id, preflight = _reserve(config)
        launch_token = secrets.token_hex(32)
        environment = dict(os.environ)
        environment["SLOFORGE_MODAL_PREFLIGHT_TOKEN"] = launch_token
        environment["SLOFORGE_EXP004_RESERVATION_ID"] = reservation_id
        command = [
            "modal",
            "run",
            str(_APP),
            "--config-path",
            str(config_path),
        ]
        started = time.monotonic()
        completed: subprocess.CompletedProcess[str] | None = None
        try:
            completed = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                # The sole coordinator's wall-clock reservation is the outer
                # bound. Never let CLI startup plus remote execution exceed it.
                timeout=_reservation_wall_seconds(config),
                env=environment,
            )
            elapsed = time.monotonic() - started
            if completed.returncode != 0:
                sys.stderr.write(completed.stdout)
                sys.stderr.write(completed.stderr)
                raise RuntimeError("Modal pilot process returned a failure")
            result = _parse_modal_result(completed.stdout)
            local_root = _settle(reservation_id, result, elapsed, config)
        except BaseException as caught:
            elapsed = time.monotonic() - started
            _record_terminal_failure(
                reservation_id=reservation_id,
                attempt_id=str(config["attempt_id"]),
                stage=("modal-process" if completed is None else "result-validation-settlement"),
                error=caught,
                elapsed_seconds=elapsed,
                completed=completed,
                reservation_wall_seconds=_reservation_wall_seconds(config),
            )
            raise RuntimeError(
                "Experiment 004 pilot failed; its reservation was conservatively settled"
            ) from caught
        remote_failed = result["result"]["status"] == "failed"
        print(
            _canonical_bytes(
                {
                    "preflight": preflight,
                    "maximum_cost_usd": _maximum_cost_usd(config),
                    "modal": result,
                    "local_root": str(local_root),
                    "scientific_status": result["result"]["scientific_status"],
                    "reservation_cleared": True,
                    "v10_finalization": (
                        "blocked_by_settled_remote_failure"
                        if remote_failed
                        else "requires explicit provider cleanup evidence"
                        if (
                            config.get("serving_methodology") == "v10-global-capacity"
                            or config.get("execution_mode") == "integrated-calibration-v10"
                        )
                        else "not_applicable"
                    ),
                }
            ).decode(),
            end="",
        )
    return 1 if remote_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
