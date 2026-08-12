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
_FUNCTION_WALL_SECONDS = 340.0
_CLIENT_RESERVATION_WALL_SECONDS = 340.0
_GPU_COUNT = 2
_GPU_USD_PER_HOUR = 2.4984
_CPU_USD_PER_CORE_SECOND = 0.0000131
_MEMORY_USD_PER_GIB_SECOND = 0.00000222
_MAXIMUM_COST_USD = (
    _CLIENT_RESERVATION_WALL_SECONDS * _GPU_COUNT / 3600.0 * _GPU_USD_PER_HOUR
    + _CLIENT_RESERVATION_WALL_SECONDS
    * (16.0 * _CPU_USD_PER_CORE_SECOND + 64.0 * _MEMORY_USD_PER_GIB_SECOND)
)
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
}
_SELECTED_LOAD_REFERENCE = (
    "artifacts/branchfabric/gpu-validation/experiment-004/calibration/selected-load.json"
)
_AGENT9_APPROVAL_REFERENCE = (
    "artifacts/branchfabric/gpu-validation/experiment-004/calibration/reviews/"
    "agent9-selected-load-approval.json"
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


def _require_budget() -> float:
    raw = os.environ.get("SLOFORGE_GPU_BUDGET_USD")
    if raw is None:
        raise RuntimeError("SLOFORGE_GPU_BUDGET_USD is required before invoking Modal")
    try:
        budget = float(raw)
    except ValueError as error:
        raise RuntimeError("SLOFORGE_GPU_BUDGET_USD must be finite and positive") from error
    if not math.isfinite(budget) or budget <= 0:
        raise RuntimeError("SLOFORGE_GPU_BUDGET_USD must be finite and positive")
    if budget < _MAXIMUM_COST_USD:
        raise RuntimeError(
            f"bounded two-A100 pilot may cost ${_MAXIMUM_COST_USD:.4f}, exceeding budget"
        )
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
        or not math.isclose(
            result.configured_rate_rps, expected_rate, rel_tol=0.0, abs_tol=1e-12
        )
    ):
        raise ValueError("v10 selected probe does not support its declared operating point")


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
        and 0.80 <= selection.spike_over_lambda_2 <= 0.90
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
    expected_fields = _V10_CONFIG_FIELDS if methodology == "v10-global-capacity" else _CONFIG_FIELDS
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
    if methodology == "v9-independent-shards":
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
    if methodology == "v9-independent-shards" and (
        gpu0_spike_rate + gpu1_spike_rate < 4.0 * control_rate
    ):
        raise ValueError("Experiment 004 serving spike offers less than 4x control load")
    bounded_number("serving_slo_maximum_ttft_seconds", lower=0.0000001, upper=10.0)
    bounded_number(
        "serving_slo_maximum_inter_token_latency_seconds",
        lower=0.0000001,
        upper=5.0,
    )
    stability = bounded_number("serving_slo_stability_window_seconds", lower=0.5, upper=5.0)
    if methodology == "v9-independent-shards" and stability > float(
        payload["temporary_serving_seconds"]
    ):
        raise ValueError("Experiment 004 serving SLO stability window exceeds its measurement")
    if methodology == "v10-global-capacity":
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
            if not isinstance(payload[field], str) or re.fullmatch(r"[0-9a-f]{64}", payload[field]) is None:
                raise ValueError(f"Experiment 004 {field} is invalid")
        _validate_v10_calibration_binding(payload)
    elif methodology != "v9-independent-shards":
        raise ValueError("Experiment 004 serving methodology is unsupported")
    maximum_wall = bounded_number("maximum_wall_seconds", lower=60, upper=900, integer=True)
    initialization = bounded_number(
        "initialization_timeout_seconds", lower=60, upper=900, integer=True
    )
    cleanup = bounded_number("cleanup_timeout_seconds", lower=10, upper=120, integer=True)
    if maximum_wall + initialization + cleanup + 40 > _FUNCTION_WALL_SECONDS:
        raise ValueError("Experiment 004 inner bounds exceed the reserved Modal Function lifetime")
    return payload


def _load_ledger() -> Any:
    from sloforge.helix.characterization.gpu_reclamation_methodology import (
        Experiment004GpuHourLedger,
    )

    if not _LEDGER.is_file():
        return Experiment004GpuHourLedger(consumed_additional_gpu_seconds=0.0)
    payload = json.loads(_LEDGER.read_text())
    # Migrate only the exact prior contract after an explicit operator
    # authorization. The next atomic reservation write persists the new hard
    # ceiling, so remote validation sees the same immutable ledger.
    if payload.get("hard_additional_gpu_seconds") == 5_400.0:
        if os.environ.get("SLOFORGE_EXP004_EXTENDED_GPU_HOURS") != "2.0":
            raise RuntimeError("v9 requires SLOFORGE_EXP004_EXTENDED_GPU_HOURS=2.0 authorization")
        payload["hard_additional_gpu_seconds"] = 7_200.0
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
    updated, preflight = reserve_gpu_invocation(
        ledger,
        reservation_id=reservation_id,
        invocation_id=str(config["attempt_id"]),
        maximum_wall_seconds=_CLIENT_RESERVATION_WALL_SECONDS,
        config_sha256=digest,
    )
    _atomic_write(_LEDGER, updated.model_dump(mode="json"))
    return reservation_id, preflight.model_dump(mode="json")


def _validate_remote_identity(
    reservation_id: str,
    result: dict[str, Any],
    config: dict[str, Any],
) -> tuple[dict[str, Any], tuple[dict[str, Any], dict[str, Any]], Any]:
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
    status_valid = (
        remote.get("status") == "provisional"
        and remote.get("scientific_status") == "pending-local-finalization"
        and remote.get("measurable_gates_pass") is True
        and remote.get("pilot_valid") is False
        if methodology == "v10-global-capacity"
        else remote.get("status") == "succeeded"
        and remote.get("scientific_status") == "valid"
        and remote.get("pilot_valid") is True
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
        allocation_seconds, upper=_CLIENT_RESERVATION_WALL_SECONDS
    )
    gpu_seconds_value = bounded_remote_number(
        gpu_seconds, upper=_CLIENT_RESERVATION_WALL_SECONDS * _GPU_COUNT
    )
    gpu_hours_value = bounded_remote_number(
        gpu_hours, upper=_CLIENT_RESERVATION_WALL_SECONDS * _GPU_COUNT / 3600.0
    )
    function_call_id = remote.get("function_call_id")
    if (
        not status_valid
        or remote.get("schema_version")
        != "sloforge.branchfabric.experiment-004-function-completion/v1"
        or remote.get("run_error") is not None
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
        or not math.isclose(
            gpu_hours_value, gpu_seconds_value / 3600.0, rel_tol=0.0, abs_tol=1e-9
        )
    ):
        raise RuntimeError("pilot remote result failed success, resource, or identity validation")
    controller = remote.get("controller")
    if not isinstance(controller, dict) or controller.get("status") != "succeeded":
        raise RuntimeError("pilot controller was not successful")
    inventory_value = controller.get("inventory_before")
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
            settled.consumed_additional_gpu_seconds
            - ledger_before.consumed_additional_gpu_seconds
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
                "schema_version": (
                    "sloforge.branchfabric.experiment-004-v10-budget-settlement/v1"
                ),
                "attempt_id": remote["attempt_id"],
                "budget": budget_evidence.model_dump(mode="json"),
                "gpu_hours_sha256": hashlib.sha256(_LEDGER.read_bytes()).hexdigest(),
            },
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
) -> None:
    path = _EXPERIMENT_ROOT / f"logs/{attempt_id}-terminal-failure.json"
    _write_new(
        path,
        {
            "schema_version": "sloforge.branchfabric.experiment-004-terminal-failure/v1",
            "attempt_id": attempt_id,
            "reservation_id": reservation_id,
            "stage": stage,
            "error": {"type": type(error).__name__, "message": str(error)},
            "client_elapsed_seconds": elapsed_seconds,
            "conservative_charge_state": "full_reservation_retained_in_gpu_hour_ledger",
            "conservatively_committed_gpu_seconds": (
                _CLIENT_RESERVATION_WALL_SECONDS * _GPU_COUNT
            ),
            "reservation_cleared": False,
            "stdout_tail": None if completed is None else completed.stdout[-65_536:],
            "stderr_tail": None if completed is None else completed.stderr[-65_536:],
        },
    )


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
    _require_budget()
    config_path = args.config_path.resolve(strict=True)
    config = _validate_config(config_path)
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
                timeout=_CLIENT_RESERVATION_WALL_SECONDS,
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
            )
            raise RuntimeError(
                "Experiment 004 pilot failed; its full reservation remains conservatively charged"
            ) from caught
        print(
            _canonical_bytes(
                {
                    "preflight": preflight,
                    "maximum_cost_usd": _MAXIMUM_COST_USD,
                    "modal": result,
                    "local_root": str(local_root),
                    "v10_finalization": (
                        "requires explicit provider cleanup evidence"
                        if config.get("serving_methodology") == "v10-global-capacity"
                        else "not_applicable"
                    ),
                }
            ).decode(),
            end="",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
