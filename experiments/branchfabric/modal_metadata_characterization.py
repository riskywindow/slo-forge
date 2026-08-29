"""One-function, one-A100 coordinator for BranchFabric Experiment 003."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import sys
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any, cast

import modal
from modal_real_gpu_cow import (
    GPU_FUNCTION_CPU_CORES,
    GPU_FUNCTION_MEMORY_GIB,
    MODAL_A100_80GB_USD_PER_HOUR,
    MODEL_MOUNT,
    MODEL_REVISION,
    RESULTS_MOUNT,
    _inventory,
    _sha256,
    _validate_model_manifest,
    _write_json,
    gpu_image,
    model_volume,
    results_volume,
)

APP_NAME = "sloforge-branchfabric-metadata-characterization-003"
REMOTE_EXPERIMENT_PREFIX = "experiment-003/modal"
LOCAL_EXPERIMENT_ROOT = Path("artifacts/branchfabric/gpu-validation/experiment-003").resolve()
GPU_FUNCTION_TIMEOUT_SECONDS = 2400
GPU_FUNCTION_STARTUP_TIMEOUT_SECONDS = 900
GPU_HOUR_TARGET = 0.75
GPU_HOUR_HARD_LIMIT = 1.25
HISTORICAL_EXPERIMENT_002_GPU_HOURS = 0.5640564932644445
CPU_USD_PER_CORE_SECOND = 0.0000131
MEMORY_USD_PER_GIB_SECOND = 0.00000222
BUDGET_RESERVE_FRACTION = 0.15
SCHEMA = "sloforge.branchfabric.modal-metadata-characterization-config/v1"
CAMPAIGN_RESERVATION_GPU_HOURS = GPU_FUNCTION_TIMEOUT_SECONDS / 3600
_SAFE_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{7,95}\Z")
_LAUNCH_TOKEN = os.getenv("SLOFORGE_MODAL_PREFLIGHT_TOKEN", "")
_CLOUD_GRAPH_ENABLED = len(_LAUNCH_TOKEN) == 64 and all(
    character in "0123456789abcdef" for character in _LAUNCH_TOKEN
)
_CLI_RUN = "modal" in Path(sys.argv[0]).as_posix().lower() and "run" in sys.argv[1:]
if _CLI_RUN and not _CLOUD_GRAPH_ENABLED:
    raise RuntimeError("Experiment 003 Modal hydration requires the budget-first launcher")

app = modal.App(APP_NAME, tags={"project": "sloforge", "experiment": "branchfabric-003"})
_run_campaign_function: Any = None


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n"
    ).encode()


def _replace_local_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("xb") as handle:
        handle.write(_canonical_bytes(value))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _validate_trial(row: object) -> dict[str, Any]:
    if not isinstance(row, dict) or set(row) != {
        "schema_version",
        "attempt_id",
        "model",
        "model_revision",
        "tokenizer_revision",
        "runtime",
        "runtime_version",
        "fanout",
        "prefix_length",
        "suffix_length",
        "seed",
        "baseline_mode",
        "implementation",
        "tracing_level",
        "maximum_wall_seconds",
        "initialization_timeout_seconds",
        "cleanup_timeout_seconds",
        "gpu_memory_utilization",
        "gpu_sample_interval_seconds",
        "pair_id",
        "pair_order_position",
        "pair_order_rule",
        "pair_order_seed",
        "campaign_id",
    }:
        raise ValueError("Experiment 003 trial has missing or unsupported fields")
    expected = {
        "schema_version": SCHEMA,
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "model_revision": "a09a35458c702b33eeacc393d103063234e8bc28",
        "tokenizer_revision": "a09a35458c702b33eeacc393d103063234e8bc28",
        "runtime": "vllm",
        "runtime_version": "0.23.0",
        "prefix_length": 16_384,
        "suffix_length": 256,
    }
    if any(row.get(name) != value for name, value in expected.items()):
        raise ValueError("Experiment 003 trial identity or shape is not pinned")
    if row.get("fanout") not in {1, 8, 16, 32}:
        raise ValueError("Experiment 003 fanout is outside the bounded matrix")
    if row.get("baseline_mode") not in {"independent", "shared_root"}:
        raise ValueError("Experiment 003 has an invalid baseline mode")
    if row.get("implementation") not in {"baseline", "optimized"}:
        raise ValueError("Experiment 003 has an invalid implementation")
    if row.get("tracing_level") not in {"disabled", "minimal", "full"}:
        raise ValueError("Experiment 003 has an invalid tracing level")
    attempt = row.get("attempt_id")
    if not isinstance(attempt, str) or _SAFE_IDENTIFIER.fullmatch(attempt) is None:
        raise ValueError("Experiment 003 attempt_id is not path safe")
    bounded_exact = {
        "maximum_wall_seconds": 240,
        "initialization_timeout_seconds": 300,
        "cleanup_timeout_seconds": 60,
        "gpu_memory_utilization": 0.8,
        "gpu_sample_interval_seconds": 1.0,
    }
    if any(row.get(name) != value for name, value in bounded_exact.items()):
        raise ValueError("Experiment 003 trial resource or timeout bounds are invalid")
    seed = row.get("seed")
    if not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed < 1 << 63:
        raise ValueError("Experiment 003 seed must be an unsigned 63-bit integer")
    return cast(dict[str, Any], json.loads(json.dumps(row)))


def validate_campaign(payload: object) -> dict[str, Any]:
    if not isinstance(payload, dict) or set(payload) != {
        "schema_version",
        "campaign_id",
        "trials",
        "maximum_campaign_seconds",
    }:
        raise ValueError("Experiment 003 campaign has an invalid shape")
    if payload.get("schema_version") != "sloforge.branchfabric.modal-campaign-003/v1":
        raise ValueError("Experiment 003 campaign schema is invalid")
    campaign_id = payload.get("campaign_id")
    if not isinstance(campaign_id, str) or _SAFE_IDENTIFIER.fullmatch(campaign_id) is None:
        raise ValueError("Experiment 003 campaign_id is invalid")
    maximum = payload.get("maximum_campaign_seconds")
    if not isinstance(maximum, int) or not 60 <= maximum <= GPU_FUNCTION_TIMEOUT_SECONDS:
        raise ValueError("Experiment 003 campaign bound is invalid")
    trials = payload.get("trials")
    if not isinstance(trials, list) or not 1 <= len(trials) <= 24:
        raise ValueError("Experiment 003 campaign requires 1..24 sequential children")
    normalized = [_validate_trial(row) for row in trials]
    attempts = [row["attempt_id"] for row in normalized]
    if len(attempts) != len(set(attempts)):
        raise ValueError("Experiment 003 attempt IDs must be unique")
    if any(row["campaign_id"] != campaign_id for row in normalized):
        raise ValueError("Experiment 003 trial campaign IDs disagree")
    return {
        "schema_version": payload["schema_version"],
        "campaign_id": campaign_id,
        "trials": normalized,
        "maximum_campaign_seconds": maximum,
    }


def _authorization(campaign: dict[str, Any], budget_usd: float) -> dict[str, Any]:
    return {
        "schema_version": "sloforge.branchfabric.modal-campaign-authorization/v1",
        "campaign_sha256": hashlib.sha256(_canonical_bytes(campaign)).hexdigest(),
        "maximum_gpu_seconds": GPU_FUNCTION_TIMEOUT_SECONDS,
        "budget_usd": budget_usd,
    }


def _validate_authorization(campaign: dict[str, Any], payload: object) -> None:
    if not isinstance(payload, dict) or payload != _authorization(
        campaign, float(payload.get("budget_usd", 0.0))
    ):
        raise RuntimeError("Modal campaign lacks an exact coordinator authorization")
    if not math.isfinite(float(payload["budget_usd"])) or float(payload["budget_usd"]) <= 0:
        raise RuntimeError("Modal campaign budget authorization is invalid")


def _publish_campaign(campaign_id: str, local_root: Path) -> tuple[str, str]:
    remote_prefix = f"{REMOTE_EXPERIMENT_PREFIX}/{campaign_id}"
    final = RESULTS_MOUNT / remote_prefix
    staging = RESULTS_MOUNT / f"{remote_prefix}.staging"
    if final.exists() or staging.exists():
        raise FileExistsError(f"immutable campaign prefix exists: {remote_prefix}")
    staging.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(local_root, staging)
    manifest = {
        "schema_version": "sloforge.branchfabric.modal-campaign-result/v1",
        "campaign_id": campaign_id,
        "remote_prefix": remote_prefix,
        "artifacts": _inventory(staging),
    }
    _write_json(staging / "REMOTE_MANIFEST.json", manifest)
    os.replace(staging, final)
    results_volume.commit()
    return remote_prefix, _sha256(final / "REMOTE_MANIFEST.json")


def run_gpu_campaign(
    campaign_payload: dict[str, Any], authorization_payload: dict[str, Any]
) -> dict[str, Any]:
    """Run every fresh child serially inside one physical A100 Function."""

    campaign = validate_campaign(campaign_payload)
    _validate_authorization(campaign, authorization_payload)
    call_id = modal.current_function_call_id()
    started_ns = time.monotonic_ns()
    started_utc = _utc_now()
    root = Path("/tmp/sloforge-branchfabric-003") / campaign["campaign_id"]
    root.mkdir(parents=True, exist_ok=False)
    snapshot = MODEL_MOUNT / "snapshots" / MODEL_REVISION
    status = "failed"
    error: dict[str, str] | None = None
    records: list[dict[str, Any]] = []
    gpu_identity: dict[str, Any] | None = None
    immutable_threshold_mib: int | None = None
    try:
        _validate_model_manifest(snapshot)
        from gpu_cow_controller import run_controller, sample_driver_gpu
        from gpu_validation_campaign import (
            _assert_controller_result,
            _initial_campaign_gate,
            _post_child_gate,
            _pre_child_gate,
        )

        baseline = _initial_campaign_gate(
            sampler=sample_driver_gpu,
            interval_seconds=0.2,
            allowance_mib=1024,
            sleep=time.sleep,
        )
        _write_json(root / "campaign-hbm-baseline.json", baseline)
        gpu_identity = cast(dict[str, Any], baseline["gpu_identity"])
        immutable_threshold_mib = int(baseline["cleanup_threshold_mib"])
        worker = Path("/opt/sloforge/experiments/branchfabric/gpu_cow_worker.py")
        for position, trial in enumerate(campaign["trials"]):
            if (time.monotonic_ns() - started_ns) / 1e9 >= campaign["maximum_campaign_seconds"]:
                raise TimeoutError("Experiment 003 campaign wall-time bound reached")
            trial_root = root / "trials" / f"{position:02d}-{trial['attempt_id']}"
            record: dict[str, Any] = {
                "position": position,
                "attempt_id": trial["attempt_id"],
                "config": trial,
                "status": "running",
                "pre_child_gate": None,
                "controller": None,
                "campaign_post_child_gate": None,
                "error": None,
            }
            try:
                record["pre_child_gate"] = _pre_child_gate(
                    sampler=sample_driver_gpu,
                    expected_identity=gpu_identity,
                    threshold_mib=immutable_threshold_mib,
                    interval_seconds=0.2,
                    sleep=time.sleep,
                )
                result = run_controller(
                    config_payload=trial,
                    work_root=trial_root,
                    worker_path=worker,
                    model_snapshot=snapshot,
                    gpu_sampler=sample_driver_gpu,
                    postflight_timeout_s=60.0,
                    baseline_interval_s=0.2,
                    monitor_interval_s=1.0,
                )
                record["controller"] = result
                _assert_controller_result(result, expected_identity=gpu_identity)
                post = _post_child_gate(
                    sampler=sample_driver_gpu,
                    expected_identity=gpu_identity,
                    threshold_mib=immutable_threshold_mib,
                    interval_seconds=1.0,
                    timeout_seconds=60.0,
                    maximum_samples=63,
                    sleep=time.sleep,
                )
                record["campaign_post_child_gate"] = post
                if not post["passed"]:
                    raise RuntimeError("immutable campaign HBM cleanup gate failed; matrix aborted")
                record["status"] = "succeeded"
            except Exception as trial_error:
                record["status"] = "failed"
                record["error"] = {
                    "type": type(trial_error).__name__,
                    "message": str(trial_error),
                }
                records.append(record)
                _write_json(root / "trial-records" / f"{position:02d}.json", record)
                raise
            records.append(record)
            _write_json(root / "trial-records" / f"{position:02d}.json", record)
        status = "succeeded"
    except Exception as caught:
        error = {"type": type(caught).__name__, "message": str(caught)}
    ended_ns = time.monotonic_ns()
    manifest = {
        "schema_version": "sloforge.branchfabric.modal-campaign-manifest-003/v1",
        "campaign_id": campaign["campaign_id"],
        "status": status,
        "function_call_id": call_id,
        "requested_gpu": "A100-80GB",
        "gpu_identity": gpu_identity,
        "same_gpu_enforced_before_every_child": status == "succeeded",
        "immutable_cleanup_threshold_mib": immutable_threshold_mib,
        "planned_children": len(campaign["trials"]),
        "started_children": len(records),
        "completed_children": sum(record.get("status") == "succeeded" for record in records),
        "records": records,
        "function_started_at_utc": started_utc,
        "function_ended_at_utc": _utc_now(),
        "function_duration_seconds": (ended_ns - started_ns) / 1e9,
        "gpu_allocation_lower_bound_seconds": (ended_ns - started_ns) / 1e9,
        "error": error,
    }
    _write_json(root / "campaign-manifest.json", manifest)
    remote_prefix, remote_sha = _publish_campaign(campaign["campaign_id"], root)
    return {**manifest, "remote_prefix": remote_prefix, "remote_manifest_sha256": remote_sha}


def _budget() -> float:
    raw = os.getenv("SLOFORGE_GPU_BUDGET_USD")
    if raw is None:
        raise RuntimeError("SLOFORGE_GPU_BUDGET_USD is required before Modal hydration")
    value = float(raw)
    proposed_hours = GPU_FUNCTION_TIMEOUT_SECONDS / 3600
    proposed_cost = proposed_hours * MODAL_A100_80GB_USD_PER_HOUR
    proposed_cost += GPU_FUNCTION_TIMEOUT_SECONDS * (
        GPU_FUNCTION_CPU_CORES * CPU_USD_PER_CORE_SECOND
        + GPU_FUNCTION_MEMORY_GIB * MEMORY_USD_PER_GIB_SECOND
    )
    if (
        not math.isfinite(value)
        or value <= 0
        or proposed_hours > GPU_HOUR_HARD_LIMIT
        or proposed_cost > value * (1 - BUDGET_RESERVE_FRACTION)
    ):
        raise RuntimeError("Experiment 003 Modal reservation exceeds its authorized budget")
    return value


def _new_gpu_ledger() -> dict[str, Any]:
    return {
        "schema_version": "sloforge.branchfabric.gpu-hours/v1",
        "experiment": "experiment-003",
        "historical_experiment_002_gpu_hours": HISTORICAL_EXPERIMENT_002_GPU_HOURS,
        "target_additional_gpu_hours": GPU_HOUR_TARGET,
        "hard_additional_gpu_hours": GPU_HOUR_HARD_LIMIT,
        "consumed_gpu_hours": 0.0,
        "gpu_active_intervals": [],
        "modal_in_flight_reservations": [],
    }


def _load_gpu_ledger(path: Path) -> dict[str, Any]:
    ledger = (
        cast(dict[str, Any], json.loads(path.read_text())) if path.is_file() else _new_gpu_ledger()
    )
    if (
        ledger.get("schema_version") != "sloforge.branchfabric.gpu-hours/v1"
        or ledger.get("experiment") != "experiment-003"
        or ledger.get("hard_additional_gpu_hours") != GPU_HOUR_HARD_LIMIT
        or not isinstance(ledger.get("gpu_active_intervals"), list)
        or not isinstance(ledger.get("modal_in_flight_reservations"), list)
    ):
        raise ValueError("Experiment 003 GPU-hour ledger is malformed or has stale limits")
    consumed = ledger.get("consumed_gpu_hours")
    if (
        not isinstance(consumed, (int, float))
        or isinstance(consumed, bool)
        or not math.isfinite(float(consumed))
        or not 0 <= float(consumed) <= GPU_HOUR_HARD_LIMIT
    ):
        raise ValueError("Experiment 003 GPU-hour ledger consumed time is invalid")
    intervals = cast(list[object], ledger["gpu_active_intervals"])
    interval_hours = 0.0
    for item in intervals:
        if not isinstance(item, dict):
            raise ValueError("Experiment 003 GPU-hour interval is malformed")
        seconds = item.get("model_load_benchmark_cleanup_seconds")
        if (
            not isinstance(seconds, (int, float))
            or isinstance(seconds, bool)
            or not math.isfinite(float(seconds))
            or float(seconds) < 0
        ):
            raise ValueError("Experiment 003 GPU-hour interval duration is invalid")
        interval_hours += float(seconds) / 3600
    if abs(interval_hours - float(consumed)) > 1e-9:
        raise ValueError("Experiment 003 consumed GPU hours disagree with raw intervals")
    return ledger


def _ledger_compute_cost_usd(ledger: dict[str, Any]) -> float:
    total = 0.0
    for item in cast(list[dict[str, Any]], ledger["gpu_active_intervals"]):
        seconds = float(item["model_load_benchmark_cleanup_seconds"])
        total += seconds / 3600 * MODAL_A100_80GB_USD_PER_HOUR
        total += seconds * (
            GPU_FUNCTION_CPU_CORES * CPU_USD_PER_CORE_SECOND
            + GPU_FUNCTION_MEMORY_GIB * MEMORY_USD_PER_GIB_SECOND
        )
    return total


def _reserve_campaign(
    ledger_path: Path,
    *,
    campaign_id: str,
    budget_usd: float,
) -> tuple[dict[str, Any], str]:
    ledger = _load_gpu_ledger(ledger_path)
    reservations = cast(list[dict[str, Any]], ledger["modal_in_flight_reservations"])
    if reservations:
        raise RuntimeError("Experiment 003 has an unresolved Modal reservation")
    consumed = float(ledger["consumed_gpu_hours"])
    if consumed + CAMPAIGN_RESERVATION_GPU_HOURS > GPU_HOUR_HARD_LIMIT:
        raise RuntimeError("Experiment 003 reservation exceeds the A100-hour hard limit")
    support_per_second = (
        GPU_FUNCTION_CPU_CORES * CPU_USD_PER_CORE_SECOND
        + GPU_FUNCTION_MEMORY_GIB * MEMORY_USD_PER_GIB_SECOND
    )
    maximum_call_cost = (
        CAMPAIGN_RESERVATION_GPU_HOURS * MODAL_A100_80GB_USD_PER_HOUR
        + GPU_FUNCTION_TIMEOUT_SECONDS * support_per_second
    )
    if _ledger_compute_cost_usd(ledger) + maximum_call_cost > budget_usd * (
        1 - BUDGET_RESERVE_FRACTION
    ):
        raise RuntimeError("Experiment 003 cumulative Modal cost exceeds the USD budget")
    reservation_id = f"exp003-reservation-{uuid.uuid4().hex}"
    reservations.append(
        {
            "reservation_id": reservation_id,
            "campaign_id": campaign_id,
            "maximum_gpu_hours": CAMPAIGN_RESERVATION_GPU_HOURS,
            "maximum_total_cost_usd": maximum_call_cost,
            "created_at_utc": _utc_now(),
            "function_call_id": None,
        }
    )
    _replace_local_json(ledger_path, ledger)
    return ledger, reservation_id


def _attach_function_call(ledger_path: Path, *, reservation_id: str, call_id: str) -> None:
    ledger = _load_gpu_ledger(ledger_path)
    reservations = cast(list[dict[str, Any]], ledger["modal_in_flight_reservations"])
    matches = [item for item in reservations if item.get("reservation_id") == reservation_id]
    if len(matches) != 1 or matches[0].get("function_call_id") is not None:
        raise RuntimeError("Experiment 003 reservation is absent, duplicated, or attached")
    matches[0]["function_call_id"] = call_id
    matches[0]["attached_at_utc"] = _utc_now()
    _replace_local_json(ledger_path, ledger)


def _annotate_reservation_failure(
    ledger_path: Path,
    *,
    reservation_id: str,
    stage: str,
    error: BaseException,
) -> None:
    ledger = _load_gpu_ledger(ledger_path)
    reservations = cast(list[dict[str, Any]], ledger["modal_in_flight_reservations"])
    matches = [item for item in reservations if item.get("reservation_id") == reservation_id]
    if len(matches) != 1:
        raise RuntimeError("Experiment 003 failed reservation is absent or duplicated")
    matches[0]["last_client_failure"] = {
        "stage": stage,
        "type": type(error).__name__,
        "message": str(error),
        "observed_at_utc": _utc_now(),
        "reservation_retained_pending_provider_audit": True,
    }
    _replace_local_json(ledger_path, ledger)


def _record_terminal_campaign(
    ledger_path: Path,
    *,
    reservation_id: str,
    campaign_id: str,
    call_id: str,
    gpu_seconds: float,
    local_call_elapsed_seconds: float,
    status: str,
) -> dict[str, Any]:
    if (
        not math.isfinite(gpu_seconds)
        or gpu_seconds < 0
        or gpu_seconds > GPU_FUNCTION_TIMEOUT_SECONDS
    ):
        raise ValueError("Experiment 003 terminal GPU duration is invalid")
    ledger = _load_gpu_ledger(ledger_path)
    reservations = cast(list[dict[str, Any]], ledger["modal_in_flight_reservations"])
    matches = [item for item in reservations if item.get("reservation_id") == reservation_id]
    if (
        len(matches) != 1
        or matches[0].get("campaign_id") != campaign_id
        or matches[0].get("function_call_id") != call_id
    ):
        raise RuntimeError("Experiment 003 terminal result does not match its reservation")
    new_consumed = float(ledger["consumed_gpu_hours"]) + gpu_seconds / 3600
    if new_consumed > GPU_HOUR_HARD_LIMIT + 1e-12:
        raise RuntimeError("Experiment 003 terminal result exceeds the hard GPU-hour limit")
    support_cost = gpu_seconds * (
        GPU_FUNCTION_CPU_CORES * CPU_USD_PER_CORE_SECOND
        + GPU_FUNCTION_MEMORY_GIB * MEMORY_USD_PER_GIB_SECOND
    )
    gpu_cost = gpu_seconds / 3600 * MODAL_A100_80GB_USD_PER_HOUR
    cast(list[dict[str, Any]], ledger["gpu_active_intervals"]).append(
        {
            "campaign_id": campaign_id,
            "reservation_id": reservation_id,
            "function_call_id": call_id,
            "gpu": "A100-80GB",
            "gpu_count": 1,
            "model_load_benchmark_cleanup_seconds": gpu_seconds,
            "local_call_elapsed_seconds": local_call_elapsed_seconds,
            "gpu_cost_usd": gpu_cost,
            "support_cost_usd": support_cost,
            "cost_usd": gpu_cost + support_cost,
            "status": status,
        }
    )
    ledger["consumed_gpu_hours"] = new_consumed
    ledger["modal_in_flight_reservations"] = [
        item for item in reservations if item.get("reservation_id") != reservation_id
    ]
    ledger["updated_at_utc"] = _utc_now()
    _replace_local_json(ledger_path, ledger)
    return ledger


def _materialize(campaign_id: str, expected_sha: str) -> Path:
    prefix = PurePosixPath(f"{REMOTE_EXPERIMENT_PREFIX}/{campaign_id}")
    target = LOCAL_EXPERIMENT_ROOT / "modal" / campaign_id
    staging = target.with_name(f".{campaign_id}.{uuid.uuid4().hex}.staging")
    if target.exists():
        raise FileExistsError(f"immutable local campaign already exists: {target}")
    staging.mkdir(parents=True, exist_ok=False)
    for entry in results_volume.iterdir(prefix.as_posix(), recursive=True):
        remote_path = PurePosixPath(entry.path.lstrip("/"))
        relative = remote_path.relative_to(prefix)
        local = staging / relative.as_posix()
        kind = getattr(entry.type, "name", str(entry.type))
        if kind == "DIRECTORY":
            local.mkdir(parents=True, exist_ok=True)
        elif kind == "FILE":
            local.parent.mkdir(parents=True, exist_ok=True)
            with local.open("xb") as handle:
                for chunk in results_volume.read_file(entry.path):
                    handle.write(chunk)
        else:
            raise RuntimeError(f"unsupported Modal Volume entry: {kind}")
    if _sha256(staging / "REMOTE_MANIFEST.json") != expected_sha:
        raise RuntimeError("materialized Experiment 003 campaign manifest hash disagrees")
    target.parent.mkdir(parents=True, exist_ok=True)
    os.replace(staging, target)
    return target


def main(campaign_path: str) -> None:
    if _run_campaign_function is None:
        raise RuntimeError("Modal campaign function is unavailable without budget preflight")
    budget = _budget()
    campaign = validate_campaign(json.loads(Path(campaign_path).resolve(strict=True).read_text()))
    LOCAL_EXPERIMENT_ROOT.mkdir(parents=True, exist_ok=True)
    ledger_path = LOCAL_EXPERIMENT_ROOT / "gpu-hours.json"
    _, reservation_id = _reserve_campaign(
        ledger_path,
        campaign_id=campaign["campaign_id"],
        budget_usd=budget,
    )
    started = time.monotonic()
    # A durable FunctionCall ID is required to distinguish a terminal failure
    # from a disconnected client while a paid GPU task remains active. Exactly
    # one call is spawned and synchronously awaited; no campaign overlap exists.
    try:
        gpu_call = _run_campaign_function.spawn(campaign, _authorization(campaign, budget))
    except Exception as spawn_error:
        _annotate_reservation_failure(
            ledger_path,
            reservation_id=reservation_id,
            stage="spawn_outcome_ambiguous",
            error=spawn_error,
        )
        raise
    call_id = str(gpu_call.object_id)
    try:
        _attach_function_call(ledger_path, reservation_id=reservation_id, call_id=call_id)
    except Exception as attach_error:
        _annotate_reservation_failure(
            ledger_path,
            reservation_id=reservation_id,
            stage="function_call_attachment_failed",
            error=attach_error,
        )
        raise
    try:
        result = gpu_call.get()
    except Exception as get_error:
        _annotate_reservation_failure(
            ledger_path,
            reservation_id=reservation_id,
            stage="function_call_get_failed_call_may_be_active",
            error=get_error,
        )
        raise
    elapsed = time.monotonic() - started
    if not isinstance(result, dict) or result.get("function_call_id") != call_id:
        # The call is terminal because get() returned, but its duration cannot
        # be trusted. Conservatively charge the full reservation before failing.
        _record_terminal_campaign(
            ledger_path,
            reservation_id=reservation_id,
            campaign_id=campaign["campaign_id"],
            call_id=call_id,
            gpu_seconds=float(GPU_FUNCTION_TIMEOUT_SECONDS),
            local_call_elapsed_seconds=elapsed,
            status="invalid_terminal_result_charged_at_reservation_maximum",
        )
        raise RuntimeError("Experiment 003 Modal function returned an invalid terminal result")
    try:
        gpu_seconds = float(result["gpu_allocation_lower_bound_seconds"])
        remote_sha = str(result["remote_manifest_sha256"])
        terminal_status = str(result["status"])
        if (
            not math.isfinite(gpu_seconds)
            or not 0 <= gpu_seconds <= GPU_FUNCTION_TIMEOUT_SECONDS
            or len(remote_sha) != 64
            or any(character not in "0123456789abcdef" for character in remote_sha)
            or terminal_status not in {"succeeded", "failed"}
        ):
            raise ValueError("terminal result fields are outside their bounds")
    except (KeyError, TypeError, ValueError) as error:
        _record_terminal_campaign(
            ledger_path,
            reservation_id=reservation_id,
            campaign_id=campaign["campaign_id"],
            call_id=call_id,
            gpu_seconds=float(GPU_FUNCTION_TIMEOUT_SECONDS),
            local_call_elapsed_seconds=elapsed,
            status="invalid_terminal_result_charged_at_reservation_maximum",
        )
        raise RuntimeError("Experiment 003 Modal function returned invalid fields") from error
    ledger = _record_terminal_campaign(
        ledger_path,
        reservation_id=reservation_id,
        campaign_id=campaign["campaign_id"],
        call_id=call_id,
        gpu_seconds=gpu_seconds,
        local_call_elapsed_seconds=elapsed,
        status=terminal_status,
    )
    local = _materialize(campaign["campaign_id"], remote_sha)
    if terminal_status != "succeeded":
        raise RuntimeError(f"Experiment 003 Modal campaign failed; artifacts at {local}")
    print(json.dumps({"result": result, "local": str(local), "ledger": ledger}, sort_keys=True))


if _CLOUD_GRAPH_ENABLED:
    _run_campaign_function = app.function(
        image=gpu_image,
        gpu="A100-80GB",
        volumes={
            str(MODEL_MOUNT): model_volume.with_mount_options(read_only=True),
            str(RESULTS_MOUNT): results_volume,
        },
        cpu=GPU_FUNCTION_CPU_CORES,
        memory=int(GPU_FUNCTION_MEMORY_GIB * 1024),
        timeout=GPU_FUNCTION_TIMEOUT_SECONDS,
        startup_timeout=GPU_FUNCTION_STARTUP_TIMEOUT_SECONDS,
        retries=0,
        max_containers=1,
        buffer_containers=0,
        single_use_containers=True,
    )(run_gpu_campaign)
    _modal_local_entrypoint = app.local_entrypoint()(main)


__all__ = [
    "APP_NAME",
    "GPU_HOUR_HARD_LIMIT",
    "GPU_HOUR_TARGET",
    "app",
    "main",
    "run_gpu_campaign",
    "validate_campaign",
]
