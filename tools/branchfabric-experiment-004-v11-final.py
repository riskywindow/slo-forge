#!/usr/bin/env python3
"""Sole paid-resource coordinator for Experiment 004 integrated v11 attempt c.

This program is deliberately narrower than the historical Experiment 004
launcher.  It admits one content-addressed integrated-v11 config, owns the
only ledger reservation, launches the fail-closed Modal surface once, verifies
the downloaded immutable bundle, and clears the reservation exactly once.
Provider cleanup is a separate post-return gate: this coordinator emits the
identities needed by that audit and never claims that provider cleanup passed.
"""

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
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, cast

from sloforge.helix.characterization.gpu_reclamation_methodology import (
    ArtifactSampleRef,
    Experiment004GpuHourLedger,
    charge_failed_gpu_reservation,
    reserve_gpu_invocation,
    settle_gpu_invocation,
)
from sloforge.helix.characterization.gpu_reclamation_v11_methodology import (
    V11_INTEGRATED_RESERVATION_WALL_SECONDS,
    Experiment004V11IntegratedConfig,
    canonical_json_bytes,
)

_ROOT = Path(__file__).resolve().parents[1]
_EXPERIMENT_ROOT = _ROOT / "artifacts/branchfabric/gpu-validation/experiment-004"
_LEDGER = _EXPERIMENT_ROOT / "gpu-hours.json"
_LOCK = _EXPERIMENT_ROOT / "gpu-hours.lock"
_APP = _ROOT / "experiments/branchfabric/modal_gpu_reclamation_integrated_v11.py"
_LOCAL_RESULT_PARENT = _EXPERIMENT_ROOT / "v11-final/integrated/raw"
_FAILURE_ROOT = _EXPERIMENT_ROOT / "v11-final/integrated/failures"
_PROVIDER_INPUTS = _EXPERIMENT_ROOT / "v11-final/integrated/provider-cleanup-audit-inputs.json"

ATTEMPT_ID = "exp004-v11-integrated-s41-c"
APP_NAME = "sloforge-branchfabric-gpu-reclamation-004-v11-integrated"
RESULTS_VOLUME = "sloforge-branchfabric-results"
MODEL_VOLUME = "sloforge-model-cache"
REMOTE_PREFIX = "experiment-004/v11/integrated/modal"
GPU_COUNT = 2
FUNCTION_WALL_SECONDS = V11_INTEGRATED_RESERVATION_WALL_SECONDS
FUNCTION_STARTUP_SECONDS = 180.0
POST_CONTROLLER_RESERVE_SECONDS = 10.0
COORDINATOR_PROCESS_GRACE_SECONDS = 60.0
GPU_PRICE_PER_HOUR_USD = 2.4984
CPU_PRICE_PER_CORE_SECOND_USD = 0.0000131
MEMORY_PRICE_PER_GIB_SECOND_USD = 0.00000222
MAXIMUM_ESTIMATED_COST_USD = FUNCTION_WALL_SECONDS * (
    GPU_COUNT * GPU_PRICE_PER_HOUR_USD / 3600.0
    + 16.0 * CPU_PRICE_PER_CORE_SECOND_USD
    + 64.0 * MEMORY_PRICE_PER_GIB_SECOND_USD
)
_RESULT_FIELDS = frozenset(
    {
        "schema_version",
        "status",
        "scientific_status",
        "attempt_id",
        "reservation_id",
        "reservation_commitment_sha256",
        "config_sha256",
        "function_call_id",
        "requested_gpu",
        "gpu_count",
        "controller_and_analysis_interval_seconds",
        "gpu_allocation_seconds_status",
        "hardware_comparability",
        "bound_artifacts",
        "absolute_deadlines",
        "controller",
        "in_function_cleanup",
        "run_error",
        "completed_at_utc",
        "gpu_allocation_seconds",
        "gpu_seconds",
        "gpu_hours",
        "remote_prefix",
        "remote_manifest_sha256",
    }
)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n"
    ).encode()


def _ledger_bytes(ledger: Experiment004GpuHourLedger) -> bytes:
    """Match the ledger mounted by the Modal launcher byte-for-byte."""

    return (
        json.dumps(
            ledger.model_dump(mode="json"),
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
        )
        + "\n"
    ).encode()


def _atomic_write_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{secrets.token_hex(4)}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_new_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(_canonical_bytes(payload))
        handle.flush()
        os.fsync(handle.fileno())


def _load_ledger(path: Path) -> Experiment004GpuHourLedger:
    return Experiment004GpuHourLedger.model_validate_json(
        path.resolve(strict=True).read_text(), strict=True
    )


def _default_seal_verifier(
    config: Experiment004V11IntegratedConfig, repository_root: Path
) -> dict[str, Any]:
    experiment_modules = repository_root / "experiments/branchfabric"
    sys.path.insert(0, str(experiment_modules))
    try:
        from gpu_reclamation_integrated_controller_v11 import (  # type: ignore[import-not-found]
            verify_sealed_evidence,
        )

        return cast(dict[str, Any], verify_sealed_evidence(config, repository_root))
    finally:
        sys.path.remove(str(experiment_modules))


def _load_sealed_config(
    config_path: Path,
    *,
    repository_root: Path,
    ledger_path: Path,
    seal_verifier: Callable[
        [Experiment004V11IntegratedConfig, Path], dict[str, Any]
    ] = _default_seal_verifier,
) -> tuple[Experiment004V11IntegratedConfig, dict[str, Any]]:
    if config_path.is_symlink():
        raise ValueError("integrated v11 config must not be a symlink")
    resolved_root = repository_root.resolve(strict=True)
    resolved = config_path.resolve(strict=True)
    if resolved != resolved_root and resolved_root not in resolved.parents:
        raise ValueError("integrated v11 config must be a regular file inside the repository")
    config = Experiment004V11IntegratedConfig.model_validate_json(resolved.read_text(), strict=True)
    if config.attempt_id != ATTEMPT_ID:
        raise ValueError("sole coordinator admits only the fresh integrated attempt c")
    if config.gpu_count != GPU_COUNT or not math.isclose(
        config.maximum_wall_seconds,
        FUNCTION_WALL_SECONDS,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise ValueError("integrated v11 config must bind exactly 2xA100 for 588 seconds")
    current_ledger_sha256 = _sha256_file(ledger_path.resolve(strict=True))
    if current_ledger_sha256 != config.ledger_sha256_before_reservation:
        raise RuntimeError("sealed config does not bind the current pre-reservation ledger")
    verified = seal_verifier(config, resolved_root)
    if not isinstance(verified, dict) or verified.get("status") != "PASS":
        raise RuntimeError("integrated v11 content-addressed evidence gate is not PASS")
    return config, verified


def _authorized_budget_usd(config: Experiment004V11IntegratedConfig, root: Path) -> float:
    path = (root / config.budget_authorization).resolve(strict=True)
    if _sha256_file(path) != config.budget_authorization_sha256:
        raise RuntimeError("integrated v11 budget authorization hash drifted")
    payload = json.loads(path.read_text())
    value = payload.get("authorized_gpu_budget_usd") if isinstance(payload, dict) else None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError("integrated v11 budget authorization omits a numeric USD limit")
    authorized = float(value)
    if not math.isfinite(authorized) or authorized < MAXIMUM_ESTIMATED_COST_USD:
        raise RuntimeError("integrated v11 budget authorization is below the bounded run cost")
    return authorized


def _require_launch_budget(config: Experiment004V11IntegratedConfig, root: Path) -> float:
    raw = os.getenv("SLOFORGE_GPU_BUDGET_USD")
    try:
        value = float(raw) if raw is not None else math.nan
    except ValueError as error:
        raise RuntimeError("SLOFORGE_GPU_BUDGET_USD must be finite and positive") from error
    authorized = _authorized_budget_usd(config, root)
    if not math.isfinite(value) or value < MAXIMUM_ESTIMATED_COST_USD or value > authorized:
        raise RuntimeError(
            "SLOFORGE_GPU_BUDGET_USD must cover the bounded run without exceeding authorization"
        )
    return value


def _reserve_locked(
    config: Experiment004V11IntegratedConfig,
    *,
    ledger_path: Path,
    reservation_id: str,
) -> tuple[Experiment004GpuHourLedger, dict[str, Any], str]:
    original_bytes = ledger_path.resolve(strict=True).read_bytes()
    if _sha256_bytes(original_bytes) != config.ledger_sha256_before_reservation:
        raise RuntimeError("ledger changed between sealed-config validation and reservation")
    ledger = Experiment004GpuHourLedger.model_validate_json(original_bytes, strict=True)
    config_sha256 = _sha256_bytes(canonical_json_bytes(config))
    updated, preflight = reserve_gpu_invocation(
        ledger,
        reservation_id=reservation_id,
        invocation_id=config.attempt_id,
        maximum_wall_seconds=FUNCTION_WALL_SECONDS,
        config_sha256=config_sha256,
        gpu_count=2,
    )
    if not math.isclose(preflight.proposed_maximum_gpu_seconds, 1176.0, rel_tol=0.0, abs_tol=1e-9):
        raise RuntimeError("integrated v11 reservation is not exactly 1,176 A100-seconds")
    _atomic_write_bytes(ledger_path, _ledger_bytes(updated))
    reservation = updated.reservations[0]
    commitment = _sha256_bytes(canonical_json_bytes(reservation))
    return updated, preflight.model_dump(mode="json"), commitment


def _invoke_modal(
    config_path: Path,
    *,
    reservation_id: str,
    budget_usd: float,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> tuple[subprocess.CompletedProcess[str], float]:
    launch_token = secrets.token_hex(32)
    environment = dict(os.environ)
    environment.update(
        {
            "SLOFORGE_GPU_BUDGET_USD": format(budget_usd, ".17g"),
            "SLOFORGE_EXP004_RESERVATION_ID": reservation_id,
            "SLOFORGE_MODAL_PREFLIGHT_TOKEN": launch_token,
        }
    )
    command = ["modal", "run", str(_APP), "--config-path", str(config_path)]
    started = time.monotonic()
    completed = runner(
        command,
        check=False,
        capture_output=True,
        text=True,
        # Modal image hydration/queueing happens before the bounded Function
        # allocation interval.  Keep the paid remote bound at 588 seconds,
        # but do not kill the local coordinator while Modal is still within
        # its separately bounded 180-second startup allowance.
        timeout=(
            FUNCTION_STARTUP_SECONDS + FUNCTION_WALL_SECONDS + COORDINATOR_PROCESS_GRACE_SECONDS
        ),
        env=environment,
    )
    return completed, time.monotonic() - started


def _parse_exact_result(stdout: str) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    for line in stdout.splitlines():
        if not line.lstrip().startswith("{"):
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and set(payload) == {"result", "materialized"}:
            candidates.append(payload)
    if len(candidates) != 1:
        raise ValueError(
            f"Modal output contained {len(candidates)} exact integrated result envelopes"
        )
    return candidates[0]


def _bounded_number(value: Any, *, upper: float, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or not 0.0 < float(value) <= upper
    ):
        raise ValueError(f"{label} is outside its paid bound")
    return float(value)


def _exact_inventory(remote: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    controller = remote.get("controller")
    if not isinstance(controller, dict):
        raise ValueError("integrated result omits controller evidence")
    before = controller.get("inventory_before")
    after = controller.get("inventory_after")
    if not isinstance(before, list) or not isinstance(after, list) or len(before) != GPU_COUNT:
        raise ValueError("integrated result omits exactly two GPU inventory rows")
    stable_identity_fields = (
        "index",
        "uuid",
        "name",
        "driver_version",
        "memory_total_mib",
    )
    if len(after) != GPU_COUNT or any(
        not isinstance(before_row, dict)
        or not isinstance(after_row, dict)
        or tuple(before_row.get(field) for field in stable_identity_fields)
        != tuple(after_row.get(field) for field in stable_identity_fields)
        for before_row, after_row in zip(before, after, strict=True)
    ):
        raise ValueError("integrated GPU identity changed during the invocation")
    rows: list[dict[str, Any]] = []
    for row in before:
        if not isinstance(row, dict):
            raise ValueError("integrated GPU inventory row is malformed")
        name = str(row.get("name", ""))
        uuid = str(row.get("uuid", ""))
        if (
            "A100" not in name
            or "80GB" not in name.replace(" ", "")
            or not uuid.startswith("GPU-")
            or int(row.get("memory_total_mib", 0)) < 79_000
        ):
            raise ValueError("integrated result did not use an A100-80GB device")
        rows.append(dict(row))
    if len({str(row["uuid"]) for row in rows}) != GPU_COUNT:
        raise ValueError("integrated result repeats a physical GPU UUID")
    return rows[0], rows[1]


def _validate_remote_result(
    envelope: Mapping[str, Any],
    *,
    config: Experiment004V11IntegratedConfig,
    reservation_id: str,
    reservation_commitment_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any], tuple[dict[str, Any], dict[str, Any]]]:
    if set(envelope) != {"result", "materialized"}:
        raise ValueError("integrated result envelope differs from its exact schema")
    remote = envelope["result"]
    materialized = envelope["materialized"]
    if not isinstance(remote, dict) or set(remote) != _RESULT_FIELDS:
        raise ValueError("integrated remote completion differs from its exact schema")
    if not isinstance(materialized, dict) or set(materialized) != {"remote_path", "volume_name"}:
        raise ValueError("integrated materialization identity differs from its exact schema")

    config_sha256 = _sha256_bytes(canonical_json_bytes(config))
    expected_remote = f"{REMOTE_PREFIX}/{config.attempt_id}"
    allocation = _bounded_number(
        remote["gpu_allocation_seconds"], upper=FUNCTION_WALL_SECONDS, label="allocation time"
    )
    gpu_seconds = _bounded_number(
        remote["gpu_seconds"], upper=GPU_COUNT * FUNCTION_WALL_SECONDS, label="GPU time"
    )
    gpu_hours = _bounded_number(
        remote["gpu_hours"],
        upper=GPU_COUNT * FUNCTION_WALL_SECONDS / 3600.0,
        label="GPU hours",
    )
    interval = _bounded_number(
        remote["controller_and_analysis_interval_seconds"],
        upper=FUNCTION_WALL_SECONDS,
        label="controller interval",
    )
    deadlines = remote.get("absolute_deadlines")
    deadline_fields = {
        "function_entry_monotonic_ns",
        "controller_deadline_monotonic_ns",
        "function_deadline_monotonic_ns",
        "post_controller_reserve_seconds",
    }
    if not isinstance(deadlines, dict) or set(deadlines) != deadline_fields:
        raise ValueError("integrated absolute deadlines differ from their exact schema")
    entry = deadlines["function_entry_monotonic_ns"]
    controller_deadline = deadlines["controller_deadline_monotonic_ns"]
    function_deadline = deadlines["function_deadline_monotonic_ns"]
    if (
        any(
            isinstance(item, bool) or not isinstance(item, int)
            for item in (entry, controller_deadline, function_deadline)
        )
        or controller_deadline - entry
        != round((FUNCTION_WALL_SECONDS - POST_CONTROLLER_RESERVE_SECONDS) * 1e9)
        or function_deadline - entry != round(FUNCTION_WALL_SECONDS * 1e9)
        or deadlines["post_controller_reserve_seconds"] != POST_CONTROLLER_RESERVE_SECONDS
    ):
        raise ValueError("integrated absolute deadlines are invalid")

    status = remote["status"]
    run_error = remote["run_error"]
    cleanup = remote["in_function_cleanup"]
    controller = remote["controller"]
    provisional = (
        status == "provisional"
        and remote["scientific_status"] == "pending-local-budget-settlement-and-provider-cleanup"
        and run_error is None
        and isinstance(controller, dict)
        and controller.get("status") == "succeeded"
        and isinstance(cleanup, dict)
        and cleanup.get("pass") is True
        and controller.get("in_function_cleanup") == cleanup
    )
    failed = (
        status == "failed"
        and remote["scientific_status"] == "invalid"
        and isinstance(run_error, dict)
        and set(run_error) == {"type", "message", "traceback"}
        and all(isinstance(run_error[key], str) and run_error[key] for key in run_error)
    )
    expected_bound = {
        "offline_gate_manifest": config.offline_gate_manifest,
        "offline_gate_manifest_sha256": config.offline_gate_manifest_sha256,
        "micro_validation_artifact": config.micro_validation_artifact,
        "micro_validation_sha256": config.micro_validation_sha256,
        "post_micro_review_manifest": config.post_micro_review_manifest,
        "post_micro_review_manifest_sha256": config.post_micro_review_manifest_sha256,
        "budget_authorization": config.budget_authorization,
        "budget_authorization_sha256": config.budget_authorization_sha256,
    }
    manifest_hash = remote["remote_manifest_sha256"]
    function_call_id = remote["function_call_id"]
    if (
        not (provisional or failed)
        or remote["schema_version"]
        != "sloforge.branchfabric.experiment-004-v11-integrated-completion/v1"
        or remote["attempt_id"] != config.attempt_id
        or remote["reservation_id"] != reservation_id
        or remote["reservation_commitment_sha256"] != reservation_commitment_sha256
        or remote["config_sha256"] != config_sha256
        or not isinstance(function_call_id, str)
        or not function_call_id
        or len(function_call_id) > 256
        or remote["requested_gpu"] != "A100-80GB:2"
        or remote["gpu_count"] != GPU_COUNT
        or remote["gpu_allocation_seconds_status"] != "pending-final-function-return"
        or interval > allocation
        or not math.isclose(gpu_seconds, GPU_COUNT * allocation, rel_tol=0.0, abs_tol=1e-6)
        or not math.isclose(gpu_hours, gpu_seconds / 3600.0, rel_tol=0.0, abs_tol=1e-9)
        or remote["bound_artifacts"] != expected_bound
        or materialized["volume_name"] != RESULTS_VOLUME
        or materialized["remote_path"] != expected_remote
        or remote["remote_prefix"] != expected_remote
        or not isinstance(manifest_hash, str)
        or re.fullmatch(r"[0-9a-f]{64}", manifest_hash) is None
    ):
        raise ValueError("integrated result failed identity, resource, or status validation")
    inventory = _exact_inventory(remote)
    return dict(remote), dict(materialized), inventory


def _download_result(
    materialized: Mapping[str, Any],
    *,
    local_root: Path,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> None:
    if local_root.exists():
        raise FileExistsError(f"immutable integrated result already exists: {local_root}")
    local_root.parent.mkdir(parents=True, exist_ok=True)
    completed = runner(
        [
            "modal",
            "volume",
            "get",
            str(materialized["volume_name"]),
            str(materialized["remote_path"]),
            str(local_root.parent),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=120.0,
    )
    if completed.returncode != 0:
        raise RuntimeError("Modal volume download failed")
    if not local_root.is_dir():
        raise FileNotFoundError("Modal volume download did not materialize the expected prefix")


def _verify_downloaded_manifest(
    local_root: Path,
    *,
    remote: Mapping[str, Any],
    materialized: Mapping[str, Any],
) -> Path:
    manifest_path = local_root / "REMOTE_MANIFEST.json"
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise FileNotFoundError("downloaded integrated remote manifest is absent or symlinked")
    if _sha256_file(manifest_path) != remote["remote_manifest_sha256"]:
        raise ValueError("returned and downloaded integrated manifest hashes differ")
    manifest = json.loads(manifest_path.read_text())
    if (
        not isinstance(manifest, dict)
        or set(manifest) != {"schema_version", "attempt_id", "remote_prefix", "artifacts"}
        or manifest["schema_version"]
        != "sloforge.branchfabric.experiment-004-v11-integrated-remote-manifest/v1"
        or manifest["attempt_id"] != remote["attempt_id"]
        or manifest["remote_prefix"] != materialized["remote_path"]
        or not isinstance(manifest["artifacts"], list)
    ):
        raise ValueError("downloaded integrated manifest identity is invalid")
    declared: dict[str, tuple[int, str]] = {}
    for item in manifest["artifacts"]:
        if not isinstance(item, dict) or set(item) != {"relative_path", "bytes", "sha256"}:
            raise ValueError("integrated manifest inventory row is malformed")
        relative = Path(str(item["relative_path"]))
        key = relative.as_posix()
        size = item["bytes"]
        digest = item["sha256"]
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or key in declared
            or isinstance(size, bool)
            or not isinstance(size, int)
            or size < 0
            or not isinstance(digest, str)
            or re.fullmatch(r"[0-9a-f]{64}", digest) is None
        ):
            raise ValueError("integrated manifest inventory contains unsafe data")
        declared[key] = (size, digest)
    tree = tuple(local_root.rglob("*"))
    if any(path.is_symlink() for path in tree):
        raise ValueError("downloaded integrated artifact tree contains a symlink")
    actual = {
        path.relative_to(local_root).as_posix(): path
        for path in tree
        if path.is_file() and path != manifest_path
    }
    if set(actual) != set(declared):
        raise ValueError("downloaded integrated artifact inventory differs from its manifest")
    for key, path in actual.items():
        size, digest = declared[key]
        if path.stat().st_size != size or _sha256_file(path) != digest:
            raise ValueError(f"downloaded integrated artifact failed integrity: {key}")

    completion = json.loads((local_root / "function-completion.json").read_text())
    final_only = {
        "gpu_allocation_seconds",
        "gpu_seconds",
        "gpu_hours",
        "remote_prefix",
        "remote_manifest_sha256",
    }
    returned_completion = {key: value for key, value in remote.items() if key not in final_only}
    if completion != returned_completion:
        raise ValueError("downloaded completion differs from the returned integrated result")
    cleanup_path = local_root / "in_function_cleanup.json"
    if isinstance(remote["in_function_cleanup"], dict):
        if not cleanup_path.is_file():
            raise FileNotFoundError("downloaded in-function cleanup evidence is absent")
        if json.loads(cleanup_path.read_text()) != remote["in_function_cleanup"]:
            raise ValueError("downloaded cleanup evidence differs from the returned result")
    return manifest_path


def _settle_verified(
    *,
    ledger_path: Path,
    reservation_id: str,
    remote: Mapping[str, Any],
    inventory: tuple[dict[str, Any], dict[str, Any]],
    manifest_path: Path,
    coordinator_elapsed_seconds: float,
) -> tuple[Experiment004GpuHourLedger, float]:
    ledger = _load_ledger(ledger_path)
    raw_ref = ArtifactSampleRef(
        artifact_reference=str(manifest_path),
        artifact_sha256=_sha256_file(manifest_path),
        sample_selector="$",
    )
    remote_allocation_seconds = float(remote["gpu_allocation_seconds"])
    settled = settle_gpu_invocation(
        ledger,
        reservation_id=reservation_id,
        function_call_id=str(remote["function_call_id"]),
        actual_gpu_models=(str(inventory[0]["name"]), str(inventory[1]["name"])),
        gpu_uuids=(str(inventory[0]["uuid"]), str(inventory[1]["uuid"])),
        # The local Modal CLI interval includes unpaid image hydration,
        # queueing, and startup, so it is not a GPU-allocation interval and
        # can legitimately exceed the 588-second Function reservation.  The
        # Function measures its own allocation interval from entry through
        # immutable publication.  Bind that same measured interval into both
        # settlement observations; retain coordinator_elapsed_seconds in the
        # coordinator artifact instead of laundering it into GPU time.
        client_elapsed_seconds=remote_allocation_seconds,
        remote_observed_allocation_seconds=remote_allocation_seconds,
        raw_manifest=raw_ref,
        gpu_price_per_hour_usd=GPU_PRICE_PER_HOUR_USD,
    )
    interval = next(
        item for item in settled.intervals if item.invocation_id == remote["attempt_id"]
    )
    _atomic_write_bytes(ledger_path, _ledger_bytes(settled))
    if coordinator_elapsed_seconds < remote_allocation_seconds:
        raise ValueError("coordinator interval cannot be shorter than remote allocation")
    return settled, interval.accounted_wall_seconds


def _charge_active_failure(
    *,
    ledger_path: Path,
    reservation_id: str,
    attempt_id: str,
    failure_root: Path,
    stage: str,
    error: BaseException,
    client_elapsed_seconds: float,
    completed: subprocess.CompletedProcess[str] | None,
) -> dict[str, Any]:
    ledger = _load_ledger(ledger_path)
    active = tuple(item for item in ledger.reservations if item.reservation_id == reservation_id)
    settled = tuple(item for item in ledger.intervals if item.invocation_id == attempt_id)
    charged = tuple(
        item for item in ledger.conservative_failure_charges if item.invocation_id == attempt_id
    )
    if not active:
        if len(settled) + len(charged) == 1:
            return {
                "status": "ALREADY_TERMINAL",
                "reservation_cleared": True,
                "attempt_id": attempt_id,
            }
        raise RuntimeError("failed invocation has no unique active or terminal ledger identity")
    if len(active) != 1 or active[0].invocation_id != attempt_id:
        raise RuntimeError("failed invocation reservation identity is inconsistent")
    evidence_path = failure_root / f"{attempt_id}-conservative-charge.json"
    evidence = {
        "schema_version": "sloforge.branchfabric.exp004-v11-final-failure-charge/v1",
        "status": "CONSERVATIVELY_CHARGED",
        "attempt_id": attempt_id,
        "reservation_id": reservation_id,
        "failure_stage": stage,
        "error": {"type": type(error).__name__, "message": str(error)},
        "client_elapsed_seconds": client_elapsed_seconds,
        "actual_gpu_seconds": None,
        "actual_gpu_seconds_status": "unavailable-or-not-trusted",
        "charged_wall_seconds": active[0].maximum_wall_seconds,
        "charged_gpu_seconds": active[0].maximum_gpu_seconds,
        "accounting_policy": "charge-full-preflight-bound-without-fabricated-measurement",
        "stdout_tail": None if completed is None else completed.stdout[-65_536:],
        "stderr_tail": None if completed is None else completed.stderr[-65_536:],
    }
    _write_new_json(evidence_path, evidence)
    updated = charge_failed_gpu_reservation(
        ledger,
        reservation_id=reservation_id,
        failure_stage=stage,
        failure_evidence=ArtifactSampleRef(
            artifact_reference=str(evidence_path),
            artifact_sha256=_sha256_file(evidence_path),
            sample_selector="$",
        ),
        gpu_price_per_hour_usd=GPU_PRICE_PER_HOUR_USD,
    )
    _atomic_write_bytes(ledger_path, _ledger_bytes(updated))
    return {
        "status": "CONSERVATIVELY_CHARGED",
        "reservation_cleared": not updated.reservations,
        "attempt_id": attempt_id,
        "charged_gpu_seconds": active[0].maximum_gpu_seconds,
        "failure_evidence": str(evidence_path),
        "failure_evidence_sha256": _sha256_file(evidence_path),
        "ledger_sha256": _sha256_file(ledger_path),
    }


def _emit_provider_cleanup_inputs(
    path: Path,
    *,
    remote: Mapping[str, Any],
    reservation_id: str,
    manifest_path: Path,
    ledger_path: Path,
) -> dict[str, Any]:
    cleanup = remote.get("in_function_cleanup")
    payload = {
        "schema_version": "sloforge.branchfabric.exp004-v11-provider-cleanup-inputs/v1",
        "status": "PENDING_EXPLICIT_PROVIDER_POST_RETURN_AUDIT",
        "attempt_id": remote["attempt_id"],
        "reservation_id": reservation_id,
        "function_call_id": remote["function_call_id"],
        "app_name": APP_NAME,
        "remote_prefix": remote["remote_prefix"],
        "remote_manifest": str(manifest_path),
        "remote_manifest_sha256": _sha256_file(manifest_path),
        "ledger_sha256_after_settlement": _sha256_file(ledger_path),
        "coordinator": {
            "pid": os.getpid(),
            "pgid": os.getpgid(0),
            "sid": os.getsid(0),
        },
        "in_function_cleanup_status": (
            "PASS" if isinstance(cleanup, dict) and cleanup.get("pass") is True else "FAIL"
        ),
        "provider_selectors": {
            "app_name": APP_NAME,
            "function_call_id": remote["function_call_id"],
            "results_volume": RESULTS_VOLUME,
            "model_volume": MODEL_VOLUME,
            "remote_prefix": remote["remote_prefix"],
        },
        "required_zero_counts": [
            "active_apps",
            "running_tasks",
            "running_containers",
            "endpoints",
            "provider_reservations",
            "owned_child_processes",
            "profilers",
        ],
        "authorized_persistent_volumes": [MODEL_VOLUME, RESULTS_VOLUME],
        "provider_cleanup_pass": None,
        "methodology": (
            "provider state must be observed after Function return; in-function cleanup "
            "is retained separately and cannot satisfy this gate"
        ),
    }
    _write_new_json(path, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-path", required=True, type=Path)
    args = parser.parse_args()
    if args.config_path.is_symlink():
        raise ValueError("integrated v11 config must not be a symlink")
    config_path = args.config_path.resolve(strict=True)
    _LOCK.parent.mkdir(parents=True, exist_ok=True)
    with _LOCK.open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        config, seal = _load_sealed_config(
            config_path,
            repository_root=_ROOT,
            ledger_path=_LEDGER,
        )
        budget = _require_launch_budget(config, _ROOT)
        local_root = _LOCAL_RESULT_PARENT / config.attempt_id
        if local_root.exists():
            raise FileExistsError(
                f"fresh integrated attempt already has a local immutable result: {local_root}"
            )
        reservation_id = f"exp004-v11-c-{secrets.token_hex(10)}"
        _ledger, preflight, commitment = _reserve_locked(
            config,
            ledger_path=_LEDGER,
            reservation_id=reservation_id,
        )
        completed: subprocess.CompletedProcess[str] | None = None
        started = time.monotonic()
        stage = "modal-launch"
        try:
            completed, elapsed = _invoke_modal(
                config_path,
                reservation_id=reservation_id,
                budget_usd=budget,
            )
            if completed.returncode != 0:
                raise RuntimeError("integrated v11 Modal process returned failure")
            stage = "result-validation"
            envelope = _parse_exact_result(completed.stdout)
            remote, materialized, inventory = _validate_remote_result(
                envelope,
                config=config,
                reservation_id=reservation_id,
                reservation_commitment_sha256=commitment,
            )
            stage = "immutable-download"
            _download_result(materialized, local_root=local_root)
            manifest = _verify_downloaded_manifest(
                local_root,
                remote=remote,
                materialized=materialized,
            )
            stage = "measured-settlement"
            settled, accounted_wall = _settle_verified(
                ledger_path=_LEDGER,
                reservation_id=reservation_id,
                remote=remote,
                inventory=inventory,
                manifest_path=manifest,
                coordinator_elapsed_seconds=elapsed,
            )
            stage = "provider-audit-inputs"
            provider_inputs = _emit_provider_cleanup_inputs(
                _PROVIDER_INPUTS,
                remote=remote,
                reservation_id=reservation_id,
                manifest_path=manifest,
                ledger_path=_LEDGER,
            )
        except BaseException as error:
            elapsed = time.monotonic() - started
            charge = _charge_active_failure(
                ledger_path=_LEDGER,
                reservation_id=reservation_id,
                attempt_id=config.attempt_id,
                failure_root=_FAILURE_ROOT,
                stage=stage,
                error=error,
                client_elapsed_seconds=elapsed,
                completed=completed,
            )
            raise RuntimeError(
                f"integrated v11 attempt terminated; accounting={charge['status']}"
            ) from error
        output = {
            "schema_version": "sloforge.branchfabric.exp004-v11-final-coordinator/v1",
            "status": "SETTLED_PENDING_PROVIDER_CLEANUP",
            "attempt_id": config.attempt_id,
            "reservation_id": reservation_id,
            "preflight": preflight,
            "seal": seal,
            "modal": envelope,
            "local_root": str(local_root),
            "accounted_wall_seconds": accounted_wall,
            "accounted_gpu_seconds": accounted_wall * GPU_COUNT,
            "coordinator_elapsed_seconds": elapsed,
            "allocation_accounting_basis": (
                "remote Function entry through immutable publication; local Modal CLI "
                "hydration/queue/startup interval retained separately"
            ),
            "reservation_cleared": not settled.reservations,
            "ledger_sha256": _sha256_file(_LEDGER),
            "provider_cleanup_inputs": provider_inputs,
        }
        print(_canonical_bytes(output).decode(), end="")
        return 1 if remote["status"] == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
