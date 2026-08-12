#!/usr/bin/env python3
"""Budget-first sole Modal coordinator for Experiment 004 capacity calibration."""

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
_APP = _ROOT / "experiments/branchfabric/modal_gpu_capacity_calibration.py"
_EXPERIMENT_ROOT = _ROOT / "artifacts/branchfabric/gpu-validation/experiment-004"
_LEDGER = _EXPERIMENT_ROOT / "gpu-hours.json"
_LOCK = _EXPERIMENT_ROOT / "gpu-hours.lock"
_CLIENT_RESERVATION_WALL_SECONDS = 212.0
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
        "schema_version",
        "attempt_id",
        "seed",
        "model",
        "model_revision",
        "tokenizer_revision",
        "runtime",
        "runtime_version",
        "dtype",
        "tracing_level",
        "serving_prompt_tokens",
        "serving_output_tokens",
        "gpu_memory_utilization",
        "warmup_seconds",
        "minimum_measurement_seconds",
        "maximum_measurement_seconds",
        "maximum_probes",
        "rate_grid_rps",
        "serving_slo_maximum_ttft_seconds",
        "worker_initialization_timeout_seconds",
        "probe_timeout_seconds",
        "tail_drain_seconds",
        "producer_queue_capacity",
        "controller_timeout_seconds",
        "worker_shutdown_timeout_seconds",
        "maximum_allocation_wall_seconds",
    }
)


def _canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n"
    ).encode()


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
        raise RuntimeError("SLOFORGE_GPU_BUDGET_USD is required before capacity calibration")
    try:
        budget = float(raw)
    except ValueError as error:
        raise RuntimeError("SLOFORGE_GPU_BUDGET_USD must be finite and positive") from error
    if not math.isfinite(budget) or budget <= 0.0:
        raise RuntimeError("SLOFORGE_GPU_BUDGET_USD must be finite and positive")
    if budget < _MAXIMUM_COST_USD:
        raise RuntimeError(
            f"bounded capacity calibration may cost ${_MAXIMUM_COST_USD:.6f}, exceeding budget"
        )
    return budget


def _validate_config(path: Path) -> dict[str, Any]:
    if path.stat().st_size > 64 * 1024:
        raise ValueError("capacity calibration config exceeds 64 KiB")
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict) or set(payload) != _CONFIG_FIELDS:
        raise ValueError("capacity calibration config fields differ from the exact contract")
    exact = {
        "schema_version": "sloforge.branchfabric.capacity-modal-config/v1",
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "model_revision": "a09a35458c702b33eeacc393d103063234e8bc28",
        "tokenizer_revision": "a09a35458c702b33eeacc393d103063234e8bc28",
        "runtime": "vllm",
        "runtime_version": "0.23.0",
        "dtype": "bf16",
        "tracing_level": "minimal",
        "serving_prompt_tokens": 256,
        "serving_output_tokens": 64,
        "maximum_probes": 3,
        "rate_grid_rps": [10.0, 15.0, 17.25, 20.0, 23.5, 25.0, 30.0, 35.0],
        "serving_slo_maximum_ttft_seconds": 2.0,
        "maximum_allocation_wall_seconds": 212.0,
    }
    if any(payload.get(field) != value for field, value in exact.items()):
        raise ValueError("capacity calibration config differs from the pinned contract")
    attempt_id = payload["attempt_id"]
    if (
        not isinstance(attempt_id, str)
        or re.fullmatch(r"[a-z0-9][a-z0-9-]{7,95}", attempt_id) is None
    ):
        raise ValueError("capacity calibration attempt ID is invalid")
    seed = payload["seed"]
    if not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed < 1 << 63:
        raise ValueError("capacity calibration seed is invalid")

    def number(field: str, lower: float, upper: float) -> float:
        value = payload[field]
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            or not lower <= float(value) <= upper
        ):
            raise ValueError(f"capacity calibration {field} is out of bounds")
        return float(value)

    number("gpu_memory_utilization", 0.000001, 0.85)
    number("warmup_seconds", 1.0, 5.0)
    minimum = number("minimum_measurement_seconds", 10.0, 20.0)
    maximum = number("maximum_measurement_seconds", 10.0, 20.0)
    if maximum < minimum:
        raise ValueError("capacity calibration measurement bounds are reversed")
    number("worker_initialization_timeout_seconds", 60.0, 120.0)
    number("probe_timeout_seconds", 15.0, 25.0)
    number("tail_drain_seconds", 1.0, 2.0)
    if payload["producer_queue_capacity"] != 256:
        raise ValueError("capacity producer queue must use the pinned bounded capacity")
    number("controller_timeout_seconds", 120.0, 150.0)
    number("worker_shutdown_timeout_seconds", 5.0, 15.0)
    return payload


def _load_ledger() -> Any:
    from sloforge.helix.characterization.gpu_reclamation_methodology import (
        Experiment004GpuHourLedger,
    )

    return Experiment004GpuHourLedger.model_validate_json(_LEDGER.read_text(), strict=True)


def _reserve(config: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    from sloforge.helix.characterization.gpu_reclamation_methodology import (
        reserve_gpu_invocation,
    )

    reservation_id = f"exp004-{secrets.token_hex(10)}"
    updated, preflight = reserve_gpu_invocation(
        _load_ledger(),
        reservation_id=reservation_id,
        invocation_id=str(config["attempt_id"]),
        maximum_wall_seconds=_CLIENT_RESERVATION_WALL_SECONDS,
        config_sha256=hashlib.sha256(_canonical_bytes(config)).hexdigest(),
    )
    _atomic_write(_LEDGER, updated.model_dump(mode="json"))
    return reservation_id, preflight.model_dump(mode="json")


def _parse_modal_result(stdout: str) -> dict[str, Any]:
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
        raise ValueError(f"capacity Modal output contained {len(candidates)} result objects")
    return candidates[0]


def _settle(reservation_id: str, result: dict[str, Any], client_elapsed_s: float) -> None:
    from sloforge.helix.characterization.gpu_reclamation_methodology import (
        ArtifactSampleRef,
        settle_gpu_invocation,
    )

    remote, inventory, reservation = _validate_remote_identity(reservation_id, result)
    materialized = result["materialized"]
    local_root = _EXPERIMENT_ROOT / "calibration/raw/modal" / str(remote["attempt_id"])
    if local_root.exists():
        raise FileExistsError(f"local immutable capacity result already exists: {local_root}")
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
    manifest = local_root / "REMOTE_MANIFEST.json"
    _validate_downloaded_manifest(
        local_root,
        remote=remote,
        materialized=materialized,
        expected_attempt_id=reservation.invocation_id,
    )
    raw_ref = ArtifactSampleRef(
        artifact_reference=str(manifest),
        artifact_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
        sample_selector="$",
    )
    _materialize_calibration_index(local_root, attempt_id=str(remote["attempt_id"]))
    settled = settle_gpu_invocation(
        _load_ledger(),
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


def _validate_remote_identity(
    reservation_id: str, result: dict[str, Any]
) -> tuple[dict[str, Any], tuple[dict[str, Any], dict[str, Any]], Any]:
    if set(result) != {"result", "materialized"}:
        raise ValueError("capacity result envelope differs from its exact contract")
    remote = result["result"]
    materialized = result["materialized"]
    if not isinstance(remote, dict) or not isinstance(materialized, dict):
        raise ValueError("capacity result envelope contains non-object payloads")
    ledger = _load_ledger()
    matches = tuple(item for item in ledger.reservations if item.reservation_id == reservation_id)
    if len(matches) != 1:
        raise RuntimeError("capacity settlement reservation is absent or duplicated")
    reservation = matches[0]
    expected_remote_path = f"experiment-004/calibration/modal/{reservation.invocation_id}"
    if (
        remote.get("status") != "succeeded"
        or remote.get("run_error") is not None
        or remote.get("attempt_id") != reservation.invocation_id
        or remote.get("reservation_id") != reservation_id
        or remote.get("requested_gpu") != "A100-80GB:2"
        or remote.get("gpu_count") != 2
        or remote.get("calibration_status") not in {"ready", "blocked"}
        or materialized.get("volume_name") != "sloforge-branchfabric-results"
        or materialized.get("remote_path") != expected_remote_path
        or remote.get("remote_prefix") != expected_remote_path
    ):
        raise RuntimeError("capacity remote result failed success or identity validation")
    controller = remote.get("controller")
    if not isinstance(controller, dict) or controller.get("status") != "succeeded":
        raise RuntimeError("capacity controller was not successful")
    inventory_value = controller.get("inventory_before")
    if not isinstance(inventory_value, list) or len(inventory_value) != 2:
        raise RuntimeError("capacity result lacks the exact two-GPU inventory")
    inventory = (inventory_value[0], inventory_value[1])
    if (
        any(not isinstance(item, dict) for item in inventory)
        or len({str(item["uuid"]) for item in inventory}) != 2
        or any(
            "A100" not in str(item["name"]) or int(item["memory_total_mib"]) < 79_000
            for item in inventory
        )
    ):
        raise RuntimeError("capacity result did not use two distinct A100-80GB GPUs")
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
        raise FileNotFoundError("downloaded capacity manifest is absent or symlinked")
    digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    if digest != remote.get("remote_manifest_sha256"):
        raise ValueError("returned and downloaded capacity manifest digests differ")
    manifest = json.loads(manifest_path.read_text())
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema_version") != "sloforge.branchfabric.capacity-remote-manifest/v1"
        or manifest.get("attempt_id") != expected_attempt_id
        or manifest.get("remote_prefix") != materialized.get("remote_path")
        or not isinstance(manifest.get("artifacts"), list)
    ):
        raise ValueError("downloaded capacity manifest identity is invalid")
    declared: dict[str, tuple[int, str]] = {}
    for item in manifest["artifacts"]:
        if not isinstance(item, dict) or set(item) != {"relative_path", "bytes", "sha256"}:
            raise ValueError("capacity manifest contains an invalid inventory row")
        relative = Path(str(item["relative_path"]))
        if relative.is_absolute() or ".." in relative.parts or relative.as_posix() in declared:
            raise ValueError("capacity manifest contains an unsafe or duplicate path")
        declared[relative.as_posix()] = (int(item["bytes"]), str(item["sha256"]))
    actual_files = tuple(
        path for path in local_root.rglob("*") if path.is_file() and path != manifest_path
    )
    if any(path.is_symlink() for path in actual_files):
        raise ValueError("downloaded capacity artifact tree contains a symlink")
    actual = {path.relative_to(local_root).as_posix(): path for path in actual_files}
    if set(actual) != set(declared):
        raise ValueError("downloaded capacity artifact inventory differs from its manifest")
    for relative_key, path in actual.items():
        expected_bytes, expected_sha = declared[relative_key]
        if (
            path.stat().st_size != expected_bytes
            or hashlib.sha256(path.read_bytes()).hexdigest() != expected_sha
        ):
            raise ValueError(f"downloaded capacity artifact failed integrity: {relative_key}")


def _artifact_reference(path: Path) -> dict[str, Any]:
    return {
        "artifact_reference": str(path),
        "artifact_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "sample_selector": "$",
    }


def _materialize_calibration_index(local_root: Path, *, attempt_id: str) -> None:
    """Index immutable downloaded raw evidence without copying or rewriting it."""

    remote_calibration = local_root / "calibration"
    for topology in ("one-gpu", "two-gpu"):
        entries = []
        for result_path in sorted((remote_calibration / topology).glob("*/result.json")):
            plan_path = result_path.with_name("plan.json")
            raw_path = result_path.with_name("raw.json")
            if not plan_path.is_file() or not raw_path.is_file():
                raise FileNotFoundError("downloaded calibration probe lacks plan/raw evidence")
            entries.append(
                {
                    "probe_id": result_path.parent.name,
                    "plan": _artifact_reference(plan_path),
                    "raw": _artifact_reference(raw_path),
                    "result": _artifact_reference(result_path),
                }
            )
        _write_new(
            _EXPERIMENT_ROOT / f"calibration/{topology}/{attempt_id}.index.json",
            {
                "schema_version": "sloforge.branchfabric.capacity-artifact-index/v1",
                "attempt_id": attempt_id,
                "topology": topology,
                "entries": entries,
            },
        )
    selection_path = remote_calibration / "selected-load.json"
    if selection_path.is_file():
        selection = json.loads(selection_path.read_text())
        _write_new(
            _EXPERIMENT_ROOT / "calibration/selected-load.json",
            {
                **selection,
                "raw_provenance": _artifact_reference(selection_path),
            },
        )


def _record_terminal_failure(
    *,
    reservation_id: str,
    attempt_id: str,
    stage: str,
    error: BaseException,
    elapsed_seconds: float,
    completed: subprocess.CompletedProcess[str] | None = None,
) -> None:
    """Preserve an explicit conservative charge without clearing the reservation."""

    path = _EXPERIMENT_ROOT / f"calibration/logs/{attempt_id}-terminal-failure.json"
    _write_new(
        path,
        {
            "schema_version": "sloforge.branchfabric.capacity-terminal-failure/v1",
            "attempt_id": attempt_id,
            "reservation_id": reservation_id,
            "stage": stage,
            "error": {"type": type(error).__name__, "message": str(error)},
            "client_elapsed_seconds": elapsed_seconds,
            "conservative_charge_state": "full_reservation_retained_in_gpu_hour_ledger",
            "conservatively_committed_gpu_seconds": (_CLIENT_RESERVATION_WALL_SECONDS * _GPU_COUNT),
            "reservation_cleared": False,
            "stdout_tail": (None if completed is None else completed.stdout[-65_536:]),
            "stderr_tail": (None if completed is None else completed.stderr[-65_536:]),
        },
    )


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
        environment = dict(os.environ)
        environment["SLOFORGE_MODAL_PREFLIGHT_TOKEN"] = secrets.token_hex(32)
        environment["SLOFORGE_EXP004_RESERVATION_ID"] = reservation_id
        started = time.monotonic()
        completed: subprocess.CompletedProcess[str] | None = None
        try:
            completed = subprocess.run(
                ["modal", "run", str(_APP), "--config-path", str(config_path)],
                check=False,
                capture_output=True,
                text=True,
                timeout=_CLIENT_RESERVATION_WALL_SECONDS,
                env=environment,
            )
            elapsed = time.monotonic() - started
            if completed.returncode != 0:
                sys.stderr.write(completed.stdout)
                sys.stderr.write(completed.stderr)
                raise RuntimeError("capacity Modal process returned a failure")
            result = _parse_modal_result(completed.stdout)
            _settle(reservation_id, result, elapsed)
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
                "capacity calibration failed; its full reservation remains conservatively charged"
            ) from caught
        print(
            _canonical_bytes(
                {"preflight": preflight, "maximum_cost_usd": _MAXIMUM_COST_USD, "modal": result}
            ).decode(),
            end="",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
