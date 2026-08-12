#!/usr/bin/env python3
"""Convert the v1 capacity failure reservation into a conservative settled charge.

This is an explicit CPU-only recovery operation.  It never calls Modal.  The
conversion is allowed only for the pinned v1 failure and only after validating
the returned result, downloaded manifest, every downloaded artifact, GPU
identity, reservation identity, and immutable configuration digest.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from sloforge.helix.characterization.gpu_reclamation_methodology import (
    ArtifactSampleRef,
    Experiment004GpuHourLedger,
    settle_gpu_invocation,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXPERIMENT_ROOT = ROOT / "artifacts/branchfabric/gpu-validation/experiment-004"
ATTEMPT_ID = "exp004-capacity-s41-v1"
RESERVATION_WALL_SECONDS = 212.0
GPU_PRICE_PER_HOUR_USD = 2.4984


def _canonical_bytes(value: Any) -> bytes:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n"
    ).encode()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_new(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(_canonical_bytes(value))
        handle.flush()
        os.fsync(handle.fileno())


def _atomic_write(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("xb") as handle:
        handle.write(_canonical_bytes(value))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _returned_envelope(terminal: dict[str, Any]) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    stdout = terminal.get("stdout_tail")
    if not isinstance(stdout, str):
        raise ValueError("v1 terminal failure lacks captured Modal stdout")
    for line in stdout.splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and set(value) == {"materialized", "result"}:
            candidates.append(value)
    if len(candidates) != 1:
        raise ValueError("v1 failure contains an ambiguous returned Modal envelope")
    return candidates[0]


def _validate_manifest(raw_root: Path, *, expected_digest: str) -> Path:
    manifest_path = raw_root / "REMOTE_MANIFEST.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise FileNotFoundError("v1 downloaded manifest is absent or symlinked")
    if _sha256(manifest_path) != expected_digest:
        raise ValueError("v1 returned and downloaded manifest digests differ")
    manifest = json.loads(manifest_path.read_text())
    expected_prefix = f"experiment-004/calibration/modal/{ATTEMPT_ID}"
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema_version")
        != "sloforge.branchfabric.capacity-remote-manifest/v1"
        or manifest.get("attempt_id") != ATTEMPT_ID
        or manifest.get("remote_prefix") != expected_prefix
        or not isinstance(manifest.get("artifacts"), list)
    ):
        raise ValueError("v1 downloaded manifest identity is invalid")
    declared: dict[str, tuple[int, str]] = {}
    for row in manifest["artifacts"]:
        if not isinstance(row, dict) or set(row) != {"relative_path", "bytes", "sha256"}:
            raise ValueError("v1 manifest contains an invalid inventory row")
        relative = Path(str(row["relative_path"]))
        key = relative.as_posix()
        if relative.is_absolute() or ".." in relative.parts or key in declared:
            raise ValueError("v1 manifest contains an unsafe or duplicate path")
        declared[key] = (int(row["bytes"]), str(row["sha256"]))
    actual_paths = tuple(
        path for path in raw_root.rglob("*") if path.is_file() and path != manifest_path
    )
    if any(path.is_symlink() for path in actual_paths):
        raise ValueError("v1 downloaded artifact tree contains a symlink")
    actual = {path.relative_to(raw_root).as_posix(): path for path in actual_paths}
    if set(actual) != set(declared):
        raise ValueError("v1 downloaded artifact inventory differs from its manifest")
    for key, path in actual.items():
        expected_bytes, expected_sha = declared[key]
        if path.stat().st_size != expected_bytes or _sha256(path) != expected_sha:
            raise ValueError(f"v1 downloaded artifact failed integrity: {key}")
    return manifest_path


def _prepare_failure_charge(
    *, experiment_root: Path, ledger: Experiment004GpuHourLedger
) -> tuple[Experiment004GpuHourLedger, dict[str, Any]]:
    calibration = experiment_root / "calibration"
    config_path = calibration / "calibration-config-v1.json"
    terminal_path = calibration / f"logs/{ATTEMPT_ID}-terminal-failure.json"
    logged_manifest_path = calibration / f"logs/{ATTEMPT_ID}-REMOTE_MANIFEST.json"
    raw_root = calibration / f"raw/modal/{ATTEMPT_ID}"
    completion_path = raw_root / "function-completion.json"
    terminal = json.loads(terminal_path.read_text())
    envelope = _returned_envelope(terminal)
    remote = envelope["result"]
    materialized = envelope["materialized"]
    if not isinstance(remote, dict) or not isinstance(materialized, dict):
        raise ValueError("v1 returned envelope contains non-object payloads")
    reservations = tuple(item for item in ledger.reservations if item.invocation_id == ATTEMPT_ID)
    if len(reservations) != 1:
        raise ValueError("v1 retained reservation is absent or duplicated")
    reservation = reservations[0]
    config_digest = hashlib.sha256(_canonical_bytes(json.loads(config_path.read_text()))).hexdigest()
    expected_prefix = f"experiment-004/calibration/modal/{ATTEMPT_ID}"
    if (
        reservation.maximum_wall_seconds != RESERVATION_WALL_SECONDS
        or reservation.config_sha256 != config_digest
        or terminal.get("attempt_id") != ATTEMPT_ID
        or terminal.get("reservation_id") != reservation.reservation_id
        or terminal.get("conservatively_committed_gpu_seconds") != 424.0
        or terminal.get("reservation_cleared") is not False
        or remote.get("attempt_id") != ATTEMPT_ID
        or remote.get("reservation_id") != reservation.reservation_id
        or remote.get("status") != "failed"
        or remote.get("run_error") is not None
        or remote.get("requested_gpu") != "A100-80GB:2"
        or remote.get("gpu_count") != 2
        or remote.get("remote_prefix") != expected_prefix
        or materialized
        != {
            "remote_path": expected_prefix,
            "volume_name": "sloforge-branchfabric-results",
        }
    ):
        raise ValueError("v1 failure identity differs from its retained reservation")
    manifest_digest = str(remote.get("remote_manifest_sha256"))
    manifest_path = _validate_manifest(raw_root, expected_digest=manifest_digest)
    if _sha256(logged_manifest_path) != manifest_digest or (
        logged_manifest_path.read_bytes() != manifest_path.read_bytes()
    ):
        raise ValueError("v1 logged and downloaded manifests differ")
    completion = json.loads(completion_path.read_text())
    identity_fields = (
        "attempt_id",
        "reservation_id",
        "function_call_id",
        "requested_gpu",
        "gpu_count",
        "gpu_allocation_seconds",
        "status",
    )
    if any(completion.get(field) != remote.get(field) for field in identity_fields):
        raise ValueError("v1 downloaded completion differs from the returned function identity")
    controller = completion.get("controller")
    if (
        not isinstance(controller, dict)
        or controller.get("status") != "failed"
        or controller.get("probe_count") != 0
        or controller.get("cleanup_error") is not None
        or controller.get("compute_processes_after") != []
        or controller.get("controller_error")
        != {
            "message": "Object of type CapacityProbePlan is not JSON serializable",
            "type": "TypeError",
        }
    ):
        raise ValueError("v1 downloaded controller failure is not the audited methodology failure")
    inventory = controller.get("inventory_before")
    if not isinstance(inventory, list) or len(inventory) != 2:
        raise ValueError("v1 failure lacks its exact two-GPU inventory")
    gpu_names = tuple(str(item.get("name")) for item in inventory)
    gpu_uuids = tuple(str(item.get("uuid")) for item in inventory)
    if (
        len(set(gpu_uuids)) != 2
        or any("A100" not in name or "80GB" not in name.replace(" ", "") for name in gpu_names)
        or any(int(item.get("memory_total_mib", 0)) < 79_000 for item in inventory)
    ):
        raise ValueError("v1 failure did not run on two distinct A100-80GB GPUs")
    ready = controller.get("worker_ready_evidence")
    if not isinstance(ready, list) or tuple(
        str(item.get("physical_gpu_uuid")) for item in ready
    ) != gpu_uuids:
        raise ValueError("v1 worker readiness does not bind the returned GPU UUIDs")
    remote_seconds = float(completion["gpu_allocation_seconds"])
    if not 0.0 < remote_seconds <= RESERVATION_WALL_SECONDS:
        raise ValueError("v1 returned allocation duration is outside its reservation")
    manifest_ref = ArtifactSampleRef(
        artifact_reference=str(manifest_path),
        artifact_sha256=manifest_digest,
        sample_selector="$",
    )
    updated = settle_gpu_invocation(
        ledger,
        reservation_id=reservation.reservation_id,
        function_call_id=str(completion["function_call_id"]),
        actual_gpu_models=(gpu_names[0], gpu_names[1]),
        gpu_uuids=(gpu_uuids[0], gpu_uuids[1]),
        client_elapsed_seconds=RESERVATION_WALL_SECONDS,
        remote_observed_allocation_seconds=remote_seconds,
        raw_manifest=manifest_ref,
        gpu_price_per_hour_usd=GPU_PRICE_PER_HOUR_USD,
    )
    record = {
        "schema_version": "sloforge.branchfabric.capacity-failure-charge-conversion/v1",
        "status": "applied",
        "attempt_id": ATTEMPT_ID,
        "reservation_id": reservation.reservation_id,
        "function_call_id": completion["function_call_id"],
        "failure_classification": "methodological-failure-before-first-probe",
        "accounting_policy": "charge-full-preflight-reservation",
        "accounted_wall_seconds": RESERVATION_WALL_SECONDS,
        "accounted_gpu_seconds": 424.0,
        "remote_observed_allocation_seconds": remote_seconds,
        "actual_gpu_models": gpu_names,
        "gpu_uuids": gpu_uuids,
        "config": {
            "artifact_reference": str(config_path),
            "config_file_sha256": _sha256(config_path),
            "canonical_config_sha256": config_digest,
        },
        "terminal_failure": {
            "artifact_reference": str(terminal_path),
            "artifact_sha256": _sha256(terminal_path),
        },
        "returned_completion": {
            "artifact_reference": str(completion_path),
            "artifact_sha256": _sha256(completion_path),
        },
        "raw_manifest": manifest_ref.model_dump(mode="json"),
        "downloaded_inventory_verified": True,
        "reservation_cleared_by_settlement": True,
        "consumed_before_gpu_seconds": ledger.consumed_additional_gpu_seconds,
        "consumed_after_gpu_seconds": updated.consumed_additional_gpu_seconds,
    }
    return updated, record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", type=Path, default=DEFAULT_EXPERIMENT_ROOT)
    parser.add_argument("--apply", action="store_true", required=True)
    args = parser.parse_args()
    experiment_root = args.experiment_root.resolve(strict=True)
    ledger_path = experiment_root / "gpu-hours.json"
    lock_path = experiment_root / "gpu-hours.lock"
    conversion_path = (
        experiment_root / f"calibration/logs/{ATTEMPT_ID}-failure-charge-conversion.json"
    )
    with lock_path.open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        if conversion_path.exists():
            raise FileExistsError("v1 failure charge conversion is immutable and already exists")
        ledger = Experiment004GpuHourLedger.model_validate_json(
            ledger_path.read_text(), strict=True
        )
        updated, record = _prepare_failure_charge(
            experiment_root=experiment_root, ledger=ledger
        )
        record["ledger_before_sha256"] = _sha256(ledger_path)
        record["ledger_after_sha256"] = hashlib.sha256(_canonical_bytes(updated)).hexdigest()
        staged_record = conversion_path.with_name(f".{conversion_path.name}.{os.getpid()}.tmp")
        _write_new(staged_record, record)
        _atomic_write(ledger_path, updated)
        os.replace(staged_record, conversion_path)
    print(_canonical_bytes(record).decode(), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
