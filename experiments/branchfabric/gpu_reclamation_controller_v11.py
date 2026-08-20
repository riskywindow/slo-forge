"""CUDA-clean single-worker controller for Experiment 004 v11 micro-validation."""

from __future__ import annotations

import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

FORBIDDEN_GPU_MODULE_PREFIXES = (
    "torch",
    "vllm",
    "triton",
    "cupy",
    "pynvml",
    "cuda",
    "numba.cuda",
)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n"
    ).encode()


def _write_new(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(_canonical_bytes(value))
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def cuda_clean_import_audit(stage: str) -> dict[str, Any]:
    loaded = sorted(
        name
        for name in sys.modules
        if any(
            name == prefix or name.startswith(prefix + ".")
            for prefix in FORBIDDEN_GPU_MODULE_PREFIXES
        )
    )
    return {
        "schema_version": "sloforge.branchfabric.v11-cuda-clean-import-audit/v1",
        "stage": stage,
        "observed_at_utc": _utc_now(),
        "pid": os.getpid(),
        "loaded_forbidden_modules": loaded,
        "cuda_clean": not loaded,
    }


def parse_single_a100_inventory(stdout: str) -> dict[str, Any]:
    rows = [line for line in stdout.splitlines() if line.strip()]
    if len(rows) != 1:
        raise RuntimeError(f"v11 micro requires exactly one visible GPU, observed {len(rows)}")
    fields = [item.strip() for item in rows[0].split(",")]
    if len(fields) != 7:
        raise RuntimeError("nvidia-smi inventory returned an unexpected field count")
    row = {
        "index": int(fields[0]),
        "uuid": fields[1],
        "name": fields[2],
        "driver_version": fields[3],
        "memory_total_mib": int(fields[4]),
        "memory_used_mib": int(fields[5]),
        "utilization_percent": int(fields[6]),
    }
    if not str(row["uuid"]).startswith("GPU-"):
        raise RuntimeError("v11 micro GPU lacks a physical UUID")
    if "A100" not in str(row["name"]) or int(row["memory_total_mib"]) < 79_000:
        raise RuntimeError(f"v11 micro requires A100-80GB, observed {row}")
    return row


def _inventory() -> dict[str, Any]:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,driver_version,memory.total,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=15.0,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"nvidia-smi inventory failed: {completed.stderr.strip()}")
    return parse_single_a100_inventory(completed.stdout)


def _compute_processes() -> tuple[dict[str, Any], ...]:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=15.0,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"nvidia-smi process query failed: {completed.stderr.strip()}")
    rows: list[dict[str, Any]] = []
    for line in completed.stdout.splitlines():
        if not line.strip() or line.lower().startswith("no running processes"):
            continue
        fields = [item.strip() for item in line.split(",")]
        if len(fields) != 4:
            raise RuntimeError("nvidia-smi process query returned an invalid row")
        rows.append(
            {
                "gpu_uuid": fields[0],
                "pid": int(fields[1]),
                "process_name": fields[2],
                "used_gpu_memory_mib": int(fields[3]),
            }
        )
    return tuple(rows)


def _terminate_process_group(child: subprocess.Popen[bytes], actions: list[dict[str, Any]]) -> None:
    process_group = child.pid
    if child.poll() is None:
        with suppress(ProcessLookupError):
            os.killpg(process_group, signal.SIGTERM)
        actions.append({"signal": "SIGTERM", "process_group": process_group})
        try:
            child.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            with suppress(ProcessLookupError):
                os.killpg(process_group, signal.SIGKILL)
            actions.append({"signal": "SIGKILL", "process_group": process_group})
            child.wait(timeout=5.0)


def run_one_gpu_micro_controller(
    *,
    config_path: Path,
    work_root: Path,
    worker_path: Path,
    model_snapshot: Path,
    absolute_deadline_ns: int,
) -> dict[str, Any]:
    """Spawn one fresh interpreter and prove its bounded postflight cleanup."""

    if not config_path.is_file() or not worker_path.is_file() or not model_snapshot.is_dir():
        raise FileNotFoundError("v11 micro controller input path is absent")
    if absolute_deadline_ns <= time.monotonic_ns():
        raise TimeoutError("v11 micro controller received an expired deadline")
    from sloforge.helix.characterization.gpu_reclamation_v11_methodology import (
        Experiment004V11MicroConfig,
        validate_bound_artifact,
    )

    config = Experiment004V11MicroConfig.model_validate_json(config_path.read_text())
    repository_root = worker_path.resolve(strict=True).parents[2]
    validate_bound_artifact(
        repository_root,
        reference=config.offline_gate_manifest,
        expected_sha256=config.offline_gate_manifest_sha256,
    )
    validate_bound_artifact(
        repository_root,
        reference=config.budget_authorization,
        expected_sha256=config.budget_authorization_sha256,
    )
    work_root.mkdir(parents=True, exist_ok=False)
    logs = work_root / "logs"
    worker_root = work_root / "worker"
    logs.mkdir()
    worker_root.mkdir()
    before_audit = cuda_clean_import_audit("before_inventory")
    if not before_audit["cuda_clean"]:
        raise RuntimeError("v11 controller imported a CUDA-owning module")
    inventory_before = _inventory()
    processes_before = _compute_processes()
    if processes_before:
        raise RuntimeError(f"one-GPU container was not exclusive before launch: {processes_before}")

    stdout_handle = (logs / "worker.stdout.log").open("xb")
    stderr_handle = (logs / "worker.stderr.log").open("xb")
    environment = dict(os.environ)
    # vLLM architecture inspection requires a numeric visible-device selector.
    environment["CUDA_VISIBLE_DEVICES"] = str(inventory_before["index"])
    environment["SLOFORGE_EXP004_PHYSICAL_GPU_UUID"] = str(inventory_before["uuid"])
    command = [
        sys.executable,
        str(worker_path),
        "--config",
        str(config_path),
        "--model-snapshot",
        str(model_snapshot),
        "--physical-gpu-uuid",
        str(inventory_before["uuid"]),
        "--work-root",
        str(worker_root),
    ]
    child: subprocess.Popen[bytes] | None = None
    cleanup_actions: list[dict[str, Any]] = []
    transaction_error: BaseException | None = None
    try:
        child = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=stdout_handle,
            stderr=stderr_handle,
            env=environment,
            start_new_session=True,
        )
        while child.poll() is None:
            if time.monotonic_ns() >= absolute_deadline_ns:
                raise TimeoutError("v11 one-GPU micro transaction exceeded its absolute bound")
            time.sleep(0.1)
        if child.returncode != 0:
            raise RuntimeError(f"v11 micro worker failed with return code {child.returncode}")
        if not (worker_root / "result.json").is_file():
            raise RuntimeError("v11 micro worker omitted its immutable result")
    except BaseException as error:
        transaction_error = error
    finally:
        if child is not None:
            _terminate_process_group(child, cleanup_actions)
        stdout_handle.close()
        stderr_handle.close()

    processes_after: tuple[dict[str, Any], ...] = ()
    cleanup_error: BaseException | None = None
    cleanup_deadline = min(absolute_deadline_ns, time.monotonic_ns() + 20_000_000_000)
    try:
        while time.monotonic_ns() < cleanup_deadline:
            processes_after = _compute_processes()
            if not processes_after:
                break
            time.sleep(0.25)
        if processes_after:
            raise RuntimeError(f"v11 micro worker left GPU processes: {processes_after}")
        inventory_after = _inventory()
        if inventory_after["uuid"] != inventory_before["uuid"]:
            raise RuntimeError("v11 micro GPU identity changed during the transaction")
    except BaseException as error:
        cleanup_error = error
        inventory_after = {}

    worker_result: dict[str, Any] | None = None
    result_path = worker_root / "result.json"
    if result_path.is_file():
        decoded = json.loads(result_path.read_text())
        if isinstance(decoded, dict):
            worker_result = decoded
    status = "succeeded" if transaction_error is None and cleanup_error is None else "failed"
    result = {
        "schema_version": "sloforge.branchfabric.experiment-004-v11-micro-controller/v1",
        "status": status,
        "controller_pid": os.getpid(),
        "worker_pid": None if child is None else child.pid,
        "worker_returncode": None if child is None else child.returncode,
        "inventory_before": inventory_before,
        "inventory_after": inventory_after,
        "compute_processes_before": processes_before,
        "compute_processes_after": processes_after,
        "cleanup_actions": cleanup_actions,
        "cuda_clean_import_audits": [
            before_audit,
            cuda_clean_import_audit("after_worker_cleanup"),
        ],
        "worker_result_sha256": _sha256(result_path) if result_path.is_file() else None,
        "worker_result": worker_result,
        "transaction_error": None
        if transaction_error is None
        else {"type": type(transaction_error).__name__, "message": str(transaction_error)},
        "cleanup_error": None
        if cleanup_error is None
        else {"type": type(cleanup_error).__name__, "message": str(cleanup_error)},
    }
    _write_new(work_root / "controller-result.json", result)
    return result


__all__ = [
    "cuda_clean_import_audit",
    "parse_single_a100_inventory",
    "run_one_gpu_micro_controller",
]
