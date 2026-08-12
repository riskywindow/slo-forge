"""CUDA-clean two-worker controller for BranchFabric Experiment 004.

The Modal Function imports this module without importing CUDA libraries.  It
executes one fresh interpreter per physical GPU, sets a single UUID in each
child's ``CUDA_VISIBLE_DEVICES``, and coordinates them with immutable barrier
files.  No fork occurs after CUDA initialization.
"""

from __future__ import annotations

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
    ).encode("utf-8")


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
        "schema_version": "sloforge.branchfabric.cuda-clean-import-audit/v1",
        "stage": stage,
        "observed_at_utc": _utc_now(),
        "observed_at_monotonic_ns": time.monotonic_ns(),
        "pid": os.getpid(),
        "loaded_forbidden_modules": loaded,
        "cuda_clean": not loaded,
    }


def _inventory() -> tuple[dict[str, Any], ...]:
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
    rows: list[dict[str, Any]] = []
    for raw in completed.stdout.splitlines():
        if not raw.strip():
            continue
        fields = [field.strip() for field in raw.split(",")]
        if len(fields) != 7:
            raise RuntimeError("nvidia-smi inventory returned an unexpected field count")
        rows.append(
            {
                "index": int(fields[0]),
                "uuid": fields[1],
                "name": fields[2],
                "driver_version": fields[3],
                "memory_total_mib": int(fields[4]),
                "memory_used_mib": int(fields[5]),
                "utilization_percent": int(fields[6]),
            }
        )
    if len(rows) != 2:
        raise RuntimeError(
            f"Experiment 004 requires exactly two visible GPUs, observed {len(rows)}"
        )
    if len({str(row["uuid"]) for row in rows}) != 2:
        raise RuntimeError("Experiment 004 received duplicate GPU UUIDs")
    for row in rows:
        if "A100" not in str(row["name"]) or int(row["memory_total_mib"]) < 79_000:
            raise RuntimeError(f"Experiment 004 requires A100-80GB, observed {row}")
    return tuple(rows)


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
        raise RuntimeError(f"nvidia-smi compute process query failed: {completed.stderr.strip()}")
    result: list[dict[str, Any]] = []
    for raw in completed.stdout.splitlines():
        if not raw.strip() or raw.lower().startswith("no running processes"):
            continue
        fields = [field.strip() for field in raw.split(",")]
        if len(fields) != 4:
            raise RuntimeError("nvidia-smi compute process query returned an unexpected row")
        result.append(
            {
                "gpu_uuid": fields[0],
                "pid": int(fields[1]),
                "process_name": fields[2],
                "used_gpu_memory_mib": int(fields[3]),
            }
        )
    return tuple(result)


def _require_no_compute_processes(processes: tuple[dict[str, Any], ...], *, phase: str) -> None:
    if processes:
        raise RuntimeError(
            f"exclusive two-GPU container has compute processes during {phase}: {processes}"
        )


def _wait_for_files(
    paths: tuple[Path, ...],
    *,
    deadline_ns: int,
    phase: str,
    processes: tuple[tuple[str, subprocess.Popen[bytes]], ...] = (),
) -> None:
    while not all(path.is_file() for path in paths):
        exited = {
            role: returncode
            for role, process in processes
            if (returncode := process.poll()) is not None
        }
        if exited:
            raise RuntimeError(f"workers exited while waiting for {phase}: {exited}")
        if time.monotonic_ns() >= deadline_ns:
            absent = [str(path) for path in paths if not path.is_file()]
            raise TimeoutError(f"timed out waiting for {phase}: {absent}")
        time.sleep(0.05)


def _process_group_exists(process_group: int) -> bool:
    try:
        os.killpg(process_group, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _wait_for_group_exit(process_group: int, *, timeout_seconds: float) -> bool:
    deadline_ns = time.monotonic_ns() + int(timeout_seconds * 1e9)
    while _process_group_exists(process_group):
        if time.monotonic_ns() >= deadline_ns:
            return False
        time.sleep(0.02)
    return True


def _terminate_group(process: subprocess.Popen[bytes], actions: list[dict[str, Any]]) -> None:
    """Terminate the complete worker process group, even if its leader exited."""

    process_group = process.pid
    if process.poll() is not None and not _process_group_exists(process_group):
        return
    actions.append(
        {
            "pid": process.pid,
            "process_group": process_group,
            "signal": "SIGTERM",
            "at_utc": _utc_now(),
        }
    )
    with suppress(ProcessLookupError):
        os.killpg(process_group, signal.SIGTERM)
    if process.poll() is None:
        with suppress(subprocess.TimeoutExpired):
            process.wait(timeout=10.0)
    if _wait_for_group_exit(process_group, timeout_seconds=10.0):
        return
    actions.append(
        {
            "pid": process.pid,
            "process_group": process_group,
            "signal": "SIGKILL",
            "at_utc": _utc_now(),
        }
    )
    with suppress(ProcessLookupError):
        os.killpg(process_group, signal.SIGKILL)
    if process.poll() is None:
        with suppress(subprocess.TimeoutExpired):
            process.wait(timeout=5.0)
    if not _wait_for_group_exit(process_group, timeout_seconds=5.0):
        raise RuntimeError(f"worker process group {process_group} survived SIGKILL")


def _run_peer_probe(
    *,
    peer_probe_path: Path,
    work_root: Path,
    gpu_uuids: tuple[str, str],
    cleanup_actions: list[dict[str, Any]],
) -> dict[str, Any]:
    output_path = work_root / "topology/cuda-peer-access.json"
    stdout_path = work_root / "logs/peer-probe.stdout.log"
    stderr_path = work_root / "logs/peer-probe.stderr.log"
    output_path.parent.mkdir(parents=True, exist_ok=False)
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_uuids)
    command = [sys.executable, str(peer_probe_path), "--output", str(output_path)]
    for gpu_uuid in gpu_uuids:
        command.extend(("--expected-gpu-uuid", gpu_uuid))
    with stdout_path.open("xb") as stdout_handle, stderr_path.open("xb") as stderr_handle:
        probe = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=stdout_handle,
            stderr=stderr_handle,
            env=environment,
            start_new_session=True,
        )
        try:
            try:
                returncode = probe.wait(timeout=60.0)
            except subprocess.TimeoutExpired as error:
                raise TimeoutError("CUDA peer-capability probe exceeded 60 seconds") from error
        finally:
            _terminate_group(probe, cleanup_actions)
    if returncode != 0:
        raise RuntimeError(f"CUDA peer-capability probe failed with return code {returncode}")
    payload = json.loads(output_path.read_text())
    if not isinstance(payload, dict):
        raise ValueError("CUDA peer-capability probe output is not a JSON object")
    if (
        payload.get("schema_version") != "sloforge.branchfabric.experiment-004-cuda-peer-probe/v1"
        or tuple(payload.get("expected_physical_gpu_uuids", ())) != gpu_uuids
        or payload.get("cuda_visible_devices") != ",".join(gpu_uuids)
        or not isinstance(payload.get("matrix"), dict)
        or payload["matrix"].get("device_count") != 2
    ):
        raise ValueError("CUDA peer-capability probe output differs from the exact GPU pair")
    directions = {
        (int(item["source_logical_device"]), int(item["destination_logical_device"]))
        for item in payload["matrix"].get("access", ())
    }
    if directions != {(0, 1), (1, 0)}:
        raise ValueError("CUDA peer-capability probe omitted a directed peer query")
    return payload


def run_two_gpu_controller(
    *,
    config_path: Path,
    work_root: Path,
    worker_path: Path,
    model_snapshot: Path,
    readiness_timeout_seconds: float,
    transaction_timeout_seconds: float,
) -> dict[str, Any]:
    """Run one bounded two-GPU causal transaction and verify child cleanup."""

    peer_probe_path = worker_path.with_name("gpu_peer_probe.py")
    if (
        not config_path.is_file()
        or not worker_path.is_file()
        or not peer_probe_path.is_file()
        or not model_snapshot.is_dir()
    ):
        raise FileNotFoundError("controller input path is absent")
    if readiness_timeout_seconds <= 0 or transaction_timeout_seconds <= 0:
        raise ValueError("controller timeouts must be positive")
    work_root.mkdir(parents=True, exist_ok=False)
    barrier_root = work_root / "barriers"
    log_root = work_root / "logs"
    barrier_root.mkdir()
    log_root.mkdir()
    before_audit = cuda_clean_import_audit("before_inventory")
    if not before_audit["cuda_clean"]:
        raise RuntimeError(f"controller imported a CUDA-owning module: {before_audit}")
    inventory_before = _inventory()
    _require_no_compute_processes(_compute_processes(), phase="worker preflight")

    children: list[tuple[str, subprocess.Popen[bytes], Any, Any]] = []
    cleanup_actions: list[dict[str, Any]] = []
    gpu_uuids = tuple(str(row["uuid"]) for row in inventory_before)
    assert len(gpu_uuids) == 2
    peer_access = _run_peer_probe(
        peer_probe_path=peer_probe_path,
        work_root=work_root,
        gpu_uuids=(gpu_uuids[0], gpu_uuids[1]),
        cleanup_actions=cleanup_actions,
    )
    after_peer_audit = cuda_clean_import_audit("after_peer_probe")
    if not after_peer_audit["cuda_clean"]:
        raise RuntimeError("controller imported a CUDA-owning module during peer preflight")
    _require_no_compute_processes(_compute_processes(), phase="peer-probe cleanup")
    transaction_error: BaseException | None = None
    try:
        for role, gpu in zip(("serving", "rollout"), inventory_before, strict=True):
            role_root = work_root / role
            role_root.mkdir()
            stdout_handle = (log_root / f"{role}.stdout.log").open("xb")
            stderr_handle = (log_root / f"{role}.stderr.log").open("xb")
            environment = dict(os.environ)
            # vLLM 0.23's architecture-inspection subprocess requires an
            # integer CUDA_VISIBLE_DEVICES entry. Keep the immutable physical
            # UUID in a separate experiment variable and result field.
            environment["CUDA_VISIBLE_DEVICES"] = str(gpu["index"])
            environment["SLOFORGE_EXP004_VISIBLE_GPU_INDEX"] = str(gpu["index"])
            environment["SLOFORGE_EXP004_PHYSICAL_GPU_UUID"] = str(gpu["uuid"])
            command = [
                sys.executable,
                str(worker_path),
                "--role",
                role,
                "--config",
                str(config_path),
                "--work-root",
                str(role_root),
                "--barrier-root",
                str(barrier_root),
                "--model-snapshot",
                str(model_snapshot),
                "--physical-gpu-uuid",
                str(gpu["uuid"]),
            ]
            try:
                child = subprocess.Popen(
                    command,
                    stdin=subprocess.DEVNULL,
                    stdout=stdout_handle,
                    stderr=stderr_handle,
                    env=environment,
                    start_new_session=True,
                )
            except BaseException:
                stdout_handle.close()
                stderr_handle.close()
                raise
            children.append((role, child, stdout_handle, stderr_handle))

        readiness_deadline = time.monotonic_ns() + int(readiness_timeout_seconds * 1e9)
        ready_paths = tuple(barrier_root / f"{role}.ready.json" for role, *_ in children)
        _wait_for_files(
            ready_paths,
            deadline_ns=readiness_deadline,
            phase="warm worker readiness",
            processes=tuple((role, child) for role, child, *_ in children),
        )
        start_ns = time.monotonic_ns() + 1_000_000_000
        _write_new(
            barrier_root / "start.json",
            {
                "schema_version": "sloforge.branchfabric.experiment-004-start/v1",
                "start_monotonic_ns": start_ns,
                "issued_at_monotonic_ns": time.monotonic_ns(),
                "issued_at_utc": _utc_now(),
            },
        )
        transaction_deadline = time.monotonic_ns() + int(transaction_timeout_seconds * 1e9)
        while any(child.poll() is None for _, child, *_ in children):
            if time.monotonic_ns() >= transaction_deadline:
                raise TimeoutError("two-GPU Experiment 004 transaction exceeded its bound")
            failed = [
                (role, child.returncode)
                for role, child, *_ in children
                if child.poll() not in {None, 0}
            ]
            if failed:
                raise RuntimeError(f"an Experiment 004 GPU worker failed: {failed}")
            time.sleep(0.1)
        returncodes = {role: child.returncode for role, child, *_ in children}
        if set(returncodes.values()) != {0}:
            raise RuntimeError(f"Experiment 004 worker return codes were invalid: {returncodes}")
        result_paths = tuple(work_root / role / "result.json" for role, *_ in children)
        _wait_for_files(
            result_paths,
            deadline_ns=transaction_deadline,
            phase="worker results",
            processes=tuple((role, child) for role, child, *_ in children),
        )
    except BaseException as caught:
        transaction_error = caught
    finally:
        for _role, child, stdout_handle, stderr_handle in children:
            try:
                _terminate_group(child, cleanup_actions)
            finally:
                stdout_handle.close()
                stderr_handle.close()

    cleanup_error: BaseException | None = None
    cleanup_deadline = time.monotonic_ns() + 30_000_000_000
    processes_after: tuple[dict[str, Any], ...] = ()
    inventory_after: tuple[dict[str, Any], ...] = ()
    try:
        while time.monotonic_ns() < cleanup_deadline:
            processes_after = _compute_processes()
            if not processes_after:
                break
            time.sleep(0.5)
        _require_no_compute_processes(processes_after, phase="worker postflight cleanup")
        inventory_after = _inventory()
        if tuple(row["uuid"] for row in inventory_after) != tuple(
            row["uuid"] for row in inventory_before
        ):
            raise RuntimeError("GPU identities changed during Experiment 004")
    except BaseException as caught:
        cleanup_error = caught
    status = "succeeded" if transaction_error is None and cleanup_error is None else "failed"
    result = {
        "schema_version": "sloforge.branchfabric.experiment-004-controller/v1",
        "status": status,
        "controller_pid": os.getpid(),
        "inventory_before": inventory_before,
        "inventory_after": inventory_after,
        "worker_pids": {role: child.pid for role, child, *_ in children},
        "worker_process_groups": {role: child.pid for role, child, *_ in children},
        "worker_returncodes": {role: child.returncode for role, child, *_ in children},
        "cleanup_actions": cleanup_actions,
        "compute_processes_after": processes_after,
        "cuda_peer_access": peer_access,
        "cuda_clean_import_audits": (
            before_audit,
            after_peer_audit,
            cuda_clean_import_audit("after_worker_cleanup"),
        ),
        "transaction_error": (
            None
            if transaction_error is None
            else {"type": type(transaction_error).__name__, "message": str(transaction_error)}
        ),
        "cleanup_error": (
            None
            if cleanup_error is None
            else {"type": type(cleanup_error).__name__, "message": str(cleanup_error)}
        ),
    }
    _write_new(work_root / "controller-result.json", result)
    return result


__all__ = ["cuda_clean_import_audit", "run_two_gpu_controller"]
