"""CUDA-clean two-worker controller for BranchFabric Experiment 004.

The Modal Function imports this module without importing CUDA libraries.  It
executes one fresh interpreter per physical GPU, sets a single UUID in each
child's ``CUDA_VISIBLE_DEVICES``, and coordinates them with immutable barrier
files.  No fork occurs after CUDA initialization.
"""

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
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        value = model_dump(mode="json")

    def json_default(item: Any) -> Any:
        nested_dump = getattr(item, "model_dump", None)
        if callable(nested_dump):
            return nested_dump(mode="json")
        raise TypeError(f"Object of type {type(item).__name__} is not JSON serializable")

    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            default=json_default,
        )
        + "\n"
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_capacity_observation_json(row: Any) -> Any:
    """Strictly decode one worker JSON observation, including enum values."""

    from sloforge.helix.characterization.gpu_capacity_calibration import (
        CapacityRequestObservation,
    )

    if not isinstance(row, dict):
        raise ValueError("capacity worker observation must be a JSON object")
    # Pydantic strict Python validation correctly rejects a bare string where
    # an Enum instance is expected.  These rows crossed the versioned JSON
    # subprocess boundary, so use the equally strict JSON decoder: it admits
    # the serialized enum value while retaining strict numeric/field checks.
    return CapacityRequestObservation.model_validate_json(_canonical_bytes(row), strict=True)


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


def _assess_sanity_guard(result: Any, *, expected_rate_rps: float) -> dict[str, Any]:
    """Apply the predeclared short stale-calibration guard, not a capacity verdict."""

    common = bool(
        result.topology.value == "gpu0-only"
        and result.configured_rate_rps == expected_rate_rps
        and result.completed_rate_span_seconds is not None
        and abs(result.completed_rate_span_seconds - 3.0) <= 1e-12
        and result.configured_arrival_rate_verified
        and result.admission_cadence_verified
        and result.routing_verified
        and result.monotonic_timestamps_verified
        and result.output_target_verified
        and result.complete_request_accounting
        and result.queue_flow_conservation_pass
        and result.p95_ttft_seconds is not None
    )
    if expected_rate_rps == 12.0:
        signals = {
            "completion_tracks_offer": (
                result.completed_rate_rps >= 0.90 * result.observed_offered_rate_rps
            ),
            "no_persistent_positive_drift": not result.queue_persistent_positive_drift,
            "p95_ttft_below_two_seconds": (
                result.p95_ttft_seconds is not None and result.p95_ttft_seconds < 2.0
            ),
            "bounded_backlog": result.maximum_queue_depth <= 25,
        }
        passed = common and all(signals.values())
        guard = "sanity_12rps_stable"
    elif expected_rate_rps == 15.0:
        signals = {
            "positive_queue_slope": result.queue_slope_requests_per_second > 0.10,
            "completed_rate_below_offered": (
                result.completed_rate_rps + 1e-12 < result.observed_offered_rate_rps
            ),
            "ttft_degradation": (
                result.p95_ttft_seconds is not None and result.p95_ttft_seconds > 0.100
            ),
            "bounded_backlog": result.maximum_queue_depth <= 25,
        }
        passed = (
            common
            and signals["bounded_backlog"]
            and any(
                signals[name]
                for name in (
                    "positive_queue_slope",
                    "completed_rate_below_offered",
                    "ttft_degradation",
                )
            )
        )
        guard = "sanity_15rps_overload"
    else:
        raise ValueError("v10 sanity guard rate must be exactly 12 or 15 rps")
    return {
        "schema_version": "sloforge.branchfabric.v10-sanity-guard-assessment/v1",
        "guard": guard,
        "expected_rate_rps": expected_rate_rps,
        "measurement_seconds": result.completed_rate_span_seconds,
        "result_verdict_ignored": result.verdict.value,
        "reason_result_verdict_ignored": (
            "three-second stale-calibration guard is intentionally not a capacity campaign"
        ),
        "signals": signals,
        "passed": passed,
        "metrics": {
            "observed_offered_rate_rps": result.observed_offered_rate_rps,
            "completed_rate_rps": result.completed_rate_rps,
            "p95_ttft_seconds": result.p95_ttft_seconds,
            "queue_depth_start": result.queue_depth_start,
            "queue_depth_end": result.queue_depth_end,
            "maximum_queue_depth": result.maximum_queue_depth,
            "queue_slope_requests_per_second": result.queue_slope_requests_per_second,
        },
    }


def _validate_measured_transaction_compilation_payloads(
    payloads: tuple[dict[str, Any], dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Require clean, complete transaction-wide compilation evidence per engine."""

    expected_fields = {
        "schema_version",
        "source",
        "role",
        "interval_start_ns",
        "interval_end_ns",
        "capture_buffer_valid",
        "events",
        "no_deferred_compilation_event",
        "passed",
    }
    by_role: dict[str, dict[str, Any]] = {}
    for payload in payloads:
        role = payload.get("role")
        observation = payload.get("measured_transaction_compilation_observation")
        if role not in {"serving", "rollout"} or not isinstance(observation, dict):
            raise RuntimeError("v10 worker omitted measured transaction compilation evidence")
        start_ns = observation.get("interval_start_ns")
        end_ns = observation.get("interval_end_ns")
        if (
            payload.get("status") != "succeeded"
            or set(observation) != expected_fields
            or observation.get("schema_version")
            != "sloforge.branchfabric.measured-transaction-compilation-observation/v1"
            or observation.get("source") != "bounded-python-logging-handler"
            or observation.get("role") != role
            or isinstance(start_ns, bool)
            or not isinstance(start_ns, int)
            or isinstance(end_ns, bool)
            or not isinstance(end_ns, int)
            or end_ns <= start_ns
            or observation.get("capture_buffer_valid") is not True
            or observation.get("events") != []
            or observation.get("no_deferred_compilation_event") is not True
            or observation.get("passed") is not True
        ):
            raise RuntimeError(f"v10 {role} measured transaction compilation gate failed")
        by_role[str(role)] = observation
    if set(by_role) != {"serving", "rollout"}:
        raise RuntimeError("v10 compilation evidence does not cover both retained engines")
    return by_role


def run_integrated_two_gpu_controller(
    *,
    config_path: Path,
    work_root: Path,
    worker_path: Path,
    model_snapshot: Path,
    repository_root: Path,
    readiness_timeout_seconds: float,
    absolute_deadline_ns: int,
) -> dict[str, Any]:
    """Run two short guards and v10 with one retained pair of vLLM engines.

    Authorization is fully sealed before allocation.  This controller only
    loads and verifies the content-addressed artifact; it has no review channel
    and cannot wait for an external approval.
    """

    started_ns = time.monotonic_ns()
    if (
        isinstance(absolute_deadline_ns, bool)
        or not isinstance(absolute_deadline_ns, int)
        or absolute_deadline_ns <= started_ns
    ):
        raise ValueError("integrated controller requires a future absolute deadline")
    overall_deadline_ns = absolute_deadline_ns

    from gpu_capacity_calibration_controller import (
        _validate_readiness_payloads,
        _validate_reset_payloads,
    )

    from sloforge.helix.characterization.gpu_capacity_calibration import (
        CapacityProbeRaw,
        ProbeTopology,
        build_probe_plan,
        evaluate_probe,
    )
    from sloforge.helix.characterization.gpu_reclamation_v10_authorization import (
        verify_authorization_file,
    )

    config = json.loads(config_path.read_text())
    if config.get("execution_mode") != "integrated-calibration-v10":
        raise ValueError("integrated controller requires its explicit execution mode")
    if not worker_path.is_file() or not model_snapshot.is_dir():
        raise FileNotFoundError("integrated worker or pinned model snapshot is absent")
    authorization_path = repository_root / str(config["authorization_artifact"])
    authorization = verify_authorization_file(
        authorization_path,
        repository_root=repository_root,
        expected_artifact_hash=str(config["authorization_artifact_hash"]),
    )
    if authorization["status"] != "APPROVED":
        raise RuntimeError("v10 authorization status is not APPROVED")
    work_root.mkdir(parents=True, exist_ok=False)
    base_config_path = work_root / "base-config.json"
    _write_new(base_config_path, config)
    _write_new(
        work_root / "authorization/verified.json",
        {
            "schema_version": "sloforge.branchfabric.v10-authorization-verification/v1",
            "status": "APPROVED",
            "artifact_reference": str(config["authorization_artifact"]),
            "artifact_hash": authorization["artifact_hash"],
            "source_evidence_hashes_verified": True,
            "remote_review_exchange_permitted": False,
            "asynchronous_review_wait_permitted": False,
            "verified_at_monotonic_ns": time.monotonic_ns(),
        },
    )
    barrier_root = work_root / "barriers"
    capacity_root = barrier_root / "capacity"
    log_root = work_root / "logs"
    capacity_root.mkdir(parents=True)
    log_root.mkdir()
    before_audit = cuda_clean_import_audit("integrated-before-inventory")
    if not before_audit["cuda_clean"]:
        raise RuntimeError("integrated controller imported a CUDA-owning module")
    inventory_before = _inventory()
    _require_no_compute_processes(_compute_processes(), phase="integrated worker preflight")
    cleanup_actions: list[dict[str, Any]] = []
    peer_access = _run_peer_probe(
        peer_probe_path=worker_path.with_name("gpu_peer_probe.py"),
        work_root=work_root,
        gpu_uuids=tuple(str(row["uuid"]) for row in inventory_before),  # type: ignore[arg-type]
        cleanup_actions=cleanup_actions,
    )
    if time.monotonic_ns() >= overall_deadline_ns:
        raise TimeoutError("integrated preflight exhausted the absolute controller deadline")
    children: list[tuple[str, subprocess.Popen[bytes], Any, Any]] = []
    controller_error: BaseException | None = None
    results: list[Any] = []
    transaction_compilation_by_role: dict[str, dict[str, Any]] = {}
    try:
        for role, gpu in zip(("serving", "rollout"), inventory_before, strict=True):
            role_root = work_root / role
            role_root.mkdir()
            stdout_handle = (log_root / f"{role}.stdout.log").open("xb")
            stderr_handle = (log_root / f"{role}.stderr.log").open("xb")
            environment = dict(os.environ)
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
        child_processes = tuple((role, child) for role, child, *_ in children)
        ready_paths = tuple(barrier_root / f"{role}.ready.json" for role, *_ in children)
        _wait_for_files(
            ready_paths,
            deadline_ns=min(
                overall_deadline_ns,
                started_ns + round(readiness_timeout_seconds * 1e9),
            ),
            phase="integrated strict readiness",
            processes=child_processes,
        )
        ready_payloads = _validate_readiness_payloads(
            payloads=tuple(json.loads(path.read_text()) for path in ready_paths),
            expected_inventory=inventory_before,
        )
        readiness_path = work_root / "readiness/both-engines-ready.json"
        _write_new(
            readiness_path,
            {
                "schema_version": "sloforge.branchfabric.both-engines-ready/v1",
                "BOTH_ENGINES_READY": True,
                "attempt_id": config["attempt_id"],
                "observed_at_monotonic_ns": time.monotonic_ns(),
                "engine_evidence": ready_payloads,
            },
        )
        sanity_assessments: list[dict[str, Any]] = []
        guard_seconds = float(config["sanity_guard_measurement_seconds"])
        for probe_index, rate in enumerate((12.0, 15.0)):
            plan = build_probe_plan(
                probe_id=f"v10-sanity-{int(rate)}rps",
                seed=int(config["seed"]),
                topology=ProbeTopology.GPU0_ONLY,
                configured_rate_rps=rate,
                start_ns=time.monotonic_ns() + 250_000_000,
                warmup_seconds=float(config["warmup_seconds"]),
                measurement_seconds=guard_seconds,
            )
            _write_new(
                capacity_root / f"probe-{probe_index:02d}.command.json",
                {
                    "schema_version": "sloforge.branchfabric.capacity-probe-command/v1",
                    "reason": (
                        "preauthorized stale-calibration sanity guard; not capacity calibration"
                    ),
                    "plan": plan,
                },
            )
            raw_paths = tuple(
                capacity_root / f"probe-{probe_index:02d}.{device}.raw.json"
                for device in ("gpu0", "gpu1")
            )
            reset_paths = tuple(
                capacity_root / f"probe-{probe_index:02d}.{device}.reset.json"
                for device in ("gpu0", "gpu1")
            )
            _wait_for_files(
                raw_paths + reset_paths,
                deadline_ns=min(
                    overall_deadline_ns,
                    time.monotonic_ns() + round(float(config["probe_timeout_seconds"]) * 1e9),
                ),
                phase=f"integrated probe {probe_index}",
                processes=child_processes,
            )
            raw_payloads = tuple(json.loads(path.read_text()) for path in raw_paths)
            resets = _validate_reset_payloads(
                payloads=tuple(json.loads(path.read_text()) for path in reset_paths),
                expected_probe_id=plan.probe_id,
            )
            observations = tuple(
                _validate_capacity_observation_json(row)
                for payload in raw_payloads
                for row in payload["observations"]
            )
            expected_tail_end_ns = plan.measurement_end_ns + round(
                float(config["tail_drain_seconds"]) * 1e9
            )
            raw = CapacityProbeRaw(
                plan=plan,
                observations=observations,
                tail_drain_end_ns=expected_tail_end_ns,
                probe_end_ns=max(int(payload["probe_end_ns"]) for payload in raw_payloads),
                tail_drain_seconds=float(config["tail_drain_seconds"]),
            )
            result = evaluate_probe(
                raw, slo_ttft_seconds=float(config["serving_slo_maximum_ttft_seconds"])
            )
            gpu1_payload = next(
                payload for payload in raw_payloads if payload.get("device") == "gpu1"
            )
            if gpu1_payload.get("observations") != []:
                raise RuntimeError("GPU1 received work during a GPU0-only sanity guard")
            assessment = _assess_sanity_guard(result, expected_rate_rps=rate)
            if assessment["passed"] is not True:
                raise RuntimeError(f"{assessment['guard']} failed: {assessment['signals']}")
            probe_root = work_root / "sanity" / f"{int(rate)}-rps"
            _write_new(probe_root / "plan.json", plan)
            _write_new(probe_root / "raw.json", raw)
            _write_new(probe_root / "worker-shards.json", raw_payloads)
            _write_new(probe_root / "result.json", result)
            _write_new(probe_root / "assessment.json", assessment)
            _write_new(probe_root / "request-state-reset.json", resets)
            results.append(result)
            sanity_assessments.append(assessment)
        _write_new(
            work_root / "sanity/sanity-result.json",
            {
                "schema_version": "sloforge.branchfabric.v10-sanity-result/v1",
                "status": "PASSED",
                "seed": int(config["seed"]),
                "guard_count": 2,
                "adaptive_capacity_search_performed": False,
                "two_gpu_capacity_probe_performed": False,
                "assessments": sanity_assessments,
            },
        )
        selected_source = next(
            item
            for item in authorization["source_calibration_artifact_hashes"]
            if item["role"] == "selected_load"
        )
        selected_source_path = repository_root / selected_source["artifact_reference"]
        selected_path = work_root / "calibration/selected-load.json"
        _write_new(selected_path, json.loads(selected_source_path.read_text()))
        selected_sha256 = _sha256(selected_path)
        if selected_sha256 != selected_source["artifact_sha256"]:
            raise RuntimeError("copied selected-load evidence differs from authorization")
        _write_new(
            work_root / "calibration/calibration-result.json",
            {
                "schema_version": "sloforge.branchfabric.v10-stale-calibration-result/v1",
                "status": "AUTHORIZED_AND_SANITY_CHECKED",
                "capacity_campaign_repeated": False,
                "authorization_artifact_hash": authorization["artifact_hash"],
                "selected_load_sha256": selected_sha256,
                "sanity_guard_assessments": sanity_assessments,
                "v10_launch_permitted": True,
                "review_required_on_live_path": False,
            },
        )
        _write_new(
            capacity_root / "final-reset.command.json",
            {
                "schema_version": "sloforge.branchfabric.integrated-final-reset-command/v1",
                "selected_load_sha256": selected_sha256,
                "authorization_artifact_hash": authorization["artifact_hash"],
                "issued_at_monotonic_ns": time.monotonic_ns(),
            },
        )
        final_reset_paths = tuple(
            capacity_root / f"final-reset.{device}.json" for device in ("gpu0", "gpu1")
        )
        _wait_for_files(
            final_reset_paths,
            deadline_ns=overall_deadline_ns,
            phase="integrated final request reset",
            processes=child_processes,
        )
        final_resets = _validate_reset_payloads(
            payloads=tuple(json.loads(path.read_text()) for path in final_reset_paths),
            expected_probe_id="final-calibration-reset",
        )
        continuity = {
            str(payload["device"]): {
                "pid": int(payload["pid"]),
                "engine_nonce": str(payload["engine_nonce"]),
                "physical_gpu_uuid": str(payload["physical_gpu_uuid"]),
            }
            for payload in ready_payloads
        }
        for payload in final_resets:
            device = str(payload["device"])
            if any(payload.get(key) != continuity[device][key] for key in continuity[device]):
                raise RuntimeError(
                    "worker/engine identity changed between readiness and final reset"
                )
        effective_config = dict(config)
        effective_config_path = work_root / "effective-config.json"
        _write_new(effective_config_path, effective_config)
        _write_new(
            barrier_root / "transaction.command.json",
            {
                "schema_version": "sloforge.branchfabric.integrated-transaction-command/v1",
                "effective_config": effective_config,
                "selection_sha256": selected_sha256,
                "authorization_artifact_hash": authorization["artifact_hash"],
                "sanity_result_sha256": _sha256(work_root / "sanity/sanity-result.json"),
                "issued_at_monotonic_ns": time.monotonic_ns(),
            },
        )
        handoff_paths = tuple(
            barrier_root / f"{role}.transaction-ready.json" for role in ("serving", "rollout")
        )
        _wait_for_files(
            handoff_paths,
            deadline_ns=overall_deadline_ns,
            phase="retained-engine transaction handoff",
            processes=child_processes,
        )
        handoffs = tuple(json.loads(path.read_text()) for path in handoff_paths)
        for payload in handoffs:
            device = str(payload["device"])
            if payload.get("passed") is not True or any(
                payload.get(key) != continuity[device][key] for key in continuity[device]
            ):
                raise RuntimeError("worker/engine identity changed before v10 transaction")
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
        while any(child.poll() is None for _, child, *_ in children):
            if time.monotonic_ns() >= overall_deadline_ns:
                raise TimeoutError("integrated capacity/v10 transaction exceeded its bound")
            failed = [
                (role, child.returncode)
                for role, child, *_ in children
                if child.poll() not in {None, 0}
            ]
            if failed:
                raise RuntimeError(f"an integrated GPU worker failed: {failed}")
            time.sleep(0.1)
        result_paths = tuple(work_root / role / "result.json" for role, *_ in children)
        _wait_for_files(
            result_paths,
            deadline_ns=overall_deadline_ns,
            phase="integrated worker results",
            processes=child_processes,
        )
        transaction_compilation_by_role = _validate_measured_transaction_compilation_payloads(
            tuple(json.loads(path.read_text()) for path in result_paths)  # type: ignore[arg-type]
        )
        _write_new(work_root / "calibration/final-request-state-reset.json", final_resets)
        manifest_artifacts = sorted(
            {
                base_config_path,
                effective_config_path,
                *result_paths,
                barrier_root / "transaction.command.json",
                *handoff_paths,
                *(path for path in (work_root / "calibration").rglob("*") if path.is_file()),
                *(path for path in (work_root / "sanity").rglob("*") if path.is_file()),
                work_root / "authorization/verified.json",
                readiness_path,
            },
            key=lambda path: str(path),
        )
        _write_new(
            work_root / "integrated-run-manifest.json",
            {
                "schema_version": "sloforge.branchfabric.integrated-run-manifest/v2",
                "attempt_id": config["attempt_id"],
                "base_config_sha256": _sha256(base_config_path),
                "selected_load_sha256": selected_sha256,
                "authorization_artifact_hash": authorization["artifact_hash"],
                "authorization_verified_before_workers": True,
                "live_review_exchange_performed": False,
                "sanity_result_sha256": _sha256(work_root / "sanity/sanity-result.json"),
                "worker_identity_by_device": continuity,
                "engine_reload_count": 0,
                "measured_transaction_compilation_free": True,
                "measured_transaction_compilation_by_role": transaction_compilation_by_role,
                "single_worker_pair": True,
                "sanity_and_transaction_same_allocation": True,
                "broad_capacity_calibration_performed": False,
                "artifacts": [
                    {
                        "relative_path": str(path.relative_to(work_root)),
                        "sha256": _sha256(path),
                    }
                    for path in manifest_artifacts
                ],
            },
        )
    except BaseException as caught:
        controller_error = caught
        if not (barrier_root / "abort.json").exists():
            _write_new(
                barrier_root / "abort.json",
                {
                    "schema_version": "sloforge.branchfabric.integrated-abort/v1",
                    "error": type(caught).__name__,
                    "issued_at_monotonic_ns": time.monotonic_ns(),
                },
            )
    finally:
        for _role, child, stdout_handle, stderr_handle in children:
            try:
                _terminate_group(child, cleanup_actions)
            finally:
                stdout_handle.close()
                stderr_handle.close()
    processes_after = _compute_processes()
    cleanup_error: BaseException | None = None
    try:
        _require_no_compute_processes(processes_after, phase="integrated worker postflight")
    except BaseException as caught:
        cleanup_error = caught
    inventory_after = _inventory()
    status = "succeeded" if controller_error is None and cleanup_error is None else "failed"
    result = {
        "schema_version": "sloforge.branchfabric.experiment-004-integrated-controller/v1",
        "status": status,
        "controller_pid": os.getpid(),
        "inventory_before": inventory_before,
        "inventory_after": inventory_after,
        "worker_pids": {role: child.pid for role, child, *_ in children},
        "worker_returncodes": {role: child.returncode for role, child, *_ in children},
        "probe_count": len(results),
        "sanity_guard_count": len(results),
        "sanity_status": "passed" if len(results) == 2 and controller_error is None else "failed",
        "measured_transaction_compilation_free": (
            set(transaction_compilation_by_role) == {"serving", "rollout"}
        ),
        "measured_transaction_compilation_by_role": transaction_compilation_by_role,
        "cuda_peer_access": peer_access,
        "cleanup_actions": cleanup_actions,
        "compute_processes_after": processes_after,
        "controller_error": None
        if controller_error is None
        else {"type": type(controller_error).__name__, "message": str(controller_error)},
        "cleanup_error": None
        if cleanup_error is None
        else {"type": type(cleanup_error).__name__, "message": str(cleanup_error)},
        "started_ns": started_ns,
        "ended_ns": time.monotonic_ns(),
    }
    _write_new(work_root / "controller-result.json", result)
    return result


__all__ = [
    "_assess_sanity_guard",
    "_validate_measured_transaction_compilation_payloads",
    "cuda_clean_import_audit",
    "run_integrated_two_gpu_controller",
    "run_two_gpu_controller",
]
