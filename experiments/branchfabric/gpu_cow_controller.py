"""CUDA-clean process controller for BranchFabric real-GPU validation.

This module is deliberately standard-library only.  The Modal Function imports it
in its long-lived interpreter, records a driver-visible baseline, and launches the
CUDA/vLLM runtime in a fresh Python child.  Process death, rather than allocator
cache APIs, is the HBM cleanup boundary.
"""

from __future__ import annotations

import json
import os
import signal
import statistics
import subprocess
import sys
import time
import traceback
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

FORBIDDEN_GPU_MODULE_PREFIXES = (
    "torch",
    "vllm",
    "triton",
    "cupy",
    "flashinfer",
    "cuda",
    "pycuda",
    "pynvml",
    "nvidia.cuda",
    "numba.cuda",
)
HBM_RECOVERY_ALLOWANCE_MIB = 1024
BASELINE_SAMPLE_COUNT = 3
POSTFLIGHT_TIMEOUT_SECONDS = 30.0
POSTFLIGHT_SAMPLE_INTERVAL_SECONDS = 1.0
STABLE_RECOVERY_SAMPLE_COUNT = 3

CommandRunner = Callable[[Sequence[str], float], dict[str, Any]]
GpuSampler = Callable[[str | None], dict[str, Any]]


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n"
    ).encode("utf-8")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _canonical_bytes(value)
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _run_command(argv: Sequence[str], timeout: float = 30.0) -> dict[str, Any]:
    started_ns = time.monotonic_ns()
    completed = subprocess.run(
        list(argv),
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return {
        "argv": list(argv),
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "started_at_utc": _utc_now(),
        "observed_at_monotonic_ns": time.monotonic_ns(),
        "duration_ns": time.monotonic_ns() - started_ns,
    }


def parse_gpu_inventory(stdout: str) -> dict[str, Any]:
    """Parse the exact one-GPU CSV shape used by the lifecycle gate."""

    rows = [row.strip() for row in stdout.splitlines() if row.strip()]
    if len(rows) != 1:
        raise ValueError(f"expected exactly one visible GPU row, observed {len(rows)}")
    values = [value.strip() for value in rows[0].split(",")]
    if len(values) != 7:
        raise ValueError("nvidia-smi GPU inventory had an unexpected shape")
    try:
        return {
            "index": int(values[0]),
            "uuid": values[1],
            "name": values[2],
            "driver_version": values[3],
            "memory_total_mib": int(values[4]),
            "memory_used_mib": int(values[5]),
            "utilization_percent": int(values[6]),
        }
    except ValueError as error:
        raise ValueError("nvidia-smi GPU inventory contained a non-integer field") from error


def parse_compute_apps(stdout: str) -> list[dict[str, Any]]:
    """Parse driver-visible CUDA compute processes without importing NVML bindings."""

    records: list[dict[str, Any]] = []
    for row in stdout.splitlines():
        stripped = row.strip()
        if not stripped or stripped.lower().startswith("no running processes"):
            continue
        values = [value.strip() for value in stripped.split(",")]
        if len(values) != 4:
            raise ValueError(f"nvidia-smi compute-app row had an unexpected shape: {row!r}")
        try:
            records.append(
                {
                    "gpu_uuid": values[0],
                    "host_pid": int(values[1]),
                    "process_name": values[2],
                    "used_gpu_memory_mib": int(values[3]),
                }
            )
        except ValueError as error:
            raise ValueError(f"nvidia-smi compute-app row was invalid: {row!r}") from error
    return records


def sample_driver_gpu(
    expected_uuid: str | None = None,
    *,
    command_runner: CommandRunner = _run_command,
) -> dict[str, Any]:
    """Take one driver-visible sample using only nvidia-smi subprocesses."""

    observed_at_monotonic_ns = time.monotonic_ns()
    observed_at_utc = _utc_now()
    inventory_command = command_runner(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,driver_version,memory.total,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        10.0,
    )
    process_command = command_runner(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ],
        10.0,
    )
    if inventory_command["returncode"] != 0 or process_command["returncode"] != 0:
        raise RuntimeError("nvidia-smi driver-visible sampling failed")
    gpu = parse_gpu_inventory(str(inventory_command["stdout"]))
    if expected_uuid is not None and gpu["uuid"] != expected_uuid:
        raise RuntimeError(
            f"visible GPU identity changed from {expected_uuid!r} to {gpu['uuid']!r}"
        )
    return {
        "observed_at_utc": observed_at_utc,
        "observed_at_monotonic_ns": observed_at_monotonic_ns,
        "gpu": gpu,
        "compute_processes": parse_compute_apps(str(process_command["stdout"])),
        "commands": {
            "inventory": inventory_command,
            "compute_apps": process_command,
        },
    }


def forbidden_import_audit(*, stage: str) -> dict[str, Any]:
    """Inspect the current interpreter without importing any candidate module."""

    loaded = sorted(
        name
        for name in sys.modules
        if any(
            name == prefix or name.startswith(prefix + ".")
            for prefix in FORBIDDEN_GPU_MODULE_PREFIXES
        )
    )
    return {
        "schema_version": "sloforge.branchfabric.forbidden-gpu-import-audit/v1",
        "stage": stage,
        "audited_at_utc": _utc_now(),
        "audited_at_monotonic_ns": time.monotonic_ns(),
        "controller_pid": os.getpid(),
        "forbidden_prefixes": list(FORBIDDEN_GPU_MODULE_PREFIXES),
        "loaded_forbidden_modules": loaded,
        "cuda_clean": not loaded,
        "source": "sys.modules",
    }


def _process_tree_procfs() -> list[dict[str, Any]]:
    root = Path("/proc")
    if not root.is_dir():
        raise FileNotFoundError("/proc is unavailable")
    records: list[dict[str, Any]] = []
    for entry in root.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            stat = (entry / "stat").read_text()
            close = stat.rfind(")")
            if close < 0:
                continue
            fields = stat[close + 2 :].split()
            cmdline_raw = (entry / "cmdline").read_bytes()
            command = " ".join(
                part.decode("utf-8", errors="replace") for part in cmdline_raw.split(b"\0") if part
            )
            records.append(
                {
                    "pid": int(entry.name),
                    "ppid": int(fields[1]),
                    "pgid": int(fields[2]),
                    "sid": int(fields[3]),
                    "state": fields[0],
                    "start_token": fields[19],
                    "command": command,
                }
            )
        except (FileNotFoundError, PermissionError, ProcessLookupError, ValueError, IndexError):
            continue
    return sorted(records, key=lambda item: int(item["pid"]))


def _process_tree_ps() -> list[dict[str, Any]]:
    completed = subprocess.run(
        ["ps", "-axo", "pid=,ppid=,pgid=,sess=,state=,command="],
        check=False,
        capture_output=True,
        text=True,
        timeout=10.0,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"ps process-tree query failed: {completed.stderr.strip()}")
    records: list[dict[str, Any]] = []
    for row in completed.stdout.splitlines():
        values = row.strip().split(maxsplit=5)
        if len(values) < 5:
            continue
        try:
            records.append(
                {
                    "pid": int(values[0]),
                    "ppid": int(values[1]),
                    "pgid": int(values[2]),
                    "sid": int(values[3]),
                    "state": values[4],
                    "start_token": None,
                    "command": values[5] if len(values) == 6 else "",
                }
            )
        except ValueError:
            continue
    return sorted(records, key=lambda item: int(item["pid"]))


def process_tree_snapshot() -> dict[str, Any]:
    try:
        records = _process_tree_procfs()
        source = "procfs"
    except FileNotFoundError:
        records = _process_tree_ps()
        source = "ps"
    return {
        "schema_version": "sloforge.branchfabric.process-tree/v1",
        "observed_at_utc": _utc_now(),
        "observed_at_monotonic_ns": time.monotonic_ns(),
        "source": source,
        "controller_pid": os.getpid(),
        "processes": records,
    }


def _descendant_pids(processes: Sequence[dict[str, Any]], root_pid: int) -> set[int]:
    by_parent: dict[int, set[int]] = {}
    for process in processes:
        by_parent.setdefault(int(process["ppid"]), set()).add(int(process["pid"]))
    descendants: set[int] = set()
    frontier = [root_pid]
    while frontier:
        parent = frontier.pop()
        for child in by_parent.get(parent, set()):
            if child in descendants:
                continue
            descendants.add(child)
            frontier.append(child)
    return descendants


def _owned_processes(
    snapshot: dict[str, Any],
    *,
    child_pid: int,
    child_pgid: int,
    tracked: dict[int, str | None],
) -> list[dict[str, Any]]:
    processes = list(snapshot["processes"])
    descendants = _descendant_pids(processes, child_pid)
    owned: list[dict[str, Any]] = []
    for process in processes:
        pid = int(process["pid"])
        start_token = process.get("start_token")
        previously_tracked = pid in tracked
        tracked_start_token = tracked.get(pid)
        same_tracked_process = (
            previously_tracked
            and tracked_start_token is not None
            and start_token is not None
            and tracked_start_token == start_token
        )
        candidate = (
            (pid == child_pid and not previously_tracked)
            or pid in descendants
            or int(process["pgid"]) == child_pgid
            or same_tracked_process
        )
        if not candidate:
            continue
        if (
            previously_tracked
            and tracked_start_token is not None
            and start_token is not None
            and tracked_start_token != start_token
        ):
            # The numeric PID has been recycled.  Never claim or signal it as
            # experiment-owned based only on its former identity.
            continue
        owned.append(process)
        if not previously_tracked:
            tracked[pid] = start_token if isinstance(start_token, str) else None
    return sorted(owned, key=lambda item: int(item["pid"]))


def _wait_for_owned_exit(
    *,
    child_pid: int,
    child_pgid: int,
    tracked: dict[int, str | None],
    timeout_s: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    deadline = time.monotonic() + timeout_s
    last = process_tree_snapshot()
    while True:
        owned = _owned_processes(
            last,
            child_pid=child_pid,
            child_pgid=child_pgid,
            tracked=tracked,
        )
        if not owned or time.monotonic() >= deadline:
            return owned, last
        time.sleep(0.1)
        last = process_tree_snapshot()


def _signal_exact_processes(
    processes: Sequence[dict[str, Any]], sig: signal.Signals
) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    for process in sorted(processes, key=lambda item: int(item["pid"]), reverse=True):
        pid = int(process["pid"])
        expected_start_token = process.get("start_token")
        pidfd: int | None = None
        try:
            if hasattr(os, "pidfd_open") and hasattr(signal, "pidfd_send_signal"):
                pidfd = os.pidfd_open(pid, 0)
                if isinstance(expected_start_token, str):
                    current = next(
                        (item for item in _process_tree_procfs() if int(item["pid"]) == pid),
                        None,
                    )
                    if current is None or current.get("start_token") != expected_start_token:
                        continue
                signal.pidfd_send_signal(pidfd, sig)
                signal_method = "pidfd_send_signal"
            else:
                if isinstance(expected_start_token, str):
                    current = next(
                        (item for item in _process_tree_procfs() if int(item["pid"]) == pid),
                        None,
                    )
                    if current is None or current.get("start_token") != expected_start_token:
                        continue
                os.kill(pid, sig)
                signal_method = "kill"
            actions.append(
                {
                    "pid": pid,
                    "pgid": int(process["pgid"]),
                    "start_token": expected_start_token,
                    "command": process.get("command", ""),
                    "signal": sig.name,
                    "signal_method": signal_method,
                    "identity_verified": isinstance(expected_start_token, str),
                    "at_utc": _utc_now(),
                }
            )
        except ProcessLookupError:
            continue
        finally:
            if pidfd is not None:
                os.close(pidfd)
    return actions


def evaluate_cleanup_samples(
    samples: Sequence[dict[str, Any]],
    *,
    threshold_mib: int,
    required_consecutive: int = STABLE_RECOVERY_SAMPLE_COUNT,
) -> dict[str, Any]:
    consecutive = 0
    maximum_consecutive = 0
    winning_start: int | None = None
    winning_end: int | None = None
    for index, sample in enumerate(samples):
        qualifies = (
            sample.get("error") is None
            and int(sample["gpu"]["memory_used_mib"]) <= threshold_mib
            and sample.get("compute_processes") == []
        )
        if qualifies:
            consecutive += 1
            maximum_consecutive = max(maximum_consecutive, consecutive)
            if consecutive == 1:
                winning_start = index
            if consecutive >= required_consecutive:
                winning_end = index
                break
        else:
            consecutive = 0
            winning_start = None
    return {
        "passed": winning_end is not None,
        "required_consecutive_samples": required_consecutive,
        "winning_sample_start_index": winning_start if winning_end is not None else None,
        "winning_sample_end_index": winning_end,
        "maximum_consecutive_qualifying_samples": maximum_consecutive,
    }


def _sample_or_error(sampler: GpuSampler, expected_uuid: str) -> dict[str, Any]:
    try:
        return sampler(expected_uuid)
    except Exception as error:  # Preserve the full bounded sampling history.
        return {
            "observed_at_utc": _utc_now(),
            "observed_at_monotonic_ns": time.monotonic_ns(),
            "error": {"type": type(error).__name__, "message": str(error)},
        }


def _semantic_child_valid(
    child_manifest: dict[str, Any] | None,
    *,
    expected_pid: int | None = None,
    expected_pgid: int | None = None,
) -> bool:
    if not isinstance(child_manifest, dict) or child_manifest.get("status") != "succeeded":
        return False
    manifest_pid = child_manifest.get("child_pid", child_manifest.get("pid"))
    manifest_pgid = child_manifest.get("child_pgid", child_manifest.get("pgid"))
    if expected_pid is not None and manifest_pid != expected_pid:
        return False
    if expected_pgid is not None and manifest_pgid != expected_pgid:
        return False
    invariants = child_manifest.get("semantic_invariants")
    return (
        isinstance(invariants, dict)
        and bool(invariants)
        and all(value is True for value in invariants.values())
        and child_manifest.get("final_runtime_assigned_kv_bytes") == 0
    )


def run_controller(
    *,
    config_payload: dict[str, Any],
    work_root: Path,
    worker_path: Path,
    model_snapshot: Path,
    gpu_sampler: GpuSampler = sample_driver_gpu,
    postflight_timeout_s: float = POSTFLIGHT_TIMEOUT_SECONDS,
    baseline_interval_s: float = 0.2,
    monitor_interval_s: float = 1.0,
) -> dict[str, Any]:
    """Run one exact child and enforce the driver-visible cleanup gate."""

    controller_started_ns = time.monotonic_ns()
    controller_started_utc = _utc_now()
    lifecycle = work_root / "lifecycle"
    logs = work_root / "logs"
    wrapper = work_root / "wrapper"
    for directory in (lifecycle, logs, wrapper):
        directory.mkdir(parents=True, exist_ok=False)
    error_record: dict[str, Any] | None = None
    child: subprocess.Popen[str] | None = None
    child_pid: int | None = None
    child_pgid: int | None = None
    child_started_ns: int | None = None
    child_started_utc: str | None = None
    child_exit_ns: int | None = None
    child_exit_utc: str | None = None
    return_code: int | None = None
    timed_out = False
    termination_actions: list[dict[str, Any]] = []
    forced_termination_required = False
    tracked: dict[int, str | None] = {}
    baseline_samples: list[dict[str, Any]] = []
    child_hbm_samples: list[dict[str, Any]] = []
    postflight_samples: list[dict[str, Any]] = []
    process_during_samples: list[dict[str, Any]] = []
    baseline_median_mib: int | None = None
    threshold_mib: int | None = None
    actual_gpu: dict[str, Any] | None = None
    expected_uuid: str | None = None
    child_manifest: dict[str, Any] | None = None
    worker_argv: list[str] | None = None
    process_after = process_tree_snapshot()
    import_audits = [forbidden_import_audit(stage="controller_entry_before_baseline")]
    _write_json(lifecycle / "process-tree-before.json", process_after)
    try:
        if not import_audits[0]["cuda_clean"]:
            raise RuntimeError(
                "CUDA-clean controller import audit failed before baseline: "
                + ", ".join(import_audits[0]["loaded_forbidden_modules"])
            )
        for index in range(BASELINE_SAMPLE_COUNT):
            sample = gpu_sampler(expected_uuid)
            if sample.get("error") is not None:
                raise RuntimeError(f"baseline GPU sample failed: {sample['error']}")
            gpu = sample["gpu"]
            if expected_uuid is None:
                expected_uuid = str(gpu["uuid"])
                actual_gpu = dict(gpu)
            if "A100" not in str(gpu["name"]) or int(gpu["memory_total_mib"]) < 80_000:
                raise RuntimeError(f"Modal returned unsupported GPU hardware: {gpu}")
            if sample["compute_processes"]:
                raise RuntimeError(
                    f"visible GPU was not exclusive before child launch: {sample['compute_processes']}"
                )
            sample["baseline_sample_index"] = index
            baseline_samples.append(sample)
            if index + 1 < BASELINE_SAMPLE_COUNT:
                time.sleep(baseline_interval_s)
        baseline_median_mib = int(
            statistics.median(int(sample["gpu"]["memory_used_mib"]) for sample in baseline_samples)
        )
        threshold_mib = baseline_median_mib + HBM_RECOVERY_ALLOWANCE_MIB
        if expected_uuid is None:
            raise AssertionError("baseline sampling did not establish a GPU UUID")
        _write_json(
            lifecycle / "hbm-baseline.json",
            {
                "schema_version": "sloforge.branchfabric.hbm-baseline/v1",
                "samples": baseline_samples,
                "baseline_samples_mib": [
                    int(sample["gpu"]["memory_used_mib"]) for sample in baseline_samples
                ],
                "baseline_median_mib": baseline_median_mib,
                "cleanup_threshold_mib": threshold_mib,
                "allowance_mib": HBM_RECOVERY_ALLOWANCE_MIB,
                "gpu_uuid": expected_uuid,
            },
        )
        import_audits.append(forbidden_import_audit(stage="immediately_before_child_launch"))
        if not import_audits[-1]["cuda_clean"]:
            raise RuntimeError(
                "CUDA-clean controller import audit failed before child launch: "
                + ", ".join(import_audits[-1]["loaded_forbidden_modules"])
            )
        controller_input = wrapper / "controller-input.json"
        _write_json(controller_input, config_payload)
        child_manifest_path = lifecycle / "child-manifest.json"
        event_log_path = lifecycle / "child-events.jsonl"
        worker_argv = [
            sys.executable,
            str(worker_path),
            "--config",
            str(controller_input),
            "--work-root",
            str(work_root),
            "--model-snapshot",
            str(model_snapshot),
            "--gpu-uuid",
            str(expected_uuid),
            "--event-log",
            str(event_log_path),
            "--child-manifest",
            str(child_manifest_path),
        ]
        environment = os.environ.copy()
        environment["PYTHONUNBUFFERED"] = "1"
        stdout_path = logs / "child.stdout.log"
        stderr_path = logs / "child.stderr.log"
        child_started_ns = time.monotonic_ns()
        child_started_utc = _utc_now()
        with (
            stdout_path.open("x", encoding="utf-8") as stdout_handle,
            stderr_path.open("x", encoding="utf-8") as stderr_handle,
        ):
            child = subprocess.Popen(
                worker_argv,
                cwd=str(worker_path.parent),
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=stdout_handle,
                stderr=stderr_handle,
                text=True,
                start_new_session=True,
            )
            child_pid = child.pid
            # start_new_session=True makes the child the new process-group
            # leader.  Deriving PGID from the returned PID avoids a race with
            # a worker that validates its input and exits before getpgid().
            child_pgid = child_pid
            initial_during = process_tree_snapshot()
            _owned_processes(
                initial_during,
                child_pid=child_pid,
                child_pgid=child_pgid,
                tracked=tracked,
            )
            process_during_samples.append(initial_during)
            deadline = time.monotonic() + float(
                int(config_payload.get("initialization_timeout_seconds", 1200))
                + int(config_payload.get("maximum_wall_seconds", 3600))
                + int(config_payload.get("cleanup_timeout_seconds", 60))
            )
            next_sample = 0.0
            while child.poll() is None:
                now = time.monotonic()
                if now >= deadline:
                    timed_out = True
                    snapshot = process_tree_snapshot()
                    owned = _owned_processes(
                        snapshot,
                        child_pid=child_pid,
                        child_pgid=child_pgid,
                        tracked=tracked,
                    )
                    termination_actions.extend(_signal_exact_processes(owned, signal.SIGTERM))
                    forced_termination_required = bool(termination_actions)
                    try:
                        child.wait(timeout=10.0)
                    except subprocess.TimeoutExpired:
                        snapshot = process_tree_snapshot()
                        owned = _owned_processes(
                            snapshot,
                            child_pid=child_pid,
                            child_pgid=child_pgid,
                            tracked=tracked,
                        )
                        termination_actions.extend(_signal_exact_processes(owned, signal.SIGKILL))
                        forced_termination_required = bool(termination_actions)
                        child.wait(timeout=5.0)
                    break
                if now >= next_sample:
                    sample = _sample_or_error(gpu_sampler, expected_uuid)
                    sample["phase"] = "child_live"
                    child_hbm_samples.append(sample)
                    snapshot = process_tree_snapshot()
                    owned = _owned_processes(
                        snapshot,
                        child_pid=child_pid,
                        child_pgid=child_pgid,
                        tracked=tracked,
                    )
                    if len(process_during_samples) < 512:
                        process_during_samples.append({**snapshot, "owned_processes": owned})
                    next_sample = now + monitor_interval_s
                remaining_sleep_s = max(0.0, min(0.1, next_sample - time.monotonic()))
                if remaining_sleep_s:
                    time.sleep(remaining_sleep_s)
            return_code = child.wait(timeout=5.0)
            child_exit_ns = time.monotonic_ns()
            child_exit_utc = _utc_now()
            stdout_handle.flush()
            os.fsync(stdout_handle.fileno())
            stderr_handle.flush()
            os.fsync(stderr_handle.fileno())
        owned_remaining, process_after = _wait_for_owned_exit(
            child_pid=child_pid,
            child_pgid=child_pgid,
            tracked=tracked,
            timeout_s=5.0,
        )
        if owned_remaining:
            termination_actions.extend(_signal_exact_processes(owned_remaining, signal.SIGTERM))
            forced_termination_required = True
            owned_remaining, process_after = _wait_for_owned_exit(
                child_pid=child_pid,
                child_pgid=child_pgid,
                tracked=tracked,
                timeout_s=10.0,
            )
        if owned_remaining:
            termination_actions.extend(_signal_exact_processes(owned_remaining, signal.SIGKILL))
            forced_termination_required = True
            owned_remaining, process_after = _wait_for_owned_exit(
                child_pid=child_pid,
                child_pgid=child_pgid,
                tracked=tracked,
                timeout_s=5.0,
            )
        if owned_remaining:
            raise RuntimeError(
                "exact experiment-owned processes survived SIGKILL: "
                + ", ".join(str(item["pid"]) for item in owned_remaining)
            )
        if child_manifest_path.is_file():
            child_manifest = json.loads(child_manifest_path.read_text())
        postflight_deadline = time.monotonic() + postflight_timeout_s
        next_postflight = time.monotonic()
        while time.monotonic() <= postflight_deadline:
            now = time.monotonic()
            if now < next_postflight:
                time.sleep(min(next_postflight - now, 0.1))
                continue
            sample = _sample_or_error(gpu_sampler, expected_uuid)
            sample["elapsed_since_child_exit_ms"] = (
                time.monotonic_ns() - child_exit_ns
            ) / 1_000_000
            postflight_samples.append(sample)
            evaluation = evaluate_cleanup_samples(
                postflight_samples,
                threshold_mib=threshold_mib,
            )
            if evaluation["passed"]:
                break
            next_postflight += POSTFLIGHT_SAMPLE_INTERVAL_SECONDS
        cleanup_evaluation = evaluate_cleanup_samples(
            postflight_samples,
            threshold_mib=threshold_mib,
        )
        import_audits.append(forbidden_import_audit(stage="after_child_exit_and_hbm_postflight"))
        _write_json(
            lifecycle / "forbidden-import-audit.json",
            {
                "schema_version": "sloforge.branchfabric.forbidden-gpu-import-audits/v1",
                "audits": import_audits,
                "cuda_clean": all(item["cuda_clean"] for item in import_audits),
            },
        )
        start_index = cleanup_evaluation["winning_sample_start_index"]
        end_index = cleanup_evaluation["winning_sample_end_index"]
        time_to_recovery_ms = (
            None
            if start_index is None
            else float(postflight_samples[int(start_index)]["elapsed_since_child_exit_ms"])
        )
        stable_confirmation_ms = (
            None
            if end_index is None
            else float(postflight_samples[int(end_index)]["elapsed_since_child_exit_ms"])
        )
        cleanup_gate_pass = bool(
            cleanup_evaluation["passed"]
            and not forced_termination_required
            and return_code == 0
            and _semantic_child_valid(
                child_manifest,
                expected_pid=child_pid,
                expected_pgid=child_pgid,
            )
            and all(item["cuda_clean"] for item in import_audits)
        )
        cleanup_gate = {
            "schema_version": "sloforge.branchfabric.gpu-runtime-cleanup-gate/v1",
            "cleanup_gate": "PASS" if cleanup_gate_pass else "FAIL",
            "GPU_RUNTIME_LIFECYCLE_CLEAN": cleanup_gate_pass,
            "process_hbm_recovery": "verified" if cleanup_evaluation["passed"] else "failed",
            "baseline_median_mib": baseline_median_mib,
            "cleanup_threshold_mib": threshold_mib,
            "allowance_mib": HBM_RECOVERY_ALLOWANCE_MIB,
            "stable_sample_evaluation": cleanup_evaluation,
            "time_to_hbm_recovery_ms": time_to_recovery_ms,
            "time_to_stable_confirmation_ms": stable_confirmation_ms,
            "child_return_code": return_code,
            "child_semantics_valid": _semantic_child_valid(
                child_manifest,
                expected_pid=child_pid,
                expected_pgid=child_pgid,
            ),
            "parent_cuda_clean": all(item["cuda_clean"] for item in import_audits),
            "no_owned_descendant_remained": True,
            "no_compute_process_remained_in_winning_samples": bool(cleanup_evaluation["passed"]),
            "forced_gpu_process_kill_required": forced_termination_required,
            "termination_actions": termination_actions,
            "timed_out": timed_out,
        }
        _write_json(lifecycle / "cleanup-gate.json", cleanup_gate)
    except Exception as error:
        error_record = {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
        }
        remaining_after_error: list[dict[str, Any]] = []
        if child is not None and child_pid is not None and child_pgid is not None:
            if child.poll() is None:
                snapshot = process_tree_snapshot()
                owned = _owned_processes(
                    snapshot,
                    child_pid=child_pid,
                    child_pgid=child_pgid,
                    tracked=tracked,
                )
                termination_actions.extend(_signal_exact_processes(owned, signal.SIGTERM))
                forced_termination_required = bool(termination_actions)
                try:
                    child.wait(timeout=10.0)
                except subprocess.TimeoutExpired:
                    snapshot = process_tree_snapshot()
                    owned = _owned_processes(
                        snapshot,
                        child_pid=child_pid,
                        child_pgid=child_pgid,
                        tracked=tracked,
                    )
                    termination_actions.extend(_signal_exact_processes(owned, signal.SIGKILL))
                    forced_termination_required = bool(termination_actions)
                    child.wait(timeout=5.0)
            return_code = child.returncode
            if child_exit_ns is None:
                child_exit_ns = time.monotonic_ns()
                child_exit_utc = _utc_now()
            remaining_after_error, process_after = _wait_for_owned_exit(
                child_pid=child_pid,
                child_pgid=child_pgid,
                tracked=tracked,
                timeout_s=5.0,
            )
            if remaining_after_error:
                termination_actions.extend(
                    _signal_exact_processes(remaining_after_error, signal.SIGTERM)
                )
                forced_termination_required = True
                remaining_after_error, process_after = _wait_for_owned_exit(
                    child_pid=child_pid,
                    child_pgid=child_pgid,
                    tracked=tracked,
                    timeout_s=10.0,
                )
            if remaining_after_error:
                termination_actions.extend(
                    _signal_exact_processes(remaining_after_error, signal.SIGKILL)
                )
                forced_termination_required = True
                remaining_after_error, process_after = _wait_for_owned_exit(
                    child_pid=child_pid,
                    child_pgid=child_pgid,
                    tracked=tracked,
                    timeout_s=5.0,
                )
            if (
                expected_uuid is not None
                and threshold_mib is not None
                and child_exit_ns is not None
                and not postflight_samples
            ):
                postflight_deadline = time.monotonic() + postflight_timeout_s
                next_postflight = time.monotonic()
                while time.monotonic() <= postflight_deadline:
                    now = time.monotonic()
                    if now < next_postflight:
                        time.sleep(min(next_postflight - now, 0.1))
                        continue
                    sample = _sample_or_error(gpu_sampler, expected_uuid)
                    sample["elapsed_since_child_exit_ms"] = (
                        time.monotonic_ns() - child_exit_ns
                    ) / 1_000_000
                    postflight_samples.append(sample)
                    if evaluate_cleanup_samples(postflight_samples, threshold_mib=threshold_mib)[
                        "passed"
                    ]:
                        break
                    next_postflight += POSTFLIGHT_SAMPLE_INTERVAL_SECONDS
        else:
            process_after = process_tree_snapshot()
        if child is not None and not any(
            item["stage"] == "after_child_exit_and_hbm_postflight" for item in import_audits
        ):
            import_audits.append(
                forbidden_import_audit(stage="after_child_exit_and_hbm_postflight")
            )
        if not (lifecycle / "forbidden-import-audit.json").exists():
            _write_json(
                lifecycle / "forbidden-import-audit.json",
                {
                    "schema_version": "sloforge.branchfabric.forbidden-gpu-import-audits/v1",
                    "audits": import_audits,
                    "cuda_clean": all(item["cuda_clean"] for item in import_audits),
                },
            )
        if not (lifecycle / "cleanup-gate.json").exists():
            _write_json(
                lifecycle / "cleanup-gate.json",
                {
                    "schema_version": "sloforge.branchfabric.gpu-runtime-cleanup-gate/v1",
                    "cleanup_gate": "FAIL",
                    "GPU_RUNTIME_LIFECYCLE_CLEAN": False,
                    "error": error_record,
                    "process_hbm_recovery": (
                        "unavailable"
                        if threshold_mib is None or not postflight_samples
                        else (
                            "verified"
                            if evaluate_cleanup_samples(
                                postflight_samples, threshold_mib=threshold_mib
                            )["passed"]
                            else "failed"
                        )
                    ),
                    "no_owned_descendant_remained": not remaining_after_error,
                    "parent_cuda_clean": all(item["cuda_clean"] for item in import_audits),
                    "forced_gpu_process_kill_required": forced_termination_required,
                    "termination_actions": termination_actions,
                    "timed_out": timed_out,
                },
            )
    finally:
        if not (lifecycle / "hbm-baseline.json").exists():
            _write_json(
                lifecycle / "hbm-baseline.json",
                {
                    "schema_version": "sloforge.branchfabric.hbm-baseline/v1",
                    "samples": baseline_samples,
                    "baseline_samples_mib": [
                        int(sample["gpu"]["memory_used_mib"])
                        for sample in baseline_samples
                        if "gpu" in sample
                    ],
                    "baseline_median_mib": baseline_median_mib,
                    "cleanup_threshold_mib": threshold_mib,
                },
            )
        _write_json(
            lifecycle / "hbm-postflight.json",
            {
                "schema_version": "sloforge.branchfabric.hbm-postflight/v1",
                "child_live_samples": child_hbm_samples,
                "post_child_exit_samples": postflight_samples,
                "sample_interval_seconds": POSTFLIGHT_SAMPLE_INTERVAL_SECONDS,
                "maximum_wait_seconds": postflight_timeout_s,
            },
        )
        _write_json(
            lifecycle / "process-tree-during.json",
            {
                "schema_version": "sloforge.branchfabric.process-tree-series/v1",
                "child_pid": child_pid,
                "child_pgid": child_pgid,
                "tracked_owned_processes": [
                    {"pid": pid, "start_token": start_token}
                    for pid, start_token in sorted(tracked.items())
                ],
                "samples": process_during_samples,
            },
        )
        _write_json(lifecycle / "process-tree-after.json", process_after)
    cleanup_gate_payload = json.loads((lifecycle / "cleanup-gate.json").read_text())
    controller_end_ns = time.monotonic_ns()
    controller_manifest = {
        "schema_version": "sloforge.branchfabric.gpu-cow-controller-manifest/v1",
        "attempt_id": config_payload.get("attempt_id"),
        "controller_pid": os.getpid(),
        "controller_pgid": os.getpgid(0),
        "controller_start_utc": controller_started_utc,
        "controller_end_utc": _utc_now(),
        "controller_duration_seconds": (controller_end_ns - controller_started_ns) / 1e9,
        "child_pid": child_pid,
        "child_pgid": child_pgid,
        "child_start_utc": child_started_utc,
        "child_end_utc": child_exit_utc,
        "child_duration_seconds": (
            None
            if child_started_ns is None or child_exit_ns is None
            else (child_exit_ns - child_started_ns) / 1e9
        ),
        "child_return_code": return_code,
        "child_stdout_path": "logs/child.stdout.log",
        "child_stderr_path": "logs/child.stderr.log",
        "child_manifest_path": "lifecycle/child-manifest.json",
        "worker_argv": worker_argv,
        "baseline_median_mib": baseline_median_mib,
        "cleanup_threshold_mib": threshold_mib,
        "parent_cuda_clean": all(item["cuda_clean"] for item in import_audits),
        "forced_gpu_process_kill_required": forced_termination_required,
        "termination_actions": termination_actions,
        "status": "succeeded" if cleanup_gate_payload["GPU_RUNTIME_LIFECYCLE_CLEAN"] else "failed",
        "error": error_record,
    }
    _write_json(lifecycle / "controller-manifest.json", controller_manifest)
    return {
        "status": controller_manifest["status"],
        "actual_gpu": actual_gpu,
        "summary": None if child_manifest is None else child_manifest.get("summary"),
        "child_manifest": child_manifest,
        "cleanup_gate": cleanup_gate_payload,
        "controller_manifest": controller_manifest,
        "error": error_record or (None if child_manifest is None else child_manifest.get("error")),
    }


__all__ = [
    "BASELINE_SAMPLE_COUNT",
    "FORBIDDEN_GPU_MODULE_PREFIXES",
    "HBM_RECOVERY_ALLOWANCE_MIB",
    "POSTFLIGHT_TIMEOUT_SECONDS",
    "STABLE_RECOVERY_SAMPLE_COUNT",
    "evaluate_cleanup_samples",
    "forbidden_import_audit",
    "parse_compute_apps",
    "parse_gpu_inventory",
    "process_tree_snapshot",
    "run_controller",
    "sample_driver_gpu",
]
