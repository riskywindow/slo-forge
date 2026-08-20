"""CUDA-clean, fail-closed controller for the one integrated v11 transaction.

This module deliberately does not import CUDA, PyTorch, vLLM, the v10 worker,
or the v10 serving implementation.  It retains the frozen experiment's
two-process and immutable-file protocol while binding the paid launch to the
strict v11 configuration and its content-addressed scientific evidence.
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import math
import os
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Mapping, Sequence
from contextlib import suppress
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path
from typing import Any

from gpu_reclamation_integrated_worker_v11 import (
    build_live_v10_config,
    expanded_runtime_config,
    validate_sanity_guard_pair,
)

from sloforge.helix.characterization.gpu_reclamation_v11_methodology import (
    V11_INTEGRATED_RESERVATION_WALL_SECONDS,
    Experiment004V11IntegratedConfig,
    validate_bound_artifact,
)

FORBIDDEN_GPU_MODULE_PREFIXES = (
    "torch",
    "vllm",
    "triton",
    "cupy",
    "pynvml",
    "cuda",
    "numba.cuda",
)
ROLES = ("serving", "rollout")
DEVICES = ("gpu0", "gpu1")
SANITY_RATES_RPS = (12.0, 15.0)
ABSOLUTE_WALL_SECONDS = 588.0
ORIGINAL_INTEGRATED_ATTEMPT_ID = "exp004-v11-integrated-s41-a"
PRIOR_INVALID_INTEGRATED_ATTEMPT_ID = "exp004-v11-integrated-s41-b"
REPLACEMENT_INTEGRATED_ATTEMPT_ID = "exp004-v11-integrated-s41-c"
INTEGRATED_PRELAUNCH_BINDINGS = {
    "methodology": "python/sloforge/helix/characterization/gpu_reclamation_v11_methodology.py",
    "worker": "experiments/branchfabric/gpu_reclamation_integrated_worker_v11.py",
    "controller": "experiments/branchfabric/gpu_reclamation_integrated_controller_v11.py",
    "launcher": "experiments/branchfabric/modal_gpu_reclamation_integrated_v11.py",
    "trigger": "experiments/branchfabric/gpu_reclamation_integrated_trigger_v11.py",
    "worker_test": "tests/python/test_gpu_reclamation_integrated_worker_v11.py",
    "controller_test": "tests/python/test_gpu_reclamation_integrated_controller_v11.py",
    "launcher_test": "tests/python/test_modal_gpu_reclamation_integrated_v11.py",
    "trigger_test": "tests/python/test_gpu_reclamation_integrated_trigger_v11.py",
    "trigger_race_test": "tests/python/test_gpu_reclamation_integrated_trigger_v11_races.py",
    "cross_runtime_review": (
        "artifacts/branchfabric/gpu-validation/experiment-004/v11/reviews/"
        "agent03-integrated-cross-runtime-audit.json"
    ),
    "make_check": (
        "artifacts/branchfabric/gpu-validation/experiment-004/v11-final/offline-fixes/"
        "make-check.json"
    ),
}
REPLACEMENT_EVIDENCE_BINDINGS = {
    "prior_attempt_cleanup": (
        "artifacts/branchfabric/gpu-validation/experiment-004/v11/runtime/"
        "exp004-v11-integrated-s41-b/postflight-cleanup-and-settlement.json"
    ),
}
FINAL_PRE_GPU_REVIEW_ROLES = {
    1: "early-trigger-implementation",
    2: "trigger-race-review",
    3: "process-cleanup-implementation",
    4: "process-lifecycle-review",
    5: "classification-logic-review",
    6: "scientific-validity-review",
    7: "gpu-budget-review",
    8: "integrated-methodology-review",
}
PROCESS_LIFECYCLE_PHASES = (
    "RUNNING",
    "QUIESCE",
    "ENGINE_STOP",
    "WORKER_STOP",
    "CHILD_REAP",
    "PGID_EMPTY",
    "CUDA_RELEASED",
    "FUNCTION_RETURN",
)
PROFILER_COMMAND_TOKENS = ("nsys", "ncu", "nvprof")
RESOURCE_TRACKER_COMMAND_TOKEN = "multiprocessing.resource_tracker"
MAX_RECLAIM_TRIGGER_REACTION_NS = 100_000_000
MAX_SCIENTIFIC_TRIGGER_DEPTH = 30
PR_SET_CHILD_SUBREAPER = 36
PR_GET_CHILD_SUBREAPER = 37


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _canonical_bytes(value: Any) -> bytes:
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        value = dump(mode="json")

    def default(item: Any) -> Any:
        nested = getattr(item, "model_dump", None)
        if callable(nested):
            return nested(mode="json")
        raise TypeError(f"Object of type {type(item).__name__} is not JSON serializable")

    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            default=default,
        )
        + "\n"
    ).encode()


def _write_new(path: Path, value: Any) -> None:
    """Publish one immutable barrier using an exclusive hard link."""

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
        "schema_version": "sloforge.branchfabric.cuda-clean-import-audit/v1",
        "stage": stage,
        "observed_at_utc": _utc_now(),
        "observed_at_monotonic_ns": time.monotonic_ns(),
        "pid": os.getpid(),
        "loaded_forbidden_modules": loaded,
        "cuda_clean": not loaded,
    }


def _parse_inventory(stdout: str) -> tuple[dict[str, Any], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw in stdout.splitlines():
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
        raise RuntimeError(f"integrated v11 requires exactly two visible GPUs, got {len(rows)}")
    if len({row["uuid"] for row in rows}) != 2:
        raise RuntimeError("integrated v11 received duplicate physical GPU UUIDs")
    if len({row["index"] for row in rows}) != 2:
        raise RuntimeError("integrated v11 received duplicate visible GPU indices")
    for row in rows:
        if "A100" not in str(row["name"]) or int(row["memory_total_mib"]) < 79_000:
            raise RuntimeError(f"integrated v11 requires A100-80GB, observed {row}")
    return rows[0], rows[1]


def _subprocess_timeout(*, deadline_ns: int | None, ceiling_seconds: float) -> float:
    if deadline_ns is None:
        return ceiling_seconds
    remaining = (deadline_ns - time.monotonic_ns()) / 1e9
    if remaining <= 0.0:
        raise TimeoutError("integrated v11 exhausted its absolute deadline")
    return min(ceiling_seconds, remaining)


def _inventory(*, deadline_ns: int | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,driver_version,memory.total,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=_subprocess_timeout(deadline_ns=deadline_ns, ceiling_seconds=15.0),
    )
    if completed.returncode != 0:
        raise RuntimeError(f"nvidia-smi inventory failed: {completed.stderr.strip()}")
    return _parse_inventory(completed.stdout)


def _parse_compute_processes(stdout: str) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    for raw in stdout.splitlines():
        if not raw.strip() or raw.lower().startswith("no running processes"):
            continue
        fields = [field.strip() for field in raw.split(",")]
        if len(fields) != 4:
            raise RuntimeError("nvidia-smi compute query returned an unexpected row")
        rows.append(
            {
                "gpu_uuid": fields[0],
                "pid": int(fields[1]),
                "process_name": fields[2],
                "used_gpu_memory_mib": int(fields[3]),
            }
        )
    return tuple(rows)


def _compute_processes(*, deadline_ns: int | None = None) -> tuple[dict[str, Any], ...]:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=_subprocess_timeout(deadline_ns=deadline_ns, ceiling_seconds=15.0),
    )
    if completed.returncode != 0:
        raise RuntimeError(f"nvidia-smi compute query failed: {completed.stderr.strip()}")
    return _parse_compute_processes(completed.stdout)


def _require_no_compute_processes(processes: tuple[dict[str, Any], ...], *, phase: str) -> None:
    if processes:
        raise RuntimeError(f"unexpected GPU compute processes during {phase}: {processes}")


def _stable_inventory_identity(
    before: Sequence[Mapping[str, Any]], after: Sequence[Mapping[str, Any]]
) -> bool:
    keys = ("index", "uuid", "name", "driver_version", "memory_total_mib")
    return len(before) == len(after) == 2 and all(
        all(left.get(key) == right.get(key) for key in keys)
        for left, right in zip(before, after, strict=True)
    )


def _wait_for_files(
    paths: tuple[Path, ...],
    *,
    deadline_ns: int,
    phase: str,
    processes: tuple[tuple[str, subprocess.Popen[bytes]], ...] = (),
    observe_processes: Any | None = None,
) -> None:
    while not all(path.is_file() for path in paths):
        if observe_processes is not None:
            observe_processes(phase)
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


def _process_tree_procfs() -> tuple[dict[str, Any], ...]:
    root = Path("/proc")
    if not root.is_dir():
        raise FileNotFoundError("/proc is unavailable")
    rows: list[dict[str, Any]] = []
    for entry in root.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            raw = (entry / "stat").read_text()
            close = raw.rfind(")")
            if close < 0:
                continue
            fields = raw[close + 2 :].split()
            command = " ".join(
                item.decode("utf-8", errors="replace")
                for item in (entry / "cmdline").read_bytes().split(b"\0")
                if item
            )
            rows.append(
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
    return tuple(sorted(rows, key=lambda item: int(item["pid"])))


def _process_tree_ps() -> tuple[dict[str, Any], ...]:
    completed = subprocess.run(
        ["ps", "-axo", "pid=,ppid=,pgid=,sess=,state=,command="],
        check=False,
        capture_output=True,
        text=True,
        timeout=5.0,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"ps process-tree query failed: {completed.stderr.strip()}")
    rows: list[dict[str, Any]] = []
    for raw in completed.stdout.splitlines():
        fields = raw.strip().split(maxsplit=5)
        if len(fields) < 5:
            continue
        try:
            rows.append(
                {
                    "pid": int(fields[0]),
                    "ppid": int(fields[1]),
                    "pgid": int(fields[2]),
                    "sid": int(fields[3]),
                    "state": fields[4],
                    "start_token": None,
                    "command": fields[5] if len(fields) == 6 else "",
                }
            )
        except ValueError:
            continue
    return tuple(sorted(rows, key=lambda item: int(item["pid"])))


def _process_tree() -> tuple[dict[str, Any], ...]:
    try:
        return _process_tree_procfs()
    except FileNotFoundError:
        return _process_tree_ps()


def _descendant_pids(processes: Sequence[Mapping[str, Any]], root_pid: int) -> set[int]:
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


def _thread_snapshot() -> tuple[dict[str, Any], ...]:
    return tuple(
        {
            "name": thread.name,
            "ident": thread.ident,
            "native_id": thread.native_id,
            "daemon": thread.daemon,
            "alive": thread.is_alive(),
        }
        for thread in threading.enumerate()
    )


def _ipc_snapshot() -> tuple[str, ...]:
    root = Path("/dev/shm")
    if not root.is_dir():
        return ()
    try:
        return tuple(sorted(item.name for item in root.iterdir()))
    except PermissionError:
        return ()


class _ChildSubreaper:
    """Temporarily adopt orphaned worker helpers so the controller can reap them."""

    def __init__(self) -> None:
        self.supported = sys.platform.startswith("linux")
        self.initially_enabled: bool | None = None
        self.enabled = False
        self.restored = False
        self.error: str | None = None
        self._libc: Any | None = None
        if not self.supported:
            return
        try:
            libc = ctypes.CDLL(None, use_errno=True)
            current = ctypes.c_int()
            if libc.prctl(PR_GET_CHILD_SUBREAPER, ctypes.byref(current), 0, 0, 0) != 0:
                raise OSError(ctypes.get_errno(), "PR_GET_CHILD_SUBREAPER failed")
            self.initially_enabled = bool(current.value)
            if not self.initially_enabled and libc.prctl(PR_SET_CHILD_SUBREAPER, 1, 0, 0, 0) != 0:
                raise OSError(ctypes.get_errno(), "PR_SET_CHILD_SUBREAPER failed")
            self._libc = libc
            self.enabled = True
        except OSError as caught:
            self.error = str(caught)

    def restore(self) -> None:
        if not self.supported:
            self.restored = True
            return
        if not self.enabled or self._libc is None or self.initially_enabled is None:
            return
        if not self.initially_enabled and self._libc.prctl(PR_SET_CHILD_SUBREAPER, 0, 0, 0, 0) != 0:
            raise OSError(ctypes.get_errno(), "restoring child-subreaper state failed")
        self.restored = True

    def evidence(self) -> dict[str, Any]:
        return {
            "supported": self.supported,
            "initially_enabled": self.initially_enabled,
            "enabled_for_experiment": self.enabled,
            "restored_before_return": self.restored,
            "error": self.error,
        }


class _ProcessLifecycleAudit:
    """Track experiment-owned processes without importing a CUDA runtime."""

    def __init__(self) -> None:
        self.subreaper = _ChildSubreaper()
        self.parent_pid = os.getpid()
        self.parent_pgid = os.getpgid(0)
        self.parent_sid = os.getsid(0)
        self.initial_processes = _process_tree()
        self.initial_descendants = _descendant_pids(self.initial_processes, self.parent_pid)
        self.initial_identities = {
            int(item["pid"]): item.get("start_token") for item in self.initial_processes
        }
        self.initial_threads = _thread_snapshot()
        self.initial_ipc = _ipc_snapshot()
        self.worker_roots: dict[int, dict[str, Any]] = {}
        self.owned: dict[int, dict[str, Any]] = {}
        self.reaped: dict[int, dict[str, Any]] = {}
        self.transitions: list[dict[str, Any]] = []
        self.last_observed_ns = 0

    def register_worker(self, role: str, process: subprocess.Popen[bytes]) -> None:
        self.worker_roots[process.pid] = {
            "role": role,
            "pid": process.pid,
            "pgid": os.getpgid(process.pid),
            "sid": os.getsid(process.pid),
        }
        self.observe("worker-spawn", force=True)

    def transition(self, phase: str) -> None:
        index = len(self.transitions)
        if index >= len(PROCESS_LIFECYCLE_PHASES) or PROCESS_LIFECYCLE_PHASES[index] != phase:
            raise RuntimeError(f"invalid process lifecycle transition to {phase}")
        self.transitions.append(
            {
                "phase": phase,
                "observed_at_utc": _utc_now(),
                "observed_at_monotonic_ns": time.monotonic_ns(),
            }
        )

    def observe(self, stage: str, *, force: bool = False) -> tuple[dict[str, Any], ...]:
        now = time.monotonic_ns()
        if not force and now - self.last_observed_ns < 250_000_000:
            return ()
        self.last_observed_ns = now
        processes = _process_tree()
        descendants = _descendant_pids(processes, self.parent_pid)
        worker_descendants = {pid: _descendant_pids(processes, pid) for pid in self.worker_roots}
        worker_groups = {int(item["pgid"]) for item in self.worker_roots.values()}
        worker_sessions = {int(item["sid"]) for item in self.worker_roots.values()}
        observed: list[dict[str, Any]] = []
        for process in processes:
            pid = int(process["pid"])
            start_token = process.get("start_token")
            tracked = self.owned.get(pid)
            if (
                tracked is not None
                and tracked.get("start_token") is not None
                and start_token is not None
                and tracked["start_token"] != start_token
            ):
                continue
            new_controller_descendant = pid in descendants and (
                pid not in self.initial_descendants
                or (
                    self.initial_identities.get(pid) is not None
                    and start_token is not None
                    and self.initial_identities[pid] != start_token
                )
            )
            candidate = (
                pid in self.worker_roots
                or any(pid in values for values in worker_descendants.values())
                or int(process["pgid"]) in worker_groups
                or int(process["sid"]) in worker_sessions
                or new_controller_descendant
                or tracked is not None
            )
            if not candidate or pid == self.parent_pid:
                continue
            role = next(
                (
                    str(root["role"])
                    for root_pid, root in self.worker_roots.items()
                    if pid == root_pid
                    or pid in worker_descendants[root_pid]
                    or int(process["pgid"]) == int(root["pgid"])
                    or int(process["sid"]) == int(root["sid"])
                ),
                "helper",
            )
            command = str(process.get("command", ""))
            process_kind = (
                "profiler"
                if _profiler_process(process)
                else (
                    "multiprocessing_resource_tracker"
                    if RESOURCE_TRACKER_COMMAND_TOKEN in command
                    else ("worker" if pid in self.worker_roots else "helper")
                )
            )
            first_seen = now if tracked is None else int(tracked["first_seen_ns"])
            record = {
                **dict(process),
                "role": role,
                "process_kind": process_kind,
                "cuda_owning_candidate": role in ROLES,
                "first_seen_ns": first_seen,
                "last_seen_ns": now,
                "last_observed_stage": stage,
            }
            self.owned[pid] = record
            observed.append(record)
        return tuple(observed)

    def mark_reaped(
        self,
        pid: int,
        *,
        exit_status: int | None,
        method: str = "popen-waitpid",
    ) -> None:
        self.reaped[pid] = {
            "exit_status": exit_status,
            "reap_method": method,
            "reap_timestamp_utc": _utc_now(),
            "reap_timestamp_monotonic_ns": time.monotonic_ns(),
        }

    def currently_owned(self, *, stage: str) -> tuple[dict[str, Any], ...]:
        self.observe(stage, force=True)
        current = {int(item["pid"]): item for item in _process_tree()}
        rows: list[dict[str, Any]] = []
        for pid, tracked in self.owned.items():
            process = current.get(pid)
            if process is None:
                if pid not in self.reaped:
                    self.mark_reaped(
                        pid,
                        exit_status=None,
                        method="process-absence-confirmed-parent-or-worker-reaped",
                    )
                continue
            if (
                tracked.get("start_token") is not None
                and process.get("start_token") is not None
                and tracked["start_token"] != process["start_token"]
            ):
                continue
            rows.append({**tracked, **process})
        return tuple(sorted(rows, key=lambda item: int(item["pid"])))


def _process_group_exists(process_group: int) -> bool:
    try:
        os.killpg(process_group, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _wait_for_group_exit(process_group: int, *, deadline_ns: int) -> bool:
    while _process_group_exists(process_group):
        if time.monotonic_ns() >= deadline_ns:
            return False
        time.sleep(0.02)
    return True


def _poll_and_reap_workers(
    processes: Sequence[subprocess.Popen[bytes]],
    *,
    lifecycle: _ProcessLifecycleAudit | None = None,
) -> None:
    """Reap leaders before testing PGID emptiness.

    A dead-but-unreaped session leader keeps ``killpg(pgid, 0)`` observable on
    Linux.  The previous ordering therefore reported a surviving group after
    SIGKILL even though the sole remaining member was a zombie.
    """

    for process in processes:
        returncode = process.poll()
        if returncode is not None and lifecycle is not None and process.pid not in lifecycle.reaped:
            lifecycle.mark_reaped(process.pid, exit_status=returncode)


def _terminate_group(
    process: subprocess.Popen[bytes],
    actions: list[dict[str, Any]],
    *,
    deadline_ns: int,
) -> None:
    """Bound teardown by the same absolute paid-allocation deadline."""

    process_group = process.pid
    _poll_and_reap_workers((process,))
    if process.poll() is not None and not _process_group_exists(process_group):
        return
    actions.append(
        {
            "pid": process.pid,
            "process_group": process_group,
            "signal": "SIGTERM",
            "forced": False,
            "target": "worker-process-group",
            "at_utc": _utc_now(),
        }
    )
    with suppress(ProcessLookupError):
        os.killpg(process_group, signal.SIGTERM)
    term_deadline = min(deadline_ns, time.monotonic_ns() + 5_000_000_000)
    if process.poll() is None:
        with suppress(subprocess.TimeoutExpired):
            process.wait(timeout=max(0.0, (term_deadline - time.monotonic_ns()) / 1e9))
    _poll_and_reap_workers((process,))
    if _wait_for_group_exit(process_group, deadline_ns=term_deadline):
        return
    actions.append(
        {
            "pid": process.pid,
            "process_group": process_group,
            "signal": "SIGKILL",
            "forced": True,
            "target": "worker-process-group",
            "at_utc": _utc_now(),
        }
    )
    with suppress(ProcessLookupError):
        os.killpg(process_group, signal.SIGKILL)
    kill_deadline = min(deadline_ns, time.monotonic_ns() + 3_000_000_000)
    if process.poll() is None:
        with suppress(subprocess.TimeoutExpired):
            process.wait(timeout=max(0.0, (kill_deadline - time.monotonic_ns()) / 1e9))
    _poll_and_reap_workers((process,))
    if not _wait_for_group_exit(process_group, deadline_ns=kill_deadline):
        raise RuntimeError(f"worker process group {process_group} survived SIGKILL")


def _terminate_groups(
    processes: Sequence[subprocess.Popen[bytes]],
    actions: list[dict[str, Any]],
    *,
    deadline_ns: int,
    lifecycle: _ProcessLifecycleAudit | None = None,
) -> None:
    """Signal all role groups in parallel so one child cannot consume teardown."""

    _poll_and_reap_workers(processes, lifecycle=lifecycle)
    groups = {
        process.pid: process
        for process in processes
        if process.poll() is None or _process_group_exists(process.pid)
    }
    for process_group in groups:
        actions.append(
            {
                "pid": process_group,
                "process_group": process_group,
                "signal": "SIGTERM",
                "forced": False,
                "target": "worker-process-group",
                "at_utc": _utc_now(),
            }
        )
        with suppress(ProcessLookupError):
            os.killpg(process_group, signal.SIGTERM)
    term_deadline = min(deadline_ns, time.monotonic_ns() + 5_000_000_000)
    while any(_process_group_exists(group) for group in groups):
        _poll_and_reap_workers(tuple(groups.values()), lifecycle=lifecycle)
        if time.monotonic_ns() >= term_deadline:
            break
        time.sleep(0.02)
    _poll_and_reap_workers(tuple(groups.values()), lifecycle=lifecycle)
    survivors = [group for group in groups if _process_group_exists(group)]
    for process_group in survivors:
        actions.append(
            {
                "pid": process_group,
                "process_group": process_group,
                "signal": "SIGKILL",
                "forced": True,
                "target": "worker-process-group",
                "at_utc": _utc_now(),
            }
        )
        with suppress(ProcessLookupError):
            os.killpg(process_group, signal.SIGKILL)
    while any(_process_group_exists(group) for group in survivors):
        _poll_and_reap_workers(tuple(groups.values()), lifecycle=lifecycle)
        if time.monotonic_ns() >= deadline_ns:
            remaining = [group for group in survivors if _process_group_exists(group)]
            raise RuntimeError(f"worker process groups survived SIGKILL: {remaining}")
        time.sleep(0.02)
    for process in groups.values():
        if process.poll() is None:
            with suppress(subprocess.TimeoutExpired):
                process.wait(timeout=max(0.0, (deadline_ns - time.monotonic_ns()) / 1e9))
        _poll_and_reap_workers((process,), lifecycle=lifecycle)


def _signal_exact_owned(
    processes: Sequence[Mapping[str, Any]],
    sent_signal: signal.Signals,
    actions: list[dict[str, Any]],
) -> None:
    """Signal tracked escaped helpers by exact, still-matching PID identity."""

    current = {int(item["pid"]): item for item in _process_tree()}
    for tracked in sorted(processes, key=lambda item: int(item["pid"]), reverse=True):
        pid = int(tracked["pid"])
        observed = current.get(pid)
        if observed is None:
            continue
        if (
            tracked.get("start_token") is not None
            and observed.get("start_token") is not None
            and tracked["start_token"] != observed["start_token"]
        ):
            continue
        actions.append(
            {
                "pid": pid,
                "process_group": int(observed["pgid"]),
                "signal": sent_signal.name,
                "forced": sent_signal == signal.SIGKILL,
                "target": "exact-owned-pid",
                "role": tracked.get("role", "helper"),
                "at_utc": _utc_now(),
            }
        )
        with suppress(ProcessLookupError):
            os.kill(pid, sent_signal)


def _wait_for_owned_exit(
    lifecycle: _ProcessLifecycleAudit,
    processes: Sequence[subprocess.Popen[bytes]],
    *,
    deadline_ns: int,
    stage: str,
) -> tuple[dict[str, Any], ...]:
    while True:
        _poll_and_reap_workers(processes, lifecycle=lifecycle)
        survivors = lifecycle.currently_owned(stage=stage)
        if not survivors or time.monotonic_ns() >= deadline_ns:
            return survivors
        time.sleep(0.02)


def _reap_owned_children(
    lifecycle: _ProcessLifecycleAudit,
    processes: Sequence[subprocess.Popen[bytes]],
    *,
    deadline_ns: int,
) -> tuple[dict[str, Any], ...]:
    """Reap worker leaders and subreaper-adopted descendants within the bound."""

    leaders = {process.pid: process for process in processes}
    while time.monotonic_ns() < deadline_ns:
        _poll_and_reap_workers(processes, lifecycle=lifecycle)
        progress = False
        for pid in sorted(lifecycle.owned):
            if pid in leaders or pid in lifecycle.reaped:
                continue
            try:
                reaped_pid, status = os.waitpid(pid, os.WNOHANG)
            except ChildProcessError:
                continue
            if reaped_pid:
                lifecycle.mark_reaped(
                    reaped_pid,
                    exit_status=os.waitstatus_to_exitcode(status),
                    method="controller-waitpid-adopted-child",
                )
                progress = True
        survivors = lifecycle.currently_owned(stage="child-reap")
        if not survivors:
            return ()
        if not progress:
            time.sleep(0.02)
    return lifecycle.currently_owned(stage="child-reap-deadline")


def _profiler_process(process: Mapping[str, Any]) -> bool:
    words = str(process.get("command", "")).split()
    executables = {Path(word).name.lower() for word in words}
    return bool(executables & set(PROFILER_COMMAND_TOKENS))


def _owned_child_evidence(
    lifecycle: _ProcessLifecycleAudit,
    actions: Sequence[Mapping[str, Any]],
    processes: Sequence[subprocess.Popen[bytes]],
) -> list[dict[str, Any]]:
    leaders = {process.pid: process for process in processes}
    rows: list[dict[str, Any]] = []
    for pid, tracked in sorted(lifecycle.owned.items()):
        matching_actions = [
            action
            for action in actions
            if int(action.get("pid", -1)) == pid
            or int(action.get("process_group", -1)) == int(tracked["pgid"])
        ]
        last_signal = matching_actions[-1]["signal"] if matching_actions else None
        reap = lifecycle.reaped.get(pid, {})
        leader = leaders.get(pid)
        rows.append(
            {
                "pid": pid,
                "ppid": int(tracked["ppid"]),
                "pgid": int(tracked["pgid"]),
                "sid": int(tracked["sid"]),
                "role": tracked["role"],
                "process_kind": tracked["process_kind"],
                "command": tracked["command"],
                "cuda_owning_candidate": tracked["cuda_owning_candidate"],
                "termination_signal": last_signal,
                "exit_status": (
                    leader.returncode if leader is not None else reap.get("exit_status")
                ),
                "reap_timestamp_utc": reap.get("reap_timestamp_utc"),
                "reap_timestamp_monotonic_ns": reap.get("reap_timestamp_monotonic_ns"),
                "reap_method": reap.get("reap_method"),
            }
        )
    return rows


def _build_in_function_cleanup(
    *,
    lifecycle: _ProcessLifecycleAudit,
    actions: Sequence[Mapping[str, Any]],
    processes: Sequence[subprocess.Popen[bytes]],
    surviving_children: Sequence[Mapping[str, Any]],
    surviving_groups: Sequence[int],
    compute_processes_after: Sequence[Mapping[str, Any]],
    cuda_release_verified: bool,
    pipes_closed: bool,
    cleanup_errors: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    final_threads = _thread_snapshot()
    initial_thread_keys = {
        (item["ident"], item["native_id"], item["name"]) for item in lifecycle.initial_threads
    }
    leaked_threads = [
        item
        for item in final_threads
        if item["alive"]
        and (item["ident"], item["native_id"], item["name"]) not in initial_thread_keys
    ]
    final_ipc = _ipc_snapshot()
    leaked_ipc = sorted(set(final_ipc) - set(lifecycle.initial_ipc))
    profilers = [item for item in surviving_children if _profiler_process(item)]
    resource_trackers = [
        item
        for item in surviving_children
        if RESOURCE_TRACKER_COMMAND_TOKEN in str(item.get("command", ""))
    ]
    zombies = [item for item in surviving_children if "Z" in str(item.get("state", ""))]
    forced_kills = [item for item in actions if item.get("signal") == "SIGKILL"]
    phases = [str(item["phase"]) for item in lifecycle.transitions]
    subreaper = getattr(lifecycle, "subreaper", None)
    subreaper_evidence = (
        subreaper.evidence()
        if subreaper is not None
        else {
            "supported": False,
            "initially_enabled": None,
            "enabled_for_experiment": False,
            "restored_before_return": True,
            "error": None,
        }
    )
    subreaper_pass = bool(
        subreaper_evidence["error"] is None
        and (
            not subreaper_evidence["supported"]
            or (
                subreaper_evidence["enabled_for_experiment"]
                and subreaper_evidence["restored_before_return"]
            )
        )
    )
    passed = bool(
        phases == list(PROCESS_LIFECYCLE_PHASES)
        and not surviving_children
        and not surviving_groups
        and cuda_release_verified
        and not compute_processes_after
        and not profilers
        and not resource_trackers
        and not zombies
        and not leaked_threads
        and not leaked_ipc
        and subreaper_pass
        and pipes_closed
        and not cleanup_errors
        and all(process.returncode is not None for process in processes)
    )
    return {
        "schema_version": "sloforge.branchfabric.in-function-cleanup/v1",
        "status": "PASS" if passed else "FAIL",
        "pass": passed,
        "parent_pid": lifecycle.parent_pid,
        "parent_pgid": lifecycle.parent_pgid,
        "parent_sid": lifecycle.parent_sid,
        "child_subreaper": subreaper_evidence,
        "lifecycle": lifecycle.transitions,
        "required_lifecycle": list(PROCESS_LIFECYCLE_PHASES),
        "owned_children": _owned_child_evidence(lifecycle, actions, processes),
        "termination_actions": list(actions),
        "forced_kills": forced_kills,
        "forced_kill_required": bool(forced_kills),
        "surviving_children": list(surviving_children),
        "surviving_process_groups": list(surviving_groups),
        "profiler_processes_after": profilers,
        "serving_workers_after": [
            item for item in surviving_children if item.get("role") == "serving"
        ],
        "rollout_workers_after": [
            item for item in surviving_children if item.get("role") == "rollout"
        ],
        "resource_tracker_processes_after": resource_trackers,
        "zombie_processes_after": zombies,
        "parent_reaped_all_owned_children": not surviving_children
        and all(process.returncode is not None for process in processes),
        "owned_ipc_resources_released": not leaked_ipc,
        "initial_ipc_resources": list(lifecycle.initial_ipc),
        "final_ipc_resources": list(final_ipc),
        "leaked_ipc_resources": leaked_ipc,
        "initial_threads": list(lifecycle.initial_threads),
        "final_threads": list(final_threads),
        "leaked_threads": leaked_threads,
        "pipes_closed": pipes_closed,
        "compute_processes_after": list(compute_processes_after),
        "cuda_released": cuda_release_verified and not compute_processes_after,
        "cleanup_errors": list(cleanup_errors),
        "recorded_at_utc": _utc_now(),
        "recorded_at_monotonic_ns": time.monotonic_ns(),
    }


def _load_json_object(path: Path, *, label: str) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _status_is_pass(value: Any) -> bool:
    return isinstance(value, str) and (
        value.upper() in {"PASS", "PASSED", "MICRO_VALIDATION_PASS"}
        or value.upper().startswith("PASS_")
    )


def _review_entries(manifest: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    raw = manifest.get("reviews", manifest.get("post_micro_reviews"))
    if not isinstance(raw, list) or not all(isinstance(item, dict) for item in raw):
        raise RuntimeError("post-micro manifest must contain a review list")
    return tuple(raw)


def _agent_number(entry: Mapping[str, Any]) -> int | None:
    raw = entry.get("agent", entry.get("agent_id", entry.get("reviewer")))
    if isinstance(raw, int) and not isinstance(raw, bool):
        return raw
    if isinstance(raw, str):
        digits = "".join(character for character in raw if character.isdigit())
        if digits:
            return int(digits)
    return None


def _verify_final_pre_gpu_gate(
    manifest: Mapping[str, Any], repository_root: Path, *, attempt_id: str
) -> dict[str, Any]:
    """Verify the outer, post-fix eight-review gate for the fresh retry.

    Review artifacts cannot hash a manifest that already contains their own
    hashes without creating a cycle.  The config therefore seals this outer
    post-micro manifest, which in turn seals each review and the fresh
    ``make check`` evidence.  No review is accepted by filename alone.
    """

    gate = manifest.get("final_integrated_pre_gpu_gate")
    if (
        not isinstance(gate, dict)
        or gate.get("status") != "PASS"
        or gate.get("attempt_id") != attempt_id
    ):
        raise RuntimeError("fresh integrated retry lacks its final pre-GPU PASS gate")
    reviews = gate.get("reviews")
    if not isinstance(reviews, list) or len(reviews) != len(FINAL_PRE_GPU_REVIEW_ROLES):
        raise RuntimeError("fresh integrated retry does not seal all eight pre-GPU reviews")
    by_agent: dict[int, Mapping[str, Any]] = {}
    verified: list[dict[str, Any]] = []
    for entry in reviews:
        if not isinstance(entry, dict):
            raise RuntimeError("fresh integrated retry review binding is malformed")
        agent = _agent_number(entry)
        if agent is None or agent in by_agent:
            raise RuntimeError("fresh integrated retry review agents are invalid or duplicated")
        by_agent[agent] = entry
    if set(by_agent) != set(FINAL_PRE_GPU_REVIEW_ROLES):
        raise RuntimeError("fresh integrated retry review coverage is incomplete")
    for agent, expected_role in FINAL_PRE_GPU_REVIEW_ROLES.items():
        entry = by_agent[agent]
        reference = entry.get("artifact")
        expected_hash = entry.get("sha256")
        if (
            entry.get("role") != expected_role
            or entry.get("status") != "PASS"
            or not isinstance(reference, str)
            or not isinstance(expected_hash, str)
        ):
            raise RuntimeError(f"fresh integrated retry Agent {agent} binding is not PASS")
        path = validate_bound_artifact(
            repository_root,
            reference=reference,
            expected_sha256=expected_hash,
        )
        evidence = _load_json_object(path, label=f"fresh pre-GPU Agent {agent} evidence")
        expected_role_tokens = {
            token
            for token in expected_role.split("-")
            if token not in {"implementation", "review"}
        }
        evidence_role_tokens = {
            token
            for token in "".join(
                character.lower()
                if character.isascii() and character.isalnum()
                else " "
                for character in (
                    str(evidence.get("role", ""))
                    + " "
                    + str(evidence.get("schema_version", ""))
                )
            ).split()
        }
        if (
            evidence.get("status") != "PASS"
            or _agent_number(evidence) != agent
            or not expected_role_tokens <= evidence_role_tokens
            or evidence.get("gpu_or_cloud_invoked") is True
            or evidence.get("gpu_resources_consumed") is True
        ):
            raise RuntimeError(f"fresh integrated retry Agent {agent} evidence is not PASS offline")
        verified.append(
            {
                "agent": agent,
                "role": expected_role,
                "artifact": reference,
                "sha256": expected_hash,
            }
        )
    make_check = gate.get("make_check")
    if (
        not isinstance(make_check, dict)
        or make_check.get("status") != "PASS"
        or make_check.get("command") != "make check"
        or not isinstance(make_check.get("artifact"), str)
        or not isinstance(make_check.get("sha256"), str)
    ):
        raise RuntimeError("fresh integrated retry lacks fresh make check PASS evidence")
    make_check_path = validate_bound_artifact(
        repository_root,
        reference=str(make_check["artifact"]),
        expected_sha256=str(make_check["sha256"]),
    )
    make_check_evidence = _load_json_object(
        make_check_path, label="fresh integrated retry make-check evidence"
    )
    if (
        make_check_evidence.get("status") != "PASS"
        or make_check_evidence.get("command") != "make check"
    ):
        raise RuntimeError("fresh integrated retry make check payload is not PASS")
    return {
        "schema_version": "sloforge.branchfabric.exp004-v11-final-pre-gpu-verification/v1",
        "status": "PASS",
        "attempt_id": attempt_id,
        "reviews": verified,
        "make_check": {
            "artifact": make_check["artifact"],
            "sha256": make_check["sha256"],
        },
    }


def verify_sealed_evidence(
    config: Experiment004V11IntegratedConfig, repository_root: Path
) -> dict[str, Any]:
    """Verify every content-addressed prerequisite before GPU inventory."""

    references = {
        "offline": (config.offline_gate_manifest, config.offline_gate_manifest_sha256),
        "micro": (config.micro_validation_artifact, config.micro_validation_sha256),
        "post_micro": (
            config.post_micro_review_manifest,
            config.post_micro_review_manifest_sha256,
        ),
        "budget": (config.budget_authorization, config.budget_authorization_sha256),
    }
    paths = {
        name: validate_bound_artifact(
            repository_root, reference=reference, expected_sha256=expected_hash
        )
        for name, (reference, expected_hash) in references.items()
    }
    offline = _load_json_object(paths["offline"], label="offline gate manifest")
    micro = _load_json_object(paths["micro"], label="micro-validation artifact")
    post_micro = _load_json_object(paths["post_micro"], label="post-micro review manifest")
    budget = _load_json_object(paths["budget"], label="budget authorization")

    gates = offline.get("production_integration_gates")
    gate_values: list[Any] = []
    if isinstance(gates, dict):
        gate_values = list(gates.values())
    elif isinstance(gates, list):
        gate_values = [item.get("status") if isinstance(item, dict) else item for item in gates]
    if (
        offline.get("status") != "PASS"
        or len(gate_values) != 4
        or not all(
            _status_is_pass(item.get("status") if isinstance(item, dict) else item)
            for item in gate_values
        )
    ):
        raise RuntimeError("offline manifest does not seal all four production gates PASS")
    if isinstance(gates, list):
        for item in gates:
            if not isinstance(item, dict):
                raise RuntimeError("offline production gate commitment is malformed")
            reference = item.get("artifact")
            expected_hash = item.get("sha256")
            if not isinstance(reference, str) or not isinstance(expected_hash, str):
                raise RuntimeError("offline production gate lacks artifact/hash commitment")
            validate_bound_artifact(
                repository_root,
                reference=reference,
                expected_sha256=expected_hash,
            )

    correctness = micro.get("correctness")
    required_correctness = (
        "allocator_epoch_pass",
        "post_free_ownership_pass",
        "fresh_destination_allocations_pass",
        "destination_mapping_commitment_pass",
        "integrity_pass",
        "movement_accounting_pass",
        "all_branches_resumed",
    )
    movement = micro.get("movement")
    if (
        micro.get("status") != "MICRO_VALIDATION_PASS"
        or micro.get("scientifically_admissible_measurement") is not True
        or micro.get("integrated_v11_allowed") is not True
        or not isinstance(correctness, dict)
        or any(correctness.get(name) is not True for name in required_correctness)
        or correctness.get("first_resumed_token_exact_matches") != 8
        or correctness.get("first_resumed_token_total") != 8
        or not isinstance(movement, dict)
        or not isinstance(movement.get("full_physical_amplification"), (int, float))
        or isinstance(movement.get("full_physical_amplification"), bool)
        or not math.isfinite(float(movement["full_physical_amplification"]))
        or float(movement["full_physical_amplification"]) > 25.0
    ):
        raise RuntimeError("micro-validation artifact does not admit integrated v11")

    micro_attempt = micro.get("attempt_id")
    reviews = _review_entries(post_micro)
    agents = {_agent_number(item) for item in reviews}
    if (
        post_micro.get("status") != "PASS"
        or post_micro.get("integrated_v11_admission") != "APPROVED_EXACTLY_ONCE"
        or post_micro.get("attempt_id") != micro_attempt
        or len(reviews) != 4
        or agents != {9, 10, 11, 12}
        or any(not _status_is_pass(item.get("status")) for item in reviews)
    ):
        raise RuntimeError("post-micro manifest does not seal PASS reviews 9-12")
    for entry in reviews:
        reference = entry.get("artifact", entry.get("path"))
        expected_hash = entry.get("sha256")
        if not isinstance(reference, str) or not isinstance(expected_hash, str):
            raise RuntimeError("post-micro review entry lacks artifact/hash commitment")
        validate_bound_artifact(
            repository_root,
            reference=reference,
            expected_sha256=expected_hash,
        )

    prelaunch = post_micro.get("integrated_prelaunch")
    required_prelaunch_labels = set(INTEGRATED_PRELAUNCH_BINDINGS)
    if not isinstance(prelaunch, dict) or prelaunch.get("status") != "PASS":
        raise RuntimeError("post-micro manifest does not seal integrated prelaunch PASS")
    prelaunch_bindings = prelaunch.get("bindings")
    if not isinstance(prelaunch_bindings, list):
        raise RuntimeError("integrated prelaunch bindings are malformed")
    by_label: dict[str, dict[str, Any]] = {}
    for entry in prelaunch_bindings:
        if not isinstance(entry, dict) or not isinstance(entry.get("label"), str):
            raise RuntimeError("integrated prelaunch binding is malformed")
        label = str(entry["label"])
        if label in by_label:
            raise RuntimeError("integrated prelaunch binding labels are not unique")
        by_label[label] = entry
    if set(by_label) != required_prelaunch_labels:
        raise RuntimeError("integrated prelaunch bindings are incomplete")
    for label, entry in by_label.items():
        reference = entry.get("artifact")
        expected_hash = entry.get("sha256")
        if not isinstance(reference, str) or not isinstance(expected_hash, str):
            raise RuntimeError("integrated prelaunch binding lacks artifact/hash commitment")
        if reference != INTEGRATED_PRELAUNCH_BINDINGS[label]:
            raise RuntimeError("integrated prelaunch binding path substitution")
        validate_bound_artifact(
            repository_root,
            reference=reference,
            expected_sha256=expected_hash,
        )
    make_check = _load_json_object(
        validate_bound_artifact(
            repository_root,
            reference=str(by_label["make_check"]["artifact"]),
            expected_sha256=str(by_label["make_check"]["sha256"]),
        ),
        label="integrated prelaunch make-check evidence",
    )
    if make_check.get("status") != "PASS" or make_check.get("command") != "make check":
        raise RuntimeError("integrated prelaunch make check is not PASS")

    replacement = post_micro.get("integrated_replacement_authorization")
    if replacement is None and config.attempt_id != ORIGINAL_INTEGRATED_ATTEMPT_ID:
        raise RuntimeError("replacement config lacks replacement-specific authorization")
    final_pre_gpu_gate: dict[str, Any] | None = None
    if replacement is not None:
        if (
            not isinstance(replacement, dict)
            or replacement.get("status") != "APPROVED_EXACTLY_ONCE"
            or config.attempt_id != REPLACEMENT_INTEGRATED_ATTEMPT_ID
            or replacement.get("replacement_attempt_id") != config.attempt_id
            or replacement.get("prior_attempt_id") != PRIOR_INVALID_INTEGRATED_ATTEMPT_ID
            or replacement.get("prior_attempt_id") == config.attempt_id
            or replacement.get("prior_attempt_scientific_status")
            != "INVALID_PRE_RECLAIM_BOUNDED_BACKLOG_ABORT"
            or replacement.get("ledger_sha256_before_reservation")
            != config.ledger_sha256_before_reservation
        ):
            raise RuntimeError("integrated replacement authorization is inconsistent")
        final_pre_gpu_gate = _verify_final_pre_gpu_gate(
            post_micro,
            repository_root,
            attempt_id=config.attempt_id,
        )
        for label in ("prior_attempt_cleanup",):
            binding = replacement.get(label)
            if not isinstance(binding, dict) or binding.get("status") != "PASS":
                raise RuntimeError(f"integrated replacement {label} is not PASS")
            reference = binding.get("artifact")
            expected_hash = binding.get("sha256")
            if not isinstance(reference, str) or not isinstance(expected_hash, str):
                raise RuntimeError(f"integrated replacement {label} lacks artifact/hash")
            if reference != REPLACEMENT_EVIDENCE_BINDINGS[label]:
                raise RuntimeError(f"integrated replacement {label} path substitution")
            evidence_path = validate_bound_artifact(
                repository_root,
                reference=reference,
                expected_sha256=expected_hash,
            )
            evidence = _load_json_object(evidence_path, label=f"replacement {label}")
            if label == "prior_attempt_cleanup":
                provider = evidence.get("provider_cleanup")
                settlement = evidence.get("settlement")
                remote = evidence.get("remote_evidence")
                if (
                    evidence.get("schema_version")
                    != "sloforge.branchfabric.experiment-004-v11-integrated-postflight/v1"
                    or evidence.get("status")
                    != "PASS_CLEANUP_SCIENTIFICALLY_INVALID_BUDGET_CHARGED"
                    or evidence.get("attempt_id") != PRIOR_INVALID_INTEGRATED_ATTEMPT_ID
                    or evidence.get("scientific_status")
                    != "INVALID_PRE_RECLAIM_BOUNDED_BACKLOG_ABORT"
                    or evidence.get("scientifically_valid_integrated_transaction") is not False
                    or evidence.get("failure_stage")
                    != "PRE_RECLAIM_GPU0_OVERLOAD_BACKLOG_SAFETY_ABORT"
                    or evidence.get("reclaim_trigger_reached") is not False
                    or evidence.get("v11_export_started") is not False
                    or evidence.get("v11_release_started") is not False
                    or evidence.get("v11_import_started") is not False
                    or evidence.get("cleanup_pass") is not True
                    or not isinstance(provider, dict)
                    or provider.get("pass") is not True
                    or any(
                        provider.get(field) != 0
                        for field in (
                            "running_tasks",
                            "running_containers",
                            "active_apps",
                            "endpoints",
                            "provider_reservations",
                            "owned_child_processes",
                            "profilers",
                        )
                    )
                    or not isinstance(settlement, dict)
                    or settlement.get("active_ledger_reservations_after") != 0
                    or settlement.get("ledger_sha256_after_settlement")
                    != config.ledger_sha256_before_reservation
                    or not isinstance(remote, dict)
                    or remote.get("manifest_verification_pass") is not True
                ):
                    raise RuntimeError("integrated replacement cleanup payload is inconsistent")

    scopes = budget.get("authorized_scope")
    if (
        budget.get("schema_version")
        != "sloforge.branchfabric.experiment-004-budget-authorization/v1"
        or budget.get("authorized_cumulative_gpu_seconds", 0) < 2 * ABSOLUTE_WALL_SECONDS
        or not isinstance(scopes, list)
        or not any("integrated reclamation" in str(item) for item in scopes)
    ):
        raise RuntimeError("budget authorization does not cover integrated v11")

    return {
        "schema_version": "sloforge.branchfabric.exp004-v11-sealed-evidence-verification/v1",
        "status": "PASS",
        "config_attempt_id": config.attempt_id,
        "micro_attempt_id": micro_attempt,
        "ledger_sha256_before_reservation": config.ledger_sha256_before_reservation,
        "micro_full_physical_amplification": movement["full_physical_amplification"],
        "micro_exact_first_token_matches": 8,
        "review_agents": sorted(agents),
        "integrated_prelaunch_labels": sorted(by_label),
        "replacement_authorization_verified": replacement is not None,
        "final_pre_gpu_gate": final_pre_gpu_gate,
        "artifacts": {
            name: {"reference": references[name][0], "sha256": references[name][1]}
            for name in references
        },
        "verified_at_monotonic_ns": time.monotonic_ns(),
    }


def _validate_engine_start(
    payloads: Sequence[Mapping[str, Any]],
    *,
    children: Sequence[tuple[str, subprocess.Popen[bytes], Any, Any]],
    inventory: Sequence[Mapping[str, Any]],
    launch_started_ns: int,
    observed_ns: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if len(payloads) != 2 or len({item.get("role") for item in payloads}) != 2:
        raise RuntimeError("retained-engine start evidence must cover two distinct roles")
    by_role = {str(item.get("role")): item for item in payloads}
    for (role, child, *_), gpu in zip(children, inventory, strict=True):
        item = by_role.get(role)
        timestamp = None if item is None else item.get("engine_started_ns")
        if (
            item is None
            or item.get("schema_version") != "sloforge.branchfabric.retained-engine-start/v1"
            or item.get("pid") != child.pid
            or item.get("physical_gpu_uuid") != gpu.get("uuid")
            or isinstance(timestamp, bool)
            or not isinstance(timestamp, int)
            or not launch_started_ns <= timestamp <= observed_ns
        ):
            raise RuntimeError(f"invalid retained-engine start evidence for {role}")
    return dict(by_role["serving"]), dict(by_role["rollout"])


def _validate_handoffs(
    payloads: Sequence[Mapping[str, Any]],
    *,
    continuity: Mapping[str, Mapping[str, Any]],
    selection_sha256: str,
    sanity_result_sha256: str,
    authorization_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if len(payloads) != 2:
        raise RuntimeError("transaction handoff requires exactly two workers")
    by_device = {str(item.get("device")): item for item in payloads}
    if set(by_device) != set(DEVICES):
        raise RuntimeError("transaction handoff device coverage is invalid")
    for device in DEVICES:
        item = by_device[device]
        hashes = item.get("pre_gpu_evidence_hashes")
        if (
            item.get("schema_version") != "sloforge.branchfabric.v11-transaction-ready/v1"
            or item.get("passed") is not True
            or item.get("selected_load_sha256") != selection_sha256
            or not isinstance(hashes, dict)
            or hashes.get("sanity_result_sha256") != sanity_result_sha256
            or hashes.get("authorization_artifact_hash") != authorization_sha256
            or item.get("authorization_artifact_hash") != authorization_sha256
            or any(item.get(key) != continuity[device].get(key) for key in continuity[device])
        ):
            raise RuntimeError(f"transaction handoff identity/commitment failed for {device}")
    return dict(by_device["gpu0"]), dict(by_device["gpu1"])


def _validate_worker_results(
    payloads: Sequence[Mapping[str, Any]],
    *,
    expected: Mapping[str, tuple[int, str]],
    attempt_id: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if len(payloads) != 2:
        raise RuntimeError("integrated v11 results do not cover both worker roles")
    validated: list[dict[str, Any]] = []
    for role, payload in zip(ROLES, payloads, strict=True):
        pid, uuid = expected[role]
        compilation = payload.get("measured_transaction_compilation_observation")
        interval_start = (
            compilation.get("interval_start_ns") if isinstance(compilation, dict) else None
        )
        interval_end = compilation.get("interval_end_ns") if isinstance(compilation, dict) else None
        expected_schema = (
            "sloforge.branchfabric.experiment-004-v11-gpu0-result/v1"
            if role == "serving"
            else "sloforge.branchfabric.experiment-004-v11-gpu1-result/v1"
        )
        if (
            payload.get("schema_version") != expected_schema
            or payload.get("status") != "succeeded"
            or payload.get("role") != role
            or payload.get("attempt_id") != attempt_id
            or payload.get("pid") != pid
            or payload.get("physical_gpu_uuid") != uuid
            or not isinstance(compilation, dict)
            or compilation.get("schema_version")
            != "sloforge.branchfabric.measured-transaction-compilation-observation/v1"
            or compilation.get("source") != "bounded-python-logging-handler"
            or compilation.get("role") != role
            or isinstance(interval_start, bool)
            or not isinstance(interval_start, int)
            or isinstance(interval_end, bool)
            or not isinstance(interval_end, int)
            or interval_end <= interval_start
            or compilation.get("capture_buffer_valid") is not True
            or compilation.get("passed") is not True
            or compilation.get("events") != []
            or compilation.get("no_deferred_compilation_event") is not True
        ):
            raise RuntimeError(f"integrated v11 {role} result failed identity/status gates")
        if role == "serving":
            _validate_gpu0_scientific_result(payload)
        else:
            _validate_gpu1_scientific_result(payload)
        validated.append(dict(payload))
    _validate_cross_role_scientific_consistency(validated[0], validated[1])
    return validated[0], validated[1]


def _all_true_fields(value: Any, required: Sequence[str], *, label: str) -> None:
    if not isinstance(value, dict) or any(value.get(field) is not True for field in required):
        raise RuntimeError(f"integrated v11 {label} evidence is incomplete or failed")


def _validate_gpu0_scientific_result(payload: Mapping[str, Any]) -> None:
    _all_true_fields(
        payload.get("scientific_gates"),
        (
            "control_stability_pass",
            "gpu0_overload_pass",
            "bounded_backlog_pass",
            "two_gpu_service_gt_offered_pass",
            "queue_drain_pass",
            "slo_restoration_pass",
            "slo_stability_pass",
            "gpu0_active_during_restore_pass",
        ),
        label="GPU0 scientific gate",
    )
    interference = payload.get("gpu0_restore_interference")
    control = payload.get("gpu0_control_interval")
    if not isinstance(interference, dict):
        raise RuntimeError("integrated v11 GPU0 restore interference evidence is absent")
    arrivals = interference.get("arrivals")
    completions = interference.get("completions")
    emitted = interference.get("emitted_tokens")
    live_samples = interference.get("live_vllm_queue_depth_samples")
    offered_samples = interference.get("offered_outstanding_samples")
    nested_control = interference.get("control_interval")
    restore_interval = interference.get("measured_restore_interval")
    if (
        not isinstance(control, dict)
        or control != nested_control
        or control.get("schema_version") != "sloforge.branchfabric.v11-control-interval/v1"
        or control.get("passed") is not True
        or control.get("duration_seconds") != 5.0
        or control.get("expected_arrivals") != 45
        or control.get("arrivals") != 45
        or control.get("offered_rate_per_second") != 9.0
        or float(control.get("completed_rate_per_second", 0.0)) < 8.1
        or float(control.get("p95_ttft_ns", math.inf)) >= 2_000_000_000
        or int(control.get("maximum_total_outstanding", 20)) >= 20
        or int(control.get("terminal_total_outstanding", 5)) > 4
        or control.get("completion_tracks_offer") is not True
        or control.get("p95_ttft_below_slo") is not True
        or control.get("queue_bounded_below_normal_trigger") is not True
        or control.get("terminal_queue_at_recovery_threshold") is not True
    ):
        raise RuntimeError("integrated v11 GPU0 control-interval evidence is invalid")
    interval_start = (
        restore_interval.get("start_ns") if isinstance(restore_interval, dict) else None
    )
    interval_end = (
        restore_interval.get("end_ns") if isinstance(restore_interval, dict) else None
    )
    interval_duration = (
        restore_interval.get("duration_seconds")
        if isinstance(restore_interval, dict)
        else None
    )
    interval_arrivals = (
        restore_interval.get("arrivals") if isinstance(restore_interval, dict) else None
    )
    interval_completions = (
        restore_interval.get("completions") if isinstance(restore_interval, dict) else None
    )
    interval_tokens = (
        restore_interval.get("emitted_tokens") if isinstance(restore_interval, dict) else None
    )
    interval_samples = (
        restore_interval.get("live_vllm_queue_depth_samples")
        if isinstance(restore_interval, dict)
        else None
    )
    if (
        not isinstance(restore_interval, dict)
        or restore_interval.get("schema_version")
        != "sloforge.branchfabric.v11-gpu0-restore-interference/v1"
        or restore_interval.get("passed") is not True
        or restore_interval.get("methodology_stall_absent") is not True
        or any(
            isinstance(item, bool) or not isinstance(item, int)
            for item in (
                interval_start,
                interval_end,
                interval_arrivals,
                interval_completions,
                interval_tokens,
            )
        )
        or interval_end <= interval_start
        or not isinstance(interval_duration, (int, float))
        or isinstance(interval_duration, bool)
        or not math.isclose(
            float(interval_duration), (interval_end - interval_start) / 1e9, abs_tol=1e-12
        )
        or interval_arrivals <= 0
        or interval_completions <= 0
        or interval_tokens <= 0
        or restore_interval.get("ttft_sample_count") != interval_arrivals
        or not math.isclose(
            float(restore_interval.get("observed_arrival_rate_per_second", math.nan)),
            interval_arrivals / float(interval_duration),
            abs_tol=1e-12,
        )
        or not math.isclose(
            float(restore_interval.get("completion_rate_per_second", math.nan)),
            interval_completions / float(interval_duration),
            abs_tol=1e-12,
        )
        or not math.isclose(
            float(restore_interval.get("token_rate_per_second", math.nan)),
            interval_tokens / float(interval_duration),
            abs_tol=1e-12,
        )
        or not isinstance(interval_samples, (tuple, list))
        or not interval_samples
        or any(
            not isinstance(item, dict)
            or not interval_start <= int(item.get("observed_ns", -1)) <= interval_end
            for item in interval_samples
        )
    ):
        raise RuntimeError("integrated v11 GPU0 actual-restore interference is invalid")
    if (
        interference.get("active_during_restore_pass") is not True
        or isinstance(arrivals, bool)
        or not isinstance(arrivals, int)
        or arrivals <= 0
        or completions != arrivals
        or emitted != arrivals * 64
        or interference.get("offered_rps") != 9.0
        or not isinstance(live_samples, (tuple, list))
        or not live_samples
        or not all(isinstance(item, dict) for item in live_samples)
        or not isinstance(offered_samples, (tuple, list))
        or not offered_samples
    ):
        raise RuntimeError("integrated v11 GPU0 was not measurably active during restore")
    serving = payload.get("serving")
    if not isinstance(serving, dict):
        raise RuntimeError("integrated v11 GPU0 serving trace is absent")
    trigger = serving.get("reclamation_trigger_evidence")
    recovery = serving.get("serving_recovery_evidence")
    events = trigger.get("events") if isinstance(trigger, dict) else None
    trigger_depth = trigger.get("queue_depth_at_trigger") if isinstance(trigger, dict) else None
    emitted_ns = events.get("RECLAIM_TRIGGER_EMITTED") if isinstance(events, dict) else None
    predicate_ns = (
        events.get("TRIGGER_PREDICATE_SATISFIED") if isinstance(events, dict) else None
    )
    overload_ns = events.get("OVERLOAD_DETECTED") if isinstance(events, dict) else None
    reaction_ns = trigger.get("controller_reaction_latency_ns") if isinstance(trigger, dict) else None
    if (
        not isinstance(trigger, dict)
        or trigger.get("schema_version")
        != "sloforge.branchfabric.reclamation-trigger-evidence/v2"
        or trigger.get("overload_confirmed") is not True
        or trigger.get("positive_queue_slope") is not True
        or trigger.get("offered_rate_exceeds_completed_rate") is not True
        or trigger.get("queue_trigger") != 20
        or trigger.get("queue_abort") != 64
        or isinstance(trigger_depth, bool)
        or not isinstance(trigger_depth, int)
        or not 20 <= trigger_depth <= MAX_SCIENTIFIC_TRIGGER_DEPTH
        or trigger.get("emergency_ceiling_headroom_requests") != 64 - trigger_depth
        or any(
            isinstance(item, bool) or not isinstance(item, int)
            for item in (overload_ns, predicate_ns, emitted_ns, reaction_ns)
        )
        or not overload_ns <= predicate_ns <= emitted_ns
        or reaction_ns != emitted_ns - predicate_ns
        or not 0 <= reaction_ns <= MAX_RECLAIM_TRIGGER_REACTION_NS
        or trigger.get("triggered_ns") != emitted_ns
    ):
        raise RuntimeError("integrated v11 GPU0 early-trigger evidence is invalid")
    windows = recovery.get("stability_windows") if isinstance(recovery, dict) else None
    if (
        not isinstance(recovery, dict)
        or recovery.get("schema_version")
        != "sloforge.branchfabric.serving-recovery-evidence/v1"
        or recovery.get("completed_rate_per_second", 0.0)
        <= recovery.get("offered_rate_per_second", math.inf)
        or recovery.get("queue_depth_slope_per_second", 0.0) >= 0.0
        or not isinstance(windows, list)
        or len(windows) != 5
        or any(
            not isinstance(window, dict)
            or window.get("passed") is not True
            or window.get("p95_ttft_ns") is None
            or float(window["p95_ttft_ns"]) > 2_000_000_000
            or int(window.get("queue_depth_end", 5)) > 4
            or int(window.get("ttft_sample_count", 0)) <= 0
            or int(window.get("end_ns", 0)) - int(window.get("start_ns", 0))
            != 1_000_000_000
            for window in windows
        )
        or any(
            int(left["end_ns"]) != int(right["start_ns"])
            for left, right in pairwise(windows)
        )
    ):
        raise RuntimeError("integrated v11 GPU0 recovery/SLO evidence is invalid")


def _validate_non_overlapping_partition(
    value: Any, *, chain: str, expected_stages: Sequence[str]
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RuntimeError(f"integrated v11 {chain} critical path is absent")
    stages = value.get("stages")
    if (
        value.get("schema_version")
        != "sloforge.branchfabric.v11-non-overlapping-critical-path/v1"
        or value.get("chain") != chain
        or value.get("passed") is not True
        or value.get("overlap_double_count_ns") != 0
        or not isinstance(stages, list)
        or [item.get("stage") for item in stages if isinstance(item, dict)]
        != list(expected_stages)
    ):
        raise RuntimeError(f"integrated v11 {chain} critical path is malformed")
    previous_end: int | None = None
    total = 0
    for item in stages:
        start = item.get("start_ns")
        end = item.get("end_ns")
        duration = item.get("wall_time_ns")
        if (
            isinstance(start, bool)
            or not isinstance(start, int)
            or isinstance(end, bool)
            or not isinstance(end, int)
            or isinstance(duration, bool)
            or not isinstance(duration, int)
            or end < start
            or duration != end - start
            or (previous_end is not None and start != previous_end)
        ):
            raise RuntimeError(f"integrated v11 {chain} critical path overlaps or has gaps")
        total += duration
        previous_end = end
    if (
        not stages
        or value.get("start_ns") != stages[0]["start_ns"]
        or value.get("end_ns") != stages[-1]["end_ns"]
        or value.get("wall_time_ns") != total
        or total != int(value["end_ns"]) - int(value["start_ns"])
    ):
        raise RuntimeError(f"integrated v11 {chain} critical path does not conserve time")
    return value


def _validate_gpu1_scientific_result(payload: Mapping[str, Any]) -> None:
    topology = payload.get("topology")
    if not isinstance(topology, dict) or topology != {
        "branch_count": 8,
        "shared_blocks": 1024,
        "private_blocks": 128,
        "total_blocks": 1152,
        "logical_state_bytes": 1_056_964_608,
    }:
        raise RuntimeError("integrated v11 GPU1 topology differs from exact workload")
    _all_true_fields(
        payload.get("correctness"),
        (
            "allocator_epoch_pass",
            "ownership_release_pass",
            "engine_step_binding_pass",
            "integrity_pass",
            "destination_mapping_commitment_pass",
            "fresh_destination_allocations_pass",
            "all_branches_resumed",
            "first_token_exact_8_of_8",
            "movement_accounting_pass",
            "gpu0_active_during_restore_protocol_pass",
        ),
        label="GPU1 correctness gate",
    )
    continuation = payload.get("continuation")
    movement = payload.get("movement")
    passes = payload.get("state_passes")
    if (
        not isinstance(continuation, dict)
        or continuation.get("exact_matches") != 8
        or not isinstance(continuation.get("minimum_tokens_per_branch"), int)
        or isinstance(continuation.get("minimum_tokens_per_branch"), bool)
        or continuation["minimum_tokens_per_branch"] < 8
        or not isinstance(movement, dict)
        or movement.get("movement_accounting_complete") is not True
        or movement.get("logical_state_bytes") != 1_056_964_608
        or not isinstance(movement.get("full_physical_bytes"), int)
        or isinstance(movement.get("full_physical_bytes"), bool)
        or movement["full_physical_bytes"] <= 0
        or not isinstance(passes, list)
        or not passes
        or not all(isinstance(item, dict) for item in passes)
    ):
        raise RuntimeError("integrated v11 GPU1 measured continuation/movement evidence failed")
    timings = payload.get("timings")
    timeline = payload.get("trigger_timeline")
    temporary = payload.get("temporary_memory")
    required_timing_fields = (
        "reclaim_trigger_ns",
        "rollout_admission_stop_ns",
        "state_quiescence_ns",
        "source_authentication_ended_ns",
        "source_pipeline_ended_ns",
        "state_publish_ended_ns",
        "source_release_ended_ns",
        "export_started_ns",
        "export_ended_ns",
        "hbm_reclaim_confirmed_ns",
        "gpu1_serving_ready_ns",
        "restore_trigger_ns",
        "restore_started_ns",
        "checkpoint_authentication_started_ns",
        "checkpoint_authentication_ended_ns",
        "destination_staging_ended_ns",
        "native_restore_pipeline_ended_ns",
        "restore_native_complete_ns",
        "continuation_started_ns",
        "first_resumed_token_ns",
        "all_branches_resumed_ns",
        "continuation_complete_ns",
    )
    if (
        not isinstance(timings, dict)
        or any(
            isinstance(timings.get(field), bool) or not isinstance(timings.get(field), int)
            for field in required_timing_fields
        )
        or not (
            timings["reclaim_trigger_ns"]
            <= timings["rollout_admission_stop_ns"]
            <= timings["export_started_ns"]
            <= timings["state_quiescence_ns"]
            <= timings["source_authentication_ended_ns"]
            <= timings["source_pipeline_ended_ns"]
            <= timings["state_publish_ended_ns"]
            <= timings["source_release_ended_ns"]
            <= timings["export_ended_ns"]
            <= timings["hbm_reclaim_confirmed_ns"]
            <= timings["gpu1_serving_ready_ns"]
            <= timings["restore_trigger_ns"]
            <= timings["restore_started_ns"]
            <= timings["checkpoint_authentication_started_ns"]
            <= timings["checkpoint_authentication_ended_ns"]
            <= timings["destination_staging_ended_ns"]
            <= timings["native_restore_pipeline_ended_ns"]
            <= timings["restore_native_complete_ns"]
            <= timings["continuation_started_ns"]
            <= timings["first_resumed_token_ns"]
            <= timings["all_branches_resumed_ns"]
            <= timings["continuation_complete_ns"]
        )
        or not isinstance(timeline, dict)
        or timeline.get("passed") is not True
        or timeline.get("trigger_precedes_state_quiescence") is not True
        or not isinstance(temporary, dict)
        or any(
            isinstance(temporary.get(field), bool)
            or not isinstance(temporary.get(field), int)
            or temporary[field] < 0
            for field in (
                "peak_pinned_host_temporary_bytes",
                "peak_pageable_host_temporary_bytes",
                "peak_gpu_temporary_bytes",
            )
        )
    ):
        raise RuntimeError("integrated v11 GPU1 timing/temporary-memory evidence failed")
    critical_paths = payload.get("critical_paths")
    if not isinstance(critical_paths, dict):
        raise RuntimeError("integrated v11 critical-path evidence is absent")
    reclaim_partition = _validate_non_overlapping_partition(
        critical_paths.get("reclamation"),
        chain="RECLAIM_TRIGGER_TO_GPU1_SERVING_READY",
        expected_stages=(
            "branch_quiesce",
            "source_authentication",
            "fused_gather_repack_d2h",
            "publish_and_release_preconditions",
            "source_release",
            "hbm_reclaim_confirmation",
            "gpu1_serving_ready",
        ),
    )
    restore_partition = _validate_non_overlapping_partition(
        critical_paths.get("restore"),
        chain="RESTORE_TRIGGER_TO_FIRST_RESUMED_TOKEN",
        expected_stages=(
            "restore_preconditions",
            "checkpoint_authentication",
            "destination_allocation_staging",
            "fused_h2d_direct_scatter_raw_validation_and_admission",
            "canonical_pass_validation_and_commit",
            "first_resumed_token",
        ),
    )
    if (
        reclaim_partition["start_ns"] != timings["reclaim_trigger_ns"]
        or reclaim_partition["end_ns"] != timings["gpu1_serving_ready_ns"]
        or restore_partition["start_ns"] != timings["restore_trigger_ns"]
        or restore_partition["end_ns"] != timings["first_resumed_token_ns"]
        or critical_paths.get("residual_source_chain_ns")
        != timings["source_pipeline_ended_ns"] - timings["state_quiescence_ns"]
        or critical_paths.get("residual_restore_chain_ns")
        != timings["restore_native_complete_ns"] - timings["restore_started_ns"]
        or not isinstance(critical_paths.get("stage_overlap_policy"), str)
        or not critical_paths["stage_overlap_policy"]
    ):
        raise RuntimeError("integrated v11 critical-path boundaries are inconsistent")
    pass_ids: set[str] = set()
    capture_count = 0
    restore_count = 0
    for record in passes:
        pass_id = record.get("pass_id")
        wall_start = record.get("wall_start_ns")
        wall_end = record.get("wall_end_ns")
        if (
            record.get("schema_version") != "sloforge.branchfabric.state-pass-record/v11"
            or not isinstance(pass_id, str)
            or not pass_id
            or pass_id in pass_ids
            or isinstance(wall_start, bool)
            or not isinstance(wall_start, int)
            or isinstance(wall_end, bool)
            or not isinstance(wall_end, int)
            or wall_end < wall_start
            or isinstance(record.get("physical_touch_bytes"), bool)
            or not isinstance(record.get("physical_touch_bytes"), int)
            or record["physical_touch_bytes"] < 0
        ):
            raise RuntimeError("integrated v11 GPU1 canonical StatePassRecord is invalid")
        pass_ids.add(pass_id)
        if pass_id.startswith("capture:"):
            capture_count += 1
            if not timings["export_started_ns"] <= wall_start <= wall_end <= timings["export_ended_ns"]:
                raise RuntimeError("integrated v11 capture pass escaped its measured wall chain")
        elif pass_id.startswith("restore:"):
            restore_count += 1
            if not timings["restore_started_ns"] <= wall_start <= wall_end <= timings["restore_native_complete_ns"]:
                raise RuntimeError("integrated v11 restore pass escaped its measured wall chain")
        else:
            raise RuntimeError("integrated v11 StatePassRecord has an unknown chain prefix")
    movement_by_phase = payload.get("movement_by_phase")
    if not isinstance(movement_by_phase, dict) or set(movement_by_phase) != {
        "source_export",
        "rollout_restore",
    }:
        raise RuntimeError("integrated v11 per-phase movement accounting is absent")
    export_movement = movement_by_phase["source_export"]
    restore_movement = movement_by_phase["rollout_restore"]
    phase_movements = (export_movement, restore_movement)
    movement_integer_fields = (
        "full_physical_bytes",
        "external_movement_bytes",
        "avoidable_physical_bytes",
        "required_physical_bytes",
        "diagnostic_physical_bytes",
        "record_count",
    )
    if any(
        not isinstance(item, dict)
        or item.get("movement_accounting_complete") is not True
        or item.get("logical_state_bytes") != 1_056_964_608
        or any(
            isinstance(item.get(field), bool)
            or not isinstance(item.get(field), int)
            or item[field] < 0
            for field in movement_integer_fields
        )
        or not math.isclose(
            float(item.get("full_physical_amplification", math.nan)),
            item["full_physical_bytes"] / 1_056_964_608,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or not math.isclose(
            float(item.get("external_movement_amplification", math.nan)),
            item["external_movement_bytes"] / 1_056_964_608,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or not math.isclose(
            float(item.get("avoidable_amplification", math.nan)),
            item["avoidable_physical_bytes"] / 1_056_964_608,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        for item in phase_movements
    ):
        raise RuntimeError("integrated v11 per-phase movement accounting is invalid")
    logical = movement["logical_state_bytes"]
    if (
        capture_count == 0
        or restore_count == 0
        or export_movement["record_count"] != capture_count
        or restore_movement["record_count"] != restore_count
        or any(
            export_movement[field] + restore_movement[field] != movement.get(field)
            for field in movement_integer_fields
        )
        or movement.get("record_count") != len(passes)
        or sum(int(record["physical_touch_bytes"]) for record in passes)
        != movement.get("full_physical_bytes")
        or any(
            isinstance(movement.get(field), bool)
            or not isinstance(movement.get(field), int)
            or movement[field] < 0
            for field in (
                "full_physical_bytes",
                "external_movement_bytes",
                "avoidable_physical_bytes",
                "required_physical_bytes",
            )
        )
        or not math.isclose(
            float(movement.get("full_physical_amplification", math.nan)),
            movement["full_physical_bytes"] / logical,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or not math.isclose(
            float(movement.get("external_movement_amplification", math.nan)),
            movement["external_movement_bytes"] / logical,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or not math.isclose(
            float(movement.get("avoidable_amplification", math.nan)),
            movement["avoidable_physical_bytes"] / logical,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ):
        raise RuntimeError("integrated v11 movement totals are incomplete or inconsistent")


def _validate_cross_role_scientific_consistency(
    gpu0: Mapping[str, Any], gpu1: Mapping[str, Any]
) -> None:
    """Bind cross-process monotonic evidence into one causal transaction."""

    gpu0_serving = gpu0["serving"]
    trigger = gpu0_serving["reclamation_trigger_evidence"]
    recovery = gpu0_serving["serving_recovery_evidence"]
    gpu1_serving = gpu1.get("serving")
    timings = gpu1["timings"]
    gpu1_requests = gpu1_serving.get("requests") if isinstance(gpu1_serving, dict) else None
    if not isinstance(gpu1_requests, (tuple, list)) or not gpu1_requests:
        raise RuntimeError("integrated v11 GPU1 useful-serving request evidence is absent")
    useful_tokens = [
        row.get("first_token_ns")
        for row in gpu1_requests
        if isinstance(row, dict) and row.get("first_token_ns") is not None
    ]
    if not useful_tokens or any(isinstance(item, bool) or not isinstance(item, int) for item in useful_tokens):
        raise RuntimeError("integrated v11 GPU1 useful-serving timestamps are invalid")
    first_useful_ns = recovery.get("gpu1_first_useful_ns")
    windows = recovery["stability_windows"]
    slo_restoration_ns = int(windows[0]["start_ns"])
    gpu0_restore = gpu0_serving.get("restore_start")
    if (
        trigger["events"]["RECLAIM_TRIGGER_EMITTED"] != timings["reclaim_trigger_ns"]
        or isinstance(first_useful_ns, bool)
        or not isinstance(first_useful_ns, int)
        or first_useful_ns not in useful_tokens
        or not isinstance(gpu0_restore, dict)
        or gpu0_restore.get("observed_ns") != timings["restore_trigger_ns"]
        or not (
            timings["hbm_reclaim_confirmed_ns"]
            <= first_useful_ns
            <= slo_restoration_ns
            <= timings["restore_trigger_ns"]
        )
    ):
        raise RuntimeError("integrated v11 cross-role timestamp/accounting evidence conflicts")


def _worker_command(
    *,
    worker_path: Path,
    role: str,
    config_path: Path,
    model_snapshot: Path,
    role_root: Path,
    barrier_root: Path,
    physical_gpu_uuid: str,
) -> list[str]:
    if role not in ROLES:
        raise ValueError("integrated v11 worker role is invalid")
    return [
        sys.executable,
        str(worker_path),
        "--role",
        role,
        "--config",
        str(config_path),
        "--model-snapshot",
        str(model_snapshot),
        "--work-root",
        str(role_root),
        "--barrier-root",
        str(barrier_root),
        "--physical-gpu-uuid",
        physical_gpu_uuid,
    ]


def _transaction_command(
    *,
    config: Experiment004V11IntegratedConfig,
    effective_config: Mapping[str, Any],
    selection_sha256: str,
    sanity_result_sha256: str,
) -> dict[str, Any]:
    expected = expanded_runtime_config(config)
    if dict(effective_config) != expected:
        raise RuntimeError("v11 transaction effective config differs from expanded mapping")
    return {
        "schema_version": "sloforge.branchfabric.integrated-transaction-command/v1",
        "effective_config": expected,
        "selection_sha256": selection_sha256,
        "authorization_artifact_hash": config.budget_authorization_sha256,
        "sanity_result_sha256": sanity_result_sha256,
        "issued_at_monotonic_ns": time.monotonic_ns(),
    }


def _wait_for_postflight_zero_compute(*, deadline_ns: int) -> tuple[dict[str, Any], ...]:
    """Tolerate bounded NVML process-list lag without extending the paid cap."""

    while True:
        processes = _compute_processes(deadline_ns=deadline_ns)
        if not processes:
            return processes
        if time.monotonic_ns() >= deadline_ns:
            _require_no_compute_processes(processes, phase="integrated v11 postflight")
        time.sleep(0.1)


def _partition_controller_deadlines(
    *,
    started_ns: int,
    caller_deadline_ns: int,
    cleanup_reserve_seconds: int,
) -> tuple[int, int]:
    if cleanup_reserve_seconds <= 0 or cleanup_reserve_seconds >= ABSOLUTE_WALL_SECONDS:
        raise ValueError("integrated v11 cleanup reserve is outside the absolute wall bound")
    controller_deadline_ns = min(
        caller_deadline_ns,
        started_ns + round(ABSOLUTE_WALL_SECONDS * 1e9),
    )
    operation_deadline_ns = controller_deadline_ns - cleanup_reserve_seconds * 1_000_000_000
    if operation_deadline_ns <= started_ns:
        raise TimeoutError("integrated v11 caller left no bounded operation window")
    return operation_deadline_ns, controller_deadline_ns


def run_integrated_v11_controller(
    *,
    config_path: Path,
    work_root: Path,
    worker_path: Path,
    model_snapshot: Path,
    repository_root: Path,
    absolute_deadline_ns: int,
) -> dict[str, Any]:
    """Run exactly one two-A100 v11 integrated transaction under a 588s cap."""

    started_ns = time.monotonic_ns()
    if (
        isinstance(absolute_deadline_ns, bool)
        or not isinstance(absolute_deadline_ns, int)
        or absolute_deadline_ns <= started_ns
    ):
        raise ValueError("integrated v11 requires a future absolute deadline")
    if V11_INTEGRATED_RESERVATION_WALL_SECONDS != ABSOLUTE_WALL_SECONDS:
        raise RuntimeError("integrated v11 methodology wall bound drifted from 588 seconds")
    if not config_path.is_file() or not worker_path.is_file() or not model_snapshot.is_dir():
        raise FileNotFoundError("integrated v11 controller input path is absent")
    root = repository_root.resolve(strict=True)
    config = Experiment004V11IntegratedConfig.model_validate_json(
        config_path.read_text(), strict=True
    )
    operation_deadline_ns, cleanup_deadline_ns = _partition_controller_deadlines(
        started_ns=started_ns,
        caller_deadline_ns=absolute_deadline_ns,
        cleanup_reserve_seconds=config.cleanup_timeout_seconds,
    )
    sealed = verify_sealed_evidence(config, root)
    # Constructing the typed live contract is an additional exact-methodology
    # check.  The worker protocol itself consumes the expanded JSON mapping.
    build_live_v10_config(config)
    effective_config = expanded_runtime_config(config)
    if time.monotonic_ns() >= operation_deadline_ns:
        raise TimeoutError("integrated v11 sealed preflight exhausted its operation deadline")

    # Load CPU-only capacity contracts after the controller's own sealed-input
    # checks.  These modules define JSON models; neither owns a CUDA context.
    from gpu_capacity_calibration_controller import (
        _validate_readiness_payloads,
        _validate_reset_payloads,
    )
    from gpu_reclamation_controller import _assess_sanity_guard

    from sloforge.helix.characterization.gpu_capacity_calibration import (
        CapacityProbeRaw,
        ProbeTopology,
        build_probe_plan,
        evaluate_probe,
    )

    work_root.mkdir(parents=True, exist_ok=False)
    barrier_root = work_root / "barriers"
    capacity_root = barrier_root / "capacity"
    log_root = work_root / "logs"
    capacity_root.mkdir(parents=True)
    log_root.mkdir()
    base_config_path = work_root / "base-config.json"
    _write_new(base_config_path, config)
    _write_new(work_root / "authorization/sealed-evidence.json", sealed)

    before_audit = cuda_clean_import_audit("v11-integrated-before-inventory")
    if not before_audit["cuda_clean"]:
        raise RuntimeError("integrated v11 controller imported a CUDA-owning module")
    inventory_before = _inventory(deadline_ns=operation_deadline_ns)
    _require_no_compute_processes(
        _compute_processes(deadline_ns=operation_deadline_ns),
        phase="integrated v11 preflight",
    )

    children: list[tuple[str, subprocess.Popen[bytes], Any, Any]] = []
    lifecycle = _ProcessLifecycleAudit()
    cleanup_actions: list[dict[str, Any]] = []
    cleanup_errors: list[dict[str, Any]] = []
    controller_error: BaseException | None = None
    cleanup_error: BaseException | None = None
    readiness_deadline_ns: int | None = None
    engine_start_evidence: tuple[dict[str, Any], ...] = ()
    ready_payloads: tuple[dict[str, Any], ...] = ()
    sanity_pair: dict[str, Any] | None = None
    result_payloads: tuple[dict[str, Any], ...] = ()
    inventory_after: tuple[dict[str, Any], ...] = ()
    processes_after: tuple[dict[str, Any], ...] = ()
    surviving_children: tuple[dict[str, Any], ...] = ()
    surviving_groups: tuple[int, ...] = ()
    cuda_release_verified = False
    in_function_cleanup: dict[str, Any] | None = None
    try:
        lifecycle.transition("RUNNING")
        launch_started_ns = time.monotonic_ns()
        for role, gpu in zip(ROLES, inventory_before, strict=True):
            role_root = work_root / role
            role_root.mkdir()
            stdout_handle = (log_root / f"{role}.stdout.log").open("xb")
            stderr_handle = (log_root / f"{role}.stderr.log").open("xb")
            environment = dict(os.environ)
            environment["CUDA_VISIBLE_DEVICES"] = str(gpu["index"])
            environment["SLOFORGE_EXP004_VISIBLE_GPU_INDEX"] = str(gpu["index"])
            environment["SLOFORGE_EXP004_PHYSICAL_GPU_UUID"] = str(gpu["uuid"])
            command = _worker_command(
                worker_path=worker_path,
                role=role,
                config_path=config_path,
                model_snapshot=model_snapshot,
                role_root=role_root,
                barrier_root=barrier_root,
                physical_gpu_uuid=str(gpu["uuid"]),
            )
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
            lifecycle.register_worker(role, child)
        processes = tuple((role, child) for role, child, *_ in children)

        engine_paths = tuple(barrier_root / f"{role}.engine-started.json" for role in ROLES)
        _wait_for_files(
            engine_paths,
            deadline_ns=operation_deadline_ns,
            phase="retained-engine startup",
            processes=processes,
            observe_processes=lifecycle.observe,
        )
        engine_start_evidence = _validate_engine_start(
            tuple(_load_json_object(path, label="engine-start") for path in engine_paths),
            children=children,
            inventory=inventory_before,
            launch_started_ns=launch_started_ns,
            observed_ns=time.monotonic_ns(),
        )
        readiness_origin_ns = min(item["engine_started_ns"] for item in engine_start_evidence)
        readiness_deadline_ns = min(
            operation_deadline_ns,
            readiness_origin_ns + config.initialization_timeout_seconds * 1_000_000_000,
        )
        ready_paths = tuple(barrier_root / f"{role}.ready.json" for role in ROLES)
        _wait_for_files(
            ready_paths,
            deadline_ns=readiness_deadline_ns,
            phase="integrated v11 readiness",
            processes=processes,
            observe_processes=lifecycle.observe,
        )
        ready_payloads = tuple(
            _validate_readiness_payloads(
                payloads=tuple(
                    _load_json_object(path, label="engine readiness") for path in ready_paths
                ),
                expected_inventory=inventory_before,
            )
        )
        _write_new(
            work_root / "readiness/both-engines-ready.json",
            {
                "schema_version": "sloforge.branchfabric.both-engines-ready/v1",
                "BOTH_ENGINES_READY": True,
                "attempt_id": config.attempt_id,
                "engine_evidence": ready_payloads,
                "observed_at_monotonic_ns": time.monotonic_ns(),
            },
        )

        assessments: list[dict[str, Any]] = []
        for probe_index, rate in enumerate(SANITY_RATES_RPS):
            plan = build_probe_plan(
                probe_id=f"v11-sanity-{int(rate)}rps",
                seed=config.seed,
                topology=ProbeTopology.GPU0_ONLY,
                configured_rate_rps=rate,
                start_ns=time.monotonic_ns() + 250_000_000,
                warmup_seconds=config.warmup_seconds,
                measurement_seconds=config.sanity_guard_measurement_seconds,
            )
            _write_new(
                capacity_root / f"probe-{probe_index:02d}.command.json",
                {
                    "schema_version": "sloforge.branchfabric.capacity-probe-command/v1",
                    "reason": "v11 preauthorized short stale-calibration sanity guard",
                    "plan": plan,
                },
            )
            raw_paths = tuple(
                capacity_root / f"probe-{probe_index:02d}.{device}.raw.json" for device in DEVICES
            )
            reset_paths = tuple(
                capacity_root / f"probe-{probe_index:02d}.{device}.reset.json" for device in DEVICES
            )
            _wait_for_files(
                raw_paths + reset_paths,
                deadline_ns=min(
                    operation_deadline_ns,
                    time.monotonic_ns() + round(config.probe_timeout_seconds * 1e9),
                ),
                phase=f"v11 sanity probe {probe_index}",
                processes=processes,
                observe_processes=lifecycle.observe,
            )
            raw_payloads = tuple(
                _load_json_object(path, label="capacity raw shard") for path in raw_paths
            )
            if {item.get("device") for item in raw_payloads} != set(DEVICES):
                raise RuntimeError("v11 sanity raw shards do not cover gpu0/gpu1")
            resets = _validate_reset_payloads(
                payloads=tuple(
                    _load_json_object(path, label="capacity reset") for path in reset_paths
                ),
                expected_probe_id=plan.probe_id,
            )
            if {item.get("device") for item in resets} != set(DEVICES):
                raise RuntimeError("v11 sanity reset shards do not cover gpu0/gpu1")
            observations = tuple(
                row for payload in raw_payloads for row in payload.get("observations", ())
            )
            encoded_rows = _canonical_bytes(list(observations))
            from sloforge.helix.characterization.gpu_capacity_calibration import (
                CapacityRequestObservation,
            )

            typed_observations = tuple(
                CapacityRequestObservation.model_validate_json(_canonical_bytes(row), strict=True)
                for row in json.loads(encoded_rows)
            )
            expected_tail_end_ns = plan.measurement_end_ns + round(config.tail_drain_seconds * 1e9)
            raw = CapacityProbeRaw(
                plan=plan,
                observations=typed_observations,
                tail_drain_end_ns=expected_tail_end_ns,
                probe_end_ns=max(int(payload["probe_end_ns"]) for payload in raw_payloads),
                tail_drain_seconds=config.tail_drain_seconds,
            )
            result = evaluate_probe(raw, slo_ttft_seconds=config.serving_slo_ttft_seconds)
            gpu1_payload = next(item for item in raw_payloads if item.get("device") == "gpu1")
            if gpu1_payload.get("observations") != []:
                raise RuntimeError("GPU1 received work during a GPU0-only sanity guard")
            assessment = _assess_sanity_guard(result, expected_rate_rps=rate)
            if assessment.get("passed") is not True:
                raise RuntimeError(f"v11 sanity guard failed: {assessment}")
            probe_root = work_root / "sanity" / f"{int(rate)}-rps"
            _write_new(probe_root / "plan.json", plan)
            _write_new(probe_root / "raw.json", raw)
            _write_new(probe_root / "worker-shards.json", raw_payloads)
            _write_new(probe_root / "result.json", result)
            _write_new(probe_root / "assessment.json", assessment)
            _write_new(probe_root / "request-state-reset.json", resets)
            assessments.append(assessment)
        sanity_pair = validate_sanity_guard_pair(assessments)
        sanity_path = work_root / "sanity/sanity-result.json"
        _write_new(sanity_path, sanity_pair)
        sanity_sha256 = _sha256(sanity_path)

        selection_path = work_root / "calibration/selected-load.json"
        _write_new(
            selection_path,
            {
                "schema_version": "sloforge.branchfabric.v11-stale-calibration-selection/v1",
                "lambda_1_rps": config.lambda_1_rps,
                "lambda_spike_rps": config.lambda_spike_rps,
                "lambda_2_rps": config.lambda_2_rps,
                "sanity_result_sha256": sanity_sha256,
                "broad_capacity_calibration_performed": False,
            },
        )
        selection_sha256 = _sha256(selection_path)
        _write_new(
            capacity_root / "final-reset.command.json",
            {
                "schema_version": "sloforge.branchfabric.integrated-final-reset-command/v1",
                "selected_load_sha256": selection_sha256,
                "authorization_artifact_hash": config.budget_authorization_sha256,
                "issued_at_monotonic_ns": time.monotonic_ns(),
            },
        )
        final_reset_paths = tuple(
            capacity_root / f"final-reset.{device}.json" for device in DEVICES
        )
        _wait_for_files(
            final_reset_paths,
            deadline_ns=operation_deadline_ns,
            phase="v11 final request reset",
            processes=processes,
            observe_processes=lifecycle.observe,
        )
        final_resets = _validate_reset_payloads(
            payloads=tuple(
                _load_json_object(path, label="final reset") for path in final_reset_paths
            ),
            expected_probe_id="final-calibration-reset",
        )
        if {item.get("device") for item in final_resets} != set(DEVICES):
            raise RuntimeError("v11 final reset shards do not cover gpu0/gpu1")
        continuity = {
            str(payload["device"]): {
                "pid": int(payload["pid"]),
                "engine_nonce": str(payload["engine_nonce"]),
                "physical_gpu_uuid": str(payload["physical_gpu_uuid"]),
            }
            for payload in ready_payloads
        }
        if set(continuity) != set(DEVICES):
            raise RuntimeError("readiness identities do not cover gpu0/gpu1")
        for payload in final_resets:
            device = str(payload["device"])
            if (
                payload.get("selected_load_sha256") != selection_sha256
                or payload.get("authorization_artifact_hash") != config.budget_authorization_sha256
                or any(payload.get(key) != continuity[device][key] for key in continuity[device])
            ):
                raise RuntimeError("worker identity changed before final reset")

        effective_path = work_root / "effective-config.json"
        _write_new(effective_path, effective_config)
        _write_new(
            barrier_root / "transaction.command.json",
            _transaction_command(
                config=config,
                effective_config=effective_config,
                selection_sha256=selection_sha256,
                sanity_result_sha256=sanity_sha256,
            ),
        )
        handoff_paths = tuple(barrier_root / f"{role}.transaction-ready.json" for role in ROLES)
        _wait_for_files(
            handoff_paths,
            deadline_ns=operation_deadline_ns,
            phase="v11 transaction handoff",
            processes=processes,
            observe_processes=lifecycle.observe,
        )
        _validate_handoffs(
            tuple(_load_json_object(path, label="transaction handoff") for path in handoff_paths),
            continuity=continuity,
            selection_sha256=selection_sha256,
            sanity_result_sha256=sanity_sha256,
            authorization_sha256=config.budget_authorization_sha256,
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
        while any(child.poll() is None for _, child, *_ in children):
            lifecycle.observe("integrated-v11-transaction")
            if time.monotonic_ns() >= operation_deadline_ns:
                raise TimeoutError("integrated v11 transaction exceeded the 588-second bound")
            failed = [
                (role, child.returncode)
                for role, child, *_ in children
                if child.poll() not in {None, 0}
            ]
            if failed:
                raise RuntimeError(f"an integrated v11 worker failed: {failed}")
            time.sleep(0.1)
        returncodes = {role: child.returncode for role, child, *_ in children}
        if set(returncodes.values()) != {0}:
            raise RuntimeError(f"integrated v11 worker return codes failed: {returncodes}")
        result_paths = tuple(work_root / role / "result.json" for role in ROLES)
        _wait_for_files(
            result_paths,
            deadline_ns=operation_deadline_ns,
            phase="integrated v11 worker results",
            processes=processes,
            observe_processes=lifecycle.observe,
        )
        expected = {
            role: (child.pid, str(gpu["uuid"]))
            for (role, child, *_), gpu in zip(children, inventory_before, strict=True)
        }
        result_payloads = _validate_worker_results(
            tuple(_load_json_object(path, label="worker result") for path in result_paths),
            expected=expected,
            attempt_id=config.attempt_id,
        )
        manifest_paths = sorted(
            {
                base_config_path,
                effective_path,
                sanity_path,
                selection_path,
                barrier_root / "transaction.command.json",
                *(work_root / role / "result.json" for role in ROLES),
            },
            key=str,
        )
        _write_new(
            work_root / "integrated-run-manifest.json",
            {
                "schema_version": "sloforge.branchfabric.v11-integrated-run-manifest/v1",
                "attempt_id": config.attempt_id,
                "single_worker_pair": True,
                "engine_reload_count": 0,
                "sanity_and_transaction_same_allocation": True,
                "broad_capacity_calibration_performed": False,
                "absolute_wall_seconds": ABSOLUTE_WALL_SECONDS,
                "sealed_evidence": sealed,
                "artifacts": [
                    {
                        "relative_path": str(path.relative_to(work_root)),
                        "sha256": _sha256(path),
                    }
                    for path in manifest_paths
                ],
            },
        )
    except BaseException as caught:
        controller_error = caught
        abort = barrier_root / "abort.json"
        if not abort.exists():
            with suppress(FileExistsError):
                _write_new(
                    abort,
                    {
                        "schema_version": "sloforge.branchfabric.integrated-abort/v1",
                        "error": type(caught).__name__,
                        "issued_at_monotonic_ns": time.monotonic_ns(),
                    },
                )
    finally:
        process_objects = tuple(child for _role, child, *_ in children)
        try:
            lifecycle.transition("QUIESCE")
            lifecycle.observe("quiesce", force=True)
            lifecycle.transition("ENGINE_STOP")
            graceful_deadline_ns = min(
                cleanup_deadline_ns,
                time.monotonic_ns() + 1_000_000_000,
            )
            _wait_for_owned_exit(
                lifecycle,
                process_objects,
                deadline_ns=graceful_deadline_ns,
                stage="engine-stop-grace",
            )
            lifecycle.transition("WORKER_STOP")
            _terminate_groups(
                process_objects,
                cleanup_actions,
                deadline_ns=cleanup_deadline_ns,
                lifecycle=lifecycle,
            )
        except BaseException as caught:
            cleanup_errors.append(
                {
                    "stage": "worker-stop",
                    "type": type(caught).__name__,
                    "message": str(caught),
                }
            )
            if cleanup_error is None:
                cleanup_error = caught
        finally:
            try:
                escaped = lifecycle.currently_owned(stage="escaped-helper-term")
                grouped = {int(item["pgid"]) for item in lifecycle.worker_roots.values()}
                escaped = tuple(item for item in escaped if int(item["pgid"]) not in grouped)
                if escaped:
                    _signal_exact_owned(escaped, signal.SIGTERM, cleanup_actions)
                    escaped = _wait_for_owned_exit(
                        lifecycle,
                        process_objects,
                        deadline_ns=min(
                            cleanup_deadline_ns,
                            time.monotonic_ns() + 1_000_000_000,
                        ),
                        stage="escaped-helper-term-wait",
                    )
                    escaped = tuple(item for item in escaped if int(item["pgid"]) not in grouped)
                if escaped:
                    _signal_exact_owned(escaped, signal.SIGKILL, cleanup_actions)
                lifecycle.transition("CHILD_REAP")
                surviving_children = _reap_owned_children(
                    lifecycle,
                    process_objects,
                    deadline_ns=cleanup_deadline_ns,
                )
            except BaseException as caught:
                cleanup_errors.append(
                    {
                        "stage": "child-reap",
                        "type": type(caught).__name__,
                        "message": str(caught),
                    }
                )
                if cleanup_error is None:
                    cleanup_error = caught
            for _role, _child, stdout_handle, stderr_handle in children:
                stdout_handle.close()
                stderr_handle.close()
            try:
                lifecycle.subreaper.restore()
            except BaseException as caught:
                cleanup_errors.append(
                    {
                        "stage": "child-subreaper-restore",
                        "type": type(caught).__name__,
                        "message": str(caught),
                    }
                )
                if cleanup_error is None:
                    cleanup_error = caught
            try:
                lifecycle.transition("PGID_EMPTY")
                _poll_and_reap_workers(process_objects, lifecycle=lifecycle)
                surviving_children = lifecycle.currently_owned(stage="pgid-empty")
                surviving_groups = tuple(
                    sorted(
                        int(item["pgid"])
                        for item in lifecycle.worker_roots.values()
                        if _process_group_exists(int(item["pgid"]))
                    )
                )
                if surviving_children or surviving_groups:
                    raise RuntimeError(
                        "experiment-owned processes remain after bounded cleanup: "
                        f"pids={[item['pid'] for item in surviving_children]}, "
                        f"pgids={list(surviving_groups)}"
                    )
            except BaseException as caught:
                cleanup_errors.append(
                    {
                        "stage": "pgid-empty",
                        "type": type(caught).__name__,
                        "message": str(caught),
                    }
                )
                if cleanup_error is None:
                    cleanup_error = caught

    try:
        processes_after = _wait_for_postflight_zero_compute(deadline_ns=cleanup_deadline_ns)
        inventory_after = _inventory(deadline_ns=cleanup_deadline_ns)
        if not _stable_inventory_identity(inventory_before, inventory_after):
            raise RuntimeError("physical GPU identity changed during integrated v11")
        cuda_release_verified = True
    except BaseException as caught:
        cleanup_errors.append(
            {
                "stage": "cuda-released",
                "type": type(caught).__name__,
                "message": str(caught),
            }
        )
        if cleanup_error is None:
            cleanup_error = caught
    lifecycle.transition("CUDA_RELEASED")
    after_audit = cuda_clean_import_audit("v11-integrated-after-worker-cleanup")
    if not after_audit["cuda_clean"]:
        caught = RuntimeError("controller imported a CUDA-owning module")
        cleanup_errors.append(
            {
                "stage": "cuda-clean-parent",
                "type": type(caught).__name__,
                "message": str(caught),
            }
        )
        if cleanup_error is None:
            cleanup_error = caught
    ended_ns = time.monotonic_ns()
    if ended_ns > cleanup_deadline_ns:
        caught = TimeoutError("integrated v11 controller exceeded its absolute deadline")
        cleanup_errors.append(
            {
                "stage": "absolute-deadline",
                "type": type(caught).__name__,
                "message": str(caught),
            }
        )
        if cleanup_error is None:
            cleanup_error = caught
    lifecycle.transition("FUNCTION_RETURN")
    in_function_cleanup = _build_in_function_cleanup(
        lifecycle=lifecycle,
        actions=cleanup_actions,
        processes=tuple(child for _role, child, *_ in children),
        surviving_children=surviving_children,
        surviving_groups=surviving_groups,
        compute_processes_after=processes_after,
        cuda_release_verified=cuda_release_verified,
        pipes_closed=all(
            stdout_handle.closed and stderr_handle.closed
            for _role, _child, stdout_handle, stderr_handle in children
        ),
        cleanup_errors=cleanup_errors,
    )
    _write_new(work_root / "in_function_cleanup.json", in_function_cleanup)
    if in_function_cleanup["pass"] is not True and cleanup_error is None:
        cleanup_error = RuntimeError("integrated v11 in-function cleanup gate failed")
    status = (
        "succeeded"
        if controller_error is None
        and cleanup_error is None
        and in_function_cleanup["pass"] is True
        else "failed"
    )
    result = {
        "schema_version": "sloforge.branchfabric.experiment-004-v11-integrated-controller/v1",
        "status": status,
        "attempt_id": config.attempt_id,
        "controller_pid": os.getpid(),
        "started_ns": started_ns,
        "ended_ns": ended_ns,
        "operation_deadline_ns": operation_deadline_ns,
        "cleanup_deadline_ns": cleanup_deadline_ns,
        "absolute_wall_seconds": ABSOLUTE_WALL_SECONDS,
        "sealed_evidence": sealed,
        "inventory_before": inventory_before,
        "inventory_after": inventory_after,
        "stable_physical_gpu_identity": _stable_inventory_identity(
            inventory_before, inventory_after
        ),
        "worker_pids": {role: child.pid for role, child, *_ in children},
        "worker_process_groups": {
            str(item["role"]): int(item["pgid"]) for item in lifecycle.worker_roots.values()
        },
        "worker_session_ids": {
            str(item["role"]): int(item["sid"]) for item in lifecycle.worker_roots.values()
        },
        "worker_returncodes": {role: child.returncode for role, child, *_ in children},
        "engine_start_evidence": engine_start_evidence,
        "readiness_evidence": ready_payloads,
        "readiness_deadline_ns": readiness_deadline_ns,
        "sanity_guard_pair": sanity_pair,
        "worker_results": result_payloads,
        "cleanup_actions": cleanup_actions,
        "in_function_cleanup": in_function_cleanup,
        "compute_processes_after": processes_after,
        "cuda_clean_import_audits": (before_audit, after_audit),
        "controller_error": None
        if controller_error is None
        else {"type": type(controller_error).__name__, "message": str(controller_error)},
        "cleanup_error": None
        if cleanup_error is None
        else {"type": type(cleanup_error).__name__, "message": str(cleanup_error)},
    }
    _write_new(work_root / "controller-result.json", result)
    return result


__all__ = [
    "ABSOLUTE_WALL_SECONDS",
    "cuda_clean_import_audit",
    "run_integrated_v11_controller",
    "verify_sealed_evidence",
]
