#!/usr/bin/env python3
"""Failure-safe remote lifecycle coordinator for BranchFabric GPU validation.

This tool is intended to be copied to and invoked *on* the selected remote host.
It never opens SSH connections, installs packages, or launches GPU workloads.
Every mutating cleanup operation is scoped by an exact REMOTE_ROOT marker and is
disabled when ``SLOFORGE_COORDINATOR_TEST_MODE=1``.
"""

from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import hashlib
import json
import os
import pathlib
import re
import shlex
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import time
import uuid
from collections.abc import Callable, Sequence
from typing import Any

SCHEMA = "sloforge.branchfabric.remote-coordinator/v1"
ROOT_PREFIX = "sloforge-real-cow-"
ROOT_PATTERN = re.compile(r"^sloforge-real-cow-[A-Za-z0-9]{8}$")
SUBDIRECTORIES = (
    "repo",
    "env",
    "cache",
    "hf",
    "torch",
    "triton",
    "tmp",
    "model",
    "runtime",
    "traces",
    "results",
    "logs",
    "pids",
    "environment",
)
CAPTURE_LIMIT_BYTES = 1_048_576
COMMAND_TIMEOUT_S = 30.0
_UTC = dt.timezone(dt.timedelta(0))  # Python 3.10-compatible bootstrap spelling.


class CoordinatorError(RuntimeError):
    pass


def test_mode() -> bool:
    return os.environ.get("SLOFORGE_COORDINATOR_TEST_MODE") == "1"


def canonical_json(value: object) -> str:
    return json.dumps(value, indent=2, sort_keys=True) + "\n"


def atomic_json(path: pathlib.Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(canonical_json(value), encoding="utf-8")
    os.replace(temporary, path)


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def remote_base() -> pathlib.Path:
    if test_mode():
        raw = os.environ.get("SLOFORGE_COORDINATOR_TEST_BASE")
        if not raw:
            raise CoordinatorError("test mode requires SLOFORGE_COORDINATOR_TEST_BASE")
        base = pathlib.Path(raw).resolve(strict=True)
        if not base.is_dir():
            raise CoordinatorError("test base must be an existing directory")
        return base
    return pathlib.Path("/tmp")


def validate_root(raw: str, *, must_exist: bool = True) -> pathlib.Path:
    candidate = pathlib.Path(raw)
    if not candidate.is_absolute():
        raise CoordinatorError("REMOTE_ROOT must be absolute")
    base = remote_base()
    if candidate.parent.resolve(strict=True) != base:
        raise CoordinatorError(f"REMOTE_ROOT must be an immediate child of {base}")
    if not ROOT_PATTERN.fullmatch(candidate.name):
        raise CoordinatorError("REMOTE_ROOT name does not match the experiment template")
    if not must_exist:
        return candidate
    root = candidate.resolve(strict=True)
    if root != candidate or not root.is_dir() or root.is_symlink():
        raise CoordinatorError("REMOTE_ROOT must be a real directory, not a symlink")
    marker_path = root / ".sloforge-experiment-root.json"
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    if marker.get("schema_version") != SCHEMA or marker.get("remote_root") != str(root):
        raise CoordinatorError("REMOTE_ROOT ownership marker is invalid")
    if marker.get("uid") != os.getuid():
        raise CoordinatorError("REMOTE_ROOT is not owned by the invoking user")
    return root


def bounded_command(
    command: Sequence[str], *, timeout_s: float = COMMAND_TIMEOUT_S
) -> dict[str, Any]:
    started = time.monotonic_ns()
    try:
        result = subprocess.run(
            list(command),
            check=False,
            capture_output=True,
            timeout=timeout_s,
        )
        stdout = result.stdout[:CAPTURE_LIMIT_BYTES]
        stderr = result.stderr[:CAPTURE_LIMIT_BYTES]
        return {
            "command": list(command),
            "returncode": result.returncode,
            "stdout": stdout.decode("utf-8", errors="replace"),
            "stderr": stderr.decode("utf-8", errors="replace"),
            "stdout_truncated": len(result.stdout) > CAPTURE_LIMIT_BYTES,
            "stderr_truncated": len(result.stderr) > CAPTURE_LIMIT_BYTES,
            "duration_ns": time.monotonic_ns() - started,
        }
    except (OSError, subprocess.TimeoutExpired) as error:
        return {
            "command": list(command),
            "returncode": None,
            "stdout": "",
            "stderr": str(error),
            "stdout_truncated": False,
            "stderr_truncated": False,
            "duration_ns": time.monotonic_ns() - started,
        }


def capture_environment(root: pathlib.Path, label: str) -> pathlib.Path:
    if not re.fullmatch(r"[A-Za-z0-9_.-]{1,96}", label):
        raise CoordinatorError("capture label must contain only safe bounded characters")
    captures: dict[str, Any] = {
        "hostname": bounded_command(["hostname"]),
        "user": bounded_command(["id", "-un"]),
        "identity": bounded_command(["id"]),
        "date_utc": bounded_command(["date", "-u", "+%Y-%m-%dT%H:%M:%SZ"]),
        "kernel": bounded_command(["uname", "-a"]),
        "disk": bounded_command(["df", "-Pk", str(root), "/tmp"]),
    }
    if test_mode():
        captures["nvidia_smi"] = {
            "skipped": True,
            "reason": "disabled by SLOFORGE_COORDINATOR_TEST_MODE",
        }
        captures["gpu_inventory"] = captures["nvidia_smi"]
        captures["gpu_processes"] = captures["nvidia_smi"]
    elif shutil.which("nvidia-smi"):
        captures["nvidia_smi"] = bounded_command(["nvidia-smi"])
        captures["gpu_inventory"] = bounded_command(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,name,driver_version,memory.total,memory.used,"
                "utilization.gpu,utilization.memory,pstate,power.draw,clocks.sm,clocks.mem",
                "--format=csv,noheader,nounits",
            ]
        )
        captures["gpu_processes"] = bounded_command(
            [
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
                "--format=csv,noheader,nounits",
            ]
        )
    else:
        unavailable = {"skipped": True, "reason": "nvidia-smi is unavailable"}
        captures["nvidia_smi"] = unavailable
        captures["gpu_inventory"] = unavailable
        captures["gpu_processes"] = unavailable
    nvcc = shutil.which("nvcc")
    captures["cuda_compiler"] = (
        bounded_command([nvcc, "--version"])
        if nvcc and not test_mode()
        else {"skipped": True, "reason": "nvcc unavailable or local test mode"}
    )
    document = {
        "schema_version": SCHEMA,
        "kind": "environment-capture",
        "label": label,
        "remote_root": str(root),
        "captured_at_utc": dt.datetime.now(_UTC).isoformat(),
        "captures": captures,
    }
    output = root / "environment" / f"{label}.json"
    atomic_json(output, document)
    return output


def command_bootstrap(_: argparse.Namespace) -> int:
    base = remote_base()
    if test_mode():
        root = pathlib.Path(tempfile.mkdtemp(prefix=ROOT_PREFIX, dir=base))
        if not ROOT_PATTERN.fullmatch(root.name):
            replacement = base / f"{ROOT_PREFIX}{uuid.uuid4().hex[:8]}"
            root.rename(replacement)
            root = replacement
    else:
        result = subprocess.run(
            ["mktemp", "-d", "/tmp/sloforge-real-cow-XXXXXXXX"],
            check=True,
            capture_output=True,
            text=True,
            timeout=COMMAND_TIMEOUT_S,
        )
        root = pathlib.Path(result.stdout.strip())
    print(f"REMOTE_ROOT={root}", flush=True)
    root.chmod(0o700)
    marker = {
        "schema_version": SCHEMA,
        "kind": "experiment-root-marker",
        "remote_root": str(root),
        "uid": os.getuid(),
        "user": os.environ.get("USER", "unknown"),
        "nonce": uuid.uuid4().hex,
        "created_at_utc": dt.datetime.now(_UTC).isoformat(),
    }
    atomic_json(root / ".sloforge-experiment-root.json", marker)
    for subdirectory in SUBDIRECTORIES:
        (root / subdirectory).mkdir(mode=0o700)
    environment = {
        "SLOFORGE_REMOTE_ROOT": str(root),
        "XDG_CACHE_HOME": str(root / "cache"),
        "HF_HOME": str(root / "hf"),
        "HUGGINGFACE_HUB_CACHE": str(root / "hf" / "hub"),
        "TRANSFORMERS_CACHE": str(root / "hf" / "transformers"),
        "TORCH_HOME": str(root / "torch"),
        "TRITON_CACHE_DIR": str(root / "triton"),
        "TMPDIR": str(root / "tmp"),
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "DO_NOT_TRACK": "1",
        "WANDB_MODE": "disabled",
        "WANDB_DISABLED": "true",
    }
    env_lines = [
        f"export {name}={shlex.quote(value)}" for name, value in sorted(environment.items())
    ]
    (root / "environment" / "remote-env.sh").write_text(
        "\n".join(env_lines) + "\n", encoding="utf-8"
    )
    atomic_json(root / "environment" / "remote-env.json", environment)
    empty_registry = {"schema_version": SCHEMA, "remote_root": str(root), "entries": []}
    for name in ("processes.json", "containers.json", "shared-memory.json"):
        atomic_json(root / "pids" / name, empty_registry)
    manifest = capture_environment(root, "bootstrap")
    print(f"BOOTSTRAP_MANIFEST={manifest}", flush=True)
    return 0


def command_capture(args: argparse.Namespace) -> int:
    root = validate_root(args.remote_root)
    print(capture_environment(root, args.label))
    return 0


def csv_rows(text: str, width: int) -> list[list[str]]:
    rows: list[list[str]] = []
    for line in text.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) == width:
            rows.append(fields)
    return rows


def command_verify_idle_gpu(args: argparse.Namespace) -> int:
    root = validate_root(args.remote_root)
    label = f"pre-gpu-{dt.datetime.now(_UTC).strftime('%Y%m%dT%H%M%S%fZ')}"
    capture_environment(root, label)
    output = root / "environment" / "selected-idle-gpu.json"
    if test_mode() or not shutil.which("nvidia-smi"):
        atomic_json(
            output,
            {
                "schema_version": SCHEMA,
                "status": "unavailable",
                "reason": "GPU inspection disabled in test mode or nvidia-smi unavailable",
                "selected_gpu": None,
            },
        )
        raise CoordinatorError("cannot select an idle GPU without live nvidia-smi inspection")
    gpu_query = bounded_command(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu,utilization.memory",
            "--format=csv,noheader,nounits",
        ]
    )
    process_query = bounded_command(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ]
    )
    if gpu_query["returncode"] != 0 or process_query["returncode"] != 0:
        atomic_json(
            output,
            {
                "schema_version": SCHEMA,
                "status": "unavailable",
                "reason": "nvidia-smi inspection failed",
                "gpu_query": gpu_query,
                "process_query": process_query,
                "selected_gpu": None,
            },
        )
        raise CoordinatorError("nvidia-smi inspection failed; no GPU selected")
    busy_uuids = {row[0] for row in csv_rows(str(process_query["stdout"]), 4)}
    candidates: list[dict[str, Any]] = []
    requested = None if args.gpu_index is None else str(args.gpu_index)
    for index, gpu_uuid, name, total, used, gpu_util, memory_util in csv_rows(
        str(gpu_query["stdout"]), 7
    ):
        try:
            observation = {
                "index": int(index),
                "uuid": gpu_uuid,
                "name": name,
                "memory_total_mib": int(total),
                "memory_used_mib": int(used),
                "gpu_utilization_percent": int(gpu_util),
                "memory_utilization_percent": int(memory_util),
                "compute_process_present": gpu_uuid in busy_uuids,
            }
        except ValueError:
            continue
        if requested is not None and index != requested:
            continue
        if (
            not observation["compute_process_present"]
            and observation["gpu_utilization_percent"] == 0
            and observation["memory_utilization_percent"] == 0
            and observation["memory_used_mib"] <= args.maximum_used_mib
        ):
            candidates.append(observation)
    selected = min(candidates, key=lambda item: int(item["index"])) if candidates else None
    atomic_json(
        output,
        {
            "schema_version": SCHEMA,
            "status": "idle_gpu_selected" if selected else "blocked_no_idle_gpu",
            "captured_at_utc": dt.datetime.now(_UTC).isoformat(),
            "selection_policy": {
                "no_compute_process": True,
                "gpu_utilization_percent": 0,
                "memory_utilization_percent": 0,
                "maximum_used_mib": args.maximum_used_mib,
                "single_observation_no_wait": True,
            },
            "selected_gpu": selected,
            "gpu_query": gpu_query,
            "process_query": process_query,
        },
    )
    if selected is None:
        raise CoordinatorError("no genuinely idle GPU was found; no process was killed")
    print(f"SELECTED_GPU_INDEX={selected['index']}")
    print(f"SELECTED_GPU_UUID={selected['uuid']}")
    return 0


def read_process(pid: int) -> dict[str, Any] | None:
    proc = pathlib.Path("/proc") / str(pid)
    try:
        stat_text = (proc / "stat").read_text(encoding="utf-8")
        closing = stat_text.rfind(")")
        fields = stat_text[closing + 2 :].split()
        status_text = (proc / "status").read_text(encoding="utf-8")
        environ_raw = (proc / "environ").read_bytes()
        command_raw = (proc / "cmdline").read_bytes()
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        return None
    uid_line = next(line for line in status_text.splitlines() if line.startswith("Uid:"))
    uid = int(uid_line.split()[1])
    environment: dict[str, str] = {}
    for item in environ_raw.split(b"\0"):
        if b"=" in item:
            key, value = item.split(b"=", 1)
            environment[key.decode(errors="replace")] = value.decode(errors="replace")
    boot_seconds = 0
    for line in pathlib.Path("/proc/stat").read_text(encoding="utf-8").splitlines():
        if line.startswith("btime "):
            boot_seconds = int(line.split()[1])
            break
    start_ticks = int(fields[19])
    ticks_per_second = os.sysconf("SC_CLK_TCK")
    start_time = dt.datetime.fromtimestamp(
        boot_seconds + start_ticks / ticks_per_second, tz=_UTC
    ).isoformat()
    command = " ".join(
        part.decode("utf-8", errors="replace") for part in command_raw.split(b"\0") if part
    )
    return {
        "pid": pid,
        "ppid": int(fields[1]),
        "process_group": int(fields[2]),
        "uid": uid,
        "command": command,
        "command_sha256": hashlib.sha256(command_raw).hexdigest(),
        "purpose": None,
        "start_time_utc": start_time,
        "start_time_ticks": start_ticks,
        "remote_root_environment": environment.get("SLOFORGE_REMOTE_ROOT"),
    }


def locked_registry_update(
    root: pathlib.Path, name: str, update: Callable[[dict[str, Any]], None]
) -> None:
    lock_path = root / "pids" / f".{name}.lock"
    with lock_path.open("a", encoding="utf-8") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        path = root / "pids" / name
        document = json.loads(path.read_text(encoding="utf-8"))
        update(document)
        atomic_json(path, document)
        fcntl.flock(lock, fcntl.LOCK_UN)


def command_register_process(args: argparse.Namespace) -> int:
    root = validate_root(args.remote_root)
    process = read_process(args.pid)
    if process is None:
        raise CoordinatorError("process does not exist or cannot be inspected")
    if process["uid"] != os.getuid():
        raise CoordinatorError("refusing to register a process owned by another user")
    if process["remote_root_environment"] != str(root):
        raise CoordinatorError("process lacks the exact SLOFORGE_REMOTE_ROOT ownership marker")
    process["purpose"] = args.purpose

    def add(document: dict[str, Any]) -> None:
        entries = document["entries"]
        if any(entry["pid"] == args.pid for entry in entries):
            raise CoordinatorError("process PID is already registered")
        entries.append(process)
        entries.sort(key=lambda entry: (entry["start_time_ticks"], entry["pid"]))

    locked_registry_update(root, "processes.json", add)
    print(canonical_json(process), end="")
    return 0


def process_matches(record: dict[str, Any], root: pathlib.Path) -> bool:
    current = read_process(int(record["pid"]))
    return bool(
        current
        and current["uid"] == os.getuid()
        and current["start_time_ticks"] == record["start_time_ticks"]
        and current["command_sha256"] == record["command_sha256"]
        and current["remote_root_environment"] == str(root)
    )


def destructive_allowed() -> None:
    if test_mode():
        raise CoordinatorError("destructive subcommands are disabled in local test mode")


def command_shutdown_processes(args: argparse.Namespace) -> int:
    destructive_allowed()
    root = validate_root(args.remote_root)
    document = json.loads((root / "pids" / "processes.json").read_text(encoding="utf-8"))
    records = [record for record in document["entries"] if process_matches(record, root)]
    for record in reversed(records):
        os.kill(int(record["pid"]), signal.SIGTERM)
    deadline = time.monotonic() + args.grace_seconds
    while time.monotonic() < deadline and any(process_matches(record, root) for record in records):
        time.sleep(0.1)
    for record in reversed(records):
        if process_matches(record, root):
            os.kill(int(record["pid"]), signal.SIGKILL)
    remaining = [record["pid"] for record in records if process_matches(record, root)]
    if remaining:
        raise CoordinatorError(f"owned processes remain after exact shutdown: {remaining}")
    print(canonical_json({"terminated_owned_pids": [record["pid"] for record in records]}), end="")
    return 0


def root_label(root: pathlib.Path) -> str:
    return hashlib.sha256(str(root).encode()).hexdigest()


def command_register_container(args: argparse.Namespace) -> int:
    root = validate_root(args.remote_root)
    engine = shutil.which(args.engine)
    if engine is None or args.engine not in {"docker", "podman"}:
        raise CoordinatorError("container engine must be an installed docker or podman executable")
    inspection = bounded_command(
        [
            engine,
            "inspect",
            "--format",
            '{{.Id}} {{index .Config.Labels "sloforge.remote_root_sha256"}}',
            args.container_id,
        ]
    )
    if inspection["returncode"] != 0:
        raise CoordinatorError("container inspection failed")
    fields = str(inspection["stdout"]).strip().split()
    if len(fields) != 2 or fields[1] != root_label(root):
        raise CoordinatorError("container lacks the exact experiment ownership label")
    record = {
        "engine": args.engine,
        "engine_path": engine,
        "container_id": fields[0],
        "purpose": args.purpose,
        "registered_at_utc": dt.datetime.now(_UTC).isoformat(),
    }

    def add(document: dict[str, Any]) -> None:
        if any(entry["container_id"] == record["container_id"] for entry in document["entries"]):
            raise CoordinatorError("container is already registered")
        document["entries"].append(record)

    locked_registry_update(root, "containers.json", add)
    print(canonical_json(record), end="")
    return 0


def command_shutdown_containers(args: argparse.Namespace) -> int:
    destructive_allowed()
    root = validate_root(args.remote_root)
    document = json.loads((root / "pids" / "containers.json").read_text(encoding="utf-8"))
    removed: list[str] = []
    for record in document["entries"]:
        inspection = bounded_command(
            [
                record["engine_path"],
                "inspect",
                "--format",
                '{{index .Config.Labels "sloforge.remote_root_sha256"}}',
                record["container_id"],
            ]
        )
        if inspection["returncode"] != 0:
            continue
        if str(inspection["stdout"]).strip() != root_label(root):
            raise CoordinatorError("registered container ownership label changed; refusing removal")
        result = bounded_command([record["engine_path"], "rm", "-f", record["container_id"]])
        if result["returncode"] != 0:
            raise CoordinatorError(f"failed to remove exact container {record['container_id']}")
        removed.append(record["container_id"])
    print(canonical_json({"removed_owned_containers": removed}), end="")
    return 0


def command_register_shm(args: argparse.Namespace) -> int:
    root = validate_root(args.remote_root)
    path = pathlib.Path(args.path).resolve(strict=True)
    if path.parent != pathlib.Path("/dev/shm") or path.stat().st_uid != os.getuid():
        raise CoordinatorError("shared-memory object must be an owned direct child of /dev/shm")
    marker = json.loads((root / ".sloforge-experiment-root.json").read_text(encoding="utf-8"))
    if marker["nonce"] not in path.name and root.name not in path.name:
        raise CoordinatorError("shared-memory name lacks the experiment root or nonce marker")
    record = {
        "path": str(path),
        "registered_at_utc": dt.datetime.now(_UTC).isoformat(),
    }

    def add(document: dict[str, Any]) -> None:
        if any(entry["path"] == str(path) for entry in document["entries"]):
            return
        document["entries"].append(record)

    locked_registry_update(root, "shared-memory.json", add)
    print(canonical_json(record), end="")
    return 0


def command_cleanup_external(args: argparse.Namespace) -> int:
    destructive_allowed()
    root = validate_root(args.remote_root)
    document = json.loads((root / "pids" / "shared-memory.json").read_text(encoding="utf-8"))
    removed: list[str] = []
    for record in document["entries"]:
        path = pathlib.Path(record["path"])
        if not path.exists():
            continue
        resolved = path.resolve(strict=True)
        if resolved.parent != pathlib.Path("/dev/shm") or resolved.stat().st_uid != os.getuid():
            raise CoordinatorError("registered shared-memory ownership changed; refusing removal")
        resolved.unlink()
        removed.append(str(resolved))
    print(canonical_json({"removed_owned_shared_memory": removed}), end="")
    return 0


def artifact_files(root: pathlib.Path, relative_paths: Sequence[str]) -> list[pathlib.Path]:
    files: set[pathlib.Path] = set()
    for raw in relative_paths:
        relative = pathlib.PurePosixPath(raw)
        if relative.is_absolute() or ".." in relative.parts:
            raise CoordinatorError(f"artifact path must remain relative to REMOTE_ROOT: {raw}")
        target = root.joinpath(*relative.parts)
        if not target.exists():
            raise CoordinatorError(f"required artifact path does not exist: {raw}")
        targets = (
            [target] if target.is_file() or target.is_symlink() else [target, *target.rglob("*")]
        )
        for path in targets:
            if (
                path.name.startswith(".artifact-inventory.json.")
                or path == root / "results" / "artifact-inventory.json"
            ):
                continue
            if path.is_symlink():
                resolved = path.resolve(strict=True)
                if root not in resolved.parents:
                    raise CoordinatorError(f"artifact symlink escapes REMOTE_ROOT: {path}")
            elif path.is_file():
                files.add(path)
    return sorted(files, key=lambda path: path.relative_to(root).as_posix())


def command_inventory(args: argparse.Namespace) -> int:
    root = validate_root(args.remote_root)
    requested = args.paths or ["environment", "traces", "results", "logs", "runtime"]
    files = artifact_files(root, requested)
    records = [
        {
            "path": path.relative_to(root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in files
    ]
    document = {
        "schema_version": SCHEMA,
        "kind": "artifact-inventory",
        "remote_root": str(root),
        "created_at_utc": dt.datetime.now(_UTC).isoformat(),
        "requested_paths": requested,
        "file_count": len(records),
        "total_bytes": sum(path.stat().st_size for path in files),
        "files": records,
    }
    output = root / "results" / "artifact-inventory.json"
    atomic_json(output, document)
    print(f"ARTIFACT_INVENTORY={output}")
    print(f"ARTIFACT_INVENTORY_SHA256={sha256_file(output)}")
    return 0


def command_acknowledge_copy(args: argparse.Namespace) -> int:
    root = validate_root(args.remote_root)
    inventory = root / "results" / "artifact-inventory.json"
    observed = sha256_file(inventory)
    if observed != args.inventory_sha256:
        raise CoordinatorError("artifact inventory hash does not match the caller acknowledgement")
    if not re.fullmatch(r"[0-9a-f]{64}", args.local_verification_sha256):
        raise CoordinatorError("local verification receipt hash must be SHA-256")
    atomic_json(
        root / "results" / "copy-verification.json",
        {
            "schema_version": SCHEMA,
            "kind": "copy-verification-acknowledgement",
            "inventory_sha256": observed,
            "local_verification_sha256": args.local_verification_sha256,
            "acknowledged_at_utc": dt.datetime.now(_UTC).isoformat(),
        },
    )
    return 0


def ancestor_pids() -> set[int]:
    ancestors: set[int] = set()
    pid = os.getpid()
    while pid > 1 and pid not in ancestors:
        ancestors.add(pid)
        process = read_process(pid)
        if not process:
            break
        pid = int(process["ppid"])
    return ancestors


def owned_process_audit(root: pathlib.Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    registry = json.loads((root / "pids" / "processes.json").read_text(encoding="utf-8"))
    registered = [record for record in registry["entries"] if process_matches(record, root)]
    registered_pids = {int(record["pid"]) for record in registry["entries"]}
    control = ancestor_pids()
    unregistered: list[dict[str, Any]] = []
    for entry in pathlib.Path("/proc").iterdir() if pathlib.Path("/proc").exists() else ():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        if pid in registered_pids or pid in control:
            continue
        process = read_process(pid)
        if (
            process
            and process["uid"] == os.getuid()
            and process["remote_root_environment"] == str(root)
        ):
            unregistered.append(process)
    return registered, unregistered


def container_audit(root: pathlib.Path) -> list[dict[str, str]]:
    if test_mode():
        return []
    found: dict[tuple[str, str], dict[str, str]] = {}
    label = root_label(root)
    for engine_name in ("docker", "podman"):
        engine = shutil.which(engine_name)
        if not engine:
            continue
        result = bounded_command(
            [engine, "ps", "-aq", "--filter", f"label=sloforge.remote_root_sha256={label}"]
        )
        if result["returncode"] == 0:
            for container_id in str(result["stdout"]).split():
                found[(engine_name, container_id)] = {
                    "engine": engine_name,
                    "container_id": container_id,
                }
    return list(found.values())


def gpu_process_audit(
    root: pathlib.Path, registered_processes: Sequence[dict[str, Any]]
) -> tuple[str, list[dict[str, Any]]]:
    if test_mode():
        return "skipped_local_test_mode", []
    if not shutil.which("nvidia-smi"):
        return "unavailable_nvidia_smi", []
    query = bounded_command(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ]
    )
    if query["returncode"] != 0:
        return "nvidia_smi_query_failed", []
    registered_pids = {int(record["pid"]) for record in registered_processes}
    owned: list[dict[str, Any]] = []
    for gpu_uuid, raw_pid, process_name, used_memory in csv_rows(str(query["stdout"]), 4):
        try:
            pid = int(raw_pid)
        except ValueError:
            continue
        process = read_process(pid)
        if (process and process["remote_root_environment"] == str(root)) or pid in registered_pids:
            owned.append(
                {
                    "gpu_uuid": gpu_uuid,
                    "pid": pid,
                    "process_name": process_name,
                    "used_gpu_memory_mib": used_memory,
                }
            )
    return "captured", owned


def tree_summary(path: pathlib.Path) -> dict[str, int | bool]:
    if not path.exists():
        return {"exists": False, "file_count": 0, "bytes": 0}
    files = [item for item in path.rglob("*") if item.is_file() and not item.is_symlink()]
    return {
        "exists": True,
        "file_count": len(files),
        "bytes": sum(item.stat().st_size for item in files),
    }


def audit_document(root: pathlib.Path) -> dict[str, Any]:
    registered, unregistered = owned_process_audit(root)
    process_registry = json.loads((root / "pids" / "processes.json").read_text(encoding="utf-8"))
    gpu_status, gpu_processes = gpu_process_audit(root, process_registry["entries"])
    marker = json.loads((root / ".sloforge-experiment-root.json").read_text(encoding="utf-8"))
    shm_registry = json.loads((root / "pids" / "shared-memory.json").read_text(encoding="utf-8"))
    registered_shm = [
        entry["path"] for entry in shm_registry["entries"] if pathlib.Path(entry["path"]).exists()
    ]
    discovered_shm: list[str] = []
    shm_root = pathlib.Path("/dev/shm")
    if shm_root.is_dir():
        for path in shm_root.iterdir():
            if root.name in path.name or marker["nonce"] in path.name:
                discovered_shm.append(str(path))
    sockets = [
        str(path.relative_to(root))
        for path in root.rglob("*")
        if not path.is_symlink() and stat.S_ISSOCK(path.stat().st_mode)
    ]
    containers = container_audit(root)
    return {
        "schema_version": SCHEMA,
        "kind": "remote-cleanup-audit",
        "remote_root": str(root),
        "audited_at_utc": dt.datetime.now(_UTC).isoformat(),
        "exact_path_validated": True,
        "live_registered_processes": registered,
        "live_unregistered_owned_processes": unregistered,
        "gpu_process_audit_status": gpu_status,
        "owned_gpu_processes": gpu_processes,
        "owned_containers": containers,
        "registered_shared_memory": sorted(registered_shm),
        "discovered_shared_memory": sorted(discovered_shm),
        "socket_paths_inside_remote_root": sorted(sockets),
        "cache_state": {
            name: tree_summary(root / name) for name in ("cache", "hf", "torch", "triton")
        },
        "model_state": tree_summary(root / "model"),
    }


def command_audit(args: argparse.Namespace) -> int:
    root = validate_root(args.remote_root)
    print(canonical_json(audit_document(root)), end="")
    return 0


def post_removal_audit(
    root: pathlib.Path,
    marker_nonce: str,
    registered_processes: Sequence[dict[str, Any]],
) -> list[dict[str, str]]:
    remaining: list[dict[str, str]] = []
    if root.exists() or root.is_symlink():
        remaining.append({"kind": "remote_root", "path": str(root)})
    for entry in pathlib.Path("/proc").iterdir() if pathlib.Path("/proc").exists() else ():
        if not entry.name.isdigit():
            continue
        process = read_process(int(entry.name))
        if (
            process
            and process["uid"] == os.getuid()
            and process["remote_root_environment"] == str(root)
        ):
            remaining.append({"kind": "process", "path": f"/proc/{entry.name}"})
    shm_root = pathlib.Path("/dev/shm")
    if shm_root.is_dir():
        for path in shm_root.iterdir():
            if root.name in path.name or marker_nonce in path.name:
                remaining.append({"kind": "shared_memory", "path": str(path)})
    for container in container_audit(root):
        remaining.append({"kind": "container", "path": container["container_id"]})
    gpu_status, gpu_processes = gpu_process_audit(root, registered_processes)
    if gpu_status != "captured":
        remaining.append({"kind": "gpu_audit", "path": gpu_status})
    for process in gpu_processes:
        remaining.append({"kind": "gpu_process", "path": str(process["pid"])})
    return remaining


def command_finalize(args: argparse.Namespace) -> int:
    destructive_allowed()
    root = validate_root(args.remote_root)
    verification = root / "results" / "copy-verification.json"
    if not verification.is_file():
        raise CoordinatorError(
            "finalize requires an explicit post-copy verification acknowledgement"
        )
    audit = audit_document(root)
    blocking = {
        "live_registered_processes": audit["live_registered_processes"],
        "live_unregistered_owned_processes": audit["live_unregistered_owned_processes"],
        "owned_gpu_processes": audit["owned_gpu_processes"],
        "owned_containers": audit["owned_containers"],
        "registered_shared_memory": audit["registered_shared_memory"],
        "discovered_shared_memory": audit["discovered_shared_memory"],
    }
    if any(bool(value) for value in blocking.values()):
        raise CoordinatorError(f"pre-removal audit found owned external state: {blocking}")
    if audit["gpu_process_audit_status"] != "captured":
        raise CoordinatorError("pre-removal owned GPU-process audit could not be completed")
    marker = json.loads((root / ".sloforge-experiment-root.json").read_text(encoding="utf-8"))
    process_registry = json.loads((root / "pids" / "processes.json").read_text(encoding="utf-8"))
    shutil.rmtree(root)
    remaining = post_removal_audit(root, marker["nonce"], process_registry["entries"])
    receipt = {
        "schema_version": SCHEMA,
        "kind": "remote-cleanup-receipt",
        "remote_root": str(root),
        "removed_at_utc": dt.datetime.now(_UTC).isoformat(),
        "pre_removal_audit": audit,
        "remote_root_gone": not root.exists(),
        "remaining_experiment_owned_artifacts": remaining,
    }
    print(canonical_json(receipt), end="")
    if remaining:
        raise CoordinatorError("post-removal audit found remaining experiment-owned artifacts")
    return 0


def parser() -> argparse.ArgumentParser:
    main = argparse.ArgumentParser(description=__doc__)
    commands = main.add_subparsers(dest="subcommand", required=True)
    bootstrap = commands.add_parser("bootstrap", help="create and immediately report REMOTE_ROOT")
    bootstrap.set_defaults(handler=command_bootstrap)
    capture = commands.add_parser(
        "capture-environment", help="capture a bounded read-only host snapshot"
    )
    capture.add_argument("remote_root")
    capture.add_argument("label")
    capture.set_defaults(handler=command_capture)
    idle = commands.add_parser("verify-idle-gpu", help="perform one read-only idle-GPU selection")
    idle.add_argument("remote_root")
    idle.add_argument("--gpu-index", type=int)
    idle.add_argument("--maximum-used-mib", type=int, default=16)
    idle.set_defaults(handler=command_verify_idle_gpu)
    register = commands.add_parser("register-process", help="register one exact owned PID")
    register.add_argument("remote_root")
    register.add_argument("pid", type=int)
    register.add_argument("purpose")
    register.set_defaults(handler=command_register_process)
    shutdown = commands.add_parser(
        "shutdown-processes", help="terminate only registered matching PIDs"
    )
    shutdown.add_argument("remote_root")
    shutdown.add_argument("--grace-seconds", type=float, default=10.0)
    shutdown.set_defaults(handler=command_shutdown_processes)
    container = commands.add_parser(
        "register-container", help="register an exactly labelled container"
    )
    container.add_argument("remote_root")
    container.add_argument("engine", choices=("docker", "podman"))
    container.add_argument("container_id")
    container.add_argument("purpose")
    container.set_defaults(handler=command_register_container)
    stop_containers = commands.add_parser(
        "shutdown-containers", help="remove registered owned containers"
    )
    stop_containers.add_argument("remote_root")
    stop_containers.set_defaults(handler=command_shutdown_containers)
    shm = commands.add_parser("register-shm", help="register one exact owned /dev/shm object")
    shm.add_argument("remote_root")
    shm.add_argument("path")
    shm.set_defaults(handler=command_register_shm)
    external = commands.add_parser(
        "cleanup-external", help="unlink only registered owned shm objects"
    )
    external.add_argument("remote_root")
    external.set_defaults(handler=command_cleanup_external)
    inventory = commands.add_parser("inventory-artifacts", help="hash required paths before copy")
    inventory.add_argument("remote_root")
    inventory.add_argument("paths", nargs="*")
    inventory.set_defaults(handler=command_inventory)
    copied = commands.add_parser(
        "acknowledge-copy", help="record caller-verified local copy hashes"
    )
    copied.add_argument("remote_root")
    copied.add_argument("inventory_sha256")
    copied.add_argument("local_verification_sha256")
    copied.set_defaults(handler=command_acknowledge_copy)
    audit = commands.add_parser("audit", help="perform a read-only exact ownership audit")
    audit.add_argument("remote_root")
    audit.set_defaults(handler=command_audit)
    finalize = commands.add_parser(
        "finalize", help="audit, remove only REMOTE_ROOT, and audit again"
    )
    finalize.add_argument("remote_root")
    finalize.set_defaults(handler=command_finalize)
    return main


def main() -> int:
    args = parser().parse_args()
    if getattr(args, "grace_seconds", 1.0) <= 0 or getattr(args, "grace_seconds", 1.0) > 60:
        raise CoordinatorError("grace period must be within (0, 60] seconds")
    if getattr(args, "maximum_used_mib", 0) < 0:
        raise CoordinatorError("maximum used GPU memory must be non-negative")
    return int(args.handler(args))


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (CoordinatorError, json.JSONDecodeError, OSError, subprocess.SubprocessError) as error:
        print(f"branchfabric-remote-coordinator: {error}", file=sys.stderr)
        raise SystemExit(2) from error
