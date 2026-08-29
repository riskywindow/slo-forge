#!/usr/bin/env python3
"""Sole paid-resource coordinator for Experiment 004 integrated v11.

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
import signal
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from itertools import pairwise
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
    V11_TARGETED_IDENTITY_RESERVATION_WALL_SECONDS,
    Experiment004V11IntegratedConfig,
    Experiment004V11TargetedSourceIdentityConfig,
    canonical_json_bytes,
)

_ROOT = Path(__file__).resolve().parents[1]
_EXPERIMENT_ROOT = _ROOT / "artifacts/branchfabric/gpu-validation/experiment-004"
_LEDGER = _EXPERIMENT_ROOT / "gpu-hours.json"
_LOCK = _EXPERIMENT_ROOT / "gpu-hours.lock"
_APP = _ROOT / "experiments/branchfabric/modal_gpu_reclamation_integrated_v11.py"
_LOCAL_RESULT_PARENT = _EXPERIMENT_ROOT / "v11-final/integrated/raw"
_FAILURE_ROOT = _EXPERIMENT_ROOT / "v11-final/integrated/failures"
_PROVIDER_INPUTS_PARENT = _EXPERIMENT_ROOT / "v11-final/integrated"
_PREBUILD_ROOT = _EXPERIMENT_ROOT / "v11-image-prebuild"

ATTEMPT_ID = "exp004-v11-integrated-s41-k"
TARGETED_ATTEMPT_ID = "exp004-v11-targeted-identity-s41-b"
_RETAINED_CAPACITY_PREIMPORT_BINDINGS = {
    "integrated_controller": (
        "experiments/branchfabric/gpu_reclamation_integrated_controller_v11.py",
        "c7c07ea0444874a100aa6d6da273058edd4f8cee03548e2c08c62bca5ecae2fc",
    ),
    "integrated_worker": (
        "experiments/branchfabric/gpu_reclamation_integrated_worker_v11.py",
        "a9312a017116ffc86ac311b578da67f2afb97198815f30369c8ee3e0159c2002",
    ),
    "frozen_sanity_assessor": (
        "experiments/branchfabric/gpu_reclamation_controller.py",
        "c91ccd8b3aa4b4c5a9034a5137ccc33257e3471528a96dcc2ca7920c8b7eefc9",
    ),
    "capacity_controller": (
        "experiments/branchfabric/gpu_capacity_calibration_controller.py",
        "31fb70855e49d3f59493592de84120fc82f6a950e5b5edd0e775d5ba47978aa2",
    ),
    "capacity_worker": (
        "experiments/branchfabric/gpu_capacity_calibration_worker.py",
        "2c41499a847879b813ce01470a7e32625d772fc07233753404296125c4afc225",
    ),
    "capacity_model": (
        "python/sloforge/helix/characterization/gpu_capacity_calibration.py",
        "7ffb993004fbc9b595cc3c5fc2e503ffd9dd2370befdeb8714d017badc4e0b3d",
    ),
    "integrated_controller_tests": (
        "tests/python/test_gpu_reclamation_integrated_controller_v11.py",
        "00f5f2bee68735b5b84da52c731aa181f3a2e7466fbde67d1450d1167156496f",
    ),
    "same_allocation_reproduction_tests": (
        "tests/python/test_gpu_reclamation_integrated_sanity_reproduction_v11.py",
        "18d086cd5f204aeca66a0c2894a9ce5dc5f20e61715579058777a6eadc648abe",
    ),
    "agent12_runtime_capacity_review": (
        "artifacts/branchfabric/gpu-validation/experiment-004/v11-final/reviews/"
        "agent12-attempt-j-runtime-capacity-review.json",
        "f5bef536dfe3fc3174158eb592c6c66da6635b19ca4f80ed49c359f111befa05",
    ),
    "agent13_whole_file_runtime_review": (
        "artifacts/branchfabric/gpu-validation/experiment-004/v11-final/reviews/"
        "agent13-attempt-k-whole-file-runtime-review.json",
        "34b1c8569ce2914c6319542507bf403afad96aeec1a103487c4a44cd8e117674",
    ),
}
APP_NAME = "sloforge-branchfabric-gpu-reclamation-004-v11-integrated"
RESULTS_VOLUME = "sloforge-branchfabric-results"
MODEL_VOLUME = "sloforge-model-cache"
REMOTE_PREFIX = "experiment-004/v11/integrated/modal"
GPU_COUNT = 2
FUNCTION_WALL_SECONDS = V11_INTEGRATED_RESERVATION_WALL_SECONDS
TARGETED_FUNCTION_WALL_SECONDS = V11_TARGETED_IDENTITY_RESERVATION_WALL_SECONDS
FUNCTION_STARTUP_SECONDS = 180.0
IMAGE_PREBUILD_ATTEMPTS = 3
IMAGE_PREBUILD_TIMEOUT_SECONDS = 900.0
PROVIDER_ZERO_POLLS = 10
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

V11Config = Experiment004V11IntegratedConfig | Experiment004V11TargetedSourceIdentityConfig


class _VerifiedRemoteFailure(RuntimeError):
    """Internal terminal carrying a completed conservative ledger charge."""

    def __init__(self, charge: Mapping[str, Any]) -> None:
        super().__init__("verified remote failure")
        self.charge = dict(charge)


def _is_targeted(config: V11Config) -> bool:
    return config.execution_mode == "targeted-source-identity-v11"


def _function_wall_seconds(config: V11Config) -> float:
    return TARGETED_FUNCTION_WALL_SECONDS if _is_targeted(config) else FUNCTION_WALL_SECONDS


def _maximum_estimated_cost_usd(config: V11Config) -> float:
    return _function_wall_seconds(config) * (
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
_PRE_WORKER_FAILURE_STAGES = frozenset(
    {
        "SEALED_EVIDENCE",
        "LIVE_CONTRACT",
        "CPU_CONTROL_IMPORTS",
        "PRE_WORKER_ARTIFACT_INITIALIZATION",
        "PRE_INVENTORY_CUDA_CLEAN_AUDIT",
        "GPU_INVENTORY_BEFORE_WORKERS",
        "ZERO_COMPUTE_BEFORE_WORKERS",
    }
)
_PRE_WORKER_CONTROLLER_FIELDS = frozenset(
    {
        "schema_version",
        "status",
        "attempt_id",
        "controller_pid",
        "started_ns",
        "ended_ns",
        "operation_deadline_ns",
        "cleanup_deadline_ns",
        "absolute_wall_seconds",
        "execution_mode",
        "terminal_phase",
        "cleanup_scope",
        "failure_stage",
        "sealed_evidence",
        "inventory_before",
        "inventory_after",
        "stable_physical_gpu_identity",
        "worker_pids",
        "worker_process_groups",
        "worker_session_ids",
        "worker_returncodes",
        "engine_start_evidence",
        "readiness_evidence",
        "readiness_deadline_ns",
        "sanity_guard_pair",
        "worker_results",
        "cleanup_actions",
        "in_function_cleanup",
        "compute_processes_after",
        "cuda_clean_import_audits",
        "controller_error",
        "cleanup_error",
    }
)
_IN_FUNCTION_CLEANUP_FIELDS = frozenset(
    {
        "schema_version",
        "status",
        "pass",
        "required_lifecycle",
        "lifecycle",
        "recorded_at_monotonic_ns",
        "recorded_at_utc",
        "parent_pid",
        "parent_pgid",
        "parent_sid",
        "child_subreaper",
        "initial_threads",
        "final_threads",
        "leaked_threads",
        "initial_ipc_resources",
        "final_ipc_resources",
        "leaked_ipc_resources",
        "owned_children",
        "termination_actions",
        "forced_kills",
        "forced_kill_required",
        "surviving_children",
        "surviving_process_groups",
        "profiler_processes_after",
        "serving_workers_after",
        "rollout_workers_after",
        "resource_tracker_processes_after",
        "zombie_processes_after",
        "compute_processes_after",
        "cleanup_errors",
        "parent_reaped_all_owned_children",
        "owned_ipc_resources_released",
        "pipes_closed",
        "cuda_released",
    }
)
_PROCESS_LIFECYCLE_PHASES = (
    "RUNNING",
    "QUIESCE",
    "ENGINE_STOP",
    "WORKER_STOP",
    "CHILD_REAP",
    "PGID_EMPTY",
    "CUDA_RELEASED",
    "FUNCTION_RETURN",
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


def _require_retained_capacity_preimport_closure(
    config: V11Config,
    repository_root: Path,
) -> dict[str, dict[str, str]]:
    """Verify K's independently reviewed runtime before importing its controller."""

    if config.attempt_id != ATTEMPT_ID:
        return {}
    resolved_root = repository_root.resolve(strict=True)
    verified: dict[str, dict[str, str]] = {}
    for label, (reference, expected_sha256) in _RETAINED_CAPACITY_PREIMPORT_BINDINGS.items():
        if (
            not isinstance(reference, str)
            or not reference
            or not isinstance(expected_sha256, str)
            or re.fullmatch(r"[0-9a-f]{64}", expected_sha256) is None
        ):
            raise ValueError(f"Attempt-K pre-import binding is malformed: {label}")
        path = repository_root / reference
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Attempt-K pre-import artifact is invalid: {label}")
        resolved = path.resolve(strict=True)
        if resolved_root not in resolved.parents or _sha256_file(resolved) != expected_sha256:
            raise ValueError(f"Attempt-K pre-import artifact hash mismatch: {label}")
        verified[label] = {"artifact": reference, "sha256": expected_sha256}
    return verified


def _default_seal_verifier(config: V11Config, repository_root: Path) -> dict[str, Any]:
    _require_retained_capacity_preimport_closure(config, repository_root)
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
    seal_verifier: Callable[[V11Config, Path], dict[str, Any]] = _default_seal_verifier,
) -> tuple[V11Config, dict[str, Any]]:
    if config_path.is_symlink():
        raise ValueError("integrated v11 config must not be a symlink")
    resolved_root = repository_root.resolve(strict=True)
    resolved = config_path.resolve(strict=True)
    if resolved != resolved_root and resolved_root not in resolved.parents:
        raise ValueError("integrated v11 config must be a regular file inside the repository")
    raw = json.loads(resolved.read_text())
    if (
        isinstance(raw, dict)
        and raw.get("schema_version")
        == "sloforge.branchfabric.experiment-004-v11-targeted-source-identity-config/v1"
    ):
        config: V11Config = Experiment004V11TargetedSourceIdentityConfig.model_validate(
            raw, strict=True
        )
    else:
        config = Experiment004V11IntegratedConfig.model_validate(raw, strict=True)
    expected_attempt = TARGETED_ATTEMPT_ID if _is_targeted(config) else ATTEMPT_ID
    if config.attempt_id != expected_attempt:
        raise ValueError("sole coordinator admits only the next exact v11 attempt")
    wall_seconds = _function_wall_seconds(config)
    if config.gpu_count != GPU_COUNT or not math.isclose(
        config.maximum_wall_seconds,
        wall_seconds,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise ValueError(f"v11 config must bind exactly 2xA100 for {int(wall_seconds)} seconds")
    current_ledger_sha256 = _sha256_file(ledger_path.resolve(strict=True))
    if current_ledger_sha256 != config.ledger_sha256_before_reservation:
        raise RuntimeError("sealed config does not bind the current pre-reservation ledger")
    verified = seal_verifier(config, resolved_root)
    if not isinstance(verified, dict) or verified.get("status") != "PASS":
        raise RuntimeError("integrated v11 content-addressed evidence gate is not PASS")
    return config, verified


def _authorized_budget_usd(config: V11Config, root: Path) -> float:
    path = (root / config.budget_authorization).resolve(strict=True)
    if _sha256_file(path) != config.budget_authorization_sha256:
        raise RuntimeError("integrated v11 budget authorization hash drifted")
    payload = json.loads(path.read_text())
    value = payload.get("authorized_gpu_budget_usd") if isinstance(payload, dict) else None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError("integrated v11 budget authorization omits a numeric USD limit")
    authorized = float(value)
    if not math.isfinite(authorized) or authorized < _maximum_estimated_cost_usd(config):
        raise RuntimeError("integrated v11 budget authorization is below the bounded run cost")
    return authorized


def _require_launch_budget(config: V11Config, root: Path) -> float:
    raw = os.getenv("SLOFORGE_GPU_BUDGET_USD")
    try:
        value = float(raw) if raw is not None else math.nan
    except ValueError as error:
        raise RuntimeError("SLOFORGE_GPU_BUDGET_USD must be finite and positive") from error
    authorized = _authorized_budget_usd(config, root)
    if (
        not math.isfinite(value)
        or value < _maximum_estimated_cost_usd(config)
        or value > authorized
    ):
        raise RuntimeError(
            "SLOFORGE_GPU_BUDGET_USD must cover the bounded run without exceeding authorization"
        )
    return value


def _reserve_locked(
    config: V11Config,
    *,
    ledger_path: Path,
    reservation_id: str,
) -> tuple[Experiment004GpuHourLedger, dict[str, Any], str]:
    original_bytes = ledger_path.resolve(strict=True).read_bytes()
    if _sha256_bytes(original_bytes) != config.ledger_sha256_before_reservation:
        raise RuntimeError("ledger changed between sealed-config validation and reservation")
    ledger = Experiment004GpuHourLedger.model_validate_json(original_bytes, strict=True)
    config_sha256 = _sha256_bytes(canonical_json_bytes(config))
    wall_seconds = _function_wall_seconds(config)
    updated, preflight = reserve_gpu_invocation(
        ledger,
        reservation_id=reservation_id,
        invocation_id=config.attempt_id,
        maximum_wall_seconds=wall_seconds,
        config_sha256=config_sha256,
        gpu_count=2,
    )
    if not math.isclose(
        preflight.proposed_maximum_gpu_seconds,
        GPU_COUNT * wall_seconds,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise RuntimeError("v11 reservation differs from its exact GPU-second bound")
    _atomic_write_bytes(ledger_path, _ledger_bytes(updated))
    reservation = updated.reservations[0]
    commitment = _sha256_bytes(canonical_json_bytes(reservation))
    return updated, preflight.model_dump(mode="json"), commitment


def _invoke_modal(
    config_path: Path,
    *,
    wall_seconds: float = FUNCTION_WALL_SECONDS,
    reservation_id: str,
    budget_usd: float,
    runner: Callable[..., subprocess.CompletedProcess[str]] | None = None,
) -> tuple[subprocess.CompletedProcess[str], float]:
    runner = _run_bounded_process_group if runner is None else runner
    launch_token = secrets.token_hex(32)
    environment = dict(os.environ)
    environment.update(
        {
            "SLOFORGE_GPU_BUDGET_USD": format(budget_usd, ".17g"),
            "SLOFORGE_EXP004_RESERVATION_ID": reservation_id,
            "SLOFORGE_MODAL_PREFLIGHT_TOKEN": launch_token,
        }
    )
    command = [
        "/usr/bin/caffeinate",
        "-dimsu",
        "modal",
        "run",
        str(_APP),
        "--config-path",
        str(config_path),
    ]
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
        timeout=(FUNCTION_STARTUP_SECONDS + wall_seconds + COORDINATOR_PROCESS_GRACE_SECONDS),
        env=environment,
    )
    return completed, time.monotonic() - started


def _require_caffeinate() -> None:
    path = Path("/usr/bin/caffeinate")
    if not path.is_file() or not os.access(path, os.X_OK):
        raise RuntimeError("integrated v11 requires executable /usr/bin/caffeinate")


def _process_group_exists(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    return True


def _terminate_process_group(pgid: int) -> None:
    if not _process_group_exists(pgid):
        return
    os.killpg(pgid, signal.SIGTERM)
    for _ in range(20):
        if not _process_group_exists(pgid):
            return
        time.sleep(0.05)
    if _process_group_exists(pgid):
        os.killpg(pgid, signal.SIGKILL)
    for _ in range(20):
        if not _process_group_exists(pgid):
            return
        time.sleep(0.05)
    raise RuntimeError(f"subprocess process group {pgid} survived SIGKILL")


def _run_bounded_process_group(
    command: list[str],
    *,
    check: bool,
    capture_output: bool,
    text: bool,
    timeout: float,
    env: Mapping[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    if check or not capture_output or not text:
        raise ValueError("bounded process-group runner requires check=False and captured text")
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=dict(env) if env is not None else None,
        start_new_session=True,
    )
    timed_out = False
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        if _process_group_exists(process.pid):
            os.killpg(process.pid, signal.SIGTERM)
        try:
            stdout, stderr = process.communicate(timeout=1.0)
        except subprocess.TimeoutExpired:
            if _process_group_exists(process.pid):
                os.killpg(process.pid, signal.SIGKILL)
            stdout, stderr = process.communicate()
        if _process_group_exists(process.pid):
            _terminate_process_group(process.pid)
    orphaned_group = _process_group_exists(process.pid)
    if orphaned_group:
        _terminate_process_group(process.pid)
    return subprocess.CompletedProcess(
        command,
        124 if timed_out else (125 if orphaned_group else process.returncode),
        stdout=stdout,
        stderr=(stderr or "")
        + ("\nSLOFORGE_PROCESS_GROUP_TIMEOUT_CLEANUP" if timed_out else "")
        + ("\nSLOFORGE_PROCESS_GROUP_ORPHAN_CLEANUP" if orphaned_group else ""),
    )


def _stop_modal_app_and_require_provider_zero(
    *,
    runner: Callable[..., subprocess.CompletedProcess[str]] | None = None,
) -> dict[str, Any]:
    runner = _run_bounded_process_group if runner is None else runner
    stop = runner(
        ["modal", "app", "stop", APP_NAME, "-y"],
        check=False,
        capture_output=True,
        text=True,
        timeout=60.0,
    )
    observations: list[dict[str, Any]] = []
    for _ in range(PROVIDER_ZERO_POLLS):
        apps = runner(
            ["modal", "app", "list", "--json"],
            check=False,
            capture_output=True,
            text=True,
            timeout=60.0,
        )
        containers = runner(
            ["modal", "container", "list", "--json"],
            check=False,
            capture_output=True,
            text=True,
            timeout=60.0,
        )
        try:
            app_payload = json.loads(apps.stdout) if apps.returncode == 0 else None
            container_payload = (
                json.loads(containers.stdout) if containers.returncode == 0 else None
            )
        except json.JSONDecodeError:
            app_payload = None
            container_payload = None
        stopped_only = isinstance(app_payload, list) and all(
            isinstance(row, dict)
            and isinstance(row.get("state"), str)
            and row["state"].lower() in {"stopped", "disabled"}
            and (
                (isinstance(row.get("tasks"), str) and row["tasks"] == "0")
                or (
                    isinstance(row.get("tasks"), int)
                    and not isinstance(row.get("tasks"), bool)
                    and row["tasks"] == 0
                )
            )
            for row in app_payload
        )
        containers_zero = isinstance(container_payload, list) and not container_payload
        observations.append(
            {
                "app_list_returncode": apps.returncode,
                "app_list_stdout": apps.stdout[-16_384:],
                "app_list_stderr": apps.stderr[-16_384:],
                "parsed_app_count": len(app_payload) if isinstance(app_payload, list) else None,
                "stopped_or_disabled_apps_only": stopped_only,
                "container_list_returncode": containers.returncode,
                "container_list_stdout": containers.stdout[-16_384:],
                "container_list_stderr": containers.stderr[-16_384:],
                "parsed_container_count": (
                    len(container_payload) if isinstance(container_payload, list) else None
                ),
            }
        )
        if stopped_only and containers_zero:
            return {
                "stop_returncode": stop.returncode,
                "stop_stdout": stop.stdout[-16_384:],
                "stop_stderr": stop.stderr[-16_384:],
                "provider_zero": True,
                "app_list_observations": observations,
            }
    raise RuntimeError("Modal app state did not reach provider zero after image prebuild")


def _prebuild_modal_image(
    *,
    budget_usd: float,
    runner: Callable[..., subprocess.CompletedProcess[str]] | None = None,
) -> dict[str, Any]:
    """Hydrate the exact Modal image before creating any A100 reservation."""

    _require_caffeinate()
    runner = _run_bounded_process_group if runner is None else runner
    attempts: list[dict[str, Any]] = []
    for sequence in range(IMAGE_PREBUILD_ATTEMPTS):
        token = secrets.token_hex(32)
        environment = dict(os.environ)
        environment.update(
            {
                "SLOFORGE_GPU_BUDGET_USD": format(budget_usd, ".17g"),
                "SLOFORGE_MODAL_PREFLIGHT_TOKEN": token,
            }
        )
        command = [
            "/usr/bin/caffeinate",
            "-dimsu",
            "modal",
            "deploy",
            "--name",
            APP_NAME,
            str(_APP),
        ]
        started = time.monotonic()
        timed_out = False
        try:
            completed = runner(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=IMAGE_PREBUILD_TIMEOUT_SECONDS,
                env=environment,
            )
            timed_out = completed.returncode == 124 and (
                "SLOFORGE_PROCESS_GROUP_TIMEOUT_CLEANUP" in completed.stderr
            )
        except subprocess.TimeoutExpired as error:
            timed_out = True
            completed = subprocess.CompletedProcess(
                command,
                124,
                stdout=(error.stdout or "") if isinstance(error.stdout, str) else "",
                stderr=(error.stderr or "") if isinstance(error.stderr, str) else "",
            )
        cleanup = _stop_modal_app_and_require_provider_zero(runner=runner)
        record = {
            "sequence": sequence,
            "returncode": completed.returncode,
            "timed_out": timed_out,
            "elapsed_seconds": time.monotonic() - started,
            "stdout": completed.stdout[-65_536:],
            "stderr": completed.stderr[-65_536:],
            "image_ids": sorted(
                set(re.findall(r"\bim-[A-Za-z0-9]+\b", completed.stdout + completed.stderr))
            ),
            "provider_cleanup": cleanup,
        }
        attempts.append(record)
        if completed.returncode == 0:
            return {
                "schema_version": "sloforge.branchfabric.exp004-v11-image-prebuild/v1",
                "status": "PASS",
                "gpu_reservation_created": False,
                "gpu_function_invoked": False,
                "attempt_count": len(attempts),
                "attempts": attempts,
                "provider_zero_before_gpu_reservation": True,
            }
    raise RuntimeError(
        f"Modal image prebuild failed after {IMAGE_PREBUILD_ATTEMPTS} provider-clean attempts"
    )


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


def _deterministic_seal_projection(seal: Mapping[str, Any]) -> dict[str, Any]:
    """Exclude only the verifier observation clock from a repeated seal check."""

    return {key: value for key, value in seal.items() if key != "verified_at_monotonic_ns"}


def _bounded_number(value: Any, *, upper: float, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or not 0.0 < float(value) <= upper
    ):
        raise ValueError(f"{label} is outside its paid bound")
    return float(value)


def _require_exact_pre_worker_failure(
    remote: Mapping[str, Any],
    *,
    config: V11Config,
) -> None:
    """Validate the exact inventory-free failure shape produced before workers exist."""

    controller = remote.get("controller")
    cleanup = remote.get("in_function_cleanup")
    run_error = remote.get("run_error")
    expected_controller_schema = (
        "sloforge.branchfabric.experiment-004-v11-targeted-controller/v1"
        if _is_targeted(config)
        else "sloforge.branchfabric.experiment-004-v11-integrated-controller/v1"
    )
    expected_terminal_phase = (
        "SOURCE_ALLOCATION_IDENTITY_GATE" if _is_targeted(config) else "INTEGRATED_TRANSACTION"
    )
    expected_empty_controller_fields = (
        "inventory_before",
        "inventory_after",
        "worker_pids",
        "worker_process_groups",
        "worker_session_ids",
        "worker_returncodes",
        "engine_start_evidence",
        "readiness_evidence",
        "worker_results",
        "cleanup_actions",
        "compute_processes_after",
    )
    if (
        not isinstance(controller, dict)
        or set(controller) != _PRE_WORKER_CONTROLLER_FIELDS
        or controller["schema_version"] != expected_controller_schema
        or controller["status"] != "failed"
        or controller["attempt_id"] != config.attempt_id
        or controller["execution_mode"] != config.execution_mode
        or controller["terminal_phase"] != expected_terminal_phase
        or controller["cleanup_scope"] != "PRE_WORKER_PREFLIGHT"
        or controller["failure_stage"] not in _PRE_WORKER_FAILURE_STAGES
        or controller["sealed_evidence"] is not None
        or controller["stable_physical_gpu_identity"] is not False
        or controller["readiness_deadline_ns"] is not None
        or controller["sanity_guard_pair"] is not None
        or controller["cleanup_error"] is not None
        or any(controller[field] not in ([], {}) for field in expected_empty_controller_fields)
        or not isinstance(controller["controller_error"], dict)
        or set(controller["controller_error"]) != {"type", "message"}
        or any(
            not isinstance(controller["controller_error"][field], str)
            or not controller["controller_error"][field]
            for field in ("type", "message")
        )
    ):
        raise ValueError("integrated PRE_WORKER failure controller differs from its exact schema")
    controller_times = tuple(
        controller[field]
        for field in (
            "controller_pid",
            "started_ns",
            "ended_ns",
            "operation_deadline_ns",
            "cleanup_deadline_ns",
        )
    )
    if (
        any(type(value) is not int or value <= 0 for value in controller_times)
        or controller["started_ns"] >= controller["ended_ns"]
        or controller["ended_ns"] > controller["cleanup_deadline_ns"]
        or controller["started_ns"] >= controller["operation_deadline_ns"]
        or controller["operation_deadline_ns"] >= controller["cleanup_deadline_ns"]
        or controller["absolute_wall_seconds"] != _function_wall_seconds(config)
    ):
        raise ValueError("integrated PRE_WORKER failure timing is invalid")

    if (
        not isinstance(run_error, dict)
        or set(run_error) != {"type", "message", "traceback"}
        or run_error["type"] != "RuntimeError"
        or run_error["message"] != "integrated v11 controller failed closed"
        or not isinstance(run_error["traceback"], str)
        or "in run_integrated" not in run_error["traceback"]
        or 'raise RuntimeError("integrated v11 controller failed closed")'
        not in run_error["traceback"]
        or "RuntimeError: integrated v11 controller failed closed" not in run_error["traceback"]
    ):
        raise ValueError("integrated PRE_WORKER launcher failure is invalid")

    audits = controller["cuda_clean_import_audits"]
    if (
        not isinstance(audits, list)
        or len(audits) != 1
        or not isinstance(audits[0], dict)
        or set(audits[0])
        != {
            "schema_version",
            "stage",
            "pid",
            "observed_at_monotonic_ns",
            "observed_at_utc",
            "loaded_forbidden_modules",
            "cuda_clean",
        }
        or audits[0]["schema_version"] != "sloforge.branchfabric.cuda-clean-import-audit/v1"
        or audits[0]["stage"] != "v11-integrated-pre-worker-failure-cleanup"
        or audits[0]["pid"] != controller["controller_pid"]
        or type(audits[0]["pid"]) is not int
        or type(audits[0]["observed_at_monotonic_ns"]) is not int
        or not controller["started_ns"]
        <= audits[0]["observed_at_monotonic_ns"]
        <= controller["cleanup_deadline_ns"]
        or not isinstance(audits[0]["observed_at_utc"], str)
        or not audits[0]["observed_at_utc"]
        or audits[0]["loaded_forbidden_modules"] != []
        or audits[0]["cuda_clean"] is not True
    ):
        raise ValueError("integrated PRE_WORKER CUDA-clean evidence is invalid")

    if (
        not isinstance(cleanup, dict)
        or set(cleanup) != _IN_FUNCTION_CLEANUP_FIELDS
        or controller["in_function_cleanup"] != cleanup
        or cleanup["schema_version"] != "sloforge.branchfabric.in-function-cleanup/v1"
        or cleanup["status"] != "PASS"
        or cleanup["pass"] is not True
        or cleanup["required_lifecycle"] != list(_PROCESS_LIFECYCLE_PHASES)
    ):
        raise ValueError("integrated PRE_WORKER cleanup differs from its exact schema")
    lifecycle = cleanup["lifecycle"]
    if (
        not isinstance(lifecycle, list)
        or len(lifecycle) != len(_PROCESS_LIFECYCLE_PHASES)
        or any(
            not isinstance(row, dict)
            or set(row) != {"phase", "observed_at_monotonic_ns", "observed_at_utc"}
            or row["phase"] != phase
            or type(row["observed_at_monotonic_ns"]) is not int
            or row["observed_at_monotonic_ns"] <= 0
            or not isinstance(row["observed_at_utc"], str)
            or not row["observed_at_utc"]
            for row, phase in zip(lifecycle, _PROCESS_LIFECYCLE_PHASES, strict=True)
        )
        or any(
            current["observed_at_monotonic_ns"] >= following["observed_at_monotonic_ns"]
            for current, following in pairwise(lifecycle)
        )
        or lifecycle[0]["observed_at_monotonic_ns"] < controller["started_ns"]
        or lifecycle[-1]["observed_at_monotonic_ns"] > controller["cleanup_deadline_ns"]
    ):
        raise ValueError("integrated PRE_WORKER cleanup lifecycle is invalid")
    empty_cleanup_fields = (
        "leaked_threads",
        "initial_ipc_resources",
        "final_ipc_resources",
        "leaked_ipc_resources",
        "owned_children",
        "termination_actions",
        "forced_kills",
        "surviving_children",
        "surviving_process_groups",
        "profiler_processes_after",
        "serving_workers_after",
        "rollout_workers_after",
        "resource_tracker_processes_after",
        "zombie_processes_after",
        "compute_processes_after",
        "cleanup_errors",
    )
    subreaper = cleanup["child_subreaper"]
    threads = cleanup["initial_threads"]
    valid_threads = bool(
        isinstance(threads, list)
        and threads
        and all(
            isinstance(row, dict)
            and set(row) == {"alive", "daemon", "ident", "name", "native_id"}
            and type(row["alive"]) is bool
            and type(row["daemon"]) is bool
            and type(row["ident"]) is int
            and row["ident"] > 0
            and isinstance(row["name"], str)
            and bool(row["name"])
            and type(row["native_id"]) is int
            and row["native_id"] > 0
            for row in threads
        )
    )
    if (
        any(cleanup[field] != [] for field in empty_cleanup_fields)
        or cleanup["forced_kill_required"] is not False
        or any(
            cleanup[field] is not True
            for field in (
                "parent_reaped_all_owned_children",
                "owned_ipc_resources_released",
                "pipes_closed",
                "cuda_released",
            )
        )
        or type(cleanup["parent_pid"]) is not int
        or cleanup["parent_pid"] != controller["controller_pid"]
        or type(cleanup["parent_pgid"]) is not int
        or cleanup["parent_pgid"] <= 0
        or type(cleanup["parent_sid"]) is not int
        or cleanup["parent_sid"] <= 0
        or type(cleanup["recorded_at_monotonic_ns"]) is not int
        or not lifecycle[-1]["observed_at_monotonic_ns"]
        <= cleanup["recorded_at_monotonic_ns"]
        <= controller["cleanup_deadline_ns"]
        or not isinstance(cleanup["recorded_at_utc"], str)
        or not cleanup["recorded_at_utc"]
        or not valid_threads
        or cleanup["initial_threads"] != cleanup["final_threads"]
        or not isinstance(subreaper, dict)
        or set(subreaper)
        != {
            "supported",
            "initially_enabled",
            "enabled_for_experiment",
            "restored_before_return",
            "error",
        }
        or type(subreaper["supported"]) is not bool
        or type(subreaper["initially_enabled"]) is not bool
        or type(subreaper["enabled_for_experiment"]) is not bool
        or type(subreaper["restored_before_return"]) is not bool
        or subreaper["error"] is not None
        or subreaper["restored_before_return"] is not True
        or (subreaper["supported"] is True and subreaper["enabled_for_experiment"] is not True)
    ):
        raise ValueError("integrated PRE_WORKER cleanup is not PASS-complete")

    hardware = remote.get("hardware_comparability")
    if (
        not isinstance(hardware, dict)
        or set(hardware)
        != {
            "modal_request",
            "modal_interconnect_selection_available",
            "observed_gpu_names",
            "observed_gpu_uuids",
            "frozen_v10_gpu_name",
            "direct_v10_timing_comparable",
            "timing_comparability_reason",
            "byte_and_amplification_comparable",
        }
        or hardware["modal_request"] != "A100-80GB:2"
        or hardware["modal_interconnect_selection_available"] is not False
        or hardware["observed_gpu_names"] != []
        or hardware["observed_gpu_uuids"] != []
        or hardware["frozen_v10_gpu_name"] != "NVIDIA A100-SXM4-80GB"
        or hardware["direct_v10_timing_comparable"] is not False
        or hardware["timing_comparability_reason"]
        != "controller did not produce validated inventory"
        or hardware["byte_and_amplification_comparable"] is not False
    ):
        raise ValueError("integrated PRE_WORKER hardware evidence is invalid")


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
    config: V11Config,
    reservation_id: str,
    reservation_commitment_sha256: str,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    tuple[dict[str, Any], dict[str, Any]] | None,
]:
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
    wall_seconds = _function_wall_seconds(config)
    allocation = _bounded_number(
        remote["gpu_allocation_seconds"], upper=wall_seconds, label="allocation time"
    )
    gpu_seconds = _bounded_number(
        remote["gpu_seconds"], upper=GPU_COUNT * wall_seconds, label="GPU time"
    )
    gpu_hours = _bounded_number(
        remote["gpu_hours"],
        upper=GPU_COUNT * wall_seconds / 3600.0,
        label="GPU hours",
    )
    interval = _bounded_number(
        remote["controller_and_analysis_interval_seconds"],
        upper=wall_seconds,
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
        != round((wall_seconds - POST_CONTROLLER_RESERVE_SECONDS) * 1e9)
        or function_deadline - entry != round(wall_seconds * 1e9)
        or deadlines["post_controller_reserve_seconds"] != POST_CONTROLLER_RESERVE_SECONDS
    ):
        raise ValueError("integrated absolute deadlines are invalid")

    status = remote["status"]
    run_error = remote["run_error"]
    cleanup = remote["in_function_cleanup"]
    controller = remote["controller"]
    expected_schema = (
        "sloforge.branchfabric.experiment-004-v11-targeted-source-identity-completion/v1"
        if _is_targeted(config)
        else "sloforge.branchfabric.experiment-004-v11-integrated-completion/v1"
    )
    expected_scientific_status = (
        "targeted-identity-pass-pending-cleanup"
        if _is_targeted(config)
        else "pending-local-budget-settlement-and-provider-cleanup"
    )
    provisional = (
        status == "provisional"
        and remote["scientific_status"] == expected_scientific_status
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
        or remote["schema_version"] != expected_schema
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
    inventory: tuple[dict[str, Any], dict[str, Any]] | None
    if (
        failed
        and isinstance(controller, dict)
        and controller.get("cleanup_scope") == ("PRE_WORKER_PREFLIGHT")
    ):
        _require_exact_pre_worker_failure(remote, config=config)
        inventory = None
    else:
        inventory = _exact_inventory(remote)
    return dict(remote), dict(materialized), inventory


def _download_result(
    materialized: Mapping[str, Any],
    *,
    local_root: Path,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> None:
    if local_root.exists() or local_root.is_symlink():
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
    if local_root.is_symlink() or not local_root.is_dir():
        raise FileNotFoundError("Modal volume download did not materialize the expected prefix")


def _verify_downloaded_manifest(
    local_root: Path,
    *,
    remote: Mapping[str, Any],
    materialized: Mapping[str, Any],
) -> Path:
    if local_root.is_symlink() or not local_root.is_dir():
        raise ValueError("downloaded integrated artifact root is invalid")
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
    verified_remote: Mapping[str, Any] | None = None,
    verified_manifest_path: Path | None = None,
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
    verified_failure = None
    if verified_remote is not None or verified_manifest_path is not None:
        if verified_remote is None or verified_manifest_path is None:
            raise ValueError("verified remote failure requires both result and manifest")
        verified_failure = {
            "status": verified_remote.get("status"),
            "scientific_status": verified_remote.get("scientific_status"),
            "function_call_id": verified_remote.get("function_call_id"),
            "cleanup_scope": (
                verified_remote.get("controller", {}).get("cleanup_scope")
                if isinstance(verified_remote.get("controller"), dict)
                else None
            ),
            "failure_stage": (
                verified_remote.get("controller", {}).get("failure_stage")
                if isinstance(verified_remote.get("controller"), dict)
                else None
            ),
            "controller_error": (
                verified_remote.get("controller", {}).get("controller_error")
                if isinstance(verified_remote.get("controller"), dict)
                else None
            ),
            "remote_manifest": str(verified_manifest_path),
            "remote_manifest_sha256": _sha256_file(verified_manifest_path),
            "remote_reported_allocation_seconds_diagnostic_only": verified_remote.get(
                "gpu_allocation_seconds"
            ),
            "remote_reported_gpu_seconds_diagnostic_only": verified_remote.get("gpu_seconds"),
        }
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
        "verified_remote_failure": verified_failure,
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


def _conservatively_settle_verified_failure(
    *,
    ledger_path: Path,
    reservation_id: str,
    attempt_id: str,
    failure_root: Path,
    remote: Mapping[str, Any],
    inventory: tuple[dict[str, Any], dict[str, Any]] | None,
    manifest_path: Path,
    client_elapsed_seconds: float,
    completed: subprocess.CompletedProcess[str],
) -> dict[str, Any]:
    """Terminate any verified failed result without measured-success settlement."""

    if remote.get("status") != "failed":
        raise ValueError("verified failure settlement requires failed remote status")
    if inventory is None:
        controller = remote.get("controller")
        if not isinstance(controller, dict):
            raise ValueError("verified PRE_WORKER failure omits controller")
        failure = controller.get("controller_error")
        stage = "verified-pre-worker-failure-conservative-settlement"
    else:
        failure = remote.get("run_error")
        stage = "verified-post-worker-failure-conservative-settlement"
    if (
        not isinstance(failure, dict)
        or set(failure) < {"type", "message"}
        or not isinstance(failure["type"], str)
        or not failure["type"]
        or not isinstance(failure["message"], str)
        or not failure["message"]
    ):
        raise ValueError("verified remote failure omits its primary error")
    return _charge_active_failure(
        ledger_path=ledger_path,
        reservation_id=reservation_id,
        attempt_id=attempt_id,
        failure_root=failure_root,
        stage=stage,
        error=RuntimeError(f"{failure['type']}: {failure['message']}"),
        client_elapsed_seconds=client_elapsed_seconds,
        completed=completed,
        verified_remote=remote,
        verified_manifest_path=manifest_path,
    )


def _require_integrated_pre_reservation_outputs_fresh(
    config: V11Config,
    *,
    provider_inputs_path: Path,
    failure_root: Path,
    terminal_output_paths: Sequence[Path] = (),
) -> None:
    """Reject any stale current-attempt outputs before a paid reservation."""

    if _is_targeted(config):
        return
    conservative_failure_path = failure_root / f"{config.attempt_id}-conservative-charge.json"
    collisions = tuple(
        path
        for path in (provider_inputs_path, conservative_failure_path, *terminal_output_paths)
        if path.exists() or path.is_symlink()
    )
    if collisions:
        raise FileExistsError(
            "fresh integrated attempt output collision before reservation: "
            + ", ".join(str(path) for path in collisions)
        )


def _require_local_result_output_fresh(local_root: Path) -> None:
    if local_root.exists() or local_root.is_symlink():
        raise FileExistsError(
            f"fresh integrated attempt already has a local immutable result: {local_root}"
        )


def _require_remote_attempt_prefix_fresh(
    config: V11Config,
    *,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> dict[str, Any]:
    """Prove the persistent results volume has no current-attempt publication path."""

    completed = runner(
        ["modal", "volume", "ls", RESULTS_VOLUME, REMOTE_PREFIX, "--json"],
        check=False,
        capture_output=True,
        text=True,
        timeout=60.0,
    )
    if completed.returncode != 0:
        raise RuntimeError("Modal results-volume prefix inventory failed")
    try:
        rows = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise ValueError("Modal results-volume prefix inventory is not JSON") from error
    expected_fields = {"filename", "type", "created_modified", "size"}
    if not isinstance(rows, list) or any(
        not isinstance(row, dict)
        or set(row) != expected_fields
        or any(not isinstance(row[field], str) for field in expected_fields)
        or any(not row[field] for field in expected_fields)
        or row["type"] not in {"dir", "file"}
        for row in rows
    ):
        raise ValueError("Modal results-volume prefix inventory schema is invalid")
    attempt_prefix = f"{REMOTE_PREFIX}/{config.attempt_id}"
    forbidden_paths = (
        attempt_prefix,
        f"{attempt_prefix}.staging",
        f"{attempt_prefix}.inflight",
    )
    raw_observed_rows = [str(row["filename"]) for row in rows]
    observed_rows = [path.strip("/") for path in raw_observed_rows]
    if len(set(observed_rows)) != len(observed_rows) or any(
        raw != path
        or not path.startswith(f"{REMOTE_PREFIX}/")
        or "\\" in path
        or "\x00" in path
        or any(part in {"", ".", ".."} for part in path.split("/"))
        for raw, path in zip(raw_observed_rows, observed_rows, strict=True)
    ):
        raise ValueError("Modal results-volume prefix inventory paths are invalid")
    observed = set(observed_rows)
    collisions = sorted(
        path
        for path in observed
        if any(
            path == forbidden or path.startswith(f"{forbidden}/") for forbidden in forbidden_paths
        )
    )
    if collisions:
        raise FileExistsError(
            "fresh integrated attempt remote prefix exists before reservation: "
            + ", ".join(collisions)
        )
    return {
        "schema_version": "sloforge.branchfabric.remote-prefix-freshness/v1",
        "status": "PASS",
        "volume_name": RESULTS_VOLUME,
        "listed_prefix": REMOTE_PREFIX,
        "attempt_id": config.attempt_id,
        "forbidden_paths": list(forbidden_paths),
        "observed_entries": rows,
        "collision_count": 0,
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
        targeted = _is_targeted(config)
        local_parent = (
            _EXPERIMENT_ROOT / "v11-final/targeted-repro/raw" if targeted else _LOCAL_RESULT_PARENT
        )
        failure_root = (
            _EXPERIMENT_ROOT / "v11-final/targeted-repro/failures" if targeted else _FAILURE_ROOT
        )
        provider_inputs_path = (
            _EXPERIMENT_ROOT
            / "v11-final/targeted-repro"
            / f"provider-cleanup-audit-inputs-{config.attempt_id}.json"
            if targeted
            else _PROVIDER_INPUTS_PARENT / f"provider-cleanup-audit-inputs-{config.attempt_id}.json"
        )
        local_root = local_parent / config.attempt_id
        prebuild_path = _PREBUILD_ROOT / f"{config.attempt_id}.json"
        post_prebuild_freshness_path = (
            _PREBUILD_ROOT / f"{config.attempt_id}-pre-reservation-freshness.json"
        )
        _require_local_result_output_fresh(local_root)
        attempt_suffix = config.attempt_id.rsplit("-", maxsplit=1)[-1]
        terminal_output_paths = (
            _PROVIDER_INPUTS_PARENT / f"status-attempt-{attempt_suffix}.json",
            _PROVIDER_INPUTS_PARENT / f"provider-cleanup-attempt-{attempt_suffix}.json",
        )
        _require_integrated_pre_reservation_outputs_fresh(
            config,
            provider_inputs_path=provider_inputs_path,
            failure_root=failure_root,
            terminal_output_paths=terminal_output_paths,
        )
        if prebuild_path.exists() or prebuild_path.is_symlink():
            raise FileExistsError(
                f"fresh integrated attempt already has image-prebuild evidence: {prebuild_path}"
            )
        if post_prebuild_freshness_path.exists() or post_prebuild_freshness_path.is_symlink():
            raise FileExistsError(
                "fresh integrated attempt already has post-prebuild freshness evidence: "
                f"{post_prebuild_freshness_path}"
            )
        remote_prefix_freshness = _require_remote_attempt_prefix_fresh(config)
        prebuild = _prebuild_modal_image(budget_usd=budget)
        _write_new_json(
            prebuild_path,
            {
                **prebuild,
                "attempt_id": config.attempt_id,
                "remote_prefix_freshness": remote_prefix_freshness,
            },
        )
        reloaded_config, reloaded_seal = _load_sealed_config(
            config_path,
            repository_root=_ROOT,
            ledger_path=_LEDGER,
        )
        if reloaded_config != config or _deterministic_seal_projection(
            reloaded_seal
        ) != _deterministic_seal_projection(seal):
            raise RuntimeError("integrated v11 seal changed during Modal image prebuild")
        _require_local_result_output_fresh(local_root)
        _require_integrated_pre_reservation_outputs_fresh(
            config,
            provider_inputs_path=provider_inputs_path,
            failure_root=failure_root,
            terminal_output_paths=terminal_output_paths,
        )
        post_prebuild_remote_prefix_freshness = _require_remote_attempt_prefix_fresh(config)
        before_entries = remote_prefix_freshness["observed_entries"]
        after_entries = post_prebuild_remote_prefix_freshness["observed_entries"]
        _write_new_json(
            post_prebuild_freshness_path,
            {
                "schema_version": (
                    "sloforge.branchfabric.post-prebuild-pre-reservation-freshness/v1"
                ),
                "status": "PASS",
                "attempt_id": config.attempt_id,
                "prebuild_evidence": {
                    "artifact": str(prebuild_path.relative_to(_ROOT)),
                    "sha256": _sha256_file(prebuild_path),
                },
                "before_remote_prefix_inventory_sha256": _sha256_bytes(
                    _canonical_bytes({"observed_entries": before_entries})
                ),
                "after_remote_prefix_inventory_sha256": _sha256_bytes(
                    _canonical_bytes({"observed_entries": after_entries})
                ),
                "remote_prefix_inventory_unchanged": before_entries == after_entries,
                "prebuild_remote_prefix_freshness": remote_prefix_freshness,
                "post_prebuild_remote_prefix_freshness": (post_prebuild_remote_prefix_freshness),
                "local_result_collision_count": 0,
                "terminal_output_collision_count": 0,
            },
        )
        reservation_id = (
            f"exp004-v11-target-{secrets.token_hex(8)}"
            if targeted
            else (
                f"exp004-v11-kill-{attempt_suffix}-{secrets.token_hex(10)}"
                if getattr(config, "mode", None) == "KILL_AND_RECOMPUTE"
                else f"exp004-v11-{attempt_suffix}-{secrets.token_hex(10)}"
            )
        )
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
                wall_seconds=_function_wall_seconds(config),
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
            if remote["status"] == "failed":
                stage = (
                    "verified-pre-worker-failure-conservative-settlement"
                    if inventory is None
                    else "verified-post-worker-failure-conservative-settlement"
                )
                charge = _conservatively_settle_verified_failure(
                    ledger_path=_LEDGER,
                    reservation_id=reservation_id,
                    attempt_id=config.attempt_id,
                    failure_root=failure_root,
                    remote=remote,
                    inventory=inventory,
                    manifest_path=manifest,
                    client_elapsed_seconds=elapsed,
                    completed=completed,
                )
                raise _VerifiedRemoteFailure(charge)
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
                provider_inputs_path,
                remote=remote,
                reservation_id=reservation_id,
                manifest_path=manifest,
                ledger_path=_LEDGER,
            )
        except _VerifiedRemoteFailure as error:
            raise RuntimeError(
                "integrated v11 remote failure preserved and conservatively settled; "
                f"accounting={error.charge['status']}"
            ) from error
        except BaseException as error:
            elapsed = time.monotonic() - started
            charge = _charge_active_failure(
                ledger_path=_LEDGER,
                reservation_id=reservation_id,
                attempt_id=config.attempt_id,
                failure_root=failure_root,
                stage=stage,
                error=error,
                client_elapsed_seconds=elapsed,
                completed=completed,
            )
            raise RuntimeError(
                f"integrated v11 attempt terminated; accounting={charge['status']}"
            ) from error
        output = {
            "schema_version": (
                "sloforge.branchfabric.exp004-v11-targeted-coordinator/v1"
                if targeted
                else "sloforge.branchfabric.exp004-v11-final-coordinator/v1"
            ),
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
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
