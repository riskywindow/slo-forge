#!/usr/bin/env python3
"""Budget-first launcher for the BranchFabric Modal experiment.

This process intentionally imports no Modal code. It validates the repository's paid-cloud
guard before starting the Modal CLI, so an unset/invalid budget cannot hydrate an App, build an
Image, inspect a Volume, or invoke a Function.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
import re
import secrets
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_APP = _ROOT / "experiments/branchfabric/modal_real_gpu_cow.py"
_LEDGER = _ROOT / "artifacts/branchfabric/gpu-validation/experiment-002/gpu-hours.json"
_GPU_HOUR_LIMIT = 4.0
_BUDGET_RESERVE_FRACTION = 0.15
_GPU_MAXIMUM_SECONDS = 5400 + 1800
_MODEL_MAXIMUM_SECONDS = 7200 + 900
_GPU_COUNT = 1
_GPU_USD_PER_HOUR = 2.4984
_CPU_USD_PER_CORE_SECOND = 0.0000131
_MEMORY_USD_PER_GIB_SECOND = 0.00000222
_GPU_CPU_CORES = 8.0
_GPU_MEMORY_GIB = 32.0
_MODEL_CPU_CORES = 4.0
_MODEL_MEMORY_GIB = 16.0
_REMOTE_ACTIONS = {
    "prepare-model",
    "run",
    "recover",
    "cancel",
    "materialize",
    "cleanup-staging",
    "reconcile-unattached",
}


def _arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", required=True, choices=sorted(_REMOTE_ACTIONS))
    parser.add_argument("--config-path", required=True, type=Path)
    parser.add_argument("--remote-manifest-sha256", default="")
    parser.add_argument("--provider-audit-path", type=Path)
    return parser.parse_args(argv)


def _require_budget_before_modal() -> float:
    raw = os.getenv("SLOFORGE_GPU_BUDGET_USD")
    if raw is None:
        raise RuntimeError("SLOFORGE_GPU_BUDGET_USD is required before invoking Modal")
    try:
        value = float(raw)
    except ValueError as error:
        raise RuntimeError("SLOFORGE_GPU_BUDGET_USD must be finite and positive") from error
    if not math.isfinite(value) or value <= 0:
        raise RuntimeError("SLOFORGE_GPU_BUDGET_USD must be finite and positive")
    return value


def _nonnegative_number(value: object, *, field: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
        or value < 0
    ):
        raise ValueError(f"GPU-hour ledger has invalid {field}")
    return float(value)


def _validate_config_without_modal(config: Path) -> None:
    """Reject malformed inputs before the Modal CLI imports or hydrates the App."""

    if config.stat().st_size > 1024 * 1024:
        raise ValueError("Modal benchmark config exceeds 1 MiB")
    payload = json.loads(config.read_text())
    if not isinstance(payload, dict):
        raise ValueError("Modal benchmark config must be a JSON object")
    allowed_fields = {
        "schema_version",
        "attempt_id",
        "model",
        "model_revision",
        "tokenizer_revision",
        "runtime",
        "runtime_version",
        "fanout",
        "prefix_length",
        "suffix_length",
        "seed",
        "baseline_mode",
        "tracing_level",
        "maximum_wall_seconds",
        "initialization_timeout_seconds",
        "cleanup_timeout_seconds",
        "gpu_memory_utilization",
        "gpu_sample_interval_seconds",
    }
    if set(payload) - allowed_fields:
        raise ValueError("Modal benchmark config has unsupported fields")
    expected = {
        "schema_version": "sloforge.branchfabric.modal-real-gpu-cow-config/v1",
        "runtime": "vllm",
        "runtime_version": "0.23.0",
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "model_revision": "a09a35458c702b33eeacc393d103063234e8bc28",
        "tokenizer_revision": "a09a35458c702b33eeacc393d103063234e8bc28",
    }
    for field, expected_value in expected.items():
        if payload.get(field, expected_value) != expected_value:
            raise ValueError(f"Modal benchmark config has unsupported {field}")
    attempt_id = payload.get("attempt_id")
    if (
        not isinstance(attempt_id, str)
        or re.fullmatch(r"[a-z0-9][a-z0-9-]{7,95}", attempt_id) is None
    ):
        raise ValueError("Modal benchmark config has invalid attempt_id")
    integer_fields = {
        "fanout": ({1, 2, 8, 32}, None),
        "prefix_length": ({2048, 4096, 16384}, None),
        "suffix_length": ({16, 256}, None),
        "seed": (None, (0, (1 << 63) - 1)),
        "maximum_wall_seconds": (None, (60, 3600)),
        "initialization_timeout_seconds": (None, (60, 1800)),
        "cleanup_timeout_seconds": (None, (10, 300)),
    }
    defaults = {
        "maximum_wall_seconds": 3600,
        "initialization_timeout_seconds": 1200,
        "cleanup_timeout_seconds": 60,
    }
    values: dict[str, int] = {}
    for field, (choices, bounds) in integer_fields.items():
        value = payload.get(field, defaults.get(field))
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError(f"Modal benchmark config has invalid {field}")
        if choices is not None and value not in choices:
            raise ValueError(f"Modal benchmark config has invalid {field}")
        if bounds is not None and not bounds[0] <= value <= bounds[1]:
            raise ValueError(f"Modal benchmark config has invalid {field}")
        values[field] = value
    baseline = payload.get("baseline_mode")
    tracing = payload.get("tracing_level", "full")
    if baseline not in {"independent", "shared_root"} or tracing not in {
        "disabled",
        "minimal",
        "full",
    }:
        raise ValueError("Modal benchmark config has invalid baseline/tracing mode")
    prefix = values["prefix_length"]
    fanout = values["fanout"]
    suffix = values["suffix_length"]
    if prefix in {2048, 4096}:
        if fanout != 2 or suffix != 16 or baseline != "shared_root":
            raise ValueError("Modal smoke config shape is invalid")
    elif fanout not in {1, 8, 32} or suffix != 256:
        raise ValueError("Modal 16K config shape is invalid")
    if (
        values["initialization_timeout_seconds"]
        + values["maximum_wall_seconds"]
        + values["cleanup_timeout_seconds"]
        + 300
        > 5400
    ):
        raise ValueError("Modal benchmark timeouts exceed the Function bound")
    for field, default, lower, upper, lower_is_inclusive in (
        ("gpu_memory_utilization", 0.80, 0.0, 0.90, False),
        ("gpu_sample_interval_seconds", 0.25, 0.05, 2.0, True),
    ):
        value = payload.get(field, default)
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(value)
            or (value < lower if lower_is_inclusive else value <= lower)
            or value > upper
        ):
            raise ValueError(f"Modal benchmark config has invalid {field}")


def _project_before_modal(*, action: str, budget_usd: float) -> None:
    """Apply the complete cost/hour ceiling before starting the Modal process."""

    if _LEDGER.is_file():
        ledger = json.loads(_LEDGER.read_text())
        if not isinstance(ledger, dict):
            raise ValueError("GPU-hour ledger must be a JSON object")
    else:
        ledger = {}
    consumed_hours = _nonnegative_number(
        ledger.get("consumed_gpu_hours", 0.0), field="consumed_gpu_hours"
    )
    consumed_cost = _nonnegative_number(
        ledger.get("consumed_cloud_cost_usd", 0.0), field="consumed_cloud_cost_usd"
    )
    gpu_reservations = ledger.get("modal_in_flight_reservations", [])
    model_reservations = ledger.get("modal_in_flight_model_reservations", [])
    if not isinstance(gpu_reservations, list) or not isinstance(model_reservations, list):
        raise ValueError("GPU-hour ledger has invalid reservation lists")
    reserved_hours = 0.0
    reserved_cost = 0.0
    for reservation in gpu_reservations:
        if not isinstance(reservation, dict):
            raise ValueError("GPU-hour ledger has invalid GPU reservation")
        reserved_hours += _nonnegative_number(
            reservation.get("maximum_gpu_hours"), field="maximum_gpu_hours reservation"
        )
        reserved_cost += _nonnegative_number(
            reservation.get("maximum_cloud_cost_usd"),
            field="maximum_cloud_cost_usd reservation",
        )
    for reservation in model_reservations:
        if not isinstance(reservation, dict):
            raise ValueError("GPU-hour ledger has invalid model reservation")
        reserved_cost += _nonnegative_number(
            reservation.get("maximum_cloud_cost_usd"),
            field="maximum_cloud_cost_usd reservation",
        )

    if action not in {"prepare-model", "run"}:
        return

    proposed_gpu_seconds = _GPU_MAXIMUM_SECONDS if action == "run" else 0
    proposed_model_seconds = _MODEL_MAXIMUM_SECONDS if action in {"prepare-model", "run"} else 0
    proposed_gpu_hours = proposed_gpu_seconds * _GPU_COUNT / 3600.0
    if consumed_hours + reserved_hours + proposed_gpu_hours > _GPU_HOUR_LIMIT:
        raise RuntimeError("projected Modal call would exceed the four GPU-hour hard limit")
    proposed_cost = proposed_gpu_hours * _GPU_USD_PER_HOUR
    proposed_cost += proposed_gpu_seconds * (
        _GPU_CPU_CORES * _CPU_USD_PER_CORE_SECOND + _GPU_MEMORY_GIB * _MEMORY_USD_PER_GIB_SECOND
    )
    proposed_cost += proposed_model_seconds * (
        _MODEL_CPU_CORES * _CPU_USD_PER_CORE_SECOND + _MODEL_MEMORY_GIB * _MEMORY_USD_PER_GIB_SECOND
    )
    if consumed_cost + reserved_cost + proposed_cost > budget_usd * (
        1.0 - _BUDGET_RESERVE_FRACTION
    ):
        raise RuntimeError(
            "projected Modal compute cost exceeds the cloud budget after the 15% reserve"
        )
    if gpu_reservations or model_reservations:
        raise RuntimeError("an in-flight Modal reservation must be recovered or cancelled first")


def main(argv: Sequence[str] | None = None) -> int:
    args = _arguments(argv)
    budget = _require_budget_before_modal()
    config = args.config_path.resolve(strict=True)
    try:
        config.relative_to(_ROOT)
    except ValueError as error:
        raise ValueError("config path must be inside the SLOForge repository") from error
    _validate_config_without_modal(config)
    _project_before_modal(action=args.action, budget_usd=budget)
    command = [
        sys.executable,
        "-m",
        "modal",
        "run",
        "--detach",
        str(_APP),
        "--action",
        args.action,
        "--config-path",
        str(config),
    ]
    if args.remote_manifest_sha256:
        command.extend(["--remote-manifest-sha256", args.remote_manifest_sha256])
    if args.action == "reconcile-unattached":
        if args.provider_audit_path is None:
            raise ValueError("--provider-audit-path is required for reconcile-unattached")
        provider_audit = args.provider_audit_path.resolve(strict=True)
        allowed_audit_root = (
            _ROOT / "artifacts/branchfabric/gpu-validation/experiment-002/modal/provider-audits"
        ).resolve()
        try:
            provider_audit.relative_to(allowed_audit_root)
        except ValueError as error:
            raise ValueError(
                "provider audit must be under Experiment 002 modal/provider-audits"
            ) from error
        if provider_audit.is_symlink() or not provider_audit.is_file():
            raise ValueError("provider audit must be a regular file")
        command.extend(["--provider-audit-path", str(provider_audit)])
    elif args.provider_audit_path is not None:
        raise ValueError("--provider-audit-path is only valid for reconcile-unattached")
    lock_path = (
        _ROOT / "artifacts/branchfabric/gpu-validation/experiment-002/modal/coordinator.lock"
    )
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("another BranchFabric Modal coordinator is active") from error
        environment = os.environ.copy()
        environment["SLOFORGE_MODAL_PREFLIGHT_TOKEN"] = secrets.token_hex(32)
        completed = subprocess.run(command, cwd=_ROOT, env=environment, check=False)
        return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
