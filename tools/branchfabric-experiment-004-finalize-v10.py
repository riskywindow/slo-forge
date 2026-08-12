#!/usr/bin/env python3
"""Finalize Experiment 004 v10 only after ledger settlement and Modal cleanup.

The paid remote function cannot observe its own provider teardown or the local
GPU-hour ledger settlement. This CPU-only finalizer requires both as immutable
inputs and writes the canonical scientific-validity artifact fail closed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "python"))


def _canonical_bytes(value: Any) -> bytes:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n"
    ).encode()


def _write_new(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("xb") as handle:
        handle.write(_canonical_bytes(value))
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _load_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"finalizer input is not an object: {path}")
    return payload


def main() -> int:
    from sloforge.helix.characterization.gpu_reclamation_methodology import (
        Experiment004GpuHourLedger,
    )
    from sloforge.helix.characterization.gpu_reclamation_v10_postprocess import (
        assess_v10_directory,
    )
    from sloforge.helix.characterization.gpu_reclamation_v10_validity import (
        BudgetEvidence,
        CleanupEvidence,
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-path", required=True, type=Path)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--gpu-hours-path", required=True, type=Path)
    parser.add_argument("--modal-cleanup-evidence", required=True, type=Path)
    args = parser.parse_args()
    config = _load_object(args.config_path.resolve(strict=True))
    if config.get("serving_methodology") != "v10-global-capacity":
        raise ValueError("v10 finalizer refuses a non-v10 serving methodology")
    run_root = args.run_root.resolve(strict=True)
    controller = _load_object(run_root / "controller-result.json")
    ledger = Experiment004GpuHourLedger.model_validate_json(
        args.gpu_hours_path.resolve(strict=True).read_text(), strict=True
    )
    settlement = _load_object(run_root / "analysis/budget-settlement.json")
    ledger_digest = hashlib.sha256(args.gpu_hours_path.resolve(strict=True).read_bytes()).hexdigest()
    if (
        settlement.get("attempt_id") != config["attempt_id"]
        or settlement.get("gpu_hours_sha256") != ledger_digest
    ):
        raise ValueError("budget settlement is not bound to the current GPU-hour ledger")
    matching = tuple(
        interval
        for interval in ledger.intervals
        if interval.invocation_id == str(config["attempt_id"])
    )
    if len(matching) != 1 or ledger.reservations:
        raise ValueError("v10 finalizer requires one settled invocation and zero reservations")
    invocation_gpu_seconds = matching[0].gpu_seconds
    before = ledger.consumed_additional_gpu_seconds - invocation_gpu_seconds
    budget = BudgetEvidence(
        conservative_gpu_seconds_before=before,
        invocation_gpu_seconds=invocation_gpu_seconds,
        conservative_gpu_seconds_after=ledger.consumed_additional_gpu_seconds,
        hard_ceiling_gpu_seconds=ledger.hard_additional_gpu_seconds,
        ledger_updated=True,
    )
    if settlement.get("budget") != budget.model_dump(mode="json"):
        raise ValueError("budget settlement artifact differs from the settled ledger")
    cleanup_payload = _load_object(args.modal_cleanup_evidence.resolve(strict=True))
    expected_volumes = {"sloforge-model-cache", "sloforge-branchfabric-results"}
    empty_fields = (
        "active_tasks",
        "running_containers",
        "endpoints",
        "reservations",
        "owned_child_processes",
        "profilers",
    )
    if (
        cleanup_payload.get("schema_version")
        != "sloforge.branchfabric.experiment-004-modal-cleanup-evidence/v1"
        or cleanup_payload.get("attempt_id") != config["attempt_id"]
        or cleanup_payload.get("observed_after_function_call_id")
        != matching[0].function_call_id
        or not isinstance(cleanup_payload.get("raw_provider_queries"), list)
        or not cleanup_payload["raw_provider_queries"]
        or any(cleanup_payload.get(field) != [] for field in empty_fields)
        or set(cleanup_payload.get("persistent_volumes", ())) != expected_volumes
    ):
        raise ValueError("Modal cleanup evidence is incomplete or does not prove zero resources")
    cleanup = CleanupEvidence(
        zero_active_tasks=True,
        zero_running_containers=True,
        zero_endpoints=True,
        zero_reservations=True,
        zero_owned_child_processes=True,
        zero_profilers=True,
        authorized_persistent_volumes_only=True,
    )
    scientific, movement, recovery = assess_v10_directory(
        work_root=run_root,
        config=config,
        controller=controller,
        budget=budget,
        cleanup=cleanup,
    )
    for name, payload in (
        ("movement-accounting.json", movement),
        ("serving-recovery.json", recovery),
    ):
        existing = run_root / "analysis" / name
        if existing.is_file() and existing.read_bytes() != _canonical_bytes(payload):
            raise ValueError(f"local recomputation differs from remote provisional {name}")
        if not existing.is_file():
            _write_new(existing, payload)
    _write_new(run_root / "analysis/scientific-validity.json", scientific)
    print(_canonical_bytes(scientific).decode(), end="")
    return 0 if scientific.scientifically_valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
