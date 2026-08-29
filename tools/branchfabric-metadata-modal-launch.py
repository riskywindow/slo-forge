#!/usr/bin/env python3
"""Budget-first launcher for the single-GPU Experiment 003 Modal campaign."""

from __future__ import annotations

import argparse
import json
import math
import os
import secrets
import subprocess
from collections.abc import Sequence
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "experiments/branchfabric/modal_metadata_characterization.py"
HARD_GPU_HOURS = 1.25
FUNCTION_GPU_HOURS = 2400 / 3600
FUNCTION_TIMEOUT_SECONDS = 2400
GPU_USD_PER_HOUR = 2.4984
SUPPORT_USD_PER_SECOND = 8 * 0.0000131 + 32 * 0.00000222
RESERVE_FRACTION = 0.15


def _arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-path", required=True, type=Path)
    return parser.parse_args(argv)


def _budget() -> float:
    raw = os.getenv("SLOFORGE_GPU_BUDGET_USD")
    if raw is None:
        raise RuntimeError("SLOFORGE_GPU_BUDGET_USD is required before invoking Modal")
    try:
        value = float(raw)
    except ValueError as error:
        raise RuntimeError("SLOFORGE_GPU_BUDGET_USD must be finite and positive") from error
    maximum_cost = (
        FUNCTION_GPU_HOURS * GPU_USD_PER_HOUR + FUNCTION_TIMEOUT_SECONDS * SUPPORT_USD_PER_SECOND
    )
    if not math.isfinite(value) or value <= 0 or maximum_cost > value * (1 - RESERVE_FRACTION):
        raise RuntimeError("Experiment 003 Modal call exceeds its authorized USD budget")
    return value


def _validate_without_modal(path: Path) -> None:
    if path.stat().st_size > 1024 * 1024:
        raise ValueError("Experiment 003 campaign input exceeds 1 MiB")
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError("Experiment 003 campaign input must be an object")
    if payload.get("schema_version") != "sloforge.branchfabric.modal-campaign-003/v1":
        raise ValueError("Experiment 003 campaign schema is invalid")
    maximum_campaign_seconds = payload.get("maximum_campaign_seconds")
    if maximum_campaign_seconds not in {1800, 2300}:
        raise ValueError("Experiment 003 campaign has an unsupported wall-time bound")
    trials = payload.get("trials")
    if not isinstance(trials, list) or not 1 <= len(trials) <= 24:
        raise ValueError("Experiment 003 campaign must contain 1..24 trials")
    if any(
        not isinstance(row, dict)
        or row.get("schema_version")
        != "sloforge.branchfabric.modal-metadata-characterization-config/v1"
        or row.get("fanout") not in {1, 8, 16, 32}
        or row.get("prefix_length") != 16_384
        or row.get("suffix_length") != 256
        for row in trials
    ):
        raise ValueError("Experiment 003 campaign contains an unbounded trial")
    ledger = ROOT / "artifacts/branchfabric/gpu-validation/experiment-003/gpu-hours.json"
    if ledger.is_file():
        current = json.loads(ledger.read_text())
        consumed = float(current.get("consumed_gpu_hours", 0.0))
        reservations = current.get("modal_in_flight_reservations", [])
        if reservations:
            raise RuntimeError("Experiment 003 has an unresolved Modal reservation")
        if consumed + FUNCTION_GPU_HOURS > HARD_GPU_HOURS:
            raise RuntimeError("Experiment 003 Modal call exceeds the 1.25 A100-hour hard limit")


def main(argv: Sequence[str] | None = None) -> int:
    args = _arguments(argv)
    _budget()  # Deliberately precedes campaign parsing and all Modal imports.
    campaign = args.campaign_path.resolve(strict=True)
    _validate_without_modal(campaign)
    environment = os.environ.copy()
    environment["SLOFORGE_MODAL_PREFLIGHT_TOKEN"] = secrets.token_hex(32)
    completed = subprocess.run(
        [
            "modal",
            "run",
            str(APP),
            "--campaign-path",
            str(campaign),
        ],
        cwd=ROOT,
        env=environment,
        check=False,
    )
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
