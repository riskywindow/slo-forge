"""Bounded scheduler-scenario loading at the versioned JSON boundary."""

from __future__ import annotations

from pathlib import Path

from .models import SchedulerRequest

MAX_SCHEDULER_SCENARIO_BYTES = 16 * 1024 * 1024


def load_scheduler_request(path: Path) -> SchedulerRequest:
    with path.open("rb") as handle:
        payload = handle.read(MAX_SCHEDULER_SCENARIO_BYTES + 1)
    if len(payload) > MAX_SCHEDULER_SCENARIO_BYTES:
        raise ValueError("scheduler scenario exceeds 16 MiB")
    return SchedulerRequest.model_validate_json(payload, strict=True)


__all__ = ["MAX_SCHEDULER_SCENARIO_BYTES", "load_scheduler_request"]
