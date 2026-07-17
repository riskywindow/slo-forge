"""Bounded machine-readable fault matrix loading."""

from __future__ import annotations

from pathlib import Path

from .models import FaultPlanRequest

MAX_FAULT_MATRIX_BYTES = 16 * 1024 * 1024


def load_fault_plan_request(path: Path) -> FaultPlanRequest:
    resolved = path.resolve(strict=True)
    if not resolved.is_file():
        raise ValueError("fault matrix must be a regular file")
    size = resolved.stat().st_size
    if size <= 0:
        raise ValueError("fault matrix cannot be empty")
    if size > MAX_FAULT_MATRIX_BYTES:
        raise ValueError(f"fault matrix exceeds {MAX_FAULT_MATRIX_BYTES} bytes")
    return FaultPlanRequest.model_validate_json(resolved.read_bytes(), strict=True)


__all__ = ["MAX_FAULT_MATRIX_BYTES", "load_fault_plan_request"]
