"""Bounded JSON input for Helix experience-selection scenarios."""

from __future__ import annotations

from pathlib import Path

from .models import ExperienceSelectionRequest

MAX_EXPERIENCE_SCENARIO_BYTES = 16 * 1024 * 1024


def load_experience_selection_request(path: Path) -> ExperienceSelectionRequest:
    """Load one strict request without accepting an unbounded scenario payload."""

    resolved = path.resolve(strict=True)
    if not resolved.is_file():
        raise ValueError("experience selection scenario must be a regular file")
    size = resolved.stat().st_size
    if size <= 0:
        raise ValueError("experience selection scenario cannot be empty")
    if size > MAX_EXPERIENCE_SCENARIO_BYTES:
        raise ValueError(
            f"experience selection scenario exceeds {MAX_EXPERIENCE_SCENARIO_BYTES} bytes"
        )
    return ExperienceSelectionRequest.model_validate_json(resolved.read_bytes(), strict=True)


__all__ = ["MAX_EXPERIENCE_SCENARIO_BYTES", "load_experience_selection_request"]
