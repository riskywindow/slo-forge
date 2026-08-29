"""Declared sample generator kept separate from synthesis execution."""

from __future__ import annotations


def sample_inputs(*, seed: int) -> tuple[dict[str, object], ...]:
    return (
        {"text": "hybrid", "maximum_new_tokens": 5, "seed": seed},
        {"text": "agent", "maximum_new_tokens": 4, "seed": seed + 1},
    )
