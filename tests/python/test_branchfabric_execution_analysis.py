from __future__ import annotations

import json
from pathlib import Path

import pytest

from sloforge.helix.characterization.execution_analysis import analyze_reclamation


def _row(seed: int, trace: str, baseline: str, repetition: int) -> dict[str, object]:
    scale = 100 if baseline == "existing_serial" else 80
    trace_cost = {"disabled": 0, "minimal": 1, "full": 2}[trace]
    total = scale + seed + repetition + trace_cost
    return {
        "seed": seed,
        "trace_level": trace,
        "software_baseline": baseline,
        "repetition": repetition,
        "branch_readiness_ns": total,
        "pause_checkpoint_ns": total * 2,
        "migration_ns": total * 3,
        "resume_ns": total * 2,
        "total_interruption_ns": total * 8,
        "total_wall_ns": total * 10,
        "hidden_fallback": False,
        "requested_engine": "cpu-reference",
        "actual_engine": "cpu-reference",
        "physical_gpu_capacity_reclaimed": 0,
    }


def _write_fixture(path: Path) -> None:
    rows = [
        _row(seed, trace, baseline, repetition)
        for seed in (41, 73, 113)
        for trace in ("disabled", "minimal", "full")
        for baseline in ("existing_serial", "optimized_bounded_parallel")
        for repetition in (0, 1)
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_analysis_is_raw_bound_paired_and_deterministic(tmp_path: Path) -> None:
    raw = tmp_path / "trials.jsonl"
    _write_fixture(raw)

    first = analyze_reclamation(raw)
    second = analyze_reclamation(raw)

    assert first == second
    assert first["raw_trial_count"] == 36
    effect = first["paired_software_effects"]["total_interruption_ns"]
    assert effect["paired_effect_size"]["pair_count"] == 18
    assert effect["paired_speedup_median_ci"]["lower"] > 1.0
    relevance = first["state_movement_relevance_metric"]
    assert relevance["value"] == pytest.approx(3.0 / 8.0)
    assert not first["target_hardware_measured"]
    assert not first["calibrated_hardware_model_available"]


def test_analysis_fails_closed_on_engine_substitution(tmp_path: Path) -> None:
    raw = tmp_path / "trials.jsonl"
    _write_fixture(raw)
    rows = raw.read_text(encoding="utf-8").splitlines()
    changed = json.loads(rows[0])
    changed["actual_engine"] = "fallback"
    rows[0] = json.dumps(changed)
    raw.write_text("\n".join(rows) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="fallback"):
        analyze_reclamation(raw)
