from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from sloforge.cli.main import app
from sloforge.helix.capture import (
    ArtifactWatermark,
    CaptureBoundary,
    CoordinatedCaptureRequest,
)
from sloforge.helix.capture.models import make_branch_point


def test_helix_help_exposes_transactional_workflow() -> None:
    result = CliRunner().invoke(app, ["helix", "--help"])
    assert result.exit_code == 0
    for command in ("demo", "policy", "branchpoint", "scheduler", "promote", "rollback"):
        assert command in result.stdout


def test_branchpoint_validation_command(tmp_path: Path) -> None:
    request = CoordinatedCaptureRequest(
        capture_id="capture",
        session_id="session",
        source_trajectory_id="trajectory",
        policy_epoch_id="policy:0",
        boundary=CaptureBoundary(
            action_watermark=1,
            model_token_watermark=2,
            environment_event_watermark=3,
            effect_watermark=4,
        ),
        seed=5,
        max_quiescence_polls=2,
        published_at_ms=6,
        capture_timestamp="2026-08-03T00:00:00Z",
        git_commit="a3366807e879cf17615021e32606fbf77216235c",
        continuum_version="0.1.0",
        created_at="2026-08-03T00:00:00Z",
        reason="test boundary",
    )
    branchpoint = make_branch_point(
        request,
        continuum_capsule_id="a" * 64,
        environment=ArtifactWatermark(artifact_id="environment", watermark=3, digest="b" * 64),
        effects=ArtifactWatermark(artifact_id="effects", watermark=4, digest="c" * 64),
    )
    path = tmp_path / "branchpoint.json"
    path.write_text(branchpoint.model_dump_json())
    result = CliRunner().invoke(app, ["helix", "branchpoint", "validate", str(path)])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["valid"]
    assert payload["branch_point_id"] == branchpoint.branch_point_id


def test_scheduler_cli_labels_predictions_and_rejects_unconfigured_static_policy() -> None:
    runner = CliRunner()
    workload = "scenarios/helix/resource/cpu-learning-aware.json"
    result = runner.invoke(
        app,
        ["helix", "scheduler", "simulate", "--workload", workload],
    )
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["serving_predicted_slo_feasible"] is True
    assert "serving_slo_preserved" not in payload
    assert "predicted_learning_value" in payload

    static = runner.invoke(
        app,
        [
            "helix",
            "scheduler",
            "simulate",
            "--workload",
            workload,
            "--policy",
            "static",
        ],
    )
    assert static.exit_code != 0
    assert "explicit" in static.output
    assert "static_limits" in static.output


def test_evaluation_cli_rejects_non_integer_seeds_without_writing(tmp_path: Path) -> None:
    output = tmp_path / "evaluation"
    result = CliRunner().invoke(
        app,
        [
            "helix",
            "evaluate",
            "--output",
            str(output),
            "--seeds",
            "41,not-a-seed",
        ],
    )
    assert result.exit_code != 0
    assert "comma-separated list of integers" in result.output
    assert not output.exists()
