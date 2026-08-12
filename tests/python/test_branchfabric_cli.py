from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from sloforge.cli.main import app

runner = CliRunner()


def test_characterization_help_exposes_complete_measurement_surface() -> None:
    helix = runner.invoke(app, ["helix", "--help"])
    assert helix.exit_code == 0
    assert "trace" in helix.stdout
    assert "characterize" in helix.stdout

    trace = runner.invoke(app, ["helix", "trace", "--help"])
    assert trace.exit_code == 0
    assert "branch" in trace.stdout

    characterize = runner.invoke(app, ["helix", "characterize", "--help"])
    assert characterize.exit_code == 0
    for command in (
        "run",
        "resume",
        "workload",
        "cow",
        "multicast",
        "transform",
        "metadata",
        "amdahl",
        "requirements",
        "report",
    ):
        assert command in characterize.stdout


def test_characterization_matrix_dry_run_is_bounded_and_does_not_write(tmp_path: Path) -> None:
    output = tmp_path / "run"
    result = runner.invoke(
        app,
        [
            "helix",
            "characterize",
            "run",
            "--matrix",
            "benchmarks/branchfabric/characterization.yaml",
            "--output",
            str(output),
            "--max-experiments",
            "100000",
            "--seed",
            "19",
            "--dry-run",
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["dry_run"] is True
    assert payload["case_count"] > 0
    assert payload["distribution_claim"] is False
    assert payload["seed"] == 19
    assert not output.exists()


def test_trace_branch_rejects_unimplemented_live_session_without_output(tmp_path: Path) -> None:
    output = tmp_path / "traces"
    result = runner.invoke(
        app,
        [
            "helix",
            "trace",
            "branch",
            "--session",
            "live-production-session",
            "--output",
            str(output),
        ],
    )
    assert result.exit_code != 0
    assert "arbitrary live-session attachment" in result.output
    assert "implemented" in result.output
    assert not output.exists()


def test_cow_rejects_malformed_page_sizes_before_backend_dispatch(tmp_path: Path) -> None:
    trace = tmp_path / "state.jsonl"
    trace.write_text("{}\n")
    result = runner.invoke(
        app,
        [
            "helix",
            "characterize",
            "cow",
            "--trace",
            str(trace),
            "--output",
            str(tmp_path / "cow"),
            "--page-sizes",
            "4k,bad",
        ],
    )
    assert result.exit_code != 0
    assert "--page-sizes must contain values" in result.output
    assert not (tmp_path / "cow").exists()
