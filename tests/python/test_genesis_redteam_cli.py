from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from sloforge.cli.main import app


def test_redteam_cli_runs_and_replays_minimization(tmp_path: Path) -> None:
    runner = CliRunner()
    root = tmp_path / "redteam"
    run = runner.invoke(
        app,
        ["redteam", "run", "--output", str(root), "--seed", "73129"],
    )
    assert run.exit_code == 0, run.output
    report = json.loads((root / "redteam-report.json").read_text(encoding="utf-8"))
    source = (
        root
        / "counterexamples"
        / f"{report['findings'][0]['counterexample']['counterexample_id']}.json"
    )
    output = tmp_path / "minimized.json"
    minimized = runner.invoke(
        app,
        [
            "redteam",
            "minimize",
            "--counterexample",
            str(source),
            "--output",
            str(output),
        ],
    )
    assert minimized.exit_code == 0, minimized.output
    assert json.loads(output.read_text(encoding="utf-8"))["minimized"] is True
