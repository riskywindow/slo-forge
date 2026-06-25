from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from sloforge.cli.main import app
from sloforge.genesis.ir import load_inference_genome

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "models/reference_tasks/hybrid_decoder"
runner = CliRunner()


def test_genesis_cli_exposes_complete_local_release_surface() -> None:
    result = runner.invoke(app, ["genesis", "--help"])
    assert result.exit_code == 0, result.output
    for command in (
        "benchmark",
        "compare",
        "deploy",
        "evolve",
        "promote",
        "replay",
    ):
        assert command in result.output


def test_inspect_and_initialize_zero_day_runtime(tmp_path: Path) -> None:
    inspection_dir = tmp_path / "inspection"
    inspected = runner.invoke(
        app,
        [
            "genesis",
            "inspect",
            "--reference",
            str(PACKAGE / "reference.py"),
            "--tokenizer",
            str(PACKAGE / "tokenizer.py"),
            "--contract",
            str(PACKAGE / "reference_package.json"),
            "--seed",
            "73129",
            "--output",
            str(inspection_dir),
        ],
    )
    assert inspected.exit_code == 0, inspected.output
    assert (inspection_dir / "inspection.json").is_file()
    assert (inspection_dir / "reference_package.json").is_file()

    hardware = tmp_path / "hardware.json"
    hardware.write_text(
        json.dumps({"schema_version": "1.0.0", "architecture": "cpu", "memory_bytes": 8 << 30}),
        encoding="utf-8",
    )
    run_dir = tmp_path / "run"
    initialized = runner.invoke(
        app,
        [
            "genesis",
            "initialize",
            "--inspection",
            str(inspection_dir),
            "--workload",
            str(PACKAGE / "search_samples.jsonl"),
            "--hardware",
            str(hardware),
            "--seed",
            "73129",
            "--output",
            str(run_dir),
        ],
    )
    assert initialized.exit_code == 0, initialized.output
    genome = load_inference_genome(run_dir / "inference_genome.json")
    run_manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    runtime_config = json.loads(
        (run_dir / "generated_runtime/runtime_config.json").read_text(encoding="utf-8")
    )
    assert genome.seed == 73129
    assert runtime_config["genome_hash"] == run_manifest["genome_hash"]
    assert runtime_config["generation_seed"] == 73129

    objectives = tmp_path / "objectives.json"
    objectives.write_text(
        json.dumps({"schema_version": "1.0.0", "primary": "cancellation_safe_goodput"}),
        encoding="utf-8",
    )
    synthesized = runner.invoke(
        app,
        [
            "genesis",
            "synthesize",
            "--run",
            str(run_dir),
            "--objectives",
            str(objectives),
            "--budget-usd",
            "100",
            "--seed",
            "73129",
        ],
    )
    assert synthesized.exit_code == 0, synthesized.output
    synthesis = json.loads((run_dir / "synthesis/result.json").read_text(encoding="utf-8"))
    assert synthesis["accepted_candidate_id"]
    assert synthesis["cross_layer_accepted"] is True
    assert synthesis["runtime_differential_passed"] is True
    request = json.loads((run_dir / "synthesis/request.json").read_text(encoding="utf-8"))
    assert request["budget_usd_ceiling"] == 100.0
    assert request["spent_usd"] == 0.0

    accepted_directory = run_dir / "candidates" / synthesis["accepted_candidate_id"]
    verified = runner.invoke(app, ["genesis", "verify", "--candidate", str(accepted_directory)])
    assert verified.exit_code == 0, verified.output

    rejected_directory = run_dir / "candidates" / synthesis["rejected_candidate_ids"][0]
    rejected = runner.invoke(app, ["genesis", "verify", "--candidate", str(rejected_directory)])
    assert rejected.exit_code == 1
    assert "emitted a committed token after cancellation" in rejected.output


def test_inspect_rejects_mismatched_contract(tmp_path: Path) -> None:
    unrelated = tmp_path / "contract.json"
    unrelated.write_text("{}", encoding="utf-8")
    result = runner.invoke(
        app,
        [
            "genesis",
            "inspect",
            "--reference",
            str(PACKAGE),
            "--contract",
            str(unrelated),
            "--output",
            str(tmp_path / "inspection"),
        ],
    )
    assert result.exit_code != 0
    assert "package manifest" in result.output
