from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from typer.testing import CliRunner

from sloforge.cli.main import app
from sloforge.genesis.capsule import (
    RawBenchmarkSamples,
    ValidationContext,
    build_local_capsule,
    load_capsule,
    validate_capsule,
)

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "models/reference_tasks/hybrid_decoder"
runner = CliRunner()


def _accepted_candidate(tmp_path: Path) -> Path:
    inspection = tmp_path / "inspection"
    result = runner.invoke(
        app,
        [
            "genesis",
            "inspect",
            "--reference",
            str(PACKAGE),
            "--output",
            str(inspection),
            "--seed",
            "73129",
        ],
    )
    assert result.exit_code == 0, result.output
    hardware = tmp_path / "hardware.json"
    hardware.write_text(
        json.dumps({"schema_version": "1.0.0", "architecture": "cpu", "memory_bytes": 8 << 30}),
        encoding="utf-8",
    )
    run = tmp_path / "run"
    result = runner.invoke(
        app,
        [
            "genesis",
            "initialize",
            "--inspection",
            str(inspection),
            "--workload",
            str(PACKAGE / "search_samples.jsonl"),
            "--hardware",
            str(hardware),
            "--output",
            str(run),
            "--seed",
            "73129",
        ],
    )
    assert result.exit_code == 0, result.output
    result = runner.invoke(
        app,
        ["genesis", "synthesize", "--run", str(run), "--seed", "73129"],
    )
    assert result.exit_code == 0, result.output
    synthesis = json.loads((run / "synthesis/result.json").read_text(encoding="utf-8"))
    return run / "candidates" / synthesis["accepted_candidate_id"]


def test_local_capsule_builds_from_persisted_evidence_and_validates(tmp_path: Path) -> None:
    candidate = _accepted_candidate(tmp_path)
    output = tmp_path / "capsule"
    observed_at = datetime(2026, 8, 2, 12, 0, tzinfo=UTC)
    result = build_local_capsule(candidate, output, observed_at=observed_at)

    assert result.promotion_eligible
    assert result.hardware_backed is False
    capsule = load_capsule(Path(result.capsule_path))
    context = ValidationContext.model_validate_json(
        Path(result.context_path).read_bytes(), strict=True
    )
    report = validate_capsule(capsule, output, context)
    assert report.promotion_eligible
    assert report.issues == ()
    assert capsule.unverified_assumptions
    assert all(
        claim.scope.assumptions for claim in capsule.claims if claim.category.value == "performance"
    )
    benchmark = capsule.benchmarks[0]
    samples = RawBenchmarkSamples.model_validate_json(
        (
            output
            / next(
                item.path
                for item in capsule.artifacts
                if item.artifact_id == benchmark.raw_samples_artifact_id
            )
        ).read_bytes(),
        strict=True,
    )
    assert len(samples.samples) == 7
    assert benchmark.summary.unit == "simulated_milliseconds"

    cli_output = tmp_path / "capsule-cli"
    built = runner.invoke(
        app,
        [
            "genesis",
            "capsule",
            "build",
            "--candidate",
            str(candidate),
            "--output",
            str(cli_output),
            "--timestamp",
            "2026-08-02T12:00:00Z",
        ],
    )
    assert built.exit_code == 0, built.output
    validated = runner.invoke(
        app,
        ["genesis", "capsule", "validate", str(cli_output)],
    )
    assert validated.exit_code == 0, validated.output
    assert '"promotion_eligible": true' in validated.output
