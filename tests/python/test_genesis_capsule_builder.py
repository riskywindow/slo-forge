from __future__ import annotations

import hashlib
import json
import stat
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import pytest
from typer.testing import CliRunner

from sloforge.cli.main import app
from sloforge.genesis.capsule import (
    RawBenchmarkSamples,
    ValidationContext,
    build_local_capsule,
    load_capsule,
    validate_capsule,
)
from sloforge.genesis.runtime import load_generated_runtime

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
        json.dumps(
            {
                "schema_version": "1.0.0",
                "architecture": "cpu",
                "memory_bytes": 8 << 30,
                "measured_fingerprint": hashlib.sha256(b"capsule-builder-cpu").hexdigest(),
            }
        ),
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
    assert benchmark.randomized_run_order
    by_id = {item.artifact_id: item for item in capsule.artifacts}
    bundle_path = output / by_id["generated-runtime"].path
    extracted = tmp_path / "runtime-bundle"
    with zipfile.ZipFile(bundle_path) as archive:
        archive.extractall(extracted)
        names = set(archive.namelist())
    assert {
        "runtime.py",
        "runtime_config.json",
        "correctness_harness.py",
        "policy.slo",
        "policy.bytecode.json",
        "bundle_manifest.json",
        "reference_package/reference.py",
        "reference_package/reference_package.json",
    }.issubset(names)
    bundle_manifest = json.loads((extracted / "bundle_manifest.json").read_text())
    for name, digest in bundle_manifest["entries"].items():
        assert hashlib.sha256((extracted / name).read_bytes()).hexdigest() == digest
    runtime = load_generated_runtime(extracted / "runtime_config.json", seed=73129)
    runtime.start()
    try:
        assert runtime.health()["policy"] == "deadline_cancel_batch"
        handle = runtime.submit_text(
            request_id="capsule-policy-execution",
            text="hybrid",
            maximum_new_tokens=1,
            seed=73,
            timeout_seconds=2.0,
        )
        assert list(handle.events(2.0))[-1].kind.value == "completed"
    finally:
        runtime.shutdown()
    policy_path = extracted / "policy.bytecode.json"
    policy_path.write_bytes(policy_path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="policy bytecode digest"):
        load_generated_runtime(extracted / "runtime_config.json", seed=73129)

    quality = json.loads((output / by_id["quality-evidence"].path).read_text())
    assert quality["observed"] == 1.0
    assert quality["case_count"] == len(quality["cases"]) > 0
    assert all(case["exact_match"] for case in quality["cases"])
    resource = json.loads((output / by_id["resource-evidence"].path).read_text())
    assert resource["method"] == "runtime-config-genome-state-contract-upper-bound"
    assert resource["runtime_queue_depth"] == 32
    assert (
        resource["champion_challenger_coexistence_bytes"]
        == 2 * resource["single_runtime_peak_bytes"]
    )
    for artifact in capsule.artifacts:
        assert not (stat.S_IMODE((output / artifact.path).stat().st_mode) & stat.S_IWUSR)

    symlink_target = tmp_path / "capsule-symlink-target"
    symlink_target.mkdir()
    symlink_output = tmp_path / "capsule-symlink"
    symlink_output.symlink_to(symlink_target, target_is_directory=True)
    with pytest.raises(ValueError, match="must not be a symlink"):
        build_local_capsule(candidate, symlink_output, observed_at=observed_at)

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
    manifest = next((cli_output / "manifests").glob("*.json"))
    bundled_context = cli_output / "validation_context.json"
    rejected_context = runner.invoke(
        app,
        [
            "genesis",
            "capsule",
            "validate",
            str(cli_output),
            "--context",
            str(bundled_context),
            "--expected-digest",
            manifest.stem,
        ],
    )
    assert rejected_context.exit_code != 0
    trusted_context = tmp_path / "operator-trusted-validation-context.json"
    trusted_context.write_bytes(bundled_context.read_bytes())
    validated = runner.invoke(
        app,
        [
            "genesis",
            "capsule",
            "validate",
            str(cli_output),
            "--context",
            str(trusted_context),
            "--expected-digest",
            manifest.stem,
        ],
    )
    assert validated.exit_code == 0, validated.output
    assert '"promotion_eligible": true' in validated.output
