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
    Digest,
    RawBenchmarkSamples,
    ValidationContext,
    build_local_capsule,
    load_capsule,
    seal_capsule,
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

    assert result.local_evolution_eligible
    assert not result.promotion_eligible
    assert not result.external_production_eligible
    assert result.hardware_backed is False
    assert Path(result.context_path).parent == output.parent
    assert not Path(result.context_path).is_relative_to(output)
    capsule = load_capsule(Path(result.capsule_path))
    context = ValidationContext.model_validate_json(
        Path(result.context_path).read_bytes(), strict=True
    )
    report = validate_capsule(capsule, output, context)
    assert report.local_evolution_eligible
    assert not report.promotion_eligible
    assert not report.external_production_eligible
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
    hostile_runtime_policy = json.loads(policy_path.read_bytes())
    hostile_runtime_policy["instructions"][-1]["opcode"] = "dynamic_import"
    hostile_runtime_payload = json.dumps(
        hostile_runtime_policy, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    policy_path.write_bytes(hostile_runtime_payload)
    runtime_config_path = extracted / "runtime_config.json"
    runtime_config = json.loads(runtime_config_path.read_bytes())
    runtime_config["policy_bytecode_sha256"] = hashlib.sha256(hostile_runtime_payload).hexdigest()
    runtime_config_path.write_text(json.dumps(runtime_config), encoding="utf-8")
    with pytest.raises(ValueError, match="forbidden opcode"):
        load_generated_runtime(runtime_config_path, seed=73129)

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
    bundled_context = cli_output.with_name("capsule-cli.validation-context.json")
    assert bundled_context.is_file()
    inside_context = cli_output / "validation_context.json"
    inside_context.write_bytes(bundled_context.read_bytes())
    rejected_context = runner.invoke(
        app,
        [
            "genesis",
            "capsule",
            "validate",
            str(cli_output),
            "--context",
            str(inside_context),
            "--expected-digest",
            manifest.stem,
        ],
    )
    assert rejected_context.exit_code != 0
    validated = runner.invoke(
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
    assert validated.exit_code == 0, validated.output
    assert '"local_evolution_eligible": true' in validated.output
    assert '"promotion_eligible": false' in validated.output

    comparison = runner.invoke(
        app,
        [
            "genesis",
            "compare",
            "--champion",
            str(cli_output),
            "--challenger",
            str(cli_output),
            "--champion-context",
            str(bundled_context),
            "--challenger-context",
            str(bundled_context),
            "--champion-digest",
            manifest.stem,
            "--challenger-digest",
            manifest.stem,
        ],
    )
    assert comparison.exit_code == 0, comparison.output
    assert '"same_source_model": true' in comparison.output

    controller_state = tmp_path / "deployment-state.json"
    deployed = runner.invoke(
        app,
        [
            "genesis",
            "deploy",
            "--capsule",
            str(cli_output),
            "--context",
            str(bundled_context),
            "--expected-digest",
            manifest.stem,
            "--deployment",
            "capsule-cli-test",
            "--output",
            str(controller_state),
        ],
    )
    assert deployed.exit_code == 0, deployed.output
    evolved = runner.invoke(
        app,
        [
            "genesis",
            "evolve",
            "--deployment",
            "capsule-cli-test",
            "--trigger",
            "workload-drift",
            "--controller-state",
            str(controller_state),
            "--context",
            str(bundled_context),
            "--expected-digest",
            manifest.stem,
            "--budget-usd",
            "25",
        ],
    )
    assert evolved.exit_code == 0, evolved.output
    assert '"spent_usd": 0.0' in evolved.output
    denied_promotion = runner.invoke(
        app,
        [
            "genesis",
            "promote",
            "--capsule",
            str(cli_output),
            "--context",
            str(bundled_context),
            "--expected-digest",
            manifest.stem,
            "--controller-state",
            str(controller_state),
        ],
    )
    assert denied_promotion.exit_code != 0
    assert "proof-gated challenger" in denied_promotion.output

    suite = tmp_path / "benchmark-suite.json"
    suite.write_text(json.dumps({"repetitions": 2}), encoding="utf-8")
    benchmark = runner.invoke(
        app,
        [
            "genesis",
            "benchmark",
            "--candidate",
            str(candidate),
            "--suite",
            str(suite),
            "--output",
            str(tmp_path / "benchmark"),
            "--seed",
            "73129",
        ],
    )
    assert benchmark.exit_code == 0, benchmark.output
    assert '"hardware_backed": false' in benchmark.output
    raw_samples = [
        json.loads(line)
        for line in (tmp_path / "benchmark/raw_samples.jsonl").read_text().splitlines()
    ]
    assert [item["input_seed"] for item in raw_samples] == [73129, 73129]
    assert [item["sandbox_environment_seed"] for item in raw_samples] == [73129, 73130]

    design_path = candidate / "candidate_design.json"
    original_design = design_path.read_bytes()
    tampered_design = json.loads(original_design)
    tampered_design["candidate_id"] = "candidate-tampered"
    design_path.write_text(json.dumps(tampered_design), encoding="utf-8")
    rejected_benchmark = runner.invoke(
        app,
        [
            "genesis",
            "benchmark",
            "--candidate",
            str(candidate),
            "--suite",
            str(suite),
            "--output",
            str(tmp_path / "benchmark-tampered"),
        ],
    )
    design_path.write_bytes(original_design)
    assert rejected_benchmark.exit_code != 0
    assert "failed independent verification" in rejected_benchmark.output

    trace = tmp_path / "replay.jsonl"
    trace.write_text(
        json.dumps(
            {
                "request_id": "capsule-replay",
                "text": "hybrid",
                "maximum_new_tokens": 1,
                "seed": 73,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    replay_output = tmp_path / "replay-evidence.json"
    replay = runner.invoke(
        app,
        [
            "genesis",
            "replay",
            "--capsule",
            str(cli_output),
            "--trace",
            str(trace),
            "--context",
            str(bundled_context),
            "--expected-digest",
            manifest.stem,
            "--output",
            str(replay_output),
            "--seed",
            "73129",
        ],
    )
    assert replay.exit_code == 0, replay.output
    assert json.loads(replay_output.read_text(encoding="utf-8"))["passed"] is True

    oversized_trace = tmp_path / "replay-oversized.jsonl"
    oversized_trace.write_text(
        json.dumps(
            {
                "request_id": "capsule-replay-oversized",
                "text": "hybrid",
                "maximum_new_tokens": 257,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    rejected_replay = runner.invoke(
        app,
        [
            "genesis",
            "replay",
            "--capsule",
            str(cli_output),
            "--trace",
            str(oversized_trace),
            "--context",
            str(bundled_context),
            "--expected-digest",
            manifest.stem,
            "--output",
            str(tmp_path / "replay-rejected.json"),
        ],
    )
    assert rejected_replay.exit_code != 0


def test_hostile_policy_bytecode_is_rejected_at_build_and_capsule_validation(
    tmp_path: Path,
) -> None:
    candidate = _accepted_candidate(tmp_path)
    policy_path = candidate / "policy.bytecode.json"
    original = policy_path.read_bytes()
    hostile_document = json.loads(original)
    hostile_document["instructions"][-1]["opcode"] = "dynamic_import"
    hostile = json.dumps(
        hostile_document, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    policy_path.write_bytes(hostile)
    with pytest.raises(ValueError, match="forbidden opcode"):
        build_local_capsule(
            candidate,
            tmp_path / "rejected-capsule",
            observed_at=datetime(2026, 8, 2, 12, 0, tzinfo=UTC),
        )
    policy_path.write_bytes(original)

    output = tmp_path / "valid-capsule"
    result = build_local_capsule(
        candidate,
        output,
        observed_at=datetime(2026, 8, 2, 12, 0, tzinfo=UTC),
    )
    capsule = load_capsule(Path(result.capsule_path))
    context = ValidationContext.model_validate_json(
        Path(result.context_path).read_bytes(), strict=True
    )
    policy_artifact = next(
        item for item in capsule.artifacts if item.artifact_id == "generated-policy-bytecode"
    )
    capsule_policy_path = output / policy_artifact.path
    capsule_policy_path.chmod(0o644)
    capsule_policy_path.write_bytes(hostile)
    hostile_ref = policy_artifact.model_copy(
        update={
            "digest": Digest(value=hashlib.sha256(hostile).hexdigest()),
            "size_bytes": len(hostile),
        }
    )
    resealed = seal_capsule(
        capsule.model_copy(
            update={
                "capsule_digest": None,
                "artifacts": tuple(
                    hostile_ref if item.artifact_id == hostile_ref.artifact_id else item
                    for item in capsule.artifacts
                ),
            }
        )
    )
    assert resealed.capsule_digest is not None
    report = validate_capsule(
        resealed,
        output,
        context.model_copy(update={"expected_capsule_digest": resealed.capsule_digest}),
    )
    assert not report.promotion_eligible
    assert any(
        issue.path == "artifacts.generated-policy-bytecode" and "forbidden opcode" in issue.message
        for issue in report.issues
    )
