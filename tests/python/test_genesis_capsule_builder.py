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
    ValidationContext,
    build_local_capsule,
    load_capsule,
    seal_capsule,
    validate_capsule,
)
from sloforge.genesis.capsule.builder import _trusted_temporary_output
from sloforge.genesis.runtime import load_generated_runtime

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "models/reference_tasks/hybrid_decoder"
runner = CliRunner()


def test_trusted_temporary_output_canonicalizes_only_the_orchestrator_root(
    tmp_path: Path,
) -> None:
    actual = tmp_path / "private-temporary"
    actual.mkdir()
    alias = tmp_path / "public-temporary"
    alias.symlink_to(actual, target_is_directory=True)

    output = _trusted_temporary_output(str(alias))

    assert output == actual.resolve(strict=True) / "artifacts"
    assert not output.exists()


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
    assert capsule.benchmarks == ()
    performance_claim = next(
        claim for claim in capsule.claims if claim.category.value == "performance"
    )
    assert not performance_claim.promotion_required
    assert "no performance improvement is accepted" in performance_claim.statement
    simulation_ref = next(
        item for item in capsule.artifacts if item.artifact_id == "candidate-simulation"
    )
    simulation = json.loads((output / simulation_ref.path).read_text(encoding="utf-8"))
    assert simulation["candidate_genome_hash"] == capsule.identity.candidate_genome_hash.value
    assert simulation["comparison_permitted"] is False
    by_id = {item.artifact_id: item for item in capsule.artifacts}
    bundle_path = output / by_id["generated-runtime"].path
    extracted = tmp_path / "runtime-bundle"
    with zipfile.ZipFile(bundle_path) as archive:
        archive.extractall(extracted)
        names = set(archive.namelist())
    assert {
        "runtime.py",
        "runtime_config.json",
        "tested_runtime_config.json",
        "correctness_harness.py",
        "deployment_manifest.json",
        "policy.slo",
        "policy.bytecode.json",
        "bundle_manifest.json",
        "reference_package/reference.py",
        "reference_package/reference_package.json",
    }.issubset(names)
    bundle_manifest = json.loads((extracted / "bundle_manifest.json").read_text())
    assert bundle_manifest["direct_launch_supported"] is False
    assert bundle_manifest["trusted_launcher"] == "sloforge.genesis.sandbox.execute_sandboxed"
    assert (
        bundle_manifest["tested_runtime_config_sha256"]
        == hashlib.sha256((extracted / "tested_runtime_config.json").read_bytes()).hexdigest()
    )
    for name, digest in bundle_manifest["entries"].items():
        assert hashlib.sha256((extracted / name).read_bytes()).hexdigest() == digest
    runtime = load_generated_runtime(
        extracted / "runtime_config.json", seed=73129, allow_untrusted_in_process=True
    )
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
        load_generated_runtime(
            extracted / "runtime_config.json", seed=73129, allow_untrusted_in_process=True
        )
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
        load_generated_runtime(runtime_config_path, seed=73129, allow_untrusted_in_process=True)

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
    promoted = runner.invoke(
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
    assert promoted.exit_code == 0, promoted.output
    assert json.loads(promoted.output)["phase"] == "promoted"
    assert (controller_state.parent / "runtime-gates/shadow/gate-evidence.json").is_file()
    assert (controller_state.parent / "runtime-gates/canary/gate-evidence.json").is_file()

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


def test_capsule_builder_replays_trusted_transformation_lowering(tmp_path: Path) -> None:
    candidate = _accepted_candidate(tmp_path)
    candidate_document = json.loads((candidate / "candidate.json").read_text(encoding="utf-8"))
    transformation_id = candidate_document["transformation_ids"][0]
    transformation_path = candidate / "transformations" / f"{transformation_id}.json"
    forged = json.loads(transformation_path.read_text(encoding="utf-8"))
    forged["affected_regions"] = ["kernel"]
    transformation_path.write_text(
        json.dumps(forged, sort_keys=True, separators=(",", ":")), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="trusted lowering derivation"):
        build_local_capsule(
            candidate,
            tmp_path / "forged-transformation-capsule",
            observed_at=datetime(2026, 8, 2, 12, 0, tzinfo=UTC),
        )


def test_capsule_builder_rejects_forged_runtime_harness_and_stale_identity(
    tmp_path: Path,
) -> None:
    candidate = _accepted_candidate(tmp_path)
    observed_at = datetime(2026, 8, 2, 12, 0, tzinfo=UTC)
    runtime = candidate / "generated_runtime"
    harness_path = runtime / "correctness_harness.py"
    config_path = runtime / "runtime_config.json"
    differential_path = candidate / "evidence/runtime-differential-result.json"
    manifest_path = runtime / "candidate_runtime_manifest.json"
    simulation_path = candidate / "evidence/simulation-result.json"
    original = {
        path: path.read_bytes()
        for path in (harness_path, config_path, differential_path, manifest_path, simulation_path)
    }

    def rebind_runtime_artifact(name: str) -> None:
        differential = json.loads(differential_path.read_bytes())
        differential["runtime_artifact_hashes"][name] = hashlib.sha256(
            (runtime / name).read_bytes()
        ).hexdigest()
        differential_path.write_text(
            json.dumps(differential, sort_keys=True, separators=(",", ":")), encoding="utf-8"
        )
        manifest = json.loads(manifest_path.read_bytes())
        manifest["artifacts"] = differential["runtime_artifact_hashes"]
        manifest_path.write_text(
            json.dumps(manifest, sort_keys=True, separators=(",", ":")), encoding="utf-8"
        )
        simulation = json.loads(simulation_path.read_bytes())
        simulation["runtime_manifest_sha256"] = hashlib.sha256(
            manifest_path.read_bytes()
        ).hexdigest()
        simulation_path.write_text(
            json.dumps(simulation, sort_keys=True, separators=(",", ":")), encoding="utf-8"
        )

    forged_harness = """import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--samples", type=Path, required=True)
parser.add_argument("--seed")
parser.add_argument("--timeout-seconds")
args = parser.parse_args()
cases = []
for line_number, line in enumerate(args.samples.read_text().splitlines(), 1):
    sample = json.loads(line)
    expected = sample["expected_tokens"]
    cases.append({"line": line_number, "request_seed": sample["seed"], "expected": expected,
                  "observed": expected, "exact_match": True})
print(json.dumps({"cases": cases, "failures": [], "passed": True}))
"""
    harness_path.write_text(forged_harness, encoding="utf-8")
    rebind_runtime_artifact("correctness_harness.py")
    with pytest.raises(ValueError, match="trusted generated template"):
        build_local_capsule(
            candidate,
            tmp_path / "forged-harness-capsule",
            observed_at=observed_at,
        )

    for path, payload in original.items():
        path.write_bytes(payload)
    config = json.loads(config_path.read_bytes())
    config["genome_hash"] = "0" * 64
    config_path.write_text(
        json.dumps(config, sort_keys=True, separators=(",", ":")), encoding="utf-8"
    )
    rebind_runtime_artifact("runtime_config.json")
    with pytest.raises(ValueError, match="runtime configuration identity mismatch"):
        build_local_capsule(
            candidate,
            tmp_path / "stale-runtime-identity-capsule",
            observed_at=observed_at,
        )


def test_capsule_builder_recomputes_proofs_and_final_corpus_oracle(tmp_path: Path) -> None:
    candidate = _accepted_candidate(tmp_path)
    observed_at = datetime(2026, 8, 2, 12, 0, tzinfo=UTC)

    modelcheck_path = candidate / "evidence/modelcheck-result.json"
    original_modelcheck = modelcheck_path.read_bytes()
    forged_modelcheck = json.loads(original_modelcheck)
    forged_modelcheck["bounds"]["max_depth"] = 999_999
    forged_modelcheck["transition_count"] = 1
    forged_modelcheck["invariants"] = ["forged invariant"]
    modelcheck_path.write_text(json.dumps(forged_modelcheck), encoding="utf-8")
    with pytest.raises(ValueError, match="independently recomputed bounded result"):
        build_local_capsule(
            candidate,
            tmp_path / "forged-modelcheck-capsule",
            observed_at=observed_at,
        )
    modelcheck_path.write_bytes(original_modelcheck)

    differential_path = candidate / "evidence/runtime-differential-result.json"
    forged_differential = json.loads(differential_path.read_bytes())
    forged_differential["cases"][0]["expected"] = [999]
    forged_differential["cases"][0]["observed"] = [999]
    forged_differential["cases"][0]["exact_match"] = True
    differential_path.write_text(json.dumps(forged_differential), encoding="utf-8")
    with pytest.raises(ValueError, match="independent sandbox replay"):
        build_local_capsule(
            candidate,
            tmp_path / "forged-differential-capsule",
            observed_at=observed_at,
        )
