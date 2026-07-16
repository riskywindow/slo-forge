from __future__ import annotations

import json
import shutil
from hashlib import sha256
from pathlib import Path

from typer.testing import CliRunner

from sloforge.cli.main import app
from sloforge.continuum.ir import load_capsule
from sloforge.continuum.storage import FileContentStore
from sloforge.util import git_commit

runner = CliRunner()


def test_continuum_cli_exposes_scriptable_surface() -> None:
    result = runner.invoke(app, ["continuum", "--help"])
    assert result.exit_code == 0, result.output
    for command in (
        "benchmark",
        "capsule",
        "checkpoint",
        "clone",
        "compatibility",
        "fork",
        "migrate",
        "migration",
        "pause",
        "report",
        "resume",
        "runtime",
        "state",
    ):
        assert command in result.output


def test_checkpoint_pause_resume_and_clone_cli_workflow(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "checkpoint"
    checkpointed = runner.invoke(
        app,
        [
            "continuum",
            "checkpoint",
            "--session",
            "operations-cli-session",
            "--seed",
            "83",
            "--generated-tokens",
            "6",
            "--output",
            str(checkpoint_dir),
        ],
    )
    assert checkpointed.exit_code == 0, checkpointed.output
    checkpoint_result = json.loads(
        (checkpoint_dir / "operation-result.json").read_text(encoding="utf-8")
    )
    assert checkpoint_result["durable_lease_retained"] is False
    checkpoint_capsule = load_capsule(checkpoint_dir / "capsule.json")
    assert checkpoint_capsule.identity.git_commit == git_commit(Path(__file__).resolve().parents[2])

    resumed_path = tmp_path / "resume.json"
    resumed = runner.invoke(
        app,
        [
            "continuum",
            "resume",
            "--checkpoint",
            str(checkpoint_dir / "checkpoint.json"),
            "--store",
            str(checkpoint_dir / "store"),
            "--seed",
            "83",
            "--generated-tokens",
            "2",
            "--output",
            str(resumed_path),
        ],
    )
    assert resumed.exit_code == 0, resumed.output
    resume_result = json.loads(resumed_path.read_text(encoding="utf-8"))
    assert resume_result["phase"] == "COMPLETED"
    assert resume_result["accepted_token_indices"] == [6, 7]
    assert resume_result["destination_owner_epoch"] == 2
    assert resume_result["durable_lease_retained"] is False

    clone_dir = tmp_path / "clone"
    cloned = runner.invoke(
        app,
        [
            "continuum",
            "clone",
            "--checkpoint",
            str(checkpoint_dir / "checkpoint.json"),
            "--store",
            str(checkpoint_dir / "store"),
            "--session",
            "operations-cli-clone",
            "--seed",
            "84",
            "--output",
            str(clone_dir),
        ],
    )
    assert cloned.exit_code == 0, cloned.output
    clone_capsule = load_capsule(clone_dir / "capsule.json")
    assert clone_capsule.identity.session_id == "operations-cli-clone"
    assert clone_capsule.identity.git_commit == git_commit(Path(__file__).resolve().parents[2])
    validated_clone = runner.invoke(
        app,
        ["continuum", "capsule", "validate", str(clone_dir / "capsule.json")],
    )
    assert validated_clone.exit_code == 0, validated_clone.output

    pause_dir = tmp_path / "pause"
    paused = runner.invoke(
        app,
        [
            "continuum",
            "pause",
            "--session",
            "operations-cli-pause",
            "--seed",
            "85",
            "--output",
            str(pause_dir),
        ],
    )
    assert paused.exit_code == 0, paused.output
    pause_result = json.loads((pause_dir / "operation-result.json").read_text(encoding="utf-8"))
    assert pause_result["operation"] == "pause"
    assert pause_result["transaction_terminal"] is True


def test_reference_runtime_capture_and_capsule_validation(tmp_path: Path) -> None:
    inspection = tmp_path / "runtime.json"
    inspected = runner.invoke(
        app,
        [
            "continuum",
            "runtime",
            "inspect",
            "--runtime",
            "reference-a",
            "--output",
            str(inspection),
        ],
    )
    assert inspected.exit_code == 0, inspected.output
    runtime = json.loads(inspection.read_text(encoding="utf-8"))
    assert runtime["runtime"]["runtime_name"] == "continuum-reference-token-major"
    assert runtime["exercised"] is True

    output = tmp_path / "capsule"
    captured = runner.invoke(
        app,
        [
            "continuum",
            "state",
            "capture",
            "--runtime",
            str(inspection),
            "--session",
            "cli-session",
            "--seed",
            "23",
            "--generated-tokens",
            "6",
            "--output",
            str(output),
        ],
    )
    assert captured.exit_code == 0, captured.output
    capsule = load_capsule(output / "capsule.json")
    assert capsule.identity.session_id == "cli-session"
    assert capsule.logical_state.client_delivery.last_gateway_committed_token_index == 5
    capture_result = json.loads((output / "capture-result.json").read_text(encoding="utf-8"))
    assert capture_result["transaction_id"] is None
    assert capture_result["durable_lease_retained"] is False
    assert not (output / "coordinator.db").exists()

    validated = runner.invoke(
        app,
        ["continuum", "capsule", "validate", str(output / "capsule.json")],
    )
    assert validated.exit_code == 0, validated.output
    assert json.loads(validated.output)["valid"] is True

    inspection_output = tmp_path / "state-inspection.json"
    state = runner.invoke(
        app,
        [
            "continuum",
            "state",
            "inspect",
            "--capsule",
            str(output / "capsule.json"),
            "--output",
            str(inspection_output),
        ],
    )
    assert state.exit_code == 0, state.output
    state_document = json.loads(inspection_output.read_text(encoding="utf-8"))
    assert "state/attention-kv" in state_document["logical_components"]
    assert state_document["non_portable_runtime_state"]

    conversion_output = tmp_path / "conversion-plan.json"
    compiled = runner.invoke(
        app,
        [
            "continuum",
            "conversion",
            "compile",
            "--tokens",
            "6",
            "--output",
            str(conversion_output),
        ],
    )
    assert compiled.exit_code == 0, compiled.output
    conversion = json.loads(conversion_output.read_text(encoding="utf-8"))
    assert conversion["dag"]["operations"][-1]["code"] == "validate"


def test_capsule_validation_fails_closed_on_missing_or_corrupt_external_state(
    tmp_path: Path,
) -> None:
    output = tmp_path / "capture"
    captured = runner.invoke(
        app,
        [
            "continuum",
            "state",
            "capture",
            "--session",
            "external-state-session",
            "--seed",
            "71",
            "--output",
            str(output),
        ],
    )
    assert captured.exit_code == 0, captured.output
    empty_store = tmp_path / "empty-store"
    empty_store.mkdir()
    missing = runner.invoke(
        app,
        [
            "continuum",
            "capsule",
            "validate",
            str(output / "capsule.json"),
            "--store",
            str(empty_store),
        ],
    )
    assert missing.exit_code != 0

    capture_result = json.loads((output / "capture-result.json").read_text(encoding="utf-8"))
    with FileContentStore(output / "store") as store:
        manifest = store.manifest(
            "local-continuum",
            capture_result["store_manifest_id"],
        )
        reference = manifest.chunks[0]
        store.corrupt_for_test(reference.tenant_id, reference.digest, b"corrupt")
    corrupt = runner.invoke(
        app,
        ["continuum", "capsule", "validate", str(output / "capsule.json")],
    )
    assert corrupt.exit_code != 0


def test_flagship_cli_migrate_verify_compatibility_and_fork(tmp_path: Path) -> None:
    output = tmp_path / "migration"
    migrated = runner.invoke(
        app,
        [
            "continuum",
            "migrate",
            "--mode",
            "pre-copy",
            "--seed",
            "317",
            "--output",
            str(output),
        ],
    )
    assert migrated.exit_code == 0, migrated.output
    artifact = output / "flagship.json"
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert artifact.is_file()
    assert manifest["synthetic_hardware"] is True

    verification = tmp_path / "verification.json"
    verified = runner.invoke(
        app,
        [
            "continuum",
            "migration",
            "verify",
            "--artifact",
            str(artifact),
            "--output",
            str(verification),
        ],
    )
    assert verified.exit_code == 0, verified.output
    evidence = json.loads(verification.read_text(encoding="utf-8"))
    assert evidence["valid"] is True
    assert evidence["accepted_token_indices"] == list(range(37))

    compatibility = tmp_path / "compatibility.json"
    checked = runner.invoke(
        app,
        [
            "continuum",
            "compatibility",
            "--artifact",
            str(artifact),
            "--output",
            str(compatibility),
        ],
    )
    assert checked.exit_code == 0, checked.output
    decision = json.loads(compatibility.read_text(encoding="utf-8"))
    assert decision["direct_reuse"]["compatibility_class"] == "incompatible"
    assert decision["recomputation_assisted"]["compatibility_class"] == ("recomputation_assisted")

    fork = tmp_path / "fork.json"
    forked = runner.invoke(
        app,
        [
            "continuum",
            "fork",
            "--artifact",
            str(artifact),
            "--output",
            str(fork),
        ],
    )
    assert forked.exit_code == 0, forked.output
    branches = json.loads(fork.read_text(encoding="utf-8"))
    assert len(branches["branches"]) == 2
    assert branches["checkpoint_bytes_deduplicated"] > 0

    tampered_document = json.loads(artifact.read_text(encoding="utf-8"))
    tampered_document["invariants"]["no_gateway_gap"] = False
    tampered = tmp_path / "tampered-flagship.json"
    tampered.write_text(json.dumps(tampered_document), encoding="utf-8")
    rejected = runner.invoke(
        app,
        [
            "continuum",
            "migration",
            "verify",
            "--artifact",
            str(tampered),
        ],
    )
    assert rejected.exit_code != 0
    rejected_compatibility = runner.invoke(
        app,
        [
            "continuum",
            "compatibility",
            "--artifact",
            str(tampered),
            "--output",
            str(tmp_path / "tampered-compatibility.json"),
        ],
    )
    assert rejected_compatibility.exit_code != 0

    token_tampered_document = json.loads(artifact.read_text(encoding="utf-8"))
    token_tampered_document["accepted_token_indices"][-1] -= 1
    token_tampered = tmp_path / "token-tampered-flagship.json"
    token_tampered.write_text(json.dumps(token_tampered_document), encoding="utf-8")
    token_rejected = runner.invoke(
        app,
        [
            "continuum",
            "migration",
            "verify",
            "--artifact",
            str(token_tampered),
        ],
    )
    assert token_rejected.exit_code != 0

    unsealed_dir = tmp_path / "unsealed"
    unsealed_dir.mkdir()
    shutil.copy2(artifact, unsealed_dir / "flagship.json")
    unsealed = runner.invoke(
        app,
        [
            "continuum",
            "migration",
            "verify",
            "--artifact",
            str(unsealed_dir / "flagship.json"),
        ],
    )
    assert unsealed.exit_code != 0

    bad_manifest_dir = tmp_path / "bad-manifest"
    shutil.copytree(output, bad_manifest_dir)
    bad_manifest_path = bad_manifest_dir / "manifest.json"
    bad_manifest = json.loads(bad_manifest_path.read_text(encoding="utf-8"))
    bad_manifest["sha256"] = "0" * 64
    bad_manifest_path.write_text(json.dumps(bad_manifest), encoding="utf-8")
    bad_seal = runner.invoke(
        app,
        [
            "continuum",
            "migration",
            "verify",
            "--artifact",
            str(bad_manifest_dir / "flagship.json"),
        ],
    )
    assert bad_seal.exit_code != 0

    conversion_dir = tmp_path / "conversion-tamper"
    conversion_dir.mkdir()
    conversion_document = json.loads(artifact.read_text(encoding="utf-8"))
    conversion_document["successful_migration"]["live_conversion_evidence"][
        "canonical_attention_match"
    ] = False
    conversion_artifact = conversion_dir / "flagship.json"
    conversion_artifact.write_text(json.dumps(conversion_document), encoding="utf-8")
    conversion_manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    conversion_manifest["sha256"] = sha256(conversion_artifact.read_bytes()).hexdigest()
    (conversion_dir / "manifest.json").write_text(
        json.dumps(conversion_manifest),
        encoding="utf-8",
    )
    bad_conversion = runner.invoke(
        app,
        [
            "continuum",
            "migration",
            "verify",
            "--artifact",
            str(conversion_artifact),
        ],
    )
    assert bad_conversion.exit_code != 0
