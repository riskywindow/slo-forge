from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import tarfile
from pathlib import Path
from types import ModuleType

import pytest


def _validator() -> ModuleType:
    path = Path(__file__).resolve().parents[2] / "tools/validate-clean-room-genesis.py"
    specification = importlib.util.spec_from_file_location("clean_room_validator", path)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path) -> tuple[Path, str, str, Path, Path, Path]:
    root = tmp_path / "clean"
    demo = root / "artifacts/genesis/demo"
    capsule_root = demo / "capsule"
    manifests = capsule_root / "manifests"
    artifacts = capsule_root / "artifacts"
    synthbench = root / "artifacts/synthbench/smoke"
    manifests.mkdir(parents=True)
    artifacts.mkdir()
    (synthbench / "run").mkdir(parents=True)
    revision = "a" * 40
    source_tree = "b" * 40
    (root / ".sloforge-source-commit").write_text(revision + "\n", encoding="utf-8")
    (root / ".sloforge-source-tree").write_text(source_tree + "\n", encoding="utf-8")
    (root / "GENESIS_FINAL_REPORT.md").write_text(
        "435a04799a831c3d19fce18eb816b206d23778d7\n"
        "## Known limitations and unmet evaluation gates\n",
        encoding="utf-8",
    )
    simulation = artifacts / "simulation.json"
    simulation.write_text('{"comparison_permitted":false}\n', encoding="utf-8")
    capsule: dict[str, object] = {
        "artifacts": [
            {
                "artifact_id": "candidate-simulation",
                "path": "artifacts/simulation.json",
                "size_bytes": simulation.stat().st_size,
                "digest": {"algorithm": "sha256", "value": _digest(simulation)},
            }
        ],
        "benchmarks": [],
        "capsule_digest": None,
        "claims": [
            {
                "category": "performance",
                "promotion_required": False,
                "statement": "no performance improvement is accepted in this scope",
            }
        ],
    }
    payload = json.dumps(capsule, sort_keys=True, separators=(",", ":")).encode()
    capsule_digest = hashlib.sha256(payload).hexdigest()
    capsule["capsule_digest"] = {"algorithm": "sha256", "value": capsule_digest}
    encoded = json.dumps(capsule, sort_keys=True, separators=(",", ":")).encode()
    manifest = manifests / f"{capsule_digest}.json"
    manifest.write_bytes(encoded)
    ui = demo / "genesis-ui-bundle.json"
    ui.write_text("{}\n", encoding="utf-8")
    report = demo / "GENESIS_DEMO_REPORT.json"
    report.write_text(
        json.dumps(
            {
                "runtime_differential_passed": True,
                "cross_layer_accepted": True,
                "capsule_promotion_eligible": False,
                "capsule_local_evolution_eligible": True,
                "capsule_external_production_eligible": False,
                "evolution_promoted": True,
                "active_stream_preserved": True,
                "hardware_backed": False,
                "kernel_speedup_claim_count": 0,
                "capsule_path": str(manifest),
                "capsule_digest": capsule_digest,
                "accepted_genome_hash": "c" * 64,
                "output_directory": str(demo),
                "report_path": str(report),
                "ui_bundle_path": str(ui),
            }
        ),
        encoding="utf-8",
    )
    synthbench_report = synthbench / "run/report.json"
    synthbench_report.write_text("{}\n", encoding="utf-8")
    (synthbench / "summary.json").write_text(
        json.dumps(
            {
                "valid_system_rate": 1.0,
                "exact_request_rate": 1.0,
                "report_path": str(synthbench_report),
            }
        ),
        encoding="utf-8",
    )
    wheel_capsule = root / "artifacts/genesis/wheel-capsule"
    shutil.copytree(capsule_root, wheel_capsule)
    wheel_context = root / "artifacts/genesis/wheel-capsule.validation-context.json"
    wheel_context.write_text("{}\n", encoding="utf-8")
    wheel_validation = root / "artifacts/genesis/wheel-capsule-validation.json"
    wheel_validation.write_text(
        json.dumps(
            {
                "capsule_digest": {"algorithm": "sha256", "value": capsule_digest},
                "candidate_genome_hash": "c" * 64,
                "integrity_valid": True,
                "contract_compatible": True,
                "evidence_complete": True,
                "local_evolution_eligible": True,
                "promotion_eligible": False,
                "external_production_eligible": False,
                "hardware_backed": False,
                "issues": [],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    log = root / "clean.log"
    wheel = root / "dist/sloforge.whl"
    wheel.parent.mkdir()
    log.write_text("passed\n", encoding="utf-8")
    wheel.write_bytes(b"wheel")
    evidence_bundle = root / "artifacts/genesis/clean-room-evidence.tar.gz"
    with tarfile.open(evidence_bundle, mode="w:gz") as archive:
        archive.add(report, arcname="artifacts/genesis/demo/GENESIS_DEMO_REPORT.json")
        context = demo / "capsule.validation-context.json"
        context.write_text("{}\n", encoding="utf-8")
        archive.add(context, arcname="artifacts/genesis/demo/capsule.validation-context.json")
        archive.add(capsule_root, arcname="artifacts/genesis/demo/capsule")
        archive.add(
            wheel_context,
            arcname="artifacts/genesis/wheel-capsule.validation-context.json",
        )
        archive.add(
            wheel_validation,
            arcname="artifacts/genesis/wheel-capsule-validation.json",
        )
        archive.add(wheel_capsule, arcname="artifacts/genesis/wheel-capsule")
        archive.add(synthbench, arcname="artifacts/synthbench/smoke")
    return root, revision, source_tree, log, wheel, evidence_bundle


def test_clean_room_validator_binds_contained_retained_artifacts(tmp_path: Path) -> None:
    root, revision, source_tree, log, wheel, evidence_bundle = _fixture(tmp_path)
    result = _validator().validate(
        root=root,
        revision=revision,
        source_tree=source_tree,
        log=log,
        wheel=wheel,
        evidence_bundle=evidence_bundle,
    )
    assert result["status"] == "passed"
    assert result["capsule_artifact_count"] == 1
    assert result["wheel_sha256"] == _digest(wheel)


def test_clean_room_validator_rejects_capsule_path_outside_archive(tmp_path: Path) -> None:
    root, revision, source_tree, log, wheel, evidence_bundle = _fixture(tmp_path)
    report_path = root / "artifacts/genesis/demo/GENESIS_DEMO_REPORT.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    outside = tmp_path / "host-capsule.json"
    outside.write_text("{}\n", encoding="utf-8")
    report["capsule_path"] = str(outside)
    report_path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="escapes the clean-room root"):
        _validator().validate(
            root=root,
            revision=revision,
            source_tree=source_tree,
            log=log,
            wheel=wheel,
            evidence_bundle=evidence_bundle,
        )


def test_clean_room_validator_rejects_tampered_retained_artifact(tmp_path: Path) -> None:
    root, revision, source_tree, log, wheel, evidence_bundle = _fixture(tmp_path)
    simulation = root / "artifacts/genesis/demo/capsule/artifacts/simulation.json"
    simulation.write_text('{"comparison_permitted":true}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="artifact digest mismatch"):
        _validator().validate(
            root=root,
            revision=revision,
            source_tree=source_tree,
            log=log,
            wheel=wheel,
            evidence_bundle=evidence_bundle,
        )


def test_clean_room_validator_rejects_failed_installed_wheel_validation(
    tmp_path: Path,
) -> None:
    root, revision, source_tree, log, wheel, evidence_bundle = _fixture(tmp_path)
    validation_path = root / "artifacts/genesis/wheel-capsule-validation.json"
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    validation["evidence_complete"] = False
    validation["issues"] = [{"code": "evidence_stale"}]
    validation_path.write_text(json.dumps(validation), encoding="utf-8")
    with pytest.raises(ValueError, match="evidence_complete"):
        _validator().validate(
            root=root,
            revision=revision,
            source_tree=source_tree,
            log=log,
            wheel=wheel,
            evidence_bundle=evidence_bundle,
        )
