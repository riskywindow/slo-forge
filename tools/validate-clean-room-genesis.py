#!/usr/bin/env python3
"""Validate and summarize retained Genesis clean-room evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _capsule_digest(capsule: dict[str, Any]) -> str:
    payload = {**capsule, "capsule_digest": None}
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_evidence_bundle(path: Path) -> int:
    required = {
        "artifacts/genesis/demo/GENESIS_DEMO_REPORT.json",
        "artifacts/genesis/demo/capsule.validation-context.json",
        "artifacts/genesis/wheel-capsule.validation-context.json",
        "artifacts/genesis/wheel-capsule-validation.json",
        "artifacts/synthbench/smoke/summary.json",
    }
    names: set[str] = set()
    with tarfile.open(path, mode="r:gz") as archive:
        for member in archive.getmembers():
            member_path = Path(member.name)
            if member_path.is_absolute() or ".." in member_path.parts:
                raise ValueError(f"retained evidence archive has an unsafe path: {member.name}")
            if member.issym() or member.islnk():
                raise ValueError(f"retained evidence archive contains a link: {member.name}")
            names.add(member.name.rstrip("/"))
    missing = sorted(required - names)
    if missing:
        raise ValueError(f"retained evidence archive lacks required files: {missing}")
    if not any(name.startswith("artifacts/genesis/demo/capsule/manifests/") for name in names):
        raise ValueError("retained evidence archive lacks a capsule manifest")
    if not any(name.startswith("artifacts/genesis/wheel-capsule/manifests/") for name in names):
        raise ValueError("retained evidence archive lacks the wheel-validated capsule manifest")
    return len(names)


def _inside(path: Path, root: Path, *, label: str) -> Path:
    resolved = path.resolve(strict=True)
    boundary = root.resolve(strict=True)
    if not resolved.is_relative_to(boundary):
        raise ValueError(f"{label} escapes the clean-room root: {resolved}")
    return resolved


def _object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def validate(
    *,
    root: Path,
    revision: str,
    source_tree: str,
    log: Path,
    wheel: Path,
    evidence_bundle: Path,
) -> dict[str, Any]:
    root = root.resolve(strict=True)
    if len(revision) != 40 or any(character not in "0123456789abcdef" for character in revision):
        raise ValueError("revision must be a lowercase SHA-1 object id")
    if len(source_tree) != 40 or any(
        character not in "0123456789abcdef" for character in source_tree
    ):
        raise ValueError("source tree must be a lowercase SHA-1 object id")
    if (root / ".sloforge-source-commit").read_text(encoding="utf-8").strip() != revision:
        raise ValueError("extracted source commit does not match the requested revision")
    if (root / ".sloforge-source-tree").read_text(encoding="utf-8").strip() != source_tree:
        raise ValueError("extracted source tree does not match the requested tree")

    final_report = root / "GENESIS_FINAL_REPORT.md"
    final_report_text = final_report.read_text(encoding="utf-8")
    if "435a04799a831c3d19fce18eb816b206d23778d7" not in final_report_text:
        raise ValueError("final report does not bind the recorded baseline commit")
    if "## Known limitations and unmet evaluation gates" not in final_report_text:
        raise ValueError("final report lacks the required limitation section")

    demo_root = (root / "artifacts/genesis/demo").resolve(strict=True)
    synthbench_root = (root / "artifacts/synthbench/smoke").resolve(strict=True)
    demo_report_path = demo_root / "GENESIS_DEMO_REPORT.json"
    synthbench_summary_path = synthbench_root / "summary.json"
    genesis = _object(demo_report_path)
    synthbench = _object(synthbench_summary_path)

    expected_bools = {
        "runtime_differential_passed": True,
        "cross_layer_accepted": True,
        "capsule_promotion_eligible": False,
        "capsule_local_evolution_eligible": True,
        "capsule_external_production_eligible": False,
        "evolution_promoted": True,
        "active_stream_preserved": True,
        "hardware_backed": False,
    }
    for field, expected in expected_bools.items():
        if genesis.get(field) is not expected:
            raise ValueError(f"Genesis demo field {field!r} is not {expected!r}")
    if genesis.get("kernel_speedup_claim_count") != 0:
        raise ValueError("clean CPU kernel evidence must not claim a speedup")
    if synthbench.get("valid_system_rate") != 1.0:
        raise ValueError("SynthBench clean-room valid-system rate is not 1.0")
    if synthbench.get("exact_request_rate") != 1.0:
        raise ValueError("SynthBench clean-room exact-request rate is not 1.0")

    if _inside(Path(str(genesis["output_directory"])), root, label="demo output") != demo_root:
        raise ValueError("demo output path is not the clean-room demo directory")
    if _inside(Path(str(genesis["report_path"])), root, label="demo report") != demo_report_path:
        raise ValueError("demo report path does not identify the checked report")
    _inside(Path(str(genesis["ui_bundle_path"])), demo_root, label="UI bundle")
    synthbench_report = _inside(
        Path(str(synthbench["report_path"])), synthbench_root, label="SynthBench report"
    )

    capsule_manifest = _inside(
        Path(str(genesis["capsule_path"])), demo_root, label="capsule manifest"
    )
    capsule_root = capsule_manifest.parents[1]
    manifest_paths = sorted((capsule_root / "manifests").glob("*.json"))
    if manifest_paths != [capsule_manifest]:
        raise ValueError("capsule directory must contain exactly the reported manifest")
    capsule = _object(capsule_manifest)
    capsule_digest = str(genesis["capsule_digest"])
    embedded_digest = capsule.get("capsule_digest")
    if not isinstance(embedded_digest, dict) or embedded_digest.get("algorithm") != "sha256":
        raise ValueError("capsule does not contain a SHA-256 identity digest")
    if (
        capsule_manifest.stem != capsule_digest
        or embedded_digest.get("value") != capsule_digest
        or _capsule_digest(capsule) != capsule_digest
    ):
        raise ValueError("capsule filename, reported digest, and canonical digest disagree")
    if capsule.get("benchmarks") != []:
        raise ValueError("local clean-room capsule unexpectedly contains promotion benchmarks")
    performance_claims = [
        claim
        for claim in capsule.get("claims", [])
        if isinstance(claim, dict) and claim.get("category") == "performance"
    ]
    if len(performance_claims) != 1:
        raise ValueError("capsule must contain exactly one scoped performance claim")
    if performance_claims[0].get("promotion_required") is not False:
        raise ValueError("non-improvement performance claim must not be promotion-required")
    if "no performance improvement is accepted" not in str(performance_claims[0].get("statement")):
        raise ValueError("capsule performance claim is not explicitly negative")

    artifacts = capsule.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise ValueError("capsule has no retained artifacts")
    artifact_by_id: dict[str, dict[str, Any]] = {}
    for artifact in artifacts:
        if not isinstance(artifact, dict) or not isinstance(artifact.get("artifact_id"), str):
            raise ValueError("capsule contains an invalid artifact record")
        artifact_path = Path(str(artifact.get("path", "")))
        if artifact_path.is_absolute() or ".." in artifact_path.parts:
            raise ValueError("capsule artifact path must be a contained relative path")
        retained = _inside(capsule_root / artifact_path, capsule_root, label="capsule artifact")
        digest = artifact.get("digest")
        if not isinstance(digest, dict) or digest.get("algorithm") != "sha256":
            raise ValueError("capsule artifact does not declare a SHA-256 digest")
        if _sha256(retained) != digest.get("value"):
            raise ValueError(f"capsule artifact digest mismatch: {artifact['artifact_id']}")
        if retained.stat().st_size != artifact.get("size_bytes"):
            raise ValueError(f"capsule artifact size mismatch: {artifact['artifact_id']}")
        artifact_by_id[artifact["artifact_id"]] = artifact

    simulation_ref = artifact_by_id.get("candidate-simulation")
    if simulation_ref is None:
        raise ValueError("capsule lacks candidate simulation evidence")
    simulation = _object(capsule_root / str(simulation_ref["path"]))
    if simulation.get("comparison_permitted") is not False:
        raise ValueError("synthetic candidate simulation permits a performance comparison")

    wheel_validation_path = root / "artifacts/genesis/wheel-capsule-validation.json"
    wheel_validation = _object(wheel_validation_path)
    required_wheel_results = {
        "integrity_valid": True,
        "contract_compatible": True,
        "evidence_complete": True,
        "local_evolution_eligible": True,
        "promotion_eligible": False,
        "external_production_eligible": False,
        "hardware_backed": False,
    }
    for field, expected in required_wheel_results.items():
        if wheel_validation.get(field) is not expected:
            raise ValueError(
                f"installed-wheel capsule validation field {field!r} is not {expected!r}"
            )
    if wheel_validation.get("issues") != []:
        raise ValueError("installed-wheel capsule validation retained issues")
    if wheel_validation.get("candidate_genome_hash") != genesis.get("accepted_genome_hash"):
        raise ValueError("installed-wheel capsule validates the wrong candidate genome")

    wheel_capsule_root = (root / "artifacts/genesis/wheel-capsule").resolve(strict=True)
    wheel_manifests = sorted((wheel_capsule_root / "manifests").glob("*.json"))
    if len(wheel_manifests) != 1:
        raise ValueError("wheel-validated capsule directory must contain exactly one manifest")
    wheel_capsule = _object(wheel_manifests[0])
    wheel_embedded_digest = wheel_capsule.get("capsule_digest")
    wheel_digest = _capsule_digest(wheel_capsule)
    if (
        not isinstance(wheel_embedded_digest, dict)
        or wheel_embedded_digest.get("algorithm") != "sha256"
        or wheel_embedded_digest.get("value") != wheel_digest
        or wheel_manifests[0].stem != wheel_digest
    ):
        raise ValueError("wheel-validated capsule canonical digest mismatch")
    validation_digest = wheel_validation.get("capsule_digest")
    if not isinstance(validation_digest, dict) or validation_digest.get("value") != wheel_digest:
        raise ValueError("installed-wheel validation reports the wrong capsule digest")
    wheel_artifacts = wheel_capsule.get("artifacts")
    if not isinstance(wheel_artifacts, list) or not wheel_artifacts:
        raise ValueError("wheel-validated capsule has no artifacts")
    for artifact in wheel_artifacts:
        if not isinstance(artifact, dict):
            raise ValueError("wheel-validated capsule contains an invalid artifact record")
        artifact_path = Path(str(artifact.get("path", "")))
        if artifact_path.is_absolute() or ".." in artifact_path.parts:
            raise ValueError("wheel-validated capsule artifact path is not contained")
        retained = _inside(
            wheel_capsule_root / artifact_path,
            wheel_capsule_root,
            label="wheel-validated capsule artifact",
        )
        digest = artifact.get("digest")
        if not isinstance(digest, dict) or digest.get("algorithm") != "sha256":
            raise ValueError("wheel-validated capsule artifact lacks a SHA-256 digest")
        if _sha256(retained) != digest.get("value") or retained.stat().st_size != artifact.get(
            "size_bytes"
        ):
            raise ValueError("wheel-validated capsule artifact integrity mismatch")

    log = _inside(log, root, label="clean-room log")
    wheel = _inside(wheel, root, label="built wheel")
    evidence_bundle = _inside(evidence_bundle, root, label="retained evidence bundle")
    evidence_bundle_member_count = _validate_evidence_bundle(evidence_bundle)
    return {
        "revision": revision,
        "source_tree": source_tree,
        "status": "passed",
        "log_sha256": _sha256(log),
        "wheel_sha256": _sha256(wheel),
        "evidence_bundle_sha256": _sha256(evidence_bundle),
        "evidence_bundle_member_count": evidence_bundle_member_count,
        "demo_report_sha256": _sha256(demo_report_path),
        "capsule_digest": capsule_digest,
        "wheel_capsule_digest": wheel_digest,
        "wheel_capsule_validation_sha256": _sha256(wheel_validation_path),
        "synthbench_summary_sha256": _sha256(synthbench_summary_path),
        "synthbench_report_sha256": _sha256(synthbench_report),
        "capsule_artifact_count": len(artifacts),
        "genesis_check": "passed",
        "genesis_demo": "passed",
        "synthbench_smoke": "passed",
        "package_build": "passed",
        "wheel_fresh_environment_smoke": "passed",
        "wheel_capsule_validation": "passed",
        "hardware_backed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--source-tree", required=True)
    parser.add_argument("--log", required=True, type=Path)
    parser.add_argument("--wheel", required=True, type=Path)
    parser.add_argument("--evidence-bundle", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    result = validate(
        root=arguments.root,
        revision=arguments.revision,
        source_tree=arguments.source_tree,
        log=arguments.log,
        wheel=arguments.wheel,
        evidence_bundle=arguments.evidence_bundle,
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
