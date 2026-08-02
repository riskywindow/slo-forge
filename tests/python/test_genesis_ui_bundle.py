from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from sloforge.genesis.demo import run_genesis_demo
from sloforge.genesis.reports import GenesisUiBundleError, export_genesis_ui_bundle


def _json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_flagship_emits_ui_bundle_from_exact_persisted_artifacts(tmp_path: Path) -> None:
    root = tmp_path / "demo"
    result = run_genesis_demo(
        root,
        seed=73129,
        observed_at=datetime(2026, 8, 2, 12, 0, tzinfo=UTC),
    )

    bundle_path = Path(result.ui_bundle_path)
    bundle = _json(bundle_path)
    report = _json(Path(result.report_path))
    assert bundle["artifact_type"] == "sloforge.genesis.ui-bundle/v1"
    assert bundle["summary"] == report

    accepted = root / f"run/candidates/{result.accepted_candidate_id}"
    assert bundle["genome"] == _json(accepted / "inference_genome.json")
    assert bundle["capsule"] == _json(Path(result.capsule_path))
    assert bundle["evolution"] == _json(root / "evolution/degraded-snapshot.json")
    assert bundle["lineage"] == _json(root / "lineage/report.json")

    candidates = bundle["candidates"]
    assert isinstance(candidates, list)
    candidate_by_id = {
        item["candidate"]["candidate_id"]: item
        for item in candidates
        if isinstance(item, dict) and isinstance(item.get("candidate"), dict)
    }
    for candidate_directory in sorted((root / "run/candidates").iterdir()):
        candidate = _json(candidate_directory / "candidate.json")
        candidate_id = candidate["candidate_id"]
        assert isinstance(candidate_id, str)
        assert candidate_by_id[candidate_id]["candidate"] == candidate
        assert candidate_by_id[candidate_id]["design"] == _json(
            candidate_directory / "candidate_design.json"
        )

    capsule = bundle["capsule"]
    assert isinstance(capsule, dict)
    benchmarks = capsule["benchmarks"]
    artifacts = capsule["artifacts"]
    assert benchmarks == []
    assert isinstance(artifacts, list)
    assert bundle["benchmark_definition"] is None
    assert bundle["candidate_samples"] is None
    assert bundle["baseline_samples"] is None
    simulation = bundle["performance_simulation"]
    assert isinstance(simulation, dict)
    assert simulation["comparison_permitted"] is False
    assert simulation["hardware_backed"] is False
    assert simulation["candidate_id"] == result.accepted_candidate_id
    performance_claims = [
        item
        for item in capsule["claims"]
        if isinstance(item, dict) and item.get("category") == "performance"
    ]
    assert len(performance_claims) == 1
    assert performance_claims[0]["promotion_required"] is False
    assert "no performance improvement is accepted" in performance_claims[0]["statement"]
    assert report["capsule_local_evolution_eligible"] is True
    assert report["capsule_external_production_eligible"] is False

    with pytest.raises(GenesisUiBundleError, match="refusing to overwrite"):
        export_genesis_ui_bundle(root)


def test_ui_bundle_rejects_cross_file_identity_tampering(tmp_path: Path) -> None:
    root = tmp_path / "demo"
    result = run_genesis_demo(root, seed=73129)
    candidate_path = root / f"run/candidates/{result.accepted_candidate_id}/candidate.json"
    candidate = _json(candidate_path)
    genome_hash = candidate["genome_hash"]
    assert isinstance(genome_hash, dict)
    genome_hash["value"] = "0" * 64
    candidate_path.write_text(
        json.dumps(candidate, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    second_bundle = root / "tampered-ui-bundle.json"
    report_path = Path(result.report_path)
    report = _json(report_path)
    report["ui_bundle_path"] = str(second_bundle.resolve())
    report_path.write_text(
        json.dumps(report, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(GenesisUiBundleError, match="accepted candidate genome hash differs"):
        export_genesis_ui_bundle(root, second_bundle)
    assert not second_bundle.exists()
