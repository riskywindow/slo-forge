from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from sloforge.genesis.demo import run_genesis_demo


def test_cpu_genesis_demo_is_artifact_backed_and_cross_layer(tmp_path: Path) -> None:
    result = run_genesis_demo(
        tmp_path / "demo",
        seed=73129,
        observed_at=datetime(2026, 8, 2, 12, 0, tzinfo=UTC),
    )

    assert result.runtime_differential_passed
    assert result.cross_layer_accepted
    assert result.rejected_candidate_ids
    assert result.minimized_counterexample_ids
    assert result.learned_constraint_ids
    assert result.capsule_promotion_eligible
    assert result.redteam_finding_count == result.redteam_replayed_count
    assert result.kernel_candidate_count == 2
    assert result.kernel_speedup_claim_count == 0
    assert result.evolution_promoted
    assert result.active_stream_preserved
    assert result.physical_degradation_triggered
    assert result.hardware_backed is False
    report = json.loads(Path(result.report_path).read_text(encoding="utf-8"))
    assert report["accepted_genome_hash"] == result.accepted_genome_hash
    timeline = json.loads((tmp_path / "demo/evolution/timeline.json").read_text(encoding="utf-8"))
    assert timeline["source"] == "controller_audit_records"
    assert any(item["action"] == "promote" for item in timeline["events"])
