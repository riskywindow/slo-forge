from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from sloforge.genesis.demo import run_genesis_demo
from sloforge.genesis.evaluation import run_genesis_evaluation


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
    assert not result.capsule_promotion_eligible
    assert result.capsule_local_evolution_eligible
    assert not result.capsule_external_production_eligible
    assert result.redteam_finding_count == result.redteam_replayed_count
    assert result.kernel_candidate_count == 2
    assert result.kernel_speedup_claim_count == 0
    assert result.kernel_measurement_scope == "isolated_operator_only_not_end_to_end_serving"
    assert not result.kernel_causal_attribution
    assert result.evolution_promoted
    assert result.active_stream_preserved
    assert result.physical_degradation_triggered
    assert result.hardware_backed is False
    report = json.loads(Path(result.report_path).read_text(encoding="utf-8"))
    assert report["accepted_genome_hash"] == result.accepted_genome_hash
    timeline = json.loads((tmp_path / "demo/evolution/timeline.json").read_text(encoding="utf-8"))
    assert timeline["source"] == "controller_audit_records"
    assert any(item["action"] == "promote" for item in timeline["events"])
    promoted = json.loads(
        (tmp_path / "demo/evolution/promoted-snapshot.json").read_text(encoding="utf-8")
    )
    challenger = promoted["challengers"][0]
    for stage in ("shadow", "canary"):
        artifact_path = tmp_path / f"demo/evolution/runtime-gates/{stage}/gate-evidence.json"
        payload = artifact_path.read_bytes()
        artifact = json.loads(payload)
        observation = challenger[f"{stage}_observation"]
        assert observation["evidence_digest"] == hashlib.sha256(payload).hexdigest()
        assert artifact["comparison"]["mismatches"] == []
        assert artifact["champion_sandbox"]["termination"] == "success"
        assert artifact["challenger_sandbox"]["termination"] == "success"
        assert artifact["trace_request_count"] == observation["sample_count"]


def test_demo_reset_rejects_symlink_output(tmp_path: Path) -> None:
    target = tmp_path / "existing-target"
    target.mkdir()
    marker = target / "preserve.txt"
    marker.write_text("preserve", encoding="utf-8")
    output = tmp_path / "linked-output"
    output.symlink_to(target, target_is_directory=True)
    with pytest.raises(ValueError, match="symlinked output"):
        run_genesis_demo(output, seed=73129, reset=True)
    assert marker.read_text(encoding="utf-8") == "preserve"


def test_demo_and_evaluation_reject_symlink_output_without_reset(tmp_path: Path) -> None:
    target = tmp_path / "empty-target"
    target.mkdir()
    output = tmp_path / "linked-output"
    output.symlink_to(target, target_is_directory=True)
    with pytest.raises(ValueError, match="symlinked output"):
        run_genesis_demo(output, seed=73129)
    with pytest.raises(ValueError, match="symlinked evaluation"):
        run_genesis_evaluation(output, seed=73129, count=2)
    assert not any(target.iterdir())
