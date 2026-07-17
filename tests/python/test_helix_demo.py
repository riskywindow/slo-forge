from __future__ import annotations

import json
from pathlib import Path

from sloforge.helix.demo import run_cpu_demo


def test_cpu_demo_executes_failure_training_rejection_promotion_and_rollback(
    tmp_path: Path,
) -> None:
    output = tmp_path / "helix-demo"
    summary = run_cpu_demo(output, seed=31)
    assert summary["production_failure_action"] == "naive_parse"
    assert summary["capture_consistent"]
    assert summary["source_resumed"]
    assert summary["champion_success_rate"] < summary["corrected_candidate"]["success_rate"]
    assert summary["rejected_candidate"]["promotion_state"] == "rejected"
    assert summary["promotion"]["incompatible_session_pinned"]
    assert summary["promotion"]["tenant_id"] == "tenant-helix-demo"
    assert summary["promotion"]["capsule_validation_eligible"] is True
    assert summary["promotion"]["state_compatibility_class"] == "recomputation_assisted"
    assert len(summary["promotion"]["capsule_digest"]) == 64
    assert summary["promotion"]["challenger_route_before_rollback"] == "coding-agent@1"
    assert summary["promotion"]["route_after_rollback"] == "coding-agent@0"
    assert summary["promotion"]["final_state"] == "rolled_back"
    assert (output / "promotion" / "capsule.json").is_file()
    assert (output / "promotion" / "validation.json").is_file()
    assert summary["environment_cleanup"]["branch_workspace_bytes"] == 0
    assert summary["replay"]["joint"]["matched"]
    assert summary["replay"]["environment"]["matched"]
    assert not summary["replay"]["transcript"]["transcript_establishes_state_equivalence"]
    persisted = json.loads((output / "summary.json").read_text())
    assert persisted["lineage_hash"] == summary["lineage_hash"]
    assert (output / "report.md").is_file()
