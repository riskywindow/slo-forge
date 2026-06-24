from __future__ import annotations

from pathlib import Path

from sloforge.lineage.demo import run_lineage_transfer_demo


def test_lineage_demo_transfers_and_invalidates_without_performance_claim(tmp_path: Path) -> None:
    report = run_lineage_transfer_demo(tmp_path / "lineage-demo", seed=73129)

    cases = report["cases"]
    assert cases["empty_lineage"]["lineage_seed_count"] == 0
    assert cases["unrelated_lineage"]["lineage_seed_count"] == 0
    assert report["related_seed_retrieved"] is True
    assert report["affected_evidence_count"] == 2
    assert report["stale_seed_suppressed_after_invalidation"] is True
    assert report["performance_hypothesis_evaluated"] is False
