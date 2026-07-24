from __future__ import annotations

import json
from pathlib import Path

import pytest

from sloforge.helix.characterization.workflow import analyze_cow, analyze_workload

VERTICAL = Path("artifacts/branchfabric/characterization/first-vertical-seed-41-rerun")


def test_workload_analysis_preserves_trace_evidence(tmp_path: Path) -> None:
    trace = VERTICAL / "branch-workload-trace-v1.jsonl"
    result = analyze_workload(
        trace=trace,
        output=tmp_path / "workload",
        seed=41,
        max_events=1000,
        replace=False,
    )
    artifact = tmp_path / "workload" / "workload-analysis.json"
    document = json.loads(artifact.read_text())
    assert result["artifact_reference"] == trace.as_posix()
    assert document["branch_fanout"]["p50"] == 4
    assert document["branch_lifetime_ns"]["sample_count"] == 4
    assert document["workload_provenance"] == ["SYNTHETIC"]


def test_cow_analysis_is_a_controlled_projection_not_a_distribution(tmp_path: Path) -> None:
    trace = VERTICAL / "state-operation-trace-v1.jsonl"
    result = analyze_cow(
        trace=trace,
        output=tmp_path / "cow.json",
        page_sizes=(4096, 65536),
        seed=41,
        max_events=1000,
        replace=False,
    )
    assert result["projection_count"] == 8 * 6 * 2
    assert result["distribution_claim"] is False
    recommendation = result["recommendation"]
    assert isinstance(recommendation, dict)
    assert recommendation["universal_page_size_supported"] is False
    with pytest.raises(FileExistsError):
        analyze_cow(
            trace=trace,
            output=tmp_path / "cow.json",
            page_sizes=(4096,),
            seed=41,
            max_events=1000,
            replace=False,
        )
