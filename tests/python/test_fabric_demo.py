from __future__ import annotations

from itertools import pairwise
from pathlib import Path

from sloforge.fabric.demo import FabricDemoManifest, _validate_report, _workload

ROOT = Path(__file__).parents[2]


def test_demo_workload_is_bursty_mixed_and_prioritized() -> None:
    records, workload = _workload(41)

    assert workload.request_count == len(records) == 12
    assert {record.request_class for record in records} == {
        "interactive",
        "long_context",
        "batch",
    }
    assert {record.priority for record in records} == {"high", "normal", "low"}
    gaps = [right.arrival_us - left.arrival_us for left, right in pairwise(records)]
    assert max(gaps) > 100 * min(gaps)


def test_checked_in_demo_report_is_derived_from_hashed_artifacts() -> None:
    artifact_root = ROOT / "artifacts" / "fabric-demo"
    manifest_path = artifact_root / "manifest.json"
    report_path = ROOT / "reports" / "fabric-demo" / "fabric-demo.md"

    manifest = FabricDemoManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    _validate_report(manifest_path, report_path, artifact_root)

    assert manifest.synthetic_hardware is True
    assert manifest.healthy_slo_attained is True
    assert manifest.degraded_slo_attained is False
    assert manifest.restored_slo_attained is True
    assert manifest.degraded.p95_ttft_ms > manifest.healthy.p95_ttft_ms
    assert manifest.restored.p95_ttft_ms <= manifest.p95_ttft_slo_ms
    assert manifest.counterfactuals_evaluated >= 7
    assert manifest.recovery_final_state == "COMPLETED"
    assert set(manifest.ground_truth_faults) == {
        "network_bandwidth_degradation",
        "rank_specific_gpu_slowdown",
    }
