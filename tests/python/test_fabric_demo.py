from __future__ import annotations

import json
from itertools import pairwise
from pathlib import Path

from typer.testing import CliRunner

from sloforge.cli.main import app
from sloforge.fabric.demo import FabricDemoManifest, _validate_report, _workload
from sloforge.fabric.simulation import FabricSimulationRequest
from sloforge.trace.format import load_trace
from sloforge.util import sha256_file

ROOT = Path(__file__).parents[2]
runner = CliRunner()


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
    assert manifest.live_gateway_requests == 12
    assert manifest.recovery_final_state == "COMPLETED"
    assert set(manifest.ground_truth_faults) == {
        "network_bandwidth_degradation",
        "rank_specific_gpu_slowdown",
    }
    metrics = (artifact_root / "metrics" / "degraded.prom").read_text(encoding="utf-8")
    for metric in (
        "sloforge_fabric_kv_transfer_p95_ms",
        "sloforge_fabric_collective_wait_p95_ms",
        "sloforge_fabric_network_throughput_bytes_per_second",
        "sloforge_fabric_rank_skew_ratio",
        "sloforge_fabric_prediction_absolute_error_ms",
        "sloforge_fabric_diagnosis_confidence_ratio",
        "sloforge_fabric_recovery_time_seconds",
    ):
        assert metric in metrics


def test_checked_demo_trace_and_autopsy_bundle_are_standalone_replayable(
    tmp_path: Path,
) -> None:
    artifact_root = ROOT / "artifacts" / "fabric-demo"
    records = load_trace(artifact_root / "mixed-bursty.jsonl")
    assert len(records) == 12
    assert {record.priority for record in records} == {0, 1, 2}

    for name in ("healthy", "degraded", "restored"):
        FabricSimulationRequest.model_validate_json(
            (artifact_root / "simulations" / f"{name}-request.json").read_text(encoding="utf-8"),
            strict=True,
        )
    metadata = json.loads(
        (artifact_root / "autopsy" / "replay-metadata.json").read_text(encoding="utf-8")
    )
    assert metadata["simulation_input_sha256"] == sha256_file(
        artifact_root / "simulations" / "degraded-request.json"
    )

    simulated = runner.invoke(
        app,
        [
            "fabric",
            "simulate",
            "--plan",
            str(artifact_root / "physical-plan.json"),
            "--topology",
            str(artifact_root / "topology.json"),
            "--fabric-profile",
            str(artifact_root / "fabric-profile.json"),
            "--trace",
            str(artifact_root / "mixed-bursty.jsonl"),
            "--output",
            str(tmp_path / "simulation"),
        ],
    )
    assert simulated.exit_code == 0, simulated.output or repr(simulated.exception)

    replayed = runner.invoke(
        app,
        [
            "autopsy",
            "replay",
            "--evidence",
            str(artifact_root / "autopsy"),
            "--counterfactual",
            str(artifact_root / "autopsy" / "scenarios.json"),
            "--output",
            str(tmp_path / "counterfactual-replay.json"),
        ],
    )
    assert replayed.exit_code == 0, replayed.output or repr(replayed.exception)
    assert (tmp_path / "counterfactual-replay.json").is_file()


def test_fabric_validate_fails_closed_on_material_prediction_error(tmp_path: Path) -> None:
    artifact_root = ROOT / "artifacts" / "fabric-demo"
    validation_dir = tmp_path / "validation"
    result = runner.invoke(
        app,
        [
            "fabric",
            "validate",
            "--plan",
            str(artifact_root / "physical-plan.json"),
            "--topology",
            str(artifact_root / "topology.json"),
            "--fabric-profile",
            str(artifact_root / "fabric-profile.json"),
            "--trace",
            str(artifact_root / "mixed-bursty.jsonl"),
            "--output",
            str(validation_dir),
        ],
    )
    assert result.exit_code == 1
    validation = json.loads((validation_dir / "validation.json").read_text(encoding="utf-8"))
    assert validation["valid"] is False
    assert validation["failure_reasons"]
