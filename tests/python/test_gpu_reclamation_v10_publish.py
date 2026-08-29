from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import sloforge.helix.characterization.gpu_reclamation_v10_publish as publication
from sloforge.helix.characterization.gpu_reclamation_v10_publish import (
    PLOT_SPECS,
    publish_v10_success,
)
from sloforge.helix.characterization.gpu_reclamation_v10_validity import (
    fail_closed_v10_scientific_validity,
)

NS = 1_000_000_000


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n")


def _success_validity() -> dict[str, object]:
    timestamps = {
        "CONTROL_STABLE": 5 * NS,
        "LOAD_SPIKE_BEGIN": 5 * NS,
        "GPU0_OVERLOAD_CONFIRMED": 10 * NS,
        "RECLAIM_TRIGGER": 10 * NS,
        "ROLLOUT_ADMISSION_STOP": 10 * NS,
        "BRANCH_QUIESCE_BEGIN": 10 * NS,
        "BRANCH_QUIESCE_END": 10 * NS + 20_000_000,
        "STATE_CAPTURE_BEGIN": 10 * NS + 20_000_000,
        "STATE_CAPTURE_END": 10 * NS + 50_000_000,
        "STATE_TRANSFORM_BEGIN": 10 * NS + 50_000_000,
        "STATE_TRANSFORM_END": 10 * NS + 100_000_000,
        "D2H_BEGIN": 10 * NS + 100_000_000,
        "D2H_END": 10 * NS + 200_000_000,
        "INTEGRITY_BEGIN": 10 * NS + 200_000_000,
        "INTEGRITY_END": 10 * NS + 300_000_000,
        "STATE_PUBLISH": 10 * NS + 300_000_000,
        "GPU1_STATE_RELEASE_BEGIN": 10 * NS + 300_000_000,
        "GPU1_STATE_RELEASE_END": 10 * NS + 350_000_000,
        "GPU1_HBM_RECLAIM_CONFIRMED": 10 * NS + 400_000_000,
        "GPU1_SERVING_ENABLE": 11 * NS,
        "GPU1_FIRST_USEFUL_SERVING_REQUEST": 12 * NS,
        "SERVING_QUEUE_DRAIN_BEGIN": 12 * NS,
        "SERVING_SLO_RECOVERY_BEGIN": 12 * NS,
        "SERVING_QUEUE_DRAIN_END": 20 * NS,
        "SERVING_SLO_RESTORED": 20 * NS,
        "SERVING_SLO_STABILITY_BEGIN": 20 * NS,
        "SERVING_SLO_STABILITY_END": 25 * NS,
        "TWO_GPU_SERVICE_STABLE": 25 * NS,
        "RESTORE_LOAD_REDUCED": 26 * NS,
        "GPU1_SERVING_DRAINED": 26 * NS + 500_000_000,
        "RESTORE_ELIGIBLE": 27 * NS,
        "RESTORE_TRIGGER": 27 * NS,
        "H2D_BEGIN": 27 * NS,
        "H2D_END": 27 * NS + 100_000_000,
        "STATE_IMPORT_BEGIN": 27 * NS + 100_000_000,
        "DESTINATION_NATIVE_WRITE_BEGIN": 27 * NS + 200_000_000,
        "DESTINATION_NATIVE_WRITE_END": 28 * NS,
        "STATE_VALIDATE_BEGIN": 28 * NS,
        "STATE_VALIDATE_END": 29 * NS,
        "STATE_IMPORT_END": 29 * NS,
        "BRANCH_RESUME_BEGIN": 29 * NS,
        "FIRST_RESUMED_TOKEN": 30 * NS,
        "ALL_BRANCHES_RESUMED": 31 * NS,
        "ROLLOUT_CONTINUATION_COMPLETE": 32 * NS,
    }
    stage_names = (
        ("h2d", 27 * NS, 27 * NS + 100_000_000),
        ("import", 27 * NS + 100_000_000, 29 * NS),
        ("destination-native-write", 27 * NS + 200_000_000, 28 * NS),
        ("destination-validation", 28 * NS, 29 * NS),
        ("resume", 29 * NS, 31 * NS),
    )
    branches = {f"branch-{index}": 100 + index for index in range(8)}
    return {
        **{field: True for field in publication.REQUIRED_VALIDITY_GATES},
        "scientifically_valid": True,
        "timeline": {
            "events": [
                {"phase": phase, "monotonic_timestamp_ns": timestamp}
                for phase, timestamp in timestamps.items()
            ]
        },
        "gpu0_overload": {
            "control": {
                "interval": {"start_ns": 0, "end_ns": 5 * NS},
                "completed_requests": 50,
                "ttft": {"p95": 100_000_000},
            },
            "overload": {"completed_requests": 50, "ttft": {"p95": 2_500_000_000}},
        },
        "two_gpu_excess_capacity": {
            "offered_rate_per_second": 12.0,
            "completed_rate_per_second": 16.0,
        },
        "queue_drain": {"passed": True},
        "slo_stability": {"passed": True, "duration_ns": 5 * NS},
        "state_correctness": {
            "logical_state_bytes": 1_000,
            "preserved_state_bytes": 1_000,
            "source_blocks_released": 8,
        },
        "branch_resume": {
            "expected_first_tokens": branches,
            "observed_first_tokens": branches,
            "continuation_token_counts": {key: 8 for key in branches},
        },
        "movement_accounting": {
            "logical_state_bytes": 1_000,
            "full_physical_touch_bytes": 58_000,
            "external_movement_bytes": 3_000,
            "avoidable_movement_bytes": 31_000,
            "critical_path_movement_bytes": 58_000,
            "accounting_duplicate_bytes": 0,
        },
        "gpu0_restore_activity": {
            "stage_activity": [
                {
                    "stage": name,
                    "interval": {"start_ns": start, "end_ns": end},
                    "completion_count": 1,
                    "emitted_tokens": 64,
                    "ttft_sample_count": 1,
                    "eligible_for_interference": name != "h2d",
                    "sufficient_sample": name != "h2d",
                }
                for name, start, end in stage_names
            ]
        },
        "budget": {
            "conservative_gpu_seconds_before": 100.0,
            "invocation_gpu_seconds": 60.0,
            "conservative_gpu_seconds_after": 160.0,
            "hard_ceiling_gpu_seconds": 10_800.0,
            "ledger_updated": True,
        },
        "cleanup": {"zero_active_tasks": True},
    }


def _raw_inputs(repository_root: Path, run_root: Path, experiment_root: Path) -> None:
    offered = []
    observed = []
    for index in range(320):
        arrival = index * 100_000_000
        phase = (
            "control"
            if arrival < 5 * NS
            else "gpu0-overload"
            if arrival < 12 * NS
            else "two-gpu-recovery"
            if arrival < 27 * NS
            else "restore-interference"
        )
        device = "gpu1" if phase == "two-gpu-recovery" and index % 2 else "gpu0"
        request_id = f"request-{index:04d}"
        offered.append(
            {
                "request_id": request_id,
                "phase": phase,
                "device": device,
                "scheduled_arrival_ns": arrival,
                "offered_ns": arrival,
            }
        )
        first = arrival + 30_000_000
        observed.append(
            {
                "request_id": request_id,
                "device": device,
                "service_start_ns": arrival + 20_000_000,
                "token_timestamps_ns": [first + token * 1_000_000 for token in range(64)],
                "completed_ns": arrival + 100_000_000,
            }
        )
    _write(
        run_root / "serving/result.json",
        {
            "status": "succeeded",
            "global_offered_requests": offered,
            "requests": observed,
        },
    )
    _write(
        run_root / "rollout/result.json",
        {
            "status": "succeeded",
            "temporary_serving": {"requests": []},
        },
    )
    _write(
        run_root / "analysis/serving-recovery.json",
        {"schema_version": "sloforge.branchfabric.experiment-004-v10-serving-recovery/v1"},
    )
    movement = {
        "schema_version": "sloforge.branchfabric.experiment-004-v10-movement-accounting/v1",
        "raw_state_movement_report": {
            "edges": [
                {
                    "source": "source_gpu_native_paged",
                    "destination": "pinned_host_transport",
                    "edge_kind": "link_transfer",
                    "bytes": 1_000,
                }
            ],
            "passes": [
                {
                    "record_id": "0001:capture-d2h:shared-root",
                    "state_segment": "shared-root",
                    "operation": "d2h",
                    "source_memory": "gpu_transport_buffer",
                    "destination_memory": "pinned_host_transport",
                    "bytes_read": 1_000,
                    "bytes_written": 1_000,
                    "transfer_bytes": 1_000,
                    "temporary_allocation_bytes": 0,
                    "required_unavoidable": True,
                    "start_ns": 10 * NS,
                    "end_ns": 10 * NS + 100_000_000,
                }
            ],
        },
    }
    _write(run_root / "analysis/movement-accounting.json", movement)
    _write(
        run_root / "effective-config.json",
        {
            "attempt_id": "v10-success-fixture",
            "serving_methodology": "v10-global-capacity",
            "model": "Qwen/Qwen2.5-7B-Instruct",
            "model_revision": "fixture-revision",
            "runtime": "vllm",
            "runtime_version": "0.23.0",
            "prefix_length": 16384,
            "fanout": 8,
            "suffix_length": 256,
            "lambda_1_rps": 10.0,
            "lambda_2_rps": 20.0,
            "lambda_spike_rps": 12.0,
            "gpu0_control_request_rate_per_second": 8.0,
            "gpu0_restore_request_rate_per_second": 8.0,
        },
    )
    _write(run_root / "topology/topology-commands.json", [{"command": "fixture"}])
    _write(
        run_root / "analysis/gpu-phase-ledger.json",
        {"schema_version": "fixture", "actual_invocation_gpu_seconds": 60.0},
    )
    _write(experiment_root / "v10-authorization.json", {"code_commit": "a" * 40})
    _write(experiment_root / "v10-phase-budget.json", {"required_A100_seconds": 938.0})
    samples = []
    for sequence in range(33):
        timestamp = sequence * NS
        samples.append(
            {
                "sequence": sequence,
                "sample_trigger_monotonic_ns": timestamp,
                "gpu_samples": [
                    {
                        "memory_used_bytes": 20_000_000_000 - sequence * 10_000,
                        "pcie_rx_bytes_per_second": 1_000,
                        "pcie_tx_bytes_per_second": 2_000,
                    }
                ],
                "host_sample": {
                    "host_memory_total_bytes": 100_000,
                    "host_memory_available_bytes": 50_000,
                    "system_cpu_user_ns": sequence * 100,
                    "system_cpu_system_ns": sequence * 50,
                    "system_cpu_idle_ns": sequence * 850,
                },
            }
        )
    telemetry = {"sampling": {"samples": samples}}
    _write(run_root / "serving/telemetry/resource-sampling.json", telemetry)
    _write(run_root / "rollout/telemetry/resource-sampling.json", telemetry)
    _write(
        experiment_root / "gpu-hours.json",
        {
            "consumed_additional_gpu_seconds": 160.0,
            "intervals": [
                {
                    "accounted_wall_seconds": 30.0,
                    "gpu_count": 2,
                    "gpu_price_per_hour_usd": 2.5,
                }
            ],
        },
    )
    _write(experiment_root / "manifest.json", {"historical": True})


def test_refuses_invalid_canonical_scientific_validity_without_writing(tmp_path: Path) -> None:
    repository = tmp_path / "repo"
    run_root = repository / "run"
    experiment_root = repository / "artifacts/experiment-004"
    repository.mkdir()
    _write(
        run_root / "analysis/scientific-validity.json",
        fail_closed_v10_scientific_validity("not run").model_dump(mode="json"),
    )

    with pytest.raises(ValueError, match="refuses non-scientifically-valid"):
        publish_v10_success(
            repository_root=repository,
            run_root=run_root,
            experiment_root=experiment_root,
        )

    assert not (experiment_root / "plots").exists()
    assert not experiment_root.exists()
    assert not (repository / "reports").exists()


def test_success_gate_rejects_a_prerequisite_changed_after_finalization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    validity_path = tmp_path / "scientific-validity.json"
    _write(validity_path, {"fixture": True})
    validity = _success_validity()
    kinds = (
        "authorization",
        "phase-budget",
        "engine-readiness",
        "sanity-12rps",
        "sanity-15rps",
        "budget-settlement",
        "modal-cleanup",
    )
    references = []
    for kind in kinds:
        evidence = tmp_path / f"{kind}.json"
        evidence.write_text(kind)
        digest = hashlib.sha256(evidence.read_bytes()).hexdigest()
        references.append(f"{kind}:{digest}:{evidence}")
    validity["prerequisites"] = {"evidence_references": references}
    monkeypatch.setattr(publication, "_strict_validity", lambda _payload: validity)
    (tmp_path / "authorization.json").write_text("mutated")

    with pytest.raises(ValueError, match="changed after finalization"):
        publication._require_success(validity_path)


def test_success_publication_writes_ten_hashed_plots_reports_and_future_plans(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = tmp_path / "repo"
    run_root = repository / "run"
    experiment_root = repository / "artifacts/experiment-004"
    repository.mkdir()
    _raw_inputs(repository, run_root, experiment_root)
    validity = _success_validity()
    _write(run_root / "analysis/scientific-validity.json", validity)
    monkeypatch.setattr(
        "sloforge.helix.characterization.gpu_reclamation_v10_publish._require_success",
        lambda _path: validity,
    )

    manifest = publish_v10_success(
        repository_root=repository,
        run_root=run_root,
        experiment_root=experiment_root,
    )

    assert manifest["scientifically_valid"] is True
    assert manifest["optimized_preservation_executed"] is False
    assert len(PLOT_SPECS) == 10
    for plot_id, _title, _kind in PLOT_SPECS:
        plot_path = experiment_root / "plots" / f"{plot_id}.json"
        svg_path = plot_path.with_suffix(".svg")
        payload = json.loads(plot_path.read_text())
        assert svg_path.read_text().startswith("<svg")
        assert len(payload["raw_provenance"]) == 13
        assert all(len(row["artifact_sha256"]) == 64 for row in payload["raw_provenance"])
    report_path = repository / "reports/branchfabric-gpu-validation-experiment-004-v10-final.json"
    report = json.loads(report_path.read_text())
    assert report["scientifically_valid"] is True
    assert report["movement"]["full_physical_touch_amplification"] == 58.0
    assert report["serving_interference"]["minimum_p95_samples"] == 20
    assert report["serving_interference"]["causal_interference_claimed"] is False
    assert all(
        "token_rate_interference" in phase for phase in report["serving_interference"]["phases"]
    )
    d2h = next(
        phase
        for phase in report["serving_interference"]["phases"]
        if phase["phase"] == "GPU1_PRESERVE_D2H"
    )
    assert d2h["throughput_interference_reportable"] is False
    assert d2h["throughput_interference"] is None
    assert report["branch_correctness"]["exact_first_token_matches"] == 8
    assert json.loads((experiment_root / "plots/09-state-pass-graph.json").read_text())["data"][
        "timing_semantics"
    ].endswith("deduplicated")
    decomposition = json.loads(
        (experiment_root / "plots/10-state-movement-amplification-decomposition.json").read_text()
    )
    assert {row["label"] for row in decomposition["data"]["bars"]} == {
        "external_movement",
        "avoidable_lower_bound",
        "critical_path",
        "full_physical_touch",
    }
    assert (
        repository / "reports/branchfabric-gpu-validation-experiment-004-v10-final.md"
    ).is_file()
    optimization = (
        repository / "docs/branchfabric/EXPERIMENT_004_V11_OPTIMIZATION_PLAN.md"
    ).read_text()
    campaign = (
        repository / "docs/branchfabric/EXPERIMENT_004_BASELINE_CAMPAIGN_PLAN.md"
    ).read_text()
    assert "NOT IMPLEMENTED OR EXECUTED" in optimization
    assert "KILL_AND_RECOMPUTE" in campaign
    assert "NO CAMPAIGN ARM WAS EXECUTED" in campaign
    generated = manifest["generated_files_excluding_manifest"]
    assert len(generated) == 31
    for item in generated:
        path = repository / item["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"]
    previous = manifest["superseded_manifest"]
    assert previous["artifact_sha256"]
    archived = repository / previous["artifact_reference"]
    assert hashlib.sha256(archived.read_bytes()).hexdigest() == previous["artifact_sha256"]
    baseline = json.loads(
        (experiment_root / "baseline/branchfabric-exp004-naive-baseline-v10.json").read_text()
    )
    assert baseline["status"] == "FROZEN"
    assert baseline["recommended_git_tag"] == "branchfabric-exp004-naive-baseline-v10"
