from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_BENCHMARK_ROOT = Path(__file__).parents[2] / "benchmarks" / "branchfabric"
sys.path.insert(0, str(_BENCHMARK_ROOT))
_ANALYZER_PATH = _BENCHMARK_ROOT / "analyze_gpu_validation.py"
_ANALYZER_SPEC = importlib.util.spec_from_file_location(
    "sloforge_branchfabric_gpu_validation_analysis", _ANALYZER_PATH
)
assert _ANALYZER_SPEC is not None and _ANALYZER_SPEC.loader is not None
_ANALYZER = importlib.util.module_from_spec(_ANALYZER_SPEC)
sys.modules[_ANALYZER_SPEC.name] = _ANALYZER
_ANALYZER_SPEC.loader.exec_module(_ANALYZER)

AnalysisError = _ANALYZER.AnalysisError


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n")


def _sample(observed_ns: int, memory_mib: int, *, elapsed_ms: float | None = None) -> dict:
    sample = {
        "observed_at_monotonic_ns": observed_ns,
        "observed_at_utc": "2026-08-11T00:00:00Z",
        "gpu": {
            "uuid": "GPU-test",
            "name": "NVIDIA A100 80GB PCIe",
            "memory_total_mib": 81_920,
            "memory_used_mib": memory_mib,
        },
        "compute_processes": [],
    }
    if elapsed_ms is not None:
        sample["elapsed_since_child_exit_ms"] = elapsed_ms
    return sample


def _materialize_clean_lifecycle(root: Path, *, attempt_id: str = "smoke-v6") -> tuple[dict, dict]:
    lifecycle = root / "lifecycle"
    baseline_samples = [
        _sample(1_000_000_000, 4),
        _sample(1_200_000_000, 4),
        _sample(1_400_000_000, 5),
    ]
    post_samples = [
        _sample(10_000_000_000, 8, elapsed_ms=1_000.0),
        _sample(11_000_000_000, 7, elapsed_ms=2_000.0),
        _sample(12_000_000_000, 6, elapsed_ms=3_000.0),
    ]
    controller = {
        "schema_version": "sloforge.branchfabric.gpu-cow-controller-manifest/v1",
        "attempt_id": attempt_id,
        "child_pid": 4242,
        "child_pgid": 4242,
        "child_return_code": 0,
        "parent_cuda_clean": True,
        "forced_gpu_process_kill_required": False,
        "termination_actions": [],
        "status": "succeeded",
    }
    child = {
        "schema_version": "sloforge.branchfabric.gpu-cow-child-manifest/v1",
        "attempt_id": attempt_id,
        "status": "succeeded",
        "child_pid": 4242,
        "child_pgid": 4242,
        "semantic_invariants": {
            "shared_root_physical_blocks_identical": True,
            "fork_allocated_zero_prefix_bytes": True,
        },
        "final_runtime_assigned_kv_bytes": 0,
        "multiprocessing_children_before_exit": [],
    }
    process_tree = {
        "schema_version": "sloforge.branchfabric.process-tree/v1",
        "observed_at_monotonic_ns": 500_000_000,
        "processes": [{"pid": 1, "ppid": 0, "pgid": 1}],
    }
    during = {
        "schema_version": "sloforge.branchfabric.process-tree-series/v1",
        "child_pid": 4242,
        "child_pgid": 4242,
        "tracked_owned_processes": [{"pid": 4242, "start_token": "test-start"}],
        "samples": [],
    }
    baseline = {
        "schema_version": "sloforge.branchfabric.hbm-baseline/v1",
        "samples": baseline_samples,
        "baseline_samples_mib": [4, 4, 5],
        "baseline_median_mib": 4,
        "cleanup_threshold_mib": 1028,
        "allowance_mib": 1024,
        "gpu_uuid": "GPU-test",
    }
    hbm_postflight = {
        "schema_version": "sloforge.branchfabric.hbm-postflight/v1",
        "child_live_samples": [_sample(4_500_000_000, 14_213)],
        "post_child_exit_samples": post_samples,
        "sample_interval_seconds": 1,
        "maximum_wait_seconds": 30,
    }
    cleanup = {
        "schema_version": "sloforge.branchfabric.gpu-runtime-cleanup-gate/v1",
        "cleanup_gate": "PASS",
        "GPU_RUNTIME_LIFECYCLE_CLEAN": True,
        "process_hbm_recovery": "verified",
        "baseline_median_mib": 4,
        "cleanup_threshold_mib": 1028,
        "allowance_mib": 1024,
        "stable_sample_evaluation": {
            "passed": True,
            "required_consecutive_samples": 3,
            "winning_sample_start_index": 0,
            "winning_sample_end_index": 2,
            "maximum_consecutive_qualifying_samples": 3,
        },
        "time_to_hbm_recovery_ms": 1_000.0,
        "time_to_stable_confirmation_ms": 3_000.0,
        "child_return_code": 0,
        "child_semantics_valid": True,
        "no_owned_descendant_remained": True,
        "no_compute_process_remained_in_winning_samples": True,
        "forced_gpu_process_kill_required": False,
        "termination_actions": [],
        "timed_out": False,
    }
    audits = {
        "schema_version": "sloforge.branchfabric.forbidden-gpu-import-audits/v1",
        "cuda_clean": True,
        "audits": [
            {"stage": stage, "cuda_clean": True}
            for stage in (
                "controller_entry_before_baseline",
                "immediately_before_child_launch",
                "after_child_exit_and_hbm_postflight",
            )
        ],
    }
    for name, payload in {
        "controller-manifest.json": controller,
        "child-manifest.json": child,
        "process-tree-before.json": process_tree,
        "process-tree-during.json": during,
        "process-tree-after.json": {**process_tree, "observed_at_monotonic_ns": 12_100_000_000},
        "hbm-baseline.json": baseline,
        "hbm-postflight.json": hbm_postflight,
        "cleanup-gate.json": cleanup,
        "forbidden-import-audit.json": audits,
    }.items():
        _write_json(lifecycle / name, payload)
    event_names = [
        ("child_process_started", 2_000_000_000),
        ("model_load_started", 3_000_000_000),
        ("kv_pool_allocated", 4_000_000_000),
        ("benchmark_started", 5_000_000_000),
        ("branch_fork_start", 6_000_000_000),
        ("decode_start", 7_000_000_000),
        ("branch_teardown_and_vllm_shutdown_completed", 8_000_000_000),
        ("child_manifest_flushed_exit_imminent", 8_900_000_000),
    ]
    (lifecycle / "child-events.jsonl").write_text(
        "".join(
            json.dumps(
                {
                    "schema_version": "sloforge.branchfabric.gpu-worker-event/v1",
                    "event": event,
                    "observed_at_monotonic_ns": observed_ns,
                },
                sort_keys=True,
            )
            + "\n"
            for event, observed_ns in event_names
        )
    )
    completion = {
        "GPU_RUNTIME_LIFECYCLE_CLEAN": True,
        "child_pid": 4242,
        "child_pgid": 4242,
        "child_return_code": 0,
    }
    wrapper_postflight = {
        "schema_version": "sloforge.branchfabric.modal-postflight/v2",
        "memory_recovered_within_1_gib": True,
        "unexpected_compute_processes": [],
        "cleanup_gate": "PASS",
        "GPU_RUNTIME_LIFECYCLE_CLEAN": True,
        "authoritative_artifact": "lifecycle/cleanup-gate.json",
    }
    return completion, wrapper_postflight


def test_authoritative_lifecycle_is_revalidated_and_plotted(tmp_path: Path) -> None:
    completion, postflight = _materialize_clean_lifecycle(tmp_path)
    evidence = _ANALYZER._load_lifecycle_evidence(
        tmp_path,
        attempt_id="smoke-v6",
        gpu_uuid="GPU-test",
        completion=completion,
        wrapper_postflight=postflight,
    )
    assert evidence is not None
    assert evidence.time_to_hbm_recovery_ms == 1_000.0
    trial = SimpleNamespace(
        attempt_id="smoke-v6",
        lifecycle=evidence,
        baseline_mode="shared_root",
    )
    svg = _ANALYZER._lifecycle_hbm_svg(trial)
    for annotation in (
        "parent baseline",
        "child launch",
        "model load",
        "KV pool allocation",
        "benchmark start",
        "branch fork",
        "decode",
        "branch teardown",
        "vLLM shutdown",
        "child exit",
        "HBM recovery",
    ):
        assert annotation in svg
    assert "driver-visible GPU memory (MiB)" in svg


def test_authoritative_lifecycle_rejects_gate_that_disagrees_with_raw_samples(
    tmp_path: Path,
) -> None:
    completion, postflight = _materialize_clean_lifecycle(tmp_path)
    gate_path = tmp_path / "lifecycle/cleanup-gate.json"
    gate = json.loads(gate_path.read_text())
    gate["stable_sample_evaluation"]["winning_sample_start_index"] = 1
    _write_json(gate_path, gate)
    with pytest.raises(AnalysisError, match="disagrees with raw HBM samples"):
        _ANALYZER._load_lifecycle_evidence(
            tmp_path,
            attempt_id="smoke-v6",
            gpu_uuid="GPU-test",
            completion=completion,
            wrapper_postflight=postflight,
        )


def test_legacy_attempt_without_lifecycle_artifacts_remains_readable(tmp_path: Path) -> None:
    assert (
        _ANALYZER._load_lifecycle_evidence(
            tmp_path,
            attempt_id="smoke-v5",
            gpu_uuid="GPU-test",
            completion={},
            wrapper_postflight={
                "memory_recovered_within_1_gib": False,
                "unexpected_compute_processes": [],
            },
        )
        is None
    )


@pytest.mark.parametrize(
    ("tracing_level", "lifecycle_clean"),
    [("full", None), ("full", False), ("disabled", None), ("disabled", False)],
)
def test_16k_trial_requires_authoritative_clean_lifecycle(
    monkeypatch: pytest.MonkeyPatch, tracing_level: str, lifecycle_clean: bool | None
) -> None:
    monkeypatch.setattr(_ANALYZER, "_validate_physical_invariants", lambda trial: None)
    legacy_16k = SimpleNamespace(
        tracing_level=tracing_level,
        prefix_length=16_384,
        lifecycle_clean=lifecycle_clean,
        attempt_id="legacy-16k",
    )
    with pytest.raises(AnalysisError, match="lack an authoritative clean"):
        _ANALYZER._validate_smoke_and_pairs([legacy_16k])


def test_comparison_plot_labels_readiness_measurement_origins(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    chart_calls: list[dict] = []

    def fake_chart(**kwargs):
        chart_calls.append(kwargs)
        return "<svg/>\n"

    monkeypatch.setattr(_ANALYZER, "_svg_line_chart", fake_chart)
    monkeypatch.setattr(_ANALYZER, "_time_series_svg", lambda trial: "<svg/>\n")
    monkeypatch.setattr(_ANALYZER, "_private_progression_svg", lambda trial: "<svg/>\n")
    monkeypatch.setattr(_ANALYZER, "_lifecycle_hbm_svg", lambda trial: "<svg/>\n")
    cell = {
        "fanout": 1,
        "all_branches_ready_ns": {
            "independent": {"median": 2_000_000},
            "shared_root": {"median": 200_000},
        },
        "root_inclusive_all_branches_ready_ns": {
            "independent": {"median": 2_000_000},
            "shared_root": {"median": 1_500_000},
        },
        "physical_kv_bytes": {
            "independent_observed_peak": {"median": 2 * 1024**3},
            "independent_comparable_full_fanout_terminal_estimate": {"median": 2 * 1024**3},
            "shared_full_fanout_terminal_actual": {"median": 1024**3},
            "aligned_all_branches_live": {
                "shared_actual": {"median": 1024**3},
                "theoretical_shared_floor": {"median": 1024**3},
            },
            "shared_root": {"median": 1024**3},
            "theoretical_shared_floor": {"median": 1024**3},
        },
        "decode_tokens_per_second": {
            "independent": {"median": 10.0},
            "shared_root": {"median": 9.8},
        },
    }
    trial = SimpleNamespace(
        baseline_mode="shared_root",
        tracing_level="full",
        prefix_length=16_384,
        lifecycle=object(),
        lifecycle_clean=True,
        fanout=1,
        seed=41,
        attempt_id="shared-16k-v7",
    )
    records = _ANALYZER._plots(tmp_path / "plots", [cell], [trial])
    readiness = next(call for call in chart_calls if "readiness" in call["title"].lower())
    assert "explicit measurement origins" in readiness["title"]
    assert [series[0] for series in readiness["series"]] == [
        "independent (root-inclusive prefill)",
        "shared root (root-inclusive end-to-end)",
        "shared root (post-root readiness)",
    ]
    assert "gpu-runtime-lifecycle.svg" in {record["path"] for record in records}


def test_independent_physical_accounting_separates_peak_from_terminal_estimate() -> None:
    def block(branch_id: str, *, logical_end: int) -> SimpleNamespace:
        return SimpleNamespace(
            bytes=100,
            branch_ids=(branch_id,),
            logical_token_end_exclusive=logical_end,
        )

    peak = SimpleNamespace(
        physical_assigned_bytes=3_000, blocks=(block("branch-0", logical_end=26),)
    )
    terminal = SimpleNamespace(
        physical_assigned_bytes=400,
        blocks=tuple(block("branch-7", logical_end=266) for _ in range(4)),
    )
    assertions = {
        "independent_prefix_cache_hits_absent",
        "independent_branch_physical_block_sets_disjoint",
        "independent_prefix_physical_bytes_scale_with_fanout",
        "every_branch_executes_complete_prefix_prefill",
    }
    trial = SimpleNamespace(
        attempt_id="independent-f8",
        baseline_mode="independent",
        prefix_length=10,
        suffix_length=256,
        fanout=8,
        summary=SimpleNamespace(assertions=assertions),
        physical_rows=(("suffix_64", peak), ("suffix_256", terminal)),
    )
    accounting = _ANALYZER._independent_physical_accounting(trial)
    assert accounting["observed_peak_bytes"] == 3_000
    assert accounting["observed_terminal_survivor_bytes"] == 400
    assert accounting["full_fanout_terminal_estimate_bytes"] == 3_200
    assert accounting["estimate_minus_observed_peak_bytes"] == 200
    assert accounting["estimate_is_direct_observation"] is False


def test_shared_root_inclusive_readiness_is_continuous_through_decode_admission() -> None:
    trial = SimpleNamespace(
        attempt_id="shared-f8",
        baseline_mode="shared_root",
        summary=SimpleNamespace(all_branches_ready_ns=250),
        phase_rows=(
            {"phase": "root_session_create", "wall_start_monotonic_ns": 1_000},
            {
                "phase": "fork_metadata",
                "wall_start_monotonic_ns": 2_500,
                "wall_duration_ns": 100,
            },
            {"phase": "concurrent_decode", "wall_start_monotonic_ns": 2_800},
        ),
    )
    assert _ANALYZER._shared_root_inclusive_readiness_ns(trial) == 1_950


def test_fanout_8_metadata_signal_selects_exactly_choice_e() -> None:
    cell = {
        "fanout": 8,
        "derived": {
            "state_management_fraction": {"median": 0.1716555266},
            "cpu_block_management_fraction_of_readiness": {"median": 0.4828390837},
            "physical_amplification": 1.0,
            "cow_floor_gap_bytes": 0.0,
            "decode_interference_fraction": None,
        },
    }
    classification = _ANALYZER._next_experiment_classification([cell], clean_smoke=True)
    assert classification is not None
    assert classification["choice"] == "E"
    assert classification["name"] == "HIGH-FANOUT RUNTIME METADATA CHARACTERIZATION"
    assert classification["evidence"]["decode_interference_availability"] == (
        "unavailable_cross_gpu_pair"
    )


def test_provider_cleanup_is_pending_until_current_campaign_audit_exists(
    tmp_path: Path,
) -> None:
    legacy = tmp_path / "environment/final-modal-provider-audit.json"
    _write_json(
        legacy,
        {
            "schema_version": "sloforge.branchfabric.modal-provider-audit/v1",
            "recorded_at_utc": "2026-08-11T04:31:20Z",
        },
    )
    evidence = _ANALYZER._provider_cleanup_evidence(
        tmp_path,
        [
            {
                "attempt_id": "current-v8",
                "function_call_id": "fc-current",
                "function_end_utc": "2026-08-11T16:00:00Z",
            }
        ],
    )
    assert evidence["verified"] is False
    assert evidence["status"] == "pending_unverified"
    assert evidence["legacy_provider_audit"]["status"] == (
        "historical_v5_scope_only_not_current_campaign_proof"
    )


def test_provider_cleanup_requires_current_apps_and_container_evidence(tmp_path: Path) -> None:
    audit_path = tmp_path / "environment/final-modal-provider-audit-v2.json"
    _write_json(
        audit_path,
        {
            "schema_version": "sloforge.branchfabric.modal-provider-final-audit/v2",
            "experiment": "experiment-002",
            "recorded_at_utc": "2026-08-11T16:01:00Z",
            "modal_app_list_command": ["modal", "app", "list", "--json"],
            "modal_container_list_command": ["modal", "container", "list", "--json"],
            "source_attempt_ids": ["current-v8"],
            "source_function_call_ids": ["fc-current"],
            "experiment_app_description": "sloforge-branchfabric-real-gpu-cow",
            "experiment_apps": [
                {
                    "app_id": "ap-current",
                    "description": "sloforge-branchfabric-real-gpu-cow",
                    "state": "stopped",
                    "task_count": 0,
                }
            ],
            "experiment_apps_not_stopped_or_with_tasks": [],
            "all_experiment_apps_stopped": True,
            "running_containers": [],
            "provider_hbm_after_container_teardown_sampled": False,
            "persistent_resources_retained": [
                "sloforge-model-cache",
                "sloforge-branchfabric-results",
            ],
        },
    )
    evidence = _ANALYZER._provider_cleanup_evidence(
        tmp_path,
        [
            {
                "attempt_id": "current-v8",
                "function_call_id": "fc-current",
                "function_end_utc": "2026-08-11T16:00:00Z",
            }
        ],
    )
    assert evidence["verified"] is True
    assert evidence["status"] == (
        "verified_all_experiment_apps_stopped_zero_tasks_and_no_running_containers"
    )
