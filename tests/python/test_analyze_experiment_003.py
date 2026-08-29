from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_BENCHMARK_ROOT = _ROOT / "benchmarks/branchfabric"
sys.path.insert(0, str(_BENCHMARK_ROOT))
_MODULE_PATH = _BENCHMARK_ROOT / "analyze_experiment_003.py"
_SPEC = importlib.util.spec_from_file_location(
    "sloforge_analyze_experiment_003_tests",
    _MODULE_PATH,
)
assert _SPEC is not None and _SPEC.loader is not None
_ANALYZER = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _ANALYZER
_SPEC.loader.exec_module(_ANALYZER)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")


def _sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _gpu(uuid: str = "GPU-fixture") -> dict[str, Any]:
    return {
        "index": 0,
        "uuid": uuid,
        "name": "NVIDIA A100 80GB PCIe",
        "driver_version": "580.fixture",
        "memory_total_mib": 81_920,
    }


def _sample(gpu: dict[str, Any], memory_mib: int = 8) -> dict[str, Any]:
    return {
        "gpu": {**gpu, "memory_used_mib": memory_mib, "utilization_percent": 0},
        "compute_processes": [],
        "observed_at_monotonic_ns": 1,
    }


def _stages(total_ns: int, metadata_ns: int) -> dict[str, dict[str, int]]:
    values = {stage: 0 for stage in _ANALYZER.STAGES}
    values.update(
        {
            "helix_orchestration": 5_000_000,
            "request_build": 5_000_000,
            "prefix_lookup": metadata_ns,
            "scheduler_select": 10_000_000,
            "gpu_execution": 20_000_000,
            "output_token_commit": 5_000_000,
        }
    )
    values["residual"] = total_ns - sum(values.values())
    assert values["residual"] >= 0
    return {
        stage: {
            "wall_ns": value,
            "process_cpu_ns": value,
            "thread_cpu_ns": value,
        }
        for stage, value in values.items()
    }


def _trial_spec(
    *,
    seed: int,
    mode: str,
    fanout: int = 8,
    pair_id: str | None = None,
    pair_position: int | None = None,
    pair_rule: str | None = None,
    trace: str = "minimal",
    post_root_ns: int = 100_000_000,
    metadata_ns: int = 2_000_000,
) -> dict[str, Any]:
    return {
        "seed": seed,
        "mode": mode,
        "fanout": fanout,
        "pair_id": pair_id,
        "pair_position": pair_position,
        "pair_rule": pair_rule,
        "trace": trace,
        "post_root_ns": post_root_ns,
        "metadata_ns": metadata_ns,
    }


def _paired_and_scaling_specs() -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    orders = {
        41: ("independent", "shared_root"),
        73: ("shared_root", "independent"),
        113: ("independent", "shared_root"),
    }
    for seed, order in orders.items():
        for position, mode in enumerate(order, start=1):
            specs.append(
                _trial_spec(
                    seed=seed,
                    mode=mode,
                    pair_id=f"seed-{seed}",
                    pair_position=position,
                    pair_rule=(
                        "independent_then_shared"
                        if seed == 41
                        else (
                            "shared_then_independent" if seed == 73 else "deterministic_randomized"
                        )
                    ),
                    post_root_ns=110_000_000 if mode == "independent" else 100_000_000,
                )
            )
    specs.extend(
        [
            _trial_spec(seed=149, mode="shared_root", fanout=1, post_root_ns=90_000_000),
            _trial_spec(seed=149, mode="shared_root", fanout=16, post_root_ns=120_000_000),
            _trial_spec(seed=149, mode="shared_root", fanout=32, post_root_ns=150_000_000),
            _trial_spec(
                seed=197,
                mode="shared_root",
                pair_rule="instrumentation_control",
                trace="disabled",
                post_root_ns=100_000_000,
            ),
            _trial_spec(
                seed=197,
                mode="shared_root",
                pair_rule="instrumentation_control",
                trace="minimal",
                post_root_ns=101_000_000,
            ),
            _trial_spec(
                seed=197,
                mode="shared_root",
                pair_rule="instrumentation_control",
                trace="full",
                post_root_ns=110_000_000,
            ),
        ]
    )
    return specs


def _materialize_campaign(
    root: Path,
    *,
    implementation: str,
    specs: list[dict[str, Any]],
    cleanup_pass: bool = True,
) -> Path:
    gpu = _gpu()
    threshold = 1032
    campaign_id = f"fixture-{implementation}-campaign"
    function_call_id = f"fc-{implementation}"
    records: list[dict[str, Any]] = []
    for position, spec in enumerate(specs):
        attempt = f"fixture-{implementation}-{position:02d}-{spec['mode']}-f{spec['fanout']}"
        config = {
            "schema_version": "sloforge.branchfabric.modal-metadata-characterization-config/v1",
            "attempt_id": attempt,
            "campaign_id": campaign_id,
            "pair_id": spec["pair_id"],
            "pair_order_position": spec["pair_position"],
            "pair_order_rule": spec["pair_rule"],
            "pair_order_seed": 20_260_811
            if spec["pair_rule"] == "deterministic_randomized"
            else None,
            "model": _ANALYZER.EXPECTED_MODEL,
            "model_revision": _ANALYZER.EXPECTED_REVISION,
            "tokenizer_revision": _ANALYZER.EXPECTED_REVISION,
            "runtime": "vllm",
            "runtime_version": "0.23.0",
            "fanout": spec["fanout"],
            "prefix_length": 16_384,
            "suffix_length": 256,
            "seed": spec["seed"],
            "baseline_mode": spec["mode"],
            "implementation": implementation,
            "tracing_level": spec["trace"],
            "maximum_wall_seconds": 240,
            "initialization_timeout_seconds": 300,
            "cleanup_timeout_seconds": 60,
            "gpu_memory_utilization": 0.8,
            "gpu_sample_interval_seconds": 1.0,
        }
        cleanup = {
            "schema_version": "sloforge.branchfabric.gpu-runtime-cleanup-gate/v1",
            "GPU_RUNTIME_LIFECYCLE_CLEAN": cleanup_pass,
            "cleanup_gate": "PASS" if cleanup_pass else "FAIL",
            "forced_gpu_process_kill_required": False,
        }
        child = {
            "schema_version": "sloforge.branchfabric.gpu-cow-child-manifest/v1",
            "attempt_id": attempt,
            "status": "succeeded",
            "final_runtime_assigned_kv_bytes": 0,
            "semantic_invariants": {"fixture_semantics": True},
            "versions": {
                "packages": {"vllm": "0.23.0", "torch": "2.11.0"},
                "torch_cuda_userspace": "13.0",
            },
        }
        gate = {
            "passed": True,
            "cleanup_threshold_mib": threshold,
            "samples": [_sample(gpu), _sample(gpu), _sample(gpu)],
        }
        controller = {
            "status": "succeeded",
            "actual_gpu": gpu,
            "cleanup_gate": cleanup,
            "child_manifest": child,
        }
        record = {
            "position": position,
            "attempt_id": attempt,
            "config": config,
            "pre_child_gate": gate,
            "controller": controller,
            "campaign_post_child_gate": gate,
        }
        records.append(record)
        trial_root = root / "trials" / f"{position:02d}-{attempt}"
        _write_json(trial_root / "lifecycle/cleanup-gate.json", cleanup)
        _write_json(trial_root / "lifecycle/child-manifest.json", child)
        mode_path = "independent_prefill" if spec["mode"] == "independent" else "shared_root"
        runner = trial_root / "runner" / mode_path
        physical = 8_000 if spec["mode"] == "independent" else 1_000
        shared = 0 if spec["mode"] == "independent" else 800
        private = physical if spec["mode"] == "independent" else 200
        summary = {
            "schema_version": "sloforge.branchfabric.metadata-trial-summary/v1",
            "configuration": {
                "attempt_id": attempt,
                "mode": mode_path,
                "implementation": implementation,
                "seed": spec["seed"],
                "prefix_token_ids": [7] * 16_384,
                "divergent_token_ids": [8] * spec["fanout"],
                "suffix_tokens": 256,
                "timeout_s": 240.0,
                "trace_level": spec["trace"],
            },
            "fanout": spec["fanout"],
            "root_inclusive_ready_ns": spec["post_root_ns"] + 200_000_000,
            "post_root_ready_ns": spec["post_root_ns"],
            "first_branch_ready_ns": spec["post_root_ns"] // 2,
            "all_branches_ready_ns": spec["post_root_ns"],
            "first_token_latency_ns": spec["post_root_ns"] + 1_000_000,
            "decode_throughput_tokens_per_second": 1000.0,
            "steady_decode_throughput_tokens_per_second": 900.0,
            "decode_active_tokens": spec["fanout"] * 256 - spec["fanout"],
            "decode_active_elapsed_ns": 200_000_000,
            "physical_block_count": physical // 10,
            "shared_root_block_count": 80 if shared else 0,
            "private_block_count": private // 10,
            "physical_assigned_bytes": physical,
            "shared_prefix_bytes": shared,
            "private_suffix_bytes": private,
            "final_runtime_assigned_kv_bytes": 0,
            "semantics": (
                {
                    "prefix_cache_reuse_absent": True,
                    "independent_prefix_blocks_single_owner": True,
                }
                if spec["mode"] == "independent"
                else {
                    "exact_shared_root_block_ids": True,
                    "runtime_native_refcounts_cover_fanout": True,
                    "private_suffix_blocks_are_single_owner": True,
                }
            ),
            "metadata_operation_counters": (
                None
                if spec["trace"] == "disabled"
                else {
                    "block_table_writes": spec["fanout"] * 1024,
                    "refcount_increments": spec["fanout"] * 1024,
                    "scheduler_queue_inserts": spec["fanout"],
                }
            ),
            "metadata_operation_normalization": (
                None if spec["trace"] == "disabled" else {"fanout": spec["fanout"]}
            ),
            "post_root_process_cpu_ns": spec["post_root_ns"],
            "post_root_thread_cpu_ns": spec["post_root_ns"],
        }
        _write_json(runner / "metrics/trial-summary.json", summary)
        if spec["trace"] != "disabled":
            stages = _stages(spec["post_root_ns"], spec["metadata_ns"])
            _write_json(
                runner / "metrics/readiness-decomposition.json",
                {
                    "schema_version": "sloforge.branchfabric.vllm-0230-metadata-timing/v1",
                    "spans": [],
                    "decomposition": {
                        "parent": {
                            "wall_ns": spec["post_root_ns"],
                            "process_cpu_ns": spec["post_root_ns"],
                            "thread_cpu_ns": spec["post_root_ns"],
                        },
                        "stages": stages,
                        "invariant": {
                            "sum_equals_parent": True,
                            "wall_error_ns": 0,
                            "process_cpu_error_ns": 0,
                            "thread_cpu_error_ns": 0,
                        },
                    },
                },
            )
        metadata_observation = {
            "schema_version": "sloforge.branchfabric.vllm-metadata-observation-0230/v1",
            "runtime": "vllm",
            "runtime_version": "0.23.0",
            "runtime_source_tag": _ANALYZER.EXPECTED_VLLM_SOURCE_TAG,
            "runtime_source_commit": _ANALYZER.EXPECTED_VLLM_SOURCE_COMMIT,
            "instrumentation_level": spec["trace"],
            "optimization": implementation,
            "post_root_ready_native_metric_count": (None if spec["trace"] == "disabled" else 1),
            "post_root_ready_runtime_native_metrics": (
                []
                if spec["trace"] == "disabled"
                else [{"metrics": {"num_running_reqs": spec["fanout"], "num_waiting_reqs": 0}}]
            ),
            "post_root_ready_operation_counters": summary["metadata_operation_counters"],
            "spans": [],
        }
        _write_json(
            runner / "metadata/runtime-metadata-observation.json",
            metadata_observation,
        )
        _write_json(
            runner / "native-metrics/runtime-native-metrics.json",
            {
                "schema_version": "sloforge.branchfabric.runtime-native-metrics/v1",
                "runtime": "vllm",
                "runtime_version": "0.23.0",
                "observations": (
                    []
                    if spec["trace"] == "disabled"
                    else [
                        {
                            "metrics": {
                                "num_running_reqs": spec["fanout"],
                                "num_waiting_reqs": 0,
                            }
                        }
                    ]
                ),
                "post_root_ready_observation_count": (None if spec["trace"] == "disabled" else 1),
            },
        )
        _write_json(
            runner / "raw/nvml-samples.json",
            {
                "schema_version": "sloforge.branchfabric.nvml-samples/v1",
                "gpu_uuid": gpu["uuid"],
                "interval_seconds": 0.1,
                "samples": [
                    {
                        "gpu_utilization_percent": 50,
                        "memory_utilization_percent": 20,
                        "memory_used_bytes": physical,
                    }
                ],
                "error": None,
            },
        )
        _write_json(
            runner / "raw/physical-layouts.json",
            {
                "root": None if spec["mode"] == "independent" else {"physical_assigned_bytes": 800},
                "decode": (
                    [
                        {
                            # Preserve an earlier all-request-live peak so the
                            # analyzer cannot mistake a later, partially
                            # completed final snapshot for concurrent HBM.
                            "physical_assigned_bytes": physical * 2,
                            "blocks": [
                                {"branch_ids": [f"{attempt}-branch-{index:02d}"]}
                                for index in range(spec["fanout"])
                            ],
                        }
                    ]
                    if spec["mode"] == "independent"
                    else []
                ),
                "final": {
                    "physical_assigned_bytes": physical,
                    "blocks": (
                        [
                            {"branch_ids": [f"{attempt}-branch-{index:02d}"]}
                            for index in range(spec["fanout"])
                        ]
                        if spec["mode"] == "independent"
                        else []
                    ),
                },
            },
        )
    baseline = {
        "passed": True,
        "gpu_identity": gpu,
        "samples": [_sample(gpu), _sample(gpu), _sample(gpu)],
        "baseline_median_mib": 8,
        "cleanup_threshold_mib": threshold,
        "allowance_mib": 1024,
    }
    _write_json(root / "campaign-hbm-baseline.json", baseline)
    _write_json(
        root / "campaign-manifest.json",
        {
            "schema_version": "sloforge.branchfabric.modal-campaign-manifest-003/v1",
            "campaign_id": campaign_id,
            "status": "succeeded",
            "function_call_id": function_call_id,
            "requested_gpu": "A100-80GB",
            "gpu_identity": gpu,
            "same_gpu_enforced_before_every_child": True,
            "immutable_cleanup_threshold_mib": threshold,
            "planned_children": len(records),
            "completed_children": len(records),
            "records": records,
        },
    )
    inventory = [
        {
            "relative_path": path.relative_to(root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "REMOTE_MANIFEST.json"
    ]
    _write_json(
        root / "REMOTE_MANIFEST.json",
        {
            "schema_version": "sloforge.branchfabric.modal-campaign-result/v1",
            "campaign_id": campaign_id,
            "remote_prefix": f"experiment-003/modal/{campaign_id}",
            "artifacts": inventory,
        },
    )
    return root


def test_generator_preserves_unavailable_measurements_and_creates_all_outputs(
    tmp_path: Path,
) -> None:
    baseline = _materialize_campaign(
        tmp_path / "baseline",
        implementation="baseline",
        specs=[_trial_spec(seed=149, mode="shared_root")],
    )
    optimized = _materialize_campaign(
        tmp_path / "optimized",
        implementation="optimized",
        specs=[_trial_spec(seed=149, mode="shared_root")],
    )
    artifact_root = tmp_path / "artifacts"
    reports = tmp_path / "reports"
    docs = tmp_path / "docs"

    report = _ANALYZER.generate_analysis(
        baseline_campaign_root=baseline,
        optimized_campaign_root=optimized,
        artifact_root=artifact_root,
        reports_root=reports,
        docs_root=docs,
    )

    assert report["completion_status"] == "incomplete"
    assert report["metadata_classification"]["conclusion"] == "UNAVAILABLE"
    assert report["paired_baseline"]["status"] == "unavailable"
    early = report["decode_interference"]["implementations"]["baseline"]["8"]["raw"][0][
        "early_divergence"
    ]
    assert early["status"] == "available"
    assert early["early_output_tokens"] == 8
    assert "phase_throughput_deficit_fraction" in early
    assert early["causal_concurrent_decode_interference"]["status"] == "unavailable"
    assert (
        report["decode_interference"]["implementations"]["baseline"]["8"]["raw"][0][
            "branch_create"
        ]["status"]
        == "unavailable"
    )
    assert report["native_metric_agreement"]["trials"][0]["recorder_count_agreement"] == (
        "confirms"
    )
    assert report["hbm"]["HARDWARE_COW_CAPACITY_TARGET"] == "CLOSED"
    assert not (docs / "EXPERIMENT_004_PLAN.md").exists()
    assert (docs / "METADATA_CHARACTERIZATION.md").is_file()
    assert (reports / "branchfabric-gpu-validation-experiment-003.json").is_file()
    assert (reports / "branchfabric-gpu-validation-experiment-003.md").is_file()
    assert (artifact_root / "metrics/readiness-definitions.json").is_file()
    runtime_source = json.loads(
        (artifact_root / "runtime-source/vllm-0230-runtime-source.json").read_text()
    )
    assert runtime_source["runtime_source_commit"] == _ANALYZER.EXPECTED_VLLM_SOURCE_COMMIT
    assert report["gpu_hours"]["status"] == "unavailable"
    plots = sorted((artifact_root / "plots").glob("*.svg"))
    assert len(plots) == 8
    assert all("<svg" in plot.read_text() for plot in plots)
    manifest = json.loads((artifact_root / "manifest.json").read_text())
    assert set(manifest["required_directories"]) == set(_ANALYZER.REQUIRED_ARTIFACT_DIRECTORIES)
    assert all((artifact_root / name).is_dir() for name in _ANALYZER.REQUIRED_ARTIFACT_DIRECTORIES)


def test_complete_low_metadata_campaign_closes_and_generates_experiment_004_plan(
    tmp_path: Path,
) -> None:
    baseline = _materialize_campaign(
        tmp_path / "baseline",
        implementation="baseline",
        specs=_paired_and_scaling_specs(),
    )
    optimized = _materialize_campaign(
        tmp_path / "optimized",
        implementation="optimized",
        specs=_paired_and_scaling_specs(),
    )
    report = _ANALYZER.generate_analysis(
        baseline_campaign_root=baseline,
        optimized_campaign_root=optimized,
        artifact_root=tmp_path / "artifacts",
        reports_root=tmp_path / "reports",
        docs_root=tmp_path / "docs",
    )

    assert report["paired_baseline"]["status"] == "available"
    assert report["paired_optimized"]["status"] == "available"
    assert len(report["paired_baseline"]["analysis"]["raw_pairs"]) == 3
    assert report["metadata_classification"]["conclusion"] == "METADATA_CLOSED"
    assert (
        report["hbm"]["implementations"]["baseline"]["independent_fanout_8"][
            "median_physical_kv_bytes"
        ]
        == 16_000
    )
    assert report["initial_analysis_gate"]["category"] == "A"
    assert report["recommended_next_experiment"] == "Experiment 004 capacity reclamation"
    plan = (tmp_path / "docs/EXPERIMENT_004_PLAN.md").read_text()
    assert "two NVIDIA A100 80GB GPUs" in plan
    assert "serving-load spike" in plan
    assert "transform and transfer" in plan
    assert "Do not execute this plan" in plan


def test_hardware_interest_emits_architecture_payload_without_experiment_004(
    tmp_path: Path,
) -> None:
    specs = _paired_and_scaling_specs()
    for spec in specs:
        spec["metadata_ns"] = 20_000_000
    baseline = _materialize_campaign(tmp_path / "baseline", implementation="baseline", specs=specs)
    optimized = _materialize_campaign(
        tmp_path / "optimized", implementation="optimized", specs=specs
    )
    artifact_root = tmp_path / "artifacts"
    docs = tmp_path / "docs"

    report = _ANALYZER.generate_analysis(
        baseline_campaign_root=baseline,
        optimized_campaign_root=optimized,
        artifact_root=artifact_root,
        reports_root=tmp_path / "reports",
        docs_root=docs,
    )

    assert report["metadata_classification"]["conclusion"] == ("METADATA_HARDWARE_INTEREST")
    payload = json.loads((artifact_root / "metadata-hardware-interest.json").read_text())
    assert payload["fpga_justification"] is False
    assert payload["fanouts"]["8"][0]["counted_metadata_work_units_per_second"] > 0
    assert (docs / "METADATA_HARDWARE_INTEREST.md").is_file()
    assert not (docs / "EXPERIMENT_004_PLAN.md").exists()


@pytest.mark.parametrize(
    ("fraction", "speedup", "metadata", "scheduler", "gpu", "operations", "expected"),
    [
        (0.04, 1.04, 4.0, 20.0, 30.0, 100, "METADATA_CLOSED"),
        (0.10, 1.10, 10.0, 20.0, 30.0, 100, "METADATA_SOFTWARE_ONLY"),
        (0.20, 1.25, 20.0, 5.0, 5.0, 100, "METADATA_HARDWARE_INTEREST"),
        (0.20, 1.25, 20.0, 15.0, 10.0, 100, "METADATA_HARDWARE_INTEREST"),
        (None, None, None, None, None, None, "UNAVAILABLE"),
    ],
)
def test_metadata_gate_is_exact_and_requires_regular_measured_operations(
    fraction: float | None,
    speedup: float | None,
    metadata: float | None,
    scheduler: float | None,
    gpu: float | None,
    operations: int | None,
    expected: str,
) -> None:
    assert (
        _ANALYZER.classify_metadata(
            metadata_fraction=fraction,
            ideal_free_speedup=speedup,
            metadata_wall_ns=metadata,
            scheduler_wall_ns=scheduler,
            gpu_wall_ns=gpu,
            counted_metadata_work_units=operations,
        )
        == expected
    )


def test_loader_rejects_failed_authoritative_cleanup(tmp_path: Path) -> None:
    campaign = _materialize_campaign(
        tmp_path / "baseline",
        implementation="baseline",
        specs=[_trial_spec(seed=149, mode="shared_root")],
        cleanup_pass=False,
    )
    with pytest.raises(_ANALYZER.Experiment003AnalysisError, match="cleanup"):
        _ANALYZER.load_campaign(campaign, implementation="baseline")
