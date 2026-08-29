# BranchFabric GPU Validation Experiment 003

Completion status: `complete`
Metadata conclusion: `METADATA_SOFTWARE_ONLY`

## GPU/runtime/model

- Model: `Qwen/Qwen2.5-7B-Instruct` revision `a09a35458c702b33eeacc393d103063234e8bc28`
- Runtime: vLLM `0.23.0`, PyTorch `2.11.0`, CUDA `13.0`
- Baseline GPU: `{'index': 0, 'uuid': 'GPU-bb86053c-1232-1eca-5aeb-574862b978f5', 'name': 'NVIDIA A100-SXM4-80GB', 'driver_version': '580.95.05', 'memory_total_mib': 81920}`
- Optimized GPU: `{'index': 0, 'uuid': 'GPU-03bc8fb9-13fd-508f-2fc8-be64e3f4f23c', 'name': 'NVIDIA A100 80GB PCIe', 'driver_version': '580.95.05', 'memory_total_mib': 81920}`

## Same-GPU paired status

- Baseline: `available`
- Optimized: `available`

## Readiness, metadata, scheduler, and GPU

See `metrics/readiness-decomposition-summary.json` for the exact non-overlapping stages and raw values. Scheduler and GPU stages are not counted as metadata.
`POST_ROOT_READY` ends after every branch returned first output. The legacy `FIRST_BRANCH_READY`/`ALL_BRANCHES_READY` fields are output-step-return boundaries, not direct runnable transitions; step-entry execution-start proxies are reported separately.
`GPU_EXECUTION` is GPU-inclusive worker CPU wall, not pure device time. Optimized fanout-8 first `_model_forward` CUDA-event time is reported separately; baseline lacks the later sample-token wrappers, so baseline/optimized GPU and residual components are not identically scoped.

## Software optimization and Amdahl

- Before/after: `available`
- Classification evidence: `{"aggregation_rule": "hardware-interest if any measured fanout qualifies; otherwise software-only if any measured fanout qualifies; closed only when every measured fanout closes", "conclusion": "METADATA_SOFTWARE_ONLY", "counted_metadata_work_units": 32840, "fanouts": {"16": {"conclusion": "METADATA_SOFTWARE_ONLY", "counted_metadata_work_units": 65680, "counter_semantics": "non-overlapping software counter taxonomy; not unique hardware transactions", "gpu_wall_ns": 28829906, "ideal_metadata_free_speedup": 1.1243425123577144, "metadata_fraction": 0.11059131091376394, "metadata_wall_ns": 29077783, "sample_count": 1, "scheduler_wall_ns": 3843746}, "32": {"conclusion": "METADATA_SOFTWARE_ONLY", "counted_metadata_work_units": 131360, "counter_semantics": "non-overlapping software counter taxonomy; not unique hardware transactions", "gpu_wall_ns": 45545009, "ideal_metadata_free_speedup": 1.126581278353154, "metadata_fraction": 0.11235876255479009, "metadata_wall_ns": 51422277, "sample_count": 1, "scheduler_wall_ns": 8071458}, "8": {"conclusion": "METADATA_CLOSED", "counted_metadata_work_units": 32840, "counter_semantics": "non-overlapping software counter taxonomy; not unique hardware transactions", "gpu_wall_ns": 190683538, "ideal_metadata_free_speedup": 1.0461929982720388, "metadata_fraction": 0.04415341944395938, "metadata_wall_ns": 12953524, "sample_count": 3, "scheduler_wall_ns": 2254981}}, "optimized_fanout_8_gpu_wall_ns": 190683538, "optimized_fanout_8_ideal_free_speedup": 1.0461929982720388, "optimized_fanout_8_metadata_fraction": 0.04415341944395938, "optimized_fanout_8_metadata_wall_ns": 12953524, "optimized_fanout_8_scheduler_wall_ns": 2254981, "uncertainty_caveat": "fanout 8 uses three paired seeds; fanout 16 and 32 are single diagnostic trials. The aggregate SOFTWARE_ONLY classification is conservative but its high-fanout fractions do not estimate trial-to-trial variance"}`
- Uncertainty: fanout 8 has three paired seeds; fanout 16/32 are singletons and drive the conservative SOFTWARE_ONLY aggregate.
- Before/after latency deltas are descriptive only because baseline used A100-SXM4 and optimized used A100-PCIE on different UUIDs; within-campaign independent/shared pairs remain causal.

## HBM/state sharing

- Hardware COW capacity target: `CLOSED`

## Decode interference

Causal concurrent decode interference is unavailable because no background decode stream overlapped branch creation. Early-divergence values are labeled sequential phase-productivity deficits against established-branch throughput; branch-create zero is only a structural synchronous stall. No Experiment 002 throughput is used.

## GPU-hours and cleanup

- GPU-hour ledger: `{"all_reservations_cleared": true, "consumed_gpu_hours": 1.105089655963889, "cost_note": "baseline recorded GPU cost only; optimized recorded GPU plus support cost. known_gpu_only_cost_usd is comparable across campaigns", "gpu_active_intervals": [{"campaign_id": "exp003-baseline-20260811-v1", "cost_usd": 1.36459497140142, "function_call_id": "fc-01KZSBX572AK09MYBJTSG1TWVR", "gpu": "A100-80GB", "gpu_count": 1, "local_call_elapsed_seconds": 2008.842290332992, "model_load_benchmark_cleanup_seconds": 1966.27517493}, {"campaign_id": "exp003-optimized-20260811-v3", "cost_usd": 1.7501594726759537, "function_call_id": "fc-01KZSFAHAM2Z1DZ9V40P96HW4F", "gpu": "A100-80GB", "gpu_cost_usd": 1.3963610250587601, "gpu_count": 1, "local_call_elapsed_seconds": 2028.7499277079914, "model_load_benchmark_cleanup_seconds": 2012.04758654, "reservation_id": "exp003-reservation-6ddd1b48571946a5b9e12c2d55ffd21b", "status": "succeeded", "support_cost_usd": 0.3537984476171936}], "hard_additional_gpu_hours": 1.25, "historical_experiment_002_gpu_hours": 0.5640564932644445, "in_flight_reservations": [], "known_gpu_only_cost_usd": 2.76095599646018, "known_support_cost_usd": 0.3537984476171936, "provenance": {"artifact_reference": "artifacts/branchfabric/gpu-validation/experiment-003/gpu-hours.json", "artifact_sha256": "743fb4384126bea96895b6825b9a4fc4fe5d8f13c71e3cba193fc379a6e0fe4f"}, "recorded_cost_usd_mixed_conventions": 3.1147544440773736, "status": "available", "target_additional_gpu_hours": 0.75}`
- Cleanup status: `{"campaign_reservations_cleared": true, "child_lifecycle_cleanup": "PASS", "children_verified": 22, "deployed_endpoints_absent": true, "forced_gpu_process_kill_required": false, "observed_at_utc": "2026-08-11T23:44:48Z", "only_authorized_persistent_volumes": true, "provenance": {"artifact_reference": "artifacts/branchfabric/gpu-validation/experiment-003/logs/modal-provider-cleanup.json", "artifact_sha256": "3d965269494abe4516a34b7cea943c904719d9d5882da3dff776f480a8280f31"}, "provider_zero_active_tasks": true, "status": "PASS"}`

## Next experiment

`Experiment 004 capacity reclamation`
