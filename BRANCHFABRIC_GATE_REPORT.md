# BranchFabric hardware gate report

Phase: **preliminary**.
Outcome: **FAIL_NO_BUILD**.
Hardware implementation allowed: **false**.

No threshold was loosened. CPU-reference, synthetic, simulated-hardware, and artifact-replay evidence cannot satisfy the target-hardware requirement.

## Candidate results

| Candidate | Real evidence | End-to-end relevance | Headroom | Platform | Workload value | Disposition |
|---|---|---|---|---|---|---|
| reshard_data_path | FAIL | FAIL | FAIL | FAIL | FAIL | NOT_JUSTIFIED |
| one_to_many_multicast | FAIL | FAIL | FAIL | FAIL | FAIL | NOT_JUSTIFIED |
| branch_translation_metadata | FAIL | FAIL | FAIL | FAIL | FAIL | NOT_JUSTIFIED |
| shared_root_cow | FAIL | FAIL | FAIL | FAIL | FAIL | NOT_JUSTIFIED |

## Candidate calculations and evidence

### `reshard_data_path`

- `mandatory_real_evidence`: **FAIL** — no target-relevant real hardware raw samples; fewer than two workload classes.
- `mandatory_end_to_end_relevance`: **FAIL** — no measured fraction reached a threshold; target critical path=0.016694.
- `mandatory_system_level_headroom`: **FAIL** — no lower confidence bound reached a threshold; all are unknown.
- `mandatory_platform_feasibility`: **FAIL** — missing: byte rate, operation rate, concurrency, queue depth, working set, latency target, bandwidth target, fault behavior, interface requirement, credible resource estimate, selected-target fit.
- `workload_value_data_path`: **FAIL** — missing: large_state_operations, repeated_frequency, bandwidth_pressure, temporary_memory_or_interference, full_system_leverage.

Evidence bindings:

- `artifacts/branchfabric/characterization/cpu-reference/attempts/continuum_state/attempt-000/continuum-trace/state-operation-trace-v1.jsonl` — SHA-256 `aa8c884f7d1e53ef824ceefc74528ea058c02de31ea6ab86db0d1271bd80fa7b`; CPU_REFERENCE_MODEL_STATE; n=18; raw samples.
- `artifacts/branchfabric/analysis/amdahl/cpu-reference-v3-windowed/amdahl-analysis.json` — SHA-256 `a3012b3532c50ffb792ae573f8ece0ff6b6dcc77201055c73b52817dc72b62f4`; ARTIFACT_REPLAY; n=1; derived/replay.

### `one_to_many_multicast`

- `mandatory_real_evidence`: **FAIL** — no target-relevant real hardware raw samples; fewer than two workload classes.
- `mandatory_end_to_end_relevance`: **FAIL** — no measured fraction reached a threshold; all are unknown.
- `mandatory_system_level_headroom`: **FAIL** — no lower confidence bound reached a threshold; all are unknown.
- `mandatory_platform_feasibility`: **FAIL** — missing: byte rate, operation rate, concurrency, queue depth, working set, latency target, bandwidth target, fault behavior, interface requirement, credible resource estimate, selected-target fit.
- `workload_value_fanout`: **FAIL** — missing: repeated_physical_unicast, destination_count_gt_one, identical_large_source, unicast_material_to_bytes_or_latency, transforms_preserve_benefit.

Evidence bindings:

- `artifacts/branchfabric/characterization/cpu-reference/attempts/vertical_trace/attempt-000/vertical-trace/branch-workload-trace-v1.jsonl` — SHA-256 `1647d7ecad74914851b864a770633334c9c526066ed19fe0f5c19e8c53446dfc`; CPU_REFERENCE_MODEL_STATE; n=70; raw samples.

### `branch_translation_metadata`

- `mandatory_real_evidence`: **FAIL** — no target-relevant real hardware raw samples; fewer than two workload classes.
- `mandatory_end_to_end_relevance`: **FAIL** — no measured fraction reached a threshold; target critical path=0.000948.
- `mandatory_system_level_headroom`: **FAIL** — no lower confidence bound reached a threshold; all are unknown.
- `mandatory_platform_feasibility`: **FAIL** — missing: byte rate, operation rate, concurrency, queue depth, working set, latency target, bandwidth target, fault behavior, interface requirement, credible resource estimate, selected-target fit.
- `workload_value_metadata`: **FAIL** — missing: optimized_software_metadata_is_critical, operation_rate_and_concurrency_measured, cache_lock_allocation_pressure_measured, hardware_working_set_and_queue_derived, full_system_leverage.

Evidence bindings:

- `artifacts/branchfabric/metadata/seed-20260809/metadata-study.json` — SHA-256 `d46a31b60147dc86427295de4c7de12c9078df44034aafdd41e518cfd1a1db8f`; SYNTHETIC; n=840; raw samples.

### `shared_root_cow`

- `mandatory_real_evidence`: **FAIL** — no target-relevant real hardware raw samples; fewer than two workload classes.
- `mandatory_end_to_end_relevance`: **FAIL** — no measured fraction reached a threshold; target critical path=0.000212.
- `mandatory_system_level_headroom`: **FAIL** — no lower confidence bound reached a threshold; all are unknown.
- `mandatory_platform_feasibility`: **FAIL** — missing: byte rate, operation rate, concurrency, queue depth, working set, latency target, bandwidth target, fault behavior, interface requirement, credible resource estimate, selected-target fit.
- `workload_value_state_sharing`: **FAIL** — missing: real_physical_state_measured, representative_fanout, shared_root_lifetime_material, allocation_savings_affect_capacity_or_throughput, software_cow_insufficient.

Evidence bindings:

- `artifacts/branchfabric/characterization/cpu-reference/attempts/continuum_state/attempt-000/continuum-trace/sharing-analysis.json` — SHA-256 `bf4565b2db19edfdf1a9cee7a3f41810d23885fccab910b515314ab4aece6a9e`; CPU_REFERENCE_MODEL_STATE; n=2; derived/replay.
- `artifacts/branchfabric/characterization/cpu-reference/attempts/continuum_state/attempt-000/continuum-trace/state-operation-trace-v1.jsonl` — SHA-256 `aa8c884f7d1e53ef824ceefc74528ea058c02de31ea6ab86db0d1271bd80fa7b`; CPU_REFERENCE_MODEL_STATE; n=18; raw samples.

## Required action

Terminate Hardware Path.

Functional-model and cycle-simulator work is allowed only after a hardware gate pass: **false**.
