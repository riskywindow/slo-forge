# BranchFabric hardware gate report

Phase: **final**.
Outcome: **FAIL_NO_BUILD**.
Hardware implementation allowed: **false**.

No threshold was loosened. CPU-reference, synthetic, simulated-hardware, and artifact-replay evidence cannot satisfy the target-hardware requirement.

## Candidate results

| Candidate | Real evidence | End-to-end relevance | Headroom | Platform | Workload value | Disposition |
|---|---|---|---|---|---|---|
| checkpoint_transform_transfer_chain | FAIL | PASS | FAIL | FAIL | FAIL | NOT_JUSTIFIED |
| shared_root_cow | FAIL | FAIL | FAIL | FAIL | FAIL | NOT_JUSTIFIED |
| one_to_many_multicast | FAIL | FAIL | FAIL | FAIL | FAIL | NOT_JUSTIFIED |
| branch_translation_metadata | FAIL | FAIL | FAIL | FAIL | FAIL | NOT_JUSTIFIED |

## Candidate calculations and evidence

### `checkpoint_transform_transfer_chain`

- `mandatory_real_evidence`: **FAIL** — no target-relevant real hardware raw samples; fewer than two workload classes.
- `mandatory_end_to_end_relevance`: **PASS** — passing measured fractions: state-movement time.
- `mandatory_system_level_headroom`: **FAIL** — no lower confidence bound reached a threshold; all are unknown.
- `mandatory_platform_feasibility`: **FAIL** — missing: byte rate, operation rate, latency target, bandwidth target, credible resource estimate, selected-target fit.
- `workload_value_data_path`: **FAIL** — missing: large_state_operations, bandwidth_pressure, temporary_memory_or_interference, full_system_leverage.

Evidence bindings:

- `artifacts/branchfabric/execution/reclamation/raw/trials.jsonl` — SHA-256 `795a6803c1123555ea5804ddfeca711805235b89eeb30c5e723299bd26e74408`; CPU_REFERENCE_MODEL_STATE; n=36; raw samples.
- `artifacts/branchfabric/execution/reclamation/analysis.json` — SHA-256 `cbd27f984fa412cd33fd3714b66f8978ab3311736127058b04efc2fdab0adbf0`; ARTIFACT_REPLAY; n=36; derived/replay.
- `artifacts/branchfabric/manifests/hardware-baseline.json` — SHA-256 `c7b2c3b6aee95211385a5e62a6d0a98ea5c0b39aaabfb7750046cc51fe72f238`; LOCAL_CPU_REAL; n=1; derived/replay.
- `artifacts/branchfabric/manifests/software-baseline.json` — SHA-256 `f4b84bd48383b242af4b996e8ed2e77c3d9cc1187ad8d31f4a6494bc569c8edd`; ARTIFACT_REPLAY; n=1; derived/replay.

### `shared_root_cow`

- `mandatory_real_evidence`: **FAIL** — no target-relevant real hardware raw samples; tracing overhead not controlled.
- `mandatory_end_to_end_relevance`: **FAIL** — no measured fraction reached a threshold; BranchPoint-to-readiness=0.000060 (raw-bound).
- `mandatory_system_level_headroom`: **FAIL** — no lower confidence bound reached a threshold; all are unknown.
- `mandatory_platform_feasibility`: **FAIL** — missing: byte rate, operation rate, working set, latency target, bandwidth target, fault behavior, interface requirement, credible resource estimate, selected-target fit.
- `workload_value_state_sharing`: **FAIL** — missing: real_physical_state_measured, representative_fanout, shared_root_lifetime_material, allocation_savings_affect_capacity_or_throughput, software_cow_insufficient.

Evidence bindings:

- `artifacts/branchfabric/execution/fanout/raw-samples.jsonl` — SHA-256 `af616379666cc4d9043be355976016cc06a57a87b07caa8b60b2ad69d901b810`; CPU_REFERENCE_MODEL_STATE; n=144; raw samples.
- `artifacts/branchfabric/execution/fanout/summary.json` — SHA-256 `51768b2b164caacb65d0de4ca3307737c4eda8c83cef030a89dd2597fcffebc4`; ARTIFACT_REPLAY; n=144; derived/replay.
- `artifacts/branchfabric/manifests/hardware-baseline.json` — SHA-256 `c7b2c3b6aee95211385a5e62a6d0a98ea5c0b39aaabfb7750046cc51fe72f238`; LOCAL_CPU_REAL; n=1; derived/replay.
- `artifacts/branchfabric/manifests/software-baseline.json` — SHA-256 `f4b84bd48383b242af4b996e8ed2e77c3d9cc1187ad8d31f4a6494bc569c8edd`; ARTIFACT_REPLAY; n=1; derived/replay.

### `one_to_many_multicast`

- `mandatory_real_evidence`: **FAIL** — no target-relevant real hardware raw samples; tracing overhead not controlled.
- `mandatory_end_to_end_relevance`: **FAIL** — no measured fraction reached a threshold; all are unknown.
- `mandatory_system_level_headroom`: **FAIL** — no lower confidence bound reached a threshold; all are unknown.
- `mandatory_platform_feasibility`: **FAIL** — missing: byte rate, operation rate, concurrency, queue depth, working set, latency target, bandwidth target, fault behavior, interface requirement, credible resource estimate, selected-target fit.
- `workload_value_fanout`: **FAIL** — missing: repeated_physical_unicast, identical_large_source, unicast_material_to_bytes_or_latency, transforms_preserve_benefit.

Evidence bindings:

- `artifacts/branchfabric/execution/fanout/raw-samples.jsonl` — SHA-256 `af616379666cc4d9043be355976016cc06a57a87b07caa8b60b2ad69d901b810`; CPU_REFERENCE_MODEL_STATE; n=144; raw samples.
- `artifacts/branchfabric/execution/fanout/summary.json` — SHA-256 `51768b2b164caacb65d0de4ca3307737c4eda8c83cef030a89dd2597fcffebc4`; ARTIFACT_REPLAY; n=144; derived/replay.
- `artifacts/branchfabric/analysis/transport/cpu-reference-v3.json` — SHA-256 `7760078aa431e8822225e6805e58be6862b2a47f4c1ee0f60e71983b345e3e80`; ARTIFACT_REPLAY; n=4; derived/replay.
- `artifacts/branchfabric/manifests/hardware-baseline.json` — SHA-256 `c7b2c3b6aee95211385a5e62a6d0a98ea5c0b39aaabfb7750046cc51fe72f238`; LOCAL_CPU_REAL; n=1; derived/replay.
- `artifacts/branchfabric/manifests/software-baseline.json` — SHA-256 `f4b84bd48383b242af4b996e8ed2e77c3d9cc1187ad8d31f4a6494bc569c8edd`; ARTIFACT_REPLAY; n=1; derived/replay.

### `branch_translation_metadata`

- `mandatory_real_evidence`: **FAIL** — no target-relevant real hardware raw samples; fewer than two workload classes; fewer than three seeds.
- `mandatory_end_to_end_relevance`: **FAIL** — no measured fraction reached a threshold; target critical path=0.000948 (raw-bound).
- `mandatory_system_level_headroom`: **FAIL** — no lower confidence bound reached a threshold; all are unknown.
- `mandatory_platform_feasibility`: **FAIL** — missing: byte rate, operation rate, queue depth, working set, latency target, bandwidth target, fault behavior, interface requirement, credible resource estimate, selected-target fit.
- `workload_value_metadata`: **FAIL** — missing: optimized_software_metadata_is_critical, operation_rate_and_concurrency_measured, cache_lock_allocation_pressure_measured, hardware_working_set_and_queue_derived, full_system_leverage.

Evidence bindings:

- `artifacts/branchfabric/metadata/seed-20260809/metadata-study.json` — SHA-256 `d46a31b60147dc86427295de4c7de12c9078df44034aafdd41e518cfd1a1db8f`; SYNTHETIC; n=840; raw samples.
- `artifacts/branchfabric/analysis/amdahl/cpu-reference-v3-windowed/amdahl-analysis.json` — SHA-256 `a3012b3532c50ffb792ae573f8ece0ff6b6dcc77201055c73b52817dc72b62f4`; ARTIFACT_REPLAY; n=1; derived/replay.
- `artifacts/branchfabric/characterization/cpu-reference/attempts/continuum_state/attempt-000/continuum-trace/state-operation-trace-v1.jsonl` — SHA-256 `aa8c884f7d1e53ef824ceefc74528ea058c02de31ea6ab86db0d1271bd80fa7b`; CPU_REFERENCE_MODEL_STATE; n=18; raw samples.
- `artifacts/branchfabric/manifests/hardware-baseline.json` — SHA-256 `c7b2c3b6aee95211385a5e62a6d0a98ea5c0b39aaabfb7750046cc51fe72f238`; LOCAL_CPU_REAL; n=1; derived/replay.
- `artifacts/branchfabric/manifests/software-baseline.json` — SHA-256 `f4b84bd48383b242af4b996e8ed2e77c3d9cc1187ad8d31f4a6494bc569c8edd`; ARTIFACT_REPLAY; n=1; derived/replay.

## Required action

Terminate Hardware Path.

Functional-model and cycle-simulator work is allowed only after a hardware gate pass: **false**.
