# BranchFabric GPU Validation Experiment 002

## Outcome

Status: `completed_fanout_8`. GPU-hours consumed: 0.564056.
Ledger-estimated cloud cost (upper bound): `$2.514374`.
Real physical shared-root semantics demonstrated: `true`.
Complete semantic smoke (including postflight HBM recovery): `true`.
This report does not claim FPGA justification.

## Runtime adapter and exposed state

Runtime: `vllm==0.23.0`. Model: `Qwen/Qwen2.5-7B-Instruct` at revision `a09a35458c702b33eeacc393d103063234e8bc28` with dtype `bfloat16`.
The adapter exposes CUDA-backed KV tensors, ordered physical block IDs, native refcounts, logical ranges, shared/private ownership, allocation epochs, exact-ID release evidence, KV assigned/reserved bytes, NVML/PyTorch memory, and hardware-backed state traces. The semantics are immutable shared-root plus append-only private suffixes, not general CUDA write-fault COW.

## Experiments actually run

| Attempt | Path | Fanout | Prefix | Suffix | Seed | GPU UUID | Trace | Lifecycle |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| modal-16k-f1-independent-s41-v2 | independent_prefill | 1 | 16384 | 256 | 41 | GPU-64a57df8-fee0-680d-5e7d-4bfcebb53ba0 | full | controller_worker / True |
| modal-16k-f1-shared-s41-v1 | shared_root | 1 | 16384 | 256 | 41 | GPU-deb8b618-7cad-6d61-9ad9-ed086117eee5 | full | controller_worker / True |
| modal-16k-f8-independent-s41-v1 | independent_prefill | 8 | 16384 | 256 | 41 | GPU-f012d28a-4b51-9fb0-0b37-9f17af0f7579 | full | controller_worker / True |
| modal-16k-f8-shared-s41-v1 | shared_root | 8 | 16384 | 256 | 41 | GPU-cc293a1a-588a-7834-43ff-a37738557bc6 | full | controller_worker / True |
| modal-smoke-shared-s41-v5 | shared_root | 2 | 2048 | 16 | 41 | GPU-652804ee-42ae-7c14-21ef-b8d29adacc11 | full | legacy_in_process / None |
| modal-smoke-shared-s41-v7 | shared_root | 2 | 2048 | 16 | 41 | GPU-bec5410c-8775-ff63-6588-c92d9cbf5d30 | full | controller_worker / True |

The materialized evidence contains 6 completed benchmark workload(s), 2 semantic smoke observation(s), and 4 accepted 16K trial(s). 5 workload(s) use the controller/worker boundary; 5 passed its authoritative driver-visible lifecycle gate. Earlier attempts remain listed as historical evidence.

## Real GPU semantic smoke observation

| Shared blocks / bytes | Private blocks / bytes | Root refs | Fork-added KV | All ready | Decode | Post-decode assigned KV | Post-root-delete assigned KV |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 128 / 112.00 MiB | 2 / 1.75 MiB | 2 | 0.00 MiB | 200.698 ms | 123.842 tok/s | 113.75 MiB | 0.00 MiB |
| 128 / 112.00 MiB | 2 / 1.75 MiB | 2 | 0.00 MiB | 181.133 ms | 126.971 tok/s | 113.75 MiB | 0.00 MiB |

The smoke assertions cover exact shared-root physical IDs, zero fork-time prefix allocation, private divergence, sibling survival after one branch is deleted, and complete runtime-assigned KV release after the final root is deleted.

## Paired 16K results

Physical HBM below means assigned physical KV state, not the pre-reserved vLLM KV pool or total driver-visible model memory.

| Fanout | Independent root-inclusive ready ms | Shared root-inclusive ready ms | Shared post-root ready ms | Aligned all-live phase | Independent aligned actual GiB | Shared aligned actual GiB | Aligned savings GiB | Aligned efficiency | Independent observed peak GiB | Independent full-256 projection GiB | Shared full-256 actual GiB | Projected terminal savings GiB | Projected terminal efficiency | Amplification | Floor gap MiB | Decode interference | Causal pair |
| ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 2448.125 | 3445.120 | 524.527 | suffix_256 | 0.8887 | 0.8887 | 0.0000 | 0.0000 | 0.8887 | 0.8887 | 0.8887 | 0.0000 | 0.0000 | 1.0000 | 0.00 | unavailable | False |
| 8 | 11843.974 | 2461.811 | 269.241 | suffix_64 | 7.0333 | 0.9023 | 6.1310 | 0.8717 | 7.0333 | 7.1094 | 0.9844 | 6.1250 | 0.8615 | 1.0000 | 0.00 | unavailable | False |

The shared root-inclusive readiness is a continuous wall-time measure from `root_session_create` start through `concurrent_decode` start plus the measured runtime readiness remaining after fork. It therefore includes the validation and instrumentation interval between fork completion and decode admission. The post-root value retains the fork/admission-through-decode-readiness view. The independent observed peak is the maximum direct physical snapshot; its comparable all-branches-live savings/efficiency uses the latest phase where all owners remain present in both paths. The full-fanout terminal value is explicitly an estimate from the exact surviving terminal branch pages times fanout, backed by cache-free, disjoint, full-prefill assertions. It is not mislabeled as a simultaneous direct observation.

## BranchFabric follow-up signals

- `state_management_fraction_of_readiness` at fanout 1: 0.0185 (none).
- `physical_allocation_above_shared_floor` at fanout 1: 0.0000 (none).
- `cpu_block_management_fraction_of_readiness` at fanout 1: 0.0953 (weak).
- `state_management_fraction_of_readiness` at fanout 8: 0.1717 (interest).
- `physical_allocation_above_shared_floor` at fanout 8: 0.0000 (none).
- `cpu_block_management_fraction_of_readiness` at fanout 8: 0.4828 (interest).
- `memory_copy_contends_with_decode` unavailable: copy-engine/CUPTI counters are unavailable.
- `hbm_bandwidth_reduces_rollout_concurrency` unavailable: nvidia-smi memory utilization is not attributable HBM bandwidth.
- `multi_gib_state_movement_on_critical_path` unavailable: allocation deltas do not prove physical byte movement.
- `decode_throughput_degradation` unavailable: observed per-path throughput is retained, but causal decode interference is unavailable: independent/shared calls used different physical GPU UUIDs; independent/shared calls used different physical GPU UUIDs and different A100 model variants.
- `readiness_superlinear_fanout_8_to_32` unavailable: valid fanout-8/fanout-32 shared trials on the same physical GPU UUID are required.

All accepted 16K trials used full tracing and no matched trace-disabled control was run. The state-management fraction is fork wall time divided by post-root readiness; the CPU fraction is process CPU time divided by wall readiness and is not a critical-path attribution. These are material diagnostic signals, not proof that metadata dominates execution.

## Recommended next GPU experiment

Classification: `E` — HIGH-FANOUT RUNTIME METADATA CHARACTERIZATION

E. HIGH-FANOUT RUNTIME METADATA CHARACTERIZATION: fanout-8 exposes a material unresolved runtime-metadata signal while physical KV remains at the exact sharing floor; paired trace-disabled controls are unavailable.

## Artifacts and cleanup

Every input was revalidated against its remote manifest and local-copy hash inventory. Controller/worker attempts additionally require a CUDA-clean parent, normal child and descendant termination, no forced GPU-process kill, zero final runtime-assigned KV bytes, and three consecutive process-free driver samples at or below baseline plus 1,024 MiB. Legacy in-process evidence remains readable but cannot authorize a 16K trial. Modal provider cleanup is reported separately from the stronger in-container process-lifecycle gate. Current provider cleanup status: `verified_all_experiment_apps_stopped_zero_tasks_and_no_running_containers`; verified: `true`. The earlier v5 provider audit remains historical evidence and is not promoted to current-campaign proof.
