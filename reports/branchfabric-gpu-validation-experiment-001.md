# BranchFabric GPU validation experiment 001

## Outcome

Experiment 001 stopped after remote safety and runtime-capability preflight, before model download, model load, or GPU workload launch. Repository commit `ab86d7600234f22f38fb71ae1d3a9d98bc048175` does not contain a real-runtime adapter connecting live GPU execution state to exact Helix BranchPoint capture, physical shared-root copy-on-write, and concurrent descendant decode. Running a stock runtime prefix-cache path or the CPU reference fork would have changed the system under test.

Consequently, no real-GPU BranchFabric measurements were collected. This is a runtime-integration blocker, not a positive or negative BranchFabric result, and it does not justify FPGA work.

## Machine and safety preflight

- Remote host: `mew0`; user: `rishivin`.
- GPUs detected: two NVIDIA A100 80GB PCIe devices. Both reported 0 MiB used, 0% GPU utilization, and no compute processes at preflight.
- GPU 0 (`GPU-9f432e61-badc-944e-6087-2f846d4c7e19`) was conditionally selected but never made GPU-active.
- Driver: 580.178.04. `nvidia-smi` reports driver compatibility with CUDA 13.0; no CUDA toolkit or `nvcc` was available.
- `/tmp` had approximately 95 GiB available. A single isolated root, `/tmp/sloforge-branchgpu-ZhfJJg8F`, was created.
- All experiment cache, model, environment, trace, result, log, and temporary paths were rooted beneath that directory. No global environment or cache was modified.

## Model and runtime

No model was selected, downloaded, or loaded. No runtime was used, so runtime version, model revision, tokenizer, dtype, and quantization are not applicable.

vLLM was the strongest candidate surface, but its SLOForge binding exposes native KV-transfer configuration and explicitly rejects complete Continuum live-state export. SGLang exposes prefill/decode launch configuration and likewise rejects complete export. Genesis does not publish an active execution-state export contract. The repository's future measurement plan says the required `branchfabric-real-fanout` command is intentionally absent and that a real-runtime adapter is a prerequisite.

The existing Continuum fork is a deterministic reference path: it restores, decodes, re-encodes, and publishes content-addressed checkpoint state per child. Its COW refcounts and shared digests are logical/CAS semantics, not physical GPU KV-page COW. Installing stock vLLM or SGLang would not add the missing BranchPoint/state integration.

## Requested and executed experiments

Requested configuration: 16,384-token prefix, 256-token continuation, fanouts 1, 8, and 32, starting with seed 41, on one GPU with a 4 GPU-hour hard limit.

Executed GPU experiments: none. The fanout-1 pilot, fanout-8 trial, fanout-32 adaptive point, independent-prefill baseline, optimized shared-root COW baseline, and concurrent-decode COW baseline were all not launched. Actual GPU-hours were **0.0**.

## Measurements

Branch readiness was not collected. P50, p95, p99, per-branch readiness, all-branches-ready latency, first decode start, and first emitted token are all null.

HBM and state sharing were not collected. Experimental HBM phase samples, logical and physical KV bytes, private state, allocator overhead, per-branch growth, sharing efficiency, physical amplification, memory saved, and physical bytes avoided are all null. The 0 MiB preflight reading is environment evidence, not an experimental HBM baseline.

State movement and COW were not collected. GPU copies, host/device movement, copy counts and sizes, materialized bytes, private allocations, root lifetime, and refcounts are all null.

GPU/decode interference was not collected. Decode throughput, interference fraction, SM utilization, HBM bandwidth, copy-engine utilization, power, and clocks are all null.

BranchWorkloadTrace, StateOperationTrace, dropped-event counts, trace-buffer overflow, and tracing overhead were not measured. No tracing-disabled control ran.

Because the underlying measurements are absent, sharing efficiency, physical amplification, branch-readiness overhead, state-management fraction, decode-interference fraction, prefill compute avoided, and marginal fanout metrics were not calculated. Null values must not be interpreted as zero.

## Early stop and interest conditions

The early-stop condition was `runtime_or_model_incompatibility_makes_measurement_invalid`. Stopping before the pilot avoided consuming GPU time on a system different from the one specified.

BranchFabric interest conditions are **not evaluable** and `triggered` is null. This report must not be read as saying those conditions did or did not trigger.

## Plots

The four requested plots were not generated because no valid datapoints exist. The plot-status artifacts record each omission. Zero-valued or theoretical-only plots would misrepresent missing data as a hardware result.

## Unexpected findings and limitations

Hardware availability was not the blocker; both A100s were idle during preflight. The primary blocker was the absent semantic runtime integration. Missing remote PyTorch/vLLM/SGLang packages were secondary because installing packages alone would not implement exact BranchPoint/COW behavior.

The process ledger covers long-running experiment work and is empty because no server, profiler, background job, model process, container, or GPU workload was started. Ephemeral SSH, rsync, and audit-shell PIDs were not comprehensively recorded, a limitation against the literal every-process accounting requirement. Exact post-cleanup scans nevertheless found no experiment-owned process or path.

## Next recommended experiment

Do not launch another GPU experiment until a bounded, feature-flagged vLLM or SGLang adapter has demonstrated exact live-state capture, runtime-native GPU KV page/block sharing, first-write materialization, per-branch readiness, concurrent descendant decode, and StateOperationTrace emission. Once that software prerequisite exists and is tested, rerun the 16K fanout-1 control followed by one fanout-8 trial and apply the original adaptive gate.

## Cleanup

All nine required remote evidence files were copied locally and matched against remote SHA-256 hashes before deletion. The isolated remote root was removed at 2026-08-10T22:41:14Z. A primary audit and an independent cleanup reviewer verified that the root, experiment paths, processes, GPU processes, sockets, and shared-memory objects were absent. Two root-owned `live_migrator` GPU processes appeared during the independent review and were unrelated; they were not touched. A final fresh SSH audit passed at 2026-08-10T22:59:21Z with both GPUs again at 0 MiB and no compute processes. Docker daemon visibility was permission-denied, but no containers were used and exact process/path scans were clean.

`remaining_experiment_owned_artifacts` is `[]`. The cleanup receipt is at `artifacts/branchfabric/gpu-validation/experiment-001/REMOTE_CLEANUP_RECEIPT.json`.
