# SLOForge Helix Characterization Final Report

## Scope and verdict

This phase measured SLOForge Helix and Continuum to determine whether BranchFabric hardware can be designed without guessing. The evidence does **not** support beginning BranchFabric implementation. No RTL, HLS, FPGA simulator, device driver, SmartNIC firmware, DPU kernel, CXL device, or hardware ISA executor was created.

The defensible Phase 1 architecture is measurement plus optimized software. No candidate state operation is `REQUIRED` or `HIGH_VALUE`; numeric HBM, PCIe, NIC, memory-capacity, hardware-queue, hardware-latency, and universal-page-size requirements remain unknown or unavailable. This negative conclusion is the result of measurement, not a predetermined hardware justification.

## Baseline, revisions, and machine

- Immutable baseline: `d6f77c839334a4644b4e2edf36a7543c68670a2d`, tagged `sloforge-branchfabric-characterization-baseline-d6f77c8` (`artifacts/branchfabric/baseline/record.json`).
- Final measurement producer commit: `0f6f45629c0b40af46f8641d907b32a5a8c6a562`.
- Final requirements compiler commit: `d5cb9b2e1833d908ccdde5c2980ad48c00ed6b08`.
- Host: Apple M4 Pro, model Mac16,8, 12 logical CPUs, 25,769,803,776 bytes unified memory, macOS arm64 (`artifacts/branchfabric/manifests/hardware-baseline.json`).
- The Apple integrated GPU was not a compatible Helix CUDA path. CUDA GPU count was zero; CUDA, NVIDIA driver, PyTorch, vLLM, SGLang, NCCL, NVLink, RDMA, multi-GPU, and multi-node measurements were unavailable (`artifacts/branchfabric/hardware/status-seed-20260809.json`).
- `SLOFORGE_BRANCHFABRIC_CHARACTERIZATION_GPU_BUDGET_USD` was absent. No paid resource was created and no cloud resource remains active (`artifacts/branchfabric/hardware/status-seed-20260809.json`).

## Workloads and experiment counts

The measured realistic control flow is the existing deterministic CPU coding-agent fixture. It creates an exact BranchPoint, four semantically distinct sibling actions, model and environment forks, rollouts, three prunes, one success, reward, training, five evaluation spans including shadow/canary, promotion, and rollback. It is `SYNTHETIC` workload semantics with real host timing; simulated model/KV bytes are never presented as GPU measurements.

The final vertical run contains 70 BranchWorkloadTrace events and 17 StateOperationTrace events with zero drops and zero filtered events (`artifacts/branchfabric/characterization/cpu-reference/attempts/vertical_trace/attempt-000/vertical-trace/vertical-run.json`). The Continuum lifecycle contains 18 state operations: 12 host-real and six simulated-transport events, also with zero drops (`artifacts/branchfabric/characterization/cpu-reference/attempts/continuum_state/attempt-000/continuum-trace/continuum-trace-run.json`).

Controlled studies preserve these sample counts:

- 855 declared matrix cells, all analytically evaluated, zero hardware executed, `distribution_claim=false` (`artifacts/branchfabric/characterization/cpu-reference/attempts/matrix_validate/attempt-000/matrix-study.json`).
- 288 COW/page-size projections (`artifacts/branchfabric/analysis/cow/vertical-seed-41.json`).
- 945 raw metadata microbenchmark samples across 14 operations, implementations, and thread counts (`artifacts/branchfabric/metadata/seed-20260809/metadata-study.json`).
- 60 environment trials: five fixture classes, three trace levels, one warmup plus three measured repetitions (`artifacts/branchfabric/environment/seed-20260809/environment-study.json`).
- 378 raw optimized-software-baseline samples (`artifacts/branchfabric/software-baselines/seed-20260809/software-baselines.json`).
- 21 corrected overhead trials: three level-specific warmups plus six measured trials per trace level across seeds 41, 73, and 113 (`artifacts/branchfabric/overhead/helix-cpu-3seed-2rep-v5-canonical/instrumentation-overhead.json`).
- Independent replication of five central findings (`artifacts/branchfabric/replication/analysis/comparison.json`; `docs/branchfabric/reviews/INDEPENDENT_REPLICATION.md`).

These controlled cells do not have production sampling weights and are not pooled into a workload distribution.

## Trace corpus

BranchWorkloadTrace v1 and StateOperationTrace v1 are versioned, strict, hash-sealed schemas with Python and Rust conformance. The hot path uses a bounded buffer; attempted, accepted, filtered, and dropped events are explicit. JSONL is authoritative, Perfetto export is preserved, compact manifests bind file hashes, and Parquet fails explicitly when `pyarrow` is unavailable (`docs/branchfabric/TRACE_SPECIFICATION.md`).

The designated corpus contains 40 hash-indexed artifacts. Its corpus hash is recorded in `artifacts/branchfabric/CORPUS_MANIFEST.json`; the requirements compiler independently binds eight run artifacts and reports canonical trace-corpus hash `90ae8353fe3010f0eb006e138ce99fd2ae6c6d60483082a87bddfbe1ae76549e` (`artifacts/branchfabric/requirements/branchfabric_requirements.json`). Raw lifecycle observations, both canonical traces, exact trajectories, resource traces, sharing analyses, and Perfetto views are preserved beneath `artifacts/branchfabric/characterization/cpu-reference/attempts/`.

## Instrumentation overhead

The corrected overhead harness times the same raw-plus-canonical Pydantic recorder and bounded buffer as the vertical run. It performs level-specific warmups before randomized measured trials and holds the full resource sampler constant across disabled, minimal, and full levels.

Across six measured trials per level, median hot-path wall time was 11.344 s disabled, 11.378 s minimal, and 11.390 s full. Full tracing changed the paired median by +0.333%. Including JSONL and Perfetto persistence, full tracing changed the paired median by +0.387%. Median full persistence was 6.090 ms and 210,867 bytes. Median CPU time was 2.163 s disabled and 2.179 s full; no trace events or resource samples were dropped (`artifacts/branchfabric/overhead/helix-cpu-3seed-2rep-v5-canonical/instrumentation-overhead.json`; raw SHA-256 `492e90f1ea0ca35f4db4e664efacd06f9c08e20dad7f6967982a2eedd876c2df`).

This is a low-rate 87-event macro workload. It does not validate synchronous canonicalization at a future target command rate. TTFT, model-server token latency, GPU utilization, and GPU interference are unavailable. The earlier v4 campaign and replication remain preserved but are superseded because they timed only the raw in-memory recorder and used an invalid warmup schedule.

## Branch DAG, lifetime, sharing, and divergence

The final realistic slice has one fanout-four branch group, three pruned siblings, and one success. Branch lifetime p50 is 422.585 ms; the observed maximum is 423.181 ms. Shared-root lifetime is 374.237 ms with peak refcount four (`artifacts/branchfabric/analysis/workload/cpu-reference-final.json`). These siblings are correlated; machine-readable p95/p99 are `UNKNOWN`, not pseudo-tails (`artifacts/branchfabric/requirements/branchfabric_requirements.json`).

All four exact trajectories diverge at token index 0 and action index 0. That immediate-divergence case says nothing about production-weighted early/mid/late divergence (`artifacts/branchfabric/analysis/workload/cpu-reference-final.json`).

For model state, four siblings have 10,672 logical branch bytes, 8,852 shared logical bytes, 2,668 logical unique/physical bytes, 10,672 naive independent bytes, 75% sharing efficiency, and physical amplification 1.0 (`artifacts/branchfabric/characterization/cpu-reference/attempts/vertical_trace/attempt-000/vertical-trace/sharing-analysis.json`). These are tiny simulated model bytes, not HBM capacity evidence.

The live environment backend eagerly restores four private workspaces: 594 base logical bytes become 2,376 live physical bytes. Application-level CAS checkpoints record 1,303 private dirty bytes, but this is not filesystem block COW (`artifacts/branchfabric/characterization/cpu-reference/attempts/vertical_trace/attempt-000/vertical-trace/sharing-analysis.json`).

## COW, page size, migration, and checkpointing

Helix records four logical COW events, 1,303 written bytes, and 668 replaced old bytes. It exposes no OS or GPU page-fault counter. Continuum independently records 512 COW bytes over eight logical pages, two branches, 2,729 shared bytes, 455 private bytes per branch, 3,184 physical bytes, 6,368 naive bytes, 50% sharing efficiency, and physical amplification 1.0 (`artifacts/branchfabric/characterization/cpu-reference/attempts/continuum_state/attempt-000/continuum-trace/sharing-analysis.json`).

The page study is an allocation-only deterministic projection calibrated from an eight-token fixture at 256 simulated KV bytes/token. It cannot select a universal page size because real fault latency, metadata cost, and GPU allocation are absent. The machine requirement therefore leaves recommended page size `UNKNOWN`; 4 KiB is only the projection tie-break, not an architecture requirement (`artifacts/branchfabric/analysis/cow/vertical-seed-41.json`; `artifacts/branchfabric/requirements/branchfabric_requirements.json`).

Migration is not inside the Helix transaction. The Continuum artifact separately exercises snapshot, publish, fork, COW/append, delta, incremental checkpoint, conversion, one real host copy, three simulated-network transfers, commit, and reclaim. Because pre-migration preparation and the separate pre-copy transaction are not one causal critical path, no migration-latency Amdahl result is claimed. Physical GPU/network migration, dirty-rate time series, convergence rounds, recopy waste, pause/resume, and failure recovery remain unknown.

## Environment behavior

Five deterministic fixtures separately cover tiny, medium, large, dependency-heavy, and service-heavy environments. Disabled-tracing median total trial durations are 8.434 ms, 49.312 ms, 127.960 ms, 90.766 ms, and 28.098 ms respectively (`artifacts/branchfabric/environment/seed-20260809/environment-study.json`). The study exercises CAS capture, eager restore, whole-file replacement, checkpoint, comparison, declared service metadata, and teardown. It is not reflink/page COW, a live database server snapshot, sandbox/container startup, or process reconstruction.

## Transform, transport, multicast, and integrity

The producer declares one fused `STATE_RESHARD -> STATE_REPACK -> STATE_CHECKSUM` span over 3,444 bytes with 1.665 ms inclusive host latency. Per-stage latency is non-decomposable; temporary bytes and memory-read/write counters are unknown (`artifacts/branchfabric/analysis/transform/cpu-reference-v3.json`). No transpose, quantization, compression, or encryption event occurs.

The strongest controlled transform software baseline is a staged canonical path at 5.899 ms median versus 8.906 ms for the direct convert/capture path. Whole-buffer SHA-256 is 79.333 microseconds median and 3.304 GB/s logical throughput for a controlled 256 KiB payload. The selected in-process transfer baseline uses 64 KiB chunks and concurrency four at 296.708 microseconds median and about 883.5 MB/s (`artifacts/branchfabric/software-baselines/seed-20260809/software-baselines.json`). These are host baselines, not device/NIC results.

The final transport trace has 8,114 source bytes and 8,114 logical-delivery bytes: one 3,444-byte in-process host copy and 4,670 bytes across three simulated-network sends. Concurrency and fanout are one, retry bytes are zero, and the conservative detector finds zero multicast opportunities and zero reducible bytes at fanout thresholds 2 through 128 (`artifacts/branchfabric/analysis/transport/cpu-reference-v3.json`). Repeated-unicast is therefore not a measured bottleneck.

## Metadata behavior and software baselines

Representative one-thread medians are 31.635 microseconds for actual branch creation, 67.633 microseconds for actual abort, 1.590 microseconds for isolated page lookup, 2.531 microseconds for isolated dirty update, 4.224 microseconds for allocation, 28.496 microseconds for publish, and 1.816 microseconds for free (`artifacts/branchfabric/metadata/seed-20260809/metadata-study.json`). Batched baselines improve several paths, so any metadata hardware claim must beat the batched/sharded software baseline.

These microbenchmarks measure service capacity, not Helix arrival demand. `metadata.operations_per_second`, working-set bytes, and outstanding queue depth are consequently `UNKNOWN` in the requirements file. Cache misses, hardware cycles, lock-wait attribution, allocator attribution, and memory bandwidth were unavailable, so the roofline classifier correctly reports all high-frequency resource classes as unknown (`artifacts/branchfabric/analysis/roofline/cpu-reference-unknown.json`).

## End-to-end bottlenecks

The final inclusive waterfall is:

| Stage | Wall union |
| --- | ---: |
| EVALUATE | 10.518 s |
| REWARD | 351.186 ms |
| FORK | 48.943 ms |
| CHECKPOINT | 21.465 ms |
| CAPTURE | 15.603 ms |
| ROLLOUT | 10.294 ms |
| PRUNE | 0.781 ms |
| TRAIN | 0.566 ms |
| PROMOTE | 0.507 ms |
| BRANCH_READY | four completion markers; inclusive group-fork wait carried separately |

Production, migrate, and resume stages are absent. The 11.008-second run is evaluation dominated; state operations are secondary in this fixture. Inclusive stage spans are not summed as an exclusive critical path (`artifacts/branchfabric/analysis/workload/cpu-reference-final.json`).

The flagship is intentionally a composite rather than a false causal waterfall: the Helix run provides capture through promotion/rollback, while Continuum separately provides checkpoint/transfer/commit/reclaim. It has fanout four because the real fixture has four distinct candidate actions; duplicating candidates to manufacture fanout 32--128 would not be a credible workload measurement.

## Amdahl bounds

Only lifecycle-window sensitivity is defensible:

| Window | Primitive | Fraction | Free upper bound | Evidence |
| --- | --- | ---: | ---: | --- |
| Helix lifecycle | COW | 0.3840% | 1.00385x | host-real, n=3 trace instances |
| Helix lifecycle | reclamation | 0.2426% | 1.00243x | host-real, n=3 trace instances |
| Continuum lifecycle | reshard | 1.6694% | 1.01698x | host-real, n=1 |
| Continuum lifecycle | delta extraction | 0.6070% | 1.00611x | host-real, n=1 |
| Continuum lifecycle | transfer | 0.9340% | 1.00943x | simulated transport, n=1 |
| Continuum lifecycle | commit | 0.0948% | 1.00095x | host-real, n=1 |
| Continuum lifecycle | COW | 0.0212% | 1.00021x | host-real, n=1 |

No end-to-end bound exists for branch readiness, rollout throughput, migration latency, capacity reclamation, or the full Helix learning transaction. The requirements file therefore sets every candidate primitive's `expected_end_to_end_speedup` to `UNKNOWN` (`artifacts/branchfabric/analysis/amdahl/cpu-reference-v3-windowed/amdahl-analysis.json`; `artifacts/branchfabric/requirements/branchfabric_requirements.json`).

## BranchFabric primitive and placement decision

All 30 StateOperationTrace operations are classified exactly once:

- `REQUIRED` / `HIGH_VALUE`: none.
- `SOFTWARE_ONLY`: `STATE_HASH`, `STATE_ENCRYPT`, `STATE_DECRYPT`, `STATE_ACK`, `STATE_RETRY`, `STATE_ABORT`.
- `NOT_JUSTIFIED` in the bounded corpus: `STATE_ALLOC`, `STATE_MAP`, `STATE_READ`, `STATE_WRITE`, `STATE_TRANSPOSE`, `STATE_QUANTIZE`, `STATE_DEQUANTIZE`, `STATE_COMPRESS`, `STATE_DECOMPRESS`, `STATE_MULTICAST`.
- Unresolved: publish, fork, COW, append, snapshot, delta, reshard, producer-declared repack/checksum, send/receive, commit, reclaim, and free.

The placement matrix makes no numeric CPU/GPU/FPGA/DPU/CXL score because source/destination locality, sustained bandwidth, utilization, and implementation complexity inputs are unavailable (`artifacts/branchfabric/analysis/hardware-placement.json`). The current placement is CPU software. No FPGA or SmartNIC/DPU target is recommended.

## Memory, page, queue, bandwidth, and latency requirements

The evidence-backed requirements are negative constraints:

- Memory capacity: no HBM, DDR, CXL, metadata-cache, shared-root, suffix, transform-buffer, or network-buffer capacity can be sized. The 2,668-byte simulated root is conformance evidence only.
- Page size: no universal or state-specific hardware page size is selected.
- Queue depth: observed concurrency one is a serialized-fixture lower bound. Minimum, recommended, p99, pathological maximum, and backpressure policy are `UNKNOWN`.
- Bandwidth: GPU ingress/egress, internal memory, host interface, NIC ingress/egress, and storage mean/p95/p99/burst targets are unknown or unavailable.
- Latency: no target p50, target p99, or maximum tolerable primitive latency is established because no delay-sensitivity experiment on a required end-to-end objective exists.
- Consistency: one commit and no canonical abort event cannot determine a device transaction model. Ownership, epoch fencing, cancellation, retry, failure, tenant isolation, and recovery remain software-defined.

These nulls are explicit machine-readable requirements, not omitted work (`artifacts/branchfabric/requirements/branchfabric_requirements.json`).

## Features measurement says not to build

Do not build a Branch Translation Unit, hardware COW/page table, universal page size, state HBM/CXL store, multicast engine, hash/checksum block, compression block, encryption block, quantizer, environment snapshot accelerator, FPGA data plane, or DPU pipeline from this corpus. Repack/reshard/checksum streaming and transfer offload remain measurement candidates, not architecture commitments (`docs/branchfabric/WHAT_NOT_TO_BUILD.md`).

## Replication and reviews

Independent replication reproduced 75% model sharing, immediate token/action divergence, 1,303 environment bytes, the exact Continuum 18-operation multiset, 512-byte/eight-page COW, all 14 metadata medians within 11%, one fused conversion declaration, and zero multicast opportunity. The original wall-overhead sign did not reproduce; the corrected v5 campaign resolves the instrumentation-path and warmup defects and shows a small positive effect (`docs/branchfabric/reviews/INDEPENDENT_REPLICATION.md`).

The adversarial, systems/hardware, statistical-methodology, and instrumentation-overhead reviews are preserved under `docs/branchfabric/reviews/`. Their high findings drove the matrix execution labels, complete 30-operation ISA disposition, metadata demand/capacity separation, tail withholding, fused-chain preservation, GPU fail-closed behavior, lifecycle-window Amdahl labels, branch-ready/evaluation tracing, and corrected overhead rerun. The remaining review finding is lack of broader real/hardware workloads, not a hidden positive result.

## Remaining uncertainty

The decisive missing measurements are actual KV/recurrent/sampler sizes and physical pages; real GPU fork/readiness and append; GPU-to-host, host-to-GPU, and peer-GPU transfers; HBM/PCIe/NVLink contention; real NIC/RDMA fanout and retransmission; concurrent outstanding state operations; live filesystem/database/process environment snapshots; pause/resume during a serving spike; a single causal migration/reclamation transaction; and sampled real Helix branch distributions at meaningful context and fanout.

Those measurements require compatible hardware and workloads. The active prioritizer ranks them by uncertainty, sensitivity, Amdahl leverage, and cost, and marks currently unavailable experiments ineligible rather than assigning values (`artifacts/branchfabric/analysis/active-experiment-queue.json`).

## Reproducible commands and key paths

Primary workflow:

```text
make branchfabric-trace-check
make branchfabric-characterization-cpu
make branchfabric-characterization-gpu
make branchfabric-requirements
make branchfabric-report
make branchfabric-clean-room-test
make helix-check
make check
```

Core artifacts:

- Corpus index: `artifacts/branchfabric/CORPUS_MANIFEST.json`
- Requirements: `artifacts/branchfabric/requirements/branchfabric_requirements.json`
- Generated report: `reports/branchfabric-characterization/CHARACTERIZATION_REPORT.md`
- Design handoff: `docs/branchfabric/CHARACTERIZATION_DESIGN_BRIEF.md`
- Negative findings: `docs/branchfabric/WHAT_NOT_TO_BUILD.md`
- Final Helix trace: `artifacts/branchfabric/characterization/cpu-reference/attempts/vertical_trace/attempt-000/vertical-trace/`
- Final Continuum trace: `artifacts/branchfabric/characterization/cpu-reference/attempts/continuum_state/attempt-000/continuum-trace/`
- Workload/waterfall: `artifacts/branchfabric/analysis/workload/cpu-reference-final.json`
- Corrected overhead: `artifacts/branchfabric/overhead/helix-cpu-3seed-2rep-v5-canonical/`
- Windowed Amdahl: `artifacts/branchfabric/analysis/amdahl/cpu-reference-v3-windowed/amdahl-analysis.json`
- Transform: `artifacts/branchfabric/analysis/transform/cpu-reference-v3.json`
- Transport/multicast: `artifacts/branchfabric/analysis/transport/cpu-reference-v3.json`
- Hardware status: `artifacts/branchfabric/hardware/status-seed-20260809.json`
- Review set: `docs/branchfabric/reviews/`
