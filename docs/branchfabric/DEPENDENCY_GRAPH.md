# SLOForge Helix Characterization dependency graph

Baseline source: `d6f77c839334a4644b4e2edf36a7543c68670a2d`

This graph is an execution contract. A node becomes ready only after its incoming
evidence edges validate. Raw artifacts are immutable inputs to analysis; analyses
write new derived files and may not rewrite measurements.

## Evidence classes

- `SYNTHETIC`: deterministic controlled workload or state-size fixture.
- `REPLAYED`: analysis of a retained trace without new system execution.
- `HARDWARE_BACKED_REAL`: a measured operation on the named physical hardware.
- `SIMULATED_HARDWARE`: deterministic physical-resource simulation.

Every experiment and derived claim carries exactly one of these labels. CPU
execution of a synthetic workload remains `SYNTHETIC`; the host counters around
that execution are separately `HARDWARE_BACKED_REAL` observations.

## Dependency DAG

```text
B0 immutable source, tests, demo, machine/software manifests
 |
 +--> T1 BranchWorkloadTrace v1 --------+
 +--> T2 StateOperationTrace v1 --------+--> V1 trace integrity/conformance
 +--> T3 storage + JSONL/Parquet/Perfetto+       + clock/drop validation
 |                                             |
 +--> I1 Helix lifecycle hooks ----------------+--> S1 first vertical trace
 +--> I2 Continuum state hooks ----------------+    capture -> fork -> COW
 +--> I3 environment/resource hooks -----------+    -> rollout/prune/checkpoint
                                                     -> migrate/reward/train/promote
                                                            |
                              +-----------------------------+
                              v
                   O1 disabled/minimal/full overhead
                              |
                              +--> W1 CPU workload matrix
                              +--> W2 flagship coding-agent run
                              +--> W3 environment matrix
                              +--> W4 COW/page-size/shared-root study
                              +--> W5 migration/delta/checkpoint study
                              +--> W6 metadata hot-path study
                              +--> W7 transform/integrity study
                              +--> W8 transport/multicast study
                              |
                              +--> H1 GPU capability gate
                                     +--> GPU / multi-GPU / multi-node runs
                                     +--> explicit unexercised manifests
                              |
                              v
                   A1 statistical reconstruction
                     + sharing/divergence/lifetime
                     + bottleneck decomposition
                     + Amdahl bounds
                     + state-operation roofline
                     + active experiment ranking
                              |
                              v
                   R1 evidence-backed requirements
                     + page/memory/queue/bandwidth/latency
                     + ISA classification
                     + placement matrix
                     + what-not-to-build
                              |
                              v
                   F1 independent replication and reviews
                              |
                              v
                   G1 full checks, clean-room regeneration,
                      corpus verification, final report
```

## Critical invariants

1. Instrumentation cannot change deterministic Helix or Continuum results.
2. Hot paths use bounded buffers; overflow and sampling are explicit events and
   manifest counters.
3. Measured time uses monotonic clocks. Cross-host normalization carries the
   Autopsy alignment confidence and uncertainty.
4. Physical bytes, transfers, bandwidth, cycles, and GPU time are absent rather
   than inferred when their counters are unavailable.
5. Synthetic GPU-state sizes are labeled `SYNTHETIC` at the value level; they
   never enter a hardware-backed aggregate.
6. Analysis consumes hash-verified immutable raw files and emits source artifact
   references for every summary.
7. Candidate hardware requirements are emitted only after the strongest
   reasonable software baseline is measured.
8. No node in this graph authorizes RTL, FPGA simulation, paid resources, or
   external deployment.

## Concurrency schedule

The environment exposes four total agent slots, so root plus three subagents is
the maximum. Initial clean worktrees are:

- `char/trace-schema` at `/Users/rishivinodkumar/sloforge-char-trace`
- `char/helix-instrumentation` at `/Users/rishivinodkumar/sloforge-char-helix`
- `char/continuum-instrumentation` at
  `/Users/rishivinodkumar/sloforge-char-continuum`

Root owns baseline manifests, CLI/Make integration, experiment orchestration,
artifact immutability, final statistical correctness, and requirement/report
publication. Completed lanes are replaced immediately while independent ready
nodes remain.

## Acceptance edges

| Edge | Acceptance evidence |
|---|---|
| B0 -> T/I | `make check`; isolated seed-41 CPU demo; immutable tag and manifests |
| T/I -> V1 | schema validation; Rust/Python golden round trip; ordering/hash/drop tests |
| V1 -> S1 | state and branch event graph reconciles to actual source artifacts |
| S1 -> O1 | identical Helix semantic hashes across trace levels |
| O1 -> W* | overhead and information-loss disposition published |
| W* -> A1 | repetitions, seeds, warmups, randomized run order, raw sample hashes |
| A1 -> R1 | every requirement value contains experiment, sample count, evidence class, statistic/confidence, and artifact reference |
| R1 -> F1 | five highest-leverage results replicated independently; reviewers receive immutable corpus |
| F1 -> G1 | high and reasonable medium findings closed or explicitly block the unsupported claim |

## Capability disposition at baseline

The current Apple M4 Pro host supports real CPU, unified-memory, filesystem, and
local-network observation. It has no NVIDIA/CUDA/PyTorch/vLLM/SGLang path, no
multi-GPU or multi-node inventory, and no characterization GPU budget. GPU,
multi-GPU, and multi-node nodes therefore resolve to hardware-ready commands and
explicit unexercised manifests unless compatible hardware becomes available;
their substitute experiments remain separately labeled synthetic or simulated.
