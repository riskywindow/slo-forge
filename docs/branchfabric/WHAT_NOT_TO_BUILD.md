# What Not to Build in BranchFabric

## Decision

The current corpus does **not** make any BranchFabric hardware primitive
`REQUIRED`. No proposed primitive is classified `STRONGLY JUSTIFIED` or
`JUSTIFIED` for hardware implementation. This is a negative finding, not a
claim that branch-aware state sharing is unimportant: the controlled fixtures
show sharing, but they do not establish a hardware bottleneck, a production
rate, or a GPU/network critical path.

The correct next step is to retain the StateOperationTrace vocabulary as a
software and architecture-replay contract and gather the missing hardware
measurements. It is not evidence for assigning every trace operation a hardware
opcode.

## How to read the classifications

These classifications apply to a **BranchFabric hardware implementation**, not
to whether Helix or Continuum needs the corresponding software semantics.

| Classification | Meaning in this document |
| --- | --- |
| `STRONGLY JUSTIFIED` | Repeated hardware-backed workloads show material end-to-end leverage, and a strong software baseline does not remove it. |
| `JUSTIFIED` | The evidence supports hardware implementation, although placement or detailed parameters may remain open. |
| `UNCERTAIN` | The operation exists and could matter, but the measured scale, hardware path, or end-to-end leverage is missing or nonrepresentative. |
| `NOT CURRENTLY JUSTIFIED` | The present corpus does not support spending hardware area or interface complexity on the feature. A materially different future corpus could change the decision. |
| `SHOULD REMAIN SOFTWARE` | Current host measurements and Amdahl bounds favor the measured or optimized software path; hardware is not warranted for the observed workload. |

Workload evidence and timing evidence are independent. The Helix and Continuum
fixtures are `SYNTHETIC`; host operation timing on the Apple M4 Pro is
`HARDWARE_BACKED_REAL`; model/KV sizes in the Helix fixture are simulated; and
the Continuum pre-copy network timings are `SIMULATED_HARDWARE`. The capability
report records zero compatible CUDA GPUs and no multi-GPU or multi-node path.
Therefore this document makes no HBM, PCIe, NVLink, RDMA, NIC-bandwidth, or GPU
kernel claim. See
`artifacts/branchfabric/hardware/status-seed-20260809.json`.

## Proposed feature disposition

| Proposed BranchFabric feature | Classification | Evidence-based disposition |
| --- | --- | --- |
| Hardware branch page table / Branch Translation Unit | `NOT CURRENTLY JUSTIFIED` | The only Helix group has fanout 4, all observed state-operation concurrency is 1, physical page faults are unavailable, and no cache-miss, lock-contention, or production-rate evidence establishes a metadata engine bottleneck. |
| Hardware COW engine | `SHOULD REMAIN SOFTWARE` | COW is semantically useful, but making it free bounds full-transaction speedup at 1.00345x and migration speedup at 1.00018x in the measured synthetic paths. |
| Hardware state allocation | `SHOULD REMAIN SOFTWARE` | Current Continuum software allocation measured a 4.224 microsecond median and about 236.7 thousand operations/s at one thread; no end-to-end allocation bottleneck was observed. |
| Hardware state reclamation / free | `SHOULD REMAIN SOFTWARE` | Free measured 1.816 microseconds median and reclaim 3.354 microseconds in the metadata microstudy. Free reclamation bounds speedup at 1.00249x for the synthetic full transaction and 1.00014x for migration. |
| Hardware dirty tracker | `SHOULD REMAIN SOFTWARE` | The isolated software dirty update measured 2.531 microseconds median at one thread. No real GPU dirty rate or page-fault stream was captured. |
| Hardware delta extraction | `NOT CURRENTLY JUSTIFIED` | One host-timed synthetic delta event was observed; making it free bounds the synthetic migration speedup at only 1.00574x. Real HBM dirty volume and pre-copy convergence remain unmeasured. |
| Hardware transaction commit / abort | `SHOULD REMAIN SOFTWARE` | Commit contributes 0.0756% of the measured synthetic migration path, for a free-operation bound of 1.00076x. The software metadata study also exercises actual Continuum abort and epoch-validation paths. |
| Hardware state hash / checksum | `SHOULD REMAIN SOFTWARE` | The strongest measured host baseline is whole-buffer `hashlib` SHA-256 at a 79.333 microsecond median and 3.304 GB/s logical throughput for a 256 KiB controlled payload. Hash/checksum is absent from the reported end-to-end Amdahl leverage set. |
| Hardware compression / decompression | `NOT CURRENTLY JUSTIFIED` | No `STATE_COMPRESS` or `STATE_DECOMPRESS` event appears in the measured Helix or Continuum traces, and no compression ratio or critical-path contribution was measured. |
| Hardware encryption / decryption | `NOT CURRENTLY JUSTIFIED` | No `STATE_ENCRYPT` or `STATE_DECRYPT` event, byte volume, or security-domain crossing cost was measured. |
| Hardware multicast | `NOT CURRENTLY JUSTIFIED` | Four sends were recorded, all fanout 1. The conservative dependency-aware analysis found zero multicast opportunities and zero reducible bytes for fanouts 2 through 128. |
| Network / DPU transfer offload | `UNCERTAIN` | The trace contains one 3,434-byte in-process host copy and three deterministic simulated-network deliveries totaling 4,653 bytes. The transfer-free migration bound is 1.01008x, but that result uses simulated hardware timing and is not a NIC result. |
| Hardware reshard | `UNCERTAIN` | Reshard is the largest measured state primitive in the migration Amdahl report, but it is one 3,434-byte synthetic host event. Making it free bounds that path at 1.02349x; no GPU layout or HBM movement was measured. |
| Hardware repack | `NOT CURRENTLY JUSTIFIED` | No standalone `STATE_REPACK` event was captured, so frequency, bytes, temporary storage, and end-to-end leverage are unknown. |
| Hardware transpose | `NOT CURRENTLY JUSTIFIED` | No `STATE_TRANSPOSE` event was captured. |
| Hardware quantize / dequantize | `NOT CURRENTLY JUSTIFIED` | No `STATE_QUANTIZE` or `STATE_DEQUANTIZE` event was captured and no destination-specific precision conversion was measured. |
| Fused reshard/repack/checksum/send pipeline | `UNCERTAIN` | The dependency analysis found only one single-stage `STATE_RESHARD` chain, processing 3,434 bytes. There is no measured multi-stage chain to fuse. A controlled software transform baseline reduced median duration from 8.906 ms to 5.899 ms for its small reference conversion, so software layout work remains a required comparator. |
| Environment snapshot accelerator | `SHOULD REMAIN SOFTWARE` | The reference backend eagerly restores private workspaces and uses content-addressed incremental checkpoints. The environment study covers five synthetic fixtures on the host, but measures neither filesystem page COW nor external process/server startup. |
| Dedicated shared-root HBM/CXL store | `UNCERTAIN` | A four-branch synthetic Helix group achieved 75% model-state sharing and kept its shared root live for 339.063 ms, but its 2,660-byte model state is simulated. Capacity and residence tier cannot be derived from it. |
| Fixed universal hardware page size | `NOT CURRENTLY JUSTIFIED` | The analytical allocation-only sweep selects 4 KiB, but explicitly rejects a universal recommendation because metadata cost and real GPU page-fault latency are unmeasured. |

The measurements supporting this table are preserved in:

- `artifacts/branchfabric/characterization/vertical-seed-41-final/sharing-analysis.json`
- `artifacts/branchfabric/analysis/workload/helix-coding-agent-seed-41.json`
- `artifacts/branchfabric/continuum/seed-41/sharing-analysis.json`
- `artifacts/branchfabric/continuum/seed-41/state-operation-trace-v1.jsonl`
- `artifacts/branchfabric/analysis/cow/vertical-seed-41.json`
- `artifacts/branchfabric/analysis/transport/continuum-seed-41.json`
- `artifacts/branchfabric/analysis/transform/continuum-seed-41.json`
- `artifacts/branchfabric/analysis/amdahl/cpu-reference.json`
- `artifacts/branchfabric/metadata/seed-20260809/metadata-study.json`
- `artifacts/branchfabric/environment/seed-20260809/environment-study.json`
- `artifacts/branchfabric/software-baselines/seed-20260809/software-baselines.json`

## Why sharing does not yet justify sharing hardware

The controlled four-branch Helix fixture allocated 2,660 physical model-state
bytes instead of 10,640 bytes for naive independent copies, for 75% sharing
efficiency and 1.0 physical amplification. Its branches diverged at token index
0 and action index 0. Four COW events wrote 1,303 environment bytes, while the
backend eagerly materialized 2,376 workspace bytes and exposed no physical COW
fault counter. The result demonstrates that the software state model can express
shared roots and private suffixes. It does not establish a GPU page-table or
HBM-capacity bottleneck because the state bytes are simulated and tiny. Exact
evidence is in
`artifacts/branchfabric/characterization/vertical-seed-41-final/sharing-analysis.json`
and
`artifacts/branchfabric/analysis/workload/helix-coding-agent-seed-41.json`.

The separate Continuum fixture has two branches, 3,176 logical unique and
physical allocated bytes, 6,352 naive independent bytes, 50% sharing
efficiency, 512 COW bytes across eight logical pages, and concurrency 1. This is
useful validation of lifecycle semantics, not a production distribution. Exact
evidence is in
`artifacts/branchfabric/continuum/seed-41/sharing-analysis.json` and
`artifacts/branchfabric/continuum/seed-41/continuum-trace-run.json`.

The page-size artifact contains 288 deterministic projections calibrated from
an eight-token synthetic fixture at 256 simulated KV bytes/token. Its
allocation-only answer is 4 KiB, but the artifact itself states that metadata
cost and real GPU page-fault latency are unmeasured and that a universal page
size is unsupported. A hardware page size must not be selected from this sweep.
See `artifacts/branchfabric/analysis/cow/vertical-seed-41.json`.

## Why metadata should remain software for this corpus

The metadata study preserves seven measured samples per implementation and
thread count, with no outlier removal. At one thread, representative medians
for actual Continuum software are:

| Operation | Median | Median throughput |
| --- | ---: | ---: |
| Allocation | 4.224 microseconds/op | 236,744 ops/s |
| Publish | 28.496 microseconds/op | 35,093 ops/s |
| Epoch validation | 6.680 microseconds/op | 149,708 ops/s |
| Free | 1.816 microseconds/op | 550,538 ops/s |
| Reclaim | 3.354 microseconds/op | 298,138 ops/s |
| Abort | 67.633 microseconds/op | 14,786 ops/s |

Faithful isolated page lookup measured 1.590 microseconds median and 628,993
ops/s at one thread. The software comparison also shows that batching improves
several isolated operations: for example, one-thread ancestry lookup throughput
increased by 1.230x and one-thread page lookup by 1.157x relative to the matched
global-lock comparator. These are microbenchmarks, so they do not prove that
the production working set is cache resident. They do show that a hardware
metadata unit must first beat a credible batched software baseline and must be
justified by production operation rates. Neither condition is met in the
current corpus. Source:
`artifacts/branchfabric/metadata/seed-20260809/metadata-study.json`.

## Why hashing, transforms, and multicast do not yet justify pipelines

The controlled software-baseline payload is 256 KiB with seven measured trials
after two warmups. Whole-buffer SHA-256 measured 79.333 microseconds median and
3.304 GB/s logical throughput. Hardware hashing has no measured end-to-end
fraction to accelerate. Source:
`artifacts/branchfabric/software-baselines/seed-20260809/software-baselines.json`.

The same artifact compares exact reference conversions. Its selected staged
software transform measured 5.899 ms median versus 8.906 ms for the current
direct capture/convert path. This is not a GPU kernel comparison; it indicates
that software staging/layout choices are still material. The trace-based
transform study found only one reshard stage, not the proposed
`READ -> DEQUANTIZE -> RESHARD -> REPACK -> CHECKSUM -> SEND` chain, and lacks
memory-read, memory-write, temporary-allocation, and GPU-time counters. Source:
`artifacts/branchfabric/analysis/transform/continuum-seed-41.json`.

The transport analysis selected eight events with 8,087 source and receive
payload bytes, no retry bytes, concurrency 1, and fanout 1. It found no exact
multicast opportunity. Host-only copy scheduling experiments vary in whether a
binary tree or repeated unicast wins by fanout: repeated unicast wins at 2, 8,
32, and 64, while the binary tree wins at 4, 16, and 128. Those results are
host-memory scheduling measurements, not NIC multicast measurements. Sources:
`artifacts/branchfabric/analysis/transport/continuum-seed-41.json` and
`artifacts/branchfabric/software-baselines/seed-20260809/software-baselines.json`.

## Amdahl guardrail

The Amdahl report uses eight samples across two synthetic objectives. Its
largest host-timed free-operation bound is 1.02349x for one reshard event in the
synthetic migration path. Other effectively-free bounds are:

| Objective | Primitive | Measured fraction | Effectively-free upper bound |
| --- | --- | ---: | ---: |
| Full Helix transaction | COW handling | 0.3438% | 1.00345x |
| Full Helix transaction | Reclamation | 0.2481% | 1.00249x |
| Migration | Reshard | 2.2955% | 1.02349x |
| Migration | Delta extraction | 0.5709% | 1.00574x |
| Migration | Transfer | 0.9984% | 1.01008x |
| Migration | Transaction commit | 0.0756% | 1.00076x |
| Migration | COW handling | 0.0176% | 1.00018x |
| Migration | Reclamation | 0.0139% | 1.00014x |

The transfer row uses `SIMULATED_HARDWARE` timing. All other rows time host
software executing synthetic state. No row supports a GPU, NIC, FPGA, or CXL
placement decision. Source:
`artifacts/branchfabric/analysis/amdahl/cpu-reference.json`.

## Minimal ISA candidate disposition

This table answers whether each candidate should become a hardware operation
now. Keeping an operation in StateOperationTrace for future replay does not
change its disposition.

| Candidate operation | Classification | Hardware decision now |
| --- | --- | --- |
| `STATE_ALLOC` | `SHOULD REMAIN SOFTWARE` | Use the measured allocator; preserve trace coverage. |
| `STATE_MAP` | `SHOULD REMAIN SOFTWARE` | No mapped-page hardware bottleneck was measured. |
| `STATE_PUBLISH` | `SHOULD REMAIN SOFTWARE` | One synthetic Continuum lifecycle exercises publish; production rates and hardware locality are unknown. |
| `STATE_FORK` | `UNCERTAIN` | Fork is semantically central, but measured state is tiny/simulated and environment fork is eager host restore. |
| `STATE_COW` | `SHOULD REMAIN SOFTWARE` | Keep software COW and tracing; the measured Amdahl leverage is negligible. |
| `STATE_APPEND` | `SHOULD REMAIN SOFTWARE` | One 1,130-byte synthetic host event is insufficient for a hardware append path. |
| `STATE_READ`, `STATE_WRITE` | `SHOULD REMAIN SOFTWARE` | No explicit trace events establish benefit beyond ordinary memory access. |
| `STATE_SNAPSHOT` | `UNCERTAIN` | Snapshots are observed, but model bytes are simulated and environment snapshots are host filesystem/CAS operations. |
| `STATE_DELTA` | `NOT CURRENTLY JUSTIFIED` | One synthetic host delta and a 1.00574x free bound do not warrant an opcode. |
| `STATE_RESHARD` | `UNCERTAIN` | Highest measured state leverage, but only one 3,434-byte host event and no GPU layout. |
| `STATE_TRANSPOSE` | `NOT CURRENTLY JUSTIFIED` | No observed event. |
| `STATE_REPACK` | `NOT CURRENTLY JUSTIFIED` | No standalone observed event. |
| `STATE_QUANTIZE`, `STATE_DEQUANTIZE` | `NOT CURRENTLY JUSTIFIED` | No observed event. |
| `STATE_COMPRESS`, `STATE_DECOMPRESS` | `NOT CURRENTLY JUSTIFIED` | No observed event or compression distribution. |
| `STATE_HASH`, `STATE_CHECKSUM` | `SHOULD REMAIN SOFTWARE` | Strong host SHA-256 baseline; no Amdahl contribution. |
| `STATE_ENCRYPT`, `STATE_DECRYPT` | `NOT CURRENTLY JUSTIFIED` | No observed event or measured security-domain cost. |
| `STATE_SEND`, `STATE_RECEIVE` | `UNCERTAIN` | Preserve software transport APIs; hardware placement awaits real NIC/GPU measurements. |
| `STATE_MULTICAST` | `NOT CURRENTLY JUSTIFIED` | Zero measured opportunities and zero reducible bytes. |
| `STATE_ACK`, `STATE_RETRY` | `SHOULD REMAIN SOFTWARE` | No retry traffic was observed; retain protocol handling in software. |
| `STATE_COMMIT`, `STATE_ABORT` | `SHOULD REMAIN SOFTWARE` | Transaction semantics matter, but measured hardware leverage does not. |
| `STATE_RECLAIM`, `STATE_FREE` | `SHOULD REMAIN SOFTWARE` | Low measured latency and negligible Amdahl leverage. |
| Generic `STATE_TRANSFORM` | `UNCERTAIN` | Use a software descriptor/replay abstraction until a repeated hardware-backed chain is measured. |

## Evidence that is specifically missing

The roofline report classifies all 35 analyzed samples as `unknown` because the
required host-memory, HBM, PCIe, network, latency-floor, metadata-operation,
transform-operation, and synchronization counters are incomplete. See
`artifacts/branchfabric/analysis/roofline/cpu-reference-unknown.json`.

The following measurements could change an `UNCERTAIN` or
`NOT CURRENTLY JUSTIFIED` decision:

1. Real KV/HBM state sizes and branch-ready latency on a compatible GPU.
2. GPU-to-host, host-to-GPU, and peer-GPU transfer traces with copy-engine and
   interference counters.
3. Multi-destination transfers on real topology with identical-payload proof,
   source-NIC load, destination-specific transforms, and repeated-unicast
   contention.
4. Production-like page faults, dirty rates, COW amplification, and metadata
   working-set/cache-miss profiles at fanouts beyond four.
5. Repeated transform chains at realistic byte sizes, including temporary
   traffic and GPU interference.
6. Multi-node migration and pre-copy convergence with measured retransmission
   and network occupancy.

The ranked uncertainty-reduction queue is preserved at
`artifacts/branchfabric/analysis/active-experiment-queue.json`. Paid-resource
provisioning was neither authorized nor performed.

## Final negative finding

On the current evidence, BranchFabric hardware should contain **no mandatory
primitive yet**. The trace schemas, replay contracts, and optimized software
baselines should remain the design boundary. Hardware SHA/checksum,
compression, encryption, multicast, COW/page tables, quantization, fixed page
size, transaction engines, allocators, reclaimers, and environment snapshot
engines should not be built from this corpus. Reshard/transform, snapshot/fork,
transport, and shared-root tiering remain measurement questions rather than
approved hardware features.
