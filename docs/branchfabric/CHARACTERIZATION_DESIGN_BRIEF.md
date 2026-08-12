# BranchFabric Characterization Design Brief

Status: evidence-gated handoff; BranchFabric hardware is not implemented or specified here.

Evidence notation in this document is deliberately strict. `SYNTHETIC/HARDWARE_BACKED_REAL` means that workload semantics are a deterministic synthetic fixture while elapsed time was measured on the recorded host. `SIMULATED_HARDWARE` means a deterministic transport model, not a physical link. A statement marked `UNKNOWN` or `UNAVAILABLE` is not a zero. Every quantitative statement below names its repo-relative source artifact and evidence class.

## 1. Executive summary

The completed corpus validates the trace and analysis path, but it does **not** justify building a BranchFabric device yet. The best-supported result is that Helix state sharing is real at the software/content-addressing level, while evaluation—not state movement—dominates the measured CPU fixture. The final four-branch run allocated 2,668 model-state bytes instead of 10,672 bytes under naive independent allocation, for 75% sharing efficiency and 1.0 physical amplification (`SYNTHETIC`; `artifacts/branchfabric/characterization/cpu-reference/attempts/vertical_trace/attempt-000/vertical-trace/sharing-analysis.json`). Its five measured evaluation spans occupy a 10.518-second wall union inside an 11.008-second run (`SYNTHETIC/HARDWARE_BACKED_REAL`; `artifacts/branchfabric/analysis/workload/cpu-reference-final.json`).

The Amdahl artifact now reports only **lifecycle-window** sensitivity. It deliberately makes no branch-readiness, migration, capacity-reclamation, rollout-throughput, or full-learning-transaction speedup claim. The largest host-real window bound is 1.01698x if the single 3,444-byte fused conversion span's reshard operation becomes free (`SYNTHETIC/HARDWARE_BACKED_REAL`, n=1; `artifacts/branchfabric/analysis/amdahl/cpu-reference-v3-windowed/amdahl-analysis.json`). Machine-readable end-to-end speedup requirements remain `UNKNOWN`.

No compatible CUDA GPU, multi-GPU system, NVLink path, RDMA fabric, or multi-node allocation was available, and no paid-resource budget was present (`HARDWARE_BACKED_REAL` capability evidence; `artifacts/branchfabric/hardware/status-seed-20260809.json`). Consequently, HBM capacity, PCIe/NVLink/network bandwidth, hardware queue depth, hardware latency targets, and a universal page size remain unavailable or unknown. The appropriate first phase is continued measurement plus optimized software baselines, not RTL, an FPGA data path, a SmartNIC program, or a frozen hardware ISA.

The immutable baseline is commit `d6f77c839334a4644b4e2edf36a7543c68670a2d`, tagged `sloforge-branchfabric-characterization-baseline-d6f77c8`. It passed the pre-characterization validation and CPU demo (`HARDWARE_BACKED_REAL` validation record; `artifacts/branchfabric/baseline/record.json`). Measurements ran on an Apple M4 Pro host with 12 logical CPUs and 25,769,803,776 bytes of unified memory; the integrated Metal GPU was not a Helix-compatible hardware-backed execution path (`HARDWARE_BACKED_REAL`; `artifacts/branchfabric/manifests/hardware-baseline.json`).

## 2. Measured Helix workload

The realistic control flow is the deterministic Helix coding-agent fixture, not production traffic. Its final canonical trace contains 70 branch events and 17 state events with zero dropped and zero filtered events; model-state sizes are explicitly simulated, while event timing is measured on the Apple host (`SYNTHETIC/HARDWARE_BACKED_REAL`; `artifacts/branchfabric/characterization/cpu-reference/attempts/vertical_trace/attempt-000/vertical-trace/vertical-run.json`). Independent earlier seeds 41, 73, and 113 reproduced the same sharing and immediate-divergence structure; the independent replication reproduced 75% sharing, 1,303 environment bytes, and the Continuum 18-operation multiset (`docs/branchfabric/reviews/INDEPENDENT_REPLICATION.md`).

The broader matrix declares coding-agent, reasoning, verification-heavy, capacity-reclamation, and highly-divergent classes across controlled fanout, context, suffix, environment, pressure, migration, and fault dimensions. All 855 cells are evaluated through the deterministic calibrated state model, but zero are hardware executed; `distribution_claim` is false, so no pooled workload percentile is produced (`ANALYTICAL_SYNTHETIC_STATE_MODEL`; `artifacts/branchfabric/characterization/cpu-reference/attempts/matrix_validate/attempt-000/matrix-study.json`). The separate Continuum lifecycle has 18 state events, zero drops, twelve host-real events, and six deterministic simulated-network events (`SYNTHETIC`, mixed timing classes; `artifacts/branchfabric/characterization/cpu-reference/attempts/continuum_state/attempt-000/continuum-trace/continuum-trace-run.json`).

## 3. Branch DAG characteristics

The final measured Helix branch group has fanout 4, with three pruned children and one successful child (`SYNTHETIC/HARDWARE_BACKED_REAL`; `artifacts/branchfabric/analysis/workload/cpu-reference-final.json`). Branch lifetime p50 is 422.585 ms and the observed maximum is 423.181 ms across four correlated siblings. The shared-root reference lifetime is 374.237 ms and peak refcount is 4. Machine-readable p95/p99 are withheld as `UNKNOWN` because one branch group cannot support hardware-sizing tails (`artifacts/branchfabric/requirements/branchfabric_requirements.json`).

These are point characteristics of one bounded branch group, not a production fanout or lifetime distribution. Queue concurrency recorded for fork, COW, snapshot, and free is 1 with zero recorded queue delay in this fixture (`SYNTHETIC/HARDWARE_BACKED_REAL`; same artifact). No hardware queue sizing should be extrapolated from that serialization.

## 4. State composition

The captured model fixture has 2,048 attention-KV bytes for eight observed tokens, plus 115 token-history bytes, 48 recurrent bytes, 122 sampler bytes, 166 guided-decoding bytes, and 161 client-delivery bytes (`SYNTHETIC`; `artifacts/branchfabric/analysis/cow/vertical-seed-41.json`). Its derived 256-byte-per-token KV slope is fixture-specific simulated state, not an actual model/KV layout and not GPU evidence.

The final canonical workload analyzer observes 13,340 unique model-segment bytes and 6,376 unique filesystem bytes, but its `model_state_bytes` field is zero because v1 segment identity does not split the simulated model capsule into KV/recurrent/sampler components (`SYNTHETIC`; `artifacts/branchfabric/analysis/workload/cpu-reference-final.json`). Database, process-reconstruction, transaction, integrity, and runtime-reconstructible byte distributions remain unknown. Actual KV size, HBM residency, padding, TP layout, and allocator overhead are unavailable until a compatible runtime is measured.

## 5. Sharing characteristics

For four siblings, shared logical model bytes total 8,852 and physical allocation is 2,668 bytes against 10,672 bytes of naive independent state. That yields 75% sharing efficiency and 1.0 physical amplification (`SYNTHETIC`; `artifacts/branchfabric/characterization/cpu-reference/attempts/vertical_trace/attempt-000/vertical-trace/sharing-analysis.json`). The two-branch Continuum fixture independently records 2,729 shared logical bytes, 455 private bytes per branch, 3,184 physical bytes versus 6,368 naive bytes, 50% sharing efficiency, and 1.0 amplification (`SYNTHETIC`; `artifacts/branchfabric/characterization/cpu-reference/attempts/continuum_state/attempt-000/continuum-trace/sharing-analysis.json`).

Environment sharing differs materially: the live Helix workspace eagerly restores each branch, so 594 base bytes become 2,376 live bytes across four branches with no claimed filesystem block sharing (`SYNTHETIC`; `artifacts/branchfabric/characterization/cpu-reference/attempts/vertical_trace/attempt-000/vertical-trace/sharing-analysis.json`). The content-addressed checkpoint layer deduplicates objects, but CAS bytes are not physical filesystem blocks. A future architecture must keep model-state sharing, live environment sharing, and checkpoint deduplication as separate mechanisms.

## 6. Divergence behavior

All four captured trajectories diverge at token index 0 and action index 0 (`SYNTHETIC`; `artifacts/branchfabric/analysis/workload/cpu-reference-final.json`). The branches nevertheless retain their common captured state until private writes occur. This is an intentional immediate-divergence case and cannot establish early-, middle-, late-, or production-weighted divergence distributions.

Page divergence is approximated only by logical COW/write records. Exact GPU block divergence, operating-system page-fault divergence, branch-depth dependence, and divergence conditional on branch age are unknown. The active experiment queue ranks the missing measurements by uncertainty, sensitivity, leverage, and cost rather than filling them with projections (`SYNTHETIC prioritization input`; `artifacts/branchfabric/analysis/active-experiment-queue.json`).

## 7. Copy-on-write behavior

The Helix fixture records four logical COW events, 1,303 bytes written, and 668 old bytes replaced. It records no page-fault counter and therefore does not claim zero actual page faults (`SYNTHETIC/HARDWARE_BACKED_REAL`; `artifacts/branchfabric/characterization/cpu-reference/attempts/vertical_trace/attempt-000/vertical-trace/sharing-analysis.json`). The Continuum fixture records 512 COW bytes over eight logical pages (`SYNTHETIC`; `artifacts/branchfabric/characterization/cpu-reference/attempts/continuum_state/attempt-000/continuum-trace/sharing-analysis.json`).

The controlled page projection evaluates 288 cells. Allocation-only results favor 4 KiB, 16 KiB, and 64 KiB equally often, while larger pages produce fragmentation in representative cells; for one shown 2,162,688-byte logical allocation, 2 MiB pages allocate 4,194,304 bytes and reach 1.939x amplification (`SYNTHETIC` analytical projection; `artifacts/branchfabric/analysis/cow/vertical-seed-41.json`). Because metadata cost and real page-fault latency are absent, the corpus explicitly rejects a universal page-size recommendation. Per-state granularity remains unresolved.

## 8. Metadata behavior

The metadata study exercises current Continuum paths and faithful isolated operations with deterministic order, warmups, seven measured repetitions, and one-, two-, and four-thread settings (`SYNTHETIC/HARDWARE_BACKED_REAL`; `artifacts/branchfabric/metadata/seed-20260809/metadata-study.json`). Representative one-thread medians are 31.635 microseconds for actual branch-create bookkeeping, 67.633 microseconds for actual abort, 1.590 microseconds for isolated page lookup, and 2.531 microseconds for isolated dirty update (`SYNTHETIC/HARDWARE_BACKED_REAL`; same artifact).

Batched isolated baselines improve page lookup by up to 1.322x at two threads and ancestry lookup by up to 1.363x at two threads, while sharding is frequently near parity (`SYNTHETIC/HARDWARE_BACKED_REAL`; same artifact). These comparisons are not hardware projections, but they establish that software batching must precede a Branch Translation Unit claim. Cache misses, allocator attribution, syscall attribution, resident metadata working set, and outstanding queue depth are not measured; the roofline classifier therefore marks resource dominance unknown (`SYNTHETIC/HARDWARE_BACKED_REAL`; `artifacts/branchfabric/analysis/roofline/cpu-reference-unknown.json`).

## 9. Transform behavior

One producer-declared fused transformation span is observed: `STATE_RESHARD -> STATE_REPACK -> STATE_CHECKSUM` over 3,444 bytes with 1.665 ms inclusive host latency. Per-operation latency is explicitly non-decomposable; only `STATE_RESHARD` is the primary canonical event (`SYNTHETIC/HARDWARE_BACKED_REAL`; `artifacts/branchfabric/analysis/transform/cpu-reference-v3.json`). Transpose, quantize, dequantize, compression, decompression, and encryption do not occur. Temporary allocation and memory-read/write counters are absent.

The strongest controlled software transform baseline is the trusted staged canonical path at 5.899 ms median versus 8.905 ms for direct Continuum convert-and-capture, each over seven measured samples; the direct path includes independent canonical verification, so this is not a fused-kernel speedup claim (`SYNTHETIC/HARDWARE_BACKED_REAL`; `artifacts/branchfabric/software-baselines/seed-20260809/software-baselines.json`). Fusion is a hypothesis for a future byte-scale GPU migration trace, not a justified hardware pipeline today.

## 10. Network behavior

The final state lifecycle records 8,114 source payload bytes and 8,114 logical delivery bytes with zero retry payload bytes (`SYNTHETIC`, mixed host-copy and `SIMULATED_HARDWARE`; `artifacts/branchfabric/analysis/transport/cpu-reference-v3.json`). One in-process host-memory transfer moves 3,444 bytes with 69 microseconds recorded transfer time; three modeled transfers account for 4,670 bytes and use deterministic simulated latency (`SYNTHETIC`, respectively `HARDWARE_BACKED_REAL` and `SIMULATED_HARDWARE`; same artifact).

No NIC, RDMA, PCIe, NVLink, aggregate link load, protocol overhead, retransmission behavior, or real concurrent transfer was measured. Therefore mean, tail, and burst bandwidth requirements for GPU, host, network, and storage interfaces remain unknown or unavailable (`artifacts/branchfabric/requirements/branchfabric_requirements.json`).

## 11. Multicast opportunity

Every observed `STATE_SEND` has fanout 1. The conservative multicast detector finds zero opportunities and zero exact payload-reduction bytes for thresholds from 2 through 128 destinations; four sends remain ungrouped because v1 lacks a shared non-empty payload-dependency identity (`SYNTHETIC`, mixed host/simulated transport; `artifacts/branchfabric/analysis/transport/cpu-reference-v3.json`). This supports `NOT_JUSTIFIED` for `STATE_MULTICAST` in the current bounded corpus, not the stronger assertion that production Helix can never benefit.

Host-only fanout baselines also fail to select one universal strategy: repeated unicast wins some fanouts and a binary tree wins others. At fanout 128, the selected tree median is 2.374 ms versus 2.653 ms repeated-unicast, but this is host-copy scheduling with no network link (`SYNTHETIC/HARDWARE_BACKED_REAL`; `artifacts/branchfabric/software-baselines/seed-20260809/software-baselines.json`). A multicast primitive must wait for identical multi-destination payload traces on physical GPUs/nodes.

## 12. Migration behavior

Migration is absent from the full Helix lifecycle and represented separately by Continuum (`SYNTHETIC`; `artifacts/branchfabric/characterization/cpu-reference/attempts/vertical_trace/attempt-000/vertical-trace/vertical-run.json`). The Continuum trace includes snapshot, publish, fork, logical COW/append/delta, a fused conversion declaration, transfer, commit, and reclaim; twelve operation timings are host-real and six transport timings are simulated (`SYNTHETIC`, mixed timing classes; `artifacts/branchfabric/characterization/cpu-reference/attempts/continuum_state/attempt-000/continuum-trace/continuum-trace-run.json`).

The state trace is not one migration critical path: snapshot, branch, conversion, real host copy, and a separate pre-copy transaction share the artifact. The Amdahl analysis therefore labels it a Continuum lifecycle window. In that window, free reshard is bounded at 1.01698x, free delta extraction at 1.00611x, free simulated transfer at 1.00943x, and free commit at 1.00095x (`SYNTHETIC`, host timings except `SIMULATED_HARDWARE` transfer; `artifacts/branchfabric/analysis/amdahl/cpu-reference-v3-windowed/amdahl-analysis.json`). No migration-latency bound is claimed.

## 13. Integrity behavior

Whole-payload host SHA-256 is the selected controlled baseline at 79.333 microseconds median and about 3.304 GB/s logical throughput over seven measured samples (`SYNTHETIC/HARDWARE_BACKED_REAL`; `artifacts/branchfabric/software-baselines/seed-20260809/software-baselines.json`). Chunked SHA-256 candidates are semantically equivalent but do not improve the selected median in this host study.

The lifecycle corpus does not isolate checksum/hash frequency on the branch critical path, Merkle metadata, encryption, GPU hashing, or in-flight verification. No Amdahl result establishes meaningful end-to-end integrity leverage. Inline hash/checksum hardware is therefore not recommended; it remains an evidence-gated measurement question.

## 14. Environment behavior

The environment study separately exercises tiny, medium, large, dependency-heavy, and service-heavy fixtures. Their declared logical sizes range from 1,280 bytes to 1,056,768 bytes. Under tracing disabled, the median sum of measured lifecycle-operation durations is approximately 7.792 ms, 46.282 ms, 123.521 ms, 86.960 ms, and 26.655 ms respectively, with three measured trials per workload and warmups retained (`SYNTHETIC/HARDWARE_BACKED_REAL`; `artifacts/branchfabric/environment/seed-20260809/environment-study.json`).

The implementation captures to CAS, eagerly restores a workspace, performs whole-file replacement, checkpoints, compares, reconstructs declared service metadata, and tears down. It is not filesystem reflink/page COW, a real database snapshot service, container startup, or process reconstruction benchmark. Environment acceleration should remain software until those mechanisms are measured independently.

## 15. End-to-end bottlenecks

In the final waterfall, evaluation is the dominant observed stage union at 10.518 s; reward is 351.186 ms, fork 48.943 ms, checkpoint 21.465 ms, capture 15.603 ms, rollout 10.294 ms, training 0.566 ms, and promotion 0.507 ms (`SYNTHETIC/HARDWARE_BACKED_REAL`; `artifacts/branchfabric/analysis/workload/cpu-reference-final.json`). Branch-ready completion is observed as four zero-duration markers carrying the inclusive group-fork wait latency. Production, migration, and resume are absent. These are inclusive stage unions; an exclusive critical path is explicitly unavailable.

This evidence makes state metadata/COW a secondary CPU-fixture concern relative to reward and overall policy/control work. It does not locate cycles on GPU, HBM, PCIe, or a NIC. The roofline remains unknown because the hardware ceilings and per-operation memory/network counters were not collected (`SYNTHETIC/HARDWARE_BACKED_REAL`; `artifacts/branchfabric/analysis/roofline/cpu-reference-unknown.json`).

## 16. Amdahl limits

No required end-to-end Amdahl objective has a defensible bound in this corpus. The retained sensitivity results are lifecycle-window upper bounds only. In the Helix lifecycle window, free COW is bounded at 1.00385x and free reclamation at 1.00243x across three trace instances. In the separate Continuum lifecycle window, free reshard is 1.01698x at n=1; other bounds are listed in Section 12 (`SYNTHETIC`, mixed timing classes; `artifacts/branchfabric/analysis/amdahl/cpu-reference-v3-windowed/amdahl-analysis.json`).

No Amdahl result exists for real GPU branch readiness, HBM copy, PCIe, NVLink, a physical network, full capacity reclamation, or a production learning transaction. A primitive cannot be classified `REQUIRED` on the basis of an unmeasured fraction.

## 17. Strongest software baselines

The required comparison set is now: batched/sharded metadata; trusted staged versus direct canonical transform; whole versus chunked SHA-256; chunk sizes from 4 KiB through 1 MiB with controlled in-process concurrency; and repeated-unicast versus binary-tree host fanout (`SYNTHETIC/HARDWARE_BACKED_REAL`; `artifacts/branchfabric/metadata/seed-20260809/metadata-study.json` and `artifacts/branchfabric/software-baselines/seed-20260809/software-baselines.json`).

For in-process transfer, the selected controlled case uses 64 KiB chunks and concurrency 4, with 296.708 microseconds median and about 883.5 MB/s logical throughput over seven samples. This measures a thread-safe host content store and in-process Continuum transport, not a NIC or device DMA path (`SYNTHETIC/HARDWARE_BACKED_REAL`; `artifacts/branchfabric/software-baselines/seed-20260809/software-baselines.json`). Any future accelerator comparison must match semantics and beat the selected software baseline on end-to-end objectives, not only microkernel time.

## 18. Recommended hardware primitives

None is currently `REQUIRED` or `HIGH-VALUE`. The generated requirements file intentionally has an empty `recommended_isa` list and leaves observed-but-undercharacterized operations unresolved (`SYNTHETIC` evidence compilation; `artifacts/branchfabric/requirements/branchfabric_requirements.json`).

The next evidence candidates are not hardware commitments: measure `STATE_RESHARD`/transport on real device-scale state because reshard has the highest current migration Amdahl fraction; measure fork/COW against actual runtime page tables; and measure identical multi-destination sends. Only after those experiments should a streaming transform, branch metadata engine, or multicast opcode be promoted.

## 19. Rejected hardware ideas

The machine-readable file classifies all 30 `StateOperationTrace v1` operations exactly once. Six absent control/integrity operations (`STATE_HASH`, encryption/decryption, ACK, retry, abort) remain `SOFTWARE_ONLY`; ten absent memory/transform/multicast operations are `NOT_JUSTIFIED`; fourteen observed or producer-declared operations remain unresolved. None is required or high-value (`artifacts/branchfabric/requirements/branchfabric_requirements.json`).

Branch fork, COW, append, snapshot, delta, reshard, send, commit, reclaim, publish, and free are **unresolved**, not accepted hardware operations (`SYNTHETIC`; same artifact). Environment snapshot acceleration should remain software because the measured backend is eager restore plus CAS, not a physical filesystem/database snapshot engine.

## 20. Memory hierarchy recommendation

Do not size or choose HBM, DDR, CXL, or storage-side state memory from this corpus. HBM is unavailable and DDR/CXL/host capacity requirements at matrix fanouts are unknown; declared fanouts are not executed capacity observations (`HARDWARE_BACKED_REAL` capability plus `SYNTHETIC` matrix inputs; `artifacts/branchfabric/requirements/branchfabric_requirements.json`).

For the next software/measurement phase, retain shared immutable roots in their runtime-native memory, retain private suffixes and environment state in their current owners, and keep content-addressed checkpoints in software storage. Instrument simultaneous residency, eviction eligibility, low-latency metadata working set, transform buffers, and burst buffers before selecting tiers. The small fixture’s 2,668-byte physical root is useful only for conformance, not capacity planning (`SYNTHETIC`; `artifacts/branchfabric/characterization/cpu-reference/attempts/vertical_trace/attempt-000/vertical-trace/sharing-analysis.json`).

## 21. Interface recommendation

Preserve versioned JSON/Parquet trace interchange and the canonical `BranchWorkloadTrace v1`/`StateOperationTrace v1` semantics as the measurement-facing contract. Do not freeze a device command ABI. The current trace identifies dependencies, state epochs, locations, transforms, transport, integrity, results, and drops; the full specification is in `docs/branchfabric/TRACE_SPECIFICATION.md`.

Any future device interface should be derived only after real traces establish atomicity, ordering, tenant isolation, failure recovery, payload addressing, and completion semantics. The present transaction corpus observes one commit and no abort in the bounded Continuum window, while epoch-check rate is unknown (`SYNTHETIC`; `artifacts/branchfabric/requirements/branchfabric_requirements.json`). That is inadequate to specify device consistency.

## 22. Queue-depth recommendation

Observed state-operation concurrency is 1, but this is a serialized-fixture lower bound, not a hardware minimum. Minimum queue depth, recommended depth, pathological maximum, and backpressure policy are all `UNKNOWN` (`SYNTHETIC`; `artifacts/branchfabric/requirements/branchfabric_requirements.json`).

Before queue sizing, collect outstanding-operation timelines under actual rollout saturation, concurrent migration, branch pruning, and failure recovery. Record occupancy, service time, queue delay, cancellation, dependencies, and overflow rather than substituting worker or thread counts.

## 23. Page-size recommendation

No universal page size is recommended. The allocation-only synthetic projection nominates 4 KiB as its tie-break result but explicitly says 4 KiB, 16 KiB, and 64 KiB tie in allocation-minimizing wins and that metadata/fault latency is unmeasured (`SYNTHETIC` analytical projection; `artifacts/branchfabric/analysis/cow/vertical-seed-41.json`). The machine-readable requirement therefore leaves recommended KV page size unknown (`SYNTHETIC` evidence compilation; `artifacts/branchfabric/requirements/branchfabric_requirements.json`).

The next sweep must distinguish runtime KV blocks, small metadata pages, environment filesystem blocks, and migration chunks on real backends. It must measure fault count, copied bytes, fragmentation, dirty-log precision, allocation latency, and reclamation latency together.

## 24. Bandwidth requirement

No BranchFabric bandwidth target can be derived. GPU-to-device, device-to-GPU, internal memory, network ingress/egress, host-interface, and storage-interface mean/tail/burst values are unknown or unavailable (`HARDWARE_BACKED_REAL` capability evidence and explicit null requirements; `artifacts/branchfabric/requirements/branchfabric_requirements.json`). The observed 883.5 MB/s in-process host result from Section 17 is a software baseline, not a requirement.

Bandwidth targets require time-windowed byte counters on device-scale payloads, preserving mean, p95, p99, peak, and burst duration. Network measurements must include topology routes, source-NIC load, destination conversion, retransmission, and compute overlap.

## 25. Latency requirement

No target p50, target p99, or maximum tolerable latency is established for append, commit, COW, delta, fork, free, publish, receive, reclaim, reshard, send, or snapshot (`SYNTHETIC` evidence compilation; `artifacts/branchfabric/requirements/branchfabric_requirements.json`). Observed host latency is not substituted for a target because no SLO sensitivity experiment exists.

The next experiment should inject bounded delay per operation and measure branch readiness, rollout throughput, migration latency, capacity reclamation, and full learning-transaction latency. Only the knee of those curves can set tolerable latency.

## 26. Initial FPGA target recommendation

Do not start an FPGA target. The current corpus neither supplies device-locality/bandwidth requirements nor provides a valid end-to-end state-operation Amdahl bound. The largest retained host-real lifecycle-window bound is 1.01698x for one tiny fused conversion span (`SYNTHETIC/HARDWARE_BACKED_REAL`; `artifacts/branchfabric/analysis/amdahl/cpu-reference-v3-windowed/amdahl-analysis.json`).

An FPGA should be reconsidered only after a compatible GPU experiment shows a repeated, byte-dominant, streamable transform-plus-transfer chain with measured interference and meaningful Amdahl leverage. The qualifying evidence must include actual HBM/PCIe bytes, transforms, temporary memory, queue occupancy, and a matched optimized GPU/CPU software baseline. This is a measurement gate, not a proposed design.

## 27. Future SmartNIC/DPU recommendation

Do not start a SmartNIC/DPU implementation. There is no physical multi-node trace, all observed sends have fanout 1, and the pre-copy link is simulated (`HARDWARE_BACKED_REAL` capability status plus `SYNTHETIC/SIMULATED_HARDWARE` trace; `artifacts/branchfabric/hardware/status-seed-20260809.json` and `artifacts/branchfabric/analysis/transport/cpu-reference-v3.json`).

Reconsider a DPU only if real multi-node runs show source-NIC saturation, repeated identical payloads, CPU-heavy integrity/serialization, or in-flight destination-independent transforms on a meaningful critical path. The existing hardware-status artifact preserves exact bounded future commands without provisioning resources (`HARDWARE_BACKED_REAL` capability evidence; `artifacts/branchfabric/hardware/status-seed-20260809.json`).

## 28. Risks

- Measurement bias: one deterministic coding-agent fixture may encode the intended architecture more strongly than real workloads. The immediate divergence and fanout-4 result must not be generalized (`SYNTHETIC`; `artifacts/branchfabric/analysis/workload/cpu-reference-final.json`).
- Instrumentation distortion: the corrected canonical-recorder campaign has six measured samples per level after level-specific warmups. Full tracing changes median hot-path wall time by +0.333% and end-to-end time including JSONL/Perfetto persistence by +0.387%; it persists a median 210,867 bytes and drops zero events. The workload is low-rate and cannot validate target command rates (`SYNTHETIC/HARDWARE_BACKED_REAL`; `artifacts/branchfabric/overhead/helix-cpu-3seed-2rep-v5-canonical/instrumentation-overhead.json`).
- Counter gaps: cache misses, memory traffic, temporary bytes, exclusive critical path, queue occupancy, and physical topology are absent (`artifacts/branchfabric/analysis/roofline/cpu-reference-unknown.json`).
- Scale risk: byte-sized fixtures can invert fixed-cost versus bandwidth conclusions. No state-size projection is production evidence (`SYNTHETIC`; `artifacts/branchfabric/analysis/cow/vertical-seed-41.json`).
- Semantic risk: CAS deduplication, logical COW, filesystem block COW, and GPU KV block sharing are distinct and must not be conflated.

## 29. Unknowns requiring additional hardware measurements

The highest-priority unknowns are actual KV/recurrent/sampler state size and layout; real GPU fork/readiness; GPU-to-host, host-to-GPU, and peer-GPU transfer; conversion and interference; page-fault/COW granularity; concurrent outstanding operations; device memory residency; and real network fanout/migration. CUDA, multi-GPU, and multi-node studies are all `UNAVAILABLE`, with no synthetic substitution (`HARDWARE_BACKED_REAL` capability evidence; `artifacts/branchfabric/hardware/status-seed-20260809.json`).

Also missing are real coding-agent workload distributions, long-context suffix evolution, late divergence, highly divergent negative cases, pause/resume under a serving spike, failure recovery, database/process environment state, dirty-rate time series, recopy waste, and full production-to-promotion critical paths. The matrix and active queue define reproducible candidates, but a declared cell remains `SYNTHETIC` until executed and cannot set hardware requirements (`benchmarks/branchfabric/characterization.yaml`; `artifacts/branchfabric/analysis/active-experiment-queue.json`).

## 30. Proposed BranchFabric Phase 1 architecture

Phase 1 should be a **measurement and software-reference architecture**, not BranchFabric hardware:

1. Keep `BranchWorkloadTrace v1` and `StateOperationTrace v1` as the versioned workload contract, with bounded asynchronous buffers, explicit drops, hashes, manifests, and Perfetto export (`docs/branchfabric/TRACE_SPECIFICATION.md`).
2. Keep state ownership and consistency in Helix/Continuum software. Use the strongest batched metadata, canonical transform, hashing, chunking, and fanout baselines documented in Section 17.
3. Add hardware-backed adapters without changing scheduling semantics: CUDA runtime state counters, GPU copy/compute timestamps, PCIe/NVLink/NIC counters, physical page/block identity, temporary-byte counters, and queue occupancy.
4. Run the active experiment queue on accessible compatible hardware first; provision nothing without explicit budget. Preserve raw data and classify every result as real, replayed, synthetic, or simulated.
5. Recompute roofline, Amdahl, placement, memory, page, queue, bandwidth, latency, and ISA rankings. Promote an operation to hardware only when a repeated measured workload shows material end-to-end leverage over the strongest software baseline.

This phase has no RTL, FPGA simulator, SmartNIC firmware, CXL design, device driver, or frozen ISA. That restraint is the central evidence-backed architecture decision of the present characterization.
