# Independent CPU Replication Review

## Verdict

The seed-149 replication reproduces the controlled Helix and Continuum fixture structure, the 75% model-CAS sharing result, immediate sibling divergence, the 512-byte Continuum COW result, all 14 one-thread metadata latency medians within 11%, zero observed exact multicast opportunities, and the single observed reshard sequence. Canonical trace integrity, event ordering, artifact hashes, and zero-drop accounting all pass.

The instrumentation-overhead replication does not reproduce the sign of the primary wall-time effect. The primary six-sample-per-level study reports a -0.13% paired median full-tracing change, while the independent two-sample-per-level run reports +1.34%. This is evidence that the current data do not resolve a stable wall-time penalty, not evidence of either a speedup or a regression. Full-tracing CPU time is higher in both datasets (+3.5% primary and +6.5% replication).

These results validate the deterministic CPU fixture and its trace pipeline. They do not turn its workload distribution, simulated model state, or simulated network into production or GPU-backed evidence, and they do not independently justify a BranchFabric hardware primitive.

## Scope and evidence classes

- Pinned measurement source: `f1ebbf621241ca27a40645c04f95bd421079eba8`.
- Replication seeds: Helix and Continuum `149`; metadata `20260810`; overhead workload `191` with randomized schedule seed `20260810`.
- Workloads: `SYNTHETIC` deterministic fixtures.
- Helix, Continuum host operations, and metadata timings: `HARDWARE_BACKED_REAL` CPU observations on the existing Apple M4 Pro host.
- Helix model-state sizes: simulated GPU state, not HBM measurements.
- Continuum pre-copy transport: `SIMULATED_HARDWARE`; 12 Continuum events carry host timings and 6 carry simulated-network timings.
- No GPU, PCIe, NVLink, RDMA, multi-node, or production-distribution claim is made.

The replication wrote only beneath `artifacts/branchfabric/replication/`; primary raw artifacts were not modified.

## Comparison with the primary corpus

| Finding | Primary evidence | Independent replication | Assessment |
|---|---:|---:|---|
| Helix fanout | 4 in seeds 41, 73, 113 | 4 | Exact fixture replication |
| Model-CAS sharing efficiency | 75% in all three seeds | 75% | Exact |
| Model physical bytes | 2,660-2,662 B | 2,664 B | Seed-dependent serialized size; same allocation relation |
| Environment bytes written | 1,303 B in all three seeds | 1,303 B | Exact |
| First divergent token/action | index 0 / index 0 in all three seeds | index 0 / index 0 | Exact, from four verified trajectory artifacts |
| Branch-lifetime p50 | 384.6-408.5 ms | 381.5 ms | Same order of magnitude; one replication observation is below the three-run range |
| Fork-to-first-private p50 | 88.8-95.4 ms | 90.3 ms | Within primary range |
| Shared-root lifetime | 339.1-357.6 ms | 335.0 ms | Close, but single-run timing is not a distribution |
| Continuum operation count | 18, zero drops | 18, zero drops | Exact |
| Continuum operation multiset | Same 18 operations in all primary seeds | Same multiset | Exact |
| Continuum COW | 512 B, 8 pages | 512 B, 8 pages | Exact |
| Continuum allocated bytes | 3,176-3,179 B | 3,180 B | Seed-dependent serialized size |
| Metadata, current software, one thread | 14 operations, 7 samples each | 14 operations, 3 samples each | Median absolute difference 2.42%; maximum 10.77% |
| Full tracing wall effect | -0.13% paired median, 6 pairs | +1.34%, 2 pairs | Sign not replicated; stable wall effect unresolved |
| Full tracing CPU median vs disabled | +3.5% | +6.5% | Direction replicated; magnitude uncertain |
| Transport source payload | 8,087-8,094 B | 8,106 B | Seed-dependent payload; four sends and four receives retained |
| Exact multicast opportunities | 0 in all primary Continuum runs | 0 | Exact for this fanout-one fixture only |
| Transform sequences | One single-stage `STATE_RESHARD` | One 3,440-byte single-stage `STATE_RESHARD` | Exact structure; no measured fusion opportunity |

## Helix branch sharing, divergence, and lifetime

The seed-149 trace contains 61 BranchWorkloadTrace events and 17 StateOperationTrace events. All 78 attempted events were accepted. Its branch DAG has four children, three pruned outcomes, and one success. Exact trajectories establish first divergence at token 0 and action 0; this is not inferred from hashes or timing.

The model-state fixture reports 10,656 logical branch-bytes, 2,664 physically allocated bytes, 75% sharing efficiency, and physical amplification 1.0. The filesystem backend remains an eager restore: 2,376 live-workspace bytes equal naive allocation, so no live filesystem sharing or page fault is claimed. The checkpoint records 1,303 private dirty bytes. The observed shared-root lifetime is 334,988,875 ns, and p50 branch lifetime is 381,462,334 ns.

This reproduces a deliberately immediate-divergence negative case. It does not establish a production divergence distribution or the value of shared-root hardware for long real contexts.

## Continuum state lifecycle

The seed-149 Continuum trace reproduces this exact operation multiset: two `STATE_SNAPSHOT`; four `STATE_SEND`; four `STATE_RECEIVE`; and one each of `STATE_PUBLISH`, `STATE_FORK`, `STATE_COW`, `STATE_APPEND`, `STATE_DELTA`, `STATE_RESHARD`, `STATE_COMMIT`, and `STATE_RECLAIM`.

The sharing result is 2,730 shared logical bytes, 3,180 physical bytes, 50% sharing efficiency across two branches, 512 COW bytes across eight logical pages, and 164,534 metadata bytes per branch. The few-byte changes from the primary runs track seed-dependent serialized content. Host operation timings are single observations and vary substantially; they are not treated as replicated latency percentiles.

## Metadata paths

The bounded replication retained 280 raw samples: one warmup and three measured repetitions for each applicable case at one and two threads. The comparable result uses only the 14 `current_software`, one-thread operations. The median absolute difference from the primary medians is 2.42%; the maximum is 10.77% for state-hash lookup. No sample or outlier was removed.

This reproduces the portable software measurements. It does not resolve CPU cycles, cache misses, memory bandwidth, lock wait, allocator attribution, or syscalls because those counters remain unavailable. The timed intervals also include `tracemalloc`, Python, and scheduler effects.

## Instrumentation overhead

The replication randomized six measured trials: two each for disabled, minimal, and full tracing. Semantic digests are identical across levels, every trial reports the same pinned source commit, and the raw sample hash is `d80c18aa88cfb30559fcb2c5cad8348b1fa2208cc559d428e6ab7a0be49aed61`.

Wall-time medians are 11.219 s disabled, 11.019 s minimal, and 11.368 s full. The full-versus-disabled paired median change is +1.34%, compared with -0.13% in the larger primary study. With two pairs, no warmup, and effect sizes comparable to run noise, this replication cannot estimate a stable population overhead. CPU-time medians are 1.944 s disabled and 2.071 s full; the direction of increased CPU work agrees with the primary study.

## Transport, multicast, and transforms

The Continuum replication records 8,106 source payload bytes, 8,106 logical delivery bytes, 8,106 receive bytes, and zero retry bytes. Three send/receive pairs use deterministic simulated transport and one pair uses measured in-process host memory copy. All sends have fanout one. The analyzer therefore finds zero exact multicast opportunities and four ungrouped send events.

The single transform is a 3,440-byte `STATE_RESHARD` from a host-DRAM token-major TP4 representation to a host-DRAM head-major packed TP2 representation. It has no adjacent transform stage, so the measured chain provides no intermediate-materialization saving from fusion. One tiny fixture is insufficient to infer network bandwidth requirements or to reject multicast and fused transforms for other workloads.

## Integrity and validation

Both trace manifests were revalidated against every referenced artifact. SHA-256, byte length, canonical event content hash, event ordering, event count, trace-corpus hash, and zero-drop accounting all match. The machine-readable result is `artifacts/branchfabric/replication/analysis/integrity-validation.json` with SHA-256 `692924b79728bc263aa668c3479c5ed0eef95cc6f8656c53fc403b691f03c844`.

Acceptance commands:

```text
uv run --locked pytest -q tests/python/test_branchfabric_trace_models.py tests/python/test_branchfabric_trace_io.py tests/python/test_branchfabric_transport_analysis.py tests/python/test_branchfabric_transform_analysis.py tests/python/test_branchfabric_workload_analysis.py tests/python/test_branchfabric_metadata_study.py tests/python/test_branchfabric_overhead.py
cargo test -p sloforge-helix-ir --test branchfabric_trace
```

Results: 36 Python tests and 5 Rust tests passed.

## Artifact index

- Machine-readable comparison: `artifacts/branchfabric/replication/analysis/comparison.json`, SHA-256 `b720161ea4a7f0e14908910ddeba16641a90dffe8431a1f7d45e3b793ae05a3f`.
- Helix manifest: `artifacts/branchfabric/replication/helix-seed-149/trace-manifest-v1.json`, SHA-256 `cdac890813dc5b7d24d7dcfa237efafa196e9c12a3354de19f207bccf07b939d`.
- Helix branch trace: `artifacts/branchfabric/replication/helix-seed-149/branch-workload-trace-v1.jsonl`, SHA-256 `dc3fe9108cbd44d512bb2051f9f2360f4ca6997978b06e463a3dc3086245fb35`.
- Helix state trace: `artifacts/branchfabric/replication/helix-seed-149/state-operation-trace-v1.jsonl`, SHA-256 `67957ddaf50d6eb05d8b260d1b5f95be26bc2ea60640da7b2ab3ea82da14b260`.
- Helix workload analysis: `artifacts/branchfabric/replication/analysis/helix-workload-analysis.json`, SHA-256 `cc20d0bc7fe05a64641b758532dddf83e6c3824024e8d29f1c25cf5c584b596b`.
- Continuum manifest: `artifacts/branchfabric/replication/continuum-seed-149/trace-manifest-v1.json`, SHA-256 `dcfe4c0b75928864a687ad9a7dece075d739dc782b40aa512642d9cc0df05698`.
- Continuum state trace: `artifacts/branchfabric/replication/continuum-seed-149/state-operation-trace-v1.jsonl`, SHA-256 `eafdd1e27287ad23083787a4ae6c24682dc93e6477e5934204be3ff48bc2e5a6`.
- Metadata study: `artifacts/branchfabric/replication/metadata/metadata-study.json`, SHA-256 `bb7f782c81c48cf196bfa147ad18e842ea82c86d0d04260d6d608116474b4c16`.
- Overhead raw samples: `artifacts/branchfabric/replication/overhead/overhead-raw-samples.json`, SHA-256 `d80c18aa88cfb30559fcb2c5cad8348b1fa2208cc559d428e6ab7a0be49aed61`.
- Overhead analysis: `artifacts/branchfabric/replication/overhead/instrumentation-overhead.json`, SHA-256 `03afda571abee745d68f7b502b6bc0aa24e23feea67dea9762f6a718f9cb6111`.
- Transport analysis: `artifacts/branchfabric/replication/analysis/transport-analysis.json`, SHA-256 `e2630d6eda444660a6727a4ec9b375f7d5da04c4186297872f021f293df83844`.
- Transform analysis: `artifacts/branchfabric/replication/analysis/transform-analysis.json`, SHA-256 `30a1ba0fed57a7ca5725221e873515aaf83b570b611d05fb6723e428e9fc49f0`.

## Hardware-design implication

The replicated evidence supports keeping the current conservative classification: software metadata and branch operations are real and measurable, but the CPU fixture is too small to establish system-level hardware leverage; no multicast opportunity is observed; no multi-stage transform-fusion opportunity is observed; and network/GPU paths are not hardware-backed. These measurements should constrain claims, not be retrofitted into a BranchFabric justification.
