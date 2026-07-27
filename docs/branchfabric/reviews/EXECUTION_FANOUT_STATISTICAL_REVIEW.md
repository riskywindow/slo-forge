# BranchFabric fanout execution: independent statistical review

## Verdict

The fanout vertical's negative hardware conclusion reproduces. The independent run validates the deterministic byte accounting exactly and reproduces the direction and scale of the software-baseline effect. More importantly for the hardware gate, the independently recomputed ideal zero-cost readiness headroom remains only `1.000020x` to `1.000060x`, far below the `1.15x` system-level threshold.

This is a pass for the stated negative conclusion, not a positive performance validation. Absolute CPU readiness timing moved by 4.6% to 8.4%, and two of six readiness-median confidence intervals did not overlap. The raw results are useful for rejecting hardware under this fixture; they are not sufficiently representative or statistically independent to justify hardware.

## Independent execution

I ran the committed fanout CLI without changing its implementation or configuration:

```text
uv run --locked python -m sloforge.helix.characterization.fanout_vertical \
  --output artifacts/branchfabric/execution/replication/fanout-independent \
  --seeds 41 73 113 --fanouts 8 16 32 --warmups 1 --repetitions 3
```

The source implementation is commit `aeb43fab12be7e45e12b4cc0ad99dad53152b432`. The replication ran at repository revision `9c18719b78336e415fafc5b2da30dab5c9e59c81`. The intervening committed change was the separate reclamation vertical; no fanout implementation or gate logic was changed.

Source hashes:

- Raw samples: `af616379666cc4d9043be355976016cc06a57a87b07caa8b60b2ad69d901b810`
- Fixture evidence: `cdc1b580d2127d11e264f264406a08969bda7447ac6f15b49f77eb663a6d802c`
- Summary: `51768b2b164caacb65d0de4ca3307737c4eda8c83cef030a89dd2597fcffebc4`

Replication hashes:

- Raw samples: `98af2702ee3a2d396351164c084d77d659978502aae3061993f05a93bb7f0235`
- Fixture evidence: `220ee1cb19d91514773937464c79993a6ce07d08250940da951d27a47b8fe46c`
- Summary: `da6363e4fcbeb78d15762cea0a030c288be268fc48d9d10752309b305c8e969f`

Both manifests agree with the files they describe. Each run contains 144 unique raw samples: 36 warmups and 108 measured samples. Randomized order is dense from 0 through 143. The source and replication have the exact same randomized case sequence and sample metadata. Both cover seeds 41, 73, and 113; fanouts 8, 16, and 32; two workload classes; two implementations; and three measured repetitions per seed.

## Central measurement comparison

Shared-root/COW-lazy BranchPoint-to-all-ready medians:

| Workload | Fanout | Source p50 | Replication p50 | Change | 95% CIs overlap |
|---|---:|---:|---:|---:|---|
| Coding agent | 8 | 44.792 us | 42.375 us | -5.40% | yes |
| Coding agent | 16 | 79.875 us | 73.542 us | -7.93% | no |
| Coding agent | 32 | 148.916 us | 137.250 us | -7.83% | no |
| Reasoning/verification | 8 | 43.417 us | 41.250 us | -4.99% | yes |
| Reasoning/verification | 16 | 78.083 us | 74.459 us | -4.64% | yes |
| Reasoning/verification | 32 | 150.958 us | 138.292 us | -8.39% | yes |

The requested fanout-8, fanout-16, and fanout-32 trends reproduce: readiness grows approximately linearly with sibling count and remains tiny relative to suffix execution. Absolute timing is not byte-for-byte stable and should not be treated as a portable service curve.

### Byte accounting

All deterministic byte and semantic-hash fields match for all 144 matched raw samples and all six matched fixtures. There were zero mismatches.

| Workload | Fanout | Logical branch bytes | Shared physical bytes at ready | Shared physical bytes after suffix | Ready reduction |
|---|---:|---:|---:|---:|---:|
| Coding agent | 8 | 4,257,600 | 532,200 | 645,384 | 88.89% |
| Coding agent | 16 | 8,515,200 | 532,200 | 758,569 | 94.12% |
| Coding agent | 32 | 17,030,400 | 532,200 | 984,963 | 96.97% |
| Reasoning/verification | 8 | 8,244,736 | 1,030,592 | 1,215,394 | 88.89% |
| Reasoning/verification | 16 | 16,489,472 | 1,030,592 | 1,400,186 | 94.12% |
| Reasoning/verification | 32 | 32,978,944 | 1,030,592 | 1,769,774 | 96.97% |

These are deterministic serialized/content-addressed state bytes. They are not RSS, OS pages, accelerator allocation, HBM, or physical transfer measurements. The ready-state reduction is also partly a consequence of deferring private materialization until suffix work.

### Ideal zero-cost headroom

| Workload | Fanout | Source ideal speedup | Replication ideal speedup | Difference |
|---|---:|---:|---:|---:|
| Coding agent | 8 | 1.000060060x | 1.000059766x | -0.29 ppm |
| Coding agent | 16 | 1.000053364x | 1.000051545x | -1.82 ppm |
| Coding agent | 32 | 1.000049088x | 1.000047573x | -1.51 ppm |
| Reasoning/verification | 8 | 1.000023838x | 1.000024878x | +1.04 ppm |
| Reasoning/verification | 16 | 1.000020997x | 1.000021939x | +0.94 ppm |
| Reasoning/verification | 32 | 1.000020131x | 1.000020222x | +0.09 ppm |

The maximum across both runs is `1.000060060x`. Even eliminating the optimized readiness path entirely would not approach the `1.15x` system-level gate.

## Independent derivation checks

Without calling the implementation's summary functions, I recomputed:

- all 12 p50/p90/p95/p99 readiness summaries from measured raw samples;
- all 12 deterministic percentile-bootstrap median intervals with their recorded seeds;
- all six nine-pair median speedups, relative reductions, faster-pair fractions, and physical reductions;
- all six speedup bootstrap intervals;
- all six local-lifecycle denominators and ideal zero-cost headroom values;
- every ready and post-suffix byte-conservation invariant;
- every queue-bound and observed-concurrency invariant.

Every recomputed value matches its respective source or replication summary. All six source-versus-replication software speedup intervals overlap, and every one of the nine matched pairs per workload/fanout favors the shared-root implementation in both runs.

## Measurement-bias and statistical findings

1. **Repeated timing samples are not nine independent workloads.** Each group has three seeds and three repetitions per seed. The percentile bootstrap resamples all nine timings as if exchangeable, which can understate uncertainty when repeated measurements within a seed are correlated. A hierarchical or seed-blocked bootstrap is required for a positive performance claim.

2. **Absolute host timing is environment-sensitive.** Replication medians were uniformly faster by 4.6% to 8.4%; coding fanout 16 and 32 intervals did not overlap. This does not alter the many-orders-of-magnitude headroom conclusion, but it prevents treating these nanosecond values as calibrated platform service curves.

3. **The apparent 102x–245x readiness effect measures eager materialization versus a lazy view.** It is a meaningful software design comparison, but not an accelerator speedup and not proof that equivalent work disappeared. The fanout report correctly places measured suffix materialization in the lifecycle denominator; that denominator collapses the ideal additional gain to about `1.00002x`–`1.00006x`.

4. **Full `tracemalloc` instrumentation is part of every timed readiness sample.** There is no disabled/minimal/full instrumentation-control matrix in this vertical. Absolute latency and allocator peaks may therefore be distorted. Since both baselines use the same instrumentation and the conclusion rests on the lifecycle denominator, this is not sufficient to reverse the negative gate result.

5. **The byte reduction is deterministic accounting, not physical allocation evidence.** Exact reproduction validates the arithmetic and state identity, not HBM/CXL page sharing, runtime allocator behavior, or transferable state bytes.

6. **The measured queue is serialized.** Queue capacity, requested concurrency, and observed concurrency are one in every sample. There is no observed concurrent fanout engine, multicast transport, GPU interference, or network bottleneck.

7. **The workload remains a synthetic CPU reference fixture.** Contexts are 2,048 and 3,968 tokens because the public reference adapter is capped at 4,096. There is no real transformer, 8K/16K/32K context, CUDA, PCIe, NVLink, RDMA, NIC, FPGA, DPU, causal reclamation, or end-to-end Helix transaction in this fanout evidence.

## Software-baseline assessment

`shared_root_cow_lazy` is the strongest measured software baseline and must remain the comparison point. Its effect reproduces across all matched pairs. It also makes the hardware opportunity smaller, not larger: readiness is tens to hundreds of microseconds, while suffix execution is hundreds of milliseconds to seconds.

No hardware primitive should be selected from this fanout vertical. The evidence supports continued software sharing and lazy materialization, while leaving hardware gates failed for lack of real workload/platform evidence and lack of system-level headroom.

## Reviewer disposition

- Raw and manifest integrity: **PASS**
- Randomization, seeds, warmup exclusion, and matched-pair construction: **PASS**
- Independent summary, CI, effect-size, byte, and headroom derivation: **PASS**
- Absolute CPU timing replication: **PASS WITH DRIFT**
- Positive hardware justification: **FAIL**
- Negative hardware conclusion: **REPRODUCED**

Machine-readable comparison: `artifacts/branchfabric/execution/replication/fanout-independent/comparison.json`.
