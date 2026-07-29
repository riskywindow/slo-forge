# BranchFabric final independent statistical and evidence review

## Verdict

The final evidence supports **`FAIL_NO_BUILD`**. The negative hardware conclusion is statistically and artifact-wise reproducible. No reviewed measurement is a real-transformer, GPU, GPU-transfer, physical-network, FPGA, or DPU result, and no candidate has a target-hardware headroom bound.

The strongest measured results are software results on deterministic CPU-reference state:

- `shared_root_cow_lazy` reduces deterministic serialized ready-state bytes by 88.89%, 94.12%, and 96.97% at fanout 8, 16, and 32. Its remaining readiness path is only 0.0020% to 0.0060% of the measured local lifecycle, so making that path free gives at most `1.000060x` additional speedup.
- `optimized_bounded_parallel` reduces local-file serialized transfer bytes by exactly 87.5% and improves the matched reclamation interruption median by `1.2721x` (95% percentile-bootstrap CI `1.2445x` to `1.2980x`). This is a CPU/local-file software effect, not hardware headroom.
- The same reclamation baseline improves total wall time by `1.1675x` (95% CI `1.1448x` to `1.1875x`) and migration by `2.2778x` (95% CI `2.1663x` to `2.3871x`), while branch readiness becomes 2.01% slower (`0.9803x`, 95% CI `0.9380x` to `0.9996x`).

The data-path candidate passes only the end-to-end relevance gate because optimized full-trace migration occupies a 19.90% median share of the local interruption interval. It still fails real evidence, system-level hardware headroom, platform feasibility, and workload value. All other candidates fail every mandatory gate. Hardware implementation and hardware modeling therefore remain disallowed.

## Reviewed immutable bindings

Review revision: `d113e12b5d618fa3b39b41a30f5f272bd96a18b2`.

| Artifact | SHA-256 | Review use |
|---|---|---|
| `execution/fanout/raw-samples.jsonl` | `af616379666cc4d9043be355976016cc06a57a87b07caa8b60b2ad69d901b810` | canonical fanout raw samples |
| `execution/fanout/summary.json` | `51768b2b164caacb65d0de4ca3307737c4eda8c83cef030a89dd2597fcffebc4` | fanout statistics and local headroom |
| `execution/fanout/fixture-evidence.jsonl` | `cdc1b580d2127d11e264f264406a08969bda7447ac6f15b49f77eb663a6d802c` | serialized reference-state accounting |
| `execution/replication/fanout-independent/comparison.json` | `08226e5aa35e938dc239d15fcd3bf25c7d0873d05b56066ebd8f8c9c5df83030` | independent fanout replication |
| `execution/reclamation/raw/trials.jsonl` | `795a6803c1123555ea5804ddfeca711805235b89eeb30c5e723299bd26e74408` | latest reclamation raw trials |
| `execution/reclamation/campaign.json` | `daac0cf1f595984d2eb2c5452524ec4ca174285986d934fcd42e68e43cc6287c` | campaign summaries |
| `execution/reclamation/analysis.json` | `cbd27f984fa412cd33fd3714b66f8978ab3311736127058b04efc2fdab0adbf0` | raw-bound CIs and effects |
| `overhead/helix-cpu-3seed-2rep-v5-canonical/overhead-raw-samples.json` | `492e90f1ea0ca35f4db4e664efacd06f9c08e20dad7f6967982a2eedd876c2df` | canonical tracing control samples |
| `gates/branchfabric_gate_input.json` | `4b70653b883e6a8c6a8b2cdd6f7768e33680ec7a6c1f5368a7467efb6eff8e6d` | final gate evidence input |
| `gates/branchfabric_gate_result.json` | `0a96f44800ae74c36e62e5744ed877445910119afc5ecfcc75ad4b9bbf0c62b4` | final compiled result |

The fanout and reclamation manifests match their payloads. Every gate evidence reference exists and matches its recorded digest. Independently rerunning `analyze_reclamation` produces the committed analysis byte for byte. Independently evaluating the corrected gate input produces the committed gate result byte for byte.

## Canonical fanout validation

The fanout campaign has 144 unique raw samples: 36 declared warmups and 108 measured samples. Its randomized order is dense, and it covers seeds 41, 73, and 113; fanouts 8, 16, and 32; two fixture labels; two software implementations; and three measured repetitions per seed.

The two labels are `coding_agent` and `reasoning_verification`, but every sample is explicitly `SYNTHETIC_CPU_REFERENCE_MODEL_STATE` / `CPU_REFERENCE_MODEL_STATE`. The contexts are 2,048 and 3,968 reference tokens. These labels are scenario classes, not sampled production distributions and not real-model evidence.

### Strongest fanout baseline

| Fixture label | Fanout | Shared ready bytes | Shared bytes after suffix | Ready-byte reduction | Optimized readiness p50 | Ideal additional speedup |
|---|---:|---:|---:|---:|---:|---:|
| Coding agent | 8 | 532,200 | 645,384 | 88.89% | 44.792 us | `1.000060x` |
| Coding agent | 16 | 532,200 | 758,569 | 94.12% | 79.875 us | `1.000053x` |
| Coding agent | 32 | 532,200 | 984,963 | 96.97% | 148.916 us | `1.000049x` |
| Reasoning/verification | 8 | 1,030,592 | 1,215,394 | 88.89% | 43.417 us | `1.000024x` |
| Reasoning/verification | 16 | 1,030,592 | 1,400,186 | 94.12% | 78.083 us | `1.000021x` |
| Reasoning/verification | 32 | 1,030,592 | 1,769,774 | 96.97% | 150.958 us | `1.000020x` |

All byte-conservation and semantic-hash checks pass. These are deterministic serialized/content-addressed bytes, not RSS, OS pages, allocator-resident device pages, HBM, CXL, or physical transfer bytes. Lazy COW defers private materialization into the measured suffix path; it does not eliminate the suffix work.

The independent fanout run reproduced all deterministic byte and semantic fields and the negative headroom conclusion. Absolute readiness medians shifted by 4.6% to 8.4%, with only four of six median CIs overlapping. All six software-effect CIs overlap, and every matched pair favors the lazy shared-root baseline. The drift prevents using the absolute host timings as a calibrated service curve but cannot make `1.000060x` approach the `1.15x` gate.

Every fanout sample has queue capacity one, requested concurrency one, and observed concurrency one. No identical state is physically delivered to multiple destinations, so this campaign contains no measured multicast opportunity.

## Latest reclamation validation

The latest campaign contains the exact deterministic shuffle for order seed `20260809`: three seeds times three trace levels times two baselines times two repetitions, for 36 unique trials. All derived percentiles, operation aggregates, candidate fractions, hashes, and manifest entries reproduce from raw samples.

All 36 trials:

- use `continuum-reference-cpu-local-file` as both requested and actual engine;
- report no hidden fallback;
- report zero physical GPU capacity and zero physical GPU bytes reclaimed;
- reclaim eight reference-scheduler capacity units, preserve twelve fixture ticks, and lose zero fixture ticks;
- reject stale-source and corruption probes;
- complete every transaction and set the reference-plan SLO-restored predicate;
- keep maximum queue occupancy at eight and maximum concurrency at four.

The SLO predicate and reclaimed capacity are deterministic Helix reference-scheduler outcomes, not observed production serving latency or physical GPU reclamation.

### Paired software effects

The following statistics pool all 18 matched `(seed, trace level, repetition)` pairs. Speedup is serial divided by optimized; relative change is optimized minus serial divided by serial.

| Metric | Serial median | Optimized median | Paired speedup, 95% CI | Median relative change | Cohen's dz | Rank-biserial |
|---|---:|---:|---:|---:|---:|---:|
| Branch readiness | 134.939 ms | 136.712 ms | `0.9803x` `[0.9380, 0.9996]` | +2.01% | +0.360 | +0.415 |
| Pause/checkpoint | 266.004 ms | 259.514 ms | `1.0229x` `[0.9638, 1.0708]` | -2.23% | -0.307 | -0.333 |
| Migration | 334.266 ms | 143.463 ms | `2.2778x` `[2.1663, 2.3871]` | -56.10% | -8.314 | -1.000 |
| Resume | 260.767 ms | 268.458 ms | `0.9755x` `[0.9140, 1.0635]` | +2.52% | +0.253 | +0.333 |
| Total interruption | 903.373 ms | 707.210 ms | `1.2721x` `[1.2445, 1.2980]` | -21.38% | -4.919 | -1.000 |
| Total wall | 1,271.659 ms | 1,078.466 ms | `1.1675x` `[1.1448, 1.1875]` | -14.34% | -3.969 | -1.000 |

Migration, interruption, and total wall favor the optimized baseline in every matched pair. Seed-blocked median speedups also preserve direction: migration `2.2733x`, `2.2958x`, and `2.2965x`; interruption `1.2878x`, `1.2474x`, and `1.2718x`; total wall `1.1817x`, `1.1534x`, and `1.1697x` for seeds 41, 73, and 113 respectively.

The optimized local-file baseline transfers 6,812 or 6,816 serialized bytes per trial instead of 54,496 or 54,528, an exact 87.5% reduction through content-addressed reuse. It is not a NIC, RDMA, PCIe, NVLink, or GPU-transfer result.

### Critical-path fraction

For the six optimized full-trace trials, migration is 19.90% of total interruption (95% CI 19.61% to 22.45%). The gate correctly treats this as end-to-end relevance context. It cannot become hardware headroom because there is no target-hardware service curve, calibrated model, or hardware-backed confidence bound.

The observed reclamation software interruption reduction is 21.38%; the lower speedup bound of `1.2445x` corresponds to a 19.65% reduction. Reports should not describe the lower bound as clearing a 20% hardware-reduction gate. More fundamentally, the measured comparison is software versus software, so it cannot populate a hardware-headroom field at all.

## Tracing controls

The reclamation campaign independently pairs disabled, minimal, and full trace modes for every seed, repetition, and baseline. Full-minus-disabled total-wall changes are:

| Baseline | Pairs | Median change | 95% CI |
|---|---:|---:|---:|
| Existing serial | 6 | -0.64% | -3.58% to +5.81% |
| Optimized bounded-parallel | 6 | -1.60% | -3.64% to +1.89% |

Both intervals include zero. This means no tracing penalty is resolved at this sample size; the negative point estimates are not speedups. Because the main software effects are paired across trace levels and retain direction within each seed, tracing does not explain the migration/interruption result.

The separate canonical Helix overhead campaign uses three seeds, two repetitions, one warmup per level, semantic equivalence, identical resource sampling, 58 minimal or 78 full canonical events, and zero drops. It measures a +0.333% full hot-path wall median change and +0.387% full end-to-end wall median change over six pairs. Those paired effects lack paired confidence intervals and use the synthetic Helix CPU demo, so they are context rather than a universal overhead correction.

Fanout itself times every case under `tracemalloc` and has no disabled/minimal/full matrix. The gate therefore correctly leaves tracing uncontrolled for fanout candidates. The separate Helix overhead study cannot retroactively calibrate that different timed region.

## CI and effect-size assessment

The recorded percentile-bootstrap calculations are deterministic, correctly raw-bound, and reproduce exactly. No outliers were removed. Paired effect sizes preserve the same sorted trial keys on both sides.

Two statistical limitations matter for positive claims:

1. Fanout treats nine values per fixture/fanout as exchangeable even though they are three repetitions within each of three deterministic seeds.
2. Reclamation treats 18 pairs as exchangeable even though trace modes and repetitions are clustered within three seeds and all seeds instantiate one reference workload.

A hierarchical seed/workload bootstrap or a larger independent workload sample is required for a positive generalization. These limitations do not weaken the no-build decision: fanout headroom misses by orders of magnitude, and reclamation has no qualifying hardware observation or model regardless of CI width.

## Final gate audit

| Candidate | Real evidence | Relevance | Hardware headroom | Platform | Workload value | Disposition |
|---|---|---|---|---|---|---|
| Checkpoint/transform/transfer chain | FAIL | PASS | FAIL | FAIL | FAIL | `NOT_JUSTIFIED` |
| Shared-root COW | FAIL | FAIL | FAIL | FAIL | FAIL | `NOT_JUSTIFIED` |
| One-to-many multicast | FAIL | FAIL | FAIL | FAIL | FAIL | `NOT_JUSTIFIED` |
| Branch translation metadata | FAIL | FAIL | FAIL | FAIL | FAIL | `NOT_JUSTIFIED` |

The corrected gate uses `CPU_REFERENCE_LOCAL_TRANSACTION` for reclamation and contains zero `TARGET_HARDWARE_REAL` references. It recompiles exactly to:

- outcome `FAIL_NO_BUILD`;
- no passing candidates;
- hardware implementation disallowed;
- functional model and cycle simulator disallowed;
- required action `TERMINATE_HARDWARE_PATH`;
- prior negative result preserved.

## Claim guardrails

Report-safe claims:

- Apple M4 Pro host CPU timing was measured.
- Deterministic CPU-reference model state and local-file state transactions were exercised.
- Fanout 8, 16, and 32 and three deterministic seeds were exercised.
- Shared-root lazy COW and batched bounded-parallel content reuse are the strongest measured software baselines.
- The data-path chain has local CPU-reference relevance, but no hardware-headroom or placement result.
- BranchFabric hardware is not justified and was not implemented.

Unsupported claims:

- a real transformer, production serving workload, 8K/16K/32K context, or production workload distribution was measured;
- Apple integrated-GPU inventory is GPU execution evidence;
- GPU/HBM, PCIe, NVLink, RDMA, NIC, multi-GPU, FPGA, or DPU behavior was measured;
- deterministic serialized bytes are physical memory or wire bytes;
- reference scheduler capacity units are reclaimed GPUs;
- the software speedups are accelerator speedups or calibrated hardware predictions;
- the local reference transaction is a complete production Helix learning transaction.

## Remaining evidence limitations

1. Reclamation has one deterministic reference workload class, six pairs per trace level, no real model, and no production serving latency.
2. Reclamation artifacts record the immutable execution baseline commit but do not embed the exact generating implementation revision. Git history places the latest data in `5ae509a` immediately after software commit `b6279b6`; reports should cite both the data commit and raw digest.
3. Fanout physical accounting is serialized-state arithmetic, and its absolute CPU timing drifts across independent runs.
4. Fanout tracing is not directly controlled.
5. No target hardware, physical network, calibrated cycle model, target resource estimate, or end-to-end hardware effect exists.

## Acceptance

The focused statistical/gate suite passed: 13 tests.

Independent checks also passed for exact randomized matrix order, unique trial keys, raw-to-campaign derivation, manifests, semantic/byte conservation, bounded queue/concurrency fields, engine identity, fault flags, raw-to-analysis reproduction, all gate evidence hashes, and exact gate recompilation.

Machine-readable review: `artifacts/branchfabric/execution/reviews/final-measurement-review.json`.
