# Agent 18 terminal Amdahl and break-even methodology review

**Verdict: PASS — terminal non-fabrication limits are enforceable.**

The valid attempt-D A100 micro-validation can support measured state-pipeline byte accounting and a narrowly scoped micro-stage sensitivity envelope. It cannot support the five requested integrated Amdahl paths, a fitted preservation break-even, kill/recompute economics, or any of the four final classifications. Integrated attempt B failed closed before the reclaim trigger; optimized export, source release, HBM reclamation, GPU1 service recovery, import, restore, admission, and restore interference were never reached.

## What is measured

The attempt-D micro result measures a 1,056,964,608-byte logical state at `P=16,384`, `B=8`, and `A=256`. This is 18,432 unique logical token-equivalents, hence `q=57,344` bytes per unique logical token-equivalent. Full physical work is 22,197,063,040 bytes (`21.000762818351625x`) in 10 canonical full-state pass families.

The two non-overlapping endpoint stages are:

| Micro stage | Physical bytes | Canonical passes | Wall | CUDA-event time | Sync wait | Integrity owner wall | Share of export+restore endpoint sum |
|---|---:|---:|---:|---:|---:|---:|---:|
| Source export | 9,512,709,120 | 4 | 2.613090870 s | 0.133267776 s | 0.005271845 s | 1.390905234 s | 37.0584% |
| Destination restore | 12,684,353,920 | 6 | 4.438193614 s | 0.697858271 s | 0.516524914 s | 2.114326745 s | 62.9416% |

The destination restore stage is the dominant measured micro stage by both endpoint wall time and physical bytes. That is not evidence that it occupies 62.94% of HBM reclamation, SLO restoration, or any integrated serving path.

CUDA-event time, synchronization wait, integrity wall spans, timing-owner wall spans, and endpoint wall are different timing views. They are not additive. The source and restore timing-owner wall unions are 1.501838905 s and 2.928762252 s; the remaining endpoint durations, 1.111251965 s and 1.509431362 s, cannot be classified as CPU, GPU, idle, or synchronization from these records alone.

## Defensible micro-only sensitivity

The only clean Amdahl-style calculation uses the non-overlapping endpoint sum `T=E+R=7.051284484 s`. If an entire endpoint stage `X` were accelerated, `T'(k)=T-X+X/k`. This is an optimistic whole-stage diagnostic envelope, not a realistic fused-chain or hardware projection.

| Accelerated micro endpoint | 2x | 5x | 10x | Effectively free |
|---|---:|---:|---:|---:|
| Source export only | 1.2274x | 1.4214x | 1.5004x | 1.5888x |
| Destination restore only | 1.4592x | 2.0142x | 2.3067x | 2.6984x |

All requested projections for time to HBM reclamation, useful GPU1 capacity, serving-SLO restoration, rollout restore, and the full integrated preservation transaction remain `NOT_REACHED`. The micro envelope must not be used to satisfy the hardware gate's realistic `>=1.20x` predicate.

## Break-even is symbolic only

The interpretable model remains:

`T_p = delta_p + c(P+B*A)`, where `c=a*q/beta`

`T_k = delta_k + B(P+A)/theta`

Optimized preservation wins exactly when:

`(delta_p-delta_k) + (c-B/theta)P + B(c-1/theta)A < 0`.

Attempt D identifies `q`, the one-point `a`, and a micro endpoint sum. It does not identify independent recompute throughput `theta`, kill setup `delta_k`, preserve fixed overhead `delta_p`, movement rate `beta`, or how amplification scales with `P`, `B`, and `A`. One preserve point cannot separate a fixed intercept from a byte slope. The analytical `133,120 = 8*(16,384+256)` recompute-token count is verified, but it is not a measured kill arm.

The independent continuation oracle is correctness evidence over prepared runtime state. Its elapsed time must not be used as causal prefill throughput or recompute cost. Also, the current model uses one `A` variable, so branch age and private suffix are the same degree of freedom; they cannot be fitted independently.

No numeric prefix-length or branch-count curve is admissible. Under the deliberately artificial assumptions `delta_k=0` and micro endpoint time equals integrated preserve time, the frozen-point equality would be `theta=18,878.829850371415 tokens/s`; this is a sensitivity identity only and must not be plotted or reported as measured break-even.

## Required plot handling

Plots 1–4 and 13 may compare frozen v10 movement with measured attempt-D micro movement if the v11 series is explicitly labeled micro. Plots 5–9 and 14 require `NOT_REACHED` v11 or kill cells. Plots 10–11 may show only the symbolic equations, not numeric curves. Plot 12 may optionally show the watermarked `MICRO_STAGE_ONLY` envelope above, while every integrated Amdahl path remains `NOT_REACHED`.

Missing values must be null with explicit status, never zero-filled, interpolated from v10, or replaced by micro endpoints.

## Classification constraint

None of the four allowed classifications is scientifically established:

- `SOFTWARE_WINS` lacks an integrated important-path fraction or ideal-free full-system speedup.
- `GPU_SOFTWARE_WINS` lacks realistic GPU-only and external-accelerator incremental end-to-end headroom.
- `BRANCHFABRIC_HARDWARE_INTEREST` is explicitly barred without measured integrated timings and lacks critical-path, economic, concurrency, interference, and placement evidence.
- `PRESERVATION_NOT_ECONOMIC` lacks the kill/recompute arm.

The honest terminal label is `FINAL_CLASSIFICATION=NOT_DETERMINED_FROM_AVAILABLE_EVIDENCE`. Forcing one enum member would violate its stated predicate and fabricate a conclusion. Experiment 005 must not be generated.

## Provenance

- Attempt-D scientific outcome: `4202d4d0ed936ae8364516113fca3dba2a41a85646e0a9324b047f186cc68494`
- Attempt-D worker result: `fdd9859ccec9187facd47d6efa621449899d18b009a7b15ea746e45e1139ecb8`
- Measured v10-v11 pass diff: `7c469b716b38314cec340e761896b1845ab070154069ad81b78bdc3b28fbbd99`
- Frozen v10 movement accounting: `ad31263ad118a2107cb9d8806b06052866144a63e7e79b2af27240ca5f5939f3`
- Agent 15 terminal integrated review: `0fe2b3e28d003e9506aaf8c8b1e18b1d1afafa0b4ca5992d877da5b9a3d923a2`
- Agent 16 terminal economic review: `d6eb0ba6b9727e73c5db980f6560b6812e410f8c409a453c6cc65eee44e6dae9`
- Integrated postflight cleanup/settlement: `82678f57e9e346f1b4dc68fd6df15d6685987aeda44ef1fd30e7284a5ab55180`
