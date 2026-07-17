# SLOForge Continuum CPU Evaluation

Evaluation ID: `385704604314a07571752635f41961674ccbe191ba843cbe4e567073e0431037`

This report is generated from authenticated per-seed artifacts. Observed host timings use `perf_counter_ns`; fields prefixed with `synthetic_` come from the deterministic transport/protocol model and are not hardware measurements. `artifact_derived` values are exact counts from the sealed state manifests.

## Reproduction

```sh
uv run --locked python -m sloforge.continuum.benchmarking --output artifacts/continuum/evaluation --seeds 101,202,303,404,505 --git-commit 8bba247b44d26e1be4475ca18058fcbc652adcf0 --initial-output-tokens 16 --delta-rounds 3,2 --resumed-tokens 3 --converter-repetitions 5 --reset
```

## Environment manifests

- Python: CPython 3.12.7
- Platform: macOS-15.6.1-arm64-arm-64bit
- Git commit: `8bba247b44d26e1be4475ca18058fcbc652adcf0`
- Mode: cpu_only
- Machine: arm64
- Logical CPUs: 12
- GPU exercised: false
- RDMA exercised: false

| Package | Version |
|---|---|
| numpy | 2.3.5 |
| pydantic | 2.13.4 |
| sloforge | 0.1.0 |

## Confidence intervals

All intervals are two-sided 95% Student-t intervals across independent seeds.

| Metric | Class | Unit | n | Mean | 95% CI |
|---|---|---:|---:|---:|---:|
| flagship_wall_time | observed_host | milliseconds | 5 | 112.212 | [110.703, 113.722] |
| canonical_conversion_median | observed_host | microseconds | 5 | 77.842 | [72.949, 82.735] |
| direct_conversion_median | observed_host | microseconds | 5 | 160.783 | [151.441, 170.125] |
| precopy_observed_interruption | observed_host | milliseconds | 5 | 9.330 | [4.391, 14.270] |
| stop_copy_observed_interruption | observed_host | milliseconds | 5 | 26.509 | [26.242, 26.777] |
| planner_regret | synthetic_protocol | objective_units | 5 | 0.000 | [0.000, 0.000] |
| planner_interruption_absolute_error | observed_host | milliseconds | 5 | 8.315 | [3.375, 13.254] |
| synthetic_transport_bytes_on_wire | synthetic_protocol | bytes | 5 | 20548.600 | [20286.463, 20810.737] |
| synthetic_final_delta_transfer | synthetic_protocol | microseconds | 5 | 963.200 | [962.161, 964.239] |
| checkpoint_bytes_deduplicated | artifact_derived | bytes | 5 | 10198.400 | [10187.699, 10209.101] |

## Per-seed raw results

| Seed | Flagship wall ms | Pre-copy pause ms | Stop-copy pause ms | Canonical µs | Direct µs | Selected | Planner regret | Prediction error ms | Synthetic wire bytes | Tokens | COW bytes saved |
|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|
| 101 | 113.982 | 7.472 | 26.329 | 84.667 | 171.041 | canonical_cpu | 0.000000 | 6.456 | 20848 | 25 | 10206 |
| 202 | 111.222 | 7.504 | 26.310 | 76.291 | 155.292 | canonical_cpu | 0.000000 | 6.489 | 20292 | 25 | 10197 |
| 303 | 111.426 | 16.446 | 26.845 | 77.583 | 166.625 | canonical_cpu | 0.000000 | 15.431 | 20469 | 25 | 10198 |
| 404 | 112.984 | 7.670 | 26.541 | 74.834 | 154.875 | canonical_cpu | 0.000000 | 6.655 | 20656 | 25 | 10206 |
| 505 | 111.447 | 7.558 | 26.520 | 75.833 | 156.083 | canonical_cpu | 0.000000 | 6.543 | 20478 | 25 | 10185 |

## Hypothesis outcomes

| Hypothesis | Status | Result | Scope |
|---|---|---|---|
| H1 | pass | Exact-semantic continuation passed across TP, page-size, and K/V packing changes. | Scoped to the deterministic CPU HybridDecoder reference adapters. |
| H2 | negative | Direct CPU conversion did not beat canonical CPU conversion in this matrix. | Observed CPU wall timings are environment-specific and do not imply GPU speedup. |
| H3 | pass | Observed pre-copy cutover was shorter than the CPU stop-and-copy pause in 5/5 seeds. | Both paths use CPU reference runtimes and in-memory stores on this host. |
| H4 | pass | Planner objective matched exhaustive legal-candidate enumeration in every seed; predicted-versus-observed interruption error is reported separately. | Oracle comparison is scoped to the planner's tractable deterministic candidate set; observed interruption comes from the CPU reference pre-copy path. |
| H5 | pass | Every seeded destination-validation crash rolled back with zero accepted duplicate or gap events. | Bounded deterministic destination-validation crash scenario only. |
| H6 | pass | Reference token-major to reference head-major adapters preserved logical continuation. | vLLM, SGLang, PyTorch, and Genesis migration paths were not exercised here. |
| H7 | pass | Every seed reused content-addressed checkpoint chunks before COW divergence. | In-memory content store; no remote-store startup latency was measured. |
| H8 | pass | Every changed state-producing weight revision was rejected for direct reuse. | One attention-weight dependency pattern plus recomputation evidence was exercised. |
| H9 | pass | Recurrent, sampler, and guided state migrated with attention KV state in every seed. | Reference integer recurrent update equation only. |

## Negative and unexercised results

- No GPU, RDMA, multi-node, or cloud benchmark was executed; no hardware speedup is claimed.
- vLLM, SGLang, PyTorch, and Genesis active-state migrations were not exercised.
- Direct CPU conversion won 0/5 observed per-seed median comparisons.
- Synthetic transport elapsed values model configured bandwidth and latency; they are not wall-clock measurements.

## Raw artifact provenance

- Seed 101 flagship: `raw/seed-101/flagship.json` (SHA-256 `75a533e3e4d2da88be62cbbf5505f0ce2cd47ba1557f916c93c430f4c4e0161b`)
- Seed 101 conversion: `raw/seed-101/conversion.json` (SHA-256 `09cf2a35596e5ae6014615fb3f4b6f0287514bc99f9ae354bc25f6060e1be016`)
- Seed 202 flagship: `raw/seed-202/flagship.json` (SHA-256 `86d2cfd0ab2f0c9fbf336aeea86f138beb3e08ed16f4debad23bc3bc37d1f4b5`)
- Seed 202 conversion: `raw/seed-202/conversion.json` (SHA-256 `e08c67b2931a0c212ab2660f3531a95c771c1b83252696b79ebee4beb9053012`)
- Seed 303 flagship: `raw/seed-303/flagship.json` (SHA-256 `a8c92737b362312a6ee21f3764d9e828f3943153f37c80c77af50768a5c97a27`)
- Seed 303 conversion: `raw/seed-303/conversion.json` (SHA-256 `e349144c21d6cc2d5ebef8e87dbc635a3d4ef255c06f3ee688bb011997d2bd28`)
- Seed 404 flagship: `raw/seed-404/flagship.json` (SHA-256 `cf1e6908acb26db7bdce2b377f116860ff56a558b4350a19a7bd608ca281a09c`)
- Seed 404 conversion: `raw/seed-404/conversion.json` (SHA-256 `d4e8fae3bdb16596c29d895f50c8733553619a247d79dfcadcd64f86b0063c2f`)
- Seed 505 flagship: `raw/seed-505/flagship.json` (SHA-256 `b8e20b08469a2289713c3bde972c15871d7d6bde47d2e8c72e671e8d86e71f64`)
- Seed 505 conversion: `raw/seed-505/conversion.json` (SHA-256 `d1300c58ff008f246d1bdb6a4c5ef50c700ae22eb25759fc90ad71c6c8d423c7`)
