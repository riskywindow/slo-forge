# WarmPath H6 evaluation

This CPU-only comparison resamples measured local startup stages across 11 deterministic seeds. It evaluates 1111 trials per strategy and scenario. P95 intervals are bootstrap intervals over per-seed p95 values.

| Strategy | p50 ready (ms) | p95 ready (ms) | 95% CI of seed p95 (ms) | Hourly cost (USD) | Failure rate | Eviction p95 (ms) |
|---|---:|---:|---:|---:|---:|---:|
| cold_uncached | 3.0046 | 3.1084 | [3.1084, 3.1084] | 0.000002 | 2.97% | 3.1084 |
| local_disk_cache | 0.9089 | 1.0813 | [1.0775, 1.1337] | 0.000002 | 3.60% | 1.0808 |
| page_cache | 0.7848 | 0.9630 | [0.9565, 0.9603] | 0.000000 | 2.88% | 1.0313 |
| pinned_host_memory_modeled | 0.7766 | 0.9476 | [0.9428, 0.9600] | 0.000010 | 2.70% | 1.0305 |
| warmpath | 0.7798 | 0.9554 | [0.9487, 0.9555] | 0.000000 | 3.33% | 1.0303 |
| warm_replica | 0.0000 | 0.0000 | [0.0000, 0.0000] | 0.420000 | 0.18% | 3.0267 |

## Findings

- WarmPath changed p95 readiness by +11.64% relative to the local-disk baseline (positive means faster).
- The best non-warm strategy was `pinned_host_memory_modeled`; WarmPath changed p95 by -0.82% relative to it.
- Under the configured eviction pressure, WarmPath p95 changed by +0.0748 ms.

## Measurement basis and limitations

- `cold_uncached`: declared-tier-bandwidth-floor plus bootstrapped measured checksum verification.
- `local_disk_cache`: bootstrapped measured local-NVMe fetch and verification.
- `page_cache`: bootstrapped measured page-cache fetch and verification.
- `pinned_host_memory_modeled`: bootstrapped measured host-memory copy used as a pinned-memory proxy.
- `warmpath`: compiled WarmPath placements with bootstrapped measured stages.
- `warm_replica`: modeled already-ready replica with declared hourly cost.

- The workload uses a small deterministic synthetic snapshot, not production model weights.
- Pinned host memory is a modeled proxy backed by measured ordinary host-memory copies.
- Cold-cache behavior is a conservative bandwidth-floor model because unprivileged portable cache eviction is unavailable.
- Failure and eviction probabilities are controlled sensitivity scenarios, not observed host failure rates.
- Warm-replica readiness and cost are modeled; no cloud or GPU resource was created.

![H6 p95 readiness](warmpath-evaluation-h6.svg)

Every table value is loaded from `artifacts/warmpath/evaluation/result.json`; individual trial outcomes are preserved in `artifacts/warmpath/evaluation/raw/trials.jsonl`.
