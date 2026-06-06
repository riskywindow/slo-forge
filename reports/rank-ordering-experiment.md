# Collective rank-ordering experiment

- Artifact hash: `1ed6199556658ae957aa5428e2cae85a5dd5c82b514391b3e3abb17d51c63d81`
- Physical plan: `rank-ordering-fixture`
- Collective: `tp-all-reduce`
- Calibration: **synthetic calibrated**
- Search: `exact` (6 candidates)
- Reference order: `[0, 1, 2, 3]`
- Selected order: `[0, 1, 3, 2]`
- Integration decision: **measure_on_hardware**
- Enabled by default: **false**

## Results

| Message bytes | Reference median (us) | Selected median (us) | Median benefit | CI |
|---:|---:|---:|---:|---:|
| 65536 | 35.513 | 14.160 | 59.985% | [59.554%, 60.864%] |
| 1048576 | 388.913 | 134.545 | 65.536% | [65.348%, 65.784%] |

## Integration decision

synthetic calibrated inputs produce confidence-supported modeled benefit in every regime, but the paired A/B trials are digital-twin executions; production enablement requires matched on-device measurements of both orderings

Warmup trials were retained in the raw artifact but excluded from all summaries. Trials use matched per-link perturbations, so the reported interval is paired.

> This is a deterministic synthetic-calibration result, not a GPU measurement. It cannot enable a production default.
