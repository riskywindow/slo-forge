# Collective rank-ordering experiment

- Artifact hash: `dd3d14c4b13cbf27282d45f524ff7d72ee25bb6e35bf658dbfcb0bc5dc7e9856`
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

synthetic calibrated trials show confidence-supported benefit in every regime, but production enablement requires matched hardware measurements

Warmup trials were retained in the raw artifact but excluded from all summaries. Trials use matched per-link perturbations, so the reported interval is paired.

> This is a deterministic synthetic-calibration result, not a GPU measurement. It cannot enable a production default.
