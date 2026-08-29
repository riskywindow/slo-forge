# BranchFabric GPU Validation Experiment 004 v10

Status: **SCIENTIFICALLY VALID NAIVE-PRESERVATION BASELINE**.

This report freezes the existing naive state path. No optimized-preservation, kill/recompute, Experiment 005, RTL, HLS, simulator, or FPGA work was executed.

## Calibrated serving envelope

- λ₁: `12.0` requests/s
- λ₂: `20.0` requests/s
- λ_spike: `15.0` requests/s
- One-GPU overload margin: `0.250000`
- Two-GPU reserve margin: `0.250000`
- One-GPU unstable point: `15.0` requests/s

## Control and bounded overload

- Control completions: `36`
- Control p95 TTFT: `42308889.25` ns
- Overload completions: `49`
- Overload p95 TTFT: `266840026.2` ns

## Capacity reclamation

- State reclamation: `5.222174` s
- First useful GPU1 capacity: `5.413681` s
- Stable excess two-GPU service: `18.576226` s
- Stable serving SLO restoration: `11.373208` s
- Two-GPU offered rate: `14.966710716479456` requests/s
- Two-GPU completed rate: `15.878388526620336` requests/s

## Restore and correctness

- First resumed token: `7.727983` s after restore trigger
- All branches restored: `7.727983` s
- Exact first-token matches: `8/8`
- Minimum continuation length: `8` tokens

## Frozen naive movement baseline

- Logical state: `1056964608` bytes
- Full physical touch: `61303947264` bytes (`58.000000x`)
- External movement: `3.000000x`
- Avoidable movement: `31.000000x`
- Critical-path movement: `58.000000x`
- Dominant chain: native read → repack/unpage/stack/concatenate → D2H → integrity/publish verification → H2D → import verification → native write → destination readback/validation D2H

## Serving interference and budget

GPU0 restore-phase measurements, sample eligibility, CPU/host pressure, and PCIe observations are recorded in the JSON report. Unsupported p95 values remain null.

- Cumulative Experiment 004 GPU use: `3.488007` hours
- Cumulative estimated GPU cost: `$8.714436`
- Modal cleanup gate: `PASS`

The frozen baseline is ready for the separately planned A/B/C campaign.
