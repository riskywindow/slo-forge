# Fixture runtime performance regression

## Observed regression

ForgeCI classification: **regression**.

- `p99_ttft_ms`: 100.147 → 112.164 ms (+12.00%, CI [+11.47%, +12.53%], Cliff's δ +1.000; regression)
- `throughput_rps`: 99.8534 → 89.1549 request/s (+10.71%, CI [+10.29%, +11.20%], Cliff's δ +1.000; regression)

Warmups were excluded from the measured trials. Intervals use deterministic bootstrap
resampling with multiple-metric correction; practical significance and the measured
noise floor are applied in addition to statistical significance.

## Suspected range

- Known good: `1efc47156bacf885ae1253bf7039b7e56977ee68`
- Known bad: `b67ae11c2464ab51afe38862f33950c7d9890e0c`
- First likely regressing commit: `17afc8fb225f09c2deb0d0bd4509107892bfd87a`
- Bisection confidence: 97.5%

This identifies a change point, not a proven source-code cause.

## Minimal reproducer

- Model/workload: `synthetic-moe` / `long-context`
- Shape: prompt=1, output=1
- Concurrency: 1
- Expected: both p99 TTFT and throughput degrade by at least 5%
- Confidence interval: family-wise 95% bootstrap intervals exclude zero

```console
    git checkout --detach b67ae11c2464ab51afe38862f33950c7d9890e0c
    /Users/rishivinodkumar/sloforge/.venv/bin/python3 benchmark.py
```

## Environment requirement

- Architecture: `cpu`
- CPUs: at least 1
- Memory: at least 0.25 GiB
- GPUs: 0

## Evidence artifacts

- `artifacts/helix/baseline/forgeci-demo/bisect/bisect-result.json`
- `artifacts/helix/baseline/forgeci-demo/bisect/comparisons/17afc8fb225f09c2deb0d0bd4509107892bfd87a-0.json`
- `artifacts/helix/baseline/forgeci-demo/minimization`

## Caveats

- None recorded
