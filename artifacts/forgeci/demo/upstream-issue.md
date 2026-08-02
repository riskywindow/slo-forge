# Fixture runtime performance regression

## Observed regression

ForgeCI classification: **regression**.

- `p99_ttft_ms`: 100.147 → 112.164 ms (+12.00%, CI [+11.47%, +12.53%], Cliff's δ +1.000; regression)
- `throughput_rps`: 99.8534 → 89.1549 request/s (+10.71%, CI [+10.29%, +11.20%], Cliff's δ +1.000; regression)

Warmups were excluded from the measured trials. Intervals use deterministic bootstrap
resampling with multiple-metric correction; practical significance and the measured
noise floor are applied in addition to statistical significance.

## Suspected range

- Known good: `583df29a9e065e79b10a0ff9459e0c1bf05517b5`
- Known bad: `da5d1c5ed8a27aad176e46bff4c87780fc8d84f9`
- First likely regressing commit: `2d43c4d468c9c49986cfba9f516d2a4a20a3a63d`
- Bisection confidence: 97.5%

This identifies a change point, not a proven source-code cause.

## Minimal reproducer

- Model/workload: `synthetic-moe` / `long-context`
- Shape: prompt=1, output=1
- Concurrency: 1
- Expected: both p99 TTFT and throughput degrade by at least 5%
- Confidence interval: family-wise 95% bootstrap intervals exclude zero

```console
    git checkout --detach da5d1c5ed8a27aad176e46bff4c87780fc8d84f9
    /Users/rishivinodkumar/sloforge/.venv/bin/python3 benchmark.py
```

## Environment requirement

- Architecture: `cpu`
- CPUs: at least 1
- Memory: at least 0.25 GiB
- GPUs: 0

## Evidence artifacts

- `artifacts/forgeci/demo/bisect/bisect-result.json`
- `artifacts/forgeci/demo/bisect/comparisons/2d43c4d468c9c49986cfba9f516d2a4a20a3a63d-0.json`
- `artifacts/forgeci/demo/minimization`

## Caveats

- None recorded
