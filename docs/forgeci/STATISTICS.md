# ForgeCI statistics

ForgeCI requires statistical and practical significance. One slow trial cannot
declare a regression.

## Per-run summaries

Warmups are stored separately and excluded. At least three finite measurements
are required. A seeded non-parametric bootstrap of the median produces a
confidence interval; the summary also records median, p95, median absolute
deviation, unit, and every raw sample.

## Comparison

For each bootstrap draw, ForgeCI resamples baseline and candidate independently,
computes both medians, and converts their difference to a degradation percentage
according to metric direction. It reports the point estimate and interval,
Cliff's delta, and a robust noise floor:

```text
noise_floor_percent = 100 * 1.4826 * MAD / abs(median)
practical_threshold = max(configured_threshold, noise_floor)
```

The family-wise significance level uses Bonferroni correction across declared
metrics. A regression requires a positive lower interval bound and degradation
at least the practical threshold. An effect large enough to matter but whose
interval crosses zero is inconclusive. Noise above the configured maximum is
flaky. The symmetric rules classify improvements.

## Fixture result

The deterministic fixture in `artifacts/forgeci/demo/evaluation.json` reported:

- p99 TTFT degradation 12.00%, corrected bootstrap interval
  [11.47%, 12.53%], Cliff's delta 1.0, noise floor 0.31%;
- throughput degradation 10.71%, interval [10.29%, 11.20%], Cliff's delta 1.0,
  noise floor 0.31%.

Both are fixture-generated synthetic process metrics. They are not GPU or
inference-runtime measurements.

## Interpretation limits

Bootstrap intervals quantify sampling variation within a run, not machine-to-
machine or day-to-day environment drift. Bonferroni is conservative but does not
correct an unplanned search across many benchmark definitions. A stable hardware
fingerprint is necessary, not sufficient, for causal commit attribution.

