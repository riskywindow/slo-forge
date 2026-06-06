# Cross-node time alignment

Autopsy aligns monotonic clocks without assuming synchronized nanosecond wall
time.

## Estimator

A `ClockSample` contains local monotonic time, reference monotonic time, round
trip, and capture wall time. Samples for one host are sorted by local time. The
reference observation is corrected by half the round trip, giving one offset
estimate. A bounded Theil-Sen fit over at most 256 lowest-RTT observations
estimates intercept and linear drift in parts per million. This robust fit keeps
one delayed exchange from arbitrarily changing the inferred clock relation.

Residual p95 plus half-RTT p95 is the reported uncertainty. A timestamp is
normalized as:

```text
reference_time = local_time + offset
               + (local_time - fit_reference) * drift_ppm / 1,000,000
```

Normalization rejects overflow, negative values, reversed intervals, and host
mismatch.

## Quality policy

- `good`: at least five samples and uncertainty no greater than 250 microseconds;
- `degraded`: at least three samples and uncertainty no greater than 5 ms;
- `insufficient`: all other cases.

Confidence increases with sample count but is capped by the quality class.
`align_run(require_usable=True)` refuses missing or insufficient hosts. Diagnosis
with an explicitly retained insufficient alignment warns that fine-grained
cross-host ordering is unsupported. First-divergence attribution is also omitted
when candidate event intervals overlap after alignment uncertainty is applied.

## Interpretation

The model handles steady offset and linear drift. It does not prove precision
below the uncertainty, capture PTP clock quality, or model discontinuous clock
steps. Long captures should resample clocks; events around a discontinuity need
separate alignment epochs. Simulator events use the synthetic source clock and
have exact shared time within that run.
