# ADR 0017: Make cross-node clock uncertainty explicit

- Status: accepted
- Date: 2026-08-01

## Context

Autopsy must order events from different clocks without claiming precision that
round-trip samples cannot establish.

## Decision

Estimate offset and linear drift from monotonic/reference clock samples. Use the
round-trip midpoint as the least-assumptive offset sample. Compute residual p95
plus half-RTT p95 as uncertainty and classify alignment as good, degraded, or
insufficient. Store normalized timestamps, confidence, and uncertainty on every
event.

## Consequences

Fine-grained cross-host claims can be rejected when alignment is insufficient.
The linear model will not capture abrupt clock corrections; captures must retain
wall-clock metadata and resample long runs.

