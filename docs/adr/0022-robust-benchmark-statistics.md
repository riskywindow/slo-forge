# ADR 0022: Use robust, provenance-preserving benchmark statistics

- Status: accepted
- Date: 2026-08-01

## Context

Tail performance and regression testing are noisy; a single trial or mean can
produce false claims.

## Decision

Separate warmup from measurement, retain raw samples, report median, p95/p99,
MAD, seeded bootstrap intervals, practical thresholds, noise floors, effect size,
and Bonferroni correction for multiple metrics. ForgeCI classifies noisy or
interval-crossing results as flaky/inconclusive and retries within a budget.
Every summary links to hardware/software and command evidence.

## Consequences

Regression calls require statistical and practical significance. Small fixtures
take more trials, and bootstrap intervals are not substitutes for testing across
independent machines or environments.

