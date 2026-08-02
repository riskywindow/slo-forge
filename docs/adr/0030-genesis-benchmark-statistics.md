# ADR 0030: Gate Genesis performance with raw repeated uncertainty-aware evidence

- Status: accepted
- Date: 2026-08-02

## Context

ADR 0022 establishes robust provenance-preserving benchmark statistics. Generated candidates add incentives for benchmark gaming and may appear faster because of noise, altered inputs, precision, fallback, cache state, or missing synchronization.

## Decision

Bind performance evidence to benchmark, metric/unit/direction, workload, hardware, software, warmup, noise floor, practical threshold, seed, and raw samples. Require at least seven positive observations per alternative. Use medians, seeded bootstrap confidence intervals over improvement, and a pairwise probability-of-superiority effect size.

Pass only when the complete interval exceeds both practical significance and measured noise; fail a significant regression; otherwise report inconclusive. Retain all samples and a seeded randomized-order plan. Require the collection harness to execute and attest that plan, equal warmup, synchronization, affinity, environment, precision, quality, and fallback behavior.

## Consequences

A favorable point estimate cannot promote a candidate, and noisy results remain honest. More repetitions cost time, bootstrap assumptions do not replace independent machines, and the focused statistical function cannot prove that supplied samples were collected correctly. ServingSynthBench and capsule provenance checks provide additional integrity evidence; hardware claims still require actual hardware runs.

