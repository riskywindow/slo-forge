# ADR 0018: Derive diagnosis confidence from structured evidence

- Status: accepted
- Date: 2026-08-01

## Context

Root cause must be reproducible, falsifiable, and independent of language-model
output.

## Decision

Match healthy and degraded stages, derive typed counters and rank skew, evaluate
27 thresholded signal rules, and retain supporting and contradicting evidence.
Confidence combines effect over threshold, matched sample count, bounded causal
specificity, and alignment quality. Then alter the exact degraded simulator input
and attach supported, contradicted, or inconclusive counterfactual estimates.

## Consequences

Every conclusion has machine-readable evidence and rejected alternatives.
Confidence is a deterministic engineering score, not a population probability;
its calibration must be reported separately for each evaluated fault corpus.

