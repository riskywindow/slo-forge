# ADR 0018: Derive diagnosis confidence from structured evidence

- Status: accepted
- Date: 2026-08-01

## Context

Root cause must be reproducible, falsifiable, and independent of language-model
output.

## Decision

Match healthy and degraded stages, derive typed counters and rank skew, evaluate
27 thresholded signal rules, and retain supporting and contradicting evidence.
Confidence combines effect over threshold, the sample count for that specific
stage/counter signal, and bounded causal specificity. Alignment quality gates
cross-host ordering claims and emits explicit warnings; it is not multiplied
into unrelated duration evidence. Then alter the exact degraded simulator input,
attach the best supported, contradicted, or inconclusive estimate per hypothesis,
and rerank the structured record. A no-signal comparison rejects every rule and
emits a warning.

## Consequences

Every conclusion has machine-readable evidence and rejected alternatives.
Confidence is a deterministic engineering score, not a population probability;
its calibration must be reported separately for each evaluated fault corpus.
