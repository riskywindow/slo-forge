# ADR 0027: Use a self-contained bounded explicit-state protocol checker

- Status: accepted
- Date: 2026-08-02

## Context

Streaming, cancellation, retry, state transfer, promotion, rollback, and controller recovery have interleavings that example tests miss. CI must not depend on proprietary or separately installed formal-methods tools, and bounded evidence must not be called a universal proof.

## Decision

Implement a deterministic Rust breadth-first explicit-state checker over a typed transition vocabulary. Bound requests, queue, tokens, workers, failures, depth, states, and fairness window. Report every invariant independently with model version, state/transition counts, action coverage, assumptions, completeness, truncation, and a replayable shortest counterexample.

Classify a truncated property as inconclusive and always set `universal_proof=false`. Validate request/result JSON with versioned schemas and reject oversized or invalid input before exploration. Keep optional external-formalism export separate from normal CI.

## Consequences

The checker finds and minimizes concrete protocol failures and has deterministic CI behavior. State explosion limits scope, and atomic transitions abstract real threads, networks, tensor operations, and external runtime internals. A generated protocol change is covered only after it is represented in this model.

