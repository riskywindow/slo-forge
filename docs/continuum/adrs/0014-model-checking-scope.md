# ADR 0014: Bounded explicit-state protocol checking

- Status: Accepted
- Date: 2026-08-02

## Context

Crash/retry/reorder interactions are difficult to cover with hand-written scenario tests alone.

## Decision

Explore a deterministic finite model with explicit queue, token, depth, state, timeout, and fault bounds; record all assumptions and minimize counterexamples.

## Consequences

The checker detects protocol mutations and provides reproducible evidence. Results are never described as proof outside the recorded bounds/model.
