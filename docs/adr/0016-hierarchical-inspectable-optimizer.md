# ADR 0016: Use hierarchical, inspectable physical optimization

- Status: accepted
- Date: 2026-08-01

## Context

Joint parallelism, placement, transport, and recovery selection is combinatorial.
An opaque learned policy would be difficult to constrain and explain with the
available hardware data.

## Decision

Enumerate parallelism choices, prune statically, construct placements with
deterministic topology scoring, estimate communication from measured curves, and
rank a robust objective. Retain exhaustive, random, sequential topology-unaware,
greedy topology-aware, hierarchical, and failure-robust strategies. Emit every
rejection and the Pareto frontier.

## Consequences

Decisions are reproducible and reviewable. Analytical ranking is not called a
simulator execution; validation is a separate Rust pass. Very large search spaces
will require more aggressive decomposition, but core invariants stay visible.

