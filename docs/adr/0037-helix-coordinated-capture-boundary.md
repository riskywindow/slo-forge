# ADR 0037: Publish Helix branch points through coordinated capture

## Context

Model, environment, action, and effect state advance on different clocks. Pairing independently
captured snapshots by wall time can create a counterfactual from a state that never existed.

## Decision

Use a durable phased barrier with typed per-domain watermarks, bounded quiescence, independent source
validation, and atomic `BranchPoint` publication. Persist attempts and fail terminally on mismatch;
never publish a partial branch point.

## Consequences

Counterfactuals have an auditable decision boundary and recovery is idempotent. Capture adds latency
and requires source adapters. SQLite proves local transactionality only; distributed deployment needs
a linearizable coordinator.
