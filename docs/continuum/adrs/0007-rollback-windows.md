# ADR 0007: Distinguish rollback from post-commit recovery

- Status: Accepted
- Date: 2026-08-02

## Context

Once a destination owns newer state or emits new output, an old source checkpoint is stale.

## Decision

Permit ordinary rollback only before ownership commit while the source remains valid. Classify failures after commit as destination recovery, new migration, recomputation, or operator-required.

## Consequences

Reports cannot hide data reconciliation behind the word rollback. Transactions retain separate commit and rollback watermarks.
