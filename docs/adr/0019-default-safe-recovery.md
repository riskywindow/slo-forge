# ADR 0019: Require simulation, shadow, and canary before recovery promotion

- Status: accepted
- Date: 2026-08-01

## Context

Changing placement or transport can interrupt streaming requests or amplify an
incorrect diagnosis.

## Decision

Use a persisted guarded state machine: `PROPOSED`,
`VALIDATED_IN_SIMULATION`, `BUILDING_REPLACEMENT`, `SHADOWING`, `CANARYING`,
`PROMOTING`, `DRAINING_OLD`, and `COMPLETED`, with explicit rejection, abort,
rollback, and operator-required states. Require sample counts, promotion and
error criteria, bounded deadlines, idempotency keys, graceful drain, and
preservation of started streams. External mutation requires two-sided opt-in.

## Consequences

Local and simulated recovery is safe by default and restartable from a snapshot.
Restoration is slower than an unguarded in-place change, but the transition has
observable abort and rollback points.

