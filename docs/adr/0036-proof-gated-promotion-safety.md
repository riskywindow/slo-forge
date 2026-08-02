# ADR 0036: Revalidate capsules and preserve stream ownership during promotion

- Status: accepted
- Date: 2026-08-02

## Context

A generated challenger may change after initial validation, fail under traffic, require incompatible
state, or orphan active streams. Local simulation success is not authority to mutate an external
deployment.

## Decision

Keep challengers isolated. Require an independently validated, digest-matching capsule before
shadowing; bounded shadow and canary gates; and capsule revalidation immediately before promotion.
Reject failed gates without changing the champion. Policy-only and request-boundary-safe changes
pin existing stream leases to their original capsule while routing new requests to the new champion.
Other transitions drain unless active-stream compatibility is independently established. Require
verified conversion evidence for state-conversion migration, and require an operator for explicitly
operator-required transitions.

Persist each transition before exposing it, bound state/audit collections, detect persisted-state
tampering, retain the prior champion and any runtime with active leases, and provide rollback.
External live canarying/promotion requires controller authorization plus
`SLOFORGE_GENESIS_ALLOW_LIVE_PROMOTION=1`. Capsule validation does not grant that authority and
synthetic performance evidence does not establish hardware readiness.

## Consequences

Promotion is slower and retains overlapping runtimes/state, but active streams and rollback remain
well defined. Controller restart can recover the persisted phase, and a changed capsule is rejected
at the last gate. The implemented local/simulated state machine does not by itself establish safety
for unexercised production, multi-node, state-conversion, or GPU paths.
