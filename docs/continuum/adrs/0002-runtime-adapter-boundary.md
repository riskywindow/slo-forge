# ADR 0002: Version-scoped runtime adapter boundary

- Status: Accepted
- Date: 2026-08-02

## Context

Inference runtime internals and public cache APIs change independently and expose different subsets of session state.

## Decision

Require each adapter to publish typed capabilities, explicit resource limits, runtime/build versions, and supported state/layout operations. Unsupported behavior returns typed errors with no hidden fallback.

## Consequences

Optional runtime packages remain non-mandatory and unstable APIs stay isolated. Package/API discovery is not sufficient evidence of migration support.
