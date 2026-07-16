# ADR 0016: Keep BIFROST optional and independent

- Status: Accepted
- Date: 2026-08-02

## Context

A BIFROST-like system may provide high-performance chunk storage and movement, but it is not present or stable in every Continuum environment.

## Decision

Treat BIFROST as an optional `StateTransport`/content-store backend. Continuum retains state semantics, conversion, planning, validation, and ownership. Normal CI has no BIFROST dependency.

## Consequences

An available backend can be integrated without duplicating its movement layer. The current repository makes no exercised BIFROST performance or compatibility claim.
