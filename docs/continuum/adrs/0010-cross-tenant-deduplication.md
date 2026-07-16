# ADR 0010: Disable cross-tenant deduplication

- Status: Accepted
- Date: 2026-08-02

## Context

Global content hashes can expose whether another tenant holds equal sensitive state.

## Decision

Scope chunk keys and manifests by tenant. Equal plaintext in separate tenants has independent authorization/reference namespaces by default.

## Consequences

Some storage savings are forgone. Within-tenant equality remains observable to that tenant's authority and still requires hash/metadata access control.
