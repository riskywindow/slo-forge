# ADR 0005: Durable CAS coordinator, not custom consensus

- Status: Accepted
- Date: 2026-08-02

## Context

State bytes may exist at multiple locations while only one runtime may mutate or emit accepted output.

## Decision

Use durable leases, fencing tokens, monotonic owner epochs, and compare-and-swap transitions. Provide SQLite for local execution and require an existing linearizable service for distributed deployment.

## Consequences

Local restart recovery is deterministic. Multi-node safety depends on the selected CAS service; Continuum does not implement or claim a consensus protocol.
