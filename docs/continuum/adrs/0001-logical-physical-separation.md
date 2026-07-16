# ADR 0001: Separate logical and physical state

- Status: Accepted
- Date: 2026-08-02

## Context

Runtime pages, shards, padding, and placements can change without changing what is required to continue a session. Conversely, equal tensor shapes can represent different model semantics.

## Decision

Use independent, versioned `LogicalStateSchema` and `PhysicalStateLayout` documents linked by stable semantic component IDs. Exclude raw pointers and process ephemera.

## Consequences

Layout/TP changes compile as representation transformations; model compatibility remains semantic. Adapters must do more work than expose allocations, but runtime internals do not become the portable ABI.
