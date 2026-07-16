# ADR 0008: Prefer explicit versioned dirty tracking

- Status: Accepted
- Date: 2026-08-02

## Context

Opaque memory scanning is expensive and cannot reliably identify logical updates.

## Decision

Prefer adapter instrumentation: append logs for KV/history, versions for mutable segments, and copy-on-write for forks. Allow hash comparison only as a declared reduced-efficiency fallback.

## Consequences

Adapters must integrate mutation boundaries, but deltas preserve meaning and avoid device-wide synchronization per token. Overflow/non-convergence is explicit.
