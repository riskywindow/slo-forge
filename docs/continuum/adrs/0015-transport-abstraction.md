# ADR 0015: Byte-only transport abstraction

- Status: Accepted
- Date: 2026-08-02

## Context

Deployment transports vary by host and fabric, but semantic migration rules must be consistent.

## Decision

Define `StateTransport` around bounded chunk transfer, acknowledgments, retries, deadlines, cancellation, integrity, and declared capabilities. Compatibility remains above transport.

## Consequences

In-process, file, simulated, TCP, and optional high-performance transports are interchangeable only when capabilities match. There is no silent fallback.
