# ADR 0003: Canonical logical fallback

- Status: Accepted
- Date: 2026-08-02

## Context

Optimized direct converters are harder to trust and debug.

## Decision

Maintain a canonical logical decode/re-encode CPU implementation as the trusted fallback and independent oracle. Canonical JSON and SHA-256 define stable document identity.

## Consequences

Correctness remains available when direct lowering is unsupported. The fallback can require more temporary memory and movement, so it is not always selected for production execution.
