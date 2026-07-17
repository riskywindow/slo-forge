# ADR 0050: Reuse versioned JSON for the Helix Rust/Python boundary

## Context

Helix needs shared IR and deterministic simulation without introducing in-process FFI into the serving path.

## Decision

Keep orchestration, modeling, validation, and reports in Python; keep shared wire types and deterministic
data-plane/simulation work in Rust. Exchange bounded versioned JSON over subprocess stdin/stdout and maintain
backward compatibility within an IR major version. Reserve HTTP/SSE for the running data plane.

## Consequences

The boundary is inspectable, replayable, and crash-isolated, reusing ADRs 0001 and 0013. Serialization costs
remain, and schema evolution requires conformance fixtures and explicit major migrations.
