# ADR 0013: Reuse versioned JSON over subprocess

- Status: accepted
- Date: 2026-08-01

## Context

Python owns compiler orchestration while Rust owns deterministic scheduling.
Adding PyO3 or a second service protocol would complicate packaging and failure
handling.

## Decision

Reuse SLOForge's versioned JSON-over-subprocess boundary. Fabric simulator input
and output use schema `1.0`, stdin/stdout, canonical encodings, a 64 MiB protocol
limit on the Autopsy path, explicit operation/event bounds, and timeouts. HTTP/SSE
remains exclusive to the serving data plane.

## Consequences

The Rust binary is independently testable and CPU CI exercises the production
wire format. Serialization has overhead, but physical-plan simulation is a
control-plane operation. Additive changes preserve major version one; incompatible
changes require a migration.

