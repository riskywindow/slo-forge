# ADR 0045: Keep trainers behind a narrow provenance-complete adapter

## Context

Trainer frameworks differ, but none should be allowed to reinterpret missing behavior policy, lineage,
eligibility, or staleness metadata.

## Decision

Validate a closed training batch before adapter invocation. Pass explicit parent/candidate epochs, samples,
algorithm, steps, and seed; require a hash-bound result. Keep the built-in trainer a small CPU reference.

## Consequences

Alternative trainers can be integrated without weakening admission. Distributed optimizer state and
framework-specific guarantees remain the adapter implementer's responsibility.
