# ADR 0038: Make environment state a first-class content-addressed capsule

## Context

Model checkpoints alone cannot reproduce agent behavior when files, dependencies, services, clocks,
or resource policy differ.

## Decision

Capture declared environment state into a strict `EnvironmentStateCapsule`, store file bytes by
digest, reject unsafe paths and symlinks, redact sensitive values, and reconstruct bounded tenant-scoped
branches from the capsule.

## Consequences

Declared local state is inspectable and branches share immutable content. Capsule construction costs
storage and capture time. Kernel state, credentials, networks, and undeclared remote services remain
outside the claim.
