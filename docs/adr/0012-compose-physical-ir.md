# ADR 0012: Compose the physical IR with the logical plan

- Status: accepted
- Date: 2026-08-01

## Context

Fabric needs rank, link, collective, memory, and recovery detail without breaking
the stable logical `DeploymentPlan` or duplicating model/SLO policy.

## Decision

Create a versioned `PhysicalExecutionPlan` that references the logical plan by
kind, API version, URI, digest, UID, and generation. Bind model, topology, and
profile inputs by canonical SHA-256. Keep core fields strict; permit only the
existing namespace-qualified extension object. Maintain matching Python, Rust,
JSON Schema, migration, and golden-fixture representations.

## Consequences

Logical exporters remain compatible and physical recompilation can occur without
rewriting the logical plan. Consumers must resolve evidence references and
reject unsupported major versions. The additional hashes make stale profile or
topology reuse a hard error rather than a warning.

