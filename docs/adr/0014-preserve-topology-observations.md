# ADR 0014: Preserve discovery observations before canonicalization

- Status: accepted
- Date: 2026-08-01

## Context

sysfs, cgroups, vendor tools, and container metadata can disagree. Flattening
them immediately would conceal uncertainty and capability gaps.

## Decision

Use `DiscoveryTopologyGraph` as the evidence layer. Facts are known, unknown, or
conflicting and retain every observation and provenance record. Convert to the
strict compiler `TopologyGraph` only after applying fail-closed normalization.
Never infer GPU, RDMA, GPUDirect, or transport reachability solely from absence.

## Consequences

Reports can explain exactly where a fact came from. Compiler inputs are simpler
and typed, while raw discovery remains auditable. Some environments cannot
produce a complete physical plan until the missing fact is supplied or measured.

