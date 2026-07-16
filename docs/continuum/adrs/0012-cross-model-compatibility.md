# ADR 0012: Dependency proofs govern cross-model reuse

- Status: Accepted
- Date: 2026-08-02

## Context

Cached state depends on weights and update equations even when shapes are unchanged.

## Decision

Reject changed state-producing dependencies by default. Permit unaffected output-head changes only with dependency evidence; otherwise recompute from authorized history/checkpoints or reject.

## Consequences

Continuum does not offer universal cross-model cache reuse. Dependency graphs and replay side-effect policy become part of the compatibility contract.
