# ADR 0004: Verify direct physical-to-physical conversion

- Status: Accepted
- Date: 2026-08-02

## Context

Full canonical materialization increases bytes read/written and temporary memory during live migration.

## Decision

Compile bounded chunk-level transformations directly between compatible physical layouts, but quarantine output until it matches the canonical converter for the declared domain.

## Consequences

Conversion can overlap transfer and avoid full logical allocation. Each lowering has explicit memory, exactness, and verification obligations; measured selection cannot bypass verification.
