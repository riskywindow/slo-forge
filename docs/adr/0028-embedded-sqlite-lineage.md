# ADR 0028: Store optimization lineage in transactional embedded SQLite

- Status: accepted
- Date: 2026-08-02

## Context

Genesis must retain accepted/rejected candidates, transformations, counterexamples, constraints, evidence, invalidations, and transfer outcomes across runs without requiring an external service. Partial writes or overwritten negative results would poison future search.

## Decision

Use SQLite as the default lineage store with foreign keys, WAL, `synchronous=FULL`, busy/query timeouts, bounded scans, and schema `user_version`. Store strict immutable JSON records under primary keys plus indexed relational columns for dependency, evidence-target, and constraint queries.

Insert a candidate and its new transformations/evidence atomically. Treat duplicate identity as conflict. Keep invalidation as an explicit atomic fresh-to-stale transition rather than deletion. Export bounded deterministic snapshots to portable JSON and GraphML.

## Consequences

Local use is zero-configuration, crash-consistent, inspectable, and portable. SQLite is not a horizontally replicated multi-writer graph service; large-scale concurrent operation and migration beyond schema v1 require additional design. Focused tests cover reopen, rollback, queries and exports, not production-scale load.

