# ADR 0029: Invalidate lineage evidence explicitly by dependency scope

- Status: accepted
- Date: 2026-08-02

## Context

Compiler, runtime, driver, hardware, model-contract, or workload changes can invalidate evidence without changing a transformation's source text. Silently retaining confidence would cause unsafe transfer; deleting history would hide negative and stale results.

## Decision

Attach typed dependency/version records to evidence. Record invalidation events with kind, name, bounded numeric version selector, reason, and time. Atomically mark matching fresh evidence stale and append the event reference, subject to a hard affected-record limit and query deadline.

Give stale, failed, inconclusive, expired, and future-dated evidence zero effective confidence and exclude it from transfer retrieval. Preserve the evidence, event, and graph edge for audit. Revalidation creates new evidence instead of editing the old result back to fresh.

## Consequences

Stale evidence is not silently reused and invalidation is reversible only through new evidence. Operators must issue events and ensure dependencies are recorded; the store does not watch package registries or drivers. The implemented range language is deliberately smaller than ecosystem-specific semver standards.

