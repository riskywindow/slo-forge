# ADR 0009: Content-addressed immutable state chunks

- Status: Accepted
- Date: 2026-08-02

## Context

Incremental checkpoints and forks contain substantial immutable overlap.

## Decision

Store bounded chunks by strong plaintext digest and publish immutable manifests transactionally with reference counts, TTL, integrity checks, and COW ancestry.

## Consequences

Forks and snapshots deduplicate authorized state and corruption is explicit. Hashes reveal equality within the authorization domain, and deletion remains best effort.
