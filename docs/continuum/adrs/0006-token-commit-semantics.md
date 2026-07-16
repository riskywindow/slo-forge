# ADR 0006: Sequence and epoch based output commitment

- Status: Accepted
- Date: 2026-08-02

## Context

Migration can produce duplicate, delayed, or stale-owner token events. Network delivery and gateway acceptance have different guarantees.

## Decision

Bind every event to session, owner epoch, token index, token ID, and state version. The gateway rejects stale epochs/gaps and deduplicates identical indices; client exactly-once delivery requires durable acknowledgments.

## Consequences

The exercised guarantee is exactly-once gateway acceptance. Client protocols without acknowledgments remain at-least-once with sequence-based deduplication.
