# Promotion and rollback

Promotion requires independently named lineage, reward-integrity, quality, safety, serving, and
compatibility gates, followed by isolated shadow and bounded canary evidence. A failed pre-gate,
shadow, or canary rejects the candidate without changing the champion. A passed canary permits one
atomic pointer update.

The prior champion remains the rollback parent. Rollback uses a compare-and-swap so it cannot replace
an unexpected current champion. Promotion records and events are persisted and evidence hashes are
revalidated on read. Active incompatible sessions remain pinned.

Local deterministic gate evidence demonstrates control flow, not production safety or statistical
power. External deployment needs a linearizable registry and monitored rollback trigger. See
[promotion capsule](PROMOTION_CAPSULE.md) and ADR 0047.
