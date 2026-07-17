# Learning transaction

`LearningTransactionStore` persists the lifecycle from a trigger and champion epoch through captured,
branched, rolled-out, rewarded, credited, batched, trained, evaluated, promoted, rejected, or failed
states. Every transition names evidence artifacts and appends a sequenced event in one SQLite
transaction.

The transition graph rejects skipped phases, terminal-state resumption, unknown evidence, artifact
identity reuse, and changed raw-evidence hashes. Repeating the same transition is idempotent. A failed
or rejected transaction cannot silently become promotable.

This is a local durable journal, not a cluster-wide consensus protocol. See
[promotion and rollback](PROMOTION_AND_ROLLBACK.md).
