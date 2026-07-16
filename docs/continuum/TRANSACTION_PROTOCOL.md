# Transaction and cutover protocol

Continuum separates byte availability from ownership. `DurableCoordinator` stores session leases, transactions, and journal transitions in SQLite with WAL and synchronous durability. The distributed interface requires compare-and-swap semantics; it does not implement a custom consensus protocol.

## Lease and fencing

A `SessionLease` binds session ID, owner runtime, monotonically increasing owner epoch, fencing token, expiration, coordinator version, committed state version, and committed token index. Mutations and output must present the current epoch. A stale source is fenced before ownership commit.

## State machine

The persisted phases are `PROPOSED`, `COMPATIBILITY_VALIDATED`, `DESTINATION_PREPARING`, `PRECOPYING`, `DELTA_SYNCING`, `CUTOVER_REQUESTED`, `SOURCE_QUIESCING`, `SOURCE_FROZEN`, `FINAL_DELTA_TRANSFERRING`, `DESTINATION_IMPORTING`, `DESTINATION_VALIDATING`, `COMMIT_INTENT_RECORDED`, `OWNERSHIP_COMMITTED`, `GATEWAY_SWITCHING`, `DESTINATION_ACTIVE`, `SOURCE_DRAINING`, and `COMPLETED`, plus typed rejection, rollback, loss, unavailability, and operator-required states.

Every transition is validated, journaled, and idempotent. Replayed identical transitions do not advance state; invalid or out-of-order transitions fail closed. Ownership commit uses CAS against source epoch and fencing token and atomically records the destination epoch and token watermark.

## Rollback windows

Before ownership commit, destination failure can abort and release the source fence, leaving the old epoch authoritative. The flagship injects a destination validation crash in this window, persists `ROLLED_BACK`, restarts the coordinator, and confirms the source and gateway watermark remain valid.

After owner-epoch commit, returning to the old source is not called rollback. If the destination has emitted or committed additional output, recovery requires destination restart, a new migration, or recomputation from the newest committed checkpoint. The transaction therefore records commit and rollback watermarks separately.

See [Token commit protocol](TOKEN_COMMIT_PROTOCOL.md) and [Model checking](MODEL_CHECKING.md).
