# Dirty tracking and delta protocol

The reference adapters instrument logical mutations explicitly. Each mutable segment has a monotonic state version, dirty epoch, content checksum, owner runtime/epoch, and transaction association. Appended KV tokens and token history use append-log semantics; recurrent, sampler, guided-decoding, page-table, workflow, delivery, and ownership changes produce versioned events.

## Protocol

1. A consistent initial snapshot establishes the copied version.
2. `start_dirty_tracking` opens a bounded log at a declared epoch.
3. `obtain_dirty_delta` returns monotonic segment changes while generation continues.
4. Duplicate chunks are recognized by hash; stale epochs and out-of-order versions are rejected.
5. At cutover, `quiesce` stops mutation at a legal token boundary and `export_final_delta` closes the gap.
6. Destination application is idempotent and followed by full import and continuation validation.

Dirty logs are bounded. Overflow raises `DirtyLogOverflowError`; it is not silently converted into success. The planner may select hybrid stop-and-copy or abort/restart with a new snapshot when measured dirty rate prevents convergence. Hash comparison exists only as a declared reduced-efficiency fallback for deterministic fixtures or adapters lacking instrumentation.

The design does not require a device-wide synchronization after every token. An adapter records changes at its safe runtime boundary and reports the associated overhead in raw evaluation artifacts.
