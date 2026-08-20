# Agent 11 Continuum correctness review

Verdict: **PASS — admit exactly one integrated v11 transaction, subject to the
root coordinator's budget and remaining post-micro reviews.**

This was a read-only audit of real-A100 attempt
`exp004-v11-micro-s41-d`. It did not invoke CUDA, Modal, the network, or a GPU,
and it did not modify source or the frozen v10 path.

## Evidence identity

The canonical worker result is 16,993,073 bytes with SHA-256
`fdd9859ccec9187facd47d6efa621449899d18b009a7b15ea746e45e1139ecb8`.
That hash agrees independently with the remote manifest, controller completion,
and settled GPU ledger. The configured attempt-D offline manifest recomputes to
`543aa451413357b8fbef7365d5af529d4c02138f42027c715c030a458a8b5b9a`;
all ten sealed source hashes still match the local reviewed tree.

## Allocation and release proof

All 1,152 source ownership transitions were independently checked. Block IDs
and allocator epochs are unique and exactly span 1 through 1,152. Recomputing
the canonical `(block_id, allocation_epoch)` commitment produces
`81e8dc5a6f319efd93cb6c43d86a59211a8f653c79247ccb7dd4b2436644c3ca`,
exactly matching the source summary and the destructive zero-queue evidence.

The owner/refcount topology is exact: 1,024 blocks have all eight logical
owners and refcount 8; 128 blocks have one owner and refcount 1, with exactly
16 private blocks per branch. Each block is 917,504 bytes, summing to the
measured 1,056,964,608-byte state.

Every transition has the required vLLM release operation, empty post-release
owners, zero post-release refcount, allocator availability, absent block hash,
inaccessible old session, no live branch reference, tombstoned epoch, and
absent native request table. KV assigned bytes fall from 1,056,964,608 to zero,
unassigned bytes rise by exactly the same amount, the reserved pool is
unchanged, and all 48,475 usable blocks are free. Allocator-visible state and
HBM therefore agree; this is not an HBM-only inference.

The source allocation queue was consumed under the sealed `EXPORT_CAPTURE`
gate with exactly 1,152 expected and observed IDs, equal set commitments, and
no missing, unexpected, or duplicate IDs. Its source-lifetime, ownership, and
capture-manifest commitments all join to the worker result.

The destination summary contains 1,152 committed allocation lifetimes, zero
source lifetime overlap, and zero physical block-ID overlap. The executed,
source-sealed worker forms that set only from typed allocator-issued subset
proofs after the stager rechecks epoch currency, mapping, exact byte coverage,
and completion. It consumes the final epoch proof once immediately before
enqueue.

## Lock, integrity, and admission order

The production binding is the synchronous vLLM 0.23 V1 `InprocClient`, with
asynchronous scheduling disabled and all 21 installed mutation wrappers intact.
The serialized export witness uses the same binding ID. The sealed worker uses
that same binding's `IMPORT_ADMISSION` critical section for transport
verification, fresh allocation, native write, exact validation, epoch checking,
single-use consumption, cache publication, and final enqueue.

Integrity evidence contains 1,152 unique page SHA-256 values, whole-payload and
domain-separated manifest commitments, exact 1,056,964,608-byte whole/page hash
coverage, and a source completion fence. The canonical ledger independently
shows one full-state coverage for each capture manifest hash, restore preflight
hash, owned snapshot, consumption hash, H2D, native write, and raw-bit
validation family.

The 10,454 pass IDs are unique. All 11,416 dependencies resolve and the global
graph is acyclic. Two restore subset completion fences cover exactly the full
logical state and end at `186585649837` and `187665929092` monotonic ns.
Continuation begins later, at `187842788001` ns. Source release also completes
before the first restore pass. No diagnostic/failure scrub pass occurred.

## Continuation and cleanup

Every branch produced at least eight continuation tokens. The first-token map
matches independent recomputation exactly for all eight branches: 8/8. The
oracle starts only after restored continuation completes and the restored
requests have drained.

Worker return code, controller status, transaction cleanup, and external Modal
postflight all pass. No compute process, app, task, container, endpoint,
reservation, child process, or profiler remains. A CUDA-clean controller import
succeeds after worker cleanup and device memory returns to the 4 MiB inventory
baseline.

Focused epoch, ownership, synchronization, restore, and worker tests also pass
60/60 offline.

## Auditability observations

The result preserves destination lifetimes as a 1,152-entry set commitment and
zero-overlap summaries, not a raw destination list. It also serializes the
export witness but not a standalone public import-witness object. These are not
correctness blockers because the content-addressed executed path fails closed
on the typed evidence before success can be emitted, but the integrated result
should preserve both raw destination lifetime/mapping evidence and public gate
metadata for easier standalone review.

`micro-validation/status.json` is still an older `BUDGET_BLOCKER` placeholder.
The root coordinator must supersede it with the reviewed attempt-D outcome
before launching the integrated transaction.

Final Continuum decision: **PASS. Correctness and the measured
21.000762818351625x amplification satisfy the explicit micro-validation gate,
so exactly one integrated v11 transaction is admissible.**
