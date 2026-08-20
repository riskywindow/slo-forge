# Agent 05C / 08C — source queue integrity and provenance re-review

Result: **PASS**

The strengthened source-allocation queue lifecycle is safe for the exact v11 micro transaction. It runs only after exact full-page topology, allocator lifetime, ownership, capture and manifest evidence exists; authenticates the production `EXPORT_CAPTURE` gate before and after the one destructive read; and releases the source immediately afterward without an engine step.

The queue must contain exactly 1,152 unique strict integer block IDs equal to the captured source set. `needs_kv_cache_zeroing` must be exactly false. Missing, duplicate, unrelated, malformed, API-error, gate or zeroing-mode conditions raise `V11RuntimeTeardownRequired`. The worker persists the typed affected IDs and complete teardown evidence to both failure artifacts and cannot publish success.

Successful evidence commits the expected and observed block sets, source `(block ID, epoch)` lifetimes, pre-release ownership snapshot, canonical state manifest, and production gate identity. Post-free proof still independently requires all source allocations released and inaccessible.

Fresh destination handling is unchanged: import requires an empty starting queue, drains and validates every new subset, uses allocator-issued epoch proofs, requires a final empty queue, and rejects source/destination lifetime overlap.

There is no movement-accounting contamination. Queue evidence is source-lifecycle metadata only. It emits no `StatePassRecord`; `all_passes` remains capture plus restore records, and every movement total is computed exclusively from those records.

Offline evidence: 61/61 independent focused tests PASS in 1.68 seconds and Ruff PASS. No GPU, Modal, network, or source actions were performed.
