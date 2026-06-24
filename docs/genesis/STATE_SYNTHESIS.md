# State and memory synthesis

Genesis has two related state representations. `InferenceGenome.state` records serving-wide state decisions. The restricted `state_transforms` module compiles and checks a concrete `StateRegion` and `StateTransformation` for ownership, capacity, migration, and rollback safety.

## State region and transformation

A region declares stable identity and semantic contract, ownership, owners, visibility, layout, storage tier, dtype, item size and bound, replication, consistency, mutability, checkpoint support, and compatible genome hashes. Supported transformation kinds include layout, precision, offload, replication, migration, prefetch, eviction, checkpoint, recompute.

Compilation validates unambiguous ownership, read-only replication, consensus consistency, remote consistency limitations, stable identity, unchanged semantic contract, exact/approximate quality cost, bounded migration chunks, and champion/challenger coexistence memory after a safety margin. The compiled result identifies peak coexistence bytes, migration chunk count, checked preconditions, proof obligations, and whether active streams are compatible or a request boundary is required.

## Trace verifier

The independent trace checker consumes increasing typed events: allocation, acquire/read/write, migration begin/commit/abort, release, cancellation, checkpoint and rollback. It checks:

- unique allocation and declared ownership;
- no use after release and no double release;
- acquisition before read and single-writer rules;
- no partial visibility or writes during migration;
- compatible target genomes and one pending migration;
- no migration commit while leases remain;
- checkpoint-backed rollback;
- memory and item bounds;
- eventual release at trace end when quiescence is required.

Violations contain event sequence, invariant name, and detail. The checker is bounded trace validation, not a concurrent memory-safety proof for arbitrary native code.

## Current status

Tests cover paged/offload transition compilation, coexistence rejection, cancellation and release, incompatible migration, use-after-free, double release, memory overflow, and ambiguous ownership. The `InferenceGenome` compiler also recovers declared persistent state fields and conservative per-request memory.

The module does not allocate real HBM, perform DMA/RDMA, implement remote storage, or convert a live tensor layout. Those require generated conversion artifacts, hardware evidence and runtime integration. The local synthesis fixture does not yet search this state transformation space, so the existence of a valid compiled transition is not an end-to-end performance or hot-swap result.

