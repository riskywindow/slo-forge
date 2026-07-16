# Continuum trust model

Continuum trusts semantic contracts only after their version, producer, dependencies, ownership epoch, and integrity chain validate. It does not trust tensor shape equality, runtime-local pointers, generated conversion code, transport completion, or a capsule merely because it parses.

## Trusted computing base

The local trusted base is the canonical Rust/Python ABI validator, compatibility policy, trusted CPU converter, capsule and chunk integrity verifier, durable coordinator compare-and-swap path, and gateway acceptance ledger. Runtime adapters are version-scoped trust boundaries: capture bytes and import acknowledgements remain untrusted until structural, logical, conversion, and bounded continuation verification pass. The embedded SQLite coordinator is a single-node authority; a distributed deployment must replace it with an existing linearizable CAS service rather than infer consensus from transport acknowledgements.

Generated kernels are proposals. They may execute only after their declared operation, shape, dtype, temporary-memory, and exactness contracts validate and an independent comparison against the trusted converter passes for the applicable domain. No generated code receives content-store keys.

## State categories

- Logical portable state is the minimal semantic continuation contract: histories, KV, recurrent, sampler, guided/speculative/workflow state, delivery cursor, dependencies, and exactness.
- Physical state is the versioned layout, paging, sharding, placement, quantization, chunk, and dirty-version representation.
- Reconstructible runtime state includes scheduler queues, CUDA graphs, communicators, allocator internals, threads, file descriptors, raw pointers, and process handles. It is never treated as portable.
- Externally visible state is the owner epoch and monotonically committed token sequence. Only the fenced owner may mutate accepted state or submit output to the gateway.

## Evidence and authority

SHA-256 content identity detects alteration but does not authenticate an issuer. Tenant authorization, expected capsule identity, coordinator lease, fencing token, replay policy, and—when enabled—AEAD key policy provide authority. Transport moves bytes and cannot decide compatibility or ownership. A destination is a candidate until required state validates, commit intent is durable, the lease CAS succeeds, and the gateway switches to the new epoch.

Pre-commit failure preserves the fenced or resumable source and permits rollback to the recorded boundary. After ownership commit or accepted destination output, returning to an old source is not described as rollback; it requires post-commit recovery, recomputation, or a new migration.

## Tenant and deployment boundary

Chunk deduplication is tenant-scoped by default. Bare TCP provides bounded framing, per-frame authentication, replay protection, and integrity but not confidentiality. Optional AES-256-GCM payload encryption binds ciphertext to transfer identity, sequence, attempt, plaintext digest, and size and fails closed without its separately supplied key; production peer identity still requires an approved channel such as mTLS. Storage deletion is best-effort logical deletion and cannot guarantee erasure from media, snapshots, or backups. See [SECURITY.md](SECURITY.md) and [THREAT_MODEL.md](THREAT_MODEL.md) for controls and residual risks.
