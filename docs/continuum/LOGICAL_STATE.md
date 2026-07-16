# Logical state ABI

`LogicalStateSchema` is the runtime-independent contract required to continue a session. It is immutable, strict (`extra="forbid"`), versioned as Continuum 1.0.0, canonically serialized, and represented in Python and Rust.

## Components

The schema contains typed records for:

- `ExecutionIdentity`: session/request/workflow/tenant identity; model, tokenizer, and adapter digests; creation and owner epochs.
- `TokenHistoryState`: prompt and committed output tokens, speculative suffix, positions, mask semantics, tokenizer fingerprint, and normalization contract.
- `AttentionState`: layer identity, logical K/V shapes, token ranges, head geometry, positional encoding, attention-window behavior, and dtype semantics.
- `RecurrentState`: per-layer logical shape, update equation contract, sequence position, and initialization contract. State-space and convolutional state can use the same typed component mechanism with distinct `StateKind` values.
- `SpeculativeState`: draft/verifier identities, accepted prefix, pending drafts, RNG state, cursor, and rollback boundary.
- `SamplerState`: algorithm, seed, RNG algorithm/counter, temperature, top-k/top-p, penalties, and deterministic requirement.
- `GuidedDecodingState`: automaton identity/state, tokenizer contract, accepted prefix, and pending constraint state.
- `WorkflowState`: node and branch identity, completed tool results, pending calls, side-effect class, deadline, and continuation contract.
- `ClientDeliveryState`: generated, gateway-committed, and optionally client-acknowledged watermarks; owner epoch; terminal/error state.
- `StateDependencyGraph`: component nodes and typed dependency edges used for compatibility and recomputation reasoning.

Each component descriptor includes a stable semantic ID, schema version, symbolic shape, dtype and update semantics, lifetime, ownership, exactness requirement, conversion permissions, recomputation permission, compatibility fingerprint, integrity digest, and provenance. Unknown state is represented explicitly as `StateKind.UNKNOWN`; an adapter cannot smuggle an untyped core field into the ABI. Namespaced JSON extensions are the sole extension point.

## What is not logical state

Scheduler queues, raw pointers, file descriptors, CUDA graph objects, communicator handles, runtime threads, temporary workspaces, and allocator internals are reconstructible runtime state. They are described as adapter capabilities or rebuilt after import; they are never serialized as portable session state.

## Canonical form and evolution

Canonical JSON uses sorted keys, no insignificant whitespace, UTF-8, strict finite numbers, and deterministic tuple/list rendering. SHA-256 is computed over those bytes. `migrate_document` permits explicit schema migration; unknown major versions and unknown core fields fail closed. Golden fixtures under `schemas/continuum/` are shared by Rust and Python conformance tests.

The schema describes continuation semantics, not reuse permission. A model hash, tokenizer fingerprint, dependency graph, and destination capabilities still have to pass [compatibility analysis](COMPATIBILITY.md).
