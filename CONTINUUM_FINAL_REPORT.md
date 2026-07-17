# SLOForge Continuum final implementation report

Date: 2026-08-02 (America/Los_Angeles)

## Release identity

- Baseline commit: `7e51ea7f7338755d23f889820558a4e046d6c42e`
- Baseline tag: `sloforge-continuum-baseline-7e51ea7`
- Final implementation commit: `8bba247b44d26e1be4475ca18058fcbc652adcf0`
- Final implementation tag: `sloforge-continuum-source-8bba247`
- Retained evidence commit: `d364419a77797ab5c22f01af70624cea2ceeb22c`
- Evaluation identity: `385704604314a07571752635f41961674ccbe191ba843cbe4e567073e0431037`
- Evaluation source commit: `8bba247b44d26e1be4475ca18058fcbc652adcf0`
- Clean-room source tree: `d2677e4de91db57821f9943f48ca03a3cbb439c9`

The implementation extends the existing repository. It retains the existing Python orchestration and Rust wire/protocol split, existing commands, Fabric physical planning, Autopsy evidence, Genesis generation, ForgeCI, WarmPath, and the versioned JSON subprocess boundary.

## Architecture

Continuum separates six concerns that runtime-specific KV connectors commonly conflate:

1. A runtime-independent logical continuation contract.
2. A versioned physical layout, sharding, placement, page, quantization, and dirty-version description.
3. Reconstructible process/runtime ephemera that is explicitly excluded from capsules.
4. Typed transformation and transfer programs with bounded buffering and independent verification.
5. Durable session ownership, fencing, transaction, rollback-window, and gateway token-commit state.
6. Client-visible owner epoch, token index, gateway commit, optional client acknowledgment, and terminal state.

Python owns adapters, capture, compatibility, conversion planning/execution, migration orchestration, evaluation, and reports. Rust owns the canonical shared ABI, deterministic ownership/token protocol, and explicit-state model checker. Transport moves authenticated chunks but cannot make semantic compatibility or ownership decisions.

The local coordinator uses SQLite compare-and-swap transactions and a durable event journal. The distributed coordinator boundary intentionally delegates linearizability to an external CAS system; Continuum does not implement a new consensus protocol.

## Portable state ABI

The v1 ABI implements strict Pydantic, Serde, and JSON Schema forms for:

- `LogicalStateSchema`: execution identity; token history; attention KV; recurrent state; sampler RNG/counter; guided decoding; workflow; client delivery; and a typed state-dependency graph.
- `PhysicalStateLayout`: runtime identity; segments; pages; strides; token/head/layer ordering; TP/PP/EP descriptors; placement; access patterns; checksums; compression/encryption; and dirty epochs.
- `ExecutionStateCapsule`: logical and physical state, external content references, exactness/quality contracts, compatibility restrictions, ownership/fencing data, rollback boundary, Merkle-style integrity, provenance, and scoped evidence.
- `CompatibilityReport`, `StateTransformationIR`, `MigrationPlan`, `StateTransaction`, and `MigrationVerificationEvidence`.

Canonical JSON hashing agrees across Rust and Python for the complete golden capsule and numeric/Unicode edge profiles. Unknown core fields fail closed; namespaced extensions are versioned. Validation rejects altered or missing chunks, stale pages, model/tokenizer mismatches, altered journals/evidence, and owner-epoch inconsistencies. Raw pointers, CUDA graph objects, scheduler queues, communicators, file descriptors, allocator internals, and threads are never portable state.

Exactness classes are `EXACT_BITWISE`, `EXACT_SEMANTIC`, `NUMERICALLY_EQUIVALENT`, `QUALITY_BOUNDED`, `RECOMPUTATION_ASSISTED`, and `INCOMPATIBLE`. Layout-only TP/page/order changes are accepted as exact semantic transformations. State-producing weight, tokenizer, RoPE, attention, or recurrent-update changes fail closed unless a dependency-backed recomputation or quality contract exists.

## Runtime adapter status

| Adapter | Status | Exercised scope |
|---|---|---|
| Deterministic reference A | Implemented and exercised | Paged token-major, separate K/V, TP=4, page size 3, four simulated devices; capture, dirty tracking, fencing, streaming |
| Deterministic reference B | Implemented and exercised | Paged head-major, packed K/V, TP=2, page size 5, two simulated devices; import, validation, activation, fork, streaming |
| Genesis | Partially implemented; migration unexercised | Versioned discovery, generated-runtime loading, and CPU smoke only |
| PyTorch | Partially implemented; package unavailable | Version-gated tensor/RNG binding and explicit-device verified conversion path; no active-session migration or CUDA exercise |
| vLLM | Partially implemented; package unavailable | Fail-closed capability/version probe pinned to inspected v0.23.0 public KV-transfer interfaces; no active-session migration |
| SGLang | Partially implemented; package unavailable | Fail-closed capability/version probe pinned to inspected v0.5.12 disaggregation interfaces; no active-session migration |

No claim is made that vLLM, SGLang, PyTorch, or Genesis completed a live active-state migration on this host. NIXL, Dynamo, BIFROST, RDMA, and multi-node paths are not implemented as exercised adapters.

## Conversion, storage, transport, and operations

The conversion compiler provides canonical materialization, direct source-to-destination conversion, chunked streaming conversion, transformation DAGs, measured backend selection, memory plans, verification obligations, and source/destination CPU placement. The direct path is used by the live migration import, including changed destination-layout deltas, and is compared byte-for-byte with the trusted canonical converter. An explicit CPU floating-dtype narrowing backend enforces and measures a maximum-absolute-error quality budget. The optional PyTorch backend never silently changes devices and verifies its full result against the canonical CPU path.

The content store provides in-memory, filesystem, and SQLite-backed tenant-scoped chunks, transactional publication, partial/concurrent reads, refcounts, TTL/GC, incremental snapshots, compression bounds, copy-on-write, integrity recovery, and AES-GCM wrapping. Cross-tenant deduplication is off by default. The transport layer includes in-process, file, deterministic simulated, and TCP implementations with bounded retries/deadlines, authenticated frames, replay protection, checksums, compression bounds, and optional AES-256-GCM payload confidentiality.

Checkpoint, incremental checkpoint, pause, resume, clone, fork, recomputation-assisted restore, and access-pattern-gated lazy migration are implemented. The CLI exposes full checkpoint, pause, transactional resume, clone, capture, capsule validation, conversion compile, migration plan/status/verify, fault, fork, compatibility, benchmark, and report workflows with explicit seeds.

## Transaction protocol and model checking

Ownership uses monotonic epochs, fencing tokens, durable leases, compare-and-swap updates, bounded deadlines, and persisted idempotent transitions. The gateway accepts events only from the current owner epoch, deduplicates identical indices, rejects changed duplicates and gaps, retains a client resume cursor, and commits terminal output once. The exercised guarantee is exactly-once acceptance by the SLOForge gateway; end-to-end exactly-once delivery requires an acknowledgment-capable client.

Pre-commit failure restores the still-valid source only after destination abort and cleanup evidence. Post-commit failure never revives the fenced source; it is classified as destination recovery, a new migration, or operator-required. Coordinator restart recovery replays exact evidence, rejects altered replay, observes the exclusive timeout boundary, and cannot backdate terminal transitions.

The seed-41 explicit-state checker explored 1,659 states and 3,216 transitions to depth 22. All 15 invariants passed within bounds of three messages, one token, one fault per execution, depth 32, 100,000 states, and six timeout ticks. The result is complete within those finite bounds and explicitly is not a universal proof. Covered actions include loss, duplication, reordering, crashes/restarts, partitions, stale output, cancellation, delayed acknowledgment, client disconnect, timeout, abort, and operator escalation.

## Flagship CPU migration

The seed-317 flagship streamed a HybridDecoder state containing attention KV, recurrent state, sampler state, guided-decoding state, token history, and client-delivery watermarks. It migrated between different adapter implementations, physical layouts, page sizes, and TP degrees:

- Source: token-major paged separate K/V, TP=4, page size 3.
- Destination: head-major paged packed K/V, TP=2, page size 5.
- Direct conversion: 197 source segments to 37 destination segments; 9,216 attention bytes compared with the canonical result; exact match; bounded 512-byte reported converter temporary buffer.

The first destination crashed during validation before ownership commit. The transaction reached `ROLLED_BACK`, retained source epoch 1 and the gateway watermark, and recovered through the durable coordinator. The second pre-copy transferred iterative deltas, quiesced at a token boundary, sent a nonempty final delta of 21 chunks (1,262 unique bytes; 1,294 simulated wire bytes), validated destination continuation, committed owner epoch 2, switched the gateway, and completed. The gateway accepted indices 0 through 36 with no duplicate or gap; stale epoch-1 output was rejected.

The migrated capsule was forked into two distinct owners/layouts through content-addressed checkpoint sharing and copy-on-write divergence. A same-shaped revision with changed state-producing weights was classified `INCOMPATIBLE` for direct reuse. A dependency-authorized token-history recomputation cloned the checkpoint, teacher-forced the changed model state, completed a separate ownership transaction, and emitted five accepted continuation tokens.

## Evaluation results

The retained CPU campaign uses seeds 101, 202, 303, 404, and 505. It contains 25 authenticated per-seed raw files plus the hash-indexed summary and reports. Report loading independently validates every raw artifact and recomputes all ten confidence intervals, hypothesis results, negative-result counts, and the evaluation identity.

| Metric | Mean | 95% confidence interval | Scope |
|---|---:|---:|---|
| Canonical CPU conversion median | 77.842 us | 72.949–82.735 us | Observed host timing |
| Direct CPU conversion median | 160.783 us | 151.441–170.125 us | Observed host timing |
| Pre-copy observed interruption | 9.330 ms | 4.391–14.270 ms | Observed host timing |
| Stop-and-copy observed interruption | 26.509 ms | 26.242–26.777 ms | In-memory observed baseline |
| Flagship wall time | 112.212 ms | 110.703–113.722 ms | Observed host timing |
| Checkpoint bytes deduplicated | 10,198.4 bytes | 10,187.699–10,209.101 bytes | Artifact-derived |
| Planner regret | 0.0 objective units | 0.0–0.0 | Finite synthetic candidate oracle |
| Simulated transport bytes on wire | 20,548.6 bytes | 20,286.463–20,810.737 bytes | Synthetic protocol model |

The direct CPU converter won 0/5 median comparisons, so H2 is a retained negative result and no direct-conversion speedup is claimed. Pre-copy had lower observed interruption than stop-and-copy in 5/5 seeds. Every seed accepted 25 contiguous gateway tokens with zero accepted duplicates and gaps. H1 and H3–H9 pass in the scoped reference campaign; H2 is negative. Planner zero regret applies only to its finite synthetic candidate set, not an unconstrained production oracle.

## Hardware-backed versus synthetic validation

The recorded host is Apple Silicon macOS without NVIDIA GPU, CUDA, Triton, RDMA, InfiniBand, PyTorch, vLLM, SGLang, NIXL, or a local BIFROST checkout. No GPU budget or live/multi-node/external deployment opt-in was enabled. No paid resource was created.

Implemented and exercised results are CPU reference-runtime, filesystem/SQLite/TCP-local, and deterministic simulated-fabric results. GPU conversion, real GPU capture/import, real cross-runtime active-state migration, multi-GPU resharding, RDMA/NIXL transport, and cloud deployment remain unexercised. `continuum-benchmark-gpu` and Docker smoke emit explicit `unexercised` status artifacts rather than fabricated results; Docker was unavailable because the daemon was not running.

## Verification and non-regression commands

The final source was exercised with:

```text
make check
make fabric-check
make genesis-check
make continuum-check
make demo
make fabric-demo
make autopsy-demo
make genesis-zero-day-demo
make forgeci-demo
make warmpath-demo
make continuum-demo
make continuum-migration-demo
make continuum-fault-demo
make continuum-fork-demo
make continuum-compatibility-demo
make continuum-benchmark-cpu
make continuum-benchmark-gpu
make continuum-docker-smoke
make continuum-clean-room-test
uv run --locked sloforge continuum migration modelcheck --seed 41 --output artifacts/continuum/modelcheck/result.json
```

The final full repository gate passed Python formatting/linting/strict typing, 927 Python tests with six declared optional skips, Rust formatting, workspace Clippy with warnings denied, all workspace Rust tests/doc-tests, and UI typecheck/lint/37 tests/build. The named Fabric gate passed 293 Python and 31 Rust tests. The named Genesis gate passed 363 Python tests with one optional PyTorch skip and 30 Rust tests. The committed-archive clean room passed package installation/build, 196 Continuum Python tests with one honest PyTorch/CUDA skip, Rust formatting/Clippy/20 tests, migration verification, fault rollback, fork, compatibility, and wheel/sdist construction for revision `8bba247b44d26e1be4475ca18058fcbc652adcf0`.

## Security status

Capsules and chunks are tenant-authorized and integrity-bound. External chunk references are resolved beneath canonical store roots and checked for tenant, digest, size, segment hash, and manifest membership. Content equality is never shared across tenants by default. Authenticated TCP frames bind transfer/chunk/attempt/digest/size metadata; replay, wrong keys, changed ciphertext, corrupt chunks, unsafe paths, and decompression expansion fail closed. Old owner epochs cannot mutate committed state or cross the gateway acceptance boundary. State payloads are excluded from logs, subprocess environments are bounded, transaction identifiers are non-reusable, and generated/optional conversion backends require independent canonical verification.

At-rest and optional TCP payload encryption use local AES-GCM key material supplied by the authorized caller. This is not a PKI identity system. Secure deletion remains best effort and depends on the underlying filesystem/storage stack.

## Known limitations and risks

1. The flagship executes one concrete destination-validation crash. The typed 31-fault catalog, focused protocol/storage/transport tests, and single-fault bounded model checker broaden coverage, but they are not a full real-system fault campaign for every catalog entry or correlated faults.
2. External vLLM, SGLang, PyTorch, and Genesis active-session adapters are partial and unexercised; public KV/disaggregation APIs do not expose the complete Continuum logical state contract.
3. No custom CUDA/Triton kernel or hardware-backed conversion/migration result exists. The optional PyTorch CUDA path is implemented and fail-closed but was unavailable.
4. The durable coordinator is local SQLite. Distributed deployment needs an existing linearizable CAS backend such as etcd or Kubernetes Lease.
5. Rust/Python conformance has complete-capsule and numeric/Unicode goldens, but standalone goldens for every top-level IR remain capsule-centric rather than a separate corpus per document.
6. Visualization is a static artifact-backed report, not the full interactive view set described in the aspirational design.
7. No upstream runtime/transport patch is included.

Additional scoped constraints: the quality-bounded backend covers CPU floating-dtype narrowing, not FP8 packing; constrained lazy migration is admitted only when required-before-use analysis proves legality; arbitrary dense full-attention post-copy and branch merging are rejected; and no claim is made about production client delivery beyond gateway acceptance without client acknowledgments.

## Artifact inventory

- Baseline: `artifacts/continuum/baseline/record.json`
- Flagship and static timeline: `artifacts/continuum/demo/flagship.json`
- Migration verification: `artifacts/continuum/migration-demo/verification.json`
- Fault and rollback: `artifacts/continuum/fault-demo/fault-result.json`
- Fork/COW: `artifacts/continuum/fork-demo/fork.json`
- Compatibility rejection/recomputation: `artifacts/continuum/compatibility-demo/compatibility.json`
- Evaluation summary and 25 raw files: `artifacts/continuum/evaluation/`
- Bounded model check: `artifacts/continuum/modelcheck/result.json`
- Clean-room result and hash-bound log: `artifacts/continuum/clean-room/`
- Hardware/Docker status: `artifacts/continuum/hardware/gpu-status.json`, `artifacts/continuum/docker/status.json`
- Static reports: `reports/continuum-evaluation.md`, `reports/continuum-evaluation.html`, `reports/continuum-compatibility.md`, `reports/continuum-fault-tolerance.md`, `reports/continuum-runtime-adapters.md`
- Architecture, protocols, trust model, ADRs, reproducibility, and review: `docs/continuum/`
- Paper-style report: `paper/continuum/REPORT.md`

The retained Continuum artifact tree contains 50 files and is approximately 11 MiB. Demo/evaluation work databases are removed after terminal operations; no Continuum lease, fault configuration, child process, or cloud resource remains active.
