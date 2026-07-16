# SLOForge Continuum: A portable execution-state ABI and transactional migration substrate

## Abstract

SLOForge Continuum represents an active AI execution as a versioned, content-addressed capsule that separates logical continuation state, physical runtime layout, reconstructible runtime ephemera, transaction ownership, and client-visible delivery state. A compatibility engine rejects semantically invalid reuse; a state-conversion compiler lowers safe relayouts; a live migration protocol transfers initial state and versioned deltas; and a durable coordinator fences the old writer before atomically changing the output owner. A deterministic CPU reference system exercises attention KV plus recurrent, sampler, guided-decoding, history, and delivery state across different runtime adapters, TP layouts, page sizes, and K/V packing. The flagship injects a pre-commit destination failure, recovers the valid source, retries successfully without a gateway-accepted duplicate or gap, forks state using content addressing, and rejects a same-shape model revision with changed state-producing weights. Results are scoped: optional GPU and production runtime paths are version-gated but not represented as exercised without matching hardware/packages.

## Motivation

Runtime-native KV transfer solves a valuable but narrower problem than execution portability. A streaming session also depends on recurrent/state-space values, RNG progression, sampling policy, repetition history, speculative cursors, grammar automata, workflows and tool continuation, committed output watermarks, and the right to emit the next token. Copying bytes without these semantics can resume from incompatible weights, duplicate visible output, lose an accepted token, or create two writers.

Continuum makes meaning and ownership first-class. It does not migrate a process image or replace runtime, transport, fabric, or storage systems. It supplies the portable state contract, conversion and compatibility logic, cutover protocol, and evidence needed to use those systems safely.

## State model

The model has four categories. Logical portable state captures information required by the continuation contract: token history, KV, recurrent/speculative/sampler/guided/workflow state, deadlines and delivery cursors. Physical state records pages, blocks, strides, padding, shards, placement, dtype/quantization, memory tiers, chunks, and dirty versions. Reconstructible state covers queues, handles, CUDA graphs, communicators, threads, pointers, file descriptors, and temporary workspaces. Externally visible state records emitted indices, owner epoch, gateway/client acknowledgment watermarks, terminal status, and error commitment.

Every logical component has a stable semantic ID, schema/version, symbolic shape, dtype/update semantics, lifetime, owner, exactness and conversion permissions, recomputation policy, compatibility/integrity digests, and provenance. A dependency DAG states which model computations produced stored values. This makes an output-head-only change distinguishable from a changed attention or recurrent update even when shapes match.

## Portable ABI

Continuum v1 defines strict Python and Rust wire types for `LogicalStateSchema`, `PhysicalStateLayout`, `ExecutionStateCapsule`, compatibility reports, transformation DAGs, migration plans, and transactions. Core models reject unknown fields; namespaced extensions are the only evolution mechanism. Canonical JSON has deterministic key order and encoding, and SHA-256 provides cross-language identity. Golden fixtures and explicit migrations protect compatibility within the IR major version.

An execution capsule binds logical and physical documents to model/tokenizer/adapter hashes, source runtime/plan, owner epoch, segment/page/chunk manifests, encryption/compression metadata, exactness and quality contracts, compatibility constraints, lease/fencing/transaction state, journal hash, and scoped verification evidence. Segment payloads remain external CAS chunks. Capsule and chunk validation are independent so a valid manifest cannot mask missing or corrupt data.

## Compatibility system

The engine returns one of bitwise exact, semantic exact, numerically equivalent, quality bounded, recomputation assisted, or incompatible. Its report also records rejected stronger classes, reasons, required transformations, recomputation, quality implications, unsupported state, and verification obligations.

Rules are conservative. Exact weights and changed TP/page/order may allow exact relayout. Quantization is not bitwise exact. Tokenizer, positional-encoding, attention-mask, recurrent-equation, or state-producing weight changes reject direct reuse unless a verified transformation exists. Output-head-only changes can preserve upstream state only when dependency evidence proves the head does not produce stored state. Token-history recomputation is legal only with sufficient inputs and acceptable side-effect policy.

## Conversion compiler

The typed transformation IR contains shape/layout operations, resharding, page transforms, dtype/quantization, compression/encryption, checksum, send/receive, metadata reconstruction, recomputation, destination writes, and validation. Operations carry pre/postconditions, exactness, device, ownership behavior, cost, memory, streamability, and verification.

The trusted CPU backend decodes a source into canonical logical K/V and encodes the destination. The direct backend processes bounded token chunks from source shards into destination shards, enabling pipelined transfer without full canonical allocation. In the reference live path it converts token-major separate K/V with TP=4/page=3 to head-major packed K/V with TP=2/page=5, then byte-compares destination attention segments against the independent canonical path. Converter selection uses repeated observed host measurements only after correctness verification.

## Migration planner

The planner considers stop-and-copy, pre-copy, hybrid, recomputation-assisted, and constrained lazy migration. It consumes state/access metadata, dirty and generation rates, source load, measured conversion and transfer curves, memory, destination readiness, SLO, quality, failure risk, rollback, and budget. Candidate estimates expose interruption, total time, bytes, overhead, temporary memory, cost, uncertainty, validation, rollback, and rejected reasons.

Fabric integration imports successful transfer samples with raw provenance and synthetic/measured labels. WarmPath contributes destination-ready p95. A versioned extension links those decisions to the existing PhysicalExecutionPlan without changing its major schema. Dense full-attention KV is required before resume unless an adapter proves a safe access/stall contract; dirty-rate non-convergence is explicit.

## Transaction protocol

The local durable coordinator uses SQLite WAL and full synchronization for leases, transactions, and journals. A lease identifies owner runtime, monotonically increasing epoch, fencing token, expiration, state version, and committed token index. The protocol persists proposal, compatibility, destination preparation, pre-copy, delta synchronization, source quiesce/freeze, final delta, import/validation, commit intent, CAS ownership commit, gateway switch, activation, drain, and completion plus failure states.

The source is fenced before ownership commit. CAS changes the lease to the destination epoch and final source watermark. The gateway then switches expected epoch/index; stale epochs are rejected, identical duplicates are idempotent, and gaps fail. The exercised guarantee is exactly-once gateway acceptance. Exactly-once client delivery needs durable client acknowledgments.

Pre-commit failure can release the fence and preserve the source. A post-commit failure is not mislabeled rollback: if newer destination state/output exists, recovery needs destination restart, another migration, recomputation, or operator action.

## Runtime adapters

The SDK exposes typed runtime identity/capabilities, consistent capture, explicit dirty tracking, token-boundary quiesce, bounded segments, inactive import, validation, dry-run continuation, activation, pause/resume/cancel, and fencing. Typed errors distinguish unsupported capability, unavailable version, resource limit, inconsistency, corruption, stale delta, and stale epoch.

Two deterministic CPU adapters fully exercise the SDK and materially different physical layouts. The stateful hybrid reference model produces KV and recurrent state alongside sampling, guided decoding, and streaming delivery. Genesis has a generated-runtime descriptor/binding and CPU smoke path. PyTorch has version-gated CPU tensor and RNG helpers. vLLM and SGLang have version/public-API gates and fail closed where public interfaces cannot export a complete portable execution state. Discovery is not counted as migration validation.

## Implementation

Python modules implement capture, adapters, compatibility, transformation compilation, planning, migration orchestration, storage/transport, advanced operations, evaluation, and static reports. Rust crates implement canonical ABI types, transaction semantics, and explicit-state model checking. The primary language boundary is bounded, versioned JSON over subprocess I/O. Content storage provides memory and filesystem/SQLite implementations, tenant-scoped dedup, partial reads, COW/incremental manifests, TTL/GC, crash cleanup, and optional authenticated encryption. Transport implementations include in-process, file, deterministic simulated, and bounded TCP paths.

Fork, clone, pause/checkpoint/resume, incremental ancestry, recomputation evidence, and lazy-legality assessment are separate operations. Forked sessions acquire distinct identities/epochs and share only authorized immutable chunks; arbitrary branch merge is excluded.

## Evaluation

The campaign runs multiple explicit seeds and retains exact commands, software/hardware manifests, raw flagship/converter artifacts, hashes, per-seed outcomes, Student-t confidence intervals, negative results, and adapter status. Report generation rejects changed raw artifacts.

The deterministic CPU tests establish exact reference-layout continuation, direct/canonical equality, non-KV state capture/resume, gateway sequencing, safe failed cutover, successful second migration, COW sharing, and unsafe changed-weight rejection. Observed converter and pause-window wall times are `observed_host` metrics; deterministic transfer curves and tractable planner cases are `synthetic_protocol`. The retained five-seed campaign runs both pre-copy and stop-and-copy CPU paths, and it compares the planner against exhaustive enumeration of its legal bounded candidates. Pre-copy had the shorter observed pause in every retained seed and planner regret was zero in that finite candidate set. Direct CPU conversion was a negative result: it lost every per-seed median comparison to canonical conversion. Exact values and confidence intervals are generated into `reports/continuum-*` from retained artifacts rather than copied into this static report.

## Fault tolerance

The fault catalog defines ground-truth labels, component, phase, activation interval, and expected response for source/destination/gateway/coordinator failure, transfer loss/duplicate/corruption/timeout/reorder, stale epochs, dirty overflow/non-convergence, OOM, conversion/validation/page-table failure, token duplicates/gaps, client disconnect, cancellation, clock skew, and warm regression.

The flagship activates destination crash during validation. The transaction reaches `ROLLED_BACK`, the coordinator reopens its durable journal, the source retains epoch and watermark, and streaming remains valid. The retry reaches `COMPLETED`, moves to epoch 2, and rejects stale-source output.

The Rust bounded checker explores finite message/fault interleavings and records bounds, assumptions, coverage, invariants, and minimized counterexamples. The retained seed-41 safe configuration explored 1,659 states and 3,216 transitions to maximum depth 22 with all 15 invariants passing. Exploration was complete within declared bounds of three messages, one token, one injected fault per execution, depth 32, 100,000 states, and six timeout ticks. This is bounded evidence, not an unbounded proof.

## Security

State is sensitive tenant data. Chunk identity is tenant-scoped to disable cross-tenant dedup by default. Strict schemas, size bounds, content hashes, capsule roots, AEAD-associated context, journal hashes, owner epochs, fencing, and transfer replay records address corruption/substitution/replay. Optional AES-GCM storage fails closed when unavailable. Bare TCP authenticates setup/integrity but is not confidential and must run inside an encrypted authenticated channel outside localhost.

Adapters and generated code receive narrowly scoped state but no storage keys or inherited secrets. Logs exclude payloads and token content. TTL/GC is best-effort logical deletion; physical secure erasure is not claimed.

## Related work

vLLM, SGLang, Dynamo, NIXL, NCCL, UCX/RDMA, PyTorch checkpoint/DTensor, and optional BIFROST-like systems supply runtime representation, serving composition, tensor placement, or movement. Continuum composes with them but retains the meaning of state and the ownership transaction. VM pre-copy and process checkpoint systems motivate delta transfer but migrate address-space state that Continuum deliberately reconstructs. Content-addressed checkpoint systems motivate COW but do not define streaming token commitment or cross-model semantic validity.

## Limitations

The recorded environment is Apple Silicon CPU-only. No NVIDIA GPU, CUDA/Triton kernel, RDMA, multi-node, vLLM-to-SGLang, or production latency result is claimed. The SQLite coordinator is local; bare TCP lacks confidentiality while the exercised optional AES-GCM mode uses pre-shared keys rather than production PKI peer identity. Deletion is best effort, quality-bounded production claims need workload-specific evidence, and model checking/continuation checks are bounded. Public runtime APIs may not expose all logical state and remain version-scoped.

## Future work

Future work is limited to currently unexercised extensions rather than prerequisites for the CPU thesis: verified GPU layout kernels, hardware-backed PyTorch/Genesis and public-runtime migrations where APIs permit complete state, linearizable distributed coordinator adapters, encrypted high-performance transports, richer recurrent/state-space adapters, and larger bounded/probabilistic fault campaigns. Each must preserve the same compatibility, evidence, fencing, and no-silent-fallback requirements.
