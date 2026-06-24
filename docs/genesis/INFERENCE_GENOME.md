# InferenceGenome v1

`InferenceGenome` is the canonical typed description of a serving implementation. Version `1.0.0` uses API identifier `sloforge.io/genesis/v1` and is implemented independently in Python and Rust. Core objects reject unknown fields. Extensibility is restricted to namespace-qualified entries in `extensions`; an extension cannot weaken validation of a core field.

## Eight regions

| Region | Represented decisions |
| --- | --- |
| Workflow | Typed steps and edges, model/tool/verification/branch/loop kinds, probabilities, deadlines, priorities, shared prefixes, future requests and cancellation |
| Request | Bounded admission, deadlines, batching eligibility, routing, queue discipline, cancellation, retry, streaming, tenant/workflow identity, quality tier and fallback |
| Serving | Aggregated/disaggregated topology, prefill, decode scheduling, continuous batching, speculation, cascades, chunking, migration and worker roles |
| State | Persistent state kind, key, ownership, layout, precision, retention, replication, migration, offload, checkpoint, eviction, recomputation, consistency and recovery |
| Distributed | TP/PP/DP/EP/context degrees, rank and expert placement, collective DAG, transfer path, failure domains and recovery variants |
| Tensor | Symbolic dimensions, typed values, shapes, strides, layouts, alias and state dependencies, operators, numerical contracts and rewrite history |
| Kernel | Backend, target, launch and tile configuration, resource estimates, layout/shape/dtype domains, determinism, tolerance, benchmark evidence and fallback |
| Recovery | Safe points, state contracts and conversion, transfer, active-stream behavior, rollback, shadow/canary and degraded modes |

Every mutable node carries a stable identifier, semantic contract, resource requirements, legal rewrite rules, proof obligations, hardware/software preconditions, quality and performance implications, uncertainty, hot-swap category, lineage/evidence references, extension data, and a frozen flag. A non-frozen node without legal rewrite rules is invalid; every node must carry proof obligations.

## Structural validation

The Python and Rust parsers validate closed workflow references, unique state and tensor identifiers, tensor operator references, symbolic ranges, speculative draft declarations, finite numbers, hashes, and the typed enumeration domains. Candidate lifecycle and budget validation are part of the same canonical IR package but remain distinct from the genome itself.

Canonical JSON is UTF-8 with sorted object keys, no insignificant whitespace, and finite JSON numbers. SHA-256 over those bytes is the stable content identity. Golden fixtures cover genome, transformation, candidate and counterexample documents. Cross-language tests compare canonical bytes and hashes, including floating-point edge cases, and property tests vary the explicit seed.

## Compatibility

JSON Schemas are generated under `schemas/inference_genome/`, `schemas/transformation/`, `schemas/candidate/`, and `schemas/counterexample/`. The migration layer recognizes only stable v1 and the known internal `v1alpha1`/`0.1.0` fixture. It performs explicit lossless field renames and rejects unknown versions, unknown kinds, or documents containing both old and new names. There is no optimistic “best effort” migration.

The Rust/Python boundary is versioned JSON, normally materialized as canonical files. The bounded model checker has its own request/result v1 schemas over subprocess standard input/output. These contracts avoid coupling the trusted validator to Python object identity or synthesis-agent reasoning.

## Baseline construction

`compile_inference_genome` lowers a validated zero-day inspection without importing model source. It produces a conservative CPU genome: bounded FIFO request admission, aggregated reference serving, explicit per-field persistent state, single-rank placement, reference tensor operators, a PyTorch/reference kernel record, and a drained recovery transition. Inspection obligations are retained as proof obligations. Changing the package, manifest, or explicit seed changes the genome identity.

## Current scope

The IR can represent the complete declared stack, and cross-language conformance is exercised. Representation does not prove that every possible combination has a compiler or runtime lowering. The current baseline compiler intentionally emits a conservative subset, and focused mutation modules cover selected policy, tensor, state and Fabric changes. Arbitrary extension semantics, arbitrary Python control flow, and arbitrary whole-genome transformations remain unsupported unless a typed compiler and verifier adapter exists.

