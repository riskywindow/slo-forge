# ADR 0023: Use a typed proof-obligation-carrying Genesis IR

- Status: accepted
- Date: 2026-08-02

## Context

Whole-stack synthesis needs to change scheduling, state, distributed placement, tensor algebra, kernels, and recovery without allowing generated code to redefine correctness. Untyped dictionaries would make compatibility, hashing, independent validation, and lineage transfer ambiguous.

## Decision

Define strict versioned `InferenceGenome`, `Transformation`, `Candidate`, and `Counterexample` documents in Python and Rust. Split the genome into workflow, request, serving, state, distributed, tensor, kernel, and recovery regions. Require every mutable node to carry stable identity, semantic/resource contracts, legal rewrites, proof obligations, preconditions, uncertainty, evidence/lineage references, hot-swap category, and frozen state.

Reject unknown core fields and allow extension data only under namespace-qualified keys. Use explicit lossless migrations for the known alpha fixture and reject unknown versions. Generate JSON Schemas and verify canonical JSON/SHA-256 agreement with cross-language golden and property tests.

## Consequences

Candidates and evidence are independently inspectable and hash-addressable, and failures can attach to precise regions. The IR is verbose and adding a core field requires a schema/version compatibility decision. Representation does not imply that a compiler exists for every combination; unsupported lowering remains explicit.

