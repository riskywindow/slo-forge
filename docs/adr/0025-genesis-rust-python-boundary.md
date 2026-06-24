# ADR 0025: Extend the existing JSON subprocess boundary for Genesis

- Status: accepted
- Date: 2026-08-02

## Context

ADR 0001 and ADR 0013 establish versioned JSON as the SLOForge Rust/Python boundary. Genesis adds a canonical compiler IR and bounded model checker but does not justify another RPC or in-process FFI contract.

## Decision

Keep Python responsible for model inspection, synthesis, search, statistical evaluation, and orchestration. Keep Rust responsible for the independent canonical IR implementation and deterministic explicit-state protocol checking. Exchange strict versioned JSON through canonical files or bounded subprocess standard input/output.

Define the Genesis v1 canonical JSON profile in both languages and test byte/hash agreement. Give the model checker separate request/result v1 schemas and cap subprocess input. Retain HTTP/SSE only for a running serving data plane and JSON-lines only for the generated baseline runtime process protocol.

Do not make PyO3, an embedded interpreter, or a new network service the canonical evidence path. Any future acceleration must preserve replayable JSON artifacts.

## Consequences

Evidence is language-neutral, inspectable, replayable, and isolated from crashes. Serialization overhead remains outside the latency-critical token path. Schema evolution and floating-number rendering require conformance tests. The focused boundary is exercised locally; no claim is made here about remote or multi-node transport.

