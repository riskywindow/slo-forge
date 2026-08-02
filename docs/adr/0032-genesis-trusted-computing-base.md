# ADR 0032: Keep the Genesis trusted computing base small and independently measurable

- Status: accepted
- Date: 2026-08-02

## Context

Genesis generates source, policies, kernels, runtime components, deployment files, and evidence.
Trusting the generator's reasoning would make every synthesis dependency and external agent part of
the security and correctness boundary. The project needs a boundary that can reject a malicious or
mistaken proposal without executing it in the verifier.

## Decision

Treat all generated output, synthesis/search proposals, frontend heuristics, and external runtime
code as untrusted. The narrow capsule/execution TCB consists of strict capsule
types and schema, canonical serialization and hashing, bounded immutable I/O, the independent
capsule validator, content-addressed artifact verification, and sandbox policy/execution. The wider
operational TCB additionally includes the restricted-DSL compiler/checker, model-check result
validator, local evidence issuer, benchmark acceptance logic, promotion state machine, and rollback
controller when their claims are used.

The local capsule builder is part of the conservative evidence-issuance TCB because it creates the
trusted validation context and evidence anchors. It independently recomputes bounded proof and
property documents, re-hashes the reference package, and sandbox-replays final-corpus evidence;
its result must still pass the separately structured capsule validator. Generated code is never
imported into the validator process. Every trusted parser is strict, bounded, and fail-closed. TCB
source size and direct imports are measured with the commands in
[`TRUST_MODEL.md`](../genesis/TRUST_MODEL.md); OS, interpreter, schema, and transitive dependencies
are disclosed separately.

## Consequences

A proposal cannot promote itself, and a generator defect can be caught by independent evidence and
compatibility checks. The smaller boundary is reviewable and measurable, but it still depends on
Pydantic, Python, SHA-256, OS primitives, and the kernel sandbox. Source-line count is an audit aid,
not proof of correctness. New trusted dependencies require an explicit ADR and updated measurement.
