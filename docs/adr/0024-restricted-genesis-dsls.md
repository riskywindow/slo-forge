# ADR 0024: Restrict synthesized policies and rewrites to checked DSLs

- Status: accepted
- Date: 2026-08-02

## Context

Arbitrary generated Python, CUDA, or configuration code cannot be exhaustively inspected and must not become part of the promotion authority. Many useful scheduling and rewrite decisions need only a small typed language.

## Decision

Use a loop-free scalar policy DSL with declared input/output types, finite ranges, a fixed operation limit, a checker, bounded bytecode compiler, deterministic interpreter, graph view, simplifier, seeded mutations, and exhaustive equivalence over bounded Boolean/integer domains.

Use similarly typed restricted records for tensor rewrites and state/distributed transformations. Every rule declares preconditions, semantic category, quality/resource implications, verification obligations, and fallback or rollback. Reject I/O, dynamic loading, unknown operations, unchecked floating reassociation, and out-of-domain output.

Generated general-purpose code remains untrusted and executes only in the sandbox; a DSL acceptance result does not replace differential, quality, model-check, resource, performance, capsule, or rollout gates.

## Consequences

Policy execution is bounded, deterministic, and explainable, and small domains support real counterexamples. Expressiveness is intentionally limited. Float equivalence, native kernels, live state conversion, and transformations outside implemented rules require separate verifier adapters and cannot be smuggled through an extension.

