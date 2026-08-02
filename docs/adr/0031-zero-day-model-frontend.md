# ADR 0031: Make zero-day inspection static and contract-first by default

- Status: accepted
- Date: 2026-08-02

## Context

Importing an unseen model package during discovery can execute hostile code, and arbitrary Python semantics cannot be recovered reliably. Genesis needs enough typed information to generate a conservative runtime without silently inventing batching, mutation, state, or numerical behavior.

## Decision

Require a strict versioned reference-package manifest that declares entry points, state, semantics, quality, supported input domain, custom operators, workflow, and separate search/final corpora. Hash the manifest and all declared artifacts.

Perform default graph/state inspection from Python AST without importing the package. Classify facts as declared, statically recovered, proof obligations, or unsupported diagnostics. Block genome/runtime compilation on unsupported behavior. Permit `torch.export(..., strict=True)` only through an explicit declared fixture, require callers to execute that untrusted path inside the sandbox, and record version and graph/range evidence.

## Consequences

Default inspection is safe, reproducible, and fail-closed, and the resulting genome preserves uncertainty. Packages must provide more contracts, and reflection, opaque native mutation, and arbitrary control flow remain unsupported. The HybridDecoder and generated SynthBench packages exercise the supported CPU path; general third-party and GPU export coverage is not implied.
