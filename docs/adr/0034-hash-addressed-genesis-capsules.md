# ADR 0034: Use strict hash-addressed GenesisCapsules as the promotion handoff

- Status: accepted
- Date: 2026-08-02

## Context

Promotion needs a stable connection between a candidate, generated artifacts, contracts,
dependencies, hardware, correctness evidence, raw performance evidence, operational evidence,
counterexamples, and rollback material. Mutable directory names or a single `verified` boolean
cannot preserve that connection.

## Decision

Define a versioned, closed `GenesisCapsule` schema independent of mutable synthesis state. Give
every artifact an origin, role, normalized relative path, size, and SHA-256 digest. Bind scoped
claims to typed independently issued evidence. Bind benchmark summaries to definition, baseline,
raw samples, software, workload, and hardware. Record unsupported cases and unverified assumptions.

Canonicalize the complete manifest with its digest slot set to `null`, address it by SHA-256, and
publish it immutably. Validate content, provenance, freshness, contract/hardware/dependency scope,
evidence completeness, counterexample corpus, and local promotion requirements without importing or
executing generated code. Pin the expected capsule digest in the controller.

## Consequences

Tampering and stale or incompatible evidence fail closed, and accepted/rejected artifacts are
replayable. SHA-256 provides integrity only when the expected digest is independently pinned; the
current capsule is not signed and does not authenticate its producer. More evidence increases
artifact size and validation cost, and incompatible schema changes require a major-version migration.
