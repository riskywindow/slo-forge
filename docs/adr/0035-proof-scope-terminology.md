# ADR 0035: State verification level and scope per claim

- Status: accepted
- Date: 2026-08-02

## Context

Build tests, differential examples, property tests, bounded enumeration, solver certificates, and
hardware observations provide different evidence. Collapsing them into one `verified` flag would
overstate correctness and allow evidence for one property or domain to be reused for another.

## Decision

Every capsule claim independently declares its category, result, evidence references, assumptions,
exclusions, and input/shape/dtype/hardware/dependency scope. Use these ordered level names:

1. Level 0, build: parse/type/compile/load/smoke evidence.
2. Level 1, differential: comparison with a reference over declared cases and tolerance.
3. Level 2, property: property, metamorphic, fuzz, or adversarial evidence over a declared domain.
4. Level 3, bounded exhaustive: complete exploration only within recorded finite bounds and
   fairness assumptions.
5. Level 4, solver-backed: an independently checked certificate under recorded assumptions.
6. Level 5, hardware/operational: measured hardware, shadow, canary, or recovery evidence tied to
   an exact environment.

Use “checked within scope” or “bounded evidence,” not “universally proved.” A higher level for one
claim does not raise another claim. Simulation is synthetic evidence; CPU measurements are scoped
host evidence; neither is GPU or multi-node evidence. `promotion_eligible` means that the validator
accepted the supplied context and required material, not that all claims reached Level 5 or that
live mutation is authorized.

## Consequences

Reports remain precise about what was checked and where. Users must inspect multiple claim results
instead of one convenient flag, and evidence producers must retain scope metadata. This prevents a
bounded model check, noisy point estimate, or synthetic result from being described as universal or
hardware-backed proof.
