# ADR 0026: Use deterministic multi-fidelity portfolio search

- Status: accepted
- Date: 2026-08-02

## Context

Whole-genome search is combinatorial, objectives conflict, verification is expensive, and a proposal score cannot be trusted as evidence. A single opaque optimizer would make failures, budgets, and diversity difficult to audit.

## Decision

Combine deterministic beam, evolutionary, local, and novelty proposal engines with stable content identities. Enforce mutable-region whitelists before evaluation. Evaluate candidates through ordered fidelity stages from static pruning and analytical/digital-twin evidence through compilation, reference/property/model checking, simulation, hardware/end-to-end benchmarks, shadow, and canary.

Preauthorize and account for wall, CPU, GPU, cloud, external-synthesis, candidate, compilation, benchmark, and verifier budgets. Preserve a bounded Pareto archive over correctness, quality, latency, goodput, cost, energy, startup, memory, reliability, complexity, and transition cost. Keep evaluator evidence independent of proposal heuristics.

## Consequences

Search is reproducible, budgeted, diverse, and auditable, and low-fidelity rejection protects scarce hardware experiments. It may evaluate more candidates than a specialized optimizer. The complete search engine is exercised with focused evaluators; the local cancellation synthesis fixture currently uses a smaller fixed candidate set, and real GPU/external-agent stages require explicit adapters and opt-ins.

