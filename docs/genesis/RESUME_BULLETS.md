# Genesis resume bullets

These bullets intentionally use placeholders for performance and evaluation metrics. Replace a placeholder only with a value read from a final artifact-backed report, and keep the artifact path in the source resume notes. Do not infer hardware results from CPU or simulated evidence.

- Architected SLOForge Genesis, a proof-carrying inference supercompiler spanning an eight-region serving IR, zero-day model inspection, generated streaming runtimes, restricted synthesis DSLs, independent verification, capsules, lineage, and guarded evolution.

- Implemented strict Python/Rust `InferenceGenome`, Transformation, Candidate, and Counterexample protocols with canonical cross-language JSON/SHA-256 identity, schema migration, property-tested round trips, and explicit proof obligations on every mutable genome node.

- Built a contract-first zero-day frontend and conservative runtime generator that converts unseen-style model packages into bounded deterministic streaming servers without importing source during default inspection; explicitly quarantined unresolved SSA/tensor bindings instead of inventing an algebraic graph. Populate `<VALID_SYSTEM_RATE_FROM_SYNTHBENCH_REPORT>` and `<TASK_COUNT_FROM_SYNTHBENCH_REPORT>` only after a final benchmark campaign.

- Developed a deterministic multi-fidelity synthesis engine combining beam, evolutionary, local, novelty, surrogate ranking, a 13-objective Pareto archive, frozen-region enforcement, and hard accounting across nine budget dimensions. Do not claim an Autopsy-guided speedup: H4's guided-versus-unrestricted campaign is unevaluated.

- Implemented counterexample-guided synthesis that detected a real cancellation-after-commit violation, minimized it to a three-event schedule, generalized a reusable precondition, suppressed a repeated unsafe transformation, and selected a corrected cross-layer request/serving policy.

- Built a self-contained Rust explicit-state checker for streaming, cancellation, retry, state ownership/transfer, promotion, rollback, failure, and controller recovery; report bounded state/transition coverage as `<MODELCHECK_STATE_COVERAGE>` and `<MODELCHECK_TRANSITION_COVERAGE>` from preserved checker artifacts.

- Designed an independent proof-carrying capsule validator and hostile-code sandbox that detect artifact tampering, stale evidence, dependency/hardware/contract mismatch, incomplete counterexample scope, and benchmark manipulation before promotion; required an operator-pinned digest and evidence/artifact anchors outside the capsule trust domain.

- Added a crash-consistent champion/challenger controller and exercised local evolution between two separately synthesized, independently validated capsules with shadow/canary gates, pre-promotion revalidation, active-stream pinning, rollback retention, and idempotent restart. External traffic and physical migration remain unmeasured.

- Created transactional SQLite optimization lineage with dependency-aware invalidation, time-decayed evidence confidence, negative-transfer learning, learned-constraint filtering, deterministic related-task retrieval, and JSON/GraphML export. Do not publish a lineage-efficiency metric: H5's comparative performance campaign is unevaluated.

- Built an evidence-targeted quantized-state kernel lab with restricted source generation, sandboxed stride/alias/numerical verification, randomized raw-sample benchmarking, and full-loop acceptance gates; retain `<KERNEL_END_TO_END_IMPROVEMENT>` as a placeholder because the preserved standalone CPU result is negative and no GPU speedup is claimed.

- Performance bullet template—do not use until H2/H4/H5 are actually evaluated: “Compared Genesis with configuration-only and single-layer baselines over `<SEEDS>` seeds; artifact-derived `<METRIC>` changed by `<VALUE>` within `<CONFIDENCE_INTERVAL>` at `<ARTIFACT_PATH>`.” CPU simulator values and harness timings are not substitutes for hardware-backed serving measurements.

Validation command references for resume source notes:

```text
make check
make genesis-check
make genesis-demo
make genesis-zero-day-demo
make genesis-redteam-demo
make genesis-evolution-demo
make synthbench-smoke
make genesis-evaluation
```

Command names are not metric evidence. If a target is unavailable or not exercised in the final checkout, remove it rather than implying success.
