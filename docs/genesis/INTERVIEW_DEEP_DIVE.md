# Genesis interview deep dive

## One-minute thesis

SLOForge Genesis changes the compiler question from “which known serving configuration should run?” to “what serving implementation should exist for this model, workload, hardware, quality contract, and SLO?” It analyzes a typed reference package, emits a whole-stack `InferenceGenome`, generates a conservative runtime, searches restricted implementation surfaces, independently verifies candidates, packages scoped evidence into a content-addressed capsule, and promotes only through shadow/canary/rollback controls.

The important design choice is distrust: generated source, policies, transformations, and proposal scores never become their own acceptance authority.

## Why this is a compiler system rather than an agent wrapper

The core representations and state transitions are deterministic without external LLM credentials:

- Python and Rust independently implement canonical Genesis IR and SHA-256 identity.
- The zero-day frontend recovers a typed call/state inventory, batching and symbolic domains without importing source by default; unresolved SSA bindings and tensor metadata remain an explicit non-rewriteable inventory rather than a guessed graph.
- Restricted policy, tensor, state, and Fabric transformations have typed preconditions and verifier obligations.
- Candidate lifecycle and nine resource budgets are persisted and cannot skip success stages.
- Rust performs bounded explicit-state protocol exploration over streaming, cancellation, failure, migration, promotion, rollback, and controller recovery.
- Capsule validation recomputes integrity and evidence gates independently of synthesis.

An optional synthesis agent could propose code, but the implemented local path does not need one and the trust argument never depends on its reasoning.

## End-to-end walkthrough

1. A reference manifest declares entry points, persistent state, semantic/quality contracts, supported domains, custom operators, workflow, and separate search/final corpora.
2. Static AST inspection records declared/recovered facts and refuses unsupported semantics. Its call inventory is not claimed to be SSA: unresolved input/output bindings, shapes, layouts, and aliasing remain obligations. Optional `torch.export` is a fail-closed sandboxed execution boundary.
3. The compiler lowers inspection evidence into eight genome regions: workflow, request, serving, state, distributed, tensor, kernel, and recovery.
4. Runtime synthesis emits a bounded single-worker streaming reference runtime with deterministic phase seeds, ownership/release discipline, cancellation safe points, fixed queues, metrics, and a differential harness.
5. Search proposes typed transformations and uses static-to-operational fidelity stages, hard budgets, mutable-region guards, a Pareto archive, and an independent evaluator.
6. Verification records separate semantic, quality, resource, statistical, model-check, hardware, and operational claims; there is no aggregate “verified” Boolean.
7. A capsule seals generated artifacts, raw evidence, compatibility, counterexamples, limitations, rollback, and lineage under one canonical digest. An operator-controlled context outside the capsule pins that digest and every trusted evidence issuer/artifact binding.
8. The flagship evolution path synthesizes and builds two distinct capsules, validates both, isolates the challenger, and pins active streams to their starting capsule while new work moves to the promoted challenger; the previous champion remains rollback-ready.

## The real counterexample story

The local synthesis fixture proposes a high-upside deadline batching policy that checks cancellation too early. An independent policy simulator executes an actual request schedule and observes cancelled work being scheduled for token commitment. Deterministic delta debugging reduces the failure to `admit, cancel, emit`. Genesis learns `cancel_check_before_emit == true`, persists the constraint, suppresses another member of the same unsafe family, and verifies a corrected candidate that changes both request and serving regions.

The scope matters: this is bounded cancellation-protocol evidence, not universal program correctness or a claim of global algorithmic novelty.

## Formal-methods boundary

The Rust model checker explores finite protocol state breadth first. It reports 13 invariants independently, including token conservation, retry safety, bounded queues, ownership, migration, active-request promotion, rollback, recovery, deadlock, and bounded progress. A depth/state truncation produces `inconclusive`; `universal_proof` is always false. Counterexample traces carry state fingerprints and are independently replayable, so modifying the trace is detected.

This model abstracts packet networks, native memory ordering, tensor numerics, CUDA/NCCL internals, and unbounded liveness. Those need separate evidence.

## GPU and performance position

Kernel synthesis is evidence-targeted rather than an attempt to rewrite attention. The focused CPU lab targets the HybridDecoder quantized persistent-state update from a measured synthetic microprobe, generates restricted candidates, checks strides/aliasing/non-finite/boundary cases in the sandbox, and retains randomized raw operator samples. The repeated operator loop is not end-to-end serving, so the CPU lab cannot promote a speedup claim without a separate full-stack gate.

The standalone upstream-ready bundle preserves a negative local CPU microbenchmark: its validation-first implementation is slower than the scalar reference on the measured host. The local capsule records a candidate-bound deterministic feasibility simulation but explicitly accepts no performance-improvement claim. Neither artifact is a hardware-backed serving speedup. GPU/Triton paths remain fail-closed and unexercised without real hardware and opt-in.

The multi-seed evaluation therefore leaves H2 (whole-stack performance), H4 (Autopsy-guided search efficiency), and H5 (lineage transfer efficiency) unevaluated. H7 and H9 have only local/scoped evidence. The mechanism demos are useful, but they do not substitute for the missing comparative campaigns.

## Persistent intelligence

Lineage stores tasks, accepted and rejected candidates, transformations, evidence, counterexamples, learned constraints, transfers, and invalidations in transactional SQLite. Retrieval requires explicit applicability, fresh passing evidence, time-decayed confidence, and no matching negative constraint. At least 20 percent of a seeded population remains unseeded, and every reuse requires reverification. Compiler/runtime/hardware/contract changes mark affected evidence stale rather than deleting history.

## Hard engineering tradeoffs

- Static inspection is safer but requires explicit contracts and cannot support arbitrary Python.
- Restricted DSLs improve explainability and bounded checking but limit expressiveness.
- Canonical JSON is slower than in-process FFI but makes evidence replayable across languages.
- Bounded model checking is CI-friendly but cannot establish unbounded correctness.
- Conservative coexistence/resource checks reject some feasible candidates but protect rollback capacity.
- Statistical gates require more trials and accept inconclusive outcomes instead of rewarding noise.
- Stream pinning may retain two runtimes longer, but avoids orphaning externally visible requests.
- External capsule identity and issuer authenticity require operator-distributed digests and evidence anchors; self-contained hashes alone cannot establish trust.

## What remains unclaimed

The implemented CPU vertical slice does not establish production-scale third-party model coverage, a universal transformation interpreter, real GPU speedup, multi-node deployment, external live promotion, or global novelty. Independent tensor/state/distributed/search modules are tested but are not all composed into the local cancellation fixture. The checked host exercised the macOS sandbox, not Linux bubblewrap or a Docker daemon. Hardware claims require raw hardware artifacts, and performance placeholders stay unpopulated until final artifact-derived evaluation exists.
