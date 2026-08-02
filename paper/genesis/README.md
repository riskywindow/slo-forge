# SLOForge Genesis: Proof-Carrying Synthesis of Inference Serving Implementations

## Abstract

SLOForge Genesis extends SLOForge from selecting known deployment configurations to synthesizing a
serving implementation for a typed reference model, workload, hardware contract, quality contract,
and service-level objective. The system represents implementation choices in a versioned
`InferenceGenome`; proposes policy, tensor, state, distributed, kernel, and runtime transformations;
and treats every generated artifact as untrusted. Independent differential, property, resource,
statistical, and bounded protocol checks produce scoped evidence. A hash-addressed `GenesisCapsule`
binds an accepted genome to generated artifacts, contracts, dependencies, hardware, counterexamples,
performance samples, and rollback material before a separate controller may shadow or promote it.

The implemented CPU vertical slice statically inspects a previously unseen-style HybridDecoder
package, generates a bounded streaming reference runtime, synthesizes a cancellation-sensitive
batching policy, rejects the highest heuristic-upside unsafe proposal, minimizes the actual failing
schedule, learns a reusable precondition, and validates a corrected cross-layer request/serving
candidate. The checked-in environment had no NVIDIA GPU or CUDA compiler; this paper therefore
reports no GPU, multi-node, cloud, production-traffic, energy, or speedup result. CPU benchmark
samples exercise the harness, and the capsule's positive performance statement is explicitly a
deterministic service-model result rather than hardware evidence.

## 1. Motivation

Inference serving is a coupled program, not a bag of independent knobs. Request admission affects
batch composition; batching affects state lifetime; state layout affects memory traffic and
migration; placement affects collectives; kernels constrain layouts and shapes; and rollout must
preserve active streams. A fast local change can violate numerical quality, cancel after token
commitment, exceed coexistence memory during canarying, or make champion state incompatible with a
challenger.

Existing SLOForge already compiles deployment and physical plans, simulates queueing and faults,
diagnoses causal bottlenecks, and controls guarded rollout. Genesis preserves those components and
changes the question from “which known configuration should run?” to “which scoped implementation
should exist, and what independently checked evidence permits it to run?”

The design follows three principles:

1. Generation and deployment authority are separate.
2. Every claim is scoped to contracts, inputs, bounds, dependencies, and hardware.
3. Negative results are durable compiler knowledge, not discarded search noise.

## 2. System model

A Genesis task contains a reference package `R`, workload contract `W`, hardware/fabric contract
`H`, semantic contract `S`, quality contract `Q`, objectives `O`, explicit seed `s`, and finite budget
`B`. Compilation produces a baseline genome `G0`. A transformation sequence `T` produces candidate
genome `Gi`; independent evaluators produce evidence `Ei`.

Acceptance is not a scalar model score. For every claim `c`, the capsule records:

```text
claim(c) = statement + domain + assumptions + exclusions
           + verification level + immutable evidence references
```

A candidate may advance only through contiguous lifecycle stages. Static validity, compilation,
reference tests, property tests, bounded model checking, simulation, hardware measurement, shadow,
canary, capsule acceptance, and promotion remain distinguishable. Synthetic evidence cannot satisfy
a hardware stage. Terminal failures are persisted with typed reasons.

The search budget is a vector over wall time, CPU time, GPU time, cloud cost, external synthesis
cost, candidate count, compilation count, benchmark count, and verifier time. An evaluator receives a
seeded stage request only after its worst-case reservation fits; reported usage cannot exceed the
reservation.

## 3. InferenceGenome IR

`InferenceGenome` v1 is independently implemented in Python and Rust and serialized as canonical,
versioned JSON. Its eight regions span the serving stack:

| Region | Core concerns |
| --- | --- |
| Workflow | Model/tool/verification DAGs, branches, loops, deadlines, prefixes, future work, cancellation |
| Request | Admission, priority, deadlines, batching eligibility, routing, retry, streaming, fallback |
| Serving | Prefill/decode policies, continuous batching, speculation, cascades, migration, worker roles |
| State | Ownership, layout, precision, retention, replication, migration, offload, recovery |
| Distributed | Parallel degrees, placement, collectives, transport, overlap, failure domains |
| Tensor | Operators, symbolic shapes, dtypes, strides, aliasing, numerical contracts, rewrite history |
| Kernel | Source/backend, launch/layout domains, resources, determinism, evidence, fallback |
| Recovery | Safe points, compatibility/conversion, active streams, shadow, canary, rollback |

Each mutable node has a stable identifier, semantics, resource requirements, legal rewrites, proof
obligations, hardware/software preconditions, quality implications, performance uncertainty,
hot-swap category, lineage/evidence references, and frozen state. Namespaced extensions cannot weaken
core validation. Explicit migrations cover known v1 predecessors; unknown versions fail rather than
being guessed.

The associated transformation IR records source and target patterns, semantic category, exact or
approximate status, pre/postconditions, quality and resource effects, affected regions, verifier and
benchmark obligations, rollback strategy, proposal origin, learned constraints, and ancestry. This
makes a transformation reviewable independently of the search method that proposed it.

## 4. Synthesis surfaces

### 4.1 Zero-day frontend and baseline compiler

A strict reference-package manifest declares entry points, persistent state, semantic and quality
contracts, supported domains, custom operators, and separate search/final evaluation corpora. Default
inspection parses Python AST without importing source. Recovered calls, state accesses, aliases,
control-flow boundaries, shapes, dtypes, and batching axes are distinguished from declared facts;
unknown semantics become proof obligations or blocking diagnostics.

Static inspection does not recover SSA value bindings or complete output shape/dtype/layout/alias
metadata. The initial `TensorGenome` therefore contains the declared inputs and an
`unresolved_static_call_inventory` extension listing every recovered operation and its obligations;
it emits no `TensorOperator` for those unresolved calls. This prevents tensor search from treating
source order as invented data flow. A future export-backed graph may populate operators only after it
supplies the missing bindings and metadata.

The compiler first emits a conservative CPU genome and generated runtime. The runtime has a bounded
admission queue, bounded per-request output queues, one deterministic FIFO worker, request-owned state,
basic serial microbatch collection, stable per-operation seeds, checks cancellation/deadline at every
operator and emission boundary, and emits ordered token events plus one terminal event. The generated
bundle includes source/config hashes, a correctness harness, offline deployment intent, and an artifact
manifest. It opens no listener and must execute through the Genesis sandbox.

### 4.2 Restricted policy DSL

The policy DSL is typed, deterministic, loop-free, and I/O-free. Its parser, formatter, type checker,
bytecode compiler, interpreter, graph representation, simplifier, mutation operators, and bounded
equivalence checker provide a deliberately small policy synthesis surface. Outputs are bounded and
typed. Exhaustive equivalence currently covers bounded Boolean/integer domains; it is not a solver for
arbitrary floating-point policies.

### 4.3 Tensor and state transformation

The tensor system represents symbolic shapes, dtype/layout/alias information, state dependencies,
numerical modes, and rule-specific obligations. It applies an explicit bounded rewrite set for
decomposition, fusion, layout/conversion simplification, specialization, and numerical changes. It is
not general equality saturation, and floating-point reassociation requires an approximate contract.

The state compiler represents ownership, lifetime, layout, precision, replication, migration,
checkpointing, eviction, and rollback compatibility. Its trace checker rejects ambiguous ownership,
double release, use after release, partial-state visibility, incompatible reuse, and bound violations
within the declared trace model. Real DMA, allocator integration, offload, and state conversion remain
outside the CPU fixture.

### 4.4 Distributed and kernel synthesis

Distributed synthesis reuses SLOForge Fabric validators and mutates selected properties of an already
valid physical plan. A structural change invalidates stale performance evidence and creates protocol
obligations. The module does not synthesize every possible topology or parallel degree.

The focused kernel lab targets the HybridDecoder's saturating quantized state update rather than
reimplementing attention. It generates deterministic restricted CPU candidates, validates their AST
against an allowlist, executes through the existing sandbox, tests int8 boundaries, random values,
stride/alias behavior, and non-finite input rejection, and records warmup, raw samples, content hashes,
confidence intervals, and repeated focused-operator-loop measurements. That loop is not end-to-end
model or serving execution, so it cannot promote a speedup claim by itself. The Triton
adapter is feature-gated and was not executed on the checked-in host.

### 4.5 Search

Deterministic beam, evolutionary, local, and novelty proposal engines produce typed candidate designs.
The search engine enforces frozen-region whitelists, a multi-fidelity stage ladder, nine hard budget
dimensions, and a bounded Pareto archive across correctness, quality, latency, goodput, throughput,
cost, energy, startup, memory, reliability, complexity, and transition cost. Proposal upside and
surrogate acquisition guide experiments but never count as acceptance evidence.

## 5. Counterexample-guided verification

Genesis implements propose, verify, minimize, learn, and correct as an auditable loop. A verifier
adapter owns the acceptance result; the synthesizer cannot set it. On failure, Genesis persists the
typed witness, repeatedly replays reductions, retains only reductions that reproduce the same
contract violation, generalizes a typed family precondition, and stores both the counterexample and
constraint in lineage.

The exercised cancellation fixture begins with this failing schedule:

```text
admit, schedule, prefill, decode, cancel, emit
```

The protocol simulator observes a committed token emitted after cancellation. Deterministic delta
debugging reduces the schedule to:

```text
admit, cancel, emit
```

Genesis learns `batching.cancel_check_before_emit == true`, suppresses a second candidate from the same
invalid family without a full verifier run, and independently accepts a corrected candidate that
changes both request cancellation and serving batching behavior. The rejected proposal had the
highest declared heuristic upside; no measured speed claim is attached to it.

Operator verification separately exercises rare shapes, dtypes, non-contiguous strides, aliasing,
determinism, exceptional values, and exact/approximate comparisons against a reference callable.
Quality verification measures exact-token match, top-1, top-k overlap, KL/JS divergence, and maximum
logit error against caller-supplied thresholds. Resource analysis conservatively checks declared
device/host/state/queue/buffer/process demand including champion-challenger coexistence. Statistical
performance acceptance requires the complete seeded bootstrap interval to clear both noise and
practical-significance thresholds.

The Rust explicit-state checker covers bounded admission, queues, streaming token commitment,
cancellation, retry, state ownership/transfer, worker failure, rollout, controller crash/restart,
promotion, and rollback. Breadth-first failures contain shortest replayable traces. Bounds, state and
transition counts, assumptions, invariant-specific results, and truncation are recorded; truncated
properties are inconclusive and `universal_proof` is always false.

## 6. Proof-carrying GenesisCapsules

A capsule is a strict canonical manifest whose digest covers identity, artifacts, claims, evidence,
contracts, dependencies, hardware scope, counterexamples, lineage, and rollback material. Each file is
bound by safe relative path, role, origin, SHA-256, byte length, media type, and executable status.
Generated artifacts cannot label themselves trusted.

The independent validator rejects path escape, symlinks, missing/changed files, manifest tampering,
issuer/role mismatch, expired or future evidence, incompatible contracts/hardware/dependencies,
altered benchmark summaries, incomplete promotion evidence, and absent rollback/counterexample
material. It recomputes benchmark median and tail values from finite raw samples. Promotion-required
issuer labels are checked against an operator-supplied context outside the capsule; its trusted
evidence anchors bind the evidence-record digest, issuer/version, and exact artifact digests.

SHA-256 supplies integrity and content identity, not signer identity. Deployment must pin the expected
capsule digest and trusted evidence anchors through an external operator-controlled channel. The CLI
rejects a validation context located inside the capsule root. The implementation does not claim a
public-key signature scheme or universal proof.

## 7. Autopsy-guided search

Genesis consumes typed Autopsy diagnoses with clock-alignment and confidence thresholds. A causal
bottleneck maps to an explicit mutable-region/transformation-family budget; all other genome nodes are
recursively frozen. The mutation guard rejects out-of-budget candidates before compilation or
benchmarking. Next-bottleneck, estimated verification cost, and expected-upside intervals are retained
only when supplied by diagnosis evidence.

Focused tests cover diagnosis admission, network-bottleneck attribution, recursive freezing, proposal
rejection, and comparison metrics derived from supplied guided/unguided summaries. No real
guided-versus-unguided hardware campaign is present, so the paper claims search-surface enforcement,
not measured search-efficiency improvement.

## 8. Lineage transfer

The embedded lineage graph stores genomes, transformations, positive and negative outcomes,
counterexamples, learned constraints, evidence, model/workload/hardware features, dependencies,
invalidation, and transfer events. Retrieval filters on accepted outcome, explicit applicability,
dependency preconditions, learned constraints, and fresh passing evidence. Confidence decays with age,
and dependency invalidation makes affected evidence stale.

Related transformations seed at most 80 percent of a population; deterministic unseeded proposals
preserve diversity. Every retrieved transformation has `requires_reverification=true`, and negative
transfer reduces later ranking rather than disappearing. Tests exercise related-over-unrelated
ranking, deterministic initialization, stale-evidence exclusion, invalidation, negative-transfer
penalty, and diversity. A full performance campaign comparing empty, unrelated, related, and stale
lineage has not been run.

## 9. Continuous evolution

The persisted controller models a validated champion, isolated challengers, shadow/canary gates,
promotion, retained rollback, and stream-to-capsule leases. The flagship implementation builds the
champion and challenger as two distinct capsules from separate deterministic synthesis runs. Both are
validated against their own trusted contexts; challenger validation runs at registration and again
immediately before promotion. Existing streams remain on their starting capsule for policy-only and
request-boundary swaps; new requests use the promoted champion. Other transition categories require
active-stream compatibility or a drain, and operator-required transitions cannot promote
autonomously.

Controller snapshots use bounded canonical JSON, a payload digest, `fsync`, and atomic replacement.
Repeated event identifiers are idempotent after restart. Tests cover workload-drift triggering,
capsule/canary rejection, successful local shadow and canary, promotion, rollback, active-stream
leases, restart recovery, state tampering, and the external live-promotion guard. Gate observations
bind the candidate, capsule, evidence digest, and controller seed. Shadow/canary observations are
supplied deterministic local evidence and the later fabric-degradation trigger is simulated; external
production traffic, real physical faults, and live state conversion remain unexercised.

## 10. Implementation

Python owns conservative package inspection, genome compilation, runtime generation/orchestration,
restricted synthesis surfaces, search, statistical verification, artifact/capsule handling, Autopsy
guidance, lineage, red-team generation, ServingSynthBench, and evolution control. Rust owns the
independent canonical Genesis IR implementation and the bounded explicit-state protocol checker. The
primary language boundary remains versioned canonical JSON over files or bounded subprocess
stdin/stdout.

Generated code runs without a shell in a fail-closed OS sandbox. Inputs are read-only, one artifact
directory is writable, the environment is rebuilt from an allowlist, credentials are rejected,
network and child creation are denied, resource/output/time bounds are enforced where the platform
supports them, and process groups are cleaned up. The checked Darwin host exercised the macOS
`sandbox-exec` backend. Its hard address-space limit is unavailable, so memory-hostile execution needs
an outer VM/container. The Linux bubblewrap/user-namespace backend is implemented but was not
exercised; missing capability returns `policy_unavailable` instead of silently executing unsandboxed.

The HybridDecoder reference fixture combines sliding-window attention-like computation, recurrent
state, sparse expert dispatch, a saturating int8 state transform, speculative head, custom top-2
sampler, and dynamic sequence length. It has no handwritten production serving adapter.

## 11. Evaluation

### 11.1 Evidence available at this snapshot

The immutable baseline record is
[`artifacts/genesis/baseline/record.json`](../../artifacts/genesis/baseline/record.json). It identifies
commit `435a04799a831c3d19fce18eb816b206d23778d7` and records the pre-Genesis checks shown below.

| Baseline command | Recorded outcome |
| --- | --- |
| `make check` | 362 Python passed, 3 skipped; 28 UI passed |
| `make fabric-check` | 292 Python passed; 31 Fabric Rust passed |
| `make demo` | 120 live and 120 simulated requests |
| `make fabric-demo` | passed |
| `make autopsy-demo` | passed |
| `make forgeci-demo` | passed; recorded bisection correct |
| `make warmpath-demo` | passed; recorded checksum verified |

The same record identifies macOS 15.6.1, x86_64 process architecture, 12 logical CPUs, 24 GiB memory,
Python 3.12.7, Rust 1.93.1, no NVIDIA GPU, no CUDA compiler, no Genesis GPU/synthesis budget, and all
external/live opt-ins disabled.

During the pre-handoff review on 2026-08-02, the focused CPU suite was executed as:

```console
PYTHONPATH=python .venv/bin/pytest -q \
  tests/python/test_genesis_*.py tests/python/test_synthbench.py
```

The latest review checkpoint reported `188 passed, 1 skipped`; the skip was the optional
`torch.export` path because PyTorch was not installed. This is a moving pre-handoff test observation,
not a serving-performance benchmark; the final report and preserved command log are authoritative for
the handoff count.

### 11.2 Findings supported by the CPU suite

- Static inspection did not import hostile fixture source, produced typed diagnostics/obligations,
  and kept unresolved SSA call bindings out of the executable tensor rewrite graph.
- The generated HybridDecoder runtime passed deterministic replay, exact token, streaming order,
  cancellation-before-commit, queue saturation, state ownership balance, source-tamper, health,
  metrics, subprocess timeout, and clean-shutdown tests.
- CEGIS rejected and minimized the cancellation failure, persisted a learned precondition, suppressed
  a repeated invalid family member, and selected the independently checked correction.
- Capsule tests detected manifest/artifact tampering, stale evidence, hardware/dependency mismatch,
  altered benchmark summaries, incomplete evidence, and unsafe paths.
- Search tests exercised deterministic proposal portfolios, lifecycle ordering, all nine budget
  dimensions, Pareto pruning/crowding, mutable-region enforcement, and hardware opt-in rejection.
- Red-team tests generated executable tensor, protocol, topology, resource, and benchmark-integrity
  findings and replayed the hashed regression corpus.
- Lineage tests exercised deterministic transfer/invalidation. The flagship evolution flow built two
  separate capsules and exercised local champion/challenger validation and state-machine safety.
- ServingSynthBench CPU smoke exercised public/hidden separation, deterministic task generation,
  exact validity checks, randomized run ordering, raw timing persistence, audit checks, and
  artifact-derived reporting.

### 11.3 Results not established

No Genesis artifact at paper-authoring time supports a hardware-backed latency, throughput, goodput,
cost, energy, GPU-kernel, multi-device, or optimization-speedup comparison. The kernel path exercises
real CPU timing and conservative acceptance logic; its standalone preserved result is negative. The
capsule's positive performance evidence is a deterministic service-model simulation and is labeled as
such. H2 (whole-stack performance), H4 (Autopsy-guided efficiency), and H5 (lineage-transfer
efficiency) are unevaluated. H7 and H9 are partially evaluated only in local/scoped fixtures. The
focused suite also does not establish a complete four-category search or external production
evolution.

## 12. Related work

Genesis builds on inference schedulers and runtimes, tensor compilers and rewrite systems, kernel DSLs,
counterexample-guided synthesis, property testing, proof-carrying code, bounded model checking,
distributed program planning, workflow-aware serving, and safe deployment control. Its research
position is the integration of a full-stack serving genome with independent scoped evidence and
durable negative results—not a claim to have invented the constituent mechanisms.

The detailed comparison and primary references are in
[`docs/GENESIS_RELATED_WORK.md`](../../docs/GENESIS_RELATED_WORK.md).

## 13. Limitations

- Static AST inspection does not support arbitrary reflection, dynamic import, opaque native state,
  arbitrary data-dependent control flow, or undeclared custom semantics. Its recovered calls are an
  unresolved inventory, not an SSA algebraic graph.
- The generated baseline is a serial conservative reference runtime, not an optimized HTTP/SSE,
  distributed, or GPU serving engine.
- The current local synthesis fixture composes request and serving policy; other synthesis surfaces
  are independently tested but not all joined into one unrestricted whole-genome run.
- The tensor explorer is a bounded explicit rule system, not general equality saturation. State and
  distributed modules model selected transformations rather than real data movement or arbitrary
  topology synthesis.
- NumPy property tests and the Rust protocol checker cover declared finite domains. They do not prove
  arbitrary native code, all numeric values, weak-memory behavior, CUDA/NCCL protocols, or unbounded
  liveness.
- Performance acceptance assumes an honest benchmark harness. Statistical recomputation alone cannot
  detect every timer, clock, affinity, thermal, cache, synchronization, fallback, or input-distribution
  manipulation.
- SHA-256 detects changed content only when the expected digest is pinned; it does not authenticate an
  untrusted replacement manifest.
- The exercised sandbox depends on deprecated macOS `sandbox-exec` and cannot enforce a hard memory
  ceiling. Linux bubblewrap/user namespaces and the Docker-daemon smoke were not exercised on the
  checked host; neither backend is a virtual-machine or cgroup boundary.
- Search is deterministic and bounded but not globally optimal. Lineage may reduce ranking quality;
  reused transformations always require reverification.
- The CPU evaluation contains no hardware-backed inference result and no evidence of external live
  promotion.

## 14. Security and operational safety

Genesis assumes generated code may be malicious, dependencies may be compromised, benchmarks may be
manipulated, evidence may be stale, and lineage may be poisoned. Generated source cannot modify or
serve as the validator, benchmark acceptance logic, promotion gate, raw evidence, or baseline result.
Proposal engines and transferred lineage are advisory inputs only.

Default operation is offline. No paid resource, external synthesis service, GPU, multi-node action,
privileged probe, external deployment, or live promotion is authorized without its separate opt-in
and budget. A capsule does not grant any of those capabilities. Promotion requires a freshly
validated capsule, complete evidence for the requested mode, compatible current contracts/hardware/
dependencies, and a retained rollback path.

Residual risks include host-kernel or sandbox-runtime vulnerabilities, incorrect fingerprint
collection, malicious but formally well-shaped benchmark evidence, issuer compromise before expiry,
and controller defects. These are recorded trust assumptions rather than hidden behind a `verified`
label.

## 15. Future work

Future evaluation should create artifact-backed, multi-seed comparisons for the nine hypotheses;
compose all synthesis surfaces in one bounded flagship search; run full empty/unrelated/related/stale
lineage campaigns; and measure Autopsy-guided versus unrestricted search. Compatible NVIDIA hardware
would permit the focused Triton experiment and end-to-end generated runtime benchmark; multiple
devices would permit actual Fabric and state-transfer evaluation. External shadow/canary execution
should remain a separate opt-in campaign with active-stream and rollback evidence.

Compiler work can broaden supported reference semantics, add solver-backed shape/index certificates,
connect more counterexample minimizers, and replace selected bounded rewrite sets with equality
saturation where it improves coverage without enlarging the trusted boundary. Security work should
add a maintained Linux cgroup/seccomp deployment profile and an authenticated capsule-signature layer.
None of these unexercised paths is required to interpret the scoped CPU results above as stronger than
they are.

## Artifact guide

- [Genesis architecture](../../docs/genesis/ARCHITECTURE.md)
- [InferenceGenome](../../docs/genesis/INFERENCE_GENOME.md)
- [Trust model](../../docs/genesis/TRUST_MODEL.md)
- [Zero-day frontend](../../docs/genesis/ZERO_DAY_FRONTEND.md)
- [Baseline runtime synthesis](../../docs/genesis/BASELINE_RUNTIME_SYNTHESIS.md)
- [CEGIS](../../docs/genesis/CEGIS.md)
- [Search](../../docs/genesis/SEARCH.md)
- [Autopsy-guided search](../../docs/genesis/AUTOPSY_GUIDED_SEARCH.md)
- [Lineage transfer](../../docs/lineage/TRANSFER.md)
- [Continuous evolution](../../docs/genesis/CONTINUOUS_EVOLUTION.md)
- [Executable red team](../../docs/redteam/ARCHITECTURE.md)
- [ServingSynthBench specification](../../docs/synthbench/SPECIFICATION.md)
- [Limitations](../../docs/genesis/LIMITATIONS.md)
- [Detailed related work](../../docs/GENESIS_RELATED_WORK.md)
- [Artifact-backed CPU demo](../../docs/genesis/DEMO_SCRIPT.md)
- [Adversarial security review](../../docs/genesis/RED_TEAM_REVIEW.md)
