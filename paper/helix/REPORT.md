# SLOForge Helix: Evidence-Carrying Counterfactual Learning Under Serving Constraints

## Abstract

Long-running language-model agents interact with mutable files, tools, services, cached model state,
and evolving policies. A learning system that records only prompts, responses, and terminal rewards
cannot reliably determine which policy acted, which state was reused, whether an effect was safe to
replay, why a sample was admitted, or whether a candidate may replace an active champion. SLOForge
Helix is a deterministic reference architecture for that systems problem. It coordinates model,
environment, action, and effect capture; constructs isolated counterfactual branches; records
policy-versioned trajectories and component-level reward evidence; derives branch-relative credit;
validates staleness and training lineage; schedules learning within serving-hard resource constraints;
and promotes through separately configured gates with active-session pinning and rollback. Separate
gate configuration does not establish organizationally independent evaluator governance. Artifacts are strict,
bounded, seeded, content addressed, and fail closed. The checked implementation and evaluation are
local synthetic CPU evidence. They establish executable interfaces and failure behavior, not
production sample-efficiency gains, GPU performance, universal replay, or autonomous safety.

## 1. Introduction

Serving systems and learning systems traditionally optimize different clocks. Serving protects
latency and availability for current requests; learning collects experience and changes the policy
that will serve future requests. Stateful agents make the boundary harder. One action can depend on a
model KV cache, an RNG counter, repository files, database contents, a tool observation, and a policy
version. A naive retry or counterfactual can duplicate a real effect. A delayed optimizer can train on
probabilities emitted by several policies. A promising candidate can break active sessions whose
state is incompatible with its weights.

Helix treats these as evidence and transaction problems before treating them as optimization
problems. The central rule is that no later phase receives more authority than its input evidence
supports. Capture mismatch prevents branching. Unsupported state reuse prevents rollout. Missing
behavior probabilities or mixed epochs prevent strict training. Invalid reward lineage prevents
credit. Serving infeasibility prevents learning work. Failed gates prevent promotion.

The implementation composes existing SLOForge subsystems rather than introducing a separate serving
stack. Continuum supplies execution-state capture and compatibility; Fabric supplies physical
resource contracts; Autopsy can supply structured failure signals; Genesis supplies generated-policy
and verification context; the shared controller and evidence conventions provide operational gates.

## 2. Design goals and non-goals

Helix aims to make one local learning transaction deterministic, inspectable, bounded, tenant-aware,
and recoverable. Every simulation and reference worker accepts a seed. Every major artifact has a
closed schema and content identity. Queues, documents, callbacks, resources, branches, verifier
outputs, and training steps are bounded. Production capture and external effects require explicit
authority, while cross-tenant reuse is disabled by default.

Helix is not a new foundation-model training algorithm, a replacement inference runtime, or a proof
of causal identification. It does not claim to snapshot arbitrary machines, reproduce hidden remote
services, safely execute every tool, or coordinate a multi-region promotion with SQLite. These
non-goals are part of the design rather than deferred footnotes.

## 3. Evidence-carrying intermediate representation

The canonical IR names policy epochs, environment and model capsules, branch points and groups,
trajectories, reward evidence, credit evidence, training batches, staleness reports, candidate epochs,
and learning transactions. Canonical JSON and SHA-256 make identity independent of map ordering.
Bounded loaders reject oversized or unknown documents, and Python/Rust schema fixtures preserve the
wire boundary.

Hashes provide integrity only when an expected hash is obtained through an independent path. They do
not authenticate the producer. Lineage records what an artifact claims to depend on; validators and
trust configuration decide whether those dependencies are authoritative.

## 4. Coordinated capture and branching

A branch point joins four independently advancing domains: actions/model generation, model execution
state, environment state, and effects. The durable capture coordinator quiesces sources, captures each
domain at a typed watermark, validates consistency, and publishes atomically. A mismatch, timeout, or
identity conflict terminates without a visible partial branch point.

Environment state is a content-addressed capsule of declared files, Git/dependency/runtime/service
metadata, policy, and bounds. The local backend reconstructs isolated tenant-scoped branches with path
and symlink safety. Continuum captures model state and reports which components are reused,
recomputed, replaced, or unsupported. Same-policy copy-on-write, controlled RNG mutation, and proven
cross-policy reuse are distinct strategies. An unsupported component cannot be hidden behind a
successful branch label.

This composition enables joint model/environment counterfactuals while keeping the exactness claim
narrow. Logical content sharing is not a measured reflink or GPU-memory result, and undeclared remote
state remains outside the capsule.

## 5. Effects, rollout, and replay

Effects are classified as pure, read-only, idempotent, compensatable, irreversible, or unknown.
Class-specific contracts and a watermark ledger prevent a speculative branch from treating an email,
charge, deletion, or unstable read like a pure function. Real external effects are disabled by
default; irreversible and unknown real effects cannot run speculatively.

Reference trajectories hash-chain ordered events and bind tokens/actions to behavior policy epoch,
log probability, seed/counter, branch state, environments, and effect ledger. Strict trajectories use
one epoch. Segmented trajectories explicitly describe boundaries and evidence needed to cross them.

Replay declares exact, causal, or semantic mode. Exact identity cannot downgrade silently after a
mismatch. The mode named causal compares declared event semantics and parent topology; it does not
identify causal effects. Causal and semantic modes report structured divergences and tolerances rather
than overclaiming bitwise equivalence. The comparator also does not prove that its caller restored
underlying model or environment bytes; that requires restoration and re-execution evidence.

## 6. Reward, credit, and experience selection

The reward worker runs deterministic commands and verifier-only hidden cases in a bounded sandbox.
It records source and verifier versions, input/output hashes, termination, return code, component
scores, and aggregate conservation. Hidden expected outputs do not enter the policy-visible sandbox.
Duplicate or changed reward submissions fail closed.

Branch-relative credit compares validated sibling outcomes and can emit scalar advantages and
pairwise preferences. Controlled-difference labels are supplied by the branch runner rather than
inferred by the credit code, and observational siblings are excluded from the controlled baseline.
This improves structural attribution over assigning one terminal score to every action, but it
remains vulnerable to unrecorded confounding and is not a causal proof.

Experience selection admits a subset rather than training on everything. Seeded random,
failure-only, uncertainty-only, and novelty-only baselines accompany a value-aware score using
failure, disagreement, uncertainty, novelty, rarity, safety, recurrence, Autopsy issue, capability
regression, branchability, and predicted learning value per cost. Consent, authorization, redaction,
tenant, privacy, effect, redundancy, budget, capacity, and anti-train-all gates are hard. All
candidates retain scores and exclusion reasons. The features remain predictions until evaluated.
The reference H9 campaign therefore measures downstream categorical-policy decisions after training
and treats random selection as its predeclared primary comparator; failure-, uncertainty-, and
novelty-only comparisons remain descriptive.

## 7. Staleness, batches, and trainer boundary

Staleness is evaluated per policy segment using update/time distance and supplied distribution or
compatibility evidence. The policy chooses rejection, truncation, or resampling and can require
log-probability or state recomputation. Unknown probabilities are not invented.

The training-batch builder joins trajectory, reward, credit, state, policy, split, and staleness
lineage. It rejects duplicate identities, holdout leakage, action or epoch mismatch, absent evidence,
and changed hashes. A narrow trainer adapter consumes only validated samples and emits a new immutable
epoch. The built-in CPU trainer exists to execute the contract; it is not a distributed LLM trainer.

Two additional reference campaigns exercise this boundary. H3 runs all four required objectives
over matched seeds and sample budgets with trainer replay and paired learning-curve effects. H4
derives parameter distance and action log probabilities from its actual categorical policies,
exercises current, stale, mixed, missing-probability, recomputed, and incompatible-state cases, and
requires zero invalid sample admissions. Neither campaign is evidence about neural convergence.

## 8. Serving-hard resource compilation

The resource compiler models serving and learning classes in deterministic ticks. Mandatory serving
demand is checked first against capacity, latency/queue forecasts, and budget. Learning work is then
admitted under resource, privacy, effect, staleness, deadline, branch, and preemption constraints.
Dedicated, static, utilization, FIFO, and value-aware policies provide baselines.

Idle reserved capacity may be lent and later reclaimed. Restart, checkpoint, and Continuum
preservation account for pause, storage, network, retained progress, lost work, and cost. Faults such
as traffic spike and GPU loss alter explicit capacity. The compiler's forecasts and learning values
are scenario inputs linked to evidence, not measured causal benefits.

The H6 campaign separately executes local restart, environment-only, model-only, and joint
Continuum/environment restoration. Loss and state bytes are derived from raw trajectories,
checkpoints, and restore evidence. Resume and serving values are deterministic reference accounting
ticks containing declared validation/dispatch weights, not wall time or an instrumented operation
trace.

## 9. Promotion, sessions, and rollback

A promotion capsule binds the candidate and parent epochs, training/evaluation evidence, lineage,
security results, compatibility, and rollback material. Independent lineage, reward-integrity,
quality, safety, serving, and compatibility gates precede shadow and canary. Only a passed canary may
change the champion pointer.

The local registry uses transactional compare-and-swap. A deliberately injected fault after pointer
update rolls back without exposing the candidate. Existing sessions are classified by compatibility;
incompatible or request-boundary sessions remain pinned to their original champion. The prior
champion remains the rollback parent. A distributed deployment would need a linearizable external
registry.

## 10. Security and privacy

Security boundaries cover production capture, artifact storage, tools, hidden tests, rewards,
training, and promotion. Production evidence requires tenant-matched, time-bounded authorization and
redaction evidence. Retention includes expiry, legal-hold bounds, and deletion receipts. Artifact
graphs are tenant scoped. Tool output is bounded and sanitized. Repository scanning looks for
secret-like material. External effects and cross-tenant reuse are opt-in, and unsafe speculative
effects remain forbidden.

The threat model does not cover a compromised kernel, stolen external credentials, malicious services
that violate declared contracts, or Byzantine distributed coordination. Sandbox strength depends on
the host. Capsules are not signed.

## 11. Evaluation methodology

The reference evaluation runs the actual local CPU demo over multiple explicit seeds, preserves raw
artifacts, computes intervals from executed observations, compiles every scheduler baseline, records
software/hardware manifests, and reports ten hypotheses separately. A hypothesis may be supported
within scope, partial, not supported, or not exercised. Missing GPU, production, or causal evidence is
not filled with simulator output.

The exercised evidence includes capture/recovery, environment isolation, branching and compatibility,
effect rejection, replay comparisons, deterministic reward, credit, strict and segmented staleness,
training, gate rejection, active-session pinning, promotion rollback, scheduler accounting,
experience governance, security controls, and a machine-readable CPU fault campaign spanning capture,
branching, rollout, reward, training, evaluation, promotion, and resources.

The current artifacts do not justify numeric claims here. Generated reports and raw JSON are the
authoritative source for any run-specific count, interval, or value.

## 12. Discussion

Helix's main tradeoff is evidence cost versus experimental authority. Coordinated capture pauses work;
state reports and capsules consume storage; separately configured gates delay promotion; retaining excluded
samples increases audit size. Relaxing those boundaries improves throughput but weakens the ability to
explain why an update or promotion was valid.

The architecture also separates optimization from safety. Value-aware experience and resource
policies may rank eligible work, but cannot override tenant, effect, serving, or promotion constraints.
This can leave capacity unused and reject useful evidence. Helix chooses explicit missed opportunity
over silent unsafe authority.

## 13. Related work

Helix builds on PPO/DPO and distributed actor-learner systems, interactive language-agent environments,
experience replay and prioritization, causal counterfactual reasoning, stateful inference serving,
checkpoint/content-addressing systems, SRE canary/rollback practice, and provenance standards. Its
claim is their evidence-aware composition across an agent learning transaction, not novelty of the
individual mechanisms. The detailed comparison and primary references are in
[HELIX_RELATED_WORK](../../docs/HELIX_RELATED_WORK.md).

## 14. Limitations

The test model, environment, rewards, and workload are synthetic local CPU fixtures. No production
traffic, real user data, GPU training, multi-node scheduler, or cloud promotion was exercised. The
trainer is tiny. Branch-relative comparisons do not eliminate confounding. Environment capture is
incomplete for arbitrary operating-system and remote state. Forecast-based schedules do not guarantee
real serving behavior. Local SQLite does not provide distributed consensus. Hashes are not signatures.
Some hypotheses remain partial or unexercised. These facts bound every conclusion above.

## 15. Future work and hardware co-design

Software work includes signed provenance and external trust roots; linearizable capture/promotion
coordination; stronger Linux isolation; adapters for real trainer frameworks; controlled
multi-algorithm sample-efficiency experiments; production-like but consented workloads; semantic
near-duplicate detection; calibrated learning-value models; and independent evaluator/red-team review.

Hardware-aware work should begin with measurement, not assumed speedups. Candidate directions include
device/runtime support for cheap copy-on-write KV and recurrent state; epoch-tagged sampler and cache
state; asynchronous, consistency-tagged environment snapshot engines; NIC/storage paths that preserve
content identities during branch materialization; checkpoint-aware GPU memory allocators; low-overhead
dirty-state tracking; and schedulers exposing bounded preemption/preservation costs. Fabric topology
and raw service curves could then parameterize joint serving/learning placement.

No future hardware direction is an implemented result. A valid claim would require real GPU or
accelerator experiments, raw repeated samples, hardware/software manifests, baselines, uncertainty,
serving-SLO evidence, and the same fail-closed provenance used by the CPU reference.

## 16. Conclusion

Helix demonstrates that the transaction substrate for a self-improving agent service can be specified
as an evidence-carrying state machine rather than an opaque train-and-redeploy script. The synthetic
demo does not establish autonomous improvement or generalization. Coordinated state, policy provenance,
typed effects, reward lineage, explicit staleness, serving-hard scheduling, and proof-gated promotion
make failure visible and authority reviewable. The result is intentionally a reference foundation:
useful because its claims stop where its evidence stops.
