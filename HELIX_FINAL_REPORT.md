# SLOForge Helix final implementation report

Verified on 2026-08-03 (America/Los_Angeles).

## Release identity and claim boundary

- Baseline commit: `a3366807e879cf17615021e32606fbf77216235c`
- Immutable baseline tag: `sloforge-helix-baseline-a336680`
- Final verified implementation source commit:
  `de89932164a5ecc8a73e5fc2c309796a34f4b8b4`
- Main implementation commit: `5450fadd482be4804c420b4d960554d7a765fc58`
- Validation class: `deterministic-local-cpu-synthetic`
- Host: Apple Silicon macOS, 12 logical CPUs, 24 GiB RAM

Helix is implemented as a transactional local reference substrate integrated with
the existing SLOForge, Fabric, Autopsy, Genesis, Continuum, ForgeCI, and WarmPath
systems. The retained experiment is a deterministic synthetic coding-agent
workload. It is not production traffic, a GPU result, a distributed-consensus
result, or evidence of general causal identification. Optional hardware and
external-runtime paths are described separately below and are never counted as
exercised results.

## Architecture

Helix adds a versioned state machine spanning capture, branching, rollout,
reward, credit, training, evaluation, and deployment:

1. Canonical Python and Rust IRs bind `PolicyEpoch`,
   `EnvironmentStateCapsule`, `BranchPoint`, `TrajectoryCapsule`,
   `BranchGroup`, `RewardEvidence`, `CreditAssignmentEvidence`,
   `TrainingBatchManifest`, `LearningTransaction`,
   `PolicyPromotionCapsule`, `StateReuseReport`, `StalenessReport`, and
   `BranchWorkloadTrace`. Thirteen JSON schemas, golden fixtures, stable
   canonical hashes, migrations, and cross-language conformance tests protect
   the subprocess JSON boundary.
2. A coordinated, fenced capture state machine quiesces model and environment
   state at one action boundary, freezes the effect ledger, verifies byte-backed
   model/environment/effect artifacts and four watermarks, publishes atomically,
   and resumes the source with bounded retries and crash recovery.
3. The environment backend stores tenant-scoped content in a hash-addressed
   store and restores per-branch writable overlays. Branch workspaces, service
   snapshots, SQLite state, resource limits, cleanup, retention tombstones, and
   secret redaction are explicit. This is deterministic reconstruction, not a
   claim of arbitrary process-memory portability.
4. The six-class effect ledger rejects speculative irreversible or unknown
   effects, constrains reads to captured/stable observations, isolates
   idempotent writes, and requires compensation evidence where applicable.
5. Continuum owns model-state reuse. Same-policy branches receive new owner
   epochs and exact content-addressed state forks; controlled RNG mutation is
   explicit. Cross-policy branches require a compatibility report and either
   documented recomputation/conversion or rejection. Every source state
   component receives exactly one disposition.
6. The trajectory path records per-token and per-action policy epoch, behavior
   probability, RNG counter, tool/effect provenance, event-chain hashes,
   rewards, staleness, and parent state. Strict trajectories are the default;
   mixed policies require explicit segment boundaries and an eligibility or
   off-policy disposition.
7. Python owns rollout, environment orchestration, reward, credit, dataset,
   training, evaluation, and scheduling. Durable local coordinators and trusted
   validation/state-machine components use SQLite transactions. Rust owns the
   canonical shared IR and validation boundary without introducing another RPC
   system.

The code is under `python/sloforge/helix/`, `crates/sloforge-helix-ir/`,
`schemas/helix/`, `scenarios/helix/`, and the focused Python/Rust test suites.
The architecture and trust boundary
are documented in `docs/helix/ARCHITECTURE.md` and
`docs/helix/TRUST_MODEL.md`.

## Exact-state branching and replay

The retained seed-41 run captured one consistent coding-agent decision boundary:

| Evidence | Value |
|---|---|
| BranchPoint | `2765e28ed1ba4a3ca480db7a8f1fa3ab451d5db5f39dbfa117fbe8d4b29594f8` |
| Continuum capsule | `060daf197efa6cf774c1f8b4f45a601a843026aa71e4410acf199ff22937510d` |
| Environment capsule | `9ae27338c11c2ac8ddc9afc55709936ba7a308f4aad1926d4e2df309a594e2e1` |
| Capture artifact bytes verified | true |
| Source resumed | true |

Four siblings ran from the common boundary: different RNG, a forced alternative
action, an alternative tool, and a verifier-assisted action. Three reused all
declared model components exactly. The RNG branch reused the immutable prefix
but replaced the sampler state before the first branch token. Each branch used
an isolated environment overlay and emitted a complete `StateReuseReport`.

The replay experiment compared transcript re-feed, environment-only,
model-only, and joint restoration. Joint restoration was the only scope that
verified exact captured identity; it matched model state, environment state,
RNG-bound trace identity, and committed event evidence. The three non-joint
scopes explicitly reported missing identity dimensions and made no state-
equivalence claim. This result repeated for seeds 41, 73, and 113: 3/3 joint
exact-identity checks passed and 0 non-joint cases did. H1 therefore remains
**partial**: the local restoration paths and identity contracts were exercised,
but readiness, prefill savings, and downstream divergence under a real
nondeterministic model were not measured. H2 is also **partial** because logical
CAS/overlay isolation and cleanup were exercised without reflink/page-fault or
GPU-memory amplification measurements.

Exact, causal, and semantic replay validators, divergence localization, and
witness-preserving trajectory minimization are implemented and tested. Causal
replay means preservation of declared event semantics and topology; it is not a
claim that the replay engine proved causation.

## Reward, credit, and training

The champion selected a plausible but incorrect edit and failed an executed
hidden behavior check. Sibling rewards came from isolated deterministic test
execution, not authored reward rows:

| Branch | Reward |
|---|---:|
| different RNG | 0.0 |
| forced alternative | 2.0 |
| alternative tool | 2.0 |
| verifier-assisted | 2.0 |

Training and holdout evaluators were separately content-addressed
(`790696c6…` and `9ab4cb6d…`), all admitted evaluator digests were trusted, and
raw stdout/stderr passed bounded sanitization. Reward components, confidence,
source version, trajectory, policy epoch, environment, and verifier artifacts
remain available independently of the aggregate score.

Branch comparison found the first controlled divergence and emitted
branch-relative credit evidence
`60f8124a67eb17bc558c382e1bb592ecbb7f16008598c0a32d54a1770f38359d`.
Observational branches receive zero optimization credit. The evidence states
its controlled-sibling assumptions and does not claim perfect causal
identification. The provenance-complete training batch is
`1634dddd376b04d420194d02895125b51d55fe7b7fe1999aae6ac9337f9b947c`
with content hash
`bd956cdfa0628291d25b1b4ce531446af9d9dad870e6006d75fcf271e61b43df`.

The flagship exercised deterministic branch-relative optimization. Champion
success was 0.625. A deliberately undertrained but genuinely evaluated
candidate reached 0.6667 and was rejected below the predeclared 0.75 quality
threshold. The corrected challenger reached 0.7917, an absolute paired
improvement of 0.1667 on this synthetic suite, and published checkpoint
`91d75141151e502f58f103012f9fbb950c8c64907e8e069a4faa88523ee8aadf`.
The required successful-branch distillation, sibling preference, group-relative,
and branch-relative objectives and their numerical/staleness tests are present;
only branch-relative optimization was exercised in the retained flagship, so
no comparative sample-efficiency claim is made.

A staged public-CLI flow independently built four samples, published batch
`e97731dedbcc64e1945e02d00740f78a44fe028d228d3faa1c301506d3680e98`,
ran eight branch-relative steps, and emitted candidate `coding-agent-cli@1`
with checkpoint
`7bc19ef5d030137c9c390f662c204835bac139b2169ae0637f417c292df4967a`.

## Asynchronous and policy-version semantics

Rollout workers expose bounded load/restore/run/pause/checkpoint/migrate/resume
operations and publish policy epoch, behavior log probability, resource use,
and failure evidence. Strict single-policy mode rejects policy changes.
Segmented mode permits transitions only at declared action boundaries and
records token/action ranges, compatibility action, recomputation, staleness,
eligibility, and correction. Staleness supports hard rejection, bounded
acceptance, recomputation, clipped importance weighting, truncation,
resampling, priority reduction, and evaluation-only disposition. Missing
behavior probabilities and undeclared policy mixing fail closed.

The reference rollout worker is exercised. PyTorch, Genesis, vLLM, and SGLang
adapters advertise capabilities and reject exact-state reuse when no portable
state export exists. Queues and subprocesses are bounded and timed.

## Resource compiler

The learning-aware compiler models serving, capture, environment, rollout,
reward/verifier, training, evaluation, shadow/canary, checkpoint, and migration
work over GPU, CPU, memory, storage, and network capacity. Serving reservation,
latency/queue SLOs, monetary budget, tenant/privacy policy, effect policy, and
staleness are hard constraints. Decisions record expected validated value,
failure coverage, diversity, freshness, state reuse, cost, preemption, and
prediction provenance. Capacity lending cannot consume the serving reserve;
reclamation selects restart, checkpoint, or Continuum preservation only when
the declared economics allow it.

Dedicated, static, utilization-threshold, FIFO, and Helix value-aware policies
were compiled for each of three seeds. On the supplied workload every policy
was feasible and tied: two completed work items, predicted learning value 16.0,
41,300 cost microunits, zero preemptions, and zero lost-work ticks. Those are
scenario predictions with raw provenance, not observed policy gains. H5 is
therefore partial and there is no claimed scheduling win; H6 was not exercised
because no preservation path was selected in the 15 policy/seed runs.

## Learning transaction, promotion, and rollback

`LearningTransaction` and its capture, training, and promotion sub-transactions
persist every transition, input artifact, policy epoch, cost/resource entry,
and failure state. Registries use transaction identity, tenant checks, fencing,
and atomic pointer updates to make retries idempotent and prevent duplicate or
partial champion publication.

The trusted promotion gate independently rehashed eight local gate artifacts
covering lineage, reward integrity, quality, safety, serving, compatibility,
shadow, and canary evidence. The retained capsule digest is
`f01301ff2423dce908141616e09cc276af2dbbd2d893e4d720fd89a02e7f1e31`.
The corrected challenger passed shadow and canary, became the route for a new
eligible session, and preserved its previous champion as rollback parent.
Continuum classified transition as recomputation-assisted. One incompatible
active session intentionally remained pinned to `coding-agent@0` while new
traffic used `coding-agent@1`. A separate injected regression rolled routing
back atomically to `coding-agent@0`; the final transaction state is
`rolled_back` and the failed evidence remains in the lineage.

The flagship journal contains 24 actual transition events. Its lineage export
contains 15 nodes and 17 edges connecting the production-like failure,
BranchPoint, four trajectories, rewards, credit, batch, candidates,
evaluations, promotion, active-session disposition, and rollback. H7 is
supported within this local transactional scope, not for a distributed or
multi-region control plane.

## Fault tolerance and security

The deterministic fault campaign exercised 16 Helix fault classes across
capture, branching, rollout, reward, training, evaluation, promotion, and
resource scheduling. Each record carries a ground-truth label, activation
interval, transaction stage, expected response, and actual response.
Focused tests also cover coordinator restart, fencing, corrupt snapshots and
checkpoints, illegal effects, missing log probabilities, staleness, reward
tampering, quality/serving/compatibility rejection, canary rollback, active-
session pinning, and artifact preservation.

Production capture, external side effects, external APIs, cloud deployment,
GPU use, and live promotion remain disabled without their explicit opt-ins and
budgets. Environment, Continuum, reward, batch, registry, and promotion
boundaries are tenant-scoped. Hostile-repository preflight, bounded traversal,
secret/PII redaction evidence, trusted evaluator digests, immutable hidden-test
namespace, retention receipts, checkpoint rehashing, and eight-artifact
promotion validation fail closed. SHA-256 provides integrity, not issuer
authentication; a distributed deployment still needs an authenticated registry
and linearizable coordinator.

The fresh multidisciplinary adversarial review found no unresolved high-
severity or reasonable medium-severity issue inside the exercised local
boundary. Its hiring assessment classified Helix as a transactional learning
substrate rather than a generic RL wrapper, while declining a Principal causal-
inference or empirical-RL claim because the required external empirical
evidence is absent.

## Commands and acceptance results

Baseline and non-regression commands:

- `make check`
- `make demo`
- `make fabric-check` and `make fabric-demo`
- `make autopsy-demo`
- `make genesis-check` and `make genesis-zero-day-demo`
- `make continuum-check`, `make continuum-demo`, and
  `make continuum-fault-demo`
- `make forgeci-demo`
- `make warmpath-demo`

All passed. The final `make check` result was 1,082 Python tests passed with six
declared optional no-Torch/GPU skips, Ruff format/lint over 531 files, strict
mypy over 401 sources, all workspace Rust format, warning-denied Clippy, unit,
integration, and doc tests, and 37 UI tests with one fixture skip plus a
production build. The six Genesis numerical-overflow warnings are known test
inputs and did not represent test failures.

Helix commands:

- `make helix-check`: 155 Python tests passed; Ruff and strict mypy passed over
  78 Helix sources; Rust format, warning-denied Clippy, unit tests, and five
  cross-language conformance tests passed.
- `make helix-demo`
- `make helix-branch-demo` (26 focused tests)
- `make helix-replay-demo` (5 focused tests)
- `make helix-training-demo` (22 focused tests)
- `make helix-resource-demo`
- `make helix-promotion-demo` (11 focused tests)
- `make helix-fault-demo` (30 focused tests plus the 16-fault campaign)
- `make helix-evaluation` (seeds 41, 73, and 113)
- `make helix-clean-room-test`

Every listed Helix target passed. The clean archive validated revision
`de89932164a5ecc8a73e5fc2c309796a34f4b8b4`, source tree
`f94e9e0c1df99f1aa1aec34afa2fbc500bea4a15`, and log hash
`e068745302940de358da34bffc97b3d89c44fbfcc54a7487fc5d6b3f201f26df`.
`make helix-docker-smoke` ran its capability check and recorded
`unexercised: Docker daemon unavailable`; it did not claim a pass. No Helix
worker, environment branch, training process, canary, cloud resource, or paid
job remained active after validation.

## Evaluation results

The evaluation ID is
`95857f1fa8a42c9b6b193d9b429c9241f4aa275686687390f8302cf77e5e0e62`.
It seals 22 top-level raw references and retains the complete 452-file raw
evaluation tree. Student-t intervals summarize deterministic seed sensitivity,
not independent population sampling. The three runs were identical, so the
reported intervals collapse to the observed values:

| Metric | Mean | 95% sensitivity interval |
|---|---:|---:|
| Champion success | 0.6250 | [0.6250, 0.6250] |
| Rejected candidate success | 0.6667 | [0.6667, 0.6667] |
| Corrected challenger success | 0.7917 | [0.7917, 0.7917] |
| Paired absolute delta | +0.1667 | [+0.1667, +0.1667] |

Across the three runs, joint exact identity passed three times, a real
below-threshold candidate was rejected three times, an incompatible session
was pinned three times, and rollback completed three times. The measured local
categorical-policy decision latency was 0.008623 ms for seed 41; this is not an
LLM serving benchmark.

Hypothesis disposition is intentionally conservative: H1, H2, H5, and H10 are
partial; H7 is supported within the local reference scope; H3, H4, H6, H8, and
H9 were not exercised as empirical comparisons. No missing experiment is
reported as a win.

## Hardware-backed and external adapter status

The host had no installed PyTorch, PEFT, vLLM, SGLang, NVIDIA device, cloud
credentials, or authorized budget. No GPU, multi-node, cloud, paid API, or live
production path was executed and no hardware measurements are claimed.

The tiny PyTorch trainer, PEFT/LoRA adapter, Genesis runtime adapter, vLLM and
SGLang capability adapters, Docker/Kubernetes/Modal offline environment plans,
and Continuum compatibility hooks are implemented behind capability/version
checks. PyTorch and PEFT checkpoint loaders require validated weights and
provenance; rollout adapters reject or recompute model-derived state when no
portable export exists. Current upstream versions and official contracts are
recorded in `docs/helix/EXTERNAL_API_VALIDATION.md`. These paths are
**implemented but unexercised** on this host, except for their deterministic
fixtures and fail-closed capability probes.

## Known limitations and risks

- The coding repository, categorical policy, candidate actions, failure-seed
  search, training schedule, and acceptance threshold are pre-authored and
  disclosed. The improvement is real within that fixture but does not establish
  external generalization or autonomous continual learning.
- Branch-relative evidence is a controlled sibling intervention with explicit
  assumptions, not universal causal identification. A comparative multi-
  algorithm learning-curve study was not run.
- Replay comparison validates captured identities and event evidence. It does
  not by itself attest that an arbitrary external engine restored memory bytes.
- The resource compiler is forecast-based and discrete-tick. Its supplied
  workload produced a tie, and hardware preemption/migration timing was not
  measured.
- Environment capsules reconstruct declared state and services; they do not
  capture arbitrary kernel memory or undeclared external systems. Local
  filesystem isolation is not a hostile multi-tenant kernel boundary.
- SQLite supplies durable local transactions, not distributed consensus.
  Content hashes need an independently authenticated digest registry to prove
  issuer identity.
- H3, H4, H6, H8, and H9 need controlled empirical campaigns; the current
  reports label them not exercised rather than inferring outcomes from unit
  tests.
- Docker smoke was unavailable because the daemon was not running. GPU and
  external runtime paths remain unexercised.

## Evidence and documentation index

- Baseline record: `artifacts/helix/baseline/record.json`
- Flagship summary: `artifacts/helix/demo/seed-41/summary.json`
- Flagship raw artifacts: `artifacts/helix/demo/seed-41/`
- Evaluation: `artifacts/helix/evaluation/reference/evaluation.json`
- Evaluation raw tree: `artifacts/helix/evaluation/reference/raw/`
- Resource demo: `artifacts/helix/resource-demo/`
- Fault campaign: `artifacts/helix/faults/cpu-matrix/`
- Staged CLI workflow: `artifacts/helix/cli-workflow/de89932/`
- Clean-room result: `artifacts/helix/clean-room/result.json`
- Docker disposition: `artifacts/helix/docker/status.json`
- Evaluation reports: `reports/helix-evaluation.md` and
  `reports/helix-evaluation.html`
- Focused reports: `reports/helix-replay.md`, `reports/helix-credit.md`,
  `reports/helix-resource-compiler.md`, `reports/helix-promotion.md`, and
  `reports/helix-continual-learning.md`
- Architecture and component documentation: `docs/helix/`
- Related work: `docs/HELIX_RELATED_WORK.md`
- ADRs: `docs/adr/` Helix decisions
- Paper-style report: `paper/helix/REPORT.md`
- Reproduction instructions: `docs/helix/REPRODUCIBILITY.md`
- Demo script: `docs/helix/DEMO_SCRIPT.md`
- Adversarial review: `docs/helix/FINAL_ADVERSARIAL_REVIEW.md`
- Security review: `docs/helix/SECURITY_REVIEW_FINDINGS.md`

All numeric claims in this report are sourced from the retained JSON artifacts.
No resume claim or hardware result is populated without raw provenance.
