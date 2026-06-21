# SLOForge Fabric final adversarial review

Review source: `8154c2895f38f044ce0e0f8cdf48d1907dff0d33`, followed by review fixes
`704bea5` and `4aad2ef`. The working artifact tree was also inspected independently while
the final evidence regeneration was in progress. This review did not edit the README,
completion reports, execution ledger, or generated evidence.

## Hiring verdict

**Yes. An experienced Modal or Baseten infrastructure engineer should recognize this as a
deep systems project, not a broad wrapper around serving libraries.** The strongest evidence
is the typed cross-language physical IR, deterministic resource-contention simulator,
simulator-refined topology compiler, evidence-linked differential diagnosis and
counterfactual replay, and guarded recovery state machine. These components own substantive
algorithms and invariants; they are not YAML veneers over vLLM, SGLang, Dynamo, Modal, or
Truss.

The qualifying sentence matters: this is a deep, production-shaped **reference system whose
multi-node and GPU claims are synthetic or offline on the reviewed machine**, not a
production-proven GPU cluster runtime. The repository represents that boundary honestly.
Presenting the checked synthetic results as real NCCL/RDMA or live-cloud evidence would turn
the verdict negative; neither the code nor current documentation does that.

## Severity-ranked findings

### High

No unresolved high-severity finding.

### Medium findings fixed during the final review

1. **Recruiter-facing documents described the compiler as making zero simulator calls and
   cited stale demo metrics.** The compiler now visibly refines all current Pareto/front-runner
   candidates in the Rust twin (`python/sloforge/fabric/compiler/core.py:1619-1667` and
   `:1738-1829`). Commit `99dc339` corrected the architecture, interview, paper, and resume
   claims and added an artifact-derived documentation test. This was a credibility defect,
   not a compiler implementation defect.

2. **`fabric validate` gated only p95 TTFT and lacked a positive public-CLI test.** Commit
   `8154c28` added observed/predicted p99 TPOT, both prediction intervals, optional hard TTFT
   and TPOT SLOs, retained backward-compatible TTFT aliases, and tested both fail-closed and
   passing paths (`python/sloforge/cli/fabric.py:563-696`,
   `tests/python/test_fabric_demo.py:111-185`). This does not pretend that one command observes
   cost or availability; it closes the two hard latency constraints the compiler accepts.

3. **`autopsy minimize` preserved a hypothesis label even when no matched evidence remained.**
   The public flagship command consequently accepted a zero-event, zero-rank bundle as a
   network diagnosis. Commit `704bea5` now requires the source and minimized top hypothesis
   to remain supported and requires at least one matched event
   (`python/sloforge/cli/autopsy.py:264-305`). The fixture asserts that a rank-straggler
   reproducer retains enough evidence to remain supported
   (`tests/python/test_fabric_cli.py:347-414`). Re-executing the public command reduced the
   flagship from 2,016 events/16 ranks to one event/one rank rather than to an empty bundle.

4. **The Autopsy architecture document still cited an obsolete 480-event/0.95-confidence
   run.** Commit `4aad2ef` bound event count, diagnosis, and formatted confidence to the
   checked flagship artifacts (`docs/autopsy/ARCHITECTURE.md:59-73`,
   `tests/python/test_fabric_docs.py:97-105`).

### Medium boundaries that remain explicit and should not be patched with synthetic claims

1. **GPU/fabric measurement breadth is not hardware-proven.** The measured executor is real,
   bounded, explicitly device-scoped, validates every nccl-tests row, and retains raw captures
   (`python/sloforge/fabric/profiling/runner.py:304-430`). It deliberately supports one local
   NCCL process only (`:323-327`); no NCCL, NVLink, RDMA, DeepEP, NIXL, or multi-host run was
   possible on this Apple Silicon host. Full fixture curves remain synthetic. This is the most
   important operational limitation.

2. **The digital twin's strong internal calibration is not independent hardware validation.**
   The refreshed H2 corpus reported Spearman rank correlation 0.993, 0.0284% median relative
   error, 16.295 ms mean absolute error, 73.33% prediction-interval coverage, and 1.003%
   Pareto-selection regret. Expert-skewed coverage was only 20%, and compiler predictions and
   the twin share synthetic calibration inputs. The compiler's refinement workload is one
   isolated p95-shaped request, while loaded open-loop queueing is a separate validation gate
   (`docs/FABRIC_LIMITATIONS.md:14-22,38-50,106-114`). The public validator fails closed when
   representative trace evidence violates its latency/error gates.

3. **Autopsy capture and recovery execution are reference implementations, not live cluster
   integrations.** Only simulator capture was exercised; it does not yet reconstruct full
   parent/dependency edges (`docs/autopsy/ARCHITECTURE.md:22-34`). Diagnosis confidence is an
   explicit deterministic evidence score, not a posterior
   (`python/sloforge/autopsy/diagnosis.py:264-361`). Counterfactual repair replays the exact
   degraded simulator request and ranks interval-aware outcomes
   (`python/sloforge/autopsy/counterfactual.py:265-361`). Recovery ships a simulated driver;
   disruptive actions require both plan and executor authorization, simulation validation,
   shadow and canary samples, promotion criteria, cooldown, drain, and rollback
   (`python/sloforge/recovery/state_machine.py:300-478`). No production orchestrator was
   mutated.

4. **Runtime adapters are intentionally offline and partially advisory.** Direct vLLM/SGLang
   reject unsupported multi-host rendezvous; generic Kubernetes rejects a multi-node plan
   without an atomic gang contract; Dynamo is the supported generated multi-node mechanism;
   Modal and Truss physical fields are advisory only
   (`python/sloforge/fabric/adapters/compatibility.py:196-256`,
   `docs/fabric/RUNTIME_ADAPTERS.md:71-99`). The adapters fail closed and emit `deployed:
   false`; they were schema/version validated, not admitted by a live operator.

5. **Evaluation breadth is insufficient for generalization.** The 180-trial H1 matrix honestly
   loses to its topology-aligned sequential fixture by 1.56%; Autopsy's 1.0 top-1 result spans
   only 24 deterministic cases from two fault families; both diagnosis-driven and threshold
   recovery restore all modeled cases, with declared action-time/cost inputs
   (`docs/FABRIC_LIMITATIONS.md:106-120`). These are useful regression/effect tests, not a
   production efficacy claim.

6. **WarmPath's strongest alternatives are partly modeled.** Local disk, page cache, checksum,
   host-memory copy, planner, and executor paths are measured locally. Pinned host memory is a
   measured host-copy proxy and an already-ready replica is modeled
   (`python/sloforge/warmpath/evaluation.py:186-197`). The checked WarmPath plan improves on
   local disk by 11.64% but is 0.82% slower than the modeled pinned-host alternative; the
   negative comparison is preserved.

7. **The final source-freshness gate belongs to integration.** During this review the working
   flagship manifest's 140 entries all existed and their SHA-256 digests verified, but its
   checked physical plan still recorded an earlier source commit while the root agent was
   regenerating final evidence. The final integration must commit the regenerated bundle and
   pass `make clean-room-test`; the source code correctly treats this as an evidence-freshness
   condition rather than rewriting provenance.

### Low findings and scope notes

- Pipeline-parallel rank groups are represented, but the IR does not emit layer-to-stage or PP
  send/receive schedules (`docs/FABRIC_LIMITATIONS.md:30-36`).
- The twin is flow/collective-level rather than packet-level or per-layer/per-token
  (`docs/FABRIC_LIMITATIONS.md:38-50`). That is a defensible model boundary, but it limits
  diagnosis of congestion-control and kernel-schedule effects.
- Canonical topology fingerprints include capture timestamps. This safely rejects stale
  evidence but prevents compatibility reuse after equivalent rediscovery
  (`docs/FABRIC_LIMITATIONS.md:87-90`).
- The rank-ordering low-level experiment enforces hash-backed trace evidence and rejects a
  synthetic topology labeled as measured hardware
  (`python/sloforge/fabric/performance/rank_ordering.py:767-844`). Its result remains disabled
  pending hardware measurement, so it is not an unsupported speedup claim.
- Historical audit documents retain numbers and findings from their named source snapshots.
  They should be read as audit history; current recruiter-facing documentation is now guarded
  against artifact drift.

## Depth assessment by subsystem

### Physical IR and protocol

`TopologyGraph`, `ModelGraph`, rank/expert placement, parallel groups, collective and KV plans,
memory/overlap plans, recovery variants, and `PhysicalExecutionPlan` are strict typed models
(`python/sloforge/fabric/ir/models.py:367-885`). Rust mirrors the protocol and validates
physical and recovery invariants (`crates/sloforge-fabric-protocol/src/model.rs:934-1168,
:1460-1570`). Canonical JSON, hashes, v1alpha1 migration, JSON Schemas, golden fixtures, Python
property tests, and Rust/Python round trips are present. Extension data is namespaced rather
than an unbounded untyped escape hatch.

### Topology discovery and networking

Discovery merges structured platform facts with explicit provenance and conflict handling.
Current-host discovery produced 53 nodes and 26 edges and explicitly warned that NVIDIA and
InfiniBand capabilities were unavailable; it did not infer them. Fixtures cover workstation,
NVLink/NVSwitch, multi-NUMA PCIe, InfiniBand, RoCE, MIG, degraded, container-limited, and
conflicting sources. Placement models GPU/NIC/NUMA/rail locality and shared contention domains.
This is credible topology work, while the absence of measured multi-host curves remains the
dominant gap.

### Rust communication twin

The Rust engine owns resource accounting, DAG readiness, fair-share versus exclusive
scheduling, shared contention groups, collective barriers, faults, rank slowdown,
counterfactual resource replacement, cost, uncertainty, and trace emission
(`crates/sloforge-fabric-sim/src/engine.rs:35-631`). Validation bounds operations/events and
rejects invalid dependencies, ranks, resources, curves, and counterfactuals
(`crates/sloforge-fabric-sim/src/validate.rs:7-470`). Analytical tests cover disjoint overlap,
fair-share link conservation, sharing groups, simultaneous faults, counterfactual removal,
resource unavailability, and deadlock. This is the clearest evidence that Fabric is more than
an adapter collection.

### Physical compiler

The compiler enumerates compatible TP/PP/DP/EP/disaggregation combinations, prunes rank and
memory infeasibility, constructs deterministic topology-aware or baseline placement, lowers
collectives/KV/memory/overlap, computes uncertainty/failure exposure, forms a Pareto frontier,
and repeatedly promotes the frontier and leading recovery candidates into the Rust simulator
(`python/sloforge/fabric/compiler/core.py:1670-1829`). Rejection codes and optimizer history
make decisions inspectable. It is a deterministic hierarchical heuristic, not a global
MILP/CP-SAT optimum, and the report does not claim otherwise.

### Autopsy causality

Autopsy verifies plan/topology/workload identity before matching request/stage/rank contexts,
models clock offset/drift/uncertainty, suppresses ambiguous cross-host first-divergence claims,
and derives 27 structured hypotheses without reading fault labels
(`python/sloforge/autopsy/comparison.py:112-260`,
`python/sloforge/autopsy/diagnosis.py:31-118,264-361`). Each candidate records support,
contradiction, rejection, and a non-probabilistic confidence score. The decisive causal step is
counterfactual execution through the physical twin, not natural-language correlation. The new
minimization guard ensures the compact bundle retains matched support.

### Safe recovery

Plans carry diagnosis/physical-plan references, expected improvement/cost/disruption/build
time, compatibility, traffic migration, promotion, rollback, abort, and evidence. The state
machine is bounded, idempotent, snapshot-restorable, and fail-closed for external mutation.
Shadow/canary minimum samples, error and SLO criteria, started-stream preservation, drain,
cooldown, abort, rollback, and operator-required states have direct tests. The gap is the
deliberate absence of an authorized production action driver, not missing safety logic.

### ForgeCI, WarmPath, and UI

ForgeCI implements warmup separation, repeated subprocess trials, robust summaries, bootstrap
intervals, practical/noise thresholds, multiple-metric correction, inconclusive retry, actual
Git bisection, minimization, and an issue bundle
(`python/sloforge/forgeci/statistics.py:39-174`,
`python/sloforge/forgeci/bisect.py:54-222`). Its deterministic fixture found the intentional
first bad commit across three evaluated commits with a corrected approximately 11.47%-12.53%
effect interval.

WarmPath owns an artifact DAG, compatibility/storage tiers, measured local startup stages,
cold-start simulation, deterministic placement search, eviction behavior, and a hash-verifying
local executor (`python/sloforge/warmpath/planner.py:129-390`,
`python/sloforge/warmpath/executor.py:210-359`).

The UI consumes evidence rather than copied numbers. It verifies manifest SHA-256 values and
cross-artifact plan, topology, diagnosis, recovery, timeline, simulation, and SLO identities
before rendering (`ui/src/fabric-parser.ts:386-625,668-768`). Tampering, missing/duplicate
artifacts, oversized payloads, and identity mismatches have tests.

## Commands exercised in this review

- `make fabric-check`: passed; Ruff format/lint, mypy over 118 source files, 292 Fabric/Autopsy/
  ForgeCI/WarmPath Python tests, Rust format/clippy with warnings denied, 10 protocol tests, and
  all Fabric simulator unit/analytical/property/two-node/validation tests.
- `sloforge fabric discover --output ...`: passed on Apple Silicon; 53 nodes, 26 edges, explicit
  no-GPU/no-InfiniBand warnings.
- `sloforge fabric benchmark ... --measured-host-memory`: passed and emitted raw measured
  host-memory evidence; no transport fallback was reported.
- `sloforge fabric model inspect --model synthetic`: passed with an explicit synthetic source;
  an invalid synthetic model identifier failed closed.
- `sloforge fabric compile ...`: passed in a standalone public workflow; 175 candidates, four
  Pareto candidates, 174 rejected alternatives, ten simulator calls, and deterministic
  simulator work units were emitted.
- `sloforge fabric simulate ...`: passed on the representative trace and emitted 588 physical
  operations.
- `sloforge fabric validate ...`: intentionally exited nonzero on material TTFT prediction
  error and interval miss. The command wrote a typed failure artifact instead of masking the
  result.
- `sloforge autopsy compare`, `diagnose`, `replay`, `minimize`, and `report`: passed on the
  flagship bundle. Seven counterfactuals were evaluated; the minimizer now retains matched
  diagnosis evidence.
- `sloforge recovery plan` and `recovery apply --mode shadow-canary`: passed; the local execution
  reached `COMPLETED` with zero external mutation.
- UI `typecheck`, lint, 28 tests, and production build: passed.
- Review-fix acceptance: Ruff and mypy passed; three focused Autopsy tests and five
  artifact-derived documentation tests passed.

## Artifact integrity and checked outcomes

At the reviewed flagship snapshot, all 140 manifest entries existed and independently matched
their recorded SHA-256 digest. The deterministic synthetic run reported p95 TTFT of 1,152.714
ms healthy, 7,045.952 ms under simultaneous network-path and rank-specific slowdown faults,
and 1,152.714 ms restored against a 1,267.985 ms SLO. Autopsy ranked network bandwidth
degradation first at evidence score 0.834, evaluated seven repairs, selected
`remove-both-faults`, and the guarded recovery finished `COMPLETED`. Twelve requests also
traversed the real local Rust gateway; physical cluster execution remained simulated.

The checked evaluation preserves negative results: topology-aware hierarchy reached 626.068 ms,
1.56% slower than the 616.428 ms topology-aligned sequential baseline; H2 interval coverage was
73.33% overall but only 20% for expert-skewed traffic and is internally calibrated rather than
hardware-independent;
diagnosis accuracy covers only two fixture fault families; threshold recovery had a shorter
declared action time; and WarmPath lost by 0.82% to the modeled pinned-host proxy. This evidence
discipline is a positive hiring signal.

## Final assessment

The architecture is coherent: Python compiles and analyzes; Rust owns bounded deterministic
physical execution; canonical JSON is the stable subprocess boundary; runtime/cloud adapters
lower plans without pretending to enforce unsupported placement; every diagnosis/recovery is
content-addressed back to evidence. The project demonstrates credible distributed inference,
topology, communication, simulation, causal debugging, recovery-safety, statistics, and systems
programming depth.

Acceptance from this adversarial review is **PASS WITH EXPLICIT HARDWARE BOUNDARIES**. There are
no remaining high-severity source findings and the reasonable medium defects found here were
fixed and tested. Final completion still requires the root integration run to commit freshly
generated artifacts and pass the clean-room/full-suite gates; that is intentionally not claimed
by this source review.
