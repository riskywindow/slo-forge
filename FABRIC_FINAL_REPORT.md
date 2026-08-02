# SLOForge Fabric final report

Report state: **release-candidate draft; final exact-HEAD verification is pending**.

This draft was assembled from source revision
`2781ce9d12c189c246a61bf5f2783664d9b05c7b` and the evidence present in the
working tree on 2026-08-02. Creating this report advances the repository beyond
that revision, so `2781ce9` is an inspected source snapshot, not the final
release commit. The final source commit, regenerated artifact publication
commit, and clean-archive result must replace the pending entries below before
this document is marked final.

The current Fabric evaluation records source commit
`704bea5ba9a0e0a794589c8ab3df4f38cde3bdec`, while the current flagship
physical plan records `001d53b50bcab2b1a0382c17f0e0fff19ce3add4`.
Consequently, the numerical results in this draft are evidence-backed snapshots
but are **not yet source-fresh release evidence**. Final regeneration and
atomic publication remain release blockers.

## Baseline

- Baseline commit:
  `c67e082a13fc6882d7849a862c6667a787b43a72`.
- Annotated tag: `sloforge-fabric-baseline-c67e082`; the dereferenced tag
  resolves exactly to the baseline commit.
- Baseline host: Apple M4 Pro, macOS 15.6.1 arm64, 12 logical CPUs, 24 GiB RAM.
  No NVIDIA GPU, CUDA toolkit, NCCL tests binary, RDMA device, privileged-probe
  authorization, cloud credentials, or GPU budget was available.
- Baseline validation recorded in `EXECUTION_LEDGER.md`: locked bootstrap
  passed; `make check` passed with 67 Python tests plus 3 expected no-Torch
  skips, 62 Rust tests, 11 UI tests and a production build; `make demo` passed
  with 120 live gateway requests and 120 simulated requests.
- Existing CLI, DeploymentPlan, CPU demo, gateway, optimizer, controller,
  exporters, and evidence behavior were retained. Fabric composes with the
  existing logical plan instead of replacing it.

## Architecture

SLOForge Fabric uses the repository's existing ownership boundary:

1. Python owns discovery orchestration, profiling, model-graph inspection,
   physical-plan compilation, Autopsy analysis, recovery planning, ForgeCI,
   WarmPath, deployment lowering, reports, and the CLI.
2. Rust owns the canonical Fabric wire models and the bounded deterministic
   physical digital twin. The existing Rust gateway remains the streaming data
   plane.
3. Canonical, versioned JSON over bounded subprocess stdin/stdout is the
   Rust/Python boundary. HTTP/SSE remains confined to the live request path.
4. Every physical plan references the logical DeploymentPlan, topology and
   profile fingerprints, prediction intervals, optimizer history, rejected
   alternatives, recovery variants, and evidence hashes.

The implemented data/control path is:

```text
DeploymentPlan + ModelGraph + TopologyGraph + FabricProfile + SLO/workload
            | physical compiler + bounded Rust-twin refinement
            v
PhysicalExecutionPlan + recovery variants + rejected alternatives
            | simulate / validate / deploy-adapter lowering
            v
runtime evidence -> Autopsy differential diagnosis -> counterfactual replay
            -> RecoveryPlan -> simulation -> shadow -> canary -> promotion/rollback
```

## Implementation summary

### Physical IR and topology

- `python/sloforge/fabric/ir` and
  `crates/sloforge-fabric-protocol` implement typed `TopologyGraph`,
  `ModelGraph`, parallel/rank/expert placement, collective, KV-transfer,
  memory, overlap, recovery-variant, and `PhysicalExecutionPlan` models.
- `schemas/fabric` contains JSON Schemas for the topology, model graph, fabric
  profile, physical plan, and recovery plan. Canonical JSON, stable hashes,
  the v1alpha1 migration, golden fixtures, Python property tests, and
  Rust/Python round trips are implemented.
- `python/sloforge/fabric/topology` normalizes structured local discovery with
  provenance and explicit conflicts/unknowns. Fixtures cover a workstation,
  NVLink/NVSwitch, multi-NUMA PCIe, InfiniBand, RoCE, MIG, degraded,
  container-limited, and conflicting-source topologies.

### Fabric profiling and digital twin

- `python/sloforge/fabric/profiling` emits raw samples, warmups, robust
  summaries, confidence intervals, placement, environment, timeout/failure,
  and hashes. The measured runner uses bounded process groups, bounded output,
  explicit device inventory, timeouts, and strict nccl-tests row validation;
  it never silently substitutes CPU or another transport.
- `crates/sloforge-fabric-sim` models operation DAGs, queues, calibrated compute
  and transfer curves, exclusive and fair-share resources, shared contention
  domains, barriers, collectives, KV transfer, startup, rank skew, link and
  worker failures, cost, uncertainty, counterfactual resource replacement, and
  trace export. It rejects invalid dependencies, resources, curves, ranks,
  faults, and deadlocks in otherwise valid plans.
- The simulator is intentionally a flow/collective-level model. It is neither a
  packet simulator nor a reconstruction of every model-layer kernel.

### Physical compiler

- `python/sloforge/fabric/compiler` enumerates and prunes compatible
  TP/PP/DP/EP and prefill/decode-disaggregation combinations, performs memory
  and reachability checks, assigns rank/GPU/NUMA/NIC/rail groups, lowers
  collectives and KV routes, selects overlap and recovery variants, computes
  robust objectives and a Pareto frontier, and explains rejection codes.
- Exhaustive/tiny, random, sequential, topology-unaware, greedy topology-aware,
  hierarchical, and robust-failure strategies are present.
- Current Pareto/front-runner candidates and recovery alternatives are refined
  through bounded Rust-twin calls. Calls, deterministic work units, solver
  timing, and intermediate decisions are recorded.

### Autopsy and recovery

- `python/sloforge/autopsy` provides the versioned event model, bounded
  lowest-RTT clock alignment with drift and uncertainty, matched healthy versus
  degraded comparison, 27 structured causal hypotheses, supporting and
  contradicting evidence, deterministic confidence scores, counterfactual
  replay, hypothesis rejection, and supported-evidence minimization.
- Diagnosis does not consume fault labels and does not depend on an LLM.
  Counterfactual replay runs modified causes through the physical twin and
  records selected and rejected repairs.
- `python/sloforge/recovery` emits typed recovery proposals and runs a bounded,
  idempotent, snapshot-restorable state machine with simulation validation,
  build, shadow, canary, promotion, graceful drain, abort, rollback, cooldown,
  and operator-required states. External mutation is disabled without both
  plan and executor authorization.

### Tier 2 and deployment surfaces

- `python/sloforge/forgeci` implements warmup-separated repeated trials,
  bootstrap intervals, practical/noise thresholds, multiple-metric correction,
  inconclusive retry, Git bisection, minimization, and an upstream issue bundle.
- `python/sloforge/warmpath` implements an artifact DAG, compatibility and
  storage tiers, measured local startup stages, cold-start simulation,
  deterministic placement search, eviction experiments, and a hash-verifying
  local executor.
- `python/sloforge/fabric/adapters` lowers physical plans to local, Docker,
  Kubernetes, NVIDIA Dynamo, vLLM, SGLang, Modal, and Truss representations.
  Unsupported placement or runtime semantics fail closed. The Fabric CLI now
  exposes this lowering; external deployment remains offline by default.
- `ui` parses the real artifact bundle, validates manifest hashes and
  cross-artifact identities, and renders topology, placement, communication,
  diagnosis, counterfactual, recovery, and SLO views.
- `benchmarks/fabric/rank_ordering` contains the trace-justified low-level
  rank-ordering experiment with correctness gates and raw synthetic trials. Its
  integration decision is `measure_on_hardware`; it remains disabled by
  default and is not a hardware speedup claim.

## Tests and demonstrations already executed

These results are tied to the named snapshots. They demonstrate implemented
behavior but do not replace the pending final exact-HEAD gates.

| Snapshot | Command or audit | Recorded result |
|---|---|---|
| Baseline `c67e082` | `make bootstrap` | Passed with locked Python, Cargo, and npm dependencies. |
| Baseline `c67e082` | `make check` | Passed: 67 Python tests, 3 expected no-Torch skips, 62 Rust tests, 11 UI tests, and production UI build. |
| Baseline `c67e082` | `make demo` | Passed: 120 live gateway and 120 simulated requests. |
| Integrated source `606680f`, after regenerating its worktree evidence | `make fabric-demo` | Passed; generated canonical trace, 588-operation simulations, seven counterfactuals, telemetry, runtime evidence, and derived report. |
| Same regenerated snapshot | `make fabric-check` | Passed: Ruff, strict mypy over 118 source files, 280 Python tests, warning-denied Fabric Rust Clippy, and 31 Fabric Rust tests. |
| Same regenerated snapshot | `make check` | Passed: 350 Python tests plus 3 expected GPU/Torch skips, all 98 Rust tests, and UI typecheck/lint/28 tests/build. |
| Same regenerated snapshot | public `fabric simulate` | Passed: 588 operations, 1,523 processed events, and 1,775.586 ms makespan. |
| Same regenerated snapshot | standalone `autopsy replay` | Passed: hash-verified bundle, seven scenarios, `remove-both-faults` selected, six alternatives rejected. |
| Same regenerated snapshot | public `fabric validate` | Expected fail-closed exit 1 with a structured artifact for 14.430933 relative TTFT error and interval miss. |
| Same clean-room review | `make package` and independent wheel smoke | Passed: wheel/sdist built; isolated wheel ran help, seeded trace, local topology discovery, and CPU hardware probe. |
| Same clean-room review | lock/audit checks | `uv lock --check`, offline locked Cargo metadata, and npm high-severity audit passed; npm reported zero known vulnerabilities. |
| Source through `4aad2ef` in `FABRIC_FINAL_ADVERSARIAL_REVIEW.md` | `make fabric-check` | Passed: 280 Fabric/Autopsy/ForgeCI/WarmPath Python tests plus Fabric protocol/simulator format, warning-denied Clippy, and tests. |
| Same adversarial review | public Fabric workflow | Discover, measured host-memory benchmark, model inspection, compile, simulate, intentional validation failure, Autopsy compare/diagnose/replay/minimize/report, and local shadow-canary recovery passed. The standalone compile recorded 175 candidates, 4 Pareto candidates, 174 rejected alternatives, and 10 simulator calls. |
| Same adversarial review | UI | Typecheck, lint, 28 tests, and production build passed. |
| Review fixes `704bea5` and `4aad2ef` | focused checks | Ruff and mypy passed; 3 focused Autopsy tests and 5 artifact-derived documentation tests passed. |

The earlier clean-room archive deliberately exposed a publication defect: a
literal archive of `606680f` failed three artifact-contract tests before demo
generation, then passed after regeneration. This is why source-fresh artifact
publication and a subsequent archive test remain mandatory rather than being
inferred from worktree success.

## Current evaluation snapshot

Canonical input: `artifacts/fabric/evaluation/result.json`, SHA-256
`8a0c6292ddbb88fb7353602b7fd996436ff16a00f00aa85a2fd12b62ee840c9d`.
It is marked `synthetic_cpu`, explicitly records hardware-backed validation as
`not_exercised_no_compatible_hardware`, and records source `704bea5`; therefore
all values below are provisional until final source-fresh regeneration.

The matrix contains 180 physical-plan trials: five methods, four two-node
topology fixtures, three workload regimes, and seeds 13, 29, and 47. Each method
summary contains 36 trials.

### H1: topology-aware compilation

| Method | Median p95 TTFT (ms) | Median p99 TPOT (ms) | Median SLO attainment | Median goodput (token/s) | Median communication time (ms) |
|---|---:|---:|---:|---:|---:|
| Random placement | 2,831.615 | 1.798 | 0.4375 | 58.560 | 12,734.204 |
| Sequential placement | 616.428 | 1.260 | 1.0 | 590.280 | 1,345.048 |
| Topology-unaware optimizer | 1,015.728 | 2.501 | 1.0 | 354.156 | 193.218 |
| Topology-aware greedy | 626.068 | 1.259 | 1.0 | 582.455 | 1,441.225 |
| Hierarchical compiler | 626.068 | 1.259 | 1.0 | 582.455 | 1,441.225 |

The hierarchical compiler beat random placement by 77.89% and the
topology-unaware optimizer by 38.36% on median p95 TTFT, but was 1.56% slower
than the already topology-aligned sequential fixture. H1 is therefore **not
established against the strongest fixture baseline**; the negative result is
retained.

### H2: communication-aware twin

- 180 synthetic twin comparisons.
- Median relative error: 0.0284%.
- Mean absolute error: 16.295 ms.
- Spearman rank correlation: 0.9931.
- Prediction-interval coverage: 73.33%.
- Pareto-selection regret: 1.0033% (5.488 ms).
- Median separately reported loaded-workload queueing delta: 802.787 ms.

These are internal synthetic results: compiler predictions and the simulator
share fixture calibration. They do not establish real-hardware accuracy.
Aggregate accuracy also hides a material weak regime: the expert-skewed group
has 18.44% median relative error and only 20% interval coverage, whereas the
long-context and mixed-bursty groups report full interval coverage. The public
validator remains a separate fail-closed loaded-trace gate.

### H3: Autopsy

The current evaluation contains 24 deterministic cases over two injected fault
families. Top-1 and top-3 accuracy are both 1.0, median deterministic evidence
confidence is 0.925, median diagnosis latency is 6.035 ms, and median
counterfactual residual is 0 ms. The confidence is an evidence score, not a
calibrated posterior probability. The narrow two-family fixture set does not
establish general production diagnosis accuracy.

### H4: recovery

The evaluation contains 120 recovery trials. Diagnosis-driven, threshold, and
full-deployment replacement each restore all modeled cases; no recovery and
worker restart each restore half. Median declared action times are 90.5 s for
diagnosis-driven, 60 s for threshold, 120 s for worker restart, and 180 s for
full replacement. These times and costs are policy inputs; post-action request
performance is simulated. The result establishes deterministic state-machine
and action-selection behavior, not live-cluster recovery time.

### ForgeCI and WarmPath snapshots

- `artifacts/forgeci/demo/evaluation.json` records the deterministic fixture's
  intentional 12.00% p99-TTFT regression, corrected 97.5% interval
  11.4676%-12.5349%, zero inconclusive rate, three unique commits evaluated,
  and the exact expected first bad fixture commit. This is not an external
  runtime bisection.
- `artifacts/warmpath/evaluation/result.json` records a 0.955416 ms WarmPath p95
  for a small synthetic snapshot: 11.641% faster than measured local-disk cache
  but 0.822% slower than the modeled pinned-host-memory proxy. Page-cache and
  local file stages are measured; pinned memory is an ordinary host-copy proxy
  and the ready warm replica is modeled.

## Current-host hardware evidence

The current machine is Apple Silicon and has no compatible GPU fabric. The
following facts are actual local discovery/CPU measurements, not synthetic GPU
results:

- `artifacts/fabric/local/topology.json`, SHA-256
  `06998272c03f70f8f4134b5461ae2e0b7952d90d1574014f01c3cc3672c65f33`,
  captured 2026-08-02T07:42:01Z: 53 typed nodes and 26 edges comprising one
  host, one CPU socket, one NUMA domain, 25 visible NIC interfaces, and 25
  network-rail nodes. It explicitly warns that no NVIDIA GPU or InfiniBand
  device was visible and does not infer either capability.
- `artifacts/fabric/local/measurements/host-memory.json`, SHA-256
  `e452b05ea7564a4e4f7fd8c826e937433c24690562c985c6d53f93d57d8852aa`,
  is a measured 1 MiB `numpy.copyto` host-memory case using 3 warmups and 11
  samples on macOS 15.6.1 arm64/Python 3.12.7/NumPy 2.3.5. Median latency was
  12.416 us, p95 12.646 us, p99 12.663 us, median absolute deviation 0.251 us,
  and the recorded 95% median interval was [11.333, 12.625] us.
- `artifacts/fabric/local/profile/host-memory.json`, SHA-256
  `9fbb0b2c0a4998204b224c9855acf722d4532303a737ab70b4240108847bfdd1`,
  is a separate 9-sample measured calibration case with 3 warmups and median
  12.834 us. It is not presented as a GPU, PCIe, NVLink, or RDMA benchmark.

## Synthetic versus hardware-backed status

| Area | Status on this host |
|---|---|
| Local CPU/platform topology discovery | **Hardware-backed, exercised.** |
| Local host-memory copy profiling | **Hardware-backed, exercised.** |
| Two-node/eight-GPU topology, NVLink/NVSwitch, PCIe, dual rail, MoE, P/D disaggregation | **Deterministic synthetic fixtures, exercised.** |
| Physical compiler, Rust twin, physical faults, Autopsy counterfactuals, and recovery | **Synthetic physical execution, exercised.** |
| Existing local Rust HTTP/SSE gateway | **Real local process execution, exercised; backend physical GPUs are simulated.** |
| ForgeCI fixture | **Real local Git/subprocess/statistical workflow against a deterministic fixture repository.** |
| WarmPath | **Measured local file/page-cache/host-copy stages plus modeled pinned-memory/warm-replica alternatives.** |
| CUDA/NVML/DCGM/CUPTI, real GPU compute and peer transfer | **Implemented/optional or fixture-tested; unexercised.** |
| NCCL, NVLink/NVSwitch, InfiniBand, RoCE, RDMA, DeepEP, NIXL, real multi-node runtime | **Runner/adapters or schemas implemented; unexercised.** |
| Privileged GPU-clock, traffic-control, eBPF, and host-network fault injection | **Opt-in guarded; unexercised and not authorized.** |
| vLLM, SGLang, and NVIDIA Dynamo execution | **Version-checked/offline lowering; live execution unexercised.** |
| Kubernetes admission and rollout | **Offline manifest/schema validation only.** |
| Modal and Truss deployment | **Offline import/schema/advisory metadata validation only; no credentials or paid resources used.** |

No cloud resource, external deployment, host-wide clock change, traffic-control
rule, or production mutation was created by this work.

## Known limitations and risks

1. No real NVIDIA, NCCL, NVLink, IB/RoCE/RDMA, or multi-node measurement exists.
   Synthetic fabric curves cannot support a hardware performance claim.
2. The measured NCCL runner deliberately supports one explicit local process;
   full heterogeneous primitive, channel, rank-order, and multi-host sweeps
   require compatible hardware and remain unexercised.
3. H2 is synthetic internal validation with shared calibration. Expert-skew
   interval coverage remains poor, and loaded queueing is evaluated separately.
4. Pipeline-parallel groups exist, but the current IR does not emit a complete
   layer-to-stage and PP send/receive schedule. Compute/communication overlap is
   a calibrated flow-level approximation.
5. Autopsy capture exercised simulator/runtime artifacts, not privileged
   CUPTI/DCGM/eBPF or synchronized real multi-node traces. Its confidence score
   is deterministic evidence weight, not probability.
6. Recovery execution uses the guarded local simulated driver. There is no
   authorized production actuator and no live orchestrator mutation claim.
7. Direct vLLM/SGLang adapters reject unsupported multi-host rendezvous;
   generic Kubernetes rejects plans without an atomic gang contract; Modal and
   Truss placement metadata is advisory.
8. The generated deployment image remains an operator-supplied runtime
   contract; no published engine-bearing GPU OCI image was exercised here.
9. WarmPath pinned-host and warm-replica alternatives are partly modeled; the
   local experiment uses a small synthetic artifact rather than model weights.
10. The rank-ordering experiment is synthetic and disabled. No speedup is
    claimed pending matched hardware trials.
11. Canonical topology fingerprints include capture metadata. This safely
    rejects stale profiles but limits reuse across equivalent rediscovery.
12. Current generated artifacts have stale source provenance. Until atomic
    regeneration, publication, and a clean archive pass, this draft must not be
    treated as the release completion record.

## Artifact and documentation locations

- Baseline and execution history: `EXECUTION_LEDGER.md`, tag
  `sloforge-fabric-baseline-c67e082`.
- Flagship bundle: `artifacts/fabric-demo/manifest.json` and
  `reports/fabric-demo/`.
- Physical plans and optimizer evidence:
  `artifacts/fabric-demo/physical-plan.json`,
  `artifacts/fabric-demo/physical-plan-topology-unaware.json`, and
  `artifacts/fabric-demo/optimizer.json`.
- Simulation, Autopsy, and recovery evidence:
  `artifacts/fabric-demo/simulations/`, `artifacts/fabric-demo/autopsy/`, and
  `artifacts/fabric-demo/recovery/`.
- Current synthetic evaluation: `artifacts/fabric/evaluation/result.json`,
  `artifacts/fabric/evaluation/manifest.json`, and
  `reports/fabric-evaluation.md`.
- Current-host evidence: `artifacts/fabric/local/topology.json` and
  `artifacts/fabric/local/{measurements,profile}/host-memory.json`.
- ForgeCI: `artifacts/forgeci/demo/evaluation.json`,
  `artifacts/forgeci/demo/upstream-issue.md`, and
  `reports/forgeci-evaluation.md`.
- WarmPath: `artifacts/warmpath/manifest.json`,
  `artifacts/warmpath/evaluation/result.json`, and
  `reports/warmpath-evaluation.md`.
- Architecture and component documentation: `docs/fabric`, `docs/autopsy`,
  `docs/recovery`, `docs/forgeci`, and `docs/warmpath`.
- Cross-cutting security, reproducibility, limits, related work, demo, interview,
  and resume material: `docs/FABRIC_*.md`.
- Paper extension: `paper/fabric_extension/SLOFORGE_FABRIC.md`.
- Reviews: `ARCHITECTURE_DISTRIBUTED_NETWORK_REVIEW.md`,
  `SIMULATOR_OPTIMIZER_REVIEW.md`, `AUTOPSY_STATISTICAL_REVIEW.md`,
  `RUNTIME_ADAPTER_REVIEW.md`, `FABRIC_GPU_PERFORMANCE_REVIEW.md`,
  `FABRIC_SECURITY_CONCURRENCY_REVIEW.md`, `FABRIC_UI_REVIEW.md`,
  `FABRIC_DOCUMENTATION_REVIEW.md`, `FABRIC_CLEANROOM_REVIEW.md`, and
  `FABRIC_FINAL_ADVERSARIAL_REVIEW.md`.

At the draft snapshot, the flagship manifest listed 140 artifacts; every file
existed and all 140 SHA-256 values matched. The manifest itself has SHA-256
`f30bd33d4ee4bcafbaa7baf6a98c6e4a0482b5795bd703d1c4a2d69c44bbacc1`.
This integrity result does not cure its stale source provenance.

## Final-release verification table

Rows marked pending have **not** been rerun at the eventual final source and
artifact-publication commits. They are deliberately not claimed.

| Release gate | Status | Required release evidence |
|---|---|---|
| Freeze and record final source commit | **PENDING** | Full SHA after all source/review fixes. |
| Regenerate original SLOForge CPU evidence | **PENDING** | `make demo` at final source. |
| Regenerate flagship physical evidence | **PENDING** | `make fabric-demo` at final source; plan/evaluation provenance must name that source. |
| Standalone Autopsy demonstration | **PENDING** | `make autopsy-demo` at final source. |
| ForgeCI fixture demonstration | **PENDING** | `make forgeci-demo` at final source. |
| WarmPath local demonstration | **PENDING** | `make warmpath-demo` at final source. |
| H1-H6 evaluation regeneration | **PENDING** | `make extension-evaluation`; atomically update raw results, reports, plots, manifests, and hashes. |
| Existing and extension fast checks | **PENDING** | `make check` and `make fabric-check` at final source with exact counts recorded. |
| Docker smoke | **PENDING** | `make docker-smoke` at final source, or an exact environmental failure record. |
| CPU benchmark and explicit GPU-unavailable path | **PENDING** | Final CPU command plus GPU command that records unavailable hardware without a fabricated result. |
| Public CLI workflow | **PENDING** | discover, benchmark, model inspect, compile, explain, simulate, fail-closed and success validation, export, Autopsy, and recovery commands against final artifacts. |
| Artifact publication | **PENDING** | Commit regenerated `artifacts/` and `reports/`; reconcile every manifest, evidence, plan, diagnosis, recovery, and UI cross-reference hash. |
| Clean archive | **PENDING** | `make clean-room-test` from the committed publication revision; must pass before generation as well as after. |
| Source quality scan | **PENDING** | Review TODO/FIXME/placeholder/stub/`NotImplementedError`/bare `pass`/fake-result/hard-coded-benchmark matches in core paths. |
| Safety and release hygiene | **PENDING** | Secret/path scan, no orphan process, no active fault/tc rule, no paid resource, no external mutation, `git diff --check`, and clean status. |
| Final report verification | **PENDING** | Replace draft/current-source wording, record source and publication commits, verify all cited paths/hashes/numbers, and commit this report. |
