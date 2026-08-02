# SLOForge Fabric completion-contract audit

Audit date: 2026-08-01 (America/Los_Angeles)  
Source observed during audit: `d523a925f7e55f8d4514690971f0a84e225263a3` through `62f5adf4c605a9ad0b73856300c058a027ab3978` (other agents were committing and regenerating evidence concurrently)  
Audit work directories: `/tmp/sloforge-completion-help.I4kayH`, `/tmp/sloforge-completion-fixture.6YLhWO`, `/tmp/sloforge-completion-forgeci.W2kzqC`

This is a fail-closed inventory, not a statement that the completion contract is met.

Status meanings:

- **PASS**: concrete implementation, test, and exercised evidence were found.
- **PARTIAL**: substantial implementation exists, but a required behavior, integration, artifact, or validation is absent.
- **UNEXERCISED**: implementation or command generation exists, but this machine could not execute the hardware, privileged, cloud, or full-system path.
- **FAIL**: the required workflow is absent or an exercised workflow failed.

## Executive verdict

SLOForge Fabric is a substantial deterministic physical simulator, compiler IR, causal-analysis fixture system, recovery state machine, ForgeCI implementation, and WarmPath implementation. It is not yet completion-contract complete. The most important blockers are:

1. **FAIL — the flagship workload is not a valid input to the public simulator CLI.** `python/sloforge/fabric/demo.py:105-111` emits `arrival_us` and string priorities, while `python/sloforge/cli/fabric.py:91-106` calls the canonical trace loader, which requires `arrival_ms` and integer priority. Exercising `sloforge fabric simulate --trace artifacts/fabric-demo/mixed-bursty.jsonl` failed before simulation. `tests/python/test_fabric_demo.py:11-22` validates burstiness but never round-trips the artifact through `load_trace` or `fabric simulate`.
2. **FAIL — checked flagship Autopsy evidence cannot be replayed.** `sloforge autopsy replay` requires a `--simulation-input` (`python/sloforge/cli/autopsy.py:140-164`), but the demo stores only `simulations/{healthy,degraded,restored}.json`; its tracked artifact list at `python/sloforge/fabric/demo.py:999-1022` omits every `FabricSimulationRequest`. `autopsy/counterfactuals.json` retains only `simulation_input_sha256`. There is no sufficient checked artifact from which to reproduce the input.
3. **FAIL — `fabric validate` is fail-open.** An exercised fixture returned predicted p95 TTFT `128.047 ms`, observed `10443.409 ms`, and relative error `80.55895` (8,055.9%), yet exited zero. `python/sloforge/cli/fabric.py:450-465` writes error numbers but has no SLO/error threshold, `valid` field, or nonzero failure path.
4. **PARTIAL — the physical compiler is not simulator-in-the-loop.** `python/sloforge/fabric/compiler/core.py:1460-1461,1557-1558` records `simulator_calls=0` and `solver_time_ms=0.0`; `tests/python/test_fabric_compiler.py:94-95` requires zero calls. Every checked evaluation trial also reports zero compiler simulator calls. The compiler evaluates analytical curve estimates, then the demo simulates only after selection. This does not satisfy the required “evaluate with simulator, refine promising candidates” compiler loop or meaningful solver-time accounting.
5. **PARTIAL — measured Fabric profiling stops at host memory.** `python/sloforge/cli/fabric.py:155-176` rejects measured quick/full suites. `python/sloforge/fabric/profiling/adapters.py:1-4` explicitly says NCCL/ibverbs/NVML adapters only build commands and never execute; there is no bounded executor/result ingester that produces canonical measured GPU/collective/link curves. The current machine evidence is one measured host `memcpy` result; all GPU, NVLink, RDMA, NCCL, DeepEP, and NIXL paths are unexercised.
6. **PARTIAL — Autopsy capture is simulator normalization, not deployment capture.** The required UX names `--deployment` and `--duration-seconds`; the implemented command requires simulator input/output and a plan (`python/sloforge/cli/autopsy.py:52-86`). Models can represent live/imported evidence, but there are no deployed adapters for CUPTI/NCCL/NVML/DCGM/NIC/CPU/container evidence collection.
7. **PARTIAL — canonical schemas and cross-language conformance cover only five Fabric documents.** JSON Schemas and Rust golden conformance exist for TopologyGraph, ModelGraph, FabricProfile, PhysicalExecutionPlan, and RecoveryPlan. There are strict Python models but no checked schemas/golden Rust round trips for Autopsy event/run/diagnosis/counterfactual/minimization or WarmPath artifact graph/profile/plan/execution.
8. **PARTIAL — physical deployment exporters are library-only and offline.** Local, Docker, Kubernetes, Dynamo, Modal, and Truss lowering is tested through `export_physical_plan`; vLLM and SGLang are runtime command generation. There is no `sloforge fabric export` command and no physical deployment export in `fabric-demo`. Generic Kubernetes rejects multi-node gang scheduling by design. Modal/Truss are advisory metadata only.
9. **PARTIAL — physical faults are a typed simulation catalog, not distinct fault mechanisms.** Thirty required labels are present, but `python/sloforge/fabric/faults.py:137-188` lowers them to four aggregate mechanisms. The schema permits only `execution_mode="simulation"` (`:104-108`). No `tc`/network-namespace/process-control integration exists.
10. **PARTIAL — required physical Prometheus metric coverage is narrow.** Autopsy events and simulator traces carry many physical signals, but `python/sloforge/fabric/demo.py:500-512` exports only p95 TTFT, p99 TPOT, and p95 end-to-end metrics. Required collective, link, rank-skew, expert, GPU, NUMA, prediction, diagnosis, and recovery metrics are not exported as Prometheus series.
11. **PARTIAL — report provenance is strong but not uniformly closed.** Fabric demo and Fabric evaluation have hash validators; WarmPath evaluation hashes tracked files; ForgeCI preserves per-run hashes and a bisection digest. The Fabric demo manifest omits healthy/restored Prometheus files, raw profile files, environment, process logs, healthy/degraded Autopsy runs, and simulation requests. At audit time `artifacts/fabric/evaluation/result.json` recorded source commit `5460ab8...`, older than the audited source; the artifact validator checks content hashes, not source freshness.

## Public command inventory

All 40 root/group/subcommand `--help` paths exercised by this audit exited zero. The following table maps every requested extension command.

| Command | Status | Implementation and exercised evidence |
|---|---|---|
| `sloforge fabric discover` | PASS | `python/sloforge/cli/fabric.py:120-138`; real current-host artifact `artifacts/fabric/local/topology.json`; nine fixture families in `tests/fixtures/topologies`; audit discovered the two-node InfiniBand fixture. |
| `sloforge fabric benchmark` | PARTIAL | `python/sloforge/cli/fabric.py:141-205`; audit synthetic quick run emitted 97 measurements; portable measured host-memory path passes. Measured quick/full is explicitly rejected. |
| `sloforge fabric model inspect` | PASS | `python/sloforge/cli/fabric.py:208-241`; audit synthetic MoE inspection emitted 18 layers; offline local-model path in `python/sloforge/fabric/model_graph`. |
| `sloforge fabric compile` | PARTIAL | `python/sloforge/cli/fabric.py:244-333`; audit emitted 2 Pareto candidates, 15 rejected candidates, and 3 recovery variants. Compiler makes zero simulator calls and reports zero solver time. |
| `sloforge fabric explain` | PASS | `python/sloforge/cli/fabric.py:336-361`; audit rendered parallelism, roles, bottleneck, predicted metrics, recovery variants, and evidence count. |
| `sloforge fabric simulate` | PARTIAL | `python/sloforge/cli/fabric.py:364-427`; audit canonical trace run emitted 224 operations and a Chrome trace. Flagship demo trace is incompatible with this command. |
| `sloforge fabric validate` | FAIL | `python/sloforge/cli/fabric.py:430-465`; audit produced an 8,055.9% relative error with exit code zero and no validity/SLO decision. |
| `sloforge autopsy capture` | PARTIAL | `python/sloforge/cli/autopsy.py:52-96`; audit normalized a simulator result. No deployment/duration or live evidence collection path. |
| `sloforge autopsy compare` | PASS | `python/sloforge/cli/autopsy.py:99-114`; audit compared checked healthy/degraded evidence. |
| `sloforge autopsy diagnose` | PASS | `python/sloforge/cli/autopsy.py:117-137`; audit produced a typed ranked diagnosis and comparison. |
| `sloforge autopsy replay` | PARTIAL | `python/sloforge/cli/autopsy.py:140-175`; audit replayed an ad hoc preserved simulator request and safely rejected a non-improving repair. Checked flagship bundle lacks the required request. |
| `sloforge autopsy minimize` | PASS | `python/sloforge/cli/autopsy.py:178-211`; audit generated a diagnosis-preserving reduced bundle. |
| `sloforge autopsy report` | PASS | `python/sloforge/cli/autopsy.py:214-240`; audit generated Markdown from typed diagnosis evidence. |
| `sloforge recovery plan` | PASS | `python/sloforge/cli/recovery.py:43-74`; audit generated quarantine-rail, NIC-affinity, and rank-placement actions with confidence 0.95 and external mutation disabled. |
| `sloforge recovery apply` | PASS (simulated) | `python/sloforge/cli/recovery.py:77-154`; audit completed 12 guarded audit records and attempted 3 actions with zero external mutations. Only simulated driver ships (`python/sloforge/recovery/executor.py:34-85`). |
| `sloforge forgeci run` | PASS | `python/sloforge/cli/forgeci.py:31-54`; audit ran the deterministic matrix successfully in a temporary clone. |
| `sloforge forgeci bisect` | PASS | `python/sloforge/cli/forgeci.py:57-100`; audit evaluated 3 commits and identified `fc422c...` with confidence 0.975. The requested direct `--metric/--threshold-percent` UX is represented inside the matrix rather than as flags. |
| `sloforge warmpath profile` | PASS | `python/sloforge/cli/warmpath.py:32-74`; audit measured 16 local stage/tier combinations with raw artifacts. |
| `sloforge warmpath compile` | PASS | `python/sloforge/cli/warmpath.py:77-146`; audit evaluated 48 candidates and emitted a typed plan with 4 placements. |

Original SLOForge command groups remain registered and the CLI test preserves `trace`, `hardware`, profile/optimize/explain/replay/serve/chaos/export/report/demo (`tests/python/test_fabric_cli.py:45-58`).

## Make target inventory

| Target | Status | Concrete evidence |
|---|---|---|
| `make bootstrap` | PASS definition / not rerun by this audit | `Makefile:8-15`; locked Python, Rust, and UI installation. |
| `make check` | UNEXERCISED by this audit | `Makefile:17-30`; full lint/type/test/UI pipeline. Audit instead ran 101 selected Python tests and all Fabric Rust tests. Parent integration must supply the final clean result. |
| `make demo` | UNEXERCISED by this audit | Original demo target preserved at `Makefile:32-33`; checked original evidence exists. |
| `make fabric-check` | PASS constituents / not invoked as one target | `Makefile:35-42`; audit directly ran 97 selected Python tests plus 4 evidence tests and 31 Fabric Rust tests. Ruff/mypy/clippy must be supplied by final integration. |
| `make fabric-demo` | PARTIAL | `Makefile:44-45`; checked report validates and the synthetic flagship run restores its SLO, but its emitted workload cannot feed the public CLI and it omits simulation requests. |
| `make autopsy-demo` | PARTIAL | `Makefile:47-48` simply reruns `sloforge.fabric.demo` into another directory; it does not exercise the public Autopsy capture/replay CLI. |
| `make forgeci-demo` | PASS | `Makefile:50-51`; checked deterministic regression/bisection evidence and bundle validated; audit separately ran CLI matrix and bisection. |
| `make warmpath-demo` | PASS | `Makefile:53-54`; checked local profile/plan/execution/manifest and report; audit separately ran profile/compile. |
| `make extension-evaluation` | PASS synthetic / stale source stamp | `Makefile:56-59`; H1-H4, H6, and ForgeCI reports exist and validators pass. Fabric result source commit is older than audited HEAD. |
| `make benchmark-cpu` | PASS existing-system path | `Makefile:64-66`; outside new Fabric audit scope but retained. |
| `make benchmark-gpu` | UNEXERCISED | `Makefile:68-69`; current machine records explicit unavailability rather than fake measurements. |
| `make docker-smoke` | PARTIAL for Fabric | `Makefile:74-84`; validates the original gateway/mock compose path, not physical-plan process placement, two-node Docker, or Fabric faulting. |
| `make clean-room-test` | PARTIAL | `Makefile:61-62`, `tools/clean-room-fabric.sh`; archives HEAD, bootstraps, checks, and runs fabric-demo, but does not run Autopsy CLI, ForgeCI, WarmPath, extension evaluation, Docker, or artifact replayability. |

## Schemas, IR, migrations, and fixtures

| Contract item | Status | Concrete evidence |
|---|---|---|
| TopologyGraph schema/model/golden | PASS | `schemas/fabric/topology-graph-v1.schema.json`, `python/sloforge/fabric/ir/models.py`, `crates/sloforge-fabric-protocol/src/model.rs`, `tests/fixtures/fabric/topology-graph-v1.json`. |
| ModelGraph schema/model/golden | PASS | `schemas/fabric/model-graph-v1.schema.json` and matching Python/Rust/golden fixture. |
| FabricProfile schema/model/golden | PASS | `schemas/fabric/fabric-profile-v1.schema.json` and matching Python/Rust/golden fixture. |
| PhysicalExecutionPlan schema/model/golden | PASS | `schemas/fabric/physical-execution-plan-v1.schema.json`; typed parallelism, placement, experts, collectives, KV, memory, overlap, recovery variants, metrics, rejected alternatives, and evidence in `python/sloforge/fabric/ir/models.py:480-883`. |
| RecoveryPlan schema/model/golden | PASS | `schemas/fabric/recovery-plan-v1.schema.json`; Python/Rust models and `tests/fixtures/fabric/recovery-plan-v1.json`. |
| Canonical serialization/stable hashes | PASS | Python canonical I/O in `python/sloforge/fabric/ir/io.py`; Rust implementation in `crates/sloforge-fabric-protocol/src/canonical.rs`; fixed cross-language hashes in `crates/sloforge-fabric-protocol/tests/conformance.rs`. |
| Migration/backward compatibility | PARTIAL | Topology v1alpha1 golden migration exists in both languages; Python also tests old physical field names. There are no migration fixtures for ModelGraph/Profile/Recovery, Autopsy, or WarmPath. |
| Namespaced extension field | PASS | Strict core models plus namespaced `Extensions`; unknown fields fail in `tests/python/test_fabric_ir.py` and Rust conformance. |
| Autopsy event/run/comparison/diagnosis schemas | PARTIAL | Strict versioned Python models at `python/sloforge/autopsy/models.py:100-390`; no JSON Schema files, golden fixtures, migrations, or Rust conformance. |
| Counterfactual/minimization schemas | PARTIAL | Strict Python models in `python/sloforge/autopsy/counterfactual.py` and `minimize.py`; no checked JSON Schemas/golden cross-language fixtures. |
| WarmPath graph/profile/plan/execution schemas | PARTIAL | Strict versioned Python models in `python/sloforge/warmpath/models.py`; no checked JSON Schemas/golden fixtures/Rust conformance. |

All nine required topology fixtures are present and exercised by topology tests:

| Required fixture | Status | File |
|---|---|---|
| Single-GPU workstation | PASS | `tests/fixtures/topologies/single_gpu_workstation.json` |
| Eight-GPU NVLink server | PASS | `tests/fixtures/topologies/eight_gpu_nvlink.json` |
| Multi-NUMA PCIe server | PASS | `tests/fixtures/topologies/multi_numa_pcie.json` |
| Two-node InfiniBand cluster | PASS | `tests/fixtures/topologies/two_node_infiniband.json` |
| RoCE cluster | PASS | `tests/fixtures/topologies/roce_cluster.json` |
| MIG host | PASS | `tests/fixtures/topologies/mig_host.json` |
| Partially degraded topology | PASS | `tests/fixtures/topologies/degraded_topology.json` |
| Container-limited visibility | PASS | `tests/fixtures/topologies/limited_container.json` |
| Conflicting sources | PASS | `tests/fixtures/topologies/conflicting_sources.json` |

## Tier 1 subsystem map

| Tier 1 requirement | Status | Implementation/test/artifact evidence |
|---|---|---|
| PhysicalExecutionPlan IR | PASS | Python/Rust typed IR, schema, golden, conformance, migrations, compiler output. |
| GPU/network topology discovery | PARTIAL | Real CPU/NUMA/NIC/current-host discovery with provenance plus synthetic GPU fixtures in `python/sloforge/fabric/topology`; no GPU existed here. Discovery relies on structured sysfs plus bounded `nvidia-smi`; it does not integrate DCGM, hwloc libraries, libibverbs APIs, or NCCL dumps. |
| Fabric microbenchmarking | PARTIAL | Deterministic complete synthetic suite and measured host-memory probe; raw samples/statistics/hashes. GPU/collective/network adapters are command builders only. |
| Communication-aware digital twin | PASS synthetic | Rust resource DAG, capacity/fairness/contention, collectives, KV, overlap, ranks, faults, deadlock validation, event limits, Chrome traces in `crates/sloforge-fabric-sim`; 21 simulator tests plus 1 protocol test group passed in this audit. |
| Topology-aware physical optimizer | PARTIAL | Exhaustive, random placement, topology-unaware, greedy topology-aware, hierarchical, robust-failure strategies in `python/sloforge/fabric/compiler/core.py`; memory/topology/runtime constraints and Pareto output. No simulator refinement, measurement acquisition, solver timing, or true incremental recompile. |
| SLOForge Autopsy | PARTIAL | Canonical event/alignment/differential/diagnosis models and deterministic rules exist. Capture is simulator-only and checked evidence is not independently replayable. |
| Counterfactual diagnosis | PASS when input preserved | Subprocess simulator replay, hypothesis selection/rejection, evidence hashes, and adversarial reranking tests in `python/sloforge/autopsy/counterfactual.py` and `tests/python/test_autopsy_counterfactual.py`. |
| Recovery planning | PASS simulated | Diagnosis-to-action planner and typed proposal in `python/sloforge/recovery/planner.py`; evidence-linked tests. |
| Shadow/canary/promotion/rollback | PASS simulated | Guarded state machine, minimum samples, cooldown, stream preservation, idempotency, persistence/restart, rollback tests in `tests/python/test_fabric_recovery.py`. No real runtime action driver ships. |
| Flagship demonstration | PARTIAL | Two hosts/eight GPUs, NUMA/NVLink/PCIe/two rails, MoE, disaggregation, mixed bursty workload, two faults, diagnosis, 7 counterfactuals, recovery, live Rust gateway, traces/metrics/report/UI. Public CLI incompatibility and missing replay request prevent cohesive artifact replay. |

## Tier 2 subsystem map

| Tier 2 requirement | Status | Concrete evidence |
|---|---|---|
| ForgeCI matrix | PASS | Strict versioned YAML at `tests/fixtures/forgeci/runtime-compatibility.yaml`; bounded runner in `python/sloforge/forgeci/runner.py`. |
| Statistical regression detection | PASS | Warmup separation, repeated trials, bootstrap intervals, effect size, practical threshold, noise/inconclusive/retry handling in `python/sloforge/forgeci/statistics.py` and `tests/python/test_forgeci.py`. |
| Git bisection | PASS | `python/sloforge/forgeci/bisection.py`; audit found intentional first-bad commit in 3 evaluations at confidence 0.975. |
| ForgeCI minimizer/upstream report | PASS API/demo | `minimize_benchmark` and `render_upstream_issue`; checked `artifacts/forgeci/demo/upstream-issue.md`. No separate CLI commands, but `forgeci-demo` exercises them. |
| WarmPath startup profiler | PASS local | Actual local fetch/verify measurements and environment capture in `python/sloforge/warmpath/profiler.py`. |
| Artifact DAG/storage tiers/compatibility | PASS | Typed DAG, cycle checks, portable/non-portable constraints, storage capacity/security/cost/failure models in `python/sloforge/warmpath/models.py`. |
| WarmPath planner/simulator/executor | PASS local | Exhaustive/beam planner, cold-start simulation, cache/restore/rebuild/local execution, checksums, eviction in `python/sloforge/warmpath`; audit profile/compile passed. |
| WarmPath cloud path | UNEXERCISED | No cloud resource was created. Modal support remains advisory metadata; there is no cloud restore executor. |
| Low-level contribution | PASS synthetic / UNEXERCISED hardware | Trace-gated rank-ordering experiment in `python/sloforge/fabric/performance/rank_ordering.py`; 56 raw paired samples, aggregate modeled median improvement 62.71% with 95% interval 60.90-65.15%, explicitly synthetic. Decision is `measure_on_hardware`, `enabled_by_default=false`; no GPU speedup claim. |
| Expanded evaluation/paper | PASS synthetic / PARTIAL provenance | H1-H4, H5, H6 reports and paper extension exist. Fabric evaluation has 180 plan trials and raw artifacts; source commit is stale relative to audit HEAD. |

## Physical fault inventory

All required labels exist in `PhysicalFaultType` (`python/sloforge/fabric/faults.py:26-56`) and emit ground-truth labels/active intervals. Their concrete behavior is simulation-only and collapses into four mechanisms (`:137-188`).

| Fault | Status | Bound simulator mechanism |
|---|---|---|
| GPU process crash | PARTIAL | Resource unavailable |
| Worker crash | PARTIAL | Resource unavailable |
| GPU clock throttle | PARTIAL | Resource-rate multiplier |
| GPU compute slowdown | PARTIAL | Resource-rate multiplier |
| GPU memory-bandwidth slowdown | PARTIAL | Resource-rate multiplier |
| PCIe bandwidth degradation | PARTIAL | Resource-rate multiplier |
| NVLink bandwidth degradation | PARTIAL | Resource-rate multiplier |
| NIC bandwidth degradation | PARTIAL | Resource-rate multiplier |
| Network latency increase | PARTIAL | Resource-rate multiplier; no independent jitter/latency distribution field |
| Network jitter increase | PARTIAL | Resource-rate multiplier; no jitter distribution |
| Aggregate packet loss | PARTIAL | Resource-rate multiplier |
| Network rail loss | PASS simulated | Resource unavailable; analytical rail-loss test |
| Network bandwidth degradation | PASS simulated | Resource-rate multiplier; flagship ground truth |
| Rank-specific slowdown | PASS simulated | Rank multiplier |
| Rank-specific GPU slowdown | PASS simulated | Rank multiplier; flagship ground truth |
| Collective delay | PASS simulated | Collective multiplier |
| Collective failure | PARTIAL | Resource unavailable, not collective-specific failure semantics |
| NCCL initialization delay | PARTIAL | Collective multiplier, not startup/group-init state |
| Expert load skew | PARTIAL | Rank multiplier, not token-to-expert redistribution |
| Hot-expert concentration | PARTIAL | Rank multiplier, not expert routing/capacity behavior |
| Prefill worker loss | PARTIAL | Resource unavailable |
| Decode worker loss | PARTIAL | Resource unavailable |
| KV-transfer slowdown | PASS simulated | Resource-rate multiplier |
| Startup spike | PARTIAL | Resource-rate multiplier |
| NUMA misplacement | PARTIAL | Resource-rate multiplier, not affinity rebinding |
| CPU scheduling delay | PARTIAL | Resource-rate multiplier |
| Container CPU throttling | PARTIAL | Resource-rate multiplier |
| Storage-read slowdown | PARTIAL | Resource-rate multiplier |
| Simulated OOM | PARTIAL | Resource unavailable |
| Memory-fragmentation rejection | PARTIAL | Resource unavailable |

`tests/python/test_fabric_faults.py:44-108` proves every enum value binds and unknown targets fail, but it does not prove distinct causal semantics. There is one checked physical scenario, `scenarios/fabric/dual-fault-demo.yaml`; the demo consumes it directly, but no public `fabric chaos` command consumes physical scenarios.

## Recovery action inventory

Every required action name is represented by `RecoveryActionKind` at `python/sloforge/fabric/ir/models.py:941-964`, with scope classification at `:967-973`:

| Required action | Status |
|---|---|
| Stop routing, drain worker, restart worker | PASS plan model; simulated executor |
| Quarantine GPU, NIC, rail | PASS plan model; simulated executor |
| Replace replica | PASS plan model; simulated executor |
| Change rank placement/order, NIC affinity, NUMA affinity | PASS plan model/compiler variant; simulated executor |
| Move expert group, replicate hot experts | PASS plan model; planner coverage is diagnosis-dependent; simulated executor |
| Change prefill/decode worker ratio | PASS plan model (`change_worker_ratio`); simulated executor |
| Switch KV transport, collective implementation/channel configuration | PARTIAL | Switch kinds exist; channel change is encoded through generic switch-collective action rather than a dedicated kind. |
| Reduce communication/request concurrency | PASS plan model; simulated executor |
| Shed low-priority traffic | PASS plan model; simulated executor |
| Switch alternate parallelism or aggregation | PASS plan model and recovery variants; simulated executor |
| Rebuild on healthy topology / multi-node to smaller replicas | PARTIAL | Generic rebuild/switch-parallelism actions exist; no real build/deploy action driver. |
| Degraded model/precision policy fallback | PASS guarded model (`degraded_model`); planner excludes it unless allowed. |

The proposal includes diagnosis reference, expected interval metrics, cost, disruption/build time, confidence, compatibility, migration, promotion/rollback/abort criteria, and evidence (`RecoveryPlan`, `python/sloforge/fabric/ir/models.py:1004-1024`).

## Required metric/export inventory

| Metric/export | Status | Evidence |
|---|---|---|
| TTFT, token latency/TPOT, end-to-end latency | PASS | Physical metrics, simulator request latencies, demo Prometheus, reports. |
| Queue time | PARTIAL | Original gateway metrics/events; not emitted in Fabric demo Prometheus. |
| Prefill and decode time | PASS trace / PARTIAL metrics | Simulator/Autopsy event types and stage deltas; absent from Fabric Prometheus. |
| KV-transfer time | PASS trace / PARTIAL metrics | Simulator operation and Autopsy diagnosis signal; absent from Prometheus. |
| Collective duration/wait/bytes | PASS trace/model / PARTIAL metrics | Collective operations/events and comparison signals; absent from Prometheus. |
| Rank skew, expert imbalance | PASS diagnosis / PARTIAL metrics | Autopsy comparison/rules; absent from Prometheus. |
| Network throughput/latency/errors/link utilization | PARTIAL | Link curves, resource utilization, selected counters; no complete live collection or Prometheus surface. |
| PCIe/NVLink throughput | PARTIAL | Curves/synthetic profile and event types; no hardware measurement here or Prometheus export. |
| GPU clocks/power/utilization/memory | PARTIAL | Discovery/NVML command fields and Autopsy counters; no live Fabric export on this host. |
| CPU launch delay/NUMA locality | PASS simulator/diagnosis / PARTIAL metrics | Typed resources and rules; absent from Prometheus. |
| Worker startup time | PASS profile/simulator / PARTIAL metrics | Startup primitive and WarmPath stages; absent from Fabric Prometheus. |
| Physical-plan version/topology fingerprint | PASS artifacts/traces | Plan and manifest; not Prometheus labels. |
| Prediction error | PASS evaluation/validate / PARTIAL operational metric | Evaluation and validation JSON; no Prometheus metric and validation is fail-open. |
| Diagnosis confidence | PASS artifact/report / PARTIAL operational metric | Diagnosis/manifest; no Prometheus series. |
| Recovery action/canary/rollback/recovery time/restored margin | PASS audit/timeline / PARTIAL operational metric | Recovery state/audit and timeline; no Prometheus series. |
| OpenTelemetry-shaped traces | PASS synthetic | `artifacts/fabric-demo/traces/otel.json`; archival JSON shape, not an OTLP exporter. |
| Perfetto/Chrome trace | PASS | `artifacts/fabric-demo/traces/degraded.perfetto.json`, simulator CLI traces, gateway trace. |
| Structured JSON logs | PASS gateway | `artifacts/fabric-demo/runtime/logs/*.stdout.log`; process logs are not included in the demo manifest. |
| Machine-readable decisions | PASS | Optimizer, diagnosis, counterfactual, recovery proposal/execution, timeline. |
| Static report/UI | PASS | `reports/fabric-demo/fabric-demo.md`; Fabric UI parsers/components/tests under `ui/src/fabric-*`, `ui/tests/fabric-*`. |

## Deployment/runtime target inventory

| Target | Status | Evidence and boundary |
|---|---|---|
| Local | PASS offline lowering | `python/sloforge/fabric/adapters/core.py:224-262`; exact CPU/GPU/NIC/rail bindings and fail-on-mismatch launch plan tested. No executor launches this generated plan. |
| Docker | PASS offline lowering / UNEXERCISED Fabric smoke | Bounded resources, UUIDs, cpuset, read-only topology in `core.py:264-320`; tests validate YAML. Root Docker smoke uses original compose. |
| Kubernetes | PARTIAL | Node affinity, anti-affinity, spread, GPU/RDMA resources, rollout and plan ConfigMap emitted/tested. Multi-node plans are rejected because no gang API is emitted. |
| NVIDIA Dynamo | PASS offline schema / UNEXERCISED runtime | v1beta1 DGD lowering and subset-schema test; no cluster execution. |
| vLLM | PASS version-gated command generation / UNEXERCISED runtime | TP/PP/DP/EP and NIXL connector command tests; `deploy/fabric/validated-versions.json` says command generation only. |
| SGLang | PASS version-gated command generation / UNEXERCISED runtime | EP/disaggregation/NIC command tests; command generation only. |
| Modal | PARTIAL | Fails closed unless advisory metadata is explicitly allowed; emits `modal.Image.env` metadata only; no physical-placement enforcement or cloud execution. |
| Truss | PARTIAL | Fails closed unless advisory metadata is allowed; config overlay metadata only; no physical-placement enforcement or live validation here. |
| DeepEP/NIXL/UCX/NCCL/ibverbs | UNEXERCISED | Inventory/version probes and some command builders; no result executor/ingester except synthetic profile. |

`DeploymentTarget` includes local, Docker, Kubernetes, Dynamo, Modal, and Truss; `RuntimeKind` includes native, vLLM, SGLang, and Dynamo (`python/sloforge/fabric/adapters/models.py:30-48`). The public Fabric CLI does not expose this lowering API.

## Flagship demo artifact contract

| Required artifact/behavior | Status | Checked evidence |
|---|---|---|
| Synthetic two-host/eight-GPU topology | PASS | `artifacts/fabric-demo/topology.json` |
| Synthetic calibrated raw Fabric measurements | PASS | `fabric-profile-raw/profile.json` and raw JSON files; only canonical profile is tracked in demo manifest. |
| Model graph, logical plan, topology-aware physical plan | PASS | `model-graph.json`, `logical-deployment-plan.json`, `physical-plan.json` |
| Topology-unaware baseline/Pareto optimizer | PASS | `physical-plan-topology-unaware.json`, `optimizer.json` |
| Bursty mixed, prioritized, long/short workload | PARTIAL | `mixed-bursty.jsonl` content is correct for the demo but not canonical trace-compatible. |
| Gateway/controller/service replay | PARTIAL | Live Rust gateway with 12 requests and metrics/trace/logs. Recovery controller is a deterministic simulated state machine; it does not reconfigure the live gateway. |
| Network degradation and rank GPU slowdown | PASS simulated | Scenario + ground-truth events; checked manifest lists both. |
| Healthy/degraded/restored simulations | PASS | `simulations/*.json`; restored p95 TTFT is below artifact-derived demo SLO. |
| Autopsy evidence/comparison/diagnosis | PASS generated / PARTIAL manifest | Run artifacts exist, but healthy/degraded run files are not manifest-tracked. |
| Counterfactual alternatives/selection | PASS generated / FAIL standalone replay | 7 alternatives evaluated and selected; simulator request omitted. |
| Recovery proposal/shadow/canary/promotion | PASS simulated | `recovery/proposal.json`, `recovery/execution.json`, `timeline.json` |
| Raw JSON/Parquet | PASS JSON | Raw structured JSON; no Parquet is required when JSON is used. |
| OpenTelemetry, Prometheus, Perfetto, structured logs | PASS files / PARTIAL coverage | Files exist; metric breadth and manifest coverage are incomplete. |
| Markdown report and plot | PASS | Hash-derived report and SVG; `tests/python/test_fabric_demo.py:25-45`. |
| Artifact-derived timeline | PASS | `timeline.json` with evidence URIs. |
| UI consumes real artifacts | PASS | `ui/tests/fabric-generated.ts` loads checked bundle; parser rejects missing/tampered/oversized inputs. |
| Report hard-code validation | PASS for tracked values | `_validate_report` and tests verify hashes/values. It cannot validate omitted artifacts. |

At audit time the checked manifest reported synthetic p95 TTFT `1157.369 ms` healthy, `7047.976 ms` degraded, and `1157.369 ms` restored, with `network_bandwidth_degradation` at confidence `0.834`. These are synthetic values and are not GPU performance claims.

## Evaluation contract map

| Hypothesis | Status | Evidence |
|---|---|---|
| H1 topology-aware compilation | PASS synthetic | 180 plan trials across 4 topology fixtures, 3 workloads, 3 seeds, and 5 methods in `artifacts/fabric/evaluation/result.json`; reports and plots. |
| H2 digital-twin ranking/error | PASS synthetic | 180 twin trials, error/rank-correlation/regret/coverage summaries and raw compressed simulations. |
| H3 Autopsy fault identification | PARTIAL breadth | Evaluation covers network-bandwidth degradation and rank straggler across topologies/seeds, reporting top-1/top-3. Unit/adversarial tests cover all diagnosis rules, but there is no evaluation matrix over all required physical fault types. |
| H4 recovery efficiency | PASS synthetic | 5 recovery strategies per diagnosis trial; restoration/cost/drop/rollback summaries. Real live gateway reconfiguration is not measured. |
| H5 ForgeCI | PASS fixture | Intentional regression detection, false-positive/noise handling, bisection, minimization, issue output. |
| H6 WarmPath | PASS local/synthetic model | Six strategies, two scenarios, eleven seeds, 101 trials; local source artifacts with modeled tier timings. `all_timings_hardware_measured=false`. |

Generated reports exist at `reports/fabric-evaluation.{md,html}`, `reports/autopsy-evaluation.md`, `reports/forgeci-evaluation.md`, and `reports/warmpath-evaluation.md`. Raw artifacts, environment manifests, commands, and limitations exist. Fabric evaluation source commit freshness must be regenerated after final code integration.

## Test-category inventory

| Required category | Status | Concrete tests |
|---|---|---|
| Rust unit/integration | PASS | Fabric curve unit; analytical, CLI, two-node, validation suites. |
| Rust property tests | PASS | `crates/sloforge-fabric-sim/tests/properties.rs`; protocol canonical-order proptest. |
| Deterministic simulation/toy cases | PASS | Analytical fairness, serialization, barrier, fault, deadlock, faster-than-wall-clock tests. |
| Concurrency tests | PASS broader runtime | Gateway concurrency/cancellation/queue contract plus recovery thread-safety tests. Fabric simulator is synchronous deterministic. |
| Paused-time tests | PARTIAL | No `tokio::time::pause`/`start_paused` tests found; deterministic simulated time substitutes for Fabric simulator. |
| Trace ingestion/malformed input | PASS | Bounded JSON subprocess, schema validation, Autopsy input bounds, YAML alias rejection, UI artifact bounds. |
| Rust fuzz targets | UNEXERCISED/not present | No `cargo-fuzz` target found. Property tests exist; no dedicated parser fuzz harness. |
| Resource cleanup | PASS targeted | Bounded subprocess cleanup/timeout tests; gateway/demo process teardown; clean-room trap. |
| Python pytest | PASS targeted | 97 selected extension/evidence tests plus 4 checked evaluation tests passed during audit. |
| Python Hypothesis | PARTIAL extension breadth | Fabric IR seed property exists; original trace/IR properties exist. No Hypothesis strategies for Autopsy/WarmPath/ForgeCI schemas. |
| Python strict typing/Ruff | UNEXERCISED by audit | Configured in `make check`/`fabric-check`; final integration must record current result. |
| Optional GPU/multi-node markers | PARTIAL | GPU/expensive markers exist; real multi-node hardware tests were not possible. |
| Cross-language IR/hash/migration | PASS for five Fabric docs | `crates/sloforge-fabric-protocol/tests/conformance.rs`; 10 tests passed. Autopsy/WarmPath omitted. |
| CPU Fabric demo | PASS report assertions / PARTIAL cohesive CLI | `tests/python/test_fabric_demo.py`; public workload replay gap remains. |
| Autopsy demo/diagnosis | PASS fixture / PARTIAL public capture | Differential, alignment, diagnosis, counterfactual, minimization tests. |
| Recovery/shadow/canary/rollback/restart/stream preservation | PASS simulated | `tests/python/test_fabric_recovery.py`. |
| Docker smoke | UNEXERCISED by audit / PARTIAL Fabric scope | Root target exists; not physical-plan Docker. |
| Simulated two-node | PASS | Rust `two_node.rs`; topology/compiler/demo tests. |
| Exporter validation | PASS offline | `tests/python/test_fabric_adapters.py`; current subset schemas and capability boundaries. |
| ForgeCI fixture bisection | PASS | Audit CLI run plus `tests/python/test_forgeci.py`. |
| WarmPath local demo | PASS | Audit CLI profile/compile plus tests and checked manifest. |
| Clean-room install | UNEXERCISED by audit | Script exists but scope is partial as noted above. |
| Artifact-derived report validation | PASS tracked / PARTIAL coverage | Fabric demo/evaluation and WarmPath validation pass; ForgeCI lacks one top-level all-artifact manifest; demo omits required replay inputs/logs/runs. |
| Frontend checks/components/build | PASS tests exist / UNEXERCISED by audit | Fabric parser/components/loader tests and production build are in `make check`. |

Audit test results:

- `97 passed in 16.75s` for targeted Fabric CLI/demo/local evidence/IR/adapters/faults/recovery/Autopsy/ForgeCI/WarmPath/report tests.
- `4 passed in 0.73s` for checked Fabric evaluation, WarmPath evaluation/report, and ForgeCI demo evidence.
- Fabric Rust: 10 protocol conformance tests, 1 library unit test, 20 simulator integration/property tests; all passed.
- CLI help: all 40 enumerated paths exited zero.
- Cheap fixture workflow: discover/benchmark/model/compile/explain/simulate completed; validate exposed the fail-open error; Autopsy compare/diagnose/minimize/report and ad hoc replay completed; recovery plan/apply completed; ForgeCI run/bisect completed; WarmPath profile/compile completed.

## Completion-contract bullet audit

### Existing system

| Bullet | Status | Evidence |
|---|---|---|
| Original tests pass | UNEXERCISED by audit | `make check` exists; targeted extension tests pass. Final full run required. |
| Original CPU demo works | UNEXERCISED by audit | `make demo` preserved; checked original artifacts exist. |
| Existing exporters functional | PASS tests | Original exporter tests plus Fabric adapter tests. |
| Existing schemas compatible/migrated | PASS | Original IR untouched; Fabric added separate versioned IR/reference. |
| No unexplained performance regression | UNEXERCISED | No before/after benchmark tied to baseline commit in current Fabric evidence. |

### Physical compiler

| Bullet | Status | Evidence |
|---|---|---|
| PhysicalExecutionPlan implemented | PASS | Typed Python/Rust/schema/golden. |
| TopologyGraph implemented | PASS | Typed Python/Rust/schema/golden. |
| ModelGraph extensions implemented | PASS | Typed layer/MoE/communication graph. |
| Rust-Python conformance passes | PASS | 10 conformance tests passed. |
| Discovery works on current machine | PASS | Real Apple host/NUMA/NIC artifact; no GPU claimed. |
| Synthetic multi-node fixtures work | PASS | Nine fixtures and two-node tests. |
| Fabric benchmark artifacts reproducible | PASS synthetic/host memory | Raw samples/hashes/tamper tests. |
| Communication-aware simulator implemented | PASS | Rust operation/resource DAG. |
| Contention and failures supported | PASS | Analytical/two-node/fault tests. |
| Physical optimizer produces valid plan | PASS | Audit compile and tests. |
| Aware/unaware baselines exist | PASS | Strategies and checked plans/evaluation. |
| Recovery variants generated | PASS | Audit compile emitted 3. |
| Decisions explainable | PASS | Rejection codes/history/explain CLI; solver timing and simulator calls are not meaningful. |

### Autopsy

| Bullet | Status | Evidence |
|---|---|---|
| Evidence capture works | PARTIAL | Simulator capture only; no deployment/live capture. |
| Canonical normalization works | PASS | Strict event/run models and capture adapter. |
| Cross-node time alignment exists | PASS | Offset/drift/uncertainty/quality models and tests. |
| Healthy/degraded comparison works | PASS | Audit and tests. |
| Diagnosis for required fixture faults | PARTIAL | All rule families unit-tested; evaluation covers only two physical fault classes. |
| Counterfactual replay works | PARTIAL | Works with preserved request; flagship request missing. |
| Hypothesis rejection recorded | PASS | `rejected_reason`, replay rejected scenarios, adversarial tests. |
| Diagnosis confidence recorded | PASS | Typed diagnosis/manifest/report. |
| Minimal reproducer works | PASS | CLI minimize and ForgeCI minimizer. |
| Reports derived from artifacts | PASS tracked | Diagnosis report hashes evidence; demo validator. |

### Recovery

| Bullet | Status | Evidence |
|---|---|---|
| RecoveryPlan exists | PASS | Python/Rust/schema/golden. |
| Guarded state machine exists | PASS | Proposed through completed/rollback/operator states. |
| Simulation validation | PASS simulated | Required first transition. |
| Shadow, canary, promotion, rollback | PASS simulated | Tests and demo audit. |
| Streaming requests preserved | PASS simulated/controller test | Drain waits for started streams; gateway no-retry behavior is separately tested. |
| Controller restart tested | PASS | Snapshot dump/restore/idempotency test. |
| No external mutation by default | PASS | Simulated driver only; external plans become operator-required. |

### Flagship demo

| Bullet | Status | Evidence |
|---|---|---|
| `make fabric-demo` CPU path | PASS execution artifacts / PARTIAL replayability | Checked generated run passes report invariants; CLI input mismatch. |
| Compiles physical plan | PASS | Checked plan/optimizer. |
| Injects two distinct physical faults | PASS | Network degradation and rank GPU slowdown. |
| Autopsy identifies faults | PARTIAL wording | Top diagnosis identifies network cause; rank slowdown appears as secondary evidence, not two separate diagnostic episodes. |
| Counterfactual alternatives evaluated | PASS | 7 evaluations. |
| Recovery selected | PASS | Selected scenario and proposal. |
| Shadow/canary steps run | PASS simulated | Recovery audit/timeline. |
| SLO restoration demonstrated | PASS synthetic | Artifact-derived p95 TTFT. |
| Timeline artifact-derived | PASS | Evidence URIs and report validator. |
| Static report generated | PASS | Markdown/HTML evaluation/static demo report. |
| UI consumes real artifacts | PASS | Generated bundle tests. |

### ForgeCI

All eight completion bullets—fixture matrix, statistical detection, intentional regression, correct bisection, minimized reproducer, and upstream report—are **PASS** in deterministic fixture mode. Concrete evidence is `tests/python/test_forgeci.py`, `artifacts/forgeci/demo`, and the audit CLI run. Attribution is limited to fixture metrics/inputs rather than CUDA/kernel/collective traces, which are unavailable here.

### WarmPath

All eight local completion bullets—startup profiler, DAG, storage model, planner, executor, benchmark, artifact-derived results, cloud separation—are **PASS** for the portable local reference path. Hardware/cloud snapshot restore is **UNEXERCISED**.

### Low-level work

The trace-justified rank-ordering experiment, correctness/invariant tests, raw samples, confidence intervals, honest synthetic label, and disabled-by-default decision are **PASS**. On-device benefit is **UNEXERCISED**.

### Quality

| Bullet group | Status |
|---|---|
| `make check`, `make fabric-check`, Rust fmt/clippy, Python Ruff/mypy, frontend check, Docker smoke, clean-room | UNEXERCISED as complete final commands by this audit; constituent tests listed above pass. |
| Rust/Python tests | PASS targeted; final all-workspace run required after concurrent changes. |
| No secrets/orphans/fault configuration/billable resources | PASS from repository scan and process-safe implementations; no paid/privileged operation was performed. |
| Core paths contain no placeholders | PASS lexical scan: no TODO/FIXME/placeholder/NotImplemented/fake-result matches in core; UI test `stub*` identifiers are test APIs, not core stubs. No dedicated fuzz target. |
| Numbers traceable to artifacts | PARTIAL | Tracked demo/evaluation values validate; omitted simulation requests/logs/runs and stale source commit weaken complete provenance. |

### Evaluation and documentation

Fabric, Autopsy, ForgeCI, and WarmPath evaluation reports, environment/software manifests, commands, raw artifacts, limitations, related work, security/reproducibility docs, demo script, interview deep dive, resume bullets, and paper extension are present. This audit classifies evaluation as **PASS synthetic / PARTIAL provenance** and documentation as **PASS presence**. Hardware manifests explicitly report no compatible NVIDIA/RDMA hardware. No hardware measurement is fabricated.

## Exact fixes required before claiming the completion contract

1. Make `FabricDemo` emit the canonical workload trace schema (or make the public loader accept a versioned Fabric workload format), add a test that runs `fabric simulate` on the checked demo workload, and correct the nonexistent `workloads/mixed-bursty.jsonl` documentation reference.
2. Persist and manifest-hash healthy/degraded/restored `FabricSimulationRequest` documents. Make `autopsy replay` consume a checked demo evidence bundle without reconstructing inputs out of band.
3. Give `fabric validate` explicit SLO and/or prediction-error acceptance criteria, a typed validity result, and nonzero exit on violation; add success and failure CLI tests.
4. Integrate bounded digital-twin evaluation into hierarchical candidate refinement, record actual elapsed solver time/simulator calls, and test that the primary strategy evaluates at least selected/promising candidates through the simulator.
5. Implement bounded execution and strict parsers for at least NCCL tests and the read-only GPU/NIC probes, producing canonical measured artifacts. Keep hardware tests optional and fail without hidden fallback.
6. Either implement a non-privileged live/import capture adapter for gateway/runtime/NVML/NIC traces or rename `capture` to make its simulator-only scope explicit and add the required import/live adapter commands.
7. Add checked JSON Schemas/golden fixtures/migrations for Autopsy and WarmPath public artifacts; add Rust conformance if they cross the Rust/Python boundary.
8. Expose physical-plan export through the CLI and exercise local/Docker generated artifacts in a Fabric system smoke test. Preserve fail-closed cloud advisory boundaries.
9. Separate fault semantics where causal diagnosis depends on them (jitter, packet loss, expert skew, startup, NUMA, OOM) and add a safe opt-in Linux executor or explicitly scope completion claims to simulation.
10. Export the required physical metric set, with plan/topology identifiers and evidence links, in Prometheus and structured telemetry; include all logs/metrics/runs in the demo manifest.
11. Regenerate evaluation and reports after the final source commit and require source-commit freshness in the final artifact verification step.

