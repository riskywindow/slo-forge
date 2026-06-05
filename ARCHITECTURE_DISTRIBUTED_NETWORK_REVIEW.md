# SLOForge Fabric architecture, distributed-systems, and networking review

Review date: 2026-08-01
Review scope: physical IR, discovery normalization, fabric profiling, compiler and placement,
Rust physical simulator, Python lowering, runtime adapter boundaries, recovery planner/executor,
and the deterministic Fabric artifacts.
Review commits: `57594ec`, `c2bc152`, `5f2b366`, `beba063`, `2828433`, `92f71a4`.

## Verdict

An experienced inference-infrastructure engineer should regard this as a deep systems project,
not a broad wrapper around existing libraries. The decisive evidence is the combination of a
typed physical IR, provenance-preserving topology discovery, measured service curves, an actual
Rust discrete-event resource scheduler, physical-plan compilation, counterfactual fault replay,
and a guarded recovery state machine. External runtimes are deliberately adapter targets; they
are not where placement, contention, diagnosis, or recovery semantics live.

That verdict is bounded. Multi-node GPU/RDMA behavior was exercised with deterministic physical
fixtures on this machine, not real multi-node hardware. The Python workload lowerer is presently
a calibrated flow-level model, not a layer-by-layer execution trace. The generic Kubernetes
exporter must remain fail-closed for cross-host gangs until it emits a real PodGroup or
LeaderWorkerSet contract. These limits do not invalidate the architecture, but they must remain
visible in reports and benchmark claims.

No unresolved high-severity correctness issue remains in the reviewed scope after the patches
listed below. Remaining medium findings are explicit model-fidelity or enforcement boundaries.

## Credibility assessment

| Area | Assessment | Evidence |
|---|---|---|
| Physical IR | Credible | `PhysicalExecutionPlan` composes rank, expert, collective, KV, memory, overlap, evidence, failure exposure, and recovery variants in strict typed models (`python/sloforge/fabric/ir/models.py:812`). Cross-document hashes and Rust/Python golden conformance tests passed. |
| Placement and rank groups | Credible after fixes | TP and PP groups are replica-local, DP groups join corresponding local ranks, EP groups do not cross replica fault domains, and disaggregation assigns complete replicas to roles (`python/sloforge/fabric/compiler/core.py:370`, `python/sloforge/fabric/compiler/core.py:409`). Divisibility and runtime-feature pruning is explicit (`python/sloforge/fabric/compiler/core.py:795`). |
| PCIe/NVLink/NIC/rail behavior | Credible at flow level | The compiler walks topology paths and derives transport/rails (`python/sloforge/fabric/compiler/core.py:544`); the simulator materializes link resources and sharing groups (`python/sloforge/fabric/simulation.py:555`, `crates/sloforge-fabric-sim/src/engine.rs:305`). Measured message-size curves override analytical path estimates when a compatible profile exists. |
| Resource contention | Credible | Rust distinguishes exclusive and bounded fair-share scheduling, reserves both physical resources and contention groups, detects deadlock, and maintains resource accounting (`crates/sloforge-fabric-sim/src/model.rs:78`, `crates/sloforge-fabric-sim/src/engine.rs:185`, `crates/sloforge-fabric-sim/src/engine.rs:305`). |
| Fault domains and counterfactuals | Credible after fixes | Rank, resource, sharing-group, rail, collective, startup, and failure effects are deterministic and scoped. Rank counterfactuals now alter the same GPU/HBM resource class as rank faults (`crates/sloforge-fabric-sim/src/engine.rs:125`). |
| Discovery | Credible and epistemically safe | Facts are known, unknown, or conflicting, always carry observations and provenance, and unknown capabilities cannot acquire inferred values (`python/sloforge/fabric/topology/models.py:52`, `python/sloforge/fabric/topology/models.py:79`). Current-host discovery explicitly warns when host topology or RDMA is unavailable. |
| Recovery transitions | Credible control skeleton | Simulation validation precedes build, shadow and canary have sample/SLO gates, deadlines terminate safely, external mutation requires dual authorization, observations are monotonic/idempotent, and started streams block old-worker drain (`python/sloforge/recovery/state_machine.py:221`). |
| Runtime boundary | Credible if fail-closed | The physical compiler owns intent; adapters perform version/capability checks and offline rendering. Exact placement is rejected where a target only supports advisory metadata (`python/sloforge/fabric/adapters/compatibility.py:47`). |

## Findings and dispositions

| Severity | Finding | Evidence | Disposition |
|---|---|---|---|
| High | Disaggregated plans split a single TP/PP replica between prefill and decode, producing pools that could not execute complete model/collective graphs. | Former role assignment split rank ranges; current complete-replica assignment is at `python/sloforge/fabric/compiler/core.py:370`. | Fixed in `57594ec`; DP must be at least two and complete replicas are assigned to worker roles. Regression test: `test_disaggregation_assigns_complete_data_replicas_to_worker_pools`. |
| High | Per-request simulation fanned each request through every DP replica and divided plan latency by model-parallel rank count a second time. This understated compute while serializing independent replicas. | Replica selection and per-rank compute lowering are now at `python/sloforge/fabric/simulation.py:599` and `python/sloforge/fabric/simulation.py:615`. | Fixed in `57594ec`; requests deterministically select one eligible replica and preserve already-lowered plan latency. |
| High | Collective critical-path cost was summed across independent DP replicas, KV used one global role path, and decode-side collectives were incorrectly staged before KV while also present in opaque TPOT. | Replica-critical-path aggregation is at `python/sloforge/fabric/compiler/core.py:611`; role staging is at `python/sloforge/fabric/simulation.py:535`. | Fixed in `57594ec` and `c2bc152`; take the slowest replica group, create replica-pair KV routes, and lower only prefill/mixed legacy collectives before KV. |
| High | Data parallelism incorrectly reduced per-rank KV memory and per-request latency; PP incorrectly provided request-latency speedup. | Memory sharding is at `python/sloforge/fabric/compiler/core.py:492`; per-request performance semantics are at `python/sloforge/fabric/compiler/core.py:880`. | Fixed in `57594ec`; DP duplicates model/KV state, TP/PP shard memory, and only TP reduces modeled request compute latency. |
| High | The physical compiler allowed TP incompatible with KV heads, EP groups spanning replicas, and unsupported disaggregation/EP features. | Feasibility checks are at `python/sloforge/fabric/compiler/core.py:795`; group formation at `python/sloforge/fabric/compiler/core.py:409`. | Fixed in `57594ec`; explicit rejection codes and tests added. |
| High | Every expert was placed only once across all ranks, leaving independent/disaggregated replicas without a full expert set. | Replica-aware ownership is at `python/sloforge/fabric/compiler/core.py:1045`. | Fixed in `92f71a4`; each expert has one owner per complete DP replica and maximum replica count matches the assignment. |
| Medium | Selected and recovery E2E metrics hard-coded 64 output tokens, corrupting other workload constraints and communication fraction. | `python/sloforge/fabric/compiler/core.py:1020`, callers at `:1116` and `:1240`. | Fixed in `92f71a4`; selected and recovery variants use `output_tokens_p95`. |
| Medium | Expert-skew input existed only as an implicit workload concept and could not alter expert traffic without changing unrelated collectives. | Typed shape and all-to-all-only scaling are at `python/sloforge/fabric/simulation.py:312` and `:692`. | Fixed in `c2bc152`; bounded `expert_skew_factor` affects all-to-all message shape only. |
| Medium | `ScaleRank` counterfactual also scaled communication operations tagged with a rank, unlike the actual rank-slowdown fault, so causal repair estimates were not comparable. | Resource-scoped transformation is at `crates/sloforge-fabric-sim/src/engine.rs:125`. | Fixed in `5f2b366`; scope is GPU compute/HBM only, with analytical regression coverage. |
| Medium | Kubernetes ConfigMap data serialized as YAML binary and returned export manifests disagreed with persisted manifests because of an impossible self-hash. | Renderer and manifest contract in `python/sloforge/fabric/adapters/core.py`. | Fixed in `beba063`; canonical JSON is text and persisted/returned manifests are identical. |
| Medium | Runtime adapter validation checked only that binding IDs existed, not node type or host ownership, allowing a host field to name a GPU or a rank to bind a remote host's NIC. | `python/sloforge/fabric/adapters/models.py`. | Fixed in `2828433`; host/GPU/NUMA/NIC/rail type and ownership are validated before rendering. |
| Medium | Generic multi-node Kubernetes output cannot guarantee gang startup: ordinary Deployments do not implement Grove/LWS just because an enum names one. | `python/sloforge/fabric/adapters/compatibility.py:70`. | Fixed fail-closed in the shared adapter-review change: cross-host plans must use the validated DynamoGraphDeployment target until the generic exporter emits an enforceable gang contract. |
| Medium, open | `ParallelismPlan` identifies PP rank groups but has no explicit layer-to-stage assignment, and the compiler does not emit PP send/receive operations. A PP plan is therefore not yet a fully executable layer map. | IR fields at `python/sloforge/fabric/ir/models.py:504`; group generation at `python/sloforge/fabric/compiler/core.py:409`. | Accepted fidelity limit. The compiler validates available stage boundaries, but PP artifacts must be treated as placement intent rather than a complete runtime schedule. No hardware claim should be based on PP until this is modeled. |
| Medium, open | Python lowering represents decode as one calibrated GPU operation and does not materialize per-token decode, expert-dispatch/combine, or per-layer overlap DAGs, although the Rust protocol supports those operation kinds. | Opaque decode duration at `python/sloforge/fabric/simulation.py:644`; Rust kinds in `crates/sloforge-fabric-sim/src/model.rs:126`. | Accepted flow-level digital-twin boundary. It is adequate for deterministic ranking/fault fixtures; it cannot support kernel-level overlap claims without imported runtime traces. |
| Medium, open | Multi-hop transfer duration sums every demanded resource and reserves the path for that sum. This is a conservative store-and-forward approximation and may overestimate cut-through/pipelined fabrics. | `crates/sloforge-fabric-sim/src/engine.rs:710`. | Accepted calibrated-model limit. Held-out error must be broken down by hop count/contention, and physical measurements should override analytical curves. |
| Medium, open | NIC choice prefers same-NUMA and highest advertised speed rather than solving the actual GPU-to-NIC shortest path or rail load balance. | `python/sloforge/fabric/compiler/core.py:331`. | Accepted greedy-heuristic limit. Transport evaluation subsequently walks graph paths, but the placement search can miss a better NIC affinity. Baselines must not call this globally optimal. |
| Medium, open | Canonical compiler profile binding hashes the full canonical topology, including capture timestamps, even though raw discovery separately computes a time-independent hardware fingerprint. This prevents safe reuse across a fresh equivalent discovery. | Strict binding at `python/sloforge/fabric/compiler/core.py:146`; capture-bearing fields at `python/sloforge/fabric/ir/models.py:367`; stable raw fingerprint at `python/sloforge/fabric/topology/models.py:179`. | Safe but operationally restrictive. It rejects stale evidence rather than misusing it; compatible-profile reuse needs a distinct hardware identity plus compatibility policy, not weaker validation. |
| Medium, open | A rollback transition is audited and traffic remains guarded, but the local executor does not execute inverse actions referenced by `rollback_action_id`; it is state rollback, not general infrastructure undo. | Rollback transition at `python/sloforge/recovery/state_machine.py:277`; action execution at `python/sloforge/recovery/executor.py:86`. | Safe by default because external mutation is disabled. Reports must not claim automated infrastructure rollback beyond the local simulated driver until compensating actions are executed and tested. |

## Distributed-systems invariants reviewed

- Determinism: placement, workload-to-replica routing, simulator ordering, faults, and
  counterfactuals use explicit seeds or stable ordering; Python hash randomization is not used for
  placement.
- Resource conservation: exclusive resources serialize, fair-share resources enforce maximum
  concurrency, sharing-group capacity is acquired/released with physical demands, and invalid
  demand/cycle/collective barriers fail explicitly.
- Failure isolation: EP/TP/PP mandatory groups stay inside a data-replica boundary; rank bindings
  carry host fault domains; recovery variants exclude failed resources; no external mutation is
  enabled by default.
- Stream preservation: promotion stops new traffic before old-worker drain, and the terminal drain
  waits while started streams remain active.
- Evidence integrity: discovery facts retain source/confidence; fabric samples retain invocation,
  warmup/sample counts, environment, and hashes; reports reference generated artifacts.

## Networking depth reviewed

The topology model distinguishes PCIe, NVLink/NVSwitch, NIC-to-network, sharing group,
contention domain, directionality, duplex behavior, health, measured curves, and provenance. The
compiler uses those edges for transport and duration selection rather than assigning one global
bandwidth. The simulator separately schedules physical edges and oversubscription groups and can
degrade either a resource or a shared domain. Synthetic fixtures cover NVLink, multi-NUMA PCIe,
InfiniBand, RoCE, MIG visibility, degraded edges, container restrictions, and conflicting facts.

This is credible communication/fabric engineering at a calibrated flow/collective level. It is
not a packet simulator and should not claim packet-loss, congestion-control, or microsecond
cross-node fidelity beyond the aggregate curves supplied to it.

## Acceptance evidence

Executed during this review:

```text
uv run pytest -q tests/python/test_fabric_ir.py tests/python/test_fabric_topology.py \
  tests/python/test_fabric_profiling.py tests/python/test_fabric_compiler.py \
  tests/python/test_fabric_simulation_bridge.py tests/python/test_fabric_adapters.py \
  tests/python/test_fabric_recovery.py tests/python/test_autopsy_diagnosis.py \
  tests/python/test_autopsy_alignment.py tests/python/test_autopsy_counterfactual.py
98 passed

uv run pytest -q tests/python/test_fabric_compiler.py
11 passed

uv run ruff check python/sloforge/fabric/compiler/core.py tests/python/test_fabric_compiler.py
All checks passed

uv run mypy python/sloforge/fabric/compiler/core.py
Success: no issues found

cargo test -p sloforge-fabric-protocol
10 conformance tests passed

cargo clippy -p sloforge-fabric-sim --all-targets --all-features -- -D warnings
passed

cargo test -p sloforge-fabric-sim
all unit, analytical, CLI, property, two-node, and validation tests passed
```

The original broad Python result predates the final two compiler regression tests; the focused
11-test compiler run includes them. GPU, NVLink, InfiniBand, and RDMA hardware were not available
to this reviewer. Their adapters and schema paths are reviewed and fixture-tested, but real
multi-node transport performance remains unexercised rather than inferred.
