# SLOForge Fabric: Topology-aware physical inference compilation, causal debugging, and guarded recovery

## Abstract

Logical inference deployment choices do not determine physical execution. Rank
placement, parallel groups, NUMA and NIC affinity, collective paths, KV transfer,
and recovery topology can dominate latency and availability. We present SLOForge
Fabric, an extension of the SLOForge deployment compiler that emits a versioned
`PhysicalExecutionPlan`, validates it with a deterministic communication-aware
Rust twin, diagnoses deviations with structured differential and counterfactual
analysis, and recompiles or recovers through a shadow/canary/rollback state
machine. The system preserves topology and measurement provenance, separates
synthetic calibration from hardware measurement, and lowers plans into existing
runtimes without replacing their data planes. On a deterministic CPU-only
two-host/16-rank synthetic demonstration, two simultaneous physical faults raised
simulated p95 TTFT from 955.883 ms to 2484.092 ms; seven counterfactuals selected
a combined repair and the guarded recovery restored 955.883 ms. This is a
synthetic systems validation, not a multi-GPU performance claim. A local ForgeCI
fixture bisected an injected 12.00% TTFT regression, and WarmPath exhaustively
evaluated 243 local artifact placements. Real NVIDIA and RDMA validation was
unavailable on the evaluation host.

## 1. Motivation

An inference service may satisfy a logical resource model and still miss its SLO
because tensor ranks cross a weak PCIe boundary, an all-to-all shares a network
rail, the CPU launch thread is remote from a GPU, decode waits for KV transfer,
or one rank throttles a collective. A replica-count controller cannot repair all
of these failures. Operators need a shared representation that connects compiler
choices, physical evidence, runtime traces, causal hypotheses, and recovery.

Fabric treats physical deployment as compilation. Its frontend is an existing
logical `DeploymentPlan`, typed model and topology graphs, calibrated fabric
profile, workload/SLO/failure policy, and budget. Its IR fixes ranks, groups,
paths, memory, overlap, and recovery variants. Its validation pass simulates
physical resources and actual workload shapes. Its runtime verifier compares
healthy and degraded evidence and tests proposed causes by replaying explicit
counterfactuals.

## 2. System model

Let a topology be a directed multigraph (G=(V,E)). Vertices represent hosts,
sockets, NUMA domains, memory, GPUs, switches, NICs, rails, storage, and optional
remote memory. Each edge (e) has direction, sharing domain, health, a latency
curve (L_e(m)), bandwidth curve (B_e(m)), confidence, and provenance for
message size (m).

A model graph is an ordered DAG of layers and communication requirements with
parameter, activation, and KV bytes. A physical candidate (x) chooses
parallelism degrees, rank/expert assignment, collective algorithms/order,
transfer routes, memory allocations, and overlap windows. It is feasible only if
degree, partition, memory, transport, reachability, architecture, runtime, budget,
and SLO constraints hold.

The compiler ranks a robust objective:

\[
J(x)=C(x)+\lambda_l V_{SLO}(x)+\lambda_t R_{tail}(x)
+\lambda_c T_{comm}(x)+\lambda_f E_{failure}(x)
+\lambda_r C_{transition}(x).
\]

The implementation exposes cost, violation, tail, communication, failure, and
transition estimates. It does not hide them in a learned model.

## 3. Physical execution IR

The v1 IR composes with the logical plan through a content-addressed document
reference. `TopologyGraph`, `ModelGraph`, and `FabricProfile` are separately
hashed. `ParallelismPlan` describes TP/PP/DP/EP/context groups, disaggregation,
and replicas. `RankPlacement` binds every rank to host, GPU, NUMA, optional NIC
and rail, CPU affinity, role, replica, and fault domain. Expert, collective, KV,
memory, and communication-overlap plans provide their respective mappings and
uncertainty.

Predicted latency, throughput, goodput, cost, availability, and communication
overhead are intervals. The envelope preserves optimizer history, rejected
alternatives, predicted bottleneck, failure exposure, evidence, compiler/Git
identity, and alternate recovery variants.

Strict Python and Rust models and JSON Schema reject unknown core fields.
Canonical JSON provides stable SHA-256. Namespaced extensions are the only open
map. v1 migrations and golden cross-language fixtures protect compatibility.

## 4. Topology characterization

Discovery first creates an observation graph. A fact can be known, unknown, or a
conflict and retains each source, command, timestamp, and confidence. Structured
OS files, cgroups, platform APIs, bounded `nvidia-smi`, network metadata, and
installed-version facts are normalized when available. Canonicalization creates
compiler nodes only from sufficient evidence.

The profiler measures shape curves rather than one peak. Each result records
primitive, bytes, ranks, concurrency, direction, process/GPU/NUMA/NIC placement,
warmup and raw samples, median/tails/MAD/bootstrap interval, command, timeout,
environment, hardware identity, success/failure, and hash. Synthetic fixtures
are explicitly labeled and cannot be loaded as measured results. The portable
measured path currently covers bounded host-memory copies.

## 5. Communication-aware simulation

The Rust twin consumes an operation DAG and physical resources. Operations cover
CPU launch, compute, HBM, P2P, collective, expert exchange, KV transfer, storage,
startup, and synchronization. Resources are exclusive or fair-share and may
participate in a sharing group representing a switch or rail.

An operation is ready when all dependencies complete and its earliest-start time
arrives. The scheduler admits it when all demanded resources have capacity. For
fair sharing, active operations divide calibrated rate; the next completion or
fault boundary becomes the next event. A collective dependency does not release
until its group work completes, making rank skew visible. Explicit stable keys
break equal-time ties.

Timed faults reduce resource/rank/collective rate or remove a resource. A
counterfactual applies transformations to a copy of the request. Validation
checks cycles, references, curve monotonicity, capacity, replacements, bounds,
and schema before execution. Output includes operation wait/service time,
resource utilization/bytes/concurrency, cost, uncertainty, and a Perfetto trace.

The model is flow-level. Packet loss and congestion-control behavior enter only
through calibrated aggregate curves.

## 6. Physical optimization

The compiler enumerates permitted parallelism/disaggregation combinations and
statically prunes invalid products. Memory feasibility includes weights, KV,
activations, runtime workspace, communication buffers, pinned memory,
fragmentation, and safety margin.

Placement strategies include exhaustive finite search, seeded random baseline,
stable sequential topology-unaware baseline, greedy topology-aware placement,
hierarchical search, and robust failure recompilation. Topology scoring uses
healthy graph paths, message-size latency, bottleneck bandwidth, host/NUMA
locality, and stable tie-breaking. Feasible candidates are compared on TTFT,
TPOT, cost, and failure exposure to construct a Pareto frontier.

The ranking pass is analytical over calibrated curves and truthfully records zero
event-simulator calls. The selected plan must pass the separate Rust simulation
and deployment-validation passes. This separation makes fidelity and cost
visible rather than labeling every estimate a simulation.

## 7. Causal diagnosis

Autopsy's event schema binds request and operation timing to host, process,
thread, rank, replica, GPU, NIC, NUMA, physical resource, counters, dependencies,
clock, and evidence hash. Cross-node alignment estimates affine offset and drift;
residual p95 plus half-RTT p95 is retained as uncertainty. Insufficient alignment
prevents fine-order claims.

Healthy and degraded runs are matched by topology, plan, workload, stage,
operation, rank, and request class. The comparator produces stage and counter
deltas, rank skew, and first divergence. Twenty-seven deterministic rules cover
queue, startup, CPU/GPU, NUMA/PCIe/NVLink/network, collective, expert,
prefill/decode/KV, health, and plan faults. Every hypothesis records support,
contradiction, confidence, and rejection.

Counterfactual replay removes a fault, rescales a physical curve/rank, or changes
a resource and reruns the exact degraded input. An improvement interval entirely
above zero supports a cause; entirely below/equal zero contradicts it; crossing
zero is inconclusive. Selection minimizes the remaining healthy-reference gap.
No LLM participates in diagnosis.

## 8. Recovery

Recovery maps the diagnosis and compiled recovery variant to scoped,
idempotent actions. The state machine requires simulation validation, builds a
replacement, shadows, canaries, promotes, and drains old workers. Confidence,
sample, SLO, error, timeout, cooldown, rollback, and abort rules are typed plan
fields. Started streams are preserved; expiry during a non-empty drain yields
operator-required rather than forced termination. A persisted snapshot and
processed idempotency keys make controller restart safe.

External mutation is false by default. Plan and executor must both authorize it,
and paid resources require a separate hard budget.

## 9. ForgeCI

ForgeCI defines a versioned matrix of repository, revision, build, environment,
hardware, model/workload/physical plan, metrics, warmups, repetitions, budget,
and timeout. Trials run in bounded process groups and retain stdout/stderr,
environment, hardware fingerprint, raw values, and hash.

Comparisons report median, MAD, p95, seeded bootstrap median/effect intervals,
Cliff's delta, robust noise floor, practical threshold, and Bonferroni-corrected
classification. Bisection uses an isolated clone, retries inconclusive commits,
preserves every candidate artifact, and emits a minimized upstream reproducer.

## 10. WarmPath

WarmPath models startup as a dependency graph of checksummed artifacts with
compatibility and security constraints. Storage tiers carry capacity, bandwidth,
latency, cost, locality, encryption/trust, and restore failure. Candidates choose
eager/lazy/rebuild/keep-warm materialization and warm replicas.

The compiler uses exhaustive search below a bound and deterministic beam search
above it. Seeded cold-start trials estimate p50/p95, failure, and stage intervals.
The local executor supports NVMe, page cache, and host memory with root-contained
paths, bounded reads, checksum verification, atomic writes, and LRU eviction.

## 11. Implementation

Python implements strict IR orchestration, topology/profile normalization, model
inspection, physical compiler, Autopsy, recovery, adapters, ForgeCI, WarmPath,
and reports. Rust implements the physical simulator and matching IR protocol.
The boundary is versioned JSON over subprocess. Runtime adapters generate local,
Docker, Kubernetes, Dynamo, Modal, and Truss artifacts offline and fail closed on
unvalidated versions or unenforceable requirements.

Tests include Python unit/property-like invariants, strict schema and artifact
checks, Rust unit/integration/property tests, analytical queue cases,
resource-conservation and malformed-plan checks, cross-language golden fixtures,
fixture topology conflicts, counterfactual diagnosis, recovery restart/rollback,
ForgeCI bisection, and WarmPath checksum/eviction.

## 12. Evaluation

### 12.1 Flagship physical fault and recovery

The CPU-only demonstration uses a synthetic two-host, 16-GPU topology, a
synthetic MoE graph, synthetic fabric curves, and 12 bursty mixed requests. It
injects network bandwidth reduction and rank-specific slowdown. According to
`artifacts/fabric-demo/manifest.json`, healthy simulated p95 TTFT was 955.883 ms,
degraded p95 TTFT 2484.092 ms, and restored p95 TTFT 955.883 ms. Degraded p99 TPOT
was 5.000 ms versus 1.250 ms healthy/restored. Seven counterfactuals selected
removing both faults; the guarded recovery completed.

Autopsy ranked `rank_straggler` first with report confidence 0.90. It did not rank
the simultaneous network fault in the top three. The combined counterfactual,
not the direct network signal, established that both modeled faults were needed
for full restoration. This is a negative multi-fault attribution result.

### 12.2 ForgeCI fixture

`artifacts/forgeci/demo/evaluation.json` reports correct first-commit bisection in
a four-commit local fixture after three unique commits, zero inconclusive rate,
and confidence 0.975. Injected p99 TTFT degradation was 12.00% with corrected
interval [11.47%, 12.53%]; throughput degradation was 10.71% with interval
[10.29%, 11.20%]. Both are synthetic process metrics.

### 12.3 WarmPath local reference

`artifacts/warmpath/manifest.json` records 243 exhaustive candidates, three eager
critical artifacts, one deferred artifact, checksum success, and measured local
execution. The payload is a deterministic synthetic snapshot; local read and
verification times are real and can vary by run.

### 12.4 Rank ordering

`reports/rank-ordering-experiment.md` records a synthetic calibrated exact search
over six orders and two message regimes. Because no matched hardware experiment
exists, the integration decision is `measure_on_hardware` and enablement is false.

### 12.5 Hardware status

The evaluation host had no NVIDIA GPU or RDMA fabric. Consequently there is no
GPU, NVLink, NCCL, InfiniBand, RoCE, DeepEP, NIXL, or real multi-node result. The
hardware paths are implemented behind optional/version-checked adapters and are
validated only by schemas, generated commands, and deterministic fixtures.

## 13. Related work

Fabric builds on Megatron-style parallelism, vLLM/SGLang serving, Dynamo
orchestration, NCCL collectives, DeepEP expert communication, NIXL transfer,
disaggregated serving systems, simulation-based capacity planning,
OpenTelemetry/Perfetto evidence formats, performance benchmark suites, and
snapshot/caching mechanisms. Its contribution is the typed composition of
physical compilation, provenance, calibrated validation, counterfactual
diagnosis, and guarded repair. A detailed comparison and primary links are in
`docs/FABRIC_RELATED_WORK.md`.

## 14. Limitations

Physical performance is synthetic on this host. Compiler search is heuristic and
analytical, and real-hardware residual calibration is absent. The flagship
selected EP=1, so it does not validate measured expert-parallel collectives.
Autopsy has multi-fault signal dilution and affine clock assumptions. Live
privileged telemetry and production recovery drivers are unexercised. Cloud
adapters are offline; Modal and Truss placement metadata is advisory. ForgeCI's
fixture is deliberately deterministic. WarmPath does not restore GPU/process or
provider-private snapshot state. Full details are in
`docs/FABRIC_LIMITATIONS.md`.

## 15. Ethics and operational safety

The system avoids paid resources and external mutation by default, labels
synthetic data, retains negative results, and keeps diagnosis independent of an
LLM. Privileged probes and host/network/GPU fault injection require separate
opt-ins. Recovery preserves started streams and offers rollback and operator
escalation. Topology and traces may reveal sensitive fleet/workload data and
require access control. Hashes provide integrity, not provenance signatures.

## 16. Future work

The next empirical requirement is matched NVIDIA multi-GPU and multi-node
calibration across message size, contention, and runtime versions, followed by
held-out prediction and diagnosis calibration. Algorithmically, simulator-guided
candidate refinement, stronger multi-fault causal interaction models, wider
model-graph inspection, and authorized Kubernetes/Dynamo action drivers are the
clearest extensions. These are not claimed as completed here.

