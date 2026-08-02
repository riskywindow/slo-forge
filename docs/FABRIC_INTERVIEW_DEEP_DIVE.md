# Fabric interview deep dive

## The problem

A logical serving plan says which model/runtime and how many replicas. It does
not say which GPU each rank uses, whether a tensor group crosses a slow PCIe or
network boundary, which NUMA domain launches it, how prefill sends KV to decode,
or what plan survives a failed rail. Fabric turns those physical choices into a
versioned compiler IR and validates them against measured or explicitly synthetic
curves.

## Why this is a compiler

The frontend is logical plan + model graph + topology + fabric profile + workload
+ SLO + failure policy. Static passes prove degree, memory, reachability,
transport, and compatibility feasibility. Placement and strategy passes build
rank groups, bindings, collectives, KV routes, memory, overlap, and recovery
variants. The output is a canonical physical IR. Runtime adapters are backends;
the Rust twin and canary are validation passes.

This separation matters: Kubernetes YAML is not the product. The same physical
plan can be simulated, diagnosed, recovered, or lowered to a different runtime.

## Hard engineering decisions

### Preserve raw topology evidence

Multiple discovery sources disagree in containers. Fabric stores known, unknown,
and conflict observations before canonicalization. This prevents a missing
`nvidia-smi` from turning into "no GPU" or an RDMA device from turning into an
unverified GPUDirect path.

### Use flow-level simulation

The available measurements support message-size latency/bandwidth curves, not
packet dynamics. The simulator therefore schedules DAG operations over exclusive
and fair-share resources with explicit sharing groups. This captures barriers,
stragglers, overlap, and bottleneck contention while keeping deterministic CPU CI.

### Keep multi-fidelity selection honest

The compiler first scores the finite candidate set from calibrated curves, then
promotes the analytical Pareto set and leading recovery alternatives into the
Rust event simulator. Simulator-refined metrics and feasibility feed back into
selection, with every call and deterministic work unit retained in the optimizer
artifact. A separate loaded-trace validation pass remains necessary because the
ranking loop intentionally simulates an isolated p95-shape request, not an
engine-specific continuous-batching scheduler.

### Make RCA falsifiable

Autopsy keeps all 27 hypotheses with support and contradiction. Counterfactual
replay modifies the exact degraded simulation input. A repair whose interval
crosses zero is inconclusive, not a root cause. A language model is unnecessary.

### Recover without breaking streams

Recovery builds a replacement, shadows, canaries, promotes, and drains. It is
idempotent across controller restart and refuses to kill started streams on a
deadline. External mutation requires explicit two-sided authorization.

## What the artifacts show

The synthetic flagship in `artifacts/fabric-demo/manifest.json` injected a rail
rate reduction and rank slowdown. Simulated p95 TTFT rose from 1152.714 ms to
7045.952 ms and returned to 1152.714 ms after the selected combined
counterfactual; seven alternatives were evaluated and recovery completed.
Autopsy ranked network bandwidth degradation first and the rank straggler
second. That is one deterministic synthetic two-fault run, not evidence of
calibrated confidence or general multi-fault accuracy.

The broader artifact-derived matrix is deliberately less flattering. Across 180
synthetic plan trials, the already topology-aligned sequential fixture was
fastest at 616.428 ms median p95 TTFT; the hierarchical compiler was 1.56%
slower at 626.068 ms. The internally calibrated twin reached 0.993 rank
correlation, 0.0284% median relative error, and 73.33% interval coverage, but
coverage fell to 20% for expert-skewed traffic and the predictions share
synthetic calibration inputs with the twin. Autopsy
was top-1/top-3 correct on 24 cases from two fault families, while threshold
recovery matched diagnosis-driven restoration with a shorter declared action
time. Those results support the evidence workflow, not claims of optimizer or
controller superiority.

The offline adapter boundary is deliberately narrow. Direct vLLM and SGLang
plans reject multi-host replicas; Dynamo's tagged v1beta1 DGD is the multi-node
engine path. vLLM expert parallelism must match the runtime TP-by-DP product,
SGLang expert parallelism never implies DP attention, native vLLM NIXL transfer
flags are kept distinct from Dynamo wrapper disaggregation flags, and generic
Kubernetes multi-node output is rejected without a gang contract. Modal and
Truss receive advisory metadata only.

ForgeCI's local fixture found the exact first regressing commit with corrected
bootstrap intervals. WarmPath evaluated 243 artifact placements and executed a
checksummed local plan. The rank-ordering experiment remained disabled because
its apparent gain was synthetic-only.

## Questions an expert should ask

**How do you prevent stale calibration?** The profile carries the canonical
topology hash and compilation rejects a mismatch.

**What happens when two flows share a PCIe switch?** Their resources reference a
common `SharingGroup`; the fair-share scheduler divides capacity and records wait
and overlap.

**How is a collective different from P2P?** It names all participating ranks and
an algorithm; group completion propagates the slowest scheduled work to its
dependents.

**Does confidence mean probability of root cause?** No. It is a deterministic
ranking score. Counterfactual support and empirical accuracy evaluation are
separate.

**Why not an LLM optimizer?** The constraints and evidence need stable,
inspectable decisions. Search and curve equations can be tested and rejected
candidate-by-candidate.

**What is not validated?** Real NVIDIA/RDMA discovery and measurement,
multi-node runtime execution, production action drivers, provider snapshots, and
hardware rank-ordering benefit on the current machine.
