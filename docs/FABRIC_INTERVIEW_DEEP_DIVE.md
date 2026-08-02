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

### Keep analytical selection honest

The compiler scores finite candidates from calibrated curves, but does not call
the event simulator in its ranking loop. It emits zero simulator calls and makes
validation a separate pass. That avoids overstating fidelity and makes it clear
where future hardware-guided refinement would attach.

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
rate reduction and rank slowdown. Simulated p95 TTFT rose from 955.883 ms to
2484.092 ms and returned to 955.883 ms after the selected combined
counterfactual; seven alternatives were evaluated and recovery completed. The
important caveat is that Autopsy top-1 was the rank straggler while the network
fault did not reach top-3 in the aggregate comparison.

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

