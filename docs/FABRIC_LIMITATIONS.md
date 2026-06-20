# Fabric limitations

This document distinguishes implemented code from validation performed on the
current machine.

## Hardware validation

The available host was Apple Silicon without NVIDIA GPU, CUDA, NVLink, NCCL,
RDMA, InfiniBand, or RoCE. Current-host CPU/NUMA/network discovery and measured
host-memory/local-file paths ran. GPU and multi-node transport paths are typed,
version-checked, and fixture-tested but unexercised on hardware. The flagship
fabric curves are explicitly synthetic calibrated values.

## Physical compiler

The compiler uses an inspectable finite/hierarchical analytical ranking and then
refines its winner, Pareto set, and leading recovery alternatives in the Rust
physical simulator. The refinement workload is one isolated p95-shaped request;
representative open-loop queueing remains a separate, fail-closed `fabric
validate` pass. Placement uses deterministic graph search and heuristics rather
than MILP/CP-SAT, and collective environment choices have not been selected from
hardware trials.

The flagship intentionally fixes TP=8, PP=1, DP=2, and EP=4 for both aware and
unaware compiles. This isolates rank-placement behavior and guarantees that its
prefill/decode KV route crosses the synthetic two-node fabric. It exercises MoE
collective lowering, cross-node KV/network contention, and rank slowdown, not
measured expert-parallel execution or proof that those degrees are optimal.

Model inspection reads local configuration and safetensors metadata; it does not
trace arbitrary model code or verify every runtime partition constraint.

The IR identifies PP rank groups but does not explicitly map layers to stages or
emit PP send/receive schedules. PP output is placement intent rather than a
complete executable layer schedule. NIC selection is a deterministic
same-NUMA/high-speed heuristic, not a globally optimal GPU-to-NIC/rail solve.

## Simulator

The twin is flow-level, not packet-level. Packet loss, congestion control, NIC
queue behavior, GPU clock dynamics, and OS scheduling are aggregate calibrated
effects. Prediction intervals propagate declared curve uncertainty and have not
been calibrated against real multi-node residuals on this host. Unknown regimes
can be badly estimated.

Python lowering represents decode as one calibrated operation rather than a
per-token/per-layer DAG, and it does not materialize layer-level expert exchange,
pipeline bubbles, or PP send/receive. Multi-hop path duration is conservatively
additive rather than cut-through. Canonical profiles currently group several
collective primitives, limiting algorithm-specific calibration.

## Autopsy

Simulator-operation capture is the only exercised capture path; it currently
does not reconstruct parent/dependency edges from simulator outcomes.
Privileged live CUPTI/DCGM/eBPF/NIC ingestion is optional and was not exercised.
Clock alignment assumes affine drift between samples. Confidence is a
deterministic ranking score, not a posterior.

The simultaneous-fault flagship ranked network bandwidth degradation first and
the rank slowdown second; the combined counterfactual was required to restore
the healthy reference. This is one deterministic two-fault interaction and does
not establish calibrated diagnosis confidence or broad multi-fault accuracy.

Minimization currently reduces ranks, events, counters, and fault intervals, not
model layers, physical parallelism, runtime commands, or runtime source code.

## Recovery

The guarded state machine, simulated action driver, rollback, idempotency,
restart, and stream-preserving drain are tested. No production orchestrator was
mutated. Expected improvement is counterfactual/model-derived; an authorized
runtime driver must still verify replacement readiness and real canary metrics.
`ROLLED_BACK` is a guarded traffic/state transition in the local reference; the
executor does not apply general inverse infrastructure actions.

## Adapters

Runtime lowerings are offline. Exact GPU UUID, NIC, and rail enforcement depends
on node runtime and device plugins. Modal and Truss accept only advisory physical
metadata in this integration. Generated artifacts are not evidence of vendor
support or successful deployment. Direct vLLM/SGLang lowerings reject
multi-host replicas; Dynamo DGD is the supported multi-node engine path. Generic
Kubernetes export rejects multi-node physical plans because it does not emit an
atomic gang contract. The DGD subset has not been admitted by a live operator.

Profile/plan binding currently hashes capture-bearing canonical topology fields,
including timestamps. This safely rejects stale evidence but also prevents reuse
after an otherwise equivalent fresh discovery; compatibility-aware reuse is not
implemented.

## ForgeCI and WarmPath

ForgeCI's demonstrated bisection uses an intentionally deterministic four-commit
CPU fixture, not a large external runtime under real hardware noise. WarmPath
uses synthetic snapshot bytes and local NVMe/page-cache/host-memory execution;
GPU HBM/process snapshot restore, peer serving, remote tiers, and cloud snapshot
execution were not exercised.

## Low-level experiment

The rank-ordering experiment found a large benefit only on synthetic calibrated
curves. Its integration decision is `measure_on_hardware` and it is disabled by
default. No production speedup is claimed.

## Checked synthetic evaluation

The 180-trial H1 matrix did not show a win over its already topology-aligned
sequential fixture: the sequential median p95 TTFT was 616.434 ms and the
hierarchical compiler was 626.120 ms, 1.57% slower. The twin's rank correlation
was 0.881, but median relative error was 28.35% and prediction-interval coverage
was 0%. The latter is a calibration failure, not evidence that the intervals are
conservative.

Autopsy top-1/top-3 accuracy was 1.0 across 24 deterministic cases from only two
fault families; the 0.925 median score is not a probability. Diagnosis-driven
and threshold policies both restored all modeled cases, while threshold recovery
had the shorter declared action time. Recovery cost/time inputs are policy
constants and post-action request behavior is simulated. None of these results
generalize to real hardware or production recovery.
