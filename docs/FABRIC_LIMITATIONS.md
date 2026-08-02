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

The compiler uses an inspectable finite/hierarchical analytical ranking. It does
not call the Rust simulator during candidate ranking and truthfully reports zero
simulator calls; queue/contention simulation is a separate validation pass. It
uses deterministic graph search and heuristic placement rather than MILP/CP-SAT,
and it does not actively select NCCL environment tuning from hardware trials.

The current flagship selected a 16-stage pipeline with TP=1 and EP=1. Its
synthetic model graph contains MoE experts, but the selected plan has no
expert-parallel collective operations. The demo validates cross-node KV/network
contention and rank slowdown, not measured expert-parallel execution.

Model inspection reads local configuration and safetensors metadata; it does not
trace arbitrary model code or verify every runtime partition constraint.

## Simulator

The twin is flow-level, not packet-level. Packet loss, congestion control, NIC
queue behavior, GPU clock dynamics, and OS scheduling are aggregate calibrated
effects. Prediction intervals propagate declared curve uncertainty and have not
been calibrated against real multi-node residuals on this host. Unknown regimes
can be badly estimated.

## Autopsy

Simulator capture is complete; privileged live CUPTI/DCGM/eBPF/NIC ingestion is
optional and was not exercised. Clock alignment assumes affine drift between
samples. Confidence is a deterministic ranking score, not a posterior.

The simultaneous-fault flagship ranked the injected rank slowdown first but did
not rank the injected network degradation in its top three; the combined
counterfactual was required to restore the healthy reference. Therefore the demo
does not establish perfect multi-fault top-k accuracy.

Minimization currently reduces ranks, events, and counters, not model layers or
runtime source code.

## Recovery

The guarded state machine, simulated action driver, rollback, idempotency,
restart, and stream-preserving drain are tested. No production orchestrator was
mutated. Expected improvement is counterfactual/model-derived; an authorized
runtime driver must still verify replacement readiness and real canary metrics.

## Adapters

Runtime lowerings are offline. Exact GPU UUID, NIC, and rail enforcement depends
on node runtime and device plugins. Modal and Truss accept only advisory physical
metadata in this integration. Generated artifacts are not evidence of vendor
support or successful deployment.

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

