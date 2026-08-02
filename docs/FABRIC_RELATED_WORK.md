# Fabric related work

SLOForge Fabric combines physical compilation, a calibrated communication twin,
structured causal debugging, and guarded recovery. It uses existing runtimes and
communication libraries where possible and does not claim to replace them.

## Parallelism and placement

[Megatron-LM](https://github.com/NVIDIA/Megatron-LM) and related systems develop
tensor, pipeline, data, context, and expert parallel execution. Fabric treats
those mechanisms as a target search space: it compiles degrees, groups, rank
ordering, physical placement, paths, memory, and recovery variants but does not
reimplement distributed transformer kernels.

[Alpa](https://www.usenix.org/conference/osdi22/presentation/zheng-lianmin)
automates parallelization across devices, while
[AlpaServe](https://www.usenix.org/conference/osdi23/presentation/li-zuohan)
uses model parallelism for statistical multiplexing in serving. Fabric's narrower
claim is topology- and SLO-aware physical inference-plan synthesis with explicit
evidence, runtime validation, and online repair.

[DistServe](https://www.usenix.org/conference/osdi24/presentation/zhong-yinmin)
and [Splitwise](https://dl.acm.org/doi/10.1145/3620665.3640423) motivate separating
prefill and decode. Fabric represents that separation, binds each role to ranks,
NICs and rails, models KV transfer and backpressure, and can compile an aggregated
recovery variant. It does not replace their schedulers.

## Inference engines and distributed runtimes

[vLLM](https://github.com/vllm-project/vllm),
[SGLang](https://github.com/sgl-project/sglang), and
[NVIDIA Dynamo](https://github.com/ai-dynamo/dynamo) provide serving/runtime
mechanisms. Fabric adapters lower only validated plan fields into version-checked
runtime surfaces. Dynamo remains responsible for executing its graph; vLLM and
SGLang remain responsible for kernels, cache management, batching, and request
scheduling. `deploy/fabric/validated-versions.json` records the exact offline API
surface used.

## Communication and KV movement

[NCCL](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/) implements GPU
collectives; [DeepEP](https://github.com/deepseek-ai/DeepEP) implements expert-
parallel communication; [NIXL](https://github.com/ai-dynamo/nixl) provides a
transfer abstraction used by disaggregated runtimes. Fabric profiles or imports
their behavior, chooses algorithms/paths where exposed, and validates the
resulting plan. It does not reproduce transport protocols.

The flow-level twin is closer to a calibrated resource scheduler than a packet
network simulator. That tradeoff is deliberate: it represents shared links,
collective barriers, and contention at the fidelity supported by profiler curves.

## Serving simulators and configuration search

[Vidur](https://proceedings.mlsys.org/paper_files/paper/2024/hash/1a5c7bda70e8b79c0382914b569a8fdb-Abstract-Conference.html)
uses simulation for LLM serving capacity planning, and SLOForge's original twin
models request queues, prefill, and decode. Fabric extends the resource graph down
to ranks, collectives, KV paths, NUMA/PCIe/NVLink/NIC/rail contention, failures,
and physical recovery. The compiler uses inspectable hierarchical search rather
than claiming a novel general-purpose optimizer.

## Tracing and root-cause analysis

[OpenTelemetry](https://opentelemetry.io/docs/specs/otel/) and
[Perfetto](https://perfetto.dev/docs/) define interoperable telemetry and trace
formats. Autopsy normalizes their evidence into a versioned physical event graph,
adds uncertain cross-node alignment, matched differential signals, explicit
hypothesis rejection, and counterfactual simulator replay. Its diagnosis is
rule- and evidence-based, not LLM-generated RCA.

## Regression testing

[TorchBench](https://github.com/pytorch/benchmark) and
[MLPerf Inference](https://mlcommons.org/benchmarks/inference-datacenter/) provide
important workloads and benchmark methodology. ForgeCI focuses on commit-level
repeated trials, robust practical-significance classification, artifact-preserving
Git bisection, and minimal upstream reproducers tied to a physical-plan/hardware
fingerprint. Its normal-CI fixture is deliberately small and is not an MLPerf
submission.

## Cold start and snapshots

Container layers, page cache, warm pools, and provider snapshots all reduce cold
start. [Modal memory snapshots](https://modal.com/docs/guide/memory-snapshot) are
one provider mechanism. WarmPath instead compiles a portable artifact DAG across
storage/memory tiers with compatibility, security, cost, readiness, failure, and
eviction constraints. It can generate provider metadata where documented but
does not clone a provider runtime or claim portable GPU-state restoration.

## Claimed contribution

The contribution is the composition and evidence discipline: a typed physical
plan; provenance-preserving topology/profile inputs; deterministic communication
simulation; explainable constrained placement; structured differential and
counterfactual diagnosis; and a recovery state machine linked to the same plan
and evidence. Individual parallelism algorithms, collectives, transfer systems,
trace formats, statistical tests, and cache mechanisms are credited to their
respective systems.

