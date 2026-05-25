# Related work and differentiation

SLOForge combines ideas from predictable serving, automatic configuration, LLM runtime scheduling and deployment tooling. Its contribution is integration around a typed, evidence-bearing deployment compiler and runtime validation loop. It does not claim to invent model parallelism, continuous batching, paged KV caches, chunked prefill, prefill/decode disaggregation, predictive autoscaling or Pareto optimization.

## Serving systems

### Clockwork

[Clockwork](https://arxiv.org/abs/2006.02464) builds predictable DNN serving from controlled execution and centralized scheduling, using known model execution times to meet tight request SLOs and isolate performance. SLOForge shares the premise that measured execution structure is more useful than a black-box latency label. Clockwork is a serving system and scheduler; SLOForge compiles configurations across engines/hardware/backends and carries uncertainty, evidence and offline deployment artifacts. SLOForge's gateway is less controlling than Clockwork and does not provide its GPU-level execution predictability.

### INFaaS

[INFaaS](https://www.usenix.org/conference/atc21/presentation/romero) selects among model variants, hardware and optimizations from performance/accuracy requirements and combines VM- and model-level autoscaling. It is the closest conceptual predecessor for configuration selection. SLOForge narrows its thesis to generative inference, explicitly separates prefill/decode/startup, uses a versioned deployment IR and simulator, and targets exportable open-source plans rather than a model-less managed abstraction. SLOForge's current CPU evaluation does not reproduce INFaaS's scale or published cost results.

### AlpaServe

[AlpaServe](https://arxiv.org/abs/2302.11665) studies model parallelism as statistical multiplexing for multi-model serving and searches placement/parallelization under bursty demand. SLOForge can represent tensor/pipeline parallelism and bursty workloads, but its current topology search does not implement AlpaServe's multi-model placement optimizer. SLOForge adds deployment evidence, multiple runtime adapters and operational policies; it should use AlpaServe as a baseline when evaluating multi-model cluster placement.

### SpotServe

[SpotServe](https://arxiv.org/abs/2311.15566) adapts parallelization to preemptible GPU availability, minimizes migration cost and recovers inference progress. SLOForge models backend failures, capacity loss, cold starts, replica changes, canary and rollback, but it neither checkpoints decoder state nor migrates in-flight inference. Its cost model currently assumes declared hourly price without a spot interruption distribution. A future spot backend should compare directly rather than claim equivalent fault tolerance.

## LLM execution engines and schedulers

### vLLM

[vLLM/PagedAttention](https://arxiv.org/abs/2309.06180) addresses KV-cache fragmentation and sharing to enable high-throughput continuous batching. SLOForge uses vLLM as an optional profiled engine; it does not replace vLLM or reimplement PagedAttention. It chooses vLLM configurations and deployment policies from workload/SLO evidence.

### SGLang

[SGLang](https://arxiv.org/abs/2312.07104) provides a frontend/runtime for structured language-model programs and optimizations including RadixAttention and compressed FSMs. SLOForge's scope is deployment synthesis for request workloads, not a language for multi-call programs. Prefix sharing and adapter metadata can inform future SGLang-specific profiling, but the current optimizer does not model RadixAttention reuse in detail.

### DistServe

[DistServe](https://www.usenix.org/conference/osdi24/presentation/zhong-yinmin) disaggregates prefill and decode onto different GPUs and co-optimizes resource allocation/parallelism for TTFT and TPOT goodput. SLOForge adopts separate service curves and SLO metrics, but its current plan has one colocated replica topology and does not move KV state between prefill/decode pools. Prefill/decode disaggregation would be an engine/topology extension, with DistServe as the direct baseline.

### Sarathi-Serve

[Sarathi-Serve](https://www.usenix.org/conference/osdi24/presentation/agrawal) introduces chunked prefill and stall-free scheduling to reduce prefill/decode interference while sustaining large batches. SLOForge represents chunked prefill in its IR and simulator and searches its enablement and chunk/token limits. The scheduling technique belongs to Sarathi-Serve; SLOForge compiles whether and how an underlying runtime should use it. Its simulator is an approximation, not a cycle-accurate Sarathi implementation.

## Positioning table

| System | Primary optimization locus | SLOForge relationship |
|---|---|---|
| Clockwork | controlled execution and request scheduling | borrows measurement/predictability principle |
| INFaaS | model/hardware/optimization variant selection | closest automatic-selection precedent |
| AlpaServe | multi-model placement and parallelism | future topology/search baseline |
| SpotServe | preemptible capacity and stateful recovery | fault/cost neighbor; no state recovery here |
| vLLM | KV memory and continuous batching runtime | optional engine backend |
| SGLang | structured-program runtime and prefix reuse | optional engine backend |
| DistServe | prefill/decode disaggregation | future topology backend; separate curves already represented |
| Sarathi-Serve | chunked-prefill scheduling | represented engine setting, not invented here |

## Simulators, optimizers and deployment tools

SLOForge's simulator follows the tradition of queueing/digital-twin tools such as Vidur: measured phase curves drive faster-than-real-time discrete events. Its distinctive requirement is that the same artifacts feed compiler selection, gateway policy, chaos replay and a hash-verified evidence report. The optimizer uses conventional monotonic interpolation, residual-radius heuristics, exhaustive/random/successive-halving-style baselines and Pareto dominance. Split-conformal calibration is intended but not implemented in the reviewed revision. Novelty is not claimed for those algorithms individually.

Kubernetes, Modal and Truss already materialize and operate serving applications. SLOForge is not a replacement control plane for any of them. Its current emitters package a subset of one versioned plan and validate what can be validated without credentials; they are not yet semantics-preserving lowerings for every engine field. Billing, identity, secrets and provider lifecycle remain with the deployment environment.

## Relationship to adjacent projects

- BIFROST is a transport/storage fabric for movable KV-cache state; SLOForge could model it as a topology capability but does not transport KV state.
- Dooly is a configuration-agnostic, redundancy-aware operation profiler; SLOForge's profiler is deployment-candidate-oriented and could consume Dooly measurements.
- Switchyard routes across existing backends; SLOForge compiles and validates which backends/configurations should exist and a policy for its own gateway.

The boundary is intentional: SLOForge's central artifact is an evidence-backed `DeploymentPlan`, not a new inference kernel, proprietary-provider router or generic cluster manager.
