# SLOForge: Compiling Inference Deployments from SLOs and Evidence

## Abstract

Generative-model serving exposes a large configuration space spanning inference runtime, precision, parallelism, batching, concurrency, replica topology, routing, warm capacity and autoscaling. A configuration that maximizes isolated throughput can violate time-to-first-token (TTFT), inter-token latency (ITL), availability or cost objectives under a mixed, bursty workload. Existing serving engines optimize important mechanisms inside a runtime, while deployment platforms materialize operator-supplied configuration. The missing layer is a reproducible process that converts workload, hardware, price and service-level objectives into an executable deployment plan and validates that plan under queueing and faults.

We present SLOForge, an open-source inference deployment compiler and adaptive runtime. SLOForge defines a typed, versioned `DeploymentPlan` intermediate representation; performs staged budgeted profiling; calibrates separate monotonic prefill, decode and startup models with uncertainty; searches configurations under hard constraints; and emits a Pareto frontier, selected plan, rejected alternatives and content-addressed evidence. A deterministic Rust discrete-event simulator models priority queues, replicas, cold starts, chunked prefill, approximate continuous batching, decode, deadlines, failures and cost. A separate asynchronous Rust gateway implements OpenAI-compatible streaming with bounded admission, cancellation propagation, health checks, circuit breakers and SLO-aware routing. Python components implement controller baselines, fault diagnosis, deployment backends and hash-verified reports.

On a 120-request CPU/mock workload, SLOForge evaluated 540 configurations, promoted 24 profile-derived trials and selected a 17-point feasible frontier. The live gateway completed 120/120 localhost requests during injected faults. The faulted simulator nevertheless attained only 59.17% of deadlines, while the selected steady-profile estimate had satisfied its requested TTFT/ITL bounds. This negative result illustrates the central design claim: deployment selection without evidence-scoped runtime validation is insufficient. No compatible NVIDIA hardware was available, so we make no claim about model or GPU performance.

## 1. Motivation

An inference deployment is a coupled systems configuration. Increasing concurrency may improve throughput but increase ITL; large batches improve accelerator utilization but delay short interactive requests; reducing warm capacity saves idle cost but exposes cold-start tails; adding replicas lowers queueing but changes cost per token; a runtime setting that is optimal for short prompts may be dominated for long contexts. Hardware SKU labels alone do not capture memory bandwidth, startup state, library version or topology.

Users often address this space with a benchmark spreadsheet and hand-written YAML. That workflow loses raw samples, hides rejected alternatives and separates offline tuning from production routing and rollback. A generic proxy cannot recover those decisions from request traffic, and a cloud exporter cannot infer them from an image.

SLOForge frames the problem as compilation:

```text
frontend:  model + engine choices + workload + hardware + SLO + budget
passes:    feasibility -> profile -> calibrate -> predict -> optimize -> validate
IR:        versioned DeploymentPlan plus EvidenceBundle
backends:  local, Docker, Kubernetes, Modal, Truss
runtime:   simulator, gateway, controller, fault diagnosis and rollback
```

This framing creates explicit semantics and a review boundary. A plan says not only which runtime to launch, but also the measurements and uncertainty supporting its limits, its admission and routing rules, and the conditions under which it should change or roll back.

## 2. Goals and non-goals

SLOForge aims to: (1) make configuration synthesis typed and reproducible; (2) spend measurement budget progressively; (3) model prefill, decode and startup separately; (4) select under tail-latency, goodput, availability and cost constraints; (5) validate queue/fault behavior in deterministic replay and a real streaming data plane; and (6) materialize multiple deployment targets without cloud dependency.

It is not a foundation model, inference engine, KV-state transport, proprietary API router, complete Kubernetes controller or SaaS platform. vLLM, SGLang, Transformers and TensorRT-LLM remain engines. SLOForge profiles and configures them. It also does not claim to invent continuous batching, chunked prefill, prefill/decode disaggregation, Pareto search or predictive scaling.

## 3. Architecture

Python owns orchestration, profiling adapters, calibration, optimization, control, exporters and reporting. Rust owns shared wire types, the simulator, load generator, gateway and low-latency telemetry. The canonical language boundary is versioned JSON over a bounded subprocess. HTTP/SSE exists only in the running data plane.

```text
                    +---------------------------+
 specs + trace ---> | Python compiler/profiler  |
 hardware probe --> | model + optimizer         |
                    +-------------+-------------+
                                  |
                        plan + evidence + hashes
                       /          |              \
                      v           v               v
              +-----------+ +-----------+ +---------------+
              | Rust sim  | | Rust HTTP | | target export |
              | event Q   | | SSE gw    | | and validate  |
              +-----+-----+ +-----+-----+ +---------------+
                    |             |
                    +------+------+
                           v
                 controller + chaos + report
```

The subprocess boundary adds serialization only to experiment/control operations, not tokens in flight. It makes inputs archivable, diffable, hashable and independently executable. Rust and Python golden fixtures verify compatible schema and canonical hashing.

### 3.1 Intermediate representation

The `sloforge.io/v1` plan contains `ModelSpec`, `EngineSpec`, `HardwareSpec`, `WorkloadSpec`, `SLOSpec`, `BudgetSpec` and deployment policies. Strict types reject unknown fields. Cross-field validators enforce dtype permission, batching/engine agreement, tensor-parallel agreement and topology/capacity consistency. Namespaced extensions are the only extensibility mechanism.

Every metric estimate stores a point, lower/upper bound, empirical coverage value, unit, sample count and stage-specific measurement IDs. TTFT binds prefill/load probes, ITL binds decode/load probes, startup binds startup probes, load-derived metrics bind load probes, and hardware has its own reference. Requested nominal coverage stays in the curve model rather than being relabeled as achieved confidence. Provenance binds profile, optimizer run, workload digest, hardware fingerprint, evidence URI, compiler version and Git state. The `EvidenceBundle` includes environment, assumptions, measurement references, calibration metrics, optimizer decisions, rejected candidates, benchmark results and artifact hashes.

Canonical JSON sorts object keys, uses compact separators, preserves UTF-8 and rejects non-finite values. SHA-256 of those bytes is the cross-language content identifier. Known alpha documents migrate through explicit lossless renames; unknown versions fail.

## 4. System model

Let a request (r) arrive at time (a_r), with prompt length (p_r), requested output length (o_r), class (c_r), priority π_r and optional deadline (d_r). A candidate configuration (x) selects engine/hardware properties, replica count (R_x), per-replica concurrency (C_x), batch token limit (B_x), chunking and routing. The profiler supplies:

```text
P_x(p, b)      prefill service time
D_x(n, ctx)    one decode-step service time
S_x(cache)     startup-time distribution
F_x            failure probability
H_x            hourly price
```

The primary constraints are tail quantiles of TTFT, ITL and E2E, deadline/fluidity goodput, availability and cost. For a metric center μ and uncertainty radius (u), an upper-bound constraint (L) is admitted when:

```text
μ + k u <= L
```

and a lower bound (G) when μ - k u >= G, where (k) is the configured safety multiplier. The optimization objective is evaluated only among candidates satisfying all conservative constraints.

### 4.1 Queue approximation

For trace duration (T), arrival rate is (N/T). A candidate's approximate request capacity derives from replicas, active sequences and p95 service E2E. The implementation uses an explicit nonlinear queue multiplier as utilization approaches one, with measured prefill/decode terms and transparent routing, batching and chunking factors. This approximation is deliberately inspectable and is later challenged by discrete-event replay.

### 4.2 Cost and goodput

Predicted output throughput is the lesser of arrivals and service capacity multiplied by mean output length. Deadline-aware goodput counts output tokens for requests whose predicted TTFT plus subsequent ITLs fit their deadline. Cost per million tokens is replica hourly cost divided by hourly output tokens. Simulator cost integrates declared replica price across simulated duration.

## 5. Profiling and calibration

Profiling advances through five fidelities. Static feasibility estimates weight and KV memory and preserves rejected candidates. Hardware characterization records raw sample arrays and a fingerprint. Startup probes include process/import/readiness and cache state. Short probes build prompt-length prefill and active-sequence decode grids. Representative traffic captures TTFT, ITL, E2E, availability and goodput on trace shapes.

Every sample records candidate, stage, shape, latency, warmup/failure flags and seed. A mutable budget meter reserves projected seconds and dollar cost before each probe. Exceeding either bound aborts rather than silently truncating a candidate.

For each dimension, non-warm successful samples are grouped by coordinate. A seeded split holds out calibration observations at every coordinate, fits a monotonic median to the remaining observations, interpolates within the grid and extrapolates with nonnegative terminal slope. The radius is the finite-sample conformal rank over absolute calibration residuals at nominal 95% coverage. Representative-load observations are a separate held-out test for prefill/decode MAPE and empirical coverage. The small coordinate-stratified split does not imply independent requests or cross-hardware generalization, so nominal and achieved coverage remain separate.

Real adapters exist for Transformers, vLLM and SGLang on explicitly requested CUDA hardware. They parse incremental OpenAI SSE, bound server readiness and requests, collect Torch/Perfetto traces and can generate Nsight commands. These adapters were not exercised in the present CPU study.

## 6. Configuration optimization

The optimizer enumerates valid topology/concurrency/batch/chunking/routing combinations. Each configuration receives a stable content-derived ID. Low-fidelity prediction computes all metrics and uncertainty. Acquisition ranks direction-aware objective improvement plus uncertainty, with penalties for constraint misses. The proposal budget records a budgeted acquisition prefix. Direct measured fidelity is reserved for exact profiler load-test shapes; scaled topology, batching and routing configurations remain predictions.

Pareto dominance minimizes TTFT, ITL, E2E, cost and cold-start p95 while maximizing goodput, throughput and availability. A point is retained when no other feasible point is no worse in every metric and strictly better in at least one.

Four outcomes are reported: exhaustive, seeded random, successive halving and uncertainty-aware acquisition. All failed constraints and null incumbents remain visible. These baselines expose entries from one shared predicted table and are therefore search-plumbing comparisons, not measurement-efficiency evidence. Fresh independently budgeted full-configuration trials are required for a strong H1 claim.

## 7. Discrete-event digital twin

The Rust simulator uses an ordered event heap and ChaCha8 sampling. Events cover arrival, startup completion, prefill chunks, decode iterations, deadline, cancellation and timed actions. Replica state includes warm/healthy flags, bounded queue, maximum active sequences, service multiplier, request error probability, network delay and cost.

Routing supports round robin, least outstanding, earliest finish and SLO slack. Requests progress through queue, prefill and iterative decode; approximate continuous batching charges steps against the active set and context distribution. Terminal outcomes distinguish completion, deadline, cancellation, rejection, backend failure and simulated OOM.

Fault actions include crash/recovery, service/startup slowdown, request errors, replica add/remove, capacity/queue loss, network latency/jitter and context OOM. Output contains per-request phase timings, aggregate tails/goodput/cost, processed-event count, Chrome trace events and provenance. Tests cover exact single-server timing, deterministic bytes, priorities, deadlines, cancellation, crash rerouting, dynamic capacity, routing and faster-than-wall-clock execution.

## 8. Streaming gateway

The Axum/Tokio gateway accepts completion and chat requests and returns SSE or buffered responses. A global semaphore bounds admitted requests; a semaphore per backend bounds work. Queue, connect, health and request timeouts are finite. The incremental SSE parser bounds assembled events and rejects malformed/incomplete streams.

Routing observes backend health, outstanding work, capacity, learned service time, price, request priority and remaining deadline slack. Health checks and a failure-threshold/cooldown circuit breaker remove bad backends. Retries are bounded and legal only before output begins. Dropping a client body drops its execution stream and capacity leases, propagating cancellation.

Prometheus metrics expose request/attempt/route/stream counters, health/outstanding/circuit gauges and TTFT/E2E histograms. A bounded in-process collector writes routing and request Chrome trace events. Contract tests exercise partial SSE, disconnects, cancellation, slow consumers, saturation, health recovery and circuit opening.

## 9. Adaptive control and diagnosis

The predictive baseline constructs an exponentially weighted arrival forecast plus positive trend and an upper uncertainty envelope. It evaluates replicas, concurrency and routing against a TTFT queue model and chooses the least-cost action inside a safety margin. Minimum samples, cooldown and hourly change budget can force a hold. Routing/variant changes enter canary and restore the prior action if observed TTFT exceeds the promotion limit.

The reactive baseline changes one replica across utilization hysteresis thresholds. Both produce complete decision records with state, forecast, alternatives, safety checks, action, outcome and rollback. For the CPU experiment, their selected replica/concurrency actions are converted to timed actions in otherwise identical calibrated Rust-simulator scenarios; request deadline misses and integrated cost are taken from those replays rather than the controller's internal TTFT estimate.

Fault diagnosis consumes phase timings, queues, arrivals/capacity, warm fraction, backend health/errors and memory rejections. Guarded rules distinguish overload, gateway queueing, insufficient warm capacity, cold start, prefill, decode, unhealthy backend and infeasible configuration. The evaluation pairs injected cases with no-fault negative windows, emits a confusion matrix and directly times classifier execution; this is not end-to-end detection latency. The explanation contains observed counters and approximate counterfactual improvement; no LLM is involved.

## 10. Deployment and reporting

Offline emitters produce local launcher, Docker/Compose, Kubernetes Helm, Modal and Truss artifacts. Target validators parse configs, inspect required probes/decorators, compile generated Python and optionally invoke local tools. They never deploy. Modal/Truss generation pins SDK/runtime dependencies and emits engine-specific Transformers, vLLM, SGLang, TensorRT-LLM or explicit-mock loading and generation code. These are still partial provider lowerings: non-mock cloud execution, distributed topology and the full controller contract remain unexercised.

The report generator accepts an evidence bundle and SHA-256 artifact index. It resolves repository-local paths, recomputes all hashes, validates the canonical plan digest, loads typed inputs and only then renders Markdown, HTML, SVG plots, Prometheus text, OTEL-shaped archival JSON and Chrome trace JSON. A static TypeScript UI parses the generated report artifact with runtime validation and visualizes plan/frontier/controller/fault data without a service or database.

## 11. Evaluation

### 11.1 Method

<!-- All evaluation values in this section come from ../reports/demo/evaluation.json, ../artifacts/demo/optimization/result.json, ../artifacts/demo/gateway/replay.json, ../artifacts/demo/simulator/replay.raw.json, ../artifacts/demo/controller/evaluation.json, ../artifacts/demo/chaos/result.json and ../artifacts/demo/hardware/local-cpu.json. -->

The evaluated host was an Apple M4 Pro with 12 logical CPUs and 24 GiB RAM on macOS 15.6.1 arm64. The CPU probe measured 74.72 GB/s median host copy and 1,637.29 GFLOP/s median FP32 384 GEMM. Three Rust HTTP mock backends represented balanced, latency-oriented and economy service curves. A seeded bursty workload contained 120 short-interactive and long-context requests with multiple priorities and output lengths.

This setup tests orchestration, protocols, queueing, faults and provenance. It does not execute a language model and cannot answer GPU efficiency hypotheses.

### 11.2 Profiling and prediction

The profiler spent 41.460405 modeled seconds and 0.013295 USD within a 180-second/0.20-USD budget. Candidate p95 TTFT ranged from 199.78 ms (fast) to 515.92 ms (economy). The selected balanced model had 7.57% held-out prefill MAPE and 54.17% interval coverage. Coverage is substantially below a typical 95% target and is a negative calibration result.

### 11.3 Search

Search evaluated 540 configurations and promoted 24, yielding 17 feasible non-dominated points. The selected configuration used balanced mock service, 2 replicas, concurrency 4, 512 batched tokens, chunked prefill, SLO-slack routing and 1 warm replica. Its profile-derived metrics were 222.812 ms p95 TTFT, 5.609 ms p99 ITL, 450.173 tokens/s throughput, 97.452 deadline-aware goodput tokens/s and 1.35750 USD/million tokens.

Exhaustive, seeded random and uncertainty-aware search reached the same objective in this small space. Successive halving found no feasible incumbent in 24 trials. H1 is therefore not supported by this run.

### 11.4 Live gateway and simulation

Live gateway replay completed all 120 requests while injecting slowdown, crash, recovery and cold-start commands. Final p95 TTFT was 1,605.45 ms and p99 ITL 12.35 ms. Attempt metrics recorded queue errors and retries, demonstrating why final success and backend-attempt health require separate denominators.

The calibrated faulted simulator processed 3,445 events across 22.25 simulated seconds, completed 71 requests, recorded 4 backend failures and 49 deadline misses. It reported 5,035.21 ms p95 TTFT, 16.02 ms p99 ITL, 332.04 tokens/s throughput, 59.17% availability/attainment and 0.01360 USD cost. The gap from the selected steady-profile point is the main result: selection evidence did not cover the injected fault/load regime.

### 11.5 Controller and diagnosis

Across 16 windows, predictive control made 2 scale actions with 0 SLO-violation windows, 1 cold exposure and 0.015889 USD modeled cost. Reactive control made 2 actions with 1 violation, 2 cold exposures and 0.014056 USD cost. The one-seed result suggests a cost/reliability trade, not general superiority. No canary or rollback occurred.

The closed-set diagnosis evaluator applied all 8 fault types and matched all expected labels, with 0 false positives and 17.54 ms mean modeled diagnosis latency. Because the injector and classifier share known counter semantics, this establishes deterministic plumbing, not production incident accuracy.

### 11.6 Hypotheses

| Hypothesis | Current result |
|---|---|
| H1: multi-fidelity reduces GPU measurement | not established; CPU strategies consume one shared prediction table rather than independent paid trials |
| H2: compiled plan beats defaults | not established; default/manual/compiled rows are calibrated predictions, not measured real-engine comparisons |
| H3: predictive handles bursts better | one-seed matched Rust-twin comparison; live-provider and multi-seed validity unmeasured |
| H4: injected bottlenecks are identified | closed-set label agreement plus no-fault controls; external validity unmeasured |

## 12. Related work

Clockwork builds predictable model serving from controlled execution; INFaaS automatically selects model/hardware/optimization variants; AlpaServe searches model-parallel placement for statistical multiplexing; SpotServe adapts distributed serving to preemptible capacity and recovers state. SLOForge shares their concern for SLO-aware configuration but centers a portable versioned plan, uncertainty/evidence and runtime validation. It does not match their cluster-scale scheduling or state-recovery mechanisms.

vLLM contributes PagedAttention and KV-cache-efficient continuous batching; SGLang contributes structured-program execution and prefix-aware runtime mechanisms. SLOForge treats both as engines. DistServe demonstrates prefill/decode disaggregation for TTFT/TPOT goodput; Sarathi-Serve contributes chunked prefill and stall-free scheduling. SLOForge models separate phases and represents chunking but currently uses colocated topology and approximate scheduling.

The full comparison and primary links are in [`docs/RELATED_WORK.md`](../docs/RELATED_WORK.md).

## 13. Limitations

The present evidence is CPU/mock and single-seed. Real GPU engines and cloud targets are implemented paths but unexercised. Nominal and empirical coverage differ, especially for decode. Only three exact load-test shapes are measured; the rest of the configuration space is predicted. The simulator omits KV transfer, cache eviction, collective contention and cycle-level GPU overlap. Controller actions are compared in the Rust twin rather than applied to live provider capacity. Diagnosis is closed set despite negative controls. The gateway needs trusted ingress for TLS/auth/quotas. Modal and Truss have engine-specific code generation but have not executed their non-mock paths here.

These limitations are not incidental footnotes: they bound every reported number and motivate the validation-first architecture.

## 14. Future work

The next evaluation should run multiple seeds on a small licensed open-weight model across Transformers, vLLM and SGLang, with fresh full-configuration trials and held-out burst traces. Modeling work should increase independent calibration data, add E2E/startup coverage and explicitly model heteroscedastic residuals. Simulator work should add prefill/decode pools, KV-transfer bandwidth, cache state and collective contention. Control work should apply actions to live local replicas and force canary failure/rollback. Exporter work should execute each generated engine path in pinned no-cloud environments before any provider deployment.

A focused Triton fused-logits experiment is present behind an explicit opt-in flag, with reference correctness tests, warmup and robust timing. It must remain disabled until a compatible GPU run demonstrates benefit in its intended regime; absence of a GPU result is not a speedup.

## 15. Conclusion

SLOForge demonstrates that inference deployment can be treated as an evidence-bearing compilation problem. Its strongest current finding is a failure: a configuration selected under representative steady measurements did not maintain deadlines under faulted replay. By making plan assumptions, uncertainty, raw measurements, runtime traces and rejected alternatives part of one reproducible artifact chain, SLOForge creates a foundation for improving that outcome without hiding it.

## References

1. A. Gujarati et al. [Serving DNNs like Clockwork: Performance Predictability from the Bottom Up](https://arxiv.org/abs/2006.02464). OSDI 2020.
2. F. Romero et al. [INFaaS: Automated Model-less Inference Serving](https://www.usenix.org/conference/atc21/presentation/romero). USENIX ATC 2021.
3. Z. Li et al. [AlpaServe: Statistical Multiplexing with Model Parallelism for Deep Learning Serving](https://arxiv.org/abs/2302.11665). OSDI 2023.
4. X. Miao et al. [SpotServe: Serving Generative Large Language Models on Preemptible Instances](https://arxiv.org/abs/2311.15566). ASPLOS 2024.
5. W. Kwon et al. [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180). SOSP 2023.
6. L. Zheng et al. [SGLang: Efficient Execution of Structured Language Model Programs](https://arxiv.org/abs/2312.07104). 2024.
7. Y. Zhong et al. [DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving](https://www.usenix.org/conference/osdi24/presentation/zhong-yinmin). OSDI 2024.
8. A. Agrawal et al. [Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve](https://www.usenix.org/conference/osdi24/presentation/agrawal). OSDI 2024.
