# Deployment compiler

SLOForge treats serving configuration as compilation: a typed frontend is lowered through measurement-aware passes into a versioned deployment IR, then materialized by target backends and validated at runtime.

## Frontend

The logical input is:

```text
CompileInput = ModelSpec + Engine choices + HardwareCatalog
             + WorkloadSpec + SLOSpec + BudgetSpec
```

The CLI can build those objects from a trace, hardware probe, profile directory and compact SLO expression. Core Pydantic models are strict and immutable. Invalid precision, duplicate workload classes, inconsistent tensor parallelism, nonsensical distribution bounds and unknown fields fail before candidate search.

## Intermediate representation

The canonical document header is:

```json
{
  "schema_version": "1.0.0",
  "api_version": "sloforge.io/v1",
  "kind": "DeploymentPlan"
}
```

Important invariants include:

- engine dtype belongs to the model's allowed precision set;
- engine and batching active-sequence and token bounds agree;
- replica and engine tensor parallelism agree;
- warm capacity does not exceed maximum replicas;
- route weights and workload class weights each sum to one;
- every measured estimate cites measurement IDs;
- extension fields use keys such as `vendor.example/feature`, never anonymous dictionaries.

Canonical JSON sorts keys, removes insignificant whitespace, preserves UTF-8 and rejects non-finite values. SHA-256 of those bytes is the content address used by provenance. Writes use a temporary file, `fsync`, and atomic replacement.

## Passes

### Trace normalization

The trace frontend checks unique request IDs, nondecreasing arrivals, positive prompt/output counts, valid cancellation times and at least two classes/priorities for the generated demo workload. Trace generation is seeded.

### Hardware characterization

The CPU probe captures host identity, memory/cgroup bounds and sample arrays for memory copy and GEMM. The CUDA path explicitly requests device index zero, uses NVML/`nvidia-smi` and Torch when available, and refuses an implicit CPU fallback.

### Static feasibility

Candidate weight storage is estimated from parameter count and dtype. A conservative KV budget derived from model scale and maximum batched tokens is added with headroom. Candidates that exceed declared memory are retained as rejected alternatives with a reason and receive no expensive probes.

### Profiling and calibration

Feasible candidates advance through startup, prefill, decode and representative load probes. Warmups remain in raw data with `warmup=true`. The model stores separate monotonic prefill and decode curves and startup quantiles. At each service-curve coordinate, a seeded partition leaves a calibration subset out of curve fitting; its absolute residuals use the finite-sample conformal rank for nominal 95% coverage. Representative-load observations are a separate held-out test used to report prefill/decode MAPE and empirical interval coverage. These intervals remain configuration- and workload-scoped rather than cross-hardware guarantees.

### Candidate expansion

The current search cross-products candidate runtime, one through `max_replicas`, supported concurrency values, token budgets of 512/2048/8192, chunked-prefill setting and round-robin/earliest-finish/SLO-slack routing. Invalid combinations such as a small non-chunked token budget are pruned.

### Prediction and uncertainty

Prediction combines service curves with prompt/output quantiles, arrival rate, replica/concurrency capacity, an explicit queueing penalty, routing and batching factors, failures, price and warm capacity. The uncertainty radius is propagated into every objective and constraint. This is a systems model, not an opaque learned network.

### Constrained selection

All candidates get predicted fidelity. Acquisition ranks potential Pareto improvement plus uncertainty and penalizes constraint misses. The configured trial budget limits the proposal history; it does not manufacture measurements. Only the exact one-replica, concurrency-one, non-chunked, round-robin, 2,048-token shapes executed during profiling can use direct load measurements, so the CPU profile contributes three measured anchors. Replica, batching, concurrency, chunking and routing variants remain `predicted`. A hard constraint is satisfied only when its safety-adjusted value fits the requested bound. No feasible candidate produces an error containing the closest candidate and exact misses.

### Plan synthesis

The selected configuration becomes replica, routing, admission, batching, autoscaling, cold-start, canary and rollback policies. Metric estimates cite only supporting stages: prefill/load for TTFT, decode/load for ITL, startup for cold start, and load for availability, throughput, goodput and derived cost. Hardware has a separate reference, nominal coverage remains in the curve model, and empirical held-out coverage becomes the estimate confidence field. Rejected candidates and every acquisition step are copied into evidence.

## Backends

- `local` emits gateway JSON and a launcher.
- `docker` adds a Dockerfile, Compose services, health checks and a model-cache volume.
- `kubernetes` emits a Helm chart with probes, resources, Prometheus annotations, rolling-update controls and HPA.
- `modal` emits a pinned Python app using current class/lifecycle/web decorators, engine-specific Transformers/vLLM/SGLang/TensorRT-LLM/mock loading and generation paths, and offline import/AST validation.
- `truss` emits `config.yaml` plus an engine-specific `Model` implementation and requirements, validates against the vendored schema subset, and uses the installed Truss handle when available.

These generators materialize intent; they do not deploy. Cloud execution is outside normal CI and requires explicit credentials and budget authorization.

## Validation

Validation happens at four boundaries: strict Python construction, JSON Schema validation, strict Rust deserialization/semantic validation, and target-specific exporter checks. Historical `v1alpha1` fixtures migrate through explicit renames. Unknown schema versions fail. The stable subprocess protocol is documented in [Architecture](ARCHITECTURE.md).

## What compilation does not prove

A plan is evidence-scoped. Passing the compiler says the document is internally consistent and selected under the supplied measurements and model. It does not prove performance on an unprofiled workload, driver, accelerator, region or runtime version. Runtime replay is part of validation, not a ceremonial post-step.
