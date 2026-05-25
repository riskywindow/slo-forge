# Resume bullets

<!-- All metrics below are CPU/mock only. Sources: ../reports/demo/evaluation.json, ../artifacts/demo/optimization/result.json, ../artifacts/demo/gateway/replay.json, ../artifacts/demo/simulator/replay.raw.json, ../artifacts/demo/controller/evaluation.json, ../artifacts/demo/chaos/result.json, and ../artifacts/demo/evidence/artifact-index.json. Do not remove that qualifier or reuse these as GPU/model-throughput claims. -->

- Built SLOForge, a Rust/Python inference deployment compiler that turns typed model, workload, hardware, cost and latency constraints into a versioned deployment plan, uncertainty-aware Pareto frontier, five offline deployment targets and a hash-verified evidence bundle.

<!-- Source: ../artifacts/demo/simulator/replay.raw.json -->

- Implemented a deterministic Rust discrete-event LLM serving simulator covering priority queues, cold/warm replicas, chunked prefill, continuous-batching approximation, decode, deadlines, cancellation, canary traffic, capacity changes, eight fault classes and cost; processed 4,970 events for a 120-request CPU/mock replay with complete per-request and Chrome-trace artifacts.

<!-- Source: ../artifacts/demo/gateway/replay.json and ../artifacts/demo/gateway/metrics.prom -->

- Engineered an asynchronous Rust OpenAI-compatible SSE gateway with bounded admission/backend queues, cancellation propagation, four routing policies, health checks, circuit breakers and pre-output-only retries; completed 120/120 compiled-topology localhost mock requests at 170.429 ms p95 TTFT during scheduled slowdown, crash/recovery and cold-start injections while preserving 2 recoverable backend errors.

<!-- Source: ../artifacts/demo/optimization/result.json -->

- Created a budget-aware uncertainty-ranked optimizer that evaluated 540 CPU/mock configurations, distinguished 3 direct profile anchors from 537 predictions, recorded 24 acquisition proposals and retained a 4-point non-dominated selection frontier with 27 feasible alternatives and explicit rejection diagnostics.

<!-- Source: ../artifacts/demo/controller/evaluation.json -->

- Built predictive and reactive capacity-controller baselines with minimum-sample, cooldown, hysteresis, change-budget, canary and rollback guards; in matched seed-41 Rust-twin scenarios, predictive control recorded 0 deadline misses versus 2 for reactive control at 0.001458 USD greater simulated cost.

<!-- Source: ../artifacts/demo/chaos/result.json -->

- Developed counter-based fault diagnosis for arrival overload, queueing, warm-capacity shortage, cold start, prefill/decode dominance, backend health and configuration infeasibility; matched 8/8 closed-set CPU/mock injections, rejected 8/8 no-fault controls and measured 0.005666 ms mean classifier execution while documenting that this is not field detection latency.

<!-- Source: ../artifacts/demo/evidence/artifact-index.json -->

- Designed a reproducible evidence pipeline that verified SHA-256 for 27 indexed raw/derived inputs before generating Markdown/HTML evaluation, Prometheus metrics, OpenTelemetry-shaped spans, Perfetto/Chrome traces and Pareto/controller plots.

## Short portfolio version

Built an open-source SLO-driven inference deployment compiler and adaptive runtime in Rust/Python, with a typed cross-language IR, deterministic digital twin, streaming gateway, budgeted Pareto optimizer, guarded controller, fault diagnosis and offline Docker/Kubernetes/Modal/Truss generation. The reproducible CPU/mock demo exercises 120 requests, 540 candidate configurations and 8 labeled faults; GPU performance is explicitly unclaimed pending compatible hardware.
