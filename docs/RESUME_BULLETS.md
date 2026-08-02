# Resume bullets

<!-- All metrics below are CPU/mock only. Sources: ../reports/demo/evaluation.json, ../artifacts/demo/optimization/result.json, ../artifacts/demo/gateway/replay.json, ../artifacts/demo/simulator/replay.raw.json, and ../artifacts/demo/chaos/result.json. Do not remove that qualifier or reuse these as GPU/model-throughput claims. -->

- Built SLOForge, a Rust/Python inference deployment compiler that turns typed model, workload, hardware, cost and latency constraints into a versioned deployment plan, uncertainty-aware Pareto frontier, five offline deployment targets and a hash-verified evidence bundle.

- Implemented a deterministic Rust discrete-event LLM serving simulator covering priority queues, cold/warm replicas, chunked prefill, continuous-batching approximation, decode, deadlines, cancellation, canary traffic, capacity changes, eight fault classes and cost; processed 3,445 events for a 120-request CPU/mock replay with complete per-request and Chrome-trace artifacts.

- Engineered an asynchronous Rust OpenAI-compatible SSE gateway with bounded admission/backend queues, cancellation propagation, four routing policies, health checks, circuit breakers and pre-output-only retries; completed 120/120 localhost mock requests during scheduled slowdown, crash/recovery and cold-start injections while preserving attempt-level failure metrics.

- Created a budget-aware multi-fidelity optimizer that evaluated 540 CPU/mock configurations, analytically promoted 24 uncertainty-ranked candidates and retained a 17-point non-dominated frontier with explicit rejected-alternative diagnostics and profile-measurement provenance.

- Built predictive and reactive capacity-controller baselines with minimum-sample, cooldown, hysteresis, change-budget, canary and rollback guards; on the seed-41 CPU/mock burst trace, predictive control recorded 0 SLO-violation windows versus 1 for reactive control at 0.001833 USD higher modeled cost.

- Developed counter-based fault diagnosis for arrival overload, queueing, warm-capacity shortage, cold start, prefill/decode dominance, backend health and configuration infeasibility; correctly classified 8/8 closed-set injected CPU/mock faults with 17.54 ms mean modeled diagnosis latency, while documenting the closed-set limitation.

- Designed a reproducible evidence pipeline that verified SHA-256 for 15 raw/derived inputs before generating Markdown/HTML evaluation, Prometheus metrics, OpenTelemetry-shaped spans, Perfetto/Chrome traces and Pareto/controller plots.

## Short portfolio version

Built an open-source SLO-driven inference deployment compiler and adaptive runtime in Rust/Python, with a typed cross-language IR, deterministic digital twin, streaming gateway, budgeted Pareto optimizer, guarded controller, fault diagnosis and offline Docker/Kubernetes/Modal/Truss generation. The reproducible CPU/mock demo exercises 120 requests, 540 candidate configurations and 8 labeled faults; GPU performance is explicitly unclaimed pending compatible hardware.
