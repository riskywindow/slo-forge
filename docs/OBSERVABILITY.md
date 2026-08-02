# Observability and evidence

SLOForge instruments compiler decisions, simulation events, gateway requests, controller windows and faults. The report layer verifies hashes before deriving presentation artifacts.

## Signals

| Signal | Producer | Format |
|---|---|---|
| Raw probe samples | hardware/profiler | JSON or JSONL |
| Gateway metrics | Rust gateway | Prometheus text |
| Gateway route/request spans | Rust telemetry | Chrome trace JSON / JSONL |
| Simulation timeline | Rust simulator | Chrome/Perfetto trace JSON |
| Controller decisions | Python controller | typed JSON |
| Fault evidence | Python diagnosis | typed JSON |
| Report spans | report generator | OTEL JSON envelope |
| Compiler/evidence logs | Python/Rust | structured JSON/text logs |

Tracked quantities include TTFT, ITL, E2E, queue, startup, prefill, decode, goodput, availability, failures/rejections, backend health/outstanding work, price-derived cost, prediction error, controller actions and rollbacks.

## Metric semantics

Attempt-level gateway errors, final request availability and simulator availability are intentionally separate:

- gateway counters count every backend attempt;
- gateway replay availability is successful final requests divided by sent requests;
- simulator availability is completed requests divided by input requests under simulated deadlines/failures;
- report replay SLO attainment is `(requests - deadline_misses) / requests`.

Comparing these without naming the denominator is an analysis bug. The CPU demo, for example, had 100% gateway final success but 59.17% faulted-simulator attainment.

## Trace correlation

Request IDs appear in gateway response headers and trace arguments. Valid W3C trace context is propagated to the backend. Gateway trace phases include routing and request terminal events. Simulator records queue/prefill/decode/fault phases by replica. Controller and chaos traces use a stable demo trace ID and carry action/fault attributes.

The generated [`reports/demo/otel-traces.json`](../reports/demo/otel-traces.json) is OTEL-shaped archival JSON, not proof that an OTLP exporter delivered it to a collector. The live Rust gateway's [`gateway-trace.json`](../artifacts/demo/runtime/gateway-trace.json) is the direct runtime trace.

## Prometheus

Live gateway metrics use `sloforge_gateway_` names and bounded label sets for backend, policy, failure kind and output-started state. The report adds plan/model/controller/diagnosis gauges in [`metrics.prom`](../reports/demo/metrics.prom). Histograms use explicit buckets and include count and sum.

Do not attach request ID, prompt, tenant or arbitrary extension values as Prometheus labels; that would create unbounded cardinality and leak content.

## Artifact verification

The demo builds an index of input path, media type and SHA-256. Report generation:

1. resolves every path inside the repository;
2. recomputes each SHA-256;
3. loads and validates the `EvidenceBundle`;
4. checks the evidence plan digest against canonical plan bytes;
5. loads typed optimizer/model/controller/chaos inputs;
6. computes metrics and renders Markdown, HTML, SVG, Prometheus, OTEL and Chrome trace files.

The current report verified 15 indexed artifacts. Any modified raw input fails report generation instead of silently refreshing a number.

## Bottleneck explanation

Explanation is counter-driven. The classifier considers arrival/capacity ratio, gateway/backend queueing, warm fraction, startup, prefill/decode contribution, backend health/errors and memory rejection. Guarded ordering prioritizes concrete infeasibility and health evidence before generic overload. Each diagnosis reports the winning evidence, confidence and an approximate counterfactual improvement.

## Production guidance

- Export live metrics to a scrape target with network and auth policy outside this gateway.
- Configure trace retention and sampling; the in-process trace collector is bounded but memory resident.
- Treat prompt text as sensitive and keep it out of logs and metrics.
- Bind health/metrics endpoints to trusted networks or front them with an authenticated proxy.
- Preserve artifact hashes and plan/profile IDs in every rollout annotation.
- Alert separately on final request failure, backend-attempt error, queue saturation, SLO miss and controller action rate.
