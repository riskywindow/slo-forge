# Rust gateway

`sloforge-gateway` is the asynchronous streaming data plane. It accepts OpenAI-compatible completion and chat-completion requests, chooses a backend, forwards the request as SSE, and exposes health, metrics and trace artifacts.

## API surface

- `POST /v1/completions`
- `POST /v1/chat/completions`
- `GET /health`
- `GET /metrics`

Both inference paths accept streaming or buffered responses. Streaming emits `data: <JSON>\n\n` records and terminates with `data: [DONE]\n\n`. Buffered completions and chat responses use the corresponding OpenAI response shape. SLOForge-only deadline and priority metadata is stripped before forwarding.

The gateway preserves a valid incoming W3C `traceparent` or creates one, preserves or creates `x-request-id`, and returns both headers. Structured public errors carry type, request ID and retryability.

## Bounded resources and backpressure

Global admission uses an owned Tokio semaphore. Each backend has its own capacity semaphore. Admission capacity, backend capacity, request body size, SSE event size and trace event count are configured finite bounds. Queue acquisition and the full request have deadlines. A client that stops reading naturally backpressures the Axum body; dropping it drops the execution stream and releases admission/backend permits.

The SSE parser is incremental across arbitrary transport chunk boundaries, supports CRLF, bounds the assembled multiline event, requires valid UTF-8/JSON and recognizes `[DONE]`. An EOF without a terminal event is an error.

## Routing

Four deterministic policies are implemented:

- weighted round robin;
- least outstanding requests;
- estimated earliest finish, computed in capacity-sized waves;
- SLO-slack-aware, which treats a deadline miss as dominant and otherwise trades predicted finish against price with a priority-dependent cost weight.

Unavailable, circuit-open and explicitly excluded backends are never selected. Estimated service time is updated with successful observations.

## Failure semantics

Health checks periodically mark backends available or unavailable. Consecutive failures open a circuit breaker until cooldown. Connect, queue and request timeouts are distinct metrics/error types. A bounded retry may occur for queue/open/stream failure only before the first output event. Once output begins, malformed SSE, disconnect or timeout is returned to the client as an in-stream error and is never retried, avoiding duplicated tokens.

Graceful shutdown listens for SIGTERM or Ctrl-C, stops accepting traffic through Axum's graceful shutdown, aborts the owned health-check task on drop and writes the configured trace artifact.

## Telemetry

Prometheus output includes request and retry counters, backend error counters, route decisions, stream events, backend outstanding/health/circuit gauges and TTFT/request-duration histograms. A bounded collector emits routing and terminal request events as Chrome trace JSON or JSONL. Logs are structured through `tracing`.

The repository includes OTEL-shaped report spans, but the live gateway currently records an internal Chrome-trace representation rather than exporting spans to an OTLP collector. This distinction matters for production deployment.

## Contract tests

Tests run deterministic HTTP mock backends and cover:

- context propagation, streaming order, metrics and trace events;
- retry before output and no retry after partial output;
- malformed SSE and backend disconnects on both sides of first output;
- admission saturation and permit release after client drop;
- cancellation of partially consumed streams;
- slow consumer ordering under bounded buffering;
- malformed requests and structured errors;
- buffered chat response compatibility;
- active health removal/recovery and circuit breaker opening.

Run `cargo test -p sloforge-gateway`. No proprietary backend or network service is required.

## CPU demo behavior

<!-- Metrics sources: ../artifacts/demo/gateway/replay.json and ../artifacts/demo/gateway/metrics.prom -->

The gateway replay materialized the compiled three-replica fast-mock topology and sent the original trace at its profiled arrival timing. All 120 requests completed despite scheduled slowdown, crash/recovery and cold-start injections. It observed 170.429 ms p95 TTFT, 9.429 ms p99 ITL and 290.455 ms p95 E2E. Prometheus data records 2 backend-status errors and 2 pre-output retries on the crashed replica while the replay summary records 100% final availability. That difference is expected: attempt errors and final request outcomes are different denominators.

The demo runs on localhost with deterministic mock token generation. It validates transport and control behavior, not inference quality or public-internet security.
