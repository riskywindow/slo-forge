# Deterministic simulator

`sloforge-sim` is a Rust discrete-event digital twin calibrated from profile artifacts. It is a standalone binary and library, with `sloforge.runtime.simulator` providing the bounded Python subprocess wrapper.

## Model

Simulation input contains a seed, one measured service curve, replicas, requests, timed actions, routing policy, canary weight and maximum event count. The service curve separates:

```text
prefill_ms = intercept
           + prompt_tokens * prefill_ms_per_prompt_token
           + batch_items * prefill_ms_per_batch_item
           + chunks * chunk_overhead_ms

decode_step_ms = intercept
               + active_sequences * decode_ms_per_active_sequence
               + context_tokens * decode_ms_per_context_token
```

Startup is constant, uniform or empirical. Sampling uses seeded ChaCha8. Python derives slopes and intercepts from the selected candidate's calibrated points and carries the measurement artifact URI into simulator provenance.

## Event semantics

The priority queue orders arrivals, startup completion, prefill chunks, decode steps, cancellations, deadlines and scenario actions. Each request is assigned to a replica queue using round robin, least outstanding, estimated earliest finish or SLO-slack-aware routing. Lower numerical arrival time wins; request priority and a deterministic sequence break ties.

A warm replica may immediately schedule work. A cold replica samples startup before serving. Prefill may run in chunks. Decode approximates continuous batching by charging a step from the current active set and context. Terminal states are completed, deadline exceeded, cancelled, rejected, backend failed or simulated OOM.

The output reports per-request arrival, queue, prefill, decode, TTFT, ITL, completion, generated tokens and deadline status, plus aggregate tail percentiles, throughput, goodput, availability, cost and event count. Trace events are Chrome/Perfetto-compatible microsecond records.

## Dynamic actions and faults

The engine supports backend crash/recovery, service slowdown, startup slowdown, request error probability, replica add/remove, active-sequence capacity loss, queue saturation, network latency/jitter and maximum-context OOM. The demo injects slowdown, crash, cold recovery and startup slowdown into simulator time; the separate chaos classifier exercises all eight labeled fault types.

Cost integrates each replica's hourly price across simulated duration. A removed or failed replica stops contributing service as dictated by event state. Canary-eligible requests may be assigned to canary replicas according to the configured weight.

## Determinism and safety

- Every run requires a seed in its input.
- Unknown fields and invalid distributions are rejected.
- Replica IDs and request IDs must be valid and unique.
- Queue, active sequence and event counts are bounded.
- Input canonical SHA-256, simulator version, curve ID, measurement URI and seed appear in provenance.
- The Python wrapper has a 120-second subprocess timeout and surfaces stderr on failure.

## Tests

Unit and integration tests cover determinism, queueing, routing, cancellations, deadlines, chunking, fault actions, cost and command-line artifacts. Regression fixtures live under `crates/sloforge-sim/tests/fixtures`. Run:

```bash
cargo test -p sloforge-sim
cargo run -q -p sloforge-sim -- simulate \
  --config crates/sloforge-sim/tests/fixtures/scenario.json \
  --output /tmp/sloforge-sim.json \
  --chrome-trace /tmp/sloforge-sim-trace.json
```

## CPU demo result

<!-- Metrics source: ../artifacts/demo/simulator/replay.raw.json; provenance source: ../artifacts/demo/simulator/replay.raw.json#/provenance -->

The checked-in seeded run processed 4,970 events for 120 requests over 16.378 simulated seconds. It completed 117 requests and recorded 3 failed/deadline-missed requests, with 717.771 ms p95 TTFT, 12.697 ms p99 ITL, 688.313 output tokens/s and 0.030026 USD simulated cost. Availability/SLO attainment was 97.5%. Those faulted-load results are retained rather than replaced by the optimizer's 192.349 ms TTFT prediction or the live gateway's successful summary.

The result is CPU/mock evidence. It says the state machine and calibrated replay run; it is not evidence for GPU service accuracy. Selected-curve prefill/decode coverage was 91.667%/100% on only 12 held-out observations each, and the simulator's faulted TTFT was far outside the steady prediction; both facts caution against treating the twin as a universal predictor.
