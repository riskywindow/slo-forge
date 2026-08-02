# Interview deep dive

## Thirty-second thesis

SLOForge compiles model, workload, hardware, price and SLO specifications into an evidence-backed deployment plan. It measures phase-specific service curves, searches configuration space with uncertainty, validates the plan in a deterministic digital twin and a real Rust streaming gateway, and preserves failures and rejected alternatives in a reproducible bundle.

## Why a compiler?

A YAML generator maps fields. A compiler owns semantics: typed frontend constraints, an intermediate representation, lowering passes, target backends, diagnostics, provenance and post-lowering validation. SLOForge's IR makes runtime choice, batching, routing, autoscaling, cold start and rollback one reviewable artifact rather than unrelated flags.

The analogy is useful only because the implementation earns it. Unknown fields are rejected, cross-field invariants are enforced, old alpha documents migrate explicitly, canonical bytes have cross-language hash tests, and every backend emits a validation manifest.

## Hard systems decisions

### Why Rust plus Python?

Rust owns work where bounded concurrency, cancellation and deterministic event ordering are central: gateway, SSE parsing, telemetry, load generation and simulation. Python owns ecosystem-heavy orchestration, statistics, engine processes, optimization and exports. Versioned JSON subprocesses make the boundary observable and evidence-friendly. PyO3 would reduce serialization but make failure isolation and reproduction harder, with no benefit on the HTTP hot path.

### Why separate prefill and decode?

Prompt processing scales with prompt/batch shape and tends to be compute-heavy; decode is iterative and depends on active sequences/context, often memory-bandwidth-sensitive. One latency scalar cannot reason about TTFT versus ITL or chunked prefill. Separate curves also create a clean path toward DistServe-style topology without claiming that disaggregation is already implemented.

### How is cancellation correct?

Admission and backend capacity are owned semaphore permits. The response body owns the execution stream, which owns the HTTP stream and lease. When the client drops the body, Rust drop cascades through those resources. A contract test partially consumes an SSE response, drops it, then verifies new work obtains the sole capacity permit. Upstream engines can still ignore TCP disconnect, so this is correct gateway propagation, not guaranteed GPU preemption.

### Why never retry after the first token?

Once output is visible, retrying on another backend can duplicate or diverge the prefix. The gateway retries only queue/open/stream failures before first data. Malformed or disconnected partial streams return an in-stream structured error. Contract tests prove the fallback backend is not invoked after one token.

### How is determinism maintained?

Every algorithmic path takes a seed. The Rust simulator uses ChaCha8, validates event ordering and has a byte-for-byte same-seed test. Canonical JSON uses sorted compact keys and finite values. Timestamps and live localhost timings are isolated in envelopes because pretending wall-clock HTTP is deterministic would be dishonest.

## Performance reasoning

The optimizer predicts phase service, converts it to request capacity, estimates utilization, applies a queue blow-up near saturation, and incorporates routing, batching, replication, warm capacity, failure and price. Hard constraints include uncertainty radii. This makes failure explanations inspectable: a candidate can miss because prefill grows with prompt p95, because utilization drives queueing, because cold capacity expands startup exposure or because availability is too low.

The model is intentionally simple. The selected service curve achieved 91.667% prefill coverage versus 95% nominal, while decode covered 100%; the harsher faulted simulator still reached 717.771 ms p95 TTFT versus the plan's 192.349 ms prediction. A stronger system would use more independent calibration runs, cover E2E/startup explicitly and execute the Rust twin for every acquisition proposal.

## Demo result, honestly stated

<!-- Metrics source: ../reports/demo/evaluation.json, ../artifacts/demo/optimization/result.json and ../artifacts/demo/simulator/replay.raw.json -->

The CPU/mock run evaluated 540 configurations, retained 3 direct measured anchors and recorded 24 acquisition proposals. It selected predicted configuration `cfg-aad9cd4cfa41`: 3 fast-mock replicas at concurrency 6 with chunked prefill and SLO-slack routing, at 192.349 ms p95 TTFT and $2.47718/million modeled tokens. The compiled-topology gateway completed 120 streamed requests under live faults at 170.429 ms p95 TTFT. The harsher simulator completed 117/120 (97.5% deadline attainment). Matched Rust-twin replay recorded 0 predictive misses versus 2 reactive misses at 0.001458 USD greater predictive cost. The classifier matched 8 closed-set injections and rejected all 8 no-fault controls.

The strongest story is the negative result: a plan that looks feasible from representative probes can fail under queue/fault dynamics. SLOForge makes that mismatch a first-class artifact rather than a hidden benchmark footnote.

## Questions a systems reviewer may ask

### Is the optimizer actually Bayesian optimization?

No. The primary strategy is an uncertainty-aware Pareto acquisition ranking over an enumerated space. It is easier to audit and compare than dressing a heuristic as an exotic optimizer. The implementation records exhaustive, random and halving baselines, and the current run does not show primary-strategy superiority.

### Are the budgeted proposals real measurements?

No. The 24-step acquisition history ranks analytical predictions. Measured fidelity is reserved for the three exact load-test shapes that the profiler actually executed; scaled replica, concurrency, batching, chunking and routing variants remain predicted. The profile itself is real HTTP timing against deterministic mock processes. A GPU paper evaluation must distinguish probe measurements, predictions, simulator trials and fresh engine trials in its trial accounting.

### Is the simulator validated?

Toy tests establish exact single-server timing, priority order, deadlines/cancellation/rejection, crash rerouting, dynamic actions, routing balance and at-least-10× faster-than-trace execution. The selected run's held-out prefill/decode MAPE was 8.401%/1.009% with 91.667%/100% interval coverage, but production accuracy and unmeasured-hardware generalization remain unproven. Both facts belong in the answer.

### What would you improve first?

Increase independent calibration data, close controller actions against live replica processes, add fresh configuration trials, implement prefill/decode topology and KV transfer cost, execute every engine-specific cloud export in a no-deploy environment, and run a multi-seed GPU matrix. Those directly attack observed uncertainty rather than adding a dashboard.

### How is this different from a proxy?

The gateway is one backend. The project thesis lives upstream and around it: hardware/profile evidence, typed plan IR, calibration, candidate pruning, constrained frontier, simulator, adaptive decisions, fault diagnosis and reproducible exporters. Removing the gateway would still leave a useful deployment compiler; removing the compiler would leave only a proxy.

## Code tour

Start at `python/sloforge/ir/models.py`, then `profiler/core.py`, `models/service_curve.py`, `optimizer/core.py` and `compiler/plan.py`. Follow plan execution into `crates/sloforge-sim/src/engine.rs` and `crates/sloforge-gateway/src/gateway.rs`. Finish with `controller/core.py`, `faults/engine.py` and `reports/generate.py` to show the evidence loop.
