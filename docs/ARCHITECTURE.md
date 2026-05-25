# Architecture

SLOForge is a deployment compiler with a measured runtime feedback loop. Python owns the compiler and experiment plane; Rust owns latency-sensitive request handling, trace generation, and deterministic simulation. The canonical handoff is versioned JSON, not in-process object sharing.

```text
Model + workload + hardware + SLO + budget
                    |
                    v
        Python compiler / profiler
   feasibility -> calibration -> optimizer
                    |
       DeploymentPlan + EvidenceBundle
          /         |             \
         v          v              v
 Rust simulator  Rust gateway   offline exporters
         |          |              |
 event trace    HTTP/SSE data   local, Docker,
 and metrics    plane + OTEL    Helm, Modal, Truss
         \          /
          v        v
       replay, controller, diagnosis, report
```

## Component boundaries

| Component | Language | Responsibility | Stable interface |
|---|---|---|---|
| IR and compiler | Python and Rust | Strict types, migration, cross-field validation, plan compilation | `sloforge.io/v1` JSON |
| Profiler | Python | Static memory checks, staged startup/prefill/decode/load probes, budget accounting | `sloforge.profile/v1` |
| Performance model | Python | Monotonic service curves, load-sample error and residual interval diagnostics | `sloforge.models/v1` |
| Optimizer | Python | Candidate enumeration, uncertainty-adjusted constraints, Pareto selection | `sloforge.optimization/v1` |
| Simulator | Rust | Deterministic discrete-event replay, failures, cost, Chrome trace | simulator JSON subprocess protocol |
| Gateway | Rust | OpenAI-compatible HTTP/SSE, bounded admission, routing, failure handling | HTTP/SSE and Prometheus |
| Controller | Python | Forecasting, guarded action evaluation, canary/rollback state | `sloforge.controller-evaluation/v1` |
| Fault diagnosis | Python | Labeled injection, counter-based classification and counterfactual score | `sloforge.chaos-result/v1` |
| Exporters | Python | Offline materialization and validation of deployment targets | `sloforge.export/v1` manifests |
| Reports | Python | Hash verification and derived Markdown, HTML, SVG, OTEL and metrics | `sloforge.report/v1` |

The Python/Rust boundary is JSON over bounded subprocess execution because it is inspectable, replayable, and naturally archived. Serialization is outside the gateway request path. HTTP/SSE is reserved for the running data plane. See [ADR 0001](adr/0001-rust-python-boundary.md).

## Data flow

1. A trace is validated as ordered JSONL with explicit request IDs, arrival offsets, token counts, class, priority and optional deadline/cancellation metadata.
2. The hardware probe captures a fingerprint plus raw microbenchmark samples. CUDA is used only when explicitly requested and available.
3. Profiling rejects infeasible candidates, probes startup/prefill/decode/load stages, labels warmups, and debits time and dollar budgets.
4. Calibration separates prefill and decode curves, enforces monotonic medians, and records held-out MAPE and interval coverage.
5. Optimization enumerates engine/topology/batching/routing combinations, evaluates all at low fidelity, promotes a budgeted subset, applies uncertainty-adjusted hard constraints, and emits a non-dominated frontier.
6. Compilation turns the selected evaluation into a strict `DeploymentPlan`; the accompanying `EvidenceBundle` points back to measurements, optimizer decisions, rejected candidates and artifact hashes.
7. The simulator or gateway replays traffic. Runtime artifacts feed controller evaluation, fault diagnosis and reports.
8. The report generator verifies the artifact index and plan digest before deriving any displayed metric.

## Control flow

The online policy operates in fixed windows. It estimates workload composition and arrival rate, forecasts an upper-rate envelope, evaluates replica/concurrency/routing actions, and chooses the cheapest action whose predicted TTFT remains inside a safety margin. Minimum samples, cooldown, and hourly change budgets may force a hold. Routing or variant changes are canaried; an observed TTFT beyond the promotion limit restores the previous state and enters rollback cooldown.

The checked-in CPU evaluation exercised scale decisions, but not a canary or rollback: zero canary windows and zero rollbacks were recorded. The state machine and rollback path are implemented and tested separately; the evaluation must not be read as evidence that rollback happened.

## Failure handling

- Gateway admission and backend capacity are semaphores with queue deadlines; there is no unbounded core queue.
- Backends have health checks, circuit breakers and request timeouts. Retries are limited to pre-output failures; a partial stream is never replayed.
- Dropping a client response drops the execution stream, the backend response and its capacity lease, propagating cancellation through Rust ownership.
- Managed demo processes run in their own process groups and are terminated with a bounded graceful interval followed by a kill.
- Simulator inputs have a maximum event count, validated positive distributions and explicit fault actions.
- Export generation is offline. No generated cloud app is invoked or deployed by validation.

## Deployment IR

`DeploymentPlan` is strict and frozen in Python and has a matching Rust representation. It combines model, engine, hardware, workload, SLO and budget specifications with replica, routing, admission, batching, autoscaling, cold-start, canary and rollback policies. Every metric estimate contains a point, interval, confidence, unit, sample count and measurement IDs. In the current CPU artifact, the serialized `0.95` confidence is a requested nominal level, not an empirically achieved coverage guarantee; observed prefill interval coverage was only 54.17%. Extension keys must be namespace-qualified. Unknown fields are rejected.

The version-1 schema is at [`schemas/deployment-plan-v1.schema.json`](../schemas/deployment-plan-v1.schema.json). Migrations accept only known alpha formats, never optimistic unknown versions. Golden fixtures and conformance tests exercise canonical serialization and SHA-256 equality across Python and Rust.

## Evidence and reproducibility

The `EvidenceBundle` stores the environment manifest, assumptions, measurement references, calibration metrics, optimizer history, rejections, benchmark records, artifact hashes, Git state and generation time. A separate artifact index binds friendly names to paths, media types and hashes. The demo report verified 15 indexed inputs before rendering. Deterministic algorithms take an explicit seed; environment timestamps remain nondeterministic envelope data.

## Current evidence boundary

The checked-in run used three deterministic HTTP mock backends on an Apple M4 Pro CPU. It validates orchestration, real HTTP streaming, bounded routing, fault injection, simulation, control and provenance. It does not measure Qwen weights, NVIDIA GPUs, vLLM, SGLang, Transformers or a cloud deployment. Detailed numbers and their sources are in [Reproducibility](REPRODUCIBILITY.md) and [Limitations](LIMITATIONS.md).
