# Limitations

This document is deliberately conservative. The checked-in system is a complete CPU/mock vertical slice with real Rust transport and simulation, but several production and GPU claims remain unexercised.

## Evidence scope

- The evaluation used deterministic mock token generation shaped like a small model, not Qwen weights. The IR model ID is `sloforge/mock-qwen3-0.6b-shape` and records `model_is_mock` provenance.
- No NVIDIA GPU, Transformers inference, vLLM server, SGLang server, TensorRT-LLM engine, Modal deployment or Baseten deployment was run.
- CPU hardware microbenchmarks cover host copy and one GEMM shape, not kernel launch, PCIe, device memory, multi-GPU collectives or topology.
- The single seed-41 run is engineering evidence, not a statistically powered performance study.

## Prediction quality

The selected service curve has 8.401% held-out prefill MAPE and 91.667% coverage on 12 representative-load observations; decode has 1.009% MAPE and 100% coverage on the corresponding 12 observations. Prefill is below the nominal 95% target and decode's perfect result has a very small denominator. E2E and startup lack distinct held-out coverage estimates. Cross-hardware generalization is neither implemented nor claimed.

Only the three exact profiled CPU/mock shapes are labeled measured. The other replica/concurrency/batching/routing configurations are analytical predictions and are not fresh launches. Cost is derived from declared hourly price and observed or predicted output rate. Availability comes from mock observations or modeled failure rates and should not be mapped to provider availability.

The selected prediction met p95 TTFT and p99 ITL constraints, and live compiled-topology replay met both at 170.429 ms and 9.429 ms. The harsher faulted simulator nevertheless reached 717.771 ms p95 TTFT and missed 3/120 deadlines. This fault-model sensitivity means the plan should not be promoted unchanged to production solely from the successful localhost run.

## Simulator fidelity

Continuous batching, prefill chunking and decode are approximated by service curves and discrete steps. The simulator has one service curve per scenario, colocated prefill/decode topology and no KV-transfer cost, cache eviction dynamics, tensor collective contention, CUDA kernel overlap or network protocol model. It is deterministic and useful for relative experiments, not cycle accurate.

## Gateway

The gateway has no authentication, TLS, tenant quota, content moderation or persistent state. It should run behind trusted ingress. Live trace export is internal Chrome JSON rather than direct OTLP. Retry policy assumes a completion request is safe only before output; engines with external side effects need stronger idempotency semantics.

Cancellation is propagated by dropping streams and permits, but upstream engines may continue compute if their HTTP server ignores disconnects. Gateway queueing is bounded globally and per backend, not fair-queued per tenant.

## Controller and diagnosis

The controller chooses actions with an analytical TTFT model, then both policies' actions are replayed through identical calibrated Rust-simulator scenarios for deadline-miss and cost comparison. The loop still does not operate a live replica manager, and the checked-in demo triggered no canary and no rollback. Provider action application and failure reconciliation are not implemented.

The diagnosis result is a closed-set test whose counter mutations and expected labels are known to the classifier. Eight negative-control windows exercise normal-state abstention, and classifier execution is timed directly, but label agreement and zero synthetic false positives do not predict production incident accuracy or end-to-end detection latency. Confidence values are heuristic and not empirically calibrated probabilities.

## Exporters

Local/Docker/Kubernetes exports are statically validated but were not all deployed in the checked-in run. Modal and Truss outputs compile/validate offline only. Their generated code has engine-specific Transformers, vLLM, SGLang, TensorRT-LLM and explicit-mock paths, but those non-mock cloud paths were not executed on this host. Provider autoscaling, region, distributed topology, canary and rollback semantics are only partially mapped.

The Kubernetes export is a compact Helm chart; it is not an operator with reconciliation against `DeploymentPlan`. Production clusters still need image publication, secrets, policy, storage, GPU scheduling and observability configuration.

## Evaluation gaps

H1 is not established because the CPU/mock strategies consume shared predictions rather than independent paid measurements. H2 has calibrated default/manual/compiled predictions across five regimes but no measured real-engine comparator; these predictions include losses and are not a benchmark win. H3 is a one-seed Rust-twin comparison, not live provider adaptation. H4 is closed-set synthetic diagnosis despite the added negative windows. The GPU report records explicit unavailability rather than performance numbers because compatible hardware was absent.

The focused GPU logits benchmark has reference/correctness/timing code, but no GPU result is recorded and the optimization is not enabled by default.

## Operational gaps

There is no authentication/billing UI, distributed controller store, high-availability compiler service, artifact object store, provider credentials manager or automated cloud cost reconciler. Those are intentionally outside the flagship systems thesis unless required to validate a deployment.
