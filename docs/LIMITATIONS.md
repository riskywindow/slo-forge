# Limitations

This document is deliberately conservative. The checked-in system is a complete CPU/mock vertical slice with real Rust transport and simulation, but several production and GPU claims remain unexercised.

## Evidence scope

- The evaluation used deterministic mock token generation shaped like a small model, not Qwen weights. The IR model ID is `sloforge/mock-qwen3-0.6b-shape` and records `model_is_mock` provenance.
- No NVIDIA GPU, Transformers inference, vLLM server, SGLang server, TensorRT-LLM engine, Modal deployment or Baseten deployment was run.
- CPU hardware microbenchmarks cover host copy and one GEMM shape, not kernel launch, PCIe, device memory, multi-GPU collectives or topology.
- The single seed-41 run is engineering evidence, not a statistically powered performance study.

## Prediction quality

The selected service curve has 7.57% held-out prefill MAPE but only 54.17% interval coverage. The uncertainty interval is therefore under-calibrated on this run. Cross-hardware generalization is neither implemented nor claimed.

The optimizer's “measured” configuration evaluations analytically transform candidate profile summaries; they do not launch every replica/concurrency/batching combination. Cost is modeled from declared hourly price and output rate. Availability derives from mock failure rates and should not be mapped to provider availability.

The selected steady-profile estimate met p95 TTFT and p99 ITL constraints, yet the faulted simulator replay attained only 59.17% of request deadlines. This exposes workload/fault mismatch and means the plan should not be promoted unchanged to production.

## Simulator fidelity

Continuous batching, prefill chunking and decode are approximated by service curves and discrete steps. The simulator has one service curve per scenario, colocated prefill/decode topology and no KV-transfer cost, cache eviction dynamics, tensor collective contention, CUDA kernel overlap or network protocol model. It is deterministic and useful for relative experiments, not cycle accurate.

## Gateway

The gateway has no authentication, TLS, tenant quota, content moderation or persistent state. It should run behind trusted ingress. Live trace export is internal Chrome JSON rather than direct OTLP. Retry policy assumes a completion request is safe only before output; engines with external side effects need stronger idempotency semantics.

Cancellation is propagated by dropping streams and permits, but upstream engines may continue compute if their HTTP server ignores disconnects. Gateway queueing is bounded globally and per backend, not fair-queued per tenant.

## Controller and diagnosis

The controller evaluation uses an analytical TTFT observation model rather than closing the loop against a live replica manager. Scale actions are evaluated, but the checked-in demo triggered no canary and no rollback. Provider action application and failure reconciliation are not implemented.

The diagnosis result is a closed-set test whose counter mutations and expected labels are known to the classifier. Its 100% accuracy does not predict production incident accuracy. Confidence values are heuristic and not empirically calibrated probabilities.

## Exporters

Local/Docker/Kubernetes exports are statically validated but were not all deployed in the checked-in run. Modal and Truss outputs compile/validate offline only. Both cloud model wrappers currently use Transformers even when another engine appears in the plan, so engine-specific runtime parity is incomplete. Provider autoscaling, region, canary and rollback semantics are only partially mapped.

The Kubernetes export is a compact Helm chart; it is not an operator with reconciliation against `DeploymentPlan`. Production clusters still need image publication, secrets, policy, storage, GPU scheduling and observability configuration.

## Evaluation gaps

H1 is not established: exhaustive, random and uncertainty-aware search found the same best objective in the tiny CPU/mock space, while successive halving found no feasible point. H2 has no measured engine-default or manual-static GPU baseline. H3 is one synthetic trace. H4 is closed-set synthetic diagnosis. There is no GPU benchmark report because compatible hardware was unavailable.

The focused GPU logits benchmark has reference/correctness/timing code, but no GPU result is recorded and the optimization is not enabled by default.

## Operational gaps

There is no authentication/billing UI, distributed controller store, high-availability compiler service, artifact object store, provider credentials manager or automated cloud cost reconciler. Those are intentionally outside the flagship systems thesis unless required to validate a deployment.
