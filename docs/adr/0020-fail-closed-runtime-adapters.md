# ADR 0020: Keep runtime adapters narrow and fail closed

- Status: accepted
- Date: 2026-08-01

## Context

Kubernetes, Dynamo, vLLM, SGLang, Modal, and Truss expose different physical
controls and evolve independently.

## Decision

Lower only fields verified in the recorded version range. Keep unstable runtime
flags in isolated adapters with semantic-version checks. Emit capability metadata
that distinguishes exact, runtime-dependent, and advisory controls. Modal and
Truss physical data is advisory only; Dynamo remains the execution mechanism for
its generated graph. Direct vLLM/SGLang lowerings reject multi-host replicas;
Dynamo DGD is the multi-node engine path. Direct process groups use role-local
DP=1. vLLM EP must match the runtime's `TP_SIZE * DP_SIZE`, while SGLang EP does
not imply DP attention. Only native vLLM NIXL KV connector fields are emitted;
Dynamo wrapper-only disaggregation flags remain isolated. Generic Kubernetes
multi-node export is rejected without a gang contract. All generation is offline.

## Consequences

Unsupported combinations fail rather than silently weakening a physical plan.
Adapters require periodic official-source validation. Generated artifacts are
not evidence that a runtime was installed or executed. The DGD JSON Schema is a
strict tagged-v1.3 subset, not live operator admission, and intentionally omits
development-branch-only fields.
