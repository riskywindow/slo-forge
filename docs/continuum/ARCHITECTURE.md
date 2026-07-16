# SLOForge Continuum architecture

Continuum treats a running inference session as portable logical state plus an explicitly described physical realization. It does not copy process memory, CUDA handles, scheduler queues, or raw pointers. A migration is legal only when compatibility analysis, conversion verification, destination validation, and ownership transfer all succeed.

```text
runtime adapter A                         runtime adapter B
  logical state ----\                 /---- logical import
  physical layout ---+-- capsule ----+----- physical allocation
  dirty log ---------/      |        \---- reconstruct ephemera
                            CAS
                     content-addressed store
                              |
  Fabric measurements -> planner -> conversion DAG -> StateTransport
                              |
                  durable coordinator + gateway ledger
```

## Responsibility boundaries

- Python owns adapter orchestration, semantic compatibility, conversion compilation, migration planning, evaluation, and reporting.
- Rust owns the canonical shared wire representation, durable protocol semantics, deterministic explicit-state exploration, and the existing gateway data-plane boundary.
- Versioned JSON is the primary Rust/Python boundary. Python and Rust serialize the same strict Continuum v1 fixtures and agree on canonical SHA-256 hashes.
- A `StateTransport` moves authenticated chunks. It never decides whether their meaning is compatible.
- Fabric contributes measured path rates and topology evidence. WarmPath contributes destination-readiness estimates. Neither can override semantic restrictions.

## End-to-end protocol

1. The source adapter publishes capabilities and captures a consistent token-boundary snapshot.
2. `captured_to_capsule_inputs` separates logical components from `PhysicalStateLayout`; `publish_capture` stores exact segment bytes and seals an `ExecutionStateCapsule`.
3. Compatibility analysis classifies reuse and emits reasons, rejected classes, conversion requirements, recomputation requirements, and verification obligations.
4. The compiler produces a bounded, typed transformation DAG. The optimized direct TP/layout converter is independently compared with the canonical decode/re-encode path on the actual captured bytes.
5. During pre-copy the source continues to generate while versioned dirty deltas are transferred. The source then quiesces at a token boundary and exports one final delta.
6. The destination imports and validates state and performs a bounded dry-run continuation check.
7. The source is fenced before a compare-and-swap owner-epoch change. The gateway switches epochs and accepts only the next token index.
8. Failed pre-commit attempts retain a valid source. A successful commit makes the destination the only accepted output owner.

## Implemented execution paths

The deterministic CPU reference path uses attention KV, recurrent, sampler, guided-decoding, token-history, and client-delivery state. It migrates from paged token-major separate K/V with logical TP=4 and page size 3 to paged head-major packed K/V with logical TP=2 and page size 5. Simulated device names are layout metadata; they are not evidence of GPU execution.

The repository also contains version-gated PyTorch, Genesis, vLLM, and SGLang bindings. The current CPU campaign exercises the two reference adapters and Genesis loading/smoke behavior; it does not claim a completed vLLM-to-SGLang live migration. See [Runtime adapter SDK](RUNTIME_ADAPTER_SDK.md) and [Limitations](LIMITATIONS.md).

## Trust boundaries

Runtime adapters and generated conversion programs are untrusted inputs until checked. Capsules are strict, authenticated manifests; tenant identity scopes content addressing; stale owner epochs are rejected; and sensitive payload bytes are excluded from event logs. See [Security](SECURITY.md) and [Threat model](THREAT_MODEL.md).
