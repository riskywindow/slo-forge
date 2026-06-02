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
its generated graph. All generation is offline.

## Consequences

Unsupported combinations fail rather than silently weakening a physical plan.
Adapters require periodic official-source validation. Generated artifacts are
not evidence that a runtime was installed or executed.

