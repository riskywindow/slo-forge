# Continuous evolution

Genesis evolves an accepted deployment through a persisted champion/challenger state machine. The controller does not trust synthesis output and never treats candidate generation as promotion authority.

## Safety boundary

The champion is a validated capsule with an available rollback artifact. Every challenger starts with an `IsolationContract` that forbids network access, cloud credentials, host devices, and live traffic. An independently supplied capsule-validator callback checks the hash-addressed capsule when the challenger is registered and again immediately before promotion. A failed or changed capsule returns to the evolving state without changing the champion.

External live canarying and promotion require both `live_promotion_authorized=true` in controller configuration and `SLOFORGE_GENESIS_ALLOW_LIVE_PROMOTION=1` in the process environment. Simulated and non-live local evolution do not mutate an external deployment.

## State machine

The path is:

`IDLE -> EVOLVING -> CHALLENGER_READY -> SHADOWING -> SHADOW_VALIDATED -> CANARYING -> READY_TO_PROMOTE -> PROMOTED`

Shadow and canary observations independently require minimum sample counts, bounded error rate, TTFT and TPOT ratios, quality regression within contract, and zero interrupted streams. Failed gates retain their evidence and reject the candidate. Post-promotion failures may transition to `ROLLED_BACK` while keeping any streams pinned to the runtime on which they began.

Policy-only and request-boundary swaps preserve active streams. Existing stream leases remain assigned to their original capsule, new requests use the promoted champion, and the prior champion remains retained for rollback. Transitions that are neither boundary-safe nor explicitly active-stream-compatible require a full drain. Operator-required transitions cannot be promoted autonomously.

## Persistence and crash recovery

Every transition is written before the new snapshot is exposed. `EvolutionStore` uses a bounded JSON envelope, canonical payload SHA-256, a same-directory temporary file, `fsync`, and atomic replacement. Restore rejects malformed, oversized, or digest-mismatched state. Event identifiers are persisted and bounded, making a repeated event after controller restart idempotent.

Snapshots preserve the trigger, challenger population, independent capsule reports, shadow and canary evidence, stream leases, retained rollback runtimes, and a bounded audit trail. They do not contain fabricated benchmark measurements: gate observations must come from the benchmark or traffic harness invoking the controller.

## Deterministic local exercise

`run_local_evolution_fixture` accepts a controller whose validator is already wired to real capsule artifacts, a challenger specification, and an explicit controller seed. It injects deterministic workload drift, runs shadow and canary gates, revalidates the capsule, and promotes only if all evidence passes. Tests also cover capsule rejection, canary rejection, request-boundary and policy swaps, rollback, crash recovery, persisted-state tampering, and the external live-promotion guard.

The controller and local/simulated observations are exercised; external production traffic is not implied. Gate observations are supplied by a harness rather than collected by this controller. Real workload-drift monitoring, physical fault injection, state-conversion migration, multi-node draining and external live promotion remain unexercised unless a separate artifact records them. The environment guard prevents those paths from being inferred from local success.
