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

The flagship demo establishes the chronology around that fixture explicitly: it persists the workload-drift trigger first, then executes the separate challenger synthesis and capsule build, records a content-hashed synthesis event bound to the trigger snapshot, and only then registers, shadows, canaries, and promotes the challenger. The combined timeline is derived from controller audit records plus that synthesis artifact; it is not a prewritten transcript.

The controller and local/simulated observations are exercised; external production traffic is not implied. Gate observations are supplied by a harness rather than collected by this controller. Real workload-drift monitoring, physical fault injection, state-conversion migration, multi-node draining and external live promotion remain unexercised unless a separate artifact records them. The environment guard prevents those paths from being inferred from local success.

## H7 comparative campaign

The artifact-backed H7 campaign compares four bounded strategies for every declared seed:

1. `no_adaptation`, which observes drift and remains degraded through the logical horizon;
2. `threshold_controller`, a deterministic scale-and-restart comparator;
3. `physical_replan_only`, which addresses the physical-plan surface but leaves workload drift unresolved in the model; and
4. `genesis_evolution`, which executes the real persisted controller, capsule gate, local runtime replay, stream lease, promotion, and optional rollback path.

The first three arms are explicit comparator models. Only the Genesis arm invokes `EvolutionController`. The campaign injects both a workload-drift observation and a Fabric-health degradation observation. The latter exercises trigger classification and coalescing; it does not alter a host link, GPU, or network device. Its event therefore cannot support a hardware fault-recovery claim.

For the Genesis arm, each seed:

- opens an externally visible stream on the champion before evolution;
- registers a digest-mismatched challenger reference and records its independent capsule rejection;
- registers the valid challenger and revalidates it before shadow, canary, and promotion;
- collects shadow and canary evidence by replaying both generated capsule runtimes inside the local sandbox;
- keeps the original stream pinned to the original capsule while new work follows the champion decision; and
- executes rollback for the configured deterministic rollback-seed subset.

External traffic, external deployment, and live promotion remain disabled. An H7 campaign always records `actual_hardware_experiments=0` unless a future schema and validator explicitly add hardware evidence.

### H7 artifact contract

The conventional artifact layout is:

```text
artifacts/genesis/evaluation/campaigns/h7/
  report.json
  raw-events.jsonl
  genesis-runs/<seed>/
    controller-state.json
    final-controller-snapshot.json
    runtime-gates/shadow/
      trace.json
      gate-evidence.json
      champion/runtime-observation.json
      challenger/runtime-observation.json
    runtime-gates/canary/
      trace.json
      gate-evidence.json
      champion/runtime-observation.json
      challenger/runtime-observation.json
```

`report.json` uses `sloforge.genesis.h7-campaign/v1`; raw events use `sloforge.genesis.h7-event/v1`. The report binds the canonical event stream, final controller snapshots, gate observations, capsule identities, and evidence roots by SHA-256. It separately labels:

- `logical_time_to_slo_restoration_ticks`, which is deterministic synthetic controller time;
- local runtime request observations, which are sandboxed CPU semantics with host timing retained in gate artifacts; and
- actual hardware experiments, which are zero in this campaign.

Dropped requests, interrupted streams, adaptation work units, rollback count, invalid-challenger count, local runtime observation count, and active-stream preservation are recomputed from raw events and controller state. Adaptation work units are a declared comparison cost, not dollars, CPU seconds, or GPU time.

### H7 validator and replay boundary

`validate_evolution_campaign` fails closed. It validates the strict report and event schemas, constrains campaign-owned paths to the artifact root, checks all hashes, reconstructs comparator events from seeds, and reconstructs Genesis events from gate observations and rollback status. It then derives run metrics and aggregates again.

For every Genesis seed, the validator also reopens the final controller snapshot, checks the workload trigger and coalesced Fabric degradation, verifies the rejected and accepted challenger records, confirms promotion or rollback, checks the retained visible stream lease, revalidates champion and challenger capsules, and repeats transition-compatibility validation. Publishable gate evidence is passed to `local_gate_evidence_validator`, which checks its raw trace and observations and reruns both generated runtime semantics in the sandbox. Unit-test gate fixtures require an explicit opt-in and are rejected by the publication path.

The campaign's comparison is therefore artifact-backed but deliberately scoped. It can establish behavior of this local deterministic exercise. It cannot establish production SLO-restoration time, real network-degradation recovery, multi-node safety, or a hardware performance advantage.
