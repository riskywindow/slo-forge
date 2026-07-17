# SLOForge Helix dependency graph

Status: implementation graph, 2026-08-03.

Helix composes the existing Continuum, Genesis, Fabric, Autopsy, recovery,
ForgeCI, and WarmPath contracts. It does not introduce another RPC system or
replace their working state machines.

## Dependency order

1. Canonical Helix IR, stable hashes, strict validators, and content references
   establish the transactional trust boundary.
2. Environment capture, the typed effect ledger, and Continuum execution-state
   capture establish a coordinated decision boundary.
3. A committed `BranchPoint` unlocks copy-on-write environment branches,
   Continuum model-state forks, compatibility analysis, and explicit
   recomputation or rejection.
4. Valid branch groups unlock strict-policy rollout, exact/causal/semantic
   replay, deterministic reward evidence, and structured branch comparison.
5. Reward and replay evidence unlock branch-relative credit, provenance-complete
   training batches, staleness checks, and trainer adapters.
6. Validated candidate policy epochs unlock separately configured quality,
   reward-integrity, serving, and active-session compatibility gates. Separate
   configuration is not evidence of independent evaluator governance.
7. A complete `PolicyPromotionCapsule` unlocks isolated shadow, bounded canary,
   atomic champion routing, champion-pinned sessions, monitoring, and rollback.
8. Stable work/resource evidence unlocks the learning-aware Fabric/SLOForge
   resource compiler, capacity lending/reclamation, and evaluation campaigns.
9. Stable raw artifacts unlock Autopsy/ForgeCI integration, static reports,
   security review, clean-room validation, and the final research report.

## Initial vertical-slice critical path

```text
PolicyEpoch + EnvironmentStateCapsule + effect ledger
       + Continuum ExecutionStateCapsule
                         |
             coordinated capture barrier
                         v
                    BranchPoint
                         |
      Continuum COW fork + environment COW branches
                         v
              strict-policy sibling rollouts
                         |
        deterministic verifier + RewardEvidence
                         v
        branch-relative CreditAssignmentEvidence
                         |
             TrainingBatchManifest validation
                         v
             deterministic tiny trainer
                         |
       candidate PolicyEpoch + separately configured gates
                         v
    local shadow/canary -> atomic route -> rollback parent
```

## Workstream ownership

The execution environment exposes four total agent slots. Root plus three
subagents is the maximum concurrency, so ready work is scheduled continuously
through those three lanes with disjoint file ownership. Root owns manifests,
CLI composition, integration, the flagship demo, and final correctness.

- IR lane: `python/sloforge/helix/ir`, `schemas/helix`,
  `crates/sloforge-helix-ir`, golden/conformance tests.
- Environment/effect lane: `python/sloforge/helix/environments`,
  `python/sloforge/helix/effects`, environment/effect tests.
- Capture/branch/replay lane: `python/sloforge/helix/capture`,
  `python/sloforge/helix/branching`, `python/sloforge/helix/replay`, focused
  tests.
- Root integration lane: root manifests, CLI, rollout/reward/credit/training,
  transaction/promotion, scheduler, demos, reports, and integration tests.

Optional GPU, cloud, external API, production capture, external side effects,
external deployment, and live promotion remain disabled unless their explicit
Helix opt-ins and budgets are present.
