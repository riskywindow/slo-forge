# Migration planner

The planner evaluates stop-and-copy, pre-copy, hybrid pre-copy, recomputation-assisted, and constrained lazy candidates under semantic, capability, memory, budget, SLO, rollback, and access-order constraints.

Inputs include current state size, dirty and generation rates, source load, destination readiness, measured conversion/transfer rates with provenance, temporary memory, failure probability, exactness/quality requirements, and objective weights. Outputs include the selected strategy, candidate estimates and rejected reasons, chunking/concurrency, pre-copy rounds, predicted interruption/total time, bytes, source/destination overhead, memory, cost, quality loss, uncertainty, validation, and rollback actions.

## Evidence integration

`fabric_transfer_rates` extracts successful KV-transfer, send/receive, host-copy, and GPU-P2P measurements from a `FabricProfile`, retaining raw artifact URI/hash, sample count, coefficient of variation, and synthetic-versus-measured mode. `plan_with_fabric_and_warmpath` replaces generic transfer curves with those rates and incorporates WarmPath's predicted p95 destination readiness.

`ContinuumMigrationExtension` can be attached through the existing PhysicalExecutionPlan extension map without changing its major schema. It references capsule, compatibility, conversion, and migration artifacts plus topology fingerprints, segment deadlines, required-before-use state, and legal conversion placements.

## Guardrails

- A destination is never warmed with active session state before compatibility is known.
- Dense full-attention state is required before use unless an adapter proves an ordered/stall-safe access contract.
- Dirty-rate non-convergence is explicit; fixed-round pre-copy is not assumed safe.
- Synthetic Fabric inputs remain labeled synthetic and cannot support real-network performance claims.
- Prediction error is computed only where predicted and observed artifacts describe the same metric.
