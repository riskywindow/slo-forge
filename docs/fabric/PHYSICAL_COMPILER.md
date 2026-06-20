# Physical compiler

The compiler transforms a logical plan reference, `ModelGraph`, `TopologyGraph`,
`FabricProfile`, workload/SLO constraints, assumptions, objective, and seed into
a `PhysicalExecutionPlan` plus a Pareto frontier and complete candidate trace.

## Passes

1. Verify that the fabric profile fingerprint equals the canonical topology hash.
2. Enumerate allowed TP, PP, DP, EP, and prefill/decode-disaggregation choices.
3. Reject degree products, expert divisibility, disaggregation, memory, runtime,
   transport, goodput, availability, latency, cost, and GPU-capacity violations.
4. Select healthy GPUs and construct rank groups.
5. Place ranks and bind NUMA, NIC, rail, CPU affinity, role, replica, and fault
   domain.
6. Select measured paths for collectives and KV transfer; construct memory and
   overlap plans.
7. Score latency, communication, cost, tail risk, failure exposure, and
   reconfiguration cost with explicit equations and uncertainty.
8. Promote the analytical winner, non-dominated frontier, and leading recovery
   alternatives into isolated-request validation through the Rust physical
   simulator. Recompute latency, communication, goodput, cost, and objective
   values from the simulator result, then repeat promotion until every final
   Pareto candidate has been validated.
9. Reject plans whose operation DAG cannot be lowered or validated, select the
   requested robust objective, and generate up to three simulator-validated
   recovery variants.
10. Emit all rejected candidates and optimizer decisions.

## Strategies

`exhaustive` explores the finite configured space. `random_placement` is a
seeded content-hash baseline. `topology_unaware` takes stable sequential devices.
`greedy_topology_aware` grows a placement by shortest-path/locality score.
`hierarchical` applies the same topology-aware placement after static pruning.
`robust_failure` excludes or penalizes degraded fault domains and compiles a
replacement choice.

The path model uses graph search over healthy bidirectional edges. Curve latency
is selected near the message size; bottleneck bandwidth is the minimum on the
path. Unknown bandwidth is deliberately punitive rather than optimistic.

NIC choice is a deterministic same-NUMA/high-advertised-speed heuristic; path
evaluation happens afterward, but placement search can miss a better
GPU-to-NIC or rail-balanced assignment. The hierarchical strategy is inspectable
and topology-aware, not a proof of global optimality. Canonical collective
profiles currently aggregate several collective primitives into one category,
so per-algorithm selection is only as specific as the imported profile.

## Objective and uncertainty

Candidate objective terms are normalized expected cost, TTFT/TPOT violation,
tail risk, communication time, failure exposure, and transition cost. The exact
selected weights depend on `CompilerObjective`: minimize cost, minimize latency,
maximize goodput, or robust balanced. Metric intervals expand estimates by the
declared relative measurement uncertainty.

The analytical ranking pass is followed by bounded, fail-closed calls to the
same Rust discrete-event simulator used by `fabric simulate`. Compiler
refinement intentionally uses one isolated p95-shaped request: the physical twin
models exclusive operation resources and does not reproduce an engine's
continuous-batching scheduler. `sloforge fabric validate` remains the separate
representative-workload queueing and contention gate.

`PhysicalCompileResult.simulator_calls` and
`simulator_validated_candidate_ids` record the exact calls. Its
`solver_time_ms` is observed local wall time and is diagnostic, so it is outside
the canonical `PhysicalExecutionPlan` hash. Optimizer history instead records a
deterministic processed-event work estimate (1,000 reference events per reported
millisecond) to keep repeated plan hashes stable; the unambiguous raw count is
`deterministic_solver_work_units` on the outer result. Simulator requests have a
30-second timeout, explicit request/response bounds, and bounded failure text.

## Explainability

Each rejected candidate has stage, reason code, explanation, and violated
constraints. Each feasible unselected candidate is retained as a dominated
objective alternative. `optimizer_history` contains stable sequence, phase,
decision, reason, solver time, and simulator call count. The plan records the
predicted dominant bottleneck and host fault exposure.

The deterministic flagship fixes a 16-rank TP=8/PP=1/DP=2/EP=4 disaggregated
shape for both aware and unaware strategies so that it compares physical
placement over the same multi-node job. Its optimizer trace and recovery
variants are in `artifacts/fabric-demo/optimizer.json` and
`physical-plan.json`. This synthetic-demo shape is not evidence that those
degrees are suitable for real hardware.
