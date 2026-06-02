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
8. Extract the non-dominated frontier, select the requested robust objective,
   and generate up to three recovery variants.
9. Emit all rejected candidates and optimizer decisions.

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

## Objective and uncertainty

Candidate objective terms are normalized expected cost, TTFT/TPOT violation,
tail risk, communication time, failure exposure, and transition cost. The exact
selected weights depend on `CompilerObjective`: minimize cost, minimize latency,
maximize goodput, or robust balanced. Metric intervals expand estimates by the
declared relative measurement uncertainty.

This is an inspectable analytical ranking pass over calibrated curves. It is not
the discrete-event simulator. `simulator_calls` is therefore zero in compiler
results and optimizer history. `sloforge fabric simulate`/`validate` is the next
explicit pass and supplies queueing/contention validation before execution or
recovery.

## Explainability

Each rejected candidate has stage, reason code, explanation, and violated
constraints. Each feasible unselected candidate is retained as a dominated
objective alternative. `optimizer_history` contains stable sequence, phase,
decision, reason, solver time, and simulator call count. The plan records the
predicted dominant bottleneck and host fault exposure.

The deterministic flagship compile emitted a 16-rank disaggregated plan, 174
rejected alternatives, and three recovery variants in
`artifacts/fabric-demo/physical-plan.json`. Those counts are synthetic-demo
results, not evidence that the selected degrees are suitable for real hardware.

