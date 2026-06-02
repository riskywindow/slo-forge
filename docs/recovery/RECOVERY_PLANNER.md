# Recovery planner

The recovery planner consumes a `DiagnosisRecord`, current
`PhysicalExecutionPlan`, workload-aware policy, and optional compiled recovery
variant, and emits the typed `RecoveryPlan` defined in the Fabric IR.

## Diagnosis-to-action mapping

Every `BottleneckKind` has an explicit action template. Examples include traffic
concurrency/shedding for overload, NUMA rebinding for CPU launch or locality,
GPU quarantine and rank replacement for compute/HBM/clock faults, rail/NIC
quarantine and affinity changes for network faults, rank reordering or collective
switching for collective faults, expert move/replication for skew, worker-ratio
changes for pool saturation, and KV transport switching for transfer faults.

Actions have stable order, scope (`request_path`, `worker_local`,
`replica_local`, `new_replica`, `deployment_rebuild`, or `operator_required`),
targets, idempotency key, timeout, rollback link, and external-mutation flag.

## Variant and objective selection

The planner chooses a compatible compiled `RecoveryVariant` whose trigger matches
the diagnosis and minimum confidence. It minimizes degraded TPOT upper bound,
transition cost, transition time, and stable ID. When a variant implies new
placement, that action is appended to the diagnosis-local repair.

Expected improvement comes from the attached counterfactual when present;
otherwise it is a conservative bounded function of diagnosis confidence. The
proposal contains intervals for SLO improvement, cost, disruption, build time,
availability, and communication change. These are planning estimates, not
observed outcomes.

## Traffic and safety policy

The proposal fixes shadow/canary fractions, minimum samples, started-stream
preservation, maximum inflight streams at drain, promotion criteria, error and
TPOT rollback thresholds, readiness abort threshold, compatibility constraints,
evidence links, and external authorization state.

The flagship proposal in `artifacts/fabric-demo/recovery/proposal.json` contains
four local simulated actions: stop routing, drain, restart, and alternate rank
placement. It has 3 shadow and 4 canary minimum samples and preserves started
streams. It must not be interpreted as an applied production change.

