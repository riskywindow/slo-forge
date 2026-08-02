# Recovery safety

Recovery is a control-plane mutation and defaults to simulation.

## Guards

- A disruptive proposal cannot leave `PROPOSED` without counterfactual
  simulation validation and minimum confidence.
- External actions require both plan authorization and executor configuration.
- Action IDs and observations are idempotent; snapshots survive controller
  restart.
- Shadow and canary have independent sample minimums and promotion metrics.
- Error/SLO rollback and readiness abort criteria are explicit IR fields.
- State and total deadlines are bounded.
- New requests can stop routing, but started streams are preserved through drain.
- The recovery layer never retries output that has started.
- Cooldown limits repeated transitions.
- Every transition and action is auditable and evidence-linked.

## Environment controls

Local demos use `ExecutionTarget.SIMULATED`. External deployment mutation remains
false by default and requires `SLOFORGE_ALLOW_EXTERNAL_DEPLOYMENT_MUTATION` in the
surrounding deployment command. Paid resources additionally require a positive
`SLOFORGE_GPU_BUDGET_USD`. Privileged probes, network faults, and GPU clock
changes have separate false-by-default controls.

## Failure behavior

An action failure aborts before traffic migration. Canary regression rolls back
the guarded traffic/state transition in the local reference implementation.
The executor does not generally invoke inverse infrastructure actions referenced
by `rollback_action_id`; automated external compensating rollback is therefore
not claimed.
If old workers cannot drain without interrupting streams, the machine enters
`OPERATOR_REQUIRED` rather than killing them. Invalid or stale snapshots fail
hash validation. Audit truncation retains the newest bounded records and does
not erase the plan/evidence references.

## Demonstrated versus unexercised

Simulation validation, action idempotency, shadow, canary, promotion, drain,
rollback, timeout, and restart behavior are covered by deterministic tests. The
flagship local simulated recovery completed. No external Kubernetes, Dynamo,
Modal, Truss, GPU, NIC, or cloud resource was mutated on the current machine.
