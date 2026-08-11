# Fabric Simulator and Optimizer Review

Review date: 2026-08-01  
Scope: `crates/sloforge-fabric-sim/**`, `python/sloforge/fabric/simulation.py`,
`python/sloforge/fabric/compiler/**`, and focused tests.  This was a CPU-only
code and synthetic-fixture review; it is not a GPU, NCCL, RDMA, or multi-node
hardware validation.

## Invariants reviewed

- deterministic event ordering, bounded events and operations, acyclic graphs,
  non-negative time, explicit deadlock, exclusive admission, fair-share
  capacity, shared contention domains, resource accounting, and fault windows;
- collective barriers and algorithms, rank and link fault scope, permanent and
  temporary resource loss, counterfactual removal/scaling/replacement, and
  transformed-plan validation;
- canonical Python-to-Rust lowering, TP/PP/DP worker staging, request-to-replica
  assignment, required transport reachability, KV routes and byte volume;
- static feasibility, memory sharding/replication, degree enumeration,
  deterministic placement, fault-domain availability, hard SLO constraints,
  objective math, Pareto dominance, optimizer history, recovery alternatives,
  and truthful simulator-call accounting.

## Findings and dispositions

| Severity | Finding | Disposition |
|---|---|---|
| High | Counterfactual transformations were not revalidated, and unknown rank/collective targets could silently do nothing. | Fixed in `bcb213a`: validate original targets, revalidate the transformed graph, and fail on capacity/type violations. |
| High | A transfer traversing multiple edges in one sharing group counted itself once per edge; independent faults on separate hops multiplied into a fictitious quadratic slowdown. | Fixed in `bcb213a`: one demand per flow/contention domain and slowest-hop flow semantics, with analytical tests. |
| High | PP stages were emitted as concurrent full-plan compute operations, creating impossible overlap and incorrect GPU occupancy. | Fixed in `5e61816`: recover PP stage indices, split plan-level compute across stages, and add explicit stage dependencies for prefill and decode. |
| High | Required collective/KV communication could disappear when a path or permitted transport was unavailable. | Fixed in `5e61816`: candidates are rejected with explicit path/transport reason codes. |
| High | Repeated same-size profile measurements could become adjacent extrapolation points with a zero byte span, amplifying sample variance into multi-hour predicted communication. | Fixed in `5e61816`: robustly collapse duplicate byte sizes, interpolate within measured support, and extrapolate only non-negative marginal serialization cost. |
| High | Availability treated replicas on a shared host as independent and did not express the conjunction of disaggregated prefill/decode pools. | Fixed in `5e61816`: exact cached Boolean evaluation over per-fault-domain availability, including shared-domain correlation. `base_availability` is explicitly interpreted per independent fault domain. |
| Medium | Greedy rank ordering optimized a global sequence rather than complete TP/PP replica groups; robust-failure placement was not behaviorally distinct. | Fixed in `5e61816`: group-aware replica placement plus a distinct complete-replica host-spreading policy for robust-failure mode. |
| Medium | “Exhaustive” degree search visited only powers of two and fixed non-power-of-two degrees could vanish from the search space. | Fixed in `5e61816`: exhaustive mode visits every degree in tiny spaces; fixed constraints are always included. |
| Medium | Pareto dominance omitted goodput, communication, and availability, and optimizer history labeled every feasible candidate as Pareto-promoted. | Fixed in `5e61816`: all required minimize/maximize directions participate and history distinguishes dominated candidates. |
| Medium | Hard constraints used point estimates even though the emitted plan carried prediction intervals. | Fixed in `5e61816`: latency, goodput, and cost constraints use conservative interval bounds. Availability remains driven by its separate fault-domain assumption rather than an unrelated latency residual. |
| Medium | KV routing covered only a subset of unequal prefill/decode pool pairings, and lowering modeled only the inflight cap rather than total compiler-known KV bytes. | Fixed in `5e61816`: emit every pool pairing and preserve the compiler-known chunk count/byte volume under bounded plan backpressure. |
| Medium | Recovery counterfactual replacement could preserve capacity validity while changing an operation to an incompatible physical resource kind. | Fixed after the main review commits: operation/resource compatibility is revalidated after transformation. |

No reviewed high-severity finding remains open.

## Remaining modeling boundaries

- The simulator is a calibrated flow/collective-level digital twin, not a packet
  simulator. A multi-edge operation uses additive calibrated path service time
  and slowest-resource fair sharing; it does not model per-packet store-and-forward.
- Communication overlap is represented by the compiler's measured overlap
  windows and by concurrent dependency-graph operations. The current opaque
  request lowering does not reconstruct individual model-layer kernels, so
  layer-by-layer overlap and pipeline microbatch bubbles remain approximate.
- The compiler records zero simulator calls because simulation is a separate,
  explicit validation pass. Wall-clock solver time is measured by evaluation;
  deterministic plan IR does not embed nondeterministic timing samples.
- The canonical profile currently groups several collective primitives into one
  `collective` category. Duplicate sizes are robustly aggregated, but separate
  all-reduce/all-to-all calibration would require a backward-compatible profile
  schema evolution.
- Prediction intervals use supplied/calibrated relative uncertainty rather than
  claiming hardware generalization. The checked-in CPU fixture evaluation is
  synthetic internal validation only.

## Acceptance evidence

Commands executed successfully during this review:

```text
cargo fmt --all -- --check
cargo test -p sloforge-fabric-sim
  1 unit + 10 analytical + 1 CLI + 2 property + 1 two-node + 6 validation tests passed
cargo clippy -p sloforge-fabric-sim --all-targets --all-features -- -D warnings
  passed with zero warnings
uv run ruff format --check python/sloforge/fabric/compiler/core.py python/sloforge/fabric/simulation.py tests/python/test_fabric_compiler.py tests/python/test_fabric_simulation_bridge.py
uv run ruff check python/sloforge/fabric/compiler/core.py python/sloforge/fabric/simulation.py tests/python/test_fabric_compiler.py tests/python/test_fabric_simulation_bridge.py
  passed
uv run mypy --strict python/sloforge/fabric/compiler/core.py python/sloforge/fabric/simulation.py
  passed, no issues in 2 files
uv run pytest -q tests/python/test_fabric_compiler.py tests/python/test_fabric_simulation_bridge.py tests/python/test_fabric_evaluation.py::test_artifact_derived_evaluation_round_trip
  27 passed
```

An isolated one-seed synthetic matrix (`seed=13`, temporary `/tmp` artifacts,
not the final checked-in evaluation) produced 9.11% median relative digital-twin
error, 53.33% interval coverage, 0.8684 rank correlation, and 1.741% Pareto
selection regret. Median p95 TTFT was 550.24 ms for both hierarchical/greedy and
the already topology-aligned sequential fixture, 540.40 ms for the topology-
unaware optimizer that was also allowed to change parallelism, and 1072.45 ms
for random placement. This is an honest provisional negative result: group-aware
placement prevented pathological rank mixing but did not beat the fixture's
already optimal sequential order. Final benchmark claims must come from the
regenerated committed evaluation artifacts, not these temporary numbers.
