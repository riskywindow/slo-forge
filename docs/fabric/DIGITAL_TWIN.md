# Communication-aware digital twin

The Fabric twin is a deterministic, flow-level discrete-event simulator in
`crates/sloforge-fabric-sim`. It consumes schema `1.0` JSON and emits operation
outcomes, per-resource metrics, applied faults/counterfactuals, uncertainty, and
Chrome/Perfetto trace events.

## Resource model

Resources are CPU core groups, NUMA memory, GPU compute, GPU HBM, GPU copy
engines, NVLink, NVSwitch, PCIe, NIC queues, network rails, and storage paths.
An exclusive resource serializes operations. A fair-share resource admits up to
`max_concurrency` and divides calibrated capacity. `SharingGroup` represents a
cross-resource bottleneck such as a PCIe switch or oversubscribed rail.

Every resource references a monotone message-size service curve. Each curve
records `measured`, `synthetic`, or `analytical` provenance, artifact URI/hash,
environment fingerprint, capture time, and uncertainty.

## Operation DAG

Operations include CPU launch, GPU compute, HBM access, point-to-point transfer,
collective, expert dispatch/combine, KV transfer, storage fetch, startup, and
synchronization. An operation has dependencies, earliest start, ranks, resource
demands, request identity, and uncertainty. Completion releases dependents. A
collective completes only after its scheduled shared work completes; rank
slowdown therefore becomes group delay.

Python's `build_simulation_request` lowers a `PhysicalExecutionPlan`, topology,
profile, and typed workload shapes to this protocol. `run_simulation` invokes
the Rust binary with a bounded input, output limit, timeout, and process group.

The Rust protocol can express per-layer, per-token, and expert operations, but
the current Python request lowerer does not reconstruct that full runtime DAG.
It lowers decode as one calibrated GPU operation and represents pipeline stages
at stage granularity; layer-level communication overlap, per-token expert
dispatch/combine, pipeline microbatch bubbles, and PP send/receive remain
approximate. A multi-edge demand uses additive path service time and slowest-link
fair sharing, a conservative flow/store-and-forward approximation rather than a
cut-through packet model.

## Faults and counterfactuals

Timed faults use half-open intervals and ground-truth labels. Implemented effects
are resource-rate reduction, resource unavailability, rank slowdown, and
collective delay. Counterfactuals can remove a fault, rescale a resource curve,
rescale a rank, or replace a resource. Modifiers apply to a copy of the input;
the original evidence remains immutable.

## Determinism and safety

The seed, stable sort keys, bounded event count, bounded operation count, and
deterministic tie-breaking define reproducible ordering. Validation rejects
duplicate resources/operations, missing dependencies, cycles, invalid demands,
impossible replacements, non-positive rates, non-monotone curves, and unsupported
schema versions. Resource conservation, non-negative durations, dependency
ordering, sharing limits, analytical queue cases, two-node transfer cases, and
counterfactual behavior are covered by Rust tests.

## Output interpretation

`makespan_us` is elapsed simulated time; `total_work_us` is summed operation work.
`overlap_efficiency` reports work hidden by concurrency, not GPU occupancy.
Resource utilization is busy time divided by makespan and is bounded to one.
Prediction bounds aggregate declared curve uncertainty. They are model
intervals, not independently calibrated real-hardware confidence intervals.

The CPU flagship artifacts are
`artifacts/fabric-demo/simulations/{healthy,degraded,restored}.json`. They are
synthetic validation of scheduling, contention, fault propagation, and recovery.
Real-hardware residual evaluation has not been performed on this machine.
