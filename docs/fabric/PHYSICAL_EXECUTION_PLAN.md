# PhysicalExecutionPlan

`PhysicalExecutionPlan` is the canonical, versioned physical IR. It composes
with a logical `DeploymentPlan` by immutable `DocumentReference`; it does not
copy or reinterpret the logical deployment contract.

## Envelope and identity

The current schema is `1.0.0` with API version `sloforge.io/fabric/v1`. A plan
contains `kind`, `plan_id`, compiler version, Git commit, logical-plan reference,
model graph hash, topology fingerprint, fabric profile hash, reproducibility
metadata, and evidence references. `canonical_json` sorts object keys and uses a
stable representation; `canonical_hash` is SHA-256 over those bytes.

JSON Schema lives at
`schemas/fabric/physical-execution-plan-v1.schema.json`. Python models live in
`python/sloforge/fabric/ir/models.py`; Rust wire models live in
`crates/sloforge-fabric-protocol/src/model.rs`. Golden documents are under
`tests/fixtures/fabric/`. Cross-language conformance checks canonical bytes and
hashes, not merely field presence.

## Composed documents

- `TopologyGraph` contains typed host, socket, NUMA, memory, GPU, NVSwitch,
  PCIe, NIC, rail, storage, and optional remote-memory nodes. Edges include
  directionality, duplex behavior, latency/bandwidth curves, sharing and
  contention domains, health, confidence, timestamp, environment, and source.
- `ModelGraph` describes ordered layers, attention/feed-forward/MoE structure,
  experts, parameter and activation bytes, KV bytes per token, communication
  requirements, partition constraints, precision, and runtime features.
- `FabricProfile` contains benchmark series, raw-sample references, invocation,
  placement, environment, hardware/software manifests, status, and failure
  reason.

## Execution mapping

`ParallelismPlan` fixes tensor, pipeline, data, expert, and context degrees,
disaggregation, communication groups, and replica groups. `RankPlacement` maps
each integer rank to host, GPU, NUMA domain, optional NIC and rail, CPU affinity,
worker role, replica, and fault domain.

The v1 pipeline representation identifies PP rank groups and available stage
boundaries but does not include an explicit layer-to-stage map or PP send/receive
schedule. PP output is therefore physical placement intent for runtime lowering,
not a complete executable layer schedule. Runtime adapters must validate their
own partition mapping rather than infer one from rank order.

`ExpertPlacement` assigns model experts to ranks with replication, load
assumption, hot-expert flag, capacity factor, and rebalance constraints.
`CollectivePlan` describes collective kind, ranks, message model, algorithm,
transport, channels, rail, ordering, duration interval, overlap window,
synchronization dependency, and fallback. `KVTransferPlan` describes
producer/consumer ranks, path, serialization, chunking, overlap, ownership,
eviction, retry, fallback, transport, and expected latency/cost.

`MemoryPlan` accounts per rank for weights, KV, activations, CUDA graphs,
workspace, communication buffers, pinned host memory, local and remote artifact
bytes, safety margin, and fragmentation allowance. `CommunicationOverlapPlan`
links compute and communication operations to streams, dependencies, expected
overlap, contention, and a serialization fallback.

## Outcomes and recovery

Predicted TTFT, TPOT, end-to-end latency, throughput, goodput, cost,
availability, and communication overhead are intervals with unit and confidence.
The plan records a predicted bottleneck, fault-domain exposure, optimizer trace,
all rejected alternatives, and `RecoveryVariant` objects. Each variant has typed
triggers, alternate parallelism/placement/transport/worker ratio, degraded-mode
metrics, transition time/cost, rebuild requirements, and compatibility rules.

## Validation invariants

Core validation rejects duplicate IDs, missing graph references, invalid degree
products, repeated ranks, memory totals that disagree with components, invalid
group membership, self-referential links, unsorted curves, inverted intervals,
and unhashed evidence. The compiler additionally binds a profile to the exact
canonical topology hash.

Namespaced extensions are the only open map. A namespace must be a reverse-DNS
identifier; extension values remain JSON but cannot replace a core field.

## Migrations and compatibility

`python/sloforge/fabric/ir/migrations.py` and
`crates/sloforge-fabric-protocol/src/migration.rs` normalize supported v1 input
into the current minor representation. Major-version changes require an
explicit migration and new fixtures. Readers reject an unsupported major rather
than guessing. Writers always emit the current schema.
