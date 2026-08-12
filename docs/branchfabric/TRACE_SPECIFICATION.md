# BranchFabric Characterization Trace Specification

## Status and scope

This document specifies the event-oriented `BranchWorkloadTrace v1`,
`StateOperationTrace v1`, and their `TraceManifest v1`. The normative machine
schemas are:

- `schemas/branchfabric/branch-workload-trace-v1.schema.json`
- `schemas/branchfabric/state-operation-trace-v1.schema.json`
- `schemas/branchfabric/trace-manifest-v1.schema.json`

The Python models in `sloforge.helix.characterization.trace.models` and the Rust
wire models are conformance implementations of those schemas. The event stream
defined here is separate from the legacy Helix scheduled-workload document at
`schemas/helix/branch-workload-trace-v1.schema.json`; neither format is a
substitute for the other.

The v1 state-operation stream is intended to remain usable as input to later
software ISA replay, architecture exploration, and benchmark generation. It is
measurement input, not a BranchFabric implementation or hardware ISA.

## Common event contract

Both event kinds are strict objects: undeclared properties are rejected, scalar
coercion is rejected by the reference models, numeric values must be finite,
and every required property must be present even when its value is `null`, zero,
or `unknown`. The optional `attributes` object is bounded to 64 entries. Its
keys contain 1--128 characters, and the reference model limits string values to
4,096 characters.

Every event carries these common groups:

| Group | Required fields | Contract |
| --- | --- | --- |
| Version | `schema_version`, `kind`, `trace_producer_version`, `collection_level` | Identifies the exact event schema, event discriminator, producer, and active trace level. |
| Evidence | `provenance`, `timing_measurement_class` | Classifies the workload independently from the source of timing values. |
| Trace identity | `trace_id`, `session_id`, `branch_group_id` | Joins streams without manufacturing a branch group before one exists. `branch_group_id` is a required nullable key. |
| Execution identity | `host`, `process_id`, `rank`, `device` | Locates the producer. Nullable rank or device means it was not applicable or not observed. |
| Time | `monotonic_timestamp_ns`, `normalized_timestamp_ns`, `duration_ns`, `clock_source`, `alignment_confidence` | Preserves the local monotonic time, producer-normalized time, duration, clock identity, and confidence in time alignment. |
| Integrity and order | `event_sequence`, `content_hash` | Makes ordering, mutation, filtering, and buffer loss observable. |

`provenance` and `timing_measurement_class` each use exactly one of:

- `SYNTHETIC`
- `REPLAYED`
- `HARDWARE_BACKED_REAL`
- `SIMULATED_HARDWARE`

The two fields answer different questions. For example, a deterministic CPU
fixture timed with `time.perf_counter_ns` remains `SYNTHETIC` workload evidence
while its local host timing can be `HARDWARE_BACKED_REAL`. A modeled network
delay is `SIMULATED_HARDWARE` timing and must not be relabeled because other
operations in the same run used real host clocks.

Supported clock sources are `monotonic`, `monotonic_raw`, `perf_counter`,
`cuda_global_timer`, `cupti`, and `synthetic`. A normalized timestamp is only as
portable as the producer's recorded alignment. Cross-host ordering must not be
inferred when alignment confidence and manifest evidence do not support it.
Durations on different events can describe the same measured span. Producers
may place a shared `timing_span_id` and duration-attribution marker in
`attributes`; analyses must deduplicate those spans rather than summing them.

## BranchWorkloadTrace v1

### Purpose

`BranchWorkloadTrace v1` records high-level Helix branch, learning, state, and
movement observations. Its schema version is
`sloforge.branchfabric.branch-workload-event/v1`, and `kind` is
`BranchWorkloadTraceEvent`.

### Required fields

In addition to the common fields, every branch-workload event contains the
following required keys. Nullable keys remain required so that absence is
explicit.

| Group | Required fields | Meaning |
| --- | --- | --- |
| Branch semantics | `branch_id`, `parent_branch_id`, `policy_epoch`, `environment_id`, `transaction_id` | Branch ancestry and the policy, environment, and transaction contexts known at the event. |
| Operation and state | `operation_type`, `logical_state_id`, `physical_state_id`, `state_segment`, `page`, `version`, `source_epoch`, `destination_epoch` | Operation identity and the affected logical/physical representation. |
| Size | `logical_bytes`, `physical_bytes`, `compressed_bytes`, `transferred_bytes`, `metadata_bytes` | Distinct byte domains; zero means no bytes observed for that field, not that other byte domains are interchangeable. |
| Location | `location`, `source_location`, `destination_location` | State residence and movement endpoints. |
| Sharing | `shared_root`, `private_suffix`, `cow_allocation` | Declares shared-root, private-suffix, and allocation-causing COW semantics. |
| Performance | `queue_delay_ns`, `execution_latency_ns`, `transfer_latency_ns`, `transform_latency_ns`, `wait_latency_ns`, `cpu_cycles`, `cpu_time_ns`, `gpu_duration_ns` | Separates queue, execution, transfer, transform, wait, CPU, and GPU observations. Nullable counters were not available or not applicable. |
| Transport | `transport_type`, `transport_source`, `transport_destination`, `chunk_size_bytes`, `fanout`, `retransmission`, `error` | Describes the movement path, granularity, destination count, retry traffic, and failure detail. |
| Hardware | `gpu_model`, `nic`, `numa_node`, `pcie_path`, `network_rail`, `memory_tier` | Event-local hardware placement when measured. |

The operation vocabulary is:

- Branch and learning lifecycle: `BRANCH_POINT`, `CAPTURE`,
  `ENVIRONMENT_FORK`, `BRANCH_FORK`, `BRANCH_READY`, `BRANCH_DIVERGENCE`,
  `BRANCH_PRUNE`, `BRANCH_ABORT`, `BRANCH_COMMIT`, `BRANCH_COMPLETE`,
  `BRANCH_MIGRATION`, `CHECKPOINT`, `ROLLOUT`, `REWARD`, `TRAIN`, `EVALUATE`,
  `CANARY`, `PROMOTE`, `ROLLBACK`, and `LEARNING_TRANSACTION_STAGE`.
- State lifecycle and movement: every operation in the StateOperationTrace v1
  vocabulary below.

The state segment vocabulary is `model`, `token_history`, `kv`, `recurrent`,
`sampler`, `guided_decoding`, `workflow`, `environment`, `filesystem`,
`database`, `process_reconstruction`, `transaction`, `integrity`,
`runtime_reconstructible`, and `unknown`.

Locations are `gpu_hbm`, `gpu_peer_hbm`, `host_dram`, `pinned_memory`,
`local_nvme`, `remote_storage`, `nic`, `transport_buffer`, and `unknown`.
Transport types are `none`, `memory_copy`, `pcie`, `nvlink`, `nccl`, `tcp`,
`rdma`, `shared_memory`, `storage`, and `synthetic`.

`cow_allocation=true` is valid only for `STATE_COW`. A branch cannot name itself
as its parent. An `error` string, when present, must be non-empty.

## StateOperationTrace v1

### Purpose

`StateOperationTrace v1` is the lower-level architecture replay stream. Its
schema version is `sloforge.branchfabric.state-operation-event/v1`, and `kind`
is `StateOperationTraceEvent`.

### Required fields

In addition to the common fields, every state-operation event contains:

| Group | Required fields | Meaning |
| --- | --- | --- |
| State and isolation | `logical_state_id`, `branch_id`, `tenant_id`, `security_domain`, `state_epoch` | Stable logical state identity, optional branch, isolation domains, and version epoch. |
| Operation | `operation_type`, `state_segment`, `source_physical_representation`, `destination_physical_representation` | Operation and source/destination representation contracts. |
| Geometry | `bytes`, `alignment_bytes`, `page_size_bytes`, `chunk_size_bytes`, `fanout` | Operation byte count and physical granularities. Zero geometry means not applicable for a nonpaged or unchunked operation; it never means a fabricated one-byte granularity. |
| Scheduling | `dependency_event_ids`, `concurrency`, `queue_delay_ns` | Explicit dependencies, outstanding-operation context, and time waiting before execution. |
| Timing | `operation_latency_ns`, `cpu_time_ns`, `gpu_time_ns`, `transfer_time_ns` | Separates end-to-end operation, CPU, GPU, and movement time. The fields are not assumed to be additive. |
| Result | `result`, `failure` | One of `success`, `failure`, `retry`, or `skipped`, plus failure detail when and only when the result is `failure`. |
| Placement | `source_location`, `destination_location`, `transport_type` | Physical endpoints and movement mechanism. |

The exact operation vocabulary is:

`STATE_ALLOC`, `STATE_MAP`, `STATE_PUBLISH`, `STATE_FORK`, `STATE_COW`,
`STATE_APPEND`, `STATE_READ`, `STATE_WRITE`, `STATE_SNAPSHOT`, `STATE_DELTA`,
`STATE_RESHARD`, `STATE_TRANSPOSE`, `STATE_REPACK`, `STATE_QUANTIZE`,
`STATE_DEQUANTIZE`, `STATE_COMPRESS`, `STATE_DECOMPRESS`, `STATE_HASH`,
`STATE_CHECKSUM`, `STATE_ENCRYPT`, `STATE_DECRYPT`, `STATE_SEND`,
`STATE_RECEIVE`, `STATE_MULTICAST`, `STATE_ACK`, `STATE_RETRY`, `STATE_COMMIT`,
`STATE_ABORT`, `STATE_RECLAIM`, and `STATE_FREE`.

Dependencies must be unique. The Continuum adapter represents an observation
dependency as `<trace_id>:<source-observation-sequence>`. Consumers must not
infer an unrecorded dependency merely from timestamp proximity.

## Collection levels, filtering, and overflow

The hot-path buffer is bounded, thread-safe, and does not perform storage I/O.
It assigns `event_sequence` to each attempted event before level filtering,
deterministic sampling, or buffer admission.

- `disabled` admits no events. Every attempted event is counted as filtered.
- `minimal` admits declared lifecycle, COW, transaction, checkpoint, migration,
  and selected state-transfer boundaries.
- `full` admits every operation subject to configured deterministic sampling and
  buffer capacity.

The exact minimal branch set is `BRANCH_POINT`, `BRANCH_FORK`,
`ENVIRONMENT_FORK`, `BRANCH_READY`, `BRANCH_PRUNE`, `BRANCH_ABORT`,
`BRANCH_COMMIT`, `BRANCH_COMPLETE`, `BRANCH_MIGRATION`, `CHECKPOINT`,
`STATE_COW`, `REWARD`, `TRAIN`, `PROMOTE`, `ROLLBACK`, and
`LEARNING_TRANSACTION_STAGE`. The exact minimal state set is `STATE_FORK`,
`STATE_COW`, `STATE_SNAPSHOT`, `STATE_DELTA`, `STATE_SEND`, `STATE_RECEIVE`,
`STATE_COMMIT`, and `STATE_ABORT`.

Sampling is either `none` with stride 1 or `deterministic_stride` with an
explicit stride greater than 1 and seed. A full buffer rejects the new event;
it does not evict an older event. Consequently, accepted events preserve attempt
order, while sequence gaps expose filtered, sampled, or dropped attempts.

The manifest must satisfy:

```text
attempted_events = accepted_events + dropped_events + filtered_events
sum(event_counts.values()) = accepted_events
```

`highest_event_sequence` is the highest attempted sequence, including an
attempt that was later filtered or dropped. It is `null` only when there were no
attempts. A trace with nonzero `dropped_events` is valid evidence of a lossy
capture, but an analysis must disclose the loss and must not treat the corpus as
complete.

## Integrity and ordering

Canonical JSON is UTF-8, has lexicographically sorted keys, contains no
insignificant whitespace, rejects non-finite numbers, and preserves Unicode.
To seal an event:

1. Set `content_hash` to 64 zero characters.
2. Canonically encode the complete event.
3. Compute lowercase SHA-256 over those bytes.
4. Store the digest in `content_hash`.

Readers recompute this digest before analysis. The streaming JSONL reader also
requires strictly increasing `event_sequence` values independently for each
`trace_id`. It does not require contiguity because gaps carry loss or sampling
information.

The trace corpus hash is SHA-256 over a canonical object containing `trace_id`
and the sorted artifact identities. Each identity includes format, URI, byte
length, artifact SHA-256, and event count. Because URI is part of that identity,
relocating an artifact requires a new manifest and corpus hash.

## Storage formats

### JSONL

JSONL is the authoritative streaming representation. It contains one sealed
canonical JSON object per nonempty line. The reference writer:

- validates event integrity before writing;
- batches at 1,024 events or 8 MiB by default;
- rejects a record larger than 16 MiB;
- writes incrementally and computes artifact bytes, event count, and SHA-256;
- refuses to replace an existing file unless overwrite is explicitly enabled.

The reader validates schema/model constraints, hashes, and order without loading
the corpus into memory. This is the required path for million-event traces.

### Parquet

Parquet is an optional bulk format requiring PyArrow. There is no silent JSONL
fallback: absence of PyArrow produces an explicit `ParquetUnavailableError`,
and the run records Parquet as unavailable. The current writer uses Zstandard
compression and bounded batches. It indexes `kind`, `trace_id`,
`event_sequence`, `monotonic_timestamp_ns`, and `operation_type`, while
retaining the complete canonical event in `event_json`. Parquet-to-JSONL and
Parquet-to-Perfetto conversions revalidate every canonical event.

### Perfetto/Chrome trace JSON

Perfetto is derived visualization output. It is not raw measurement evidence.
Duration-bearing events become complete events; zero-duration events become
instant events. The export preserves the trace, branch, sequence, content hash,
clock, alignment, provenance, and timing classification in arguments. Branch
and state events use separate categories. Analyses requiring exact integer
nanoseconds or complete fields must read JSONL or Parquet, not the visualization.

## Trace manifest

`TraceManifest v1` binds a corpus to its run context. Required manifest content
includes:

- trace/session identity, creation time, seed, producer, evidence class, and
  collection level;
- buffer capacity, attempted/accepted/dropped/filtered accounting, highest
  attempted sequence, operation counts, and sampling configuration;
- host, machine, CPU, logical CPU count, memory, NUMA, devices, PCIe paths,
  network rails, and clock settings;
- OS, kernel, Python, optional Rust, and versioned component provenance;
- every artifact's format, URI, byte length, SHA-256, and event count;
- the trace corpus hash.

Unknown hardware must be recorded as unknown or absent in the appropriate
field, not replaced with a nominal device or transport. A manifest records what
was available and measured; it does not turn synthetic state sizes into
hardware-backed measurements.

## Conformance and compatibility

Conforming producers must pass the JSON Schema, strict Python model, content
hash, sequence, buffer-accounting, JSONL round-trip, and Rust/Python wire
conformance tests. Derived-format conversion must preserve event identity and
integrity.

Because v1 schemas reject additional properties and enumerate operations, an
uncoordinated new field or operation is not a compatible v1 extension. Such a
change requires coordinated schema/model/wire updates and a versioning decision.
Existing v1 event meanings, evidence labels, zero/null semantics, and hashing
rules must not be reinterpreted.
