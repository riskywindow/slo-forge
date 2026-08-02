# Autopsy evidence model

The canonical event schema is `sloforge.autopsy.event/v1`; a capture uses
`sloforge.autopsy.run/v1`.

## Event identity and scope

An event records a stable event ID, typed event kind, host, optional process,
thread, rank, replica, GPU, NIC, NUMA domain, request, model, and operation. The
implemented types cover request and queue stages, scheduler, prefill/decode/token
stream, CPU launch, GPU compute/memory/kernel, collective/wait, expert
dispatch/combine, KV and network/PCIe/NVLink transfer, storage, startup,
readiness, controller, recovery, fault, and health.

Raw start/end timestamps are integer nanoseconds and name their source clock.
Normalized timestamps are optional but must appear as a pair with alignment
confidence and uncertainty. The model rejects negative or reversed intervals.

## Causality and physical binding

`parent_event_id` encodes nesting and `dependency_event_ids` encodes causal
predecessors. Self-dependencies, duplicate dependencies, and references outside
the run are invalid. `ResourceRef` names a physical or logical resource, optional
topology node, and contention domain.

Counters are typed name/value/unit triples with finite values and unique names
per event. Each event has an `EvidenceRef` with source, artifact URI, SHA-256, and
optional record index. This prevents a report from citing an untraceable scalar.

## Capture envelope

`AutopsyRun` includes topology, plan, and workload SHA-256 fingerprints;
reference host; events; per-host alignments; exact fault intervals; artifact
references; and warnings. Sources are `live`, `simulator`, `synthetic_fixture`,
or `imported_trace`. Ground-truth labels are evidence for evaluation only and are
not consumed by diagnosis rules.

## Differential and diagnosis records

`DifferentialComparison` retains matched-stage and counter deltas, rank skew,
first divergence, sample count, alignment quality, and warnings.
`DiagnosisRecord` retains all hypotheses, not just top-k. An evidence statement
stores observed value, threshold, relation, support/contradiction, event IDs, and
plain-language explanation. Counterfactual estimates include expected and bound
improvements and the remaining healthy gap.

The flagship evidence chain is indexed by
`artifacts/fabric-demo/manifest.json`; the manifest contains SHA-256 values for
healthy/degraded runs, comparison, diagnosis, counterfactuals, recovery, traces,
metrics, and timeline.

