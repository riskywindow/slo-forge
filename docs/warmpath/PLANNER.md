# WarmPath planner and local executor

WarmPath minimizes weighted p95 readiness time, hourly storage/warm capacity
cost, and restore failure probability subject to compatibility, security,
capacity, startup SLO, and budget constraints.

## Candidate generation

For every artifact, the planner enumerates eligible tier/materialization pairs:
eager restore, lazy restore, rebuild, or keep-warm. Readiness dependencies cannot
be deferred. Compatibility violations retain rebuild candidates when permitted
and are recorded as rejected snapshot alternatives.

If the Cartesian product and warm-replica options fit `exhaustive_limit`, every
candidate is evaluated. Otherwise a deterministic beam retains the lowest
partial duration choices. Capacity and hourly cost are checked before seeded
cold-start simulation. The selected key is objective, p95 readiness, cost, then
stable tier/mode ordering.

The simulation separately retains p50, p95, interval, failure probability, raw
trials, and stage predictions. The plan records optimizer name, seed, evaluated
count, rejected candidates, graph/profile hashes, and evidence.

## Local execution

`LocalWarmPathExecutor` checks plan/host/graph identity and exact placement
coverage. It enforces per-operation time and artifact-byte limits, protects
against path traversal, reads in bounded chunks, verifies source and destination
SHA-256, atomically fills caches and outputs, tracks LRU access, and evicts only
unprotected entries. Restore failures can be seeded or explicitly injected.

Lazy artifacts are marked deferred and unverified until consumed. A keep-warm
placement materializes no bytes in the reference executor. Execution records
stage status, time, bytes, checksum, source/destination, evictions, success, and
failure reason.

## Current demo evidence

`artifacts/warmpath/manifest.json` records an exhaustive 243-candidate run over
four deterministic snapshot artifacts. It selected three eager placements and
one deferred artifact with no warm replica. The manifest reports measured local
ready time and predicted p50/p95, but the snapshot itself is synthetic. Exact
values should be read from the manifest because repeated local runs can vary.

