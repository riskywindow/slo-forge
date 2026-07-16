# Continuum interview deep dive

## The problem

Inference runtimes normally own active state as private allocations. Copying KV pages alone cannot preserve RNG progression, recurrent state, guided decoding, workflow continuation, token delivery watermarks, or exclusive output ownership. Continuum makes those semantics explicit and uses a compiler plus transaction protocol to move them.

## Why two schemas?

Logical state answers what information continuation requires. Physical layout answers how a runtime stores it. Keeping them distinct lets TP resharding, page changes, and K/V packing compile as exact relayouts without treating runtime pointers as an ABI. It also makes incompatibility visible when state-producing equations change despite identical tensor shapes.

## Why a capsule is not a checkpoint file

A capsule is a canonical manifest plus logical/physical/transaction contracts and evidence. Segment bytes live in a tenant-scoped CAS and can be incremental or shared by forks. The capsule binds content, versions, owner epoch, journal, exactness, and required verification. A valid capsule still needs authorization and compatibility before resume.

## Direct conversion versus canonical conversion

Canonical decode/re-encode is simple and trusted but requires full logical materialization. Direct conversion reads a bounded logical token slice from source shards and writes destination shards immediately. Its output is quarantined until the canonical path produces identical destination bytes. This preserves a small trusted base while allowing compiler optimization.

## What makes cutover transactional?

Source and destination can both hold bytes, but only one epoch may mutate committed state or produce accepted output. The source quiesces and is fenced, destination state validates, the coordinator atomically changes the lease epoch/token watermark, and the gateway then changes its expected epoch. Duplicate indices are idempotent, gaps and stale epochs fail closed.

## Failure subtlety

Before the CAS commit, destination failure can restore a still-valid source. After commit, especially after new destination output, returning to old source is not rollback; it is another recovery transaction or recomputation. Continuum records these windows explicitly instead of claiming universal rollback.

## Why it is broader than KV transfer

The exercised capsule holds KV plus recurrent, sampler, guided-decoding, history, and delivery state. The flagship changes adapter, layout, TP, packing, page size, and epoch; transfers live deltas; injects and recovers a failed cutover; forks using COW; and rejects a semantically invalid same-shape revision. Transport is only the byte-moving layer.

## Evidence boundaries

Direct/canonical equality, transaction artifacts, gateway indices, and fork storage derive from deterministic CPU runs. Fabric/network time may be synthetic; host converter timings are observed; GPU/runtime integration is reported as unexercised where packages or hardware are absent. The model checker explores a finite state space with declared bounds and assumptions.
