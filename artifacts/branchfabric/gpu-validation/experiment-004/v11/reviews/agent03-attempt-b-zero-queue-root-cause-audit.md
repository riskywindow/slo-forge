# Agent 03 attempt-b zero-queue root-cause audit

Recommendation: **FAIL — remediate offline before attempt c**

Audit time: `2026-08-19T01:17:15Z`

This was a read-only audit. It did not invoke vLLM, CUDA, Modal, the network,
or a GPU. It did not modify the frozen v10 path.

## Failure classification

Attempt b failed safely before destination allocation or admission at:

```text
V11RuntimeTeardownRequired:
v11 restore consumed preexisting zero-queue evidence; runtime teardown required
```

The failure is a source/destination allocation-ledger lifecycle gap, not
evidence of corrupt captured bytes, an engine-step race, or an invalid reason
to relax the restore guard.

vLLM 0.23's `KVCacheManager.allocate_slots` reports newly allocated native
blocks through the destructive `take_new_block_ids()` queue. The frozen v10
stager already models the relevant production distinction: when
`scheduler.needs_kv_cache_zeroing` is false, old allocation notifications may
remain; when it is true, pending IDs are a zeroing obligation and must not be
silently consumed. The v11 stager intentionally strengthened its entry rule to
reject every preexisting ID so destination allocation evidence cannot be mixed
with an older lifetime.

The micro worker created the exact live source state, exported and released it,
then reached restore without ever separating the source allocation
notifications from future destination allocation notifications. No supported
request allocation occurs between source release and the failing restore-entry
check. The strongest code-and-timeline inference is therefore that the pending
queue contains source-lifetime allocation notifications retained by the
non-zeroing production scheduler path.

The attempt-b failure artifact did not serialize the affected block IDs or the
runtime `needs_kv_cache_zeroing` value. Consequently, it does **not** prove that
the observed queue was exactly the 1,152 current source IDs. That exact-match
claim must remain a fail-closed assertion in the remediation and be recorded by
the next run; it must not be assumed from the exception text.

## Ruling on the proposed drain

A one-time destructive drain under `EXPORT_CAPTURE` is lifecycle-correct and
scientifically preferable to v10's broad "ignore stale queue when zeroing is
disabled" behavior, provided every condition below is enforced:

1. `scheduler.needs_kv_cache_zeroing` is explicitly and observably `False`.
   If it is true, do not consume the queue; abort because the IDs can represent
   pending native zeroing work.
2. The sealed production witness is valid for the exact scheduler and manager
   with operation `EXPORT_CAPTURE`.
3. The expected set is cross-checked across the canonical capture plan, the
   allocator-issued epoch map, and the pre-release ownership snapshot.
4. The expected set contains exactly 1,152 unique, nonnegative live source
   block IDs for the fixed 16K/fanout-8 topology.
5. The optimized capture and its final D2H/integrity completion fence have
   passed before the queue is destructively drained.
6. The observed queue contains exactly those 1,152 IDs once each: no missing,
   duplicate, or unrelated ID.
7. The successful queue evidence is persisted with count, sorted-ID hash,
   zeroing flag, gate identity, and the source plan/ownership commitments.
8. Source release occurs only after this exact evidence. Restore then requires
   an empty entry queue and continues to require every destination allocation
   drain to equal the new destination-table delta.

This ordering cleanly closes two lifetimes:

```text
source allocate -> source queue -> capture/ownership proof -> exact drain -> release
destination allocate -> destination queue -> exact drain -> write/validate -> admission
```

No source ID is blessed as a destination ID, and no unrelated ID is ignored.

## Defects in the current offline patch

The current `consume_exact_source_allocation_queue_v11` implementation has the
right exact-set core, but it is not yet sufficient for attempt c:

- It drains before pre-release ownership capture and before optimized capture
  completes. Move it after both successful proofs and immediately before
  `release_source_and_prove_v11`.
- It accepts only a manager and an expected tuple. It does not require or
  validate the sealed `EXPORT_CAPTURE` witness or the exact scheduler
  identity.
- It does not assert `scheduler.needs_kv_cache_zeroing is False`.
- `manager.take_new_block_ids` is a mutating production surface but is absent
  from `Vllm0230EngineStepBinding._REQUIRED_SURFACES`. Add it so all direct and
  scheduler-driven drains traverse the installed bounded lock.
- A destructive mismatch raises plain `RuntimeError` and discards the observed
  IDs. Raise typed teardown-required evidence and serialize observed count/hash,
  missing IDs, unexpected IDs, duplicates, expected count/hash, gate identity,
  and zeroing flag before process cleanup.
- The unit test proves only tuple/set behavior on a local manager. It does not
  prove witness rejection, zeroing-mode rejection, binding exclusion of a
  concurrent drain, ordering after capture completion, or separation between
  source and destination queues.

Because the mismatch path is destructive, a typed mandatory teardown remains
the correct response. The worker's process-finally cleanup makes the current
code operationally fail closed, but it does not provide the required proof or
provenance.

## Mandatory offline tests

Before attempt c, add deterministic tests for:

- exact 1,152-ID success with `needs_kv_cache_zeroing=False`;
- missing, extra, duplicate, and stale/reused source IDs;
- `needs_kv_cache_zeroing=True` rejection before destructive drain;
- absent, stale, wrong-operation, foreign-scheduler, and foreign-manager gate
  witnesses;
- a competing direct `take_new_block_ids` call blocked by `EXPORT_CAPTURE`;
- source ownership/capture completion preceding the single drain and source
  release following it;
- an empty restore-entry queue after source drain;
- exact destination allocation-delta drains whose union is the 1,152 fresh
  destination blocks; and
- structured mismatch evidence surviving worker failure serialization.

The existing relevant suite still passes (`42 passed in 1.33s`), which confirms
that attempt b exposed a missing production lifecycle case rather than a known
test regression. That suite is not sufficient until the cases above exist.

## Go/no-go

The exact-set source drain design is **APPROVED IN PRINCIPLE**. The current
implementation is **FAIL / NO-GO FOR ATTEMPT C** until the witness, zeroing-mode,
ordering, binding-surface, typed-evidence, and lifecycle-test requirements are
implemented and independently re-audited.

Audited attempt-b failure SHA-256:
`17871ec58c1664d5b9e146e94ff50b97219e01ec0bbff417b25c77a5d72a3007`.
