# Agent 03 attempt-c recorder lifecycle audit

Recommendation: **FAIL — no further GPU attempt before offline remediation**

Audit time: `2026-08-19T01:52:31Z`

This was a read-only production-lifecycle review. It did not invoke vLLM,
CUDA, Modal, the network, or a GPU, and it did not modify the frozen v10 path.

## Root cause

Attempt c crossed the prior source-allocation gate and reached direct native
restore. It then failed while opening a CUDA pass record for raw-bit
destination validation:

```text
RuntimeError: v11 pending CUDA pass bound exceeded
```

The same full recorder was passed to failure scrubbing, so the scrub could not
open its first diagnostic CUDA record. The inner write helper and outer stager
both attempted fail-closed scrub; neither could prove it. The stager therefore
raised `V11RuntimeTeardownRequired`, did not admit restored branches, and the
worker/container cleanup reclaimed the process. Attempt c is scientifically
invalid, but its safety response was correct.

The bound exhaustion is not evidence of bad state bytes. It is a deterministic
ledger-cardinality mismatch. The restore planner may allocate physical block
IDs in descending or otherwise nonconsecutive order. `_consecutive_runs`
coalesces only ascending `+1` runs, so raw-bit validation emits as many as one
CUDA record per page per layer. For 1,152 pages and 28 Qwen layers, validation
alone can produce 32,256 CUDA records. Adding H2D, per-layer index upload and
scatter records places the exact workload around 34,000 CUDA records, well
above the recorder's 4,096 simultaneously pending-event bound.

The problem is simultaneous live CUDA event objects, not the scientific need
to retain all resolved canonical records.

## Incremental-resolution ruling

Incremental pending-event resolution is **safe inside the held
`IMPORT_ADMISSION` critical section** if it is non-final and preserves one
authoritative ledger.

`CUDA Event.synchronize()` may release the Python GIL while waiting, but it does
not release `Vllm0230EngineStepBinding`'s re-entrant lock. Competing production
engine-step, scheduler, allocator, block-table, cache-publication, and queue
mutations therefore remain excluded. The sealed witness's owner thread,
acquisition generation, scheduler identity, and manager identity do not change.
Completing previously launched writes and validations only strengthens the
pre-admission fence.

A correct recorder needs two distinct operations:

- `flush_pending()` resolves currently pending CUDA events into immutable
  `V11StatePassRecord`s, clears only live event references, retains all claimed
  pass IDs and resolved records, and remains open for append.
- final `resolve()` flushes the remaining events and permanently freezes the
  recorder.

Dependencies may cross incremental flush boundaries. Final authoritative
validation must run once over the merged ledger, proving global pass-ID
uniqueness, dependency closure/acyclicity, timing ownership, device/branch
identity, tier conservation, and physical-byte conservation. No profiler or
copy counter may be added to these totals.

Incremental flush may occur automatically at a documented normal-lane
watermark or explicitly at bounded chunk boundaries. Either design must keep
the maximum live normal CUDA events bounded, retain a separately bounded total
record limit sized for the exact topology, and record resolution wait time
without double-counting it as another physical-memory pass.

## Emergency scrub capacity ruling

Failure scrub must not depend on unused capacity in the nominal measurement
lane. It requires a **nonborrowable, bounded emergency accounting lane**.

For the current 28-layer scrub, one invocation emits at most:

```text
1 destination-index upload
+ 28 native zero passes
+ 28 raw-zero verification passes
= 57 CUDA passes
```

The write helper can scrub a failed subset and the outer stager can then scrub
the full staged mapping. Those identities can differ, so the safe nested bound
is 114 CUDA passes. Reserve at least 128; 256 is the conservative bounded
choice. Normal capture/restore work must be unable to consume this reserve.

Emergency use must be explicit and limited to `DIAGNOSTIC` scrub operations.
It must retain event/stream/byte/tier evidence in the same authoritative ledger
with disjoint pass IDs. Exhaustion or CUDA failure in the emergency lane must
raise typed teardown and prevent allocator reuse or admission.

## Adversarial offline tests required

Before another GPU allocation, require tests for:

1. a tiny normal bound that flushes several times, remains appendable, and
   yields the same final records/totals as one-shot resolution;
2. dependencies that cross flush boundaries and still validate globally;
3. duplicate IDs rejected across already-flushed and pending batches;
4. final resolution freezing append while non-final flush does not;
5. pending-event and total-record bounds enforced independently;
6. `IMPORT_ADMISSION` witness validity before and after a blocking incremental
   flush;
7. a competing engine step and direct allocator/queue mutation remaining
   blocked while event synchronization releases the GIL;
8. nominal work unable to borrow the emergency reserve;
9. scrub succeeding with the nominal lane exactly full;
10. two nested 57-pass scrubs fitting the documented reserve;
11. emergency-reserve exhaustion producing typed teardown and no admission or
    allocator reuse; and
12. the exact 1,152-page/28-layer descending-destination cardinality completing
    without a pending-bound failure.

The currently focused offline tests pass (`53 passed in 1.44s`) but contain no
incremental resolution or emergency-reserve test, so they do not close this
production failure.

Final decision: **incremental resolution is synchronization-safe; emergency
scrub capacity is mandatory; current code is NO-GO until both are implemented,
tested, and independently re-reviewed.**

Attempt-c failure SHA-256:
`1fb2cb6f296030e301f0864cf240550ff721b248d128adf9af22ee48e2887b59`.
