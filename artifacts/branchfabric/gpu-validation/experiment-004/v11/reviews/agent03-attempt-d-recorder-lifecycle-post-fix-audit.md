# Agent 03 attempt-d recorder lifecycle post-fix audit

Recommendation: **PASS — attempt d is lifecycle-admissible, subject to the root coordinator's budget and remaining independent gates**

Audit time: `2026-08-19T02:02:47Z`

This read-only review did not invoke CUDA, Modal, the network, or a GPU and did
not modify the frozen v10 path. It supersedes Agent 03's attempt-c recorder
lifecycle FAIL.

## Blocker closure

Attempt c failed because one restore recorder retained more than 4,096 live
CUDA event pairs. Its scrub then tried to use that same full lane and could not
open its first diagnostic record. The corrected recorder now separates three
bounds:

- 3,840 normal live CUDA passes;
- a nonborrowable 256-pass diagnostic emergency lane, within a 4,096 live-event
  hard bound; and
- 65,536 total canonical passes, with the last 256 unavailable to normal work.

At the normal live-event watermark, the recorder synchronizes already-launched
end events and appends immutable records before removing their pending entries.
It retains the global pass-ID set and accumulated records and remains
appendable. Only final `resolve()` freezes it. A partial materialization failure
leaves the unmaterialized suffix retryable, while already-materialized events
are not synchronized or recorded twice.

The observed attempt-c production ordering needs 8,128 forward CUDA records.
The adversarial admitted ordering with singleton destination runs needs 35,904.
Both fit below the 65,280 normal total-record limit, while no longer requiring
all events to remain live simultaneously.

## Production lock verdict

Incremental materialization is safe under the real `IMPORT_ADMISSION` binding.
Every restore CUDA launch that can trigger it executes inside the worker's one
enclosing production critical section. `CUDA Event.synchronize()` may release
the Python GIL, but it does not exit `Vllm0230EngineStepBinding._hold_lock`,
release its RLock, or change the sealed witness's owner or acquisition
generation. The installed engine, scheduler, KV-manager, block-table,
admission-queue, release, and allocation-evidence wrappers therefore continue
to exclude competing mutations throughout the wait.

The final recorder resolution can remain outside the critical section. Each
write subset already executes a compute-stream completion fence and exact
raw-bit validation before returning its typed proof; the stager admits requests
only after all subset, epoch, mapping, and ownership checks pass. Final
resolution merely materializes completed timing events and does not mutate or
expose runtime state.

## Dependency and scrub verdict

The verified transport token now binds the exact page-to-preflight-pass map and
the chunk bound. Every consumption snapshot depends on the actual preflight
pass covering its pages, including subsets beginning inside a larger preflight
chunk. Missing, duplicate, forged, or differently planned provenance fails
closed, and final authoritative validation checks the merged dependency DAG.

The production stager is the sole failure-scrub owner. The worker sets
`caller_guarantees_outer_scrub=True` for both write subsets, and the stager
catches either callback failure and performs exactly one scrub over every
currently staged destination allocation using the same authoritative recorder.
For Qwen's 28 layers, a scrub uses 57 CUDA records plus one CPU completion-fence
record: 58 total. The 256-pass protected reserve covers even the conservative
historical two-scrub bound of 116. Only `DIAGNOSTIC` work may enter that reserve.
Scrub, free, reset, or final-enqueue failure produces typed runtime teardown and
never admits the group.

## Verification

The accounting-focused suite passed `14/14`. The combined v11 gate suite passed
`93/93` in 1.77 seconds. Ruff lint, Ruff format checking, and `git diff --check`
passed for all reviewed files.

Coverage includes multi-drain exact-once behavior, cross-drain dependencies and
duplicate rejection, partial-drain retry, separate live/total bounds, reserve
nonborrowing and exhaustion, the 8,128- and 35,904-record cardinalities, exact
preflight provenance, single authoritative scrub ownership after either subset
fails, typed teardown failures, and threaded exclusion of every production
mutation surface.

The 35,904-pass test uses fake CUDA events; real-A100 behavior and all measured
performance values remain unvalidated until the designated coordinator runs
attempt d. This is not a claim of a GPU result.

Final decision: **PASS. The recorder-exhaustion and scrub-starvation blockers
are closed offline without weakening the production engine-step lock.**
