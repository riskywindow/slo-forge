# v11 Source Allocation Root Cause

## Outcome

The attempt-C failure is classified as **GATE_BUG**.

The source identity gate compared two different domains:

- the current authenticated source capture plan; and
- vLLM's retained, unscoped `take_new_block_ids()` allocation-event queue.

GPU1 readiness legitimately allocated and freed KV blocks before rollout creation. Readiness and later request-state resets cleared live requests and prefix-cache state, but they never drained or phase-separated the allocation-event queue. The queue therefore described allocator history, not the identity of the 1,152 allocations live at capture.

This is not classified as `BENIGN_ALLOCATOR_CHURN` because no current source representation change was shown. Benign readiness churn merely exposed the faulty assertion. It is not `REAL_LIFECYCLE_BUG`: the exact live semantic source proof had already passed under the production export gate.

## Frozen evidence and limitation

The failed execution is `exp004-v11-integrated-s41-c`, execution commit `af8f5d474421637c257ddac8f9596870cc11ede5`, with terminal analysis commit `6b36d097ac739097c5ad63f7179439903e0118b5`.

The immutable inventory is [failed-run-manifest.json](../../artifacts/branchfabric/gpu-validation/experiment-004/v11-final/allocation-forensics/failed-run-manifest.json). It hashes all 57 non-compiler attempt-C files, the remote manifest, and terminal scientific-validity artifact. Existing failed evidence was not modified.

Attempt C did not retain the exact expected or observed allocator sets. `V11RuntimeTeardownRequired` held `affected_block_ids` and `teardown_evidence` in memory, but the integrated worker failure writer serialized only exception type, message, traceback, and time. Consequently the following cannot be recovered:

- failed-run logical-page to physical-block mappings;
- allocator epochs for the 1,152 source blocks;
- raw zero-queue values, count, or commitment;
- exact missing, unexpected, or duplicate IDs;
- per-event allocator timestamps, engine steps, threads, and runtime request IDs; and
- the pre-release ownership snapshot, capture manifest, and page hashes.

The [lifecycle reconstruction](../../artifacts/branchfabric/gpu-validation/experiment-004/v11-final/allocation-lifecycle/source-allocation-lifecycles.json) therefore contains exactly 1,152 logical rows while explicitly setting unavailable physical fields to `null`. It does not substitute identities from the successful micro run.

## First causal divergence

The earliest responsible runtime event predates `ROLLOUT_CREATED`. The first GPU1 readiness request, `readiness-gpu1-00000`, entered the one-request readiness batch at monotonic time `170168081065`. Its first token was observed at `170771993885`. Its first `KVCacheManager.allocate_slots` call occurred inside that interval and appended allocation IDs to the retained manager queue.

The exact allocation timestamp is unrecoverable because the allocator-issued record was process-local. The bounded event and evidence limitation are recorded in [first-allocation-divergence.json](../../artifacts/branchfabric/gpu-validation/experiment-004/v11-final/allocation-forensics/first-allocation-divergence.json).

GPU1's later 12-rps and 15-rps sanity shards admitted no requests. Their reset artifacts again prove only live request and prefix-cache reset. They do not prove allocation-event-queue reset.

## Why the old invariant failed

The old gate required:

```text
observed = tuple(manager.take_new_block_ids())
len(observed) == 1,152
len(set(observed)) == 1,152
set(observed) == set(current_capture_plan_block_ids)
```

That assertion was too strong in the wrong dimension and too weak in the correctness dimensions:

- It rejected valid historical allocation events unrelated to the current source lifetime.
- It compared block numbers without allocator epochs.
- It did not itself prove logical mapping, ownership, refcount, liveness, device, runtime, or model identity.

The gate also destructively drained its diagnostic evidence before throwing.

## Proof that current source semantics had passed

Inside one production `EXPORT_CAPTURE` critical section, attempt C completed these steps before calling the faulty gate:

1. Read all eight live runtime branch tables.
2. Build the canonical page plan with allocator-issued epochs.
3. Validate the exact topology: 1,024 shared pages, 128 private pages, and 1,056,964,608 logical bytes.
4. Require the live layout's block set to equal the capture-plan block set.
5. Require current allocator epochs for all blocks.
6. Require native request-table owners and global referrers to equal the expected logical owners.
7. Require native refcount to equal owner-set cardinality.
8. Require the branch group to retain the declared shared root.
9. Require allocator-visible assigned HBM bytes to equal the source extent.
10. Capture and authenticate the source state.

Only then did it drain `take_new_block_ids()`. Source release was the next statement and never ran. Thus the failed equality does not show a stale, freed, reused, foreign-owned, or remapped source allocation.

## Correct source identity invariant

A `SourceCaptureCommit` is valid only when, for every logical page:

```text
logical_page
-> physical_block_id
-> allocator_epoch
-> token_range
-> branch_group
-> exact expected_owner_set
-> expected_refcount
-> device
-> runtime_instance
-> model_and_policy_identity
```

At commit and immediately before export read:

- the logical page must still resolve to the same block and epoch;
- the allocation must be live and not tombstoned;
- the native request tables must contain exactly the expected source owners;
- the native refcount must equal the expected owner-set cardinality;
- shared pages must be owned by all eight expected branches and no foreign request;
- private pages must be exclusively owned by their expected branch;
- allocator-visible assigned bytes must agree with the committed source extent; and
- the production engine-step binding and allocator mutation generation must remain unchanged.

A reused physical block fails even when its integer block ID is unchanged:

```text
(block_id = X, epoch = A) != (block_id = X, epoch = B)
```

Historical counters, timestamps, diagnostic queue order, and unrelated allocation generations are excluded from identity.

## Allocator quiescence

`SERVING_QUEUE_ZERO` is not allocator quiescence.

The legal commit boundary requires:

1. rollout admission stopped;
2. no branch append or engine step;
3. scheduler running/waiting state stable with exactly the retained source requests;
4. deferred request teardown, cache release, refcount, and block-table work affecting source pages complete;
5. the production engine-step mutation lock held;
6. a stable allocator mutation generation before and after commit; and
7. no mutation affecting committed source state through export read.

No arbitrary sleep is valid evidence.

## Implemented smallest correct fix

The movement pipeline remains unchanged. The correction now:

- create and persist `SourceCaptureCommit` from the already-proven ownership/capture snapshot;
- validate semantic allocation identity against that commit;
- persist allocator exception context before teardown;
- retires and persists pre-commit `take_new_block_ids()` history without making a semantic identity claim;
- treat subsequent queue drains only as allocation provenance and zero-queue hygiene; and
- fails if an allocation notification or semantic allocation mutation appears after commit.

The integrated failure serializer now preserves `affected_block_ids` and
`teardown_evidence`. A bounded allocator lifecycle journal records allocation,
epoch issue, table binding, owner/reference changes, request finish, free, and
cache release with timestamps, process/thread identity, and caller labels. At
capture it binds exact `(block_id, allocation_epoch)` lifetimes back to logical
pages. Prefix-cache resets now notify the epoch source, so calibration lifetime
tombstones are retained correctly.

A separate targeted execution mode replays readiness, the 12/15-rps sanity
guards, rollout creation, early overload, quiescence, `SourceCaptureCommit`, and
the 1,152-page semantic gate, then stops before export or source release.

## Verification status

Offline forensic artifact validation can prove record cardinality, logical topology, checksums, timestamp ordering, classification consistency, and absence of fabricated physical fields. It cannot retroactively prove the missing attempt-C physical histories.

The final focused attempt-D authorization suite passes 181 tests. The complete
repository check passes: formatting, Ruff, mypy, 2,047 Python tests with eight
environment skips, Cargo formatting/clippy/tests, and UI
typecheck/lint/tests/build.

Targeted real-runtime attempt `exp004-v11-targeted-identity-s41-b` passed the
exact lifecycle at backlog 21. All 1,152 allocations passed logical mapping,
block plus epoch, owners, refcounts, device, and live status; the committed
semantic SHA-256 was
`28168c11daf69cdc89e1ddf369a8c828fc274315d37c234c480ba63c5ab67086`;
and zero post-commit allocator events occurred. This confirms `GATE_BUG` and
rejects a hidden source lifecycle race for the reproduced transition.

The one subsequent full integrated retry failed during Modal image build before
function entry, so it neither confirms nor contradicts the allocator proof and
produces no integrated measurement. `HARDWARE_GATE_NOT_REACHED` remains
mandatory.
