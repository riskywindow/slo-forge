# v11 source-allocation identity invariant

Status: `DEFINED_AND_TARGETED_REAL_A100_PROVEN`

Invariant version: `sloforge.continuum.vllm-v11-source-capture-commit/v1`

## Semantic identity

At the legal capture boundary, every logical source page is canonicalized by
`logical_page_id` and bound to:

- its block-table slot, logical token range, and valid-token extent;
- its physical vLLM block ID and allocator-issued allocation epoch;
- the exact logical branch-owner set derived from the canonical branch tables;
- the corresponding capture-time runtime request-owner set;
- `expected_refcount = len(expected runtime owner set)`;
- live native request-table membership and the absence of foreign referrers;
- device, runtime instance, runtime/adapter/model/tokenizer/dtype/policy identity;
- the committed branch token-history hash and computed-token boundary; and
- the held production engine-step binding ID and acquisition generation.

The pair `(block_id, allocation_epoch)` is indivisible. Reuse of a physical
block with a new epoch is a different allocation and fails validation.

For a shared page, the expected logical owners are the complete declared branch
group. Its native refcount must equal the exact runtime-owner set, every branch
must retain its reference, and no foreign runtime request may refer to the
block. For a private page, the declared owner set contains exactly one branch,
the runtime-owner set contains that branch's live request, and refcount is one.

The following are provenance, not semantic identity: the contents or order of
`take_new_block_ids()`, timestamps, inspection snapshot epochs, monotonic
diagnostic counters, allocator-wide issue counters, cache-history sequence
numbers, and record enumeration order. Reordering records passes when the
normalized logical-page mapping is unchanged. Reordering a live branch block
table changes the token-to-allocation mapping and fails.

## Snapshot time

`SourceCaptureCommit` is created from current runtime block tables and current
allocator epochs after branch quiescence. It is not a rollout-creation
snapshot. Legitimate lifecycle changes before this boundary are represented by
the committed current state. The commit is sealed in-process and has a
canonical semantic SHA-256 that excludes non-semantic diagnostic history.

The transaction validates the commit before the first export read, then rebuilds
the current plan and revalidates after the final export read. A source semantic
mutation between commit and read completion aborts the transaction; it is never
silently accepted as allocator churn.

## Allocator quiescence

`ALLOCATOR_QUIESCENT` means all of the following hold:

1. The last synchronous GPU1 engine step has returned.
2. Every source branch is `PAUSED` at its committed computed-token boundary.
3. Scheduler waiting and skipped-waiting queues are empty.
4. Scheduler requests, scheduler running requests, and native manager request
   tables contain exactly the eight live source runtime requests.
5. Asynchronous scheduling is disabled.
6. The production `EXPORT_CAPTURE` critical section is held by the validating
   thread and covers engine, scheduler, allocator, cache, and request-table
   mutation surfaces.
7. The current logical mapping, live `(block, epoch)` records, owner sets, and
   refcounts pass under that witness.

Source requests intentionally remain installed and live. A serving queue depth
of zero, an empty allocation-notification queue, or an empty scheduler request
table is not allocator quiescence.

## Allocation-notification history

vLLM 0.23.0 `take_new_block_ids()` is an epoch-free destructive notification
history. The retained engine can populate it during readiness and sanity work.
v11 now retires and persists that history before `SourceCaptureCommit`, labels
it `semantic_identity_claimed=false`, and establishes an empty baseline. Any
notification after the commit is independently fatal. The history queue is
never compared with the 1,152 live source allocations.

## CPU proof matrix

The local tests prove:

- unchanged block, epoch, mapping, owners, and refcounts pass;
- same block with a new epoch fails;
- an unauthorized physical remap or table-slot reorder fails;
- an expected shared/private lifecycle passes only with a newly derived current
  owner set and refcount;
- foreign ownership, freed state, reuse, post-commit mutation, stale creation
  manifests, non-paused branches, pending scheduler cleanup, async scheduling,
  and stale engine-step witnesses fail;
- diagnostic history changes and canonical record reordering pass; and
- pre-commit readiness history does not affect semantic identity, while any
  post-commit notification fails.

The implementation lives in
`python/sloforge/continuum/adapters/vllm_reclamation_v11_ownership.py`; the
integrated ordering is in
`experiments/branchfabric/gpu_reclamation_integrated_worker_v11.py`.

## Real-runtime proof

Targeted attempt `exp004-v11-targeted-identity-s41-b` exercised retained-engine
readiness, 12/15-rps sanity, live rollout creation, overload trigger at backlog
21, admission stop, allocator quiescence, and `SourceCaptureCommit`. It passed
1,152/1,152 logical mappings, block-plus-epoch identities, owner sets,
refcounts, device identities, and live-allocation checks. There were zero
post-commit notifications or semantic mutations. Export and source release did
not start. In-function and provider cleanup passed.
