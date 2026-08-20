# Agent 01 allocator epoch audit

Verdict: **PASS**

Scope: offline review only. I did not invoke Modal, a GPU, or the network.

## Result

Gate A meets the production integration requirements. Epochs are emitted from the live `KVCacheManager.allocate_slots` interception using the allocator result and installed runtime block-table delta. Continuum import cannot supply an epoch: it can only request a sealed proof from the manager-attached source, validate that proof against current runtime ownership, and consume it once.

The lifecycle is fail closed. Native free and prefix-cache reset tombstone the exact `(block_id, allocation_epoch)` generation; reuse of a physical block advances the manager-wide monotonic epoch; a stale, freed, consumed, forged, unowned, reordered, or destination-mismatched proof is rejected before admission.

The destination commitment binds the checkpoint manifest commitment, payload digest, destination target, ordered logical-page-to-block mapping, and ordered allocator-issued epochs. Gate B separately binds the same allocator records into source ownership evidence.

## Requirement matrix

| Requirement | Result | Decisive evidence |
| --- | --- | --- |
| Actual runtime allocation path | PASS | Issuance occurs only while unwinding the installed live manager's `allocate_slots`; allocator return plus native table delta are authoritative. |
| Not supplied by Continuum import | PASS | Import has no epoch parameter or issuance API; proof source and seal are manager-bound. |
| Lifetime unique and non-reusable | PASS | Positive monotonic epochs, exact live-pair set, free/reset tombstones, and higher generation on reuse. |
| Ownership evidence binding | PASS | Gate B obtains allocator proof for the exact source IDs and compares every captured epoch. |
| Destination commitment binding | PASS | Commitment hashes ordered block epochs with manifest, payload, target, and mapping. |
| Current before admission | PASS | Final `require_current` and `consume_once` execute immediately before enqueue under the production admission critical section. |
| Stale/freed/reused/replayed proof rejection | PASS | Runtime ownership refresh, live-set membership, source/seal identity, generation equality, and consumed-set checks all fail closed. |

## Tests

The focused command passed 7/7:

```text
PYTHONPATH=python python -m pytest -q \
  tests/python/test_vllm_allocator_epochs.py \
  tests/python/test_vllm_live_adapter.py::test_external_computed_pages_receive_allocation_epochs_when_return_is_empty \
  tests/python/test_vllm_reclamation_capture_plan.py::test_capture_plan_requires_all_live_allocation_epochs \
  tests/python/test_vllm_reclamation_v11.py::test_streaming_stager_requires_typed_fused_evidence_and_cleans_up

7 passed in 0.88s
```

An additional read-only negative probe confirmed rejection of an ordered destination-block mismatch, a proof marked non-runtime-issued, and a second consumption of the same allocation lifetime. `git diff --check` passed for the audited files.

The required cases are covered: valid allocation; stale and freed allocation; reused physical block ID with a newer epoch; single-use/double-import replay at the exact admission primitive; and mismatched destination identity, page coverage, mapping, or epoch commitment.

## Audit boundaries

Offline fixtures execute the production hook and stager code without a GPU. The real A100 micro-validation must still assert that all 1,152 native vLLM blocks carry allocator proof, all fresh destination generations differ from the released source lifetimes, and admission consumes those generations exactly once. This is a required runtime confirmation, not an unresolved Gate A defect.

The annotated v10 tag still dereferences to frozen execution commit `c5fa2f169e8a343ef7a86ebc814b3a23a7de6b39`.

Machine-readable evidence and audited file hashes are in `agent01-allocator-epoch-audit.json`.
