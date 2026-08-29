# Experiment 004 v11 Offline Software Pipeline Optimization Review

Status: **OFFLINE REVIEW COMPLETE; NO GPU OR MODAL EXECUTION; NO IMPLEMENTATION CHANGED**

Scope: source transform, restore transform, bounded/pinned buffering, vLLM 0.23.0 native-page I/O, and zero-fill/security. Evidence is the frozen v10 code and the measured `exp004-v10-naive-s41-v7` artifacts. This review does not approve a GPU launch; local equivalence and adversarial tests remain mandatory first.

## Verdict

Implement v11 as an explicitly selected, versioned pipeline while leaving the body of the v10 naive transaction and `vllm_reclamation.py` unchanged. The smallest correct high-leverage change set is:

1. bounded direct native-page-to-canonical chunks;
2. two-slot asynchronous chunk scheduling into one canonical pinned checkpoint allocation;
3. bounded canonical-to-native scatter, with no complete canonical GPU restore tensor;
4. exact device-side destination comparison while each expected chunk is still resident;
5. one streamed publication integrity proof plus one pre-restore payload proof, instead of repeated `state.verify()` scans;
6. full-page overwrite for complete pages and zeroing only invalid tails of partial pages.

This removes most avoidable movement without requiring a custom kernel initially. A Triton/CUDA gather, scatter, or comparison kernel is justified only if bounded real-GPU profiling shows PyTorch's `index_select`/`index_copy_` intermediates keep amplification above the material-improvement gate or dominate the residual path.

## Frozen v10 evidence

The measured state is 1,056,964,608 logical bytes and 1,056,964,608 physical source bytes across 1,152 pages. Consequently every measured page is full; the baseline has zero padding bytes.

The implementation sites that create the dominant passes are:

- `experiments/branchfabric/gpu_reclamation_worker.py::_run_rollout_transaction`, source read/transform/D2H at lines 1661-1794, repeated transport verification at lines 1795-1889 and 2026-2064, restore conversion/write at lines 2123-2234, and full destination recapture at lines 2236-2478.
- `python/sloforge/continuum/adapters/vllm_reclamation.py::read_native_pages_to_gpu_intermediate`, which performs per-layer `index_select`, `movedim`, and `contiguous` over the complete physical state.
- `transform_native_read_to_canonical_device`, which loops over pages, makes valid-token contiguous tensors, stacks layers, and concatenates every page.
- `CanonicalKvTransportState.verify` and `build_host_transport_state`, which each materialize a complete pageable `bytes` object and compute aggregate and per-page SHA-256 digests.
- `convert_canonical_device_to_native_pages`, which creates a zero-filled full physical page for every logical page, overlays valid tokens, permutes/contiguates, and stacks.
- `write_native_pages_to_destination`, which finally performs the required native pool write.

The measured host allocation peak includes 1,056,964,608 pinned checkpoint bytes and 1,908,408,320 simultaneously live pageable validation bytes. Total pageable allocation churn was 6,341,787,648 bytes. The validation path separately moved 1,056,964,608 bytes D2H; v10 external movement is therefore D2H capture + H2D restore + D2H validation = 3x.

## Pass families to remove

These labels must disappear or be replaced by bounded chunk records in the v10-v11 pass diff:

| Family | Frozen v10 labels | v11 treatment |
|---|---|---|
| Full source materialization | `capture-source-native-read`, `capture-native-axis-contiguous`, `capture-unpage-valid-tokens`, `capture-stack-layers`, `capture-concatenate-pages` | bounded page gather directly into the final canonical chunk shape; no full native or full GPU-canonical intermediate |
| Repeated host verification | `capture-integrity-manifest`, `capture-integrity-hash-reads`, `transport-publish-validation`, `transport-publish-hash-reads`, `restore-transport-validation`, `restore-transport-hash-reads`, `restore-import-validation`, `restore-import-hash-reads` | one ordered streamed publication proof; one pre-H2D payload proof; metadata/root verification is O(manifest), not O(payload) |
| Full restore materialization | `restore-zero-native-pages`, `restore-overlay-valid-tokens`, `restore-native-axis-contiguous`, `restore-stack-native-pages` | direct per-layer `index_copy_`/narrowed writes from bounded canonical chunks; zero only invalid tails |
| Full destination recapture | `validation-destination-native-read`, `validation-native-axis-contiguous`, `validation-unpage-valid-tokens`, `validation-stack-layers`, `validation-concatenate-pages`, `validation-d2h`, `validation-expected-page-concatenation`, `validation-host-tensor-compare`, `validation-recapture-host-lifetime` | exact device comparison against the still-live expected chunk; a scalar mismatch result crosses to the host; full recapture is diagnostic-only after failure |

The required endpoint work remains: read every logical source byte, publish every logical byte to checkpoint storage, transfer every logical byte D2H and H2D, write every logical destination byte, verify the retained payload before import, and independently read/verify every logical destination byte before visibility.

## Exact implementation path

### New versioned adapter module

Add `python/sloforge/continuum/adapters/vllm_reclamation_v11.py`. Reuse the frozen v1 manifest models, capture plan, identity validation, native geometry validation, and allocation semantics from `vllm_reclamation.py`; do not change the v10 converter functions.

The new module should define these bounded interfaces:

- `V11ChunkPlan`: immutable ordered page descriptors, exact payload offsets, one `valid_tokens` extent per dense run, maximum chunk bytes, and explicit seed-independent deterministic ordering.
- `plan_transport_chunks(page_order, geometry, *, maximum_chunk_bytes)`: split only at page boundaries; split again when `valid_tokens` changes so each dense chunk has shape `[page, layer, valid_token, kv, head, dim]`; prove complete, unique, gap-free byte coverage.
- `V11TransportIntegrity`: domain-separated SHA-256 state commitment over the canonical v1 manifest bytes, including model/tokenizer/policy identity, token histories, branch ancestry/ownership, page offsets/extents, per-page digests, and whole-payload digest.
- `V11CanonicalTransportState`: the unchanged canonical manifest, one contiguous pinned CPU `uint8` checkpoint tensor, the commitment, and immutable chunk descriptors. The pinned payload is checkpoint state, not a second temporary copy.
- `capture_native_to_transport_v11(...)`: two bounded GPU chunk slots, a compute stream, a copy stream, explicit CUDA events, and D2H into disjoint views of the final pinned checkpoint. Hash a host view only after its D2H-complete event. Never call `.numpy().tobytes()` on the complete payload.
- `restore_transport_to_native_v11(...)`: verify retained host payload once, use two bounded GPU chunk slots, overlap H2D of chunk N+1 with scatter/validation of N, and return only after every write and comparison event has completed.
- `write_and_validate_native_subset_v11(...)`: accept the stager's disjoint destination map, coalesce adjacent canonical payload ranges, scatter each chunk, exact-compare native logical bytes against the same GPU chunk, and validate zero tails. On mismatch, fail before visibility and retain chunk/page diagnostics.
- `Vllm0230StreamingRestoreStager`: preserve `Vllm0230RestoreStager` allocation, fresh-incarnation, cleanup, and admission rules, but use one fused `write_and_validate_pages(mapping)` callback. Separate write and validation callbacks cannot remain bounded because validation would otherwise require retaining or retransferring the complete first-branch subset.

The first implementation should use PyTorch operations. For each layer in a chunk, move the native block dimension to the front as a view, permute the remaining semantic axes to `(token, kv, head, dim)`, `index_select` only the bounded source page IDs, and copy into `chunk[:, layer]`. Release each per-layer gather before proceeding to the next layer. Do not build a list containing all layer-sized gathers.

For restore, use `index_copy_` directly into each native layer from `chunk[:, layer]` after the inverse semantic permutation. A complete page receives no prior zero-fill because every byte is overwritten. For a partial page, copy only `[0:valid_tokens]` and zero `[valid_tokens:block_size_tokens]` in the actual destination native tensor before validation.

### Transaction isolation

Keep `experiments/branchfabric/gpu_reclamation_worker.py::_run_rollout_transaction` byte-for-byte behaviorally intact for `PRESERVE_NAIVE`. Add a new `experiments/branchfabric/gpu_reclamation_v11_pipeline.py` that owns v11 capture, restore, integrity, and v11 StatePassRecord emission. Dispatch to it only under an explicit `state_pipeline: "optimized-preserve-v11"` configuration value before entering the v10 transaction path. An absent or unknown value must fail closed; it must never silently select v11.

Prefer a separate `Experiment004V11Config`/entry point in `experiments/branchfabric/modal_gpu_reclamation_v11.py` over widening the frozen v10 `Experiment004PilotConfig.mode`, whose current only allowed value is `PRESERVE_NAIVE`. The controller may share process-launch and serving code, but the emitted config and result schemas need distinct v11 versions.

### Buffer choice

The measured native page is exactly 917,504 bytes. Default to 32 pages per chunk: 29,360,128 bytes (28 MiB), 36 chunks for the baseline state, two GPU slots totaling 56 MiB, and at most two in-flight pinned checkpoint views. Search only 16, 32, and 64 pages during bounded micro-validation; keep the default unless another size materially improves overlap without increasing interference.

The final 1,056,964,608-byte host checkpoint is irreducible while GPU1 is repurposed. Retain it as a single pinned allocation and stream into disjoint views. This avoids the extra pinned-to-pageable and pageable-to-pinned copies that a reusable two-buffer plus pageable checkpoint design would add. Report separately:

- checkpoint-resident host bytes;
- pipeline-temporary host bytes;
- total live host bytes.

This makes the comparison with v10's approximately 2.965 GB simultaneous pinned-plus-pageable peak honest. The expected v11 host peak is the 1.057 GB checkpoint plus bounded metadata/small diagnostics, not a claim that the checkpoint itself vanished.

## Integrity and visibility contract

The v11 path must preserve all current checks and replace only their physical implementation:

- Compute ordered per-page SHA-256 and whole-payload SHA-256 while each completed D2H chunk is already CPU-visible. A fused/native multi-digest routine may later share a memory read, but two ordinary `hashlib` calls must be accounted as two hash reads.
- Compute a domain-separated SHA-256 commitment over the completed canonical manifest. Because the manifest contains payload/page digests plus identity, ancestry, token history, ownership, ordering, offsets, and logical extents, wrong-branch or metadata substitution changes the commitment.
- Before restore allocation/H2D, validate the small manifest/commitment and make exactly one ordered pass over retained payload bytes to recompute the whole-payload SHA-256. Do not invoke the v10 `state.verify()` again inside the stager.
- Exact device comparison is stronger than a probabilistic destination hash. It must compare the logical canonical bit pattern for every layer/page/valid token and independently prove that every invalid tail byte is zero. Only one mismatch scalar and diagnostic metadata return to the host.
- On any mismatch, do not enqueue requests or publish additional prefix-cache entries. Preserve the failing chunk/page artifact, use the v10 full recapture only as a diagnostic, and fail the optimized run.
- Source blocks may be destroyed only after every capture copy event and CPU integrity update has completed. Destination pages may be cached/enqueued only after every write, tail-zero, and exact-comparison event has completed. These fences are correctness operations and must be recorded.

## Zero-fill and tenant-security ruling

Removing the measured full-state zero-fill is safe only under a byte-coverage proof, not merely because vLLM currently masks unused positions.

For each newly allocated destination block and every layer:

1. all K and V bytes for tokens `[0, valid_tokens)` are overwritten from the authenticated checkpoint;
2. all K and V bytes for tokens `[valid_tokens, block_size_tokens)` are explicitly zeroed;
3. the union of written/zeroed ranges equals the full native physical page;
4. the union of logical destinations equals the exact set drained from vLLM's new-block/zero queue;
5. no request becomes runnable before this coverage and validation complete.

For the measured baseline, every page is complete, so rule 1 covers the entire state and zero-fill bytes fall from 1,056,964,608 to zero. Randomized partial-page tests must begin with tenant-distinct nonzero sentinels and prove the tail is zero after restore. Preserve the current failure when `needs_kv_cache_zeroing` is true and unrelated blocks were already pending; do not drain or bless another request's zero queue.

## Local test plan (mandatory before GPU allocation)

Add `tests/python/test_vllm_reclamation_v11.py` and use explicit deterministic seeds, at minimum 41, 73, and 113.

1. Generate CPU BF16 native pools with randomized physical page order, 1/2/4 layers, different legal block dimensions, multiple block counts, K and V, full and partial final pages, and shared/private fanout layouts.
2. Assert byte-for-byte equality of the v10 `native_pages_to_host_transport` payload and manifest against v11 export for identical canonical inputs.
3. Restore v10 and v11 into fresh pools initialized with different sentinels; assert logical equality, complete-page equality, zero partial tails, and untouched non-destination blocks.
4. Assert chunk coverage is gap-free and bijective; reject missing, duplicate, overlapping, out-of-order, oversized, empty, and mixed-extent malformed chunk plans.
5. Flip one payload bit, corrupt one destination element after write, reorder pages/chunks, remove a chunk, duplicate a chunk, alter one owner/branch, change token history, and change policy/model identity. Every case must fail before admission.
6. Exercise random destination block IDs and non-leading block dimensions. Assert writes mutate the actual backing pool rather than an advanced-index temporary.
7. Mock CUDA events/streams to prove at most two slots are in flight, a slot cannot be reused before copy completion, source release waits for all D2H completion, and admission waits for all restore validation completion.
8. Reuse and extend current stager tests: exact zero-queue coverage, unrelated pending-zero failure, shared-root maps to one destination, fresh runtime incarnations, allocation failure cleanup, validation failure cleanup, and partial enqueue failure cleanup.
9. Add movement-ledger tests with exact per-chunk source/destination bytes and allocation lifetimes. No byte savings may be inferred from logical sizes without emitted v11 StatePassRecords.
10. Run the relevant adapter, capture-plan, worker-movement, and Experiment 004 harness tests after each major local change, then `make check` only after the full local implementation is stable.

## Risks and stop conditions

- **High: noncontiguous PyTorch output may silently allocate another chunk.** Profile allocation/HBM and record the `index_select` output and canonical chunk separately. It is acceptable while bounded; it is not acceptable to materialize the full state.
- **High: preserving separate write/validate callbacks forces full-subset retention or retransfers.** Use the fused streaming callback.
- **High: asynchronous capture can race source destruction.** Require explicit completion events before release.
- **High: asynchronous restore can expose partial state through the prefix cache.** Preserve the exclusive engine-step gate and validate before `cache_blocks`/enqueue.
- **High: omitting tail zeroing leaks stale tenant state.** Full-page overwrite may skip zeroing; partial pages may not.
- **Medium: canonical page order interleaves private branches.** The restore planner must coalesce selected adjacent ranges but support noncontiguous first-branch subsets without copying them into a full host temporary.
- **Medium: Python page-by-page/layer-by-layer loops can erase latency gains.** Vectorize full-page runs by layer and chunk; use per-page fallback only for small partial runs.
- **Medium: `torch.cuda.empty_cache()` may synchronize.** Release bounded scratch once after capture, measure the fence, and ensure scratch caching cannot reduce useful serving capacity.
- **Stop before full transaction** if micro-validation does not materially lower physical amplification, if any 8/8/token/integrity gate fails, if host peak is not bounded as designed, or if production-serving interference increases materially.

## Expected review outcome

This design is capable of removing the majority of v10's avoidable movement while retaining exact logical equality, SHA-256 transport integrity, fresh native allocation, tenant-safe tails, and atomic group visibility. The principal residual expected in the initial PyTorch path is bounded `index_select` gather traffic and exact device comparison traffic. Only measured residuals may justify a Triton/CUDA follow-up.
