# Experiment 004 v11 adapter integration/correctness review

## Verdict

**REJECT FOR GPU MICRO-VALIDATION pending the high-severity findings below.**

The bounded CPU implementation is substantially improved and its current local tests pass. The remaining blockers are integration contracts: trustworthy/replay-safe destination evidence, canonical measured state-pass records, fail-closed cleanup, atomic admission, and exact CUDA-device binding. No GPU, Modal, or remote compute was used for this review.

## Reviewed snapshot

- `python/sloforge/continuum/adapters/vllm_reclamation_v11.py`
  - SHA-256: `20694a115a718df64cce9b5b616181204272f1fb619be0c22d57fd0bed58097d`
- `tests/python/test_vllm_reclamation_v11.py`
  - SHA-256: `f471b7bd3bb4c282a505084b3d89219d4fadc808d0cec65176b7deed2d2f43be`
- frozen v10 adapter `python/sloforge/continuum/adapters/vllm_reclamation.py`
  - SHA-256: `fa8f59b6e6755d0eb5c6c008d26df48b5fcbb4ad85f2af3b84233808c0ba953f`

Offline tests:

```text
PYTHONPATH=python pytest -q \
  tests/python/test_vllm_reclamation_v11.py \
  tests/python/test_vllm_reclamation_adapter.py \
  tests/python/test_vllm_reclamation_capture_plan.py

31 passed in 0.76s
```

The v11 file is a separate implementation and the v10 adapter remained byte-for-byte unchanged during this review. The 18 targeted v10 adapter/capture-plan tests pass.

## High-severity findings

### H1. `V11StatePassRecord` is not the canonical measured ledger and cannot support the v11 movement gate

`V11StatePassRecord` at lines 84-99 is a second, permissive dataclass rather than `sloforge.helix.characterization.gpu_reclamation_accounting.StatePassRecord`. The records emitted by `capture_native_to_transport_v11` at lines 611-660 and `restore_transport_to_native_v11` at lines 1077-1137 are aggregate estimates, not records for each real chunk/pass. They omit:

- `state_segment`, `branch_group`, logical offset/coverage, and per-chunk identity;
- measured start/end timestamps, CPU time, CUDA-event time, device, and synchronization;
- transfer direction, unique byte-event IDs, temporary allocation IDs/domains, and dependency/fused-chain identity;
- a validated required/avoidable flag and the canonical memory-domain enums.

The restore validation record also uses the synthetic combined source `destination_gpu_native_paged+bounded_gpu_transport_chunk`, which is not one canonical `MemoryDomain`. The accounting library therefore cannot validate or deduplicate these rows. Temporary allocation accounting is peak metadata attached to aggregate operations rather than allocation events.

The source endpoint byte formula is now directionally correct: bounded gather records physical extent `P` read plus `P` write, and repack records logical `L` read plus `L` write. That improvement does not make the ledger measured or canonical.

**Required change:** instrument each actual chunk operation with the existing canonical `StatePassRecord`, including CUDA events/fences and CPU wall/process time. Use separate records (sharing explicit event IDs when appropriate) when an operation has two physical sources. Feed those records directly through `build_state_movement_report`; add a test that the report accepts them and that exact formulas close for full and partial pages. Do not derive the headline amplification from the current adapter dataclass.

### H2. Sealed destination and scrub evidence can be replayed for a different state or block mapping

`V11SubsetValidationEvidence` (lines 147-157) and `V11ScrubEvidence` (lines 160-167) contain logical page IDs and booleans but no commitment to:

- the manifest/verified transport;
- the ordered `logical_page_id -> destination_block_id` mapping;
- destination tensor/device identity or allocation epoch;
- the exact expected logical, tail, and physical byte totals.

`require_subset_evidence` at lines 1175-1188 validates only set equality, booleans, and the process-local proof seal. The cleanup check at lines 1271-1280 has the same limitation. A genuine sealed result from an earlier restore attempt with the same logical page IDs can therefore be returned after fresh blocks are allocated; it will admit or free blocks that were not the blocks proven by that evidence. This is a realistic retry/stale-callback failure, not an adversarial Python-module-access concern.

**Required change:** add a domain-separated SHA-256 commitment over manifest commitment, verified payload digest, ordered page IDs, destination block IDs, destination device/target identity, geometry, and allocation epoch where available. Bind both validation and scrub evidence to it. Recompute the expected mapping commitment in the stager and check all byte totals, ordered unique page coverage, and the commitment. Add stale-evidence, changed-block-ID, changed-state, duplicate-page, and reordered-page rejection tests.

### H3. Cleanup is not fail-closed when free/reset operations fail

The `import_group_v11` exception path at lines 1269-1305 correctly avoids freeing pages when scrub proof fails. However, after a successful scrub it suppresses every `manager.free` exception (lines 1292-1296), removes the scheduler request anyway, and ignores the return value or exception semantics of `reset_prefix_cache` (lines 1298-1300). This can leave KV ownership in the manager while deleting the scheduler's ownership record, and the caller receives the original restore error rather than a mandatory teardown/quarantine signal.

On scrub failure the function raises a generic `V11IntegrityError` containing teardown text, but the type does not force the controller to destroy the runtime. The adapter itself leaves manager tables allocated and removes scheduler records. Continuing that engine is unsafe unless the caller has an explicit mandatory teardown path.

**Required change:** collect cleanup failures, require `reset_prefix_cache() is True` when present, and raise a dedicated `V11RuntimeTeardownRequired` carrying the affected block IDs whenever scrub, free, queue removal, or cache reset cannot be proven. The worker/controller must catch that type and terminate the destination engine before any more requests execute. Add injected scrub/free/reset/queue-removal failure tests and assert either complete clean state or mandatory engine teardown with blocks never returned to reuse.

### H4. Atomic admission depends on an unexpressed external engine-step gate

The internal ordering is good: all destination subsets are written and validated before `num_computed_tokens` is set, and before the first final enqueue (lines 1261-1267). But final admission is still a loop of individual `_enqueue_waiting_request` calls. Unlike the v10 method's detailed docstring, `import_group_v11` neither accepts a gate witness nor asserts that scheduling is paused. If scheduler progress can interleave, one restored branch becomes visible before the remainder of the group.

**Required change:** make the engine-step/admission gate an explicit precondition in the callable integration contract (preferably a controller-owned context/capability), assert it in the adapter, and release it only after the full enqueue loop. Add an interleaving scheduler test showing that no branch can be selected between group enqueues. The production worker must call v11 only through this gated path.

### H5. CUDA configuration binds only a device type, not the selected GPU

`V11PipelineConfig.expected_device_type` distinguishes only `cpu` from `cuda`. Capture and restore compare `{tensor.device.type}` to that value, so tensors on the wrong CUDA index satisfy configuration and a mixed-device tuple is not rejected by preflight. Streams are then created from `tensors[0].device`, while later layers may reside elsewhere. This violates the repository rule against silently switching devices/engines and weakens topology provenance.

**Required change:** bind config to an exact `torch.device`/CUDA index and expected GPU UUID (or inject a validated device-identity evidence object). Preflight must require every layer on the same exact device before allocating host state or mutating destination pages. Record that identity in every measured pass. Add wrong-index and mixed-device tests; the real-A100 micro-run must also compare the configured UUID with NVML/runtime evidence.

## Medium-severity findings

### M1. Zero-queue checks destructively consume unrelated evidence

`pending_before_restore = tuple(self.manager.take_new_block_ids())` occurs before the guarded cleanup block at lines 1190-1192. A nonempty result is consumed and then the method raises without returning, zeroing, or otherwise preserving those IDs. The later unexpected-allocation check also consumes IDs before raising. If these are unrelated pages awaiting zeroing/accounting, their obligation is lost.

Use a non-destructive emptiness assertion where possible. If the vLLM hook is necessarily destructive, enter the guarded transaction first and return the IDs to the zeroing controller or fail with a teardown-required/quarantine result that explicitly retains them.

### M2. Returned import evidence loses the fused proof and cleanup/admission provenance

`import_group_v11` returns the v10 `StagedGroupImport`, which contains allocations and drained IDs but not the two sealed subset proofs, mapping commitments, fences, admission timestamps, or cleanup state. The controller must currently capture callback side effects out of band, making the artifact chain easy to break.

Return a v11-specific staged-import result containing ordered subset evidence, exact mapping commitment, verified transport commitment, allocation/admission intervals, gate identity, and final cleanup disposition. This result should be the source of the correctness artifact rather than reconstructed dictionaries.

### M3. Memory stats do not separate required pinned residency from temporary pinned buffers

The CUDA capture owns the entire final checkpoint as a pinned tensor while restore also allocates bounded pinned snapshots. `V11PipelineStats` reports checkpoint resident host bytes and peak host temporary bytes, but not pinned versus pageable residency, process RSS, vLLM pool, or CUDA/HBM temporary peaks. A later report could incorrectly describe `peak_host_temporary_bytes` as total pinned pressure.

Add explicit `checkpoint_pinned_host_bytes`, `peak_pinned_staging_bytes`, `peak_pageable_host_bytes`, and measured RSS/HBM fields (the latter from runtime instrumentation). Keep the retained checkpoint separate from temporary amplification in plots and comparisons.

### M4. The local suite does not yet exercise the complete integration surface

The current tests provide valuable bitwise v10/v11 export equality, logical restore equality, partial-page zeroing, raw-bfloat16 comparison, tensor and NumPy-alias corruption, block-dimension preflight, and a real helper-produced fused proof. Remaining required local cases include:

- eight branches and the exact 1,024-shared/128-private page shape (with small geometry where needed);
- stale/replayed validation and scrub evidence against changed destination mappings;
- cleanup failures for scrub, free, prefix-cache reset, and final enqueue;
- preexisting/late zero-queue IDs;
- wrong branch ancestry/history, missing/duplicate/reordered manifest chunks, and wrong destination allocation epoch;
- canonical pass-ledger closure and deduplication;
- mocked CUDA stream/event ordering, exact-device rejection, and bounded-buffer lifetime.

These should be added before the bounded real-A100 micro-validation; no broad tuning sweep is needed.

### M5. Semantic capture-plan validation happens after source movement begins

`capture_native_to_transport_v11` validates native geometry and simple page-order uniqueness before gathering, but the cross-check between page ownership, branch tables, token ranges, ancestry, and logical offsets is obtained only when `CanonicalKvTransportManifest` is constructed after all chunks were copied and hashed. Invalid semantic input therefore performs the complete source-state movement before failing.

Construct/validate the destination-independent manifest skeleton or reuse `CanonicalCapturePlan` before allocating/copying the payload. This is primarily fail-fast behavior, but it also prevents an invalid plan from consuming the reclamation critical path.

## Positive findings retained

- v10 isolation is real in this snapshot: separate module, unchanged v10 hash, and passing v10 tests.
- Source and destination tensor geometry are preflighted before destination mutation.
- Restore authenticates an owned bounded host snapshot immediately before consumption, closing both ordinary tensor-mutation and NumPy-alias mutation holes.
- Destination validation compares raw bfloat16 bits, not floating-point equality, and failure scrubs all mapped native pages.
- Full pages are overwritten without zero-fill; partial unreadable tails are zeroed and verified before evidence is issued.
- CUDA mode explicitly requires pinned transport storage, async copies, and at least two bounded slots.
- Internal admission ordering validates all state before setting computed-token visibility or enqueueing requests.

## Approval condition

Do not allocate an A100 for v11 micro-validation until H1-H5 are either fixed or, for H4, proven by an explicit production integration invariant plus a targeted interleaving test. Re-run the 31-test offline set and the new adversarial cases, preserve the v10 adapter hash, then request movement-accounting/Continuum/CUDA review on the canonical evidence emitted by that exact snapshot.
