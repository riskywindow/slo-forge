# Agent 05 v11 integrity audit

Result: **PASS** for the offline pre-GPU integrity gate.

This review does not claim a real-A100 result. It establishes that the isolated v11 adapter and micro worker have fail-closed integrity and correctness mechanics sufficient to proceed to the separately authorized micro-validation.

## Findings

| Requirement | Result | Evidence |
|---|---:|---|
| Whole-state SHA-256 | PASS | Capture hashes deterministic canonical chunk order; streaming restore preflight recomputes and compares the complete payload digest. |
| Per-page SHA-256 | PASS | Capture, preflight, and consumption-time checks cover each immutable page range with exact chunk closure. |
| Manifest commitment | PASS | Domain-separated SHA-256 of canonical manifest bytes is bound into verified transport, destination evidence, and scrub evidence. |
| Consumption-time authentication | PASS | Each bounded restore chunk is copied to an owned snapshot and page-authenticated immediately before consumption; NumPy alias mutation is rejected even when the tensor version counter is unchanged. |
| BF16 raw-bit validation | PASS | Destination comparisons use `int16` views, retaining NaN payload and signed-zero distinctions. |
| Partial-tail zero proof | PASS | Only partial tails are zeroed, every tail is raw-zero checked, and logical plus tail bytes must exactly equal physical selected bytes. |
| Destination/device/mapping/epoch commitment | PASS | Tensor backing, geometry, device index, GPU UUID, ordered page map, physical block IDs, and allocator epochs are committed and independently checked by the sealed production stager. |
| Failure scrub and typed teardown | PASS | Failed writes scrub allocated pages and fence/verify zero; cleanup uncertainty escalates to `V11RuntimeTeardownRequired` with affected blocks. |
| Admission ordering | PASS | Transport verification, fresh allocation, native writes, integrity, table construction, current-epoch validation, single-use epoch consumption, and enqueue execute under the production import gate with rechecks. |
| Fresh allocation lifetime | PASS | The worker requires all 1,152 destinations and rejects any source/destination `(block_id, allocation_epoch)` overlap. A reused physical ID is accepted only with a later allocator epoch. |
| 8/8 independent recompute methodology | PASS | Exactly eight restored branches must emit at least eight tokens; after successful drain and prefix-cache reset, all immutable histories are independently prefetched with identical effective seeds and all eight first tokens must match. |

## Verification

- Focused offline suite: 59 passed, 0 failed.
- Adversarial integrity subset: 15 passed, 0 failed.
- Ruff: PASS.
- Python compilation: PASS.
- `git diff --check`: PASS.
- Modal/GPU/network work by this reviewer: none.

The adversarial subset covered transport corruption, range omission/duplication/reordering, destination bit corruption, direct and NumPy-alias mutation after verification, raw BF16 edge patterns, forged/incomplete fused evidence, scrub/free/reset/enqueue cleanup failures, admission-gate loss, consumed epochs, freed/stale epochs, and physical-ID reuse under a new epoch.

## Scope boundary

The real-A100 correctness outcome remains deliberately unasserted. The GPU run must still demonstrate the exact 16K/fanout-8/1,152-block topology, actual allocator lifetimes, successful source release, fresh destination lifetimes, at least eight continuation tokens per branch, and 8/8 first-token equality. The worker's result validator fails closed if any of those conditions is false.

Machine-readable evidence, including reviewed file SHA-256 values, is in `agent05-v11-integrity-audit.json`.
