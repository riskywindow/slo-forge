# Agent 07c — attempt-c budget and scientific admissibility

Status: **PASS**, conditional only on `/root` mechanically resealing the exact current hashes before the atomic reservation.

No GPU, Modal, network, or reservation action was performed by this reviewer.

## Decision

Exactly one `exp004-v11-micro-s41-c` invocation is budget-authorized and scientifically admissible at **1 × A100-80GB for at most 525 seconds**.

Attempt b is retained as a settled failed scientific run, not as a micro-validation result. It reached real A100 model/state execution but failed before restore measurement, continuation, complete movement accounting, or a published outcome. The retry repairs that deterministic lifecycle defect without selecting on amplification or correctness, so one unchanged-seed/workload attempt c is remedial rather than cherry-picked.

## Budget

The authoritative ledger contains 13,236.993626085978 consumed GPU-seconds and zero reservations under the 21,600-second authorization. Attempt c requires 525 GPU-seconds; with the 15% planning margin, its envelope is 603.75 GPU-seconds.

| Quantity | GPU-seconds |
|---|---:|
| Current remaining | 8,363.006373914022 |
| Attempt-c exact bound | 525.0 |
| Attempt-c plus margin | 603.75 |
| Projected cumulative with margin | 13,840.743626085978 |
| Remaining after margin | 7,759.256373914022 |

The exact maximum costs $0.36435 at the ledger rate; the planning envelope costs $0.4190025. Both fit the explicit $40 authorization.

## Offline closure

The cause-aligned source-queue remediation is independently approved for production synchronization and for integrity/provenance. It requires `needs_kv_cache_zeroing is False`, exactly 1,152 authenticated source IDs, the sealed production gate, lifecycle/ownership/manifest commitments, typed teardown on any mismatch, and the unchanged strict destination queue/epoch checks. It adds metadata evidence only and no movement bytes.

Final offline evidence:

- Agent03 post-fix review: PASS, `785f896089a1d3169b9c08910832ec99120887a08fee2bd76fc80ea0cd8d004a`
- Agent05c/08c integrity review: PASS, `721fe9985971a108c506a6c6f128c0c5d6ac983aaa4f2f182d9a206e6700d4a7`
- Full `make check`: PASS — 1,771 Python tests, 8 optional skips, Rust/UI PASS, `325cb8174fc362aaea2aa46f6c472ad0ac61f97adbc18b681c70089ff34c025e`
- Current ledger: `bd9c1f919f33d1c63f0edf4bc86134ad78e73d00638280c68d55111202ff19b3`

## Sole remaining condition

`/root` must create a new immutable attempt-c manifest and config binding this audit, the two post-fix reviews, final test evidence, current source/test hashes, authorization, and current ledger. The config must preserve seed 41 and every scientific workload/runtime parameter, use a unique result prefix and fresh one-time token, then receive one atomic 1-A100/525-second reservation with retries disabled.

Any hash, ledger, cleanup, workload, GPU-count, duration, or coordinator change voids this PASS. Any retry after attempt c requires a new independent budget and scientific audit.
