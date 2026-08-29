# Agent 04 attempt-D CUDA pass-accounting audit

**Result: PASS.** Gate D is restored offline. Exactly one cause-aligned 1×A100 attempt D is scientifically admissible after immutable reseal and atomic reservation by the designated GPU coordinator.

## Attempt-C root cause

The failure is a deterministic accounting-capacity defect, not state corruption. The first destination subset completed with 4,080 pending CUDA records. The remaining subset requires another 4,048, for 8,128 CUDA records and 8,294 records including preflight, snapshot/authentication, and completion-fence CPU records. The fixed 4,096 live-event list therefore failed during a `raw_bit_equal` launch in the remaining subset. The full ledger was valid in size; its live-event retention strategy was not.

The admitted singleton-run destination ordering is larger: 1,152 pages × 28 raw-validation records plus 64 × (one H2D + 28 index uploads + 28 scatters) = 35,904 CUDA records, or 36,070 with the 166 CPU records. That is why merely raising the pending-event limit to 8,192 would not be a production fix.

## Remediation verdict

The recorder retains a 4,096 hard live-event bound partitioned into 3,840 normal slots and a nonborrowable 256-record `DIAGNOSTIC` emergency lane. At the normal threshold it synchronizes and materializes pending events into canonical records, appends each record before removing its pending event, and remains appendable. Final `resolve()` alone freezes the ledger and is idempotent.

If event synchronization fails partway through a drain, the materialized prefix remains exact once and only the failed suffix remains pending for retry/fail-closed handling. Normal work cannot enter the reserve. A diagnostic scrub can still be launched if normal materialization fails, and reserve exhaustion is explicit. One 28-layer production scrub is 57 CUDA records plus one CPU fence; the 256-record lane also covers the historical two-scrub 116-record case.

Canonical records remain the sole source for physical bytes, tier traffic, D2H/H2D/link-control movement, CUDA event time, and synchronization time. Incremental materialization creates no synthetic memory pass. Profiler observations and copy counters are not added to the ledger.

The shared restore recorder now also binds every consumed page to its actual full-state preflight pass, so subset chunk boundaries cannot create unknown DAG dependencies. The production stager owns exactly one full-allocation failure scrub; standalone callers retain the safe local scrub default.

Independent post-fix reviews also pass: Agent 03's engine-step/recorder lifecycle audit (`5126c1dec9a619a1b1b6776beee5ba25f58d624cfad1e569a683e418c8cb3d23`) and Agent 05d's dependency/scrub integrity audit (`28f5c3f06c2143be2942349ea21be5215847a7bc2188069e01c24b4e70c914f2`).

## Offline evidence

The six focused allocator/ownership/synchronization/accounting/adapter/runtime files passed 74/74 tests. The accounting file passed 14/14, including multi-flush exact-once behavior, cross-flush DAG closure and duplicate rejection, partial synchronization failure recovery, concrete one-sync-per-end-event assertions, protected reserve exhaustion, event-construction rollback, the 8,128-pass attempt-C shape, and the 35,904-pass singleton-run ordering. Ruff lint and format checks passed.

Audited accounting source SHA-256: `fe8c913af0414ec402ce230ffb19e8db9474bfe19b8ad539f80f0419c99b23ec`.

Audited accounting test SHA-256: `068e01ef8a8c3bfd487a2b57daec21c233336f9fcaf9e103abcd3c90d3cfb929`.

## Attempt-D authorization decision

The ledger has consumed 13,498.623626085977 of 21,600 authorized GPU-seconds, leaving 8,101.376373914023 with zero active reservations. One 1×A100 invocation predicted at 525 seconds requires 525 GPU-seconds; with the required 15% planning margin it consumes 603.75 seconds of headroom and leaves 7,497.626373914023 seconds. At $2.4984/A100-hour, predicted cost is $0.36435 and conservative cost with margin is $0.4190025.

Attempt D is scientifically admissible because the changes are limited to measured-pass retention, cleanup reserve semantics, exact dependency provenance, and single-scrub ownership. They do not alter the model, exact 16K/fanout-8 topology, bytes, continuation oracle, or go/no-go criteria.

Before invocation, the coordinator must reseal the exact source/test/review hashes into an immutable attempt-D config and manifest, atomically reserve exactly one 1×A100-80GB/525-second slot, and retain fail-closed settlement and complete Modal cleanup. This reviewer invoked no GPU, Modal, or network resource.
