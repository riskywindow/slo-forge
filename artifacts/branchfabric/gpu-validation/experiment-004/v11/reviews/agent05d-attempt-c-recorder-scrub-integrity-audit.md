# Agent 05d — attempt-C recorder/scrub integrity audit

Status: **PASS** for exactly one bounded replacement micro-validation. This is not a v11 performance result and does not authorize the integrated experiment.

## Attempt-C disposition

Attempt C failed deterministically in instrumentation: the restore recorder reached its former 4,096-pending-pass bound before the next pass was claimed or launched. The same saturated recorder then rejected both scrub entry points before scrub work began. No correctness mismatch, successful pass ledger, amplification result, or latency result was produced.

The failure remained fail closed. The stager admitted no branches and reported all 1,152 destination blocks (IDs 1,153–2,304) in `V11RuntimeTeardownRequired`. Because scrub was unproven, the stager did not free those allocations for reuse. Its first prefix-cache reset correctly returned false while the 1,152 references remained. Worker-finally cleanup then detached and freed the requests and made one post-free reset, which succeeded. Provider postflight found zero active apps, tasks, containers, endpoints, reservations, child processes, and profilers; no remote compute process remained, CUDA-clean import passed, and the A100 reported 4 MiB used.

## Remediation verdict

The current recorder keeps a hard 4,096 live-event bound, reserves 256 slots exclusively for diagnostics, and incrementally converts normal pending events into canonical records at a 3,840-pass threshold. Materialization does not freeze the recorder. A pending item is removed only after its record is appended; failures retain the unmaterialized suffix; final `resolve()` drains the suffix once and is idempotent. The separate 65,536 total-record bound protects 256 records for cleanup. Attempt C's 8,128 expected restore CUDA passes and a conservative 35,904-pass singleton-run topology both fit.

Failure scrubbing now has one production owner. Standalone subset writes keep local scrub by default, while the production callback explicitly delegates to the stager, which scrubs all allocations once, authenticates typed completion-fenced evidence, and only then frees. Tests inject both first-subset and remaining-subset failures and prove one scrub, unique pass IDs, exact zeroed-byte conservation, and no admission.

The audit also found and closed a latent pass-DAG defect: full-state preflight chunks and restore-subset chunks can have different boundaries. `V11VerifiedTransport` now binds each page to the actual preflight pass that authenticated it; consumption depends on the deduplicated exact covering IDs. Arbitrary inside-chunk subset starts validate, while missing, duplicate, forged, and invalid-bound provenance fail before writes.

Incremental materialization creates no physical-memory operation and therefore no extra movement record. Actual scrubs remain distinct `DIAGNOSTIC` passes. There is no pass loss or byte double count in an admissible ledger.

## Offline evidence

The combined independent suite passed 70/70:

```text
PYTHONPATH=python:experiments/branchfabric python -m pytest -q \
  tests/python/test_vllm_reclamation_v11_accounting.py \
  tests/python/test_vllm_reclamation_v11.py \
  tests/python/test_gpu_reclamation_runtime_v11.py \
  tests/python/test_vllm_reclamation_v11_ownership.py \
  tests/python/test_vllm_reclamation_v11_sync.py

70 passed in 1.67s
```

Ruff lint and format checks passed for all six changed implementation/test files.

## Attempt-D admissibility

One replacement micro is scientifically admissible. Attempts B and C stopped at deterministic integration/instrumentation defects before a successful outcome, and the source changes are narrow and independent of any observed amplification or correctness result. The ledger has 8,101.376373914023 GPU-seconds remaining under the 21,600-second hard authorization. A 525-second one-GPU invocation plus 15% margin consumes at most 603.75 GPU-seconds, leaving 7,497.626373914023 GPU-seconds. Conservative incremental cost is $0.4190025 at $2.4984/GPU-hour.

Attempt D must retain seed 41, the pinned model/runtime, exact 16K/fanout-8/suffix-256 workload, 1,152 pages, 1,056,964,608 logical bytes, 525-second timeout, minimal instrumentation, no automatic Modal retry, and a fresh one-time authorization/reservation. A complete canonical DAG is mandatory before any movement/latency result is published. Recurrence of the same recorder, scrub-accounting, or preflight-dependency class is terminal for this retry sequence.

The full machine-readable evidence and current hashes are in the companion JSON.
