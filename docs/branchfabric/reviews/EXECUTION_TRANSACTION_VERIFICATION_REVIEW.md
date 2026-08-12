# BranchFabric Execution Transaction Verification Review

## Verdict

**PASS WITH SCOPE LIMITATIONS.** The two medium findings from the first review are
resolved for the exercised path: aggregate queue waiting is deadline-bounded, and
a destination validation failure after prepare/import now produces durable abort
and rollback records, removes staged state, releases the source fence, and permits
a clean retry. The optimized reclamation baseline also now batches identical
content-addressed chunks once per branch group.

The software improvements reproduce, but they do not justify hardware. The latest
gate artifact observed during this review remains `FAIL_NO_BUILD`, has no passing
candidates, forbids hardware implementation and simulation, preserves the negative
result, and requires `TERMINATE_HARDWARE_PATH`.

This review exercised the code at repository revision
`d113e12b5d618fa3b39b41a30f5f272bd96a18b2`, including the queue/rollback change
`d82f857755c336107431f88d56b094f7d6f53c98` and the batched-transfer change
`b6279b647a8ef3c96e9865ce9523b786f703a73a`. It is CPU-reference evidence only.

## Reviewed scope and immutable inputs

The review inspected:

- `python/sloforge/helix/characterization/reclamation.py`
- `python/sloforge/continuum/operations/resume.py`
- `python/sloforge/continuum/transaction/coordinator.py`
- `python/sloforge/continuum/transaction/token_ledger.py`
- `tests/python/test_branchfabric_reclamation.py`
- `tests/python/test_continuum_operations.py`
- the regenerated 36-trial reclamation corpus and analysis
- the latest on-disk BranchFabric gate result and report

The current committed reclamation corpus is internally consistent:

| Artifact | SHA-256 |
| --- | --- |
| `campaign.json` | `daac0cf1f595984d2eb2c5452524ec4ca174285986d934fcd42e68e43cc6287c` |
| `environment.json` | `e39cfada6e71977c1926c89cc974eeee1611ba788d199c532d7059acc4966dbe` |
| `raw/trials.jsonl` | `795a6803c1123555ea5804ddfeca711805235b89eeb30c5e723299bd26e74408` |
| `analysis.json` | `cbd27f984fa412cd33fd3714b66f8978ab3311736127058b04efc2fdab0adbf0` |
| `MANIFEST.json` | `cf1e1f363ce95d894416c3abbf56d9735fb55c48db8387126251e376ec846a79` |

The manifest entries match the named campaign, environment, and raw files. The
raw evidence contains exactly 36 distinct trials: seeds 41, 73, and 113; disabled,
minimal, and full tracing; serial and optimized baselines; and two repetitions.
Every trial restored the modeled SLO, rejected corrupt state and stale source
output, completed all eight resume transactions, used the requested CPU/local-file
engine, and reported no hidden fallback.

## Independent post-change trials

I ran two new full-trace transactions from current source:

| Baseline | Seed / repetition | Canonical trial SHA-256 | Queue / concurrency | Wall / interruption |
| --- | --- | --- | --- | --- |
| `existing_serial` | 73 / 191 | `c5726b21cbd4cda00b41faa5b541bc8dd395e5266eab068c62dff4f56d1048d4` | 8 / 1 | 1,167,757,125 ns / 819,900,333 ns |
| `optimized_bounded_parallel` | 113 / 192 | `2896fc8a9458bea7d919d7496970d2e91c6e75d28ed6cbc4695a593860c3fcc2` | 8 / 4 | 1,071,487,708 ns / 708,548,167 ns |

Both stopped admissions, restored the modeled serving SLO, lost zero rollout work,
rejected corrupted transfer state, rejected fenced-source output, and completed
all eight migrations. These are two host samples, not a performance estimate or
hardware gate input.

## Aggregate queue deadline

`_bounded_map` now establishes one monotonic deadline for submission, shutdown,
and thread joins. It no longer calls the unbounded `Queue.join`. Queue capacity is
eight; the serial and optimized trials observed maximum concurrency one and four,
respectively, and neither exceeded occupancy eight.

An independent blocked-task probe used a 10 ms aggregate timeout against a task
whose natural wait was 200 ms. The call returned `TimeoutError: bounded reclamation
worker deadline expired` after 14.99 ms, before the task's natural timeout. The
focused aggregate-deadline test passed as well.

This bounds caller waiting, not execution cancellation: Python cannot forcibly
stop an in-flight worker thread. A timed-out daemon worker may continue until its
task returns. That is a remaining harness-cleanup limitation, not an observed
violation in the 36-trial or independent campaigns.

## Post-prepare rollback and retry

An independent probe injected `VALIDATE_IMPORT` failure after destination prepare,
state import, and source fencing. The original `InjectedFailureError` propagated,
and all pre-commit safety observations held before retry:

- the transaction journal ended `ABORTING`, then `ROLLED_BACK`;
- no recoverable/incomplete transaction remained;
- the coordinator lease remained on source epoch 1;
- the gateway watermark remained 5;
- the source returned to `paused`, rather than remaining fenced;
- prepared-destination count and visible destination-session count were both zero.

After clearing the injected failure, retry completed, ownership advanced exactly
once to epoch 2, the destination became active, and the source became fenced. This
closes the prior post-prepare fault-coverage finding for the tested validation
failure. It does not prove every process-crash, cleanup-failure, or post-ownership-
commit recovery path.

## Batched content-addressed transfer

The serial path still performs eight per-branch validated local-file transfers.
The optimized path sorts unique chunk digests and performs one validated
`branch-group` transfer. The new accounting separates logical requested bytes from
unique transferred bytes.

| Trial | Logical requested | Unique transferred | Reuse saved | Transfer samples |
| --- | ---: | ---: | ---: | ---: |
| Serial seed 73 / rep 191 | 54,528 B | 54,528 B | 0 B | 8 |
| Optimized seed 113 / rep 192 | 54,496 B | 6,812 B | 47,684 B (87.5%) | 1 |

Across both the committed and independently regenerated 36-trial campaigns, all
18 serial trials transferred their complete logical byte count and all 18
optimized trials transferred exactly one eighth of it. The optimized minimal and
full traces each contain one transfer sample per trial with branch ID
`branch-group`; serial minimal and full traces contain eight. Disabled tracing
correctly records no operation samples while preserving the byte counters.

This is a strong software baseline for identical content. It is not physical
multicast: the fixture's eight branches share the same content-addressed chunks,
the transport is local filesystem, and no NIC, GPU, PCIe, NVLink, RDMA, FPGA, or
DPU was exercised. High-divergence state would reduce the reuse fraction.

## Campaign and analysis regeneration

I independently regenerated the same randomized 36-trial matrix and reran the
5,000-repetition deterministic bootstrap analysis.

| Regenerated artifact | SHA-256 |
| --- | --- |
| `campaign.json` | `f75d9f04b451402a447142093e6da78cecaba5144dec23bfe851473c9999610c` |
| `environment.json` | `e39cfada6e71977c1926c89cc974eeee1611ba788d199c532d7059acc4966dbe` |
| `raw/trials.jsonl` | `50f6796e9d57e9e885662f76900448fc6991846461f040ad1426e50a7c8c4a29` |
| `analysis.json` | `f075f98dc48892d9107f0ca77d5fa8e703b8e4483afb00094e08179c8b383761` |
| `MANIFEST.json` | `efd6c32af1b9ed38f2aeb163b2edcde1ee3aef12d3da0e5cf18ffc5018963c99` |

Raw and derived hashes are expected to differ because host durations are measured;
the analysis also binds the raw path and hash. Across 36 matched trial identities,
all 22 deterministic scheduler, state-byte, transfer-byte, outcome, queue, engine,
fault, and fallback fields matched with zero mismatches.

The software conclusions reproduce:

| Paired software speedup | Committed median (95% CI) | Independent median (95% CI) |
| --- | --- | --- |
| Migration | 2.278x (2.166–2.387x) | 2.195x (2.078–2.327x) |
| Total interruption | 1.272x (1.245–1.298x) | 1.248x (1.206–1.289x) |
| Total wall | 1.168x (1.145–1.188x) | 1.149x (1.126–1.194x) |

The independent optimized full-trace migration phase is 22.69% of interruption,
and optimized transfer accounts for 6.28% of total wall time at the campaign
median. The result demonstrates that software batching removes repeated fixture
movement. It supplies no target-hardware service curve, calibrated hardware model,
or real-platform lower confidence bound. The total-wall software speedup is not a
hardware projection, and its independent lower bound is below 1.15x in any case.

## Verification and remaining findings

Focused Python reclamation, Continuum operation, execution-analysis, and no-build
acceptance passed with 15 tests. The Rust suites passed all five bounded
state-modelcheck tests and all nine state-transaction tests; their doc tests also
passed.
The pre-existing ownership-order, early-commit rejection, durable progress guards,
gateway checks, Rust transaction tests, and bounded model-check results remain
applicable; none was weakened by these changes.

Remaining limitations:

- Raw trial fault fields are booleans rather than typed rejection codes and durable
  journal/lease/gateway hashes.
- The deadline bounds waiting but does not cancel an already running daemon task.
- The new post-prepare probe covers validation failure, not every partial-import,
  rollback-cleanup, process-reset, completion-loss, or post-commit recovery case.
- Reference model state, CPU scheduler reclamation, and local-file transport cannot
  substitute for production transformer allocation or physical accelerator/fabric
  behavior.
- The campaign contains one reference reclamation workload class; it cannot meet
  the mandatory two-real-workload-class gate.

## Final assessment

The bounded aggregate wait, pre-commit rollback/retry path, and batched
content-addressed transfer pass independent review for their declared local
CPU-reference scope. The optimized software baseline materially improves this
fixture and further weakens the case for a hardware transfer engine.

The latest gate disposition remains correct: **no BranchFabric hardware is
justified, and the hardware path must terminate**.

Machine-readable review evidence:
`artifacts/branchfabric/execution/replication/reclamation-independent-replication.json`.
