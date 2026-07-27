# BranchFabric Execution Transaction Verification Review

## Verdict

**PASS WITH SCOPE LIMITATIONS.** No high-severity state-integrity, ownership, or
visibility defect was found in the exercised CPU-reference reclamation path.
Independent full-trace serial and optimized transactions completed with the
expected fault rejections and bounded observed concurrency. The review found
two reasonable medium-severity hardening gaps and one low-severity evidence gap;
none changes the negative hardware conclusion.

This is a review of the local CPU reference transaction at source revision
`3ad119a7e70ed1258acc15eb11d3eed13904ea4c`. It is not evidence for GPU
reclamation, GPU state movement, a physical network, FPGA, DPU, or BranchFabric
hardware.

## Reviewed scope and immutable inputs

The review inspected:

- `python/sloforge/helix/characterization/reclamation.py`
- `python/sloforge/continuum/operations/resume.py`
- `python/sloforge/continuum/transaction/coordinator.py`
- `python/sloforge/continuum/transaction/token_ledger.py`
- `tests/python/test_branchfabric_reclamation.py`
- the focused Continuum operation, runtime, transaction, recovery, and protocol tests
- the bounded Rust state-transaction and state-modelcheck crates
- `artifacts/branchfabric/execution/reclamation/`

The committed reclamation corpus reproduced its manifest exactly:

| Artifact | SHA-256 |
| --- | --- |
| `campaign.json` | `0ca238f362feef351679e8426d6a763300949f7202d738b913287931e1a453c6` |
| `environment.json` | `e39cfada6e71977c1926c89cc974eeee1611ba788d199c532d7059acc4966dbe` |
| `raw/trials.jsonl` | `5cd3dd64e0e0ff01286f8cb7b893ae0ac690c6a923a0e069102db8eb093a340d` |

The campaign declaration and raw JSONL agree on 36 trials covering seeds 41,
73, and 113; disabled, minimal, and full tracing; both software baselines; and
two repetitions. All 36 raw trials report SLO restoration, corrupt-state and
stale-source rejection, and completed resume transactions. The maximum recorded
queue occupancy is eight and maximum concurrency is four.

## Independent replication

The authoritative independent result is
`artifacts/branchfabric/execution/replication/reclamation-independent-replication.json`.
It embeds the complete raw trial documents, their canonical hashes, the source
corpus hashes, the ordering observations, and the independent guard probes.

| Baseline | Seed / repetition | Trace | Canonical trial SHA-256 | Queue / concurrency | Result |
| --- | --- | --- | --- | --- | --- |
| `existing_serial` | 73 / 91 | full | `e9788dcb45477287c7dc2c3c288c4ea66938e849e21d82428a6e91b96f07b355` | 8 / 1 | SLO restored; corruption and stale source rejected; 8/8 transactions completed |
| `optimized_bounded_parallel` | 113 / 92 | full | `962a0ce5d7688e5c18c1ffe05572f521365266f3aebc584b7c52a89610aa086e` | 8 / 4 | SLO restored; corruption and stale source rejected; 8/8 transactions completed |

Host timings varied, as expected: the serial trial recorded 1,265,143,750 ns
total wall time and 909,877,167 ns interruption; the optimized trial recorded
1,194,095,500 ns and 827,205,458 ns. These are two independent host samples,
not a performance estimate or hardware gate input.

## State-semantics and transaction findings

The exercised high-level ordering is correct:

1. The checkpoint artifact and current lease epoch, fencing token, runtime,
   and committed watermark are verified.
2. The source is required to be paused and is fenced before import completes.
3. Destination import is structurally and continuation-hash validated.
4. Commit intent, state hash, commit watermark, and rollback watermark are
   durably recorded.
5. Ownership is changed by a lease-and-transaction compare-and-swap.
6. The gateway owner epoch advances exactly once at the committed watermark.
7. Only then is the destination activated and allowed to emit output.

Independent method instrumentation observed this exact sequence for all 16
migrated branches across the two fresh trials:
`destination_validated`, `ownership_committed`, `gateway_switched`, then
`destination_activated`. Each validation reported structural and continuation
validity. This is evidence for the exercised `resume_checkpoint` orchestration,
not a claim that the gateway and coordinator independently implement consensus.

An independent premature-commit probe called `commit_ownership` while the
transaction remained `PROPOSED`. It raised `InvalidTransition` and left the
source lease byte-for-byte unchanged. Thus ownership cannot commit before a
persisted commit intent through the reviewed coordinator API.

The new durable progress CAS was also probed independently. After committing
state version 7 and token index 5:

- state-version regression was rejected as `CoordinatorConflict`;
- a mismatched fencing token was rejected as `StaleOwner`;
- the durable watermarks remained exactly version 7 / token 5 after both failures.

The focused gateway tests independently reject stale owner epochs, gaps,
conflicting duplicates, and state-version regression. The trial's stale-source
field itself proves the fenced source runtime cannot generate after cutover; it
does not by itself prove gateway stale-epoch rejection.

## Integrity, security, and visibility

The transaction binds the expected tenant, model contract, segment digests,
continuation hash, owner epoch, fencing token, transaction identifier, and
watermark. Local-file transfer has a finite deadline and validates content
digests. A corrupting read wrapper changed transferred bytes and restore failed
before the state could enter the resume transaction. No corrupted state became
visible and no ownership or gateway mutation was performed for that corrupt
probe.

The driver-independent visibility boundary is the ordered combination of
coordinator ownership CAS, gateway epoch switch, and runtime activation. The
standalone `GatewayCommitLedger` is intentionally not coupled to the
coordinator; callers must use the reviewed orchestration. Distributed consensus
is not implemented or claimed.

No cross-tenant or hostile filesystem campaign was part of this vertical.
Those properties continue to rely on the existing tenant-scoped content-store,
transport, and Continuum security suites.

## Bounded and formal verification

Focused acceptance passed:

```text
10 Python reclamation/transaction/runtime/recovery tests passed
5 sloforge-state-modelcheck tests passed
9 sloforge-state-transaction tests passed
all selected Rust doc tests passed
```

The optimized path declares four workers; the serial path declares one. The
work queue has capacity eight, submissions and reads use 30-second timeouts,
thread joins use 30-second timeouts, and the independent trials observed no
capacity or worker-bound violation. The general Rust model checker also passed
its complete declared finite fault exploration, including stale output,
monotonic epochs/watermarks, ownership fencing, and bounded queues. This is a
bounded result, not a universal proof and not a trace-specific model of the
Python reclamation runner.

## Findings

### MEDIUM — the aggregate queue drain has no deadline

`_bounded_map` bounds queue capacity and per-operation queue/thread waits, but
calls `pending.join()` without a timeout. If a worker exits between dequeue and
`task_done`, the aggregate caller can wait indefinitely and never reach the
timed thread joins. The current task body uses `finally: task_done()` and both
replications terminated, so this is a robustness gap rather than an observed
failure. Replace the unbounded join with a deadline-aware completion condition
or equivalent bounded executor protocol.

### MEDIUM — fault coverage stops before transactional abort/recovery

The corruption injection is checked during `restore_reference_capture`, before
`resume_checkpoint` begins a transaction. It proves corrupt input is rejected
before visibility, but does not exercise corruption or partial input after
destination preparation/import, transaction abort, reset, completion loss, or
recovery from a partially staged destination. Add one post-prepare corruption or
partial-import fault and retain evidence that the destination was aborted, the
source lease and gateway remained current, and recovery exposes no staged state.

### LOW — raw trial booleans do not preserve rejection codes or journals

The runner catches broad `Exception` for corruption and emits only booleans for
corruption, stale-source rejection, and transaction completion. The independent
review can corroborate these with focused tests and dynamic ordering, but the
committed campaign alone cannot distinguish the intended integrity error from
an unrelated exception or reconstruct durable transaction order. Persist the
typed error code plus transaction/journal/lease/gateway hashes in future raw
trials.

## Final assessment

The reviewed CPU-reference vertical provides credible causal software evidence
for pausing eight branches, checkpointing and transferring exact state,
logically reclaiming CPU scheduler capacity, restoring the modeled serving SLO,
and resuming without work loss. Its exercised ownership, stale-writer,
regression, integrity, and visibility properties passed independent checks.

It does **not** establish physical GPU reclamation, real-model state behavior,
GPU or network transfer behavior, or a hardware opportunity. The correct
hardware disposition remains no-build unless separately authorized real
workloads and hardware measurements pass the mandatory gates.
