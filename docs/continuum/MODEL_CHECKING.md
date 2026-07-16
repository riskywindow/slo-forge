# Bounded protocol model checking

`sloforge-state-modelcheck` is a deterministic Rust explicit-state explorer for source, destination, gateway, coordinator, client, transport queue, ownership lease, validation state, and transaction journal.

The safe default request uses seed-controlled successor ordering and bounds of three queued messages, one token, one fault per execution, depth 32, 100,000 maximum states, and six timeout ticks. It explores message reorder/duplicate/loss, source/destination/gateway/coordinator crash, partition, delayed acknowledgment, stale-owner output, cancellation, timeout, and client disconnect.

The invariants cover unique state/output owner, monotonic epochs/watermarks, rejection of stale/duplicate/gapped output, validation before activation, source fencing, valid pre-commit abort and completed destination, bounded progress to completion/abort/operator-required, absence of valid bounded deadlock, bounded queues, and idempotent replay.

The retained safe run described in the execution ledger used seed 41 and explored 1,659 states and 3,216 transitions to maximum depth 22 under bounds of three messages, one token, one injected fault per execution, depth 32, 100,000 states, and six timeout ticks. All 15 recorded invariants passed and exploration was complete within those bounds. These are bounded exploration results, not a universal proof. Every evidence document includes bounds, assumptions, coverage, invariant statuses, and a minimized counterexample when a mutated protocol fails.

Run:

```bash
cargo run -q -p sloforge-state-modelcheck -- --safe 41
```

Protocol mutation switches deliberately disable fencing, atomic owner CAS, stale-output rejection, deduplication, gap rejection, validation-before-activation, idempotence, or queue bounds to confirm the checker detects relevant violations.
