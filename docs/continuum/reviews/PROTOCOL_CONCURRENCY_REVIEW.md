# Continuum protocol, concurrency, and formal-method review

Date: 2026-08-02

Scope: `python/sloforge/continuum/transaction/`, `crates/sloforge-state-transaction`, and `crates/sloforge-state-modelcheck`. The review covered ownership CAS, lease expiry, fencing, rollback windows, gateway token acceptance, SQLite crash/concurrency behavior, snapshot recovery, model-check scope, invariant sensitivity, and evidence integrity.

## Result

No open high-severity defect remains in the reviewed code. Five high-severity and four medium-severity findings were fixed. The focused Python suite passes 22 tests; the Rust protocol crate passes 9 tests; the model checker passes 5 tests; strict Python typing, Ruff, and Rust Clippy with warnings denied pass.

The expanded safe model-check run for seed 41 passed all 15 invariants after exploring 1,659 states and 3,216 transitions to maximum depth 22. Exploration was complete within the declared bounds: three messages, one token, one fault per execution, depth 32, 100,000 states, and six timeout ticks. This is bounded evidence, not a universal proof.

## Severity-ranked findings

### High — fixed: post-commit failure could enter the pre-commit rollback path

The Python transition table allowed `OWNERSHIP_COMMITTED -> ABORTING -> ROLLED_BACK`. The Rust state machine had the same issue through `DESTINATION_LOST -> ABORTING`. That could label an old source as valid after the destination epoch became authoritative.

Both implementations now derive whether ownership was ever committed from persisted history and reject `ABORTING`, `ROLLED_BACK`, and `FAILED_BEFORE_COMMIT` after that boundary. Post-commit loss must proceed through destination recovery, `FAILED_AFTER_COMMIT`, or `OPERATOR_REQUIRED`. Adversarial tests verify that the destination lease remains authoritative.

### High — fixed: replay identity was not exact in the Rust state machine

Rust treated every request for the current phase as an idempotent replay, even when logical tick or reason differed. Ownership commit itself was not retry-idempotent.

Same-phase replay now requires an exact match to the last persisted transition. Ownership commit accepts an exact retry only when destination owner, epoch, state version, token watermark, and derived lease expiration all match; a differing replay fails with `IdempotencyConflict`. Python multi-connection transition retries now re-read a winning CAS and return success only for an exact event match.

### High — fixed: gateway duplicate and concurrency semantics were incomplete

The Python SQLite connection was thread-bound and unprotected, so simultaneous calls through one ledger were unsafe. A byte-identical terminal-token retry was rejected because terminal state was checked before duplicate identity. The Rust ledger compared only token ID, allowing a repeated index with altered state version, transaction ID, or terminal flag.

The Python ledger now uses a reentrant lock, `check_same_thread=False`, `BEGIN IMMEDIATE`, and WAL/FULL durability for file-backed databases. Duplicate identity is checked before terminal rejection. Both implementations compare the full token event, reject regressing state commit versions, reject stale epochs and gaps, and require contiguous owner epochs. Two independent SQLite connections are tested concurrently: exactly one accepts a token and the other returns an exact duplicate.

### High — fixed: Rust coordinator recovery admitted invalid ownership state

The Rust reference coordinator permitted more than one active transaction for a session and recovery did not reject that condition. Recovery validation also omitted journal sequencing, legal edges, post-commit rollback history, evidence presence, identity bounds, watermark ordering, and capacity bounds.

Transaction creation and snapshot recovery now enforce one active migration per session. Recovery validates journal sequence and phase continuity, exclusive deadlines, phase legality, rollback windows, source/destination lease consistency, cutover evidence, watermarks, identities, hashes, and fixed coordinator capacities. Commit now checks that the source lease is still unexpired at the CAS point.

### High — fixed: model-check mutations did not exercise all claimed invariants

The original checker only crashed components in a few phases, did not attempt a token gap, suppressed the validation invariant when validation enforcement was disabled, and allowed some protocol mutations to deadlock before reaching the unsafe behavior. Result validation also accepted altered aggregate status or assumptions.

Crash and restart actions now cover pre-copy, quiesce, import, validation, commit, gateway switch, and active phases as applicable. The checker explicitly attempts gap output. Validation and fencing mutations can reach unsafe activation/commit, and the invariant checks remain unconditional. Tests confirm independent sensitivity to non-atomic ownership CAS, stale-output acceptance, duplicate acceptance, gap acceptance, missing validation, missing fencing, and altered counterexample/result evidence.

### Medium — fixed: lease and transaction deadline boundaries differed

Python treated lease expiration as exclusive but transaction expiration as inclusive; Rust authorized a lease at its expiration tick. All reviewed implementations now use an exclusive deadline: authority is invalid when `now >= expiration` or `now >= transaction timeout`.

### Medium — fixed: cutover evidence was mutable or recordable too early in Rust

Cutover evidence could be recorded before destination validation and overwritten with a different value. It also did not persist the token cutover and rollback boundaries before commit.

Evidence is now accepted only in `DestinationValidating`, requires at least one state hash for a validated destination, records both watermarks, is exact-idempotent, and is a prerequisite for commit intent. Ownership CAS must match the recorded commit watermark.

### Medium — fixed: token and coordinator storage were not explicitly bounded in Python

The gateway token table and per-session transaction history had no application-level bound. They now fail closed at configurable/per-protocol limits: at most 65,536 retained gateway token records and 256 transaction records per session. The Rust reference coordinator also declares and enforces session and transaction capacity bounds.

### Medium — fixed: client acknowledgment cursor lacked an independent durable operation

Python could mark a token acknowledged during acceptance but could not persist a delayed acknowledgment. `acknowledge_client` now rejects cursors ahead of the gateway or behind the prior client cursor and handles an exact replay without advancing the database version.

## Remaining assumptions and limitations

- The coordinator and gateway use separate durable ledgers. Correct cutover relies on the adapter contract that source quiesce/fencing drains output through the declared watermark before ownership CAS. A future distributed implementation should add an explicit gateway cutover barrier or make the gateway consult the authoritative lease. This assumption is now machine-readable in model-check evidence.
- Python records source fencing through the guarded phase sequence; Rust records a caller-supplied fencing-evidence flag. A coordinator cannot independently prove that a runtime honored a physical writer fence. Runtime adapters remain version-scoped trust boundaries, while the gateway epoch check provides the independent post-switch acceptance fence.
- The local Python coordinator deliberately fails closed after 256 retained transactions for one session. It has no archival/compaction API in this implementation.
- Rust `OwnershipCoordinator` is an in-memory deterministic CAS reference whose snapshot persistence is caller-owned. The exercised durable implementation is the Python SQLite coordinator; no custom distributed consensus system is claimed.
- The model checker abstracts payload bytes and authenticated state validation, and assumes durable coordinator/gateway recovery. The retained run explores one token and at most one injected fault per execution; correlated multi-fault executions are outside this result.
- The checker proves reachability to a terminal/recoverable outcome within the finite graph under its stated weak-fairness assumption. It is not a universal liveness proof and does not enumerate SQLite instruction-level interleavings.

## Acceptance evidence

Commands executed:

```text
PYTHONPATH=python .venv/bin/pytest -q \
  tests/python/test_continuum_protocol_review.py \
  tests/python/test_continuum_transaction.py \
  tests/python/test_continuum_migration.py \
  tests/python/test_continuum_operations.py \
  tests/python/test_continuum_flagship.py
# 22 passed

.venv/bin/ruff check python/sloforge/continuum/transaction \
  tests/python/test_continuum_protocol_review.py
# passed

.venv/bin/mypy --strict python/sloforge/continuum/transaction \
  tests/python/test_continuum_protocol_review.py
# passed

cargo test -p sloforge-state-transaction -p sloforge-state-modelcheck
# state transaction: 9 passed; model checker: 5 passed

cargo clippy -p sloforge-state-transaction -p sloforge-state-modelcheck \
  --all-targets -- -D warnings
# passed

cargo run -q -p sloforge-state-modelcheck -- --safe 41
# passed: 1,659 states; 3,216 transitions; max depth 22; 15/15 invariants
```

The protocol-review regression tests are in `tests/python/test_continuum_protocol_review.py`. They cover exclusive deadlines, post-commit rollback rejection, exact terminal replay, state-version monotonicity, client acknowledgment monotonicity, fixed storage bounds, and concurrent SQLite linearization across independent connections.
