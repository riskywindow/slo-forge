# Continuum Release and Clean-Room Review

Date: 2026-08-02
Reviewer: independent release/clean-room lane
Scope: Make and CLI surfaces, clean-room/Docker/GPU gates, retained evidence,
cleanup state, and release staging in the existing dirty worktree.

## Executive result

**Not release-ready at the reviewed baseline HEAD.** The implementation and focused
evidence checks are healthy, but the source must be committed before any retained
Continuum evidence is regenerated. At review time, every current demo/evaluation
artifact names baseline commit `7e51ea7f7338755d23f889820558a4e046d6c42e`, whose
tree does not contain Continuum, and the clean-room script deliberately archives
that same HEAD. This is a release-provenance blocker, not a test failure.

The three gated scripts now remove stale status, Docker builds an exact
`git archive` context, and a registered CUDA test performs PyTorch CUDA conversion
with independent canonical comparison. Those corrections were inspected after
landing in the shared worktree.

## Findings

### High: retained evidence is not yet bound to the implemented source tree

The current evaluation summary records baseline HEAD `7e51ea7f...`, while all
Continuum sources are uncommitted. `tools/clean-room-continuum.sh` correctly uses
`git archive "$revision"`, so running it before a source commit would omit Continuum.
The earlier Docker dirty-context ambiguity is fixed: Docker now also builds an exact
archive of HEAD.

Required fix:

1. Selectively stage and commit the Continuum source tree without staging unrelated
   baseline-demo churn.
2. Run every retained demo, model check, and CPU campaign from that commit.
3. Run clean-room against that commit and retain its revision/tree/hash receipt.
4. If any source changes after regeneration, invalidate and repeat the evidence run.

Do not describe the currently retained artifacts as results for the Continuum source
tree.

### High: durable demo databases retain logically active leases

No Continuum child process was running during review, and no host-wide fault rule was
found. However, every retained `coordinator.sqlite` contains a lease with logical
expiration `120000`; the flagship/demo model clock stops below that boundary. The
destination lease is therefore still valid in its declared logical domain, even
though the process has exited. `state capture` also persists a newly proposed
migration transaction without moving it to a terminal phase.

This fails the literal release gate that no lease remain active and makes cleanup
depend on process disappearance instead of a persisted protocol action.

Required fix:

- Add a fenced, compare-and-swap lease release/expiry operation that preserves audit
  history and rejects stale releasers.
- End CLI capture transactions in an explicit terminal capture/abort state.
- Have demos expire/release branch and destination leases during clean shutdown.
- Add a release audit command/test that opens every retained coordinator, asserts
  every transaction is terminal, and proves every lease is expired/released at the
  recorded final logical time.

### High: the scriptable CLI does not yet close the advertised plan-to-execution loop

All existing top-level commands are exposed and tested, but `migration plan` emits a
plan that `continuum migrate` cannot consume. `migrate` only runs the fixed reference
pre-copy flagship. There are no CLI commands for pause, resume, checkpoint,
incremental checkpoint, or clone even though their Python operations exist. The
benchmark command accepts a seed list, not the requested matrix document.

Required fix: expose the implemented operations through typed CLI commands, accept a
compiled plan in migration execution, reject strategy/capability mismatches, and add
one CLI system test that captures/checkpoints, compiles/plans, migrates or resumes,
and validates the final capsule/transaction. If the flagship command remains, name it
explicitly as a demo rather than presenting it as the general plan executor.

### Medium: clean-room failure evidence is discarded

The clean-room script now removes a prior `passed` result before a run, so a failed
rerun cannot leave a stale success receipt. On failure, however, the EXIT trap removes
the temporary tree and its log without publishing a failed receipt or diagnostic log.
The script also has no outer wall-clock timeout.

Required fix: copy the log and atomically publish a `failed` receipt from the trap,
including revision, source tree, failing stage, and log SHA-256. Bound the overall
run with a portable timeout/worker wrapper. Retain `passed` only after all assertions
and copies complete.

### Medium: the evidence summary is hash-checked but not a complete release seal

The reviewed summary's 20 raw references all match their recorded sizes and SHA-256
digests, and all five root report copies exactly match the generated report copies.
Tampered raw files are rejected by tests. Remaining closure gaps are:

- `evaluation_id` commits only to seed, Git string, and flagship hashes, not the
  conversion, stop-and-copy, planner, environment, or observed timing records.
- the summary itself has no retained parent/index digest;
- the report-set hash manifest exists only as transient CLI output in the current Make
  workflow;
- the Markdown provenance section lists flagship/conversion hashes but omits the
  stop-and-copy and planner hashes;
- the report says "authenticated" although plain SHA-256 references provide integrity,
  not origin authentication.

Required fix: produce a release evidence index bound to the source commit and source
tree. Include hashes and sizes for the evaluation summary, all raw inputs, generated
reports, demos, model-check output, Docker/GPU status, and clean-room receipt. Retain
the `ReportSet` manifest. Say `hash-verified` unless a signature/MAC is actually used.

### Medium: raw SQLite work directories are an unsafe commit boundary

The untracked artifact tree contains coordinator/gateway databases plus WAL/SHM
sidecars. Several SHM files are non-empty, and the work databases are not referenced
by the evaluation summary. Blindly staging `artifacts/continuum/` would retain
ephemeral SQLite coordination files whose transaction journal hashes are not indexed.
Conversely, dropping all work databases leaves capsule transaction-journal digests
without independently retained journal events.

Required fix: export canonical bounded transaction and gateway journals to JSON,
hash-reference them from the demo/evaluation manifest, checkpoint/close SQLite, and
exclude `work/`, `*.sqlite-wal`, and `*.sqlite-shm` from the release evidence commit.
Retain a SQLite snapshot only when it is intentionally indexed and validated.

### Medium: hardware/container receipts need stronger exercised-path provenance

Default Docker/GPU gating is honest: missing Docker is `unexercised`, disabled GPU is
`unexercised`, and unavailable CUDA-enabled PyTorch is `unavailable`. Docker now uses
an exact Git archive and records the revision; the CUDA marker is registered and one
test is collected. An exercised GPU status currently records only a path label, not
the revision, hardware/software manifest, raw timing, or result hash. The target is
named `benchmark` but the test proves conversion correctness, not throughput.

Required fix: when hardware is exercised, emit a typed receipt with revision/tree,
GPU identity, driver/CUDA/PyTorch versions, exact command, test result hash, and raw
conversion measurements. Keep correctness-only status distinct from benchmark status.
The Docker exercised receipt should likewise hash the verified in-container migration
manifest, not rely only on process exit.

### Low: reproducibility documentation names the wrong summary file

`docs/continuum/REPRODUCIBILITY.md` says `evaluation.json`; the implementation writes
`evaluation-summary.json`.

## Selective staging guardrail

The reviewed worktree contained 1,056 changed paths: 386 modified, 568 deleted, and
102 untracked. Most modified/deleted paths are regenerated pre-Continuum demo
artifacts and must not be swept into the release commit.

Use an explicit allowlist, then inspect the staged set:

```bash
git add -- \
  Cargo.toml Cargo.lock Makefile README.md docs/reports/EXECUTION_LEDGER.md pyproject.toml \
  python/sloforge/cli/main.py python/sloforge/cli/continuum.py \
  python/sloforge/continuum adapters/continuum \
  crates/sloforge-continuum-ir crates/sloforge-state-transaction \
  crates/sloforge-state-modelcheck schemas/continuum scenarios/continuum \
  deploy/docker/Continuum.Dockerfile \
  tools/clean-room-continuum.sh tools/continuum-docker-smoke.sh \
  tools/continuum-gpu-benchmark.sh \
  tests/python/test_continuum*.py \
  docs/continuum docs/CONTINUUM_RELATED_WORK.md paper/continuum

git diff --cached --check
git diff --cached --name-status
```

Do not use `git add -A`, `git add .`, or stage the whole artifact directory. Add the
post-commit evidence index and its explicitly referenced files in a separate evidence
closure commit.

## Checks executed

- `bash -n` passed for all three Continuum release scripts.
- `uv run --locked pytest --collect-only -q -m continuum_gpu` collected exactly one
  CUDA equivalence test after marker registration.
- Focused CLI/evaluation/report tests passed: **7 passed**.
- The current evaluation summary's **20/20** raw references matched size and SHA-256.
- The five published root reports exactly matched their generated evaluation copies.
- `sloforge continuum report` regenerated and returned five valid hashed report
  references.
- The retained model-check JSON is strict JSON and reports 1,659 states, 3,216
  transitions, 15 passing invariants, depth 22, and explicitly scopes the result as
  bounded rather than universal.
- Process inspection found no SLOForge/Continuum test or demo child process. The only
  matching long-lived process was an editor extension helper, outside Continuum.

These checks validate the current mechanics; they do not cure the source-commit
provenance or lease-release blockers above.

## Release acceptance commands

Run after the selective source commit and cleanup fixes:

```bash
git diff --cached --check
make continuum-check
make continuum-demo
make continuum-migration-demo
make continuum-fault-demo
make continuum-fork-demo
make continuum-compatibility-demo
make continuum-benchmark-cpu
make continuum-benchmark-gpu
make continuum-docker-smoke

uv run --locked sloforge continuum migration modelcheck \
  --seed 41 --output artifacts/continuum/modelcheck/result.json

make check
make fabric-check
make genesis-check
make demo
make fabric-demo
make autopsy-demo
make genesis-zero-day-demo
make forgeci-demo
make warmpath-demo

make continuum-clean-room-test
```

Then independently verify:

```bash
test "$(jq -r .revision artifacts/continuum/clean-room/result.json)" = "$(git rev-parse HEAD)"
test "$(jq -r .source_tree artifacts/continuum/clean-room/result.json)" = "$(git rev-parse HEAD^{tree})"
test "$(jq -r .status artifacts/continuum/clean-room/result.json)" = passed

uv run --locked sloforge continuum report \
  --evaluation artifacts/continuum/evaluation/evaluation-summary.json \
  --output artifacts/continuum/evaluation/report-set.json

git diff --check
git status --short
```

The final release report should distinguish the tested source revision from any later
documentation/evidence-only closure commit and should cite only artifacts indexed by
the release evidence manifest.
