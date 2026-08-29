# Agent 03 integrated cross-runtime audit

Verdict: **PASS — the reviewed controller/launcher contract admits exactly one
integrated v11 transaction, subject to the designated coordinator's current
reservation, full-repository test, settlement, and Modal-cleanup gates.**

This was a read-only source and offline-test audit. It did not invoke Modal,
CUDA, a GPU, or the network, and it did not modify the frozen v10 path.

## Cross-runtime contract

The final controller, launcher, worker, and strict methodology hashes are
sealed in the accompanying JSON. The controller's prelaunch boundary now also
requires exactly nine uniquely labelled content-addressed bindings, with an
exact label-to-repository-path map: the methodology; worker, controller, and
launcher; their three focused tests; this Agent03 JSON review; and a fresh
integrated-prelaunch `make check` artifact. Missing, extra, duplicate,
path-substituted, and hash-tampered bindings fail closed, as does a make-check
artifact without exact PASS status and `make check` command. The controller
launches exactly two child
roles with the worker's complete CLI contract: role, strict config, pinned
model snapshot, per-role work root, shared barrier root, and physical GPU UUID.
Engine-start, readiness, per-probe reset, final reset, transaction-ready, and
result records are bound to retained process identity, engine nonce, physical
UUID, selected-load hash, sanity-result hash, and budget-authorization hash.
The transaction command must equal the authoritative expanded v11 config, so a
late load or methodology change fails closed.

The two sanity guards are exactly the requested 12 rps stable and 15 rps
overload probes, each with a three-second measurement window. The controller
explicitly records that no broad recalibration occurred. The retained frozen
v10 global-serving source hashes remain unchanged.

## Scientific result boundary

Success is not accepted from process return code alone. Each result must match
its role-specific schema, child PID, attempt, physical UUID, and measured
no-deferred-compilation interval. GPU0 must present overload, bounded backlog,
excess two-GPU service, negative/drained queue recovery, SLO restoration and
stability, plus nonempty live vLLM queue samples and completed 9-rps serving
during restore. GPU1 must present the exact 16K/fanout-8, 1,152-block,
1,056,964,608-byte topology; Gate A/B/C/D-equivalent correctness fields;
fresh destination lifetimes; integrity and mapping commitments; measured
movement and StatePassRecords; at least eight continuation tokens per branch;
and 8/8 independent-recompute first-token matches.

The controller also rejects any launch whose content-addressed prerequisites
do not prove all four offline gates, the passing real-A100 micro-validation,
PASS reviews 9–12, and the post-micro manifest's explicit
`APPROVED_EXACTLY_ONCE` admission.

Any non-original integrated attempt additionally requires an explicit
replacement authorization, and the only admitted replacement identity is
exactly attempt B with reservation B. The verifier binds the replacement and prior
attempt IDs, the exact scientific-null status, and the config's pre-reservation
ledger hash. Cleanup, packaging, and budget evidence must use exact repository
paths; both their manifest wrappers and embedded payloads must pass. Cleanup
must prove the original attempt, zero provider state, no active reservation,
and the same charged-ledger hash. Packaging must prove the exact tests mount,
functional layout test, and launcher/test hashes. Budget must bind the exact
replacement attempt/reservation/ledger and a single authorized replacement
with hard-ceiling, margin, and cost passes. Tests reject absent authorization,
wrapper or embedded FAIL, wrong attempts, wrong ledger, nonzero provider state,
path substitution, and hash tamper.
It also rejects a coordinated attempt-C reseal even when the budget payload is
rewritten consistently, and status-like prefixes such as `PASS_REVOKED` do not
pass exact replacement status checks.

## Budget, isolation, deadlines, and cleanup

The Modal graph requests exactly `A100-80GB:2` for at most 588 seconds, with no
retry, no warm buffer, one single-use container, no endpoint, and only the two
preexisting authorized volumes. The launcher no longer imports the unrelated
micro-validation Modal graph and has no `create_if_missing=True` path. Torch
2.11.0, vLLM 0.23.0, Transformers 5.14.1, Pydantic 2.13.4, and Modal 1.5.3 are
pinned; the Modal SDK is checked locally before spawn and again remotely.
The remote image now copies the 11 MiB repository `tests` tree to
`/opt/sloforge/tests`, closing the observed pre-inventory failure in which the
exact prelaunch verifier could not resolve its three bound test paths. This is
a packaging-only change; it does not change the worker, controller,
methodology, runtime pins, or scientific configuration.
The functional packaging test constructs an `/opt/sloforge`-equivalent tree,
copies all three bound integrated tests into their exact paths, and validates
each path and SHA-256 with the production `validate_bound_artifact` function.
The final layout test covers every integrated-prelaunch and replacement
evidence binding, including the exact prior-attempt cleanup/settlement file
that is now copied explicitly into the image.

The local reservation must be the sole exact two-GPU reservation and must
reconstruct to the config's pre-reservation ledger hash. Remote authorization
binds that reservation, the immutable config, one-time preflight token, 588
wall seconds, and 1,176 GPU-seconds.

The function partitions its deadline into non-overlapping windows: controller
operation through second 568, controller cleanup through second 578, and
immutable publication through Modal's second-588 cap. Both worker process
groups are signaled concurrently inside the same cleanup window. Inventory and
compute-process checks consume only remaining deadline, and controller success
requires zero compute processes, stable GPU identity, and a CUDA-clean parent.
Remote publication uses a staging tree, content manifest, atomic rename, volume
commit, and refuses overwrite. Provider cleanup and ledger settlement are
correctly left provisional for the designated coordinator rather than claimed
complete inside the paid function.

## Offline verification

The focused integrated controller, launcher, worker, and methodology suite
passes **92/92**. Ruff check, Ruff format check, and Python compilation pass.

The broader 167-test runtime set has one host-scheduler-sensitive test in the
unchanged frozen-v10 synthetic serving protocol. It failed under the full suite
when an arrival exceeded the 100 ms lateness/burst bound, then passed **1/1** in
isolation. A second full-suite run reproduced only that load-sensitive test;
166 other tests passed. No timing bound or frozen source was changed. This does
not identify a controller/launcher contract defect, but the root coordinator
must still record the separately mandated full `make check` PASS before a paid
launch.

Final launch-admission decision: **PASS for exactly one integrated v11 run.**
The final post-micro manifest must bind this review's final hash and the fresh
post-change make-check hash. This review deliberately does not bind that
subsequently generated manifest, keeping the commitment graph acyclic. The
designated coordinator must then bind the resulting manifest, current hashes,
and reservation, then
settle the ledger and verify no Modal app, task, container, endpoint,
reservation, child process, or profiler remains after return. Direct v10 timing
comparison is allowed only if both observed devices are exactly
`NVIDIA A100-SXM4-80GB`; otherwise only the byte/amplification comparison is
unconfounded.
