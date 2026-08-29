# Agent 03 integrated packaging-replacement audit

Verdict: **PASS for one replacement transaction, but GPU NO-GO until the
updated package is content-addressed and newly reserved.**

FunctionCall `fc-01M0C4SKRP86EM7VH6CYNZ7WG1` was a real paid two-A100
allocation, so it is not treated like the earlier local import failure. It ran
remotely for 57.046 seconds and is conservatively charged the full 1,176
GPU-seconds. The ledger now contains no active reservation and retains
6,662.486373914022 authorized GPU-seconds.

Scientifically, attempt A is null. It failed in the content-addressed
prerequisite verifier because `/opt/sloforge/tests` was absent, before GPU
inventory, model load, workers, vLLM state, serving load, reclamation, or any
measurement. It therefore does not satisfy or consume the requirement for one
scientifically valid integrated transaction. Its failed completion and full
failure charge must remain immutable.

The fix is correctly scoped: copy the 11 MiB, symlink-free repository `tests`
tree to `/opt/sloforge/tests` in the Modal image. That supplies the exact three
test paths already required by the prelaunch map and changes no worker,
controller, methodology, model, runtime pin, load, or measurement behavior.
The functional image-layout test constructs an `/opt/sloforge`-equivalent
tree, copies all three bound integrated test artifacts into their exact paths,
and authenticates every SHA-256 with the production artifact validator. The
launcher now also bundles the exact prior cleanup/settlement artifact, and the
functional layout test authenticates every integrated-prelaunch and
replacement-evidence binding. The integrated suite passes 92/92; Ruff and
compilation pass.

The replacement controller now semantically authenticates the replacement
authorization rather than accepting its presence alone. It requires the exact
replacement/prior IDs, null-before-runtime status, charged-ledger continuity,
exact cleanup/packaging/budget paths, and PASS in both the wrapper and embedded
evidence. Embedded cleanup proves zero provider state and ledger equality;
packaging proves the exact mount/layout and current hashes; budget proves the
single replacement identity and ceiling/margin/cost gates. Adversarial tests
cover absence, wrapper and embedded FAIL, wrong attempts, wrong ledger,
nonzero provider state, path substitution, and hash tamper.
The final identity is narrowed to exactly attempt B/reservation B; a coordinated
attempt-C rewrite is rejected, as are status prefixes such as `PASS_REVOKED`.

A replacement cannot launch from the existing seal. The launcher and launcher
test hashes changed, and attempt A's immutable result prefix already exists.
Before allocating another GPU, the coordinator must hash the updated Agent03
cross-runtime review, run a fresh full `make check`, update the exact prelaunch
manifest, create a new immutable attempt/config bound to ledger SHA-256
`84175458f5e0b2f54fdc7c1e2c37c7d9b855f4b3771127299bcbeb4ad827f603`,
and atomically create one new reservation. Modal retries remain zero.

The remaining budget covers another 1,176-GPU-second maximum reservation and
its 15% planning margin, leaving 5,310.086373914022 GPU-seconds. Thus one
replacement is both scientifically and economically admissible after those
closure gates pass.
