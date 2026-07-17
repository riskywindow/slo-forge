# Helix security and reward-integrity review

Review date: 2026-08-03

Scope: the Python Helix implementation under `python/sloforge/helix`, its existing tests,
and the new standalone controls under `python/sloforge/helix/security`. This was a static,
offline review. It did not access production data, credentials, external services, or paid
resources. Severity describes impact if Helix is used with mutually untrusted tenants or
training candidates; it is not a claim that exploitation has occurred.

## Bounded threat model

Protected assets are production traces and environment state, tenant-private repository and
CAS content, secrets and PII, model/checkpoint integrity, reward and training lineage, hidden
tests, and external systems reachable from a worker.

In-scope adversaries are a malicious submitted repository, generated program, tool result,
reward submission, or lineage document; a caller attempting cross-tenant access or production
capture without consent; and accidental operator misconfiguration. The attacker may control
filenames, symlinks, Git-local metadata, subprocess output, reward values, claimed identifiers,
and duplicate/replayed requests. The attacker is not assumed to have host root, direct write
access to the trusted verifier registry, or the ability to break SHA-256. A compromised kernel,
hypervisor, Python interpreter, package supply chain, or trusted authorization issuer is out of
scope. Denial of service beyond the configured entry, byte, process, and time bounds is also out
of scope.

Trust boundaries reviewed:

1. production/cross-tenant input to environment capture;
2. hostile repository and symlink content to capture or reward execution;
3. tool/verifier output to durable evidence and logs;
4. capsule/checkpoint/artifact bytes to reuse and training;
5. reward, hidden-test, and lineage claims to training/promotion;
6. speculative/training execution to networks, filesystems, credentials, and external effects;
7. artifact expiry and deletion completion.

The new scanner and sanitizer are deliberately bounded: repository scans default to 100,000
entries, 512 MiB total, 64 MiB per file, and depth 128; redaction accepts at most 1 MiB; retained
tool output is at most 1 MiB; reports contain at most 256 unique violations; and the durable
submission registry has an explicit record limit. All time-based validators take `checked_at_ms`
as an input rather than reading a nondeterministic clock.

## Final remediation status

All high-severity findings below were found during implementation and closed before final
acceptance. Capture now performs a tool-free hostile-repository preflight and passively reads
bounded Git metadata; production capture requires a scoped tenant-bound grant; tenant identity is
carried through capture, rewards, batches, registries, and promotion; reward workers require a
pinned evaluator digest and sanitize retained output; environment capsules have tenant-scoped
tombstone receipts and a declared retention policy; coordinated capture publishes only byte-backed
`VerifiedCaptureArtifact` values; and promotion rehashes all eight local gate artifacts in a
tenant-bound trusted capsule. The historical entries are retained to make the threat review and
remediation lineage auditable.

The remaining limitations are lower-level deployment boundaries: CAS byte garbage collection is
conservative, multi-host authorization needs an authenticated registry, filesystem isolation is
not a substitute for a hostile multi-tenant kernel boundary, and SQLite is not distributed
consensus.

## Severity-ranked historical findings

No confirmed critical issue was identified under the stated local-worker assumptions. The
validators label some policy violations critical because consuming a forged artifact or crossing
a tenant boundary must fail closed.

### High

**HELIX-SEC-001 — Host Git is invoked on an untrusted repository before a security preflight.**

`environments/backend.py` runs `git rev-parse`, `git ls-files`, and `git status` against the
captured repository. Local Git configuration can enable executable helpers such as `core.fsmonitor`
or redirect hooks. The command has bounded output and timeout, but its environment does not make
all repository-local execution settings inert. A hostile repository could therefore attempt code
execution in the capture worker outside the Genesis sandbox.

Action: run `scan_repository` before any Git invocation, reject active hooks/config/filter
settings, and either parse metadata without Git or execute Git inside the same strict kernel
sandbox with system/global/local execution features disabled. The new scanner itself never
invokes Git or another tool, but it is not yet wired into `EnvironmentBackend.capture`.

**HELIX-SEC-002 — Production capture authorization is a process-level boolean, not a scoped grant.**

`EnvironmentBackend(allow_production_capture=True)` permits every subsequent request marked
`production=True`. The resulting policy records the configuration boolean but not who approved
the capture, its request scope, validity window, tenant, or a trusted approval artifact. The
experience selector requires consent/redaction claims for production evidence, but it does not
retroactively prove capture authorization.

Action: require `validate_capture_access` with a per-request, tenant-bound, expiring
`ProductionCaptureGrant`, and supply approval digests from an independently trusted registry.

**HELIX-SEC-003 — Tenant scope is strong in environment storage but absent at several composite boundaries.**

The environment CAS, capsule load/restore, parent capture, branch, and effect ledger perform
tenant checks. However, coordinated capture requests/branch points, `RewardRun`, reference
training batches, and policy promotion evidence do not consistently carry a tenant identity.
Application code can therefore accidentally combine otherwise valid artifacts from different
tenants before the environment backend sees them.

Action: apply tenant authorization to every composite input, keep cross-tenant reuse disabled,
and validate the complete manifest/reward graph with `validate_artifact_graph` and
`validate_reward_claim` before training or promotion.

**HELIX-SEC-004 — Reward authority is caller-asserted rather than independently trusted.**

`VerifierCommand` lets the caller supply `argv`, `source_version`, expected return code, and pass/
fail scores. `HiddenCase` similarly accepts a runner path, arguments, expected output, and scores.
The Genesis sandbox strongly limits host impact and fails closed when kernel isolation is
unavailable, but containment does not establish that the evaluator or scoring policy is trusted.
A malicious or mistaken caller can create valid-looking reward evidence for an attacker-chosen
test.

Action: content-address evaluator binaries/specifications, admit only trusted evaluator digests,
bind each component to exact source evidence and trajectory bytes, enforce score bounds, and
require a trusted hidden-boundary digest. `validate_reward_claim` implements these checks as a
preflight layer.

**HELIX-SEC-005 — Reward stdout/stderr excerpts can persist secrets, PII, and terminal controls.**

The sandbox sanitizes inherited environment variables and bounds aggregate output. However,
`DeterministicRewardWorker` copies the last 4,096 characters of verifier stdout and stderr into
`RewardComponentResult` without secret/PII scanning or control-sequence normalization. A verifier
can print sensitive repository content or log/terminal controls.

Action: pass output through `sanitize_tool_output`, retain its zero-residual redaction evidence,
and reject unverified output before it enters reward evidence, reports, logs, or terminals.

**HELIX-SEC-006 — Environment and reward artifacts have no enforceable retention/deletion lifecycle.**

The environment backend persists tenant CAS objects and capsule JSON and can clean branch
workspaces, but it has no artifact expiry, deletion request, replica accounting, bounded legal
hold, tombstone, or deletion receipt. This is material for production capture and privacy
commitments.

Action: attach `RetentionPolicy`/`ArtifactLifecycle` to every retained artifact, schedule expiry,
delete all replicas within a bounded grace period, and verify a tamper-evident deletion receipt
with `validate_retention`.

**HELIX-SEC-007 — Coordinated capture trusts callback-provided environment/effect digests.**

The coordinator verifies the Continuum checkpoint and cross-source watermarks, then binds the
callback-provided `ArtifactWatermark` identifiers/digests into a branch point. It does not retrieve
and hash environment/effect bytes at publication time. A compromised callback can therefore
publish a correctly sealed branch point whose declared non-model digest is not backed by the
referenced bytes.

Action: require sealed `ArtifactManifest` values, verify bytes immediately before atomic
publication, and retain the exact manifests as publication inputs.

### Medium

**HELIX-SEC-008 — Duplicate reward detection is in-memory only.**

`DeterministicRewardWorker._submissions` rejects a repeated reward/trajectory pair only for one
worker object. Restarts, multiple workers, and a new reward ID for identical content bypass it.

Action: use the tenant/namespace-scoped SQLite `SubmissionRegistry`. It rejects both identity
rebinds and content aliases and survives worker restarts; distributed workers still require a
linearizable shared implementation.

**HELIX-SEC-009 — Redaction is useful but incomplete and has no evidence contract.**

Environment capture excludes common secret paths and redacts declared secret values/key names in
structured metadata. It does not scan ordinary captured file bytes for PII, relies on callers to
declare secret values and paths, and does not produce a source/sanitized digest pair with residual
scan counts. Git path metadata and symlink targets can also disclose sensitive names.

Action: apply a tenant-approved classifier before capture, exclude/redact files as units, and
retain only hash-based `RedactionEvidence`. The reference redactor covers declared secrets plus
common email, phone, and payment-card forms; production deployments need a calibrated classifier
and false-positive/false-negative review.

**HELIX-SEC-010 — Filesystem path checks are not race-free against a concurrent local attacker.**

Lexical traversal rejection, symlink-parent checks, non-following directory scans, and destination
checks prevent straightforward escapes. They use path-based check-then-open operations, however,
so a same-host actor able to mutate directories concurrently may race a validated component into
a symlink.

Action: use directory descriptors and no-follow/open-beneath primitives (or an isolated immutable
mount) for hostile, concurrently mutable trees. Continue rejecting special files and unexpected
symlinks.

**HELIX-SEC-011 — Promotion gate evidence is hash-labeled but not fetched or authority-verified.**

Promotion requires named lineage, reward-integrity, quality, safety, serving, and compatibility
gates, checks their arithmetic, and hashes the supplied evidence tuple. The registry does not
retrieve each `artifact_hash` from an immutable store or verify who produced it. A caller that can
create a promotion can supply self-consistent claims.

Action: resolve each gate artifact by digest, verify its producer/tenant/lineage manifest and
authorization, and only then admit it to the promotion transaction.

### Low

**HELIX-SEC-012 — Security claims use SHA-256 integrity seals, not signer authentication.**

Content hashes reliably detect accidental or post-publication changes when a trusted digest is
already known. They do not identify the issuer. The new models therefore require callers to pass
trusted approval, evaluator, hidden-boundary, or legal-hold digest sets rather than treating a
self-hash as authority.

Action: back trusted digest sets with an authenticated registry or signed transparency log in a
multi-host deployment. Key management and signature infrastructure are outside this offline
package.

## Existing controls worth preserving

- Genesis hostile-code execution requires enforced network and filesystem isolation, sanitizes
  the environment, avoids a shell, bounds output/time/processes/artifacts, and fails closed when
  the kernel sandbox is unavailable.
- Environment content is tenant-namespaced, content-addressed, size-verified, and rehashed on read.
- Environment capture and restore reject traversal, external symlink targets, special entries,
  symlink destinations, and cross-tenant parents/restores/branches.
- The effect ledger performs no external operation itself, rejects real writes by default,
  requires tenant match and write evidence, and forbids irreversible/unknown speculative effects.
- Helix IR models are strict and immutable; `LearningTransaction` cross-checks embedded trajectory,
  reward, credit, sample, policy, and staleness digests. The reference dataset also rejects missing
  rewards, duplicate trajectories/rewards, holdout overlap, and mixed behavior policy epochs.
- Capture journals, branch points, environment capsules, training batches, checkpoints, promotion
  evidence, and policy records use deterministic content identities at important boundaries.

## New standalone protection package

`sloforge.helix.security` provides independently callable, fail-closed controls without changing
existing Helix call sites:

- scoped production grants, tenant authorization, and cross-tenant reuse rejection;
- hash-only secret/PII redaction evidence and hostile tool-output normalization;
- bounded retention, legal-hold, deletion, replica, and receipt validation;
- capsule/checkpoint/artifact byte manifests and acyclic exact-digest lineage validation;
- trusted-evaluator reward claims, score bounds, hidden-test boundary attestations, and source
  evidence verification;
- a tool-free malicious repository scanner for symlinks, special files, active Git hooks/config,
  filters, external submodules, path controls, and resource limits;
- enforced execution isolation validation for network, filesystem, environment, write namespace,
  and external effects; and
- a durable, tenant-scoped duplicate-submission registry.

These controls are wired into the local reference capture, reward, dataset, and promotion paths,
which treat violations as terminal. External adapters must provide equivalent enforcement before
they may claim the same trust class.
