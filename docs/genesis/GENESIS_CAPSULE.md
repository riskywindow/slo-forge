# GenesisCapsule

A `GenesisCapsule` is the immutable, hash-addressed handoff between untrusted synthesis and the
independent validation and rollout path. It carries generated artifacts together with narrowly
scoped claims and the evidence needed to check those claims. A capsule is not a universal proof, a
signature, or permission to send live traffic.

The canonical Python types are in
[`python/sloforge/genesis/capsule/models.py`](../../python/sloforge/genesis/capsule/models.py), and
the version 1 public schema is
[`schemas/genesis_capsule/genesis-capsule-v1.schema.json`](../../schemas/genesis_capsule/genesis-capsule-v1.schema.json).
All core values are strict and immutable; unknown fields are rejected.

## Manifest structure

Identity binds the candidate genome, reference model, tokenizer, workload and hardware contracts,
compiler/verifier versions, Git commit, dependency lock, generation time, and optional parent
capsule. Compatibility separately names exact allowed hardware fingerprints, architectures, minimum
device count, dependency versions/digests, and restrictions.

Each artifact has a stable identifier, role, origin, normalized relative path, media type,
executable bit, byte size, and SHA-256 digest. Roles distinguish generated source/runtime/policy/
kernel/deployment/state-conversion artifacts from rollback, evidence, raw benchmark samples,
dependency locks, model-check results, and the counterexample corpus. Origins distinguish trusted,
generated-untrusted, external-runtime, verified-evidence, performance-evidence, and formal-or-bounded
material. An origin is provenance metadata, not an authorization bit.

Each evidence record declares its class, independent issuer, issuer version, result, verification
level, immutable artifact references, observation/expiry time, seed, and assumptions. Each claim
declares a category, statement, result, level, evidence references, and a scope containing input,
shape, dtype, exact hardware fingerprints, dependencies, assumptions, and exclusions. Unsupported
cases and unverified assumptions are first-class manifest fields.

Benchmark records bind their definition, candidate and baseline raw samples, software manifest,
workload fingerprint, hardware fingerprint, warmup, repetitions, randomized ordering, noise floor,
confidence interval, effect size, regression probability, and practical threshold. The validator
recomputes the declared median and tail statistic from finite raw samples. It does not trust a
summary merely because the manifest was re-sealed.

## Canonicalization and addressing

Canonical JSON is UTF-8, sorted by key, compact, and rejects non-finite numbers. The capsule digest
is SHA-256 over the complete manifest with `capsule_digest` set to JSON `null`; sealing then inserts
that digest. Publication uses the digest as the filename and refuses to replace different content.
Artifacts are separately content-addressed and materialized through non-symlink, publish-once paths.

SHA-256 detects content changes only when the expected digest is pinned independently. It does not
authenticate the publisher. The current implementation is hash-addressed, not cryptographically
signed. Production key distribution and signature verification remain outside the implemented
scope.

## Independent validation

Validation receives the capsule, its artifact root, and a freshly constructed `ValidationContext`.
The validator never imports or executes generated code. It accumulates deterministic issues for:

- an unsealed or changed manifest;
- missing, symlinked, escaping, size-mismatched, or digest-mismatched artifacts;
- absent, failed, future-dated, expired, incorrectly issued, role-incompatible, or insufficient-
  level evidence;
- promotion claims whose complete statement and scope are not externally anchored, claims outside
  their hardware scope, or claims unsupported by the matching evidence class;
- source model, tokenizer, workload contract, hardware contract, hardware fingerprint,
  architecture, device count, verifier, dependency lock, or dependency mismatch;
- benchmark provenance or recomputed-statistic mismatch; and
- missing generated runtime, deployment, rollback, semantic, quality, resource, performance,
  operational, or counterexample-corpus material required by the local promotion-completeness gate.

A counterexample corpus may contain zero found failures only when it declares the domains searched.
Omitting the corpus is rejected. A complete capsule can still state negative results, exclusions,
and unsupported cases without inflating their verification level.

## H6 promotion-attack campaign

The dedicated H6 campaign in
[`capsule_attacks.py`](../../python/sloforge/genesis/evaluation_campaigns/capsule_attacks.py)
exercises the production `validate_capsule` entry point, rather than a campaign-local acceptance
predicate. For each explicit seed it builds a strict, promotion-complete capsule conformance
vector, verifies that the unmodified vector is eligible under its trusted validation context, and
then evaluates these ten attacks:

| Attack | Mutation | Expected fail-closed control |
| --- | --- | --- |
| Modified runtime artifact | Changes generated-runtime bytes after sealing | Artifact size/digest integrity |
| Hardware fingerprint mismatch | Supplies a current fingerprint outside the declared scope | Hardware and claim-scope compatibility |
| Dependency version mismatch | Supplies a different installed runtime version | Dependency compatibility |
| Stale evidence | Advances trusted validation time past the evidence horizon | Evidence freshness |
| Incomplete evidence | Removes quality evidence while retaining its required claim | Evidence and required-class completeness |
| Altered benchmark summary | Re-seals a median not derivable from the raw samples | Independent benchmark recomputation |
| Altered quality evidence | Changes replayable expected/observed cases without their anchored digest | Artifact integrity and evidence completeness |
| Missing counterexample corpus | Removes the required corpus from a re-sealed manifest | Counterexample-corpus completeness |
| Invalid model-check scope | Moves the operational claim outside the validating hardware fingerprint | Claim-scope compatibility and external whole-claim anchor |
| Incompatible state migration | Replaces the anchored state-conversion bytes with an incompatible source genome | State-conversion artifact integrity |

The default matrix is three seeds by ten attack classes. Every mutation record preserves its seed,
description, changed manifest/context paths, changed-file before/after hashes, expected issue codes,
and the complete validator report. The campaign validator reopens the stored capsule, context, and
reports, revalidates them, reconstructs the original conformance vector in a new temporary tree,
reapplies each mutation, and requires the rebuilt capsule, context, mutation record, issue set, and
validation report to match. Changing a supporting campaign artifact is itself rejected by its
recorded SHA-256 digest.

Some re-sealed attacks deliberately substitute the re-sealed digest into the supplied context.
That exercises evidence anchors and semantic completeness after manifest hashing has passed; it
does not weaken the operational requirement that the deployment controller independently pin the
expected capsule digest.

Promotion-required evidence anchors bind issuer records and artifact digests. Separate trusted
claim anchors bind the canonical digest of the complete claim, including its statement, evidence
references, verification level, assumptions, exclusions, and input/shape/dtype/hardware scope.
Consequently, re-sealing a narrower or broader claim scope cannot authorize a claim that the
external validation authority did not approve.

The campaign has an exact, narrow proof scope. Its promotion-complete capsule is a validator
conformance fixture. Its benchmark numbers are deterministic statistical test vectors, not
measurements. `hardware_backed_runs` and `gpu_hours` are both zero, and no deployment or performance
claim is made. The invalid-model-check-scope case checks whether a scoped operational claim contains
the current hardware; it does not establish model-checker soundness or replay a transition system.
The state-migration case proves that an anchored conversion artifact cannot be changed unnoticed;
the capsule validator does not independently execute that conversion or prove semantic state
compatibility.

Run the campaign and its artifact-replay checks with:

```bash
PYTHONDONTWRITEBYTECODE=1 uv run pytest -q \
  tests/python/test_genesis_capsule_attack_campaign.py
```

## Verification levels and promotion meaning

Claims independently use levels 0 through 5: build, differential, property, bounded exhaustive,
solver-backed, and hardware/operational. Level ordering does not merge different properties: a
Level 4 shape certificate does not establish Level 4 numerical equivalence, and a Level 5 shadow
result does not establish universal semantic correctness. See
[`ADR 0035`](../adr/0035-proof-scope-terminology.md).

`promotion_eligible=true` means that this validator found no integrity, compatibility, evidence-
completeness, or local gate issue under the supplied context. It does not itself mutate a
deployment and does not imply Level 5 hardware evidence. The local capsule builder deliberately
emits `hardware_backed=false` and scopes performance to deterministic service-model simulation.
External shadow/canary/promotion additionally requires appropriate environment-specific evidence,
controller gates, immediate revalidation, transition compatibility, and explicit live authorization.

## Build and validate

Build from an accepted, property-tested candidate into a new empty output directory:

```bash
sloforge genesis capsule build \
  --candidate artifacts/genesis/run-001/candidates/candidate-id \
  --output artifacts/genesis/capsules/candidate-id \
  --timestamp 2026-08-02T00:00:00Z
```

Validate the directory or its digest-named manifest:

```bash
sloforge genesis capsule validate artifacts/genesis/capsules/candidate-id
```

The builder copies persisted evidence, seals the manifest, invokes the independent validator, and
publishes only if validation succeeds. Reusing the original timestamp and byte-identical inputs
reproduces the content-addressed payload. Validation refreshes current time and the repository lock;
an optional hardware contract can narrow compatibility, but the current CLI still reports
`hardware_backed=false` because it does not perform a hardware benchmark.

Exercise schema, tampering, stale-evidence, compatibility, benchmark-provenance, symlink, and
immutable-publication behavior with:

```bash
PYTHONDONTWRITEBYTECODE=1 uv run pytest -q tests/python/test_genesis_capsule.py
```

The tests use explicit fixture evidence. They validate the gate mechanics, not a real production
deployment or GPU result.

## Compatibility

Version 1 core fields are closed. Additive experimental data belongs in a future explicitly typed
extension or schema revision; incompatible semantics require a new major version and migration.
Consumers must reject unsupported major versions rather than silently dropping proof obligations.
The capsule is the stable trust-boundary projection of a candidate, not an alias for mutable search
state.
