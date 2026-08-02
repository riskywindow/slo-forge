# SLOForge Genesis security review report

Review date: 2026-08-02  
Platform exercised: macOS 15.6.1, Darwin 24.6.0  
Baseline commit inspected: `2db12849b1003b7a1fcd3f5c5d95acad5c6bdcec`

## Outcome

The reviewed Genesis trust boundary now prevents the two most consequential
self-authentication failures: a capsule cannot establish its own expected
identity, and promotion-required evidence cannot establish its own issuer trust.
The operator must supply the expected capsule digest and typed trusted-evidence
anchors outside the capsule. The validator binds each promotion claim to the
canonical evidence-record digest, issuer and version, and exact artifact
digests.

The zero-day frontend no longer imports an untrusted reference package in the
orchestration process. Import and `torch.export` occur in the restricted Genesis
sandbox worker. Evolution observations are bound to their candidate, capsule,
evidence, and controller seed; replay is rejected; external promotion requires
level-5 operational evidence, trusted gate validation, and the existing explicit
live-promotion opt-in.

## Trusted boundary changes

- External capsule digest and evidence anchors are mandatory validation inputs.
- Capsule-local validation contexts are rejected by the CLI.
- Candidate-genome identity and the minimum promotion evidence level are carried
  in validation reports and checked by evolution.
- Resealed evidence with an allowed but untrusted issuer is rejected.
- Hostile Torch reference code is dispatched to a no-network, read-only-input,
  bounded sandbox request.
- Capsule and evidence files use exclusive creation and immutable modes; capsule
  roots and intermediate components reject symlinks.
- Evolution persistence uses random same-directory atomic temporary files,
  restrictive permissions, and non-regular/symlink rejection.
- Generated artifact directories must be dedicated and empty on the sandbox,
  red-team, and kernel-lab paths reviewed.

## Validation evidence

Focused trust tests:

```text
47 passed, 1 skipped
```

Sandbox tests:

```text
11 passed
```

Static validation of touched files:

```text
ruff check: passed
ruff format --check: passed
mypy: Success: no issues found in 20 source files
```

The skip is the actual Torch export integration because PyTorch is not installed
in the exercised environment. A hostile-reference unit test confirms that the
trusted orchestrator does not import the target and that it constructs a strict
sandbox request.

A wider selected suite reported 77 passes, one Torch skip, and one unrelated
concurrent failure in a kernel-demo expectation after that workstream began
accepting two kernel claims. This report does not classify the wider suite as
green and does not use those kernel results as security or performance evidence.

## Adversarial evidence

- Wrong operator-supplied capsule digest: rejected as manifest tampering.
- Internally consistent, resealed evidence with a different allowed issuer:
  rejected as untrusted evidence.
- Capsule-bundled context used as trust root: rejected by the CLI.
- Gate observation for another candidate/capsule/seed: rejected.
- External gate without trusted evidence validation: rejected.
- External capsule below verification level 5: rejected.
- Symlinked capsule, output, demo, evaluation, or evolution-store paths: rejected
  by the relevant regression checks.
- Undeclared macOS host-file access: denied by the exercised sandbox policy.

## Open risks

1. **High — in-process red-team callbacks.** The generic red-team runner invokes
   target callbacks in-process. Only trusted fixture callbacks are safe. Any
   generated-candidate adapter must invoke the callback through the Genesis
   sandbox with a hard timeout.
2. **Medium — direct runtime adapter use.** The runtime adapter imports source
   in-process. Existing generated-runtime callers sandbox that work, but the API
   still needs a sandbox-only public entry point or trusted-source guard.
3. **Medium — validation/execution TOCTOU.** Promotion revalidates immediately,
   but production execution should consume a digest-pinned CAS materialization
   or rehash the exact executable at launch.
4. **Medium — macOS memory isolation.** The exercised backend cannot impose a
   hard address-space limit. Memory-hostile generated code requires an outer
   container or VM boundary.
5. **Medium — operational trust distribution.** The expected digest, trusted
   evidence anchors, and external gate validator are trusted inputs. Copying a
   capsule's suggested context without independent verification would defeat the
   intended boundary.

## Validation scope

Implemented and exercised locally:

- macOS sandbox filesystem/network/process/output/timeout controls;
- capsule hashing, external trust anchoring, tamper and forgery rejection;
- local evolution evidence binding and external-promotion gating logic;
- symlink and preexisting-output defenses;
- deterministic red-team fixture generation, replay, and minimization.

Implemented but not exercised here:

- actual `torch.export` execution in the frontend worker;
- Linux bubblewrap sandbox behavior.

Not claimed by this report:

- GPU or multi-GPU execution;
- hardware-backed performance evidence;
- live production traffic or external deployment;
- cloud or paid synthesis execution;
- universal correctness or universal sandbox escape resistance.

The detailed finding matrix and attack scenarios are in
`docs/genesis/RED_TEAM_REVIEW.md`.
