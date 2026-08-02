# Genesis security and operational safety

Genesis assumes generated source, policies, kernels, transformations, binaries, deployment files,
and synthesis-agent output can be malicious. Compilation and benchmark success do not grant trust.
The security objective is to keep proposal, execution, evidence production, capsule validation, and
promotion authority separate and to fail closed when an isolation or evidence requirement is absent.

The detailed boundary is in [`TRUST_MODEL.md`](TRUST_MODEL.md); capsule rules are in
[`GENESIS_CAPSULE.md`](GENESIS_CAPSULE.md).

## Assets and adversaries

Protected assets include reference models and contracts, tokenizer/weight inputs, verifier and
acceptance code, baseline and raw evidence, capsule signing/addressing state, lineage records,
deployment credentials, active-stream state, the current champion, and its rollback artifact.

Threats include malicious generated code, accidental unsafe code, compromised dependencies,
filesystem or credential discovery, network exfiltration, subprocess/fork abuse, resource and output
exhaustion, dynamic loading, capsule replacement, stale evidence, hardware or model mismatch,
benchmark manipulation, lineage poisoning, and unsafe promotion. The project does not assume the
synthesis process is honest or competent.

## Generated-code execution

The executor accepts an argv vector and never invokes a shell. Inputs are explicit, non-symlink,
read-only paths; output is a separate explicit artifact directory. The working directory must lie
inside a declared input. The environment is rebuilt from an allowlist, credential-shaped variable
names are rejected, `HOME` and `TMPDIR` are isolated under the output directory, user site packages
are disabled, stdin is closed, and `PYTHONHASHSEED` is explicit.

Wall time, CPU time, open files, output bytes, artifact bytes, artifact entries, and process cleanup
are bounded. Address-space and Linux process-count limits are best effort without cgroups. Output is
drained incrementally, and timeout/output-limit termination kills the process group. Accepted output
trees contain only bounded regular files; symlinks and device/special files are rejected.

Network and filesystem isolation require a supported OS backend. The macOS backend uses
`sandbox-exec`; it was exercised in this workspace for network denial, protected-root read denial,
write confinement, child-process denial, credential sanitization, timeout cleanup, output flooding,
and symlink rejection. It is deprecated platform machinery, does not create a device namespace, and
is not a VM boundary. System-readable files outside protected roots may remain readable. The Linux
bubblewrap adapter unshares namespaces and provides a private `/dev`, but it was not executed on
Linux in this workspace. If the selected host lacks a supported/working backend, strict execution
returns `policy_unavailable` without running generated code. Windows strict execution is not
implemented.

Kernel Lab adds an AST allowlist before sandbox execution, rejecting imports, dynamic loading,
filesystem/process access, arbitrary attributes, and unexpected functions for that restricted
generated-kernel language. The general OS sandbox alone does not prove that a process cannot load a
library from an otherwise readable declared input. Generated build inputs therefore remain
untrusted, and their outputs require independent verification. Network denial and write confinement
prevent ordinary online package installation; the system does not claim to defeat a malicious
library already present in readable runtime or declared input paths.

GPU or host-device access is not intentionally granted by default. The current Triton adapter
reports capability and opt-in status only. `SLOFORGE_GENESIS_ALLOW_GPU=1` cannot turn CPU evidence
into GPU evidence; a separately isolated hardware harness and new evidence are required.

## Integrity and evidence controls

The content-addressed artifact store streams into a bounded temporary object, hashes before atomic
publication, makes the object read-only, verifies collisions, rejects symlinked store paths, and
rehashes on reads/materialization. Capsule manifests bind identity, dependencies, compatibility,
artifacts, evidence, claims, raw benchmark provenance, unsupported cases, and assumptions.

The independent validator rejects changed manifests/artifacts, stale or future evidence, unsupported
issuers, insufficient verification levels, mismatched contracts/hardware/dependencies/verifier,
unsafe paths, incomplete evidence, altered benchmark statistics, and absent counterexample corpora.
Hash addressing is not publisher authentication: deployment control must pin the expected digest,
and signatures are not yet implemented.

Benchmark-integrity attacks are independently searched by the executable red team, including
missing synchronization, timer misuse, warmup/input/cache differences, hidden fallback, precision or
quality mismatch, omitted failures, clock/affinity/background-process changes, discarded samples,
and definition mismatch. See
[`docs/redteam/BENCHMARK_INTEGRITY.md`](../redteam/BENCHMARK_INTEGRITY.md).

Lineage is proposal input, never proof authority. Transferred transformations and learned constraints
must be rechecked in the new model, workload, hardware, and dependency context. Dependency
invalidation marks affected evidence stale instead of silently reusing it.

## Promotion safety

Generated output cannot edit verifier code, benchmark acceptance logic, the capsule validator,
promotion gate, or immutable raw evidence through the sandbox output path. Challengers remain
isolated and must pass capsule validation before shadowing. Shadow and canary gates require bounded
sample counts, error/latency/quality criteria, and zero interrupted streams. The capsule is validated
again immediately before promotion.

Existing streams remain pinned to the capsule on which they began for policy-only and
request-boundary-safe transitions; new requests use the promoted champion. Other transitions drain
unless independently declared active-stream-compatible. State conversion requires evidence,
operator-required transitions cannot promote autonomously, and the old champion remains retained for
rollback. Persisted controller state is content-checked and restartable.

External live canarying/promotion requires both controller authorization and
`SLOFORGE_GENESIS_ALLOW_LIVE_PROMOTION=1`. Paid GPU, external synthesis, multi-node, privileged
probe, external deployment, and live-promotion controls are separate false-by-default authorities.
No capsule grants them. Budget variables are hard ceilings, not opt-ins.

## Evidence status

| Path | Status | Security claim allowed |
| --- | --- | --- |
| Canonical capsule parsing, hashing, validation, and tamper tests | Implemented and exercised locally | Integrity and compatibility behavior for tested fixtures |
| macOS generated-code sandbox | Implemented and exercised locally | Tested denial/confinement and bounded-process behavior on the recorded Darwin host |
| Linux bubblewrap sandbox | Implemented but unexercised here | No Linux isolation result |
| Local champion/challenger controller | Implemented and exercised with deterministic local/simulated observations | State-machine and gate behavior for tested cases |
| Deterministic service-model performance evidence | Implemented and exercised | Synthetic prediction evidence only |
| Local CPU Kernel Lab | Implemented and exercised | CPU-scoped correctness/timing when raw artifacts are retained |
| CUDA/Triton GPU execution and multi-node isolation | Not exercised | No hardware/GPU security or performance claim |
| External live promotion | Guarded but unexercised | No production safety claim |

## Verification commands

```bash
PYTHONDONTWRITEBYTECODE=1 uv run pytest -q \
  tests/python/test_genesis_sandbox.py \
  tests/python/test_genesis_artifacts.py \
  tests/python/test_genesis_capsule.py \
  tests/python/test_genesis_evolution.py \
  tests/python/test_genesis_redteam.py
```

Passing these tests does not establish kernel-level sandbox correctness, protection against a
compromised Python interpreter/Pydantic/OS, real GPU isolation, external deployment safety, or
universal correctness of generated code.

## Response to a failed control

On any sandbox, integrity, compatibility, evidence, shadow, or canary failure: stop the candidate,
kill its process group, retain raw stdout/stderr and artifacts within configured limits, record the
typed failure/counterexample, reject or roll back the challenger, keep the prior champion, invalidate
dependent evidence where applicable, and require fresh verification. Never weaken a policy or
silently select a different engine/device to make a candidate pass.
