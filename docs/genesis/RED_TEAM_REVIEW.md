# Genesis adversarial security review

Date: 2026-08-02

This review treats generated models, policies, kernels, runtimes, evidence, and
capsules as hostile. It covers the Genesis Python trust boundary, capsule and
promotion gates, generated-code sandboxing, artifact output handling, and the
executable red-team path. The reviewed checkout was based on Git commit
`2db12849b1003b7a1fcd3f5c5d95acad5c6bdcec` with the Genesis implementation in
the working tree.

## Threat model

An attacker may control a reference model package, generated source, generated
artifacts, candidate metadata, capsule contents, benchmark results, or lineage
records. The attacker may attempt to execute code in the orchestration process,
read undeclared host files or credentials, use the network, create child
processes, overwrite trusted evidence, replace artifacts through symlinks,
forge evidence provenance, replay gate observations, or promote a candidate
whose evidence applies to another genome.

The security boundary assumes the operator supplies the expected capsule digest
and trusted evidence anchors through a channel outside the untrusted capsule.
It also assumes that the external gate-evidence validator is part of the trusted
computing base and validates evidence from an operator-controlled store.

## Findings

| ID | Severity | Status | Finding and disposition |
| --- | --- | --- | --- |
| GEN-SEC-001 | Critical | Fixed | A capsule could self-assert a trusted issuer. Promotion-required evidence is now checked against externally supplied typed evidence anchors covering the evidence-record digest, issuer identity and version, and the exact artifact digest set. A resealed forged record is rejected. |
| GEN-SEC-002 | Critical | Fixed | Capsule identity could be accepted using only capsule-local metadata. Validation now requires an externally supplied expected capsule digest. The CLI requires an external validation context and rejects a context located inside the capsule root. |
| GEN-SEC-003 | High | Fixed | The zero-day Torch frontend imported the target module in the trusted orchestrator. Target import and `torch.export` now run only in the restricted sandbox worker. Sandbox failure is fail-closed and untrusted stderr is not reflected into the exception. |
| GEN-SEC-004 | High | Fixed | Evolution gate observations were not cryptographically bound to a candidate, capsule, or evidence record. Observations now bind all three digests plus the deterministic controller seed, reject evidence-digest replay, and pass through a trusted external validator for external deployments. |
| GEN-SEC-005 | High | Fixed | External promotion did not require operational evidence. External champion, challenger, rollback, and gate validation now require verification level 5 (`HARDWARE_OPERATIONAL`), in addition to the existing live-promotion opt-in. |
| GEN-SEC-006 | High | Fixed | Promotion could accept a validation report for another genome. Capsule reports now expose the candidate-genome hash and the controller compares it with the challenger specification. |
| GEN-SEC-007 | High | Fixed | The macOS sandbox allowed undeclared system reads. Its generated policy now restricts reads to declared roots and the executable/runtime roots needed to start the worker; a regression test attempts an undeclared host read. |
| GEN-SEC-008 | Medium | Fixed | Capsule artifacts and resettable demo outputs were exposed to symlink replacement or mutable overwrite. Capsule construction uses exclusive creation, synchronization, and read-only modes; validation rejects root and intermediate symlinks; affected demo and evaluation resets reject symlinks. |
| GEN-SEC-009 | Medium | Fixed | Sandbox, kernel-lab, and red-team artifact directories could contain preexisting output. They now require an empty dedicated directory, limiting evidence substitution and accidental overwrite. |
| GEN-SEC-010 | Medium | Fixed | Evolution persistence used a predictable temporary name and did not consistently reject symlinked storage. It now uses a same-directory random temporary file, atomic replacement, restrictive mode, and symlink/non-regular-file checks while preserving compare-and-swap behavior. |
| GEN-SEC-011 | High | Open | `redteam.runner` accepts Python callbacks and evaluates them in-process. The bundled deterministic fixture is trusted, but an adapter that passes generated candidate code directly would bypass sandbox limits and per-evaluation hard timeouts. Generated or externally sourced targets must be invoked through the Genesis sandbox before this API is used on hostile code. |
| GEN-SEC-012 | Medium | Open | `runtime.adapter` imports a source module in-process. Current generated-runtime differential and kernel paths place this operation inside a sandbox, but direct library use is unsafe for hostile source. A sandbox-only public entry point or an explicit trusted-source guard remains necessary. |
| GEN-SEC-013 | Medium | Open | Validation and execution are separate filesystem operations. The evolution controller revalidates immediately before promotion, but a production deployer should execute a digest-pinned content-addressed materialization or rehash the exact material at process launch to remove the remaining time-of-check/time-of-use window. |
| GEN-SEC-014 | Medium | Platform limitation | The exercised macOS backend enforces filesystem, network, process, wall-time, CPU-time, file-descriptor, output, and child cleanup controls. A hard address-space limit is not available through this backend, so memory-hostile workloads require an outer VM/container boundary. |
| GEN-SEC-015 | Medium | Unexercised | The Linux bubblewrap backend was not executed on this Darwin host. Its presence is not counted as validated isolation. |
| GEN-SEC-016 | Medium | Unexercised | The Torch export worker was unit-tested for trusted-orchestrator isolation, but the actual `torch.export` path was skipped because PyTorch is absent from the local environment. |

## Adversarial scenarios exercised

- A reference module that raises immediately when imported was submitted to the
  Torch frontend. The trusted process did not import it; the request reached the
  sandbox boundary with network disabled, a read-only package root, a dedicated
  output directory, and bounded CPU, wall time, processes, file descriptors,
  and output.
- A valid capsule was resealed after replacing a semantic evidence record with a
  record claiming another allowed issuer. The internal hashes were consistent,
  but the independently supplied evidence anchor did not match, so validation
  returned `EVIDENCE_UNTRUSTED`.
- A capsule was validated against the wrong operator-supplied digest. Validation
  returned `MANIFEST_TAMPERED`.
- A capsule attempted to use its own bundled context as the trust root. The CLI
  rejected the context because it resolved inside the capsule directory.
- Gate observations with the wrong candidate, capsule digest, or controller seed
  were rejected. External gates without trusted validation, and external
  capsules below level 5, were rejected.
- Capsule, demo, evaluation, persistence, sandbox, red-team, and kernel output
  paths were exercised against symlinks or nonempty directories by regression
  tests.
- The deterministic red-team fixture produced executable, replayable findings
  and minimized counterexamples. This validates the fixture pipeline, not the
  safety of arbitrary in-process callback targets described in GEN-SEC-011.

## Review commands and results

The focused trust-boundary suite completed with 47 passing tests and one skip:

```text
uv run pytest -q tests/python/test_genesis_capsule.py \
  tests/python/test_genesis_evolution.py \
  tests/python/test_genesis_frontend.py \
  tests/python/test_genesis_capsule_builder.py
47 passed, 1 skipped
```

The sandbox suite completed with 11 passing tests. The broader reviewed surface
completed with 77 passing tests, one Torch-related skip, and one concurrent
kernel-demo assertion failure: the test expected zero accepted kernel speedup
claims while another workstream had begun emitting two accepted claims. That
failure was outside this review's trust-boundary changes and is not reported as
passing here.

Static checks for the touched trust-boundary files completed successfully:

```text
ruff check: passed
ruff format --check: passed
mypy: Success: no issues found in 20 source files
```

Repository scans found no direct network-client imports in the reviewed Genesis,
red-team, and Genesis CLI Python surfaces. The only ordinary subprocess call
outside the sandbox executor was the capsule builder's bounded, argument-vector
`git rev-parse` query. No `shell=True`, `os.system`, `eval`, or `exec` use was
found in that scope.

## Scope and residual-risk statement

This is bounded implementation evidence, not a proof of universal isolation.
The review did not exercise Linux sandboxing, GPUs, live production traffic,
external deployment, cloud credentials, or paid synthesis services. The open
in-process callback/import hazards must not receive generated or externally
sourced code until routed through the sandbox. Operator-controlled digest and
evidence-anchor distribution, external gate validation, and deployment-time
artifact pinning remain part of the trusted operational procedure.
