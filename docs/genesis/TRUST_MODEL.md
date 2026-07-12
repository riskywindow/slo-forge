# Genesis trust model

## Security objective

Genesis treats generated source, policies, transformations, binaries, and agent output as
hostile. A candidate is never trusted because it compiled or because the component that proposed
it reported success. The independent capsule validator admits only content-addressed artifacts
whose claims have scoped evidence, and generated programs run through a capability-checked local
sandbox. Promotion is a separate decision from synthesis.

This is a scoped assurance model. A passing bounded check is evidence only within its recorded
bounds; a differential test covers only its declared inputs; hardware measurements apply only to
their exact hardware, workload, dependency lock, and benchmark definition. Genesis does not emit a
universal `verified` bit.

## Asset classes

Capsules distinguish the following origins in their canonical manifest:

- `trusted`: parsers, canonical serializers, validators, bounded checkers, benchmark acceptance,
  promotion control, and rollback control;
- `generated_untrusted`: generated source, policy, kernel, runtime, binary, deployment, or state
  conversion output;
- `external_runtime`: model frameworks, compiler/runtime libraries, and other sourced code not
  maintained inside the trusted boundary;
- `verified_evidence`: immutable differential, quality, resource, operational, property, or fuzz
  results accepted by a named independent issuer;
- `performance_evidence`: raw samples and their workload, software, and hardware provenance;
- `formal_or_bounded_evidence`: solver certificates or explicit-state results, including their
  bounds and assumptions.

An evidence artifact is not executable authority. A generated artifact cannot identify itself as
trusted, and generated code is never imported by the capsule validator.

## Trusted computing base

The Python trust lane consists of:

- `genesis/capsule/models.py`: strict immutable parsers and scoped claim types;
- `genesis/capsule/canonical.py`: deterministic JSON and SHA-256 manifest sealing;
- `genesis/capsule/io.py`: bounded parsing and publish-once digest-named manifests;
- `genesis/capsule/validator.py`: independent integrity, provenance, compatibility, freshness, and
  promotion checks;
- `genesis/capsule/builder.py`: local evidence issuance, reference-package recomputation, bounded
  proof recomputation, and fresh final-corpus sandbox replay before sealing;
- `genesis/artifacts/store.py`: atomic immutable SHA-256 object publication;
- `genesis/sandbox/models.py` and `genesis/sandbox/executor.py`: policy construction, sanitized
  process setup, resource limits, bounded output, timeout, and process-group cleanup;
- the bounded synthesis checkers, evolution evidence/controller, SynthBench runner/integrity
  validator, and kernel correctness/benchmark acceptance modules named by the command below;
- `schemas/genesis_capsule/genesis-capsule-v1.schema.json`: the public wire schema.

On 2026-08-02 the narrow capsule/artifact/sandbox subset measured 3,960 physical lines. The
conservative local evidence and promotion envelope measured 8,830 lines. Reproduce the conservative
count from the repository root with:

```bash
wc -l \
  python/sloforge/genesis/capsule/{models,canonical,io,validator,builder}.py \
  python/sloforge/genesis/artifacts/store.py \
  python/sloforge/genesis/sandbox/{models,executor}.py \
  python/sloforge/genesis/synthesis/{local,fixture}.py \
  python/sloforge/genesis/evolution/{evidence,controller}.py \
  python/sloforge/synthbench/{runner,runtime_runner,integrity}.py \
  python/sloforge/genesis/kernel_lab/{executor,benchmark}.py
rg '^import |^from ' python/sloforge/genesis/{capsule,artifacts,sandbox} -g '*.py' \
  | sed -E 's/^.*:(from|import) ([A-Za-z0-9_.]+).*$/\2/' \
  | cut -d. -f1 | sort -u
```

The source count excludes the JSON Schema, Python interpreter, operating-system kernel,
`sandbox-exec`/bubblewrap implementation, and transitive dependencies. The only direct
non-standard import in the narrow Python TCB is Pydantic; internal `sloforge` imports and Python
standard-library modules appear in the dependency command. The sandbox additionally depends on
the OS process/resource APIs and either macOS `sandbox-exec` or Linux `bubblewrap`. Hashing uses
the standard-library SHA-256 implementation. The local capsule builder is inside this conservative
envelope because it issues trusted evidence anchors; its output must also pass the separately
structured capsule validator. Candidate proposal, search scoring, generated runtimes, PyTorch,
Triton, external coding agents, and their reasoning remain outside this TCB.

TCB size is reported as an approximate auditable surface, not as a proof of absence of bugs.
Generated line counts, vendored dependency lines, the OS kernel, Python interpreter, and crypto
implementation are reported separately rather than hidden in the source-line number.

## Capsule admission

`GenesisCapsule` is strict and rejects unknown fields. Its canonical digest covers the complete
manifest with the digest slot set to JSON `null`, avoiding a self-referential hash. Artifacts use
safe relative paths and include role, origin, SHA-256 digest, size, media type, and executable bit.
The validator rejects missing files, symlinks, path escape, size changes, byte changes, and a
changed manifest.

Every claim declares a category, statement, input/shape/dtype/hardware/dependency scope,
assumptions, exclusions, result, and one of six independent verification levels. Claims refer to
immutable evidence IDs. The validator checks that evidence:

- exists and resolves to integrity-checked files;
- was produced by an issuer permitted for its evidence class;
- has a compatible artifact role and sufficient level;
- passed, is not future-dated, and has not expired;
- matches the claim category and current hardware scope.

For every promotion-required claim, the external validation context also pins a canonical digest
of the complete claim. This prevents a re-sealed capsule from broadening an evidence-backed scope,
changing assumptions or exclusions, or redirecting evidence references while retaining valid
artifact and evidence-record anchors. The context is control-plane authority and must be constructed
outside the untrusted capsule directory.

Promotion additionally requires generated runtime, deployment, rollback, semantic, quality,
resource, performance, and operational evidence; a counterexample corpus; and at least one
provenance-complete benchmark. The current model, tokenizer, workload contract, hardware contract,
hardware fingerprint and architecture, verifier version, exact dependency lock, and dependency
versions must match. An empty counterexample corpus is legal only when it declares the domains
searched. Omitting the corpus is not legal.

Raw benchmark evidence binds samples to the benchmark definition, software manifest, workload,
and hardware digests. The gate checks sample and repetition counts, randomized ordering, and
recomputes the reported median and tail statistic from finite raw samples. It rejects a summary
that was altered and then re-sealed.

SHA-256 gives content integrity and addressing, not signer identity. A caller must pin the expected
capsule digest through the deployment control plane. Signature/key distribution can be layered on
the manifest without weakening hash checks; it is not claimed by this implementation.

## H6 adversarial validation of the trust boundary

The H6 proof-carrying promotion campaign treats
`genesis/capsule/validator.py` as the production trust decision and treats the campaign generator,
mutations, fixture manifests, and re-sealed attacker output as untrusted inputs. The campaign does
not add an alternative verifier or accept a candidate from a hard-coded outcome. It first requires
the unmodified, promotion-complete conformance capsule to pass `validate_capsule`, then requires the
same validator to reject each mutation.

The default campaign spans three explicit seeds and ten attack classes:

- changed generated-runtime bytes;
- a mismatched current hardware fingerprint;
- a mismatched installed dependency version;
- evidence evaluated after its validity horizon;
- a missing required quality-evidence record;
- a re-sealed benchmark summary inconsistent with raw samples;
- changed replayable quality-evidence bytes;
- a missing counterexample corpus;
- an operational/model-check claim outside the current hardware scope; and
- changed state-conversion bytes naming an incompatible source genome.

Each result includes the validator's exact typed issue codes and immutable hashes for the mutation,
attacked capsule, validation context, and validation report. A separate campaign validator reopens
those artifacts and calls the production validator again. It then builds the baseline capsule in a
fresh temporary tree, reapplies the mutation, and compares the independently rebuilt mutation,
capsule, context, issue set, and report. This prevents the summary from becoming authority for its
own result.

The model-check-scope attack is rejected both when the current hardware lies outside the mutated
scope and because the mutated complete claim no longer matches its externally trusted claim anchor.
The anchor closes scope-broadening attacks too; checking only that the current hardware appears in a
manifest-provided scope would let an untrusted publisher enlarge that scope.

Several re-sealed attacks also substitute the attacker-selected manifest digest into the test
context. This is intentional adversarial depth: the inner evidence-anchor, completeness, scope, and
statistics controls must still reject the attack even if content addressing alone passes. It is not
the operational trust flow. In deployment, `expected_capsule_digest` remains an independently
pinned control-plane value, and an attacker cannot authorize its replacement merely by re-sealing a
manifest.

H6 is local CPU trust-boundary conformance only. It records zero hardware-backed runs and zero GPU
hours. The benchmark inputs are deterministic validation vectors, not measured performance. The
model-check attack verifies claim-scope enforcement, not the soundness of the bounded model checker.
The migration attack verifies the integrity of an externally anchored state-conversion artifact,
not semantic equivalence of source and target state. Semantic conversion replay and protocol
invariants remain obligations of the evidence issuer and promotion controller.

## Generated-code sandbox

The sandbox takes an argv vector and never invokes a shell. It receives explicit non-symlink
read-only inputs and one explicit writable artifact directory. The working directory must be
inside a declared input, while the output must be outside it. It starts with a new process group,
closed stdin, captured pipes, and a deterministic `PYTHONHASHSEED`.

The environment is rebuilt from a short allowlist. It does not inherit cloud, API, SSH, Kubernetes,
or other credentials; credential-shaped caller variables are rejected. `HOME` and `TMPDIR` point
inside the artifact directory. User site packages and bytecode writes are disabled.

On macOS, the generated process runs under a profile that denies network operations and child
forks, denies writes except to the artifact directory, and protects the user/workspace roots while
allowing declared inputs and required runtime files. This is not a system-wide read allowlist:
system-readable paths outside protected roots may remain readable. On Linux, bubblewrap creates
new namespaces, unshares networking, mounts runtime and inputs read-only, and binds only the
artifact directory writable. If a required isolation capability is missing, the result is
`policy_unavailable` and no candidate code runs.

Both paths enforce wall and CPU time, file and descriptor bounds, output byte limits, deterministic
environment metadata, and process-group cleanup. Linux retains an address-space limit. macOS uses
a parent-owned process-group RSS watchdog because interpreter mappings make `RLIMIT_AS` unreliable;
that limit is explicitly best-effort rather than a kernel/cgroup guarantee. macOS denies generated
child forking outright. Output is drained incrementally, and the artifact tree is rescanned during
execution; output, memory, entry, or aggregate-byte floods kill the process group. Final acceptance
rechecks that the artifact tree contains only bounded regular files.

Generated runtime imports additionally require the sandbox launch marker. Direct in-process loading
is rejected unless a caller uses the explicitly named test-only opt-in. The deployment and capsule
manifests advertise the trusted sandbox launcher and set direct launch support to false.

### Capability limitations

- `sandbox-exec` is deprecated by Apple and must be revalidated after OS updates.
- Bubblewrap presence does not prove that the host enables unprivileged user namespaces; setup
  failure remains fail-closed.
- OS/runtime files required to start the interpreter remain readable. The workspace/user trust
  roots and undeclared sibling files are denied, but this is not a virtual machine boundary.
- `RLIMIT_AS`, `RLIMIT_NPROC`, and a userspace RSS watchdog are not cgroup substitutes. Capability
  records identify best-effort boundaries, and production deployment still requires an outer
  container/VM memory boundary.
- Windows has no accepted backend and therefore cannot execute with strict defaults.
- Linux bubblewrap supplies a private minimal `/dev`. The macOS profile does not create a device
  namespace, so it must not be described as proof that all readable device nodes or ioctls are
  unavailable. No GPU device is intentionally granted by the orchestrator. A future GPU path
  requires an explicit, separately reviewed capability and cannot silently fall back.

### Exercised and unexercised isolation

| Boundary | Status in this workspace | Evidence scope |
| --- | --- | --- |
| macOS `sandbox-exec` | Implemented and exercised on Darwin on 2026-08-02 | Environment sanitization, deterministic seed, blocked undeclared workspace read, blocked loopback connection, blocked source write, artifact write, child-fork denial, timeout cleanup, output cap, active artifact-entry flood termination, RSS observation, credential-name rejection, and symlink-output rejection |
| Linux bubblewrap | Implemented but unexercised in this workspace | Command construction and fail-closed capability reporting exist; no Linux kernel/user-namespace execution result is claimed |
| No supported backend | Implemented as a fail-closed path | Strict requests return `policy_unavailable`; generated code is not run |
| Windows | Not implemented | Strict generated-code execution is unavailable |

The current targeted verification command is:

```bash
PYTHONDONTWRITEBYTECODE=1 uv run pytest -q \
  tests/python/test_genesis_sandbox.py \
  tests/python/test_genesis_capsule.py \
  tests/python/test_genesis_capsule_attack_campaign.py \
  tests/python/test_genesis_artifacts.py
```

That command passed 27 tests on the recorded macOS workspace. It is not evidence for Linux,
Windows, GPU isolation, container escape resistance, or protection against kernel vulnerabilities.

## Threats and controls

| Threat | Control | Residual risk |
| --- | --- | --- |
| Malicious or accidental generated code | Network denial, filesystem policy, sanitized environment, no shell, bounded resources, process cleanup | Kernel/runtime sandbox flaws |
| Credential theft | No inherited environment; user/workspace roots denied; isolated `HOME` | Credentials stored in OS-readable locations outside protected roots |
| Fork bomb or orphan | Fork denial on macOS; PID namespace and kill-on-parent on Linux; process group kill | Linux process limit is best-effort without cgroups |
| Output, disk or memory flood | Incremental output cap, `RLIMIT_FSIZE`, active artifact-tree scan, Linux address-space limit and macOS RSS watchdog | Filesystem-wide quotas and kernel-enforced macOS memory isolation are deployment-environment responsibilities |
| Capsule or artifact tampering | Pinned canonical manifest digest plus per-file size/SHA-256 | Hashes do not authenticate an unpinned replacement manifest |
| Stale proof/evidence | Required validity horizon, exact verifier/dependency/hardware/contract match | Issuer compromise before expiry |
| Benchmark manipulation | Raw samples, definition/software/workload/hardware binding, statistic recomputation, randomized-order requirement | Validator cannot independently detect malicious timers without a trusted harness |
| Hardware/model mismatch | Exact fingerprints and contract hashes at validation | Fingerprint collector correctness is part of the surrounding TCB |
| Lineage poisoning | Lineage output is untrusted proposal input; all transferred work is reverified | Search efficiency may still degrade |
| Unsafe promotion | Promotion requires a complete capsule and independent gate; live production mutation remains opt-in | Deployment controller bugs |

## Separation and operation

Generated builds receive read-only source and an empty artifact output directory. They cannot write
the verifier, benchmark acceptance code, capsule validator, promotion gate, raw baseline evidence,
or lineage database. Validation should run in a fresh process after generation, with a pinned
capsule digest and freshly collected compatibility context. Old champions and rollback artifacts
remain content-addressed and are not overwritten by a challenger.

Default operation is local and offline. No paid resource, external synthesis service, GPU device,
multi-node action, or live promotion is authorized by a capsule. Those actions require the
separate explicit opt-ins and budgets described by the project operational controls.
