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
- `genesis/artifacts/store.py`: atomic immutable SHA-256 object publication;
- `genesis/sandbox/models.py` and `genesis/sandbox/executor.py`: policy construction, sanitized
  process setup, resource limits, bounded output, timeout, and process-group cleanup;
- `schemas/genesis_capsule/genesis-capsule-v1.schema.json`: the public wire schema.

At the snapshot that introduced the trust lane, these Python files total approximately 2,110
physical lines including comments, declarations, and package exports. Reproduce the measurement
with:

```bash
wc -l python/sloforge/genesis/{capsule,artifacts,sandbox}/*.py
rg '^import |^from ' python/sloforge/genesis/{capsule,artifacts,sandbox} -g '*.py'
```

The direct non-standard Python dependency is Pydantic. The sandbox additionally depends on the
standard-library process/resource APIs and either macOS `sandbox-exec` or Linux `bubblewrap`.
Hashing uses the standard-library SHA-256 implementation. Synthesis, search, generated runtimes,
PyTorch, Triton, external coding agents, and their reasoning are outside this TCB.

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

## Generated-code sandbox

The sandbox takes an argv vector and never invokes a shell. It receives explicit non-symlink
read-only inputs and one explicit writable artifact directory. The working directory must be
inside a declared input, while the output must be outside it. It starts with a new process group,
closed stdin, captured pipes, and a deterministic `PYTHONHASHSEED`.

The environment is rebuilt from a short allowlist. It does not inherit cloud, API, SSH, Kubernetes,
or other credentials; credential-shaped caller variables are rejected. `HOME` and `TMPDIR` point
inside the artifact directory. User site packages and bytecode writes are disabled.

On macOS, the generated process runs under a profile that denies network operations and child
forks, denies writes except to the artifact directory, denies reads from the user/workspace trust
root, and re-allows only the runtime plus declared inputs and output. On Linux, bubblewrap creates
new namespaces, unshares networking, mounts runtime and inputs read-only, and binds only the
artifact directory writable. If a required isolation capability is missing, the result is
`policy_unavailable` and no candidate code runs.

Both paths enforce wall and CPU time, file and descriptor bounds, output byte limits, deterministic
environment metadata, and process-group cleanup. Address-space and Linux per-user process limits
are marked best-effort because their semantics vary without cgroups. macOS denies generated child
forking outright. Output is drained incrementally; an output flood is killed rather than first
buffered without bound. Before accepting a result, the executor walks the artifact tree with entry
and aggregate byte bounds and rejects symlinks, devices, and other non-regular output.

### Capability limitations

- `sandbox-exec` is deprecated by Apple and must be revalidated after OS updates.
- Bubblewrap presence does not prove that the host enables unprivileged user namespaces; setup
  failure remains fail-closed.
- OS/runtime files required to start the interpreter remain readable. The workspace/user trust
  roots and undeclared sibling files are denied, but this is not a virtual machine boundary.
- `RLIMIT_AS` and `RLIMIT_NPROC` are not portable cgroup substitutes. Capability records identify
  them as best-effort.
- Windows has no accepted backend and therefore cannot execute with strict defaults.
- Host-device access is absent from the default namespace/profile. A future GPU path requires an
  explicit, separately reviewed capability and cannot silently fall back.

## Threats and controls

| Threat | Control | Residual risk |
| --- | --- | --- |
| Malicious or accidental generated code | Network denial, filesystem policy, sanitized environment, no shell, bounded resources, process cleanup | Kernel/runtime sandbox flaws |
| Credential theft | No inherited environment; user/workspace roots denied; isolated `HOME` | Credentials stored in OS-readable locations outside protected roots |
| Fork bomb or orphan | Fork denial on macOS; PID namespace and kill-on-parent on Linux; process group kill | Linux process limit is best-effort without cgroups |
| Output or disk flood | Incremental output cap and `RLIMIT_FSIZE`; explicit artifact directory | Filesystem-wide quotas are deployment-environment responsibilities |
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
