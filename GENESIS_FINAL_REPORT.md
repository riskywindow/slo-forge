# SLOForge Genesis verified implementation report

Date: 2026-08-02 (America/Los_Angeles)

- Baseline commit: `435a04799a831c3d19fce18eb816b206d23778d7`
- Immutable baseline tag: `sloforge-genesis-baseline-435a047`
- Final reviewed implementation ancestry: `8dc39a2d383232449e42a06f23794f9b8e69e06b`
- Clean-room validated release commit: populated from `artifacts/genesis/clean-room/result.json`
- Final report publication commit: resolve with `git log -1 --format=%H -- GENESIS_FINAL_REPORT.md`

This report distinguishes local CPU measurements, deterministic synthetic/model evidence, bounded
verification, and unexercised hardware or external paths. It claims neither universal correctness,
global algorithmic novelty, GPU performance, nor a serving speedup unsupported by raw evidence.

## Architecture and implementation

Genesis extends the completed SLOForge, Fabric, Autopsy, recovery, ForgeCI, and WarmPath systems.
Python owns inspection, runtime generation, synthesis/search, verification orchestration, lineage,
evolution, benchmarking, and reports. Rust owns strict canonical IR conformance and deterministic
bounded protocol checking. Their primary boundary remains versioned canonical JSON over bounded,
timed subprocess stdin/stdout; HTTP/SSE remains confined to the running data plane.

Implemented and integrated:

- strict typed `InferenceGenome`, `Transformation`, `Candidate`, `Counterexample`, and
  `GenesisCapsule` protocols; eight genome regions; JSON Schemas; Python/Rust golden round trips;
  canonical SHA-256 agreement; explicit migrations, extension namespaces, proof obligations, and
  auditable lifecycle states;
- a zero-day AST frontend plus optional `torch.export` adapter, persistent-state and batching
  analysis, operator contracts, unsupported-feature obligations, typed randomized task grammar,
  and task-specific-special-casing detection;
- model-specific generated runtimes with bounded admission and output queues, deterministic
  sampler, batching, streaming, cancellation, per-request state ownership, bounded paged state,
  health/metrics, deterministic replay, failure cleanup, and clean shutdown;
- a restricted typed policy DSL and compiler/interpreter, bounded equivalence checks, tensor
  rewrites, state transformations, Fabric plan mutations, multi-fidelity deterministic search,
  hard multi-dimensional budgets, Pareto archive, Autopsy region freezing, and CEGIS;
- independent differential, property, schema-aware fuzz, quality, conservative resource,
  statistical performance, explicit-state protocol, artifact/capsule, and promotion gates;
- content-addressed artifacts, fail-closed generated-code sandbox, transactional SQLite lineage
  with invalidation/transfer, executable red team, champion/challenger evolution, ServingSynthBench,
  artifact-backed views, and a local upstream-review bundle;
- a measured reference-trace-selected CPU kernel experiment, generated source, exact numerical
  replay, isolated and generated-runtime serving benchmarks, raw samples, paired confidence gates,
  and preserved negative result. The optional Triton adapter is fail-closed and was not exercised.

No original command, protocol, or schema was replaced. Backward compatibility remains enforced
within an IR major version, with explicit migrations for incompatible alpha input.

## Trusted computing base

Generated source, policies, transformations, runtime packages, kernels, and synthesis reasoning are
untrusted. The conservative local authority includes strict parsing/canonicalization, artifact and
capsule validation/build issuance, bounded-proof recomputation, the sandbox, runtime/evolution gate
replay, benchmark-integrity reconstruction, kernel acceptance, promotion/rollback control, and the
clean-room verifier. On the final reviewed source it measures 11,086 physical lines. The narrower
capsule/artifact/sandbox subset measures 4,257 lines. Exact file lists and reproduction commands are
in `docs/genesis/TRUST_MODEL.md`; both counts exclude schemas, interpreters, the OS sandbox/kernel,
cryptographic implementation, and transitive dependencies.

The macOS `sandbox-exec` path was exercised with no network, sanitized environment, declared
read/write roots, CPU/RSS/process/output/artifact/wall-time bounds, fork denial, deterministic seed
metadata, and process-group cleanup. Linux bubblewrap is implemented but unexercised here. Runtime
and frontend loading compiles the exact hashed source bytes and rejects bytecode caches, preventing
stale or substituted `.pyc` execution. No generated program receives cloud credentials or arbitrary
repository write access.

## Generated zero-day runtime

The unseen HybridDecoder package combines sliding-window attention, recurrent/state-space behavior,
sparse MoE, quantized persistent state, a speculative head, and a custom sampler. Genesis recovered
79 operator records and five state fields, built the genome, and generated the serving runtime
without a hand-written production adapter.

Candidate `candidate-corrected-f358aac5a54f` reached `SIMULATED`; genome
`4db91427bcb35144e076ee8fd21a9b9c32b4d30fe09e25d75c68f409859a8f3b` passed exact reference
replay, state transitions, streaming, cancellation, queue/resource bounds, ownership/release,
deterministic seeds, and clean shutdown. It makes algorithmic policy changes to RequestGenome and
ServingGenome (`deadline_bucket`, SLO-slack scheduling, pre-emit cancellation check) and a bounded
64-byte paged allocator change to StateGenome. Thus the accepted candidate changes three genome
regions and two transformation categories; the measured campaign does not establish that this
combination is faster than the best isolated alternative.

Local capsule `0b7baeeba952b82de333e4bda6eba787ff570c590f7c3456a6220cbee7a1176a`
contains 17 artifacts, six evidence records, and five independently scoped claims. It is eligible
for the tested local evolution path, deliberately ineligible for external production, and carries
no hardware benchmark promotion claim.

## Verification and counterexample result

The higher modeled-upside candidate `candidate-fast-29ebf72e649f` scheduled a request for token
commitment after cancellation. The real verifier rejected it, and 15 minimizer evaluations reduced
the witness to three events: `admit`, `cancel`, `emit`. Counterexample
`counterexample-da9ead0ea96e115d40f634c5` produced family constraint
`constraint-d4cb57a99c1875a2fe18708b` (`cancel_check_before_emit == true`). The repeat candidate
`candidate-repeat-31c430a8d417` was then suppressed before verification, and the corrected
policy/state candidate was generated and accepted.

The corrected policy was exhaustively evaluated over all 66,066 declared DSL states. The separate
bounded protocol abstraction visited 20 states and 52 transitions to depth four for one request and
at most two committed tokens; its artifact explicitly records `universal_proof: false`. Capsule
construction re-hashes the reference package, binds final-corpus oracle records, recomputes bounded
evidence, and sandbox-replays the exact runtime. H6 rejected 50/50 retained attacks across 11 issue
codes, including changed binaries, wrong hardware/dependencies, stale/incomplete evidence, altered
benchmarks/quality, missing counterexamples, invalid proof scope, and state-conversion mismatch.

## Test and demo status

Final host validation before clean-room publication:

- `make check`: 728 Python tests passed, five optional PyTorch/GPU tests skipped, and six intentional
  overflow warnings; Rust formatting, warning-denied workspace Clippy, 128 Rust tests/doc-test lanes,
  UI typecheck/lint, 37 UI tests (one fixture skip), and production build passed. The final Rust/UI
  portion was rerun independently after a detached-session record was lost.
- `make genesis-check`: 362 Genesis/SynthBench Python tests passed, one optional PyTorch skip;
  Genesis Rust formatting/lint and 30 IR/model-check tests passed.
- `make fabric-check`: 292 focused Python tests and 31 Fabric Rust tests passed after a serialized
  rerun. An earlier parallel invocation raced the demo's `--reset`; its six missing-artifact errors
  are excluded rather than misreported as product failures.
- `make demo`, `make fabric-demo`, `make autopsy-demo`, `make forgeci-demo`, `make warmpath-demo`,
  and `make extension-evaluation`: passed. Core replay served 120 live and 120 simulated requests;
  Fabric/Autopsy restored the synthetic SLO; ForgeCI found its planted first regression; WarmPath
  restored and checksum-verified its local artifact fixture.
- `make genesis-demo`, `make genesis-zero-day-demo`, `make genesis-redteam-demo`,
  `make genesis-evolution-demo`, `make synthbench-smoke`, `make synthbench-evaluation`, and
  `make genesis-evaluation` plus its independent suite validator: passed.
- `make genesis-docker-smoke`: failed closed at preflight because the Docker daemon is unavailable;
  no container was started. Docker execution is implemented but unexercised on this host.
- `make genesis-clean-room-test`: final result is recorded after running against the committed
  release candidate; it uses a Git archive, fresh locked environment, built wheel, installed CLI,
  actual capsule validation, and a portable evidence tarball.

No SLOForge child process or fault configuration remained after the runs. No cloud resource,
external deployment, paid synthesis call, privileged probe, or production traffic mutation occurred.

## Real benchmark and evaluation highlights

ServingSynthBench CPU smoke generated two tasks with 1.0 valid/exact rates and retained 0.077716
seconds of aggregate request wall duration. The ten-task CPU evaluation also recorded 1.0
valid/exact rates and 0.403103046 seconds. These duration sums are neither campaign CPU time nor
production latency. Eager reference and generated Genesis paths are actual local runs; unavailable
framework/hardware lanes and explicitly labeled surrogates are not independent runtime baselines.

The independently validated H1-H9 suite has nine CPU/synthetic campaigns, zero hardware-backed
runs, and zero universal-proof claims:

- H1 supported in grammar scope: 5/5 valid and exact hidden-task runtimes, 95% Wilson lower bound
  0.5655, and zero hand-authored model-specific serving lines.
- H2 supported only in deterministic service-model scope: configuration-only minus Genesis was
  0.19775 modeled units over five paired seeds, descriptive 95% interval [0.09123, 0.30974].
- H3 mixed: full CEGIS and bounded model-check-only each escaped 0/10 scoped faults; tests-only
  escaped 10/10, fuzz-only 5/10, and five learned constraints were reused.
- H4 mixed: Autopsy guidance avoided 15 invalid candidates and 3.5328 synthetic time units versus
  unrestricted search, but used three more candidates than random-region search.
- H5 supported only in synthetic transfer scope: related lineage saved median two candidate units
  and four synthetic time units; invalidation avoided five negative transfers.
- H6 supported in local validator-conformance scope: 50/50 attacks rejected.
- H7 supported in synthetic-controller/local-runtime scope: restoration 1.0 versus 0.0 static,
  zero interrupted streams, and one exercised rollback.
- H8 supported in executable-fixture scope: 19 red-team-only contract families; 95/95 converted
  regressions independently replayed across five seeds.
- H9 not supported: best single-layer minus Genesis was -0.005625 modeled units, interval
  [-0.0057125, -0.00555]. The negative result is retained.

The trace-selected quantized-state CPU kernel experiment profiled seven reference workload trials
and attributed 14.16% of observed profile time to the target without claiming causality. Both
generated candidates passed exact correctness, but neither was accepted. The only generated-runtime
serving comparison used six requests, seven paired trials, exact token/final-state equality, and
independent replay: reference median 3,796,666 ns, candidate median 3,850,667 ns, -1.422% point
estimate, 95% paired interval [-3.495%, 1.752%]. Status: inconclusive/rejected, zero speedup claims.

## Lineage and evolution results

The standalone lineage demonstration retrieved one related transformation, kept four unseeded
candidates for diversity, ignored unrelated lineage, required reverification, and suppressed the
seed after dependency invalidation. It declares `performance_hypothesis_evaluated: false`; the H5
cost-unit result comes from the separate five-seed evaluation campaign.

The flagship workload-drift timeline is artifact-derived: trigger, separately synthesized and
capsuled challenger, registration, sandboxed shadow, sandboxed canary, revalidation, promotion,
active-stream preservation, and retained prior champion. A later simulated Fabric degradation
trigger entered evolution. The H7 campaign additionally exercises invalid-challenger rejection,
physical-degradation coalescing, rollback, and controller persistence/restart. External live
promotion remains disabled.

## Hardware-backed versus synthetic validation

This was an Apple-Silicon macOS 15.6.1 host running an x86_64 process under Rosetta, with 12 logical
CPUs and 24 GiB RAM. There was no NVIDIA inventory, CUDA compiler, Triton/PyTorch installation,
GPU or cloud budget, multi-node opt-in, external synthesis authorization, or live-promotion opt-in.
Reported runtime and kernel timings are real local CPU measurements; Fabric, controller-time,
service-model, topology degradation, H2/H4/H5/H7/H9, and some comparison lanes are explicitly
synthetic. There is no single-GPU, multi-GPU, multi-node, RDMA/NCCL, CUDA/Triton, Modal, Truss,
external-engine, cloud, or production-traffic measurement.

## Known limitations and unmet evaluation gates

- The optional `torch.export` path and GPU/Triton tests were skipped because dependencies/hardware
  are absent; AST inspection and generated pure-Python tasks were exercised.
- Tensor rewrites are typed and verified but arbitrary selected rewrites are not lowered into the
  flagship runtime. The Fabric mutation is evaluated and deliberately remains ineligible pending
  its full revalidation pipeline. The accepted flagship changes policy and state, not all four
  evaluated transformation categories.
- H2/H4/H5/H7/H9 use deterministic model or logical units and do not establish hardware serving
  performance. H3/H4 are mixed and H9 is negative.
- The local capsule lacks Level-5 hardware/production evidence and cannot be externally promoted.
- Docker, Linux bubblewrap, Windows, GPU, multi-node, paid synthesis, external deployment, and live
  production promotion are unexercised. The build backend itself is not lock-pinned; external
  package indexes and host toolchains remain supply-chain assumptions.
- Human-readable artifact records contain host-absolute paths. Hashes detect mutation and the
  clean-room evidence archive is portable, but unpacked reports may require rebasing/regeneration.
- The randomized grammar is a small affordable architecture family; fixed seeds and fixture-level
  confidence intervals do not generalize to arbitrary production models or workloads.
- The upstream-ready bundle remains local; no pull request or issue was opened.

## Artifact and documentation inventory

- Baseline record: `artifacts/genesis/baseline/record.json`
- Flagship runtime, capsule, kernel evidence, lineage, evolution, and report:
  `artifacts/genesis/demo/`
- Zero-day, red-team, and evolution demonstrations: `artifacts/genesis/zero-day-demo/`,
  `artifacts/genesis/redteam-demo/`, `artifacts/genesis/evolution-demo/`
- H1-H9 evaluation root: `artifacts/genesis/evaluation/GENESIS_EVALUATION_SUITE.json`
- Lineage transfer: `artifacts/genesis/lineage-transfer-demo/report.json`
- ServingSynthBench: `artifacts/synthbench/`
- Clean-room result/log/portable evidence: `artifacts/genesis/clean-room/`
- Architecture, trust, security, reproducibility, and limitation docs: `docs/genesis/`
- Lineage/red-team/SynthBench docs: `docs/lineage/`, `docs/redteam/`, `docs/synthbench/`
- Related work and ADRs: `docs/GENESIS_RELATED_WORK.md`, `docs/adr/`
- Paper-style report: `paper/genesis/README.md`
- Upstream-review bundle: `generated/patches/hybrid-quantized-state-update/`

All numeric claims above are derived from the named JSON/JSONL artifacts or final command logs.
Unavailable hardware paths and negative results are repeated deliberately so local CPU evidence
cannot be presented as GPU, production, or universal-proof evidence.
