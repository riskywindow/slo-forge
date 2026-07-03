# SLOForge Genesis verified implementation report

Date: 2026-08-02 (America/Los_Angeles)

Baseline commit: `435a04799a831c3d19fce18eb816b206d23778d7`

Baseline tag: `sloforge-genesis-baseline-435a047`

Final core implementation commit: `e52868bd76b486365c9c2360c88fcd996eb317a1`

Clean-room validated release commit: `46814c0f1c29301ff4220c3b56bf663c18ed9301`

Current report commit: resolve with `git log -1 --format=%H -- GENESIS_FINAL_REPORT.md`; later
report-only evidence-scope corrections do not change the validated implementation.

This report distinguishes exercised CPU evidence, synthetic evidence, and unexercised optional
hardware/external paths. It does not claim universal correctness, GPU performance, globally novel
algorithms, or a measured speedup.

## Architecture and implementation

Genesis extends SLOForge rather than replacing it. Python owns zero-day inspection, baseline
runtime generation, restricted policy compilation, tensor/state/distributed transformations,
search, CEGIS, statistical gates, lineage, evaluation, and reports. Rust owns strict canonical IR
conformance and bounded protocol model checking. The established boundary remains versioned,
canonical JSON over bounded subprocess stdin/stdout; the live gateway keeps HTTP/SSE.

Implemented components include:

- typed, versioned `InferenceGenome`, `Transformation`, `Candidate`, `Counterexample`, and
  `GenesisCapsule` surfaces with JSON Schemas, Python/Rust round trips, migrations, golden fixtures,
  canonical SHA-256 agreement, extension namespaces, and stable lifecycle states;
- a zero-day reference-package frontend with AST and optional `torch.export` inspection,
  state/operator/batching/streaming diagnostics, explicit unresolved obligations, package hashing,
  and typed randomized task generation;
- conservative model-specific runtime generation with bounded queues, state ownership, batching,
  streaming, cancellation, deterministic sampling, health/metrics, sandbox-only loading,
  differential harnesses, and clean shutdown;
- a deterministic restricted policy DSL, tensor rewrite engine, state/memory transformation IR,
  Fabric mutation bridge, multi-objective archive, budget manager, Autopsy mutation guards,
  candidate lifecycle, and counterexample-guided local synthesis;
- operator/quality/resource/performance verification, schema-aware fuzzing, numerical contracts,
  a Rust explicit-state protocol checker, and scoped verification levels;
- SQLite lineage with negative results, transfer, confidence/invalidation rules, JSON/GraphML
  export, champion/challenger evolution, guarded shadow/canary/promotion/rollback, red-team
  adversaries, ServingSynthBench, static artifact views, and an upstream-review bundle;
- a focused CPU kernel laboratory with retained sources/cases/runner outputs, independent replay,
  randomized correctness cases, repeated raw timings, confidence gates, and deliberately no
  end-to-end speedup claim.

## Trusted computing base

Generated source, policies, transformations, runtimes, kernels, and synthesis reasoning are
untrusted. Local evidence authority consists of strict parsers/canonicalization, capsule issuance
and validation, bounded proof recomputation, sandbox execution, runtime gate replay, benchmark
integrity reconstruction, kernel correctness replay, promotion control, and rollback control.

The conservative source-file envelope for these paths is 8,552 physical lines across capsule,
artifact, sandbox, synthesis-checker, evolution-evidence/controller, SynthBench harness, and kernel
acceptance files. This count excludes Python, Pydantic, the OS kernel, `sandbox-exec`/bubblewrap,
schemas, and transitive dependencies. The narrower capsule/artifact/sandbox validation subset is
3,741 lines. Exact files and reproduction commands are in `docs/genesis/TRUST_MODEL.md`.

The macOS sandbox was exercised with network denial, declared read/write roots, rebuilt
credential-free environment, CPU/RSS/output/artifact/process limits, deterministic seed metadata,
fork denial, timeout cleanup, and process-group cleanup. Linux bubblewrap is implemented but was
not exercised on this host. Windows strict execution is unavailable. No GPU device, cloud
credential, paid synthesis service, privileged probe, external deployment, or live-production
promotion was authorized.

## Generated zero-day runtime

The flagship HybridDecoder reference package contains sliding-window, recurrent/state-space,
sparse-MoE, quantized persistent-state, speculative-head, and custom-sampler behavior. Genesis
recovered 79 operator records and five persistent-state fields, constructed a genome, and generated
the candidate runtime without a hand-written production adapter.

Candidate `candidate-corrected-9e080dc3ce26` reached `SIMULATED`; its genome hash is
`f44d18e98b1001c9e528d61d707ffe1f364ed918e298179a61dbc6ca905d8860`. The runtime passed exact
final-corpus differential replay, streaming, cancellation, queue bounds, state ownership,
deterministic seed, and clean-shutdown checks. The accepted candidate modifies the RequestGenome
and ServingGenome through one policy transformation and is structurally cross-layer within those
two regions. Its StateGenome is unchanged. It does not establish a performance improvement.

The resulting local capsule is
`2051608ce0df253e1454c0a1a249d0bdb0ace5e6f6ffbae6cbfd813d31476e07`; it contains 16 artifacts,
six evidence records, and five scoped claims. It is eligible for the tested local evolution path,
but is intentionally ineligible for external production and carries no benchmark promotion claim.

## Verification and counterexample case study

The higher modeled expected-upside unsafe batching candidate `candidate-fast-c5f24ba72a3a`
scheduled work after cancellation. It had no benchmark evidence. The independent verifier rejected
it with real counterexample
`counterexample-f1a82ff1eae24eb0395f3581`; minimization retained the smallest reproducing schedule,
and generalized constraint `constraint-d4cb57a99c1875a2fe18708b` suppressed the repeated unsafe
family before the corrected candidate was selected.

The corrected policy was then exhaustively checked over all 66,066 declared integer/boolean input
states. The separate bounded protocol abstraction visited 20 states and 52 transitions to depth
four and records `universal_proof: false`. Capsule construction recomputes both documents,
re-hashes the reference package, binds each final-corpus oracle line, and sandbox-replays the exact
candidate runtime before sealing. Regression tests reject altered bounds/invariants/transitions,
equal-but-wrong expected/observed tokens, package drift, transformation drift, hostile policy
bytecode, stale evidence, hardware mismatch, and artifact tampering.

Evolution gate validation binds the still-current champion, both exact runtime bundles, trace,
raw observations, seeds, stages, and summaries, and independently sandbox-replays both runtimes
again immediately before promotion. Coherent attacks that re-hash a changed champion, trace, or
equal-but-forged observations are rejected.

## Tests and demonstrations

The final CPU release sequence exercised:

- `make check`: passed on clean-room release revision `46814c0`; Python, Rust workspace formatting,
  Clippy with warnings denied, Rust tests/doc tests, and UI type/lint/test/build all passed;
- `make genesis-check`: 270 Python tests passed, one optional PyTorch test skipped, four intentional
  numerical-overflow warnings; nine Genesis IR conformance tests and 17 model-check tests passed;
- `make genesis-demo`, `make genesis-zero-day-demo`, `make genesis-redteam-demo`, and
  `make genesis-evolution-demo`: passed;
- `make synthbench-smoke` and `make synthbench-evaluation`: passed;
- `make genesis-evaluation`: passed its content-addressed evidence revalidation;
- `uv run --locked python -m sloforge.lineage.demo ...`: passed transfer/invalidation mechanics;
- `make genesis-clean-room-test`: passed bootstrap, Genesis checks, demo, SynthBench smoke,
  package build, wheel reinstall, and installed CLI smoke from a Git archive;
- `make genesis-docker-smoke`: attempted, but the installed Docker client could not reach a Docker
  daemon; Docker execution is therefore unexercised, not passed.

The baseline record also preserves passing pre-Genesis `make check`, Fabric checks/demo, SLOForge
CPU demo, Autopsy demo, ForgeCI fixture, and WarmPath fixture. No generated-code subprocess or
fault-injection process remained after validation, and no cloud resources were created.

## Benchmark and evaluation results

ServingSynthBench CPU smoke generated two tasks and ran two execution seeds per task. Its report
records eight actual measured baseline runs (direct Python reference plus generated Genesis), 28
explicitly labelled request-order surrogate runs, and 16 unavailable hardware/framework lanes.
Valid-system and exact-request rates were 1.0. The sum of retained per-request wall-clock durations
was 0.08196783 seconds; it is not process CPU time or end-to-end campaign time.

The CPU evaluation generated ten task grammars and two execution seeds per task. It records 40
actual measured runs, 140 explicitly labelled surrogates, 80 unavailable lanes, a 1.0 valid-system
rate, and a 1.0 exact-request rate. The retained-request wall-duration sum was 0.425307152 seconds.
Every Genesis result is rebound to the descriptor, public workload or committed hidden corpus,
oracle, request, runtime manifest/files, exact retained runner response, and sandbox evidence.

The three-seed flagship evaluation recorded 1.0 accepted-runtime differential, local capsule,
real-rejection, red-team replay, and local evolution-promotion rates, with zero external-production
eligibility and zero hardware-backed runs. H6 is supported within its local tamper scope. H1, H3,
H7, and H8 are only partially evaluated. H2, H4, H5, and H9 remain not evaluated because there is
no hardware-comparable whole-stack campaign, complete ablation campaign, transfer-performance
campaign, or single-layer-versus-cross-layer performance comparison. These absences are release
limitations, not inferred successes.

The red team found 19 unique violation families per run and replayed all of them; across three
runs the report now distinguishes 57 total findings from 19 unique families. The two CPU kernel
candidates produced retained correctness and isolated-operator timing evidence but zero speedup
claims. No statistical significance or end-to-end impact is claimed.

## Lineage and evolution

The deterministic lineage demonstration retrieved one related transformation, retained four
unseeded candidates for diversity, required reverification, ignored unrelated lineage, and
suppressed the related seed after its dependency was invalidated. It explicitly sets
`performance_hypothesis_evaluated: false`; no transfer speedup is claimed.

The local workload-drift fixture generated an isolated challenger, validated its capsule, ran bound
shadow and canary gates, preserved an active stream, promoted the challenger, and retained the old
champion for rollback. After that promotion the demo classified a physical-degradation trigger and
entered `evolving`; it did not synthesize, verify, or promote a second degradation-specific
challenger. External live promotion remains opt-in and was not exercised.

## Hardware-backed versus synthetic evidence

This host was macOS x86_64 with 12 logical CPUs and 24 GiB RAM. It had no NVIDIA GPU, CUDA compiler,
Triton installation, GPU budget, cloud budget, or multi-node opt-in. All reported timings and
serving results are CPU/local or deterministic simulation. Hardware adapters, version checks, and
fail-closed commands exist, but no single-GPU, multi-GPU, multi-node, RDMA/NCCL, CUDA/Triton, Modal,
Truss, or external-engine result is claimed.

## Known limitations and unmet evaluation gates

- The optional PyTorch export frontend test was skipped because PyTorch is not installed; the AST
  frontend and generated Python task grammar were exercised.
- Arbitrary tensor rewrites are represented, checked, and costed, but the flagship generated
  runtime does not lower an arbitrary selected rewrite into model code.
- The flagship evaluates executable request/serving policy behavior and kernel-lab experiments,
  and records a physical-degradation trigger classification. State, tensor, and distributed
  transformation surfaces have focused tests but are not selected by the accepted flagship
  candidate; this is not a complete measured four-category whole-stack search.
- The local capsule is not externally promotable because it lacks repeated provenance-complete
  hardware/service benchmarks and production shadow/canary evidence.
- ServingSynthBench keeps unavailable engines and explicit request-order surrogates in the schema;
  those rows are not independent baseline implementations.
- SynthBench and Genesis evaluation artifact paths are currently host-absolute. Evidence hashes
  detect mutation, but moving an unpacked report requires path rebasing or regeneration.
- Confidence intervals exist for kernel/performance gates, but the aggregate valid-system rates do
  not yet publish task-level Wilson/bootstrap intervals. The fixed HybridDecoder seeds are not
  independent model-family samples.
- The upstream-ready bundle is local only; no issue or pull request was opened.
- Docker, Linux bubblewrap, Windows, GPU, multi-node, paid synthesis, external deployment, and live
  production promotion are unexercised.

## Artifact and documentation inventory

- Baseline: `artifacts/genesis/baseline/record.json`
- Flagship report/capsule/runtime/counterexamples: `artifacts/genesis/demo/`
- Zero-day and evolution fixtures: `artifacts/genesis/zero-day-demo/` and
  `artifacts/genesis/evolution-demo/`
- Multi-seed evaluation and hypothesis reports: `artifacts/genesis/evaluation/`
- Lineage transfer: `artifacts/genesis/lineage-transfer-demo/`
- Red team: `artifacts/genesis/redteam-demo/`
- ServingSynthBench smoke/evaluation: `artifacts/synthbench/`
- Clean-room result and log: `artifacts/genesis/clean-room/`
- Architecture/trust/reproducibility/limitations: `docs/genesis/`
- Lineage, red-team, and SynthBench specifications: `docs/lineage/`, `docs/redteam/`, and
  `docs/synthbench/`
- Related work: `docs/GENESIS_RELATED_WORK.md`
- ADRs: `docs/adr/`
- Paper-style system report: `paper/genesis/README.md`
- Generated upstream-review bundle: `generated/patches/hybrid-quantized-state-update/`

All numeric claims above are sourced from the named JSON/JSONL artifacts. Hardware and missing
campaign limitations are repeated deliberately to prevent a CPU fixture from being presented as a
production or GPU result.
