# SLOForge Fabric GPU-performance and low-level methodology review

Review date: 2026-08-01

Scope: Fabric profiling adapters, optional NVIDIA/NCCL behavior, and the
topology-aware collective rank-ordering experiment. This review did not modify
the compiler, simulator, Autopsy, recovery, root manifests, or demo/evaluation
artifacts.

## Verdict

The focused rank-ordering contribution is a valid deterministic digital-twin
experiment with a sequential reference, calibrated physical routes, paired
samples, warmup separation, robust intervals, and conservative integration
gates. It is not a GPU speedup result. The checked experiment remains disabled
by default and has status `measure_on_hardware`.

No NVIDIA GPU or communication hardware was present on the review host (Apple
arm64 Darwin). CUDA, NVML, NCCL tests, UCX, ibverbs/perftest, PyTorch, vLLM,
SGLang, Dynamo, NIXL, and DeepEP all reported `unavailable`. Accordingly, no
GPU, NVLink, NCCL, InfiniBand, or RoCE result is claimed.

## Findings fixed

1. High: `calibration_mode=measured_hardware` could previously cause modeled
   paired trials to return `enable`. The generated trials always perturb and
   evaluate digital-twin link curves; they are not on-device A/B measurements.
   All such trials can now only recommend `measure_on_hardware`. A regression
   test covers hardware-labeled inputs and verifies default disablement.
2. High: the library accepted a digest-shaped trace reference without reading
   the referenced evidence. It now requires a local materialized artifact,
   verifies SHA-256, checks every gate field against the JSON evidence, requires
   `hardware_exercised=true` for hardware-labeled traces, and bounds reads to
   16 MiB. Tests cover hash mismatch, field mismatch, and oversized evidence.
   `evidence_root` is a path-resolution base, not a containment sandbox; absolute
   and parent-relative evidence is allowed because bundles can live outside the
   repository, but remains size- and hash-checked.
3. High: NCCL test commands inherited ambient CUDA visibility while specifying
   only a GPU count. Command construction now requires one unique explicit
   device identifier per requested GPU and emits `CUDA_VISIBLE_DEVICES`.
   Empty, duplicate, comma-containing, and cardinality-mismatched device sets
   fail closed.
4. Medium: profiler documentation claimed that generic raw artifacts retained
   warmup phases and per-result hardware fingerprints. The implementation keeps
   warmup counts but not warmup durations; the enclosing profile, rather than an
   individual result, carries the topology fingerprint. Documentation now says
   exactly that.
5. Medium: the low-level correctness suite covered only one exact-search shape.
   It now exercises deterministic shuffled placements for two through six ranks,
   validates complete rings and non-regression against the reference, and covers
   a uniform topology where the result is correctly `inconclusive` rather than
   presenting a neutral result as a win.

## Methodology audit

- Reference: the physical plan's existing sequential ring is evaluated with the
  same calibrated path model as the selected order.
- Search: exhaustive rotationally-canonical enumeration is used for small rank
  groups; larger groups use a bounded deterministic nearest-neighbor plus 2-opt
  heuristic.
- Calibration: every traversed edge requires both measured/calibrated latency
  and bandwidth curves plus confidence metadata. There is no theoretical-link,
  CPU, TCP, device, or transport fallback.
- Synchronization model: collective completion is the maximum propagation plus
  contention-domain serialization over ring iterations. The experiment does not
  claim packet-level or CUDA-stream timing.
- Sampling: warmup and measurement trials are both retained in JSONL; summaries
  exclude warmups. Reference and selected routes receive matched per-link
  perturbations. Bootstrap resampling is seeded and paired.
- Statistics: results report medians, MAD, confidence intervals, raw trials, and
  limiting message-size regimes. Every configured regime must clear the
  practical-effect lower bound before hardware measurement is recommended.
- Negative/neutral handling: lack of trace justification keeps the reference;
  confidence intervals crossing zero are inconclusive; confidence-supported
  losses keep the reference. The uniform-topology test covers the neutral path.
- Enablement: no modeled experiment enables the optimization. A future matched
  on-device runner must supply actual paired A/B measurements before an enable
  decision can be implemented.

## Optional hardware path audit

`adapter_inventory` performs bounded version probes and distinguishes missing,
available, and version-unparsed tools. The NCCL, ibverbs, and NVIDIA-SMI builders
validate executable paths, arguments, read-only fields, timeouts, and explicit
environment overlays without mutating the parent environment. They only build
commands: they do not execute tools, parse hardware output, synchronize CUDA
work, or ingest NCCL samples into a `FabricProfile`.

Therefore `benchmarks/fabric/hardware-full.yaml` is a hardware matrix
specification, not an exercised hardware benchmark runner. This is documented as
an unimplemented hardware-output ingestion limitation, not described as
implemented-but-unexercised. Normal CI requires none of the optional stacks.

## Checked experiment

- Input evidence:
  `benchmarks/fabric/rank_ordering/observed-collective-trace.json`
- Canonical result:
  `benchmarks/fabric/rank_ordering/results/rank-ordering-experiment.json`
- Complete paired samples:
  `benchmarks/fabric/rank_ordering/results/rank-ordering-raw-samples.jsonl`
- Artifact-derived report: `reports/rank-ordering-experiment.md`
- Result hash: `1ed6199556658ae957aa5428e2cae85a5dd5c82b514391b3e3abb17d51c63d81`
- Calibration: synthetic calibrated
- Decision: `measure_on_hardware`, `enabled_by_default=false`

The synthetic fixture reports modeled median benefits of 59.985% at 64 KiB and
65.536% at 1 MiB. These values are only validation of the known asymmetric
fixture and route model; they are not hardware benchmark numbers or a speedup
claim.

## Acceptance evidence

Executed on the review host:

```text
uv run --locked ruff check <scoped Python and test files>
All checks passed!

uv run --locked mypy \
  python/sloforge/fabric/performance/rank_ordering.py \
  python/sloforge/fabric/profiling/adapters.py
Success: no issues found in 2 source files

uv run --locked pytest -q \
  tests/python/test_fabric_performance.py \
  tests/python/test_fabric_profiling.py
31 passed
```

The checked experiment was regenerated through
`uv run --locked python benchmarks/fabric/rank_ordering/run_experiment.py`; its
report and content hash changed only because the integration rationale now
states the modeled-versus-measured boundary explicitly.
