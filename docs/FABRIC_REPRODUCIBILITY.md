# Fabric reproducibility

Fabric separates deterministic control-plane results from variable hardware
measurements and labels synthetic calibration at every boundary.

## Locked environment

Python dependencies are locked by `uv.lock`; Rust dependencies by `Cargo.lock`;
the UI uses `package-lock.json`. Run:

```bash
make bootstrap
make check
make fabric-check
```

Every simulator, compiler, synthetic profiler, ForgeCI fixture, WarmPath
simulation, and demo accepts or records a seed. Current flagship defaults use
seed 41; ForgeCI/WarmPath internals record their derived seeds.

## Reproduce the CPU extension

```bash
make fabric-demo
make autopsy-demo
make forgeci-demo
make warmpath-demo
make extension-evaluation
```

These commands reset only their scoped artifact/report directories. The Fabric
and Autopsy demos use synthetic topology and fabric curves; ForgeCI uses a local
fixture Git repository; WarmPath profiles real local file operations over
synthetic snapshot bytes.

## Evidence chain

Each flagship artifact is hashed in `artifacts/fabric-demo/manifest.json`.
Profiles retain raw warmup and measurement samples, invocation, environment, and
placement. Simulator output retains input hash, seed, calibration artifact URIs
and kinds. Physical plans retain logical/model/topology/profile hashes and Git
commit. Recovery and diagnosis retain evidence references. WarmPath and ForgeCI
write their own manifests/tree hashes.

Reports load generated artifacts; they do not embed expected benchmark values.
To audit a quoted metric, start from:

- Fabric: `artifacts/fabric-demo/manifest.json`;
- H1-H4 matrix: `artifacts/fabric/evaluation/manifest.json` and
  `artifacts/fabric/evaluation/result.json`;
- ForgeCI: `artifacts/forgeci/demo/evaluation.json`;
- WarmPath: `artifacts/warmpath/manifest.json`;
- rank ordering: raw experiment artifact referenced by
  `reports/rank-ordering-experiment.md`.

## Hardware-backed runs

Run `sloforge fabric discover` without `--fixture`, then run the measured
`host-memory` suite or an explicitly installed optional GPU/collective adapter.
Do not combine a profile with a different topology: compilation verifies the
canonical fingerprint. Preserve driver, CUDA, NCCL, runtime, container, clock,
power, placement, command, and topology metadata with the raw samples.

The current binding hashes the complete canonical topology, including capture
timestamps. Re-discovering an equivalent host therefore requires a fresh profile
instead of reusing the old one. This is intentionally fail closed, but it is
functional reproducibility rather than a compatibility-aware calibration cache.

The current development host had no NVIDIA GPU or RDMA device. Multi-GPU and
multi-node paths are implemented and fixture-tested but no hardware result is
reported for them.

## Sources of nondeterminism

Actual scheduling, page cache, thermal state, clocks, network congestion, and
background work vary. Repeated trials and robust intervals quantify within-run
noise, not all environmental drift. Git commit and hardware fingerprint must be
held constant when comparing runs. A regenerated artifact can have different
capture timestamps and hashes even when its seeded synthetic metrics agree.
