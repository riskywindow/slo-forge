# Fabric profiler

The Fabric profiler measures curves over shape and placement. It does not reduce
a link or collective to one peak-bandwidth value.

## Artifact contract

Each `BenchmarkResult` includes primitive, message bytes, ranks, concurrency,
direction, process/GPU/NUMA/NIC placement, invocation, timeout, warmup count,
raw samples, robust summary, environment, status, and an explicit failure
reason. The enclosing `FabricProfile` binds results to a topology fingerprint.
Measurement samples carry trial index, duration, optional throughput, and seed.
The invocation retains the warmup count; the generic Fabric profiler currently
does not retain warmup durations. Summaries include median, p95, p99, median
absolute deviation, and a seeded bootstrap interval.

`save_profile` writes structured JSON and referenced raw samples, then hashes the
artifact. The canonical conversion copies curve provenance and binds the profile
to the canonical topology fingerprint.

## Modes

`benchmark_synthetic_fabric` is a deterministic fixture calibrator. Its service
times are derived from fixture link curves, message volume, rank count, primitive
volume factors, and contention. Results are always labeled synthetic. The quick
suite covers host copy, GPU P2P, all-reduce, all-to-all, expert dispatch, KV
transfer, and startup. The full suite covers all declared primitives, including
GPU local work, H2D/D2H, collective variants, expert combine, and group
initialization.

`benchmark_host_memory` is the current measured path. It performs bounded NumPy
copies on the selected CPU and records the real samples and environment. It does
not claim GPU, pinned-memory, or transport measurement.

## Optional adapters

Adapter inventory supports offline command generation and availability checks
for NCCL tests and other optional transport tools. NCCL commands require an
explicit, cardinality-checked device list which is lowered to
`CUDA_VISIBLE_DEVICES`; they cannot inherit an ambient device set. These
builders do not execute tools or parse their output. Missing CUDA, GPU, NCCL,
DeepEP, NIXL, UCX, or RDMA remains explicitly unavailable; no CPU or TCP
fallback is relabeled as the requested primitive.

The runtime compatibility ranges and official source URLs validated for
deployment lowering are recorded in `deploy/fabric/validated-versions.json`.
Normal CI does not install these optional stacks.

## Methodology

Warmup counts are retained and warmup iterations are excluded from summaries;
only the focused rank-ordering experiment currently retains warmup durations.
A benchmark has at least three measurement samples; bootstrap summaries use at
least 100 rounds. Synthetic noise uses the explicit seed. Actual benchmarks
enforce a timeout and fixed payload size. Placement and command are part of the
evidence, so two results are not pooled unless their environment and physical
binding are compatible.

## Known limitations

No NVIDIA GPU, NVLink, NCCL, InfiniBand, or RoCE measurements were executed on
the current machine. The repository currently provides strict optional-tool
inventory and command construction, not a hardware-output parser or end-to-end
NCCL result ingester. Consequently `benchmarks/fabric/hardware-full.yaml` is a
hardware-ready matrix specification, not an exercised runner. Synthetic demo
profiles must not be quoted as hardware performance.
