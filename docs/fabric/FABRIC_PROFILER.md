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

## Measured NCCL adapter

`fabric benchmark --measured --suite quick|full` has an executable NCCL-tests
path. It requires all of the following explicit inputs:

- `--adapter nccl-tests`
- `--transport nccl-local`
- an exact `--adapter-executable` path whose filename matches the operation
- one `--visible-device` GPU node ID or UUID per local rank

Every device must resolve uniquely in the supplied topology and all devices
must belong to one host. The runner lowers the exact set to
`CUDA_VISIBLE_DEVICES`, disables InfiniBand for this local transport class, and
does not inherit ambient CUDA or NCCL variables. It never substitutes a CPU,
TCP, synthetic, or different-operation benchmark. Cross-host NCCL requires a
separate bounded launcher and is intentionally rejected by this local runner.

The runner starts each repetition in a process group with closed stdin, a hard
deadline, and independently capped stdout and stderr. Timeout and output-limit
paths terminate the complete process group. It parses the standard 13-column
NCCL-tests table, rejects missing or duplicated sizes and nonzero correctness
counts, and retains both in-place and out-of-place fields in bounded capture
artifacts. Each independent process invocation contributes one out-of-place
raw sample per message size; the `-n` inner iterations remain the NCCL-tests
aggregation unit. Raw profile and capture artifacts are written before a failed
run returns an error, while a successful run also produces the canonical
`fabric-profile.json`.

Example (not executable on a machine without the named GPUs):

```bash
sloforge fabric benchmark \
  --topology artifacts/fabric/topology.json \
  --suite quick --measured \
  --adapter nccl-tests --transport nccl-local \
  --adapter-executable /opt/nccl-tests/build/all_reduce_perf \
  --visible-device GPU-aaaaaaaa --visible-device GPU-bbbbbbbb \
  --samples 7 --warmups 3 --inner-iterations 20 \
  --output artifacts/fabric/measured-all-reduce
```

An optional explicit `--nvidia-smi-executable` performs only allowlisted CSV
queries for the selected devices. The bounded query output is parsed into
environment facts and linked by a SHA-256 capture digest. The executable path
itself is hashed in every benchmark result.

Adapter inventory still supports bounded availability checks and offline
command construction for ib-perftest and other optional tools. Those builders
do not imply execution support. Missing CUDA, GPU, NCCL, DeepEP, NIXL, UCX, or
RDMA remains explicitly unavailable; no fallback is relabeled as the requested
primitive.

The runtime compatibility ranges and official source URLs validated for
deployment lowering are recorded in `deploy/fabric/validated-versions.json`.
Normal CI does not install these optional stacks.

## Methodology

Warmup counts are retained and NCCL performs them inside each invocation;
warmup durations are not reported by the standard table and are excluded from
summaries. Only the focused rank-ordering experiment currently retains warmup
durations. A benchmark has at least three independent process-level samples;
bootstrap summaries use at least 100 rounds. Synthetic noise uses the explicit
seed. Actual benchmarks enforce a timeout and bounded payload/output sizes.
Placement and command are part of the evidence, so two results are not pooled
unless their environment and physical binding are compatible.

## Known limitations

No NVIDIA GPU, NVLink, NCCL, InfiniBand, or RoCE measurements were executed on
the current machine. The local NCCL-tests runner and standard-output ingester
were exercised only against deterministic fixture executables, so this is an
implemented but hardware-unexercised path. Multi-node launch, ib-perftest pair
orchestration, NCCL subtransport verification, pinned-memory GPU transfer, and
DeepEP/NIXL ingestion remain unsupported. `benchmarks/fabric/hardware-full.yaml`
is still a hardware-ready matrix specification, not evidence of a hardware
benchmark. Synthetic demo and fixture-executable profiles must not be quoted as
hardware performance.
