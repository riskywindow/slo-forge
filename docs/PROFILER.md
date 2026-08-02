# Profiler and hardware characterization

SLOForge profiling is staged and budgeted. It records raw samples first and derives summaries second; no summary is accepted without candidate, environment, hardware and workload provenance.

## Hardware probe

`sloforge hardware probe --device cpu` captures architecture, CPU model, logical core count, system memory, cgroup limit, container indicator, OS and hostname, then runs warmed sample loops for host memory copy bandwidth and a 384-square FP32 GEMM. The output includes every sample, warmup count, units, median and p95. The stable fingerprint hashes the hardware envelope.

`--device cuda` is explicit. It checks the requested CUDA device through Torch, captures NVML/`nvidia-smi` identity and versions where available, and runs CUDA-specific probes. Failure to initialize that device is an error; there is no CPU substitution.

The current implementation does not yet benchmark all-reduce/all-gather or topology-sensitive peer links, and the CPU GEMM uses the local numerical stack. Treat those as capability gaps, not zeros.

## Stage A: static feasibility

For each `BackendCandidate`, the profiler estimates weights from parameter count and dtype and a conservative KV allocation for the maximum batch-token budget. It adds 10% headroom and rejects candidates exceeding declared memory. Rejections remain in the profile with estimates and reason, but no raw timing IDs.

## Stage B: hardware reuse

The profile binds the exact hardware JSON hash. The hardware fingerprint can be reused only when the caller decides the environment is compatible; the system never silently substitutes a different probe.

## Stage C: startup

Mock and HTTP mock profiling measure backend process readiness and cold probes. Real adapters distinguish startup/import/server readiness and capture managed-process output. Startup samples retain explicit warmup labels. The demo uses actual mock process readiness observations rather than only catalog constants.

## Stage D: service curves

Prefill is probed on a prompt grid containing 32, 128, 512, 2048 and the trace p95. Decode is probed over supported active-sequence counts 1, 2, 4 and 8. Each raw record includes the stage, token/batch/active shape, latency, peak memory when available, failure bit, sample index and seed.

The real GPU path supports:

- Transformers as an in-process correctness baseline;
- vLLM as a managed OpenAI-compatible server;
- SGLang as a managed OpenAI-compatible server.

The profiler decodes SSE incrementally, requires a terminal event, measures TTFT/ITL/E2E and bounds server startup, request duration and process teardown. `torch.profiler` can write Perfetto-compatible traces. Nsight Systems command generation is available when `nsys` is installed, but it is never a normal-CI dependency.

## Stage E: representative traffic

Load probes sample the trace across candidates, vary active concurrency, retain failures and summarize TTFT, ITL, E2E, availability and output-token goodput. The CPU demo profiles real HTTP mock processes with deterministic service behavior; the word “mock” describes the backend computation, not fabricated artifact rows.

## Budgets

`ProfilingBudget.reserve` debits projected seconds and `duration / 3600 * hourly_price` before each probe. A request exceeding either ceiling raises an error. The demo budget was 180 seconds and 0.20 USD; it recorded 41.460405 measured seconds and 0.013295 USD of modeled probe spend.

## Calibration

For each candidate, non-warm successful samples are grouped by prompt tokens or active sequences. Median points are made nondecreasing; linear interpolation is used inside the grid and a nonnegative terminal slope outside it. Residuals from those same grouped probe samples set a nominal p95 radius. All representative-load samples are then used to report prefill MAPE and empirical interval coverage; shuffling them does not create a holdout split. Startup median/p95 remain separate.

Accordingly, current `held_out_*` and `conformal_radius_ms` names describe intended semantics, not the statistical procedure actually implemented. A valid uncertainty claim requires disjoint fitting, calibration and test partitions (preferably split by request or run, not individual correlated token samples), finite-sample conformal quantiles and separate coverage checks for TTFT, ITL, E2E and cold start.

## Artifact layout

```text
profile.json           profile metadata, feasibility and summaries
measurements.jsonl     immutable raw samples
environment.json       process and package environment
hardware.json          copied hardware probe
workload.jsonl          copied normalized workload
```

## Measured CPU run

<!-- Metrics sources: ../artifacts/demo/hardware/local-cpu.json, ../artifacts/demo/profiles/qwen3-cpu-mocks/profile.json, ../reports/demo/evaluation.json -->

On an Apple M4 Pro with 12 logical CPUs and 24 GiB RAM, the probe measured median host-copy bandwidth of 74.72 GB/s and FP32 384 GEMM throughput of 1,637.29 GFLOP/s. Three mock candidates were feasible. Their profiled p95 TTFTs were 353.67 ms (balanced), 199.78 ms (fast) and 515.92 ms (economy). The selected balanced model's held-out prefill MAPE was 7.57%, but its interval coverage was only 54.17%; the latter is a negative calibration result and is reported as such.

No NVIDIA GPU was present. None of the GPU adapters or GPU performance numbers has been exercised on this machine.
