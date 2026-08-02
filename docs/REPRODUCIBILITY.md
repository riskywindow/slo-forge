# Reproducibility

SLOForge separates deterministic payloads from environment envelopes. Seeds control workload generation, profiling jitter, simulation, optimization, controller evaluation and chaos application; timestamps, ephemeral ports, process timing and host metadata may vary.

## Toolchain

- Python 3.11 or newer and `uv` 0.10.x, with dependency resolution pinned by `uv.lock`.
- Rust 1.93.1 for the full checkout, pinned by `rust-toolchain.toml`; crate metadata records a minimum supported Rust version of 1.85 and `Cargo.lock` pins dependencies.
- Node 22 and npm are required by `make bootstrap` and `make check` because the repository includes the static artifact UI. They are not runtime dependencies of the Python package or Rust binaries.
- GNU/BSD Make and a native C toolchain are needed for the full source build.
- Docker/Helm/Modal/Truss are optional validators, not normal Python/Rust test requirements.

From a clean checkout:

```bash
make bootstrap
make check
make demo
```

Activate the environment with `source .venv/bin/activate` before using bare `sloforge` commands shown below, or prefix them with `uv run --locked`.

If `make` is unavailable, the underlying development setup is equivalent to `uv sync --locked --extra dev --extra deploy`, `cargo build --workspace --locked`, Python checks, Rust format/lint/tests and `npm ci` plus UI checks. Use the checked-in Makefile as the command source of truth. Docker is needed only for `make docker-smoke`; a GPU and cloud credentials are not needed for bootstrap, checks, the CPU demo or offline exporter validation.

`make bootstrap` uses a checkout-local npm cache (`.cache/npm`) so its result does not depend on the permissions or contents of a user's global npm cache. The cache, virtual environment, Rust target tree and UI dependencies are ignored build outputs.

## Deterministic inputs

The CPU demo uses seed 41, a 120-request bursty workload, three mock-backend parameter sets and [`scenarios/cpu-demo.yaml`](../scenarios/cpu-demo.yaml). Reusing those inputs should preserve trace contents, candidate/config IDs, simulator event ordering and fault labels. Wall-clock HTTP timings will vary, so the live replay's measured latencies are not byte-for-byte deterministic.

The simulator has a stronger guarantee: identical canonical input and binary version produce identical output payload, tested byte for byte. Its provenance contains input SHA-256, service-curve ID, measurement URI and seed.

## Evidence chain

The demo report is derived through:

```text
measurements.jsonl + hardware.json + workload.jsonl
                -> profile -> service curves -> optimization
                -> DeploymentPlan + EvidenceBundle
                -> simulator/gateway/controller/chaos artifacts
                -> SHA-256 artifact index -> report
```

[`artifact-index.json`](../artifacts/demo/evidence/artifact-index.json) contains 15 named inputs. Report generation recomputes every hash and checks the canonical plan digest before reading derived metrics. The command transcript is archived at [`artifacts/demo/command-transcript.txt`](../artifacts/demo/command-transcript.txt).

## Recorded CPU environment

<!-- Facts source: ../artifacts/demo/hardware/local-cpu.json and ../artifacts/demo/profiles/qwen3-cpu-mocks/environment.json -->

The example run currently described by the generated local artifacts was captured on macOS 15.6.1 arm64, Apple M4 Pro, 12 logical CPUs and 24 GiB memory, using Python 3.12.7. Demo measurements are build outputs rather than source: run `make demo` before following links under `artifacts/demo` or `reports/demo`. CI uploads both directories as a 14-day evidence artifact keyed by commit. The profile environment could not enumerate packages because that virtual environment lacked `pip`; the error is preserved in `environment.json`. Dependency versions remain available from `uv.lock`, but this is weaker than a successful runtime package inventory.

The hardware probe's median samples were 74.72 GB/s host copy and 1,637.29 GFLOP/s for its 384-square FP32 GEMM. These numbers are host-specific and should not be used as a baseline for another machine.

## Re-running individual stages

```bash
sloforge trace generate --seed 41 --count 120 --output /tmp/workload.jsonl
sloforge trace validate /tmp/workload.jsonl
sloforge hardware probe --device cpu --samples 5 --output /tmp/hardware.json
sloforge profile --model Qwen/Qwen3-0.6B --engines mock \
  --hardware /tmp/hardware.json --trace /tmp/workload.jsonl \
  --budget-usd 0.20 --duration-s 180 --seed 41 --output /tmp/profile
sloforge optimize --profile /tmp/profile \
  --slo 'p95_ttft_ms<=250,p99_itl_ms<=45,availability>=0.70' \
  --objective minimize:cost_per_million_tokens --trials 24 --seed 41 \
  --output /tmp/plan.json
```

For simulator-only replay:

```bash
sloforge replay --plan artifacts/demo/plans/qwen3-coding-agent.json \
  --trace artifacts/demo/workloads/coding-agent.jsonl \
  --faults --seed 41 --output /tmp/replay.json
```

## GPU protocol

No GPU result is included in the current evidence. On a compatible NVIDIA host, create a fresh CUDA hardware probe, install exactly one or more explicit engine extras, profile Transformers/vLLM/SGLang and write to a new artifact directory. Never reuse the CPU/mock evidence URI or rename mock results as GPU results. Capture `nvidia-smi`, driver/CUDA versions, model revision/checksum, package inventory, raw samples and Perfetto traces.

`make benchmark-gpu` should skip with an explicit unavailable reason on a CPU host. A skip is not a passing GPU benchmark.

## Comparing runs

Compare schema version, seed, workload digest, hardware fingerprint, model revision/checksum, engine/runtime version and Git commit before comparing metrics. Report medians, tail percentiles and intervals across multiple seeds where practical. A single changed dimension invalidates causal claims unless the experiment design intended that change.
