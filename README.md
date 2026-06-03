# SLOForge

SLOForge is an SLO-driven inference deployment compiler and adaptive runtime. It turns a typed model, workload, hardware catalog, price, profiling budget, and latency/goodput/availability constraints into a versioned `DeploymentPlan`, an uncertainty-aware Pareto frontier, deployable artifacts, and a hash-bound `EvidenceBundle`.

It is not an LLM proxy or a YAML generator. The compiler performs memory feasibility pruning, measured hardware and engine profiling, service-curve calibration, budgeted candidate search, topology/routing/admission/autoscaling lowering, deployment validation, replay, fault injection, and evidence-backed reporting. Modal and Baseten Truss are optional output targets; the local compiler, Rust simulator, and Rust gateway have no cloud dependency.

## Architecture

```mermaid
flowchart LR
    F[Model + workload + hardware + SLO + budget] --> IR[Typed versioned frontend IR]
    IR --> P[Feasibility and profiling passes]
    P --> M[Calibrated service curves and uncertainty]
    M --> O[Constrained multi-objective optimizer]
    O --> DP[DeploymentPlan + rejected alternatives]
    DP --> B{Backends}
    B --> L[Local / Docker]
    B --> K[Kubernetes / Helm]
    B --> C[Modal / Truss]
    DP --> R[Rust gateway + digital twin]
    R --> T[Replay + telemetry + chaos]
    T --> CT[Predictive controller / rollback]
    CT --> E[EvidenceBundle + static report + local UI]
```

Python owns profiling orchestration, statistical models, optimization, control experiments, diagnosis, lowering, and reports. Rust owns the asynchronous streaming data plane, bounded telemetry, load generation, and deterministic discrete-event simulation. Their primary boundary is canonical JSON over a stable subprocess protocol: it is debuggable, language-neutral, and keeps a Python crash outside the latency-sensitive gateway process. See [Architecture](docs/ARCHITECTURE.md) and [ADR 0001](docs/adr/0001-rust-python-boundary.md).

## SLOForge Fabric

Fabric extends the logical compiler into a topology-aware physical execution compiler and self-healing multi-node runtime. It maps tensor, pipeline, data, and expert-parallel ranks onto concrete hosts, GPUs, NUMA domains, NICs, and rails; lowers collectives and KV transfers onto measured paths; validates the resulting `PhysicalExecutionPlan` in a deterministic communication-aware Rust twin; and binds every decision to canonical evidence hashes.

```mermaid
flowchart LR
    D[DeploymentPlan] --> C[Physical compiler]
    M[ModelGraph] --> C
    T[TopologyGraph + provenance] --> C
    P[FabricProfile curves] --> C
    C --> X[PhysicalExecutionPlan]
    X --> S[Contention-aware Rust twin]
    X --> A[Runtime adapters]
    A --> O[Canonical Autopsy evidence]
    O --> H[Causal hypotheses]
    H --> S
    S --> R[RecoveryPlan]
    R --> G[Simulation / shadow / canary / promotion]
    G --> O
```

Autopsy is a structured causal debugger, not an LLM wrapper. It aligns cross-host clocks with explicit uncertainty, matches comparable workload slices, scores supporting and contradicting evidence, removes each suspected cause in the digital twin, and records why alternatives were rejected. The recovery controller cannot mutate production by default: disruptive proposals must pass simulation, shadow, minimum-sample canary, promotion, drain, and rollback guards.

The complete no-GPU path is reproducible locally:

```console
make fabric-check
make fabric-demo
make autopsy-demo
make forgeci-demo
make warmpath-demo
make extension-evaluation
```

`make fabric-demo` uses an explicit synthetic two-host/eight-GPU-per-host topology with NUMA domains, PCIe/NVLink contention domains, two network rails, MoE expert placement, disaggregated prefill/decode workers, and calibrated synthetic link curves. It starts the real Rust gateway, replays streaming traffic, injects a rail degradation and rank-local GPU slowdown, captures canonical evidence, evaluates counterfactual repairs, and executes the selected recovery through shadow and canary promotion. Its static report and UI reject missing or hash-mismatched artifacts.

The same workflow is exposed through the CLI:

```console
sloforge fabric discover --fixture two_node_infiniband \
  --output artifacts/fabric/topology.json
sloforge fabric benchmark --topology artifacts/fabric/topology.json --suite full --synthetic \
  --output artifacts/fabric/measurements
sloforge fabric model inspect --model synthetic --synthetic-moe \
  --output artifacts/models/model-graph.json
sloforge fabric compile --deployment-plan artifacts/plans/current.json \
  --topology artifacts/fabric/topology.json \
  --fabric-profile artifacts/fabric/measurements/fabric-profile.json \
  --model-graph artifacts/models/model-graph.json \
  --slo 'p95_ttft_ms<=250,p99_tpot_ms<=45' \
  --output artifacts/physical-plans/plan.json
sloforge fabric simulate --plan artifacts/physical-plans/plan.json \
  --topology artifacts/fabric/topology.json \
  --fabric-profile artifacts/fabric/measurements/fabric-profile.json \
  --trace workloads/mixed-bursty.jsonl --output artifacts/simulations/run-001
sloforge autopsy diagnose artifacts/autopsy/degraded-run.json \
  --baseline artifacts/autopsy/healthy-run.json \
  --output artifacts/autopsy/diagnosis.json
sloforge recovery plan --diagnosis artifacts/autopsy/diagnosis.json \
  --physical-plan artifacts/physical-plans/plan.json \
  --output artifacts/recovery/proposal.json
sloforge recovery apply --proposal artifacts/recovery/proposal.json --mode shadow-canary
```

ForgeCI uses repeated trials, bootstrap intervals, an explicit noise floor, practical significance, inconclusive classification, and a portable fixture repository to validate performance bisection without external projects. WarmPath profiles a real local artifact DAG and chooses among storage, cache, host-memory, lazy-restore, and warm-capacity strategies under readiness, cost, compatibility, capacity, and failure constraints.

On this checked-in development host, topology discovery and local memory profiling are hardware-backed, but NVIDIA GPU, NVLink, NCCL, RDMA, InfiniBand/RoCE, and multi-node runtime execution are unavailable. Those adapters are version-gated and fixture-tested; the repository does not present synthetic measurements as hardware results. See the [Fabric architecture](docs/fabric/ARCHITECTURE.md), [Autopsy diagnosis model](docs/autopsy/DIAGNOSIS.md), [recovery safety model](docs/recovery/SAFETY.md), [Fabric limitations](docs/FABRIC_LIMITATIONS.md), and [extension report](paper/fabric_extension/SLOFORGE_FABRIC.md).

## Three-minute CPU quickstart

Prerequisites are Python 3.11+, [uv](https://docs.astral.sh/uv/), Rust 1.93.1, and Node 22. Node is only used by the local artifact UI, but the full repository `bootstrap` and `check` targets include that UI. No NVIDIA GPU, model download, cloud account, or credentials are needed.

```console
make bootstrap
source .venv/bin/activate
make check
make demo
```

The demo starts three real Rust mock-server processes with distinct service curves, profiles them over HTTP/SSE, calibrates the digital twin, searches 540 configurations under a 24-proposal budget, preserves only the three exactly profiled shapes as measured anchors, emits a plan and Pareto set, starts the real Rust gateway, replays 120 streaming requests, injects slowdown/crash/cold-start faults, evaluates both controller policies in the calibrated Rust twin, and creates all five deployment targets.

The primary outputs are:

- `artifacts/demo/plans/qwen3-coding-agent.json`
- `artifacts/demo/evidence/evidence.json`
- `artifacts/demo/profiles/qwen3-cpu-mocks/measurements.jsonl`
- `artifacts/demo/simulator/replay.json` and `trace.json`
- `artifacts/demo/gateway/replay.json` and `metrics.prom`
- `artifacts/demo/benchmarks/serving-baselines.json`
- `reports/demo/evaluation.md`, `evaluation.html`, and `report-data.json`

The report generator verifies every indexed SHA-256 digest, reconciles the `EvidenceBundle` hash chain, and verifies the plan's canonical digest before reading metrics. It refuses tampered or internally stale input, so report values cannot drift away from the raw run silently.

## Cohesive CLI workflow

```console
sloforge trace validate workloads/coding-agent.jsonl
sloforge hardware probe --device cpu --output artifacts/hardware/local.json

sloforge profile \
  --model sloforge/mock-qwen3-0.6b-shape \
  --engines mock \
  --hardware artifacts/hardware/local.json \
  --trace workloads/coding-agent.jsonl \
  --budget-usd 0.20 \
  --output artifacts/profiles/local

sloforge optimize \
  --profile artifacts/profiles/local \
  --slo 'p95_ttft_ms<=250,p99_itl_ms<=45' \
  --objective minimize:cost_per_million_tokens \
  --output artifacts/plans/local.json

sloforge explain artifacts/plans/local.json
sloforge replay --plan artifacts/plans/local.json --trace workloads/coding-agent.jsonl
sloforge chaos --scenario scenarios/cpu-demo.yaml
sloforge export --plan artifacts/plans/local.json --target kubernetes --output generated/k8s
sloforge report \
  --evidence artifacts/demo/evidence \
  --artifact-index artifacts/demo/evidence/artifact-index.json \
  --output reports/local
```

An optimization result identifies the chosen non-dominated candidate, its fidelity (`measured` only for an exactly executed profile shape; otherwise `predicted`), uncertainty-adjusted constraint margins, the budgeted proposal history, and every rejected candidate with reason codes. `sloforge explain` names the dominant measured/modelled bottleneck and links back to evidence.

## GPU profiling

The default open-weight path is `Qwen/Qwen3-0.6B`, whose resolved immutable revision, safetensors checksum, parameter count, context limit, and Apache-2.0 license metadata are captured at profiling time. A CUDA probe never falls back to CPU and requires an explicit hourly price; zero is valid for owned hardware.

```console
uv sync --locked --extra dev --extra gpu-transformers
sloforge hardware probe \
  --device cuda \
  --hourly-price-usd 0 \
  --output artifacts/hardware/gpu.json

sloforge profile \
  --model Qwen/Qwen3-0.6B \
  --engines transformers,vllm,sglang \
  --hardware artifacts/hardware/gpu.json \
  --trace workloads/coding-agent.jsonl \
  --budget-usd "${SLOFORGE_GPU_BUDGET_USD:-0}" \
  --output artifacts/profiles/qwen3-gpu
```

Install the separate `gpu-vllm` and `gpu-sglang` extras in engine-compatible environments when those servers are requested. The Transformers path is the direct correctness baseline; vLLM and SGLang use managed OpenAI-compatible servers. TensorRT-LLM is isolated behind its optional adapter. Torch Perfetto traces, bounded server logs, package/CUDA metadata, and generated Nsight commands are retained. `make benchmark-gpu` records an explicit unavailable artifact rather than numbers when no NVIDIA device exists.

The Triton fused logits experiment is opt-in and never selected at runtime without a measured 10% benefit:

```console
uv run python benchmarks/gpu/benchmark_fused_logits.py \
  --device cuda --enable-triton-experiment \
  --output artifacts/gpu/fused-logits.json
```

## Gateway and controller

The Axum/Tokio gateway exposes OpenAI-compatible completion and chat paths with streaming SSE, bounded admission and stream buffers, cancellation propagation, pre-output-only retry, health checks, circuit breakers, four routing policies, structured errors, Prometheus metrics, and Chrome traces. Its contract suite covers partial output, malformed SSE, slow consumers, disconnects, saturation, retries, active health removal/restoration, and client-drop cancellation.

The controller records observations, drift forecast, all evaluated actions, safety guards, chosen action, canary state, observed outcome, and rollback reason. Minimum samples, hysteresis, cooldowns, and hourly change budgets prevent oscillation. A targeted test forces a materially different routing canary to violate promotion criteria and verifies rollback.

## Deployment targets

- Local and Docker: exact gateway config, cached model volume, explicit health checks, read-only/capability-dropped containers, and a Docker Compose smoke test.
- Kubernetes: Helm chart with gateway/backend Deployments and Services, probes, safe rollout, HPA stabilization, Prometheus annotations, optional GPU resources, security contexts, and a NetworkPolicy.
- Modal: offline-imported app pinned to Modal 1.5.3, current class concurrency/container/region/snapshot primitives, and engine-specific loading paths. SLOForge never calls `deploy`.
- Truss: engine-specific model bundle checked by Truss 0.18.24 and a vendored official-schema subset without Baseten credentials. This is community integration, not official Baseten support.

No exporter creates a paid resource. `SLOFORGE_GPU_BUDGET_USD` is the only authorization ceiling for optional paid experiments; absence means local-only execution.

## Verification and benchmarks

```console
make check          # Ruff, strict mypy, pytest/Hypothesis, fmt, clippy -D warnings, Rust tests, UI checks
make benchmark-cpu  # fresh CPU run plus repository-level evidence-derived report
make benchmark-gpu  # real path when available; explicit unavailable artifact otherwise
make docker-smoke   # Linux container build plus real streaming request
```

The checked seed-41 CPU report is an integration and methodology study over explicit mock inference, not a claim about Qwen GPU performance. The selected three-replica plan predicts 192.349 ms p95 TTFT, 3.471 ms p99 ITL and $2.47718/million tokens; its compiled-topology live replay completed 120/120 requests during scheduled faults at 170.429 ms p95 TTFT and 9.429 ms p99 ITL. The harsher faulted Rust simulation still missed 3/120 deadlines. In a separate matched controller-twin comparison, predictive control had 0 misses versus 2 for reactive control at $0.001458 greater simulated cost. Held-out prefill/decode MAPE was 8.401%/1.009%, with 91.667%/100% interval coverage. H1 and production H2 remain unestablished until independently budgeted real-engine GPU trials are executed; the report's default/manual/compiled comparisons are visibly labeled calibrated predictions.

See [Reproducibility](docs/REPRODUCIBILITY.md), [Limitations](docs/LIMITATIONS.md), [Related work](docs/RELATED_WORK.md), and the [paper-style report](paper/SLOFORGE.md). The local artifact explorer is documented in [ui/README.md](ui/README.md).

## Security and scope

This research infrastructure does not implement authentication, billing, or tenant isolation. Bind the gateway privately or put it behind an authenticated ingress. Model code and weights are supply-chain inputs; the real profiler accepts safetensors, pins resolved revisions, and hashes snapshots. Generated cloud code must be reviewed before deployment. See [Security](docs/SECURITY.md).

SLOForge is distinct from KV-state transport systems, operation profilers, and provider-routing control planes: its core contribution is compiling and validating an inference deployment from workload, hardware, cost, uncertainty, and SLO constraints.

## License

SLOForge is Apache-2.0. Model weights, engines, generated base images, and cloud SDKs retain their own licenses; their versions and model license metadata are recorded in evidence.
