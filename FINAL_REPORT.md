# SLOForge final implementation report

Release verification date: 2026-08-01 (America/Los_Angeles)

Evidence source revision: `2c889e6956ac73a1a530f1abb1d7407f70219ffe`

## What was implemented

SLOForge is a working SLO-driven inference deployment compiler and adaptive runtime, not a scaffold. It accepts typed model, engine, hardware, workload, budget, and SLO inputs; profiles candidate runtimes; calibrates service curves; searches a constrained configuration space; emits a versioned `DeploymentPlan`; validates it with a deterministic Rust digital twin and live Rust streaming gateway; and produces a hash-linked `EvidenceBundle`, offline deployment artifacts, telemetry, plots, and reports.

The repository includes:

- A strict Pydantic/Serde `sloforge.io/v1alpha1` IR, JSON Schema, v1 migration, canonical serialization, golden fixtures, and cross-language conformance tests.
- A deterministic Rust discrete-event simulator for arrivals, priority queues, warm/cold replicas, separate prefill/decode, approximate continuous batching, deadlines, cancellation, failures, degraded capacity, control actions, routing, canaries, and cost.
- An async Rust OpenAI-compatible gateway with SSE, cancellation propagation, bounded admission/stream queues, backpressure, health checks, circuit breakers, safe retry rules, timeouts, graceful shutdown, metrics, and tracing.
- CPU/GPU hardware probes and a budgeted staged profiler with static memory feasibility, startup phases, service curves, Stage-E load measurements, environment capture, raw samples, Perfetto export, and explicit Transformers/vLLM/SGLang/TensorRT-LLM adapters.
- Systems-aware monotonic service models, disjoint finite-sample load calibration/test splits, uncertainty propagation, exhaustive/random/successive-halving baselines, uncertainty-aware acquisition, constrained selection, a Pareto frontier, and rejected-alternative explanations.
- Guarded predictive and reactive controllers evaluated on separate identically seeded Rust twins, with cooldowns, hysteresis, canary/promotion/rollback records, retained scenarios, and raw traces.
- Eight executable YAML fault classes, negative windows, a counter/event-order bottleneck classifier, confusion-matrix evidence, and actual classifier timing.
- Local, Docker, Kubernetes/Helm, Modal 1.5.3, and Baseten Truss 0.18.24 exporters with capability checks and offline validation. No exporter makes a network deployment in normal tests.
- Prometheus metrics, structured logs, OpenTelemetry-shaped archival traces, Chrome/Perfetto traces, a static Markdown/HTML report, an artifact-backed TypeScript UI, benchmark plots, GPU-ready commands, and a paper-style technical report.

The primary Python/Rust boundary is a versioned JSON subprocess protocol. It keeps the data-plane binaries independently deployable, makes traces and requests replayable, and avoids coupling the optimizer to Python extension ABI builds.

## Architecture summary

`ModelSpec + EngineSpec + HardwareSpec + WorkloadSpec + SLOSpec + BudgetSpec` flow through feasibility pruning, profiling, calibration, candidate generation, uncertainty-aware optimization, topology/policy construction, IR validation, backend lowering, runtime replay, control evaluation, and evidence/report generation. Python owns orchestration, statistical modeling, optimization, control experimentation, and exporters. Rust owns the gateway, simulator, load generator, protocol validation, and latency-sensitive telemetry. Every report input is located through an artifact index and checked against both its index digest and the `EvidenceBundle` hash chain.

## Release commands and results

| Command | Final result |
|---|---|
| clean archive of revision `2c889e6`; `make bootstrap` | Passed: new locked Python environment with 114 packages, all five Rust crates built from `Cargo.lock`, and 231 UI packages installed with zero npm vulnerabilities. |
| `make check` | Passed: Ruff format/lint, strict mypy over 43 Python source files, 67 Python tests with 3 expected GPU/Torch skips, Rust format/clippy with warnings denied, 62 Rust tests, UI type/lint, 11 UI tests, and production build. |
| `make demo` | Passed from a clean demo state without GPU or credentials; generated the workload, profiles, model, 540-candidate optimization, compiled live topology, five live faults, Rust twin replays, controller study, eight-class chaos study, five deployment targets, evidence, reports, telemetry, and plots. |
| `sloforge report` round trip | Passed; all 27 artifact-index digests and all 24 evidence hashes reconciled before report generation. |
| `make benchmark-cpu` | Passed; reran the complete CPU benchmark workflow and promoted machine-readable evaluation/report outputs under `reports/`. |
| `make benchmark-gpu` | Passed its no-GPU contract: status is `unavailable`, the measurement list is empty, and exact GPU commands/artifact schema are emitted. |
| `make docker-smoke` | Passed; digest-pinned Linux images built, backend and gateway became healthy, and a real streamed request reached `[DONE]`; cleanup completed. |
| package/audit/secret/path scans | Wheel and sdist build/import passed; npm and Python dependency audits reported no known vulnerabilities; no credential patterns, core empty implementations, or developer-absolute paths were found in release evidence. |

## CPU benchmark and demo results

All values below are read from `reports/demo/evaluation.json`, `reports/demo/report-data.json`, or the indexed raw artifacts. They describe deterministic mock inference backends on this Apple Silicon CPU host and must not be interpreted as Qwen3/GPU performance.

- Selected `cfg-aad9cd4cfa41`: three `mock-fast` replicas, concurrency 6, 2048 maximum batched tokens, chunked prefill, two warm replicas, and SLO-slack routing.
- Search: 540 evaluated configurations, 27 uncertainty-adjusted feasible configurations, 3 direct measured anchors, 24 promoted optimizer decisions, 513 rejected alternatives, and a 4-point Pareto frontier.
- Selected representative prediction: 192.349 ms p95 TTFT, 3.471 ms p99 ITL, 994.243 ms p95 end-to-end latency, 740.088 tokens/s goodput, 99.866% availability, and 2.477182 USD per million tokens.
- Prediction radii: 68.518 ms TTFT, 1.215 ms ITL, and 0.867014 USD per million tokens. Conservative feasibility margins were 23.392 ms TTFT, 40.922 ms ITL, and 0.2812 percentage points availability.
- Held-out model evaluation: prefill MAPE 8.401% with 91.667% empirical interval coverage; decode MAPE 1.009% with 100% coverage. Each final held-out metric used 12 samples disjoint from calibration.
- Live compiled topology: 120/120 streamed requests succeeded; p95 TTFT 170.429 ms, p99 ITL 9.429 ms, and p95 end-to-end latency 290.455 ms. Five scheduled live fault operations exercised slowdown, crash/recovery, and cold-start behavior.
- Faulted Rust simulator replay: 117/120 requests completed, 3 deadline misses, 97.5% SLO attainment, p95 TTFT 717.771 ms, p99 ITL 12.697 ms, and 0.030026 USD modeled cost. This negative tail result is retained.
- Controller twins: predictive had 0 deadline misses at 0.036939 USD; reactive had 2 misses at 0.035481 USD. The predictive policy spent about 4.1% more modeled cost in this one-seed run.
- Fault diagnosis: 8/8 injected classes were identified, with 0/8 false positives across negative windows and 0.005666 ms mean measured classifier execution time. This is a deterministic closed-set result.

H1 and H2 are not claimed as established: no fresh real-GPU experiment campaign was authorized, and mock-derived default/manual comparators cannot establish engine performance. H3 is a one-seed digital-twin result, and H4 is closed-set. The reports retain these qualifications and do not fabricate a win.

## Measured limitations and unexercised paths

- This host has no NVIDIA GPU. Real Transformers, vLLM, SGLang, TensorRT-LLM, CUDA, Triton, Nsight, P2P, or collective benchmarks were not run. The GPU report contains zero measurements.
- `SLOFORGE_GPU_BUDGET_USD` was absent. No Modal, Baseten, Kubernetes, or other paid/cloud resource was created. Modal and Truss are current-version offline validations, not deployment smoke tests.
- Generated GPU Docker/Kubernetes manifests require an operator-provided engine-bearing image. The repository's source Docker Compose path, not a GPU image, is what was smoke-tested.
- The controller applies actions to the calibrated Rust digital twin; it does not mutate a production fleet or receive external actuator acknowledgements.
- Benchmark summaries contain a single deterministic run. Raw per-request data are preserved, but singleton lower/point/upper values are not evidence of a multi-run 95% confidence interval.
- OpenTelemetry output is archival OTLP-shaped JSON rather than live export to a collector.
- The Python wheel is pure Python; complete gateway/simulator operation requires the Rust binaries on `PATH` or the source checkout.
- The mock-backend service curves and selected topology do not claim generalization to unseen hardware, engines, or model weights.

## Artifact locations

- Compiled plan: `artifacts/demo/plans/qwen3-coding-agent.json`
- Evidence and verified artifact index: `artifacts/demo/evidence/evidence.json`, `artifacts/demo/evidence/artifact-index.json`
- Raw profile measurements: `artifacts/demo/profiles/qwen3-cpu-mocks/measurements.jsonl`
- Optimizer candidates/frontier: `artifacts/demo/optimization/result.json`
- Live gateway replay and trace: `artifacts/demo/gateway/replay.json`, `artifacts/demo/runtime/gateway-trace.json`
- Faulted simulator raw output and trace: `artifacts/demo/simulator/replay.raw.json`, `artifacts/demo/simulator/trace.json`
- Controller decisions, scenarios, raw twin outputs, and traces: `artifacts/demo/controller/`
- Chaos results: `artifacts/demo/chaos/result.json`
- Local/Docker/Kubernetes/Modal/Truss outputs: `artifacts/demo/exports/`
- Human and machine reports: `reports/demo/evaluation.md`, `reports/demo/evaluation.html`, `reports/demo/evaluation.json`, `reports/demo/report-manifest.json`
- Independent `benchmark-cpu` raw evidence and source reports: `artifacts/cpu-demo/`, `reports/cpu-demo/`; promoted benchmark summary and report outputs: `reports/cpu-benchmark.json`, `reports/evaluation.md`, `reports/evaluation.html`, `reports/evaluation.json`
- Prometheus/OpenTelemetry/Chrome outputs and plots: `reports/demo/metrics.prom`, `reports/demo/otel-traces.json`, `reports/demo/trace.json`, `reports/demo/pareto.svg`, `reports/demo/controller.svg`
- GPU unavailable record and reproduction commands: `reports/gpu/`
- Review records: `FINAL_ADVERSARIAL_REVIEW.md`, `SECURITY_REVIEW.md`, `CLEANROOM_REVIEW.md`, `CLEAN_REVISION_AUDIT.md`, and `docs/REVIEWS.md`

The checked evidence is reproducible, repository-relative, and bound to the exact source revision that generated it. Later documentation/artifact publication commits do not change that source provenance.
