# Final adversarial review

Review date: 2026-08-01 (America/Los_Angeles)

Scope: compiler/optimizer semantics, calibrated uncertainty, regenerated CPU artifacts, gateway and simulator validation, controller/fault methodology, evidence integrity, offline exporters, security boundaries, and the completion contract. This review inspected the implementation and raw artifacts directly; it does not inherit the conclusions of the older `docs/REVIEWS.md` audit.

## Outcome

No unresolved high-severity defect remains in the exercised CPU compiler, gateway, simulator, evidence, or report path after the patches below. The initially regenerated run exposed three high-severity semantic errors and one reproducibility error; all were fixed in source and covered by targeted tests.

The required final regeneration has since completed from immutable source revision `2c889e6956ac73a1a530f1abb1d7407f70219ffe`. The optimizer selected three `mock-fast` replicas at concurrency six, the live gateway config contained exactly those three distinct processes and the compiled SLO-slack policy, all 120 streamed requests completed through five live injected fault operations, and actual p95 TTFT was below the requested 250 ms. The final index contains 27 artifacts with zero digest mismatches and reconciles all 24 `EvidenceBundle` hashes.

## Resolved high-severity findings

### H-01: isolated probes overrode representative-workload feasibility

The optimizer previously selected a one-replica candidate as “measured” and SLO-feasible even though its own representative-workload prediction and runtime replays missed TTFT by orders of magnitude. Stage-E measurements are isolated request-shape probes and do not observe compiler-added topology or arrival-induced queueing.

Fixed at `python/sloforge/optimizer/core.py:377-418`: direct measurements remain calibration/provenance, while constraints and objectives use representative-workload predictions. Pareto comparisons and plan/report serialization use the same prediction semantics (`python/sloforge/compiler/plan.py:253-281`, `python/sloforge/reports/generate.py:475-485`). Regression tests recompute every serialized constraint margin and assert plan metrics equal the selected representative prediction.

### H-02: acquisition promoted cheap, grossly infeasible candidates

The old cost-scaled rejection penalty ranked very cheap configurations with multi-second SLO misses above feasible candidates; a special “measured” bypass then selected outside the 24-trial promotion budget.

Fixed at `python/sloforge/optimizer/core.py:397-408,591-596`: normalized constraint violations dominate constrained acquisition, and every selectable candidate must be in the promoted trial set. On the inspected profile, exhaustive, successive-halving, and the uncertainty-aware strategy all returned feasible candidates, and the primary selection was one of its 24 promoted configurations.

### H-03: live replay did not exercise the compiled plan

The demo gateway previously routed across all three heterogeneous candidates, while the plan selected a different single-backend topology; replay also compressed arrivals by 5.6x relative to optimization. This made runtime validation incomparable with the compiled configuration.

Fixed at `python/sloforge/demo.py:515-617`: the demo starts distinct processes for only the selected backend, preserves selected replica count/concurrency/routing, and replays the original arrival process at `time_scale=1.0`. Fault scheduling now supports any non-empty replica count. The inspected corrected runtime config contained the selected three replicas and no rejected backend variant.

### H-04: “calibrated” workload interval used a service-only radius

The preliminary corrected artifact reported extremely poor prefill interval coverage. Stage-D service-only residuals calibrated the radius, while Stage-E load TTFT—including runtime/queue overhead—was the evaluation target. Using that radius for SLO selection contradicted the uncertainty-aware claim.

Fixed at `python/sloforge/models/service_curve.py:80-85,137-225`: Stage-E load records are deterministically shuffled and split into disjoint calibration and test halves; TTFT and ITL radii are expanded with finite-sample load-calibration residuals; MAPE and empirical coverage use only the untouched test half. Against the same preliminary raw profile, mock-fast test coverage became 91.67% for both TTFT and ITL (12 test samples), with no test sample reused for calibration. The selected configuration remained feasible with a 34.3 ms uncertainty-adjusted TTFT margin.

### H-05: controller twin inputs overwrote one another

Predictive and reactive replays both wrote `controller/simulator-scenario.json`, so the second run destroyed the first run’s exact reproducibility input.

Fixed at `python/sloforge/runtime/simulator.py:147-151`: every replay writes `<output-stem>.scenario.json`. The final artifact index should include or retain both scenario files alongside their raw outputs.

### H-06: report trusted two independent hash chains without reconciling them

The report verified the artifact index and the plan digest but did not ensure `EvidenceBundle.artifact_hashes` agreed with the indexed files.

Fixed at `python/sloforge/reports/generate.py:411-456`: every named evidence digest must map to the expected indexed artifact and match its SHA-256 before parsing results. This prevents a freshly indexed but internally stale evidence bundle from producing a report.

## Remaining medium-severity boundaries

### M-01: generated Docker/Kubernetes engine image is an operator contract, not a built artifact

`ExportContext.image` defaults to `ghcr.io/sloforge/runtime:0.1.0` (`python/sloforge/exporters/core.py:41`), and Docker assigns that same image to the gateway and the selected inference command (`python/sloforge/exporters/core.py:140-176`). Kubernetes likewise derives one image for both components (`python/sloforge/exporters/core.py:278-296`). Structural tests prove there is no silent mock substitution, and the repository’s source Docker Compose is genuinely smoke-tested, but generated vLLM/SGLang/Transformers manifests are not proven runnable from a published engine-bearing OCI image. A release should add separate digest-pinned gateway/backend image inputs and engine-image smoke jobs before calling those generated GPU deployments turnkey.

### M-02: controller actuator is the Rust twin, not the live gateway

The guarded predictive controller produces canary/rollback decision records and its actions are applied to the calibrated Rust simulator. It is not a long-running process that mutates the live gateway/backend fleet. Thus H3 is a reproducible policy/twin comparison, not evidence of a production actuator. Kubernetes/cloud actuator adapters and live promotion/rollback acknowledgement remain necessary before deployment automation is enabled.

### M-03: singleton benchmark estimates carry a nominal confidence field

Evidence benchmark summaries use one run, equal lower/point/upper values, and `confidence=0.95` (`python/sloforge/demo.py:255-275`). Raw per-request measurements remain available, but a degenerate single-run interval is not a 95% statistical confidence interval. Future evaluation should aggregate multiple seeds/runs and bootstrap the requested benchmark quantities, or extend `MetricEstimate` with an explicit exact/single-run interval kind.

### M-04: OpenTelemetry output is archival OTLP-shaped JSON

The repository emits validly structured resource/span JSON and direct Chrome traces, but the live gateway does not deliver spans to an OTLP collector. Existing component documentation states this boundary accurately. Production deployments need an actual OpenTelemetry exporter and collector integration.

### M-05: optional execution paths remain unexercised on this host

No NVIDIA GPU, paid cloud resource, Kubernetes cluster, or credentials were available. Transformers/vLLM/SGLang/TensorRT-LLM profiling and Modal/Truss deployment were validated through bounded adapters, current-package offline imports/schema validation, and explicit unavailable artifacts—not real performance or deployment. The GPU report must continue to contain zero measurements until hardware execution occurs.

## Security and artifact checks

- The gateway keeps bounded admission and stream queues, propagates cancellation, refuses backend redirects/credential-bearing URLs, caps error/non-streaming bodies, gates half-open circuit-breaker probes, and terminates process groups. Contract tests cover malformed/partial SSE, disconnects, queue saturation, slow consumers, cancellation, retry-before-output, and breaker recovery.
- Generated local/Docker endpoints bind to loopback; containers drop capabilities, disallow privilege escalation, and use read-only filesystems. Kubernetes disables service-account token mounting, applies non-root/seccomp controls, and restricts backend ingress. Authentication/TLS remain trusted-ingress responsibilities.
- The final artifact index contains 27 entries with zero digest mismatches. Evidence includes five measurement-stage references, held-out calibration/error records, 24 optimizer decisions, 513 rejected candidates, four benchmark records, environment versions, raw result digests, and the immutable post-review source commit.
- Modal generation now drops the IR-only `local` region rather than creating an unschedulable Modal app. Offline generation never calls `deploy`, `remote`, or a paid resource API.
- Core-source scan found no `NotImplementedError`, TODO/FIXME core stubs, silent device/engine substitution, or empty exception handlers after the review patches.

## Verification performed by this review

- `uv run ruff check python tests` — passed.
- `uv run mypy python/sloforge` — passed for 43 source files.
- `uv run pytest -q tests/python/test_profiler_optimizer.py tests/python/test_compiler_exporters.py tests/python/test_evidence_reports.py tests/python/test_controller_faults.py` — 23 passed.
- Refit/calibration probe on the regenerated raw profile — TTFT coverage 91.67%, ITL coverage 91.67%, disjoint held-out count 12; selected plan remained feasible under the widened intervals.
- Final live demo artifact audit — selected plan/topology matched gateway runtime; 120/120 streaming requests completed; five required live fault operations were recorded; 27/27 indexed SHA-256 values and 24/24 evidence hashes matched.
- Final release gates — `make check`, `make demo`, `make benchmark-cpu`, `make benchmark-gpu`, `make docker-smoke`, the report round trip, clean-archive `make bootstrap`, stale-claim scan, path/secret scan, and `git diff --check` passed. Python recorded 67 passes with three expected GPU/Torch skips, Rust recorded 62 passes, and the UI recorded 11 passes plus a production build.

All final metric and resume claims were regenerated from that post-review artifact set. The remaining medium-severity boundaries above are retained in `FINAL_REPORT.md` and `docs/LIMITATIONS.md`.
