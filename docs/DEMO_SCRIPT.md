# CPU demo script

The demo is an end-to-end systems exercise that needs no GPU or cloud credentials. It starts real local processes and sends real HTTP/SSE traffic; inference computation is deterministic mock behavior.

## Preparation

```bash
make bootstrap
make check
```

Confirm that ports on localhost are allowed and that no previous demo process is running. The demo chooses ephemeral ports and owns each child process group.

## Run

```bash
make demo
```

The equivalent direct entry point is:

```bash
uv run sloforge demo \
  --artifact-dir artifacts/demo \
  --report-dir reports/demo \
  --reset
```

Do not run the reset form against a directory containing unrelated data. The normal Make target uses the repository's dedicated demo locations.

## What happens

1. Generate a seeded 120-request bursty JSONL trace with interactive and long-context classes, multiple priorities and variable output lengths.
2. Probe CPU/memory and run raw host-copy and GEMM samples.
3. Start balanced, fast and economy Rust mock backends with distinct startup, prefill, decode, concurrency, price and failure parameters.
4. Profile their live HTTP behavior under a 180-second/0.20-USD modeled budget.
5. Fit monotonic prefill/decode curves and record held-out error/coverage.
6. Evaluate 540 configurations, record a 24-step acquisition history, retain only the three exactly profiled shapes as measured anchors, build the Pareto frontier and compile a strict plan/evidence bundle.
7. Start the Rust gateway with SLO-slack routing and replay all 120 requests.
8. During replay inject backend slowdown, crash, recovery and cold-start behavior.
9. Run the calibrated Rust simulator with slowdown, crash, cold recovery and startup slowdown.
10. Evaluate predictive and reactive controller actions in matched calibrated Rust-twin scenarios, then classify all eight labeled diagnosis faults plus eight no-fault control windows.
11. Generate local, Docker, Kubernetes, Modal and Truss exports offline.
12. Hash every indexed source artifact and derive reports only after verification.

## Presenter walkthrough

Start with the plan:

```bash
uv run sloforge explain artifacts/demo/plans/qwen3-coding-agent.json
jq '.engine,.replica_topology,.routing,.autoscaling,.predicted_metrics' \
  artifacts/demo/plans/qwen3-coding-agent.json
```

Show raw-to-derived evidence:

```bash
wc -l artifacts/demo/profiles/qwen3-cpu-mocks/measurements.jsonl
jq '.artifacts | length' artifacts/demo/evidence/artifact-index.json
jq . reports/demo/evaluation.json
```

Then inspect runtime behavior:

```bash
jq '.summary,.faults' artifacts/demo/gateway/replay.json
jq '.metrics,.provenance' artifacts/demo/simulator/replay.raw.json
jq '.predictive,.reactive' artifacts/demo/controller/evaluation.json
jq '.diagnosis_accuracy,.executions[].diagnosis' artifacts/demo/chaos/result.json
```

Open `reports/demo/evaluation.html`, `reports/demo/pareto.svg` and `reports/demo/controller.svg` in a browser. Load `artifacts/demo/simulator/trace.json` or `artifacts/demo/runtime/gateway-trace.json` in Perfetto.

## Expected interpretation

<!-- Metrics source: ../reports/demo/evaluation.json and ../artifacts/demo/gateway/replay.json -->

The seed-41 artifact selects 3 fast-mock replicas at concurrency 6, a 2,048-token batch limit, chunked prefill, SLO-slack routing and 2 warm replicas. Its calibrated prediction is 192.349 ms p95 TTFT, 3.471 ms p99 ITL and $2.47718/million tokens. Live compiled-topology replay completed 120/120 requests during scheduled faults at 170.429 ms p95 TTFT and 9.429 ms p99 ITL. The harsher faulted simulator completed 117/120 and missed 3 deadlines (97.5% attainment). In matched Rust-twin scenarios, predictive control had 0 misses versus 2 for reactive control at 0.001458 USD greater cost. Diagnosis matched 8/8 known injections and produced 0 false positives across 8 no-fault windows; mean classifier execution was 0.005666 ms.

The honest conclusion is not “the plan solved every SLO.” The live compiled-topology replay met the requested TTFT/ITL bounds, while the intentionally harsher simulator's crash/slowdown/startup scenario still missed three request deadlines and had 717.771 ms p95 TTFT. Compile plus validation preserves both results and their different fault models.

## Artifact validation

```bash
uv run sloforge report \
  --evidence artifacts/demo/evidence \
  --artifact-index artifacts/demo/evidence/artifact-index.json \
  --output /tmp/sloforge-report-check
```

Changing an indexed input should make this command fail its hash check. Do not edit demo artifacts merely to make displayed results prettier.

## Cleanup and troubleshooting

The runner terminates process groups on success and exception. If interrupted outside its handler, confirm with `ps` that no `sloforge-gateway mock-backend` remains before rerunning. Read `artifacts/demo/runtime/logs/*.stderr.log` for readiness or bind failures.

A missing GPU is expected. A cloud credential is unnecessary. Docker/Helm/Modal/Truss absence should affect only optional validators. Any silent device fallback is a bug.
