# Optimizer

The optimizer performs constrained multi-objective search over deployment configurations. It preserves uncertainty and rejections rather than reducing the search to a single score.

## Search space

For every statically feasible backend candidate, search spans:

- replica count from one through the request limit;
- concurrency from supported values among 1, 2, 4 and the backend maximum;
- maximum batched tokens of 512, 2048 or 8192;
- chunked prefill on or off where valid;
- round robin, estimated earliest finish and SLO-slack routing;
- a derived warm-replica count based on startup time.

Configuration IDs are the first 12 hex characters of canonical SHA-256 over the parameter payload, so identical inputs produce identical IDs.

## Metrics

Each evaluation contains point predictions and radii for p95 TTFT, p99 ITL, p95 E2E, deadline-aware token goodput, throughput, availability, cost per million tokens and p95 cold start. Cost uses declared replica hourly price divided by predicted output throughput. Availability derives from candidate failure rate and replication. These are scoped analytical estimates, not promises.

## Queue and service model

Prefill and decode are predicted separately from calibrated monotonic curves. Trace prompt/output quantiles determine service shape. Capacity is replica count × active concurrency divided by service E2E. Utilization feeds a queue penalty that grows sharply near saturation. Explicit factors approximate routing, chunking and batching. Warm-replica fraction increases cold-start exposure.

## Hard constraints

Compact expressions such as:

```text
p95_ttft_ms<=250,p99_itl_ms<=45,availability>=0.70
```

are parsed into typed constraints. For upper bounds the conservative value is `center + safety * radius`; for lower bounds it is `center - safety * radius`. Negative margins become human-readable rejection reasons. The selected candidate must satisfy every conservative hard constraint.

## Multi-fidelity acquisition

All configurations receive predicted evaluation. An acquisition score combines direction-aware objective improvement, one quarter of objective uncertainty and a penalty for constraint misses. The top `trial_budget` configurations define the proposal history. Direct representative-load measurements are attached only when a candidate exactly matches the profiled one-replica, concurrency-one, non-chunked, round-robin, 2,048-token shape; the CPU run therefore has three measured anchors and every scaled or policy-modified variant remains predicted. Optimizer history records proposal order, fidelity, score, incumbent and reason.

## Pareto frontier

Candidate A dominates B only if it is no worse in all latency/cost/cold-start metrics and no worse in goodput/throughput/availability, with at least one strict improvement. The frontier is computed over feasible evaluations and retains the evaluation provenance and fidelity.

## Baselines

The result records exhaustive, seeded random, successive-halving-style and uncertainty-aware outcomes. The budgeted strategies reveal predictions from the same candidate table; they do not execute independent configuration trials. Consequently, they are an algorithm smoke comparison rather than a measurement-efficiency experiment. A `null` best result means the budgeted strategy found no feasible point; it is not converted to a win or silently dropped.

## CPU evaluation

<!-- Metrics source: ../artifacts/demo/optimization/result.json and ../reports/demo/evaluation.json -->

The CPU/mock run enumerated 540 configurations: 3 exact profile shapes are measured anchors and 537 topology/policy variants remain predictions. Twenty-seven evaluations were feasible in the full table; the 24 budgeted acquisition proposals formed the feasible selection pool and yielded a 4-point non-dominated frontier. It selected predicted `cfg-aad9cd4cfa41`: fast mock backend, 3 replicas, concurrency 6, 2,048 batched tokens, chunked prefill, SLO-slack routing and 2 warm replicas. Its prediction was 192.349 ms p95 TTFT, 3.471 ms p99 ITL, 740.088 output/goodput tokens/s and $2.47718/million tokens. TTFT uncertainty radius was 68.518 ms; with the configured half-radius safety multiplier, its conservative TTFT margin was 23.392 ms.

Exhaustive, successive halving and uncertainty-aware search found the same $2.47718 objective in this small synthetic regime; seeded random reached $3.30291 in 24 proposals. Because every strategy consumed one shared prediction table rather than independently purchased measurements, this does not demonstrate H1 or reduced GPU minutes. Live compiled-topology replay met the requested TTFT/ITL bounds, while the harsher faulted simulator missed 3/120 deadlines, showing why the evidence scope and fault model remain part of deployment validation.

## Reproduction

```bash
sloforge optimize \
  --profile artifacts/demo/profiles/qwen3-cpu-mocks \
  --slo 'p95_ttft_ms<=250,p99_itl_ms<=45,availability>=0.70' \
  --objective minimize:cost_per_million_tokens \
  --trials 24 --seed 41 \
  --output /tmp/sloforge-optimization.json
```

Use the CLI help in the installed revision as the source of truth for flags. Input hashes and the seed, not wall-clock generation timestamps, determine the reproducible payload.
