# Performance evidence

Genesis accepts performance only from repeated raw samples bound to a benchmark contract. Proposal scores, simulator predictions, and isolated point estimates cannot pass this gate.

## Contract and calculation

The implemented `BenchmarkContract` records benchmark/metric/unit, optimization direction, workload fingerprint, hardware fingerprint, software-manifest hash, warmup count, practical-significance threshold, measured noise floor, bootstrap rounds and confidence level. At least seven positive samples are required for both baseline and candidate.

`evaluate_performance` computes baseline and candidate medians, percent improvement in the declared direction, a pairwise probability-of-superiority effect size, and a seeded bootstrap confidence interval over median improvement. It returns:

- `passed` only when the complete interval exceeds both practical significance and noise floor;
- `failed` when the complete interval establishes a practically significant regression;
- `inconclusive` when the interval overlaps either threshold.

The result retains all supplied samples, a deterministic randomized run-order vector derived from the seed and sample counts, seed, contract, statistics, effect size and rationale. A lower candidate point estimate alone is insufficient.

## Benchmark harness responsibilities

The statistical function evaluates already collected samples, so it cannot prove that collection followed the attached order vector. The harness must derive and execute the same seeded randomized order before measurement, apply equal warmup, synchronize asynchronous devices, retain slow/outlier samples, and capture environment, clocks, affinity, power state, failures, fallback and quality mode. Capsule validation separately binds raw sample artifacts to benchmark, software, workload and hardware digests and recomputes declared summary statistics.

ServingSynthBench CPU smoke records real `perf_counter_ns` samples and audits sample completeness, but it does not claim GPU speedup, cost or energy improvement. Hardware results must remain separately identified.

## Exercised and unexercised scope

Tests demonstrate a clearly separated passing case, a noisy inconclusive case, minimum sample rejection, and deterministic bootstrap output. They do not establish that any particular generated runtime is faster on real hardware.

The implementation does not currently calculate regression probability directly, perform paired or blocked bootstrap, characterize the noise floor itself, detect thermal/clock drift, or choose sample count from a power analysis. Those are harness and evaluation-plan responsibilities. Simulator prediction and prediction error can be capsule evidence, but are not fields in this focused statistical result.
