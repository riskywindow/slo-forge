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

The statistical function evaluates already collected samples, so it cannot prove collection order by itself. The harness must persist an execution ordinal for every baseline and candidate trial. Capsule validation reconstructs the exact combined order, aligns pairs by trial identifier rather than file order, binds raw samples to benchmark/software/workload/hardware digests, and recomputes medians, tails, bootstrap bounds, effect size and regression probability. A promotion-scoped performance claim is rejected when paired regression probability exceeds five percent or when the candidate confidence bound does not conservatively clear the practical/noise threshold.

ServingSynthBench CPU smoke runs the generated Genesis runtime in the strict local sandbox, records real `perf_counter_ns` samples from the bounded subprocess harness, reopens their hashes during comparison, and audits token truthfulness, seeds, task/package identity, execution surfaces, timing consistency, sample completeness and global randomized order. Other named CPU lanes are explicitly labeled reference-order surrogates rather than independent engine ablations. The smoke suite does not claim GPU speedup, cost or energy improvement. Hardware results must remain separately identified.

## Exercised and unexercised scope

Tests demonstrate a clearly separated passing case, a noisy inconclusive case, minimum sample rejection, and deterministic bootstrap output. They do not establish that any particular generated runtime is faster on real hardware.

The generic `evaluate_performance` helper does not currently perform a blocked bootstrap, characterize the noise floor itself, detect thermal/clock drift, or choose sample count from a power analysis. The independent capsule validator does calculate trial-aligned regression probability and applies the conservative promotion gates described above. Simulator prediction and prediction error can be capsule evidence, but the local CPU capsule intentionally contains no accepted benchmark comparison and no performance-improvement claim.
