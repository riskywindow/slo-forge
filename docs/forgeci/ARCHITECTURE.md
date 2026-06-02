# ForgeCI architecture

ForgeCI is a reproducible inference performance-regression runner. It executes a
versioned benchmark matrix, separates warmup and measurement, classifies robust
effects, bisects a local Git history, minimizes the benchmark, and writes an
upstream-ready evidence report.

## Matrix and execution

Schema `1.0.0` defines repository/revision, build commands, environment,
hardware requirement, benchmark input, metrics, warmups, repetitions, maximum
repetitions, retry count, bootstrap rounds, seed, budget, and timeout. Metrics
declare direction, practical threshold, significance, and allowed noise.

Before a run, ForgeCI observes architecture, CPU, memory, GPU inventory, RDMA,
and optional topology fingerprint and fails a mismatched requirement. Commands
run in bounded process groups with timeout and cleanup. Every trial gets a derived
seed and writes stdout/stderr. JSON metrics are parsed against the exact declared
metric names. Sensitive environment values are redacted from stored commands and
output.

`RunRecord` retains warmups, measurement trials, summaries with raw samples,
environment and hardware manifest, warnings, artifact directory, and tree hash.
`verify_run_artifact` recalculates that evidence hash.

## Comparison and bisection

Comparison uses robust, seeded bootstrap inference and produces regression,
improvement, unchanged, inconclusive, flaky, or failed for every metric and the
aggregate case. Git bisection operates on a temporary no-hardlinks clone, never
the source checkout. It caches decisions, retries inconclusive commits within a
bound, and preserves all candidate build/run/comparison artifacts.

Minimization greedily reduces prompt, output, concurrency, runtime arguments,
repetitions, and related fields only when a supplied predicate preserves the
regression. The final bundle contains exact commands, hardware need, physical
plan, commit range, expected effect, interval, and artifact references.

## Fixture evaluation

Normal CI creates a four-commit local repository with a deterministic 12% TTFT
and reciprocal throughput regression. The run in
`artifacts/forgeci/demo/evaluation.json` identified the expected first regressing
commit after evaluating three unique commits, with zero inconclusive results and
reported bisection confidence 0.975. This demonstrates the workflow, not external
runtime performance.

