# ServingSynthBench metrics and evidence

Raw CPU evidence is one canonical JSON object per request. It records task and workload hashes, environment fingerprint, baseline, run seed, repetition and execution ordinal, request identity, latency, TTFT, inter-token intervals, observed and expected tokens, exactness, timer source, and numerical precision.

The runner reloads those artifacts to calculate:

- sample count and exact-request rate;
- median and p95 request latency;
- median TTFT and inter-token interval;
- measured CPU seconds and candidate count;
- hidden-case exact rate and escaped regressions;
- aggregate valid-system rate and exact-request rate.

The integrity auditor requires the complete request-by-repetition Cartesian product. It flags discarded slow samples, duplicates, workload or environment changes, seed mismatches, altered distributions, non-monotonic timing, precision mismatch, and output disagreement. Different warmup policies are prevented structurally by one run-level configuration.

Latency samples are real local clock observations and are noisy. The CPU smoke report does not assert a statistically significant speedup, cost reduction, energy result, or hardware advantage. Hardware reports must retain their own warmup, raw repetitions, confidence intervals, effect size, noise floor, randomized order, manifest, and simulator-prediction error before making those claims.

## Unseen-task synthesis metrics

The H1 campaign uses the ServingSynthBench typed task grammar but evaluates zero-day synthesis rather than only replaying a pre-existing runtime. Its report schema is `sloforge.genesis.h1-campaign/v1`, conventionally exported at `artifacts/genesis/evaluation/campaigns/h1/report.json`.

For every generated task and declared synthesis seed, raw attempt evidence records:

- task-generation and synthesis seeds;
- task descriptor, public-package, reference-package, inspection, runtime-manifest, and runtime-configuration hashes;
- explicit unsupported diagnostics;
- compilation, correctness, and total attempt time from the host monotonic clock;
- public and held-out requests, expected and observed tokens, sandbox inputs and outputs, and exact match;
- the task-specific special-casing audit; and
- human-authored model-specific adapter lines, which must remain zero for a successful zero-day result.

Task success requires every declared synthesis seed to compile and pass all public and held-out cases. The aggregate reports task and attempt counts, time to first compiling and correct runtime, exact-case rates, unsupported rate, valid-system rate, and Wilson intervals whose sample unit is explicit. Hardware validation remains `not_exercised`; CPU sandbox success must not be converted into a GPU or distributed claim.

`validate_h1_unseen_campaign` reconstructs metrics from raw attempts. It rechecks generated task identity, public package integrity, zero-day inspection, unsupported diagnostics, runtime manifests and seeds, source special-casing, retained sandbox requests and outputs, hidden-case identity, and exact tokens. A summary-only H1 report is not acceptable evidence.

## Hypothesis campaign metric scopes

The following campaigns complement ServingSynthBench. They are kept separate because their measurement units are intentionally different.

| Campaign | Primary metrics | Statistical or replay contract | Scope restriction |
| --- | --- | --- | --- |
| H2/H9 whole stack | Deterministic objective, policy/state/category evidence, configuration-only and best-single-layer paired differences | Seeded paired bootstrap; raw variants and accepted artifacts are replayed | CPU service-model units; hardware runs are zero |
| H3 CEGIS | Detected, prevented, escaped, learned-constraint reuse, verification work | Wilson intervals over candidate-fault instances; all bounded strategy evidence is reconstructed | One cancellation-policy protocol fixture; no universal proof |
| H4 Autopsy guidance | Candidates, invalid candidates, modeled time to improvement, synthetic high-fidelity experiments, final objective, diversity | Seeded proposals, mutation surfaces, stage records, and aggregates are regenerated | Synthetic normalized utility and time units; no hardware experiment |
| H5 lineage transfer | Candidate and time units to correct/improved runtime, invalid candidates, hardware experiments, final objective, negative transfer | Frozen/evaluated SQLite stores, retrieval, reverification, transfer, and invalidation are replayed | Synthetic candidate-cost and objective units; hardware runs are zero |
| H7 evolution | Logical restoration ticks, drops, interruptions, adaptation work, rollbacks, invalid challengers, local runtime observations | Comparator events are regenerated; controller state, capsules, transitions, and sandbox runtime gates are revalidated | Mixed local CPU semantics and synthetic logical time; physical degradation is injected only |
| H8 red team | Unique normal/red-team violations, executed cases to violation, minimized/regression conversion, recurrence | Per-seed red-team reports, corpora, and regression replays are reopened and executed again | Deterministic unsafe fixture; no wall-clock performance or hardware evidence |

Campaign outputs use strict schemas and canonical hashes. Conventional report paths are `artifacts/genesis/evaluation/campaigns/<campaign>/report.json`; the report's artifact references are authoritative if a different output root is selected. Evaluation reports must invoke the matching validator before consuming a metric.

## Negative-result handling

A campaign report retains an unsupported hypothesis instead of selecting a favorable point estimate. The current H9 whole-stack evaluation records `h9_conclusion="not_supported"`: its paired deterministic-service-model evidence does not establish that the two-layer policy/state candidate beats the best single-layer candidate. This is a scoped negative result, not a hardware regression and not a claim that all cross-layer synthesis is ineffective.

Documentation does not pin generated numeric campaign results. Final evaluation and paper artifacts must read metrics from newly generated, hash-validated reports so stale values cannot survive a rerun or a changed implementation.
