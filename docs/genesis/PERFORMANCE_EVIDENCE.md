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

## Artifact-backed hypothesis campaigns

Genesis hypothesis campaigns live under an evaluation artifact root rather than embedding copied results in documentation. The conventional export root is `artifacts/genesis/evaluation/campaigns/`; each campaign writes a `report.json` plus raw inputs, observations, or transactional stores below its own directory. A report is usable only after its corresponding validator accepts the report and every referenced artifact.

| Hypothesis | Report schema and conventional path | Evidence scope |
| --- | --- | --- |
| H1 | `sloforge.genesis.h1-campaign/v1` at `campaigns/h1/report.json` | Generated unseen tasks, zero-day inspection, generated CPU runtimes, public and held-out cases, and host-monotonic compilation/correctness timing. GPU and distributed hardware are not exercised. |
| H2 and H9 | `sloforge.genesis.whole-stack-campaign/v1` at `campaigns/h2-h9/report.json` | Real candidate, genome, policy, transformation, tensor-rewrite, and Fabric-fixture artifacts evaluated by a deterministic CPU service model. Service units are not wall time or hardware measurements. |
| H3 | `sloforge.genesis.h3-campaign/v1` at `campaigns/h3/report.json` | Bounded deterministic CPU protocol fixture comparing tests, fuzzing, model checking, and full CEGIS. It is neither a universal proof nor a performance experiment. |
| H4 | `sloforge.genesis.h4-campaign/v1` at `campaigns/h4/report.json` | Autopsy-guided, random-region, and unrestricted search over a deterministic synthetic evaluator. Candidate time and objective values are declared model units; hardware experiments are zero. |
| H5 | `schema_version=1.0.0`, `evaluator_id=genesis-lineage-synthetic-v1` at `campaigns/h5/report.json` | Empty, unrelated, related, stale, and invalidated lineage scenarios using frozen SQLite stores and deterministic candidate-cost units. It does not measure model or hardware speed. |
| H7 | `sloforge.genesis.h7-campaign/v1` at `campaigns/h7/report.json` | Proof-gated local capsule replay for the Genesis arm plus deterministic logical-time comparator models. Local sandbox observations and synthetic restoration ticks remain separate; hardware experiments are zero. |
| H8 | `sloforge.genesis.h8-campaign/v1` at `campaigns/h8/report.json` | Executable unsafe-streaming fixture comparing a bounded nominal suite with existing red-team surfaces. Discovery time is executed-case count, not wall time. |

The path is not evidence by itself. Reports bind raw files by SHA-256 and use strict schemas with unknown fields forbidden. Deterministic campaign seeds are stored at report and per-attempt or per-candidate level; multi-seed campaigns reject missing, repeated, reordered, or out-of-budget seed products where their contract requires them.

## Independent campaign validation

Campaign validators reopen raw artifacts and derive conclusions rather than trusting report summaries:

- H1 revalidates task descriptors and package hashes, reruns zero-day inspection, checks unsupported diagnostics, verifies generated-runtime manifests and synthesis seeds, repeats the special-casing audit, and binds every public and hidden token result to retained sandbox request and output evidence.
- H2/H9 reopen the accepted candidate, genome, policy, physical-plan fixture, exact tensor rewrite, and raw seed rows; replay every deterministic service-model variant; and recompute paired bootstrap effects and conclusions.
- H3 checks every bounded artifact reference and size, reconstructs ground-truth fault classifications, reruns the strategy-specific checks, validates minimized counterexamples and learned constraints, and derives all aggregates from raw fault records.
- H4 regenerates seeded proposals and mutation surfaces, reenforces the Autopsy mutation guard, checks lifecycle stage records, and recomputes guided-versus-baseline deltas.
- H5 reopens the frozen and evaluated SQLite stores, repeats lineage retrieval and initialization, reconstructs candidate and transfer records, checks dependency invalidation, and derives every scenario and paired-seed result.
- H7 recomputes all comparator traces, reconstructs Genesis events from the persisted controller snapshot and gate observations, revalidates both capsules and transition compatibility, and reruns the sandboxed runtime semantics for publishable shadow and canary evidence. Test-fixture gate evidence is rejected by default.
- H8 reopens per-seed red-team reports, minimized corpora, and replay artifacts; executes the regression corpus again; and recomputes unique violations, recurrence, conversion, and conclusion fields.

Validators reject changed hashes, missing artifacts, inconsistent seeds, summary/raw disagreement, escaped campaign-root paths where the format defines a root, scope upgrades such as changing hardware-backed counts, and conclusions unsupported by recomputed evidence. Validation is scoped evidence checking, not a universal proof of the implementation or benchmark harness.

## Statistical interpretation and the H9 negative result

The campaigns do not share one interchangeable confidence unit:

- H1 reports Wilson intervals over unseen-task and synthesis-seed success units and records host-monotonic timings.
- H2/H9 use paired seed differences with a seeded 95 percent bootstrap over deterministic service-model outcomes.
- H3 Wilson intervals are descriptive over correlated candidate-fault instances.
- H4, H5, H7, and H8 report deterministic fixture aggregates or paired differences; they do not imply population-level statistical significance.

The current H9 campaign result is negative: within the declared deterministic service model, the paired confidence bound does not establish that the accepted two-layer policy/state candidate outperforms the best single-layer candidate. The report therefore records `h9_conclusion="not_supported"`. This does not invalidate the cross-layer candidate or H2's configuration-only comparison; it prevents an unsupported cross-layer-superiority claim. Numeric values must be read from a newly generated and validated `campaigns/h2-h9/report.json`, not copied into this document.
