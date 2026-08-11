# SLOForge Fabric Autopsy and statistical-methodology review

Review baseline: `bcb213a`  
Scope: canonical evidence comparison, time alignment, diagnosis ranking, counterfactual replay,
minimization, and the checked-in H1–H4 evaluation methodology. The committed large evaluation
bundle was inspected but not regenerated or edited during this review.

## Verdict

Autopsy is a deterministic evidence engine, not an LLM wrapper. Its diagnosis path does not read
`FaultInterval.fault_type` or `ground_truth`; a new adversarial test changes the injected label
while preserving observations and obtains an identical diagnosis and confidence. The subsystem
has meaningful causal structure—matched healthy/degraded evidence, physical counters, stage
deltas, first-divergence analysis, counterfactual simulation, and delta debugging—but its numeric
confidence remains a heuristic evidence score rather than a calibrated probability.

The checked-in H3 result is valid as a deterministic two-fault synthetic regression test. It is
not evidence of broad real-cluster diagnosis accuracy. The checked-in H4 result evaluates
counterfactual post-action request performance, while action time and cost are declared policy
parameters. It is not a measured recovery trial.

## Findings fixed

| Severity | Finding | Resolution and test |
|---|---|---|
| High | `compare_runs` accepted equal-workload evidence from a different topology or physical plan, invalidating attribution. | Comparisons now require identical workload, topology, physical plan, and reference host. `test_comparison_rejects_unmatched_physical_context` covers both mismatches. |
| High | Global first-divergence ordering selected a timestamp without considering clock-alignment uncertainty. Missing multi-host estimates were treated as usable by vacuous truth. | Divergent events now carry uncertainty intervals; overlapping earliest intervals produce no first divergence and an explicit warning. Multi-host ordering fails closed when estimates are missing or insufficient. |
| High | `MinimizationResult.bundle_sha256` hashed an intermediate run, not the emitted minimized bundle. | The emitted run is built first and its canonical JSON is hashed. The adversarial test recomputes and compares the digest. |
| Medium | Clock offset/drift used ordinary least squares, allowing one delayed exchange to arbitrarily move the fit. | Alignment uses a bounded Theil–Sen estimator over at most 256 minimum-RTT observations. An outlier test preserves the known 5 ppm drift and offset. |
| Medium | Hypothesis confidence used the total matched event count, so unrelated events could inflate a signal's score. | Each signal now counts only relevant stage matches and named counters. A 100-event irrelevant-noise test produces exactly the same network confidence as the one-event case. |
| Medium | Reused counter names with conflicting units were collapsed by name. | Diagnosis now rejects ambiguous units instead of silently choosing one delta. |
| Medium | Unsupported hypotheses could be ranked without clearly declaring the absence of positive evidence. | Supported hypotheses sort before rejected hypotheses; no-threshold cases emit an explicit warning. Every diagnosis states that confidence is not a calibrated probability. |
| Medium | Completed counterfactual records did not validate interval ordering or status consistency. | Model validation now derives the only valid supported/contradicted/inconclusive state from the improvement interval. |
| Medium | Counterfactual attachment could prefer a contradicted large point estimate over a supported repair for the same hypothesis. | Selection now prioritizes status, confidence, healthy-reference residual, then effect size. |
| Medium | Counterfactual evidence could invalidate the original top hypothesis without updating the diagnosis ranking. | Attached evidence now deterministically reranks hypotheses, `top_hypothesis`, `top_three`, and overall confidence. |
| Medium | Replay allowed a "degraded" run no slower than its healthy reference. | Replay rejects this non-regression case before causal repair ranking. |
| Medium | Minimization did not reduce fault scope despite the command contract. | Fault intervals now participate in deterministic delta debugging and removed fault IDs/counts are reported. |
| Medium | Diagnosis-rule tests covered only network degradation and rank skew. | A table-driven test proves that all 27 `BottleneckKind` rules can independently cross their evidence threshold and win ranking. |

## Label leakage and causal ranking

- Ground-truth fault intervals remain in simulator evidence for evaluation scoring, but
  `diagnose` and `extract_signals` consume only `DifferentialComparison` observations. The new
  label-invariance test is a direct guard against H3 target leakage.
- Physical bandwidth changes receive higher causal specificity than downstream stage delays.
  Relevant sample counts prevent unrelated downstream work from increasing that confidence.
- Rank skew and stage regressions can still be propagation effects. Counterfactual replay is
  therefore required before treating a hypothesis as a repair recommendation.
- `RemoveFault` is an exact simulator counterfactual, not an executable production repair. Results
  using it establish internal causal consistency only.

## Statistical-methodology audit

### H1 and H2

- The evaluation has three seeds across four synthetic topologies and three workload regimes.
  Point estimates are artifact-derived.
- `_bootstrap_median` resamples trial rows rather than seed blocks. Rows sharing a seed across
  workloads and topologies are not demonstrably independent, so the published bootstrap ranges
  are descriptive intervals, not confirmatory confidence intervals.
- H1 reports the primary compiler losing the aggregate latency comparison; this negative result is
  not hidden.
- H2 explicitly reports poor interval coverage (`0.067`) and identifies shared calibration inputs
  between compiler and simulator. Its rank correlation is internal synthetic ranking evidence, not
  hardware generalization.
- H1 lacks paired effect-size intervals by topology/workload. The raw per-seed rows permit that
  analysis, but the checked-in report currently gives point deltas and aggregate bootstrap ranges.

### H3

- The checked-in evaluation contains 24 deterministic cases: two fault families over four
  topologies and three seeds. Top-1 and top-3 accuracy are exact proportions for that fixture
  matrix.
- It does not provide binomial uncertainty, Brier score, expected calibration error, or a
  reliability diagram. Median confidence is not confidence calibration.
- The 27-rule table test establishes implementation reachability, not diagnosis accuracy under
  confounding. Broader multi-fault and no-fault fixtures are required before claiming broad H3
  accuracy.

### H4

- All policy rows share the same degraded and counterfactual Rust simulations, which makes request
  performance comparable.
- Action durations and costs are declared constants, not repeated measurements. Generic full
  replacement and threshold recovery are assigned the restored counterfactual when their policy
  says they repair the fault. Thus restoration-rate differences partly encode policy assumptions.
- Dropped requests, interrupted streams, build-time variance, and action failures are not sampled
  by this H4 matrix. The report correctly labels it a simulation result rather than live failover.

## Remaining limitations

1. Clock uncertainty includes residual and half-RTT error but cannot observe fixed path asymmetry.
   It also does not model uncertainty growth beyond the clock-sample window.
2. Diagnosis confidence has monotonic, bounded evidence semantics but no empirical calibration
   dataset. It must not be interpreted as a probability of correctness.
3. Counterfactual repair semantics are validated by the physical simulator, but exact
   `RemoveFault` scenarios are oracle-like synthetic controls.
4. Minimization reduces events, ranks, counters, and fault intervals. Model shape, workload shape,
   and parallelism are referenced by hashes rather than embedded in `AutopsyRun`, so this API
   cannot reduce those dimensions by itself.
5. Evaluation bootstrap and effect-size issues cannot be corrected without regenerating the
   committed result and report bundle; this review deliberately preserved that provenance.

## Acceptance evidence

Executed commands:

```text
.venv/bin/ruff check python/sloforge/autopsy tests/python/test_autopsy_adversarial.py
MYPYPATH=python .venv/bin/mypy --strict python/sloforge/autopsy tests/python/test_autopsy_adversarial.py
PYTHONPATH=python .venv/bin/pytest -q tests/python/test_autopsy_alignment.py tests/python/test_autopsy_diagnosis.py tests/python/test_autopsy_counterfactual.py tests/python/test_autopsy_adversarial.py
PYTHONPATH=python .venv/bin/pytest -q tests/python/test_fabric_demo.py tests/python/test_fabric_cli.py tests/python/test_fabric_recovery.py tests/python/test_fabric_evaluation.py -k 'not artifact_derived_evaluation_round_trip'
```

Observed results:

- Ruff: passed.
- strict mypy: passed for nine Autopsy source/test modules.
- focused Autopsy tests: 50 passed.
- Fabric integration compatibility tests: 20 passed, one expensive regeneration test deselected.
