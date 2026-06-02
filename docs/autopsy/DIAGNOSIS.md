# Diagnosis engine

The diagnosis engine is a deterministic rules-and-counterfactual system. Its
output is complete without natural-language generation.

## Differential features

Healthy and degraded runs must share topology, plan, and workload fingerprints.
Events are grouped by type, operation, rank, and request. The comparator reports
matched medians and relative deltas, unmatched counts, counter medians, first
divergence, and maximum rank skew. Rank skew uses physical, prefill, decode, and
transfer durations, avoiding request-only averages that hide a slow rank.

Signals include queue, startup/load, CPU launch, kernel count, GPU compute/HBM
and clocks, NUMA, PCIe/NVLink/network, rank skew, collective wait/algorithm,
expert imbalance, prefill/decode saturation, KV transfer, worker health/crash,
topology mismatch, and plan-assumption error.

## Hypotheses

All 27 bottlenecks are evaluated on every comparison. A rule defines signal,
threshold, relation, target, and bounded causal-specificity bonus. A crossed
threshold yields supporting evidence; otherwise the same evidence is retained as
contradicting with a rejection reason.

Confidence combines distance over threshold, logarithmic matched-sample factor,
and a bounded bonus for direct physical counters. It is capped at 0.99. This
score orders engineering hypotheses; it is not a posterior probability.
Alignment insufficiency adds a warning rather than silently weakening timestamp
claims.

## Counterfactual validation

A hypothesis is strengthened only by an explicit repair simulation. Improvement
is degraded makespan minus counterfactual makespan. Bounds combine the two
simulator prediction intervals. Positive lower improvement is `supported`;
non-positive upper improvement is `contradicted`; an interval crossing zero is
`inconclusive`. Confidence is reduced by uncertainty width and repair direction.

Selection minimizes remaining distance to a healthy reference, then prefers
higher confidence and a stable scenario ID. Rejected scenarios and simulator
failures remain in the replay artifact.

## Current validation scope

Unit fixtures exercise every diagnosis rule. The flagship simultaneous-fault run
correctly detected the rank straggler as top-1 and needed a combined repair for
full restoration, but did not rank the injected network bandwidth fault in its
top three. See `artifacts/fabric-demo/autopsy/diagnosis.json`. Cross-fault accuracy
must therefore be read from the evaluation corpus, not inferred from the number
of implemented rules.

