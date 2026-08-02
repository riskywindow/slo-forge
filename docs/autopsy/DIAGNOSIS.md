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

Confidence combines distance over threshold, a logarithmic sample factor for the
hypothesis's own stage/counter signal, and a bounded bonus for direct physical
counters. It is capped at 0.99. Unsupported rules remain at or below 0.20. This
score orders engineering hypotheses; it is not a posterior probability.
Alignment insufficiency gates cross-host ordering and adds a warning rather than
silently weakening unrelated stage-duration evidence. A comparison with no
crossed rule records that no causal hypothesis has evidence.

## Counterfactual validation

A hypothesis is strengthened only by an explicit repair simulation. Improvement
is degraded makespan minus counterfactual makespan. Bounds combine the two
simulator prediction intervals. Positive lower improvement is `supported`;
non-positive upper improvement is `contradicted`; an interval crossing zero is
`inconclusive`. Confidence is reduced by uncertainty width and repair direction.

Replay selection minimizes remaining distance to a healthy reference, then
prefers expected improvement and a stable scenario ID. Attachment keeps the best
status/confidence/residual estimate for each hypothesis and reranks the diagnosis
after causal outcomes. Rejected scenarios and simulator failures remain in the
replay artifact.

## Current validation scope

Unit fixtures exercise every diagnosis rule. In the flagship simultaneous-fault
run, `network_bandwidth_degradation` is top-1 and `rank_straggler` is top-2; a
combined repair is needed for full restoration. See
`artifacts/fabric-demo/autopsy/diagnosis.json`. This is one deterministic
synthetic interaction, so cross-fault accuracy must be read from the evaluation
corpus, not inferred from the number of implemented rules.
