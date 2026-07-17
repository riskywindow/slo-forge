# Helix H9 experience-selection campaign

Campaign ID: `787dbbfd992b493fa4b36d2d2c04c95bb08ecad112c6c9fc38f19f6e75135fdf`

H9 status: **inconclusive**

Measured effects are paired sampled holdout outcomes after reference training. Selector scores and expected learning values are not treated as measured gain.

## Strategy summaries

| Strategy | Cost mean [95% sensitivity] | Diversity | Redundancy | Repaired | Introduced | Before | After | Paired change |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| random | 16.0 [16.0, 16.0] | 1.000 | 0.167 | 44.33 | 2.00 | 0.339 | 0.669 | +0.331 [+0.168, +0.493] |
| failure_only | 16.0 [16.0, 16.0] | 1.000 | 0.250 | 56.00 | 0.00 | 0.339 | 0.776 | +0.438 [+0.386, +0.489] |
| uncertainty_only | 16.0 [16.0, 16.0] | 1.000 | 0.000 | 41.33 | 8.67 | 0.339 | 0.594 | +0.255 [+0.166, +0.345] |
| novelty_only | 16.0 [16.0, 16.0] | 1.000 | 0.250 | 59.00 | 4.33 | 0.339 | 0.766 | +0.427 [+0.378, +0.476] |
| helix_value_aware | 16.0 [16.0, 16.0] | 1.000 | 0.500 | 67.00 | 0.00 | 0.339 | 0.862 | +0.523 [+0.402, +0.645] |

## Paired Helix comparisons

| Baseline | Helix-minus-baseline mean | 95% Student-t sensitivity interval | Seeds |
| --- | ---: | ---: | ---: |
| random | +0.193 | [-0.088, +0.474] | 3 |
| failure_only | +0.086 | [+0.008, +0.164] | 3 |
| uncertainty_only | +0.268 | [+0.181, +0.356] | 3 |
| novelty_only | +0.096 | [+0.023, +0.170] | 3 |

## Raw provenance

The campaign JSON contains a path-and-SHA-256 ledger for 86 scenario, candidate, selection, training, and evaluation artifacts.

## Limitations

- The categorical reference policy cannot condition its action probabilities on the observation.
- Declared diversity and semantic-redundancy groups are synthetic scenario labels.
- The holdout evaluates bounded action choices rather than an external repository or production workload.
- This is one deterministic local synthetic scenario, not a production estimate.
- Seeds are deterministic sensitivity cases and are not claimed to be independent population draws.
- Student-t intervals summarize seed sensitivity at small n; they are not multiplicity-adjusted hypothesis tests.
- Selection features and expected learning values are input predictions; only paired holdout decisions are reported as measured gain.
- Semantic redundancy uses declared scenario groups; the selector itself deduplicates only exact content fingerprints.
- Failure-only uses the reference selector's declared secondary ranking behavior when more candidates qualify than fit.
