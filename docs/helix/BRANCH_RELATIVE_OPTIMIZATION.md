# Branch-relative optimization

The implemented vertical slice optimizes a policy from controlled sibling outcomes: capture one
decision boundary, vary an intervention, verify and replay outcomes, compute branch-relative credit,
build a provenance-complete batch, and run a bounded trainer adapter.

This reduces obvious terminal-reward overassignment and makes pairwise preferences available. It does
not guarantee an unbiased policy gradient or establish better sample efficiency than PPO, DPO,
rejection sampling, or train-on-all baselines.

The H3 reference campaign now runs successful-branch distillation, pairwise preference,
group-relative, and branch-relative optimization over the same declared seeds and example budgets.
Its decision rule compares each evaluation seed's mean targeted task success across the complete
budget curve. Any non-positive branch-relative paired mean yields `not_supported`; the remaining
case stays `inconclusive` because the deterministic trainer does not vary initialization or sample
order by seed and larger sample budgets also receive more optimizer steps. The categorical samples carry branch identifiers and behavior log
probabilities, but they are controlled synthetic provenance fixtures rather than trajectories from
the flagship exact-state repository branch group.

See [credit assignment](CREDIT_ASSIGNMENT.md), [experience selection](EXPERIENCE_SELECTION.md), and
[reproducibility](REPRODUCIBILITY.md).
