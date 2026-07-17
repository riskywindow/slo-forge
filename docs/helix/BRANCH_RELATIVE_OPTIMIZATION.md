# Branch-relative optimization

The implemented vertical slice optimizes a policy from controlled sibling outcomes: capture one
decision boundary, vary an intervention, verify and replay outcomes, compute branch-relative credit,
build a provenance-complete batch, and run a bounded trainer adapter.

This reduces obvious terminal-reward overassignment and makes pairwise preferences available. It does
not guarantee an unbiased policy gradient or establish better sample efficiency than PPO, DPO,
rejection sampling, or train-on-all baselines. The checked evaluation exercises only the reference
method; a controlled multi-algorithm learning-curve experiment remains absent.

See [credit assignment](CREDIT_ASSIGNMENT.md), [experience selection](EXPERIENCE_SELECTION.md), and
[reproducibility](REPRODUCIBILITY.md).
