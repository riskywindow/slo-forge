# Trainer adapter

The trainer boundary accepts provenance-complete samples and returns a new immutable policy epoch plus
step evidence. The local `ReferenceTrainer` supports bounded deterministic policy-gradient, clipped
importance-ratio, KL-regularized, and pairwise-preference updates over the tiny reference policy.

Learning rate, clipping, KL coefficient, step count, action vocabulary, behavior epoch, sample
eligibility, staleness, and preference pairs are validated. Mixed behavior epochs and an entirely
rejected sample set fail closed. The candidate epoch and metrics preserve seed and parent identity.

This is a CPU reference adapter, not a distributed LLM trainer. It does not implement optimizer-state
sharding, mixed precision, gradient checkpointing, or production convergence guarantees. See ADR
0045 and [limitations](LIMITATIONS.md).
