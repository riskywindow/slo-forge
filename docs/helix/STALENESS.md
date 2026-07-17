# Staleness semantics

`assess_staleness` evaluates behavior-policy provenance against a target policy and explicit policy.
It reports per-segment update distance, elapsed time, KL/importance-weight evidence when supplied,
transition compatibility, and bounded aggregate metrics. Policies choose hard rejection, truncation,
or resampling and may require log-probability or state recomputation.

Strict samples need one behavior epoch. Segmented samples need dense, non-overlapping ranges and
evidence for every boundary. Missing or unproven Continuum compatibility rejects affected training;
unknown log probabilities are not synthesized. Importance weights are clipped only under an explicit
policy and the unclipped evidence remains visible.

Update count is an imperfect proxy for distribution shift. The reference reports it honestly and
does not claim a universal safe threshold. See ADR 0042 and [training batch](TRAINING_BATCH.md).
