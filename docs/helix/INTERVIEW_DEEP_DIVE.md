# Helix interview deep dive

## The problem

Online learning systems often collapse capture, reward, optimization, and deployment into one opaque
loop. Helix asks a harder systems question: what evidence and state boundaries are necessary before a
serving policy may learn from a failure without silently mixing policies, leaking tenants, replaying
effects, starving serving, or partially promoting a candidate?

## The design answer

Helix makes every boundary typed and content addressed. A coordinated barrier binds model,
environment, action, and effect watermarks. Counterfactual branches declare exact reuse,
recomputation, and unsupported state. Trajectories carry behavior log probabilities and policy epochs.
Rewards retain verifier provenance. Credit is sibling-relative. Batches are lineage complete.
Staleness is a disposition, not a hidden filter. Serving remains a hard scheduling constraint.
Promotion requires separately configured gates and a rollback parent; the reference implementation
does not establish organizationally independent evaluator governance.

## Difficult tradeoffs

- Exact capture increases coordination latency; loose capture weakens controlled comparisons and
  prevents exact restoration claims.
- More branch reuse saves work; unsupported reuse corrupts experiments. Reports make the boundary explicit.
- Rich production evidence may improve relevance; consent, privacy, retention, and tenant isolation dominate.
- Learning-value scheduling can use idle capacity; serving forecast error requires reclamation and bounded loss.
- Hashes make artifacts inspectable; they do not authenticate the issuer or prove semantic correctness.

## What the evidence supports

The test suite exercises strict schemas, crash recovery, capture mismatch, branch compatibility,
effect rejection, replay divergence, reward integrity, policy mixing/staleness, training, promotion
rollback, scheduler faults, experience governance, and a complete CPU fault matrix. The evaluation is
honest about synthetic CPU scope and unexercised GPU/production hypotheses.
