# Credit assignment

`assign_branch_relative_credit` compares sibling outcomes from one branch group instead of assigning
the terminal score indiscriminately to every action. It records baseline-relative reward, weighted
advantage, exclusion reasons, causal support, and pairwise chosen/rejected preferences.

Credit is admitted only when branch, trajectory, action, and reward provenance agree. The reference
method is deterministic and structurally auditable. It does not solve causal identification: sibling
branches can still differ in uncontrolled or unrecorded state, and a reward difference is not by
itself proof that one action caused the outcome.

Observational outcomes remain in the credit artifact for accounting but receive zero optimization
weight and cannot enter pairwise preferences. Controlled intervention labels are caller-supplied
evidence classifications, not proof that the intervention was the only changed variable. Any
`process_score` must come from evidence independent of terminal reward; deriving it from the same
reward would count the outcome twice. Decision-local decay is based on the post-intervention suffix,
while group-relative optimization uses the unlocalized advantage.

Use replay and branch minimization to strengthen causal support. See ADR 0044 and
[branch-relative optimization](BRANCH_RELATIVE_OPTIMIZATION.md).
