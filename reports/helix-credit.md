# SLOForge Helix: H3 and H10

## H3: not_supported

Comparative sample-budget behavior of four objectives on one categorical CPU reference fixture.

- Observation: At 8 examples, mean targeted success was 0.778646 for successful-branch distillation, 0.802083 for pairwise preference, 0.328125 for group-relative, and 0.348958 for branch-relative optimization.
- Observation: The paired campaign decision rule classified H3 as not_supported.
- Limitation: Training examples are controlled synthetic branch-provenance samples, not GPU-hour-normalized exact-state flagship trajectories.
- Limitation: This reference result does not establish comparative performance for neural policies.
- Artifact: `raw/campaigns/h3-training-algorithms/campaign.json`

## H10: partial

Controlled exact-state sibling rewards and credit provenance.

- Observation: Every branch group shared one committed BranchPoint and preserved component rewards, first divergence, intervention type, assumptions, and exclusions.
- Limitation: No independent non-sibling variance baseline was run.
- Limitation: Controlled sibling differences improve comparability but do not prove universal causal identification.
- Artifact: `raw/flagship/seed-41/credit/branch-relative.json`
- Artifact: `raw/flagship/seed-73/credit/branch-relative.json`
- Artifact: `raw/flagship/seed-113/credit/branch-relative.json`
