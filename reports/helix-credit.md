# SLOForge Helix: H3 and H10

## H3: not_exercised

Comparative sample efficiency across four training algorithms.

- Observation: The flagship exercised branch-relative optimization only.
- Limitation: A controlled multi-algorithm learning-curve experiment is absent.
- Artifact: `raw/flagship/seed-41/summary.json`
- Artifact: `raw/flagship/seed-73/summary.json`
- Artifact: `raw/flagship/seed-113/summary.json`

## H10: partial

Controlled exact-state sibling rewards and credit provenance.

- Observation: Every branch group shared one committed BranchPoint and preserved component rewards, first divergence, intervention type, assumptions, and exclusions.
- Limitation: No independent non-sibling variance baseline was run.
- Limitation: Controlled sibling differences improve comparability but do not prove universal causal identification.
- Artifact: `raw/flagship/seed-41/credit/branch-relative.json`
- Artifact: `raw/flagship/seed-73/credit/branch-relative.json`
- Artifact: `raw/flagship/seed-113/credit/branch-relative.json`
