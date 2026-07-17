# SLOForge Helix: H4 and H7

## H4: supported_within_scope

Fail-closed policy provenance and bounded staleness semantics.

- Observation: The campaign executed 21 policy-version cases with 0 invalid sample acceptances.
- Observation: 12 cases passed the staleness layer, while strict batch and trainer boundaries separately rejected mixed or unsupported old-policy evidence.
- Limitation: The reference trainer intentionally cannot consume bounded off-policy or segmented mixed-policy evidence.
- Limitation: The recomputed-log-probability case uses declared deterministic fixture evidence rather than recomputing a neural policy probability from token history.
- Limitation: Training stability under a large asynchronous neural optimizer was not measured.
- Artifact: `raw/campaigns/h4-staleness/campaign.json`

## H7: supported_within_scope

Local atomic promotion state machine and evidence preservation.

- Observation: Each local synthetic run rejected an executed below-threshold candidate, promoted a passing candidate after shadow/canary, pinned an incompatible session, and restored the champion during rollback.
- Limitation: No live distributed gateway or multi-region control plane was exercised.
- Artifact: `raw/flagship/seed-41/summary.json`
- Artifact: `raw/flagship/seed-73/summary.json`
- Artifact: `raw/flagship/seed-113/summary.json`
