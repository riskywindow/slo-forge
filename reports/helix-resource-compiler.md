# SLOForge Helix: H5 and H6

## H5: partial

Deterministic scheduler feasibility under identical scenario inputs.

- Observation: Dedicated, static, utilization, FIFO, and value-aware policies were compiled for every evaluation seed.
- Observation: Hard feasibility against supplied serving latency and queue predictions is reported independently for every policy, seed, and tick.
- Limitation: Learning values are scenario predictions with provenance, not observed policy gains.
- Limitation: No GPU-hour causal claim is made.
- Artifact: `raw/scheduler/seed-41/dedicated.json`
- Artifact: `raw/scheduler/seed-41/static.json`
- Artifact: `raw/scheduler/seed-41/utilization.json`
- Artifact: `raw/scheduler/seed-41/fifo.json`
- Artifact: `raw/scheduler/seed-41/helix_value_aware.json`
- Artifact: `raw/scheduler/seed-73/dedicated.json`
- Artifact: `raw/scheduler/seed-73/static.json`
- Artifact: `raw/scheduler/seed-73/utilization.json`
- Artifact: `raw/scheduler/seed-73/fifo.json`
- Artifact: `raw/scheduler/seed-73/helix_value_aware.json`
- Artifact: `raw/scheduler/seed-113/dedicated.json`
- Artifact: `raw/scheduler/seed-113/static.json`
- Artifact: `raw/scheduler/seed-113/utilization.json`
- Artifact: `raw/scheduler/seed-113/fifo.json`
- Artifact: `raw/scheduler/seed-113/helix_value_aware.json`

## H6: supported_within_scope

Executed local rollout preservation during deterministic capacity reclamation.

- Observation: A preservation path was selected in 0/15 scheduler policy-seed runs.
- Observation: Terminate/restart lost 4.000 tokens and 1.000 tool-work units on average.
- Observation: Joint Continuum plus environment restoration lost 0.000 tokens and 0.000 tool-work units on average.
- Limitation: Resume and restoration values are deterministic executed-operation ticks, not hardware wall time.
- Limitation: No hardware-backed rollout migration timing was measured.
- Artifact: `raw/scheduler/seed-41/dedicated.json`
- Artifact: `raw/scheduler/seed-41/static.json`
- Artifact: `raw/scheduler/seed-41/utilization.json`
- Artifact: `raw/scheduler/seed-41/fifo.json`
- Artifact: `raw/scheduler/seed-41/helix_value_aware.json`
- Artifact: `raw/scheduler/seed-73/dedicated.json`
- Artifact: `raw/scheduler/seed-73/static.json`
- Artifact: `raw/scheduler/seed-73/utilization.json`
- Artifact: `raw/scheduler/seed-73/fifo.json`
- Artifact: `raw/scheduler/seed-73/helix_value_aware.json`
- Artifact: `raw/scheduler/seed-113/dedicated.json`
- Artifact: `raw/scheduler/seed-113/static.json`
- Artifact: `raw/scheduler/seed-113/utilization.json`
- Artifact: `raw/scheduler/seed-113/fifo.json`
- Artifact: `raw/scheduler/seed-113/helix_value_aware.json`
- Artifact: `raw/campaigns/h6-preservation/campaign.json`
