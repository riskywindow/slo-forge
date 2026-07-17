# SLOForge Helix reference evaluation

Evaluation ID: `db8d73223397e48487e23652db7f1bd24229754033276f11ed18a17aa279b792`

This report is generated from deterministic local CPU artifacts. Synthetic results are not presented as GPU or production measurements.

## Flagship result

Intervals are two-sided Student-t 95% intervals over paired deterministic seed runs. They describe seed sensitivity in this fixture, not population uncertainty.

Effect size: paired absolute success-rate difference (corrected challenger minus champion).

| Metric | Mean | Student-t 95% interval | Seed runs |
|---|---:|---:|---:|
| Champion task success | 0.625 | [0.625, 0.625] | 3 |
| Rejected challenger task success | 0.667 | [0.667, 0.667] | 3 |
| Corrected challenger task success | 0.792 | [0.792, 0.792] | 3 |
| Corrected minus champion paired effect | 0.167 | [0.167, 0.167] | 3 |

## Scheduler comparison

All scheduler values and SLO results below are modeled predictions, not measurements.

| Seed | Policy | Predicted SLO feasible | Completed | Predicted value | Cost (microunits) | Lost ticks |
|---:|---|---:|---:|---:|---:|---:|
| 41 | dedicated | true | 2 | 16.000 | 41300 | 0 |
| 41 | static | true | 2 | 16.000 | 41300 | 0 |
| 41 | utilization | true | 2 | 16.000 | 41300 | 0 |
| 41 | fifo | true | 2 | 16.000 | 41300 | 0 |
| 41 | helix_value_aware | true | 2 | 16.000 | 41300 | 0 |
| 73 | dedicated | true | 2 | 16.000 | 41300 | 0 |
| 73 | static | true | 2 | 16.000 | 41300 | 0 |
| 73 | utilization | true | 2 | 16.000 | 41300 | 0 |
| 73 | fifo | true | 2 | 16.000 | 41300 | 0 |
| 73 | helix_value_aware | true | 2 | 16.000 | 41300 | 0 |
| 113 | dedicated | true | 2 | 16.000 | 41300 | 0 |
| 113 | static | true | 2 | 16.000 | 41300 | 0 |
| 113 | utilization | true | 2 | 16.000 | 41300 | 0 |
| 113 | fifo | true | 2 | 16.000 | 41300 | 0 |
| 113 | helix_value_aware | true | 2 | 16.000 | 41300 | 0 |

## Hypotheses

### H1: partial

State identity and replay contracts, not downstream trajectory fidelity.

- Observation: Joint restoration passed exact identity in 3/3 runs.
- Observation: Non-joint modes passed exact identity in 0 cases.
- Observation: Transcript equality explicitly did not establish hidden model/environment equivalence.
- Limitation: Readiness, prefill work, and divergence under a nondeterministic real model were not measured.
- Limitation: Reference policy and repository are synthetic local CPU fixtures.
- Limitation: The result does not establish behavior for arbitrary models, tools, or production traffic.

### H2: partial

CAS-backed logical copy-on-write isolation and cleanup.

- Observation: All branch workspaces were isolated and cleaned after each run.
- Limitation: No reflink/page-fault or GPU-memory amplification benchmark was run on this host.
- Limitation: Reference policy and repository are synthetic local CPU fixtures.
- Limitation: The result does not establish behavior for arbitrary models, tools, or production traffic.

### H3: not_supported

Comparative sample-budget behavior of four objectives on one categorical CPU reference fixture.

- Observation: At 8 examples, mean targeted success was 0.778646 for successful-branch distillation, 0.802083 for pairwise preference, 0.328125 for group-relative, and 0.348958 for branch-relative optimization.
- Observation: The paired campaign decision rule classified H3 as not_supported.
- Limitation: Training examples are controlled synthetic branch-provenance samples, not GPU-hour-normalized exact-state flagship trajectories.
- Limitation: This reference result does not establish comparative performance for neural policies.

### H4: supported_within_scope

Fail-closed policy provenance and bounded staleness semantics.

- Observation: The campaign executed 21 policy-version cases with 0 invalid sample acceptances.
- Observation: 12 cases passed the staleness layer, while strict batch and trainer boundaries separately rejected mixed or unsupported old-policy evidence.
- Limitation: The reference trainer intentionally cannot consume bounded off-policy or segmented mixed-policy evidence.
- Limitation: The recomputed-log-probability case uses declared deterministic fixture evidence rather than recomputing a neural policy probability from token history.
- Limitation: Training stability under a large asynchronous neural optimizer was not measured.

### H5: partial

Deterministic scheduler feasibility under identical scenario inputs.

- Observation: Dedicated, static, utilization, FIFO, and value-aware policies were compiled for every evaluation seed.
- Observation: Hard feasibility against supplied serving latency and queue predictions is reported independently for every policy, seed, and tick.
- Limitation: Learning values are scenario predictions with provenance, not observed policy gains.
- Limitation: No GPU-hour causal claim is made.

### H6: supported_within_scope

Executed local rollout preservation during deterministic capacity reclamation.

- Observation: A preservation path was selected in 0/15 scheduler policy-seed runs.
- Observation: Terminate/restart lost 4.000 tokens and 1.000 tool-work units on average.
- Observation: Joint Continuum plus environment restoration lost 0.000 tokens and 0.000 tool-work units on average.
- Limitation: Resume and restoration values are deterministic executed-operation ticks, not hardware wall time.
- Limitation: No hardware-backed rollout migration timing was measured.

### H7: supported_within_scope

Local atomic promotion state machine and evidence preservation.

- Observation: Each local synthetic run rejected an executed below-threshold candidate, promoted a passing candidate after shadow/canary, pinned an incompatible session, and restored the champion during rollback.
- Limitation: No live distributed gateway or multi-region control plane was exercised.

### H8: partial

Continual repair and forgetting over changing task distributions.

- Observation: Across 3 seed runs and 3 sequential tasks, 3 candidates passed and 6 candidates were rejected by the retention gate.
- Observation: The unweighted mean per-seed recurrence rate for accepted repairs was 0.000000.
- Observation: Later targeted candidates improved their current task but were rejected for excessive prior-capability loss.
- Limitation: The categorical policy lacks an observation representation, which limits forward adaptation.
- Limitation: Gate success demonstrates protected retention, not continual-learning superiority.

### H9: inconclusive

Measured downstream learning value from governed experience selection.

- Observation: Helix value-aware selection produced mean paired measured success change 0.523438 with 95% seed-sensitivity interval [0.402239, 0.644636].
- Observation: Helix minus random had paired mean 0.192708 with interval [-0.088084, 0.473500].
- Observation: The campaign decision rule classified H9 as inconclusive.
- Limitation: The campaign uses one local synthetic candidate pool and categorical reference trainer.
- Limitation: Small-n Student-t intervals are sensitivity summaries and are not multiplicity-adjusted hypothesis tests.

### H10: partial

Controlled exact-state sibling rewards and credit provenance.

- Observation: Every branch group shared one committed BranchPoint and preserved component rewards, first divergence, intervention type, assumptions, and exclusions.
- Limitation: No independent non-sibling variance baseline was run.
- Limitation: Controlled sibling differences improve comparability but do not prove universal causal identification.

## Raw provenance

Every executed summary and scheduler plan used above is content-addressed here. Paths are relative to the evaluation output directory.

- `b396a99d45e33ff41027de5eaff89684f5a79eef04774a5e51c585c3b365a2f3`  `raw/campaigns/h3-training-algorithms/campaign.json`
- `a8baa7d68021e2783a0cad096b58bfe4ba940e235da3d1c07916d7a475ab21b7`  `raw/campaigns/h4-staleness/campaign.json`
- `13f4d3baab4eaa382d5f0426b20bb56c652f0b1038c67fbac25a4ef1281896fa`  `raw/campaigns/h6-preservation/campaign.json`
- `400ecfd553c41ce2b215e8e1dcf6e5f5f96c91467b34301e6dcf852b760792ad`  `raw/campaigns/h8-continual-learning/campaign.json`
- `db930756589ef85460989bca5ea5de564a60833c5a8047a5440a38fc845bc090`  `raw/campaigns/h9-experience-selection/campaign.json`
- `c370a90eda19eaaeb3be3168c799444793ef65ea090f48fef4232018b48e3639`  `raw/flagship/seed-113/credit/branch-relative.json`
- `b0da2a3363c546f3d86b5b08e108ba07f2a9cc920952321b4f81e13e306b8094`  `raw/flagship/seed-113/summary.json`
- `efd659b9de2eb9ea9266588f899f1bf2be8d883338cb27764cf09e1029fb66db`  `raw/flagship/seed-41/credit/branch-relative.json`
- `869199e118bd4c6149f9d7edc9f0dde14d4d68d9c8ff5165e4ad14a2e0a7ebdd`  `raw/flagship/seed-41/summary.json`
- `f6ce4e5feeeea491e8d08c034a1a5bb7b970204d168e171cbadd5eb978904553`  `raw/flagship/seed-73/credit/branch-relative.json`
- `5eae8843b654245de6fe3a94ea20c8663f8f118b9e2fd4359e22dd3fbedf0757`  `raw/flagship/seed-73/summary.json`
- `da241bb2e8c19c529f90d409adf1af2115e669517a92edbfa91d981fbdd02f6a`  `raw/scheduler/seed-113/dedicated.json`
- `731432ffe9d198ec73c36d22fc3939d1612af6145328d8726b5c7af097105ecf`  `raw/scheduler/seed-113/fifo.json`
- `7e396eb0ef4852c5a2fe020f0e9d6feeadadab7b5e32892acb81c8ed2b2bd67c`  `raw/scheduler/seed-113/helix_value_aware.json`
- `2ea80ff2671b6f803ab849b91463729b6fc6516b7fb4f53104dec9e909a318cd`  `raw/scheduler/seed-113/static.json`
- `e0aa21ee9fdc9e28ace7443184cbda8cdc944fb5d9333b7c6d6ad4affa7d4064`  `raw/scheduler/seed-113/utilization.json`
- `31ec28abddd2c549810be41c1fbd9ee0fef9ef5d7838172712d71559269765f2`  `raw/scheduler/seed-41/dedicated.json`
- `4e3a0467e813b60a3f81a40660d04c8212bbca26148de25d390e9be580eaddc6`  `raw/scheduler/seed-41/fifo.json`
- `b5dd0a4894b66cf1d69249631a741a31833eaacd05f557661d6830ec94166912`  `raw/scheduler/seed-41/helix_value_aware.json`
- `6db329bcdcaabeefaa97dd93615951dcd0e62725ea5ec196059842e5a27308bf`  `raw/scheduler/seed-41/static.json`
- `6e3ef0ce5587ea63f79b8c025df6bca8f2913d02449b4357d481e1c2ce5aa07f`  `raw/scheduler/seed-41/utilization.json`
- `5ea61a15ec6e692813d99746b9f51435d72dabfe882fbd323cc62b8100a1fa26`  `raw/scheduler/seed-73/dedicated.json`
- `4f3f811bd67bc0126d11c0ab00ea092f545452ee5d961e805086c713e28a0585`  `raw/scheduler/seed-73/fifo.json`
- `5c9d3266adbae426cc92310deec746fd0c48e62b24ac5237a9580b8278139ef9`  `raw/scheduler/seed-73/helix_value_aware.json`
- `c6fc020f2d382be6f9bfc4a725af1ffd80fe54fcdbae1f98f8063942e74bea3f`  `raw/scheduler/seed-73/static.json`
- `255f59ed1e4dfb5c7e89d2343b598515e528d6e1afe62310d22a8cfa288036cf`  `raw/scheduler/seed-73/utilization.json`
- `9e1b5c66fa1c7d26b28dd5a8e67cd5e75e1847fb36f7ce54df37263138c78809`  `raw/scheduler/source-workload.json`

## Limitations

- No GPU or hardware-backed model path was available on this host.
- No production data, external API, live promotion, or external side effect was authorized.
- Student-t intervals summarize deterministic seed sensitivity with a small sample; seeds are not asserted to be independent population draws.
- H1-H10 statuses are descriptive evidence classifications, not multiplicity-adjusted statistical hypothesis tests.
- Predicted scheduler learning value and predicted serving-SLO feasibility are not observed quality or serving measurements.
- The static baseline uses one declared resource-vector unit per learning class because the source workload is authored for the value-aware policy.
