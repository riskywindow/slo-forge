# SLOForge Helix: H8 and H9

## H8: partial

Continual repair and forgetting over changing task distributions.

- Observation: Across 3 seed runs and 3 sequential tasks, 3 candidates passed and 6 candidates were rejected by the retention gate.
- Observation: The unweighted mean per-seed recurrence rate for accepted repairs was 0.000000.
- Observation: Later targeted candidates improved their current task but were rejected for excessive prior-capability loss.
- Limitation: The categorical policy lacks an observation representation, which limits forward adaptation.
- Limitation: Gate success demonstrates protected retention, not continual-learning superiority.
- Artifact: `raw/campaigns/h8-continual-learning/campaign.json`

## H9: inconclusive

Measured downstream learning value from governed experience selection.

- Observation: Helix value-aware selection produced mean paired measured success change 0.523438 with 95% seed-sensitivity interval [0.402239, 0.644636].
- Observation: Helix minus random had paired mean 0.192708 with interval [-0.088084, 0.473500].
- Observation: The campaign decision rule classified H9 as inconclusive.
- Limitation: The campaign uses one local synthetic candidate pool and categorical reference trainer.
- Limitation: Small-n Student-t intervals are sensitivity summaries and are not multiplicity-adjusted hypothesis tests.
- Artifact: `raw/campaigns/h9-experience-selection/campaign.json`
