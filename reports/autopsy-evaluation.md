# SLOForge Autopsy and recovery evaluation

All faults and observations in this report come from deterministic simulator artifacts.
H3 and H4 use the mixed-bursty workload; H1 and H2 additionally cover long-context and expert-skewed regimes.

## H3 — causal diagnosis

- Top-1 diagnosis accuracy: 1.000
- Top-3 diagnosis accuracy: 1.000
- Median diagnosis latency: 5.387 ms
- Median diagnosis confidence: 0.925
- Median counterfactual healthy residual: 0.000000 ms

## H4 — recovery policy

| method | restoration rate | median action seconds | median estimated cost USD | incorrect action rate |
|---|---:|---:|---:|---:|
| no_recovery | 0.375 | n/a | 0.000000 | 0.000 |
| restart_affected_worker | 0.875 | 120.000 | 0.133333 | 0.500 |
| replace_full_deployment | 1.000 | 180.000 | 3.200000 | 0.000 |
| threshold_recovery | 1.000 | 60.000 | 0.066667 | 0.000 |
| diagnosis_driven | 1.000 | 90.500 | 0.000000 | 0.000 |

Action time and cost are declared recovery-policy parameters. Post-recovery TTFT is derived from counterfactual Rust simulation; this is not a live-cluster failover test.

Raw evidence is indexed by `artifacts/fabric/evaluation/manifest.json`.
