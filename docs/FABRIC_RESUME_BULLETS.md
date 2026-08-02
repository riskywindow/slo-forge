# SLOForge Fabric resume bullets

The measurements below are explicitly labeled synthetic or local fixture
results. Source comments are retained beside each bullet so metrics can be
audited before use.

<!-- source: artifacts/fabric-demo/manifest.json and artifacts/fabric-demo/physical-plan.json; deterministic synthetic hardware, seed 41 -->
- Built a topology-aware physical inference compiler and deterministic Rust
  communication twin that mapped a synthetic two-host/16-rank deployment,
  preserved 174 rejected alternatives and three recovery variants, and validated
  faults through artifact-hashed traces and reports.

<!-- source: artifacts/fabric-demo/manifest.json and artifacts/fabric-demo/recovery/execution.json; synthetic simulation, not GPU measurement -->
- Implemented structured causal diagnosis and guarded self-healing that evaluated
  seven synthetic counterfactual repairs, moved through simulation, shadow,
  canary, promotion, and stream-preserving drain, and restored simulated p95 TTFT
  from 2484.092 ms to 955.883 ms after simultaneous network/rank faults.

<!-- source: artifacts/forgeci/demo/evaluation.json; deterministic local fixture repository -->
- Built a robust performance-regression CI/bisector that identified the exact
  first regressing fixture commit after three unique commit evaluations, detecting
  a 12.00% p99-TTFT regression with a corrected 95% bootstrap interval of
  11.47%-12.53% and zero inconclusive classifications.

<!-- source: artifacts/warmpath/manifest.json; real local file timings over synthetic snapshot payload -->
- Built a cold-start artifact DAG planner and bounded checksumming executor that
  exhaustively evaluated 243 local placement candidates across NVMe, page cache,
  and host memory, then materialized three startup-critical artifacts and
  deferred one non-critical artifact.

<!-- source: reports/rank-ordering-experiment.md and its referenced raw artifact; synthetic calibrated experiment, disabled by default -->
- Evaluated a trace-justified collective rank-ordering optimization across six
  exact candidates and two message regimes, retaining it behind a
  measure-on-hardware gate because the observed benefit came only from synthetic
  calibrated curves.
