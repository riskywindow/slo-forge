# SLOForge Fabric resume bullets

The measurements below are explicitly labeled synthetic or local fixture
results. Source comments are retained beside each bullet so metrics can be
audited before use.

<!-- source: artifacts/fabric/evaluation/result.json and reports/fabric-evaluation.md; 180 deterministic synthetic plan trials, not hardware measurement -->
- Built a topology-aware physical inference compiler and deterministic Rust
  communication twin evaluated across 180 synthetic plan trials, preserving the
  negative result that its 626.120 ms median p95 TTFT was 1.57% slower than the
  already topology-aligned sequential fixture rather than claiming a win.

<!-- source: artifacts/fabric-demo/manifest.json and artifacts/fabric-demo/recovery/execution.json; synthetic simulation, not GPU measurement -->
- Implemented structured causal diagnosis and guarded self-healing that evaluated
  seven synthetic counterfactual repairs, moved through simulation, shadow,
  canary, promotion, and stream-preserving drain, and restored simulated p95 TTFT
  from 7045.952 ms to 1152.714 ms after simultaneous network/rank faults.

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
