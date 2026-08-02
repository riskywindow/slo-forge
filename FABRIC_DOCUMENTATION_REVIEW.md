# SLOForge Fabric documentation review

Review date: 2026-08-01
Scope: required Fabric, Autopsy, recovery, ForgeCI, WarmPath, ADR, README, and
paper documentation; runtime-adapter review; checked artifact claims
Validation mode: source inspection, artifact-hash validation, local-link scan,
metric reconciliation, stale-claim scan, and targeted tests

## Verdict

The required documentation set is present, substantive, and internally linked.
The review corrected stale flagship values, Autopsy semantics, recovery action
and audit counts, runtime-adapter ranges, and several open simulator/compiler
boundaries. It also added the complete negative H1/H2 outcome instead of
presenting topology-aware compilation or uncertainty calibration as a win.

All quoted performance values below are derived from repository artifacts.
Fabric GPU/network/multi-node results remain synthetic; no documentation claims
hardware execution that did not occur.

## Required-document inventory

The review inspected every required document:

- root and paper: `README.md`, `paper/fabric_extension/README.md`, and
  `paper/fabric_extension/SLOFORGE_FABRIC.md`;
- Fabric: `docs/fabric/ARCHITECTURE.md`,
  `docs/fabric/PHYSICAL_EXECUTION_PLAN.md`,
  `docs/fabric/TOPOLOGY_DISCOVERY.md`, `docs/fabric/FABRIC_PROFILER.md`,
  `docs/fabric/DIGITAL_TWIN.md`, `docs/fabric/PHYSICAL_COMPILER.md`, and
  `docs/fabric/RUNTIME_ADAPTERS.md`;
- Autopsy: `docs/autopsy/ARCHITECTURE.md`,
  `docs/autopsy/EVIDENCE_MODEL.md`, `docs/autopsy/TIME_ALIGNMENT.md`,
  `docs/autopsy/DIAGNOSIS.md`, `docs/autopsy/COUNTERFACTUAL_REPLAY.md`, and
  `docs/autopsy/MINIMIZATION.md`;
- recovery: `docs/recovery/RECOVERY_PLANNER.md`,
  `docs/recovery/STATE_MACHINE.md`, and `docs/recovery/SAFETY.md`;
- ForgeCI: `docs/forgeci/ARCHITECTURE.md`, `docs/forgeci/STATISTICS.md`, and
  `docs/forgeci/BISECTION.md`;
- WarmPath: `docs/warmpath/ARCHITECTURE.md`,
  `docs/warmpath/ARTIFACT_GRAPH.md`, and `docs/warmpath/PLANNER.md`;
- cross-cutting: `docs/FABRIC_RELATED_WORK.md`,
  `docs/FABRIC_SECURITY.md`, `docs/FABRIC_REPRODUCIBILITY.md`,
  `docs/FABRIC_LIMITATIONS.md`, `docs/FABRIC_DEMO_SCRIPT.md`,
  `docs/FABRIC_INTERVIEW_DEEP_DIVE.md`, and
  `docs/FABRIC_RESUME_BULLETS.md`;
- ADRs: `docs/adr/0012-*` through `docs/adr/0022-*`, covering IR composition,
  the JSON subprocess boundary, topology evidence, simulator resources,
  optimizer design, clock alignment, diagnosis confidence, recovery safety,
  runtime adapters, privileged telemetry, and benchmark statistics.

The review also reconciled these implementation reviews:
`ARCHITECTURE_DISTRIBUTED_NETWORK_REVIEW.md`,
`SIMULATOR_OPTIMIZER_REVIEW.md`, `AUTOPSY_STATISTICAL_REVIEW.md`,
`RUNTIME_ADAPTER_REVIEW.md`, and `SECURITY_REVIEW.md`.

## Artifact and claim reconciliation

| Claim area | Canonical source | Reviewed conclusion |
| --- | --- | --- |
| Flagship demo | `artifacts/fabric-demo/manifest.json` and indexed artifacts | Synthetic p95 TTFT 947.314 ms healthy, 2470.966 ms degraded, and 947.314 ms restored; seven counterfactuals; completed guarded recovery. |
| H1 compiler comparison | `artifacts/fabric/evaluation/result.json` | 180 trials. Sequential was fastest at 616.434 ms; hierarchical was 626.120 ms, 1.57% slower. H1 is not supported against this already aligned fixture. |
| H2 twin | same result | Spearman 0.881 and Pareto regret 2.06%, but 28.35% median relative error and 0% interval coverage. Interval calibration failed. |
| H3 Autopsy | same result and `reports/autopsy-evaluation.md` | Top-1/top-3 accuracy 1.0 over 24 deterministic cases from two fault families; median 0.925 confidence is an evidence score, not probability. |
| H4 recovery | same result and report | Diagnosis-driven, threshold, and full replacement restored all modeled cases. Threshold had the shortest declared median action time. Times/costs are policy inputs and post-action requests are simulated. |
| ForgeCI | `artifacts/forgeci/demo/evaluation.json` | Local deterministic fixture only; 12.00% p99-TTFT regression and exact fixture bisection, not external runtime performance. |
| WarmPath | `artifacts/warmpath/manifest.json` and `artifacts/warmpath/evaluation/result.json` | 243-candidate local demo and H6 local/synthetic-payload evaluation; pinned-memory and warm-replica paths remain modeled where documented. |
| Rank ordering | `reports/rank-ordering-experiment.md` and referenced raw artifact | Synthetic-only result; integration decision remains `measure_on_hardware`, disabled by default. |

Resume metrics retain adjacent HTML source comments. The five bullets are
synthetic/local-fixture claims and do not imply GPU measurement.

## Corrected semantic boundaries

Autopsy documentation now matches the implementation: bounded lowest-RTT
Theil-Sen clock fitting, uncertainty-aware first-divergence ordering, exact
event/operation/rank/request matching, signal-specific sample counts, no-signal
rejection, per-hypothesis counterfactual attachment and reranking, and fault-
interval minimization. Simulator capture does not claim dependency edges it does
not reconstruct, and diagnosis does not consume ground-truth labels.

The simulator/compiler documentation now preserves the accepted review limits:
PP has no explicit layer-to-stage/send-receive schedule; decode lowering is an
opaque calibrated operation; layer/token expert exchange and overlap are
approximate; multi-hop service is conservatively additive; NIC placement is a
heuristic; and collective profile categories are aggregated. The topology hash
currently includes capture metadata, safely preventing but also limiting
equivalent-host profile reuse.

Runtime lowering now documents the exact fail-closed boundary from
`RUNTIME_ADAPTER_REVIEW.md` and `deploy/fabric/validated-versions.json`: vLLM
0.26.x, SGLang 0.5.x, Dynamo 1.2-1.3 v1beta1 DGD, Kubernetes 1.31-1.36 stable
fields, Modal 1.5.3, and Truss 0.18.24. Direct vLLM/SGLang reject multi-host
replicas; Dynamo is the multi-node engine path; generic Kubernetes rejects a
multi-node plan without a gang contract; Modal/Truss placement is advisory; and
the DGD subset is offline schema validation, not live operator admission.

Recovery documentation distinguishes controller/traffic rollback from general
inverse infrastructure actions. Security documentation records that the mock
fault API is disabled unless a scoped test/demo config explicitly opts in and
that gateway output/time/capacity bounds are enforced.

## Validation executed

```text
PYTHONPATH=python .venv/bin/pytest -q \
  tests/python/test_fabric_docs.py tests/python/test_fabric_demo.py \
  tests/python/test_autopsy_alignment.py tests/python/test_autopsy_diagnosis.py \
  tests/python/test_autopsy_counterfactual.py tests/python/test_autopsy_adversarial.py
55 passed

PYTHONPATH=python .venv/bin/pytest -q \
  tests/python/test_fabric_docs.py tests/python/test_fabric_demo.py \
  tests/python/test_fabric_evaluation.py::test_checked_in_evaluation_reports_match_raw_artifacts
6 passed

FabricDemoManifest + _validate_report
demo_manifest=valid, indexed artifacts=21

validate_evaluation_artifacts
evaluation_manifest=valid, plan/diagnosis/recovery trials=180/24/120

custom Markdown local-link and artifact-metric reconciliation scan
45 Markdown files inspected, 0 broken relative links
README/paper/resume/limitations metrics matched the JSON artifacts

rg -i 'TODO|FIXME|placeholder|stub|NotImplementedError|fake result|hard-coded benchmark' \
  <reviewed documentation>
0 matches

git diff --check -- README.md docs paper/fabric_extension
passed
```

The report validators recompute artifact and report hashes. A later release
regeneration may update capture timestamps and the recorded source commit; final
publication must retain the regenerated manifest/result/report set atomically.
