# BranchFabric gated execution dependency graph

Status: active. Hardware implementation is locked until the machine-readable
gate artifact reports that every mandatory gate and at least one workload-value
gate passed from hardware-backed evidence.

## Fixed execution boundary

- Baseline commit: `46955be24d49af7090429444a0ef68f9a5695283`.
- Immutable tag: `sloforge-branchfabric-execution-baseline-46955be`.
- Available concurrency: four total agent slots, so root plus three subagents is
  the maximum. Eight simultaneous agents are unavailable on this runner.
- BranchFabric GPU, multi-GPU, multi-node, RTL, FPGA-build, accelerator, and
  external-resource authorizations are absent. Authorized spend is zero.
- No RTL, target-specific hardware directory, functional hardware model, or
  cycle simulator may start unless the gate artifact says `PASS`. Synthetic or
  CPU-only evidence cannot satisfy the real-evidence gate.

## Dependency graph

```text
P0 baseline + historical negative audit
  |-- P1A real/highest-fidelity model-state fanout (8/16/32; 3 seeds; 2 classes)
  |-- P1B causal local capacity-reclamation transaction + queue/concurrency trace
  |-- P1C independent corpus/measurement replication + statistical review
  `-- P1D strongest reasonable metadata/transform/transfer/sharing baselines
          |
          v
     P2 evidence normalization + tracing controls
          |
          v
     P3 gate compiler + independent adversarial review
          |
          +-- FAIL --> P4A verified no-build reports + future commands
          |
          `-- PASS --> P4B semantic model/simulator/target selection
                              |
                              v
                         authorized minimal hardware only
```

On this host, P4B is unreachable unless new authorized compatible hardware
evidence appears during the run. The default and expected completion branch is
P4A; this is a gate consequence, not a predetermined relabeling of a metric.

## Active ownership and acceptance edges

| Lane | Owner | Branch/worktree | Owned files | Depends on | Acceptance |
|---|---|---|---|---|---|
| P0/P1C negative-result replication | `negative_replication` | `branchfabric/negative-replication`; isolated worktree | execution replication review and new replication artifacts only | sealed corpus | reproduce the five central findings and challenge any positive reinterpretation |
| P1A model-state/fanout vertical | `fanout_vertical` | `branchfabric/fanout-vertical`; isolated worktree | new fanout workload module/tests/artifacts only | existing Helix/Continuum state semantics | deterministic 8/16/32 siblings, two classes, seeds 41/73/113, raw samples and sharing/readiness |
| P1B reclamation/concurrency vertical | `reclamation_vertical` | `branchfabric/reclamation-vertical`; isolated worktree | new causal transaction module/tests/artifacts only | Continuum transaction and Helix scheduler | serving spike -> pause/checkpoint/reclaim/preserve/resume with queue percentiles and no accepted corruption |
| P1D/gate/integration | root | `main` | shared manifests, ledgers, baselines, gate compiler, reports, final integration | P1A-P1C | targeted tests, independently regenerated gate, `make check` |

Completed agents are replaced immediately while an independent ready lane
exists. Hardware lanes are deliberately absent from the ready set until a gate
`PASS` exists.

