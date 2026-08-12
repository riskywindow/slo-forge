# SLOForge Helix Characterization integration graph

Baseline source: `d6f77c839334a4644b4e2edf36a7543c68670a2d`

History reconciled through: `8a9a505`

Host boundary: Apple M4 Pro CPU/unified memory; no compatible CUDA GPU,
multi-GPU allocation, multi-node allocation, RDMA path, or paid GPU budget.

This document is the current integration graph. The earlier
`docs/branchfabric/DEPENDENCY_GRAPH.md` preserves the launch plan. Neither graph
is measurement evidence. Evidence comes only from the referenced, hash-checked
artifacts.

## Dependency DAG

```text
B0  immutable baseline, source tag, hardware/software manifests
 |
 +--> S0  BranchWorkloadTrace v1 + StateOperationTrace v1 schemas
 |     +--> S1 Python/Rust conformance, canonical hashes, JSONL/Perfetto,
 |              bounded-buffer drop/filter accounting
 |
 +--> I0  Helix lifecycle instrumentation -----------------------+
 +--> I1  Continuum state lifecycle instrumentation -------------+--> V0
 +--> I2  environment/resource instrumentation ------------------+    vertical
 |                                                                  validation
 +--> M0  controlled matrix and calibrated synthetic projection -+
                                                                    |
                 +--------------------------------------------------+
                 v
 O0  disabled/minimal/full overhead with semantic/source guards
                 |
                 +--> C0 aligned Helix CPU seeds 41/73/113
                 +--> C1 Continuum CPU seeds 41/73/113
                 +--> C2 environment CPU study
                 +--> C3 metadata CPU study
                 +--> C4 strongest CPU software baselines
                 +--> H0 hardware capability gate
                       +--> H1 GPU measurement: UNAVAILABLE
                       +--> H2 multi-GPU measurement: UNAVAILABLE
                       +--> H3 multi-node measurement: UNAVAILABLE
                 |
                 v
 A0  workload DAG/divergence/waterfall + COW/page replay
 A1  transform/integrity + transport/multicast replay
 A2  statistical reconstruction + Amdahl/roofline/placement
 A3  active experiment ranking
                 |
                 v
 R0  evidence-bound requirements JSON and ISA classification
 R1  negative findings and hardware placement recommendations
 R2  characterization report and hardware handoff brief
                 |
                 v
 Q0  five-measurement independent replication
 Q1  multidisciplinary and adversarial reviews
                 |
                 v
 G0  make check + helix-check + trace check + CPU regeneration
     + requirements/report regeneration + clean-room archive
```

An unavailable hardware node is terminal for this host, not a failed CPU
dependency. It contributes an explicit unknown to `R0`; it contributes no
latency, bandwidth, size, or speedup value.

## Node register

| Node | Owner | State | Inputs | Outputs | Acceptance |
|---|---|---|---|---|---|
| B0 | root | complete, tracked | Helix release `d6f77c8` | baseline record, manifests, immutable tag | baseline `make check`; isolated seed-41 demo |
| M0 | root | implemented | baseline fixture semantics | controlled YAML matrix; calibrated projections | matrix/projection focused tests |
| S0-S1 | root/trace-schema lane | implemented | B0 | Python trace package, three JSON schemas, Rust wire types | `make branchfabric-trace-check` |
| I0 | root/Helix instrumentation lane | implemented, exercised | S0 | Helix lifecycle event stream | lifecycle and semantic-equivalence tests |
| I1 | root/Continuum instrumentation lane | implemented, exercised | S0 | Continuum state-operation event stream | Continuum harness and adapter tests |
| I2 | root | implemented, exercised | S0, I0/I1 | environment events and bounded resource traces | environment/resource tests |
| V0 | root | tracked first slice; aligned corpus uncommitted | S1, I0-I2 | branch/state JSONL, Perfetto, manifest, sharing analysis | runner tests; zero-drop/hash/semantic checks |
| O0 | root | complete, tracked valid v4 | V0 | randomized three-level raw samples and overhead report | overhead tests and report validity guards |
| C0-C4 | root, metadata_study, cli_workflow | executed; current outputs uncommitted | O0 and study-specific fixtures | CPU raw samples/traces and summaries | study-focused tests plus artifact integrity audit |
| H0-H3 | metadata_study | capability gate executed; hardware studies unavailable | B0 manifests and absent budget | explicit status, no metrics, exact future commands | hardware-study tests; GPU Make target disposition |
| A0 | network_transform + root | implementation complete; draft outputs uncommitted | C0/C1 | workload DAG, divergence, queues, waterfall, COW replay | workload/workflow tests; source hash audit |
| A1 | network_transform | implementation complete; draft outputs uncommitted | C1 | transform chain and transport/multicast analyses | transform/transport tests; source hash audit |
| A2 | root | models complete; final bound reports pending | C0-C4, A0/A1, explicit H unknowns | statistics, Amdahl bounds, roofline class, placement matrix | focused model tests; every datum bound to evidence |
| A3 | root | implementation complete; final queue pending | uncertainty, sensitivity, Amdahl leverage, cost | deterministic ranked next-experiment queue | prioritizer tests and reproducibility check |
| R0-R2 | requirements_report | in progress | reviewed A0-A3 and H0-H3 | requirements JSON, negative findings, design brief, final report | requirements schema; reference/hash audit; Make targets |
| Q0-Q1 | future review lanes | planned | frozen R0-R2 and corpus | replication/review records and scoped fixes | five replications; high/reasonable-medium closure |
| G0 | root | planned | all applicable nodes | committed reproducible release corpus | complete Make/clean-room command set |

## Evidence flow and immutability

```text
raw event/sample
  -> raw-file SHA-256
  -> trace/study manifest
  -> replay analysis with source references
  -> requirement value with experiment, class, count and statistic
  -> prose claim with exact artifact link
```

The allowed transitions are:

1. A measurement lane creates a new output directory. It never rewrites a
   prior raw directory.
2. An analysis lane reads raw files and writes a distinct derived artifact.
3. A requirements lane may emit a number only when the source artifact,
   evidence class, sample count and statistic are present. Otherwise it emits
   an explicit unknown or unavailable value.
4. A report cites the requirements/analysis artifact; it does not introduce a
   new measurement.
5. Replication reads the frozen corpus or creates a new independent raw output.
   It does not modify either source.

## Corpus checkpoints

| Checkpoint | State | Paths |
|---|---|---|
| Baseline | tracked | `artifacts/branchfabric/baseline/`; `artifacts/branchfabric/manifests/` |
| First vertical | tracked | `artifacts/branchfabric/characterization/first-vertical-seed-41-rerun/` |
| Valid overhead | tracked | `artifacts/branchfabric/overhead/helix-cpu-3seed-2rep-v4/` |
| Invalid overhead attempts | retained/excluded | earlier `helix-cpu-3seed-2rep*` directories with `INVALIDATED_ANALYSIS.json` |
| Aligned Helix corpus | uncommitted | `artifacts/branchfabric/characterization/vertical-seed-{41-final,73,113}/` |
| Continuum corpus | uncommitted | `artifacts/branchfabric/continuum/seed-{41,73,113}/` |
| CPU studies | uncommitted | `artifacts/branchfabric/{environment,metadata,software-baselines}/` |
| Draft replay analyses | uncommitted | `artifacts/branchfabric/analysis/{cow,workload,transform,transport}/` |
| Hardware disposition | uncommitted; no performance metrics | `artifacts/branchfabric/hardware/status-seed-20260809.json` |
| Requirements/reports/reviews | pending | `artifacts/branchfabric/requirements/`; `reports/branchfabric-characterization/`; final BranchFabric documents |

Uncommitted does not mean invalid. It means the artifact is not yet an
immutable release input and therefore cannot support a final completion claim.

## Capability gates and future hardware edges

The current hardware status record reports zero compatible CUDA GPUs and no
multi-node/RDMA allocation. Its GPU, multi-GPU and multi-node studies are
`UNAVAILABLE`, contain empty metric arrays, use no synthetic substitution, and
performed no provisioning. The future commands embedded in that record are
bounded, seed-explicit and non-provisioning:

- GPU: output `artifacts/branchfabric/characterization/hardware/gpu`, seed
  `20260809`, timeout 3,600 seconds, with explicit local GPU opt-in.
- Multi-GPU: output
  `artifacts/branchfabric/characterization/hardware/multi_gpu`, same seed and
  timeout, requiring at least two compatible CUDA GPUs and NCCL.
- Multi-node: output
  `artifacts/branchfabric/characterization/hardware/multi_node`, same seed and
  timeout, requiring an allocated multi-node network fabric.

None of these edges authorize paid resources. If a compatible host becomes
available, the hardware study must create a new exercised artifact and pass the
same evidence/reference gates before any requirement is revised.

## Critical path to publication

1. Freeze and hash-audit the currently uncommitted CPU corpus.
2. Regenerate Amdahl, roofline, placement and active-experiment outputs from
   that frozen corpus, preserving explicit hardware unknowns.
3. Compile `branchfabric_requirements.json`; reject every unbound value.
4. Generate the characterization report, design brief and
   `WHAT_NOT_TO_BUILD.md` from the compiled evidence.
5. Independently replicate the five highest-leverage CPU measurements.
6. Complete architecture, Continuum, GPU, network, storage/memory, statistics,
   bias, instrumentation-overhead, FPGA, SmartNIC/DPU and adversarial reviews.
7. Resolve all high-severity and reasonable medium-severity findings.
8. Run the complete integrated and clean-room gates from the committed source.

## Acceptance command map

```sh
make check
make helix-check
make branchfabric-trace-check
make branchfabric-characterization-cpu
make branchfabric-characterization-gpu
make branchfabric-characterization
make branchfabric-requirements
make branchfabric-report
make branchfabric-clean-room-test
```

The final gate additionally verifies that no generated hardware code, RTL,
FPGA simulator, hardware driver, active experiment process, or cloud resource
exists. Characterization completion does not authorize BranchFabric
implementation.
