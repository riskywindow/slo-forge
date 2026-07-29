# BranchFabric verified no-build report

Decision: `FAIL_NO_BUILD`  
Required action: `TERMINATE_HARDWARE_PATH`  
Hardware implementation allowed: **false**

## Executive decision

No BranchFabric hardware primitive is justified by the evidence available in
this execution. No target-specific RTL, FPGA/DPU data path, hardware driver,
BranchFabric functional hardware model, or cycle-level simulator was created.
The selected placement is optimized CPU software. This is a verified negative
hardware result, not a deferred claim that hardware would help.

The result follows directly from absent authorization and evidence: no CUDA GPU,
multi-GPU system, GPU PCIe path, NVLink, RDMA, physical test NIC, FPGA, DPU, or
accelerator toolchain was available or authorized. All BranchFabric authorization
flags were absent, both BranchFabric budgets were zero, and external paid
resources were not created.

## Baseline and preserved prior result

- Immutable baseline: `46955be24d49af7090429444a0ef68f9a5695283`, tag
  `sloforge-branchfabric-execution-baseline-46955be`.
- Host: Apple M4 Pro Mac16,8, 12 logical CPUs, 25,769,803,776 bytes memory,
  integrated 16-core Metal GPU; CUDA-compatible GPU count zero.
- Toolchain: macOS 15.6.1, Python 3.12.7, Rust/Cargo 1.93.1, uv 0.10.2,
  Node 22.16.0, npm 11.4.2, Apple clang 16.0.0.
- The original 40-artifact characterization corpus reproduced with aggregate
  SHA-256 `9e000d81afc102c5a05666d6fb48859cb4e136a72ad6fc59f5ce086bfaa0b5d3`.
  Requirements and generated report reproduced byte-for-byte.
- The prior negative result remains visible: four siblings, serialized
  concurrency one, transfer fanout one, zero multicast opportunity, zero
  `REQUIRED` or `HIGH_VALUE` operations, and a maximum individual-operation
  ideal-free Amdahl bound of 1.016977x.

## Highest-fidelity available workloads

### Branch fanout

The execution ran 144 raw CPU-reference samples over seeds 41, 73, and 113,
randomized order, one declared warmup and three measured repetitions. It used
two controlled workload classes and fanouts 8, 16, and 32:

| Workload | Context | Suffix/divergence | Serialized root state |
|---|---:|---|---:|
| coding-agent reference | 2,048 tokens | 24-token immediate RNG divergence | 532,200–532,205 B |
| reasoning/verification reference | 3,968 tokens | 32-token late divergence after 16 shared tokens | 1,030,592–1,030,593 B |

These are deterministic Continuum HybridDecoder reference states, not a real
transformer allocation. Byte totals are deterministic serialized and
content-addressed accounting, not RSS, physical pages, HBM, or GPU allocation.
The public reference adapter is bounded at 4,096 tokens, so 8K, 16K, and 32K
contexts were unsupported and were not attempted. Queue concurrency remained
one and no state was physically delivered to multiple workers.

### Causal local reclamation transaction

The execution ran 36 randomized trials: three seeds, two repetitions, three
tracing controls, and two matched software baselines. Each trial causally
compiled a Helix serving-spike plan; stopped rollout admission; forked eight
branches; appended private suffixes; paused them; extracted deltas; wrote
incremental checkpoints; transformed layouts; transferred hash-validated
content; committed ownership; resumed all branches; and rejected stale-source
and corrupt-state faults.

This transaction reclaimed eight modeled CPU scheduler units and exactly zero
physical GPU capacity or GPU bytes. The serving SLO was a deterministic
scheduler-model result, not an observed production serving-latency trace. It is
therefore integration and software evidence, not the required physical
GPU-reclamation or full Helix learning-transaction evidence.

## Strongest reasonable software baselines

1. Shared-root, content-addressed, lazy-COW branch readiness replaced eager
   per-sibling private materialization. Across the six workload/fanout cells,
   median readiness improved 102.28x–237.71x and ready-time serialized byte
   accounting fell 88.89%, 94.12%, and 96.97% for fanouts 8, 16, and 32.
2. Reclamation used bounded four-worker checkpoint, transform, and resume
   stages with an aggregate 30-second deadline and queue capacity eight.
3. The optimized transfer formed one sorted unique content-addressed chunk set
   for the whole branch group and performed one integrity-checked batch transfer.
   It moved 6,812–6,816 unique bytes rather than 54,496–54,528 logical repeated
   bytes, an exact 87.5% byte reduction.
4. Transaction hardening durably aborts prepared destination state, rolls the
   coordinator through `ABORTING -> ROLLED_BACK`, releases a pre-commit source
   fence, leaves gateway/ownership epochs unchanged, and permits a clean retry.

The matched local reclamation software result was:

| Metric | Existing serial | Optimized | Paired result |
|---|---:|---:|---:|
| interruption median | 903.37 ms | 707.21 ms | 1.2721x; 95% CI 1.2445–1.2980x |
| empirical p99/max interruption | 987.07 ms | 775.82 ms | 21.40% lower descriptively |
| migration median | 334.27 ms | 143.46 ms | 2.2778x; 95% CI 2.1663–2.3871x |
| total-wall median | 1,271.66 ms | 1,078.47 ms | 1.1675x; 95% CI 1.1448–1.1875x |
| branch-readiness median | 134.94 ms | 136.71 ms | 0.9803x; optimized path 2.01% slower |

These are host CPU/local-file software effects. None is a hardware prediction
or hardware speedup.

## Gate result

Thresholds were not changed. The final evidence-bound result contains four
candidate audits and zero passing candidates:

| Candidate | Real evidence | End-to-end relevance | System headroom | Platform feasibility | Workload value | Result |
|---|---|---|---|---|---|---|
| checkpoint/transform/transfer chain | FAIL | PASS | FAIL | FAIL | FAIL | `NOT_JUSTIFIED` |
| shared-root/COW hardware | FAIL | FAIL | FAIL | FAIL | FAIL | `NOT_JUSTIFIED` |
| one-to-many multicast | FAIL | FAIL | FAIL | FAIL | FAIL | `NOT_JUSTIFIED` |
| branch-translation metadata unit | FAIL | FAIL | FAIL | FAIL | FAIL | `NOT_JUSTIFIED` |

The local optimized migration phase is 19.90% of local interruption (95% CI
19.61%–22.45%), so the data-path chain passes only the relevance screen. It
still fails mandatory real evidence, calibrated system-level hardware headroom,
platform feasibility, and data-path workload value. A software improvement is
not inserted into the hardware-headroom field.

The optimized fanout readiness path is at most 0.006006% of its measured local
lifecycle. Even making it free gives at most 1.000060x local lifecycle speedup.
No physical repeated unicast exists, and optimized metadata remains only
0.0948% of its measured target window in the historical characterization.

## Architecture and implementation disposition

- Selected: CPU shared-root/lazy-COW; bounded parallel checkpoint/transform/
  resume; batched content-addressed transfer reuse; existing Continuum
  transactional ownership and integrity semantics.
- Rejected: GPU copy/transform kernel, multicast engine, metadata accelerator,
  shared-root HBM/CXL store, FPGA, SmartNIC, and DPU.
- Not created by policy: BranchStateGraph hardware extension, BranchFabric
  command ISA, functional hardware model, cycle simulator, target driver, RTL,
  HLS, FPGA image, or DPU program. Those artifacts are conditional on a gate
  pass and would be misleading on this completion branch.
- Optional Continuum and Helix non-BranchFabric paths remain the only execution
  paths; no hidden fallback or CPU substitution is presented as hardware.

## Verification

- Final `make check`: 1,268 Python tests passed, six declared optional hardware
  skips; Ruff format/lint and Mypy passed; Rust formatting, warning-denied
  Clippy, all workspace tests/doc tests passed; UI typecheck/lint, 37 tests with
  one fixture skip, and production build passed.
- Focused BranchFabric gate, analysis, reclamation, fanout, and no-build tests
  passed. The final gate regenerated from raw evidence as `FAIL_NO_BUILD`.
- A clean archive of `979a1df` bootstrapped, passed 156 Python and five Rust
  trace tests, completed all six CPU characterization stages, regenerated the
  negative requirements/report, and built both distribution packages.
- The final core, Fabric, Autopsy, ForgeCI, WarmPath, Genesis zero-day,
  Continuum demo/migration/fault/fork/compatibility, and Helix CPU
  demonstrations all passed.
- Rust Continuum protocol model checking passed five tests within its declared
  bound; Rust state-transaction verification passed nine tests.
- All 36 reclamation trials completed transactions, rejected stale sources and
  corrupted state, reported no hidden fallback, and reclaimed zero physical GPU
  bytes. Injected post-import validation failure now proves durable rollback and
  successful retry without an epoch or gateway-watermark change.
- Independent reviews reproduced the historical Amdahl bound, complete fanout
  summary, reclamation effects, 87.5% transfer-byte reuse, all gate hashes, and
  the zero-candidate no-build outcome.
- The final adversarial closure verified all 18 gate references, all four
  producer-source hashes, byte-exact gate/report regeneration, and 23 focused
  tests. It concluded that a neutral architect would not select this hardware.

## End-to-end Helix result

There is no hardware-backed end-to-end Helix improvement result. The only new
causal transaction is a deterministic CPU-reference/local-file preservation and
resume vertical. Its optimized software interruption improved by 21.38% in
paired trials, while branch readiness itself regressed by 2.01%. It did not run
production serving, a real transformer, reward/training/evaluation/promotion,
or physical GPU reassignment. Previous Helix and Continuum demonstrations remain
non-regression requirements and pass, but they cannot supply the missing
BranchFabric hardware counterfactual.

## Remaining limitations and evidence needed to reopen

- No real transformer or production hybrid state allocation; no 8K/16K/32K
  contexts; no HBM allocation or allocator behavior.
- No CUDA, PCIe-GPU, NVLink, GPU-to-GPU, RDMA, physical NIC, multi-GPU,
  multi-node, FPGA, or DPU measurement.
- No physical one-to-many state delivery or multicast opportunity.
- No production serving spike or causal physical-GPU reclamation.
- Reclamation has one deterministic reference workload class and 18 matched
  pairs per baseline. Its p99 is descriptive, not a stable production tail.
- The original measurement schemas stored the semantic baseline revision; the
  execution provenance manifest separately binds generator source hashes,
  generator revisions, publication revisions, raw artifacts, and analyses.
- No billable resource was authorized or used. Future work must satisfy the
  authorization preflight and measurement contract in the future plan before
  the gate can be rerun.

## Artifacts

- Gate: `artifacts/branchfabric/gates/branchfabric_gate_result.json` and
  `BRANCHFABRIC_GATE_REPORT.md`.
- Fanout: `artifacts/branchfabric/execution/fanout/`.
- Reclamation: `artifacts/branchfabric/execution/reclamation/`.
- Replication/reviews: `artifacts/branchfabric/execution/replication/` and
  `docs/branchfabric/reviews/`.
- Final adversarial closure:
  `artifacts/branchfabric/execution/reviews/hardware-adversarial-closure.json`
  and
  `docs/branchfabric/reviews/EXECUTION_HARDWARE_ADVERSARIAL_CLOSURE.md`.
- Provenance and ledgers: `artifacts/branchfabric/execution/PROVENANCE_MANIFEST.json`
  and `artifacts/branchfabric/execution/ledgers/`.
- Future campaign: `docs/branchfabric/FUTURE_MEASUREMENT_PLAN.md`.
- Final consolidated report: `BRANCHFABRIC_FINAL_REPORT.md`.
