# SLOForge BranchFabric final report

Final outcome: `FAIL_NO_BUILD`

Hardware justified: **no**

Completion branch: verified no-build

## Revisions and decision integrity

- Immutable execution baseline:
  `46955be24d49af7090429444a0ef68f9a5695283`.
- Final implementation source:
  `d113e12b5d618fa3b39b41a30f5f272bd96a18b2`.
- Final batched-reclamation evidence publication:
  `5ae509add7154daa92816de5f5b4a9c3a3d2de0a`.
- Final no-build report publication:
  `26ecc48ca8c2e9302a6744e98cb4ed4013c90e96`.
- No gate threshold was loosened. No synthetic or CPU-reference metric was
  reclassified as target-hardware evidence.
- The historical negative characterization remains in the corpus and reports.

The original characterization found only four controlled siblings, synthetic
state sizes, fanout and concurrency one, no multicast opportunity, no causal
reclamation transaction, no compatible GPU/fabric/accelerator measurement,
zero `REQUIRED`/`HIGH_VALUE` operations, and at most 1.016977x ideal-free
individual-operation Amdahl improvement. This execution preserved that result
and added higher-fanout CPU-reference and causal local transaction evidence. It
did not manufacture the missing hardware evidence.

## Hardware and software environment

The measured host was an Apple M4 Pro Mac16,8 with 12 logical CPUs,
25,769,803,776 bytes memory, and an integrated 16-core Metal GPU. Host CPU time
and local filesystem behavior were measured. The Apple GPU was not used as a
substitute CUDA target.

There was no CUDA GPU, PyTorch, Triton, vLLM, SGLang, NVLink, GPU PCIe transfer
path, multi-GPU or multi-node allocation, RDMA, physical test NIC, FPGA, DPU,
accelerator toolchain, or corresponding driver. All BranchFabric authorization
flags were absent, `SLOFORGE_BRANCHFABRIC_TARGET` defaulted to `auto`, both
BranchFabric budgets were zero, spend was $0, and no external resource was
created.

The exercised toolchain was macOS 15.6.1, Python 3.12.7, uv 0.10.2, Rust/Cargo
1.93.1, Node 22.16.0, npm 11.4.2, and Apple clang 16.0.0.

## Workloads and measurements

### CPU-reference fanout

- Two controlled workload classes: coding-agent immediate divergence at 2,048
  tokens and reasoning/verification late divergence at 3,968 tokens.
- Fanouts 8, 16, and 32; seeds 41, 73, and 113; randomized order; one warmup
  plus three measured repetitions; 144 raw records.
- Root serialized state: approximately 532 KiB and 1,031 KiB, including
  reference KV, recurrent, sampler, guided-decoding, page metadata, and runtime
  metadata accounting.
- The bytes are deterministic serialization/content-addressing, not resident
  physical memory or HBM. The runtime is a deterministic HybridDecoder fixture,
  not a production transformer. The public adapter capped context at 4,096, so
  8K/16K/32K were not supported.
- No physical multi-destination transfer occurred; queue concurrency was one.

### CPU-reference causal reclamation

- One reference reclamation workload, eight siblings, seeds 41/73/113, two
  repetitions, disabled/minimal/full tracing, serial/optimized baselines, and
  randomized run order produced 36 raw trials.
- The causal local sequence executed Helix scheduling, admission stop, fork,
  append/COW, pause, final delta, incremental checkpoint, cross-layout
  transform, integrity-checked transfer, ownership commit, stale-epoch fencing,
  resume, and corruption rejection.
- It reclaimed eight modeled CPU scheduler units but zero physical GPU capacity
  and zero GPU bytes. Its modeled serving SLO is not a production latency trace.
- Maximum host concurrency was four and maximum queue occupancy eight.

## Strongest software baselines

The final baseline uses shared-root lazy COW for readiness, content-addressed
immutable chunks, bounded four-worker checkpoint/transform/resume stages, an
aggregate 30-second queue deadline, direct reference layout conversion, one
sorted batched transfer of the unique group chunk set, SHA-256 integrity checks,
and Continuum transactional publication.

Against eager private materialization, shared-root readiness improved
102.28x–237.71x and reduced ready-time serialized bytes by 88.89%–96.97%.
After optimization, readiness was at most 0.006006% of its local lifecycle;
making it free yields at most 1.000060x additional local speedup.

The batched group transfer reduced repeated local transfer bytes from
54,496–54,528 B to 6,812–6,816 B, exactly 87.5%. Across 18 matched reclamation
pairs, the optimized path produced:

| Metric | Existing median | Optimized median | Paired estimate |
|---|---:|---:|---:|
| migration | 334.27 ms | 143.46 ms | 2.2778x, 95% CI 2.1663–2.3871x |
| interruption | 903.37 ms | 707.21 ms | 1.2721x, 95% CI 1.2445–1.2980x |
| total wall | 1,271.66 ms | 1,078.47 ms | 1.1675x, 95% CI 1.1448–1.1875x |
| branch readiness | 134.94 ms | 136.71 ms | 0.9803x, 95% CI 0.9380–0.9996x |

The interruption reduction is 21.38%; branch readiness regressed 2.01%. These
are CPU/local-file software measurements, not a hardware speedup or prediction.

## Gate calculations and outcome

The checkpoint/transform/transfer chain passed only end-to-end relevance: the
optimized full-trace migration phase was 19.90% of local interruption, with a
95% interval of 19.61%–22.45%. It failed real evidence, system-level hardware
headroom, platform feasibility, and data-path value. No target-hardware service
curve or calibrated hardware model exists, so no software speedup was copied
into a hardware-headroom field.

| Primitive | Real evidence | Relevance | Hardware headroom | Feasibility | Value gate | Classification |
|---|---|---|---|---|---|---|
| checkpoint/transform/transfer | FAIL | PASS | FAIL | FAIL | FAIL | `NOT_JUSTIFIED` |
| shared-root/COW hardware | FAIL | FAIL | FAIL | FAIL | FAIL | `NOT_JUSTIFIED` |
| multicast | FAIL | FAIL | FAIL | FAIL | FAIL | `NOT_JUSTIFIED` |
| metadata accelerator | FAIL | FAIL | FAIL | FAIL | FAIL | `NOT_JUSTIFIED` |

Final gate: zero passing candidates, `hardware_implementation_allowed=false`,
`functional_model_or_cycle_simulator_allowed=false`, outcome `FAIL_NO_BUILD`,
required action `TERMINATE_HARDWARE_PATH`.

## Architecture and implementation status

Selected architecture: CPU software using existing Continuum state semantics
and Helix control. Rejected placements: GPU kernel, FPGA, DPU/SmartNIC,
multicast engine, metadata unit, and CXL/HBM shared-root store.

Implemented in this execution:

- fail-closed typed evidence gates with raw SHA-256/metric bindings;
- deterministic fanout and causal reclamation measurement verticals;
- raw-bound bootstrap/effect analysis and independent replication;
- bounded queue deadlines and batched content-addressed transfer reuse;
- durable pre-commit resume rollback and successful retry;
- a no-build verifier, provenance manifest, ledgers, reviews, and future
  real-hardware measurement contract.

Not implemented, because the gate failed: BranchStateGraph extension, hardware
command IR, hardware functional model, cycle simulator, target driver, RTL, HLS,
FPGA image, DPU program, or `hardware/branchfabric`/`sim/branchfabric` tree.

## Verification status

- `make check`: 1,268 Python passed, six optional hardware skips; Ruff, Mypy,
  Rust format, warning-denied Clippy, all Rust workspace tests/doc tests, UI
  typecheck/lint, 37 UI tests with one fixture skip, and production build passed.
- `make helix-check`: 188 Python plus Helix Rust tests passed.
- `make branchfabric-trace-check`: 156 Python and five Rust trace tests passed.
- `make branchfabric-clean-room-test`: a clean archive of `979a1df`
  bootstrapped, passed 156 Python and five Rust trace tests, completed all six
  CPU characterization stages, regenerated the negative requirements/report,
  and built both distribution packages.
- The final historical CPU demonstration set passed: core, Fabric, Autopsy,
  ForgeCI, WarmPath, Genesis zero-day, Continuum demo/migration/fault/fork/
  compatibility, and Helix.
- Rust Continuum state model checking: five tests passed within declared
  bounds. Rust transaction protocol: nine tests passed.
- Post-import validation fault: destination staging removed, durable journal
  reached `ABORTING -> ROLLED_BACK`, source returned to paused, lease epoch and
  gateway watermark stayed unchanged, no recoverable transaction remained, and
  a retry completed at epoch two.
- All 36 canonical reclamation trials completed eight transactions, restored
  the modeled SLO, rejected stale-source/corruption faults, and reported no
  hidden fallback.
- Independent regeneration found zero mismatches in deterministic fields,
  reproduced all gate evidence hashes and calculations, and affirmed no-build.
- The final adversarial closure at `ea3c505` verified all 18 gate references,
  all four producer-source hashes, byte-exact gate/report regeneration, and 23
  focused tests. It answered that a neutral workload-selected architecture
  would not choose this hardware.

## End-to-end Helix result

No BranchFabric hardware path exists, so there is no hardware-enabled end-to-end
Helix improvement. The new causal CPU-reference transaction shows a 21.38%
software interruption reduction but no physical serving traffic, GPU release,
real transformer state, rollout reward/training/evaluation/promotion, or
trajectories-per-GPU-hour measurement. Prior Helix/Continuum demonstrations and
all existing tests remain passing non-regressions; they are not relabeled as a
BranchFabric hardware result.

## Hardware-backed versus simulated status

- Hardware-backed: Apple M4 Pro host inventory and CPU wall-clock/local-file
  behavior only.
- CPU-reference: model state, fanout, layout transform, preservation, and
  reclamation transaction.
- Artifact replay: statistics, Amdahl calculations, gate evaluation, and
  independent deterministic checks.
- Hardware simulated: none in the passing evidence and none used to pass a gate.
- Physical accelerator: none implemented or measured.

## Negative workload and findings

The coding workload diverges immediately, providing the high-divergence
negative case; shared-root lifetime is short and only 163 new suffix bytes are
identical across destinations. No physical multicast is present. The optimized
fanout readiness path has negligible residual local leverage. Metadata remains
0.0948% of its historical target window. No candidate has measured GPU
interference, HBM pressure, physical state amplification, repeated unicast,
hardware operation rate, bandwidth target, or target resource envelope.

The independent adversarial answer is no: absent the BranchFabric name and
prior proposal, the measured workload would select optimized CPU software, not
an accelerator. The work demonstrates disciplined hardware/software gating and
transactional software integration; it does not yet demonstrate a physical
hardware/software co-design speedup.

## Remaining limitations and unexercised paths

- Real transformer/hybrid production state and 8K/16K/32K contexts.
- CUDA/HBM/copy-engine/interference, PCIe, NVLink, multi-GPU, multi-node,
  RDMA/NIC, FPGA, and DPU paths.
- Physical identical-state delivery to multiple destinations.
- Production serving spike and causal physical-GPU reclamation.
- Full BranchFabric-affected Helix learning transaction and trajectories per
  GPU-hour.
- Reclamation has one reference workload class and 18 matched pairs per
  baseline; empirical p99 is the sample maximum.
- The canonical raw schemas retain the semantic baseline; the separate
  provenance manifest binds generator source hashes and publication revisions.
- Aggregate Python deadlines bound caller wait but cannot forcibly cancel an
  already executing daemon task.

## Artifact locations

- Baseline: `artifacts/branchfabric/execution/baseline/record.json`.
- Historical corpus: `artifacts/branchfabric/CORPUS_MANIFEST.json`.
- Fanout raw/summary: `artifacts/branchfabric/execution/fanout/`.
- Reclamation raw/analysis: `artifacts/branchfabric/execution/reclamation/`.
- Final gate: `artifacts/branchfabric/gates/branchfabric_gate_result.json` and
  `BRANCHFABRIC_GATE_REPORT.md`.
- Replication/reviews: `artifacts/branchfabric/execution/replication/` and
  `docs/branchfabric/reviews/`.
- Final adversarial closure:
  `artifacts/branchfabric/execution/reviews/hardware-adversarial-closure.json`
  and
  `docs/branchfabric/reviews/EXECUTION_HARDWARE_ADVERSARIAL_CLOSURE.md`.
- Provenance/ledgers: `artifacts/branchfabric/execution/PROVENANCE_MANIFEST.json`
  and `artifacts/branchfabric/execution/ledgers/`.
- Verified no-build: `BRANCHFABRIC_NO_BUILD_REPORT.md` and
  `artifacts/branchfabric/execution/no-build-verification.json`.
- Future evidence: `docs/branchfabric/FUTURE_MEASUREMENT_PLAN.md`.
