# BranchFabric final hardware adversarial closure

Status: **PASS_FINAL_NO_BUILD**

This read-only closure supersedes the preliminary evidence snapshot in
`docs/branchfabric/reviews/EXECUTION_HARDWARE_ADVERSARIAL_REVIEW.md`. It does
not erase that review or its findings; it verifies their final disposition
against main commit `53dae56390f7b6c2b4e4ba963ab6ea45543b10eb` and the exact
current artifacts. No raw evidence, source, gate logic, prior review, or final
report was modified.

## Closure verdict

The verified result is **no-build**. No BranchFabric candidate passes all
mandatory gates plus a workload-value gate. Hardware implementation is false,
functional-model/cycle-simulator permission is false, and the required action
is `TERMINATE_HARDWARE_PATH`.

The strongest measured improvements are already software improvements:

- fanout uses `shared_root_cow_lazy`; its measured readiness improvement over
  eager private materialization is large, but optimized readiness is at most
  `0.000060056175263484067` of the local lifecycle. Making it free yields only
  `1.0000600597822242x` ideal local-lifecycle speedup;
- reclamation uses bounded parallel checkpoint/transform/resume plus one batch
  of unique content-addressed transfer chunks. Across 18 matched trials, median
  interruption speedup is `1.2720844760027905x`, with 95% bootstrap interval
  `1.2445075107013515x`--`1.2979926622703524x`;
- the batched transfer reduces 54,496--54,528 logical bytes to 6,812--6,816
  unique local-file bytes: exactly 87.5% saved in every optimized trial.

These results show that software batching and reuse were necessary. They do
not constitute target-hardware headroom. The gate correctly leaves all six
hardware-headroom lower bounds null.

## Exact artifact closure

| Artifact | Verified SHA-256 |
|---|---|
| Gate input | `4b70653b883e6a8c6a8b2cdd6f7768e33680ec7a6c1f5368a7467efb6eff8e6d` |
| Gate result | `0a96f44800ae74c36e62e5744ed877445910119afc5ecfcc75ad4b9bbf0c62b4` |
| Gate report | `8c0f76e435b9fdaffd8b8db9fc2e7d7c74ef959dcd150d680a8427af5d17234f` |
| Reclamation raw trials | `795a6803c1123555ea5804ddfeca711805235b89eeb30c5e723299bd26e74408` |
| Reclamation analysis | `cbd27f984fa412cd33fd3714b66f8978ab3311736127058b04efc2fdab0adbf0` |
| Fanout raw samples | `af616379666cc4d9043be355976016cc06a57a87b07caa8b60b2ad69d901b810` |
| Fanout summary | `51768b2b164caacb65d0de4ca3307737c4eda8c83cef030a89dd2597fcffebc4` |
| Execution provenance manifest | `555f682bf3801104e996b7a7d8f14e2f3bff2dc7912735868f72d19f6dfda0d0` |
| No-build verification | `f2e87da54be0bb2ae1b37b9d141788b959ce58a88bf4a00ba2f4db2eb4e03965` |

All 18 evidence references in the gate input exist and match their bound
hashes. The gate result and report reproduce byte-for-byte. The no-build
verifier independently returns `PASS_VERIFIED_NO_BUILD`. The provenance
manifest's four generator revisions reproduce the declared source hashes,
closing the earlier producer-identity finding.

## Final candidate matrix

| Candidate | Real evidence | Relevance | Hardware headroom | Platform | Value | Disposition |
|---|---|---|---|---|---|---|
| `checkpoint_transform_transfer_chain` | FAIL | PASS | FAIL | FAIL | FAIL | `NOT_JUSTIFIED` |
| `shared_root_cow` | FAIL | FAIL | FAIL | FAIL | FAIL | `NOT_JUSTIFIED` |
| `one_to_many_multicast` | FAIL | FAIL | FAIL | FAIL | FAIL | `NOT_JUSTIFIED` |
| `branch_translation_metadata` | FAIL | FAIL | FAIL | FAIL | FAIL | `NOT_JUSTIFIED` |

The checkpoint/transform/transfer chain passes relevance only because optimized
full-trace migration is a raw-bound `0.1990330675356517` of interruption. It
still has one CPU-reference workload class, 6,816-byte working set, no target
hardware samples, no calibrated hardware service curve, no hardware headroom,
no byte/operation-rate requirement, no bandwidth/latency target, and no
credible resource estimate. A relevance PASS alone cannot authorize hardware.

## Authorization and physical-hardware closure

Every BranchFabric target, budget, GPU, multi-GPU, multi-node, RTL, FPGA-build,
and external-resource variable is absent. Effective budgets are zero. The spend
ledger is empty, and the resource ledger records that no external GPU, FPGA,
DPU, or multi-node resource was created.

`hardware/branchfabric` and `sim/branchfabric` are absent. No BranchFabric RTL,
functional model, cycle simulator, bitstream, FPGA/DPU image, CUDA result,
physical GPU transfer, NVLink, RDMA, or physical multicast result exists. Every
reclamation trial uses the requested CPU-reference local-file engine, reports
no hidden fallback, and reclaims zero physical GPU capacity and bytes. The
campaign has `hardware_claim=false` and `gpu_measurement=false`; the analysis
has `target_hardware_measured=false` and
`calibrated_hardware_model_available=false`.

## Earlier adversarial findings

- `BF-HA-H4` is resolved: the final gate binds fanout and reclamation evidence.
- `BF-HA-M1` is resolved: the provenance manifest distinguishes semantic
  baseline, generator revision, publication revision, analyzer revision, and
  dirty-state policy; every declared source hash verifies from Git.
- `BF-HA-M2` is resolved: the strongest local transfer baseline performs one
  unique-digest batch and demonstrates 87.5% byte reuse.
- `BF-HA-M3` is resolved for gate use: matched paired effects, deterministic
  bootstrap intervals, and tracing controls exist. Small-sample p99 values
  remain explicitly descriptive and are not hardware-headroom bounds.

No unresolved finding can make the hardware case stronger. The remaining
limitations are the evidence gaps that require no-build: reference model state,
contexts below 8K, no compatible CUDA GPU, no multi-GPU or physical fabric, one
synthetic reclamation workload, zero physical GPU reclamation, and no
end-to-end production serving measurement.

## Required adversarial answer

**Would this hardware have been selected from the measured workload if the
BranchFabric name and prior architecture proposal did not already exist?**

**No.** A neutral workload-selected design would retain shared-root/lazy-COW,
bounded parallel execution, and batched content-addressed reuse in software.
It would request real model/GPU/network evidence before selecting an FPGA,
DPU, multicast engine, GPU kernel, metadata accelerator, or CXL/HBM store.

## Modal/Baseten hiring answer

**Does this demonstrate real hardware-software co-design, or is it a synthetic
accel attached to a software project without end-to-end relevance?**

It demonstrates credible hardware-software **design judgment and gate
discipline**, but not a physical accelerator implementation or hardware-backed
end-to-end speedup. It is **not** a synthetic accelerator attached to the
project: the evidence gate prevented one from being built. This is a strong
systems and negative-design-selection signal, but it must not be represented as
delivered FPGA, DPU, or GPU co-design hardware.

## Acceptance

Focused Amdahl, fanout, reclamation, paired-analysis, gate, and no-build tests
passed: 23 tests. Gate recompilation, no-build verification, evidence-hash
closure, producer-source verification, JSON parsing, and `git diff --check`
also passed.

Final reviewer decision:
`PASS_FINAL_NO_BUILD_REJECT_HARDWARE_RETAIN_OPTIMIZED_SOFTWARE`.

