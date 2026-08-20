# Agent 20 — Adversarial hardware terminal review

## Required answer

**No.** After giving software every reasonable chance within the authorized experiment, I would not independently spend engineering time investigating custom BranchFabric hardware for the remaining bottleneck.

Final classification: **SOFTWARE_WINS**

This is a conservative terminal hardware closure, not a measured end-to-end software victory. The mandated classification set has no `INCONCLUSIVE` option. `SOFTWARE_WINS` is the least overclaiming disposition because software measurably removed most of the physical pathology, while the integrated run produced no state-path critical-share, Amdahl, placement, or economic evidence that could support custom hardware.

## What software actually proved

The exact real-A100 micro-validation restored the 1,056,964,608-byte, 16K-prefix, fanout-8 state with 8/8 exact first resumed tokens and at least eight continuation tokens per branch. Allocator epochs, post-free ownership, fresh destination lifetimes, mapping commitments, integrity, and movement accounting all passed.

Against frozen v10, measured v11 changed:

- full physical amplification: 58× → 21.000762818×, removing 39,106,884,224 bytes or 63.79%;
- external amplification: 3× → 2.000257098×;
- avoidable amplification: 31× → 12.000762818×, removing 20,081,521,280 avoidable bytes;
- full-state passes: 30 → 10;
- peak host temporary memory: 2,965,372,928 → 143,552,512 bytes conservatively, a 20.66× reduction. The 58,720,256-byte number is pinned staging only.

The measured PCIe-A100 micro phase walls were 2.613090870 s export and 4.438193614 s restore. They are single instrumented observations and are not directly latency-comparable with frozen SXM4 v10.

## What remains

The micro trace shows a regular residual chain, but its dominant pieces are mixed:

- export is dominated by CPU publish SHA-256 (1.390905234 s aggregate) rather than the 0.133267776 s CUDA-event sum;
- restore includes 2.114326745 s of CPU SHA work and 4,480 small raw-bit validation launches;
- raw validation accounts for 0.442472927 s, or about 63.4% of restore CUDA-event work, making larger-run CUDA/Triton fusion the clearest next software target.

These overlapping aggregates cannot be summed as disjoint stages. More importantly, micro-local cost is not the same as share of an economically important end-to-end path.

## Why custom hardware fails the gate

The terminal integrated attempt aborted in GPU0's serving methodology before reclaim. It emitted no reclaim trigger and executed no v11 export, release, import, StatePassRecord, restore, or continuation work. Therefore there is no measured residual share of:

- HBM reclamation;
- useful GPU1 capacity;
- serving SLO restoration;
- rollout interruption;
- full preservation latency;
- trajectories per GPU-hour or dollar cost.

BranchFabric hardware interest requires all eight criteria. Only strong micro-level software optimization is established. The evidence does not establish a residual chain at ≥15% of an important integrated critical path, ≥1.20× realistic end-to-end economic improvement, integrated burst/concurrency targets, or an advantage from placement outside normal GPU execution. The residual is also mixed CPU hashing and small GPU launches, not a proven external bandwidth/dataflow bottleneck.

Coarse micro-only Amdahl is calculable from non-overlapping source, restore, and continuation walls. For the full micro preservation-plus-continuation interval, hypothetical 2×/5×/10×/free acceleration of the entire state pipeline gives 1.945×/4.491×/7.967×/35.267×. Those values accelerate the whole measured state path and are not evidence for any isolated hardware block. Every integrated impact—HBM reclamation, useful GPU1 capacity, SLO restoration, rollout restore, and the economic preservation transaction—remains null because the integrated timing DAG does not exist.

The preservation side has one measured 7.051284484 s state-path point. Under the explicit simplifying assumptions that kill and preserve fixed overheads are equal and preserve time stays locally constant, preserve would win below 18,878.83 recomputed tokens/s at the frozen workload. Actual recompute throughput, kill fixed overhead, and the break-even surface remain unidentified because kill/recompute did not run. Filling those nulls from v10 timings or projected bandwidth would fabricate evidence.

## Why the other labels are weaker

`GPU_SOFTWARE_WINS` would overclaim that CUDA/Triton captures most realistic headroom and that an external accelerator adds less than 1.15× end to end. The micro trace suggests a GPU fusion candidate, but the integrated evidence needed for those claims is absent.

`BRANCHFABRIC_HARDWARE_INTEREST` is expressly forbidden without measured integrated v11 timings and fails multiple mandatory criteria.

`PRESERVATION_NOT_ECONOMIC` cannot be selected because the kill/recompute arm and measured break-even fit do not exist.

`SOFTWARE_WINS` therefore means: the measured software result survives, while the custom-hardware hypothesis does not clear its burden of proof. It does **not** mean that remaining state work was measured below 10% of integrated paths or that ideal-free full-system speedup was measured below 1.15×.

## Terminal disposition

Hardware interest does not survive. Do not generate Experiment 005 and do not begin FPGA implementation. If work were ever resumed outside this terminal experiment, first batch/fuse raw validation and reduce CPU hash/launch overhead in ordinary software, then require a valid integrated timing DAG before reconsidering custom hardware.
