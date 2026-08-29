# BranchFabric Hardware Gate

Final classification: **SOFTWARE_WINS**.

This is the least-overclaiming required exact-one terminal disposition and a conservative custom-hardware closure, not a claim of measured end-to-end integrated software victory. The integrated failure left all four strict classification predicates unestablished; in particular, the strict `SOFTWARE_WINS` end-to-end performance predicate was not measured.

The measured software path reduced full physical work from 58× to 21.000762818×, removed 39.107 GB of work, reduced full-state pass families from 30 to 10, lowered temporary host memory by 20.657×, and preserved exact 8/8 continuation correctness.

The remaining micro path is a mixed source chain (`native read → gather/repack → hash → D2H`) and restore chain (`H2D → direct native scatter → raw-bit/hash validation`). Complete micro export and restore stages account for 37.06% and 62.94% of measured micro state-path wall time. These coarse shares do not establish a hardware target.

The canonical residual record is `artifacts/branchfabric/gpu-validation/experiment-004/v11/amdahl/residual-chains.json`. It preserves logical and physical bytes, wall/CUDA/synchronization views, an explicit null for unidentifiable CPU time, the micro-only sensitivity envelope, and null/NOT_REACHED cells for every required integrated path. `artifacts/branchfabric/gpu-validation/experiment-004/v11/economics/headroom-status.json` records each required economic impact as null rather than estimating it.

The only integrated v11 attempt failed before the reclaim trigger. Thus the following hardware-interest predicates have no supporting measurement:

- residual state-chain share of a successful reclamation, useful-capacity, SLO-restoration, or restore critical path;
- at least 1.20× realistic improvement to an economically important end-to-end metric;
- integrated burst rate, latency target, or concurrency;
- a stable placement advantage outside normal GPU execution;
- effect on trajectories per GPU-hour, rollout interruption, or recomputation waste.

The observed residual also contains 4,480 small GPU raw-bit validation records and clear CUDA fusion/batching opportunities. Giving software the next opportunity is more defensible than investigating custom hardware.

Adversarial answer: **No, I would not independently spend engineering time investigating custom hardware from this evidence.**

Hardware interest does not survive, so Experiment 005 was not generated. No FPGA implementation was started.
