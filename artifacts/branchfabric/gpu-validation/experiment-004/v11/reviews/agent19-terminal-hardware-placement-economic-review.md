# Agent 19 terminal hardware-placement and economic review

Status: **PASS — close the campaign fail-closed, with no final classification and no Experiment 005.**

The measured A100 micro result is real and meaningful: v11 reduced full physical amplification from frozen v10's 58× to 21.000762818×, a 63.791788% reduction, with exact 8/8 continuation. That is sufficient for the completed micro gate, but not for an end-to-end software/hardware conclusion.

The only authorized integrated attempt failed before the reclaim trigger. Optimized export, release, restore, integrated residual timing, and GPU0 restore interference were never measured. Kill/recompute was consequently forbidden and not run. There are no measured recompute tokens or GPU seconds, no fitted preservation break-even, no non-overlapping residual Amdahl result, and no hardware-economic headroom.

None of the four mandated final classifications is evidence-eligible:

- `SOFTWARE_WINS` is not established because neither a residual pipeline share below 10% nor an ideal-free full-system speedup below 1.15× was measured.
- `GPU_SOFTWARE_WINS` is not established because realistic GPU-native headroom and external-accelerator incremental benefit were not measured on an integrated path.
- `BRANCHFABRIC_HARDWARE_INTEREST` is prohibited without measured integrated timings and fails the ≥15% critical-share, ≥1.20× economic-benefit, burst-rate, latency-target, concurrency, and outside-GPU-placement predicates.
- `PRESERVATION_NOT_ECONOMIC` is not established because kill/recompute was not run and therefore cannot be shown to dominate.

Selecting any one would fabricate a missing predicate. The correct status is `TERMINAL_NOT_REACHED_INTEGRATED_INVALID_NO_RETRY` with `final_classification: null`. This means the exact-one-classification completion contract was not achieved; it does not convert missing evidence into `SOFTWARE_WINS`.

My answer to the adversarial hardware question is **no on current evidence**: I would not independently spend custom-hardware engineering time on this bottleneck. That is a refusal to advance hardware, not proof that software has met the `SOFTWARE_WINS` thresholds.

Experiment 005 must remain absent, and no GPU/host/PCIe/NIC/DPU/CXL/FPGA placement ranking is defensible. `BRANCHFABRIC_SOFTWARE_CLOSURE.md` should not be generated because software closure was not measured; `PRESERVATION_POLICY.md` should not be generated because preservation non-economics was not measured. The terminal closure belongs in the hardware-gate document and final report as an evidence-closure outcome.

The current hardware-gate document and final report still say budget blocker and predate the real-A100 micro result. They must be updated to record the micro PASS, the integrated pre-reclaim invalid result, no retry, null downstream metrics, no classification, and no Experiment 005. No FPGA work is authorized.

The reviewer invoked no Modal, GPU, or network resource and changed no source file.
