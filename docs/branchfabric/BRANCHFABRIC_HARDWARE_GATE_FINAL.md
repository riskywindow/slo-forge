# BranchFabric Hardware Gate Final

Final classification: **`HARDWARE_GATE_NOT_REACHED`**.

This is the only classification supported by the available evidence. The single
integrated v11 attempt failed the exact source-allocation identity gate before
source release and HBM reclamation. It therefore produced no valid integrated
movement amplification, residual source/restore chain, critical-path share,
Amdahl headroom, realistic accelerator lower bound, or end-to-end CUDA/Triton
comparison. The gated kill/recompute baseline was not run.

The early trigger and cleanup fixes did pass. Those successes establish that the
normal trigger no longer waits for the emergency ceiling and that the Function
tears down its own process hierarchy, but they do not establish the v11
preservation transaction or its economics.

The missing mandatory evidence prevents all four substantive decisions:

- `SOFTWARE_WINS` cannot be claimed because residual integrated headroom is not measured.
- `GPU_SOFTWARE_WINS` cannot be claimed because no integrated residual chain exists for a CUDA/Triton placement comparison.
- `BRANCHFABRIC_HARDWARE_INTEREST` cannot be claimed from micro-validation alone.
- `PRESERVATION_NOT_ECONOMIC` cannot be claimed without the measured kill/recompute arm.

v11 was not frozen. Experiment 005 was not generated, and no FPGA work began.
