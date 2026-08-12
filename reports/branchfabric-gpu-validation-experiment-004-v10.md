# BranchFabric GPU Validation Experiment 004 v10

Status: **VALID BLOCKER**. V10 was not launched and is not scientifically valid.

The complete offline v9 audit found that the reported 54× was not the canonical physical-touch metric. The raw ledger recorded 51× memory-endpoint traffic plus 3× PCIe-link traffic, while omitting a real 4× host verification inside restore import. Corrected movement is 55× endpoint touch (58,133,053,440 bytes), 3× external movement (3,170,893,824 bytes), and 58× when all requested read/write/move surfaces are summed (61,303,947,264 bytes). The conservative avoidable lower bound is 31×.

The prior 2× D2H is real: 1,056,964,608 bytes for checkpoint capture plus disjoint 954,204,160-byte and 102,760,448-byte destination-validation recaptures. Duplicate accounting is zero; K and V are already included in the logical denominator.

Capacity calibration produced no observations. Attempt v1 failed after both engines became ready because a nested `CapacityProbePlan` was not JSON serializable. The one permitted corrective attempt v2 timed out during the 115-second engine-readiness bound while both workers were still compiling the vLLM graph. Both attempts have `probe_count = 0`, clean compute postflight, immutable hash-verified raw artifacts, and a full conservative charge of 424 A100-seconds each.

Consequently λ₁, λ₂, and λ_spike are unknown. Agent 9 approval cannot exist, v10's launch gate remains closed, and no v10 control, overload, recovery, restore, resume, or movement metric is reported. No plots were generated from invalid data. The baseline campaign and optimization-candidate documents gated on a valid v10 were not generated.

The Experiment 004 ledger is 6,431.795790918 conservative A100-seconds (1.786609942 A100-hours), with 768.204209082 seconds remaining and zero reservations. Another 424-second calibration reservation cannot coexist with the protected 680-second v10 reservation under the 7,200-second hard ceiling.

Recommended next experiment: only with additional authorization, run a new immutable capacity-calibration attempt with an engine-readiness bound that covers the observed vLLM graph-compilation path. Do not run v10 or kill/recompute versus naive versus optimized preservation until measured one-/two-GPU capacity and an independently approved spike exist.

Final Modal state is clean: both Experiment 004 calibration apps are stopped with zero tasks, zero containers, zero endpoints, and zero reservations. Only `sloforge-model-cache` and `sloforge-branchfabric-results` remain.
