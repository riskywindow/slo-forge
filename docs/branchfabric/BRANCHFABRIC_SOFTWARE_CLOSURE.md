# BranchFabric Software Closure

Experiment 004 closes the custom-hardware inquiry on current evidence.

The optimized software pipeline is real and measured: 58× became 21.000762818× with exact continuation correctness, 63.7918% fewer physical bytes, 10 rather than 30 full-state passes, and 20.657× lower total temporary host memory.

The integrated experiment did not produce a successful preservation transaction. It failed at the pre-reclaim backlog guard, before any v11 state operation. That prevents a measured claim that the state pipeline is below 10% of successful serving recovery, and it prevents a realistic integrated hardware speedup estimate. It also means kill/recompute economics remain unknown.

Accordingly, `SOFTWARE_WINS` is used only as the least-overclaiming required exact-one hardware-closure disposition; its strict performance predicate was not measured. Custom hardware is not justified, Experiment 005 is not authorized, and the result must not be described as an integrated end-to-end v11 win. A future software experiment would first need to repair and independently validate the bounded reclaim-trigger methodology; this task authorizes no retry.
