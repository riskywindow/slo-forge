# BranchFabric GPU Validation Experiment 004 v11

1. **v10 frozen baseline status:** PASS. Execution `c5fa2f169e8a343ef7a86ebc814b3a23a7de6b39`; analysis `1c51853e10809686d4368037153927f25e834117`; local annotated tag resolves to the execution commit; frozen adapter hash is unchanged.
2. **Main sources of 58× amplification:** capture/publish 21×, restore through native write 20×, destination recapture/host compare 17×. Equivalently: GPU intermediates 29×, host intermediates 23×, native endpoints 3×, links 3×.
3. **v11 software optimizations implemented:** isolated bounded native gather/repack, 28 MiB deterministic chunks, two-slot pinned asynchronous pipeline, owned consumption snapshots, direct native scatter, no full canonical GPU restore tensor, full-page zero-fill elimination, and no full destination D2H recapture.
4. **Validation/integrity changes:** complete/per-page SHA-256, manifest commitment, consumption-time authentication, raw-bit BF16 destination equality, tail raw-zero proof, destination/mapping/device/allocation commitments, failure scrub, typed teardown, and admission-gate witness. GPU launch remains blocked on single-use allocator epoch, post-free ownership proof, real engine-step lock integration, and canonical per-chunk measured records.
5. **v10 vs v11 full physical amplification:** 58× measured vs unavailable. The v11 algorithm projects 21×; that projection is not a measurement.
6. **v10 vs v11 external movement amplification:** 3× measured vs unavailable. The v11 algorithm projects 2×.
7. **v10 vs v11 avoidable amplification:** 31× measured vs unavailable.
8. **v10 vs v11 full-state pass count:** 30 measured conceptual passes vs unavailable; v11 has nine projected aggregate algorithm stages.
9. **v10 vs v11 peak temporary memory:** v10 peak host temporary 2,965,372,928 bytes; v11 measured value unavailable. The two-slot host staging bound is 58,720,256 bytes, separate from the required resident checkpoint.
10. **v10 vs v11 HBM reclamation time:** 5.222174431 s vs unavailable.
11. **v10 vs v11 first useful GPU1 capacity:** 5.413681427 s vs unavailable.
12. **v10 vs v11 serving SLO restoration:** 11.373208336 s vs unavailable.
13. **v10 vs v11 rollout restore latency:** 7.727983415 s vs unavailable.
14. **GPU0 serving interference:** v10 recorded 71/71 completions and 4,507 tokens during restore, about 9.05 completions/s and 574.27 tokens/s. v11 unavailable.
15. **kill/recompute result:** not run; current authorization explicitly excludes it.
16. **naive preserve result:** frozen scientifically valid v10, including 8/8 exact first-token matches.
17. **optimized preserve result:** local implementation and tests exist; GPU result unavailable.
18. **preservation break-even:** analytical form complete. Independent frozen-workload recompute is 133,120 tokens. Numeric coefficients are unavailable.
19. **residual dominant state chain:** unavailable without a measured v11 timing DAG.
20. **residual Amdahl headroom:** unavailable; 2×/5×/10×/free projections remain null rather than reusing v10 timing.
21. **final classification:** NOT REACHED — BUDGET_BLOCKER. No allowed classification is selected without the required measured arms.
22. **BranchFabric hardware gate rationale:** hardware interest is not established. The necessary residual chain, ≥15% critical-path share, and ≥1.20× economic improvement are not measured.
23. **Experiment 005 status:** not generated because the `BRANCHFABRIC_HARDWARE_INTEREST` gate was not reached.
24. **GPU-hours and cost:** v11 used 0 A100-seconds, 0 GPU-hours, and $0.00. Remaining ledger capacity is 943.176 A100-seconds; the primary v11 envelope with margin is 3,474.15, a 2,530.974-second shortfall. `SLOFORGE_GPU_BUDGET_USD` is unset.
25. **Key artifact paths:** `artifacts/branchfabric/gpu-validation/experiment-004/v11-analysis/naive-pass-graph.json`, `artifacts/branchfabric/gpu-validation/experiment-004/v11/manifest.json`, `artifacts/branchfabric/gpu-validation/experiment-004/v11/v11-gpu-budget.json`, `docs/branchfabric/V11_SOFTWARE_STATE_PIPELINE.md`, `docs/branchfabric/PRESERVATION_BREAK_EVEN.md`, and `docs/branchfabric/BRANCHFABRIC_HARDWARE_GATE.md`.
26. **Modal cleanup status:** PASS. Zero apps, containers, endpoints, reservations, child runtime processes, and profiler processes.
