# Experiment 003 adversarial hardware-interest review

Review date: 2026-08-11

Scope: final report JSON/Markdown, non-overlapping decomposition, metadata-operation scaling, Amdahl projections, CPU diagnostic, decode-interference analysis, HBM analysis, and Experiment 004 plan. This review made no Modal calls and consumed no GPU resources.

## Regenerated-artifact re-review disposition

Re-reviewed after regeneration at 2026-08-11 18:30 local artifact time.

**PASS — no remaining high-severity or reasonable medium-severity finding in the corrected areas.**

- The prior decode-interference finding is resolved. `causal_concurrent_decode_interference` is explicitly unavailable; the measured sequential quantity is now `phase_throughput_deficit_fraction`, and branch-create zero remains labeled only as a structural synchronous stall.
- The prior aggregate-operation finding is resolved. `counted_metadata_work_units` excludes prompt-token writes, scheduler events, output commits, and classification/outcome counters. The excluded values remain separately reported, `all_counter_events_non_unique` retains the complete audit total, and the classification explicitly says the work units are not unique hardware transactions. At optimized fanout 8 these are 32,840 counted metadata work units, 25 scheduler events, 131,080 prompt-token writes, eight output commits, and 180,345 all-counter events.
- The prior fanout-8 Amdahl finding is resolved. The analyzer computes each trial's Amdahl projection independently and takes the median of the three trial projections. The summary carries all three raw artifact references and reports `sample_count: 3`. The median ideal-free speedup is 1.046193x.
- The CPU diagnostic caveat is resolved for reporting purposes. The artifact explicitly states that the full control combines per-block spans and cProfile, that cumulative times are strongly instrumentation-distorted, and that it identifies code paths rather than primary latency magnitudes.
- The cross-campaign before/after values remain explicitly descriptive because the baseline was A100 SXM4 and optimized was A100 PCIe. No causal confidence claim was added.
- The hardware conclusion remains conservative. Fanout 8 closes at median metadata fraction 4.415% despite ideal-free 1.046x; fanouts 16 and 32 remain software-only at 11.059%/1.124x and 11.236%/1.127x. No measured fanout satisfies the hardware-interest conjunction of metadata greater than 15% and ideal-free speedup at least 1.15x.

The findings below are retained as the historical first-pass audit trail; their current dispositions are recorded above.

## Verdict

`METADATA_HARDWARE_INTEREST` is not supported. The conservative `METADATA_SOFTWARE_ONLY` conclusion is consistent with the stated gate: optimized metadata is 4.42% at fanout 8 and approximately 11.1% at fanouts 16 and 32; ideal-free POST_ROOT_READY speedups are 1.045x, 1.124x, and 1.127x. No fanout reaches both the greater-than-15% metadata threshold and the at-least-1.15x ideal-free threshold. Root-inclusive ideal-free speedups are only 1.005x, 1.012x, and 1.020x. Scheduler contribution remains separately classified and is not evidence for metadata hardware.

The conclusion should remain software-only after the reporting corrections below. Experiment 004 is the required next experiment under the decision contract.

## Findings

### RESOLVED (formerly HIGH) — The reported decode-interference percentages are not causal concurrent decode interference

Evidence:

- `analysis/decode-interference.json` defines early-divergence throughput as `early_outputs / POST_ROOT_READY`, where `early_outputs = fanout*256 - decode_active_tokens`.
- At optimized fanout 8 this is eight first outputs divided by approximately 293–303 ms and compared with the same cohort's later steady throughput, producing approximately 92.6% "interference."
- The branch-create value of 100% is explicitly described in the artifact as a synchronous structural host stall with no overlapping GPU submission, execution, or output.
- There is no independent control decode stream running through branch creation whose phase-specific throughput is compared with simultaneous branch creation.

Impact:

This measures phase productivity/first-token readiness relative to established decode, not causal slowdown of ongoing decode. Calling the 92.6% value `decode_interference_fraction` can be read as GPU interference even though the method folds orchestration, request construction, metadata, scheduler, and first-execution latency into the denominator. The report's caveat helps, but `status: available` and the interference label still overstate the evidence.

Required disposition:

- Report causal concurrent decode interference as unavailable for this matrix.
- Retain these values only under a name such as `early_divergence_phase_productivity_gap`, with the exact formula and synchronous-cohort limitation.
- Keep branch-create zero labeled only as a structural synchronous stall. Do not report it as concurrent GPU slowdown.

### RESOLVED (formerly MEDIUM) — `direct_metadata_operation_count` is an all-counter event sum, not a unique metadata-operation count

Evidence:

- The optimized fanout-8 reported total is 180,345.
- Its constituents include 131,080 `prompt_token_writes`, 25 scheduler events (`scheduler_candidate_evaluations`, queue inserts/removals, and scan), and eight `output_token_commits`.
- Excluding only those request-token, scheduler, and output counters leaves 49,232 events.
- That remainder still intentionally counts several observations of the same prefix block separately: candidate, bound, hash lookup, hash hit, table write, and refcount increment. It is therefore not a unique transaction count or a hardware request rate.
- At constant 1,024-prefix-block length, the exact O(N) total follows directly from fixed per-branch counter increments. The artifact correctly notes that O(N) and O(N x prefix_blocks) are not separately identifiable.

Impact:

The per-operation counters are useful and direct, but the total is unsuitable as a hardware workload size and violates the intended semantic separation when scheduler operations are included under a metadata label. It must not be used to derive an accelerator operation rate or queue requirement.

Required disposition:

- Rename 180,345/360,689/721,377 to `all_direct_counter_events` or equivalent.
- Publish separate totals for request construction, prefix/block metadata, scheduler, private allocation, and output commit.
- Preserve the per-counter O(N) result, but state that total-event linearity is structural at fixed prefix length, not an independently discovered complexity law.

### RESOLVED (formerly MEDIUM) — The full-trace CPU diagnostic is dominated by diagnostic instrumentation and cannot establish the production CPU hot path

Evidence:

- The only cProfile run is the full-trace fanout-8 diagnostic.
- The tracing-control artifact reports full/disabled POST_ROOT_READY = 2.254x, a 125.4% overhead. Full tracing automatically enables cProfile, so this is combined full-trace-plus-cProfile overhead, not an isolated full-trace overhead measurement.
- The profile contains 16,594 adapter `span` calls, 16,621 `process_time_ns` calls, and thousands of context-manager entries/exits. Adapter instrumentation occupies several of the highest cumulative rows.
- vLLM `get_computed_blocks` has 0.572 s cumulative time but only 0.00166 s self time; the cumulative value includes descendants and instrumentation. Likewise, scheduler `schedule` has 0.360 s cumulative but only 0.000208 s self time.
- The diagnostic does not isolate GIL wait, allocator/mutex wait, native-extension time, or cache-miss behavior.

Impact:

The diagnostic confirms traversal through per-block Python lookup wrappers, but it cannot quantify the minimally traced production hot path or assign cumulative time to vLLM itself. It is not evidence for a regular hardware engine and should not be used for lock/GIL claims.

Required disposition:

- Label the flamegraph/equivalent as a call-path diagnostic under combined full-trace+cProfile distortion.
- Base latency attribution on minimal exclusive spans, and use self time—not cumulative time—when describing profiler-owned work.
- State that GIL, lock, allocator, and native-extension attribution remains unavailable.

### RESOLVED (formerly MEDIUM) — Fanout-8 Amdahl aggregation and provenance use inconsistent summaries

Evidence:

- The optimized fanout-8 decomposition reports median POST_ROOT_READY 302.553485 ms and median metadata 12.953524 ms.
- Their ratio is 4.2814%, and the Amdahl artifact uses it to compute the 1.044729x ideal-free speedup.
- The decomposition/classification instead reports 4.4153%, the median of the three per-trial metadata fractions.
- The Amdahl fanout-8 provenance points to the seed-41 trial, although the synthesized parent median is from seed 113 and the metadata median is from seed 73.

Impact:

The gate outcome does not change: both summaries remain below 5%, and ideal-free remains below 1.05x. However, a synthesized median must not claim provenance to one raw trial, and a classification record should not combine two aggregation conventions without naming them.

Required disposition:

- Either compute Amdahl per trial and summarize the resulting speedups, or consistently use ratio-of-medians.
- Attach provenance to all contributing trials and identify the aggregation rule.
- Keep `metadata_fraction_median_of_ratios` distinct from `ratio_of_median_stage_times`.

### ACCEPTED WITH EXPLICIT CAVEAT (formerly MEDIUM) — Before/after readiness improvement is descriptive, not a causal same-device optimization effect

Evidence:

- The baseline campaign used `GPU-bb860...`, NVIDIA A100-SXM4-80GB.
- The optimized campaign used `GPU-03bc...`, NVIDIA A100 80GB PCIe.
- `before_after` correctly records `same_gpu_uuid: false` and `same_exact_gpu_model: false`, but also emits a median readiness difference of -110.787 ms and a median metadata difference of -1.471 ms.

Impact:

Within each campaign, independent/shared pairs are valid same-GPU comparisons. Across campaigns, exact optimization deltas are confounded by physical GPU variant and UUID, even though the user-authorized rerun requirement specified the same GPU family. The direction is compatible with the software change, but the precise delta is not causal.

Required disposition:

- Preserve the existing descriptive caveat in every final mention of before/after numbers.
- Do not attach a causal confidence interval or claim that 110.8 ms is solely due to the optimization.

### LOW — Fanout-16/32 software-only observations have no repeat uncertainty

Evidence:

- Optimized fanouts 16 and 32 each have one scaling sample; fanout 8 has three.
- The 11.06% and 11.24% metadata fractions and 1.124x/1.127x ideal-free projections are therefore point observations, not population estimates.
- Minimal tracing overhead is measured once at fanout 8 (1.84%), and no fanout-16/32 disabled controls exist.

Impact:

`METADATA_SOFTWARE_ONLY` is a conservative aggregate because the high-fanout points fall well below the hardware gate and tracing overhead can only inflate measured software metadata leverage. It should nevertheless be presented as bounded-run evidence, not a stable percentile or distribution.

Required disposition:

- State sample counts adjacent to high-fanout fractions.
- Do not infer p95 or p99 from them.
- Do not extrapolate beyond fanout 32 or varying prefix length.

## Explicit checks

- FPGA/hardware overclaim: none in the current conclusion. No FPGA justification is supported.
- Scheduler-as-metadata error: exclusive scheduler wall time is correctly separate; only the aggregate operation-count label incorrectly mixes scheduler counters into a "metadata" total.
- Unsupported p95/p99: none found. Paired, scaling, and trace-control artifacts set p95/p99 to null and explain the insufficient sample sizes.
- HBM/COW conclusion: supported. Shared prefix remains at the theoretical floor with zero floor gap; the capacity hardware target remains closed.
- Experiment 004 recommendation: supported and required by the classification contract. The plan does not execute hardware work and correctly scopes two-GPU capacity reclamation and cross-layout movement.

## Final recommendation

Keep `METADATA_SOFTWARE_ONLY`, close any Branch Translation Unit/FPGA inference, and proceed to the Experiment 004 plan. Correct the decode-interference status, aggregate-operation label, CPU-profile caveat, and Amdahl provenance before final handoff. None of these corrections can turn the current data into `METADATA_HARDWARE_INTEREST`.
