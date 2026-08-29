# Experiment 003 final GPU methodology review

Final re-review scope: same-GPU validity, lifecycle cleanup, provider inventory, tracing controls, GPU/interference attribution, statistics, GPU-hour accounting, and scientific caveats. No Modal command or GPU workload was invoked for either review pass.

## Disposition

No high or reasonable medium GPU-methodology finding remains. The provider cleanup evidence, cost convention, interference semantics, singleton uncertainty, and cross-variant causal boundary are fixed and machine-validated. The reviewed artifacts support `METADATA_SOFTWARE_ONLY` without another GPU run.

## Final caveat verification — resolved

`$.metadata_classification.fanouts` now records sample counts 3, 1, and 1 for fanouts 8, 16, and 32. Its `uncertainty_caveat` explicitly says the high-fanout fractions are single diagnostic trials and do not estimate trial-to-trial variance. The Markdown places that limitation directly beside the classification and identifies the aggregate as conservative.

The Markdown also now states that the median before/after latency deltas are descriptive only because the baseline used an A100-SXM4 and the optimized campaign used an A100 PCIe GPU on a different UUID. It separately preserves the causal status of the within-campaign independent/shared pairs. This matches `$.before_after.comparison` and the raw `same_exact_gpu_model = false` / `same_gpu_uuid = false` fields.

## Resolved and verified

### Provider cleanup — resolved

`artifacts/branchfabric/gpu-validation/experiment-003/logs/modal-provider-cleanup.json` is persisted with SHA-256 `a5d219383970b8dd30529efaf1c0a9e21763f9bc6a49ccc86f19dd12f5d62135`, matching report provenance. It records:

- both Experiment 003 apps stopped with zero tasks;
- an empty container inventory;
- exactly the authorized `sloforge-branchfabric-results` and `sloforge-model-cache` volumes.

`_modal_cleanup_summary` validates schema and inventory contents, and completion now requires `provider_zero_active_tasks`, `deployed_endpoints_absent`, and `only_authorized_persistent_volumes`. All three are true, so `completion_status = complete` is truthful with respect to the persisted provider evidence.

### Cost convention — resolved

The report no longer presents the heterogeneous interval sum as one comparable total. It identifies:

- `known_gpu_only_cost_usd = 2.7609559965` across both campaigns;
- `known_support_cost_usd = 0.3537984476` for the optimized campaign only;
- `recorded_cost_usd_mixed_conventions = 3.1147544441` with an explicit note that baseline recorded GPU cost only.

Duration accounting remains consistent: 1.105089656 additional A100-hours exceeded the 0.75-hour target but stayed below the 1.25-hour hard limit, and reservations are empty.

### Decode/GPU interference semantics — resolved

The artifact schema is now `sloforge.branchfabric.experiment-003-decode-phase-productivity/v2`. It explicitly marks causal concurrent decode interference unavailable because no background decode stream overlapped branch creation. Early output divided by POST_ROOT_READY is labeled a sequential phase-productivity deficit; branch-create zero is labeled a synchronous structural stall; benchmark-wide NVML sampling is disclosed. No causal GPU slowdown or phase-isolated utilization claim remains, and no Experiment 002 throughput is reused.

### Same-GPU and lifecycle validity — verified

- Baseline primary pairs used Function call `fc-01KZSBX572AK09MYBJTSG1TWVR` and only SXM4 UUID `GPU-bb86053c-1232-1eca-5aeb-574862b978f5`.
- Optimized primary pairs used Function call `fc-01KZSFAHAM2Z1DZ9V40P96HW4F` and only PCIe UUID `GPU-03bc8fb9-13fd-508f-2fc8-be64e3f4f23c`.
- Seeds 41, 73, and 113 realize independent/shared, shared/independent, and deterministic-randomized independent/shared order in both campaigns.
- All 22 completed children have three matching pre-child samples and three qualifying post-child samples at 4 MiB used HBM, no compute process, and the expected UUID.
- Every authoritative lifecycle gate records clean child exit, no descendant, zero final runtime-assigned KV bytes, no forced GPU-process kill, HBM recovery in 81.99–114.56 ms, and stable confirmation in 2.078–2.098 s.
- The baseline adaptive tail timeout omitted only its minimal/full trace controls after all primary pairs and fanouts 1/16/32 completed. The report continues to disclose `campaign_status = failed` and `adaptive_tail_timeout = true`; optimized controls supply the required disabled/minimal/full triplet.

### Trace/statistical boundaries — verified

The single same-GPU seed-191 control triplet measures minimal overhead at 1.84% and combined full tracing/cProfile overhead at 125.44%; the report identifies the full diagnostic composition and suppresses unsupported tail percentiles. Paired fanout-8 analysis retains all three raw differences, uses deterministic whole-pair bootstrap resampling, and marks effect sizes/tails as descriptive or insufficient where appropriate.

## Final reviewer judgment

The same-GPU causal independent/shared comparisons, cleanup gates, hard GPU-hour limit, optimized-campaign metadata classification, and conclusion-facing limitations are methodologically usable. No high or reasonable medium GPU-methodology issue remains.
