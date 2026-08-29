# BranchFabric Metadata Characterization

This document is generated exclusively from materialized Experiment 003 artifacts. Missing measurements are reported as unavailable and no Experiment 002 cross-GPU throughput result is reused as causal evidence.

## Result

- Completion status: `complete`
- Metadata conclusion: `METADATA_SOFTWARE_ONLY`
- HBM COW capacity target: `CLOSED`
- Same-GPU paired baseline: `available`
- Same-GPU paired optimized: `available`

## Measurement boundaries

POST_ROOT_READY is decomposed only from non-overlapping exclusive stage time. It ends after every branch returned first output. The legacy FIRST_BRANCH_READY and ALL_BRANCHES_READY fields are output-step-return boundaries, not direct runnable transitions; step-entry execution-start proxies are reported separately. Scheduler admission, waiting, and selection are reported as scheduler work, never metadata. Counted metadata work units use a non-overlapping software taxonomy and are not hardware transactions.

GPU_EXECUTION is GPU-inclusive worker CPU wall, not pure device time. The optimized fanout-8 first GPUModelRunner._model_forward CUDA-event median is available in the decomposition JSON. Baseline predates the sample_tokens wrappers used by the optimized campaign, so baseline/optimized GPU and residual stages are not identically scoped and are not used as causal before/after evidence.

## Known unavailable evidence

Causal concurrent decode interference is unavailable because no background decode stream overlapped branch creation. The reported early-divergence ratio is a sequential phase-productivity deficit; a branch-create throughput of zero is a structural synchronous stall, not concurrent GPU slowdown. Numeric vLLM-native comparisons use only POST_ROOT-frozen scheduler snapshots; native metrics do not provide exact exclusive-span timing equivalents.
