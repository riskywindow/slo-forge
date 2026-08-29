# Experiment 003 runtime-semantics review

Reviewer scope: final, independent audit of the pinned vLLM 0.23.0 request/KV/scheduler path, the Experiment 003 timing and counter semantics, the shared/private physical-state evidence, and the optimized immutable-root-hash path. This review made no Modal call and launched no GPU work.

## Re-review disposition after report regeneration

No high-severity runtime-semantics finding remains. Numeric native/custom value
comparisons, a non-overlapping metadata work-unit taxonomy, and explicit legacy
readiness-boundary labels were added. The physical-state, HBM-floor,
optimization-correctness, timing-exclusivity, and scheduler-versus-metadata
findings remain sound.

The native-metric finding is resolved. Numeric comparisons now use only the
embedded POST_ROOT_READY-frozen native observations; the shared-root values
confirm request count, query-token count, and 16-token-block hit count. The
serialized and recorder post-root counts agree (17/17 for independent and 1/1
for shared), while tracing-disabled trials are explicitly unavailable. No
major native/custom disagreement remains.

The consolidated Markdown report now correctly labels the legacy readiness
fields as first-output boundaries, execution-start values as proxies, and
`GPU_EXECUTION` as GPU-inclusive worker CPU wall rather than pure device time.
It also identifies the separate first-model-forward CUDA event and the
baseline/optimized `sample_tokens` scope mismatch. The report JSON contains the
13,254,656 ns optimized fanout-8 median CUDA-event value.

The generated `METADATA_CHARACTERIZATION.md` and its generator now carry the
same output-boundary/runnable-proxy, GPU-inclusive worker-wall, direct
CUDA-event, and baseline `sample_tokens` scope qualifiers. The last
documentation finding is resolved. No high-severity or reasonable
medium-severity runtime-semantics finding remains.

## Confirmed semantics and invariants

### Exact vLLM path and configuration

- `VllmLiveInternalView` requires vLLM 0.23.0, V1 `InprocClient`, TP/PP/DP 1, synchronous FCFS scheduling, no speculative decoding, no LoRA, no KV/EC connector, one ordinary full-attention cache group, and CUDA-backed KV tensors (`python/sloforge/continuum/adapters/vllm_live.py:210-339`). The invocation records block size 16.
- The documented request path is consistent with the hooks: adapter `_submit` -> `LLMEngine.add_request` -> `EngineCore.preprocess_add_request` -> scheduler `add_request`; `LLMEngine.step` -> `Scheduler.schedule` -> prefix lookup/slot allocation -> executor/runner execution -> sampling -> scheduler/frontend output commit.
- Internal vLLM objects remain confined to the version-specific adapter. Persisted observations contain scalar fields, IDs, and JSON-compatible native statistics; generic Continuum IR is not polluted.

### Non-overlapping timing

- `SpanHierarchy` rejects sibling wall overlap and invalid containment and computes exclusive time by subtracting nested children (`python/sloforge/continuum/adapters/vllm_0230_metadata.py:392-474`). `TimingDecomposition` requires exact equality across wall/process/thread dimensions (`:322-335`).
- Every retained decomposition reports zero wall/process/thread error. Scheduler admission/wait/select are independent additive stages and are excluded from `METADATA_STAGES` (`benchmarks/branchfabric/analyze_experiment_003.py:68-77`). No scheduler queueing is mislabeled metadata.
- `SCHEDULER_WAIT` is correctly qualified as direct timestamps with inferred attribution: the final submit boundary to first schedule entry, not summed per-request queue residence.

### Counters and lifecycle

- At optimized fanout 8 readiness, direct/scoped counts match the source invariants: 8 allocations, 8,192 complete-prefix probes/hits, 8,192 bound root blocks, 8,200 refcount increments, 8,200 worker-table writes, eight queue inserts/removals, and one scheduler scan.
- Full and minimal diagnostic trials both report 8,192 prefix probes/hits and zero misses, validating the minimal-mode derived probe count against the full direct wrapper.
- The lifecycle snapshot is separated from the frozen readiness snapshot. For optimized fanout 8 it records 8,320 total refcount increments and decrements, 1,152 blocks reaching zero (1,024 root plus 128 private), 128 total post-root private allocations, and final assigned KV bytes of zero. This is internally balanced.

### Shared/private ownership and HBM floor

- `_validate_shared_branch_tables` requires each ordered request-table prefix and initial cache-hit ID list to equal the published root, requires 16,384 cached tokens, rejects private-page aliasing, and validates the native root refcount (`python/sloforge/continuum/adapters/vllm_live.py:1775-1824`). This is independent of the hash-template counter.
- The retained optimized fanout-8 layouts show 1,024 root blocks at refcount 8 and private blocks at refcount 1. At the 256-token boundary there are 128 private pages.
- With a 917,504-byte physical page, the 1,024-page root is 939,524,096 bytes. Fanout-8 final shared assigned KV is 939,524,096 + 128 * 917,504 = 1,056,964,608 bytes, exactly the reported theoretical floor. The same zero-floor-gap invariant is present for fanouts 1, 16, and 32, baseline and optimized. `HARDWARE_COW_CAPACITY_TARGET = CLOSED` is semantically justified.

### Optimization correctness

- Root hashes are captured from the original vLLM hasher and published only after the aligned completed root has been validated (`python/sloforge/continuum/adapters/vllm_metadata_0230.py:443-451`; `vllm_live.py:1314-1358`).
- Template use is limited to a root-linked, root-plus-one, pure-token request with no salt, LoRA, multimodal feature, or prompt embeddings, and returns a fresh list container (`vllm_metadata_0230.py:493-531`). All subsequent native lookup/refcount/allocation/block-table work still executes.
- Both before and after artifacts preserve exact shared-root IDs, private single-owner suffix pages, zero prefix duplication, successful native release, and HBM cleanup. Cross-campaign timing deltas are correctly labeled descriptive because the baseline and optimized campaigns used different GPU UUIDs and different A100 variants.

## Final disposition

Accepted. The exact request/KV path, physical-sharing/HBM result, narrow
optimization correctness, non-overlap invariant, native/custom comparisons,
counter taxonomy, readiness labels, GPU-wall semantics, and separation of
scheduler work from metadata have no remaining high or reasonable-medium
finding. No rerun is needed.
