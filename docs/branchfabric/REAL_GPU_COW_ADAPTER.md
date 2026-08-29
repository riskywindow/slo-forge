# Real GPU shared-root adapter

## Scope

The BranchFabric real-runtime adapter is the narrow process-local boundary in
`sloforge.continuum.adapters.real_runtime`. Its selected implementation targets
only vLLM 0.23.0 V1 synchronous offline inference. It observes and sequences the
runtime's existing GPU paged-KV state; it does not make that state portable and
does not change vLLM's allocator or attention kernels.

This document specifies what the adapter may claim. Fixture tests validate its
state machine and accounting, but cannot establish GPU residence or physical
sharing. Experiment 002 v5 subsequently established the bounded page-level
semantics on a real A100; it did not pass the separate driver-visible HBM
cleanup gate and therefore did not authorize the 16K matrix.

## Experiment 002 observed semantics

The v5 Modal smoke used vLLM 0.23.0, Qwen2.5-7B-Instruct in bfloat16, and one
`NVIDIA A100 80GB PCIe`. One 2,048-token prefix occupied exactly 128 physical
blocks of 917,504 bytes each across all 28 attention layers. Both live branches
then referenced those identical block IDs at native refcount two. Metadata fork
allocated no additional KV page; divergence allocated disjoint blocks 129 and
130, one per branch.

Destroying branch A made block 129 allocator-available while every root block
and branch B's block 130 remained owned. Final branch/root destruction made all
130 observed experiment block IDs allocator-available, cleared their hashes,
restored the full 55,959-block usable pool, and reduced assigned experiment KV
bytes to zero. No `STATE_COW` event occurred: the demonstrated mechanism is an
aligned immutable shared prefix plus private append, not write-fault COW.

The complete smoke remains failed. Postflight observed 14,213 MiB held by the
Function process versus a 4 MiB baseline after adapter cleanup, attributable to
retained model/CUDA-graph runtime state rather than assigned experiment KV. The
runner therefore stopped before 16K. An exact child-process teardown (or an
equally strong in-process release proof) is required before rerunning the smoke.

## State boundary

The adapter maps runtime-native state into three disjoint classes.

| Class | Recorded | Never serialized |
| --- | --- | --- |
| Logical | exact model/tokenizer revisions, token IDs and hash, positions, attention-layer identities, half-open KV token ranges, decoding policy and epoch | renderer/tokenizer implementation objects |
| Physical | runtime block ID and index, KV group, layers, CUDA device, dtype, tokens and bytes per block, logical range, owning branches, native refcount, allocation epoch, shared/private class, residency | K/V tensor contents or data pointers |
| Runtime-ephemeral | opaque session, engine, request, scheduler, allocator, worker and profiler handles held inside the adapter | all pointers, CUDA handles, streams, IPC objects, allocator objects, request objects |

The portable Continuum adapter SDK is not used to save or restore this state.
The `RealRuntimeStateAdapter` schema is a bounded observation schema, not an
execution-state capsule.

## Required runtime shape

The constructor must verify before model work:

```text
vllm distribution version == 0.23.0
VLLM_ENABLE_V1_MULTIPROCESSING == 0
engine client class == vllm.v1.engine.core_client.InprocClient
tensor_parallel_size == 1
pipeline_parallel_size == 1
CUDA device count in adapter scope == 1
KV cache group count == 1
KV cache spec == full attention
speculative decoding == disabled
LoRA == disabled
sliding/hybrid/Mamba state == absent
P/D, offload, and remote KV connector == disabled
```

After initialization it verifies that the model runner's `kv_caches` are CUDA
tensors, that its `kv_cache_config` agrees with the scheduler manager's config,
and that `num_blocks * physical_page_size_bytes` agrees with the full physical
pool geometry. Here one scheduler block ID addresses one page in every layer KV
tensor in the validated group; the physical page size is therefore the sum of
those per-layer pages, not one `FullAttentionSpec.page_size_bytes`. The null
block is excluded from all state accounting.

The experiment identity is:

- runtime: `vllm==0.23.0`;
- runtime source: `0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665`;
- model and tokenizer: `Qwen/Qwen2.5-7B-Instruct`;
- immutable revision: `a09a35458c702b33eeacc393d103063234e8bc28`;
- scope: one adapter, engine process, GPU, model, dtype, and policy epoch.

The logical identity remains the official Hugging Face model/tokenizer ID and
immutable revision even when the remote run has no network access during model
load. `VllmLiveStateAdapter.create` therefore requires a separate absolute
`execution_model_path`. It fails before importing vLLM or allocating a model
unless that path resolves to a local directory named by the exact
`RuntimeModelIdentity.model_revision`, contains `config.json`,
`tokenizer_config.json`, and safetensor weights, and is passed as both vLLM's
`model` and `tokenizer` argument. The `revision` and `tokenizer_revision`
arguments must still equal the official logical identity. The execution path is
runtime-ephemeral provenance and never replaces or enters portable Continuum
model identity.

For the Modal experiment, the CPU-only preparation Function downloads the
immutable snapshot beneath the dedicated `sloforge-model-cache` Volume at
`/models/snapshots/<revision>`. The GPU Function mounts that Volume read-only
and passes the resolved snapshot directory explicitly. No fallback to a hub ID,
another revision, a global cache, or a second tokenizer is permitted.

## Runtime access isolation

One version-specific module owns all accesses below:

```text
LLM.llm_engine
  -> LLMEngine.engine_core: InprocClient
    -> InprocClient.engine_core: EngineCore
      -> EngineCore.scheduler
        -> requests / running
        -> kv_cache_manager
          -> coordinator.single_type_managers[0].req_to_blocks
          -> get_computed_blocks / allocate_slots
          -> get_blocks / get_block_ids
          -> free / reset_prefix_cache
          -> block_pool.blocks
      -> EngineCore.model_executor.driver_worker.worker.model_runner
        -> kv_caches / kv_cache_config / device
```

`KVCacheBlock.block_id`, `ref_cnt`, `block_hash`, and `is_null` and
`KVCacheGroupSpec.kv_cache_spec.block_size` and `page_size_bytes`, and
`KVCacheGroupSpec.layer_names` are the exact metadata dependencies. Core Continuum and Helix
code sees only the pointer-free models in `real_runtime.py`.

## Session and branch protocol

### Root

1. `start_session` creates an adapter session with fixed tokens, seed, model,
   policy epoch, and timeout.
2. `prefill_session` adds one vLLM request and steps the synchronous engine
   until the complete, block-aligned prefix has computed KV.
3. `pause_at_safe_decode_boundary` observes only between calls to
   `LLMEngine.step()`; there is no in-flight model execution at this boundary.
4. The priming request finishes or is aborted through vLLM's normal path after
   the complete root pages are published to the prefix cache.
5. `create_shared_root_reference` records the cached root's ordered block IDs,
   ranges, page bytes, hash-presence state, and native refcount, which is
   normally zero when no request owns the cached pages. Branch submission then
   occurs in `run_concurrent_branches` before the next engine scheduling or
   allocation event.

The prefix length must be divisible by the resolved block size. A partial final
block is not shared and invalidates the experiment's “16K shared root” claim.

The adapter deliberately does not keep the priming request live: it would
resume decoding when concurrent branches step. It also does not call
`BlockPool.touch()` or mutate `KVCacheBlock.ref_cnt`; artificial pins would
perturb vLLM's common-prefix optimization and allocator measurements. The root
is therefore evictable during the short interval between priming completion
and branch admission. The run uses an isolated engine with verified headroom,
admits children immediately, and fails if the root's IDs, hashes, or residency
change. The adapter-level `SharedRootReference` is ownership-equivalent
metadata, not a native allocator reference.

### Branches

`fork_same_policy_session` does not copy metadata or KV tensors. It creates a
new vLLM request containing the identical root token history followed by the
branch's first divergent token. vLLM's normal `get_computed_blocks()` and
`allocate_slots()` path must attach the exact root `KVCacheBlock` objects to the
new request and allocate new suffix blocks.

The adapter asserts at the first branch-ready boundary that:

- every full-prefix block ID equals the corresponding root block ID;
- every shared block is backed by the one CUDA KV pool;
- native `ref_cnt` and the independently derived request-table owner count
  agree;
- no suffix block ID belongs to the root or another divergent branch;
- computed-token accounting shows no optimized-path re-prefill.

`run_concurrent_branches` admits all siblings before repeated engine steps.
The adapter records four distinct per-branch milestones on one monotonic clock:

- request admission is the first successful native `allocate_slots` call;
- first-decode-token start is the host-side lower bound after native admission
  for the engine step later proven to have produced that branch's first output
  token;
- branch readiness is the return of that engine step, which proves the full
  prompt/prefix work completed rather than merely allocating an early chunk;
- first-token emission is the subsequent observation of that output by the
  adapter.

The adapter fails the run if any milestone is missing or causally misordered.
For chunked independent prefill, early allocation is therefore reported as
admission and is never mislabeled as full branch readiness. All branches use
the same scheduler and actually decode concurrently.
For path comparisons, the runner uses the same timing origin on both sides:
shared readiness includes aggregate fork-session metadata creation, while the
independent baseline includes aggregate independent-session metadata creation.
Snapshots at suffix 1, 16, 64, and 256 tokens show private allocation growth.

### Destruction

`destroy_branch` aborts only the exact live request ID (a request may already
have completed), retains the former private block IDs, and reads those IDs back
from `BlockPool`. Every former private block must have native `ref_cnt == 0` and
be allocator-available; cached full blocks may still retain `block_hash` and be
eviction candidates. Sibling root mappings must remain unchanged. The operation
is idempotent at the adapter boundary.

`destroy_session` retires the adapter's root-reference metadata only after its
branch references are gone. Its final idle prefix-cache reset must clear the
hashes on every retained experiment block, leave every exact block ID at native
`ref_cnt == 0`, and restore `BlockPool.get_num_free_blocks()` to the complete
non-null pool capacity. `cleanup_runtime` is an idempotent failure-safe path: it
aborts remaining owned requests, attempts an idle prefix-cache reset, restores
the instrumented methods, and shuts down the owned engine. Only the normal
`destroy_session` gate proves exact-ID and whole-pool recovery; the fallback
cleanup path does not turn an earlier failed proof into a valid run.

## Capability behavior

| Capability | Observable result |
| --- | --- |
| `inspect_logical_state` | token history, positions, layer identity and policy provenance |
| `inspect_physical_kv_layout` | unique physical blocks with group/layer/range/owner/refcount/byte data |
| `inspect_gpu_memory_state` | NVML process/device bytes, PyTorch allocated/reserved bytes, and assigned/reserved KV bytes |
| `identify_shared_prefix_blocks` | ordered intersection proven equal to the root range, not a content-only match |
| `create_shared_root_reference` | cached-root identity plus an adapter ownership-equivalent lifecycle reference; no allocator pin |
| `allocate_private_suffix_state` | new vLLM allocations observed after divergence |
| `observe_block_allocation` / `observe_block_release` | session-scoped layout snapshots tied to scheduler steps; disappearance alone is not release evidence |
| `inspect_block_release_evidence` | exact former IDs, allocation epochs, native refcounts, hash presence, allocator availability, and aggregate free/usable pool counts |
| `report_block_refcounts_or_equivalent` | native `KVCacheBlock.ref_cnt` and request-table cross-check |
| `capture_branchpoint` | conservative same-runtime/same-policy/process-local contract |
| trace emission | real runtime allocation, reference, append, divergence, release, and destruction events |

## Sharing proof and terminology

The adapter implements an immutable physical shared root with append-only
private suffixes. Autoregressive attention does not rewrite prior-token KV, so
this gives the experiment the required COW-equivalent behavior without a
write-fault mechanism. It does not implement general CUDA-memory COW, mutable
tensor snapshots, arbitrary session cloning, or portable Helix state.

A run may claim the bounded physical GPU shared-root page semantics only when
these observations are present in local artifacts:

1. the model runner's KV backing tensors are on the selected CUDA device;
2. the prefix was computed exactly once;
3. both branches' request tables contain the exact same ordered root block IDs;
4. prefix physical assigned bytes remain one copy as fanout grows;
5. suffix block IDs and assigned bytes grow after divergence;
6. deleting one branch frees only its private state and preserves a sibling;
7. the root IDs remain resident from prime through all active branches;
8. deleting the last branch, retiring the adapter reference, and resetting the idle prefix cache
   restores allocator availability for every experiment page.

Prefix-cache latency or token-hit counters alone are insufficient. If any item
is missing, the report must say the physical sharing claim was not demonstrated.
Passing the complete semantic smoke additionally requires driver-visible HBM to
return to the permitted baseline after runtime cleanup. Failure of that separate
gate forbids 16K execution even when page sharing and page release were observed.

## Physical and HBM accounting

For the selected one-group configuration, a physical block's byte cost is the
sum of the pages it addresses across all layer KV tensors. The adapter derives
that value from `KVCacheConfig.kv_cache_tensors[*].size / num_blocks`, checks it
against the sum of the group's per-layer spec page sizes, and independently
checks `kv_pool_reserved_bytes` against the actual CUDA tensor sizes.

```text
physical_assigned_bytes = count(unique assigned non-null block IDs) * page_size_bytes
shared_prefix_bytes = count(unique root block IDs) * page_size_bytes
private_suffix_bytes = count(unique live non-root block IDs) * page_size_bytes
kv_unassigned_bytes = kv_pool_reserved_bytes - physical_assigned_bytes
```

NVML process/device used memory and PyTorch allocated/reserved memory are
reported separately. Because vLLM reserves the KV pool at engine startup,
branching may change assigned capacity without changing driver-visible HBM.
Neither measurement may be substituted for the other.

The theoretical shared-path floor is one root plus the minimum page-rounded
private suffix for each live branch. Independent state is the union of all
branch block sets with prefix sharing disabled. Derived metrics must use
physical assigned bytes for sharing efficiency and amplification and disclose
page rounding.

## Independent-prefill baseline

The independent engine has automatic prefix caching disabled. Every other
scientifically relevant setting is identical: vLLM version and source, model
and tokenizer revision, dtype, block size, maximum context, GPU, scheduler
limits, branch prompts, suffix target, sampling configuration, and seeds.

The baseline run is invalid unless:

- cached-token counts are zero for every branch;
- prefix block-ID intersections between independent branches are empty;
- each branch performs its own complete 16K prefill;
- no state from an optimized engine or previous trial is reachable.

Optimized and independent engines are never alive on the GPU concurrently.
Their fixed model/KV-pool footprints are measured separately, and comparisons
use state deltas and assigned pages rather than assuming identical NVML
baselines.

## Tracing

Each runtime snapshot has a monotonically increasing allocation/snapshot epoch.
Diffs emit real `BranchWorkloadTrace` and `StateOperationTrace` events for:

- prefix allocation and block publication;
- branch creation and shared-root references;
- native refcount changes;
- private block allocation and first divergence;
- KV append at selected token checkpoints;
- branch release/private reclaim or free, emitted only after native exact-ID evidence;
- final root release, prefix reset, and runtime cleanup.

Physical events include runtime/version, device, group, layer IDs, block ID,
page bytes, logical half-open token range, branch IDs, shared/private class,
native refcount, and measured event duration when available. A block disappearing
from a selected session scope never emits `STATE_FREE`. A refcount-zero cached
eviction candidate emits `STATE_RECLAIM`; `STATE_FREE` additionally requires
native allocator availability and a cleared hash. Recorder self-work and trace
extraction are reported separately. End-to-end tracing perturbation is not
claimed or subtracted without paired disabled/full trials.

## Semantic limitations

- The state is valid only within the owning vLLM 0.23.0 `EngineCore` process.
- A block ID is ephemeral and may be reused after free; allocation epoch is
  therefore part of its trace identity.
- Native `ref_cnt` describes active allocator ownership. Prefix-cached blocks
  may remain physically resident with zero active references until eviction or
  explicit idle reset.
- The primed root is evictable before the first branch acquires native refs.
  Isolation, headroom, immediate admission, and identity revalidation mitigate
  but do not remove that runtime property; eviction invalidates the run.
- Only full, block-aligned prefix pages are shared by the experiment contract.
- Sampler state is reproduced from explicit deterministic inputs; it is not
  exported as a portable runtime object.
- No guarantee is made for TP/PP greater than one, hybrid or sliding attention,
  speculative decoding, LoRA, P/D, offload, remote connectors, migration, or a
  different vLLM version.
- This adapter and its traces are experiment instrumentation, not BranchFabric
  hardware and not evidence for an FPGA.

## Failure and cleanup contract

Every operation has a bounded timeout and exact owned request/session IDs.
Unsupported configuration, semantic mismatch, runtime error, CUDA error, OOM,
instrumentation error, or assertion failure stops further measurement and
enters idempotent cleanup. No device/runtime fallback is allowed.

Modal execution uses one single-use ephemeral Function container. Hot traces,
temporary caches, CUDA contexts, and subprocess state live only on its local
filesystem; exact experiment-owned children are terminated before return.
Final immutable artifacts are fsynced, copied to
`sloforge-branchfabric-results`, committed, hash-verified after local
materialization, and treated as a remote cache rather than the canonical
report. Only `sloforge-model-cache` and `sloforge-branchfabric-results` may
persist. No host-level `REMOTE_ROOT` or SSH audit applies to this backend. The
older `mew0` receipt under Experiment 002 is historical evidence only.
