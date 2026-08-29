# BranchFabric GPU runtime adapter decision

## Decision

BranchFabric GPU Validation Experiment 002 selects **vLLM 0.23.0**, official
tag commit `0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665`, as the only runtime for this
experiment. The adapter is process-local, version-scoped, and restricted to a
single V1 synchronous engine on one GPU. It uses vLLM's existing paged-KV
allocator and prefix-cache ownership; it does not patch vLLM, copy KV tensors,
or introduce a new allocator.

SGLang 0.5.15.post1 is technically feasible, but is rejected for this
experiment. Its native session-radix cache has suitable physical sharing and
close-time release semantics, but the repository is locked to SGLang 0.5.2,
the existing adapter evidence is for 0.5.12, the current release requires a
substantially newer CUDA 13/Torch 2.11 dependency stack, and exact physical
page observations require an SGLang scheduler patch. Those are unnecessary
risks when the locked vLLM release exposes the required state in-process.

This decision does not select vLLM or SGLang for portable Continuum state. The
existing transport adapters remain separate and continue to reject complete
portable execution-state export.

## Versions and sources inspected

| Runtime | Version or revision | Status in this repository | Decision |
| --- | --- | --- | --- |
| vLLM | `0.23.0`, commit `0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665` | `uv.lock` resolves this version; existing Continuum and Helix probes already cover it | Selected and pinned exactly |
| vLLM | `0.26.0`, commit `568afb3` | Outside the repository's `>=0.9,<0.24` compatibility interval | Rejected for this experiment; no benefit justifies internal-API and dependency churn |
| SGLang | `0.5.2` | Version resolved by `uv.lock` | Rejected; lacks the inspected current session-radix surface |
| SGLang | `0.5.12` / `0.5.12.post1` | Existing adapter source lock is 0.5.12 | Rejected; existing binding is P/D configuration only |
| SGLang | `0.5.15.post1`, commit `0b3bb0cbe31873994c9f989fddfe2f87ca839fdd` | Current source inspected independently | Feasible alternate, rejected for this experiment |

Primary vLLM source locks are:

- [V1 offline engine](https://github.com/vllm-project/vllm/blob/v0.23.0/vllm/v1/engine/llm_engine.py)
- [In-process engine client](https://github.com/vllm-project/vllm/blob/v0.23.0/vllm/v1/engine/core_client.py)
- [Engine core](https://github.com/vllm-project/vllm/blob/v0.23.0/vllm/v1/engine/core.py)
- [V1 scheduler](https://github.com/vllm-project/vllm/blob/v0.23.0/vllm/v1/core/sched/scheduler.py)
- [KV-cache manager](https://github.com/vllm-project/vllm/blob/v0.23.0/vllm/v1/core/kv_cache_manager.py)
- [Block pool](https://github.com/vllm-project/vllm/blob/v0.23.0/vllm/v1/core/block_pool.py)
- [KV block metadata](https://github.com/vllm-project/vllm/blob/v0.23.0/vllm/v1/core/kv_cache_utils.py)
- [KV cache geometry](https://github.com/vllm-project/vllm/blob/v0.23.0/vllm/v1/kv_cache_interface.py)
- [GPU KV backing tensors](https://github.com/vllm-project/vllm/blob/v0.23.0/vllm/v1/worker/gpu_model_runner.py)
- [vLLM automatic prefix caching design](https://docs.vllm.ai/en/v0.23.0/design/prefix_caching/)

Primary SGLang source locks are:

- [Radix cache](https://github.com/sgl-project/sglang/blob/v0.5.15.post1/python/sglang/srt/mem_cache/radix_cache.py)
- [Session radix cache](https://github.com/sgl-project/sglang/blob/v0.5.15.post1/python/sglang/srt/mem_cache/session_radix_cache.py)
- [Memory pools](https://github.com/sgl-project/sglang/blob/v0.5.15.post1/python/sglang/srt/mem_cache/memory_pool.py)
- [Paged allocator](https://github.com/sgl-project/sglang/blob/v0.5.15.post1/python/sglang/srt/mem_cache/allocator/paged.py)
- [Scheduler](https://github.com/sgl-project/sglang/blob/v0.5.15.post1/python/sglang/srt/managers/scheduler.py)

## Required vLLM configuration

The real-state adapter fails closed unless all of the following are true:

- distribution version is exactly `0.23.0`;
- V1 is active and `VLLM_ENABLE_V1_MULTIPROCESSING=0` was set before import;
- `LLM.llm_engine.engine_core` is an `InprocClient` with an in-process
  `EngineCore`;
- tensor and pipeline parallel sizes are one and exactly one CUDA device is in
  adapter scope;
- automatic prefix caching is enabled for the optimized path;
- the model resolves to one full-attention KV-cache group;
- speculative decoding, LoRA, sliding-window or hybrid attention, Mamba state,
  P/D disaggregation, offload, and remote KV connectors are disabled;
- model, tokenizer, revisions, dtype, block size, seed, decoding policy, and
  policy epoch match across the root and all branches.

The selected model is `Qwen/Qwen2.5-7B-Instruct`, immutable revision
`a09a35458c702b33eeacc393d103063234e8bc28`. The runtime must verify that the
model advertises at least a 16K context and that the tokenizer and weights come
from that revision. Any necessary model substitution must be recorded as a new
decision; it must not happen silently.

Remote execution loads a local immutable snapshot through the adapter's
explicit `execution_model_path`; this does not change the logical model or
tokenizer identity. The path must be an absolute local snapshot directory named
by the pinned revision, contain model/tokenizer configuration and safetensor
weights, and be supplied to vLLM for both model and tokenizer loading while the
official `revision` and `tokenizer_revision` remain pinned. A missing or
mismatched local snapshot fails before vLLM import and GPU allocation.

## Exact internal interfaces depended upon

All private access is confined to the vLLM-0.23 adapter. No vLLM object enters
portable Continuum schemas.

| Interface | Adapter use |
| --- | --- |
| `vllm.entrypoints.llm.LLM.llm_engine` | Own the offline engine and deterministic lifecycle |
| `vllm.v1.engine.llm_engine.LLMEngine.add_request()` and `step()` | Admit requests and pause only between synchronous scheduler/model steps |
| `LLMEngine.engine_core` / `vllm.v1.engine.core_client.InprocClient.engine_core` | Reach the same-process `EngineCore`; reject multiprocess clients |
| `EngineCore.scheduler.requests` and `.running` | Inspect live logical requests, token progress, and lifecycle |
| `EngineCore.scheduler.kv_cache_manager` | Map requests to allocated/cache-hit blocks |
| `KVCacheManager.get_computed_blocks()` and `allocate_slots()` | Runtime-native prefix reuse and suffix allocation performed by vLLM, never reimplemented by SLOForge |
| `KVCacheManager.get_blocks()` and `get_block_ids()` plus `KVCacheManager.coordinator.single_type_managers[0].req_to_blocks` | Exact request-to-physical-block ownership for the asserted single group |
| `KVCacheManager.free()` and `reset_prefix_cache()` | Exact branch release and final cached-root cleanup |
| `KVCacheManager.block_pool.blocks` and `BlockPool.get_num_free_blocks()` | Exact former-ID refcount/hash evidence plus aggregate free/usable pool recovery |
| `vllm.v1.core.kv_cache_utils.KVCacheBlock.block_id`, `.ref_cnt`, `.block_hash`, `.is_null` | Stable process-local block identity, native active ownership, cache identity, and null-block exclusion |
| `vllm.v1.kv_cache_interface.KVCacheConfig.num_blocks`, `.kv_cache_groups`, and `.kv_cache_tensors[*].size` | Assert one group and derive the full physical bytes addressed by one scheduler block ID across all layer tensors |
| `KVCacheGroupSpec.kv_cache_spec.block_size` and `.page_size_bytes`, plus `KVCacheGroupSpec.layer_names` | Tokens per block, per-layer byte cross-checks, and layer identity |
| `EngineCore.model_executor.driver_worker.worker.model_runner.kv_caches`, `.kv_cache_config`, and `.device` | Prove that block IDs address real CUDA KV backing tensors and cross-check reserved bytes |

These are not a supported public state-export API. The adapter therefore checks
the distribution version, module paths, concrete in-process client, field
presence, group geometry, device, and tensor residence at startup. A failed
check terminates the run before measurement.

## Runtime comparison

### vLLM feasibility

vLLM's V1 prefix cache is physical. `get_computed_blocks()` returns
`KVCacheBlock` objects already resident in the GPU KV pool. `allocate_slots()`
adds those same objects to the new request's per-group block table and increments
their native `ref_cnt`; it allocates new blocks only for uncached tokens. A
block ID selects a page in the CUDA KV tensors configured by `KVCacheConfig`.

This is sufficient for the required experiment without modifying vLLM. A
priming request prefills the root once and then finishes or is aborted through
the normal scheduler path, publishing its full blocks to the prefix cache.
Sibling requests are admitted immediately with the identical, block-aligned
prefix and a unique first suffix token. They must receive exactly the primed
root block IDs and new suffix blocks. Branch destruction calls the normal
request-abort/free path. After the last branch is gone,
`reset_prefix_cache()` removes the experiment's cached root.

The adapter does not manually increment `KVCacheBlock.ref_cnt` or call
`BlockPool.touch()` to pin the root: doing so would perturb vLLM's common-prefix
and eviction behavior. Between priming completion and first branch admission,
the cached root is resident but evictable and normally has native refcount zero.
The isolated engine, immediate admission, and required HBM headroom make that
window small; block IDs and hashes are rechecked and the run fails if eviction
or reuse occurs. Once branches are admitted, their native references preserve
the root while they are active.

### SGLang feasibility

SGLang 0.5.15.post1 also has real physical sharing. `RadixCache.match_prefix()`
returns GPU KV slot indices; `ReqToTokenPool.req_to_token` maps request positions
to those slots; `MHATokenToKVPool` owns the CUDA K/V buffers; and
`PagedTokenToKVPoolAllocator` allocates and frees their pages. The top-level
`session_id`, `SessionRadixCacheMixin._tag_session_leaf()`, and
`release_radix_session()` provide branch-scoped release with shared ancestors
preserved.

However, SGLang's public internal-state response reports aggregate memory only.
An observation-only scheduler patch would be needed to export radix nodes,
slot/page IDs, active request mappings, and ownership. Its session tags are
also ordinary evictable cache entries rather than pins. Combined with the
repository lock mismatch and CUDA 13/Torch 2.11 packaging delta, this makes it
the riskier implementation for Experiment 002.

## Capability matrix

`native` means vLLM supplies the operation. `adapter` means SLOForge sequences
or observes native operations without changing allocator semantics.

| `RealRuntimeStateAdapter` capability | Support | Mechanism |
| --- | --- | --- |
| start/prefill/pause/resume session | adapter + native | `LLMEngine.add_request()` and synchronous `step()` boundary |
| inspect logical state | adapter | scheduler request token and progress fields |
| inspect physical layout | native metadata | `req_to_blocks`, `get_block_ids()`, `KVCacheBlock` |
| inspect GPU memory | adapter | NVML/PyTorch plus KV tensor/config accounting |
| identify shared prefix | adapter | ordered block-ID intersection and logical-range check |
| create root reference/fork same policy | adapter + native | record primed cached root; immediately submit same aligned prefix; vLLM prefix lookup/reuse |
| allocate private suffix | native | `KVCacheManager.allocate_slots()` during extension/decode |
| concurrent branches | native | admit siblings before repeated engine steps |
| observe allocation/release | native metadata | snapshot/diff manager, pool, and native refcounts |
| report refcounts | native | `KVCacheBlock.ref_cnt`, cross-checked against request block tables |
| capture BranchPoint | adapter | immutable pointer-free snapshot of root IDs and policy identity |
| emit both trace kinds | adapter | timestamped diffs of real runtime state |
| destroy branch/session/runtime | adapter + native | exact request abort/free, prefix reset, engine shutdown; idempotent wrapper |

Unsupported by design are arbitrary mutation of shared blocks, portable or
cross-runtime state, migration, TP resharding, multi-node operation, LoRA
conversion, post-copy, and BranchFabric hardware.

## Semantic contract and terminology

The optimized path implements **aligned immutable shared-root, append-only
private-suffix semantics**. It is COW-equivalent for autoregressive KV state:
shared prefix pages are never modified, while divergent tokens allocate new
pages. It is not generic GPU virtual-memory COW, a CUDA page-fault mechanism,
or a way to fork arbitrary runtime objects. Reports must use “GPU COW” only
after exact shared block IDs and GPU backing are demonstrated, and must retain
this qualification.

The experiment BranchPoint declares:

```text
same_runtime_only = true
same_policy_only = true
same_process_or_adapter_scope = one vLLM 0.23.0 InprocClient/EngineCore process
cross_runtime_portable = false
```

The adapter's root-reference count is an ownership-equivalent lifecycle count,
not a fabricated allocator pin. Runtime pointers, CUDA handles, worker objects, and tensors remain ephemeral.
Only token identities, logical ranges, block IDs, group/layer labels, sizes,
native refcounts, allocation epochs, state classification, and provenance are
written to artifacts.

## Baseline and cache isolation

The independent baseline uses the same version, model revision, tokenizer,
dtype, GPU, block size, scheduler limits, seeds, and decoding settings, but
automatic prefix caching is disabled in a separately initialized engine. This
is the one intentional configuration difference. The adapter asserts that no
prefix block ID is referenced by more than one independent branch and records
that cached-token counts are zero.

The optimized engine enables prefix caching, prefills the root exactly once,
and asserts that every branch reuses the full block-aligned root without
executing prefix tokens again. Timing separately records native request
admission, the host-side post-admission lower bound for the engine step proven
to yield the first decode token, branch readiness when that step returns, and
adapter observation of the first emitted token. A first chunk allocation is
never treated as full-prefix readiness; state-inspection time is recorded
separately.

Engine state is never reused across baseline and optimized trials. Prefix cache
reset, idle state, pool free-count recovery, and a fresh measurement epoch are
required before another trial.

## HBM accounting

vLLM preallocates its KV pool, so NVML and PyTorch reserved HBM need not grow as
logical branches acquire pages. Every report therefore keeps three byte domains
separate:

1. process/device resident HBM from NVML and PyTorch;
2. runtime KV-pool reserved bytes from the CUDA tensors and `KVCacheConfig`;
3. assigned physical state, computed from unique non-null block IDs multiplied
   by the sum of per-layer physical pages addressed by each scheduler block ID.

Shared-prefix bytes count each unique root block once. Private-suffix bytes
count unique non-root blocks owned by live branches. Logical attribution may
sum a shared block over branches, but physical assigned bytes may not. The
adapter fails if assigned bytes exceed pool bytes, tensor/config sizes disagree,
or a supposedly private block is shared between divergent branches.

## Stability and stop conditions

The internal layout is unstable across vLLM releases. Any version mismatch,
multiprocess client, multiple KV group, unsupported attention/state type,
missing field, non-CUDA KV tensor, block-size inconsistency, unexpected block
aliasing, hidden baseline cache hit, hidden optimized re-prefill, or failed
release assertion invalidates the run. There is no fallback to CPU state,
serialized Continuum state, another model, another GPU, or SGLang.
