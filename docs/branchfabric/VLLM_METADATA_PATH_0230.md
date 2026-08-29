# vLLM 0.23.0 Branch Metadata Path

## Scope and source identity

This document is version-scoped to `vllm==0.23.0`, Git tag `v0.23.0`, commit
`0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665`. The source paths and line numbers
below refer to that commit. They must be revalidated before use with any other
vLLM version.

The SLOForge path is narrower than generic vLLM. It requires the V1 engine,
`InprocClient`, synchronous scheduling, one process, one GPU, one full-attention
KV-cache group, no speculative decoding, TP/PP/DP 1, a 16-token block size, and
prefix caching for the shared-root mode. These constraints are enforced by
`VllmLiveInternalView` in
`python/sloforge/continuum/adapters/vllm_live.py:209-350`. The adapter pins the
runtime and declares its internal dependencies at
`python/sloforge/continuum/adapters/vllm_live.py:60-79`.

The source was inspected from the release tag rather than reconstructed from
API memory. Experiment 002 independently confirms that this is the live path:
its retained stack traces enter `vllm/v1/engine/llm_engine.py`, construct an
`InprocClient`, and reach `vllm/v1/worker/gpu_model_runner.py`. Its invocation
also records `disable_log_stats=true`, which matters for the native-metrics
availability described below.

## Correction to the Experiment 002 signal

Experiment 002's reported 17.17% `state_management_fraction_of_readiness` at
fanout 8 is not a measured vLLM prefix/refcount fraction. The numerator is the
runner's `fork_metadata` wall interval at
`benchmarks/branchfabric/gpu_validation_runner.py:1494-1537`. That interval:

1. calls `VllmLiveStateAdapter.fork_same_policy_session()` for each branch;
2. calls `inspect_logical_state()` and `_assert_logical_prefix()` for each
   branch; and
3. ends before `run_concurrent_branches()` submits any branch to vLLM.

`fork_same_policy_session()` is explicitly metadata-only before runtime
admission (`python/sloforge/continuum/adapters/vllm_live.py:1339-1403`). At 16K
it copies the root token tuple and appends the divergent token at line 1375.
The validation path constructs another token tuple and hashes every token with
an 8-byte encoding at `vllm_live.py:1016-1036` and
`python/sloforge/continuum/adapters/real_runtime.py:530-533`. Full tracing adds a
branch event at `vllm_live.py:1386-1401`.

For the accepted fanout-8 run, `fork_metadata` was 46,216,678 ns and
`all_branches_ready_ns` was 269,240,839 ns. Their quotient is the reported
17.17%. The same artifact reports first/all request admission at 72,251,001 ns
and 78,559,480 ns after `run_concurrent_branches()` began, proving that native
runtime admission happened later. Experiment 003 must classify the old
numerator as Helix/SLOForge orchestration and validation until a new direct
measurement says otherwise. These intervals are consecutive, not nested: the
old quotient is a diagnostic ratio and cannot be treated as the fraction of an
additive child span within one `POST_ROOT_READY` parent.

## Exact request-to-first-output call chain

The shared-root branch has 16,385 prompt tokens: the 16,384-token immutable root
plus one divergent token. With a 16-token block size, the root comprises 1,024
full blocks.

### SLOForge construction and submission

1. `VllmLiveStateAdapter.run_concurrent_branches()` records its start, then
   submits branches sequentially with `_submit()`
   (`python/sloforge/continuum/adapters/vllm_live.py:1546-1599`).
2. `_submit()` builds `SamplingParams`, materializes
   `{"prompt_token_ids": list(session.token_ids)}`, and calls
   `LLMEngine.add_request()` (`vllm_live.py:916-937`).
3. `LLMEngine.add_request()` calls `InputProcessor.process_inputs()`, assigns
   the internal randomized request ID, creates frontend `RequestState` through
   `OutputProcessor.add_request()`, then calls `InprocClient.add_request()`
   (`vllm/v1/engine/llm_engine.py:209-268`).
4. `InputProcessor.process_inputs()` validates and preprocesses the token-id
   prompt, clones sampling parameters, and creates `EngineCoreRequest`
   (`vllm/v1/engine/input_processor.py:242-385`). `assign_request_id()` performs
   internal ID randomization at lines 230-240 unless explicitly disabled.
5. `OutputProcessor.add_request()` constructs frontend output/detokenizer state
   and updates its request tables (`vllm/v1/engine/output_processor.py:512-541`).

All objects in this section are Python runtime objects. The SLOForge session
and root references are project-owned. `EngineCoreRequest`, frontend
`RequestState`, and all objects below are vLLM types and must not enter generic
Continuum IR.

### Engine request creation and scheduler queue admission

1. `InprocClient.add_request()` calls
   `EngineCore.preprocess_add_request()` and then `EngineCore.add_request()`
   (`vllm/v1/engine/core_client.py:296-299`).
2. `EngineCore.preprocess_add_request()` constructs scheduler `Request` through
   `Request.from_engine_core_request()` (`vllm/v1/engine/core.py:819-841`).
3. `Request.__init__()` copies prompt IDs, initializes scheduling state, and
   calls `update_block_hashes()` (`vllm/v1/request.py:59-182`). The block hasher
   iterates every full 16-token block and computes a chained content hash
   (`vllm/v1/core/kv_cache_utils.py:659-710`). For a 16,385-token branch this is
   1,024 block-hash computations during request construction, before scheduler
   selection or prefix-cache lookup.
4. `EngineCore.add_request()` delegates to `Scheduler.add_request()`
   (`vllm/v1/engine/core.py:341-376`).
5. `Scheduler.add_request()` inserts the request into `requests` and calls
   `_enqueue_waiting_request()` (`vllm/v1/core/sched/scheduler.py:1801-1823`).
   For this unblocked FCFS workload `_enqueue_waiting_request()` calls
   `waiting.add_request()` (`scheduler.py:1659-1664`), which is a deque append in
   `vllm/v1/core/sched/request_queue.py:75-84`.

There is no scheduler mutex on this bounded `InprocClient` path. Submission and
scheduling run serially in the interpreter. The `multiprocessing.Lock` created
by `UniProcExecutor` is passed to the optional shared multimodal receiver cache
(`vllm/v1/executor/uniproc_executor.py:45-68` and
`vllm/v1/worker/worker_base.py:289-309`); it is not used by the text-only
scheduler/KV path. Python interpreter/GIL cost remains possible and must be
profiled, but should not be reported as a measured lock wait without evidence.

### Scheduler selection, prefix lookup, and physical metadata

`VllmLiveStateAdapter` calls `LLMEngine.step()` after all branches are submitted
(`vllm_live.py:1417-1423`). The exact path is:

1. `LLMEngine.step()` -> `InprocClient.get_output()`
   (`vllm/v1/engine/llm_engine.py:287-325`).
2. `InprocClient.get_output()` -> `EngineCore.step_fn()` -> `post_step()`
   (`vllm/v1/engine/core_client.py:288-291`). In this configuration `step_fn` is
   ordinary `EngineCore.step()` (`vllm/v1/engine/core.py:217-220`).
3. `EngineCore.step()` calls `Scheduler.schedule()` before worker execution
   (`vllm/v1/engine/core.py:443-472`).
4. `Scheduler.schedule()` scans the waiting queue. For each new branch it peeks
   the queue head, then calls `KVCacheManager.get_computed_blocks()`
   (`vllm/v1/core/sched/scheduler.py:562-613`).
5. `KVCacheManager.get_computed_blocks()` caps the hit length at
   `request.num_tokens - 1` and calls the coordinator's
   `find_longest_cache_hit()` (`vllm/v1/core/kv_cache_manager.py:196-236`). This
   cap is exactly 16,384 tokens for the branch workload.
6. Qwen2.5's validated single full-attention cache group selects
   `UnitaryKVCacheCoordinator.find_longest_cache_hit()`
   (`vllm/v1/core/kv_cache_coordinator.py:379-448`) and
   `FullAttentionManager.find_longest_cache_hit()`
   (`vllm/v1/core/single_type_kv_cache_manager.py:521-569`). The latter performs
   one sequential `BlockPool.get_cached_block()` lookup per root block and stops
   at the first miss. A full hit therefore performs 1,024 direct cache lookups
   per branch (`vllm/v1/core/block_pool.py:184-209`).
7. The scheduler calculates one token of new work and calls
   `KVCacheManager.allocate_slots()` (`scheduler.py:680-772`).
8. `allocate_slots()` performs capacity calculation, installs cached blocks,
   and allocates new blocks (`vllm/v1/core/kv_cache_manager.py:238-436`). The
   coordinator delegates to `SingleTypeKVCacheManager`.
9. `SingleTypeKVCacheManager.allocate_new_computed_blocks()` calls
   `BlockPool.touch()` and extends `req_to_blocks`
   (`vllm/v1/core/single_type_kv_cache_manager.py:182-245`). `BlockPool.touch()`
   iterates the 1,024 hit blocks, removes a refcount-zero cached block from the
   free list when necessary, and increments every refcount
   (`vllm/v1/core/block_pool.py:402-417`). Thus the initial shared admission has
   1,024 directly countable refcount increments per branch.
10. `SingleTypeKVCacheManager.allocate_new_blocks()` computes the missing block
    count, calls `BlockPool.get_new_blocks()`, and extends `req_to_blocks`
    (`single_type_kv_cache_manager.py:259-290`). At 16,385 prompt tokens it
    allocates exactly one private block per branch. `BlockPool.get_new_blocks()`
    increments that new block's refcount (`block_pool.py:333-363`).
11. On success the scheduler pops the FCFS queue, appends the request to
    `running`, changes its status to `RUNNING`, and creates scheduler-to-worker
    `NewRequestData` (`scheduler.py:803-910` and
    `vllm/v1/core/sched/output.py:30-65`). `KVCacheBlocks.get_block_ids()`
    materializes Python block-ID lists (`kv_cache_manager.py:57-84`).

The adapter's existing observer wraps only `KVCacheManager.allocate_slots()`
and `free()` (`python/sloforge/continuum/adapters/vllm_live.py:744-819`). Its
`request_admitted_ns` timestamp at line 783 is direct evidence of successful slot
allocation, but it is earlier than the subsequent queue pop/status transition
and does not decompose lookup, refcount, allocation, or scheduler-selection
time.

For the first scheduler admission, assuming the validated full 16K hit, no
capacity deferral, and all requested branches selected in that step, the source
code gives the following counter invariants. These are expectations to assert
against direct hooks, not substitutes for capturing the counters:

| Fanout | Request full-block hashes | Cache lookups | Shared-block refcount increments | Private block allocations | Total initial refcount increments | CPU block-table ID writes | Queue inserts / removals |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1,024 | 1,024 | 1,024 | 1 | 1,025 | 1,025 | 1 / 1 |
| 8 | 8,192 | 8,192 | 8,192 | 8 | 8,200 | 8,200 | 8 / 8 |
| 16 | 16,384 | 16,384 | 16,384 | 16 | 16,400 | 16,400 | 16 / 16 |
| 32 | 32,768 | 32,768 | 32,768 | 32 | 32,800 | 32,800 | 32 / 32 |

In symbols, the dominant initial metadata counts are `N * prefix_blocks` for
request hashes, cache lookups, and shared-root touches; `N` for private-page
allocations; and `N * (prefix_blocks + 1)` for the initial worker block-table
writes. Later private-suffix allocation during the 256-token decode must be
counted separately at its actual allocation steps; it is not part of these
first-admission invariants.

### Scoped immutable-root hash optimization

The optimized software path replaces only the initial per-branch request-hash
construction. It captures the source request's output from the exact
`EngineCore.request_block_hasher` callable, publishes the first 1,024 hashes
only after the adapter has verified the aligned, resident immutable root, and
returns a fresh list containing those hashes for an eligible 16,385-token
branch. `BlockHash` is `NewType("BlockHash", bytes)` at
`vllm/v1/core/kv_cache_utils.py:43`, so sharing the immutable hash values while
copying the list container preserves `Request.update_block_hashes()` semantics.

Eligibility is version/configuration scoped: one pure-token full-attention
group, no LoRA or multimodal hash keys, no cache salt, an exact root-plus-one
prompt shape, and an adapter branch tied to the published root. Later hasher
calls fall through to vLLM's original chained hasher, using the template's last
hash as the previous-block value. The optimization eliminates 1,024 hash
computations per branch; it does not eliminate the subsequent 1,024 prefix
cache probes, refcount updates, physical allocation, or worker block-table
writes. The tracing-disabled control installs only this semantic hasher hook;
all timing and counter hooks remain absent.

### Worker block table, GPU submission, and first output

1. After `schedule()` returns, `EngineCore.step()` calls
   `model_executor.execute_model(scheduler_output, non_block=True)`
   (`vllm/v1/engine/core.py:454-461`).
2. `UniProcExecutor.execute_model()` calls `collective_rpc()`, which invokes
   `WorkerWrapperBase.execute_model()` directly with `run_method()`
   (`vllm/v1/executor/uniproc_executor.py:79-119`,
   `vllm/v1/serial_utils.py:486-510`, and
   `vllm/v1/worker/worker_base.py:340-345`). In this in-process executor,
   `non_block=True` does not create a worker thread; ordinary results are wrapped
   in an already-completed `Future` after the direct call.
3. `GPUWorker.execute_model()` calls `GPUModelRunner.execute_model()`
   (`vllm/v1/worker/gpu_worker.py:805-894`).
4. `GPUModelRunner._update_states()` creates `CachedRequestState` and adds each
   new request to the persistent `InputBatch`
   (`vllm/v1/worker/gpu_model_runner.py:1125-1257,1436-1447`).
5. `InputBatch.add_request()` copies the prompt IDs and calls
   `MultiGroupBlockTable.add_row()` (`vllm/v1/worker/gpu_input_batch.py:335-413`).
   That dispatches to each child `BlockTable.add_row()`; its `append_row()`
   writes every physical block ID into a pinned CPU NumPy table
   (`vllm/v1/worker/block_table.py:102-122,267-289`). The direct counter observes
   the outer `add_row`/`append_row` arguments and applies each child table's
   `blocks_per_kv_block` expansion factor. This is 1,025 CPU integer writes per
   branch on the validated initial 16K-root-plus-divergence admission.
6. `_prepare_inputs()` starts an asynchronous CPU-to-GPU copy of the block table
   with `commit_block_table()` and later prepares slot mappings and other input
   tensors (`gpu_model_runner.py:1872-1910,2079-2113` and
   `block_table.py:166-167,312-314`).
7. `GPUModelRunner.execute_model()` builds attention metadata, then enters
   `_model_forward()` at lines 4258-4287. `_model_forward()` calls the model at
   `gpu_model_runner.py:3718-3748`. A CPU timestamp at this call boundary is a
   direct launch-call observation, not proof of when the GPU began executing.
8. The runner computes logits, stores its ephemeral execution state, and returns
   `None` (`gpu_model_runner.py:4289-4366`). `EngineCore.step()` consequently
   calls `model_executor.sample_tokens()` (`core.py:461-464`).
9. `GPUModelRunner.sample_tokens()` samples, performs bookkeeping, and returns a
   CPU-visible `ModelRunnerOutput` (`gpu_model_runner.py:4383-4596`). Because
   async scheduling is disabled, its bookkeeping/D2H path synchronizes output
   before return. GPU kernel duration versus CPU launch and output wait requires
   CUDA events or a diagnostic profiler; plain wall timing of the wrapper is not
   pure GPU time.
10. `Scheduler.update_from_output()` commits sampled tokens and creates
    `EngineCoreOutput` (`vllm/v1/core/sched/scheduler.py:1329-1569`).
11. `InprocClient.get_output()` returns to `LLMEngine.step()`, whose
    `OutputProcessor.process_outputs()` detokenizes/commits frontend output
    (`vllm/v1/engine/output_processor.py:576-693`).
12. SLOForge `_process_outputs()` copies output token IDs and records its first
    token timestamp (`python/sloforge/continuum/adapters/vllm_live.py:939-959`).

Experiment 002's `first_decode_token_started_ns` is not a direct GPU-start
timestamp. The adapter records the CPU start of the `LLMEngine.step()` call that
eventually returns the first token, then clamps it to the allocation-observer
timestamp (`vllm_live.py:1594-1599,1662-1667`). Experiment 003 must use a new
worker hook/CUDA event for direct GPU evidence and retain this old value only as
an inferred lower-resolution boundary.

### Release path

On request completion, `Scheduler._free_blocks()` calls
`KVCacheManager.free()` (`vllm/v1/core/sched/scheduler.py:1907-1910`). This
delegates through the coordinator to
`SingleTypeKVCacheManager.free()` (`vllm/v1/core/kv_cache_manager.py:438-447`,
`vllm/v1/core/kv_cache_coordinator.py:264-272`, and
`vllm/v1/core/single_type_kv_cache_manager.py:363-378`).
`BlockPool.free_blocks()` decrements every request-owned block refcount and
returns newly unreferenced blocks to the free queue
(`vllm/v1/core/block_pool.py:419-441`). These are direct decrement/free
operation-count insertion points.

## Version-scoped instrumentation map

The implemented recorder installs and removes instance-level wrappers from
`VllmLiveStateAdapter`; it does not patch portable Continuum types. It validates
`vllm==0.23.0` and the adapter validates the exact configuration contract before
the hooks are installed. Hooks write fixed-size counters and bounded timestamps.
Full per-probe spans are emitted only in diagnostic runs. Each observation also
serializes `stage_measurement_semantics` and `operation_counter_semantics`, so an
analysis does not have to infer directness from a counter name.

| Measurement | Exact insertion point | Evidence | Notes |
|---|---|---|---|
| Helix branch construction | `VllmLiveStateAdapter.fork_same_policy_session` | Direct | Separate tuple construction, trace emission, and runner logical validation. |
| Logical validation | `inspect_logical_state`, `token_history_sha256`, runner `_assert_logical_prefix` | Direct | This is not vLLM metadata. Count encoded/hashed tokens. |
| Request construction | adapter `_submit`; `LLMEngine.add_request`; `InputProcessor.process_inputs`; `OutputProcessor.add_request` | Direct | Use exclusive timings because these calls are nested. Count prompt list copies and frontend state allocations. |
| Request block hashing | `EngineCore.request_block_hasher`, passed through `EngineCore.preprocess_add_request` into `Request.update_block_hashes` | Direct | Count hashes returned by each hasher call; 1,024 hashes per baseline 16K branch. The optimized path records an explicit root-template hit instead. |
| Scheduler queue insert | `Scheduler._enqueue_waiting_request` and `RequestQueue.add_request` | Direct | Count one insert per request and record enqueue timestamp. |
| Scheduler queue removal/admission | `Scheduler.schedule` returned `scheduled_new_reqs` | Derived-exact for this workload | Counts newly admitted queue removals after successful scheduling. It deliberately does not claim to count preempted/resumed/skipped queue traffic; those paths are excluded by the validated Experiment 003 configuration. |
| Scheduler scan/candidate evaluation | `Scheduler.schedule`; `KVCacheManager.get_computed_blocks` | Direct calls, scoped interpretation | `scheduler_scans` is the number of schedule calls. `scheduler_candidate_evaluations` is the number of prefix-lookup candidate calls and equals loop candidates only for the validated unblocked FCFS/no-LoRA/no-connector path. |
| Prefix lookup | `KVCacheManager.get_computed_blocks`; `UnitaryKVCacheCoordinator.find_longest_cache_hit`; `FullAttentionManager.find_longest_cache_hit`; `BlockPool.get_cached_block` | Derived-exact in MINIMAL; direct in FULL | MINIMAL derives probes as hits plus one on an early miss from the returned aligned hit-token count and the pinned loop bound, avoiding 8,192 Python wrappers at fanout 8. FULL wraps each probe and records its hit/miss result. The two counts must agree on a diagnostic run. |
| Refcount increments | `BlockPool.touch` and `BlockPool.get_new_blocks` | Direct | Count sequence lengths and actual `ref_cnt` changes. Avoid a per-block timer in minimal mode. |
| Refcount decrements/frees | `BlockPool.free_blocks` | Direct | Count decrements and blocks reaching zero separately. |
| Physical allocation | `BlockPool.get_new_blocks`; `SingleTypeKVCacheManager.allocate_new_blocks` | Direct | `block_allocations` counts every returned physical block. `private_suffix_allocations` counts coordinator allocations only when the adapter request scope names a shared root, so independent-prefix pages are not mislabeled as suffixes. |
| Scheduler block-ID materialization | `KVCacheBlocks.get_block_ids`; `NewRequestData.from_request` | Not separately timed | Its uncovered wall time remains scheduler-select exclusive time. The actual downstream worker-table integer writes are counted directly. |
| Scheduler prefix binding | `UnitaryKVCacheCoordinator.allocate_new_computed_blocks`, called with keyword arguments by `KVCacheManager.allocate_slots` | Direct | Count `new_computed_blocks` from the named argument. This is `req_to_blocks` binding, not a worker table write. |
| Worker block-table writes | `GPUModelRunner.input_batch.block_table` (`MultiGroupBlockTable.add_row`/`append_row`) | Direct | Count submitted IDs times each child `BlockTable.blocks_per_kv_block`; 1,025 actual CPU integer writes per initial branch is expected for the validated non-hybrid one-group layout. |
| CPU-to-GPU block-table copy | `BlockTable.commit_block_table`; `CpuGpuBuffer.copy_to_gpu` if needed | Included, not isolated | The copy is inside the direct `GPUModelRunner.execute_model` wall span. No standalone copy event is emitted in MINIMAL, so copy completion and bandwidth must not be claimed from this decomposition. |
| Runtime dispatch | `UniProcExecutor.execute_model` and `UniProcExecutor.sample_tokens` | Direct CPU wall | Exclusive time is in `GPU_SUBMISSION`; it shows in-process dispatch, not GPU start. |
| First GPU work | `GPUModelRunner._model_forward` plus CUDA events or the existing `gpu_model_runner: forward` profiler range | Direct only with CUDA event/profiler | CPU entry alone is a launch-call proxy. Resolve events after the existing output synchronization; do not add a synchronization to minimal runs. |
| Worker execute and sampling/output synchronization | `GPUModelRunner.execute_model` and `GPUModelRunner.sample_tokens` | Direct CPU wall, GPU-inclusive | Both halves are classified as `GPU_EXECUTION`. They contain CPU preparation/bookkeeping and GPU/output waits, so they are not pure device time. The first `_model_forward` CUDA event is a separate non-additive direct device diagnostic. |
| Scheduler token commit | `Scheduler.update_from_output` | Direct | Count requests/tokens committed. |
| Frontend output commit | `OutputProcessor.process_outputs` | Direct | Includes detokenization and `RequestOutput` construction. |
| SLOForge first-output observation | adapter `_process_outputs` | Direct | Preserve separately from vLLM frontend commit. |

Per-request queue residence intervals overlap request construction for later
branches. They are useful auxiliary measurements but must not be summed as
non-overlapping children of `POST_ROOT_READY`. For additive wall decomposition,
record the union of queue-only idle intervals, or assign the sequential top-level
timeline to request construction/admission and report per-request native queue
latencies outside the additive hierarchy. Likewise, nested vLLM function wall
times require exclusive-time subtraction before addition.

A defensible non-overlapping critical-path partition for this exact synchronous
path is:

1. Helix orchestration and validation before `_submit`;
2. exclusive request construction and queue insertion across all submissions;
3. queue-only gap after the last submission, if any;
4. exclusive scheduler selection;
5. prefix lookup;
6. block/refcount/private allocation;
7. scheduler-to-worker metadata materialization;
8. worker preparation and CPU-to-runtime launch;
9. GPU execution/output synchronization wall interval, with CUDA-event GPU time
   reported as a nested diagnostic rather than added again;
10. scheduler output/token commit;
11. frontend and SLOForge output commit; and
12. residual.

The sum check should use CPU monotonic wall boundaries on this partition. CUDA
elapsed time, per-request queue latency, and nested profiler spans are
cross-checks, not additional summands.

`SCHEDULER_WAIT` deserves special care: its two timestamps are direct, but its
attribution is inferred. It is exactly the gap from the end of the last branch
submission to entry into the first `Scheduler.schedule()` call. It is not the
sum of per-request queue residence and does not include scheduler policy work.

`POST_ROOT_READY` counters freeze when all branches return their first output.
The recorder remains installed during steady decode and destruction; a second
snapshot persists cumulative lifecycle counters, including later suffix
allocation and refcount decrements/frees. The readiness snapshot and lifecycle
snapshot are separate artifacts so teardown work cannot silently contaminate
the primary metadata numerator.

The shared-root semantics check is independent of the optimization counters.
For every admitted branch, `VllmLiveStateAdapter._validate_shared_branch_tables`
requires the request table's ordered prefix IDs and the allocation observer's
initial cache-hit IDs to equal the published root exactly, requires exactly
16,384 cached tokens, rejects shared private-page IDs, and checks native root
refcounts after every branch is present. The optimized hash-template path must
therefore still pass the same physical-prefix validation as the baseline; a
template hit alone is never accepted as proof of sharing.

## vLLM-native metrics in 0.23.0

The public V1 `LLM.get_metrics()` API delegates to
`LLMEngine.get_metrics()` and the in-memory Prometheus registry
(`vllm/entrypoints/llm.py:847-857`, `vllm/v1/engine/llm_engine.py:369-371`, and
`vllm/v1/metrics/reader.py:70-143`). It asserts that stat logging is enabled.
Experiment 002 set `disable_log_stats=true`, so these metrics were unavailable
there. Experiment 003 must deliberately enable them for native-metric trials
and measure that choice's overhead; it must not silently compare a stats-on
trial with a stats-off performance trial.

Applicable metric names declared in `vllm/v1/metrics/loggers.py` are:

- `vllm:num_requests_running` and `vllm:num_requests_waiting`, with
  `vllm:num_requests_waiting_by_reason` for capacity versus deferred waiting;
- `vllm:kv_cache_usage_perc`;
- `vllm:prefix_cache_queries` and `vllm:prefix_cache_hits` (token counts, not
  hash-table operation counts);
- `vllm:num_preemptions`;
- `vllm:prompt_tokens`, `vllm:prompt_tokens_by_source`, and
  `vllm:prompt_tokens_cached`;
- `vllm:generation_tokens`;
- `vllm:time_to_first_token_seconds`;
- `vllm:request_queue_time_seconds`,
  `vllm:request_inference_time_seconds`,
  `vllm:request_prefill_time_seconds`, and
  `vllm:request_decode_time_seconds`; and
- `vllm:request_prefill_kv_computed_tokens`.

Optional sampled KV residency metrics are
`vllm:kv_block_lifetime_seconds`,
`vllm:kv_block_idle_before_evict_seconds`, and
`vllm:kv_block_reuse_gap_seconds`; they exist only when the observability
`kv_cache_metrics` option is enabled and are eviction/reuse samples, not direct
refcount-operation latency metrics (`loggers.py:936-1000`). Enabling them in a
primary minimal run would change instrumentation and must be controlled.

`Scheduler.make_stats()` produces native running/waiting counts, KV usage, and
prefix statistics only when `log_stats` is true
(`vllm/v1/core/sched/scheduler.py:2023-2059`). Native request queue, prefill, and
decode durations are derived from `QUEUED`, `SCHEDULED`, and first/last-token
timestamps (`vllm/v1/metrics/stats.py:404-472`). They independently cross-check
coarse custom timing but do not decompose scheduler selection, hash lookup,
refcount handling, block-table construction, or worker launch.

| Custom measurement | Native confirmation in 0.23.0 |
|---|---|
| Queue insertion/removal timestamps | Partial: request queue-time histogram and running/waiting gauges |
| Prefix hit tokens | Confirms: prefix query/hit and cached-prompt-token counters |
| Hash lookup count/latency | Unavailable; direct hook required |
| Refcount increments/decrements | Unavailable; direct hook required |
| Physical allocations/frees | Partial: KV usage and optional lifetime/reuse samples |
| Scheduler scan/candidate count/selection time | Unavailable; direct hook required |
| Block-table writes and CPU-to-GPU copies | Unavailable; direct hook required |
| GPU forward duration | Unavailable in ordinary metrics; CUDA events/profiler required |
| TTFT/prefill/decode/output | Partial confirmation from native request histograms and token counters |

## Public versus internal interfaces

`vllm.LLM.get_metrics()` is a documented public V1 entrypoint. The SLOForge
adapter methods are project APIs. `LLMEngine` is explicitly a legacy
compatibility engine, and every `vllm.v1.engine`, `vllm.v1.core`, scheduler,
KV-manager, block-pool, worker, input-batch, and block-table object named above
is an internal implementation detail. Instrumentation of those objects is safe
only under the exact-version/configuration guard and must fail closed on drift.

No vLLM internal object or class should appear in generic Continuum IR or the
version-independent trace schema. Persist only scalar timings, counters,
versioned string labels, physical integer block IDs already covered by the live
adapter contract, and provenance naming the exact insertion point.
