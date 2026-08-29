# Agent 03 independent engine-step synchronization audit

Status: **PASS (offline production integration gate)**

Final audit time: `2026-08-19T00:34:57Z`

This review was read-only with respect to source and did not invoke vLLM,
CUDA, Modal, the network, or any GPU resource. The frozen v10 path and tag were
not modified.

## Production binding

`Vllm0230EngineStepBinding` accepts only the exact vLLM 0.23.0 V1 synchronous
`InprocClient` object graph and validates the identities
`LLMEngine -> InprocClient -> EngineCore -> Scheduler -> KVCacheManager`. It
rejects async scheduling and missing required mutation methods.

The one bounded re-entrant lock is installed directly on the live instance
methods, including:

- `llm_engine.step` and request add/abort;
- EngineCoreClient and EngineCore add/step/abort surfaces when present;
- scheduler add, enqueue, schedule, output update, and finish;
- KV-manager allocate, free, cache publication, and prefix reset; and
- waiting/skipped-waiting queue insertion and removal.

This is not an unrelated synthetic mutex. Once installed on the live validated
view, ordinary engine steps, scheduler reassignment/output commit, branch
admission/cache publication, allocation/free, and block-table mutation all
traverse the same lock. Required surfaces are mandatory, wrapper displacement
is detected, acquisition is bounded, and close while held is rejected.

The sealed witness binds the exact binding UUID, scheduler identity, manager
identity, owner thread, operation, and acquisition generation. A witness is
invalid outside the held section and is rejected if stale, foreign,
cross-operation, cross-runtime, or displaced.

## Actual v11 runtime integration

The isolated production worker constructs the binding directly from the live
adapter view before creating the rollout state:

```text
adapter = _create_adapter(...)
binding = Vllm0230EngineStepBinding(adapter._view, ...)
```

The `EXPORT_CAPTURE` critical section is a single lexical and runtime scope
covering all of the following:

1. live request/token/block-table observation through `_runtime_capture_inputs`;
2. canonical capture-plan construction from those live allocations;
3. exact pre-release ownership/refcount/allocator-epoch proof;
4. optimized native capture, hashing, D2H, and its completion fence; and
5. source branch/root release plus exact post-free ownership proof.

The export witness is passed to both the CUDA capture helper and ownership
release proof. The helpers revalidate it. No engine step, scheduler selection
or output commit, branch append/admission, KV allocation/free, prefix-cache
publication, or queue mutation can run concurrently on another thread.

The `IMPORT_ADMISSION` critical section is one scope covering:

1. streaming transport authentication;
2. ordinary creation of eight fresh restore request incarnations;
3. request detachment and complete native block-table allocation;
4. allocator-issued lifetime proof;
5. direct H2D/native write, tail zero, raw-bit validation, and completion fence;
6. mapping/destination/integrity evidence validation;
7. single-use allocator-epoch consumption; and
8. serial scheduler enqueue of all restored branches.

The write and failure-scrub callbacks close over the exact `import_gate` and
pass it to the CUDA helpers. The stager independently validates that same
witness at entry, around each callback, before epoch consumption, before every
queue insertion, and after final insertion. Requests stay detached from the
runnable queues until native bytes, integrity, complete block tables,
ownership, mapping commitments, and epochs are valid. Any gate loss triggers
fail-closed cleanup or typed runtime teardown.

After critical-section exit, continuation uses the now-wrapped production
`llm_engine.step`; each ordinary step therefore acquires the same installed
binding normally. Binding teardown restores the previously installed live
adapter wrappers before adapter cleanup.

## Test evidence

The concurrency suite covers direct wrapper installation, export exclusion
against engine step/scheduler selection/branch enqueue/allocation/free,
re-entrant import mutation, bounded timeout, stale/foreign/wrong-operation
witness rejection, async/non-Inproc rejection, and wrapper-displacement
detection.

The isolated runtime integration test parses the production worker and proves
that exactly the two required critical sections exist and that their lexical
scopes contain the required export/release and import/write/scrub calls. It
also proves all restore operations share one authoritative pass recorder.
This structural test supplements rather than replaces the production call
site: the worker itself binds `adapter._view`, not a test double.

Command:

```text
PYTHONPATH=python python -m pytest -q \
  tests/python/test_gpu_reclamation_runtime_v11.py \
  tests/python/test_vllm_reclamation_v11_sync.py \
  tests/python/test_vllm_reclamation_v11.py \
  tests/python/test_vllm_reclamation_v11_ownership.py
```

Result: `45 passed in 1.17s`.

Ruff lint and format checks passed for all seven audited implementation/test
files.

## Audited hashes

- production v11 worker: `ef21c2c159b727f350534e1307b3ea287fa9788b6c7c7b42be53512a2bdd5e45`
- sync implementation: `095384e5d892d6a1899a9dc1ddc9e2c2bea9d3dbcc3d8c87176c381aab2afbd0`
- v11 adapter/integration: `ee8c0ca43068265c40c2d7fdb3ddd1d5730b93dc2aac98275bd7510e740c868e`
- runtime integration tests: `f234ff79e9f3aacd3e0960ffb8838b61d4987ad4609615f6879cdc57074d490c`
- synchronization tests: `35daa70da4006bcd5cf412d870eb30bc0b921b0d0715b15dd2c44b8d3faf5ebc`
- v11 adapter tests: `c81463989638186cd2e1f8792ed1833a982eeeda92cabe8d8d111ad4cd4c7f3b`
- ownership tests: `2e45a183923de79acf198db5f7ed153712ef9bf268ba23f4e308143ff32efe84`

The offline gate is PASS. Real vLLM/CUDA execution remains correctly deferred
to the separately authorized A100 micro-validation; that execution must still
record `binding.evidence().passed == true`, and the micro result fails closed
otherwise.
