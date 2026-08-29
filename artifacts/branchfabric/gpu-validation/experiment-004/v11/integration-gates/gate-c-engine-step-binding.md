# Gate C — production engine-step synchronization binding

Status: **PASS (offline production integration gate)**

Scope: vLLM 0.23.0 V1, synchronous `InprocClient`, one in-process engine. Async
scheduling, multiprocessing clients, missing mutation methods, wrapper
displacement, foreign witnesses, stale witnesses, cross-thread witnesses, and
unbounded acquisition are rejected.

## Exact binding

`Vllm0230EngineStepBinding` is installed on the validated
`VllmLiveInternalView`. It uses one bounded re-entrant critical section and
wraps the live instance's production entry points after other instrumentation
has been installed:

- `llm_engine.step` and `llm_engine.add_request`;
- scheduler admission, enqueue, selection, and output-commit methods;
- KV-manager allocate, free, cache-publication, and prefix-reset methods;
- waiting/skipped-waiting queue insert/remove methods;
- available abort/finish and in-process EngineCore/EngineCoreClient mutation
  methods.

These are the methods through which the validated vLLM 0.23 engine mutates KV
pages, request scheduling, branch-visible cached blocks, and block tables. The
gate is consequently on the production call path; it is not an unrelated
boolean probe or test-only mutex. Nested calls made by `LLMEngine.step` and the
restore stager re-enter the same lock.

## Export capture

The controller enters:

```python
with binding.critical_section("EXPORT_CAPTURE", gate_id=gate_id) as gate:
    # derive the source table and capture native pages here
```

CUDA capture checks the sealed witness before the first source read and again
after the final D2H fence/hash. While held, `LLMEngine.step`, scheduler
selection/output commit, branch cache publication, allocation/free, and queue
mutation cannot execute on another thread. Therefore source KV mutation,
scheduler reassignment, branch append, and block-table modification cannot race
with capture.

## Import and admission

The controller enters the same installed binding with
`IMPORT_ADMISSION`. The restore stager checks the sealed witness at entry,
before every queue insertion, and after the final insertion. The stager orders:

1. ordinary runtime allocation and block-table construction;
2. allocator-epoch proof;
3. native write and completion fence;
4. exact raw-bit/tail/integrity evidence;
5. destination/mapping/ownership/epoch validation;
6. single-use epoch consumption;
7. serial scheduler admission.

No engine step can observe a detached destination request before step 7. The
witness is tied to the exact scheduler, manager, binding UUID, owner thread,
operation, and lock acquisition generation, and expires on critical-section
exit.

## Offline evidence

Targeted concurrency tests cover real-wrapper installation, export exclusion
against engine step/scheduler reassignment/branch admission/block allocation/
free, re-entrant import mutation, stale and foreign witnesses, operation
mismatch, wrapper displacement, async/non-Inproc rejection, and bounded lock
timeout. These tests exercise production-shaped objects because the offline
host cannot import CUDA vLLM; passing them is necessary but does not replace
the real-runtime structural validation performed when the live binding is
installed. The combined allocator/ownership/v11/synchronization targeted suite
passed 40/40 under the local PyTorch environment. Ruff and mypy passed for the
Gate C source surfaces.

GPU resources were not invoked for this gate.
