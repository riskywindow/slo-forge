# Experiment 004 v11 worker/controller integration audit

Status: **OFFLINE DESIGN COMPLETE; NO MODAL, GPU, OR REMOTE COMPUTE USED**

Scope: `gpu_reclamation_worker.py`, the controller/Modal/launcher path, the vLLM 0.23 restore stager, movement accounting, and their tests. This review makes no implementation change and does not authorize a GPU launch.

## Decision

Use a separately versioned v11 execution path. Do not widen the frozen v10 config class or reinterpret an existing v10 mode. The safest minimal graph is:

```text
branchfabric-experiment-004-v11-launch.py
  -> modal_gpu_reclamation_v11.py
       -> micro: gpu_reclamation_v11_controller.run_one_gpu_micro_controller
       -> full:  gpu_reclamation_v11_controller.run_integrated_two_gpu_v11_controller
            -> gpu_reclamation_v11_worker.py
                 -> gpu_reclamation_v11_pipeline.py
                      -> vllm_reclamation_v11.py
```

The v11 modules may import CPU-safe helpers and frozen transport models from the v10 modules, but the existing v10 entrypoints continue to parse only their current config and execute only their current naive transaction. A missing or unknown `pipeline_version` must fail closed in every v11 entrypoint.

This separate-path recommendation is not cosmetic. The current v10 path is deliberately sealed in four independent places:

- `modal_gpu_reclamation.py:84-277` accepts only schema v1 and `mode="PRESERVE_NAIVE"`.
- `tools/branchfabric-experiment-004-launch.py:35-116,617-856` requires exact field sets, the v10 schema/mode, and v10-specific immutable evidence.
- `gpu_reclamation_controller.py:652-705` accepts only `execution_mode="integrated-calibration-v10"` and the v10 authorization schema.
- `gpu_reclamation_worker.py:1495-2803` hard-codes the naive conversion/verification path, while `main` at lines 2820-3051 recognizes only the v10 schema and integrated mode.

Changing those conditions in place creates a high risk of accidentally making a v10 configuration select v11. New v11 entrypoints are less code than proving every widened v10 branch remains sealed.

## Frozen source boundaries

At audit time, HEAD was analysis commit `1c51853e10809686d4368037153927f25e834117`; annotated tag `branchfabric-exp004-naive-baseline-v10` peeled to execution commit `c5fa2f169e8a343ef7a86ebc814b3a23a7de6b39`.

The following current files define the naive state path and should not be refactored for v11:

| File | Audited SHA-256 | Frozen behavior |
|---|---|---|
| `experiments/branchfabric/gpu_reclamation_worker.py` | `e6858e445ddbca912877fc67155dd9422c82d6f6fd4dcbf71871831dee15008a` | naive transaction and v10 worker protocol |
| `experiments/branchfabric/gpu_reclamation_controller.py` | `c91ccd8b3aa4b4c5a9034a5137ccc33257e3471528a96dcc2ca7920c8b7eefc9` | frozen two-GPU and integrated v10 protocol |
| `experiments/branchfabric/modal_gpu_reclamation.py` | `b62f43259fada4228611b5e82982c039f43b8b089a3fe842f29aaabfd91b713d` | v10 Modal graph and strict config |
| `tools/branchfabric-experiment-004-launch.py` | `5f694b86a4a08f2b60aebac142feb01410239805c2c784a318107122847e41c3` | v10 authorization, budget, settlement, and launch |
| `python/sloforge/continuum/adapters/vllm_reclamation.py` | `fa8f59b6e6755d0eb5c6c008d26df48b5fcbb4ad85f2af3b84233808c0ba953f` | canonical v1 transport, naive converters, and separate write/validate stager |

The recommended v11 module should reuse these stable objects without changing their semantics:

- `CanonicalKvTransportManifest`, `CanonicalBranchTable`, and `CanonicalKvPageDescriptor` at `vllm_reclamation.py:43-218`;
- `NativeKvGeometry`, `RuntimeBranchCaptureInput`, and `CanonicalCapturePlan` at lines 247-331;
- `build_canonical_capture_plan` at lines 334-464;
- `StagedBranchAllocation` and `StagedGroupImport` at lines 466-509, unless v11 adds a strictly additive evidence field in a new v11 result type;
- `validate_native_tensors` and `validate_transport_identity` at lines 573-599 and 1295-1312.

Do not call the naive converters at lines 621-944 or 957-1033 from the measured v11 path. They remain the local reference oracle.

## Exact v11 config contract

Define separate strict Pydantic models in `modal_gpu_reclamation_v11.py`; do not add fields to `Experiment004PilotConfig`.

Common fixed fields:

```text
schema_version = "sloforge.branchfabric.experiment-004-v11-modal-config/v1"
pipeline_version = "v11"
mode = "PRESERVE_OPTIMIZED"
model = "Qwen/Qwen2.5-7B-Instruct"
model_revision = tokenizer_revision = "a09a35458c702b33eeacc393d103063234e8bc28"
runtime = "vllm"
runtime_version = "0.23.0"
prefix_length = 16384
fanout = 8
suffix_length = 256
seed = explicit non-negative int < 2**63
chunk_pages in {16, 32, 64}, default 32
buffer_count = 2
integrity = "sha256-domain-separated-chunks-v1"
```

Use two exact execution models rather than one union with optional fields:

1. `Experiment004V11MicroConfig`
   - `execution_mode="state-pipeline-micro-v11"`
   - `gpu_count=1`, `requested_gpu="A100-80GB:1"`
   - no serving-rate, recovery, spike, or calibration fields
   - `tracing_level` may be `minimal` or `full`; only `minimal` is causal-timing eligible
   - bounds cover one engine load, 16K/fanout-8 preparation, export, source release, restore, 8-token continuation, recompute oracle, and cleanup
2. `Experiment004V11IntegratedConfig`
   - `execution_mode="integrated-reclamation-v11"`
   - `gpu_count=2`, `requested_gpu="A100-80GB:2"`
   - exact v10 serving values: 12/15/20 rps, 20 trigger, 64 abort, 256-token serving prompt, 64 output tokens, five-second SLO stability
   - `tracing_level="minimal"`
   - v11 authorization, v11 phase-budget, micro-validation result, and post-micro reviewer approvals are immutable hash-bound inputs

Validation rules:

- `pipeline_version` is required; no default.
- Schema, pipeline version, mode, and execution mode must agree exactly.
- Micro rejects every serving field. Integrated rejects micro-only diagnostic settings.
- Unknown fields are forbidden.
- `chunk_pages * physical_page_bytes` must stay below a configured byte bound; page count alone is not a sufficient safety check for another geometry.
- The full config is invalid unless the micro artifact reports exact 8/8 correctness, complete movement records, material physical-amplification improvement, and the required review hashes.
- The v10 launcher must reject this schema as an extra/unknown contract, and the v11 launcher must reject all v10 schema-v1 configurations.

## Modal and controller path

Create a separate Modal app name and result prefix, for example:

```text
APP_NAME = "sloforge-branchfabric-gpu-reclamation-004-v11"
REMOTE_EXPERIMENT_PREFIX = "experiment-004/v11/modal"
```

`modal_gpu_reclamation_v11.py` should register two single-use functions from the already baked `modal_real_gpu_cow.gpu_image`:

- micro: `gpu="A100-80GB:1"`, one container, no retries;
- integrated: `gpu="A100-80GB:2"`, one container, no retries.

The local entrypoint validates the config first and spawns exactly one of these functions. The app never makes a second GPU call itself. Reuse the existing model and result volumes with `create_if_missing=False`; the existing image already copies `python/` and `experiments/branchfabric/` at image-build time (`modal_real_gpu_cow.py:443-460`), so no install or model download is needed while GPUs are held.

Implement `gpu_reclamation_v11_controller.py` with CPU-only imports. It may reuse `_inventory`, `_compute_processes`, `_terminate_group`, `_wait_for_files`, `_assess_sanity_guard`, and compilation-evidence validation from `gpu_reclamation_controller.py`; it must not import torch or vLLM.

Micro controller behavior:

1. Require exactly one A100-80GB inventory row.
2. Do not run the two-device peer probe.
3. Spawn exactly one v11 worker with role `rollout` and its physical UUID.
4. Wait for engine/state readiness, issue one start command, wait for one bounded result, terminate the process group, and prove zero compute processes.
5. Record driver/CUDA/runtime/model identity and one-GPU topology. Do not manufacture P2P/NVLink evidence for a one-GPU run.

Integrated controller behavior:

1. Preserve the current two-engine readiness and 12/15-rps short guards from `gpu_reclamation_controller.py:727-968`.
2. Preserve final request reset and retained-engine continuity from lines 969-1034.
3. Bind the transaction command to v11 authorization, the exact v11 config hash, the selected-load hash, the micro result hash, and reviewer hashes.
4. Launch two v11 workers, one per physical GPU; only the rollout role selects the v11 state pipeline.
5. Preserve compilation observation, absolute deadlines, cleanup, and exact two-A100 topology checks.

The micro run cannot use the current GPU ledger unchanged: `GpuInvocationReservation`, `GpuActiveInterval`, `GpuConservativeFailureCharge`, and `GpuBudgetPreflight` are all hard-coded to `Literal[2]` in `gpu_reclamation_methodology.py:579-654,749-796`. The minimal backward-compatible shared infrastructure change is to permit `gpu_count in {1,2}`, require `len(actual_gpu_models) == len(gpu_uuids) == gpu_count`, pass `gpu_count` explicitly to reserve/settle/failure-charge operations, and retain a default of 2 for every existing v10 caller. Add tests proving serialized v10 records and v10 arithmetic are unchanged. Use the same `gpu-hours.lock` for both launchers and make the canonical cumulative ledger include v11 consumption; a disconnected v11-only ledger would allow later v10 work to overspend the configured cumulative authorization.

## Worker execution modes

Create `gpu_reclamation_v11_worker.py` rather than inserting branches throughout `_run_rollout_transaction`.

It may import the following non-state helpers from the frozen worker: `_create_adapter`, `_prepare_rollouts`, `_runtime_capture_inputs`, `_add_restore_requests`, `_validate_exact_source_release`, `_sampling_params`, `_CacheSaltEngine`, `_run_raw_serving`, `_run_integrated_calibration_phase`, `_run_measured_v10_transaction`, `_stage_row`, `_causalize_stages`, and artifact helpers. The v11 worker must own config validation, telemetry GPU-count validation, result schemas, and state-pipeline selection.

Micro execution sequence:

```text
load one warmed engine
prepare the exact 16K/fanout-8 rollout
quiesce at a token boundary
capture plan + source allocation evidence
v11 bounded export + streamed integrity
release exact source blocks and confirm allocator state
create fresh restore request incarnations
one pre-restore retained-payload verification
fused bounded H2D + native write + native validation
atomic admission
8 continuation tokens on all 8 branches
independent greedy recompute first-token oracle
movement/memory/latency artifacts
cleanup
```

There is no serving role, spike, reclaim trigger, queue-drain gate, or SLO claim in micro mode. Its result schema must say `causal_serving_evidence=false` and cannot be passed to the v11 scientific-validity postprocessor as a full transaction.

Integrated execution uses the same serving/reclamation orchestration as v10, but delegates capture/restore to `run_v11_pipeline_transaction`. Its result includes `pipeline_version="v11"`; postprocessing must reject a result whose config, worker result, movement report, or pipeline evidence disagrees on that value.

## Fused write-and-validate stager contract

The v10 stager at `vllm_reclamation.py:1133-1286` calls `state.verify()` and then invokes separate `write_pages` and `validate_pages` callbacks twice. Do not add a skip-verification boolean to that API. A boolean would make it possible to bypass v10 correctness and would not prove which transport was verified.

Add `Vllm0230StreamingRestoreStager` in `vllm_reclamation_v11.py`. It can subclass `Vllm0230RestoreStager` only to reuse `__init__`, `detach_new_request`, and `_allocate`; it implements a new import method and never calls the v10 `import_group`.

Recommended signature:

```python
def import_verified_group(
    self,
    verified: VerifiedV11Transport,
    *,
    runtime_request_ids: dict[str, str],
    expected_identity: dict[str, str],
    write_and_validate_pages: Callable[
        [dict[str, int]], V11SubsetWriteValidationEvidence
    ],
    allocation_observer: Callable[[str, int, int], None] | None = None,
) -> V11StagedGroupImport:
    ...
```

`VerifiedV11Transport` is returned only by the one pre-restore verifier. It binds the exact manifest commitment, payload digest, payload storage identity/size, verification timestamp, and complete chunk coverage. The stager rechecks small metadata and `validate_transport_identity`, but does not rescan payload bytes.

`write_and_validate_pages` has these semantics:

- Input maps a disjoint logical-page subset to fresh native block IDs.
- The callback may internally overlap H2D for chunk N+1 with write/validation for chunk N, but it must not return before every write, tail zero, device validation, and relevant CUDA event for the subset has completed.
- Success returns a strict evidence object, not `bool` and not an arbitrary truthy value.
- Evidence contains manifest-ordered page IDs, exact destination bindings or their canonical SHA-256, chunk IDs, logical bytes written/validated, tail bytes zeroed/validated, mismatch count fixed to zero, and completion time/event provenance.
- The stager verifies evidence coverage equals the input mapping exactly and that no page or chunk was duplicated.
- A mismatch raises and persists a diagnostic before returning; the stager never treats diagnostic recapture as optimized success.
- The worker updates `imported_subsets` only after successful fused evidence, unlike the v10 write callback at `gpu_reclamation_worker.py:2233-2234`, which can mark a subset before the separate validator runs.

Preserve the exact v10 admission order:

```text
allocate first branch
fused write+validate shared-root/branch-0 subset
publish validated root through cache_blocks(first request)
allocate remaining branches using only that validated root
fused write+validate remaining private subsets
assert zero queue empty and full page coverage bijective
publish remaining hashes
set num_computed_tokens
enqueue the complete detached group while the engine-step gate is held
```

On any exception, preserve the v10 cleanup at lines 1259-1281: remove every restore incarnation from both waiting queues, free all tables, remove scheduler request objects, reset the prefix cache, and re-raise.

## v11 movement ledger labels

Emit one record per real chunk/segment event. Use a label suffix with a zero-padded deterministic chunk sequence; never infer bytes later from logical state size. When two records describe two reads inside one fused CUDA operation, give them the same operation/timing-group ID but distinct byte-event IDs so timings are not summed and bytes are not deduplicated.

| Stable label prefix | Operation | Exact accounted state bytes | Required |
|---|---|---|---|
| `v11-capture-native-gather-chunk` | `FUSED_NATIVE_TO_HOST` or `TRANSFORM` | native source bytes actually read; bounded GPU chunk bytes written | yes |
| `v11-capture-d2h-chunk` | `D2H` | GPU chunk read, pinned checkpoint view written, one D2H link event | yes |
| `v11-capture-integrity-chunk-read` | `CHECKSUM` | host bytes actually read by chunk/page SHA-256; no fictitious payload write | yes |
| `v11-capture-fence-before-source-release` | synchronization sidecar, not a movement record unless bytes are touched | zero bytes; all D2H/hash completion events | yes |
| `v11-restore-preflight-integrity-read` | `CHECKSUM` | retained pinned payload bytes read exactly once before allocation/H2D | yes |
| `v11-restore-h2d-chunk` | `H2D` | pinned checkpoint read, GPU chunk written, one H2D link event | yes |
| `v11-restore-native-write-chunk` | `FUSED_HOST_TO_NATIVE` or `WRITE` | expected GPU chunk bytes read and logical destination native bytes written | yes |
| `v11-restore-tail-zero-chunk` | `WRITE` | only invalid-tail destination bytes written; zero for the measured all-full-page state | yes/security |
| `v11-restore-validation-native-read-chunk` | `VALIDATE` | logical destination native bytes read | yes |
| `v11-restore-validation-expected-read-chunk` | `VALIDATE` | expected GPU chunk bytes read only if exact device compare is implemented; omit for a validated logical SHA-256 comparison | implementation-dependent |
| `v11-restore-tail-validation-read-chunk` | `VALIDATE` | invalid-tail destination bytes read; zero for the measured state | yes/security when tails exist |
| `v11-restore-subset-fence-before-publish` | synchronization sidecar | zero bytes; callback completion event | yes |
| `v11-diagnostic-recapture-*` | corresponding read/repack/D2H operations | actual bytes only after a mismatch | diagnostic-only; makes the run fail |

Do not emit the following v10 labels on a successful v11 path: `capture-native-axis-contiguous`, `capture-unpage-valid-tokens`, `capture-stack-layers`, `capture-concatenate-pages`, `transport-publish-validation`, `transport-publish-hash-reads`, `restore-import-validation`, `restore-import-hash-reads`, `restore-zero-native-pages` for full pages, `restore-native-axis-contiguous`, `restore-stack-native-pages`, or any normal `validation-d2h`/host comparison label.

The persistent pinned checkpoint is retained state, not pipeline-temporary memory. Report it separately as `checkpoint_resident_host_bytes`. Only the two bounded GPU slots, any bounded pinned staging slots, and operation-local scratch count toward `pipeline_temporary_bytes`. If the existing v1 movement schema cannot express synchronization, representations, executor, CUDA time, and fused-chain membership, retain the v1 byte records unchanged and emit an enriched v11 sidecar keyed by `record_id`; do not overload byte fields or put zero-byte pseudo-passes into `StatePassRecord` (the current validator rejects them at `gpu_reclamation_accounting.py:246-255`).

Report both:

- raw `StatePassRecord` count (chunk/segment records), and
- `full_state_equivalent_pass_count`, derived only when a deterministic label/timing group has gap-free, non-overlapping coverage of the complete logical state.

Without the second value, chunking would misleadingly appear to increase the v11 full-state pass count.

## Required tests before any GPU allocation

Add new test modules; do not rewrite the existing v10 tests.

`tests/python/test_vllm_reclamation_v11.py`:

- v10 reference export versus v11 export: identical canonical payload and v1 manifest for seeds 41, 73, and 113;
- v10 versus v11 restore logical equality with randomized page order, block dimension, layers, full pages, partial pages, K/V, and shared/private splits;
- complete-page overwrite skips zeroing; partial tails begin as tenant-distinct nonzero sentinels and become zero;
- fused callback sequence is `write+validate subset0 -> cache root -> allocate remainder -> write+validate subset1 -> cache/admit`;
- false/forged/incomplete evidence is rejected;
- callback mismatch, allocation failure, cross-page alias, zero-queue pollution, and partial enqueue failure clean the entire group;
- `state.verify()` is not called inside `import_verified_group`; a verified-transport token is mandatory;
- corrupt, missing, duplicate, or reordered chunks/pages and wrong branch/token/model/policy identity fail before admission.

`tests/python/test_gpu_reclamation_v11_worker.py`:

- `pipeline_version` is mandatory and equals `v11` in config, result, transport evidence, and movement report;
- micro mode never enters serving/spike/barrier code;
- integrated mode uses the v11 pipeline only for rollout and preserves GPU0 serving flow;
- imported-page coverage updates only after fused validation success;
- every chunk record has exact ranges, unique event IDs, bounded allocations, and a fused-chain/timing sidecar;
- diagnostic recapture exists only on failure and invalidates optimized success;
- source release waits for all capture completion events; publication waits for all subset validation events.

`tests/python/test_experiment_004_v11_gpu_harness.py`:

- Modal graph registers exactly one-A100 micro and two-A100 integrated functions, both single-use/no retry;
- selecting one execution mode spawns exactly one corresponding function;
- one-GPU budget reserve/settlement arithmetic is exact and cumulative with v10 usage;
- v10 configs still validate byte-for-byte through the v10 launcher and are rejected by the v11 launcher;
- v11 configs are rejected by the v10 launcher;
- micro cannot carry serving fields; integrated cannot run without hash-bound micro/review/budget evidence;
- controller imports remain CUDA-clean and cleanup includes descendant/profiler processes.

Keep all current v10 tests, especially:

- separate write/validate stager semantics and cleanup in `test_vllm_reclamation_adapter.py:356-619`;
- movement byte/link uniqueness in `test_gpu_reclamation_worker_movement.py:110-210`;
- strict v10 config and exact two-A100 Modal graph in `test_experiment_004_gpu_harness.py:365-607`;
- retained-engine and transaction-command behavior in `test_gpu_reclamation_worker_integrated.py:24-309`.

## Integration gates and stop conditions

Before micro allocation, require local property tests, exact reference equivalence, config/authorization hashes, `v11-gpu-budget.json`, and no active reservation. Micro success requires real A100 export/restore, exact 8/8 continuation, movement records, bounded memory, and material amplification improvement. A mismatch or absent evidence fails closed; it does not fall back to v10 and report v11 success.

Only after the post-micro movement/CUDA/Continuum reviews approve the immutable micro artifact may the full two-GPU v11 config become valid. The controller must not wait for reviews while GPUs are allocated.

## High-severity integration findings

1. **A `pipeline_version` field added to the current v10 config would break its exact-field contract and weaken isolation.** Use a new schema and parser.
2. **A `skip_verify=True` option on `Vllm0230RestoreStager.import_group` would expose an unsafe correctness bypass.** Require a typed preverified transport in a v11-only method.
3. **Keeping separate v11 write and validate callbacks either retains full subsets or retransfers them.** Use one synchronous-at-return fused callback.
4. **The current GPU ledger cannot truthfully reserve or settle one GPU.** Generalize count with default 2 and exact-cardinality checks before micro launch.
5. **Chunk record count is not full-state pass count.** Publish both raw records and gap-free full-state-equivalent groups.
6. **A persistent checkpoint is not temporary pipeline memory.** Separate checkpoint residency from bounded scratch or the claimed memory reduction will be misleading.
7. **Do not reuse the v10 authorization artifact as v11 code authorization.** Reuse its measured load evidence by hash, but seal a v11 authorization against the v11 commit and micro/review artifacts.

## Final audit verdict

`APPROVE_IMPLEMENTATION_WITH_ISOLATED_V11_PATH`

The proposed path preserves v10 behavior, supplies a real one-GPU micro mode, removes duplicate verification without creating a bypass, keeps atomic vLLM visibility semantics, and provides unambiguous per-chunk movement evidence. GPU execution remains blocked until the new local tests, budget artifact, and required pre-GPU reviews pass.
