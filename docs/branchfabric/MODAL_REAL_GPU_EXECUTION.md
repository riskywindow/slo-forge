# BranchFabric real-GPU execution on Modal

## Scope and status

BranchFabric GPU Validation Experiment 002 uses Modal exclusively for real-GPU execution. It
must not SSH to `mew0`. The implementation is
`experiments/branchfabric/modal_real_gpu_cow.py`; it defines an ephemeral app named
`sloforge-branchfabric-real-gpu-cow`, a CPU-only model-preparation function, and one
single-container GPU function that executes one typed benchmark configuration.

The run was authorized with `SLOFORGE_GPU_BUDGET_USD=40`. Five bounded smoke attempts
allocated one exact `A100-80GB` each; only v5 completed the benchmark workload. Cumulative
accounting is 0.123306008368 A100-hours and $1.11371127858 of conservative cloud cost, with no
in-flight model or GPU reservation after result materialization.

V5 ran on `NVIDIA A100 80GB PCIe` UUID
`GPU-652804ee-42ae-7c14-21ef-b8d29adacc11`. Its 2,048-token, two-branch workload demonstrated
128 identical physical vLLM KV blocks shared at native refcount two, zero additional prefix
bytes at metadata fork, and one 917,504-byte private suffix page per branch. Exact-ID evidence
also showed branch-private and final-root KV allocator release.

The invocation nevertheless failed its mandatory cleanup postflight: driver-visible memory was
14,213 MiB versus a 4 MiB preflight baseline, held by the Function's PID 1 after in-process
engine cleanup. This is retained model/CUDA-graph state, distinct from the experiment KV pages,
whose assigned bytes were zero. No 16K Function was admitted. The next implementation must run
the engine in an exactly tracked child process or otherwise prove complete in-process teardown,
then rerun only the smoke without weakening the 1-GiB recovery gate.

The two explicitly allowed persistent resources contain the pinned model and immutable,
hash-verified attempt artifacts:

- `sloforge-model-cache` for one pinned model snapshot and its content manifest;
- `sloforge-branchfabric-results` for immutable completed result prefixes.

There is no deployed web endpoint, scheduled function, autoscaled benchmark worker, or
long-lived runtime server. Modal Function containers were single-use; the result does not infer
cleanup success from container ephemerality because the explicit v5 HBM postflight failed.

## Official API audit

The code is fail-closed on `modal==1.5.3`, the current official SDK release inspected on
2026-08-10. The following current official interfaces were checked before implementation:

- [Modal SDK changelog](https://modal.com/docs/sdk/py/changelog): 1.5.3 and programmatic
  FunctionCall log retrieval;
- [GPU guide](https://modal.com/docs/guide/gpu): exact string `gpu="A100-80GB"`; no fallback
  list and no `gpu="any"`;
- [App API](https://modal.com/docs/sdk/py/latest/modal.App): `App.function` resource,
  timeout, retry, and container-count arguments;
- [Image API](https://modal.com/docs/reference/modal.Image): `from_registry`,
  `uv_pip_install`, `apt_install`, `entrypoint`, and `add_local_dir(copy=True)`;
- [Volume guide](https://modal.com/docs/guide/volumes): per-mount read-only options,
  explicit `commit`, and programmatic reads;
- [model-weight guide](https://modal.com/docs/guide/model-weights): CPU-side pinned
  `snapshot_download` into a Volume before GPU allocation;
- [FunctionCall API](https://modal.com/docs/sdk/py/latest/FunctionCall): `spawn`, durable
  `object_id`, bounded `get`, `cancel(terminate_containers=True)`, and `logs.fetch`;
- [resource pricing](https://modal.com/pricing): A100-80GB, physical CPU-core, memory, and
  Volume rates used by the pre-launch cost ceiling.

The locally installed 1.5.3 signatures were also inspected directly. Unsupported SDK
versions fail before any remote call.

## Pinned environment

The GPU Image begins with
`nvidia/cuda:13.0.2-cudnn-runtime-ubuntu24.04`, adds Python 3.12, and installs normal
dependencies while the Image is built. It pins `uv==0.10.2`, `torch==2.11.0`,
`transformers==5.14.1`, `vllm==0.23.0`, `pydantic==2.13.4`, `psutil==7.2.2`, and
`nvidia-ml-py==13.610.43`. No benchmark function runs pip or downloads model files.
PyTorch's profiler/CUPTI support, NVML bindings, and process accounting are present, but the
current runner does not claim copy-engine utilization or attributable HBM bandwidth without
actual profiler counters.

The selected open model is `Qwen/Qwen2.5-7B-Instruct` at immutable model and tokenizer
revision `a09a35458c702b33eeacc393d103063234e8bc28`. It supports the 16K workload and needs
no Hugging Face credential. If a future model needs authentication, it must use a Modal
Secret; plaintext tokens are forbidden.

## Model preparation and validation

The CPU-only function downloads the exact snapshot, constructs a deterministic 16,384-token
prefix and 32 distinct valid divergence tokens, hashes every model/tokenizer/input file, and
commits the final snapshot to `sloforge-model-cache`. A GPU function mounts this Volume
read-only, revalidates every recorded size and SHA-256, and operates offline. An incomplete
CPU staging directory is removed only by the owning preparation path before retry.

## GPU function contract

`ModalBenchmarkConfig` binds the model, runtime, fanout, prefix length, suffix length, seed,
baseline mode, trace level, timeouts, and sampler interval. The function requests exactly
one `A100-80GB`, has retries disabled, caps the app at one benchmark container, uses a
single-use container, and writes hot traces under `/tmp`. It records:

- FunctionCall ID and function-body/model-ready/benchmark/end timestamps;
- exact `nvidia-smi` GPU name, UUID, driver, HBM, utilization, and process inventory;
- Python, CUDA userspace, PyTorch, vLLM, transformers, Modal, and instrumentation versions;
- vLLM model-initialization, memory-profile, dummy-execution, and CUDA-graph warmup time;
- raw live-KV layouts, block/refcount/release evidence, GPU samples, state traces, and trial
  summary from the real-runtime adapter;
- exact child-process termination and a postflight HBM/process recovery assertion.

High-frequency files remain container-local until the trial ends. The function fsyncs them,
copies them to a unique staging prefix in `sloforge-branchfabric-results`, generates an
inventory, atomically renames the prefix, explicitly commits the Volume, and returns the
manifest hash. Local materialization anchors the downloaded manifest to that returned hash
and verifies every byte before promotion into Experiment 002.

Modal allocates the exact requested GPU family, but separate Function calls are not promised
the same physical UUID. Every trial records the UUID. Derived paired results must disclose
this limitation and may not say “same physical GPU” unless the recorded UUIDs match.

## Adaptive and cost gates

The coordinator permits only this order:

1. 2K/4K shared-root smoke, fanout 2, suffix 16;
2. same-seed 16K fanout-1 independent and shared paths;
3. same-seed 16K fanout-8 only after both fanout-1 paths succeed;
4. fanout-32 only after both fanout-8 paths and a local, explicit HBM/invariant/GPU-budget
   approval artifact.

There is deliberately no checked-in fanout-32 launch configuration. A failed returned trial
raises after its artifacts and accounting are materialized, so automation cannot silently
continue.

Before a GPU spawn, the local coordinator reserves the provider-enforced worst case of the
function timeout plus startup timeout. The ledger includes in-flight reservations. A failed,
timed-out, disconnected, or untracked terminal call is conservatively charged the entire
reservation; successful calls use a client-elapsed upper bound, lower-bounded by the
container observation. CPU and memory charges for model preparation and GPU support are
included in the dollar ceiling. GPU work cannot exceed four A100-equivalent hours, and the
15% budget reserve is mandatory.

The local dollar ledger is a conservative compute ceiling for declared GPU, CPU, and memory.
Persistent Volume storage is governed separately by the Modal workspace budget: current
official pricing includes 1 TiB/month free and the two experiment Volumes are expected to
remain far below it, but the experiment does not freeze future prices or regional multipliers.
No paid invocation is allowed unless both the external workspace budget and the local
`SLOFORGE_GPU_BUDGET_USD` ceiling are acceptable to the operator.

## Explicit invocation

These commands are intentionally not executed by generation or tests. The budget-first
launcher imports no Modal code and projects the full model-preparation and GPU reservation
against the dollar guard, its 15% reserve, the durable in-flight reservations, and the
four-GPU-hour ledger before it can start `modal run`; the Modal entrypoint repeats the detailed
ledger/cost checks.
Remote paid functions also require a config-bound, reservation-bound authorization payload.
Without the launcher's one-use preflight token, importing the experiment module registers no
Modal Functions or local entrypoint at all—the GPU body remains a plain Python function with
no `.spawn`/`.remote` surface. This prevents accidental direct paid invocation through the
module; the workspace's administrative access controls remain the security boundary against a
deliberate operator rewriting or bypassing repository code.

```bash
export SLOFORGE_GPU_BUDGET_USD=<authorized-total-usd>
uv run --locked python tools/branchfabric-modal-launch.py \
  --action prepare-model \
  --config-path experiments/branchfabric/configs/modal-smoke-shared-s41-v1.json
uv run --locked python tools/branchfabric-modal-launch.py \
  --action run \
  --config-path experiments/branchfabric/configs/modal-smoke-shared-s41-v1.json
```

The launcher uses Modal's detached App mode so a client disconnect does not silently kill an
accepted Function. FunctionCall IDs and reservations are durable locally. `--action recover`
reattaches with `FunctionCall.from_id`, finalizes conservative accounting, and materializes a
completed result; `--action cancel` cancels the exact call and consumes its worst-case
reservation. Ambiguous spawn or unconfirmed cancellation leaves its durable reservation in
place and prevents a subsequent paid call; elapsed wall time alone is not terminal evidence
because Modal queue time is outside the Function timeout. `--action cleanup-staging` removes
only the exact incomplete result staging prefix after first proving no final result and no
matching in-flight model or GPU reservation exists.

If an ambiguous spawn never yielded a FunctionCall ID or an attempt-bound final publication,
the reservation remains fail-closed. It may be reconciled only after an operator records a
strict `modal-provider-terminal-audit/v1` artifact under Experiment 002's
`modal/provider-audits/` directory proving the provider has no live or pending call. The
`reconcile-unattached` action binds that audit to the exact attempt/reservation, stores its
SHA-256 in the terminal ledger interval, and applies the full conservative charge.

Only the two named Volumes persist. The model cache is not deleted after a run, by explicit
experiment requirement. Completed result prefixes remain a remote cache after verified local
materialization. All other files, CUDA contexts, and processes disappear with the ephemeral
single-use Function container; no host-level cleanup or SSH audit applies.
