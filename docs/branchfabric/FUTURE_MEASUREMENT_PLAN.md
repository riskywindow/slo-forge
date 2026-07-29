# BranchFabric future measurement plan

## Decision boundary

The current gate is `FAIL_NO_BUILD`. This plan does not authorize GPU, multi-GPU,
multi-node, RTL, FPGA, DPU, or external-resource use. It describes the minimum
new evidence that could justify rerunning the gate. A larger synthetic fixture,
CPU projection, simulator, HLS estimate, or isolated copy benchmark cannot do so.

The next campaign must begin from the optimized software paths measured in this
execution: shared-root/COW/lazy fork, bounded parallel checkpoint/transform/resume,
one batched transfer of the unique
content-addressed chunk set, direct or staged canonical transform selected by
measured workload, adaptive transfer chunks, and whole-buffer vectorized SHA-256.

## Authorization and budget preflight

All listed values default to false or zero. Stop before provisioning unless the
requested resource and spend are explicitly authorized.

```sh
test "${SLOFORGE_BRANCHFABRIC_ALLOW_EXTERNAL_RESOURCES:-false}" = true
test "${SLOFORGE_BRANCHFABRIC_ALLOW_GPU:-false}" = true
test -n "${SLOFORGE_BRANCHFABRIC_TARGET:-}"
test "${SLOFORGE_BRANCHFABRIC_GPU_BUDGET_USD:-0}" != 0
```

Multi-GPU, multi-node, RTL, and FPGA runs additionally require their matching
flags:

```sh
test "${SLOFORGE_BRANCHFABRIC_ALLOW_MULTI_GPU:-false}" = true
test "${SLOFORGE_BRANCHFABRIC_ALLOW_MULTI_NODE:-false}" = true
test "${SLOFORGE_BRANCHFABRIC_ALLOW_RTL:-false}" = true
test "${SLOFORGE_BRANCHFABRIC_ALLOW_FPGA_BUILD:-false}" = true
test "${SLOFORGE_BRANCHFABRIC_ACCELERATOR_BUDGET_USD:-0}" != 0
```

Keep at least 15% of either authorized budget unused until independent
replication. Record provider, instance, region, start/stop time, hourly price,
and accrued spend before and after every run. The commands below consume an
already-authorized allocation; none provisions paid resources.

## Stage 1: capability and existing bounded harness

On an already allocated compatible CUDA host, capture the hardware/software
manifest first. The existing characterization gate uses its historical alias,
so set it only after the new execution-level authorization has passed:

```sh
test "${SLOFORGE_BRANCHFABRIC_ALLOW_GPU:-false}" = true
export SLOFORGE_BRANCHFABRIC_CHARACTERIZATION_ALLOW_GPU=1
nvidia-smi -q > artifacts/branchfabric/execution/hardware/nvidia-smi-q.txt
nvidia-smi topo -m > artifacts/branchfabric/execution/hardware/nvidia-topology.txt
uv run --locked sloforge helix characterize run \
  --matrix benchmarks/branchfabric/characterization.yaml \
  --output artifacts/branchfabric/characterization/hardware/gpu-seed-41 \
  --hardware gpu --seed 41 --max-experiments 100000 --timeout-seconds 3600
```

Repeat with seeds 73 and 113 into distinct, immutable output directories. This
existing harness is only a capability and regression step. It does not provide
the missing real-transformer, high-fanout, or causal GPU-reclamation evidence
and cannot by itself pass the gate.

## Stage 2: real runtime integration required before measurement

Add a bounded, feature-flagged vLLM or SGLang adapter that emits exact
StateOperationTrace events and distinguishes logical state from allocator,
resident HBM, pinned-host, and transport bytes. The adapter must reject a
requested engine/device mismatch instead of falling back. Its manifest must
bind model revision, tokenizer, runtime revision, CUDA/driver/NCCL versions,
GPU UUIDs, topology, page/block size, state layout hashes, and profiler version.

Use the smallest real model that exercises the deployed state path. For each of
coding-agent and reasoning/verification workloads, run contexts 8K, 16K, and
32K, suffix/divergence variants, fanout 8, 16, and 32, and 64 when budget allows.
The campaign command contract must be:

```sh
uv run --locked sloforge helix branchfabric-real-fanout \
  --engine vllm --model-revision "$MODEL_REVISION" \
  --workload benchmarks/branchfabric/real-workloads.json \
  --contexts 8192,16384,32768 --fanouts 8,16,32 \
  --seed 41 --repetitions 5 --trace-levels disabled,minimal,full \
  --output artifacts/branchfabric/execution/hardware/fanout-seed-41
```

That subcommand is intentionally not present today; implementing and testing
the real-runtime measurement adapter is a prerequisite, not evidence that may
be replaced with the CPU reference CLI.

Collect Nsight Systems/CUPTI counters for GPU utilization, HBM and copy-engine
activity, PCIe, launch overhead, and inference interference. With two or more
authorized GPUs, record CUDA peer-access, NCCL, PCIe, and NVLink paths
separately; do not infer one from another. With authorized nodes, record actual
TCP/RDMA/NIC delivery and identical-state repeated unicast by destination.

## Stage 3: causal physical-GPU reclamation

Implement a real-runtime variant of the bounded local transaction, then run one
causal sequence: serving plus rollout load, exact BranchPoints, eight or more
long-lived siblings, injected serving spike, admission stop, pause/checkpoint,
physical GPU release to serving, state movement/preservation, serving-SLO
restoration, ownership commit, and rollout resume. The required future command
contract is:

```sh
uv run --locked sloforge helix branchfabric-real-reclamation \
  --engine vllm --serving-workload benchmarks/branchfabric/serving-spike.json \
  --rollout-workload benchmarks/branchfabric/real-workloads.json \
  --siblings 8 --seed 41 --repetitions 5 \
  --trace-levels disabled,minimal,full \
  --output artifacts/branchfabric/execution/hardware/reclamation-seed-41
```

Repeat for seeds 73 and 113 in randomized order. Preserve raw timing, GPU
allocation, work-lost/work-preserved, ownership, integrity, and fault receipts.
The destination GPU must actually become available to serving; a logical CPU
scheduler unit is not a GPU-reclamation measurement.

## Stage 4: evidence and gate acceptance

An independent reviewer must rerun at least five central measurements from
fresh allocations. Each candidate metric must bind its raw paths, sample
selectors, workload classes, seeds, derivation, confidence level, and
target-hardware timing provenance in
`artifacts/branchfabric/gates/branchfabric_gate_input.json`.

Then run:

```sh
make branchfabric-gate
uv run --locked pytest -q tests/python/test_branchfabric_gates.py
```

Do not start a functional hardware model, cycle simulator, target selection,
driver, RTL, HLS, FPGA build, or DPU implementation unless the regenerated
artifact reports all of:

- `outcome: PASS`;
- `hardware_implementation_allowed: true`;
- `functional_model_or_cycle_simulator_allowed: true`;
- every mandatory gate and at least one workload-value gate `PASS` for the
  same candidate.

If the result remains `FAIL_NO_BUILD`, terminate the hardware path again. Do
not loosen thresholds, relabel CPU/model evidence, or count an ideal zero-cost
operation bound as calibrated hardware headroom.
