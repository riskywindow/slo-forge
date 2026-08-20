# v11 Integrated Result

Status: **scientifically invalid**
Classification consequence: `HARDWARE_GATE_NOT_REACHED`

Exactly one integrated v11 attempt (`exp004-v11-integrated-s41-c`) ran on two
distinct NVIDIA A100-SXM4-80GB GPUs. Both engines became ready. The bounded
12-rps stable guard, 15-rps overload guard, and five-second 9-rps control
interval passed without measured-transaction compilation.

The corrected overload controller emitted the normal trigger at queue depth 21,
leaving 43 requests below the independent emergency ceiling of 64. The trigger
followed predicate satisfaction by 0.29008 ms, and rollout admission stopped
20.103482 ms after the trigger.

The rollout side validated the eight-branch, 1,056,964,608-byte logical topology,
performed the pre-release ownership/refcount/epoch proof, and completed a
transient v11 source export. It then failed the destructive production
allocation zero-queue identity check: the drained allocation history did not
equal the current authenticated 1,152-block capture plan. Retained-engine
readiness and calibration allocations had not been epoch-separated or drained;
prefix-cache reset alone did not reset this allocator queue.

The fail-closed teardown happened before source release. Consequently there is
no valid integrated HBM-reclamation, useful-GPU1, SLO-restoration, restore,
movement-amplification, critical-path, or branch-resume measurement. The queue
later reached the independent depth-64 abort because GPU1 never became useful.
Micro-validation values are not substituted for these missing integrated data.

v11 is not frozen. The gated kill/recompute arm was not run.

In-function cleanup passed independently of the scientific failure: all owned
children were reaped, process groups were empty, CUDA-owning processes were
gone, IPC and pipes were released, and no forced kill was required. Modal
postflight also found zero active apps, tasks, containers, endpoints, queues,
dictionaries, or GPU-ledger reservations.

Primary evidence is under
`artifacts/branchfabric/gpu-validation/experiment-004/v11-final/integrated/`.
