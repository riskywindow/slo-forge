# BranchFabric Experiment 004 v11 integrated final

Final classification: `HARDWARE_GATE_NOT_REACHED`.

The offline early-trigger, in-function cleanup, and classification-policy fixes
all passed, followed by a full `make check` pass. Exactly one integrated v11
Modal invocation ran on two distinct NVIDIA A100-SXM4-80GB devices. The 12-rps
stability guard, 15-rps overload guard, engine-readiness gate, and five-second
9-rps control interval passed.

The early trigger worked: it emitted at backlog 21, leaving 43 requests of
headroom below the independent depth-64 abort. Predicate-to-emission latency was
0.29008 ms and rollout admission stopped 20.103482 ms after emission.

The transaction validated the 1,056,964,608-byte logical topology and completed
a transient source export, then failed before reclamation. GPU1's production
allocation zero-queue did not exactly match the authenticated capture plan,
causing a `V11RuntimeTeardownRequired`. Retained-engine readiness and calibration
allocations had not been epoch-separated or drained; resetting the prefix cache
did not reset that production allocation history. Source release, HBM
reclamation, useful GPU1
capacity, SLO restoration, rollout restore, integrated StatePass accounting,
and branch-resume correctness were therefore not reached. With no useful GPU1
capacity, the serving queue later reached the independent depth-64 abort.

The integrated run is scientifically invalid. v11 is not frozen. Under the
precommitted dependency, kill/recompute was not run. Measured break-even,
integrated residual Amdahl headroom, a realistic accelerator projection, and a
decisive CUDA/Triton comparison are unavailable. Micro-validation numbers are
not substituted.

In-function cleanup itself passed: the full lifecycle completed, every owned
child was reaped, no process groups, zombies, profilers, workers, IPC resources,
or CUDA compute processes survived, and no SIGKILL was needed. Provider
postflight also passed with zero active apps, tasks, containers, endpoints,
queues, dictionaries, or ledger reservations.

The completed FunctionCall records 436.059700712 actual A100-seconds
(0.1211276946 A100-hours, $0.3026254323 at the recorded $2.4984/A100-hour).
Because post-return validation initially failed, the ledger conservatively
charged the full 1,176 A100-seconds (0.3266666667 A100-hours, $0.816144); the
read-only accounting recovery does not rewrite that conservative settlement.
