# BranchFabric Experiment 004 v11 integrated result final

Status: `HARDWARE_GATE_NOT_REACHED`

The allocator mismatch is offline-classified as `GATE_BUG` and the corrected
snapshot-time semantic identity pipeline passes the full local verification
matrix. The old gate equated an epoch-free retained allocation-notification
history with the current live source allocation set. The first causal history
event was the first KV allocation for `readiness-gpu1-00000`, before rollout
creation.

Targeted attempt `exp004-v11-targeted-identity-s41-b` exercised the exact real
retained-engine lifecycle and passed. At backlog 21, with 43 requests of trigger
headroom, allocator quiescence was established and a `SourceCaptureCommit` was
created under the production engine-step binding. All 1,152 allocations passed
logical mapping, block plus allocator epoch, owners, refcounts, device, and live
status. There were zero post-commit allocator events. The targeted run stopped
before export and release as required, and both cleanup gates passed.

The one authorized full retry, `exp004-v11-integrated-s41-d`, did not enter the
remote function. The local Modal client lost its heartbeat connection while the
image was being built, after which the provider stopped the app and reported an
externally terminated image build. There is no function call ID, controller,
worker, CUDA context, allocator lifecycle, or integrated result bundle for D.
The attempt was conservatively charged 1,176 A100-seconds, leaving zero active
reservations and 2,202.024986768025 authorized A100-seconds.

Consequently there is no scientifically valid integrated v11 measurement.
Micro and targeted values are not substituted for integrated values. HBM
reclamation, useful GPU1 capacity, SLO restoration, restore latency, movement
accounting, residual shares, and branch continuation remain unmeasured for the
corrected integrated pipeline. The exactly-once retry is consumed, v11 is not
frozen, and kill/recompute remains forbidden.

Evidence:

- [root cause](V11_SOURCE_ALLOCATION_ROOT_CAUSE.md)
- [source invariant](V11_SOURCE_IDENTITY_INVARIANT.md)
- [targeted PASS](../../artifacts/branchfabric/gpu-validation/experiment-004/v11-final/targeted-repro/status-attempt-b.json)
- [integrated attempt-D status](../../artifacts/branchfabric/gpu-validation/experiment-004/v11-final/integrated/status-attempt-d.json)
- [integrated attempt-D failure](../../artifacts/branchfabric/gpu-validation/experiment-004/v11-final/integrated/failures/exp004-v11-integrated-s41-d-conservative-charge.json)
- [local verification](../../artifacts/branchfabric/gpu-validation/experiment-004/v11-final/source-identity/local-tests.json)
- [post-fix hardware classification](../../artifacts/branchfabric/gpu-validation/experiment-004/v11-final/hardware-gate/final-classification-post-fix.json)
