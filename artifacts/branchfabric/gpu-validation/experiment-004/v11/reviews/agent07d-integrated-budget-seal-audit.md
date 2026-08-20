# Agent 07d integrated-v11 budget and seal audit

Status: **PASS — approve exactly one integrated v11 transaction.**

No GPU, Modal, or network operation was performed by this reviewer. The approval is for the designated root GPU coordinator only and remains subject to atomic reservation, settlement, and postflight cleanup.

## Budget decision

The authoritative ledger is schema-valid, has no active reservation, and is sealed by raw-file SHA-256 `be29bc61ffd9c6df62e8c0dc9236308674d0c4c0f6d40bb746202ebff272988f`.

- Authorized ceiling: 21,600 GPU-seconds = 6.0 A100-hours; authorized paid budget: $40.
- Accounted use: 13,761.513626085978 GPU-seconds = 3.822642673912772 A100-hours.
- Remaining before reservation: 7,838.486373914022 GPU-seconds.
- Exact plan: 2 A100-80GB GPUs × 588 seconds = 1,176 GPU-seconds.
- Fifteen-percent planning projection: 1,352.4 GPU-seconds.
- Projected cumulative use with margin: 15,113.913626085978 GPU-seconds.
- Remaining after margin: 6,486.086373914022 GPU-seconds.
- Exact reservation cost at $2.4984/A100-hour: $0.816144; margin-inclusive planning cost: $0.9385656.

Both the exact reservation and conservative planning projection remain below the configured ceiling. The existing ledger cost is $9.550490456503669 including the full conservative charge for attempt A.

## Scientific and runtime seal

The integrated config at `artifacts/branchfabric/gpu-validation/experiment-004/v11/runtime/exp004-v11-integrated-s41-a/config.json` passes strict model validation and the production controller's sealed-evidence verifier.

- Config file SHA-256: `67a6c1bf3f5928e077b9f347e82fe710ff759b607a2b5dd41d6aea6a74c6b4c1`.
- Config canonical SHA-256: `88293cd24640a15ec90d75b14aa43a5f7a7beaf02262eaff5ff533e25968988b`.
- Post-micro manifest SHA-256: `e7b48588923919928e5b2178b93ea1f55e6e94f233609e6fec564b5156b82d67`.
- All nine integrated-prelaunch source, test, review, and make-check hashes match.
- Offline gates A-D pass; micro attempt D passes with 8/8 exact first tokens and admits integrated v11; Agents 9-12 and the cross-runtime audit pass.
- The fresh full repository check passes: 1,842 Python tests, Rust fmt/clippy/tests/doc-tests, and UI typecheck/lint/tests/build.
- This reviewer independently reran the focused seal suite: 84 passed, 0 failed; Ruff passed.

The Modal surface requests exactly `A100-80GB:2` for 588 seconds, has no retry, uses one single-use container, defines no endpoint, and references only the two pre-existing authorized volumes. Local and remote Modal SDK checks require 1.5.3. The local launcher reconstructs the exact pretty-serialized pre-reservation ledger hash, binds the canonical config hash, and remotely binds the reservation, preflight token, and complete 1,176 GPU-second maximum.

A fresh provider preflight at `2026-08-19T04:27:51Z` also passes: Modal SDK 1.5.3, zero apps, tasks, containers, endpoints, reservations, child processes, and profilers, with only `sloforge-model-cache` and `sloforge-branchfabric-results` present. Its artifact SHA-256 is `e1fad3fdbdeb5c687637a42448f4a45eca7a68a1780829299cc76b1060a349c8`.

## Attempt history and frozen baseline

Micro attempts A-D are fully accounted. Attempt A is conservatively charged 525 GPU-seconds; B, C, and D are settled at 155.17, 261.63, and 262.89 GPU-seconds. Every attempt's cleanup artifact reports no surviving app, task, container, endpoint, reservation, child process, or profiler. The only authorized persistent resources are `sloforge-model-cache` and `sloforge-branchfabric-results`.

Seven frozen v10 runtime files were compared byte-for-byte by Git blob against `branchfabric-exp004-naive-baseline-v10`; all match the tag dereferenced at `c5fa2f169e8a343ef7a86ebc814b3a23a7de6b39`. No v10 runtime mutation was used to obtain this PASS.

## Mandatory conditions

The root coordinator must create the sole exact reservation against the canonical config hash and pre-reservation ledger hash, use the explicit authorized paid-budget environment, and launch once. The run must fail closed on sanity-guard, hardware, scientific-validity, timeout, or cleanup failure. After return, the coordinator must settle conservatively and independently record zero Modal resources before any later GPU arm.
