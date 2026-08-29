# Agent 13 — Attempt-B terminal serving-methodology review

Status: **FAIL — scientifically invalid, terminal, no retry authorized**

Attempt `exp004-v11-integrated-s41-b` failed in GPU0's pre-reclaim 15-rps serving phase. This was a serving-methodology/backlog-trigger abort, not a v11 optimized state-pipeline failure.

## Exact failure sequence

Both A100-SXM4-80GB engines passed readiness. The short 12-rps stability guard and 15-rps overload guard both passed, and both workers published authenticated transaction-ready barriers. GPU1 had prepared eight rollout branches and was waiting for `v10-reclaim-trigger.json`.

The measured transaction started at monotonic timestamp `204943491072`. GPU0 ran the five-second 9-rps control phase, then began the 15-rps spike at `209943491072`. The trigger rule required both positive queue slope and total outstanding in the inclusive range 20–25, but trigger evaluation was forbidden until three seconds after the spike.

On the recorded trajectory:

- Total outstanding first reached 20 at spike +0.800 s.
- The last recorded point at 25 was spike +2.192894367 s.
- Trigger evaluation first became eligible at spike +3.000 s.
- At that point the reconstructed three-second window had 45 offers (15 rps), 27 completions (9 rps), queue depth 13→31, and slope +6 requests/s. Overload and material service deficit were clear, but the bounded 20–25 predicate was already false.
- The producer therefore continued without publishing a reclaim trigger.
- At spike +7.682677963 s, total outstanding reached 65 against the 64-request ceiling: 64 driver-active requests, one request in the external GPU0 queue, and live vLLM state of 16 running plus 48 waiting.

The serving worker then raised `calibrated v10 backlog exceeded its bounded safety ceiling during pre-reclaim` and failed closed. The controller produced no worker results.

## Root cause

For this measured service trajectory, the fixed three-second minimum overload probe and the required 20–25 total-outstanding trigger band were temporally incompatible. The queue traversed the permitted trigger band before the implementation was allowed to evaluate it. Once evaluation became eligible, the queue was already above 25 and continued rising until the hard abort ceiling fired.

The 15-rps overload signal itself was real. The formal overload gate nevertheless cannot pass because its bounded-backlog conjunction never passed at an eligible evaluation time. The bounded safety abort behaved as designed.

The control phase produced 45 requests, all of which eventually completed before abort, with descriptive p95 TTFT of 69.580015 ms. No canonical control-stability result was published, so `control_stable` remains false fail-closed; post-hoc descriptive metrics cannot replace the missing result.

## v11 state-pipeline reachability

No optimized v11 preservation operation ran. GPU1's integrated function waits for `v10-reclaim-trigger.json` before entering the export critical section. That barrier was never created. Consequently:

- no EXPORT_CAPTURE critical section ran;
- no gather/repack, hashing, D2H, state publication, or StatePassRecord generation ran;
- no source release, ownership proof, or HBM reclamation ran;
- GPU1 never served recovery traffic;
- no IMPORT_ADMISSION, H2D, scatter, integrity check, fresh destination allocation, or continuation oracle ran;
- no integrated v11 movement or latency measurement exists.

Preparing the eight live rollout branches is prerequisite workload creation, not state-pipeline execution. This failure neither validates nor falsifies the optimized state pipeline; it prevents integrated validation.

## Scientific-validity disposition

Passed before the abort:

- `two_engine_ready`
- `sanity_12rps_pass`
- `sanity_15rps_pass`
- `budget_pass` after conservative full-bound settlement
- `cleanup_pass` after final provider postflight

Explicit failure:

- `bounded_backlog_pass`

Not emitted or not reached and therefore false fail-closed:

- `control_stable`
- `gpu0_overload_pass` (descriptive overload existed, formal bounded trigger did not)
- `state_capture_pass`
- `allocator_epoch_pass`
- `ownership_release_pass`
- `hbm_reclaim_pass`
- `gpu1_useful_capacity_pass`
- `two_gpu_service_gt_offered_pass`
- `queue_drain_pass`
- `slo_restoration_pass`
- `slo_stability_pass`
- `gpu0_active_during_restore_pass`
- `fresh_destination_allocations_pass`
- `integrity_pass`
- `branch_resume_pass`
- `movement_accounting_pass`

Therefore `scientifically_valid = false`.

The controller also reported that rollout process group 9 survived its local SIGKILL check, so its internal cleanup/identity closure failed. Final provider postflight subsequently proved zero active apps, tasks, containers, endpoints, provider reservations, owned child processes, and profilers. That secondary cleanup issue did not cause the scientific abort and cannot repair it.

## Terminal recommendation

Do not retry attempt B. Preserve it as an invalid pre-reclaim serving-methodology transaction. Do not attribute the failure to v11 export/import and do not use it for integrated v11 timing, movement, correctness, v10 comparison, or hardware-interest claims.

Primary evidence:

- `controller-result.json` SHA-256 `26392f35aa4f0fbb5aa349f2e6ea7b799528702e32ccadf2dfb33865f649251b`
- `v10-gpu0-partial-failure.json` SHA-256 `15623fe28138f59b86d63aab60a26b1d09cc8f7389e7c92d0b82f819a3700c33`
- `abort.json` SHA-256 `1b59141d3daf2e66fd439d3801641283952d9c18d435db4033f65ec81b949acb`
- remote manifest SHA-256 `374c12f875ffd3c4a93014420720abe69b525b5caed189b3090df748c628f64c` (460/460 files present and hash-valid)
- final postflight/settlement SHA-256 `82678f57e9e346f1b4dc68fd6df15d6685987aeda44ef1fd30e7284a5ab55180`
