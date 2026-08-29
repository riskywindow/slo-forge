# BranchFabric GPU Validation Experiment 004 v11 — Final

Status: **terminal; integrated v11 scientifically invalid; no retry authorized**.

Final classification: **SOFTWARE_WINS**, used solely as the least-overclaiming exact-one terminal disposition and a conservative custom-hardware closure. The integrated failure left all four strict classification predicates unestablished, so this is not a claim of measured end-to-end software victory.

## Production integration gates

All four offline gates passed independent review: allocator-issued single-use epochs, post-free ownership/refcount proof, production vLLM engine-step lock binding, and canonical measured CUDA `StatePassRecord` accounting. The final prelaunch `make check` passed 1,859 Python tests with 8 skips, plus Rust and UI checks.

## Real-A100 micro-validation

Attempt `exp004-v11-micro-s41-d` ran the exact 16,384-token shared prefix, fanout 8, 256-token private suffix, 1,152-block, 1,056,964,608-byte workload on a real NVIDIA A100 80GB PCIe. Allocator epochs, post-free ownership, fresh destination lifetimes, integrity, destination commitments, and admission all passed. All eight branches generated at least eight continuation tokens and matched independent recomputation on the first resumed token, 8/8.

Measured v11 results:

- Full physical work: 22,197,063,040 bytes, **21.000762818×** logical.
- External movement: 2,114,200,960 bytes, **2.000257098×**.
- Avoidable physical work: 12,684,381,568 bytes, **12.000762818×**.
- Full-state pass families: 10; raw canonical records: 10,454.
- Export: 2.613090870 s wall, 0.133267776 s CUDA.
- Restore: 4.438193614 s wall, 0.697858271 s CUDA.
- Conservative host temporary peak: 143,552,512 bytes, comprising 58,720,256 pinned and 84,832,256 pageable bytes.
- GPU temporary peak: 59,769,344 bytes.

Compared with frozen v10, v11 removed 39,106,884,224 physical bytes (63.7918%), 20 of 30 full-state passes, and 2,821,820,416 temporary host bytes (20.6571× lower). The byte comparison is valid; the PCIe micro timing is not directly comparable with the frozen SXM4 integrated timing.

## Integrated v11 terminal outcome

Attempt `exp004-v11-integrated-s41-b` authenticated two distinct NVIDIA A100-SXM4-80GB devices. Both engines became ready, and the 12-rps stable and 15-rps overload sanity guards passed. The serving phase then reached 65 outstanding requests against the bounded safety ceiling of 64: 16 running, 48 waiting, and one external outstanding request. It failed closed before the reclaim trigger.

Consequently, integrated export, ownership release, HBM reclamation, GPU1 recovery, restore, movement accounting, GPU0 restore interference, and integrated 8/8 continuation correctness were not reached. Their values are null, not zero and not measurements. The failure was a serving-trigger/backlog methodology incompatibility, not a v11 state-pipeline failure.

The controller reported an in-function cleanup failure because rollout process-group liveness remained after its bounded kill check. The provider post-return audit nevertheless found zero active apps, tasks, containers, endpoints, reservations, child processes, or profilers. All 460 remote artifacts (20,098,327 bytes) match the hash-verified, content-addressed manifest.

The requested direct timing pairs therefore remain: v10 HBM reclamation 5.222174431 s versus v11 not reached; v10 first useful GPU1 capacity 5.413681427 s versus v11 not reached; v10 SLO restoration 11.373208336 s versus v11 not reached; and v10 first resumed token 7.727983415 s versus integrated v11 not reached. None of these null v11 cells is a zero or an improvement.

## Kill/recompute, break-even, and Amdahl

The kill/recompute arm was forbidden because integrated v11 did not pass. The analytical 133,120-token count was therefore not reported as measured. A numerical preservation break-even surface cannot be fitted without measured recompute throughput and setup overhead. At the one measured workload, export plus restore was 7.051284484 s; under the explicitly conditional assumptions of equal fixed overheads and locally constant preserve time, preservation beats recomputation below 18,878.83 tokens/s.

Only micro-stage Amdahl sensitivity is defensible. Accelerating the complete 2.613090870-s export stage yields 1.227×/1.421×/1.500×/1.589× full micro state-path speedup at 2×/5×/10×/free. Accelerating the complete 4.438193614-s restore stage yields 1.459×/2.014×/2.307×/2.698×. These are coarse whole-stage diagnostics, not integrated economic headroom or isolated hardware kernels.

The dominant measured micro chain is destination authentication → H2D → direct scatter/native write → raw-bit/hash validation: 1,056,964,608 logical bytes, 12,684,353,920 physical bytes, 4.438193614 s endpoint wall, 0.697858271 s CUDA-event time, and 0.516524914 s synchronization attribution. Its separately attributable CPU time is null because the CPU, integrity, synchronization, and endpoint timing views overlap; it occupies 62.9416% of the non-overlapping micro export-plus-restore state path. The source chain accounts for the remaining 37.0584%. Their shares of successful integrated reclamation, restore, and full preservation are all null/NOT_REACHED.

The required economic impacts—trajectories per GPU-hour, reclaimed-capacity availability, rollout interruption, wasted recomputation, and dollar-cost improvement—are likewise null/NOT_IDENTIFIABLE. The campaign cost is measured, but no valid integrated preservation/kill pair exists from which to attribute an acceleration effect.

## Hardware gate

Custom hardware interest does not survive. Software reduced measured amplification from 58× to 21.000762818× with exact correctness, but no integrated state-path share, economic improvement, burst rate, latency target, concurrency target, or placement advantage was measured. The final adversarial answer was: **No, I would not independently spend engineering time investigating custom hardware from this evidence.** Experiment 005 was not generated, and no FPGA work was performed.

## GPU budget and cleanup

Known actual v11 usage is at least 0.545029240 A100-hours ($1.361701), while conservative v11 accounting is 0.987969444 A100-hours ($2.468343). The cumulative campaign ledger is 4.475976007 conservative A100-hours ($11.182778), with an actual bounded range of 4.033035802–4.178869136 hours. The ledger retains 1.524023993 authorized A100-hours and has zero active reservations.
