# Agent 10 — attempt-D CUDA performance review

Status: **PASS**. Attempt D satisfies the micro go/no-go and supports exactly one integrated v11 transaction. It does not establish a hardware conclusion.

## Canonical timing and record integrity

The real-A100 result contains 10,454 unique records: 10,216 CUDA records and 238 CPU records. Every CUDA record has device, stream, start/end event, and event duration; every CPU record has null CUDA fields. All 10,454 pass IDs are unique, all dependencies resolve, the dependency graph is acyclic, and every timing group has exactly one owner. Independent sums reproduce the published 22,197,063,040 physical bytes, 2,114,200,960 external bytes, 831,126,047 ns of CUDA-event work, and 521,796,759 ns of synchronization attribution exactly.

The semantic full-state pass count is 10, down from v10's 30. This is distinct from the 10,454 per-chunk/per-layer/per-run execution records. Export contributes 2,160 records; restore contributes 8,294, including the expected 8,128 CUDA restore passes that triggered the corrected attempt-C recorder bound.

## Measured phase decomposition

| Phase | Wall | CUDA event sum | Sync attribution | Record timestamp span |
|---|---:|---:|---:|---:|
| Export | 2.613091 s | 0.133268 s | 0.005272 s | 1.640384 s |
| Restore | 4.438194 s | 0.697858 s | 0.516525 s | 4.261723 s |
| Combined | 7.051284 s | 0.831126 s | 0.521797 s | — |

These columns overlap. CUDA-event sum is aggregate device work, not a concurrency-adjusted critical-path interval. Synchronization includes event-resolution waits and synchronous scalar validation waits. Neither can be added to phase wall time as a disjoint waterfall stage.

The worker's 3.505232-second `integrity_time_ns` is specifically SHA-only: 1.390905 seconds of export publish hashing, 1.397200 seconds of restore preflight hashing, and 0.717127 seconds of consumption hashing. It excludes 4,480 raw-bit destination comparisons, which account for 0.485535 seconds of aggregate launch wall, 0.442473 seconds of CUDA events, and 0.502565 seconds of synchronization attribution. An inclusive hash-plus-raw validation work sum is 3.990767 seconds, but that is aggregate work rather than critical-path latency because the pipeline overlaps operations.

## Residual CUDA chains

Export's measured device chain—native gather, repack, and D2H—uses 0.133268 seconds of CUDA events. D2H moves the 1,056,964,608-byte payload in 0.042120 seconds of copy-engine time, or 25.09 GB/s. The dominant recorded export component is the 1.390905-second CPU publish hash, not the copy engine.

Restore uses 0.063086 seconds for H2D (16.75 GB/s), 0.116518 seconds for 1,792 destination-index uploads, 0.075781 seconds for direct scatter, and 0.442473 seconds for raw-bit validation. Raw validation is 63.4% of restore CUDA-event work. Its median is 72.336 microseconds and p95 is 113.44 microseconds, but one cold/outlier event reaches 53.590 milliseconds. Larger fused scatter/validation runs are the clearest GPU-software diagnostic candidate. The CPU-side preflight and consumption hashes total 2.114327 seconds and remain the dominant measured restore work.

## Temporary memory

The measured pinned staging peak is exactly 58,720,256 bytes, matching the projected two-slot bound. Pageable temporary peaks at 84,832,256 bytes, so the conservative total host-temporary peak is 143,552,512 bytes, excluding the 1,056,964,608-byte resident checkpoint. GPU temporary peaks at 59,769,344 bytes. Against v10's 2,965,372,928-byte host peak, the conservative reduction is 20.66× and 2,821,820,416 bytes.

The pageable value comes from a 10 ms RSS sampler after subtracting resident checkpoint and pinned staging; a shorter pageable spike could be missed. The 58,720,256-byte number is therefore the measured pinned bound, not total host temporary memory.

## Profiling and admission

The causal state phases used minimal tracing, CUDA events, monotonic wall timing, and bounded RSS sampling. No PyTorch profiler, Nsight Systems, or Nsight Compute was active; postflight profiler count is zero. vLLM's internal CUDA-graph memory profiling occurred during engine initialization, outside export/restore timing.

Measured full amplification is 21.0007628× with all correctness gates passing, so the result is **MEANINGFUL** and meets the `<=25×` integrated-admission rule. Relative to v10's 58×, it removes 39,106,884,224 physical bytes, 36.9992× amplification, and 63.79% of physical traffic. External amplification is 2.0002571× and avoidable amplification is 12.0007628×.

Proceed with exactly one integrated v11 transaction, subject to three reporting constraints:

1. Preserve minimal instrumentation and validate the complete canonical pass DAG before publishing movement.
2. Label 3.505232 seconds as SHA-only integrity and report raw-bit validation separately; do not add overlapping timing views.
3. The micro ran on an A100 80GB PCIe while frozen v10 names A100-SXM4-80GB. Direct latency claims require matching hardware/topology or explicit hardware-confounding language. Byte and amplification comparisons remain comparable.

Integrated serving measurements remain mandatory before any BranchFabric hardware-interest decision.
