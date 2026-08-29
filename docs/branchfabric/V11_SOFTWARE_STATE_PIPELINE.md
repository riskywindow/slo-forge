# BranchFabric Experiment 004 v11 software state pipeline

Status: local state equivalence passes; real-A100 validation is blocked by the current GPU authorization.

The v10 naive path is frozen at execution commit `c5fa2f169e8a343ef7a86ebc814b3a23a7de6b39` and analysis commit `1c51853e10809686d4368037153927f25e834117`. The local annotated tag `branchfabric-exp004-naive-baseline-v10` dereferences to the execution commit. Its adapter remains byte-for-byte unchanged.

## Why v10 reached 58×

The corrected offline DAG contains 270 segment records, 30 conceptual full-state passes, 44 execution nodes, and 46 dependency edges. It conserves exactly 61,303,947,264 physical bytes for 1,056,964,608 logical bytes.

- Capture and publication: 21×.
- Restore through native write: 20×.
- Destination recapture and host comparison: 17×.
- By memory role: 29× GPU intermediates, 23× host intermediates, 3× native endpoints, and 3× links.

Shared measured-operation intervals have one timing owner. Their 21 internal semantic edges do not contribute independently to latency, so critical-path or Amdahl consumers cannot double-count them.

## v11 implementation

The isolated implementation is `python/sloforge/continuum/adapters/vllm_reclamation_v11.py`. It does not change or wrap the frozen v10 converter in place.

Capture preflights all native dimensions, the complete logical manifest, the exact CUDA index, and GPU UUID before allocating or moving state. It then gathers native pages in deterministic page/valid-extent chunks, repacks directly into canonical order, and asynchronously copies into the final pinned checkpoint. The default 28 MiB chunk corresponds to 32 full pages of the frozen workload; two bounded slots are used.

Restore verifies retained transport, takes an owned bounded host snapshot, authenticates each page again at consumption time, asynchronously transfers it, and scatters directly into fresh native pages. It never reconstructs a complete canonical GPU tensor. Full pages skip zero-fill because every readable byte is overwritten. A partial page zeros and raw-validates only its unreadable tail.

Destination validation compares BF16 raw bits, preserving NaN payloads and distinguishing signed zero. It does not recapture the full destination to host. Its evidence commits to the manifest, payload digest, ordered page-to-block mapping, destination tensor identities, geometry, exact device identity, and allocation epoch. Stale-evidence tests identify a remaining requirement: production must issue an unforgeable, single-use allocation epoch and must not accept caller-reused epochs.

On a write failure, every touched destination page is raw-zeroed and fenced. Cleanup errors raise `V11RuntimeTeardownRequired`. A remaining integration gate is post-free allocator ownership proof: a silent no-op `free()` cannot yet be distinguished from successful release. Likewise, the admission witness must be bound to the real controller-owned engine-step lock rather than an arbitrary probe before GPU use.

## Local verification

The targeted offline suite passed 47 tests, including 19 v11-specific tests. It covers randomized fixture seeds and page order, partial tails, v10/v11 bitwise transport equality, v10/v11 logical restore equality, raw BF16 edge cases, missing/duplicated/reordered transport ranges, tensor and NumPy-alias mutation, destination corruption, stale mapping evidence, cleanup failures, and admission-gate loss. The targeted frozen-v10 and canonical-accounting tests also pass.

The algorithm-derived baseline-workload projection is 21× full physical work and 2× external link movement, versus measured v10 values of 58× and 3×. This is not a GPU measurement and is ineligible as a headline result. Real PyTorch/CUDA allocator behavior, event timing, overlap, HBM temporary peaks, canonical per-chunk StatePassRecords, single-use allocation provenance, and post-free ownership evidence remain micro-validation prerequisites.

## GPU gate

No v11 GPU run was launched. The ledger has 943.176 A100-seconds left, while the primary campaign requires 3,474.15 A100-seconds with margin. More importantly, the current explicit authorization excludes optimized preservation and kill/recompute, and `SLOFORGE_GPU_BUDGET_USD` is unset. See `artifacts/branchfabric/gpu-validation/experiment-004/v11/v11-gpu-budget.json`.
