# Agent 08 — v11 micro-validation methodology audit

Result: **PASS**

Scope: offline, source read-only review of the one-A100 v11 micro worker, controller, Modal boundary, immutable configuration, canonical pass accounting, budget binding, and focused tests. This reviewer performed no GPU, Modal, cloud, network, or paid action.

## Decision

The smallest scientifically valid real-A100 v11 micro-validation design is ready for the designated GPU coordinator once the consolidated offline gate is hashed and the exact one-GPU reservation is created atomically. The audit found no source-level blocker.

This is a methodology-readiness result only. It does not claim real-A100 correctness, amplification, bytes, latency, temporary memory, or cleanup evidence.

## Exact workload and runtime

The immutable configuration and runtime fail closed on all required identities:

- one `A100-80GB` GPU; one visible CUDA device; A100 name; compute capability 8.0; at least 79,000 MiB; stable GPU UUID
- `Qwen/Qwen2.5-7B-Instruct` at model and tokenizer revision `a09a35458c702b33eeacc393d103063234e8bc28`
- vLLM `0.23.0`, PyTorch `2.11.0`, BF16
- 16,384 shared-prefix tokens, fanout 8, 256 private divergent tokens per branch, and at least 8 restored continuation tokens per branch
- 16-token blocks: 1,024 shared plus 128 private equals exactly 1,152 full physical pages
- exactly 1,056,964,608 logical state bytes

The worker derives the capture plan from real live vLLM block tables, runtime-issued allocation epochs, and physical pages. It rejects any state that differs from that complete geometry.

## Lifecycle and correctness

Export runs inside the production capture critical section. The worker requires the full post-free ownership/refcount proof before export completes. Import uses new runtime allocations and rejects any source/destination `(block ID, allocation epoch)` lifetime overlap; a physical ID may reappear only with a distinct later allocator epoch.

All eight restored branches must emit at least eight tokens. After they drain, the worker resets the prefix cache and independently prefills every immutable prompt history with its original effective seed. It requires exact first-resumed-token equality for 8/8 branches. Any failed allocator, ownership, synchronization, mapping, integrity, freshness, accounting, topology, continuation, or oracle condition fails the result closed.

## Measurement method

The causal run uses minimal instrumentation: CUDA events, monotonic wall clocks, bounded 10 ms RSS sampling, and actual tensor-allocation extents. It enables no heavy profiler, so heavy-profiler overhead is not applicable and must not be estimated.

Canonical `StatePassRecord` is the single accounting authority. The validator enforces required fields, operation-site CUDA events, one timing owner, exact device and branch-group identity, and valid dependency edges. Full, external, avoidable, D2H/H2D, and tier bytes are summed from records without adding profiler or transfer counters again.

The result surface contains logical, physical, external, avoidable and tier bytes; amplifications; full-state pass count; pinned/pageable/GPU temporary peaks; export/restore wall and CUDA time; integrity and synchronization time; and the complete serialized pass stream.

Export wall time covers capture through source release and proof. Restore wall time covers transport verification through admission. CUDA time uses non-overlapping event owners. Synchronization time uses explicit recorded waits. Host and GPU temporary attribution uses sampled or live allocated extents rather than the projected 58,720,256-byte staging bound.

## Bounded invocation and budget

The execution bounds are 180 seconds for initialization, 320 seconds for the transaction, 20 seconds for cleanup, and 525 seconds maximum wall time. The controller uses a fresh process group with TERM/KILL escalation and verifies that the sole GPU has no compute process afterward. Modal requests exactly `A100-80GB:1`, permits one single-use container, no buffer container, no retry, and no endpoint.

The reviewed authorization permits 6.0 cumulative A100-hours. At review time the ledger recorded 12,556.823626 A100-seconds used and 9,043.176374 seconds remaining. The micro envelope is 525 GPU-seconds; with a 15% planning margin it is 603.75 A100-seconds. Agent 08 created no reservation. Only `/root` may reserve and launch.

## Strict post-result gate

- `<=8x`: EXCELLENT
- `>8x` and `<=12x`: STRONG
- `>12x` and `<=25x`: MEANINGFUL
- `>25x` and `<=40x`: WEAK; require adversarial review before any exception decision
- `>40x`: FAILED_TO_REMOVE_PATHOLOGY; stop without an integrated allocation

Normal continuation requires correctness plus measured amplification no greater than 25x. Any exception above 25x requires strong measured evidence that the residual bytes are unavoidable and latency remains scientifically interesting. The gate must use the immutable real-A100 `StatePassRecord` totals, never the projected 21x value.

## Mandatory coordinator obligations

Before launch:

1. Consolidate all eight PASS reviews and full `make check` evidence, hash the gate manifest, and bind that exact hash into the immutable micro configuration.
2. Re-read the ledger under its lock, recompute remaining authorization, and reserve exactly one A100 for at most 525 GPU-seconds. Do not reserve later stages.
3. Set the authorized paid-launch value, reservation identity, and preflight token only in the designated `/root` coordinator process. Invoke once with no retry.

After launch:

1. Validate the immutable result and apply the strict gate before a separate integrated reservation.
2. Settle actual and conservative GPU-seconds and cost.
3. Prove no active app, task, container, endpoint, reservation, owned child process, or profiler remains. Only `sloforge-model-cache` and `sloforge-branchfabric-results` may persist.
4. Replace or explicitly supersede the stale pre-authorization `BUDGET_BLOCKER` micro status so it cannot be mistaken for the current authorization.

These are runtime closeout requirements, not missing source behavior. The controller can prove child GPU-process cleanup inside its container; provider resource cleanup necessarily becomes evidence only after the invocation.

## Offline verification

- Ruff: PASS
- Python compilation: PASS
- static Modal/CUDA boundary audit: PASS
- focused suite: **76 passed, 0 failed in 1.75 seconds**
- no GPU, Modal, network, or source edit performed

Focused suite:

```text
PYTHONPATH=python:experiments/branchfabric python -m pytest -q \
  tests/python/test_gpu_reclamation_runtime_v11.py \
  tests/python/test_gpu_reclamation_v11_methodology.py \
  tests/python/test_gpu_reclamation_v11_budget_ledger.py \
  tests/python/test_vllm_reclamation_v11_accounting.py \
  tests/python/test_vllm_reclamation_v11.py \
  tests/python/test_vllm_reclamation_v11_ownership.py \
  tests/python/test_vllm_reclamation_v11_sync.py \
  tests/python/test_vllm_allocator_epochs.py
```

The authoritative JSON artifact contains the reviewed source, test, authorization, and ledger SHA-256 commitments.
