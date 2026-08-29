# Agent 03 attempt-c zero-queue post-fix audit

Status: **PASS — offline remediation is fail-closed**

Audit time: `2026-08-19T01:26:50Z`

This was an independent read-only re-review. It did not invoke vLLM, CUDA,
Modal, the network, or a GPU, and it did not modify the frozen v10 path.

## Decision

Every blocker in the Agent 03 attempt-b root-cause review is closed in the
current tree. The exact source-allocation queue drain is now a valid,
fail-closed lifecycle transition for the fixed vLLM 0.23.0 16K/fanout-8
micro-validation. The remediation is approved for the designated
coordinator's attempt-c budget/methodology gate.

This approval does not assume that the next real runtime will expose 1,152
queued IDs. It approves the assertion: attempt c may proceed only if the real
runtime reports `needs_kv_cache_zeroing is False` and the queue is exactly the
1,152 authenticated live source block IDs. Any other observation terminates
the runtime with structured failure evidence.

## Blocker closure

### Ordering and lifetime separation — PASS

The production worker now orders one `EXPORT_CAPTURE` critical section as:

```text
exact topology and source epochs
-> pre-release ownership snapshot
-> optimized capture and completion fence
-> exact source allocation-queue drain
-> source release and post-free proof
```

No engine step occurs between the drain and release. Restore retains its strict
empty-entry check. Each later destination allocation is immediately drained
and compared with its fresh block-table delta, so source and destination
allocation notifications cannot be mixed.

### Zeroing mode — PASS

The source drain requires the production scheduler value to be exactly the
boolean `False`. Missing, truthy, or `True` values are rejected before
`take_new_block_ids()` is called. This preserves pending zero work on runtimes
where cache zeroing is required.

### Production synchronization — PASS

The helper validates a sealed `EXPORT_CAPTURE` witness against the exact
scheduler and manager both before and after the destructive drain.
`manager.take_new_block_ids` is now a required
`Vllm0230EngineStepBinding` mutation surface. A competing direct drain is
therefore serialized by the same bounded production lock as engine step,
scheduler mutation, allocation, and free. The concurrency test explicitly
demonstrates exclusion of this surface.

### Exact source identity — PASS

The expected queue accepts exactly 1,152 unique, nonnegative strict integers.
The worker derives them from the canonical capture plan's allocator-issued
source epochs. The pre-release ownership call independently requires that
exact block/epoch map, and the queue drain occurs only after capture succeeds.
The evidence binds three lowercase SHA-256 commitments:

- ordered source allocation lifetimes;
- the complete public pre-release ownership snapshot; and
- the completed capture manifest.

Missing, malformed, duplicate, unexpected, or incomplete identities fail
closed.

### Typed destructive-failure evidence — PASS

After a destructive drain, any mismatch raises
`V11RuntimeTeardownRequired`. Its evidence contains gate ID/binding/operation,
zeroing mode, expected count/hash, observed count/hash, missing IDs, unexpected
IDs, duplicate IDs, commitment hashes, and whether the drain occurred. The
worker serializes both `affected_block_ids` and `teardown_evidence` into its
immutable failure and result artifacts; it cannot emit success from that path.

Successful evidence is retained under
`source_allocations.allocation_zero_queue` and contains the exact-match and
held-gate assertions.

### Offline tests — PASS

The new tests cover exact 1,152-ID success; empty/missing/extra/duplicate,
negative, boolean, string, and exception cases; zeroing-required rejection;
source drain ordering; structured failure persistence; required lock-surface
installation; and concurrent direct-drain exclusion. Existing sealed-witness
tests cover stale, foreign, wrong-operation, cross-runtime, and displaced
bindings.

Independent command:

```text
PYTHONPATH=python python -m pytest -q \
  tests/python/test_gpu_reclamation_runtime_v11.py \
  tests/python/test_gpu_reclamation_v11_budget_ledger.py \
  tests/python/test_gpu_reclamation_v11_methodology.py \
  tests/python/test_vllm_allocator_epochs.py \
  tests/python/test_vllm_reclamation_v11.py \
  tests/python/test_vllm_reclamation_v11_accounting.py \
  tests/python/test_vllm_reclamation_v11_ownership.py \
  tests/python/test_vllm_reclamation_v11_sync.py
```

Result: `80 passed in 1.71s`.

Ruff lint and format checks passed for the worker, v11 adapter, synchronization
binding, runtime integration tests, and synchronization tests.

## Current audited hashes

- production v11 worker: `bb2a8ba7fa4b6c1398e08635aa8b401270d68b301d0997422362b55ddc8fe867`
- v11 adapter/typed teardown: `7b5752b37897f880f451068ea69e2439c979dbb3324ea0f8c15c354ecde0dc67`
- production synchronization binding: `05dc160d881f0b772d77db4f6b96298fe4290a7de81be9abcc876e83cada2820`
- runtime integration tests: `3adbc2f2e1e0f424f23e6984a75846cc2c1d06c35fa390402f684a5cab587d45`
- synchronization tests: `02ac7d54b8bec7bd0de84e5a25696cc28dc7a7cbfae6eec4f6f3a73a0fec9aa5`
- v11 pipeline tests: `c81463989638186cd2e1f8792ed1833a982eeeda92cabe8d8d111ad4cd4c7f3b`
- ownership tests: `8ac0f86cb1b475d55d63a2ff5896e96b9ba2b5d2e549ecfc5b5215e87c7e08b7`
- allocator-epoch tests: `2f3751a7c38893a112282605f00f116518e4b7ac8603d7af60f22e76bd78f85b`
- canonical accounting tests: `776a9b3ad0ee057524d0652f68688dc2aa7ce811c1b4450ebeac74f5d7fff2af`
- methodology tests: `7d473282c07c3f4aa1b22f1f2ea100f06f4a2ac9203935ee9eb2a8209a8e277d`
- budget-ledger tests: `d6380b4c1b42b8d932cae6a90f945c0c689f775ed1b034f9dcec89d6c1fce72f`

Repository HEAD remains the frozen analysis commit
`1c51853e10809686d4368037153927f25e834117`; the frozen v10 execution tag
resolves to `c5fa2f169e8a343ef7a86ebc814b3a23a7de6b39`.

Final recommendation: **PASS_OFFLINE_REMEDIATION; attempt c may enter the
separate budget and launch-authorization gate.**
