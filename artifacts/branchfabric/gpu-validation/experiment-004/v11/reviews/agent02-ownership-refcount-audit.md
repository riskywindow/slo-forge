# Agent 02 post-free ownership/refcount audit

## Verdict

**PASS — Gate B is approved for the pre-GPU integration gate.**

The final implementation proves exact pre-release ownership and exact post-release absence across the adapter session graph, scheduler request table, native KV manager request-to-block table, allocator epoch source, native block pool, and allocator-visible HBM pool accounting. It is bound to the real vLLM release path under the production `EXPORT_CAPTURE` engine-step gate and fails closed when any layer disagrees.

No Modal, GPU, network, or frozen-v10 mutation was used. The reviewer modified no source code.

## Resolved blocking finding B-01

The initial review found that a detached `SingleTypeKVCacheManager.req_to_blocks` table could survive after scheduler removal while the old proof returned `passed=True`. The implementation was corrected before this final verdict.

The proof now reads the exact pinned-vLLM ownership map at `KVCacheManager.coordinator.single_type_managers[0].req_to_blocks`. Before release, all declared source runtime IDs must be keys in that native table. After release, every old runtime ID must be absent. The result is persisted as `native_manager_request_tables_absent`, copied to every block transition as `native_request_tables_absent`, and included in both aggregate and per-block `passed` predicates.

I reran the original adversarial scenario with both old tables deliberately retained. The corrected gate rejected it:

```text
{'rejected': True,
 'error': 'v11 source release failed the post-free ownership gate',
 'surviving_manager_tables': ['branch.0', 'branch.1']}
```

The focused test suite now contains this failure mode as `test_post_free_proof_rejects_stale_native_request_block_tables`.

## Requirement audit

| Requirement | Result | Evidence |
|---|---:|---|
| Exact source allocation coverage | PASS | Layout block IDs equal the caller-bound allocator-epoch map and exact expected count. |
| Block ID and allocation epoch | PASS | Per-block snapshot and transition records bind both; allocator proof is refreshed before release and tombstoned after free. |
| Pre-release owner and refcount | PASS | Logical owners, runtime owners, scheduler referrers, native `ref_cnt`, and native request-table membership must agree. |
| Real release operation | PASS | Exact production adapter calls `destroy_branch`/`destroy_session`, which invoke abort/free/reset. |
| Post-release owner and refcount | PASS | Source native refcounts are zero; scheduler referrers, old runtime requests, and old native request tables are absent. |
| Allocator availability | PASS | Every source block must be non-null, zero-refcount, hash-free, and allocator-available; the complete usable pool must be recovered. |
| No live branch/session/global reference | PASS | Destroyed session state, root-reference absence, scheduler scan, and exact `req_to_blocks` key absence are all required. |
| Old-session inaccessibility | PASS | Old adapter layouts are empty and native manager ownership tables for all old runtime IDs are absent. |
| Allocator/HBM agreement | PASS | Pre/post KV assigned/unassigned bytes and source physical bytes close exactly; HBM-pool observation cannot substitute for ownership evidence. |
| Production gate binding | PASS | Capture and release require an active, same-thread `EXPORT_CAPTURE` witness for the exact live scheduler and manager. |
| Exact 1,152-block enforcement | PASS | The production API requires exact count plus one allocator epoch per source block; a sampled or incomplete set fails before release. Real records will be materialized in the micro-run. |

## Offline checks

```text
uv run --locked pytest -q \
  tests/python/test_vllm_allocator_epochs.py \
  tests/python/test_vllm_reclamation_v11_sync.py \
  tests/python/test_vllm_reclamation_v11_ownership.py

22 passed in 0.79s
```

```text
uv run --locked ruff check \
  python/sloforge/continuum/adapters/vllm_reclamation_v11_ownership.py \
  tests/python/test_vllm_reclamation_v11_ownership.py

All checks passed!
```

Python compilation of the ownership, epoch, and live-adapter modules also passed. The separate adversarial rerun rejected the retained native tables exactly as required.

## Reviewed hashes

| File | SHA-256 |
|---|---|
| `vllm_reclamation_v11_ownership.py` | `ec28e4ff1362c81e7ab2ec2095ce509aaf625f892ae08e26d1708a87b846228f` |
| `test_vllm_reclamation_v11_ownership.py` | `2e45a183923de79acf198db5f7ed153712ef9bf268ba23f4e308143ff32efe84` |
| `vllm_live.py` | `149d423303dc91201826902bb406c5c6de88aed043933a23e28a4f39fed3b2f0` |
| `vllm_allocator_epochs.py` | `14e819a5cf3ebd5f73ad2a1b9ab6b975508b6b549422d0568cee1d5dd7a8a497` |
| `vllm_reclamation_v11_sync.py` | `095384e5d892d6a1899a9dc1ddc9e2c2bea9d3dbcc3d8c87176c381aab2afbd0` |

Working-tree HEAD was the v10 analysis commit `1c51853e10809686d4368037153927f25e834117`; the frozen execution tag still dereferenced to `c5fa2f169e8a343ef7a86ebc814b3a23a7de6b39`.

## Approval

Gate B is **PASS** for offline integration. The later real-A100 micro-validation must still emit all 1,152 per-block transitions from this exact production proof and make `native_manager_request_tables_absent`, allocator/HBM agreement, and every other aggregate Gate-B boolean mandatory for the micro go/no-go decision.
