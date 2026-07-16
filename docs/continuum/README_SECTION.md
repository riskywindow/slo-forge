## SLOForge Continuum

Continuum makes an active AI execution a portable, proof-carrying state capsule. It separates logical continuation state from runtime physical layout, compiles verified state conversion, transfers versioned deltas, and changes the sole state/output owner through a durable fenced transaction.

The deterministic CPU flagship migrates KV plus recurrent, sampler, guided-decoding, history, and delivery state from token-major separate K/V (logical TP=4) to head-major packed K/V (logical TP=2). It injects a destination crash before commit, recovers the valid source, retries, commits a new epoch without a gateway-accepted duplicate or gap, forks the migrated state with copy-on-write, and rejects unsafe same-shape cross-model reuse.

```text
source adapter -> logical + physical capsule -> compatibility -> conversion DAG
       |                    |                       |              |
    dirty log          tenant-scoped CAS        planner      StateTransport
       \____________________ pre-copy + final delta ______________/
                                      |
                         durable lease/fence + gateway ledger
                                      |
                              destination adapter
```

### Three-minute CPU demo

```bash
uv sync --locked --extra dev
make continuum-migration-demo
make continuum-fault-demo
make continuum-fork-demo
```

For a scriptable live migration and independent verification:

```bash
uv run sloforge continuum migrate --mode pre-copy --seed 317 \
  --session session-flagship-001 --output artifacts/continuum/flagship --reset
uv run sloforge continuum migration verify \
  --artifact artifacts/continuum/flagship/flagship.json
uv run sloforge continuum compatibility \
  --artifact artifacts/continuum/flagship/flagship.json \
  --output artifacts/continuum/flagship/compatibility.json
```

The compatibility artifact records both the rejected changed-state-producer reuse and the separately legal recomputation-assisted action. The flagship artifact records the pre-commit failure/rollback, successful cutover, fork ancestry, real chunk counts, and gateway indices.

### Validation status

| Path | Status |
|---|---|
| deterministic reference A to B, cross-layout and cross-TP | implemented and CPU-exercised |
| direct versus canonical CPU conversion | implemented, byte-verified, host-timed |
| Genesis generated runtime binding | implemented with CPU smoke; no live hardware migration claim |
| PyTorch, vLLM, SGLang | version-gated bindings; full live migration unexercised on this host |
| GPU, RDMA, multi-node | unexercised; no performance claim |

The multi-seed benchmark preserves raw artifacts, exact commands, manifests, negative results, and 95% confidence intervals while keeping observed-host and synthetic-protocol metrics separate. Generated reports, rather than README prose, are the source of numeric benchmark claims.

Start with [the architecture](docs/continuum/ARCHITECTURE.md), [the demo script](docs/continuum/DEMO_SCRIPT.md), [compatibility rules](docs/continuum/COMPATIBILITY.md), and [limitations](docs/continuum/LIMITATIONS.md). GPU, RDMA, multi-node, vLLM, and SGLang execution status is artifact-backed; no unavailable hardware path is represented as exercised.
