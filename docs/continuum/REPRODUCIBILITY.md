# Reproducibility

Every Continuum simulation, migration demo, model-check run, fault scenario, and evaluation campaign takes an explicit seed. Protocol artifacts use logical event ordering and canonical serialization; observed host timings are recorded separately because wall-clock samples are not deterministic.

## Clean CPU workflow

```bash
uv sync --locked --extra dev
make continuum-check
make continuum-migration-demo
make continuum-fault-demo
make continuum-fork-demo
make continuum-compatibility-demo
```

Run the multi-seed campaign directly when investigating its exact inputs:

```bash
uv run --locked python -m sloforge.continuum.benchmarking \
  --output artifacts/continuum/evaluation \
  --seeds 101,202,303 \
  --git-commit "$(git rev-parse HEAD)" \
  --initial-output-tokens 16 \
  --delta-rounds 3,2 \
  --resumed-tokens 3 \
  --converter-repetitions 5
```

The resulting `evaluation-summary.json` records the exact command, software and hardware manifests, per-seed raw artifact paths and SHA-256 digests, confidence intervals, hypothesis status, negative results, and adapter exercise status. Static Markdown/HTML reports refuse to render if a referenced raw artifact is changed.

## Metric classes

- `artifact_derived`: counts and invariants read from canonical event/capsule/store artifacts.
- `observed_host`: wall-clock measurements from the machine described in the hardware manifest.
- `synthetic_protocol`: deterministic transport/fabric/model timing, never presented as physical-network performance.

GPU, RDMA, multi-node, and optional runtime tests require explicit environment opt-ins. Absence of hardware or packages is recorded as not exercised; it is not replaced by a fabricated result or silent CPU fallback.

## Canonical conformance

Run Python/Rust golden and hash checks with `make continuum-check`. The ABI schemas and golden fixture under `schemas/continuum/` must round-trip and retain their expected hashes across language implementations and compatible schema migrations.
