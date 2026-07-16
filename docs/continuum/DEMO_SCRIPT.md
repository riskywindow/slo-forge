# CPU flagship demonstration

The flagship is local, deterministic, and requires no GPU, cloud credentials, root access, external model API, RDMA, or paid resource.

## Three-minute path

```bash
uv sync --locked --extra dev
uv run sloforge continuum migrate \
  --mode pre-copy \
  --seed 317 \
  --session session-flagship-001 \
  --output artifacts/continuum/flagship \
  --reset

uv run sloforge continuum migration verify \
  --artifact artifacts/continuum/flagship/flagship.json \
  --output artifacts/continuum/flagship/verification.json

uv run sloforge continuum compatibility \
  --artifact artifacts/continuum/flagship/flagship.json \
  --output artifacts/continuum/flagship/compatibility.json
```

## What actually happens

1. A token-major, separate-K/V, logical TP=4 source creates a seeded hybrid session with KV, recurrent, sampler, guided-decoding, token history, and delivery state.
2. Tokens stream through the gateway ledger and establish a committed watermark.
3. A consistent capsule is content-addressed and a direct TP/layout converter is compared byte-for-byte with the canonical converter.
4. The source generates during pre-copy. Versioned deltas move over the deterministic simulated transport.
5. A destination crash is injected during validation before commit. The transaction rolls back; a restarted coordinator recovers the journal; the source remains the valid epoch/output owner.
6. A fresh migration prepares the head-major, packed-K/V, logical TP=2 destination, transfers final delta at a token boundary, validates continuation, fences source, commits epoch 2, switches the gateway, and resumes.
7. The verifier checks a contiguous, duplicate-free accepted token sequence and rejection of stale-source output.
8. The migrated checkpoint forks into two content-sharing branches under different layouts.
9. A same-shape model with changed state-producing weights is rejected for direct reuse; a separate token-history recomputation-assisted plan is recorded.

All timeline events, chunk counts, watermarks, fault labels, conversion evidence, and branch storage counts are generated from the run. `manifest.json` hashes the flagship artifact and marks hardware as synthetic.

## Inspect lower-level pieces

```bash
uv run sloforge continuum runtime inspect --runtime reference-a \
  --output artifacts/continuum/runtime-a.json
uv run sloforge continuum state capture --session session-capture --seed 317 \
  --runtime artifacts/continuum/runtime-a.json \
  --output artifacts/continuum/capture
uv run sloforge continuum capsule validate \
  artifacts/continuum/capture/capsule.json
uv run sloforge continuum migration modelcheck --seed 41 \
  --output artifacts/continuum/modelcheck.json
```
