# Baseline runtime synthesis

Genesis generates a correct, conservative runtime before optimization. `generate_baseline_runtime(package, inspection, output, seed=...)` refuses a stale inspection and any inspection containing unsupported semantics. The generation seed contributes to the runtime identity.

## Generated bundle

Each model-specific bundle contains:

- `runtime.py`: a bounded JSON-lines stdin/stdout streaming runner and importable application factory;
- `runtime_config.json`: content identity, exact reference-package root, generation provenance, and runtime bounds;
- `correctness_harness.py`: deterministic sample replay with optional exact-token comparison;
- `deployment_manifest.json`: offline deployment intent, health/metrics routes, read-only inputs, writable artifact location, queue bound, and mandatory request timeout;
- `artifact_manifest.json`: SHA-256 identities for generated artifacts.

Generation is offline and does not install packages, access the network, deploy resources, or execute reference code. Existing non-identical files are never overwritten.

At load time, Genesis recomputes the complete reference-package identity. Changed source, tokenizer, contracts, or evaluation samples invalidate the runtime before model import.

## Runtime protocol

The conservative runner accepts at most 65,536 bytes per JSON request. Every inference request supplies a request ID, text, maximum output length, seed, and timeout. It emits one JSON object per token followed by exactly one terminal event. Health and metrics are available as `/healthz` and `/metrics` application routes and as `health` and `metrics` control operations on the JSON-lines process boundary.

The in-process runtime provides:

- a fixed-capacity admission queue and fixed-capacity per-request output queue;
- a deterministic FIFO worker and bounded microbatch collection;
- explicit `CREATED`, `QUEUED`, `PREFILLING`, `DECODING`, and terminal lifecycle states;
- declared model loading, state allocation, prefill, decode, sampler, and tokenizer calls only;
- per-request single-owner state allocation and release in a `finally` path;
- cancellation and deadline checks before prefill, after prefill, before and after every decode operation, before sampling commitment, and before token emission;
- stable SHA-256-derived seeds for allocation, prefill, every decode step, and every sample;
- ordered token sequence numbers and bounded deterministic output;
- duplicate active-request rejection, queue-saturation rejection, clean non-daemon worker shutdown, and child-free execution;
- health and operational counters for submissions, terminal states, batches, tokens, state allocations, and releases.

The runtime deliberately uses one worker and conservative reference operations. Basic batching collects independently eligible requests but executes them serially so that it cannot invent model-specific batched semantics. Later synthesis stages may replace this policy only after creating and discharging new proof obligations.

## Correctness and failure behavior

The HybridDecoder fixture exercises deterministic replay, exact expected token streams, streaming order, cancellation-before-commit, queue saturation, state ownership balance, source tamper detection, health and metrics routes, and clean shutdown. The correctness harness is a real bounded subprocess in tests with an explicit timeout; the generated runtime itself does not spawn children.

Run:

```bash
PYTHONPATH=python pytest -q tests/python/test_genesis_runtime.py
```

## Trust boundary and limits

Reference code and generated bundles remain untrusted. Import and execution must occur inside the Genesis sandbox; runtime generation alone does not authorize traffic or promotion. The application does not open a network listener. A deployment adapter may expose HTTP/SSE only after sandbox, capsule, and promotion validation.

Python cannot safely preempt an arbitrary native or Python operator in the worker thread. Request deadlines are enforced at every operator boundary, while the outer Genesis sandbox supplies the hard wall-clock and process cleanup boundary for a hung operator. The baseline supports request-owned state and explicitly rejects unsupported semantic recovery; state migration, distributed execution, and optimized kernels belong to later verified synthesis stages.
