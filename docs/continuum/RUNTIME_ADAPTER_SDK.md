# Runtime adapter SDK and status

`ContinuumRuntimeAdapter` is a version-scoped, fail-closed boundary. It exposes typed values and bounded byte segments, not runtime pointers. Every adapter publishes a `CapabilityMatrix` containing operations, state kinds, layouts, dirty-tracking strategies, and explicit resource limits.

## Required lifecycle

An adapter supports some declared subset of:

1. inspection (`identity`, `capabilities`, session metadata and watermarks);
2. consistent capture (`begin_consistent_snapshot`, segment/page/logical reads, `end_consistent_snapshot`);
3. live tracking (`start_dirty_tracking`, versioned delta, token-boundary quiesce, final delta);
4. import (`prepare_destination`, allocate/import/apply delta, rebuild ephemera, validate, abort/release);
5. continuation (`dry_run_next_token`, activate/resume/pause/drain/cancel);
6. fencing (`release_source_fence` and owner-epoch checks on every mutation/output).

Unsupported behavior raises a typed `UnsupportedCapabilityError`; absence and version mismatch have distinct errors. Limits bound sessions, snapshots, dirty events, segment bytes, and stream buffers.

## Adapter status

| Adapter | Implementation | Exercised here | Honest boundary |
|---|---|---|---|
| deterministic token-major | full SDK | CPU capture, dirty tracking, migration, rollback, streaming | simulated devices only |
| deterministic head-major | full SDK | CPU import, validation, activation, fork, streaming | simulated devices only |
| Genesis | versioned descriptor/binding | generated baseline load and CPU smoke | no full active-state migration claim |
| PyTorch | version-gated CPU tensor/RNG binding | package discovery; host package may be absent | not a complete live scheduler/session adapter |
| vLLM | version-gated public-API binding | static discovery when installed | public KV-transfer interfaces do not provide a universal execution capsule |
| SGLang | version-gated public PD binding | static discovery when installed | public disaggregation interfaces do not expose all logical state |

`probe_*` reports `implemented_and_exercised`, `implemented_but_unexercised`, `partially_implemented`, or unavailable/version-incompatible state based on the local environment. Public API discovery is not migration validation.

## Adapter author checklist

- Bind the model, tokenizer, positional encoding, adapter, and state-producer hashes.
- Serialize every persistent logical component, including RNG and output watermarks.
- Mark reconstructible runtime state rather than attempting to copy it.
- Instrument explicit segment/page versions; declare hash comparison only as a reduced-efficiency fallback.
- Reject stale epochs before mutating state or emitting output.
- Import into an inactive candidate, validate, then activate only after coordinator commit.
- Add deterministic fixtures, corruption cases, resource-limit tests, and exact version gates.
