# SLOForge Fabric security and concurrency review

Reviewed 2026-08-01 on macOS/arm64. The review covered recovery persistence and
idempotency, subprocess and process-group cleanup, bounded ingress and queues,
gateway streaming cancellation, filesystem and symlink behavior, artifact
integrity, YAML/JSON parsing, privileged/cloud mutation controls, secrets,
command construction, and Fabric runtime adapters.

## Result

No unresolved high-severity finding remains in the reviewed local and
simulation paths. The shipped recovery executor remains non-mutating by
default, physical faults remain simulation-only, and cloud adapters remain
offline metadata generators. The focused Rust and Python checks and a fresh
CPU-only Fabric demo passed.

## Resolved findings

| Original severity | Finding | Resolution and evidence |
|---|---|---|
| High | A recovery driver could return a successful attempt for a different action, and a tampered snapshot could omit or substitute action evidence. | The executor now binds returned action and idempotency identities to the requested action, bounds driver details, records arbitrary driver exceptions as failures, rejects duplicate plan keys and batch attempts, and requires restored applied actions to exactly match successful attempt evidence. See `python/sloforge/recovery/executor.py` and `python/sloforge/recovery/state_machine.py`. |
| High | Recovery snapshots were written directly and parsed without a recovery-specific input bound. A final symlink could redirect a CLI snapshot write. | Snapshots are limited to 8 MiB and atomically written through a mode-0600 sibling temporary file, `fsync`, and `os.replace`; the final symlink is replaced instead of followed. Restore validates bounded history and plan/action integrity. Concurrency and final-symlink tests are in `tests/python/test_fabric_recovery.py`. |
| High | ForgeCI captured unbounded child stdout/stderr with `communicate()`, permitting memory exhaustion. | Two bounded drainers retain at most 1 MiB per stream while continuing to drain pipes. Truncation fails a benchmark/build closed. Timeouts terminate the process group, readers are joined, and a surviving reader is an explicit error. `tests/python/test_forgeci.py` exercises output exhaustion and orphan prevention. |
| High | Fabric adapter lowering accepted a physical plan and a different topology artifact, while exporting the plan's unverified fingerprint. | `FabricAdapterContext` now verifies `plan.topology_fingerprint == canonical_hash(topology)` after structural placement validation. A mismatch test covers this provenance boundary. |
| Medium | Gateway configuration files and several capacities/timeouts were not bounded above; request `max_tokens` had no gateway-level ceiling. | Gateway/mock configs are capped at 1 MiB. Admission, buffers, request size, trace capacity, retries, timeouts, circuit-breaker settings, backend capacity, output tokens, and mock delays/counts are bounded. Oversized output-token requests are rejected before admission. |
| Medium | The mock backend's unauthenticated `/admin/fault` route was always exposed. | The route is absent unless `fault_api_enabled: true`. Only deterministic demo/test configs opt in. Fault request bodies and parameters are bounded. This endpoint is still test infrastructure and must not be exposed publicly. |
| Medium | Gateway replay created one asynchronous task per trace request. | Replay now accepts at most 100,000 requests and uses at most 128 workers over a queue bounded to 256 entries. `asyncio.TaskGroup` cancels sibling tasks on failure and results remain input-ordered. A fresh flagship demo exercised this path. |
| Medium | ForgeCI inherited the complete host environment into arbitrary benchmark commands. | Common credential/token/key/cookie markers are removed from the ambient environment. Matrix variables remain explicit; variables marked sensitive are redacted from logs and manifests. Commands remain argument vectors and never use a shell. |
| Medium | ForgeCI matrices, physical fault scenarios, generic fault scenarios, and Autopsy control documents admitted unbounded YAML or YAML reference expansion. | These loaders now enforce 4 MiB input bounds, use `safe_load`, and reject anchors and aliases before construction. Corresponding adversarial tests pass. |
| Medium | WarmPath accepted Windows-style parent traversal in an otherwise portable relative path. | Artifact paths normalize both separator styles and reject parent/current segments, drive prefixes, absolute paths, and control bytes. The local executor continues to resolve and enforce containment. |
| Medium | Adapter output could be a symlink or contain stale symlink artifacts that were followed during hashing. | Export rejects a symlink output directory and any symlink below it before validation and digest generation. Final generated file writes already use atomic replacement. |

## Concurrency and cancellation invariants verified

- `DeterministicRecoveryExecutor.tick`, snapshotting, and persistence serialize
  under an `RLock`; 64 simultaneous duplicate observations produce one audit
  record and one processed idempotency key.
- Restored action attempts are part of the pending-action guard, so a new
  `SimulatedActionDriver` does not replay persisted actions.
- The gateway uses bounded global admission and per-backend semaphores. Client
  stream drop releases both permits and cancels backend work.
- Gateway retries occur only before output begins. Partial malformed SSE,
  partial disconnects, slow consumers, response bounds, queue saturation, and
  cancellation are covered by the Rust contract suite.
- Health-check tasks abort on drop. Demo-managed gateway/backend processes use
  separate process groups and TERM/KILL cleanup; the fresh demo left no
  SLOForge process behind.
- ForgeCI command capture drains stdout and stderr concurrently, caps retained
  output, terminates the process group on timeout, and refuses to continue if a
  reader fails to stop.

## Trust boundaries and residual risks

1. ForgeCI executes commands from an explicitly trusted local benchmark matrix.
   It is not an untrusted-code sandbox: benchmark code can read local files or
   access the network with the invoking user's authority. Ambient secret names
   are filtered, but isolation requires an external container/VM boundary.
2. `ActionDriver.apply` is synchronous under the recovery executor lock. A
   future production driver must enforce its own I/O timeout and durably honor
   the supplied idempotency key. SLOForge currently ships only the bounded,
   non-mutating simulated driver; no external production mutation driver is
   enabled. In-process timeouts cannot safely cancel an already-running
   side-effecting Python call.
3. Process-group cleanup is strongly exercised on POSIX. Windows uses the
   standard new-process-group/terminate path but does not yet use a Windows Job
   Object, so independently detached descendants are outside the current
   guarantee.
4. Artifact SHA-256 values provide content integrity and provenance matching,
   not publisher authenticity. Evidence from an untrusted producer still needs
   a signed supply-chain envelope outside this repository.
5. ForgeCI upstream bundles and some local reports preserve absolute artifact
   paths for reproducibility. They contain no credential values, but paths may
   expose workstation names and should be scrubbed before public upload.
6. Gateway TLS, authentication, and authorization remain the responsibility of
   a trusted ingress/service mesh. The mock fault API is opt-in but intentionally
   unauthenticated test infrastructure.
7. Atomic writers replace final symlinks, but caller-selected parent directories
   are a local trust boundary. The CLI is not intended to run setuid or write
   into attacker-controlled directory trees with elevated authority.

## Privileged, cloud, and adapter disposition

- Physical Fabric fault schemas permit only `execution_mode: simulation`; this
  review found no host clock, traffic-control, or RDMA mutation command in that
  path.
- The portable recovery CLI rejects every plan that authorizes or requires an
  external mutation.
- Modal and Truss Fabric adapters emit advisory offline metadata, and the
  local/Docker/Kubernetes/Dynamo adapters emit argument arrays and manifests;
  none deploys resources. Runtime commands are constructed as typed argument
  vectors rather than shell strings.
- The adapter composition check now binds every export to the exact canonical
  topology hash as well as typed host/GPU/NUMA/NIC/rail references.
- No credential-like material matching private-key, AWS key, GitHub token, or
  OpenAI-style key patterns was found in tracked non-artifact source during the
  review.

## Verification

Executed successfully:

```text
cargo fmt --all -- --check
cargo clippy -p sloforge-gateway --all-targets -- -D warnings
cargo test -p sloforge-gateway
  18 unit tests passed
  16 gateway contract tests passed

uv run ruff format --check <reviewed Python and test paths>
uv run ruff check <reviewed Python and test paths>
uv run mypy python/sloforge/recovery python/sloforge/forgeci \
  python/sloforge/runtime/gateway_replay.py python/sloforge/cli/common.py \
  python/sloforge/fabric/faults.py python/sloforge/faults \
  python/sloforge/warmpath python/sloforge/fabric/adapters
  Success: no issues found in 37 source files

uv run pytest -q tests/python/test_fabric_recovery.py \
  tests/python/test_forgeci.py tests/python/test_fabric_faults.py \
  tests/python/test_controller_faults.py tests/python/test_warmpath.py \
  tests/python/test_fabric_adapters.py tests/python/test_bounded_http.py \
  tests/python/test_fabric_cli.py
  74 passed

uv run --locked python -m sloforge.fabric.demo \
  --artifact-dir <temporary>/artifacts --report-dir <temporary>/reports --reset
  completed; two injected faults, seven counterfactuals, recovery COMPLETED,
  restored SLO attained, 36 files emitted, no child process remained
```

The review did not create cloud resources, alter host networking or GPU clocks,
or exercise privileged probes.
