# Security and concurrency review

Review date: 2026-08-01

Scope: Rust gateway cancellation, backpressure, SSE and circuit-breaker behavior; Python
subprocess ownership, parsers, real-engine profiling and artifact paths; Docker, Kubernetes,
Modal and Truss generation; dependency health, licenses and secret scanning.

## Result

No unresolved high-severity finding remains in the reviewed scope. The review fixed the
following concrete issues and added regression coverage:

- **Bounded backend data:** error bodies are read incrementally to a 1 KiB cap
  (`crates/sloforge-gateway/src/backend.rs:231`), buffered non-streaming completions have a total
  response bound, and Python mock/replay error reads stop at 512 bytes. Trace, IR, profile and
  chaos input readers now impose file, line and record limits.
- **Circuit-breaker concurrency:** an expired breaker permits exactly one half-open probe and
  releases that reservation on cancellation (`crates/sloforge-gateway/src/backend.rs:90-203`).
  Queue saturation no longer increments backend failure state. Tests exercise both properties.
- **Request isolation:** the gateway refuses backend redirects, rejects credential-bearing or
  path-confusing backend URLs, and does not disclose backend response bodies to clients or logs
  (`crates/sloforge-gateway/src/gateway.rs:96-103`,
  `crates/sloforge-gateway/src/config.rs:145-175`).
- **Bounded health concurrency:** fleet health probes run with a fan-out of 16 instead of
  serializing an entire configured fleet (`crates/sloforge-gateway/src/gateway.rs:145-179`).
- **Real-profiler process ownership:** server logs retain a bounded tail, POSIX process groups are
  terminated even if the launcher exits, readiness does not consume response bodies or accept
  redirects, and SSE response/event sizes and content type are enforced
  (`python/sloforge/profiler/gpu_tools.py:196-284,332-450`).
- **Untrusted model metadata:** JSON metadata reads are bounded, safetensors are required, header
  sizes are capped, tensor offsets are validated, and index path traversal is rejected
  (`python/sloforge/profiler/real.py:809-918`).
- **Deployment integrity:** Docker now runs the selected engine rather than always substituting a
  mock, requests the exact GPU count, publishes the gateway on loopback, and applies read-only,
  no-new-privileges and capability-drop controls. Kubernetes adds non-root/seccomp/container
  hardening, disables service-account token mounting and limits backend ingress to gateway pods
  (`python/sloforge/exporters/core.py:119-204,322-428`). Device and replica topology mismatches
  fail validation before generation.
- **Baseline engine backpressure:** the Transformers correctness server reserves its bounded slot
  before committing HTTP 200 headers, reports saturation as HTTP 429, and releases the slot when
  its streaming iterator is cancelled.
- **Process cleanup:** demo process construction closes log handles on failure and cleanup attempts
  every child even when one shutdown reports an error (`python/sloforge/demo.py:50-106`).

## Remaining findings

### Medium: gateway authentication and TLS are deployment-boundary responsibilities

The gateway router intentionally exposes inference, health and metrics without authentication
(`crates/sloforge-gateway/src/gateway.rs:131-141`). This matches the project's stated non-goal of
building generic SaaS authentication, but a Kubernetes Service, ingress or cloud endpoint must not
be treated as an Internet security boundary without an external mTLS/authentication proxy and
network policy. Local and generated Docker exports now bind only to loopback; the Kubernetes
backend is not directly reachable outside gateway-labeled pods.

### Medium: backend destinations trust compiler/operator authority

Config validation prevents credentials, redirects, queries, fragments and path confusion but
still permits private and link-local HTTP origins (`crates/sloforge-gateway/src/config.rs:145-175`).
Private origins are required for local and cluster inference, so a generic SSRF denylist would
break the product. Treat DeploymentPlan/config write access as code-execution-equivalent and use
namespace/firewall egress controls around any service that accepts plans from an untrusted party.

### Medium: generated runtime image references are not required to be OCI digests

`ExportContext.image` accepts a tag (`python/sloforge/exporters/core.py:41`). The repository's
Docker smoke image is digest pinned, package lock files are present, and generated engine packages
are version pinned, but the exported standalone Dockerfile also uses a mutable Debian tag and
production plan promotion should resolve and record OCI digests before deployment. Enforcing
digest-only input was not applied because it would invalidate the current offline default image
and development workflow.

### Low: Windows descendant-process termination is weaker than POSIX

`ManagedEngineServer` uses a process group on POSIX but only terminates the direct process on
Windows (`python/sloforge/profiler/gpu_tools.py:425-448`). GPU serving is exercised on Linux; a
Windows port should add a Job Object before claiming equivalent descendant cleanup.

### Low: several fixed-tool diagnostic subprocesses capture output without a byte ceiling

Git/package capture and fixed system probes use argv arrays and timeouts but `subprocess.run` can
still retain all output from a compromised executable (`python/sloforge/util.py:46-79,96-113`).
Real inference servers, HTTP bodies and core artifact parsers are bounded. A future common runner
can drain diagnostic output into a bounded tail without changing artifact semantics.

## Dependency, license and secret checks

- `uv pip check --python .venv/bin/python`: 114 packages checked, compatible.
- `npm audit --prefix ui --audit-level=high`: 0 vulnerabilities, including UI development tools.
- Rust dependencies are lock-file resolved; the repository CI runs `rustsec/audit-check@v2`.
  A local `cargo-audit` executable was not installed on this machine.
- Direct Python/runtime licenses reviewed: BSD-3-Clause (httpx, Jinja2 classifier, NumPy), MIT
  (jsonschema, Pydantic, PyYAML, Rich, Typer, Truss), Apache-2.0 (Modal). Hypothesis is MPL-2.0 and
  development-only. Rust's `r-efi` offers MIT or Apache-2.0 as permissive alternatives; no
  AGPL/SSPL/BUSL-only dependency was found.
- Credential-pattern scan across tracked-source candidates (excluding build environments and
  dependency directories): no AWS, GitHub, Slack, OpenAI-style or private-key credential matched.
- The GitHub workflow uses only the scoped `${{ secrets.GITHUB_TOKEN }}` expression; no secret
  value is stored in the repository.

## Verification

- `uv run ruff format python/sloforge tests/python`
- `uv run ruff check python/sloforge tests/python` — passed.
- `uv run mypy python/sloforge` — strict check passed for 43 source files.
- `uv run pytest -q` — 64 passed, 3 GPU-kernel tests skipped because PyTorch/GPU is unavailable.
- `cargo fmt --all --check` — passed.
- `cargo clippy --workspace --all-targets --all-features --locked -- -D warnings` — passed.
- `cargo test --workspace --all-features --locked` — 62 tests passed.
- Dynamic gateway contracts cover partial SSE, malformed SSE, backend disconnect, redirect
  refusal, error redaction, total buffered-response limits, queue saturation, cancellation,
  half-open concurrency, retries before output only, health removal/recovery and slow consumers.
