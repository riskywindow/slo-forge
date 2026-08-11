# Clean-room setup and reproducibility review

Review date: 2026-08-01 (America/Los_Angeles)

Scope: source bootstrap, Python/Rust packaging, lock files, CI, UI installation, Docker smoke, generated-artifact retention, dependency licenses and credential hygiene on a macOS arm64 host without NVIDIA hardware or cloud credentials. The CPU demo itself was deliberately not rerun during this review because the integration owner was regenerating it concurrently.

## Outcome

The credential-free source bootstrap, Python wheel/sdist, Python/Rust checks, CPU-safe GPU status command and Linux Docker streaming smoke pass after the fixes below. A wheel installed into an independent virtual environment can generate and validate a seeded trace. The Docker test builds the gateway from a fresh Linux build stage, starts a mock backend and gateway, waits for both health checks, and observes the streamed `[DONE]` marker.

The two release-process caveats found during this review were resolved before publication: source revision `2c889e6956ac73a1a530f1abb1d7407f70219ffe` was frozen and clean-archive bootstrapped, and the regenerated repository-relative CPU evidence was retained. The subsequent release gates are recorded in `docs/reviews/core/CLEAN_REVISION_AUDIT.md` and `docs/reports/FINAL_REPORT.md`.

## Concrete fixes applied

- Made `make bootstrap` fail with actionable prerequisite messages, synchronize Python with `--locked`, and use a checkout-local npm cache. The original bootstrap failed because the user's global npm cache contained root-owned files; the second and subsequent runs succeeded without changing global permissions.
- Added `--locked` to every Make-driven Python execution so Make targets cannot silently rewrite `uv.lock`.
- Added a `make package` target that builds both the Python sdist and wheel from the sdist.
- Added Python package authorship, URLs, classifiers, SPDX metadata and `py.typed`; verified the wheel contains the Apache license and typing marker.
- Completed the standard Apache-2.0 `LICENSE` text, whose appendix was absent.
- Added Rust crate descriptions/repository metadata and version requirements for internal path dependencies, making publish order and package metadata explicit.
- Pinned Docker frontend and base images by digest. The compose smoke retains health checks, read-only filesystems and a non-root UID.
- Pinned every GitHub Action to a commit, disabled persisted checkout credentials, added a Python 3.11/3.12 package smoke matrix on Ubuntu/macOS, and retained demo evidence/reports as a 14-day CI artifact.
- Added ignores for `.env*`, private key files, local npm cache, Playwright inspection output and TypeScript incremental-build state.
- Corrected clean-checkout prerequisites and command flags in the README/reproducibility/optimizer documentation. Bare `sloforge` examples now have an explicit environment-activation instruction, and the report example points at the complete demo evidence index it actually consumes.
- Added the CPU-host GPU benchmark check: it records `status=unavailable`, an empty measurement list and reproduction commands instead of fabricated performance numbers.

## Commands and results

| Command | Result |
|---|---|
| `make bootstrap` (initial) | Failed at `npm ci` with `EACCES` in the global `~/.npm` cache; this directly motivated the local-cache fix. Python lock resolution and Rust build had succeeded. |
| `make bootstrap` (after fix) | Passed twice. `uv sync --locked --extra dev --extra deploy`, `cargo fetch/build --locked`, and `npm ci` all completed; npm audited 232 packages with 0 vulnerabilities. |
| `uv lock --check` | Passed; 285 locked package records resolved without changing the lock. |
| `uv build` / `make package` | Passed. Produced `sloforge-0.1.0.tar.gz` and a wheel built from that sdist. The sdist contains the monorepo source and excludes environments, demo build output and TypeScript build state. |
| independent wheel install and `sloforge trace generate/validate --seed 7 --count 8` | Passed in `/tmp/sloforge-cleanroom-venv`; 28 runtime packages installed and the eight-record trace validated. |
| `uv run --locked ruff format --check python tests` | Passed: 57 files formatted. |
| `uv run --locked ruff check python tests` | Passed. |
| `uv run --locked mypy python/sloforge` | Passed in strict mode: 43 source files. |
| `uv run --locked pytest -q` | Passed: 64 tests; 3 GPU/Torch tests skipped because Torch is not installed. |
| `cargo fmt --all --check` | Passed. |
| `cargo clippy --workspace --all-targets --all-features --locked -- -D warnings` | Passed. |
| `cargo test --workspace --all-features --locked` | Passed: 62 Rust tests across gateway, loadgen, protocol, simulator and telemetry; no failures. |
| `make docker-smoke` | Passed. Digest-pinned Linux images built, both containers became healthy, `/health` succeeded and a real SSE request reached `[DONE]`; the trap removed containers, network and volumes. |
| `make benchmark-gpu` | Passed its CPU-host behavior: `reports/gpu/evaluation.json` says `unavailable`, reports no devices and contains zero measurements. |
| `npm audit --prefix ui --audit-level=high` | Passed with 0 vulnerabilities. |
| `uvx --from pip-audit==2.10.1 pip-audit` over the locked runtime plus deploy extras | Passed: no known vulnerabilities found. The pinned scheduled CI job repeats this against the current advisory service. |
| workflow YAML parse | Both CI files parsed and contained the expected jobs. |
| credential-pattern scan | No AWS, GitHub, OpenAI, Slack token or private-key pattern found outside ignored/generated dependency trees. The only generic token reference is GitHub's ephemeral `${{ secrets.GITHUB_TOKEN }}` in the pinned RustSec action. |

The UI's individual typecheck/lint/test/build result is recorded after the UI owner finishes its concurrent patch; the initial review correctly caught a missing root `check` script and an ES library mismatch for `Array.toSorted` and sent both concrete failures to that owner.

## Dependency-license review

- Cargo metadata enumerated 234 unique name/license rows. None lacked license metadata. No mandatory strong-copyleft dependency was found; `r-efi` offers `MIT OR Apache-2.0 OR LGPL-2.1-or-later`, so the permissive alternative applies.
- npm enumerated 179 installed package rows. The only missing license field was the SLOForge UI root package itself; its owner was asked to add `Apache-2.0`. No installed third-party Node package lacked a license field or declared a strong-copyleft license.
- The synchronized Python environment contained 114 distributions. Ten had incomplete primary metadata fields; installed license files/classifiers resolve them as BSD (Jinja2, prompt-toolkit), Apache-2.0 (google-crc32c, synchronicity), MIT (httpx-ws, jaraco.classes, loguru, markdown-it-py, mdurl) and MPL-2.0 (pathspec). NumPy is BSD-3-Clause and its binary metadata discloses GCC runtime libraries under the GCC Runtime Library Exception plus libquadmath under LGPL; these are bundled runtime components, not SLOForge source relicensing.
- Direct deploy dependencies are Modal 1.5.3 (Apache-2.0) and Truss 0.18.24 (MIT). Model weights and engine packages remain separately licensed inputs and are not installed by the default environment.

This is an engineering compatibility review, not legal advice. A release should preserve generated SBOM/license output and repeat the review whenever either lock changes.

## Remaining issues and boundaries

1. **Resolved: immutable source revision.** Revision `2c889e6956ac73a1a530f1abb1d7407f70219ffe` was archived and `make bootstrap` passed from the clean archive. The final evidence bundle names that exact revision.
2. **Resolved: generated evidence retention.** The regenerated `artifacts/demo` and `reports/demo` trees are repository-relative, digest verified, secret/path scanned, and retained. Stale pre-hardening output is excluded.
3. **The Python wheel is not a native-binary bundle.** Pure-Python trace/schema/compiler features work independently, but `serve` and simulator replay need `sloforge-gateway`/`sloforge-sim` on `PATH` or a source checkout with Cargo. The sdist includes the Rust workspace, and the source quickstart is the supported complete installation. Publishing platform wheels with Rust binaries is future release engineering, not silently claimed here.
4. **Initial Rust publication has an order.** Protocol, simulator and telemetry crates package locally. Gateway/loadgen manifests are package-ready but Cargo cannot resolve their internal versioned dependencies from crates.io until telemetry/simulator are published first. This is expected for a first multi-crate release and should be automated if crates.io publication becomes a goal.
5. **Docker OS packages are not snapshot-repository pinned.** Base images are digest-pinned and Cargo is locked, but `apt-get` installs current `ca-certificates`/`curl` from Debian repositories. For bit-for-bit rebuild claims, replace this with a dated Debian snapshot or a distroless health probe; current claims are functional reproducibility, not bit identity.
6. **Optional paths remain intentionally unexercised.** No NVIDIA GPU, Modal/Baseten credentials, paid resources, Helm cluster or real model download was used. Offline exporter tests and the no-measurement GPU status artifact cover those boundaries without presenting them as runtime validation.
