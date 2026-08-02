# SLOForge Fabric clean-room and packaging review

Review date: 2026-08-02 (America/Los_Angeles)

Audited source revision: `606680ff79e99cd051c5568c0f9b763aeb8fc4d4`

Host: macOS 15.6.1 arm64, Apple M4 Pro; Python selected by `uv` 3.12.7,
`uv` 0.10.2, Rust/Cargo 1.93.1, Node 22.16.0, npm 11.4.2. The host has
no NVIDIA GPU, RDMA fabric, or authorized cloud budget.

Revisions `ce704da` and `a98685b` landed after the source archive was created.
The former is documentation-only; the latter expands the repository clean-room
script. Neither is silently included in these results. The final release owner
must run that expanded gate after publishing the regenerated evidence.

## Outcome

The integrated source bootstraps, packages, runs the complete physical Fabric
demo, and passes the Python/Rust/UI suites after its demo evidence is generated.
The public physical simulator, standalone seven-scenario Autopsy replay, and
fail-closed validation contract all work from that regenerated bundle. An
independently installed wheel also runs trace, local topology discovery, and CPU
hardware-probe commands without importing workspace source.

However, revision `606680f` is **not a passing clean checkout as committed**.
Its checked-in `artifacts/fabric-demo` bundle predates the new artifact contract.
An initial `make check`, executed before any generation step, failed three tests:

- `metrics/degraded.prom` lacked newly required physical Fabric metrics;
- `mixed-bursty.jsonl` still used `arrival_us` and string priorities rather than
  the canonical `TraceRequest` schema;
- fail-closed validation therefore stopped at trace parsing instead of writing
  its structured `validation.json` result.

That initial run reported 347 passing Python tests, 3 failures, and 3 legitimate
GPU/Torch skips. Running `make fabric-demo` regenerated the missing evidence;
the same tests then passed, followed by the full suite. This isolates the defect
to evidence publication, but does not waive it: the regenerated artifacts and
report must be committed before the revision can be called clean-room passing.

## Isolation method

The tree was exported without Git metadata or working-tree files:

```bash
review_root=$(mktemp -d /tmp/sloforge-final-cleanroom.XXXXXX)
git archive 606680ff79e99cd051c5568c0f9b763aeb8fc4d4 | tar -x -C "$review_root"
test ! -e "$review_root/.git"
```

The archive began without `.venv`, `target`, `ui/node_modules`, or `ui/dist`.
Each was created inside the temporary tree. The wheel smoke used another virtual
environment under `/tmp` and ran outside the source archive. Shared download
caches could provide immutable archives, so this is a clean-checkout test, not
an air-gapped installation test.

## Commands and results

| Command | Result |
|---|---|
| `make bootstrap` | **PASS**. Locked `uv` synchronization installed 114 distributions; locked Cargo built all seven workspace crates; `npm ci` installed 231 packages and reported zero vulnerabilities. |
| `make check` before demo generation | **FAIL** as described above: 347 Python passed, 3 artifact-contract tests failed, and 3 GPU tests skipped. Ruff and strict mypy had passed before pytest stopped the target. |
| `make fabric-demo` | **PASS**. Generated a canonical trace, 588-operation request simulations, hashed profiler evidence, seven counterfactuals, structured telemetry, runtime evidence, and a derived report. |
| `pytest -q tests/python/test_fabric_demo.py` after generation | **PASS**, 4 tests including standalone replay and fail-closed validation. |
| `make fabric-check` after generation | **PASS**. Ruff, strict mypy over 118 source files, 280 Python tests, Fabric Rust fmt/clippy with warnings denied, and 31 Rust tests passed. |
| `make check` after generation | **PASS**. Ruff format/lint covered 155 files; strict mypy covered 118 source files; pytest reported 350 passed and 3 GPU/Torch skips; Rust workspace fmt/clippy and all 98 tests passed; UI typecheck/lint, 28 tests, and production build passed. |
| public `sloforge fabric simulate` | **PASS**. The public CLI loaded plan/topology/profile/canonical trace and emitted 588 operations, 1,523 processed events, 1775.586 ms makespan, and its result artifact. |
| standalone `sloforge autopsy replay` | **PASS**. Replay resolved and hash-verified its self-contained evidence bundle, evaluated 7 scenarios, selected `remove-both-faults`, and recorded all 6 rejected alternatives. |
| public `sloforge fabric validate` | **EXPECTED FAIL-CLOSED**. Exited 1 and still wrote `validation.json`; it rejected relative error 14.430933 over the 0.25 limit and an observed p95 TTFT outside the plan interval. |
| `uv lock --check` | **PASS**. The checked lock resolved 285 records without mutation. |
| `cargo metadata --locked --offline --format-version 1` | **PASS**. Cargo resolved exclusively from `Cargo.lock` and the local cache. |
| `npm audit --prefix ui --audit-level=high` | **PASS**, zero known vulnerabilities in the installed UI tree. |
| `make package` | **PASS**. Built a 373 KiB pure-Python wheel and 17 MiB source distribution. The wheel contains 124 files, `py.typed`, console entry point, and Apache-2.0 license; the sdist carries the full monorepo build inputs. |
| independent wheel install and CLI smoke | **PASS**. Installed 28 runtime distributions; `sloforge --help`, a 37-record seeded trace roundtrip, `fabric discover`, and the CPU hardware probe exited zero. |

The installed-wheel trace reproduced SHA-256
`9d3648041209489211456e4c579151e1807de35c5e36c4e0ee64c441172191d5`.
Wheel-based local discovery reported 53 typed nodes and 26 edges and did not
fabricate an NVIDIA device.

The regenerated CPU demo recorded healthy/degraded/restored p95 TTFT values of
1152.714/7045.952/1152.714 ms under its 1267.985 ms synthetic SLO. It diagnosed
`network_bandwidth_degradation`, retained both injected ground-truth faults,
evaluated seven repairs, selected removal of both physical faults, and completed
the guarded recovery state machine. These are calibrated deterministic synthetic
outputs, not measurements of this Apple host as a GPU fabric.

## Lock and dependency-license inventory

All lockfiles were exercised, not merely checked for existence:

- `uv.lock`: SHA-256 `44d926eb3b8da5765b5817d50570c336faf2ffc90d4382fd56fb68b2a88ba5f4`;
- `Cargo.lock`: SHA-256 `ccf749c5eb3a2aefcdbf64547f24ca19467a625763cd5370a47979a26b983e80`;
- `ui/package-lock.json`: SHA-256
  `a4fa86fb1607ee8749d9ba323462885fc43b78f4814c51bdb4379a19cc2c7a6c`.

Cargo metadata enumerated 244 packages and none lacked manifest license metadata.
Seven local crates inherit Apache-2.0 from the workspace. Declarations use
permissive choices/combinations; optional LGPL alternatives are paired with MIT
or Apache-2.0, and no mandatory strong-copyleft Cargo dependency was found.

The npm lock enumerated 231 dependency rows and every row declared a license;
npm's live audit reported zero known vulnerabilities. The synchronized Python
environment contained 113 third-party distributions. Only
`google-crc32c==1.8.0` omitted a primary metadata field; its installed wheel
contains the complete Apache-2.0 text under its `.dist-info/licenses` directory.

The repository `LICENSE` is Apache-2.0 with SHA-256
`632f5eaeebffe4ccfaa0a68750baa65016ebd82d69a31cbea0eda7b7cf878d70`.
This is an engineering inventory, not legal advice. Model weights and optional
engine/GPU extras require separate review before redistribution.

## Findings and residual boundaries

1. **Open release blocker — stale committed Fabric demo evidence.** The source
   regenerates valid evidence and then passes all tests, but `make check` fails
   in the literal committed archive before generation. Publish the regenerated
   bundle and rerun the archive gate at that exact revision.
2. **Fixed in prior audit — WarmPath interval invariant.** Revision `62f5adf`
   clamps a bootstrap interval around its own median, with a deterministic ULP
   regression. The final source retains the fix and all 11 WarmPath tests pass.
3. **Source checkout is the complete distribution.** The wheel contains pure
   Python. Gateway and Rust-simulator launch commands require their binaries on
   `PATH` or a source build; the sdist includes those sources and locks.
4. **No Docker validation in this review.** Docker status must come from the
   integration owner's final gate.
5. **No GPU, RDMA, multi-node hardware, privileged fault, cloud, Modal runtime,
   or Truss runtime validation occurred.** CPU-only synthetic execution and
   actual local CPU/topology probes do not validate NVIDIA/NCCL/NVLink/IB/RoCE.
6. **Platform scope.** This audit ran on one macOS arm64 host. It does not infer
   success on environments that were not exercised.
7. **Advisory scope.** npm's live audit was queried. No equivalent live Python
   or Rust advisory service was queried during this bounded rerun.

Until finding 1 is resolved and the final committed revision is re-exported and
tested, SLOForge Fabric must not claim a passing clean-room checkout.
