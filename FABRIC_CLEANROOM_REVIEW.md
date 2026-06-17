# SLOForge Fabric clean-room and packaging review

Review date: 2026-08-02 (America/Los_Angeles)

Audited source revision: `62f5adf4c605a9ad0b73856300c058a027ab3978`

Host: macOS 15.6.1 arm64, Apple M4 Pro; Python selected by `uv` 3.12.7,
`uv` 0.10.2, Rust/Cargo 1.93.1, Node 22.16.0, npm 11.4.2. The host has
no NVIDIA GPU, RDMA fabric, or authorized cloud budget.

## Outcome

The committed revision bootstraps, packages, passes the complete Python/Rust/UI
quality suite, passes the Fabric-specific suite, and completes the CPU-only
Fabric demo from source-only archives. A wheel installed into an independent
virtual environment also runs the deterministic trace workflow, current-host
topology discovery, and CPU hardware probe without importing workspace source.

The first archived full check exposed one release-blocking nondeterministic
failure. Three repeated sub-millisecond samples caused linear percentile
interpolation to place a bootstrap bound one floating-point ULP above its own
median; `StageMeasurement` correctly rejected the contradictory interval. Commit
`62f5adf` clamps the reported interval around its point estimate and adds the
exact timer-sample regression. Targeted validation passed with 11 WarmPath tests,
and every clean-room command below was repeated against the fixed revision.

## Isolation method

The audited tree was exported without Git metadata or working-tree files:

```bash
review_root=$(mktemp -d /tmp/sloforge-cleanroom-review2.XXXXXX)
git archive 62f5adf4c605a9ad0b73856300c058a027ab3978 | tar -x -C "$review_root"
test ! -e "$review_root/.git"
```

The archive began without `.venv`, `target`, `ui/node_modules`, or `ui/dist`.
Each was created inside the temporary tree. The package smoke used a second
virtual environment under `/tmp` and ran from outside the source archive.
Download caches could supply immutable archives, so this validates a clean
checkout installation, not an air-gapped install.

## Commands and results

| Command | Result |
|---|---|
| `make bootstrap` | **PASS**. `uv sync --locked --extra dev --extra deploy` installed 114 distributions; `cargo fetch/build --workspace --locked` built all seven crates; `npm ci` installed 231 packages and reported zero vulnerabilities. |
| `make check` | **PASS**. Ruff format/lint passed for 153 files, strict mypy passed for 117 source files, pytest reported 305 passed and 3 GPU/Torch skips, Rust fmt and workspace clippy with `-D warnings` passed, 98 Rust tests passed, and UI typecheck/lint/21 tests/production build passed. |
| `make fabric-check` | **PASS**. The focused Python suite reported 235 passed; Fabric Rust fmt, clippy with warnings denied, conformance, analytical, property, CLI, two-node, and validation tests passed. |
| `make clean-room-test` | **PASS**. The target created another source-only archive, bootstrapped it, repeated `fabric-check`, regenerated the complete CPU-only Fabric demo, and asserted degraded-SLO failure plus restored-SLO success. It printed `clean-room validation passed for 62f5adf...`. |
| `uv lock --check` | **PASS**. The checked lock resolved 285 package records without mutation. |
| `cargo metadata --locked --offline --format-version 1` | **PASS**. Cargo resolved the workspace exclusively from `Cargo.lock` and the local cache. |
| `npm audit --prefix ui --audit-level=high` | **PASS**, zero known vulnerabilities in the installed UI tree. |
| `make package` | **PASS**. Built a 361 KiB pure-Python wheel and 16 MiB source distribution from the archive. The wheel contains 123 files, `py.typed`, console entry point, and Apache-2.0 license; the sdist contains the Rust workspace, all three locks, toolchain pin, UI, Makefile, and clean-room script. |
| independent wheel install | **PASS**. Installed 28 runtime distributions into `/tmp/sloforge-wheel-review.../venv`; `sloforge --help`, deterministic trace generation/validation, `fabric discover`, and CPU `hardware probe` all exited zero. |

The installed-wheel trace contained 37 records and reproduced SHA-256
`9d3648041209489211456e4c579151e1807de35c5e36c4e0ee64c441172191d5`.
Wheel-based topology discovery described the actual local host as 53 typed nodes
and 26 edges with zero fabricated GPU devices.

The clean-room demo compiled topology-aware and topology-unaware plans, exercised
the Rust gateway, injected network degradation plus rank-specific slowdown,
identified `network_bandwidth_degradation`, evaluated seven counterfactuals,
completed shadow/canary recovery, and restored p95 TTFT from 7047.976 ms to
1157.369 ms under its 1273.106 ms synthetic SLO. These are deterministic
synthetic demo outputs, not measurements of this Apple host as a GPU fabric.

## Lock and dependency-license inventory

The three lockfiles were present and exercised rather than only inspected:

- `uv.lock`: SHA-256 `44d926eb3b8da5765b5817d50570c336faf2ffc90d4382fd56fb68b2a88ba5f4`;
  locked synchronization and `uv lock --check` passed.
- `Cargo.lock`: SHA-256 `ccf749c5eb3a2aefcdbf64547f24ca19467a625763cd5370a47979a26b983e80`;
  locked online build and offline metadata resolution passed.
- `ui/package-lock.json`: SHA-256
  `a4fa86fb1607ee8749d9ba323462885fc43b78f4814c51bdb4379a19cc2c7a6c`;
  `npm ci`, audit, tests, and build passed.

Cargo metadata enumerated 244 packages. Every third-party manifest declared a
license; seven SLOForge workspace members inherit Apache-2.0 from the workspace.
The declarations are permissive choices/combinations including MIT, Apache-2.0,
BSD, ISC, Unicode-3.0, Zlib, CDLA-Permissive-2.0, and BSL-1.0. Two dependencies
offer LGPL-2.1-or-later as one alternative alongside MIT or Apache-2.0; no
mandatory strong-copyleft Cargo dependency was found.

The npm lock enumerated 231 installed dependency rows and every row contained a
license: 196 MIT, 14 Apache-2.0, 11 ISC, 7 BSD-2-Clause, 2 BSD-3-Clause, and 1
Python-2.0. The root UI package itself declares Apache-2.0 in `package.json`.

The synchronized Python environment contained 113 third-party distributions.
All but `google-crc32c==1.8.0` exposed a primary license field or classifier;
that wheel includes its complete Apache-2.0 text at
`google_crc32c-1.8.0.dist-info/licenses/LICENSE`. The inventory otherwise
reported permissive licenses plus MPL-2.0 components and NumPy's disclosed
binary-runtime exceptions. This is an engineering inventory, not legal advice;
model weights and optional engine/GPU extras require their own release review.

The repository `LICENSE` is Apache-2.0 with SHA-256
`632f5eaeebffe4ccfaa0a68750baa65016ebd82d69a31cbea0eda7b7cf878d70`.

## Findings and residual boundaries

1. **Fixed — WarmPath bootstrap interval invariant.** The one-ULP failure was a
   genuine host-timer-dependent clean-room defect. Commit `62f5adf` supplies the
   production fix and deterministic regression.
2. **Source checkout is the complete distribution.** The wheel intentionally
   contains pure Python only. Commands that launch the gateway or Rust simulator
   still require the corresponding binaries on `PATH` or a source build. The
   sdist contains those sources and locks; no platform-native wheel claim is
   made.
3. **No Docker result in this review.** `make docker-smoke` was not run here;
   Docker evidence must be taken from the integration owner's final gate.
4. **No GPU, RDMA, multi-node, privileged fault, cloud, Modal runtime, or Truss
   runtime validation occurred.** The CPU-only synthetic path and actual local
   CPU/topology probes passed. They do not validate NVIDIA/NCCL/NVLink/IB/RoCE
   execution.
5. **Platform scope.** This review executed on one macOS arm64 host. CI and
   Docker cover separate environments, but their status is not inferred from
   this run.
6. **Advisory freshness.** npm's live audit was queried. No equivalent live
   Python or Rust advisory service was queried during this bounded review; the
   pinned CI security workflow remains the authoritative recurring check.

No additional packaging fix is required by the exercised clean-room workflow.
Because other integration work may land after this audited revision, the final
release owner must rerun `make clean-room-test` on the exact source revision
recorded in `FABRIC_FINAL_REPORT.md`.
