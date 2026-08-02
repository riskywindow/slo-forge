# Clean revision audit

Audit date: 2026-08-01 (America/Los_Angeles)

Audited revision: `d5d683925de35ab7d823afa06ebb3179e73ab9ff`

## Scope and method

This audit examines the literal committed tree at the revision above, not the
working directory. The working directory contained later modified and generated
files when this audit ran. They were excluded by exporting the revision with:

```bash
audit_dir=$(mktemp -d /tmp/sloforge-clean-d5d6839.XXXXXX)
git archive HEAD | tar -x -C "$audit_dir"
```

The resulting archive was `/tmp/sloforge-clean-d5d6839.RGiwtU`. It contained no
`.git` directory or pre-existing `.venv`, `target`, `ui/node_modules`, or
`ui/dist`. Python environments, the Cargo target directory, the npm cache and UI
dependencies used below were created inside that temporary archive. Global
download caches may still have supplied immutable dependency archives; no
environment or build output from the source working directory was reused.

## Commands and results

### Locked Python bootstrap, package, and isolated import

Executed from the archive root:

```bash
uv sync --locked --extra dev --extra deploy
uv build --out-dir audit-dist
uv venv --python python3 .wheel-venv
uv pip install --python .wheel-venv/bin/python \
  audit-dist/sloforge-0.1.0-py3-none-any.whl
.wheel-venv/bin/python -c \
  'import sloforge, sloforge.cli.main; print("IMPORT_OK", sloforge.__file__)'
```

Result: **PASS** (all commands exited 0).

- `uv sync --locked` created a new archive-local `.venv`, resolved the checked
  lock in 0.81 ms, built SLOForge and installed 114 packages. It explicitly
  ignored an unrelated active `VIRTUAL_ENV` because it did not match the local
  project environment.
- `uv build` produced both
  `sloforge-0.1.0.tar.gz` and
  `sloforge-0.1.0-py3-none-any.whl`.
- The wheel installed into a second new environment with 28 runtime packages.
- The import resolved to
  `/private/tmp/sloforge-clean-d5d6839.RGiwtU/.wheel-venv/lib/python3.12/site-packages/sloforge/__init__.py`,
  demonstrating that the installed wheel, rather than workspace source, was
  imported.

### Deterministic trace roundtrip

Executed with the installed wheel's console entry point:

```bash
mkdir -p .audit
.wheel-venv/bin/sloforge trace generate \
  --seed 713 --count 37 --output .audit/trace.jsonl
.wheel-venv/bin/sloforge trace validate .audit/trace.jsonl
wc -l .audit/trace.jsonl
shasum -a 256 .audit/trace.jsonl
```

Result: **PASS** (all commands exited 0).

- Generation and validation independently reported 37 requests, duration
  4706.902 ms, mean rate 7.860796761861623 requests/s and peak one-second rate
  14 requests/s.
- Both summaries reported the same priority counts (`0`: 10, `1`: 20, `2`: 7)
  and request-class counts (`interactive`: 30, `long-context`: 7).
- The file contained exactly 37 JSONL records.
- SHA-256:
  `9d3648041209489211456e4c579151e1807de35c5e36c4e0ee64c441172191d5`.

### Locked Rust workspace build

```bash
CARGO_TARGET_DIR=.audit-target cargo build --workspace --locked
```

Result: **PASS** (exit 0). Cargo compiled all five workspace crates and finished
the development build in 20.51 seconds using a fresh archive-local target tree.

### Lockfile-clean UI build

```bash
npm ci --prefix ui --cache .audit-npm-cache
npm run build --prefix ui
```

Result: **PASS** (both commands exited 0).

- `npm ci` installed 231 packages from `package-lock.json` and reported zero
  vulnerabilities.
- TypeScript and Vite production build passed.
- Output sizes were 1.03 kB HTML (0.60 kB gzip), 15.82 kB CSS (4.41 kB gzip)
  and 38.19 kB JavaScript (12.38 kB gzip).

These Python, Rust and Node commands independently cover the substantive steps
behind `make bootstrap`: locked Python synchronization, locked Cargo resolution
and build, and `npm ci`. The literal `make bootstrap` wrapper was not invoked a
second time after those steps.

## Source path and secret scan

The scan operated only on the 174 exported source files. It pruned every
environment or build-output directory created by this audit:

```bash
find . \
  \( -path './.venv' -o -path './.wheel-venv' \
     -o -path './.audit-target' -o -path './.audit-npm-cache' \
     -o -path './ui/node_modules' -o -path './ui/dist' \
     -o -path './audit-dist' -o -path './.audit' \) -prune \
  -o -type l -print

rg --hidden -l -I \
  '(-----BEGIN (RSA |EC |OPENSSH )?PRIVATE KEY-----|AKIA[0-9A-Z]{16}|gh[pousr]_[A-Za-z0-9]{30,}|xox[baprs]-[A-Za-z0-9-]{20,}|sk-[A-Za-z0-9]{20,})' \
  . <the same generated-directory exclusions>

rg --hidden -l -I \
  '(/Users/[A-Za-z0-9._-]+/|/home/[A-Za-z0-9._-]+/|[A-Za-z]:\\\\Users\\\\)' \
  . <the same generated-directory exclusions>
```

Results:

- Source symlinks: **0**. There are therefore no source symlink targets capable
  of escaping the archive root.
- Files matching the high-confidence private-key/cloud-token patterns: **0**.
- Files containing macOS, Linux or Windows developer-absolute paths: **0**.
- The only committed files under generated-result-like roots were
  `reports/gpu/commands.txt`, `reports/gpu/evaluation.json`, and
  `reports/gpu/evaluation.md`. No `artifacts/` files were present in the exported
  revision.

This is a high-confidence pattern scan, not a proof that arbitrary prose cannot
encode a credential in an unrecognized format.

## Post-review source revision addendum

After the adversarial fixes were committed, the integration owner repeated the literal bootstrap from a second clean archive of source revision `2c889e6956ac73a1a530f1abb1d7407f70219ffe`:

```bash
audit_dir=$(mktemp -d /tmp/sloforge-final-audit.XXXXXX)
git archive 2c889e6956ac73a1a530f1abb1d7407f70219ffe | tar -x -C "$audit_dir"
cd "$audit_dir"
make bootstrap
```

Result: **PASS** (exit 0). The clean archive created a new locked Python environment and installed 114 packages, built all five Rust workspace crates from `Cargo.lock` in a fresh target directory, and installed 231 UI packages from `package-lock.json`; npm reported zero vulnerabilities. This addendum covers the exact source revision embedded in the final evidence bundle. The full lint/test/demo/Docker matrix was subsequently executed in the integration checkout and is recorded in `FINAL_REPORT.md`.

The earlier audit remains useful because it additionally exercised isolated wheel/sdist installation, trace roundtrip, source-only secret/path scanning, and the production UI build. No source manifest or packaging regression was introduced between the audited revisions.

## Limitations and conclusion

The clean archive was not used to run the full test/lint matrix, CPU demo,
Docker smoke test, GPU path or cloud deployment commands. Those are outside this
bounded clean-revision check. The wheel's runtime dependencies were resolved in
the isolated installation from the package metadata; the development bootstrap
itself was separately constrained by `uv sync --locked`.

Within the executed scope, revision `d5d6839` is independently bootstrapable and
packageable: locked Python synchronization, Python sdist/wheel creation,
installed-wheel import, deterministic trace generation/validation, locked Rust
workspace build and lockfile-clean UI production build all passed from a clean
archive. No conclusion in this document applies to commits or uncommitted source
changes made after `d5d6839`.
