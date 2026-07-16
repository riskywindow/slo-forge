#!/usr/bin/env bash
set -euo pipefail

repository_root="$(git rev-parse --show-toplevel)"
revision="$(git -C "$repository_root" rev-parse HEAD)"
source_tree="$(git -C "$repository_root" rev-parse "${revision}^{tree}")"
clean_root="$(mktemp -d "${TMPDIR:-/tmp}/sloforge-genesis-clean.XXXXXX")"
result_root="${SLOFORGE_GENESIS_CLEANROOM_RESULT_ROOT:-$repository_root/artifacts/genesis/clean-room}"
case "$result_root" in
  /*) ;;
  *) result_root="$repository_root/$result_root" ;;
esac
cleanup() {
  exit_status=$?
  if test "$exit_status" -ne 0 && test -f "$clean_root/genesis-clean-room.log"; then
    cp "$clean_root/genesis-clean-room.log" "$result_root/.run.log.failed"
    mv "$result_root/.run.log.failed" "$result_root/run.log"
  fi
  rm -rf -- "$clean_root"
  return "$exit_status"
}
trap cleanup EXIT INT TERM

# A previous successful result must never survive a failed current attempt as the
# apparent latest result. The authoritative result is replaced atomically only
# after every check below succeeds.
mkdir -p "$result_root"
python3 - "$result_root/result.json" "$revision" "$source_tree" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
temporary = path.with_name(f".{path.name}.running")
temporary.write_text(
    json.dumps(
        {
            "revision": sys.argv[2],
            "source_tree": sys.argv[3],
            "status": "running",
        },
        indent=2,
        sort_keys=True,
    )
    + "\n",
    encoding="utf-8",
)
temporary.replace(path)
PY

git -C "$repository_root" archive "$revision" | tar -x -C "$clean_root"
printf '%s\n' "$revision" > "$clean_root/.sloforge-source-commit"
printf '%s\n' "$source_tree" > "$clean_root/.sloforge-source-tree"

# Do not let an activated environment, caller PYTHONPATH, external Cargo target,
# or production Genesis opt-in change what the archived source builds or runs.
unset PYTHONPATH PYTHONHOME VIRTUAL_ENV UV_PROJECT UV_PROJECT_ENVIRONMENT UV_WORKING_DIR
unset CARGO_BUILD_TARGET RUSTC_WRAPPER RUSTFLAGS CARGO_ENCODED_RUSTFLAGS MAKEFLAGS MFLAGS
unset SLOFORGE_GENESIS_GPU_BUDGET_USD SLOFORGE_GENESIS_SYNTHESIS_BUDGET_USD
export UV_PROJECT_ENVIRONMENT="$clean_root/.venv"
export CARGO_TARGET_DIR="$clean_root/target"
export SLOFORGE_GENESIS_ALLOW_EXTERNAL_SYNTHESIS=0
export SLOFORGE_GENESIS_ALLOW_GPU=0
export SLOFORGE_GENESIS_ALLOW_MULTI_NODE=0
export SLOFORGE_GENESIS_ALLOW_PRIVILEGED_PROBES=0
export SLOFORGE_GENESIS_ALLOW_EXTERNAL_DEPLOYMENT=0
export SLOFORGE_GENESIS_ALLOW_LIVE_PROMOTION=0

cd "$clean_root"
{
  make bootstrap
  make genesis-check
  make genesis-demo
  make synthbench-smoke
  make package

  # The seeded demo uses a historical deterministic timestamp, so a real
  # current-time validator must reject that capsule as stale. Rebuild the same
  # accepted candidate with a fresh evidence horizon for the installed-wheel
  # validation instead of weakening the production freshness check.
  accepted_candidate_id="$($clean_root/.venv/bin/python -c \
    'import json, sys; print(json.load(open(sys.argv[1]))["accepted_candidate_id"])' \
    "$clean_root/artifacts/genesis/demo/GENESIS_DEMO_REPORT.json")"
  fresh_timestamp="$($clean_root/.venv/bin/python -c \
    'from datetime import UTC, datetime; print(datetime.now(UTC).isoformat())')"
  fresh_capsule="$clean_root/artifacts/genesis/wheel-capsule"
  fresh_context="$clean_root/artifacts/genesis/wheel-capsule.validation-context.json"
  fresh_build="$clean_root/artifacts/genesis/wheel-capsule-build.json"
  "$clean_root/.venv/bin/sloforge" genesis capsule build \
    --candidate "$clean_root/artifacts/genesis/demo/run/candidates/$accepted_candidate_id" \
    --output "$fresh_capsule" \
    --trust-output "$fresh_context" \
    --timestamp "$fresh_timestamp" > "$fresh_build"

  wheel_count="$(find "$clean_root/dist" -maxdepth 1 -type f -name '*.whl' -print | wc -l | tr -d '[:space:]')"
  test "$wheel_count" = 1
  wheel_path="$(find "$clean_root/dist" -maxdepth 1 -type f -name '*.whl' -print)"

  # Install the wheel in a fresh environment whose runtime dependencies come
  # from the checked lock. Reusing .venv would allow its editable checkout to
  # mask missing wheel files.
  runtime_requirements="$clean_root/.wheel-runtime-requirements.txt"
  uv export --locked --no-dev --no-emit-project --no-annotate --no-header \
    --output-file "$runtime_requirements"
  uv venv --python "$clean_root/.venv/bin/python" "$clean_root/.wheel-venv"
  uv pip sync --python "$clean_root/.wheel-venv/bin/python" "$runtime_requirements"
  uv pip install --python "$clean_root/.wheel-venv/bin/python" --no-deps "$wheel_path"

  mkdir -p "$clean_root/.wheel-smoke"
  cd "$clean_root/.wheel-smoke"
  "$clean_root/.wheel-venv/bin/python" - "$clean_root/.wheel-venv" <<'PY'
import pathlib
import sys
import sysconfig

import sloforge

venv = pathlib.Path(sys.argv[1]).resolve()
module = pathlib.Path(sloforge.__file__).resolve()
purelib = pathlib.Path(sysconfig.get_paths()["purelib"]).resolve()
assert purelib.is_relative_to(venv), (purelib, venv)
assert module.is_relative_to(purelib), (module, purelib)
PY
  "$clean_root/.wheel-venv/bin/sloforge" --help >/dev/null

  capsule_path="$($clean_root/.venv/bin/python -c \
    'import json, sys; print(json.load(open(sys.argv[1]))["capsule_path"])' \
    "$fresh_build")"
  capsule_digest="$($clean_root/.venv/bin/python -c \
    'import json, sys; print(json.load(open(sys.argv[1]))["capsule_digest"])' \
    "$fresh_build")"
  "$clean_root/.wheel-venv/bin/sloforge" genesis capsule validate "$capsule_path" \
    --context "$fresh_context" \
    --expected-digest "$capsule_digest" \
    > "$clean_root/artifacts/genesis/wheel-capsule-validation.json"
} 2>&1 | tee "$clean_root/genesis-clean-room.log"

wheel_path="$(find "$clean_root/dist" -maxdepth 1 -type f -name '*.whl' -print)"
evidence_bundle="$clean_root/artifacts/genesis/clean-room-evidence.tar.gz"
tar -czf "$evidence_bundle" -C "$clean_root" \
  artifacts/genesis/demo/GENESIS_DEMO_REPORT.json \
  artifacts/genesis/demo/capsule.validation-context.json \
  artifacts/genesis/demo/capsule \
  artifacts/genesis/wheel-capsule-build.json \
  artifacts/genesis/wheel-capsule.validation-context.json \
  artifacts/genesis/wheel-capsule-validation.json \
  artifacts/genesis/wheel-capsule \
  artifacts/synthbench/smoke
"$clean_root/.wheel-venv/bin/python" "$clean_root/tools/validate-clean-room-genesis.py" \
  --root "$clean_root" \
  --revision "$revision" \
  --source-tree "$source_tree" \
  --log "$clean_root/genesis-clean-room.log" \
  --wheel "$wheel_path" \
  --evidence-bundle "$evidence_bundle" \
  --output "$clean_root/artifacts/genesis/clean-room-result.json"

# Publish retained evidence and the log first, then the digest-binding result
# last. Readers therefore either see status=running or a complete result whose
# hashes can be checked and whose capsule can be replayed after extraction.
cp "$evidence_bundle" "$result_root/.evidence.tar.gz.tmp"
mv "$result_root/.evidence.tar.gz.tmp" "$result_root/evidence.tar.gz"
cp "$clean_root/genesis-clean-room.log" "$result_root/.run.log.tmp"
mv "$result_root/.run.log.tmp" "$result_root/run.log"
cp "$clean_root/artifacts/genesis/clean-room-result.json" "$result_root/.result.json.tmp"
mv "$result_root/.result.json.tmp" "$result_root/result.json"
echo "Genesis clean-room validation passed for $revision"
