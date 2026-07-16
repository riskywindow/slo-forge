#!/usr/bin/env bash
set -euo pipefail

repository_root="$(git rev-parse --show-toplevel)"
revision="$(git -C "$repository_root" rev-parse HEAD)"
source_tree="$(git -C "$repository_root" rev-parse "${revision}^{tree}")"
clean_root="$(mktemp -d "${TMPDIR:-/tmp}/sloforge-continuum-clean.XXXXXX")"
result_root="${SLOFORGE_CONTINUUM_CLEANROOM_RESULT_ROOT:-$repository_root/artifacts/continuum/clean-room}"
mkdir -p "$result_root"
rm -f -- "$result_root/run.log" "$result_root/result.json"
cleanup() {
  exit_status=$?
  if test "$exit_status" -ne 0 && test -f "$clean_root/continuum-clean-room.log"; then
    cp "$clean_root/continuum-clean-room.log" "$result_root/run.log"
    python3 - "$result_root/result.json" "$revision" "$source_tree" "$result_root/run.log" <<'PY'
import hashlib
import json
import pathlib
import sys

output = pathlib.Path(sys.argv[1])
log = pathlib.Path(sys.argv[4])
output.write_text(
    json.dumps(
        {
            "schema_version": "sloforge.continuum.clean-room/v1",
            "status": "failed",
            "revision": sys.argv[2],
            "source_tree": sys.argv[3],
            "log_sha256": hashlib.sha256(log.read_bytes()).hexdigest(),
        },
        indent=2,
        sort_keys=True,
    )
    + "\n"
)
PY
  fi
  rm -rf -- "$clean_root"
  return "$exit_status"
}
trap cleanup EXIT INT TERM

git -C "$repository_root" archive "$revision" | tar -x -C "$clean_root"
unset PYTHONPATH PYTHONHOME VIRTUAL_ENV UV_PROJECT UV_PROJECT_ENVIRONMENT UV_WORKING_DIR
unset CARGO_BUILD_TARGET RUSTC_WRAPPER RUSTFLAGS CARGO_ENCODED_RUSTFLAGS MAKEFLAGS MFLAGS
unset SLOFORGE_CONTINUUM_GPU_BUDGET_USD
export UV_PROJECT_ENVIRONMENT="$clean_root/.venv"
export CARGO_TARGET_DIR="$clean_root/target"
export SLOFORGE_CONTINUUM_ALLOW_GPU=0
export SLOFORGE_CONTINUUM_ALLOW_MULTI_NODE=0
export SLOFORGE_CONTINUUM_ALLOW_RDMA=0
export SLOFORGE_CONTINUUM_ALLOW_NETWORK_FAULTS=0
export SLOFORGE_CONTINUUM_ALLOW_EXTERNAL_DEPLOYMENT=0
export SLOFORGE_CONTINUUM_ALLOW_LIVE_MIGRATION=0

cd "$clean_root"
{
  make bootstrap
  make continuum-check
  make continuum-migration-demo
  make continuum-fault-demo
  make continuum-fork-demo
  make continuum-compatibility-demo
  make package
} 2>&1 | tee "$clean_root/continuum-clean-room.log"

python3 - "$clean_root" "$revision" "$source_tree" <<'PY'
import hashlib
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
revision = sys.argv[2]
source_tree = sys.argv[3]
verification = json.loads((root / "artifacts/continuum/migration-demo/verification.json").read_text())
fault = json.loads((root / "artifacts/continuum/fault-demo/fault-manifest.json").read_text())
fork = json.loads((root / "artifacts/continuum/fork-demo/fork.json").read_text())
compatibility = json.loads((root / "artifacts/continuum/compatibility-demo/compatibility.json").read_text())
assert verification["valid"] is True
assert fault["final_phase"] == "ROLLED_BACK"
assert len(fork["branches"]) == 2
assert compatibility["direct_reuse"]["compatibility_class"] == "incompatible"
log = root / "continuum-clean-room.log"
result = {
    "schema_version": "sloforge.continuum.clean-room/v1",
    "status": "passed",
    "revision": revision,
    "source_tree": source_tree,
    "log_sha256": hashlib.sha256(log.read_bytes()).hexdigest(),
    "continuum_check": "passed",
    "migration_demo": "passed",
    "fault_demo": "passed",
    "fork_demo": "passed",
    "compatibility_demo": "passed",
    "package_build": "passed",
}
(root / "continuum-clean-room-result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
PY
cp "$clean_root/continuum-clean-room.log" "$result_root/run.log"
cp "$clean_root/continuum-clean-room-result.json" "$result_root/result.json"
echo "Continuum clean-room validation passed for $revision"
