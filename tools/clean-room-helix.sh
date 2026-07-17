#!/usr/bin/env bash
set -euo pipefail

repository_root="$(git rev-parse --show-toplevel)"
revision="$(git -C "$repository_root" rev-parse HEAD)"
source_tree="$(git -C "$repository_root" rev-parse "${revision}^{tree}")"
clean_root="$(mktemp -d "${TMPDIR:-/tmp}/sloforge-helix-clean.XXXXXX")"
result_root="${SLOFORGE_HELIX_CLEANROOM_RESULT_ROOT:-$repository_root/artifacts/helix/clean-room}"
mkdir -p "$result_root"
cleanup() {
  exit_status=$?
  if test "$exit_status" -ne 0 && test -f "$clean_root/helix-clean-room.log"; then
    cp "$clean_root/helix-clean-room.log" "$result_root/run.log"
  fi
  rm -rf -- "$clean_root"
  return "$exit_status"
}
trap cleanup EXIT INT TERM

git -C "$repository_root" archive "$revision" | tar -x -C "$clean_root"
printf '%s\n' "$revision" > "$clean_root/.sloforge-source-commit"
unset PYTHONPATH PYTHONHOME VIRTUAL_ENV UV_PROJECT UV_PROJECT_ENVIRONMENT UV_WORKING_DIR
unset CARGO_BUILD_TARGET RUSTC_WRAPPER RUSTFLAGS CARGO_ENCODED_RUSTFLAGS MAKEFLAGS MFLAGS
unset SLOFORGE_HELIX_GPU_BUDGET_USD SLOFORGE_HELIX_TRAINING_BUDGET_USD
unset SLOFORGE_HELIX_EXTERNAL_API_BUDGET_USD
export UV_PROJECT_ENVIRONMENT="$clean_root/.venv"
export CARGO_TARGET_DIR="$clean_root/target"
export SLOFORGE_HELIX_ALLOW_GPU=0
export SLOFORGE_HELIX_ALLOW_MULTI_NODE=0
export SLOFORGE_HELIX_ALLOW_EXTERNAL_API=0
export SLOFORGE_HELIX_ALLOW_PRODUCTION_CAPTURE=0
export SLOFORGE_HELIX_ALLOW_EXTERNAL_SIDE_EFFECTS=0
export SLOFORGE_HELIX_ALLOW_EXTERNAL_DEPLOYMENT=0
export SLOFORGE_HELIX_ALLOW_LIVE_PROMOTION=0

cd "$clean_root"
{
  make bootstrap
  make helix-check
  make helix-demo
  make helix-resource-demo
  make package
} 2>&1 | tee "$clean_root/helix-clean-room.log"

"$clean_root/.venv/bin/python" - "$clean_root" "$revision" "$source_tree" <<'PY'
import hashlib
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
summary = json.loads((root / "artifacts/helix/demo/seed-41/summary.json").read_text())
resource = json.loads((root / "artifacts/helix/resource-demo/plan.json").read_text())
assert summary["capture_consistent"] is True
assert summary["rejected_candidate"]["promotion_state"] == "rejected"
assert summary["promotion"]["incompatible_session_pinned"] is True
assert summary["promotion"]["final_state"] == "rolled_back"
assert resource["ticks"]
assert all(tick["serving_slo_satisfied"] is True for tick in resource["ticks"])
log = root / "helix-clean-room.log"
result = {
    "schema_version": "sloforge.helix.clean-room/v1",
    "status": "passed",
    "revision": sys.argv[2],
    "source_tree": sys.argv[3],
    "log_sha256": hashlib.sha256(log.read_bytes()).hexdigest(),
    "summary_sha256": hashlib.sha256(
        (root / "artifacts/helix/demo/seed-41/summary.json").read_bytes()
    ).hexdigest(),
    "resource_plan_id": resource["plan_id"],
}
(root / "helix-clean-room-result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
PY
cp "$clean_root/helix-clean-room.log" "$result_root/run.log"
cp "$clean_root/helix-clean-room-result.json" "$result_root/result.json"
echo "Helix clean-room validation passed for $revision"
