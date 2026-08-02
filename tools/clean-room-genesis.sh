#!/usr/bin/env bash
set -euo pipefail

repository_root="$(git rev-parse --show-toplevel)"
revision="$(git -C "$repository_root" rev-parse HEAD)"
clean_root="$(mktemp -d "${TMPDIR:-/tmp}/sloforge-genesis-clean.XXXXXX")"
result_root="${SLOFORGE_GENESIS_CLEANROOM_RESULT_ROOT:-$repository_root/artifacts/genesis/clean-room}"
cleanup() {
  rm -rf -- "$clean_root"
}
trap cleanup EXIT INT TERM

git -C "$repository_root" archive "$revision" | tar -x -C "$clean_root"
printf '%s\n' "$revision" > "$clean_root/.sloforge-source-commit"
{
  make -C "$clean_root" bootstrap
  make -C "$clean_root" genesis-check
  make -C "$clean_root" genesis-demo
  make -C "$clean_root" synthbench-smoke
  make -C "$clean_root" package
  wheel_path="$(find "$clean_root/dist" -maxdepth 1 -type f -name '*.whl' -print -quit)"
  test -n "$wheel_path"
  uv pip install --python "$clean_root/.venv/bin/python" --no-deps --force-reinstall "$wheel_path"
  "$clean_root/.venv/bin/sloforge" --help >/dev/null
} 2>&1 | tee "$clean_root/genesis-clean-room.log"

python3 - "$clean_root" "$revision" <<'PY'
import hashlib
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
revision = sys.argv[2]
genesis = json.loads((root / "artifacts/genesis/demo/GENESIS_DEMO_REPORT.json").read_text())
synthbench = json.loads((root / "artifacts/synthbench/smoke/summary.json").read_text())
assert genesis["runtime_differential_passed"] is True
assert genesis["cross_layer_accepted"] is True
assert genesis["capsule_promotion_eligible"] is True
assert genesis["active_stream_preserved"] is True
assert genesis["hardware_backed"] is False
assert synthbench["valid_system_rate"] == 1.0
assert synthbench["exact_request_rate"] == 1.0
(root / "artifacts/genesis/clean-room-result.json").write_text(
    json.dumps(
        {
            "revision": revision,
            "log_sha256": hashlib.sha256(
                (root / "genesis-clean-room.log").read_bytes()
            ).hexdigest(),
            "genesis_check": "passed",
            "genesis_demo": "passed",
            "synthbench_smoke": "passed",
            "package_build": "passed",
            "wheel_install_smoke": "passed",
            "hardware_backed": False,
        },
        indent=2,
        sort_keys=True,
    )
    + "\n",
    encoding="utf-8",
)
PY
mkdir -p "$result_root"
cp "$clean_root/artifacts/genesis/clean-room-result.json" "$result_root/result.json"
cp "$clean_root/genesis-clean-room.log" "$result_root/run.log"
echo "Genesis clean-room validation passed for $revision"
