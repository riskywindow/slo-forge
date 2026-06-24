#!/usr/bin/env bash
set -euo pipefail

repository_root="$(git rev-parse --show-toplevel)"
revision="$(git -C "$repository_root" rev-parse HEAD)"
clean_root="$(mktemp -d "${TMPDIR:-/tmp}/sloforge-genesis-clean.XXXXXX")"
cleanup() {
  rm -rf -- "$clean_root"
}
trap cleanup EXIT INT TERM

git -C "$repository_root" archive "$revision" | tar -x -C "$clean_root"
printf '%s\n' "$revision" > "$clean_root/.sloforge-source-commit"
make -C "$clean_root" bootstrap
make -C "$clean_root" genesis-check
make -C "$clean_root" genesis-demo
make -C "$clean_root" synthbench-smoke
make -C "$clean_root" package

python3 - "$clean_root" "$revision" <<'PY'
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
            "genesis_check": "passed",
            "genesis_demo": "passed",
            "synthbench_smoke": "passed",
            "package_build": "passed",
            "hardware_backed": False,
        },
        indent=2,
        sort_keys=True,
    )
    + "\n",
    encoding="utf-8",
)
PY
echo "Genesis clean-room validation passed for $revision"
