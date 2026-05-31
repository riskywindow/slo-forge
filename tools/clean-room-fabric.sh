#!/usr/bin/env bash
set -euo pipefail

repository_root="$(git rev-parse --show-toplevel)"
revision="$(git -C "$repository_root" rev-parse HEAD)"
clean_root="$(mktemp -d "${TMPDIR:-/tmp}/sloforge-fabric-clean.XXXXXX")"
cleanup() {
  rm -rf -- "$clean_root"
}
trap cleanup EXIT INT TERM

git -C "$repository_root" archive "$revision" | tar -x -C "$clean_root"
make -C "$clean_root" bootstrap
make -C "$clean_root" fabric-check
make -C "$clean_root" fabric-demo
python3 - "$clean_root" "$revision" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
revision = sys.argv[2]
manifest = json.loads((root / "artifacts/fabric-demo/manifest.json").read_text())
assert manifest["degraded_slo_attained"] is False
assert manifest["restored_slo_attained"] is True
(root / "artifacts/clean-room-result.json").write_text(
    json.dumps(
        {
            "revision": revision,
            "fabric_check": "passed",
            "fabric_demo": "passed",
            "synthetic_hardware": manifest["synthetic_hardware"],
        },
        indent=2,
        sort_keys=True,
    )
    + "\n",
    encoding="utf-8",
)
PY
echo "clean-room validation passed for $revision"
