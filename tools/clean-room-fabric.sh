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
make -C "$clean_root" check
make -C "$clean_root" fabric-check
make -C "$clean_root" demo
make -C "$clean_root" fabric-demo
make -C "$clean_root" forgeci-demo
make -C "$clean_root" warmpath-demo
make -C "$clean_root" package

uv --directory "$clean_root" run --locked sloforge fabric simulate \
  --plan "$clean_root/artifacts/fabric-demo/physical-plan.json" \
  --topology "$clean_root/artifacts/fabric-demo/topology.json" \
  --fabric-profile "$clean_root/artifacts/fabric-demo/fabric-profile.json" \
  --trace "$clean_root/artifacts/fabric-demo/mixed-bursty.jsonl" \
  --output "$clean_root/artifacts/clean-room-simulation"
uv --directory "$clean_root" run --locked sloforge autopsy replay \
  --evidence "$clean_root/artifacts/fabric-demo/autopsy" \
  --counterfactual "$clean_root/artifacts/fabric-demo/autopsy/scenarios.json" \
  --output "$clean_root/artifacts/clean-room-autopsy-replay.json"
if uv --directory "$clean_root" run --locked sloforge fabric validate \
  --plan "$clean_root/artifacts/fabric-demo/physical-plan.json" \
  --topology "$clean_root/artifacts/fabric-demo/topology.json" \
  --fabric-profile "$clean_root/artifacts/fabric-demo/fabric-profile.json" \
  --trace "$clean_root/artifacts/fabric-demo/mixed-bursty.jsonl" \
  --max-relative-error 0 \
  --output "$clean_root/artifacts/clean-room-validation"; then
  echo "error: strict prediction validation unexpectedly passed" >&2
  exit 1
fi
python3 - "$clean_root" "$revision" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
revision = sys.argv[2]
manifest = json.loads((root / "artifacts/fabric-demo/manifest.json").read_text())
validation = json.loads((root / "artifacts/clean-room-validation/validation.json").read_text())
replay = json.loads((root / "artifacts/clean-room-autopsy-replay.json").read_text())
assert manifest["degraded_slo_attained"] is False
assert manifest["restored_slo_attained"] is True
assert validation["valid"] is False
assert validation["failure_reasons"]
assert replay["evaluations"] and replay["selected_scenario_id"]
(root / "artifacts/clean-room-result.json").write_text(
    json.dumps(
        {
            "revision": revision,
            "full_check": "passed",
            "fabric_check": "passed",
            "original_demo": "passed",
            "fabric_demo": "passed",
            "public_simulation": "passed",
            "standalone_autopsy_replay": "passed",
            "fail_closed_validation": "passed",
            "forgeci_demo": "passed",
            "warmpath_demo": "passed",
            "package_build": "passed",
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
