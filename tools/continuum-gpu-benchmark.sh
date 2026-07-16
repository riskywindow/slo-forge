#!/usr/bin/env bash
set -euo pipefail

repository_root="$(git rev-parse --show-toplevel)"
revision="$(git -C "$repository_root" rev-parse HEAD)"
result_dir="$repository_root/artifacts/continuum/hardware"
mkdir -p "$result_dir"
status_path="$result_dir/gpu-status.json"
log_path="$result_dir/gpu-run.log"
rm -f -- "$status_path" "$log_path"

record_failure() {
  exit_status=$?
  if test "$exit_status" -ne 0 && test ! -f "$status_path"; then
    printf '%s\n' '{"schema_version":"sloforge.continuum.hardware-status/v1","status":"failed","reason":"Continuum GPU acceptance command failed","gpu_results_claimed":false}' > "$status_path"
  fi
  return "$exit_status"
}
trap record_failure EXIT

if test "${SLOFORGE_CONTINUUM_ALLOW_GPU:-0}" != "1"; then
  printf '%s\n' '{"schema_version":"sloforge.continuum.hardware-status/v1","status":"unexercised","reason":"SLOFORGE_CONTINUUM_ALLOW_GPU is not 1","gpu_results_claimed":false}' > "$status_path"
  echo "Continuum GPU benchmark unexercised: explicit opt-in is disabled"
  exit 0
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
  printf '%s\n' '{"schema_version":"sloforge.continuum.hardware-status/v1","status":"unavailable","reason":"nvidia-smi is unavailable","gpu_results_claimed":false}' > "$status_path"
  echo "Continuum GPU benchmark unavailable: no supported NVIDIA device"
  exit 0
fi
if ! uv run --locked python -c 'import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)' >/dev/null 2>&1; then
  printf '%s\n' '{"schema_version":"sloforge.continuum.hardware-status/v1","status":"unavailable","reason":"a compatible CUDA-enabled PyTorch installation is unavailable","gpu_results_claimed":false}' > "$status_path"
  echo "Continuum GPU benchmark unavailable: CUDA-enabled PyTorch is not ready"
  exit 0
fi

uv run --locked pytest -q tests/python/test_continuum_gpu.py -m continuum_gpu 2>&1 | tee "$log_path"
uv run --locked python - "$status_path" "$log_path" "$revision" <<'PY'
import hashlib
import json
import pathlib
import sys

import torch

status = pathlib.Path(sys.argv[1])
log = pathlib.Path(sys.argv[2])
status.write_text(
    json.dumps(
        {
            "schema_version": "sloforge.continuum.hardware-status/v1",
            "status": "exercised",
            "revision": sys.argv[3],
            "gpu_results_claimed": True,
            "path": "pytorch_cuda_direct_conversion_with_canonical_verification",
            "torch_version": str(torch.__version__),
            "cuda_version": str(torch.version.cuda),
            "device_count": int(torch.cuda.device_count()),
            "device_names": [
                str(torch.cuda.get_device_name(index))
                for index in range(torch.cuda.device_count())
            ],
            "test_log_sha256": hashlib.sha256(log.read_bytes()).hexdigest(),
        },
        indent=2,
        sort_keys=True,
    )
    + "\n"
)
PY
