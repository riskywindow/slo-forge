from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
from pathlib import Path
from typing import Any

from sloforge.util import sha256_file, utc_now, write_json

CPU_OUTPUTS = (
    "evaluation.md",
    "evaluation.html",
    "evaluation.json",
    "pareto.svg",
    "controller.svg",
    "metrics.prom",
    "otel-traces.json",
    "trace.json",
    "report-data.json",
    "report-manifest.json",
)


def finalize_cpu_report(*, source: Path, output: Path) -> Path:
    """Promote a verified CPU run into the repository-level evaluation report."""
    source = source.resolve()
    manifest_path = source / "report-manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing report manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "sloforge.report/v1":
        raise ValueError("unsupported or absent report manifest schema")
    if int(manifest.get("verified_artifact_count", 0)) < 1:
        raise ValueError("report manifest does not attest any verified input artifacts")
    missing = [name for name in CPU_OUTPUTS if not (source / name).is_file()]
    if missing:
        raise FileNotFoundError(f"CPU report is incomplete: {missing}")

    output.mkdir(parents=True, exist_ok=True)
    hashes: dict[str, str] = {}
    for name in CPU_OUTPUTS:
        destination = output / name
        shutil.copy2(source / name, destination)
        hashes[name] = sha256_file(destination)
    record = output / "cpu-benchmark.json"
    write_json(
        record,
        {
            "schema_version": "sloforge.cpu-benchmark/v1",
            "generated_at": utc_now(),
            "source_report": str(source),
            "verified_artifact_count": int(manifest["verified_artifact_count"]),
            "output_hashes": hashes,
            "metrics": manifest.get("metrics", {}),
        },
    )
    return record


def _nvidia_inventory() -> tuple[bool, list[str]]:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return False, []
    try:
        completed = subprocess.run(
            [
                executable,
                "--query-gpu=name,uuid,driver_version,memory.total",
                "--format=csv,noheader",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return False, []
    devices = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    return bool(devices), devices


def benchmark_gpu(*, output: Path) -> Path:
    """Record GPU availability or the exact real-engine benchmark procedure.

    This command deliberately emits no latency or throughput values when an NVIDIA device is
    unavailable. That makes the optional-path limitation machine readable without fabricating a
    benchmark.
    """
    output.mkdir(parents=True, exist_ok=True)
    available, devices = _nvidia_inventory()
    commands = [
        "sloforge trace generate --output workloads/gpu-benchmark.jsonl --count 180 --seed 41",
        "sloforge hardware probe --device cuda --hourly-price-usd 0 "
        "--output artifacts/hardware/gpu.json",
        "sloforge profile --model Qwen/Qwen3-0.6B --engines transformers,vllm,sglang "
        "--hardware artifacts/hardware/gpu.json --trace workloads/gpu-benchmark.jsonl "
        "--budget-usd ${SLOFORGE_GPU_BUDGET_USD:-0} --output artifacts/profiles/gpu",
    ]
    status = "ready-not-executed" if available else "unavailable"
    reason = (
        "NVIDIA hardware is present; install the requested engine extras and run the recorded "
        "commands to execute measurements."
        if available
        else "No working nvidia-smi inventory was detected on this host."
    )
    payload: dict[str, Any] = {
        "schema_version": "sloforge.gpu-benchmark/v1",
        "generated_at": utc_now(),
        "status": status,
        "reason": reason,
        "devices": devices,
        "host": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": platform.python_version(),
        },
        "model": {
            "identifier": "Qwen/Qwen3-0.6B",
            "license": "Apache-2.0",
            "engines": ["transformers", "vllm", "sglang"],
        },
        "measurements": [],
        "commands": commands,
        "expected_profile_schema": "sloforge.profile/v1",
    }
    result_path = output / "evaluation.json"
    write_json(result_path, payload)
    command_path = output / "commands.txt"
    command_path.write_text("\n".join(commands) + "\n", encoding="utf-8")
    markdown = "\n".join(
        [
            "# GPU benchmark status",
            "",
            f"Status: **{status}**",
            "",
            reason,
            "",
            "No GPU performance numbers are reported by this artifact.",
            "",
            "## Reproduction commands",
            "",
            "```console",
            *commands,
            "```",
            "",
        ]
    )
    (output / "evaluation.md").write_text(markdown, encoding="utf-8")
    return result_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SLOForge reproducible benchmark helpers")
    commands = parser.add_subparsers(dest="command", required=True)
    cpu = commands.add_parser("finalize-cpu")
    cpu.add_argument("--source", type=Path, required=True)
    cpu.add_argument("--output", type=Path, required=True)
    gpu = commands.add_parser("benchmark-gpu")
    gpu.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    arguments = _parser().parse_args()
    if arguments.command == "finalize-cpu":
        print(finalize_cpu_report(source=arguments.source, output=arguments.output))
    else:
        print(benchmark_gpu(output=arguments.output))


if __name__ == "__main__":
    main()
