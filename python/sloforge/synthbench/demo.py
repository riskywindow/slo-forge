"""Reproducible CPU smoke/evaluation entry point for ServingSynthBench."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from sloforge.genesis.ir import canonical_json

from .grammar import generate_tasks
from .models import CpuRunConfiguration, GrammarConfiguration
from .runner import run_cpu_benchmark


class SynthBenchDemoResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    seed: int
    task_count: int
    task_seeds: tuple[int, ...]
    valid_system_rate: float
    exact_request_rate: float
    measured_cpu_seconds: float
    hardware_backed: bool = False
    report_path: str


def _safe_reset(output: Path, repository: Path) -> None:
    if not output.exists():
        return
    resolved = output.resolve()
    if resolved in {Path("/").resolve(), Path.home().resolve(), repository.resolve()}:
        raise ValueError(f"refusing to reset unsafe path: {resolved}")
    shutil.rmtree(resolved)


def run_synthbench_demo(
    output: Path,
    *,
    seed: int,
    count: int,
    reset: bool = False,
) -> SynthBenchDemoResult:
    if seed < 0 or count <= 0:
        raise ValueError("seed must be non-negative and count must be positive")
    repository = Path(__file__).resolve().parents[3]
    if reset:
        _safe_reset(output, repository)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"synthbench output is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    tasks = output / "tasks"
    descriptors = generate_tasks(
        GrammarConfiguration(
            seed=seed,
            count=count,
            public_cases_per_task=3,
            hidden_cases_per_task=6,
        ),
        tasks,
    )
    report = run_cpu_benchmark(
        tuple(tasks / item.task_id for item in descriptors),
        CpuRunConfiguration(
            seeds=(seed + 1, seed + 2),
            warmup_count=1,
            repetitions=2,
            maximum_tasks=count,
            maximum_runtime_seconds=max(30.0, count * 5.0),
        ),
        output / "run",
    )
    result = SynthBenchDemoResult(
        seed=seed,
        task_count=report.metrics.task_count,
        task_seeds=tuple(item.seed for item in descriptors),
        valid_system_rate=report.metrics.valid_system_rate,
        exact_request_rate=report.metrics.exact_request_rate,
        measured_cpu_seconds=report.metrics.measured_cpu_seconds,
        report_path=str((output / "run/report.json").resolve()),
    )
    (output / "summary.json").write_bytes(canonical_json(result) + b"\n")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=73129)
    parser.add_argument("--count", type=int, default=2)
    parser.add_argument("--reset", action="store_true")
    arguments = parser.parse_args(argv)
    result = run_synthbench_demo(
        arguments.output,
        seed=arguments.seed,
        count=arguments.count,
        reset=arguments.reset,
    )
    print(json.dumps(result.model_dump(mode="json"), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["SynthBenchDemoResult", "main", "run_synthbench_demo"]
