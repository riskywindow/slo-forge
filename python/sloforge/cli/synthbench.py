"""ServingSynthBench task generation, CPU execution, and artifact comparison CLI."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Annotated, Literal

import typer
from rich.console import Console

from sloforge.genesis.ir import canonical_json
from sloforge.synthbench import (
    CpuRunConfiguration,
    GrammarConfiguration,
    SynthBenchReport,
    generate_tasks,
    run_cpu_benchmark,
)

synthbench_app = typer.Typer(
    help="Generate and execute reproducible inference-system synthesis tasks.",
    no_args_is_help=True,
)
console = Console()


def _json_result(value: object) -> None:
    console.print_json(json.dumps(value, default=str))


def _task_directories(root: Path) -> tuple[Path, ...]:
    paths = tuple(sorted(path.parent for path in root.glob("task-*/task.json")))
    if not paths:
        raise typer.BadParameter(f"no generated tasks found under {root}")
    return paths


def _parse_seeds(value: str) -> tuple[int, ...]:
    try:
        seeds = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise typer.BadParameter("--seeds must be comma-separated non-negative integers") from error
    if any(seed < 0 for seed in seeds):
        raise typer.BadParameter("--seeds must be non-negative")
    return seeds


@synthbench_app.command("generate")
def generate_command(
    output: Annotated[Path, typer.Option("--output", "-o")],
    seed: Annotated[int, typer.Option("--seed", min=0)] = 73129,
    count: Annotated[int, typer.Option("--count", min=1, max=1000)] = 10,
) -> None:
    """Generate public packages and separately committed evaluator-only cases."""

    descriptors = generate_tasks(
        GrammarConfiguration(seed=seed, count=count),
        output,
    )
    _json_result(
        {
            "output": str(output.resolve()),
            "seed": seed,
            "count": len(descriptors),
            "task_ids": [item.task_id for item in descriptors],
        }
    )


@synthbench_app.command("run")
def run_command(
    tasks: Annotated[Path, typer.Option("--tasks", exists=True, file_okay=False)],
    output: Annotated[Path, typer.Option("--output", "-o")],
    system: Annotated[Literal["genesis"], typer.Option("--system")] = "genesis",
    seeds: Annotated[str, typer.Option("--seeds")] = "101,202",
    repetitions: Annotated[int, typer.Option("--repetitions", min=2, max=100)] = 3,
    warmup_count: Annotated[int, typer.Option("--warmup-count", min=0, max=100)] = 1,
    maximum_tasks: Annotated[int, typer.Option("--maximum-tasks", min=1)] = 100,
    timeout_seconds: Annotated[
        float, typer.Option("--timeout-seconds", min=1.0, max=86_400.0)
    ] = 120.0,
) -> None:
    """Run the dependency-free CPU profile with complete raw sample retention."""

    del system
    configuration = CpuRunConfiguration(
        seeds=_parse_seeds(seeds),
        warmup_count=warmup_count,
        repetitions=repetitions,
        maximum_tasks=maximum_tasks,
        maximum_runtime_seconds=timeout_seconds,
    )
    report = run_cpu_benchmark(_task_directories(tasks), configuration, output)
    _json_result(
        {
            "output": str((output / "report.json").resolve()),
            "tasks": report.metrics.task_count,
            "run_seeds": list(report.run_seeds),
            "valid_system_rate": report.metrics.valid_system_rate,
            "exact_request_rate": report.metrics.exact_request_rate,
            "measured_cpu_seconds": report.metrics.measured_cpu_seconds,
            "hardware_backed": False,
        }
    )


@synthbench_app.command("compare")
def compare_command(
    runs: Annotated[Path, typer.Option("--runs", exists=True, file_okay=False)],
    output: Annotated[Path, typer.Option("--output", "-o")],
) -> None:
    """Compare completed reports without recomputing or altering their raw evidence."""

    paths = tuple(sorted(runs.rglob("report.json")))
    if not paths:
        raise typer.BadParameter(f"no report.json artifacts found under {runs}")
    entries: list[dict[str, object]] = []
    for path in paths:
        payload = path.read_bytes()
        report = SynthBenchReport.model_validate_json(payload, strict=True)
        entries.append(
            {
                "report": str(path.resolve()),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "run_seeds": list(report.run_seeds),
                "metrics": report.metrics.model_dump(mode="json"),
            }
        )
    comparison = {
        "schema_version": "1.0.0",
        "source": "validated_synthbench_reports",
        "reports": entries,
        "hardware_comparison": "not_measured_in_cpu_profile",
    }
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "comparison.json"
    if report_path.exists():
        raise typer.BadParameter(f"refusing to overwrite comparison artifact: {report_path}")
    report_path.write_bytes(canonical_json(comparison) + b"\n")
    _json_result({"output": str(report_path.resolve()), "report_count": len(entries)})


__all__ = ["synthbench_app"]
