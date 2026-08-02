"""Executable red-team and independently replayed minimization CLI."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from sloforge.genesis.ir import Counterexample, write_canonical
from sloforge.redteam.demo import run_demo
from sloforge.redteam.fixture import UnsafeStreamingCandidate, unsafe_benchmark_comparison
from sloforge.redteam.models import RedTeamConfiguration
from sloforge.redteam.runner import run_red_team

redteam_app = typer.Typer(help="Generate, minimize, and persist executable counterexamples.")
console = Console()


def _json_result(value: object) -> None:
    console.print_json(json.dumps(value, default=str))


@redteam_app.command("run")
def run_command(
    output: Annotated[Path, typer.Option("--output", "-o")],
    seed: Annotated[int, typer.Option("--seed", min=0)] = 73129,
    budget_seconds: Annotated[float, typer.Option("--budget-seconds", min=0.1, max=600.0)] = 15.0,
    candidate: Annotated[str, typer.Option("--candidate")] = "unsafe-fastpath-v1",
) -> None:
    """Run the dependency-free fixture; arbitrary generated targets require a sandbox adapter."""

    if candidate != UnsafeStreamingCandidate.descriptor.candidate_id:
        raise typer.BadParameter("only the bounded unsafe-fastpath-v1 fixture is locally available")
    # The public fixture has tighter internal limits than a caller may supply.
    # Preserve the requested hard ceiling by refusing an under-sized budget.
    if budget_seconds < 15.0:
        raise typer.BadParameter("the fixture requires --budget-seconds >= 15")
    result = run_demo(output, seed=seed)
    _json_result(result.model_dump(mode="json"))


def _reproduction_seed(counterexample: Counterexample) -> int:
    if counterexample.reproduction.seed is None:
        raise typer.BadParameter("counterexample reproduction command has no deterministic seed")
    return counterexample.reproduction.seed


@redteam_app.command("minimize")
def minimize_command(
    counterexample_path: Annotated[
        Path, typer.Option("--counterexample", exists=True, dir_okay=False)
    ],
    output: Annotated[Path, typer.Option("--output", "-o")],
) -> None:
    """Re-execute the known target's bounded minimizer and require the same witness identity."""

    counterexample = Counterexample.model_validate_json(
        counterexample_path.read_bytes(), strict=True
    )
    if counterexample.candidate_id != UnsafeStreamingCandidate.descriptor.candidate_id:
        raise typer.BadParameter("no trusted local target adapter exists for this candidate")
    seed = _reproduction_seed(counterexample)
    report = run_red_team(
        target=UnsafeStreamingCandidate(),
        configuration=RedTeamConfiguration(
            seed=seed,
            maximum_findings=32,
            maximum_minimization_evaluations=128,
            minimization_timeout_seconds=2.0,
            run_timeout_seconds=15.0,
            tensor_cases=12,
            schedule_cases=12,
            topology_cases=8,
            resource_cases=8,
        ),
        benchmark_comparison=unsafe_benchmark_comparison(),
    )
    matching = tuple(
        finding.counterexample
        for finding in report.findings
        if finding.counterexample.counterexample_id == counterexample.counterexample_id
    )
    if len(matching) != 1:
        raise typer.BadParameter(
            "counterexample did not reproduce with the trusted fixture adapter"
        )
    if output.exists():
        raise typer.BadParameter(f"refusing to overwrite minimized counterexample: {output}")
    write_canonical(matching[0], output)
    _json_result(
        {
            "counterexample_id": matching[0].counterexample_id,
            "output": str(output.resolve()),
            "reproduced": True,
            "minimized": matching[0].minimized,
            "seed": seed,
        }
    )


__all__ = ["redteam_app"]
