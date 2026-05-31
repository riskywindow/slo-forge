"""Reproducible ForgeCI matrix and Git-bisection commands."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from sloforge.forgeci import bisect_regression, load_matrix, run_matrix
from sloforge.forgeci.models import MatrixCase
from sloforge.util import write_json

from .common import json_result

forgeci_app = typer.Typer(help="Detect and bisect statistically meaningful regressions.")


def _case(matrix_path: Path, case_id: str | None) -> MatrixCase:
    matrix = load_matrix(matrix_path)
    if case_id is None:
        if len(matrix.cases) != 1:
            raise typer.BadParameter("--case-id is required for a matrix with multiple cases")
        return matrix.cases[0]
    try:
        return next(item for item in matrix.cases if item.case_id == case_id)
    except StopIteration as error:
        raise typer.BadParameter(f"matrix contains no case {case_id!r}") from error


@forgeci_app.command("run")
def run_command(
    matrix: Annotated[Path, typer.Option("--matrix", exists=True, dir_okay=False)],
    output: Annotated[Path, typer.Option("--output", "-o")],
    repository_base: Annotated[
        Path, typer.Option("--repository-base", exists=True, file_okay=False)
    ] = Path("."),
    allow_network: Annotated[bool, typer.Option("--allow-network")] = False,
) -> None:
    result = run_matrix(
        load_matrix(matrix),
        output_directory=output,
        repository_base=repository_base.resolve(),
        allow_network=allow_network,
    )
    json_result(
        {
            "output": str(output / "matrix-run.json"),
            "matrix_id": result.matrix_id,
            "success": result.success,
            "runs": len(result.runs),
            "warnings": list(result.warnings),
        }
    )


@forgeci_app.command("bisect")
def bisect_command(
    repository: Annotated[Path, typer.Option("--repository", exists=True, file_okay=False)],
    good: Annotated[str, typer.Option("--good")],
    bad: Annotated[str, typer.Option("--bad")],
    benchmark: Annotated[
        Path, typer.Option("--benchmark", "--matrix", exists=True, dir_okay=False)
    ],
    output: Annotated[Path, typer.Option("--output", "-o")],
    case_id: Annotated[str | None, typer.Option("--case-id")] = None,
    maximum_inconclusive_retries: Annotated[
        int, typer.Option("--maximum-inconclusive-retries", min=0, max=5)
    ] = 2,
) -> None:
    case = _case(benchmark, case_id).model_copy(update={"repository": str(repository.resolve())})
    result = bisect_regression(
        repository=repository.resolve(),
        good_revision=good,
        bad_revision=bad,
        case=case,
        output_directory=output,
        maximum_inconclusive_retries=maximum_inconclusive_retries,
    )
    result_path = output / "bisect-result.json"
    write_json(result_path, result.model_dump(mode="json"))
    json_result(
        {
            "output": str(result_path),
            "first_regressing_commit": result.first_regressing_commit,
            "confidence": result.confidence,
            "commits_evaluated": len(result.steps),
            "inconclusive_commits": list(result.inconclusive_commits),
        }
    )
