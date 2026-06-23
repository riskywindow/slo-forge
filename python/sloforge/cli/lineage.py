"""CLI queries and invalidation for the embedded optimization lineage graph."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from sloforge.lineage import (
    DependencyKind,
    DependencySelector,
    InvalidationEvent,
    LineageNotFound,
    LineageStore,
    TransformationOutcome,
    TransformationQuery,
    export_graphml,
    export_json,
)

lineage_app = typer.Typer(
    help="Query, export, and invalidate Genesis optimization lineage.", no_args_is_help=True
)
console = Console()


def _json_result(value: object) -> None:
    console.print_json(json.dumps(value, default=str))


@lineage_app.command("query")
def query_command(
    database: Annotated[Path, typer.Option("--database")] = Path("artifacts/lineage/lineage.db"),
    model_family: Annotated[str | None, typer.Option("--model-family")] = None,
    hardware: Annotated[str | None, typer.Option("--hardware")] = None,
    operation: Annotated[str | None, typer.Option("--operation")] = None,
    workload: Annotated[str | None, typer.Option("--workload")] = None,
    family: Annotated[str | None, typer.Option("--family")] = None,
    outcome: Annotated[TransformationOutcome | None, typer.Option("--outcome")] = None,
    limit: Annotated[int, typer.Option("--limit", min=1, max=1000)] = 100,
) -> None:
    query = TransformationQuery(
        model_family=model_family,
        operation=operation,
        hardware_architecture=hardware,
        workload_regime=workload,
        family=family,
        outcome=outcome,
        limit=limit,
        scan_limit=max(limit, min(100_000, limit * 10)),
    )
    with LineageStore(database) as store:
        matches = store.query_transformations(query)
    _json_result(
        {
            "database": str(database),
            "count": len(matches),
            "transformations": [item.model_dump(mode="json") for item in matches],
        }
    )


@lineage_app.command("explain")
def explain_command(
    transformation: Annotated[str, typer.Option("--transformation")],
    database: Annotated[Path, typer.Option("--database")] = Path("artifacts/lineage/lineage.db"),
) -> None:
    with LineageStore(database) as store:
        record = next(
            (
                item
                for item in store.list_transformations(limit=100_000)
                if item.transformation_id == transformation
            ),
            None,
        )
        if record is None:
            raise LineageNotFound(f"unknown transformation {transformation!r}")
        evidence = store.evidence_for_transformation(transformation, limit=10_000)
        constraints = tuple(
            item
            for item in store.list_constraints(limit=10_000)
            if item.transformation_family == record.family
        )
        counterexamples = tuple(
            item
            for item in store.list_counterexamples(limit=10_000)
            if item.transformation_id == transformation
        )
    _json_result(
        {
            "transformation": record.model_dump(mode="json"),
            "evidence": [item.model_dump(mode="json") for item in evidence],
            "constraints": [item.model_dump(mode="json") for item in constraints],
            "counterexamples": [item.model_dump(mode="json") for item in counterexamples],
        }
    )


@lineage_app.command("invalidate")
def invalidate_command(
    dependency: Annotated[str, typer.Option("--dependency")],
    version_range: Annotated[str, typer.Option("--version-range")],
    kind: Annotated[DependencyKind, typer.Option("--kind")] = DependencyKind.LIBRARY,
    reason: Annotated[str, typer.Option("--reason")] = "dependency compatibility changed",
    database: Annotated[Path, typer.Option("--database")] = Path("artifacts/lineage/lineage.db"),
    maximum_evidence: Annotated[
        int, typer.Option("--maximum-evidence", min=1, max=1_000_000)
    ] = 10_000,
) -> None:
    occurred_at = datetime.now(UTC)
    identity = hashlib.sha256(
        f"{kind.value}\0{dependency}\0{version_range}\0{occurred_at.isoformat()}".encode()
    ).hexdigest()
    event = InvalidationEvent(
        invalidation_id=f"invalidation-{identity[:24]}",
        selector=DependencySelector(kind=kind, name=dependency, version_range=version_range),
        reason=reason,
        occurred_at=occurred_at,
    )
    with LineageStore(database) as store:
        affected = store.invalidate_dependency(event, maximum_evidence=maximum_evidence)
    _json_result(
        {
            "database": str(database),
            "invalidation": event.model_dump(mode="json"),
            "evidence_marked_stale": affected,
        }
    )


@lineage_app.command("export")
def export_command(
    output: Annotated[Path, typer.Option("--output", "-o")],
    database: Annotated[Path, typer.Option("--database")] = Path("artifacts/lineage/lineage.db"),
    maximum_records: Annotated[
        int, typer.Option("--maximum-records", min=1, max=1_000_000)
    ] = 100_000,
) -> None:
    exported_at = datetime.now(UTC)
    with LineageStore(database) as store:
        snapshot = export_json(
            store,
            output.with_suffix(".json"),
            exported_at=exported_at,
            maximum_records=maximum_records,
        )
        export_graphml(
            store,
            output.with_suffix(".graphml"),
            exported_at=exported_at,
            maximum_records=maximum_records,
        )
    _json_result(
        {
            "json": str(output.with_suffix(".json")),
            "graphml": str(output.with_suffix(".graphml")),
            "tasks": len(snapshot.tasks),
            "candidates": len(snapshot.candidates),
            "transformations": len(snapshot.transformations),
        }
    )


__all__ = ["lineage_app"]
