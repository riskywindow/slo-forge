"""WarmPath local profiling and cold-start planning commands."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

import typer
from pydantic import TypeAdapter

from sloforge.warmpath import (
    HostEnvironment,
    StartupProfile,
    StorageTierSpec,
    WarmPathObjective,
    compile_warmpath,
    load_graph,
    profile_local_startup,
    save_plan,
)

from .common import json_result

warmpath_app = typer.Typer(help="Profile and compile physical cold-start artifact placement.")


def _profile(path: Path) -> StartupProfile:
    source = path / "profile.json" if path.is_dir() else path
    return StartupProfile.model_validate_json(source.read_text(encoding="utf-8"), strict=True)


@warmpath_app.command("profile")
def profile_command(
    graph: Annotated[Path, typer.Option("--graph", exists=True, dir_okay=False)],
    source: Annotated[Path, typer.Option("--source", exists=True, file_okay=False)],
    host: Annotated[Path, typer.Option("--host", exists=True, dir_okay=False)],
    tiers: Annotated[Path, typer.Option("--tiers", exists=True, dir_okay=False)],
    output: Annotated[Path, typer.Option("--output", "-o")],
    profile_id: Annotated[str, typer.Option("--profile-id")] = "warmpath-local",
    warmups: Annotated[int, typer.Option("--warmups", min=0)] = 2,
    samples: Annotated[int, typer.Option("--samples", min=3)] = 7,
    seed: Annotated[int, typer.Option("--seed", min=0)] = 17,
    timeout_seconds: Annotated[float, typer.Option("--timeout-seconds", min=0.1)] = 10.0,
    maximum_artifact_bytes: Annotated[int, typer.Option("--maximum-artifact-bytes", min=1)] = 1
    << 30,
) -> None:
    artifact_graph = load_graph(graph)
    host_environment = HostEnvironment.model_validate_json(
        host.read_text(encoding="utf-8"), strict=True
    )
    storage_tiers = TypeAdapter(tuple[StorageTierSpec, ...]).validate_json(
        tiers.read_text(encoding="utf-8"), strict=True
    )
    result = profile_local_startup(
        profile_id=profile_id,
        graph=artifact_graph,
        host=host_environment,
        tiers=storage_tiers,
        source_directory=source,
        output_directory=output,
        warmups=warmups,
        sample_count=samples,
        seed=seed,
        timeout_seconds=timeout_seconds,
        maximum_artifact_bytes=maximum_artifact_bytes,
    )
    json_result(
        {
            "output": str(output / "profile.json"),
            "profile_id": result.profile_id,
            "measurements": len(result.measurements),
            "raw_artifacts": result.raw_artifact_directory,
        }
    )


@warmpath_app.command("compile")
def compile_command(
    graph: Annotated[Path, typer.Option("--graph", exists=True, dir_okay=False)],
    profile: Annotated[Path, typer.Option("--profile", exists=True)],
    output: Annotated[Path, typer.Option("--output", "-o")],
    objective: Annotated[
        Literal["minimize_p95_ready_time", "balanced"], typer.Option("--objective")
    ] = "minimize_p95_ready_time",
    hourly_cost_weight: Annotated[float, typer.Option("--hourly-cost-weight", min=0.0)] = 0.0,
    failure_risk_weight: Annotated[float, typer.Option("--failure-risk-weight", min=0.0)] = 0.0,
    maximum_p95_ready_time_ms: Annotated[
        float | None, typer.Option("--maximum-p95-ready-time-ms", min=0.001)
    ] = None,
    maximum_hourly_cost: Annotated[
        float | None, typer.Option("--maximum-hourly-cost", min=0.0)
    ] = None,
    maximum_warm_replicas: Annotated[int, typer.Option("--maximum-warm-replicas", min=0)] = 0,
    warm_replica_hourly_cost: Annotated[
        float, typer.Option("--warm-replica-hourly-cost", min=0.0)
    ] = 0.0,
    seed: Annotated[int, typer.Option("--seed", min=0)] = 17,
    simulation_trials: Annotated[int, typer.Option("--simulation-trials", min=3)] = 101,
) -> None:
    ready_weight = 1.0
    if objective == "balanced" and hourly_cost_weight == 0.0 and failure_risk_weight == 0.0:
        hourly_cost_weight = 1.0
        failure_risk_weight = 1_000.0
    plan = compile_warmpath(
        graph=load_graph(graph),
        profile=_profile(profile),
        objective=WarmPathObjective(
            ready_time_weight=ready_weight,
            hourly_cost_weight=hourly_cost_weight,
            failure_risk_weight=failure_risk_weight,
            maximum_p95_ready_time_ms=maximum_p95_ready_time_ms,
            maximum_hourly_cost=maximum_hourly_cost,
            warm_replica_hourly_cost=warm_replica_hourly_cost,
            maximum_warm_replicas=maximum_warm_replicas,
        ),
        seed=seed,
        simulation_trials=simulation_trials,
    )
    save_plan(plan, output)
    json_result(
        {
            "output": str(output),
            "plan_id": plan.plan_id,
            "p95_ready_time_ms": plan.predicted_p95_ready_time_ms,
            "prediction_interval_ms": [
                plan.prediction_interval_low_ms,
                plan.prediction_interval_high_ms,
            ],
            "hourly_cost": plan.predicted_hourly_cost,
            "evaluated_candidates": plan.evaluated_candidate_count,
            "optimizer": plan.optimizer,
        }
    )
