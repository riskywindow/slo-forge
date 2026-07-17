from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Annotated, Literal, cast

import typer
from rich.console import Console

from sloforge.compiler import (
    ModelProfileMetadata,
    compile_deployment,
    explain_plan,
    mock_qwen3_metadata,
)
from sloforge.demo import _candidate_catalog, run_demo
from sloforge.exporters import ExportContext, export_plan
from sloforge.faults import execute_scenario, load_scenario
from sloforge.hardware.probe import ProbeResult, run_probe, save_probe
from sloforge.ir import load_deployment_plan, load_evidence_bundle
from sloforge.models import fit_service_curves
from sloforge.models.service_curve import CalibratedModels
from sloforge.optimizer import OptimizationRequest, optimize, parse_slo_expression
from sloforge.profiler.core import ProfilingBudget, load_profile, profile_mock_candidates
from sloforge.reports import generate_report
from sloforge.runtime import replay_simulator
from sloforge.trace.format import (
    generate_bursty_trace,
    load_trace,
    validate_trace,
    write_trace,
)
from sloforge.util import load_json, sha256_file, write_json

from .autopsy import autopsy_app
from .continuum import continuum_app
from .fabric import fabric_app
from .forgeci import forgeci_app
from .genesis import genesis_app
from .helix import helix_app
from .lineage import lineage_app
from .recovery import recovery_app
from .redteam import redteam_app
from .synthbench import synthbench_app
from .warmpath import warmpath_app

app = typer.Typer(
    name="sloforge",
    help="Compile measured inference deployments from workload, hardware, budget, and SLO constraints.",
    no_args_is_help=True,
)
trace_app = typer.Typer(help="Validate and generate canonical workload traces.")
hardware_app = typer.Typer(help="Probe hardware without silently changing devices.")
app.add_typer(trace_app, name="trace")
app.add_typer(hardware_app, name="hardware")
app.add_typer(fabric_app, name="fabric")
app.add_typer(autopsy_app, name="autopsy")
app.add_typer(recovery_app, name="recovery")
app.add_typer(forgeci_app, name="forgeci")
app.add_typer(warmpath_app, name="warmpath")
app.add_typer(genesis_app, name="genesis")
app.add_typer(lineage_app, name="lineage")
app.add_typer(synthbench_app, name="synthbench")
app.add_typer(redteam_app, name="redteam")
app.add_typer(continuum_app, name="continuum")
app.add_typer(helix_app, name="helix")
console = Console()


def _root() -> Path:
    return Path(__file__).resolve().parents[3]


def _json_result(value: object) -> None:
    console.print_json(json.dumps(value, default=str))


@trace_app.command("validate")
def trace_validate(path: Annotated[Path, typer.Argument(exists=True, dir_okay=False)]) -> None:
    summary = validate_trace(load_trace(path))
    _json_result(summary.model_dump())


@trace_app.command("generate")
def trace_generate(
    output: Annotated[Path, typer.Option("--output", "-o")],
    seed: Annotated[int, typer.Option("--seed")] = 41,
    count: Annotated[int, typer.Option("--count", min=8)] = 180,
) -> None:
    requests = generate_bursty_trace(seed=seed, count=count)
    write_trace(output, requests)
    _json_result({"output": str(output), **validate_trace(requests).model_dump()})


@hardware_app.command("probe")
def hardware_probe(
    output: Annotated[Path, typer.Option("--output", "-o")],
    device: Annotated[Literal["cpu", "cuda"], typer.Option("--device")] = "cpu",
    samples: Annotated[int, typer.Option("--samples", min=3)] = 7,
    hourly_price_usd: Annotated[float | None, typer.Option("--hourly-price-usd", min=0)] = None,
) -> None:
    result = run_probe(device=device, samples=samples, hourly_price_usd=hourly_price_usd)
    save_probe(output, result)
    _json_result(
        {"output": str(output), "fingerprint": result.fingerprint, "warnings": result.warnings}
    )


@app.command("profile")
def profile_command(
    model: Annotated[str, typer.Option("--model")],
    engines: Annotated[str, typer.Option("--engines")],
    hardware: Annotated[Path, typer.Option("--hardware", exists=True, dir_okay=False)],
    trace: Annotated[Path, typer.Option("--trace", exists=True, dir_okay=False)],
    output: Annotated[Path, typer.Option("--output", "-o")],
    budget_usd: Annotated[float, typer.Option("--budget-usd", min=0)] = 0.0,
    duration_s: Annotated[float, typer.Option("--duration-s", min=1)] = 120.0,
    seed: Annotated[int, typer.Option("--seed")] = 41,
) -> None:
    requested = [item.strip() for item in engines.split(",") if item.strip()]
    probe = ProbeResult.model_validate_json(hardware.read_text(encoding="utf-8"))
    workload = load_trace(trace)
    if requested and all(item == "mock" or item.startswith("mock-") for item in requested):
        candidates = _candidate_catalog(probe.hardware.memory_bytes)
        if requested != ["mock"]:
            by_id = {item.candidate_id: item for item in candidates}
            unknown = [item for item in requested if item not in by_id]
            if unknown:
                raise typer.BadParameter(
                    f"unknown mock engines: {unknown}; available {sorted(by_id)}"
                )
            candidates = [by_id[item] for item in requested]
        bundle = profile_mock_candidates(
            candidates=candidates,
            trace=workload,
            trace_path=trace,
            hardware_path=hardware,
            budget=ProfilingBudget(max_duration_s=duration_s, max_cost_usd=budget_usd),
            seed=seed,
            output_dir=output,
        )
        (output / "workload.jsonl").write_bytes(trace.read_bytes())
        (output / "hardware.json").write_bytes(hardware.read_bytes())
        metadata = mock_qwen3_metadata(requested_model=model)
        write_json(output / "model-metadata.json", metadata.model_dump(mode="json"))
    else:
        try:
            from sloforge.profiler.real import profile_real_engines
        except ImportError as exc:
            raise RuntimeError(
                "real GPU profiler module is unavailable in this installation"
            ) from exc
        if probe.requested_device != "cuda":
            raise typer.BadParameter(
                "real engines require a hardware probe created with --device cuda"
            )
        bundle = profile_real_engines(
            model=model,
            engines=requested,
            hardware=probe,
            hardware_path=hardware,
            trace=workload,
            trace_path=trace,
            budget=ProfilingBudget(max_duration_s=duration_s, max_cost_usd=budget_usd),
            seed=seed,
            output_dir=output,
        )
    _json_result(
        {
            "profile_id": bundle.profile_id,
            "output": str(output),
            "candidates": len(bundle.candidates),
            "measurements": len(bundle.raw_measurements),
            "spent_cost_usd": bundle.budget.spent_cost_usd,
        }
    )


@app.command("optimize")
def optimize_command(
    profile: Annotated[Path, typer.Option("--profile", exists=True, file_okay=False)],
    slo: Annotated[str, typer.Option("--slo")],
    objective: Annotated[str, typer.Option("--objective")] = "minimize:cost_per_million_tokens",
    output: Annotated[Path, typer.Option("--output", "-o")] = Path("artifacts/plans/plan.json"),
    max_replicas: Annotated[int, typer.Option("--max-replicas", min=1)] = 4,
    trials: Annotated[int, typer.Option("--trials", min=1)] = 24,
    seed: Annotated[int, typer.Option("--seed")] = 41,
) -> None:
    profile_bundle = load_profile(profile)
    workload_path = profile / "workload.jsonl"
    hardware_path = profile / "hardware.json"
    metadata_path = profile / "model-metadata.json"
    for required in (workload_path, hardware_path, metadata_path):
        if not required.exists():
            raise typer.BadParameter(f"profile bundle is missing required artifact {required.name}")
    workload = load_trace(workload_path)
    models = fit_service_curves(profile_bundle, seed=seed)
    direction_raw, separator, metric = objective.partition(":")
    if not separator or direction_raw not in {"minimize", "maximize"}:
        raise typer.BadParameter("objective must be minimize:metric or maximize:metric")
    request = OptimizationRequest.model_validate(
        {
            "constraints": parse_slo_expression(slo),
            "objective": metric,
            "direction": direction_raw,
            "max_replicas": max_replicas,
            "trial_budget": trials,
            "seed": seed,
        }
    )
    result = optimize(profile=profile_bundle, models=models, trace=workload, request=request)
    optimization_path = output.with_suffix(".optimization.json")
    write_json(optimization_path, result.model_dump())
    metadata = ModelProfileMetadata.model_validate(load_json(metadata_path))
    hardware = ProbeResult.model_validate_json(hardware_path.read_text(encoding="utf-8"))
    compiled = compile_deployment(
        optimization=result,
        profile=profile_bundle,
        models=models,
        hardware=hardware,
        trace=workload,
        trace_path=workload_path,
        hardware_path=hardware_path,
        profile_dir=profile,
        output_path=output,
        evidence_dir=output.parent / f"{output.stem}.evidence",
        model_metadata=metadata,
        repository_root=_root(),
    )
    _json_result(
        {
            "plan": str(compiled.plan_path),
            "evidence": str(compiled.evidence_path),
            "optimization": str(optimization_path),
            "selected": result.selected.configuration.config_id,
            "pareto_candidates": len(result.pareto_frontier),
        }
    )


@app.command("explain")
def explain_command(plan: Annotated[Path, typer.Argument(exists=True, dir_okay=False)]) -> None:
    console.print(explain_plan(load_deployment_plan(plan)))


def _models_for_plan(plan_path: Path) -> CalibratedModels:
    plan = load_deployment_plan(plan_path)
    models_path = Path(plan.provenance.evidence_bundle_uri) / "models.json"
    if not models_path.is_absolute():
        models_path = _root() / models_path
    if not models_path.exists():
        raise FileNotFoundError(
            f"plan evidence is missing calibrated service curves: {models_path}"
        )
    return CalibratedModels.model_validate_json(models_path.read_text(encoding="utf-8"))


@app.command("replay")
def replay_command(
    plan: Annotated[Path, typer.Option("--plan", exists=True, dir_okay=False)],
    trace: Annotated[Path, typer.Option("--trace", exists=True, dir_okay=False)],
    output: Annotated[Path, typer.Option("--output", "-o")] = Path("artifacts/replay/result.json"),
    seed: Annotated[int, typer.Option("--seed")] = 41,
    faults: Annotated[bool, typer.Option("--faults/--no-faults")] = False,
) -> None:
    deployment = load_deployment_plan(plan)
    run = replay_simulator(
        plan=deployment,
        models=_models_for_plan(plan),
        trace=load_trace(trace),
        output_path=output,
        chrome_trace_path=output.with_name("trace.json"),
        repository_root=_root(),
        seed=seed,
        inject_required_faults=faults,
    )
    _json_result({"output": str(output), **run.summary})


@app.command("serve")
def serve_command(
    plan: Annotated[Path, typer.Option("--plan", exists=True, dir_okay=False)],
    backend: Annotated[list[str] | None, typer.Option("--backend")] = None,
    bind: Annotated[str, typer.Option("--bind")] = "127.0.0.1:8080",
) -> None:
    deployment = load_deployment_plan(plan)
    backend_urls = backend or ["http://127.0.0.1:8000"]
    config_path = plan.with_suffix(".gateway.json")
    config = {
        "bind": bind,
        "backends": [
            {
                "name": f"backend-{index}",
                "base_url": url,
                "capacity": deployment.batching.maximum_active_sequences,
                "estimated_service_ms": deployment.predicted_metrics["p95_e2e_ms"].point,
                "price_per_hour_usd": deployment.hardware.hourly_price_usd,
                "health_path": "/health",
                "weight": 1,
            }
            for index, url in enumerate(backend_urls)
        ],
        "routing_policy": {
            "round_robin": "round_robin",
            "least_outstanding": "least_outstanding",
            "earliest_finish": "estimated_earliest_finish",
            "slo_slack": "slo_slack_aware",
        }[deployment.routing.kind.value],
        "admission_capacity": deployment.admission.queue_capacity,
        "trace_output": str(plan.with_suffix(".gateway-trace.json")),
        "provenance": {"plan_id": deployment.metadata.uid},
    }
    write_json(config_path, config)
    binary = shutil.which("sloforge-gateway")
    command = [binary] if binary else ["cargo", "run", "-q", "-p", "sloforge-gateway", "--"]
    command += ["serve", "--config", str(config_path)]
    os.execvp(command[0], command)


@app.command("chaos")
def chaos_command(
    scenario: Annotated[Path, typer.Option("--scenario", exists=True, dir_okay=False)],
    output: Annotated[Path, typer.Option("--output", "-o")] = Path("artifacts/chaos/result.json"),
) -> None:
    result = execute_scenario(load_scenario(scenario))
    write_json(output, result.model_dump())
    _json_result(
        {
            "output": str(output),
            "faults": len(result.executions),
            "diagnosis_accuracy": result.diagnosis_accuracy,
        }
    )


def _export_context(plan_path: Path) -> ExportContext:
    plan = load_deployment_plan(plan_path)
    engine = "tensorrt-llm" if plan.engine.runtime == "tensorrt_llm" else plan.engine.runtime
    return ExportContext(
        plan_id=plan.metadata.uid,
        model_id=plan.model.model_id,
        model_revision=plan.model.revision,
        engine=engine,
        dtype=cast(
            Literal["float32", "float16", "bfloat16", "int8", "int4"],
            plan.engine.dtype.value,
        ),
        accelerator=plan.hardware.gpus[0].product if plan.hardware.gpus else None,
        gpu_count=len(plan.hardware.gpus),
        cpu_cores=float(plan.hardware.cpu.physical_cores),
        memory_gib=max(1, plan.hardware.system_memory_bytes // (1024**3)),
        min_replicas=plan.replica_topology.minimum_replicas,
        max_replicas=plan.replica_topology.maximum_replicas,
        concurrency=plan.batching.maximum_active_sequences,
        max_batched_tokens=plan.batching.maximum_batched_tokens,
        max_sequence_length=plan.model.maximum_sequence_length,
        regions=list(plan.replica_topology.regions),
        scaledown_window_s=int(plan.autoscaling.scale_down_cooldown_seconds),
        estimated_service_ms=plan.predicted_metrics["p95_e2e_ms"].point,
        hourly_price_usd=plan.hardware.hourly_price_usd,
    )


@app.command("export")
def export_command(
    plan: Annotated[Path, typer.Option("--plan", exists=True, dir_okay=False)],
    target: Annotated[
        Literal["local", "docker", "kubernetes", "modal", "truss"], typer.Option("--target")
    ],
    output: Annotated[Path, typer.Option("--output", "-o")],
) -> None:
    result = export_plan(
        context=_export_context(plan), target=target, output=output, repository_root=_root()
    )
    _json_result(result.model_dump())


@app.command("report")
def report_command(
    evidence: Annotated[Path, typer.Option("--evidence", exists=True)],
    output: Annotated[Path, typer.Option("--output", "-o")],
    artifact_index: Annotated[Path | None, typer.Option("--artifact-index")] = None,
) -> None:
    evidence_path = evidence / "evidence.json" if evidence.is_dir() else evidence
    index = artifact_index or evidence_path.with_name("artifact-index.json")
    bundle = load_evidence_bundle(evidence_path)
    plan_path = next(
        (
            path
            for path in _root().rglob("*.json")
            if path.is_file()
            and sha256_file(path) == bundle.artifact_hashes["deployment_plan"].value
        ),
        None,
    )
    if plan_path is None:
        raise FileNotFoundError("could not locate the plan referenced by the EvidenceBundle")
    result = generate_report(
        evidence_path=evidence_path,
        artifact_index_path=index,
        output_dir=output,
        repository_root=_root(),
        plan_explanation=explain_plan(load_deployment_plan(plan_path)),
    )
    _json_result(result.model_dump())


@app.command("demo")
def demo_command(
    artifact_dir: Annotated[Path, typer.Option("--artifact-dir")] = Path("artifacts/demo"),
    report_dir: Annotated[Path, typer.Option("--report-dir")] = Path("reports/demo"),
    reset: Annotated[bool, typer.Option("--reset")] = False,
) -> None:
    run_demo(
        repository_root=_root(),
        artifact_dir=(_root() / artifact_dir).resolve(),
        report_dir=(_root() / report_dir).resolve(),
        reset=reset,
    )


if __name__ == "__main__":
    app()
