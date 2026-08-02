"""Artifact-driven causal debugging commands."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

import typer

from sloforge.autopsy import (
    AutopsyRun,
    CounterfactualScenario,
    DiagnosisRecord,
    capture_simulation_run,
    compare_runs,
    diagnose,
    minimize_run,
    replay_counterfactuals,
)
from sloforge.autopsy.counterfactual import bind_subprocess_runner
from sloforge.fabric.ir import canonical_hash, load_physical_execution_plan
from sloforge.fabric.simulation import FabricSimulationOutput, FabricSimulationRequest
from sloforge.util import canonical_json, sha256_bytes

from .common import console, json_result, load_yaml_or_json, repository_root, write_model

autopsy_app = typer.Typer(help="Capture, compare, and causally diagnose Fabric evidence.")


def _run(path: Path) -> AutopsyRun:
    return AutopsyRun.model_validate_json(path.read_text(encoding="utf-8"), strict=True)


def _diagnosis(path: Path) -> DiagnosisRecord:
    return DiagnosisRecord.model_validate_json(path.read_text(encoding="utf-8"), strict=True)


def _scenarios(path: Path) -> tuple[CounterfactualScenario, ...]:
    decoded: Any = load_yaml_or_json(path)
    values = decoded.get("scenarios") if isinstance(decoded, dict) else decoded
    if not isinstance(values, list):
        values = [decoded]
    return tuple(
        CounterfactualScenario.model_validate_json(
            json.dumps(value, sort_keys=True, separators=(",", ":")), strict=True
        )
        for value in values
    )


@autopsy_app.command("capture")
def capture_command(
    simulation_input: Annotated[
        Path, typer.Option("--simulation-input", exists=True, dir_okay=False)
    ],
    simulation_output: Annotated[
        Path, typer.Option("--simulation-output", exists=True, dir_okay=False)
    ],
    plan: Annotated[Path, typer.Option("--plan", exists=True, dir_okay=False)],
    output: Annotated[Path, typer.Option("--output", "-o")],
    run_id: Annotated[str, typer.Option("--run-id")],
    topology_fingerprint: Annotated[str, typer.Option("--topology-fingerprint")],
    workload_fingerprint: Annotated[str | None, typer.Option("--workload-fingerprint")] = None,
) -> None:
    """Normalize an actual simulator run; privileged live capture is never implied."""

    request = FabricSimulationRequest.model_validate_json(
        simulation_input.read_text(encoding="utf-8"), strict=True
    )
    result = FabricSimulationOutput.model_validate_json(
        simulation_output.read_text(encoding="utf-8"), strict=True
    )
    physical = load_physical_execution_plan(plan)
    workload_hash = workload_fingerprint or sha256_bytes(
        canonical_json(request.model_dump(mode="json")).encode()
    )
    captured = capture_simulation_run(
        run_id=run_id,
        request=request,
        output=result,
        plan=physical,
        topology_fingerprint=topology_fingerprint,
        workload_fingerprint=workload_hash,
        artifact_path=simulation_output,
    )
    write_model(output, captured)
    json_result(
        {
            "output": str(output),
            "run_id": captured.run_id,
            "events": len(captured.events),
            "fault_intervals": len(captured.fault_intervals),
            "source": captured.source,
        }
    )


@autopsy_app.command("compare")
def compare_command(
    healthy: Annotated[Path, typer.Option("--healthy", exists=True, dir_okay=False)],
    degraded: Annotated[Path, typer.Option("--degraded", exists=True, dir_okay=False)],
    output: Annotated[Path, typer.Option("--output", "-o")],
) -> None:
    comparison = compare_runs(_run(healthy), _run(degraded))
    write_model(output, comparison)
    json_result(
        {
            "output": str(output),
            "comparison_id": comparison.comparison_id,
            "matched_events": comparison.matched_event_count,
            "first_divergence": comparison.first_divergence_event_id,
        }
    )


@autopsy_app.command("diagnose")
def diagnose_command(
    degraded: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    baseline: Annotated[Path, typer.Option("--baseline", exists=True, dir_okay=False)],
    output: Annotated[Path, typer.Option("--output", "-o")],
) -> None:
    healthy_run = _run(baseline)
    degraded_run = _run(degraded)
    comparison = compare_runs(healthy_run, degraded_run)
    record = diagnose(degraded_run, comparison=comparison, baseline=healthy_run)
    write_model(output, record)
    write_model(output.with_name("comparison.json"), comparison)
    json_result(
        {
            "output": str(output),
            "diagnosis_id": record.diagnosis_id,
            "top_hypothesis": record.top_hypothesis.value,
            "top_three": [item.value for item in record.top_three],
            "confidence": record.confidence,
        }
    )


@autopsy_app.command("replay")
def replay_command(
    evidence: Annotated[Path, typer.Option("--evidence", exists=True, dir_okay=False)],
    baseline: Annotated[Path, typer.Option("--baseline", exists=True, dir_okay=False)],
    simulation_input: Annotated[
        Path, typer.Option("--simulation-input", exists=True, dir_okay=False)
    ],
    counterfactual: Annotated[Path, typer.Option("--counterfactual", exists=True, dir_okay=False)],
    healthy_reference_us: Annotated[float, typer.Option("--healthy-reference-us", min=0.0)],
    output: Annotated[Path, typer.Option("--output", "-o")],
    timeout_seconds: Annotated[float, typer.Option("--timeout-seconds", min=0.1)] = 120.0,
) -> None:
    healthy = _run(baseline)
    degraded = _run(evidence)
    comparison = compare_runs(healthy, degraded)
    record = diagnose(degraded, comparison=comparison, baseline=healthy)
    request_value = json.loads(simulation_input.read_text(encoding="utf-8"))
    if not isinstance(request_value, dict):
        raise typer.BadParameter("simulation input must contain one JSON object")
    replay = replay_counterfactuals(
        diagnosis=record,
        simulation_request=request_value,
        scenarios=_scenarios(counterfactual),
        healthy_reference_us=healthy_reference_us,
        runner=bind_subprocess_runner(repository_root=repository_root(), timeout_s=timeout_seconds),
    )
    write_model(output, replay)
    json_result(
        {
            "output": str(output),
            "diagnosis_id": replay.diagnosis_id,
            "evaluated": len(replay.evaluations),
            "selected": replay.selected_scenario_id,
            "rejected": list(replay.rejected_scenario_ids),
        }
    )


@autopsy_app.command("minimize")
def minimize_command(
    degraded: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    baseline: Annotated[Path, typer.Option("--baseline", exists=True, dir_okay=False)],
    output: Annotated[Path, typer.Option("--output", "-o")],
) -> None:
    healthy = _run(baseline)
    source = _run(degraded)
    original = diagnose(source, comparison=compare_runs(healthy, source), baseline=healthy)

    def preserves(candidate: AutopsyRun) -> bool:
        try:
            result = diagnose(
                candidate,
                comparison=compare_runs(healthy, candidate),
                baseline=healthy,
            )
        except ValueError:
            return False
        return result.top_hypothesis is original.top_hypothesis

    minimized = minimize_run(source, preserves)
    write_model(output, minimized)
    json_result(
        {
            "output": str(output),
            "diagnosis": original.top_hypothesis.value,
            "events_before": minimized.original_event_count,
            "events_after": minimized.minimized_event_count,
            "ranks_before": minimized.original_rank_count,
            "ranks_after": minimized.minimized_rank_count,
            "bundle_sha256": minimized.bundle_sha256,
        }
    )


@autopsy_app.command("report")
def report_command(
    diagnosis_path: Annotated[
        Path, typer.Argument(exists=True, dir_okay=False, metavar="DIAGNOSIS")
    ],
    output: Annotated[Path, typer.Option("--output", "-o")],
) -> None:
    record = _diagnosis(diagnosis_path)
    hypotheses = "\n".join(
        f"- `{item.kind.value}`: confidence {item.confidence:.3f}; "
        f"{item.rejected_reason or 'supported by the recorded signals'}"
        for item in record.hypotheses
    )
    evidence = "\n".join(
        f"- `{item.artifact_uri}` (`sha256:{item.sha256}`)" for item in record.evidence
    )
    report = (
        f"# SLOForge Autopsy: {record.diagnosis_id}\n\n"
        f"Top diagnosis: `{record.top_hypothesis.value}` "
        f"(confidence {record.confidence:.3f}).\n\n"
        f"## Ranked hypotheses\n\n{hypotheses}\n\n"
        f"## Evidence\n\n{evidence or '- No evidence artifacts recorded.'}\n\n"
        f"Diagnosis canonical hash: `{canonical_hash(record)}`.\n"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report, encoding="utf-8")
    console.print(report)
