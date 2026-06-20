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
from sloforge.util import canonical_json, sha256_bytes, sha256_file

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


def _contained_bundle_path(bundle: Path, value: object, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise typer.BadParameter(f"Autopsy replay metadata {field} must be a relative path")
    candidate = (bundle / value).resolve()
    boundary = bundle.parent.resolve()
    try:
        candidate.relative_to(boundary)
    except ValueError as error:
        raise typer.BadParameter(f"Autopsy replay metadata {field} escapes its bundle") from error
    if not candidate.is_file():
        raise typer.BadParameter(f"Autopsy replay metadata {field} is missing: {candidate}")
    return candidate


def _verify_bundle_hash(metadata: dict[object, object], field: str, path: Path) -> None:
    expected = metadata.get(f"{field}_sha256")
    if not isinstance(expected, str) or sha256_file(path) != expected:
        raise typer.BadParameter(f"Autopsy replay {field} hash does not match metadata")


def _resolve_replay_bundle(
    evidence: Path,
    baseline: Path | None,
    simulation_input: Path | None,
    healthy_reference_us: float | None,
    output: Path | None,
) -> tuple[Path, Path, Path, float, Path]:
    if evidence.is_file():
        if baseline is None or simulation_input is None or healthy_reference_us is None:
            raise typer.BadParameter(
                "file evidence requires --baseline, --simulation-input, and "
                "--healthy-reference-us; pass a replay-bundle directory to resolve them"
            )
        return (
            evidence,
            baseline,
            simulation_input,
            healthy_reference_us,
            output or evidence.with_name("counterfactual-replay.json"),
        )

    metadata_path = evidence / "replay-metadata.json"
    if not metadata_path.is_file():
        raise typer.BadParameter(f"Autopsy replay bundle is missing {metadata_path.name}")
    metadata = load_yaml_or_json(metadata_path)
    if not isinstance(metadata, dict):
        raise typer.BadParameter("Autopsy replay metadata must contain one JSON object")
    degraded_path = _contained_bundle_path(evidence, metadata.get("degraded"), "degraded")
    _verify_bundle_hash(metadata, "degraded", degraded_path)
    if baseline is None:
        baseline_path = _contained_bundle_path(evidence, metadata.get("baseline"), "baseline")
        _verify_bundle_hash(metadata, "baseline", baseline_path)
    else:
        baseline_path = baseline
    input_path = simulation_input or _contained_bundle_path(
        evidence, metadata.get("simulation_input"), "simulation_input"
    )
    reference = healthy_reference_us
    if reference is None:
        raw_reference = metadata.get("healthy_reference_us")
        if isinstance(raw_reference, bool) or not isinstance(raw_reference, int | float):
            raise typer.BadParameter("Autopsy replay metadata healthy_reference_us must be numeric")
        reference = float(raw_reference)
    if reference < 0.0:
        raise typer.BadParameter("healthy reference must be non-negative")
    _verify_bundle_hash(metadata, "simulation_input", input_path)
    return (
        degraded_path,
        baseline_path,
        input_path,
        reference,
        output or evidence / "cli-counterfactual-replay.json",
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
    evidence: Annotated[Path, typer.Option("--evidence", exists=True)],
    counterfactual: Annotated[Path, typer.Option("--counterfactual", exists=True, dir_okay=False)],
    baseline: Annotated[
        Path | None, typer.Option("--baseline", exists=True, dir_okay=False)
    ] = None,
    simulation_input: Annotated[
        Path | None, typer.Option("--simulation-input", exists=True, dir_okay=False)
    ] = None,
    healthy_reference_us: Annotated[
        float | None, typer.Option("--healthy-reference-us", min=0.0)
    ] = None,
    output: Annotated[Path | None, typer.Option("--output", "-o")] = None,
    timeout_seconds: Annotated[float, typer.Option("--timeout-seconds", min=0.1)] = 120.0,
) -> None:
    degraded_path, baseline_path, input_path, reference_us, output_path = _resolve_replay_bundle(
        evidence,
        baseline,
        simulation_input,
        healthy_reference_us,
        output,
    )
    healthy = _run(baseline_path)
    degraded = _run(degraded_path)
    comparison = compare_runs(healthy, degraded)
    record = diagnose(degraded, comparison=comparison, baseline=healthy)
    request_value = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(request_value, dict):
        raise typer.BadParameter("simulation input must contain one JSON object")
    replay = replay_counterfactuals(
        diagnosis=record,
        simulation_request=request_value,
        scenarios=_scenarios(counterfactual),
        healthy_reference_us=reference_us,
        runner=bind_subprocess_runner(repository_root=repository_root(), timeout_s=timeout_seconds),
    )
    write_model(output_path, replay)
    json_result(
        {
            "output": str(output_path),
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
    original_top = original.hypotheses[0]
    if original_top.rejected_reason is not None or not original_top.supporting_evidence:
        raise typer.BadParameter(
            "the source run has no supported top diagnosis to preserve during minimization"
        )

    def preserves(candidate: AutopsyRun) -> bool:
        try:
            candidate_comparison = compare_runs(healthy, candidate)
            result = diagnose(
                candidate,
                comparison=candidate_comparison,
                baseline=healthy,
            )
        except ValueError:
            return False
        candidate_top = result.hypotheses[0]
        return (
            candidate_comparison.matched_event_count > 0
            and result.top_hypothesis is original.top_hypothesis
            and candidate_top.rejected_reason is None
            and bool(candidate_top.supporting_evidence)
        )

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
