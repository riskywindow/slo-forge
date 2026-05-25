from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

from sloforge.controller.core import ControllerEvaluation
from sloforge.faults.engine import FaultScenarioResult
from sloforge.ir import load_deployment_plan, load_evidence_bundle
from sloforge.models.service_curve import CalibratedModels
from sloforge.optimizer.core import CandidateConfiguration, Metrics, OptimizationResult
from sloforge.util import sha256_file, utc_now, write_json


class IndexedArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    path: str
    sha256: str
    media_type: str


class ArtifactIndex(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["sloforge.artifact-index/v1"] = "sloforge.artifact-index/v1"
    generated_at: str
    artifacts: list[IndexedArtifact]

    def path_for(self, name: str, *, repository_root: Path) -> Path:
        artifact = next((item for item in self.artifacts if item.name == name), None)
        if artifact is None:
            raise KeyError(f"artifact index does not contain {name!r}")
        path = repository_root / artifact.path
        actual = sha256_file(path)
        if actual != artifact.sha256:
            raise ValueError(
                f"artifact hash mismatch for {name}: expected {artifact.sha256}, got {actual}"
            )
        return path


class EvaluationMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid")

    selected_config_id: str
    selected_cost_per_million_tokens: float
    selected_p95_ttft_ms: float
    selected_p99_itl_ms: float
    pareto_candidate_count: int
    measured_trial_count: int
    prefill_held_out_mape: float
    prediction_interval_coverage: float
    decode_held_out_mape: float
    decode_interval_coverage: float
    predictive_controller_violations: int
    reactive_controller_violations: int
    predictive_controller_cost_usd: float
    reactive_controller_cost_usd: float
    diagnosis_accuracy: float
    diagnosis_false_positive_rate: float
    diagnosis_negative_window_count: int
    diagnosis_mean_latency_ms: float
    replay_request_count: int
    replay_slo_attainment: float


class ReportResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["sloforge.report/v1"] = "sloforge.report/v1"
    generated_at: str
    evidence_path: str
    metrics: EvaluationMetrics
    verified_artifact_count: int
    outputs: dict[str, str]


class ServingBaselineEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    baseline: Literal["documented-engine-default", "manual-static", "sloforge-plan"]
    regime: Literal["steady", "bursty", "short-prompts", "long-prompts", "mixed"]
    configuration: CandidateConfiguration
    fidelity: Literal["calibrated_prediction"]
    metrics: Metrics
    uncertainty: Metrics
    feasible: bool
    rejection_reasons: list[str]


class ServingBaselines(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["sloforge.serving-baselines/v1"]
    source_profile_id: str
    measurement_claim: str
    results: list[ServingBaselineEntry]


def build_artifact_index(
    *, repository_root: Path, artifacts: dict[str, tuple[Path, str]], output: Path
) -> ArtifactIndex:
    indexed: list[IndexedArtifact] = []
    for name, (path, media_type) in sorted(artifacts.items()):
        resolved = path.resolve()
        try:
            relative = resolved.relative_to(repository_root.resolve())
        except ValueError as exc:
            raise ValueError(f"artifact {path} is outside repository root") from exc
        indexed.append(
            IndexedArtifact(
                name=name,
                path=str(relative),
                sha256=sha256_file(resolved),
                media_type=media_type,
            )
        )
    index = ArtifactIndex(generated_at=utc_now(), artifacts=indexed)
    write_json(output, index.model_dump())
    return index


def _report_path(path: Path, repository_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(repository_root.resolve()))
    except ValueError:
        return str(path.resolve())


def _svg_scatter(optimization: OptimizationResult) -> str:
    width, height, padding = 720, 400, 55
    points = [item for item in optimization.evaluated_candidates if item.feasible]
    if not points:
        raise ValueError("cannot plot an empty feasible candidate set")
    xs = [item.predicted.cost_per_million_tokens for item in points]
    ys = [item.predicted.p95_ttft_ms for item in points]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    def sx(value: float) -> float:
        return padding + (value - min_x) / max(max_x - min_x, 1e-9) * (width - 2 * padding)

    def sy(value: float) -> float:
        return (
            height - padding - (value - min_y) / max(max_y - min_y, 1e-9) * (height - 2 * padding)
        )

    frontier_ids = {item.configuration.config_id for item in optimization.pareto_frontier}
    circles = "\n".join(
        f'<circle cx="{sx(x):.2f}" cy="{sy(y):.2f}" r="{5 if item.configuration.config_id in frontier_ids else 2.5}" fill="{("#ea580c" if item.configuration.config_id in frontier_ids else "#64748b")}" opacity="0.8"><title>{html.escape(item.configuration.config_id)}</title></circle>'
        for x, y, item in zip(xs, ys, points, strict=True)
    )
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="#fff"/>
<line x1="{padding}" y1="{height - padding}" x2="{width - padding}" y2="{height - padding}" stroke="#0f172a"/>
<line x1="{padding}" y1="{padding}" x2="{padding}" y2="{height - padding}" stroke="#0f172a"/>
<text x="{width / 2}" y="{height - 12}" text-anchor="middle" font-family="sans-serif" font-size="13">Cost (USD / million tokens)</text>
<text x="16" y="{height / 2}" transform="rotate(-90 16 {height / 2})" text-anchor="middle" font-family="sans-serif" font-size="13">p95 TTFT (ms)</text>
<text x="{padding}" y="{height - padding + 18}" font-family="monospace" font-size="10">{min_x:.3f}</text>
<text x="{width - padding}" y="{height - padding + 18}" text-anchor="end" font-family="monospace" font-size="10">{max_x:.3f}</text>
<text x="{padding - 5}" y="{height - padding}" text-anchor="end" font-family="monospace" font-size="10">{min_y:.1f}</text>
<text x="{padding - 5}" y="{padding}" text-anchor="end" font-family="monospace" font-size="10">{max_y:.1f}</text>
{circles}
</svg>'''


def _svg_controller(controller: ControllerEvaluation) -> str:
    width, height, padding = 720, 320, 45
    states = controller.windows
    if not states:
        raise ValueError("controller evaluation has no windows")
    max_rate = max(item.arrival_rate_rps for item in states) or 1.0
    rate_points = " ".join(
        f"{padding + index / max(1, len(states) - 1) * (width - 2 * padding):.1f},{height - padding - item.arrival_rate_rps / max_rate * (height - 2 * padding):.1f}"
        for index, item in enumerate(states)
    )
    actions = "\n".join(
        f'<line x1="{padding + decision.observed.window_index / max(1, len(states) - 1) * (width - 2 * padding):.1f}" y1="{padding}" x2="{padding + decision.observed.window_index / max(1, len(states) - 1) * (width - 2 * padding):.1f}" y2="{height - padding}" stroke="#ea580c" stroke-dasharray="3,3" opacity="0.55"/>'
        for decision in controller.predictive_decisions
        if decision.chosen.action_type != "hold"
    )
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="#fff"/>
<line x1="{padding}" y1="{height - padding}" x2="{width - padding}" y2="{height - padding}" stroke="#0f172a"/>
{actions}
<polyline points="{rate_points}" fill="none" stroke="#2563eb" stroke-width="2"/>
<text x="{width / 2}" y="{height - 10}" text-anchor="middle" font-family="sans-serif" font-size="13">Control windows (orange = predictive action)</text>
<text x="15" y="{height / 2}" transform="rotate(-90 15 {height / 2})" text-anchor="middle" font-family="sans-serif" font-size="13">arrival requests/s</text>
</svg>'''


def _prometheus(metrics: EvaluationMetrics) -> str:
    rows = [
        ("sloforge_plan_ttft_p95_milliseconds", metrics.selected_p95_ttft_ms),
        ("sloforge_plan_itl_p99_milliseconds", metrics.selected_p99_itl_ms),
        ("sloforge_plan_cost_per_million_tokens", metrics.selected_cost_per_million_tokens),
        ("sloforge_prediction_mape_ratio", metrics.prefill_held_out_mape),
        ("sloforge_prediction_interval_coverage_ratio", metrics.prediction_interval_coverage),
        ("sloforge_decode_prediction_mape_ratio", metrics.decode_held_out_mape),
        ("sloforge_decode_interval_coverage_ratio", metrics.decode_interval_coverage),
        (
            'sloforge_controller_slo_violations_total{controller="predictive"}',
            metrics.predictive_controller_violations,
        ),
        (
            'sloforge_controller_slo_violations_total{controller="reactive"}',
            metrics.reactive_controller_violations,
        ),
        ("sloforge_diagnosis_accuracy_ratio", metrics.diagnosis_accuracy),
        ("sloforge_diagnosis_false_positive_ratio", metrics.diagnosis_false_positive_rate),
        ("sloforge_diagnosis_negative_windows_total", metrics.diagnosis_negative_window_count),
        ("sloforge_replay_slo_attainment_ratio", metrics.replay_slo_attainment),
    ]
    return (
        "\n".join(f"# TYPE {name.split('{')[0]} gauge\n{name} {value}" for name, value in rows)
        + "\n"
    )


def _otel_trace(controller: ControllerEvaluation, chaos: FaultScenarioResult) -> dict[str, object]:
    spans: list[dict[str, object]] = []
    trace_id = "0" * 16 + "5a10f0a6e0000001"
    for index, decision in enumerate(controller.predictive_decisions):
        start_ns = int(decision.observed.window_start_ms * 1_000_000)
        spans.append(
            {
                "traceId": trace_id,
                "spanId": f"{index + 1:016x}",
                "name": "sloforge.controller.decision",
                "kind": 1,
                "startTimeUnixNano": str(start_ns),
                "endTimeUnixNano": str(start_ns + 2_000_000),
                "attributes": [
                    {
                        "key": "sloforge.action",
                        "value": {"stringValue": decision.chosen.action_type},
                    },
                    {
                        "key": "sloforge.replicas",
                        "value": {"intValue": str(decision.chosen.target_replicas)},
                    },
                    {"key": "sloforge.rollback", "value": {"boolValue": decision.rolled_back}},
                ],
                "status": {"code": 1},
            }
        )
    offset = len(spans)
    for index, execution in enumerate(chaos.executions):
        start_ns = int(execution.event.start_ms * 1_000_000)
        spans.append(
            {
                "traceId": trace_id,
                "spanId": f"{offset + index + 1:016x}",
                "name": "sloforge.chaos.fault",
                "kind": 1,
                "startTimeUnixNano": str(start_ns),
                "endTimeUnixNano": str(start_ns + execution.event.duration_ms * 1_000_000),
                "attributes": [
                    {
                        "key": "sloforge.fault_type",
                        "value": {"stringValue": execution.event.fault_type},
                    },
                    {
                        "key": "sloforge.diagnosis",
                        "value": {"stringValue": execution.diagnosis.predicted_label},
                    },
                ],
                "status": {"code": 1 if execution.diagnosis.correct else 2},
            }
        )
    return {
        "resourceSpans": [
            {
                "resource": {
                    "attributes": [
                        {"key": "service.name", "value": {"stringValue": "sloforge-demo"}}
                    ]
                },
                "scopeSpans": [{"scope": {"name": "sloforge", "version": "0.1.0"}, "spans": spans}],
            }
        ]
    }


def _chrome_trace(
    controller: ControllerEvaluation, chaos: FaultScenarioResult
) -> dict[str, object]:
    events: list[dict[str, object]] = []
    for decision in controller.predictive_decisions:
        events.append(
            {
                "name": f"controller:{decision.chosen.action_type}",
                "cat": "control",
                "ph": "X",
                "ts": decision.observed.window_start_ms * 1000,
                "dur": 2000,
                "pid": 1,
                "tid": 1,
                "args": {
                    "replicas": decision.chosen.target_replicas,
                    "rollback": decision.rolled_back,
                },
            }
        )
    for execution in chaos.executions:
        events.append(
            {
                "name": f"fault:{execution.event.fault_type}",
                "cat": "chaos",
                "ph": "X",
                "ts": execution.event.start_ms * 1000,
                "dur": execution.event.duration_ms * 1000,
                "pid": 1,
                "tid": 2,
                "args": {
                    "diagnosis": execution.diagnosis.predicted_label,
                    "correct": execution.diagnosis.correct,
                },
            }
        )
    return {"traceEvents": events, "displayTimeUnit": "ms"}


def _markdown(
    metrics: EvaluationMetrics,
    optimization: OptimizationResult,
    controller: ControllerEvaluation,
    chaos: FaultScenarioResult,
    plan_explanation: str,
    serving_baselines: ServingBaselines | None,
) -> str:
    baseline_rows = "\n".join(
        f"| {item.strategy} | {item.trials} | {item.best_config_id or 'none'} | {item.best_objective if item.best_objective is not None else 'n/a'} |"
        for item in optimization.baseline_outcomes
    )
    serving_rows = (
        "\n".join(
            f"| {item.baseline} | {item.regime} | {item.metrics.p95_ttft_ms:.2f} | {item.metrics.p99_itl_ms:.2f} | {item.metrics.cost_per_million_tokens:.4f} | {item.fidelity} |"
            for item in serving_baselines.results
        )
        if serving_baselines is not None
        else "| unavailable | unavailable | n/a | n/a | n/a | n/a |"
    )
    return f"""# SLOForge CPU evaluation

Generated from hashed raw artifacts. No GPU was present, so these results characterize the deterministic CPU mock environment only.

## Selected plan

{plan_explanation}

| Metric | Measured/predicted result |
|---|---:|
| p95 TTFT | {metrics.selected_p95_ttft_ms:.3f} ms |
| p99 ITL | {metrics.selected_p99_itl_ms:.3f} ms |
| Cost | ${metrics.selected_cost_per_million_tokens:.5f} / million tokens |
| Held-out prefill MAPE | {metrics.prefill_held_out_mape:.3%} |
| Prefill interval coverage | {metrics.prediction_interval_coverage:.3%} |
| Held-out decode MAPE | {metrics.decode_held_out_mape:.3%} |
| Decode interval coverage | {metrics.decode_interval_coverage:.3%} |
| Replay SLO attainment | {metrics.replay_slo_attainment:.3%} |

## H1: profiling efficiency

| Strategy | Trials | Best configuration | Best objective |
|---|---:|---|---:|
{baseline_rows}

The comparison is a CPU mock study. It validates budget accounting and search behavior; it is not evidence of reduced GPU minutes until the GPU matrix is run.

## H2: compiled plan

The selected configuration is one of {metrics.pareto_candidate_count} non-dominated feasible candidates.

| Configuration | Workload regime | p95 TTFT (ms) | p99 ITL (ms) | Cost / million tokens | Fidelity |
|---|---|---:|---:|---:|---|
{serving_rows}

These comparisons are calibrated predictions backed by raw profile measurements, not relabeled configuration measurements. Wins and losses must be interpreted with the reported uncertainty and held-out coverage.

## H3: adaptive control

- Predictive controller: {controller.predictive.slo_violations} request deadline misses in the calibrated Rust twin, ${controller.predictive.estimated_cost_usd:.6f} simulated cost, {controller.predictive.replica_oscillations} policy oscillations.
- Reactive controller: {controller.reactive.slo_violations} request deadline misses in the calibrated Rust twin, ${controller.reactive.estimated_cost_usd:.6f} simulated cost, {controller.reactive.replica_oscillations} policy oscillations.

## H4: bottleneck diagnosis

Closed-set label agreement was {chaos.diagnosis_accuracy:.2%} across {len(chaos.executions)} labeled synthetic injections. The classifier produced {chaos.false_positive_count} false positives across {chaos.negative_window_count} negative control windows ({chaos.false_positive_rate:.2%}) and mean execution latency was {chaos.mean_diagnosis_latency_ms:.3f} ms. The complete confusion matrix, counters, and failures remain visible in `chaos-result.json`; this is not an estimate of field diagnosis accuracy.

## Limitations

- The host has no NVIDIA GPU; Transformers, vLLM, and SGLang commands are implemented but unexercised here.
- Mock-backend measurements test systems logic, determinism, queueing, and fault handling; they are not model performance claims.
- Performance intervals apply only to the profiled configurations and workload.
"""


def generate_report(
    *,
    evidence_path: Path,
    artifact_index_path: Path,
    output_dir: Path,
    repository_root: Path,
    plan_explanation: str,
) -> ReportResult:
    evidence = load_evidence_bundle(evidence_path)
    index = ArtifactIndex.model_validate_json(artifact_index_path.read_text(encoding="utf-8"))
    verified = {
        item.name: index.path_for(item.name, repository_root=repository_root)
        for item in index.artifacts
    }
    evidence_index_names = {
        "deployment_plan": "plan",
        "measurements": "raw_measurements",
        "simulator_replay": "replay",
        **{
            name: name
            for name in (
                "profile",
                "hardware",
                "workload",
                "models",
                "optimization",
                "gateway_replay",
                "controller",
                "predictive_controller_twin",
                "reactive_controller_twin",
                "chaos",
                "command_transcript",
                "serving_baselines",
                "simulator_scenario",
                "simulator_raw",
                "simulator_trace",
                "predictive_controller_scenario",
                "predictive_controller_raw",
                "predictive_controller_trace",
                "reactive_controller_scenario",
                "reactive_controller_raw",
                "reactive_controller_trace",
            )
        },
    }
    indexed_by_name = {item.name: item for item in index.artifacts}
    for evidence_name, digest in evidence.artifact_hashes.items():
        index_name = evidence_index_names.get(evidence_name)
        indexed = indexed_by_name.get(index_name) if index_name is not None else None
        if indexed is None:
            raise ValueError(
                f"EvidenceBundle hash {evidence_name!r} has no corresponding indexed artifact"
            )
        if indexed.sha256 != digest.value:
            raise ValueError(
                f"EvidenceBundle hash mismatch for {evidence_name}: "
                f"expected {digest.value}, indexed {indexed.sha256}"
            )
    plan = load_deployment_plan(verified["plan"])
    if evidence.plan_digest.value != __import__(
        "sloforge.ir", fromlist=["canonical_hash"]
    ).canonical_hash(plan):
        raise ValueError("EvidenceBundle plan digest does not match the indexed DeploymentPlan")
    optimization = OptimizationResult.model_validate_json(
        verified["optimization"].read_text(encoding="utf-8")
    )
    models = CalibratedModels.model_validate_json(verified["models"].read_text(encoding="utf-8"))
    controller = ControllerEvaluation.model_validate_json(
        verified["controller"].read_text(encoding="utf-8")
    )
    chaos = FaultScenarioResult.model_validate_json(verified["chaos"].read_text(encoding="utf-8"))
    serving_baselines = (
        ServingBaselines.model_validate_json(
            verified["serving_baselines"].read_text(encoding="utf-8")
        )
        if "serving_baselines" in verified
        else None
    )
    replay = json.loads(verified["replay"].read_text(encoding="utf-8"))
    replay_count = int(replay["summary"]["request_count"])
    replay_attainment = float(replay["summary"]["slo_attainment"])
    selected_metrics = optimization.selected.predicted
    selected_model = next(
        item
        for item in models.candidates
        if item.candidate_id == optimization.selected.configuration.backend_candidate_id
    )
    metrics = EvaluationMetrics(
        selected_config_id=optimization.selected.configuration.config_id,
        selected_cost_per_million_tokens=selected_metrics.cost_per_million_tokens,
        selected_p95_ttft_ms=selected_metrics.p95_ttft_ms,
        selected_p99_itl_ms=selected_metrics.p99_itl_ms,
        pareto_candidate_count=len(optimization.pareto_frontier),
        measured_trial_count=sum(
            item.fidelity == "measured" for item in optimization.evaluated_candidates
        ),
        prefill_held_out_mape=selected_model.held_out_mape,
        prediction_interval_coverage=selected_model.interval_coverage,
        decode_held_out_mape=selected_model.decode_held_out_mape,
        decode_interval_coverage=selected_model.decode_interval_coverage,
        predictive_controller_violations=controller.predictive.slo_violations,
        reactive_controller_violations=controller.reactive.slo_violations,
        predictive_controller_cost_usd=controller.predictive.estimated_cost_usd,
        reactive_controller_cost_usd=controller.reactive.estimated_cost_usd,
        diagnosis_accuracy=chaos.diagnosis_accuracy,
        diagnosis_false_positive_rate=chaos.false_positive_rate,
        diagnosis_negative_window_count=chaos.negative_window_count,
        diagnosis_mean_latency_ms=chaos.mean_diagnosis_latency_ms,
        replay_request_count=replay_count,
        replay_slo_attainment=replay_attainment,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    markdown = _markdown(
        metrics, optimization, controller, chaos, plan_explanation, serving_baselines
    )
    (output_dir / "evaluation.md").write_text(markdown, encoding="utf-8")
    body = html.escape(markdown)
    document = f"<!doctype html><html><head><meta charset='utf-8'><title>SLOForge evaluation</title><style>body{{font-family:system-ui;max-width:1000px;margin:2rem auto;line-height:1.5}}pre{{white-space:pre-wrap}}img{{max-width:100%}}</style></head><body><h1>SLOForge CPU evaluation</h1><img src='pareto.svg' alt='Pareto frontier'><img src='controller.svg' alt='Controller timeline'><pre>{body}</pre></body></html>"
    (output_dir / "evaluation.html").write_text(document, encoding="utf-8")
    (output_dir / "pareto.svg").write_text(_svg_scatter(optimization), encoding="utf-8")
    (output_dir / "controller.svg").write_text(_svg_controller(controller), encoding="utf-8")
    (output_dir / "metrics.prom").write_text(_prometheus(metrics), encoding="utf-8")
    write_json(output_dir / "otel-traces.json", _otel_trace(controller, chaos))
    write_json(output_dir / "trace.json", _chrome_trace(controller, chaos))
    write_json(output_dir / "evaluation.json", metrics.model_dump())
    report_data = {
        "plan": plan.model_dump(mode="json"),
        "metrics": metrics.model_dump(),
        "pareto_frontier": [item.model_dump(mode="json") for item in optimization.pareto_frontier],
        "controller": controller.model_dump(mode="json"),
        "chaos": chaos.model_dump(mode="json"),
        "serving_baselines": (
            serving_baselines.model_dump(mode="json") if serving_baselines is not None else None
        ),
    }
    write_json(output_dir / "report-data.json", report_data)
    output_paths = {
        name: _report_path(output_dir / name, repository_root)
        for name in (
            "evaluation.md",
            "evaluation.html",
            "evaluation.json",
            "pareto.svg",
            "controller.svg",
            "metrics.prom",
            "otel-traces.json",
            "trace.json",
            "report-data.json",
        )
    }
    result = ReportResult(
        generated_at=utc_now(),
        evidence_path=_report_path(evidence_path, repository_root),
        metrics=metrics,
        verified_artifact_count=len(verified),
        outputs=output_paths,
    )
    write_json(output_dir / "report-manifest.json", result.model_dump())
    return result
