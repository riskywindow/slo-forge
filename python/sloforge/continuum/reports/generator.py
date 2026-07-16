"""Static reports rendered only after raw Continuum artifacts validate."""

from __future__ import annotations

import hashlib
import html
import os
from pathlib import Path
from typing import Literal

from sloforge.continuum.benchmarking.evaluation import validate_evaluation_artifacts
from sloforge.continuum.benchmarking.models import (
    ArtifactReference,
    EvaluationBundle,
    ReportSet,
)
from sloforge.continuum.conversion import ConversionSelection
from sloforge.continuum.demo import FlagshipDemoResult


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _validated_raw(
    evaluation: EvaluationBundle,
    root: Path,
) -> tuple[tuple[FlagshipDemoResult, ConversionSelection], ...]:
    validated = validate_evaluation_artifacts(evaluation, root=root)
    return tuple((item.flagship, item.conversion) for item in validated)


def _write_text(path: Path, content: str) -> None:
    payload = content.encode("utf-8")
    if not payload or len(payload) > 16 * 1024 * 1024:
        raise ValueError("report must contain 1 byte..16 MiB")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _reference(
    path: Path,
    *,
    root: Path,
    media_type: Literal["text/markdown", "text/html"],
) -> ArtifactReference:
    return ArtifactReference(
        path=path.relative_to(root).as_posix(),
        sha256=_sha256(path),
        size_bytes=path.stat().st_size,
        media_type=media_type,
    )


def _evaluation_markdown(evaluation: EvaluationBundle) -> str:
    lines = [
        "# SLOForge Continuum CPU Evaluation",
        "",
        f"Evaluation ID: `{evaluation.evaluation_id}`",
        "",
        "This report is generated from authenticated per-seed artifacts. Observed host timings "
        "use `perf_counter_ns`; fields prefixed with `synthetic_` come from the deterministic "
        "transport/protocol model and are not hardware measurements. `artifact_derived` values "
        "are exact counts from the sealed state manifests.",
        "",
        "## Reproduction",
        "",
        "```sh",
        evaluation.exact_command,
        "```",
        "",
        "## Environment manifests",
        "",
        f"- Python: {evaluation.software.python_implementation} {evaluation.software.python_version}",
        f"- Platform: {evaluation.software.platform}",
        f"- Git commit: `{evaluation.software.git_commit}`",
        f"- Mode: {evaluation.hardware.mode}",
        f"- Machine: {evaluation.hardware.machine}",
        f"- Logical CPUs: {evaluation.hardware.logical_cpu_count}",
        f"- GPU exercised: {str(evaluation.hardware.gpu_exercised).lower()}",
        f"- RDMA exercised: {str(evaluation.hardware.rdma_exercised).lower()}",
        "",
        "| Package | Version |",
        "|---|---|",
    ]
    lines.extend(f"| {item.package} | {item.version} |" for item in evaluation.software.packages)
    lines.extend(
        (
            "",
            "## Confidence intervals",
            "",
            "All intervals are two-sided 95% Student-t intervals across independent seeds.",
            "",
            "| Metric | Class | Unit | n | Mean | 95% CI |",
            "|---|---|---:|---:|---:|---:|",
        )
    )
    lines.extend(
        f"| {item.metric} | {item.metric_class} | {item.unit} | {item.sample_count} | "
        f"{item.mean:.3f} | [{item.lower:.3f}, {item.upper:.3f}] |"
        for item in evaluation.confidence_intervals
    )
    lines.extend(
        (
            "",
            "## Per-seed raw results",
            "",
            "| Seed | Flagship wall ms | Pre-copy pause ms | Stop-copy pause ms | Canonical µs | Direct µs | Selected | Planner regret | Prediction error ms | Synthetic wire bytes | Tokens | COW bytes saved |",
            "|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|",
        )
    )
    lines.extend(
        f"| {item.seed} | {item.observed_flagship_wall_ns / 1_000_000:.3f} | "
        f"{item.observed_precopy_interruption_ns / 1_000_000:.3f} | "
        f"{item.observed_stop_and_copy_interruption_ns / 1_000_000:.3f} | "
        f"{item.observed_canonical_conversion_median_ns / 1_000:.3f} | "
        f"{item.observed_direct_conversion_median_ns / 1_000:.3f} | "
        f"{item.selected_converter} | {item.planner_regret:.6f} | "
        f"{item.planner_interruption_absolute_error_ms:.3f} | "
        f"{item.synthetic_transport_bytes_on_wire} | "
        f"{item.gateway_accepted_tokens} | {item.checkpoint_bytes_deduplicated} |"
        for item in evaluation.per_seed
    )
    lines.extend(
        (
            "",
            "## Hypothesis outcomes",
            "",
            "| Hypothesis | Status | Result | Scope |",
            "|---|---|---|---|",
        )
    )
    lines.extend(
        f"| {item.hypothesis} | {item.status} | {item.statement} | {item.limitation} |"
        for item in evaluation.hypotheses
    )
    lines.extend(("", "## Negative and unexercised results", ""))
    lines.extend(f"- {result}" for result in evaluation.negative_results)
    lines.extend(("", "## Raw artifact provenance", ""))
    for item in evaluation.per_seed:
        lines.extend(
            (
                f"- Seed {item.seed} flagship: `{item.flagship_artifact.path}` "
                f"(SHA-256 `{item.flagship_artifact.sha256}`)",
                f"- Seed {item.seed} conversion: `{item.conversion_artifact.path}` "
                f"(SHA-256 `{item.conversion_artifact.sha256}`)",
            )
        )
    return "\n".join(lines) + "\n"


def _evaluation_html(evaluation: EvaluationBundle) -> str:
    intervals = "".join(
        "<tr>"
        f"<td>{html.escape(item.metric)}</td>"
        f"<td>{html.escape(item.metric_class)}</td>"
        f"<td>{item.sample_count}</td><td>{item.mean:.3f}</td>"
        f"<td>[{item.lower:.3f}, {item.upper:.3f}] {html.escape(item.unit)}</td>"
        "</tr>"
        for item in evaluation.confidence_intervals
    )
    seeds = "".join(
        "<tr>"
        f"<td>{item.seed}</td>"
        f"<td>{item.observed_flagship_wall_ns / 1_000_000:.3f} ms</td>"
        f"<td>{item.observed_canonical_conversion_median_ns / 1_000:.3f} µs</td>"
        f"<td>{item.observed_direct_conversion_median_ns / 1_000:.3f} µs</td>"
        f"<td>{html.escape(item.selected_converter)}</td>"
        f"<td>{item.synthetic_transport_bytes_on_wire}</td>"
        "</tr>"
        for item in evaluation.per_seed
    )
    hypotheses = "".join(
        f"<li><strong>{item.hypothesis}: {html.escape(item.status)}</strong> — "
        f"{html.escape(item.statement)} Scope: {html.escape(item.limitation)}</li>"
        for item in evaluation.hypotheses
    )
    negatives = "".join(f"<li>{html.escape(item)}</li>" for item in evaluation.negative_results)
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Continuum CPU Evaluation</title>
<style>body{{font-family:system-ui,sans-serif;max-width:1100px;margin:2rem auto;line-height:1.45}}table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #bbb;padding:.4rem;text-align:left}}code{{word-break:break-all}}.warning{{background:#fff4cc;padding:1rem}}</style></head>
<body><h1>SLOForge Continuum CPU Evaluation</h1>
<p>Evaluation ID: <code>{evaluation.evaluation_id}</code></p>
<p class="warning">Observed host timings and deterministic synthetic protocol values are labeled separately. No GPU, RDMA, cloud, or multi-node result is claimed.</p>
<h2>Reproduction</h2><pre><code>{html.escape(evaluation.exact_command)}</code></pre>
<h2>Confidence intervals</h2><table><thead><tr><th>Metric</th><th>Class</th><th>n</th><th>Mean</th><th>95% CI</th></tr></thead><tbody>{intervals}</tbody></table>
<h2>Per-seed results</h2><table><thead><tr><th>Seed</th><th>Flagship wall</th><th>Canonical</th><th>Direct</th><th>Selected</th><th>Synthetic wire bytes</th></tr></thead><tbody>{seeds}</tbody></table>
<h2>Hypotheses</h2><ul>{hypotheses}</ul><h2>Negative results</h2><ul>{negatives}</ul>
</body></html>"""


def _compatibility_markdown(
    evaluation: EvaluationBundle,
    raw: tuple[tuple[FlagshipDemoResult, ConversionSelection], ...],
) -> str:
    lines = [
        "# Continuum Compatibility Evaluation",
        "",
        "Matching shapes were held constant while attention state-producing weights changed. "
        "Direct reuse had to be rejected; token-history recomputation was allowed only with "
        "explicit dependency evidence.",
        "",
        "| Seed | Shapes match | Direct reuse | Rejection reasons | Recompute class | Components |",
        "|---:|---|---|---|---|---|",
    ]
    unsafe_acceptances = 0
    for seed_record, (flagship, _conversion) in zip(evaluation.per_seed, raw, strict=True):
        case = flagship.compatibility_case
        if case.direct_reuse.safe:
            unsafe_acceptances += 1
        reasons = ", ".join(reason.code for reason in case.direct_reuse.reasons)
        components = ", ".join(
            case.recomputation_assisted.required_recomputation[0].state_components
        )
        lines.append(
            f"| {seed_record.seed} | {str(case.shapes_match).lower()} | "
            f"{case.direct_reuse.compatibility_class.value} | {reasons} | "
            f"{case.recomputation_assisted.compatibility_class.value} | {components} |"
        )
    lines.extend(
        (
            "",
            f"Unsafe direct-reuse acceptances: **{unsafe_acceptances}**.",
            "",
            "The recomputation result is a compatibility plan with verification obligations; "
            "this campaign does not claim an executed changed-weight migration.",
        )
    )
    return "\n".join(lines) + "\n"


def _fault_markdown(
    evaluation: EvaluationBundle,
    raw: tuple[tuple[FlagshipDemoResult, ConversionSelection], ...],
) -> str:
    lines = [
        "# Continuum Fault-Tolerance Evaluation",
        "",
        "Each seed injects a destination crash at `DESTINATION_VALIDATING`, before ownership "
        "commit. The coordinator is closed and reopened before the second migration.",
        "",
        "| Seed | Fault label | Final phase | Source epoch | Duplicate accepted | Token gaps | Second migration |",
        "|---:|---|---|---:|---:|---:|---|",
    ]
    for seed_record, (flagship, _conversion) in zip(evaluation.per_seed, raw, strict=True):
        lines.append(
            f"| {seed_record.seed} | "
            f"{flagship.failed_migration.fault.definition.ground_truth_label} | "
            f"{seed_record.failed_transaction_final_phase} | "
            f"{flagship.failed_migration.source_epoch_after} | "
            f"{seed_record.gateway_duplicate_count} | {seed_record.gateway_gap_count} | "
            f"{seed_record.successful_transaction_final_phase} |"
        )
    lines.extend(
        (
            "",
            "External exactly-once delivery is not claimed. These results establish exactly-once "
            "acceptance at the SLOForge gateway for the bounded resumable protocol fixture.",
        )
    )
    return "\n".join(lines) + "\n"


def _runtime_markdown(evaluation: EvaluationBundle) -> str:
    lines = [
        "# Continuum Runtime Adapter Status",
        "",
        "| Runtime | Version | Status | Discovery exercised | Migration exercised | Limitation |",
        "|---|---|---|---|---|---|",
    ]
    lines.extend(
        f"| {item.runtime} | {item.version or 'unavailable'} | {item.adapter_status} | "
        f"{str(item.discovery_exercised).lower()} | {str(item.migration_exercised).lower()} | "
        f"{item.limitation} |"
        for item in evaluation.adapters
    )
    lines.extend(
        (
            "",
            "Only the two deterministic reference adapters performed active-state migration in "
            "this campaign. Public API discovery does not constitute migration validation.",
        )
    )
    return "\n".join(lines) + "\n"


def generate_reports(evaluation: EvaluationBundle, *, root: Path) -> ReportSet:
    """Validate every raw input before rendering the five required static reports."""

    raw = _validated_raw(evaluation, root)
    report_root = root / "reports"
    evaluation_markdown = report_root / "continuum-evaluation.md"
    evaluation_html = report_root / "continuum-evaluation.html"
    compatibility_markdown = report_root / "continuum-compatibility.md"
    fault_markdown = report_root / "continuum-fault-tolerance.md"
    runtime_markdown = report_root / "continuum-runtime-adapters.md"
    _write_text(evaluation_markdown, _evaluation_markdown(evaluation))
    _write_text(evaluation_html, _evaluation_html(evaluation))
    _write_text(compatibility_markdown, _compatibility_markdown(evaluation, raw))
    _write_text(fault_markdown, _fault_markdown(evaluation, raw))
    _write_text(runtime_markdown, _runtime_markdown(evaluation))
    return ReportSet(
        evaluation_markdown=_reference(evaluation_markdown, root=root, media_type="text/markdown"),
        evaluation_html=_reference(evaluation_html, root=root, media_type="text/html"),
        compatibility_markdown=_reference(
            compatibility_markdown, root=root, media_type="text/markdown"
        ),
        fault_tolerance_markdown=_reference(fault_markdown, root=root, media_type="text/markdown"),
        runtime_adapters_markdown=_reference(
            runtime_markdown, root=root, media_type="text/markdown"
        ),
    )
