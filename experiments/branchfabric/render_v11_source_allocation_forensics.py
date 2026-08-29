#!/usr/bin/env python3
"""Render evidence-bounded source-allocation forensic plots for v11 attempt C."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EVIDENCE = ROOT / "artifacts/branchfabric/gpu-validation/experiment-004/v11-final"


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"forensic input {path} is not an object")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _save(fig: plt.Figure, output: Path, name: str) -> dict[str, Any]:
    path = output / name
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return {
        "path": str(path.relative_to(ROOT)),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _phase_rows(lifecycle: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = lifecycle.get("phase_timeline")
    if not isinstance(rows, list):
        raise ValueError("lifecycle artifact lacks phase_timeline")
    result = {}
    for item in rows:
        if not isinstance(item, dict) or not isinstance(item.get("phase"), str):
            raise ValueError("lifecycle artifact has a malformed phase row")
        result[str(item["phase"])] = item
    return result


def render_lifecycle_timeline(
    lifecycle: dict[str, Any], divergence: dict[str, Any], output: Path
) -> dict[str, Any]:
    phases = _phase_rows(lifecycle)
    first = divergence["first_causal_event"]["timestamp"]
    lower_ns = int(first["lower_bound_inclusive_ns"])
    upper_ns = int(first["upper_bound_inclusive_ns"])
    origin_ns = lower_ns
    observed = [
        ("readiness ALLOC window", lower_ns, upper_ns, "#d95f02"),
        (
            "ROLLOUT_READY",
            int(phases["ROLLOUT_READY"]["observed_at_monotonic_ns"]),
            None,
            "#1b9e77",
        ),
        (
            "CONTROL_BEGIN",
            int(phases["CONTROL_BEGIN"]["observed_at_monotonic_ns"]),
            None,
            "#7570b3",
        ),
        (
            "OVERLOAD_DETECTED",
            int(phases["OVERLOAD_DETECTED"]["observed_at_monotonic_ns"]),
            None,
            "#e6ab02",
        ),
        (
            "TRIGGER_PREDICATE_SATISFIED",
            int(phases["TRIGGER_PREDICATE_SATISFIED"]["observed_at_monotonic_ns"]),
            None,
            "#66a61e",
        ),
        (
            "RECLAIM_TRIGGER",
            int(phases["RECLAIM_TRIGGER"]["observed_at_monotonic_ns"]),
            None,
            "#1f78b4",
        ),
        (
            "QUIESCE_BEGIN proxy",
            int(phases["QUIESCE_BEGIN"]["observed_at_monotonic_ns"]),
            None,
            "#a6761d",
        ),
        ("FAILURE", int(phases["FAILURE"]["observed_at_monotonic_ns"]), None, "#e31a1c"),
    ]
    fig, ax = plt.subplots(figsize=(12.5, 6.2))
    for index, (label, start_ns, end_ns, color) in enumerate(observed):
        y = len(observed) - index
        start = (start_ns - origin_ns) / 1e9
        if end_ns is None:
            ax.scatter(start, y, s=55, color=color, zorder=3)
            label_x = start + 0.28
        else:
            width = (end_ns - start_ns) / 1e9
            ax.broken_barh([(start, width)], (y - 0.22, 0.44), facecolors=color)
            label_x = start + width + 0.35
        ax.text(label_x, y, label, va="center", fontsize=8.5)
    ax.axvspan(
        (lower_ns - origin_ns) / 1e9,
        (upper_ns - origin_ns) / 1e9,
        color="#d95f02",
        alpha=0.12,
    )
    ax.annotate(
        "allocation-event history is contaminated here",
        xy=(((lower_ns + upper_ns) / 2 - origin_ns) / 1e9, len(observed) + 0.25),
        xytext=(3.5, len(observed) + 0.65),
        arrowprops={"arrowstyle": "->", "color": "#555555"},
        fontsize=9,
    )
    missing = [
        item["phase"]
        for item in lifecycle["phase_timeline"]
        if item["evidence_status"] == "NOT_PERSISTED"
    ]
    ax.text(
        0.0,
        0.2,
        "Not plotted as time values (not persisted): " + ", ".join(missing),
        fontsize=8,
        color="#555555",
        wrap=True,
    )
    topology = lifecycle["topology"]
    ax.text(
        0.0,
        -0.35,
        (
            f"Logical topology retained: {topology['physical_source_page_count']:,} pages "
            f"({topology['shared_page_count']:,} shared + {topology['private_page_count']:,} private), "
            f"{topology['logical_state_bytes']:,} bytes. Per-page physical lifetimes were not serialized."
        ),
        fontsize=8,
        color="#555555",
        wrap=True,
    )
    ax.set_title("Source allocation lifecycle timeline — retained attempt-C evidence")
    ax.set_xlabel("Seconds since readiness-gpu1-00000 batch start")
    ax.set_yticks([])
    ax.set_xlim(-0.5, 42.0)
    ax.set_ylim(-0.7, len(observed) + 1.25)
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    return _save(fig, output, "01-source-allocation-lifecycle-timeline.png")


def render_first_divergence(divergence: dict[str, Any], output: Path) -> dict[str, Any]:
    first = divergence["first_causal_event"]
    timestamp = first["timestamp"]
    nodes = [
        (
            "Readiness request",
            "readiness-gpu1-00000\nfirst batch, size 1",
            "#4c78a8",
        ),
        (
            "First causal allocator event",
            (
                "ALLOC via\nKVCacheManager.allocate_slots\n"
                f"bounded [{timestamp['lower_bound_inclusive_ns']}, "
                f"{timestamp['upper_bound_inclusive_ns']}] ns"
            ),
            "#f58518",
        ),
        (
            "Reset omission",
            "prefix/request state reset\ndid not drain take_new_block_ids()",
            "#eeca3b",
        ),
        (
            "Correct current proof",
            "1,152 live blocks + epochs\nowners/refcounts/mapping/HBM passed",
            "#54a24b",
        ),
        (
            "Faulty equality",
            "historical notification queue\n== current capture block-ID set",
            "#e45756",
        ),
    ]
    fig, ax = plt.subplots(figsize=(13.0, 4.8))
    ax.set_xlim(0, len(nodes) * 2.4)
    ax.set_ylim(0, 3.2)
    for index, (title, detail, color) in enumerate(nodes):
        x = 0.25 + index * 2.4
        box = FancyBboxPatch(
            (x, 1.0),
            2.0,
            1.35,
            boxstyle="round,pad=0.04",
            facecolor=color,
            edgecolor="#333333",
            linewidth=0.8,
            alpha=0.9,
        )
        ax.add_patch(box)
        ax.text(x + 1.0, 2.02, title, ha="center", va="center", fontsize=9, weight="bold")
        ax.text(x + 1.0, 1.5, detail, ha="center", va="center", fontsize=7.2, wrap=True)
        if index < len(nodes) - 1:
            ax.add_patch(
                FancyArrowPatch(
                    (x + 2.02, 1.68),
                    (x + 2.36, 1.68),
                    arrowstyle="-|>",
                    mutation_scale=12,
                    color="#444444",
                )
            )
    ax.text(
        0.25,
        0.42,
        "First divergence predates ROLLOUT_CREATED. Exact block IDs and allocator timestamp are unrecoverable because failure context was not serialized.",
        fontsize=9,
        color="#555555",
    )
    ax.text(
        9.85,
        2.7,
        "CLASS A: GATE_BUG",
        fontsize=11,
        weight="bold",
        color="#b22222",
    )
    ax.set_title("First allocation divergence — causal event, not final equality only")
    ax.axis("off")
    fig.tight_layout()
    return _save(fig, output, "02-first-allocation-divergence.png")


def render_identity_evidence_matrix(lifecycle: dict[str, Any], output: Path) -> dict[str, Any]:
    phases = lifecycle["requested_phases"]
    fields = [
        "phase timestamp",
        "logical page topology",
        "physical block ID",
        "allocation epoch",
        "owner set / refcount",
        "block-table mapping",
        "allocator event history",
        "state commitment",
    ]
    matrix = np.zeros((len(fields), len(phases)), dtype=int)
    rows = _phase_rows(lifecycle)
    for column, phase in enumerate(phases):
        status = rows[phase]["evidence_status"]
        if status in {"DIRECT", "BOUNDARY_PROXY", "ORDERING_PROVEN_TIMESTAMP_NOT_PERSISTED"}:
            matrix[0, column] = 1 if status != "DIRECT" else 2
    rollout_ready = phases.index("ROLLOUT_READY")
    failure = phases.index("FAILURE")
    matrix[1, rollout_ready : failure + 1] = 1
    pre_export = phases.index("PRE_EXPORT_IDENTITY_GATE")
    matrix[2:6, pre_export] = 1
    matrix[7, pre_export] = 1
    fig, ax = plt.subplots(figsize=(14.0, 7.0))
    cmap = ListedColormap(["#eeeeee", "#fdb863", "#5ab4ac"])
    image = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=0, vmax=2)
    del image
    ax.set_xticks(range(len(phases)), phases, rotation=55, ha="right", fontsize=8)
    ax.set_yticks(range(len(fields)), fields, fontsize=9)
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            label = {0: "missing", 1: "aggregate", 2: "direct"}[int(matrix[row, column])]
            if matrix[row, column] != 0:
                ax.text(column, row, label, ha="center", va="center", fontsize=6.5)
    ax.set_title("Allocation identity evidence by lifecycle phase — missing evidence is not zero")
    ax.set_xlabel("Lifecycle phase")
    ax.set_ylabel("Identity/evidence field")
    ax.legend(
        handles=[
            Patch(facecolor="#eeeeee", label="not persisted"),
            Patch(facecolor="#fdb863", label="aggregate/control-flow proof only"),
            Patch(facecolor="#5ab4ac", label="direct retained timestamp/evidence"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.63),
        ncol=3,
        frameon=False,
    )
    fig.subplots_adjust(left=0.2, right=0.99, top=0.92, bottom=0.48)
    return _save(fig, output, "03-allocation-identity-fields-by-phase.png")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--evidence-root", type=Path, default=DEFAULT_EVIDENCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_EVIDENCE / "plots")
    args = parser.parse_args()
    np.random.seed(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    lifecycle_path = args.evidence_root / "allocation-lifecycle/source-allocation-lifecycles.json"
    divergence_path = args.evidence_root / "allocation-forensics/first-allocation-divergence.json"
    lifecycle = _load(lifecycle_path)
    divergence = _load(divergence_path)
    plots = [
        render_lifecycle_timeline(lifecycle, divergence, args.output),
        render_first_divergence(divergence, args.output),
        render_identity_evidence_matrix(lifecycle, args.output),
    ]
    plot_data = {
        "schema_version": "sloforge.branchfabric.v11-allocation-forensics-plot-data/v1",
        "seed": args.seed,
        "attempt_id": lifecycle["attempt_id"],
        "classification": divergence["root_cause_class"],
        "topology": lifecycle["topology"],
        "phase_timeline": lifecycle["phase_timeline"],
        "first_causal_event": divergence["first_causal_event"],
        "anti_fabrication_notice": lifecycle["anti_fabrication_notice"],
        "provenance": [
            {"path": str(lifecycle_path.relative_to(ROOT)), "sha256": _sha256(lifecycle_path)},
            {
                "path": str(divergence_path.relative_to(ROOT)),
                "sha256": _sha256(divergence_path),
            },
        ],
    }
    plot_data_path = args.output / "forensic-plot-data.json"
    plot_data_path.write_text(json.dumps(plot_data, indent=2, sort_keys=True) + "\n")
    manifest = {
        "schema_version": "sloforge.branchfabric.v11-allocation-forensics-plot-manifest/v1",
        "status": "PASS",
        "seed": args.seed,
        "plot_count": len(plots),
        "plots": plots,
        "plot_data": {
            "path": str(plot_data_path.relative_to(ROOT)),
            "bytes": plot_data_path.stat().st_size,
            "sha256": _sha256(plot_data_path),
        },
        "integrated_only_plots_generated": False,
    }
    (args.output / "forensic-plot-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
