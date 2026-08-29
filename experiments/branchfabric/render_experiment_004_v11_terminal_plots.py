#!/usr/bin/env python3
"""Render the terminal Experiment 004 v11 waterfall and Sankey plots."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.sankey import Sankey

SEED = 41
ROOT = Path(__file__).resolve().parents[2]
PLOTS = ROOT / "artifacts/branchfabric/gpu-validation/experiment-004/v11/plots"

V10_HBM_RECLAMATION_SECONDS = 5.222174431
V10_RESTORE_FIRST_TOKEN_SECONDS = 7.727983415
V10_CONTINUATION_COMPLETE_SECONDS = 7.848228746
V11_EXPORT_SECONDS = 2.613090870
V11_RESTORE_SECONDS = 4.438193614
V11_CONTINUATION_FIRST_SECONDS = 0.083229142
V11_CONTINUATION_COMPLETE_SECONDS = 0.205774608

V10_TIERS_GIB = {
    "GPU temporary": 30_651_973_632 / 2**30,
    "Host temporary": 24_310_185_984 / 2**30,
    "Native endpoints": 3_170_893_824 / 2**30,
    "Links": 3_170_893_824 / 2**30,
}
V11_TIERS_GIB = {
    "GPU temporary": 7_399_019_520 / 2**30,
    "Host temporary": 9_512_948_736 / 2**30,
    "Native endpoints": 3_170_893_824 / 2**30,
    "Links": 2_114_200_960 / 2**30,
}


def _save(fig: plt.Figure, name: str) -> None:
    fig.savefig(PLOTS / name, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def render_export_reclamation_waterfall() -> None:
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    rows = [
        ("v10 integrated", V10_HBM_RECLAMATION_SECONDS, "Trigger → HBM reclaimed"),
        ("v11 micro", V11_EXPORT_SECONDS, "Optimized export complete"),
    ]
    colors = ["#7a5195", "#2f4b7c"]
    for y, ((_label, duration, annotation), color) in enumerate(zip(rows, colors, strict=True)):
        ax.broken_barh([(0.0, duration)], (y - 0.28, 0.56), facecolors=color)
        ax.plot([duration, duration], [y - 0.35, y + 0.35], color="#111111", linewidth=1.2)
        ax.text(duration + 0.08, y, f"{duration:.3f} s  {annotation}", va="center", fontsize=9)
    ax.text(
        0.02,
        2.0,
        "v11 integrated: NOT REACHED (failed before reclaim trigger)",
        va="center",
        fontsize=10,
        color="#a51c30",
        fontweight="bold",
    )
    ax.set_yticks([0, 1, 2], ["v10 integrated", "v11 micro", "v11 integrated"])
    ax.set_xlabel("Elapsed seconds from each scope's start")
    ax.set_title("v10 reclamation vs v11 export waterfall — measured scopes")
    ax.set_xlim(0, 6.4)
    ax.set_ylim(-0.7, 2.55)
    ax.grid(axis="x", alpha=0.22)
    ax.text(
        0,
        -0.62,
        "Scope warning: v10 is trigger→HBM on SXM4; v11 is micro export on PCIe. Durations are not a causal latency comparison.",
        fontsize=8,
        color="#555555",
    )
    fig.tight_layout()
    _save(fig, "05-export-reclamation-waterfall.png")


def render_restore_waterfall() -> None:
    fig, ax = plt.subplots(figsize=(11.2, 5.0))
    v10_first = V10_RESTORE_FIRST_TOKEN_SECONDS
    v10_tail = V10_CONTINUATION_COMPLETE_SECONDS - V10_RESTORE_FIRST_TOKEN_SECONDS
    ax.broken_barh([(0.0, v10_first)], (-0.28, 0.56), facecolors="#7a5195")
    ax.broken_barh([(v10_first, v10_tail)], (-0.28, 0.56), facecolors="#bc5090")
    ax.text(
        v10_first / 2,
        0,
        "trigger → first token",
        ha="center",
        va="center",
        color="white",
        fontsize=8,
    )
    ax.text(
        V10_CONTINUATION_COMPLETE_SECONDS + 0.08,
        0,
        f"{V10_CONTINUATION_COMPLETE_SECONDS:.3f} s",
        va="center",
    )

    v11_first_end = V11_RESTORE_SECONDS + V11_CONTINUATION_FIRST_SECONDS
    v11_tail = V11_CONTINUATION_COMPLETE_SECONDS - V11_CONTINUATION_FIRST_SECONDS
    ax.broken_barh([(0.0, V11_RESTORE_SECONDS)], (0.72, 0.56), facecolors="#2f4b7c")
    ax.broken_barh(
        [(V11_RESTORE_SECONDS, V11_CONTINUATION_FIRST_SECONDS)],
        (0.72, 0.56),
        facecolors="#00a6ca",
    )
    ax.broken_barh([(v11_first_end, v11_tail)], (0.72, 0.56), facecolors="#00ccbc")
    ax.text(
        V11_RESTORE_SECONDS / 2,
        1,
        "optimized restore",
        ha="center",
        va="center",
        color="white",
        fontsize=8,
    )
    ax.text(
        V11_RESTORE_SECONDS + V11_CONTINUATION_COMPLETE_SECONDS + 0.08,
        1,
        f"{V11_RESTORE_SECONDS + V11_CONTINUATION_COMPLETE_SECONDS:.3f} s",
        va="center",
    )

    ax.text(
        0.02,
        2.0,
        "v11 integrated: NOT REACHED (restore trigger absent)",
        va="center",
        fontsize=10,
        color="#a51c30",
        fontweight="bold",
    )
    ax.set_yticks([0, 1, 2], ["v10 integrated", "v11 micro", "v11 integrated"])
    ax.set_xlabel("Elapsed seconds from restore trigger (v10) or micro restore start (v11)")
    ax.set_title("v10 vs v11 restore waterfall — measured scopes")
    ax.set_xlim(0, 8.9)
    ax.set_ylim(-0.7, 2.55)
    ax.grid(axis="x", alpha=0.22)
    ax.legend(
        handles=[
            Patch(color="#2f4b7c", label="restore/state reconstruction"),
            Patch(color="#00a6ca", label="to first resumed token"),
            Patch(color="#00ccbc", label="continuation tail"),
        ],
        loc="upper right",
        frameon=False,
        fontsize=8,
    )
    ax.text(
        0,
        -0.62,
        "Scope warning: v10 is integrated SXM4; v11 is PCIe micro. The integrated v11 comparison cell is null, not zero.",
        fontsize=8,
        color="#555555",
    )
    fig.tight_layout()
    _save(fig, "06-restore-waterfall.png")


def _draw_sankey(ax: plt.Axes, title: str, tiers: dict[str, float], color: str) -> None:
    total = sum(tiers.values())
    labels = ["Physical work", *list(tiers)]
    flows = [total] + [-value for value in tiers.values()]
    sankey = Sankey(
        ax=ax,
        scale=1 / total,
        offset=0.22,
        head_angle=120,
        margin=0.75,
        format="%.2f GiB",
    )
    sankey.add(
        flows=flows,
        labels=labels,
        orientations=[0, 1, -1, 1, -1],
        trunklength=1.15,
        pathlengths=[0.35, 0.55, 0.55, 1.8, 1.8],
        facecolor=color,
        alpha=0.78,
    )
    sankey.finish()
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.axis("off")


def render_movement_sankey() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14.0, 6.2))
    _draw_sankey(axes[0], "Frozen v10 — 58x", V10_TIERS_GIB, "#7a5195")
    _draw_sankey(axes[1], "Measured v11 micro — 21.0007628x", V11_TIERS_GIB, "#2f4b7c")
    fig.suptitle(
        "Before/after state-movement Sankey by memory tier", fontsize=14, fontweight="bold"
    )
    fig.text(
        0.5,
        0.02,
        "Flow widths encode measured physical bytes. Each panel conserves its full physical-work total; links include D2H, H2D, and control bytes.",
        ha="center",
        fontsize=9,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.94))
    _save(fig, "13-before-after-movement-sankey.png")


def write_manifest() -> None:
    plots = []
    for path in sorted(PLOTS.glob("[0-9][0-9]-*.png")):
        payload = path.read_bytes()
        plots.append(
            {
                "artifact": str(path.relative_to(ROOT)),
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    data_path = PLOTS / "plot-data.json"
    data_payload = data_path.read_bytes()
    manifest = {
        "schema_version": "sloforge.branchfabric.experiment-004-v11-plot-manifest/v1",
        "status": "PASS",
        "seed": SEED,
        "plot_count": len(plots),
        "plots": plots,
        "data_artifact": {
            "artifact": str(data_path.relative_to(ROOT)),
            "bytes": len(data_payload),
            "sha256": hashlib.sha256(data_payload).hexdigest(),
        },
    }
    (PLOTS / "plot-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> None:
    PLOTS.mkdir(parents=True, exist_ok=True)
    render_export_reclamation_waterfall()
    render_restore_waterfall()
    render_movement_sankey()
    write_manifest()


if __name__ == "__main__":
    main()
