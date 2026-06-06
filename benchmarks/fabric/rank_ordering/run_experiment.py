#!/usr/bin/env python3
"""Run the reproducible topology-aware collective rank-ordering experiment."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from sloforge.fabric.performance import (
    RankOrderingExperimentInput,
    execute_rank_ordering_experiment,
    write_experiment_artifacts,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(__file__).with_name("synthetic-input.json"),
        help="strict input bundle containing topology, physical plan, trace evidence, and config",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("results"),
        help="directory for the canonical result and complete raw samples",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "reports" / "rank-ordering-experiment.md",
        help="artifact-derived Markdown report path",
    )
    return parser.parse_args()


def _resolve_artifact(uri: str) -> Path:
    candidate = Path(uri)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def main() -> None:
    args = _arguments()
    bundle = RankOrderingExperimentInput.model_validate_json(args.input.read_text(encoding="utf-8"))
    evidence_path = _resolve_artifact(bundle.trace_evidence.artifact_uri)
    evidence_digest = hashlib.sha256(evidence_path.read_bytes()).hexdigest()
    if evidence_digest != bundle.trace_evidence.artifact_sha256:
        raise SystemExit(
            f"evidence digest mismatch for {evidence_path}: expected "
            f"{bundle.trace_evidence.artifact_sha256}, observed {evidence_digest}"
        )
    experiment = execute_rank_ordering_experiment(
        bundle.physical_plan,
        bundle.topology,
        bundle.trace_evidence,
        bundle.config,
        evidence_root=PROJECT_ROOT,
    )
    paths = write_experiment_artifacts(args.output, experiment, report_path=args.report)
    print(f"decision={experiment.decision.status.value}")
    print(f"enabled_by_default={str(experiment.decision.enabled_by_default).lower()}")
    print(f"artifact_hash={experiment.artifact_hash}")
    print(f"result={paths.result_json}")
    print(f"raw_samples={paths.raw_samples_jsonl}")
    print(f"report={paths.report_markdown}")


if __name__ == "__main__":
    main()
