"""Artifact I/O for WarmPath plans and graphs."""

from __future__ import annotations

from pathlib import Path

from sloforge.util import write_json
from sloforge.warmpath.models import ArtifactGraph, ExecutionRecord, StartupProfile, WarmPathPlan


def save_graph(graph: ArtifactGraph, path: Path) -> None:
    write_json(path, graph.model_dump(mode="json"))


def load_graph(path: Path) -> ArtifactGraph:
    return ArtifactGraph.model_validate_json(path.read_text(encoding="utf-8"))


def save_plan(plan: WarmPathPlan, path: Path) -> None:
    write_json(path, plan.model_dump(mode="json"))


def load_plan(path: Path) -> WarmPathPlan:
    return WarmPathPlan.model_validate_json(path.read_text(encoding="utf-8"))


def load_execution(path: Path) -> ExecutionRecord:
    return ExecutionRecord.model_validate_json(path.read_text(encoding="utf-8"))


def save_profile(profile: StartupProfile, path: Path) -> None:
    write_json(path, profile.model_dump(mode="json"))
