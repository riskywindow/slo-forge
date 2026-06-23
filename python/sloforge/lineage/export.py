"""Portable deterministic JSON and GraphML lineage exports."""

from __future__ import annotations

import json
import os
import tempfile
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path

from .models import EvidenceTargetKind, LineageSnapshot
from .store import LineageStore

_GRAPHML = "http://graphml.graphdrawing.org/xmlns"
ET.register_namespace("", _GRAPHML)


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def export_json(
    store: LineageStore,
    path: Path,
    *,
    exported_at: datetime,
    maximum_records: int = 100_000,
) -> LineageSnapshot:
    snapshot = store.snapshot(exported_at=exported_at, maximum_records=maximum_records)
    payload = json.dumps(
        snapshot.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    _atomic_write(path, payload + b"\n")
    return snapshot


def _node(graph: ET.Element, identifier: str, kind: str, label: str) -> None:
    node = ET.SubElement(graph, f"{{{_GRAPHML}}}node", id=identifier)
    ET.SubElement(node, f"{{{_GRAPHML}}}data", key="kind").text = kind
    ET.SubElement(node, f"{{{_GRAPHML}}}data", key="label").text = label


def _edge(graph: ET.Element, identifier: str, source: str, target: str, kind: str) -> None:
    edge = ET.SubElement(graph, f"{{{_GRAPHML}}}edge", id=identifier, source=source, target=target)
    ET.SubElement(edge, f"{{{_GRAPHML}}}data", key="edge_kind").text = kind


def export_graphml(
    store: LineageStore,
    path: Path,
    *,
    exported_at: datetime,
    maximum_records: int = 100_000,
) -> LineageSnapshot:
    snapshot = store.snapshot(exported_at=exported_at, maximum_records=maximum_records)
    root = ET.Element(f"{{{_GRAPHML}}}graphml")
    for key_id, target, name in (
        ("kind", "node", "kind"),
        ("label", "node", "label"),
        ("edge_kind", "edge", "kind"),
    ):
        ET.SubElement(
            root,
            f"{{{_GRAPHML}}}key",
            {"id": key_id, "for": target, "attr.name": name, "attr.type": "string"},
        )
    graph = ET.SubElement(
        root, f"{{{_GRAPHML}}}graph", id="genesis-lineage", edgedefault="directed"
    )
    for task in snapshot.tasks:
        _node(graph, f"task:{task.task_id}", "task", task.model_family)
    for candidate in snapshot.candidates:
        _node(
            graph, f"candidate:{candidate.candidate_id}", "candidate", candidate.disposition.value
        )
    for transformation in snapshot.transformations:
        _node(
            graph,
            f"transformation:{transformation.transformation_id}",
            "transformation",
            transformation.family,
        )
    for evidence in snapshot.evidence:
        _node(graph, f"evidence:{evidence.evidence_id}", "evidence", evidence.evidence_type)
    for counterexample in snapshot.counterexamples:
        _node(
            graph,
            f"counterexample:{counterexample.counterexample_id}",
            "counterexample",
            counterexample.scope.value,
        )
    for constraint in snapshot.constraints:
        _node(graph, f"constraint:{constraint.constraint_id}", "constraint", constraint.rationale)
    for transfer in snapshot.transfers:
        _node(graph, f"transfer:{transfer.transfer_id}", "transfer", transfer.outcome.value)
    for invalidation in snapshot.invalidations:
        _node(
            graph,
            f"invalidation:{invalidation.invalidation_id}",
            "invalidation",
            invalidation.selector.name,
        )

    edge_index = 0

    def edge(source: str, target: str, kind: str) -> None:
        nonlocal edge_index
        _edge(graph, f"edge:{edge_index:08d}", source, target, kind)
        edge_index += 1

    for candidate in snapshot.candidates:
        edge(f"task:{candidate.task_id}", f"candidate:{candidate.candidate_id}", "contains")
        for parent in candidate.parent_candidate_ids:
            edge(f"candidate:{parent}", f"candidate:{candidate.candidate_id}", "parent")
    for transformation in snapshot.transformations:
        edge(
            f"candidate:{transformation.source_candidate_id}",
            f"transformation:{transformation.transformation_id}",
            "proposed",
        )
        for parent in transformation.parent_transformation_ids:
            edge(
                f"transformation:{parent}",
                f"transformation:{transformation.transformation_id}",
                "parent",
            )
        if transformation.target_candidate_id is not None:
            edge(
                f"transformation:{transformation.transformation_id}",
                f"candidate:{transformation.target_candidate_id}",
                "produced",
            )
    for evidence in snapshot.evidence:
        target_prefix = (
            "candidate"
            if evidence.target_kind is EvidenceTargetKind.CANDIDATE
            else "transformation"
        )
        edge(
            f"evidence:{evidence.evidence_id}",
            f"{target_prefix}:{evidence.target_id}",
            "supports",
        )
        for invalidation_id in evidence.invalidation_event_ids:
            edge(
                f"invalidation:{invalidation_id}",
                f"evidence:{evidence.evidence_id}",
                "invalidates",
            )
    for counterexample in snapshot.counterexamples:
        edge(
            f"counterexample:{counterexample.counterexample_id}",
            f"candidate:{counterexample.candidate_id}",
            "rejects",
        )
        if counterexample.transformation_id is not None:
            edge(
                f"counterexample:{counterexample.counterexample_id}",
                f"transformation:{counterexample.transformation_id}",
                "constrains",
            )
    for constraint in snapshot.constraints:
        edge(
            f"counterexample:{constraint.counterexample_id}",
            f"constraint:{constraint.constraint_id}",
            "generalizes",
        )
        if constraint.transformation_id is not None:
            edge(
                f"constraint:{constraint.constraint_id}",
                f"transformation:{constraint.transformation_id}",
                "restricts",
            )
    for transfer in snapshot.transfers:
        edge(
            f"transformation:{transfer.transformation_id}",
            f"transfer:{transfer.transfer_id}",
            "transferred",
        )
        edge(
            f"transfer:{transfer.transfer_id}",
            f"task:{transfer.target_task_id}",
            "targets",
        )
        for evidence_id in transfer.source_evidence_ids:
            edge(
                f"evidence:{evidence_id}",
                f"transfer:{transfer.transfer_id}",
                "justifies",
            )
    ET.indent(root, space="  ")
    _atomic_write(path, ET.tostring(root, encoding="utf-8", xml_declaration=True) + b"\n")
    return snapshot
