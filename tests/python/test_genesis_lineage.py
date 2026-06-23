from __future__ import annotations

import hashlib
import json
import xml.etree.ElementTree as ET
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from pydantic import ValidationError

from sloforge.lineage import (
    CandidateDisposition,
    CandidateRecord,
    ConstraintPredicate,
    CounterexampleRecord,
    CounterexampleScope,
    DependencyKind,
    DependencySelector,
    DependencyVersion,
    EvidenceFreshness,
    EvidenceRecord,
    EvidenceResult,
    EvidenceTargetKind,
    InvalidationEvent,
    LearnedConstraintRecord,
    LineageConflict,
    LineageLimitExceeded,
    LineageStore,
    SemanticCategory,
    TaskFeatures,
    TransferOutcome,
    TransferRecord,
    TransformationOutcome,
    TransformationQuery,
    TransformationRecord,
    effective_confidence,
    export_graphml,
    export_json,
    version_matches,
)

NOW = datetime(2026, 1, 1, tzinfo=UTC)


def _hash(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _dependency(version: str = "2.2.0") -> DependencyVersion:
    return DependencyVersion(
        kind=DependencyKind.COMPILER,
        name="triton",
        version=version,
        content_hash=_hash(f"triton-{version}"),
    )


def _task(task_id: str = "source-task") -> TaskFeatures:
    return TaskFeatures(
        task_id=task_id,
        model_family="sparse-moe",
        operator_families=("expert-dispatch", "gated-mlp"),
        workload_regimes=("bimodal-prompts",),
        hardware_architecture="sm90",
        topology_features=("nvlink",),
        dependencies=(_dependency(),),
        model_contract_hash=_hash(f"model-{task_id}"),
        workload_contract_hash=_hash(f"workload-{task_id}"),
    )


def _candidate(candidate_id: str = "candidate-a", task_id: str = "source-task") -> CandidateRecord:
    return CandidateRecord(
        candidate_id=candidate_id,
        task_id=task_id,
        genome_hash=_hash(candidate_id),
        disposition=CandidateDisposition.ACCEPTED,
        causal_bottleneck="expert dispatch metadata",
        exposed_next_bottleneck="collective latency",
        created_at=NOW,
    )


def _transformation(
    transformation_id: str = "transform-a", source_candidate_id: str = "candidate-a"
) -> TransformationRecord:
    return TransformationRecord(
        transformation_id=transformation_id,
        family="expert-dispatch-packing",
        semantic_category=SemanticCategory.EXACT,
        source_candidate_id=source_candidate_id,
        affected_regions=("tensor.expert_dispatch", "kernel.metadata_pack"),
        preconditions=("expert count is positive",),
        applicable_model_families=("sparse-moe",),
        applicable_operations=("expert-dispatch",),
        applicable_hardware=("sm90",),
        applicable_workloads=("bimodal-prompts",),
        dependency_preconditions=(
            DependencySelector(
                kind=DependencyKind.COMPILER, name="triton", version_range=">=2.0,<4.0"
            ),
        ),
        expected_benefit=0.12,
        outcome=TransformationOutcome.ACCEPTED,
        proposal_source="autopsy-guided-local-search",
        created_at=NOW,
    )


def _evidence(evidence_id: str, version: str = "2.2.0") -> EvidenceRecord:
    return EvidenceRecord(
        evidence_id=evidence_id,
        target_kind=EvidenceTargetKind.TRANSFORMATION,
        target_id="transform-a",
        evidence_type="end-to-end-benchmark",
        result=EvidenceResult.PASS,
        content_hash=_hash(evidence_id),
        model_family="sparse-moe",
        workload_regimes=("bimodal-prompts",),
        hardware_architecture="sm90",
        dependencies=(_dependency(version),),
        base_confidence=0.9,
        observed_at=NOW,
        valid_until=NOW + timedelta(days=365),
    )


def _populated_store(path: Path) -> LineageStore:
    store = LineageStore(path)
    store.record_task(_task())
    store.record_candidate(_candidate())
    store.record_transformation(_transformation())
    store.record_evidence(_evidence("evidence-old", "2.2.0"))
    store.record_evidence(_evidence("evidence-new", "3.1.0"))
    return store


def test_lineage_models_are_strict_and_version_ranges_are_real() -> None:
    document = _task().model_dump(mode="json")
    document["unknown"] = True
    with pytest.raises(ValidationError):
        TaskFeatures.model_validate(document, strict=True)
    assert version_matches("2.9.4", "<3.x")
    assert not version_matches("3.0.0", "<3.x")
    assert version_matches("3.1.0", ">=3.0,<4.0")
    assert version_matches("3.7.2", "3.x")


def test_sqlite_records_survive_reopen_and_bundle_rolls_back(tmp_path: Path) -> None:
    database = tmp_path / "lineage.sqlite3"
    with _populated_store(database) as store:
        bundle_candidate = _candidate("candidate-bundle")
        duplicate = _transformation("duplicate-transform", "candidate-bundle")
        with pytest.raises(LineageConflict):
            store.record_candidate_bundle(bundle_candidate, (duplicate, duplicate), ())
        assert {item.candidate_id for item in store.list_candidates()} == {"candidate-a"}
        with pytest.raises(ValueError, match="query limit"):
            store.list_candidates(limit=0)

    with LineageStore(database) as reopened:
        assert reopened.list_tasks() == (_task(),)
        assert reopened.list_transformations() == (_transformation(),)
        assert len(reopened.list_evidence()) == 2
        assert reopened.query_transformations(
            TransformationQuery(
                model_family="sparse-moe",
                operation="expert-dispatch",
                hardware_architecture="sm90",
                outcome=TransformationOutcome.ACCEPTED,
            )
        ) == (_transformation(),)


def test_dependency_invalidation_marks_matching_evidence_stale(tmp_path: Path) -> None:
    with _populated_store(tmp_path / "lineage.sqlite3") as store:
        event = InvalidationEvent(
            invalidation_id="invalidate-triton-v2",
            selector=DependencySelector(
                kind=DependencyKind.COMPILER, name="triton", version_range="<3.x"
            ),
            reason="Triton v2 code generation is no longer trusted",
            occurred_at=NOW + timedelta(days=30),
        )
        assert store.invalidate_dependency(event) == 1
        by_id = {item.evidence_id: item for item in store.list_evidence()}
        assert by_id["evidence-old"].freshness is EvidenceFreshness.STALE
        assert by_id["evidence-old"].invalidation_event_ids == (event.invalidation_id,)
        assert by_id["evidence-new"].freshness is EvidenceFreshness.FRESH
        assert effective_confidence(by_id["evidence-old"], as_of=NOW + timedelta(days=31)) == 0.0
        assert effective_confidence(
            by_id["evidence-new"],
            as_of=NOW + timedelta(days=90),
            half_life_days=90,
        ) == pytest.approx(0.45)


def test_oversized_invalidation_rolls_back_without_partial_staleness(tmp_path: Path) -> None:
    with _populated_store(tmp_path / "lineage.sqlite3") as store:
        event = InvalidationEvent(
            invalidation_id="bounded-invalidation",
            selector=DependencySelector(
                kind=DependencyKind.COMPILER, name="triton", version_range="*"
            ),
            reason="bounded invalidation fixture",
            occurred_at=NOW + timedelta(days=1),
        )
        with pytest.raises(LineageLimitExceeded):
            store.invalidate_dependency(event, maximum_evidence=1)
        assert store.list_invalidations() == ()
        assert all(item.freshness is EvidenceFreshness.FRESH for item in store.list_evidence())


def test_counterexamples_constraints_transfers_and_exports_are_portable(tmp_path: Path) -> None:
    with _populated_store(tmp_path / "lineage.sqlite3") as store:
        counterexample = CounterexampleRecord(
            counterexample_id="counterexample-a",
            candidate_id="candidate-a",
            transformation_id="transform-a",
            transformation_family="expert-dispatch-packing",
            scope=CounterexampleScope.DEPENDENCY,
            violated_contract="non-contiguous dispatch input is corrupted",
            minimized_input_hash=_hash("minimized-input"),
            reproduction_command=("python", "reproduce.py", "--seed", "17"),
            learned_precondition="requires contiguous metadata",
            dependencies=(_dependency(),),
            created_at=NOW,
        )
        store.record_counterexample(counterexample)
        constraint = LearnedConstraintRecord(
            constraint_id="constraint-a",
            counterexample_id=counterexample.counterexample_id,
            transformation_family=counterexample.transformation_family,
            transformation_id=counterexample.transformation_id,
            predicate=ConstraintPredicate(
                dependency_selectors=(
                    DependencySelector(
                        kind=DependencyKind.COMPILER,
                        name="triton",
                        version_range="2.x",
                    ),
                )
            ),
            rationale="avoid the minimized dependency-specific failure",
            created_at=NOW,
        )
        store.record_constraint(constraint)
        transfer = TransferRecord(
            transfer_id="transfer-a",
            target_task_id="source-task",
            transformation_id="transform-a",
            source_evidence_ids=("evidence-old",),
            retrieval_score=0.8,
            rank=1,
            seed=73129,
            outcome=TransferOutcome.REVERIFIED,
            rationale="reverification passed on the target task",
            created_at=NOW,
        )
        store.record_transfer(transfer)
        json_path = tmp_path / "lineage.json"
        graph_path = tmp_path / "lineage.graphml"
        snapshot = export_json(store, json_path, exported_at=NOW)
        export_graphml(store, graph_path, exported_at=NOW)

    decoded = json.loads(json_path.read_text(encoding="utf-8"))
    assert decoded["schema_version"] == "1.0.0"
    assert snapshot.counterexamples == (counterexample,)
    graph = ET.parse(graph_path).getroot()
    namespace = {"g": "http://graphml.graphdrawing.org/xmlns"}
    node_ids = {node.attrib["id"] for node in graph.findall(".//g:node", namespace)}
    assert "counterexample:counterexample-a" in node_ids
    assert "constraint:constraint-a" in node_ids
    assert "transfer:transfer-a" in node_ids
    assert graph.findall(".//g:edge", namespace)
