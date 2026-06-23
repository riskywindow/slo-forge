from __future__ import annotations

import hashlib
from datetime import UTC, datetime, timedelta
from pathlib import Path

from sloforge.lineage import (
    CandidateDisposition,
    CandidateRecord,
    ConstraintPredicate,
    CounterexampleRecord,
    CounterexampleScope,
    DependencyKind,
    DependencySelector,
    DependencyVersion,
    EvidenceRecord,
    EvidenceResult,
    EvidenceTargetKind,
    InvalidationEvent,
    LearnedConstraintRecord,
    LineageStore,
    SemanticCategory,
    TaskFeatures,
    TransferOutcome,
    TransferRecord,
    TransformationOutcome,
    TransformationRecord,
    initialize_search_from_lineage,
    retrieve_related_transformations,
)

NOW = datetime(2026, 1, 1, tzinfo=UTC)
AS_OF = NOW + timedelta(days=30)


def _hash(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _dependency(
    name: str, version: str, kind: DependencyKind = DependencyKind.COMPILER
) -> DependencyVersion:
    return DependencyVersion(
        kind=kind,
        name=name,
        version=version,
        content_hash=_hash(f"{name}-{version}"),
    )


def _task(
    task_id: str,
    *,
    model: str,
    operation: str,
    workload: str,
    hardware: str,
    dependency: DependencyVersion,
) -> TaskFeatures:
    return TaskFeatures(
        task_id=task_id,
        model_family=model,
        operator_families=(operation,),
        workload_regimes=(workload,),
        hardware_architecture=hardware,
        topology_features=(("nvlink",) if hardware == "sm90" else ("pcie",)),
        dependencies=(dependency,),
        model_contract_hash=_hash(f"model-{task_id}"),
        workload_contract_hash=_hash(f"workload-{task_id}"),
    )


def _candidate(candidate_id: str, task_id: str) -> CandidateRecord:
    return CandidateRecord(
        candidate_id=candidate_id,
        task_id=task_id,
        genome_hash=_hash(candidate_id),
        disposition=CandidateDisposition.ACCEPTED,
        created_at=NOW,
    )


def _transformation(
    transformation_id: str,
    candidate_id: str,
    *,
    family: str,
    outcome: TransformationOutcome = TransformationOutcome.ACCEPTED,
) -> TransformationRecord:
    return TransformationRecord(
        transformation_id=transformation_id,
        family=family,
        semantic_category=SemanticCategory.POLICY,
        source_candidate_id=candidate_id,
        affected_regions=("serving.scheduler",),
        preconditions=("bounded queue",),
        applicable_model_families=("sparse-moe",),
        applicable_operations=("expert-dispatch",),
        applicable_hardware=("sm90",),
        applicable_workloads=("bimodal-prompts",),
        dependency_preconditions=(
            DependencySelector(
                kind=DependencyKind.COMPILER,
                name="triton",
                version_range=">=2.0,<4.0",
            ),
        ),
        expected_benefit=0.15,
        outcome=outcome,
        proposal_source="deterministic-local-search",
        created_at=NOW,
    )


def _evidence(
    evidence_id: str,
    transformation_id: str,
    *,
    model: str,
    workload: str,
    hardware: str,
    dependency: DependencyVersion,
    confidence: float,
) -> EvidenceRecord:
    return EvidenceRecord(
        evidence_id=evidence_id,
        target_kind=EvidenceTargetKind.TRANSFORMATION,
        target_id=transformation_id,
        evidence_type="full-stack-benchmark",
        result=EvidenceResult.PASS,
        content_hash=_hash(evidence_id),
        model_family=model,
        workload_regimes=(workload,),
        hardware_architecture=hardware,
        dependencies=(dependency,),
        base_confidence=confidence,
        observed_at=NOW,
        valid_until=NOW + timedelta(days=365),
    )


def _store(path: Path) -> tuple[LineageStore, TaskFeatures]:
    triton = _dependency("triton", "2.2.0")
    torch = _dependency("torch", "2.8.0", DependencyKind.RUNTIME)
    related_task = _task(
        "related-source",
        model="sparse-moe",
        operation="expert-dispatch",
        workload="bimodal-prompts",
        hardware="sm90",
        dependency=triton,
    )
    unrelated_task = _task(
        "unrelated-source",
        model="dense-decoder",
        operation="dense-attention",
        workload="uniform-prompts",
        hardware="cpu",
        dependency=torch,
    )
    target = _task(
        "target-task",
        model="sparse-moe",
        operation="expert-dispatch",
        workload="bimodal-prompts",
        hardware="sm90",
        dependency=triton,
    )
    store = LineageStore(path)
    for task in (related_task, unrelated_task, target):
        store.record_task(task)
    for candidate in (
        _candidate("candidate-related", related_task.task_id),
        _candidate("candidate-unrelated", unrelated_task.task_id),
        _candidate("candidate-rejected", related_task.task_id),
    ):
        store.record_candidate(candidate)
    related = _transformation(
        "transform-related", "candidate-related", family="expert-dispatch-scheduler"
    )
    unrelated = _transformation(
        "transform-unrelated", "candidate-unrelated", family="generic-queue-policy"
    )
    rejected = _transformation(
        "transform-rejected",
        "candidate-rejected",
        family="unsafe-dispatch",
        outcome=TransformationOutcome.REJECTED,
    )
    for transformation in (related, unrelated, rejected):
        store.record_transformation(transformation)
    store.record_evidence(
        _evidence(
            "evidence-related",
            related.transformation_id,
            model="sparse-moe",
            workload="bimodal-prompts",
            hardware="sm90",
            dependency=triton,
            confidence=0.95,
        )
    )
    store.record_evidence(
        _evidence(
            "evidence-unrelated",
            unrelated.transformation_id,
            model="dense-decoder",
            workload="uniform-prompts",
            hardware="cpu",
            dependency=torch,
            confidence=0.8,
        )
    )
    store.record_evidence(
        _evidence(
            "evidence-rejected",
            rejected.transformation_id,
            model="sparse-moe",
            workload="bimodal-prompts",
            hardware="sm90",
            dependency=triton,
            confidence=0.99,
        )
    )
    return store, target


def test_related_lineage_ranks_first_and_initialization_preserves_diversity(
    tmp_path: Path,
) -> None:
    store, target = _store(tmp_path / "lineage.sqlite3")
    with store:
        first = retrieve_related_transformations(store, target, seed=73129, as_of=AS_OF, limit=10)
        second = retrieve_related_transformations(store, target, seed=73129, as_of=AS_OF, limit=10)
        assert first == second
        assert [item.transformation_id for item in first] == [
            "transform-related",
            "transform-unrelated",
        ]
        assert first[0].score > first[1].score
        assert all(item.requires_reverification for item in first)

        initialized = initialize_search_from_lineage(
            store,
            target,
            seed=73129,
            as_of=AS_OF,
            population_size=6,
            lineage_fraction=0.5,
        )
        assert initialized.lineage_seeds == first[:2]
        assert len(initialized.lineage_seeds) + len(initialized.unseeded_proposals) == 6
        assert len(initialized.unseeded_proposals) >= 3
        assert initialized == initialize_search_from_lineage(
            store,
            target,
            seed=73129,
            as_of=AS_OF,
            population_size=6,
            lineage_fraction=0.5,
        )


def test_negative_transfer_penalizes_but_does_not_hide_history(tmp_path: Path) -> None:
    store, target = _store(tmp_path / "lineage.sqlite3")
    with store:
        before = retrieve_related_transformations(store, target, seed=17, as_of=AS_OF)
        related_before = next(
            item for item in before if item.transformation_id == "transform-related"
        )
        store.record_transfer(
            TransferRecord(
                transfer_id="negative-transfer",
                target_task_id=target.task_id,
                transformation_id="transform-related",
                source_evidence_ids=("evidence-related",),
                retrieval_score=related_before.score,
                rank=1,
                seed=17,
                outcome=TransferOutcome.NEGATIVE_TRANSFER,
                rationale="target verification found a quality regression",
                created_at=AS_OF,
            )
        )
        after = retrieve_related_transformations(store, target, seed=17, as_of=AS_OF)
        related_after = next(
            item for item in after if item.transformation_id == "transform-related"
        )
        assert related_after.score < related_before.score
        assert store.list_transfers()[0].outcome is TransferOutcome.NEGATIVE_TRANSFER


def test_dependency_invalidation_prevents_silent_stale_reuse(tmp_path: Path) -> None:
    store, target = _store(tmp_path / "lineage.sqlite3")
    with store:
        assert "transform-related" in {
            item.transformation_id
            for item in retrieve_related_transformations(store, target, seed=3, as_of=AS_OF)
        }
        affected = store.invalidate_dependency(
            InvalidationEvent(
                invalidation_id="invalidate-related-triton",
                selector=DependencySelector(
                    kind=DependencyKind.COMPILER,
                    name="triton",
                    version_range="2.x",
                ),
                reason="compiler upgrade invalidates code-generation evidence",
                occurred_at=AS_OF,
            )
        )
        assert affected == 2
        remaining = retrieve_related_transformations(store, target, seed=3, as_of=AS_OF)
        assert "transform-related" not in {item.transformation_id for item in remaining}
        assert "transform-unrelated" in {item.transformation_id for item in remaining}


def test_minimized_counterexample_constraint_blocks_repeated_transfer(tmp_path: Path) -> None:
    store, target = _store(tmp_path / "lineage.sqlite3")
    with store:
        counterexample = CounterexampleRecord(
            counterexample_id="transfer-counterexample",
            candidate_id="candidate-related",
            transformation_id="transform-related",
            transformation_family="expert-dispatch-scheduler",
            scope=CounterexampleScope.DEPENDENCY,
            violated_contract="Triton 2.x cancellation schedule loses ownership",
            minimized_input_hash=_hash("minimized-schedule"),
            reproduction_command=("sloforge", "redteam", "replay", "--seed", "19"),
            learned_precondition="do not transfer this policy on Triton 2.x",
            dependencies=(_dependency("triton", "2.2.0"),),
            created_at=AS_OF,
        )
        store.record_counterexample(counterexample)
        store.record_constraint(
            LearnedConstraintRecord(
                constraint_id="avoid-triton-v2-transfer",
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
                rationale=counterexample.learned_precondition,
                created_at=AS_OF,
            )
        )
        retrieved = retrieve_related_transformations(store, target, seed=19, as_of=AS_OF)
        assert "transform-related" not in {item.transformation_id for item in retrieved}
