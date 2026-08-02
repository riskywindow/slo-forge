"""Deterministic lineage retrieval, stale-reuse, and invalidation demonstration."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from .models import (
    CandidateDisposition,
    CandidateRecord,
    DependencyKind,
    DependencySelector,
    DependencyVersion,
    EvidenceRecord,
    EvidenceResult,
    EvidenceTargetKind,
    InvalidationEvent,
    SemanticCategory,
    TaskFeatures,
    TransformationOutcome,
    TransformationRecord,
)
from .store import LineageStore
from .transfer import initialize_search_from_lineage

_NOW = datetime(2026, 8, 2, tzinfo=UTC)


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _dependency(version: str) -> DependencyVersion:
    return DependencyVersion(
        kind=DependencyKind.COMPILER,
        name="triton",
        version=version,
        content_hash=_digest(f"triton:{version}"),
    )


def _task(task_id: str, *, related: bool, version: str = "2.2.0") -> TaskFeatures:
    return TaskFeatures(
        task_id=task_id,
        model_family="sparse-moe" if related else "dense-decoder",
        operator_families=("expert-dispatch" if related else "dense-attention",),
        workload_regimes=("bimodal-prompts" if related else "uniform-prompts",),
        hardware_architecture="sm90" if related else "cpu",
        topology_features=("nvlink" if related else "pcie",),
        dependencies=(_dependency(version),),
        model_contract_hash=_digest(f"model:{task_id}"),
        workload_contract_hash=_digest(f"workload:{task_id}"),
    )


def _record_discovery(
    store: LineageStore,
    *,
    task: TaskFeatures,
    candidate_id: str,
    transformation_id: str,
    applicable_to_target: bool,
) -> None:
    store.record_task(task)
    store.record_candidate(
        CandidateRecord(
            candidate_id=candidate_id,
            task_id=task.task_id,
            genome_hash=_digest(candidate_id),
            disposition=CandidateDisposition.ACCEPTED,
            created_at=_NOW,
        )
    )
    store.record_transformation(
        TransformationRecord(
            transformation_id=transformation_id,
            family="expert-dispatch-scheduler" if applicable_to_target else "dense-queue",
            semantic_category=SemanticCategory.POLICY,
            source_candidate_id=candidate_id,
            affected_regions=("serving.scheduler",),
            preconditions=("bounded queue", "independent request state"),
            applicable_model_families=("sparse-moe" if applicable_to_target else "dense-decoder",),
            applicable_operations=(
                "expert-dispatch" if applicable_to_target else "dense-attention",
            ),
            applicable_hardware=("sm90" if applicable_to_target else "cpu",),
            applicable_workloads=(
                "bimodal-prompts" if applicable_to_target else "uniform-prompts",
            ),
            dependency_preconditions=(
                DependencySelector(
                    kind=DependencyKind.COMPILER,
                    name="triton",
                    version_range=">=2.0,<4.0",
                ),
            ),
            expected_benefit=0.12 if applicable_to_target else 0.04,
            outcome=TransformationOutcome.ACCEPTED,
            proposal_source="deterministic-local-fixture",
            created_at=_NOW,
        )
    )
    store.record_evidence(
        EvidenceRecord(
            evidence_id=f"evidence-{transformation_id}",
            target_kind=EvidenceTargetKind.TRANSFORMATION,
            target_id=transformation_id,
            evidence_type="deterministic-simulator",
            result=EvidenceResult.PASS,
            content_hash=_digest(f"evidence:{transformation_id}"),
            model_family=task.model_family,
            workload_regimes=task.workload_regimes,
            hardware_architecture=task.hardware_architecture,
            dependencies=task.dependencies,
            base_confidence=0.90,
            observed_at=_NOW,
            valid_until=_NOW + timedelta(days=90),
        )
    )


def _case(store: LineageStore, target: TaskFeatures, *, seed: int) -> dict[str, Any]:
    initialized = initialize_search_from_lineage(
        store,
        target,
        seed=seed,
        as_of=_NOW + timedelta(days=1),
        population_size=5,
        lineage_fraction=0.6,
    )
    return {
        "lineage_seed_ids": [item.transformation_id for item in initialized.lineage_seeds],
        "lineage_seed_count": len(initialized.lineage_seeds),
        "unseeded_count": len(initialized.unseeded_proposals),
        "reverification_required": all(
            item.requires_reverification for item in initialized.lineage_seeds
        ),
    }


def run_lineage_transfer_demo(output: Path, *, seed: int) -> dict[str, Any]:
    if seed < 0:
        raise ValueError("seed must be non-negative")
    if output.exists() and output.is_symlink():
        raise ValueError("lineage demo output must not be a symlink")
    output.mkdir(parents=True, exist_ok=False)
    target = _task("target-related", related=True, version="2.2.0")

    empty_path = output / "empty.sqlite3"
    with LineageStore(empty_path) as empty:
        empty.record_task(target)
        empty_case = _case(empty, target, seed=seed)

    lineage_path = output / "lineage.sqlite3"
    with LineageStore(lineage_path) as store:
        unrelated = _task("source-unrelated", related=False)
        _record_discovery(
            store,
            task=unrelated,
            candidate_id="candidate-unrelated",
            transformation_id="transformation-unrelated",
            applicable_to_target=False,
        )
        store.record_task(target)
        unrelated_case = _case(store, target, seed=seed)
        related = _task("source-related", related=True)
        _record_discovery(
            store,
            task=related,
            candidate_id="candidate-related",
            transformation_id="transformation-related",
            applicable_to_target=True,
        )
        related_case = _case(store, target, seed=seed)
        # This deliberately shows the unsafe state before invalidation: the
        # evidence remains retrievable even though an external dependency
        # change has occurred but has not yet been recorded in lineage.
        stale_without_invalidation = _case(store, target, seed=seed)
        affected = store.invalidate_dependency(
            InvalidationEvent(
                invalidation_id="invalidate-triton-2x",
                selector=DependencySelector(
                    kind=DependencyKind.COMPILER,
                    name="triton",
                    version_range="<3.0",
                ),
                reason="compiler major-version change requires revalidation",
                occurred_at=_NOW + timedelta(days=1),
            )
        )
        after_invalidation = _case(store, target, seed=seed)

    report: dict[str, Any] = {
        "schema_version": "1.0.0",
        "seed": seed,
        "scope": "deterministic lineage retrieval and invalidation mechanics; no speedup claim",
        "cases": {
            "empty_lineage": empty_case,
            "unrelated_lineage": unrelated_case,
            "related_lineage": related_case,
            "stale_dependency_before_invalidation": stale_without_invalidation,
            "stale_dependency_after_invalidation": after_invalidation,
        },
        "affected_evidence_count": affected,
        "related_seed_retrieved": "transformation-related" in related_case["lineage_seed_ids"],
        "stale_seed_suppressed_after_invalidation": "transformation-related"
        not in after_invalidation["lineage_seed_ids"],
        "performance_hypothesis_evaluated": False,
    }
    (output / "report.json").write_text(
        json.dumps(report, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=73129)
    arguments = parser.parse_args()
    report = run_lineage_transfer_demo(arguments.output, seed=arguments.seed)
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
