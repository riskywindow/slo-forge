from __future__ import annotations

from pathlib import Path

import pytest

from sloforge.autopsy.models import DiagnosisRecord
from sloforge.genesis.autopsy_guidance import (
    FrozenMutationError,
    MutationGuard,
    SearchEfficiencySummary,
    build_mutation_budget,
    compare_search_efficiency,
    freeze_genome_regions,
)
from sloforge.genesis.ir import ArtifactDigest, TransformationFamily, load_inference_genome
from sloforge.genesis.search import CandidateDesign, MutationChoice, ParameterValue

ROOT = Path(__file__).resolve().parents[2]


def _diagnosis() -> DiagnosisRecord:
    return DiagnosisRecord.model_validate_json(
        (ROOT / "tests/fixtures/autopsy/diagnosis-v1.json").read_text(encoding="utf-8")
    )


def _candidate(
    family: TransformationFamily, regions: tuple[str, ...], candidate_id: str
) -> CandidateDesign:
    return CandidateDesign.model_validate(
        {
            "candidate_id": candidate_id,
            "seed": 73129,
            "genome_hash": ArtifactDigest(value="a" * 64),
            "parent_candidate_ids": (),
            "mutations": (
                MutationChoice.model_validate(
                    {
                        "transformation_id": f"transform-{candidate_id}",
                        "family": family,
                        "regions": regions,
                        "parameters": (ParameterValue(key="choice", value="test"),),
                        "expected_upside": 1.0,
                        "invalidity_risk": 0.1,
                        "feature_delta": (1.0,),
                    }
                ),
            ),
            "feature_vector": (1.0,),
            "proposal_engine": "fixture",
        }
    )


def _all_node_frozen(value: object, expected: bool) -> bool:
    if isinstance(value, dict):
        node = value.get("node")
        if isinstance(node, dict) and node.get("frozen") is not expected:
            return False
        return all(_all_node_frozen(child, expected) for child in value.values())
    if isinstance(value, list):
        return all(_all_node_frozen(child, expected) for child in value)
    return True


def test_autopsy_maps_network_bottleneck_to_fabric_regions() -> None:
    budget = build_mutation_budget(_diagnosis())

    assert budget.mutable_regions == ("state", "distributed", "kernel", "recovery")
    assert "request" in budget.frozen_regions
    assert TransformationFamily.COMMUNICATION in budget.allowed_families
    assert budget.expected_upside is None
    assert budget.next_bottleneck is not None


def test_frozen_regions_are_enforced_on_nested_genome_nodes() -> None:
    budget = build_mutation_budget(_diagnosis())
    genome = load_inference_genome(ROOT / "tests/fixtures/genesis/inference-genome-v1.json")
    frozen = freeze_genome_regions(genome, budget)
    payload = frozen.model_dump(mode="json")

    assert _all_node_frozen(payload["request"], True)
    assert _all_node_frozen(payload["distributed"], False)
    guard = MutationGuard(budget)
    guard.validate(
        _candidate(
            TransformationFamily.COMMUNICATION,
            ("distributed",),
            "fabric-candidate",
        )
    )
    with pytest.raises(FrozenMutationError, match="frozen regions"):
        guard.validate(
            _candidate(
                TransformationFamily.SCHEDULER,
                ("request",),
                "gateway-candidate",
            )
        )


def test_guided_and_unguided_comparison_uses_observed_inputs() -> None:
    guided = SearchEfficiencySummary(
        label="autopsy-guided",
        candidates_evaluated=8,
        invalid_candidates=1,
        hardware_experiments=2,
        time_to_improvement_seconds=12.0,
        final_objective=1.2,
        distinct_transformation_families=3,
    )
    unguided = SearchEfficiencySummary(
        label="unrestricted",
        candidates_evaluated=15,
        invalid_candidates=5,
        hardware_experiments=6,
        time_to_improvement_seconds=30.0,
        final_objective=1.1,
        distinct_transformation_families=5,
    )

    comparison = compare_search_efficiency(guided, unguided)
    assert comparison.candidate_reduction == 7
    assert comparison.invalid_candidate_reduction == 4
    assert comparison.hardware_experiment_reduction == 4
    assert comparison.seconds_to_improvement_reduction == 18.0
    assert comparison.objective_delta == pytest.approx(0.1)
