"""Map causal Autopsy evidence to a bounded Genesis mutation surface."""

from __future__ import annotations

import json
from typing import Literal

from sloforge.autopsy.models import BottleneckKind, DiagnosisRecord
from sloforge.genesis.ir import InferenceGenome, TransformationFamily
from sloforge.genesis.search import CandidateDesign

from .models import (
    ALL_REGIONS,
    MutationBudget,
    Region,
    SearchEfficiencyComparison,
    SearchEfficiencySummary,
    UpsideEstimate,
)


class FrozenMutationError(ValueError):
    """A candidate attempted to mutate outside Autopsy's evidence-backed budget."""


_QUEUEING = {
    BottleneckKind.ARRIVAL_OVERLOAD,
    BottleneckKind.GATEWAY_QUEUEING,
    BottleneckKind.BACKEND_QUEUEING,
    BottleneckKind.INSUFFICIENT_WARM_CAPACITY,
    BottleneckKind.PREFILL_POOL_SATURATION,
    BottleneckKind.DECODE_POOL_SATURATION,
}
_STARTUP = {BottleneckKind.COLD_START_REGRESSION, BottleneckKind.MODEL_LOADING_REGRESSION}
_TENSOR_KERNEL = {
    BottleneckKind.CPU_LAUNCH_BOTTLENECK,
    BottleneckKind.EXCESSIVE_KERNEL_LAUNCHES,
    BottleneckKind.GPU_COMPUTE_REGRESSION,
    BottleneckKind.GPU_MEMORY_BANDWIDTH_REGRESSION,
    BottleneckKind.GPU_CLOCK_THROTTLING,
}
_FABRIC = {
    BottleneckKind.NUMA_MISPLACEMENT,
    BottleneckKind.PCIE_BOTTLENECK,
    BottleneckKind.NVLINK_DEGRADATION,
    BottleneckKind.NETWORK_BANDWIDTH_DEGRADATION,
    BottleneckKind.NETWORK_LATENCY_DEGRADATION,
    BottleneckKind.RANK_STRAGGLER,
    BottleneckKind.COLLECTIVE_IMBALANCE,
    BottleneckKind.COLLECTIVE_ALGORITHM_REGRESSION,
    BottleneckKind.TOPOLOGY_MISMATCH,
    BottleneckKind.INVALID_PHYSICAL_PLAN,
}


def _surface(
    bottleneck: BottleneckKind,
) -> tuple[tuple[Region, ...], tuple[TransformationFamily, ...]]:
    if bottleneck in _QUEUEING:
        return (
            ("workflow", "request", "serving", "state"),
            (
                TransformationFamily.WORKFLOW,
                TransformationFamily.SCHEDULER,
                TransformationFamily.BATCHING,
                TransformationFamily.CACHE_POLICY,
            ),
        )
    if bottleneck in _STARTUP:
        return (
            ("state", "kernel", "recovery"),
            (
                TransformationFamily.STATE_LAYOUT,
                TransformationFamily.KERNEL,
                TransformationFamily.RECOVERY,
                TransformationFamily.RUNTIME_CODE_PATCH,
            ),
        )
    if bottleneck in _TENSOR_KERNEL:
        return (
            ("serving", "state", "tensor", "kernel"),
            (
                TransformationFamily.ALGEBRAIC_REWRITE,
                TransformationFamily.TENSOR_DECOMPOSITION,
                TransformationFamily.OPERATOR_FUSION,
                TransformationFamily.LAYOUT,
                TransformationFamily.PRECISION,
                TransformationFamily.KERNEL,
            ),
        )
    if bottleneck in _FABRIC:
        return (
            ("state", "distributed", "kernel", "recovery"),
            (
                TransformationFamily.STATE_LAYOUT,
                TransformationFamily.DISTRIBUTED_PLAN,
                TransformationFamily.COMMUNICATION,
                TransformationFamily.KERNEL,
                TransformationFamily.RECOVERY,
            ),
        )
    if bottleneck == BottleneckKind.EXPERT_LOAD_IMBALANCE:
        return (
            ("serving", "state", "distributed", "tensor", "kernel"),
            (
                TransformationFamily.SCHEDULER,
                TransformationFamily.STATE_LAYOUT,
                TransformationFamily.DISTRIBUTED_PLAN,
                TransformationFamily.COMMUNICATION,
                TransformationFamily.LAYOUT,
                TransformationFamily.KERNEL,
            ),
        )
    if bottleneck == BottleneckKind.KV_TRANSFER_BOTTLENECK:
        return (
            ("serving", "state", "distributed"),
            (
                TransformationFamily.SCHEDULER,
                TransformationFamily.CACHE_POLICY,
                TransformationFamily.STATE_LAYOUT,
                TransformationFamily.DISTRIBUTED_PLAN,
                TransformationFamily.COMMUNICATION,
            ),
        )
    if bottleneck in {BottleneckKind.UNHEALTHY_WORKER, BottleneckKind.WORKER_CRASH}:
        return (
            ("state", "distributed", "recovery"),
            (
                TransformationFamily.STATE_LAYOUT,
                TransformationFamily.DISTRIBUTED_PLAN,
                TransformationFamily.RECOVERY,
            ),
        )
    raise ValueError(f"no Genesis attribution exists for bottleneck {bottleneck}")


def build_mutation_budget(
    diagnosis: DiagnosisRecord, *, minimum_confidence: float = 0.25
) -> MutationBudget:
    """Derive a fail-closed mutation whitelist from causal diagnosis evidence."""

    if not 0.0 <= minimum_confidence <= 1.0:
        raise ValueError("minimum_confidence must be in [0, 1]")
    if not diagnosis.sufficient_alignment:
        raise ValueError("Autopsy evidence has insufficient clock alignment")
    if diagnosis.confidence < minimum_confidence:
        raise ValueError("Autopsy diagnosis confidence is below the mutation threshold")
    mutable, families = _surface(diagnosis.top_hypothesis)
    frozen = tuple(region for region in ALL_REGIONS if region not in mutable)
    hypothesis = next(
        item for item in diagnosis.hypotheses if item.kind == diagnosis.top_hypothesis
    )
    counterfactual = hypothesis.counterfactual
    upside = (
        UpsideEstimate(
            lower_ms=counterfactual.lower_improvement_ms,
            expected_ms=counterfactual.expected_improvement_ms,
            upper_ms=counterfactual.upper_improvement_ms,
            source=counterfactual.scenario_id,
        )
        if counterfactual is not None
        else None
    )
    evidence_ids = tuple(
        sorted({f"{evidence.source}:{evidence.sha256}" for evidence in diagnosis.evidence})
    )
    next_bottleneck = diagnosis.top_three[1] if len(diagnosis.top_three) > 1 else None
    verification_cost: Literal["low", "medium", "high"] = (
        "high" if len(mutable) >= 5 else "medium" if len(mutable) >= 3 else "low"
    )
    return MutationBudget(
        diagnosis_id=diagnosis.diagnosis_id,
        bottleneck=diagnosis.top_hypothesis,
        confidence=diagnosis.confidence,
        mutable_regions=mutable,
        frozen_regions=frozen,
        allowed_families=families,
        expected_upside=upside,
        expected_verification_cost=verification_cost,
        next_bottleneck=next_bottleneck,
        evidence_ids=evidence_ids,
    )


def _set_frozen(value: object, frozen: bool) -> None:
    if isinstance(value, dict):
        node = value.get("node")
        if isinstance(node, dict) and "frozen" in node:
            node["frozen"] = frozen
        for child in value.values():
            _set_frozen(child, frozen)
    elif isinstance(value, list):
        for child in value:
            _set_frozen(child, frozen)


def freeze_genome_regions(genome: InferenceGenome, budget: MutationBudget) -> InferenceGenome:
    """Mark every node in frozen regions and unfreeze every whitelisted region."""

    payload = genome.model_dump(mode="json")
    for region in ALL_REGIONS:
        _set_frozen(payload[region], region in budget.frozen_regions)
    return InferenceGenome.model_validate_json(
        json.dumps(payload, sort_keys=True, separators=(",", ":")), strict=True
    )


class MutationGuard:
    def __init__(self, budget: MutationBudget) -> None:
        self.budget = budget

    def validate(self, candidate: CandidateDesign) -> None:
        disallowed_regions = set(candidate.affected_regions) - set(self.budget.mutable_regions)
        if disallowed_regions:
            raise FrozenMutationError(
                f"candidate {candidate.candidate_id} mutates frozen regions "
                f"{sorted(disallowed_regions)}"
            )
        disallowed_families = {
            mutation.family
            for mutation in candidate.mutations
            if mutation.family not in self.budget.allowed_families
        }
        if disallowed_families:
            raise FrozenMutationError(
                f"candidate {candidate.candidate_id} uses transformations outside the "
                f"Autopsy budget: {sorted(str(item) for item in disallowed_families)}"
            )


def compare_search_efficiency(
    guided: SearchEfficiencySummary, unguided: SearchEfficiencySummary
) -> SearchEfficiencyComparison:
    guided_time = guided.time_to_improvement_seconds
    unguided_time = unguided.time_to_improvement_seconds
    guided_objective = guided.final_objective
    unguided_objective = unguided.final_objective
    return SearchEfficiencyComparison(
        guided=guided,
        unguided=unguided,
        candidate_reduction=unguided.candidates_evaluated - guided.candidates_evaluated,
        invalid_candidate_reduction=unguided.invalid_candidates - guided.invalid_candidates,
        hardware_experiment_reduction=(unguided.hardware_experiments - guided.hardware_experiments),
        seconds_to_improvement_reduction=(
            unguided_time - guided_time
            if guided_time is not None and unguided_time is not None
            else None
        ),
        objective_delta=(
            guided_objective - unguided_objective
            if guided_objective is not None and unguided_objective is not None
            else None
        ),
        diversity_delta=(
            guided.distinct_transformation_families - unguided.distinct_transformation_families
        ),
    )


__all__ = [
    "FrozenMutationError",
    "MutationGuard",
    "build_mutation_budget",
    "compare_search_efficiency",
    "freeze_genome_regions",
]
