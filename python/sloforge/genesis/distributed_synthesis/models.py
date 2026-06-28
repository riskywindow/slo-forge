"""Restricted mutation surface over validated PhysicalExecutionPlan documents."""

from __future__ import annotations

from typing import Annotated, Final, Literal, Self, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from sloforge.fabric.ir import PhysicalExecutionPlan, canonical_hash

NonEmpty = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
PositiveInt = Annotated[int, Field(gt=0)]
Sha256Hex = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]

VerifierStage: TypeAlias = Literal[
    "fabric-static-validation",
    "fabric-resource-analysis",
    "digital-twin",
    "bounded-protocol-model-check",
    "end-to-end-benchmark",
]

REQUIRED_REVALIDATION_STAGES: Final[tuple[VerifierStage, ...]] = (
    "fabric-static-validation",
    "fabric-resource-analysis",
    "digital-twin",
    "bounded-protocol-model-check",
    "end-to-end-benchmark",
)

InvalidatedField: TypeAlias = Literal[
    "predicted_metrics",
    "bottleneck_prediction",
    "failure_exposure",
    "optimizer_history",
    "rejected_alternatives",
    "recovery_variants",
    "evidence",
]

INVALIDATED_FABRIC_FIELDS: Final[tuple[InvalidatedField, ...]] = (
    "predicted_metrics",
    "bottleneck_prediction",
    "failure_exposure",
    "optimizer_history",
    "rejected_alternatives",
    "recovery_variants",
    "evidence",
)


class DistributedModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, validate_default=True)


class CollectiveMutation(DistributedModel):
    kind: Literal["collective"] = "collective"
    transformation_id: NonEmpty
    operation_id: NonEmpty
    algorithm: Literal["ring", "tree", "recursive_doubling", "direct", "pairwise", "auto"]
    transport: Literal["shared_memory", "nvlink", "pcie", "infiniband", "roce", "tcp"]
    channel_count: Annotated[int, Field(ge=1, le=32)]
    rank_order: tuple[Annotated[int, Field(ge=0)], ...]


class KVTransferMutation(DistributedModel):
    kind: Literal["kv_transfer"] = "kv_transfer"
    transformation_id: NonEmpty
    route_id: NonEmpty
    chunk_bytes: PositiveInt
    maximum_inflight_chunks: Annotated[int, Field(ge=1, le=1024)]
    overlap_with_decode: bool
    eviction_policy: Literal["lru", "clock", "deadline"]


class OverlapMutation(DistributedModel):
    kind: Literal["overlap"] = "overlap"
    transformation_id: NonEmpty
    window_id: NonEmpty
    expected_overlap_fraction: Annotated[float, Field(ge=0.0, le=1.0)]
    stream: NonEmpty
    fallback_serialization: Literal["compute_first", "communication_first", "critical_path"]


class ExpertPlacementMutation(DistributedModel):
    kind: Literal["expert_placement"] = "expert_placement"
    transformation_id: NonEmpty
    expert_id: NonEmpty
    rank_ids: tuple[Annotated[int, Field(ge=0)], ...]
    capacity_factor: Annotated[float, Field(gt=0.0)]


class RankPlacementMutation(DistributedModel):
    kind: Literal["rank_placement"] = "rank_placement"
    transformation_id: NonEmpty
    # Element N identifies the existing binding moved to logical rank N.
    source_rank_order: tuple[Annotated[int, Field(ge=0)], ...]


DistributedMutation: TypeAlias = (
    CollectiveMutation
    | KVTransferMutation
    | OverlapMutation
    | ExpertPlacementMutation
    | RankPlacementMutation
)


class InvalidatedEvidenceReference(DistributedModel):
    """A source-plan artifact that must not be trusted for the mutated plan."""

    kind: NonEmpty
    uri: NonEmpty
    digest_sha256: Sha256Hex
    reason: Literal["distributed-plan-mutated"] = "distributed-plan-mutated"


class DistributedSynthesisResult(DistributedModel):
    schema_version: Literal["1.0.0"] = "1.0.0"
    source_plan_id: NonEmpty
    candidate_plan: PhysicalExecutionPlan
    transformation_id: NonEmpty
    affected_surface: Literal[
        "collective", "kv_transfer", "overlap", "expert_placement", "rank_placement"
    ]
    candidate_plan_hash: Sha256Hex
    source_predicted_metrics_hash: Sha256Hex
    fabric_schema_validated: Literal[True]
    evidence_state: Literal["invalidated-pending-revalidation"]
    performance_evidence_valid: Literal[False]
    eligible_for_performance_comparison: Literal[False]
    required_verifier_stages: tuple[VerifierStage, ...]
    invalidated_fields: tuple[InvalidatedField, ...]
    invalidated_evidence: tuple[InvalidatedEvidenceReference, ...]
    invalidated_evidence_uris: tuple[NonEmpty, ...]
    bounded_model_check_required: Literal[True]

    @model_validator(mode="after")
    def validate_fail_closed_evidence_state(self) -> Self:
        if self.required_verifier_stages != REQUIRED_REVALIDATION_STAGES:
            raise ValueError("distributed mutation must require the complete revalidation pipeline")
        if self.invalidated_fields != INVALIDATED_FABRIC_FIELDS:
            raise ValueError("distributed mutation must invalidate every Fabric performance field")
        if self.candidate_plan.evidence:
            raise ValueError("pending distributed candidate must not retain source evidence")
        if self.candidate_plan.failure_exposure:
            raise ValueError("pending distributed candidate must not retain failure predictions")
        if self.candidate_plan.rejected_alternatives:
            raise ValueError("pending distributed candidate must not retain source search results")
        if self.candidate_plan.recovery_variants:
            raise ValueError(
                "pending distributed candidate must not retain derived recovery variants"
            )
        if self.candidate_plan.bottleneck_prediction != (
            "invalidated-pending-distributed-revalidation"
        ):
            raise ValueError("pending distributed candidate must not retain bottleneck predictions")
        if tuple(item.uri for item in self.invalidated_evidence) != self.invalidated_evidence_uris:
            raise ValueError("invalidated evidence URI index does not match typed references")

        extension = self.candidate_plan.extensions.root.get("sloforge.dev/genesis-candidate")
        if not isinstance(extension, dict):
            raise ValueError(
                "pending distributed candidate is missing its evidence-state extension"
            )
        expected_extension_values = {
            "evidence_state": self.evidence_state,
            "performance_evidence_valid": False,
            "eligible_for_performance_comparison": False,
            "requires_bounded_model_check": True,
            "predicted_metrics_representation": (
                "source-values-zero-confidence-schema-compatibility-only"
            ),
            "required_revalidation_stages": list(REQUIRED_REVALIDATION_STAGES),
            "invalidated_fields": list(self.invalidated_fields),
            "invalidated_evidence": [
                reference.model_dump(mode="json") for reference in self.invalidated_evidence
            ],
            "source_plan_id": self.source_plan_id,
            "source_predicted_metrics_hash": self.source_predicted_metrics_hash,
            "transformation_id": self.transformation_id,
        }
        for key, expected in expected_extension_values.items():
            if extension.get(key) != expected:
                raise ValueError(f"candidate evidence-state extension has invalid {key}")

        if canonical_hash(self.candidate_plan) != self.candidate_plan_hash:
            raise ValueError("candidate plan hash does not match the invalidated plan")
        metric_payload = self.candidate_plan.predicted_metrics.model_dump(mode="json")
        if any(interval["confidence"] != 0.0 for interval in metric_payload.values()):
            raise ValueError("invalidated predicted metrics must carry zero confidence")
        history = self.candidate_plan.optimizer_history
        if len(history) != 1 or any(
            (
                entry.candidate_id != self.candidate_plan.plan_id
                or entry.phase != "feasibility"
                or entry.decision != "evaluate"
                or entry.simulator_calls != 0
            )
            for entry in history
        ):
            raise ValueError(
                "pending candidate must contain only its unsimulated feasibility event"
            )
        return self


__all__ = [
    "INVALIDATED_FABRIC_FIELDS",
    "REQUIRED_REVALIDATION_STAGES",
    "CollectiveMutation",
    "DistributedMutation",
    "DistributedSynthesisResult",
    "ExpertPlacementMutation",
    "InvalidatedEvidenceReference",
    "KVTransferMutation",
    "OverlapMutation",
    "RankPlacementMutation",
]
