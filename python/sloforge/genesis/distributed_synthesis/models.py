"""Restricted mutation surface over validated PhysicalExecutionPlan documents."""

from __future__ import annotations

from typing import Annotated, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from sloforge.fabric.ir import PhysicalExecutionPlan

NonEmpty = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
PositiveInt = Annotated[int, Field(gt=0)]


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


class DistributedSynthesisResult(DistributedModel):
    schema_version: Literal["1.0.0"] = "1.0.0"
    source_plan_id: NonEmpty
    candidate_plan: PhysicalExecutionPlan
    transformation_id: NonEmpty
    affected_surface: Literal[
        "collective", "kv_transfer", "overlap", "expert_placement", "rank_placement"
    ]
    fabric_schema_validated: bool
    performance_evidence_valid: bool
    required_verifier_stages: tuple[NonEmpty, ...]
    invalidated_evidence_uris: tuple[NonEmpty, ...]
    bounded_model_check_required: bool


__all__ = [
    "CollectiveMutation",
    "DistributedMutation",
    "DistributedSynthesisResult",
    "ExpertPlacementMutation",
    "KVTransferMutation",
    "OverlapMutation",
    "RankPlacementMutation",
]
