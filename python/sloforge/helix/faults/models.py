"""Strict contracts for deterministic, CPU-only Helix fault campaigns."""

from __future__ import annotations

import hashlib
import json
from enum import StrEnum
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

Identifier = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}$")]
Digest = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
NonEmpty = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=2048)]


def canonical_digest(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class FaultModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, validate_default=True)


class FaultStage(StrEnum):
    CAPTURE = "capture"
    BRANCHING = "branching"
    ROLLOUT = "rollout"
    REWARD = "reward"
    TRAINING = "training"
    EVALUATION = "evaluation"
    PROMOTION = "promotion"
    RESOURCE = "resource"


class FaultKind(StrEnum):
    INCONSISTENT_BOUNDARY = "inconsistent_boundary"
    ILLEGAL_EFFECT = "illegal_effect"
    MISSING_LOGPROB = "missing_logprob"
    POLICY_MISMATCH = "policy_mismatch"
    POLICY_STALENESS = "policy_staleness"
    REWARD_CORRUPTION = "reward_corruption"
    DUPLICATE_REWARD = "duplicate_reward"
    CHECKPOINT_FAILURE = "checkpoint_failure"
    LINEAGE_FAILURE = "lineage_failure"
    TRAINING_FAILURE = "training_failure"
    QUALITY_REJECT = "quality_reject"
    SERVING_REJECT = "serving_reject"
    COMPATIBILITY_REJECT = "compatibility_reject"
    PARTIAL_CHAMPION_POINTER = "partial_champion_pointer"
    TRAFFIC_SPIKE = "traffic_spike"
    GPU_LOSS = "gpu_loss"


class FaultResponse(StrEnum):
    REJECTED = "rejected"
    ABORTED = "aborted"
    QUARANTINED = "quarantined"
    DEFERRED = "deferred"
    RETAINED_CHAMPION = "retained_champion"
    ROLLED_BACK = "rolled_back"
    SERVING_PROTECTED = "serving_protected"
    CAPACITY_REDUCED = "capacity_reduced"
    CALLBACK_ERROR = "callback_error"
    CALLBACK_TIMEOUT = "callback_timeout"
    CALLBACK_MISSING = "callback_missing"


class MutationOperation(StrEnum):
    SET = "set"
    DELETE = "delete"
    SCALE = "scale"
    DUPLICATE = "duplicate"
    INTERRUPT = "interrupt"


EXPECTED_STAGE: dict[FaultKind, FaultStage] = {
    FaultKind.INCONSISTENT_BOUNDARY: FaultStage.CAPTURE,
    FaultKind.ILLEGAL_EFFECT: FaultStage.ROLLOUT,
    FaultKind.MISSING_LOGPROB: FaultStage.ROLLOUT,
    FaultKind.POLICY_MISMATCH: FaultStage.ROLLOUT,
    FaultKind.POLICY_STALENESS: FaultStage.ROLLOUT,
    FaultKind.REWARD_CORRUPTION: FaultStage.REWARD,
    FaultKind.DUPLICATE_REWARD: FaultStage.REWARD,
    FaultKind.CHECKPOINT_FAILURE: FaultStage.BRANCHING,
    FaultKind.LINEAGE_FAILURE: FaultStage.TRAINING,
    FaultKind.TRAINING_FAILURE: FaultStage.TRAINING,
    FaultKind.QUALITY_REJECT: FaultStage.EVALUATION,
    FaultKind.SERVING_REJECT: FaultStage.EVALUATION,
    FaultKind.COMPATIBILITY_REJECT: FaultStage.EVALUATION,
    FaultKind.PARTIAL_CHAMPION_POINTER: FaultStage.PROMOTION,
    FaultKind.TRAFFIC_SPIKE: FaultStage.RESOURCE,
    FaultKind.GPU_LOSS: FaultStage.RESOURCE,
}

EXPECTED_RESPONSE: dict[FaultKind, FaultResponse] = {
    FaultKind.INCONSISTENT_BOUNDARY: FaultResponse.ABORTED,
    FaultKind.ILLEGAL_EFFECT: FaultResponse.REJECTED,
    FaultKind.MISSING_LOGPROB: FaultResponse.QUARANTINED,
    FaultKind.POLICY_MISMATCH: FaultResponse.REJECTED,
    FaultKind.POLICY_STALENESS: FaultResponse.DEFERRED,
    FaultKind.REWARD_CORRUPTION: FaultResponse.QUARANTINED,
    FaultKind.DUPLICATE_REWARD: FaultResponse.REJECTED,
    FaultKind.CHECKPOINT_FAILURE: FaultResponse.ABORTED,
    FaultKind.LINEAGE_FAILURE: FaultResponse.REJECTED,
    FaultKind.TRAINING_FAILURE: FaultResponse.ABORTED,
    FaultKind.QUALITY_REJECT: FaultResponse.RETAINED_CHAMPION,
    FaultKind.SERVING_REJECT: FaultResponse.RETAINED_CHAMPION,
    FaultKind.COMPATIBILITY_REJECT: FaultResponse.RETAINED_CHAMPION,
    FaultKind.PARTIAL_CHAMPION_POINTER: FaultResponse.ROLLED_BACK,
    FaultKind.TRAFFIC_SPIKE: FaultResponse.SERVING_PROTECTED,
    FaultKind.GPU_LOSS: FaultResponse.CAPACITY_REDUCED,
}


class ActivationInterval(FaultModel):
    start_step: int = Field(ge=0, le=1_000_000)
    end_step: int = Field(gt=0, le=1_000_001)

    @model_validator(mode="after")
    def validate_order(self) -> Self:
        if self.end_step <= self.start_step:
            raise ValueError("fault activation end must be after its start")
        return self


class FaultSpec(FaultModel):
    fault_id: Identifier
    ground_truth_label: Identifier
    kind: FaultKind
    stage: FaultStage
    activation: ActivationInterval
    expected_response: FaultResponse
    evidence_sha256: Digest
    magnitude: Annotated[float, Field(gt=0.0, le=1000.0, allow_inf_nan=False)] = 1.0

    @model_validator(mode="after")
    def validate_ground_truth(self) -> Self:
        if self.stage is not EXPECTED_STAGE[self.kind]:
            raise ValueError("fault kind is assigned to the wrong Helix stage")
        if self.expected_response is not EXPECTED_RESPONSE[self.kind]:
            raise ValueError("fault kind has an incorrect expected fail-closed response")
        if self.kind is FaultKind.GPU_LOSS and self.magnitude > 1.0:
            raise ValueError("GPU loss magnitude must be at most one")
        return self


class FaultPlanRequest(FaultModel):
    schema_version: Literal["sloforge.helix.fault-plan-request/v1"] = (
        "sloforge.helix.fault-plan-request/v1"
    )
    request_id: Identifier
    seed: int = Field(ge=0, le=2**64 - 1)
    horizon_steps: int = Field(gt=0, le=1_000_000)
    callback_timeout_ms: int = Field(gt=0, le=60_000)
    max_callbacks: int = Field(gt=0, le=100_000)
    faults: Annotated[tuple[FaultSpec, ...], Field(min_length=1, max_length=100_000)]
    assumptions: Annotated[tuple[NonEmpty, ...], Field(max_length=128)] = ()

    @model_validator(mode="after")
    def validate_matrix(self) -> Self:
        ids = tuple(fault.fault_id for fault in self.faults)
        labels = tuple(fault.ground_truth_label for fault in self.faults)
        identities = tuple((fault.kind, fault.activation) for fault in self.faults)
        if len(ids) != len(set(ids)):
            raise ValueError("fault identifiers must be unique")
        if len(labels) != len(set(labels)):
            raise ValueError("ground-truth labels must be unique")
        if len(identities) != len(set(identities)):
            raise ValueError("duplicate fault kind and activation interval")
        if any(fault.activation.end_step > self.horizon_steps for fault in self.faults):
            raise ValueError("fault activation exceeds the plan horizon")
        if self.max_callbacks < len(self.faults):
            raise ValueError("callback bound must cover every planned fault")
        if len(self.assumptions) != len(set(self.assumptions)):
            raise ValueError("fault plan assumptions must be unique")
        return self


class FaultPlan(FaultModel):
    schema_version: Literal["sloforge.helix.fault-plan/v1"] = "sloforge.helix.fault-plan/v1"
    plan_id: Digest
    request_digest: Digest
    request_id: Identifier
    seed: int = Field(ge=0, le=2**64 - 1)
    horizon_steps: int = Field(gt=0, le=1_000_000)
    callback_timeout_ms: int = Field(gt=0, le=60_000)
    max_callbacks: int = Field(gt=0, le=100_000)
    faults: Annotated[tuple[FaultSpec, ...], Field(min_length=1, max_length=100_000)]
    assumptions: Annotated[tuple[NonEmpty, ...], Field(min_length=1, max_length=128)]

    @model_validator(mode="after")
    def validate_seal(self) -> Self:
        FaultPlanRequest(
            request_id=self.request_id,
            seed=self.seed,
            horizon_steps=self.horizon_steps,
            callback_timeout_ms=self.callback_timeout_ms,
            max_callbacks=self.max_callbacks,
            faults=self.faults,
            assumptions=(),
        )
        identity = self.model_dump(mode="json", exclude={"plan_id"})
        if canonical_digest(identity) != self.plan_id:
            raise ValueError("fault plan identifier is invalid")
        return self


class FaultMutation(FaultModel):
    field: NonEmpty
    operation: MutationOperation
    encoded_value: Annotated[str | None, Field(max_length=4096)] = None

    @model_validator(mode="after")
    def validate_value(self) -> Self:
        if (self.operation is MutationOperation.DELETE) != (self.encoded_value is None):
            raise ValueError("only delete mutations omit their encoded value")
        return self


class InjectedFault(FaultModel):
    injection_id: Digest
    fault_id: Identifier
    ground_truth_label: Identifier
    kind: FaultKind
    stage: FaultStage
    activation: ActivationInterval
    seed: int = Field(ge=0, le=2**64 - 1)
    expected_response: FaultResponse
    evidence_sha256: Digest
    mutations: Annotated[tuple[FaultMutation, ...], Field(min_length=1, max_length=8)]

    @model_validator(mode="after")
    def validate_seal(self) -> Self:
        identity = self.model_dump(mode="json", exclude={"injection_id"})
        if canonical_digest(identity) != self.injection_id:
            raise ValueError("injected fault identifier is invalid")
        return self


class FaultObservation(FaultModel):
    actual_response: FaultResponse
    detail: NonEmpty
    evidence_sha256: Digest


class FaultExecutionResult(FaultModel):
    fault_id: Identifier
    ground_truth_label: Identifier
    kind: FaultKind
    stage: FaultStage
    activation: ActivationInterval
    expected_response: FaultResponse
    actual_response: FaultResponse
    passed: bool
    evidence_sha256: Digest
    observation_evidence_sha256: Digest
    injection_id: Digest
    detail: NonEmpty

    @model_validator(mode="after")
    def validate_outcome(self) -> Self:
        if self.passed != (self.actual_response is self.expected_response):
            raise ValueError("fault pass flag disagrees with expected and actual responses")
        return self


class FaultCampaignResult(FaultModel):
    schema_version: Literal["sloforge.helix.fault-campaign-result/v1"] = (
        "sloforge.helix.fault-campaign-result/v1"
    )
    result_id: Digest
    plan_id: Digest
    seed: int = Field(ge=0, le=2**64 - 1)
    passed: bool
    callback_count: int = Field(ge=0, le=100_000)
    results: Annotated[tuple[FaultExecutionResult, ...], Field(min_length=1, max_length=100_000)]
    failed_fault_ids: tuple[Identifier, ...]
    assumptions: Annotated[tuple[NonEmpty, ...], Field(min_length=1, max_length=128)]

    @model_validator(mode="after")
    def validate_campaign(self) -> Self:
        ids = tuple(result.fault_id for result in self.results)
        if len(ids) != len(set(ids)):
            raise ValueError("fault campaign contains duplicate results")
        failed = tuple(result.fault_id for result in self.results if not result.passed)
        if failed != self.failed_fault_ids or self.passed != (not failed):
            raise ValueError("campaign disposition disagrees with fault results")
        if self.callback_count > len(self.results):
            raise ValueError("callback count exceeds fault result count")
        identity = self.model_dump(mode="json", exclude={"result_id"})
        if canonical_digest(identity) != self.result_id:
            raise ValueError("fault campaign result identifier is invalid")
        return self

    def require_passed(self) -> None:
        if not self.passed:
            raise FaultCampaignFailed(self.failed_fault_ids)


class FaultCampaignFailed(RuntimeError):
    def __init__(self, failed_fault_ids: tuple[str, ...]) -> None:
        self.failed_fault_ids = failed_fault_ids
        super().__init__(f"fault campaign failed closed: {', '.join(failed_fault_ids)}")


__all__ = [
    "ActivationInterval",
    "FaultCampaignFailed",
    "FaultCampaignResult",
    "FaultExecutionResult",
    "FaultKind",
    "FaultMutation",
    "FaultObservation",
    "FaultPlan",
    "FaultPlanRequest",
    "FaultResponse",
    "FaultSpec",
    "FaultStage",
    "InjectedFault",
    "MutationOperation",
    "canonical_digest",
]
