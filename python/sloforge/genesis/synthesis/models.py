"""Typed CEGIS outcomes, protocol witnesses, and generalized constraints."""

from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from sloforge.genesis.ir import (
    CounterexampleScope,
    LearnedConstraint,
    RequestEventCase,
    TransformationFamily,
)
from sloforge.genesis.search import CandidateDesign

NonEmpty = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
PositiveInt = Annotated[int, Field(gt=0)]
NonNegativeInt = Annotated[int, Field(ge=0)]


class SynthesisModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, validate_default=True)


class ProtocolWitness(SynthesisModel):
    events: tuple[RequestEventCase, ...]

    @model_validator(mode="after")
    def bounded_trace(self) -> Self:
        if not self.events or len(self.events) > 256:
            raise ValueError("protocol witness must contain between 1 and 256 events")
        if tuple(event.at_step for event in self.events) != tuple(range(len(self.events))):
            raise ValueError("protocol witness steps must be contiguous from zero")
        return self


class VerificationFailure(SynthesisModel):
    violated_contract: NonEmpty
    scope: CounterexampleScope
    transformation_id: NonEmpty | None
    expected_behavior: NonEmpty
    observed_behavior: NonEmpty
    witness: ProtocolWitness


class VerificationOutcome(SynthesisModel):
    passed: bool
    evidence_id: NonEmpty
    failure: VerificationFailure | None = None

    @model_validator(mode="after")
    def pass_failure_exclusive(self) -> Self:
        if self.passed == (self.failure is not None):
            raise ValueError(
                "verification outcome must contain failure exactly when it did not pass"
            )
        return self


class GeneralizedConstraint(SynthesisModel):
    learned: LearnedConstraint
    family: TransformationFamily
    parameter_key: NonEmpty
    required_value: NonEmpty
    rationale: NonEmpty

    def rejects(self, candidate: CandidateDesign) -> bool:
        relevant = [mutation for mutation in candidate.mutations if mutation.family is self.family]
        return any(
            mutation.parameter(self.parameter_key) != self.required_value for mutation in relevant
        )


class ConstraintDocument(SynthesisModel):
    schema_version: Literal["1.0.0"] = "1.0.0"
    constraints: tuple[GeneralizedConstraint, ...] = ()


class MinimizationResult(SynthesisModel):
    witness: ProtocolWitness
    evaluations: PositiveInt


class CegisConfiguration(SynthesisModel):
    seed: NonNegativeInt
    maximum_candidates: PositiveInt
    maximum_minimization_evaluations: PositiveInt = 256
    maximum_events: PositiveInt = 4096


class CegisEvent(SynthesisModel):
    schema_version: Literal["1.0.0"] = "1.0.0"
    sequence: NonNegativeInt
    event_type: Literal[
        "candidate_verification",
        "candidate_rejected",
        "candidate_suppressed",
        "counterexample_captured",
        "counterexample_minimized",
        "constraint_learned",
        "candidate_accepted",
    ]
    candidate_id: NonEmpty
    reason: NonEmpty
    counterexample_id: NonEmpty | None = None
    constraint_id: NonEmpty | None = None


class CegisRunResult(SynthesisModel):
    accepted_candidate_id: NonEmpty | None
    rejected_candidate_ids: tuple[NonEmpty, ...]
    suppressed_candidate_ids: tuple[NonEmpty, ...]
    counterexample_ids: tuple[NonEmpty, ...]
    constraint_ids: tuple[NonEmpty, ...]
    verifier_invocations: NonNegativeInt
    minimization_evaluations: NonNegativeInt
    events_path: NonEmpty
