"""Strict compatibility contracts for portable Continuum execution state."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from sloforge.continuum.ir import ExactnessClass as ExactnessClass

NonEmptyString = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
NonNegativeFloat = Annotated[float, Field(ge=0.0, allow_inf_nan=False)]


class CompatibilityModel(BaseModel):
    """Immutable, strict wire value used by the compatibility engine."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, validate_default=True)


class ReasonSeverity(StrEnum):
    INFO = "info"
    REQUIREMENT = "requirement"
    BLOCKING = "blocking"


class VerificationKind(StrEnum):
    STRUCTURAL = "structural"
    CONVERSION_EQUIVALENCE = "conversion_equivalence"
    CONTINUATION = "continuation"
    QUALITY = "quality"
    RECOMPUTATION = "recomputation"


class ModelSemantics(CompatibilityModel):
    """State-producing semantics, deliberately separate from physical layout."""

    schema_version: Literal["1.0.0"] = "1.0.0"
    model_id: NonEmptyString
    architecture: NonEmptyString
    weights_hash: NonEmptyString
    state_producing_weights_hash: NonEmptyString
    output_head_hash: NonEmptyString
    tokenizer_hash: NonEmptyString
    special_tokens_hash: NonEmptyString
    positional_encoding: NonEmptyString
    rope_fingerprint: NonEmptyString
    attention_mask_semantics: NonEmptyString
    sliding_window: int | None = Field(default=None, gt=0)
    layer_count: int = Field(gt=0)
    head_count: int = Field(gt=0)
    kv_head_count: int = Field(gt=0)
    head_dim: int = Field(gt=0)
    recurrent_update_fingerprint: NonEmptyString | None = None
    adapter_hash: NonEmptyString | None = None
    state_dtype: NonEmptyString = "float32"
    quantization: NonEmptyString = "none"
    sampler_algorithm: NonEmptyString = "counter_rng_v1"

    @model_validator(mode="after")
    def _validate_heads(self) -> ModelSemantics:
        if self.head_count % self.kv_head_count != 0:
            raise ValueError("head_count must be divisible by kv_head_count")
        return self


class RuntimeCapabilities(CompatibilityModel):
    runtime_name: NonEmptyString
    runtime_version: NonEmptyString
    adapter_version: NonEmptyString
    supported_state_types: tuple[NonEmptyString, ...]
    supported_dtypes: tuple[NonEmptyString, ...]
    supported_quantizations: tuple[NonEmptyString, ...] = ("none",)
    can_recompute_from_token_history: bool = False
    logical_state_contract: NonEmptyString = "sloforge-continuum-logical-state-v1"

    @model_validator(mode="after")
    def _validate_capabilities(self) -> RuntimeCapabilities:
        for name, values in (
            ("supported_state_types", self.supported_state_types),
            ("supported_dtypes", self.supported_dtypes),
            ("supported_quantizations", self.supported_quantizations),
        ):
            if not values:
                raise ValueError(f"{name} must not be empty")
            if len(set(values)) != len(values):
                raise ValueError(f"{name} contains duplicates")
        return self


class StateDependencyEvidence(CompatibilityModel):
    """Version-scoped proof about which model changes can affect stored state."""

    dependency_graph_hash: NonEmptyString
    changed_components: tuple[NonEmptyString, ...]
    state_producing_components: tuple[NonEmptyString, ...]
    affected_state_components: tuple[NonEmptyString, ...]
    recomputable_state_components: tuple[NonEmptyString, ...] = ()
    output_head_is_state_sink: bool = False
    token_history_available: bool = False

    @property
    def output_head_only(self) -> bool:
        return bool(self.changed_components) and set(self.changed_components) == {"output_head"}

    @model_validator(mode="after")
    def _validate_component_sets(self) -> StateDependencyEvidence:
        for name, values in (
            ("changed_components", self.changed_components),
            ("state_producing_components", self.state_producing_components),
            ("affected_state_components", self.affected_state_components),
            ("recomputable_state_components", self.recomputable_state_components),
        ):
            if len(set(values)) != len(values):
                raise ValueError(f"{name} contains duplicates")
        if not set(self.recomputable_state_components).issubset(self.affected_state_components):
            raise ValueError("recomputable state must be a subset of affected state")
        return self


class QualityEvidence(CompatibilityModel):
    metric: NonEmptyString
    observed_loss: NonNegativeFloat
    maximum_loss: NonNegativeFloat
    artifact_hash: NonEmptyString
    sample_count: int = Field(gt=0)

    @property
    def within_budget(self) -> bool:
        return self.observed_loss <= self.maximum_loss


class CompatibilityRequest(CompatibilityModel):
    schema_version: Literal["1.0.0"] = "1.0.0"
    source: ModelSemantics
    destination: ModelSemantics
    source_runtime: RuntimeCapabilities
    destination_runtime: RuntimeCapabilities
    source_layout_fingerprint: NonEmptyString
    destination_layout_fingerprint: NonEmptyString
    required_state_types: tuple[NonEmptyString, ...]
    required_exactness: ExactnessClass
    dependency_evidence: StateDependencyEvidence | None = None
    quality_evidence: QualityEvidence | None = None
    allow_recomputation: bool = False
    numeric_tolerance: NonNegativeFloat = 1e-5

    @model_validator(mode="after")
    def _validate_required_state(self) -> CompatibilityRequest:
        if not self.required_state_types:
            raise ValueError("required_state_types must not be empty")
        if len(set(self.required_state_types)) != len(self.required_state_types):
            raise ValueError("required_state_types contains duplicates")
        if self.dependency_evidence is not None:
            evidence_state = set(self.dependency_evidence.affected_state_components)
            if not evidence_state.issubset(self.required_state_types):
                raise ValueError("dependency evidence names state outside the required state set")
        return self


class CompatibilityReason(CompatibilityModel):
    code: NonEmptyString
    severity: ReasonSeverity
    component: NonEmptyString
    message: NonEmptyString
    evidence: tuple[NonEmptyString, ...] = ()


class VerificationObligation(CompatibilityModel):
    obligation_id: NonEmptyString
    kind: VerificationKind
    component: NonEmptyString
    method: NonEmptyString
    tolerance: NonNegativeFloat | None = None


class RejectedCompatibilityClass(CompatibilityModel):
    exactness_class: ExactnessClass
    reason_codes: tuple[NonEmptyString, ...]


class RecomputationRequirement(CompatibilityModel):
    state_components: tuple[NonEmptyString, ...]
    source: Literal["token_history", "checkpoint"]
    dependency_graph_hash: NonEmptyString


class CompatibilityDecision(CompatibilityModel):
    """Engine-internal decision; serialized artifacts use ``ir.CompatibilityReport``."""

    schema_version: Literal["1.0.0"] = "1.0.0"
    compatibility_class: ExactnessClass
    safe: bool
    reasons: tuple[CompatibilityReason, ...]
    rejected_classes: tuple[RejectedCompatibilityClass, ...]
    required_conversion: tuple[NonEmptyString, ...]
    required_recomputation: tuple[RecomputationRequirement, ...]
    unsupported_state: tuple[NonEmptyString, ...]
    quality_implications: tuple[NonEmptyString, ...]
    verification_obligations: tuple[VerificationObligation, ...]
    migration_restrictions: tuple[NonEmptyString, ...]

    @model_validator(mode="after")
    def _validate_safe_class(self) -> CompatibilityDecision:
        if self.safe == (self.compatibility_class is ExactnessClass.INCOMPATIBLE):
            raise ValueError("safe must be false exactly when compatibility_class is incompatible")
        if (
            self.compatibility_class is ExactnessClass.QUALITY_BOUNDED
            and not self.quality_implications
        ):
            raise ValueError("quality-bounded reports require explicit quality implications")
        return self
