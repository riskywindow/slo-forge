"""Typed contracts and inspection evidence for zero-day reference packages."""

from __future__ import annotations

import math
from enum import StrEnum
from pathlib import PurePosixPath
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

REFERENCE_PACKAGE_SCHEMA_VERSION: Literal["1.0.0"] = "1.0.0"
INSPECTION_SCHEMA_VERSION: Literal["1.0.0"] = "1.0.0"

NonEmpty = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
Identifier = Annotated[str, StringConstraints(pattern=r"^[a-z][a-z0-9_.-]*$")]
Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
PositiveInt = Annotated[int, Field(gt=0)]
NonNegativeInt = Annotated[int, Field(ge=0)]


class FrontendModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, validate_default=True)


class DimensionContract(FrontendModel):
    """One exact or symbolic tensor dimension with a finite supported domain."""

    name: Identifier
    minimum: PositiveInt
    maximum: PositiveInt
    multiple_of: PositiveInt = 1

    @model_validator(mode="after")
    def validate_bounds(self) -> Self:
        if self.maximum < self.minimum:
            raise ValueError("dimension maximum cannot be less than minimum")
        if self.minimum % self.multiple_of != 0 or self.maximum % self.multiple_of != 0:
            raise ValueError("dimension endpoints must be divisible by multiple_of")
        return self


class TensorDomain(FrontendModel):
    name: Identifier
    dtype: NonEmpty
    dimensions: tuple[DimensionContract, ...]
    contiguous: bool | None = None
    allowed_strides: tuple[tuple[PositiveInt, ...], ...] = ()

    @model_validator(mode="after")
    def validate_strides(self) -> Self:
        if not self.dimensions:
            raise ValueError("tensor domain must have at least one dimension")
        rank = len(self.dimensions)
        if any(len(strides) != rank for strides in self.allowed_strides):
            raise ValueError("every allowed stride vector must match tensor rank")
        return self


class ScalarDomain(FrontendModel):
    name: Identifier
    kind: Literal["integer", "float", "boolean", "string"]
    minimum: float | None = None
    maximum: float | None = None
    allowed_values: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_domain(self) -> Self:
        if any(
            value is not None and not math.isfinite(value) for value in (self.minimum, self.maximum)
        ):
            raise ValueError("scalar bounds must be finite")
        if self.minimum is not None and self.maximum is not None and self.maximum < self.minimum:
            raise ValueError("scalar maximum cannot be less than minimum")
        return self


class SupportedInputDomain(FrontendModel):
    tensors: tuple[TensorDomain, ...] = ()
    scalars: tuple[ScalarDomain, ...] = ()
    maximum_prompt_tokens: PositiveInt
    maximum_generated_tokens: PositiveInt


class StateFieldContract(FrontendModel):
    field_id: Identifier
    kind: Literal["kv", "recurrent", "convolutional", "speculative", "custom", "workflow"]
    dtype: NonEmpty
    shape: tuple[DimensionContract, ...]
    mutable: bool
    persistent_across_tokens: bool
    reset_at_request_boundary: bool
    alias_group: Identifier | None = None
    quantization: NonEmpty | None = None


class StateContract(FrontendModel):
    ownership: Literal["request", "session", "replicated_read_only"]
    fields: tuple[StateFieldContract, ...]
    mutation_atomicity: Literal["per_token", "per_chunk", "immutable"]
    cancellation_releases_state: bool
    migration_supported: bool

    @model_validator(mode="after")
    def unique_fields(self) -> Self:
        identifiers = [field.field_id for field in self.fields]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("state field identifiers must be unique")
        if self.ownership != "replicated_read_only" and not self.cancellation_releases_state:
            raise ValueError("mutable request/session state must be released on cancellation")
        return self


class SemanticContract(FrontendModel):
    token_commitment: Literal["on_emit", "on_acknowledgement"]
    deterministic_for_seed: bool
    batching_axes: tuple[Identifier, ...]
    batch_isolation: Literal["independent_requests", "shared_prefix_read_only"]
    streaming_order: Literal["strict_token_order"]
    cancellation: Literal["immediate_before_commit", "request_boundary"]
    retry_after_first_token: Literal["forbidden", "idempotent_continuation"]
    allowed_control_flow: tuple[NonEmpty, ...] = ()
    required_invariants: tuple[NonEmpty, ...]


class QualityMetricContract(FrontendModel):
    metric: Literal[
        "exact_token_match",
        "top1_agreement",
        "topk_agreement",
        "maximum_absolute_error",
        "maximum_relative_error",
        "kl_divergence",
        "sequence_agreement",
    ]
    threshold: float
    comparison: Literal["at_least", "at_most", "exact"]

    @model_validator(mode="after")
    def finite_threshold(self) -> Self:
        if not math.isfinite(self.threshold):
            raise ValueError("quality metric threshold must be finite")
        return self


class QualityContract(FrontendModel):
    metrics: tuple[QualityMetricContract, ...]
    final_evaluation_corpus: NonEmpty
    search_corpus: NonEmpty
    permit_approximation: bool

    @model_validator(mode="after")
    def separate_corpora(self) -> Self:
        if self.final_evaluation_corpus == self.search_corpus:
            raise ValueError("search and final quality corpora must be distinct")
        return self


class CustomOperatorContract(FrontendModel):
    operator_id: Identifier
    symbol: NonEmpty
    semantic_description: NonEmpty
    exact: bool
    input_domain: tuple[Identifier, ...]
    state_reads: tuple[Identifier, ...] = ()
    state_writes: tuple[Identifier, ...] = ()
    verification_obligations: tuple[NonEmpty, ...]


class WorkflowStepContract(FrontendModel):
    step_id: Identifier
    kind: Literal["model", "tool", "verification", "branch", "loop"]
    dependencies: tuple[Identifier, ...] = ()
    deadline_ms: PositiveInt | None = None
    expected_latency_ms: PositiveInt | None = None


class WorkflowContract(FrontendModel):
    steps: tuple[WorkflowStepContract, ...]

    @model_validator(mode="after")
    def validate_dag_references(self) -> Self:
        ordered_identifiers = [step.step_id for step in self.steps]
        identifiers = set(ordered_identifiers)
        if len(identifiers) != len(ordered_identifiers):
            raise ValueError("workflow step identifiers must be unique")
        dependencies = {step.step_id: step.dependencies for step in self.steps}
        for step in self.steps:
            unknown = set(step.dependencies) - identifiers
            if unknown:
                raise ValueError(
                    f"workflow step {step.step_id} has unknown dependencies: {unknown}"
                )
            if len(step.dependencies) != len(set(step.dependencies)):
                raise ValueError(f"workflow step {step.step_id} dependencies must be unique")

        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(identifier: str) -> None:
            if identifier in visiting:
                raise ValueError("workflow contract must be acyclic")
            if identifier in visited:
                return
            visiting.add(identifier)
            for dependency in dependencies[identifier]:
                visit(dependency)
            visiting.remove(identifier)
            visited.add(identifier)

        for identifier in ordered_identifiers:
            visit(identifier)
        return self


class EntryPointContract(FrontendModel):
    load_model: NonEmpty
    allocate_state: NonEmpty
    prefill: NonEmpty
    decode_step: NonEmpty
    sample: NonEmpty
    tokenize: NonEmpty
    detokenize: NonEmpty
    sample_inputs: NonEmpty
    torch_export: NonEmpty | None = None


class ReferencePackageManifest(FrontendModel):
    schema_version: Literal["1.0.0"] = REFERENCE_PACKAGE_SCHEMA_VERSION
    package_id: Identifier
    reference_module: NonEmpty
    tokenizer_module: NonEmpty
    sample_generator_module: NonEmpty
    sample_corpus: NonEmpty
    auxiliary_modules: tuple[NonEmpty, ...] = ()
    entry_points: EntryPointContract
    state_contract: StateContract
    semantic_contract: SemanticContract
    quality_contract: QualityContract
    supported_input_domain: SupportedInputDomain
    custom_operators: tuple[CustomOperatorContract, ...] = ()
    workflow: WorkflowContract | None = None
    software_preconditions: tuple[NonEmpty, ...] = ()

    @model_validator(mode="after")
    def validate_relative_paths_and_references(self) -> Self:
        for value in (
            self.reference_module,
            self.tokenizer_module,
            self.sample_generator_module,
            self.sample_corpus,
            *self.auxiliary_modules,
            self.quality_contract.final_evaluation_corpus,
            self.quality_contract.search_corpus,
        ):
            path = PurePosixPath(value)
            if path.is_absolute() or any(part in ("", ".", "..") for part in path.parts):
                raise ValueError(f"package path must be normalized and relative: {value!r}")
        if len(self.auxiliary_modules) != len(set(self.auxiliary_modules)):
            raise ValueError("auxiliary module paths must be unique")
        if any(not value.endswith(".py") for value in self.auxiliary_modules):
            raise ValueError("auxiliary modules must be Python source files")
        fields = {field.field_id for field in self.state_contract.fields}
        for operator in self.custom_operators:
            unknown = (set(operator.state_reads) | set(operator.state_writes)) - fields
            if unknown:
                raise ValueError(
                    f"custom operator {operator.operator_id} references unknown state: {unknown}"
                )
        dimensions = {
            dimension.name
            for tensor in self.supported_input_domain.tensors
            for dimension in tensor.dimensions
        }
        unknown_batching_axes = set(self.semantic_contract.batching_axes) - dimensions
        if unknown_batching_axes:
            raise ValueError(
                "semantic batching axes must reference declared tensor dimensions: "
                f"{sorted(unknown_batching_axes)}"
            )
        if len(self.semantic_contract.batching_axes) != len(
            set(self.semantic_contract.batching_axes)
        ):
            raise ValueError("semantic batching axes must be unique")
        return self


class DiagnosticSeverity(StrEnum):
    INFO = "info"
    OBLIGATION = "obligation"
    UNSUPPORTED = "unsupported"


class SourceLocation(FrontendModel):
    relative_path: NonEmpty
    line: PositiveInt
    column: NonNegativeInt


class InspectionDiagnostic(FrontendModel):
    diagnostic_id: Identifier
    severity: DiagnosticSeverity
    category: Literal[
        "unknown_semantics",
        "dynamic_control_flow",
        "aliasing",
        "custom_operator",
        "torch_export",
        "contract",
    ]
    message: NonEmpty
    location: SourceLocation | None = None
    proof_obligation: NonEmpty | None = None


class RecoveredOperator(FrontendModel):
    operator_id: Identifier
    symbol: NonEmpty
    category: Literal[
        "tensor",
        "state",
        "control",
        "sampling",
        "custom",
        "python",
        "unknown",
    ]
    inputs: tuple[NonEmpty, ...] = ()
    outputs: tuple[NonEmpty, ...] = ()
    state_reads: tuple[Identifier, ...] = ()
    state_writes: tuple[Identifier, ...] = ()
    location: SourceLocation
    custom_operator_id: Identifier | None = None


class RecoveredControlFlow(FrontendModel):
    kind: Literal["if", "for", "while", "match", "try"]
    location: SourceLocation
    declared_semantics: bool


class RecoveredAlias(FrontendModel):
    source: NonEmpty
    target: NonEmpty
    location: SourceLocation
    explicit_contract: bool


class RecoveredGraph(FrontendModel):
    operators: tuple[RecoveredOperator, ...]
    input_tensors: tuple[TensorDomain, ...]
    symbolic_dimensions: tuple[DimensionContract, ...]
    state_fields: tuple[StateFieldContract, ...]
    state_dependencies: tuple[NonEmpty, ...]
    legal_batching_axes: tuple[Identifier, ...]
    aliases: tuple[RecoveredAlias, ...]
    control_flow: tuple[RecoveredControlFlow, ...]
    custom_operator_ids: tuple[Identifier, ...]


class TorchExportEvidence(FrontendModel):
    torch_version: NonEmpty
    graph_nodes: tuple[NonEmpty, ...]
    tensor_metadata: tuple[NonEmpty, ...]
    range_constraints: tuple[NonEmpty, ...]


class InspectionResult(FrontendModel):
    schema_version: Literal["1.0.0"] = INSPECTION_SCHEMA_VERSION
    package_id: Identifier
    package_hash: Sha256
    manifest_hash: Sha256
    source_hashes: tuple[tuple[NonEmpty, Sha256], ...]
    graph: RecoveredGraph
    semantic_contract: SemanticContract
    quality_contract: QualityContract
    supported_input_domain: SupportedInputDomain
    diagnostics: tuple[InspectionDiagnostic, ...]
    torch_export: TorchExportEvidence | None = None

    @property
    def has_unsupported_behavior(self) -> bool:
        return any(item.severity == DiagnosticSeverity.UNSUPPORTED for item in self.diagnostics)
