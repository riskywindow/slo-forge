"""Versioned StateTransformationIR and bounded-memory chunk schedules."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from sloforge.continuum.compatibility import ExactnessClass

NonEmptyString = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
NonNegativeInt = Annotated[int, Field(ge=0)]
PositiveInt = Annotated[int, Field(gt=0)]
NonNegativeFloat = Annotated[float, Field(ge=0.0, allow_inf_nan=False)]


class TransformationModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, validate_default=True)


class OperationCode(StrEnum):
    SLICE = "slice"
    CONCATENATE = "concatenate"
    SPLIT = "split"
    RESHAPE = "reshape"
    PERMUTE = "permute"
    TRANSPOSE = "transpose"
    PAD = "pad"
    UNPAD = "unpad"
    INTERLEAVE = "interleave"
    DEINTERLEAVE = "deinterleave"
    PACK = "pack"
    UNPACK = "unpack"
    SHARD = "shard"
    RESHARD = "reshard"
    REPLICATE = "replicate"
    GATHER = "gather"
    SCATTER = "scatter"
    PAGE_REMAP = "page_remap"
    PAGE_COALESCE = "page_coalesce"
    PAGE_SPLIT = "page_split"
    DTYPE_CONVERT = "dtype_convert"
    QUANTIZE = "quantize"
    DEQUANTIZE = "dequantize"
    COMPRESS = "compress"
    DECOMPRESS = "decompress"
    CHECKSUM = "checksum"
    ENCRYPT = "encrypt"
    DECRYPT = "decrypt"
    COPY = "copy"
    ZERO_FILL = "zero_fill"
    RECONSTRUCT_METADATA = "reconstruct_metadata"
    RECOMPUTE = "recompute"
    SEND = "send"
    RECEIVE = "receive"
    WRITE_DESTINATION = "write_destination"
    VALIDATE = "validate"


class OwnershipBehavior(StrEnum):
    READ_ONLY = "read_only"
    PRODUCES_CANDIDATE = "produces_candidate"
    TRANSFERS_CHUNK = "transfers_chunk"
    WRITES_DESTINATION = "writes_destination"
    VALIDATES_ONLY = "validates_only"


class TargetDevice(StrEnum):
    SOURCE_CPU = "source_cpu"
    SOURCE_GPU = "source_gpu"
    DESTINATION_CPU = "destination_cpu"
    DESTINATION_GPU = "destination_gpu"
    TRANSPORT = "transport"


class ParameterValue(TransformationModel):
    """Typed operation attribute; avoids an untyped extension dictionary."""

    key: NonEmptyString
    string_value: str | None = None
    integer_value: int | None = None
    float_value: float | None = Field(default=None, allow_inf_nan=False)
    boolean_value: bool | None = None
    integer_list: tuple[int, ...] | None = None

    @model_validator(mode="after")
    def _one_value(self) -> Self:
        values = (
            self.string_value,
            self.integer_value,
            self.float_value,
            self.boolean_value,
            self.integer_list,
        )
        if sum(value is not None for value in values) != 1:
            raise ValueError("a parameter must contain exactly one typed value")
        return self


class TensorContract(TransformationModel):
    state_id: NonEmptyString
    shape: tuple[NonNegativeInt, ...]
    dtype: NonEmptyString
    layout: NonEmptyString
    exactness: ExactnessClass


class ShapeTransformation(TransformationModel):
    source_shape: tuple[NonNegativeInt, ...]
    destination_shape: tuple[NonNegativeInt, ...]


class TransformationOperation(TransformationModel):
    operation_id: NonEmptyString
    code: OperationCode
    depends_on: tuple[NonEmptyString, ...] = ()
    inputs: tuple[NonEmptyString, ...]
    outputs: tuple[NonEmptyString, ...]
    source_contract: TensorContract
    destination_contract: TensorContract
    preconditions: tuple[NonEmptyString, ...]
    postconditions: tuple[NonEmptyString, ...]
    exactness: ExactnessClass
    shape_transformation: ShapeTransformation
    source_dtype: NonEmptyString
    destination_dtype: NonEmptyString
    ownership: OwnershipBehavior
    target_device: TargetDevice
    estimated_cost_ns: NonNegativeInt
    memory_requirement_bytes: NonNegativeInt
    streamable: bool
    verification_obligation: NonEmptyString
    fallback_implementation: NonEmptyString
    parameters: tuple[ParameterValue, ...] = ()


class TransformationDAG(TransformationModel):
    operations: tuple[TransformationOperation, ...]

    @model_validator(mode="after")
    def _validate_dag(self) -> Self:
        ids = [operation.operation_id for operation in self.operations]
        if len(ids) != len(set(ids)):
            raise ValueError("operation IDs must be unique")
        known = set(ids)
        for operation in self.operations:
            unknown = set(operation.depends_on) - known
            if unknown:
                raise ValueError(f"unknown operation dependencies: {sorted(unknown)}")
            if operation.operation_id in operation.depends_on:
                raise ValueError("an operation cannot depend on itself")

        pending = {
            operation.operation_id: set(operation.depends_on) for operation in self.operations
        }
        completed: set[str] = set()
        while pending:
            ready = sorted(
                operation_id for operation_id, deps in pending.items() if deps <= completed
            )
            if not ready:
                raise ValueError("transformation operations contain a cycle")
            completed.update(ready)
            for operation_id in ready:
                del pending[operation_id]
        return self

    def topological_order(self) -> tuple[str, ...]:
        pending = {
            operation.operation_id: set(operation.depends_on) for operation in self.operations
        }
        completed: set[str] = set()
        ordered: list[str] = []
        while pending:
            ready = sorted(
                operation_id for operation_id, deps in pending.items() if deps <= completed
            )
            ordered.extend(ready)
            completed.update(ready)
            for operation_id in ready:
                del pending[operation_id]
        return tuple(ordered)


class ChunkAssignment(TransformationModel):
    chunk_id: NonEmptyString
    destination_rank: NonNegativeInt
    token_start: NonNegativeInt
    token_end: NonNegativeInt
    head_start: NonNegativeInt
    head_end: NonNegativeInt
    estimated_temporary_bytes: NonNegativeInt
    depends_on: tuple[NonEmptyString, ...] = ()

    @model_validator(mode="after")
    def _validate_ranges(self) -> Self:
        if self.token_end <= self.token_start:
            raise ValueError("chunk token range must be non-empty")
        if self.head_end <= self.head_start:
            raise ValueError("chunk head range must be non-empty")
        return self


class ChunkSchedule(TransformationModel):
    chunks: tuple[ChunkAssignment, ...]
    maximum_temporary_bytes: PositiveInt
    chunk_token_count: PositiveInt
    bounded_buffer_count: PositiveInt = 2

    @model_validator(mode="after")
    def _validate_memory_bound(self) -> Self:
        if any(
            chunk.estimated_temporary_bytes > self.maximum_temporary_bytes for chunk in self.chunks
        ):
            raise ValueError("chunk exceeds the declared temporary-memory bound")
        return self


class MemoryPlan(TransformationModel):
    destination_allocation_bytes: NonNegativeInt
    maximum_temporary_bytes: PositiveInt
    canonical_fallback_temporary_bytes: NonNegativeInt
    bounded_buffers: PositiveInt


class StateTransformationIR(TransformationModel):
    schema_version: Literal["1.0.0"] = "1.0.0"
    program_id: NonEmptyString
    source_layout_hash: NonEmptyString
    destination_layout_hash: NonEmptyString
    exactness: ExactnessClass
    dag: TransformationDAG
    chunk_schedule: ChunkSchedule
    memory_plan: MemoryPlan
    direct_conversion: bool
    canonical_fallback: NonEmptyString
    predicted_duration_ns: NonNegativeInt
    prediction_basis: Literal["measured_throughput", "unmeasured"]
