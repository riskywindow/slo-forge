"""CPU reference, direct, and streaming state-conversion compiler."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from sloforge.continuum.compatibility import ExactnessClass

from .ir import (
    ChunkAssignment,
    ChunkSchedule,
    MemoryPlan,
    OperationCode,
    OwnershipBehavior,
    ParameterValue,
    ShapeTransformation,
    StateTransformationIR,
    TargetDevice,
    TensorContract,
    TransformationDAG,
    TransformationOperation,
)
from .layouts import (
    KVLayout,
    KVLayoutKind,
    KVShard,
    LayoutValidationError,
    PhysicalKVState,
    allocate_state,
    decode_logical,
    encode_logical,
)

FloatArray = NDArray[np.floating]


class ConversionCompilationError(ValueError):
    """Raised when no legal bounded-memory conversion can be compiled."""


@dataclass(frozen=True)
class ConvertedChunk:
    chunk_id: str
    destination_rank: int
    token_start: int
    token_end: int
    head_start: int
    head_end: int
    key: FloatArray | None = None
    value: FloatArray | None = None
    packed: FloatArray | None = None

    @property
    def temporary_nbytes(self) -> int:
        return sum(
            array.nbytes for array in (self.key, self.value, self.packed) if array is not None
        )


def _validate_logical_compatibility(source: KVLayout, destination: KVLayout) -> None:
    source_semantics = (
        source.layer_count,
        source.token_count,
        source.kv_head_count,
        source.head_dim,
    )
    destination_semantics = (
        destination.layer_count,
        destination.token_count,
        destination.kv_head_count,
        destination.head_dim,
    )
    if source_semantics != destination_semantics:
        raise ConversionCompilationError(
            "layout conversion cannot change logical layers, tokens, KV heads, or head dimension"
        )
    _conversion_exactness(source, destination)


def _conversion_exactness(source: KVLayout, destination: KVLayout) -> ExactnessClass:
    """Classify only conversions implemented by the trusted CPU backend.

    Narrowing float conversions remain numerical. Integer narrowing and
    integer/float conversions can lose arbitrary state and therefore require a
    quality-bounded backend that this converter does not implement.
    """

    source_dtype = np.dtype(source.dtype)
    destination_dtype = np.dtype(destination.dtype)
    if source_dtype == destination_dtype:
        return ExactnessClass.EXACT_SEMANTIC
    if np.can_cast(source_dtype, destination_dtype, casting="safe"):
        return ExactnessClass.EXACT_SEMANTIC
    if source_dtype.kind == "f" and destination_dtype.kind == "f":
        return ExactnessClass.NUMERICALLY_EQUIVALENT
    raise ConversionCompilationError(
        "potentially lossy integer/float dtype conversion requires a quality-bounded backend"
    )


def _temporary_bytes_per_token(source: KVLayout, destination: KVLayout) -> int:
    widest_dtype = max(np.dtype(source.dtype).itemsize, np.dtype(destination.dtype).itemsize)
    return source.layer_count * source.kv_head_count * source.head_dim * 2 * widest_dtype


def _chunk_token_count(
    source: KVLayout,
    destination: KVLayout,
    maximum_temporary_bytes: int,
) -> int:
    if maximum_temporary_bytes <= 0:
        raise ConversionCompilationError("maximum_temporary_bytes must be positive")
    per_token = _temporary_bytes_per_token(source, destination)
    if maximum_temporary_bytes < per_token:
        raise ConversionCompilationError(
            f"memory bound {maximum_temporary_bytes} cannot hold one logical token ({per_token} bytes)"
        )
    if source.token_count == 0:
        return 1
    return min(source.token_count, max(1, maximum_temporary_bytes // per_token))


def _contract(
    layout: KVLayout,
    layout_name: str,
    exactness: ExactnessClass,
) -> TensorContract:
    return TensorContract(
        state_id="attention.kv",
        shape=layout.logical_shape,
        dtype=layout.dtype,
        layout=layout_name,
        exactness=exactness,
    )


def compile_conversion(
    source: KVLayout,
    destination: KVLayout,
    *,
    maximum_temporary_bytes: int,
    measured_throughput_bytes_s: float | None = None,
) -> StateTransformationIR:
    """Compile a direct streaming reshard/page-remap/pack conversion DAG."""

    _validate_logical_compatibility(source, destination)
    chunk_tokens = _chunk_token_count(source, destination, maximum_temporary_bytes)
    exactness = _conversion_exactness(source, destination)
    source_contract = _contract(source, source.kind.value, exactness)
    destination_contract = _contract(destination, destination.kind.value, exactness)
    shape = ShapeTransformation(
        source_shape=source_contract.shape,
        destination_shape=destination_contract.shape,
    )
    estimated = (
        math.ceil(source.logical_nbytes / measured_throughput_bytes_s * 1_000_000_000)
        if measured_throughput_bytes_s is not None and measured_throughput_bytes_s > 0
        else 0
    )

    specs = (
        (
            "read-reshard",
            OperationCode.RESHARD,
            (),
            OwnershipBehavior.READ_ONLY,
            "read only the destination head slice from source TP shards",
        ),
        (
            "layout-pack",
            OperationCode.PACK
            if destination.kind is KVLayoutKind.HEAD_MAJOR_PACKED
            else OperationCode.UNPACK,
            ("read-reshard",),
            OwnershipBehavior.PRODUCES_CANDIDATE,
            "transpose token/head axes and convert separate/packed K/V",
        ),
        (
            "page-remap",
            OperationCode.PAGE_REMAP,
            ("layout-pack",),
            OwnershipBehavior.PRODUCES_CANDIDATE,
            "map logical token ranges to destination page boundaries",
        ),
        (
            "write-destination",
            OperationCode.WRITE_DESTINATION,
            ("page-remap",),
            OwnershipBehavior.WRITES_DESTINATION,
            "write candidate bytes without transferring ownership",
        ),
        (
            "checksum",
            OperationCode.CHECKSUM,
            ("write-destination",),
            OwnershipBehavior.VALIDATES_ONLY,
            "hash destination metadata and state bytes",
        ),
        (
            "validate",
            OperationCode.VALIDATE,
            ("checksum",),
            OwnershipBehavior.VALIDATES_ONLY,
            "compare direct output with the trusted canonical converter",
        ),
    )
    operations = tuple(
        TransformationOperation(
            operation_id=operation_id,
            code=code,
            depends_on=depends_on,
            inputs=("attention.kv",),
            outputs=(f"attention.kv.{operation_id}",),
            source_contract=source_contract,
            destination_contract=destination_contract,
            preconditions=("source checksum verified", "logical shapes match"),
            postconditions=(postcondition,),
            exactness=exactness,
            shape_transformation=shape,
            source_dtype=source.dtype,
            destination_dtype=destination.dtype,
            ownership=ownership,
            target_device=TargetDevice.DESTINATION_CPU,
            estimated_cost_ns=estimated // len(specs),
            memory_requirement_bytes=maximum_temporary_bytes,
            streamable=True,
            verification_obligation="trusted canonical equivalence before activation",
            fallback_implementation="canonical_cpu_v1",
            parameters=(
                ParameterValue(key="chunk_tokens", integer_value=chunk_tokens),
                ParameterValue(
                    key="destination_tp", integer_value=destination.tensor_parallel_degree
                ),
                ParameterValue(
                    key="destination_page_size", integer_value=destination.page_size_tokens
                ),
            ),
        )
        for operation_id, code, depends_on, ownership, postcondition in specs
    )

    chunks: list[ChunkAssignment] = []
    previous_by_rank: dict[int, str] = {}
    for token_start in range(0, source.token_count, chunk_tokens):
        token_end = min(source.token_count, token_start + chunk_tokens)
        for rank in range(destination.tensor_parallel_degree):
            head_start = rank * destination.heads_per_shard
            head_end = head_start + destination.heads_per_shard
            chunk_id = f"rank-{rank}-tokens-{token_start}-{token_end}"
            dependencies = (previous_by_rank[rank],) if rank in previous_by_rank else ()
            actual_bytes = (
                destination.layer_count
                * (token_end - token_start)
                * destination.heads_per_shard
                * destination.head_dim
                * 2
                * np.dtype(destination.dtype).itemsize
            )
            chunks.append(
                ChunkAssignment(
                    chunk_id=chunk_id,
                    destination_rank=rank,
                    token_start=token_start,
                    token_end=token_end,
                    head_start=head_start,
                    head_end=head_end,
                    estimated_temporary_bytes=actual_bytes,
                    depends_on=dependencies,
                )
            )
            previous_by_rank[rank] = chunk_id

    # Empty KV state has no data chunks, but remains a legal conversion.
    schedule = ChunkSchedule(
        chunks=tuple(chunks),
        maximum_temporary_bytes=maximum_temporary_bytes,
        chunk_token_count=chunk_tokens,
        bounded_buffer_count=1,
    )
    program_hash = hashlib.sha256(
        f"{source.fingerprint()}:{destination.fingerprint()}:{chunk_tokens}".encode("ascii")
    ).hexdigest()[:24]
    return StateTransformationIR(
        program_id=f"continuum-kv-{program_hash}",
        source_layout_hash=source.fingerprint(),
        destination_layout_hash=destination.fingerprint(),
        exactness=exactness,
        dag=TransformationDAG(operations=operations),
        chunk_schedule=schedule,
        memory_plan=MemoryPlan(
            destination_allocation_bytes=destination.physical_nbytes,
            maximum_temporary_bytes=maximum_temporary_bytes,
            canonical_fallback_temporary_bytes=source.logical_nbytes,
            bounded_buffers=1,
        ),
        direct_conversion=True,
        canonical_fallback="canonical_cpu_v1",
        predicted_duration_ns=estimated,
        prediction_basis=(
            "measured_throughput" if measured_throughput_bytes_s is not None else "unmeasured"
        ),
    )


def canonical_convert(source: PhysicalKVState, destination: KVLayout) -> PhysicalKVState:
    """Correctness-first conversion through complete canonical K and V tensors."""

    _validate_logical_compatibility(source.layout, destination)
    key, value = decode_logical(source)
    return encode_logical(key, value, destination)


def _read_logical_chunk(
    source: PhysicalKVState,
    *,
    token_start: int,
    token_end: int,
    head_start: int,
    head_end: int,
    dtype: np.dtype[np.floating],
) -> tuple[FloatArray, FloatArray]:
    layout = source.layout
    shape = (
        layout.layer_count,
        token_end - token_start,
        head_end - head_start,
        layout.head_dim,
    )
    key = np.empty(shape, dtype=dtype)
    value = np.empty(shape, dtype=dtype)
    covered = np.zeros(head_end - head_start, dtype=np.bool_)
    for shard in source.shards:
        overlap_start = max(head_start, shard.head_start)
        overlap_end = min(head_end, shard.head_end)
        if overlap_start >= overlap_end:
            continue
        source_head_start = overlap_start - shard.head_start
        source_head_end = overlap_end - shard.head_start
        destination_head_start = overlap_start - head_start
        destination_head_end = overlap_end - head_start
        if layout.kind is KVLayoutKind.TOKEN_MAJOR_SEPARATE:
            assert shard.key is not None and shard.value is not None
            key[:, :, destination_head_start:destination_head_end, :] = shard.key[
                :, token_start:token_end, source_head_start:source_head_end, :
            ]
            value[:, :, destination_head_start:destination_head_end, :] = shard.value[
                :, token_start:token_end, source_head_start:source_head_end, :
            ]
        else:
            assert shard.packed is not None
            key[:, :, destination_head_start:destination_head_end, :] = shard.packed[
                :, source_head_start:source_head_end, token_start:token_end, 0, :
            ].transpose(0, 2, 1, 3)
            value[:, :, destination_head_start:destination_head_end, :] = shard.packed[
                :, source_head_start:source_head_end, token_start:token_end, 1, :
            ].transpose(0, 2, 1, 3)
        covered[destination_head_start:destination_head_end] = True
    if not np.all(covered):
        raise LayoutValidationError("source shards do not cover the requested logical head range")
    return key, value


def _read_packed_chunk(
    source: PhysicalKVState,
    *,
    token_start: int,
    token_end: int,
    head_start: int,
    head_end: int,
    dtype: np.dtype[np.floating],
) -> FloatArray:
    """Read shard intersections directly into packed output with one bounded buffer."""

    layout = source.layout
    packed = np.empty(
        (
            layout.layer_count,
            head_end - head_start,
            token_end - token_start,
            2,
            layout.head_dim,
        ),
        dtype=dtype,
    )
    covered = np.zeros(head_end - head_start, dtype=np.bool_)
    for shard in source.shards:
        overlap_start = max(head_start, shard.head_start)
        overlap_end = min(head_end, shard.head_end)
        if overlap_start >= overlap_end:
            continue
        source_start = overlap_start - shard.head_start
        source_end = overlap_end - shard.head_start
        destination_start = overlap_start - head_start
        destination_end = overlap_end - head_start
        if layout.kind is KVLayoutKind.TOKEN_MAJOR_SEPARATE:
            assert shard.key is not None and shard.value is not None
            packed[:, destination_start:destination_end, :, 0, :] = shard.key[
                :, token_start:token_end, source_start:source_end, :
            ].transpose(0, 2, 1, 3)
            packed[:, destination_start:destination_end, :, 1, :] = shard.value[
                :, token_start:token_end, source_start:source_end, :
            ].transpose(0, 2, 1, 3)
        else:
            assert shard.packed is not None
            packed[:, destination_start:destination_end, :, :, :] = shard.packed[
                :, source_start:source_end, token_start:token_end, :, :
            ]
        covered[destination_start:destination_end] = True
    if not np.all(covered):
        raise LayoutValidationError("source shards do not cover the requested logical head range")
    return packed


def stream_direct_conversion(
    source: PhysicalKVState,
    destination: KVLayout,
    *,
    maximum_temporary_bytes: int,
) -> Iterator[ConvertedChunk]:
    """Direct chunk conversion without materializing complete canonical state.

    ``direct_convert`` consumes chunks one at a time. Both the yielded payload and
    the converter's scratch state are bounded by the compiled schedule.
    """

    source.verify_integrity()
    program = compile_conversion(
        source.layout,
        destination,
        maximum_temporary_bytes=maximum_temporary_bytes,
    )
    dtype = np.dtype(destination.dtype)
    for assignment in program.chunk_schedule.chunks:
        if destination.kind is KVLayoutKind.TOKEN_MAJOR_SEPARATE:
            key, value = _read_logical_chunk(
                source,
                token_start=assignment.token_start,
                token_end=assignment.token_end,
                head_start=assignment.head_start,
                head_end=assignment.head_end,
                dtype=dtype,
            )
            chunk = ConvertedChunk(
                chunk_id=assignment.chunk_id,
                destination_rank=assignment.destination_rank,
                token_start=assignment.token_start,
                token_end=assignment.token_end,
                head_start=assignment.head_start,
                head_end=assignment.head_end,
                key=key,
                value=value,
            )
        else:
            packed = _read_packed_chunk(
                source,
                token_start=assignment.token_start,
                token_end=assignment.token_end,
                head_start=assignment.head_start,
                head_end=assignment.head_end,
                dtype=dtype,
            )
            chunk = ConvertedChunk(
                chunk_id=assignment.chunk_id,
                destination_rank=assignment.destination_rank,
                token_start=assignment.token_start,
                token_end=assignment.token_end,
                head_start=assignment.head_start,
                head_end=assignment.head_end,
                packed=packed,
            )
        if chunk.temporary_nbytes > maximum_temporary_bytes:
            raise ConversionCompilationError("converter exceeded its compiled memory bound")
        yield chunk


def direct_convert(
    source: PhysicalKVState,
    destination: KVLayout,
    *,
    maximum_temporary_bytes: int,
) -> PhysicalKVState:
    """Execute the direct, resharding streaming program into destination storage."""

    output = allocate_state(destination)
    chunks = stream_direct_conversion(
        source,
        destination,
        maximum_temporary_bytes=maximum_temporary_bytes,
    )
    shards_by_rank: dict[int, KVShard] = {shard.rank: shard for shard in output.shards}
    for chunk in chunks:
        shard = shards_by_rank[chunk.destination_rank]
        token_slice = slice(chunk.token_start, chunk.token_end)
        if destination.kind is KVLayoutKind.TOKEN_MAJOR_SEPARATE:
            assert shard.key is not None and shard.value is not None
            assert chunk.key is not None and chunk.value is not None
            shard.key[:, token_slice, :, :] = chunk.key
            shard.value[:, token_slice, :, :] = chunk.value
        else:
            assert shard.packed is not None and chunk.packed is not None
            shard.packed[:, :, token_slice, :, :] = chunk.packed
    return PhysicalKVState(layout=destination, shards=output.shards)
