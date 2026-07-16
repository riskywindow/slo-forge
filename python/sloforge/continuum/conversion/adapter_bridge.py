"""Execute the direct KV converter over exact reference-adapter capture bytes."""

from __future__ import annotations

import struct
from dataclasses import dataclass, replace

import numpy as np

from sloforge.continuum.adapters import (
    CapturedState,
    LayoutKind,
    StateKind,
)
from sloforge.continuum.adapters.sdk import checksum_bytes
from sloforge.continuum.reference.codec import decode_captured, encode_state
from sloforge.continuum.reference.runtime import DeterministicHybridRuntimeAdapter

from .compiler import direct_convert
from .layouts import KVLayout, KVLayoutKind, KVShard, PhysicalKVState


@dataclass(frozen=True, slots=True)
class LiveConversionEvidence:
    source_hash: str
    direct_hash: str
    source_segment_count: int
    destination_segment_count: int
    compared_attention_bytes: int
    canonical_attention_match: bool
    maximum_temporary_bytes: int


def _int32_values(payload: bytes) -> np.ndarray:
    if len(payload) % 4:
        raise ValueError("reference adapter segment is not int32 aligned")
    if not payload:
        return np.empty((0,), dtype=np.dtype("<i4"))
    return np.asarray(struct.unpack(f"<{len(payload) // 4}i", payload), dtype="<i4")


def captured_attention_state(captured: CapturedState) -> PhysicalKVState:
    """Decode only physical attention shards, without canonical KV materialization."""

    captured.verify()
    logical = captured.logical
    kind = (
        KVLayoutKind.TOKEN_MAJOR_SEPARATE
        if captured.layout.kind is LayoutKind.PAGED_TOKEN_MAJOR_SEPARATE_KV
        else KVLayoutKind.HEAD_MAJOR_PACKED
    )
    layout = KVLayout(
        kind=kind,
        tensor_parallel_degree=captured.layout.tensor_parallel_degree,
        page_size_tokens=captured.layout.page_size_tokens,
        layer_count=logical.attention_layer_count,
        token_count=len(logical.input_token_ids) + len(logical.committed_output_token_ids),
        kv_head_count=logical.attention_kv_head_count,
        head_dim=logical.attention_head_dimension,
        dtype="int32",
    )
    shards: list[KVShard] = []
    for rank in range(layout.tensor_parallel_degree):
        head_start = rank * layout.heads_per_shard
        head_end = head_start + layout.heads_per_shard
        if kind is KVLayoutKind.TOKEN_MAJOR_SEPARATE:
            key = np.zeros(
                (
                    layout.layer_count,
                    layout.padded_token_count,
                    layout.heads_per_shard,
                    layout.head_dim,
                ),
                dtype=np.int32,
            )
            value = np.zeros_like(key)
            shards.append(
                KVShard(
                    rank=rank,
                    head_start=head_start,
                    head_end=head_end,
                    key=key,
                    value=value,
                )
            )
        else:
            packed = np.zeros(
                (
                    layout.layer_count,
                    layout.heads_per_shard,
                    layout.padded_token_count,
                    2,
                    layout.head_dim,
                ),
                dtype=np.int32,
            )
            shards.append(
                KVShard(
                    rank=rank,
                    head_start=head_start,
                    head_end=head_end,
                    packed=packed,
                )
            )
    for segment in captured.segments:
        descriptor = segment.descriptor
        if descriptor.layer is None:
            continue
        shard = shards[descriptor.shard_rank]
        values = _int32_values(segment.payload).reshape(descriptor.logical_shape)
        token_slice = slice(descriptor.token_start, descriptor.token_end)
        if descriptor.state_kind is StateKind.ATTENTION_KEY:
            if shard.key is None:
                raise ValueError("key segment is incompatible with packed source layout")
            shard.key[descriptor.layer, token_slice, :, :] = values
        elif descriptor.state_kind is StateKind.ATTENTION_VALUE:
            if shard.value is None:
                raise ValueError("value segment is incompatible with packed source layout")
            shard.value[descriptor.layer, token_slice, :, :] = values
        elif descriptor.state_kind is StateKind.ATTENTION_PACKED_KV:
            if shard.packed is None:
                raise ValueError("packed segment is incompatible with separate source layout")
            shard.packed[descriptor.layer, :, token_slice, :, :] = values
    return PhysicalKVState(layout=layout, shards=tuple(shards))


def direct_convert_capture(
    captured: CapturedState,
    *,
    destination: DeterministicHybridRuntimeAdapter,
    maximum_temporary_bytes: int,
) -> tuple[CapturedState, LiveConversionEvidence]:
    """Relayout live KV bytes directly and independently compare trusted output bytes."""

    if destination.config.layout.kind is not LayoutKind.PAGED_HEAD_MAJOR_PACKED_KV:
        raise ValueError("live direct conversion currently requires a head-major destination")
    source = captured_attention_state(captured)
    destination_layout = KVLayout(
        kind=KVLayoutKind.HEAD_MAJOR_PACKED,
        tensor_parallel_degree=destination.config.layout.tensor_parallel_degree,
        page_size_tokens=destination.config.layout.page_size_tokens,
        layer_count=source.layout.layer_count,
        token_count=source.layout.token_count,
        kv_head_count=source.layout.kv_head_count,
        head_dim=source.layout.head_dim,
        dtype="int32",
    )
    direct = direct_convert(
        source,
        destination_layout,
        maximum_temporary_bytes=maximum_temporary_bytes,
    )
    state = decode_captured(
        captured,
        destination_config=destination.config,
        destination_session_id=captured.logical.session_id,
    )
    canonical = encode_state(state, destination.config)
    shards = {shard.rank: shard for shard in direct.shards}
    converted_segments = []
    compared_bytes = 0
    canonical_match = True
    for segment in canonical.segments:
        descriptor = segment.descriptor
        if descriptor.state_kind is not StateKind.ATTENTION_PACKED_KV:
            converted_segments.append(segment)
            continue
        shard = shards[descriptor.shard_rank]
        if shard.packed is None or descriptor.layer is None:
            raise ValueError("direct converter did not produce packed attention state")
        direct_array = shard.packed[
            descriptor.layer,
            :,
            descriptor.token_start : descriptor.token_end,
            :,
            :,
        ]
        payload = np.ascontiguousarray(direct_array, dtype="<i4").tobytes(order="C")
        compared_bytes += len(payload)
        canonical_match = canonical_match and payload == segment.payload
        converted_segments.append(
            replace(
                segment,
                descriptor=replace(
                    descriptor,
                    checksum=checksum_bytes(payload),
                    payload_bytes=len(payload),
                ),
                payload=payload,
            )
        )
    if not canonical_match:
        raise ValueError("direct live-state conversion differs from trusted canonical bytes")
    converted = CapturedState(
        handle=replace(captured.handle, segment_count=len(converted_segments)),
        runtime=destination.identity,
        layout=destination.config.layout,
        logical=canonical.logical,
        segments=tuple(converted_segments),
        page_table=canonical.page_table,
    )
    converted.verify()
    return converted, LiveConversionEvidence(
        source_hash=source.content_hash,
        direct_hash=direct.content_hash,
        source_segment_count=len(captured.segments),
        destination_segment_count=len(converted.segments),
        compared_attention_bytes=compared_bytes,
        canonical_attention_match=True,
        maximum_temporary_bytes=maximum_temporary_bytes,
    )
