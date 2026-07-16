"""Trusted CPU codecs for the two reference physical state layouts."""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass

from sloforge.continuum.adapters.sdk import (
    CapturedState,
    LayoutKind,
    LogicalStateManifest,
    PageTableEntry,
    RuntimeLayout,
    SegmentDescriptor,
    SegmentIntegrityError,
    SessionLifecycle,
    StateKind,
    StateSegment,
    checksum_bytes,
)
from sloforge.continuum.reference.models import (
    HybridDecoderConfig,
    HybridDecoderState,
    state_from_manifest,
)


@dataclass(frozen=True, slots=True)
class EncodedState:
    logical: LogicalStateManifest
    segments: tuple[StateSegment, ...]
    page_table: tuple[PageTableEntry, ...]


def _json_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _i32_bytes(values: list[int]) -> bytes:
    if not values:
        return b""
    return struct.pack(f"<{len(values)}i", *values)


def _i32_values(payload: bytes) -> list[int]:
    if len(payload) % 4:
        raise ValueError("int32 segment payload is not word aligned")
    if not payload:
        return []
    return list(struct.unpack(f"<{len(payload) // 4}i", payload))


def _page_dirty_epoch(state: HybridDecoderState, token_start: int, token_end: int) -> int:
    return max(state.token_dirty_epochs[token_start:token_end], default=0)


def _segment(
    *,
    segment_id: str,
    semantic_id: str,
    state_kind: StateKind,
    logical_shape: tuple[int, ...],
    dtype: str,
    encoding: str,
    layer: int | None,
    shard_rank: int,
    shard_count: int,
    token_start: int,
    token_end: int,
    head_start: int,
    head_end: int,
    page_id: int | None,
    version: int,
    dirty_epoch: int,
    payload: bytes,
) -> StateSegment:
    descriptor = SegmentDescriptor(
        segment_id=segment_id,
        semantic_id=semantic_id,
        state_kind=state_kind,
        logical_shape=logical_shape,
        dtype=dtype,
        encoding=encoding,
        layer=layer,
        shard_rank=shard_rank,
        shard_count=shard_count,
        token_start=token_start,
        token_end=token_end,
        head_start=head_start,
        head_end=head_end,
        page_id=page_id,
        version=version,
        dirty_epoch=dirty_epoch,
        required_before_resume=True,
        payload_bytes=len(payload),
        checksum=checksum_bytes(payload),
    )
    return StateSegment(descriptor=descriptor, payload=payload)


def _encode_scalar_segments(
    state: HybridDecoderState,
    config: HybridDecoderConfig,
    manifest: LogicalStateManifest,
) -> list[StateSegment]:
    token_count = state.token_count
    scalar_payloads: tuple[tuple[str, StateKind, tuple[int, ...], str, bytes], ...] = (
        (
            "recurrent",
            StateKind.RECURRENT,
            (config.layers, config.recurrent_width),
            "little_endian_i32",
            _i32_bytes([value for layer in state.recurrent_state for value in layer]),
        ),
        (
            "sampler",
            StateKind.SAMPLER,
            (1,),
            "canonical_json",
            _json_bytes(
                {
                    "algorithm": manifest.sampler.algorithm,
                    "counter": manifest.sampler.counter,
                    "seed": manifest.sampler.seed,
                    "temperature_milli": manifest.sampler.temperature_milli,
                    "top_k": manifest.sampler.top_k,
                    "top_p_millionths": manifest.sampler.top_p_millionths,
                }
            ),
        ),
        (
            "guided",
            StateKind.GUIDED_DECODING,
            (1,),
            "canonical_json",
            _json_bytes(
                {
                    "accepted_prefix_length": manifest.guided_decoding.accepted_prefix_length,
                    "automaton_hash": manifest.guided_decoding.automaton_hash,
                    "automaton_id": manifest.guided_decoding.automaton_id,
                    "state": manifest.guided_decoding.state,
                }
            ),
        ),
        (
            "history",
            StateKind.TOKEN_HISTORY,
            (token_count,),
            "canonical_json",
            _json_bytes(
                {
                    "committed_output_token_ids": manifest.committed_output_token_ids,
                    "input_token_ids": manifest.input_token_ids,
                    "uncommitted_speculative_token_ids": (
                        manifest.uncommitted_speculative_token_ids
                    ),
                }
            ),
        ),
        (
            "delivery",
            StateKind.CLIENT_DELIVERY,
            (1,),
            "canonical_json",
            _json_bytes(
                {
                    "last_client_acknowledged_token_index": (
                        manifest.client_delivery.last_client_acknowledged_token_index
                    ),
                    "last_gateway_committed_token_index": (
                        manifest.client_delivery.last_gateway_committed_token_index
                    ),
                    "last_generated_token_index": (
                        manifest.client_delivery.last_generated_token_index
                    ),
                    "stream_owner_epoch": manifest.client_delivery.stream_owner_epoch,
                    "terminal_status": manifest.client_delivery.terminal_status.value,
                }
            ),
        ),
    )
    return [
        _segment(
            segment_id=f"logical:{name}",
            semantic_id=f"logical/{name}",
            state_kind=kind,
            logical_shape=shape,
            dtype="int32" if kind is StateKind.RECURRENT else "structured",
            encoding=encoding,
            layer=None,
            shard_rank=0,
            shard_count=1,
            token_start=0,
            token_end=token_count,
            head_start=0,
            head_end=0,
            page_id=None,
            version=state.state_version,
            dirty_epoch=state.state_version,
            payload=payload,
        )
        for name, kind, shape, encoding, payload in scalar_payloads
    ]


def _encode_layout_a(
    state: HybridDecoderState,
    config: HybridDecoderConfig,
) -> tuple[list[StateSegment], list[PageTableEntry]]:
    layout = config.layout
    local_heads = config.kv_heads // layout.tensor_parallel_degree
    segments: list[StateSegment] = []
    pages: list[PageTableEntry] = []
    for layer in range(config.layers):
        for page_start in range(0, state.token_count, layout.page_size_tokens):
            page_end = min(state.token_count, page_start + layout.page_size_tokens)
            page_id = page_start // layout.page_size_tokens
            page_epoch = _page_dirty_epoch(state, page_start, page_end)
            for rank in range(layout.tensor_parallel_degree):
                head_start = rank * local_heads
                head_end = head_start + local_heads
                segment_ids: list[str] = []
                for suffix, kind, tensor in (
                    ("k", StateKind.ATTENTION_KEY, state.attention_keys),
                    ("v", StateKind.ATTENTION_VALUE, state.attention_values),
                ):
                    values = [
                        tensor[layer][token][head][dimension]
                        for token in range(page_start, page_end)
                        for head in range(head_start, head_end)
                        for dimension in range(config.head_dimension)
                    ]
                    segment_id = f"attn:l{layer}:p{page_id}:r{rank}:{suffix}"
                    segment_ids.append(segment_id)
                    segments.append(
                        _segment(
                            segment_id=segment_id,
                            semantic_id=(
                                f"attention/layer/{layer}/{suffix}/tokens/{page_start}:{page_end}"
                            ),
                            state_kind=kind,
                            logical_shape=(
                                page_end - page_start,
                                local_heads,
                                config.head_dimension,
                            ),
                            dtype="int32",
                            encoding="little_endian_i32_token_major",
                            layer=layer,
                            shard_rank=rank,
                            shard_count=layout.tensor_parallel_degree,
                            token_start=page_start,
                            token_end=page_end,
                            head_start=head_start,
                            head_end=head_end,
                            page_id=page_id,
                            version=page_epoch,
                            dirty_epoch=page_epoch,
                            payload=_i32_bytes(values),
                        )
                    )
                pages.append(
                    PageTableEntry(
                        logical_state_id=f"attention/layer/{layer}",
                        layer=layer,
                        shard_rank=rank,
                        logical_token_start=page_start,
                        logical_token_end=page_end,
                        physical_page_id=page_id,
                        segment_ids=tuple(segment_ids),
                        page_version=page_epoch,
                        dirty_epoch=page_epoch,
                        owner_epoch=state.owner_epoch,
                    )
                )
    return segments, pages


def _encode_layout_b(
    state: HybridDecoderState,
    config: HybridDecoderConfig,
) -> tuple[list[StateSegment], list[PageTableEntry]]:
    layout = config.layout
    local_heads = config.kv_heads // layout.tensor_parallel_degree
    segments: list[StateSegment] = []
    pages: list[PageTableEntry] = []
    for layer in range(config.layers):
        for page_start in range(0, state.token_count, layout.page_size_tokens):
            page_end = min(state.token_count, page_start + layout.page_size_tokens)
            page_id = page_start // layout.page_size_tokens
            page_epoch = _page_dirty_epoch(state, page_start, page_end)
            for rank in range(layout.tensor_parallel_degree):
                head_start = rank * local_heads
                head_end = head_start + local_heads
                values: list[int] = []
                for head in range(head_start, head_end):
                    for token in range(page_start, page_end):
                        values.extend(state.attention_keys[layer][token][head])
                        values.extend(state.attention_values[layer][token][head])
                segment_id = f"attn:l{layer}:p{page_id}:r{rank}:kv"
                segments.append(
                    _segment(
                        segment_id=segment_id,
                        semantic_id=(
                            f"attention/layer/{layer}/packed-kv/tokens/{page_start}:{page_end}"
                        ),
                        state_kind=StateKind.ATTENTION_PACKED_KV,
                        logical_shape=(
                            local_heads,
                            page_end - page_start,
                            2,
                            config.head_dimension,
                        ),
                        dtype="int32",
                        encoding="little_endian_i32_head_major_packed_kv",
                        layer=layer,
                        shard_rank=rank,
                        shard_count=layout.tensor_parallel_degree,
                        token_start=page_start,
                        token_end=page_end,
                        head_start=head_start,
                        head_end=head_end,
                        page_id=page_id,
                        version=page_epoch,
                        dirty_epoch=page_epoch,
                        payload=_i32_bytes(values),
                    )
                )
                pages.append(
                    PageTableEntry(
                        logical_state_id=f"attention/layer/{layer}",
                        layer=layer,
                        shard_rank=rank,
                        logical_token_start=page_start,
                        logical_token_end=page_end,
                        physical_page_id=page_id,
                        segment_ids=(segment_id,),
                        page_version=page_epoch,
                        dirty_epoch=page_epoch,
                        owner_epoch=state.owner_epoch,
                    )
                )
    return segments, pages


def encode_state(state: HybridDecoderState, config: HybridDecoderConfig) -> EncodedState:
    manifest = state.logical_manifest(config)
    if config.layout.kind is LayoutKind.PAGED_TOKEN_MAJOR_SEPARATE_KV:
        attention_segments, page_table = _encode_layout_a(state, config)
    elif config.layout.kind is LayoutKind.PAGED_HEAD_MAJOR_PACKED_KV:
        attention_segments, page_table = _encode_layout_b(state, config)
    else:
        raise ValueError(f"trusted codec does not support layout {config.layout.kind.value}")
    scalar_segments = _encode_scalar_segments(state, config, manifest)
    segments = tuple(
        sorted((*attention_segments, *scalar_segments), key=lambda item: item.descriptor.segment_id)
    )
    pages = tuple(
        sorted(
            page_table,
            key=lambda item: (item.layer, item.physical_page_id, item.shard_rank),
        )
    )
    return EncodedState(logical=manifest, segments=segments, page_table=pages)


def _check_scalar_segments(
    segments: tuple[StateSegment, ...],
    config: HybridDecoderConfig,
    manifest: LogicalStateManifest,
) -> None:
    synthetic = HybridDecoderState(
        session_id=manifest.session_id,
        request_id=manifest.request_id,
        tenant_id=manifest.tenant_id,
        seed=manifest.sampler.seed,
        owner_epoch=manifest.owner_epoch,
        input_token_ids=list(manifest.input_token_ids),
        output_token_ids=list(manifest.committed_output_token_ids),
        attention_keys=[[] for _ in range(config.layers)],
        attention_values=[[] for _ in range(config.layers)],
        token_dirty_epochs=[0]
        * (len(manifest.input_token_ids) + len(manifest.committed_output_token_ids)),
        recurrent_state=[list(layer) for layer in manifest.recurrent_state],
        sampler_counter=manifest.sampler.counter,
        guided_state=manifest.guided_decoding.state,
        gateway_committed_index=manifest.client_delivery.last_gateway_committed_token_index,
        client_acknowledged_index=manifest.client_delivery.last_client_acknowledged_token_index,
        state_version=manifest.state_version,
        lifecycle=SessionLifecycle.PREPARED,
    )
    expected = {
        segment.descriptor.segment_id: segment
        for segment in _encode_scalar_segments(synthetic, config, manifest)
    }
    actual = {
        segment.descriptor.segment_id: segment
        for segment in segments
        if segment.descriptor.state_kind
        not in {
            StateKind.ATTENTION_KEY,
            StateKind.ATTENTION_VALUE,
            StateKind.ATTENTION_PACKED_KV,
        }
    }
    if set(actual) != set(expected):
        raise ValueError("logical scalar segment set is incomplete or contains duplicates")
    for segment_id, expected_segment in expected.items():
        if actual[segment_id].payload != expected_segment.payload:
            raise SegmentIntegrityError(
                "logical scalar payload disagrees with the manifest",
                operation="decode_snapshot",
                session_id=manifest.session_id,
            )


def decode_segments(
    *,
    source_layout: RuntimeLayout,
    source_segments: tuple[StateSegment, ...],
    manifest: LogicalStateManifest,
    destination_config: HybridDecoderConfig,
    destination_session_id: str,
) -> HybridDecoderState:
    """Decode either trusted source layout to canonical in-memory reference state."""

    if manifest.model != destination_config.model:
        raise ValueError("source model contract does not match destination reference model")
    if manifest.guided_decoding.automaton_hash != destination_config.automaton_hash:
        raise ValueError("guided decoding automaton does not match destination")
    for segment in source_segments:
        if checksum_bytes(segment.payload) != segment.descriptor.checksum:
            raise SegmentIntegrityError(
                "source segment checksum mismatch",
                operation="decode_snapshot",
                session_id=manifest.session_id,
            )
    _check_scalar_segments(source_segments, destination_config, manifest)

    token_count = len(manifest.input_token_ids) + len(manifest.committed_output_token_ids)
    layers = destination_config.layers
    heads = destination_config.kv_heads
    dimension = destination_config.head_dimension
    keys = [
        [[[0 for _ in range(dimension)] for _ in range(heads)] for _ in range(token_count)]
        for _ in range(layers)
    ]
    values = [
        [[[0 for _ in range(dimension)] for _ in range(heads)] for _ in range(token_count)]
        for _ in range(layers)
    ]
    key_coverage = [
        [[False for _ in range(heads)] for _ in range(token_count)] for _ in range(layers)
    ]
    value_coverage = [
        [[False for _ in range(heads)] for _ in range(token_count)] for _ in range(layers)
    ]
    attention_segments = [
        segment
        for segment in source_segments
        if segment.descriptor.state_kind
        in {
            StateKind.ATTENTION_KEY,
            StateKind.ATTENTION_VALUE,
            StateKind.ATTENTION_PACKED_KV,
        }
    ]
    for segment in attention_segments:
        descriptor = segment.descriptor
        if descriptor.layer is None or not 0 <= descriptor.layer < layers:
            raise ValueError("attention segment has an invalid layer")
        if not 0 <= descriptor.token_start <= descriptor.token_end <= token_count:
            raise ValueError("attention segment token range is out of bounds")
        if not 0 <= descriptor.head_start <= descriptor.head_end <= heads:
            raise ValueError("attention segment head range is out of bounds")
        words = _i32_values(segment.payload)
        layer = descriptor.layer
        if source_layout.kind is LayoutKind.PAGED_TOKEN_MAJOR_SEPARATE_KV:
            if descriptor.state_kind not in {StateKind.ATTENTION_KEY, StateKind.ATTENTION_VALUE}:
                raise ValueError("token-major layout contains a non-separate KV segment")
            expected_words = (
                (descriptor.token_end - descriptor.token_start)
                * (descriptor.head_end - descriptor.head_start)
                * dimension
            )
            if len(words) != expected_words:
                raise ValueError("token-major attention payload has the wrong shape")
            cursor = 0
            target = keys if descriptor.state_kind is StateKind.ATTENTION_KEY else values
            coverage = (
                key_coverage if descriptor.state_kind is StateKind.ATTENTION_KEY else value_coverage
            )
            for token in range(descriptor.token_start, descriptor.token_end):
                for head in range(descriptor.head_start, descriptor.head_end):
                    if coverage[layer][token][head]:
                        raise ValueError("overlapping attention shard coverage")
                    target[layer][token][head] = words[cursor : cursor + dimension]
                    coverage[layer][token][head] = True
                    cursor += dimension
        elif source_layout.kind is LayoutKind.PAGED_HEAD_MAJOR_PACKED_KV:
            if descriptor.state_kind is not StateKind.ATTENTION_PACKED_KV:
                raise ValueError("head-major layout contains a non-packed KV segment")
            expected_words = (
                (descriptor.token_end - descriptor.token_start)
                * (descriptor.head_end - descriptor.head_start)
                * dimension
                * 2
            )
            if len(words) != expected_words:
                raise ValueError("head-major packed attention payload has the wrong shape")
            cursor = 0
            for head in range(descriptor.head_start, descriptor.head_end):
                for token in range(descriptor.token_start, descriptor.token_end):
                    if key_coverage[layer][token][head] or value_coverage[layer][token][head]:
                        raise ValueError("overlapping packed attention shard coverage")
                    keys[layer][token][head] = words[cursor : cursor + dimension]
                    cursor += dimension
                    values[layer][token][head] = words[cursor : cursor + dimension]
                    cursor += dimension
                    key_coverage[layer][token][head] = True
                    value_coverage[layer][token][head] = True
        else:
            raise ValueError(f"unsupported source layout {source_layout.kind.value}")
    if token_count and (
        not all(all(all(token) for token in layer) for layer in key_coverage)
        or not all(all(all(token) for token in layer) for layer in value_coverage)
    ):
        raise ValueError("attention shards do not provide complete token/head coverage")
    state = state_from_manifest(
        destination_config,
        manifest,
        destination_session_id=destination_session_id,
        attention_keys=keys,
        attention_values=values,
    )
    # Recompute from tokens independently to make plausible but internally inconsistent
    # state fail closed even when all segment checksums are self-consistent.
    replay = HybridDecoderState.create(
        destination_config,
        session_id=destination_session_id,
        request_id=manifest.request_id,
        tenant_id=manifest.tenant_id,
        seed=manifest.sampler.seed,
        owner_epoch=manifest.owner_epoch,
        input_token_ids=manifest.input_token_ids,
    )
    for expected_token in manifest.committed_output_token_ids:
        event = replay.generate(destination_config, transaction_id=None)
        if event.token_id != expected_token:
            raise ValueError("token history is inconsistent with deterministic sampler state")
        replay.acknowledge_gateway(token_index=event.token_index, owner_epoch=manifest.owner_epoch)
    if replay.continuation_hash(destination_config) != state.continuation_hash(destination_config):
        raise ValueError("decoded state fails independent continuation-state replay")
    return state


def decode_captured(
    captured: CapturedState,
    *,
    destination_config: HybridDecoderConfig,
    destination_session_id: str,
) -> HybridDecoderState:
    captured.verify()
    return decode_segments(
        source_layout=captured.layout,
        source_segments=captured.segments,
        manifest=captured.logical,
        destination_config=destination_config,
        destination_session_id=destination_session_id,
    )
