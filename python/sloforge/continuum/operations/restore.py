"""Strict reconstruction of reference adapter captures from authenticated capsules."""

from __future__ import annotations

import json
import re
import struct
from collections import defaultdict
from typing import Any, cast

from sloforge.continuum.adapters import (
    CapturedState,
    ClientTerminalStatus,
    LayoutKind,
    LogicalStateManifest,
    ModelContract,
    PageTableEntry,
    RuntimeIdentity,
    RuntimeLayout,
    SegmentDescriptor,
    SnapshotHandle,
)
from sloforge.continuum.adapters import StateKind as RuntimeStateKind
from sloforge.continuum.adapters import StateSegment as RuntimeSegment
from sloforge.continuum.adapters.sdk import (
    ClientDeliverySnapshot,
    GuidedDecodingSnapshot,
    SamplerSnapshot,
)
from sloforge.continuum.ir import (
    CompressionKind,
    EncryptionKind,
    KVPacking,
    Ordering,
)
from sloforge.continuum.storage import ContentStore

from .checkpoint import verify_checkpoint_artifact
from .models import (
    AuthorizationError,
    CapsuleRestoreError,
    CheckpointArtifact,
    UnsupportedReferenceABI,
)

_ATTENTION_ID = re.compile(
    r"^attn:l(?P<layer>[0-9]+):p(?P<page>[0-9]+):r(?P<rank>[0-9]+):(?P<part>k|v|kv)$"
)
_KNOWN_RUNTIMES = {
    (
        "continuum-reference-token-major",
        "continuum-adapter-a/1.0.0",
    ): LayoutKind.PAGED_TOKEN_MAJOR_SEPARATE_KV,
    (
        "continuum-reference-head-major",
        "continuum-adapter-b/1.0.0",
    ): LayoutKind.PAGED_HEAD_MAJOR_PACKED_KV,
}
_SCALAR_KINDS = {
    "logical:recurrent": RuntimeStateKind.RECURRENT,
    "logical:sampler": RuntimeStateKind.SAMPLER,
    "logical:guided": RuntimeStateKind.GUIDED_DECODING,
    "logical:history": RuntimeStateKind.TOKEN_HISTORY,
    "logical:delivery": RuntimeStateKind.CLIENT_DELIVERY,
}
_SCALAR_ENCODING = {
    "logical:recurrent": ("int32", "little_endian_i32"),
    "logical:sampler": ("structured", "canonical_json"),
    "logical:guided": ("structured", "canonical_json"),
    "logical:history": ("structured", "canonical_json"),
    "logical:delivery": ("structured", "canonical_json"),
}


def _json_object(payload: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CapsuleRestoreError(f"{label} is not canonical JSON") from error
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise CapsuleRestoreError(f"{label} must be a JSON object")
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    if canonical != payload:
        raise CapsuleRestoreError(f"{label} does not use canonical JSON encoding")
    return cast(dict[str, Any], value)


def _require_int(value: object, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise CapsuleRestoreError(f"{label} must be an integer")
    return value


def _restriction(capsule: CheckpointArtifact, prefix: str) -> str:
    matches = [
        value.removeprefix(prefix)
        for value in capsule.capsule.compatibility.architecture_restrictions
        if value.startswith(prefix)
    ]
    if len(matches) != 1 or re.fullmatch(r"[0-9a-f]{64}", matches[0]) is None:
        raise CapsuleRestoreError(f"capsule lacks one authenticated {prefix[:-1]} restriction")
    return matches[0]


def _runtime_identity(artifact: CheckpointArtifact) -> RuntimeIdentity:
    source = artifact.capsule.physical_state.runtime
    key = (source.runtime_name, source.adapter_version)
    if key not in _KNOWN_RUNTIMES or source.runtime_version != "1.0.0":
        raise UnsupportedReferenceABI(
            f"unsupported reference runtime contract {source.runtime_name}/"
            f"{source.runtime_version}/{source.adapter_version}"
        )
    dependencies: list[tuple[str, str]] = []
    for dependency in source.dependency_versions:
        name, separator, version = dependency.partition("=")
        if not separator or not name or not version:
            raise UnsupportedReferenceABI("runtime dependency metadata is not reversible")
        dependencies.append((name, version))
    if len(source.target_hardware) != 1:
        raise UnsupportedReferenceABI("reference runtime requires one target hardware contract")
    return RuntimeIdentity(
        runtime_name=source.runtime_name,
        runtime_version=source.runtime_version,
        adapter_version=source.adapter_version,
        build_hash=source.build_hash.value,
        dependency_versions=tuple(dependencies),
        target_hardware=source.target_hardware[0],
    )


def _runtime_layout(artifact: CheckpointArtifact) -> RuntimeLayout:
    physical = artifact.capsule.physical_state
    if len(physical.layout_descriptors) != 1:
        raise UnsupportedReferenceABI("reference capsule must contain exactly one layout")
    descriptor = physical.layout_descriptors[0]
    runtime_kind = _KNOWN_RUNTIMES[
        (physical.runtime.runtime_name, physical.runtime.adapter_version)
    ]
    expected = (
        (Ordering.TOKEN_MAJOR, KVPacking.SEPARATE)
        if runtime_kind is LayoutKind.PAGED_TOKEN_MAJOR_SEPARATE_KV
        else (Ordering.HEAD_MAJOR, KVPacking.PACKED_KV)
    )
    if (descriptor.ordering, descriptor.k_v_packing) != expected:
        raise UnsupportedReferenceABI("runtime identity and physical layout contract disagree")
    extensions = physical.extensions.root
    page_size = extensions.get("sloforge.runtime/PageSizeTokens")
    devices = extensions.get("sloforge.runtime/SimulatedDevices")
    if not isinstance(page_size, int) or isinstance(page_size, bool):
        raise UnsupportedReferenceABI("reference page size metadata is absent")
    if not isinstance(devices, list) or any(not isinstance(item, str) for item in devices):
        raise UnsupportedReferenceABI("reference device metadata is absent")
    attention_degrees = {
        shard.tensor_parallel_degree
        for segment in physical.segments
        if segment.logical_state_reference == "state/attention-kv"
        for shard in physical.shard_descriptors
        if shard.shard_id == segment.shard_id
    }
    degree = len(devices)
    if attention_degrees and attention_degrees != {degree}:
        raise UnsupportedReferenceABI("reference tensor-parallel metadata is inconsistent")
    return RuntimeLayout(
        kind=runtime_kind,
        page_size_tokens=page_size,
        tensor_parallel_degree=degree,
        ordering="token-major" if expected[0] is Ordering.TOKEN_MAJOR else "head-major",
        kv_packing="separate-k-v" if expected[1] is KVPacking.SEPARATE else "packed-k-v",
        alignment_bytes=descriptor.alignment_bytes,
        simulated_devices=tuple(cast(list[str], devices)),
    )


def _read_payloads(
    artifact: CheckpointArtifact, store: ContentStore, tenant_id: str
) -> dict[str, bytes]:
    references = {reference.digest: reference for reference in artifact.chunk_references}
    payloads: dict[str, bytes] = {}
    for manifest in artifact.capsule.segment_manifests:
        if len(manifest.chunks) != 1:
            raise UnsupportedReferenceABI("reference segment must resolve to exactly one CAS chunk")
        chunk = manifest.chunks[0]
        if chunk.tenant_security_domain != tenant_id:
            raise AuthorizationError("segment names another tenant security domain")
        if (
            chunk.compression is not CompressionKind.NONE
            or chunk.encryption is not EncryptionKind.NONE
        ):
            raise UnsupportedReferenceABI(
                "reference loader does not silently decode transformed chunks"
            )
        expected_uri = f"cas://{tenant_id}/{chunk.content_hash.value}"
        if chunk.storage_uri != expected_uri:
            raise CapsuleRestoreError("segment CAS URI is not the authenticated local reference")
        reference = references.get(chunk.content_hash.value)
        if reference is None:
            raise CapsuleRestoreError("authorized CAS publication is missing a segment")
        payload = store.read(tenant_id, reference)
        if len(payload) != chunk.size_bytes:
            raise CapsuleRestoreError("CAS payload size differs from capsule")
        payloads[manifest.segment_id] = payload
    return payloads


def _model_contract(artifact: CheckpointArtifact, expected: ModelContract) -> ModelContract:
    capsule = artifact.capsule
    logical = capsule.logical_state
    if capsule.identity.model_hash.value != expected.model_hash:
        raise AuthorizationError("destination model hash differs from capsule model")
    if capsule.identity.tokenizer_hash.value != expected.tokenizer_hash:
        raise AuthorizationError("destination tokenizer hash differs from capsule tokenizer")
    expected_adapter = logical.execution.adapter_identity
    if expected_adapter is None or expected_adapter.value != expected.adapter_hash:
        raise AuthorizationError("destination adapter identity differs from capsule")
    if _restriction(artifact, "state_producer=") != expected.state_producer_hash:
        raise AuthorizationError("destination state-producing weights differ from capsule")
    if _restriction(artifact, "recurrent_update=") != expected.recurrent_update_hash:
        raise AuthorizationError("destination recurrent update differs from capsule")
    all_tokens = (
        *logical.token_history.input_token_ids,
        *logical.token_history.committed_output_token_ids,
    )
    if any(token >= expected.vocabulary_size for token in all_tokens):
        raise AuthorizationError("capsule token history exceeds destination vocabulary")
    return expected


def _logical_manifest(
    artifact: CheckpointArtifact,
    payloads: dict[str, bytes],
    model: ModelContract,
) -> LogicalStateManifest:
    logical = artifact.capsule.logical_state
    if logical.attention is None or not logical.recurrent or logical.guided_decoding is None:
        raise UnsupportedReferenceABI(
            "reference restore requires attention, recurrent, and guidance state"
        )
    if (
        logical.workflow is not None
        or logical.speculative is not None
        or logical.unknown_components
    ):
        raise UnsupportedReferenceABI(
            "reference adapter cannot import workflow/speculative/unknown state"
        )
    recurrent_meta = logical.recurrent[0]
    if len(recurrent_meta.logical_shape) != 2:
        raise CapsuleRestoreError("reference recurrent state must have rank two")
    recurrent_payload = payloads.get("logical:recurrent")
    if recurrent_payload is None or len(recurrent_payload) % 4:
        raise CapsuleRestoreError("reference recurrent bytes are missing or unaligned")
    count = len(recurrent_payload) // 4
    values = struct.unpack(f"<{count}i", recurrent_payload) if count else ()
    layers, width = recurrent_meta.logical_shape
    if layers * width != len(values):
        raise CapsuleRestoreError("recurrent payload shape is inconsistent")
    recurrent = tuple(tuple(values[layer * width : (layer + 1) * width]) for layer in range(layers))

    history = _json_object(payloads["logical:history"], "token history segment")
    expected_history = {
        "committed_output_token_ids": list(logical.token_history.committed_output_token_ids),
        "input_token_ids": list(logical.token_history.input_token_ids),
        "uncommitted_speculative_token_ids": list(
            logical.token_history.uncommitted_speculative_tokens
        ),
    }
    if history != expected_history:
        raise CapsuleRestoreError("token history bytes disagree with canonical logical state")
    sampler = _json_object(payloads["logical:sampler"], "sampler segment")
    guided = _json_object(payloads["logical:guided"], "guided decoding segment")
    delivery = _json_object(payloads["logical:delivery"], "client delivery segment")

    state_version = logical.extensions.root.get("sloforge.runtime/StateVersion")
    continuation_hash = logical.extensions.root.get("sloforge.runtime/ContinuationHash")
    if not isinstance(state_version, int) or isinstance(state_version, bool):
        raise UnsupportedReferenceABI("capsule lacks reference state version")
    if (
        not isinstance(continuation_hash, str)
        or re.fullmatch(r"[0-9a-f]{64}", continuation_hash) is None
    ):
        raise UnsupportedReferenceABI("capsule lacks reference continuation digest")
    first_layer = logical.attention.layers[0]
    if any(
        (layer.head_count, layer.kv_head_count, layer.head_dimension)
        != (first_layer.head_count, first_layer.kv_head_count, first_layer.head_dimension)
        for layer in logical.attention.layers
    ):
        raise CapsuleRestoreError("attention layers disagree on logical shape")
    client_ack = delivery.get("last_client_acknowledged_token_index")
    if not isinstance(client_ack, int) or isinstance(client_ack, bool):
        raise UnsupportedReferenceABI("reference client acknowledgment cannot be absent")
    terminal_value = delivery.get("terminal_status")
    try:
        terminal = ClientTerminalStatus(str(terminal_value))
    except ValueError as error:
        raise CapsuleRestoreError("invalid reference terminal status") from error
    guided_hash = guided.get("automaton_hash")
    if guided_hash != logical.guided_decoding.automaton_identity.value:
        raise CapsuleRestoreError("guided automaton bytes disagree with logical state")
    if str(guided.get("state")) != logical.guided_decoding.current_automaton_state:
        raise CapsuleRestoreError("guided automaton state disagrees with logical state")
    return LogicalStateManifest(
        schema_version="continuum.logical.runtime.v1",
        session_id=logical.execution.session_id,
        request_id=logical.execution.request_id,
        tenant_id=logical.execution.tenant_id,
        model=model,
        input_token_ids=tuple(logical.token_history.input_token_ids),
        committed_output_token_ids=tuple(logical.token_history.committed_output_token_ids),
        uncommitted_speculative_token_ids=tuple(
            logical.token_history.uncommitted_speculative_tokens
        ),
        attention_layer_count=len(logical.attention.layers),
        attention_head_count=first_layer.head_count,
        attention_kv_head_count=first_layer.kv_head_count,
        attention_head_dimension=first_layer.head_dimension,
        positional_encoding_semantics=first_layer.positional_encoding_semantics,
        attention_window_semantics=first_layer.attention_window_semantics,
        recurrent_state=recurrent,
        sampler=SamplerSnapshot(
            algorithm=str(sampler.get("algorithm")),
            seed=_require_int(sampler.get("seed"), "sampler seed"),
            counter=_require_int(sampler.get("counter"), "sampler counter"),
            temperature_milli=_require_int(sampler.get("temperature_milli"), "sampler temperature"),
            top_k=_require_int(sampler.get("top_k"), "sampler top-k"),
            top_p_millionths=_require_int(sampler.get("top_p_millionths"), "sampler top-p"),
        ),
        guided_decoding=GuidedDecodingSnapshot(
            automaton_id=str(guided.get("automaton_id")),
            automaton_hash=str(guided_hash),
            state=_require_int(guided.get("state"), "guided state"),
            accepted_prefix_length=_require_int(
                guided.get("accepted_prefix_length"), "guided prefix"
            ),
        ),
        client_delivery=ClientDeliverySnapshot(
            last_generated_token_index=_require_int(
                delivery.get("last_generated_token_index"), "generated watermark"
            ),
            last_gateway_committed_token_index=_require_int(
                delivery.get("last_gateway_committed_token_index"), "gateway watermark"
            ),
            last_client_acknowledged_token_index=client_ack,
            stream_owner_epoch=_require_int(delivery.get("stream_owner_epoch"), "owner epoch"),
            terminal_status=terminal,
        ),
        owner_epoch=logical.execution.current_owner_epoch,
        state_version=state_version,
        dirty_epoch=state_version,
        continuation_hash=continuation_hash,
    )


def _segments_and_pages(
    artifact: CheckpointArtifact,
    payloads: dict[str, bytes],
    layout: RuntimeLayout,
    logical: LogicalStateManifest,
) -> tuple[tuple[RuntimeSegment, ...], tuple[PageTableEntry, ...]]:
    physical = artifact.capsule.physical_state
    shards = {shard.shard_id: shard for shard in physical.shard_descriptors}
    access = {item.access_pattern_id: item for item in physical.access_patterns}
    canonical_pages = {item.segment_id: item for item in physical.page_tables}
    segments: list[RuntimeSegment] = []
    page_groups: dict[tuple[int, int, int, int, int], list[str]] = defaultdict(list)
    page_meta: dict[tuple[int, int, int, int, int], tuple[int, int, int, int]] = {}
    local_heads = logical.attention_kv_head_count // layout.tensor_parallel_degree
    for segment in physical.segments:
        payload = payloads[segment.segment_id]
        shard = shards[segment.shard_id]
        pattern = access[segment.access_pattern_id]
        match = _ATTENTION_ID.fullmatch(segment.segment_id)
        if segment.logical_state_reference == "state/attention-kv":
            if match is None:
                raise UnsupportedReferenceABI("attention segment ID is outside reference ABI v1")
            layer = int(match.group("layer"))
            page_id = int(match.group("page"))
            rank = int(match.group("rank"))
            part = match.group("part")
            if rank != shard.rank or shard.tensor_parallel_degree != layout.tensor_parallel_degree:
                raise CapsuleRestoreError("attention segment and shard placement disagree")
            table = canonical_pages.get(segment.segment_id)
            if table is None or len(table.entries) != 1:
                raise CapsuleRestoreError("attention segment lacks one page table entry")
            entry = table.entries[0]
            expected_page = f"{segment.segment_id}/page/{page_id}"
            if entry.physical_page_id != expected_page:
                raise CapsuleRestoreError("attention page identity is inconsistent")
            token_start = entry.logical_token_range.start
            token_end = entry.logical_token_range.end_exclusive
            if part == "k":
                state_kind = RuntimeStateKind.ATTENTION_KEY
                encoding = "little_endian_i32_token_major"
            elif part == "v":
                state_kind = RuntimeStateKind.ATTENTION_VALUE
                encoding = "little_endian_i32_token_major"
            else:
                state_kind = RuntimeStateKind.ATTENTION_PACKED_KV
                encoding = "little_endian_i32_head_major_packed_kv"
            key = (layer, page_id, rank, token_start, token_end)
            page_groups[key].append(segment.segment_id)
            page_meta[key] = (
                entry.page_version,
                segment.dirty_epoch,
                entry.owner_epoch,
                entry.copy_on_write_reference_count,
            )
            head_start = rank * local_heads
            head_end = head_start + local_heads
            semantic_id = (
                f"attention/layer/{layer}/{part}/tokens/{token_start}:{token_end}"
                if part != "kv"
                else f"attention/layer/{layer}/packed-kv/tokens/{token_start}:{token_end}"
            )
        else:
            if segment.segment_id not in _SCALAR_KINDS:
                raise UnsupportedReferenceABI("unknown reference scalar segment")
            state_kind = _SCALAR_KINDS[segment.segment_id]
            _dtype, encoding = _SCALAR_ENCODING[segment.segment_id]
            layer = None
            page_id = None
            token_start = 0
            token_end = len(logical.input_token_ids) + len(logical.committed_output_token_ids)
            head_start = head_end = 0
            semantic_id = f"logical/{segment.segment_id.removeprefix('logical:')}"
        dtype = (
            "int32"
            if state_kind
            in {
                RuntimeStateKind.ATTENTION_KEY,
                RuntimeStateKind.ATTENTION_VALUE,
                RuntimeStateKind.ATTENTION_PACKED_KV,
                RuntimeStateKind.RECURRENT,
            }
            else "structured"
        )
        descriptor = SegmentDescriptor(
            segment_id=segment.segment_id,
            semantic_id=semantic_id,
            state_kind=state_kind,
            logical_shape=segment.tensor_shape,
            dtype=dtype,
            encoding=encoding,
            layer=layer,
            shard_rank=shard.rank,
            shard_count=shard.tensor_parallel_degree,
            token_start=token_start,
            token_end=token_end,
            head_start=head_start,
            head_end=head_end,
            page_id=page_id,
            version=segment.current_version,
            dirty_epoch=segment.dirty_epoch,
            required_before_resume=pattern.required_before_resume,
            payload_bytes=len(payload),
            checksum=segment.checksum.value,
        )
        segments.append(RuntimeSegment(descriptor=descriptor, payload=payload))
    pages = tuple(
        PageTableEntry(
            logical_state_id=f"attention/layer/{key[0]}",
            layer=key[0],
            shard_rank=key[2],
            logical_token_start=key[3],
            logical_token_end=key[4],
            physical_page_id=key[1],
            segment_ids=tuple(sorted(segment_ids)),
            page_version=page_meta[key][0],
            dirty_epoch=page_meta[key][1],
            owner_epoch=page_meta[key][2],
            copy_on_write_refs=page_meta[key][3],
        )
        for key, segment_ids in sorted(page_groups.items())
    )
    return tuple(sorted(segments, key=lambda item: item.descriptor.segment_id)), pages


def restore_reference_capture(
    artifact: CheckpointArtifact,
    *,
    store: ContentStore,
    expected_tenant_id: str,
    expected_model: ModelContract,
) -> CapturedState:
    """Restore only the explicitly versioned reference ABI from authorized CAS bytes."""

    verify_checkpoint_artifact(artifact)
    if artifact.capsule.identity.tenant_id != expected_tenant_id:
        raise AuthorizationError("caller is not authorized for this capsule tenant")
    runtime = _runtime_identity(artifact)
    layout = _runtime_layout(artifact)
    model = _model_contract(artifact, expected_model)
    payloads = _read_payloads(artifact, store, expected_tenant_id)
    logical = _logical_manifest(artifact, payloads, model)
    segments, pages = _segments_and_pages(artifact, payloads, layout, logical)
    captured = CapturedState(
        handle=SnapshotHandle(
            snapshot_id=artifact.capsule.identity.capsule_id,
            session_id=logical.session_id,
            owner_epoch=logical.owner_epoch,
            state_version=logical.state_version,
            dirty_epoch=logical.dirty_epoch,
            segment_count=len(segments),
        ),
        runtime=runtime,
        layout=layout,
        logical=logical,
        segments=segments,
        page_table=pages,
    )
    try:
        captured.verify()
    except ValueError as error:
        raise CapsuleRestoreError("restored reference snapshot is structurally invalid") from error
    return captured
