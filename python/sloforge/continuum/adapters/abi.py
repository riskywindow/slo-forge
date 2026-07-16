"""Lossless bridge from a live adapter capture to the canonical Continuum ABI."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from hashlib import sha256
from typing import NamedTuple

from sloforge.continuum.adapters.sdk import (
    CapturedState,
    ClientTerminalStatus,
)
from sloforge.continuum.adapters.sdk import (
    LayoutKind as RuntimeLayoutKind,
)
from sloforge.continuum.adapters.sdk import (
    StateKind as RuntimeStateKind,
)
from sloforge.continuum.adapters.sdk import (
    StateSegment as RuntimeSegment,
)
from sloforge.continuum.ir import (
    AccessPatternDescriptor,
    AccessPatternKind,
    AttentionLayerState,
    AttentionState,
    ByteRange,
    ClientDeliveryState,
    CompressionKind,
    ConversionPermission,
    Digest,
    DTypeSemantics,
    EncryptionKind,
    ExactnessClass,
    ExecutionIdentity,
    Extensions,
    ExternalChunkReference,
    GuidedDecodingState,
    KVPacking,
    LayoutDescriptor,
    LayoutKind,
    LogicalComponentSize,
    LogicalStateSchema,
    Ordering,
    OwnershipScope,
    PageTableDescriptor,
    PageTableEntry,
    PhysicalStateLayout,
    PlacementDescriptor,
    Provenance,
    RecomputationPermission,
    RecurrentState,
    RuntimeIdentity,
    SamplerState,
    SegmentManifest,
    ShardDescriptor,
    StateComponentDescriptor,
    StateDependencyEdge,
    StateDependencyGraph,
    StateDependencyNode,
    StateKind,
    StateLifetime,
    StateSegment,
    StorageLocation,
    TerminalStatus,
    TokenHistoryState,
    TokenRange,
    UnknownStateHandling,
)
from sloforge.continuum.reference.codec import decode_segments
from sloforge.continuum.reference.models import HybridDecoderConfig


@dataclass(frozen=True, slots=True)
class CapturedChunk:
    """Tenant-scoped chunk payload awaiting publication by a content store."""

    segment_id: str
    chunk_id: str
    content_hash: Digest
    tenant_security_domain: str
    payload: bytes = field(repr=False)


class CapsuleInputs(NamedTuple):
    logical_state: LogicalStateSchema
    physical_state: PhysicalStateLayout
    chunks: tuple[CapturedChunk, ...]
    segment_manifests: tuple[SegmentManifest, ...]


def _digest_bytes(payload: bytes) -> Digest:
    return Digest(value=sha256(payload).hexdigest())


def _digest_value(value: object) -> Digest:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return _digest_bytes(payload)


def _provenance(captured: CapturedState, payload_digest: Digest) -> tuple[Provenance, ...]:
    return (
        Provenance(
            producer=captured.runtime.runtime_name,
            producer_version=captured.runtime.runtime_version,
            source_uri=f"runtime-capture://{captured.handle.snapshot_id}",
            source_digest=payload_digest,
            captured_at=f"logical-epoch:{captured.logical.state_version}",
            raw_evidence_uri=f"runtime-capture://{captured.handle.snapshot_id}/segments",
        ),
    )


def _component(
    captured: CapturedState,
    *,
    semantic_id: str,
    kind: StateKind,
    symbolic_shape: tuple[str, ...],
    update_semantics: str,
    integrity: Digest,
    compatibility_label: str,
    conversion_permissions: tuple[ConversionPermission, ...],
    recomputation_permission: RecomputationPermission,
    ownership: OwnershipScope = OwnershipScope.SESSION_OWNER,
) -> StateComponentDescriptor:
    return StateComponentDescriptor(
        semantic_id=semantic_id,
        schema_version="1.0.0",
        kind=kind,
        symbolic_shape=symbolic_shape,
        dtype_semantics=(
            DTypeSemantics.INT32
            if kind in {StateKind.ATTENTION_KV, StateKind.RECURRENT}
            else DTypeSemantics.OPAQUE
        ),
        update_semantics=update_semantics,
        lifetime=StateLifetime.SESSION,
        ownership=ownership,
        exactness_requirement=ExactnessClass.EXACT_SEMANTIC,
        conversion_permissions=conversion_permissions,
        recomputation_permission=recomputation_permission,
        compatibility_fingerprint=_digest_value(
            {
                "compatibility_label": compatibility_label,
                "model_state_producer": captured.logical.model.state_producer_hash,
                "schema": "1.0.0",
            }
        ),
        integrity_hash=integrity,
        provenance=_provenance(captured, integrity),
    )


def captured_to_logical_state(captured: CapturedState) -> LogicalStateSchema:
    """Construct canonical logical semantics from independently checked live bytes."""

    captured.verify()
    manifest = captured.logical
    config = HybridDecoderConfig(
        layout=captured.layout,
        layers=manifest.attention_layer_count,
        kv_heads=manifest.attention_kv_head_count,
        head_dimension=manifest.attention_head_dimension,
        recurrent_width=len(manifest.recurrent_state[0]),
        model=manifest.model,
    )
    decoded = decode_segments(
        source_layout=captured.layout,
        source_segments=captured.segments,
        manifest=manifest,
        destination_config=config,
        destination_session_id=manifest.session_id,
    )
    token_count = len(manifest.input_token_ids) + len(manifest.committed_output_token_ids)
    token_integrity = _digest_value(
        {
            "committed_output": manifest.committed_output_token_ids,
            "input": manifest.input_token_ids,
            "position_offset": 0,
        }
    )
    attention_integrity = _digest_value(
        {"keys": decoded.attention_keys, "values": decoded.attention_values}
    )
    recurrent_integrity = _digest_value(manifest.recurrent_state)
    sampler_integrity = _digest_value(
        {
            "algorithm": manifest.sampler.algorithm,
            "counter": manifest.sampler.counter,
            "seed": manifest.sampler.seed,
        }
    )
    guided_integrity = _digest_value(
        {
            "accepted_prefix": manifest.committed_output_token_ids,
            "automaton": manifest.guided_decoding.automaton_hash,
            "state": manifest.guided_decoding.state,
        }
    )
    client_integrity = _digest_value(
        {
            "client": manifest.client_delivery.last_client_acknowledged_token_index,
            "gateway": manifest.client_delivery.last_gateway_committed_token_index,
            "generated": manifest.client_delivery.last_generated_token_index,
            "owner": manifest.client_delivery.stream_owner_epoch,
            "terminal": manifest.client_delivery.terminal_status.value,
        }
    )

    token_component = _component(
        captured,
        semantic_id="state/token-history",
        kind=StateKind.TOKEN_HISTORY,
        symbolic_shape=("tokens",),
        update_semantics="append-only committed token sequence",
        integrity=token_integrity,
        compatibility_label="tokenizer-and-normalization-v1",
        conversion_permissions=(
            ConversionPermission.EXACT_RELAYOUT,
            ConversionPermission.RECOMPUTE,
        ),
        recomputation_permission=RecomputationPermission.FORBIDDEN,
    )
    attention_component = _component(
        captured,
        semantic_id="state/attention-kv",
        kind=StateKind.ATTENTION_KV,
        symbolic_shape=("layers", "tokens", "kv_heads", "head_dimension", "k_or_v"),
        update_semantics="append K/V derived from model state-producing weights",
        integrity=attention_integrity,
        compatibility_label=(
            f"{manifest.model.state_producer_hash}:{manifest.positional_encoding_semantics}:"
            f"{manifest.attention_window_semantics}"
        ),
        conversion_permissions=(
            ConversionPermission.EXACT_RELAYOUT,
            ConversionPermission.RECOMPUTE,
        ),
        recomputation_permission=RecomputationPermission.FROM_TOKEN_HISTORY,
    )
    recurrent_component = _component(
        captured,
        semantic_id="state/recurrent",
        kind=StateKind.RECURRENT,
        symbolic_shape=("layers", "recurrent_width"),
        update_semantics="continuum-recurrent-equation-v1",
        integrity=recurrent_integrity,
        compatibility_label=manifest.model.recurrent_update_hash,
        conversion_permissions=(
            ConversionPermission.EXACT_RELAYOUT,
            ConversionPermission.RECOMPUTE,
        ),
        recomputation_permission=RecomputationPermission.FROM_TOKEN_HISTORY,
    )
    sampler_component = _component(
        captured,
        semantic_id="state/sampler",
        kind=StateKind.SAMPLER,
        symbolic_shape=("counter_state",),
        update_semantics="counter increments once per generated token",
        integrity=sampler_integrity,
        compatibility_label=manifest.sampler.algorithm,
        conversion_permissions=(ConversionPermission.OPAQUE_COPY,),
        recomputation_permission=RecomputationPermission.FROM_TOKEN_HISTORY,
    )
    guided_component = _component(
        captured,
        semantic_id="state/guided-decoding",
        kind=StateKind.GUIDED_DECODING,
        symbolic_shape=("automaton_state",),
        update_semantics="deterministic automaton transition per accepted token",
        integrity=guided_integrity,
        compatibility_label=manifest.guided_decoding.automaton_hash,
        conversion_permissions=(ConversionPermission.OPAQUE_COPY,),
        recomputation_permission=RecomputationPermission.FROM_TOKEN_HISTORY,
    )
    client_component = _component(
        captured,
        semantic_id="state/client-delivery",
        kind=StateKind.CLIENT_DELIVERY,
        symbolic_shape=("watermarks",),
        update_semantics="monotonic gateway and client acknowledgment watermarks",
        integrity=client_integrity,
        compatibility_label="continuum-token-commit-v1",
        conversion_permissions=(ConversionPermission.OPAQUE_COPY,),
        recomputation_permission=RecomputationPermission.FORBIDDEN,
        ownership=OwnershipScope.EXTERNAL_COORDINATOR,
    )
    components = (
        token_component,
        attention_component,
        recurrent_component,
        sampler_component,
        guided_component,
        client_component,
    )
    nodes = tuple(
        StateDependencyNode(
            component_id=component.semantic_id,
            state_producing_fingerprint=component.compatibility_fingerprint,
        )
        for component in components
    )
    dependencies = tuple(
        StateDependencyEdge(
            upstream_component_id="state/token-history",
            downstream_component_id=target,
            dependency_semantics=semantics,
            invalidated_by_weight_change=invalidated,
        )
        for target, semantics, invalidated in (
            ("state/attention-kv", "tokens produce attention K/V", True),
            ("state/recurrent", "tokens drive recurrent updates", True),
            ("state/sampler", "accepted tokens advance RNG counter", False),
            ("state/guided-decoding", "accepted tokens advance automaton", False),
            ("state/client-delivery", "committed tokens advance delivery watermark", False),
        )
    )
    terminal = {
        ClientTerminalStatus.OPEN: TerminalStatus.OPEN,
        ClientTerminalStatus.COMPLETED: TerminalStatus.COMPLETED,
        ClientTerminalStatus.CANCELLED: TerminalStatus.CANCELLED,
        ClientTerminalStatus.ERRORED: TerminalStatus.ERRORED,
    }[manifest.client_delivery.terminal_status]
    return LogicalStateSchema(
        execution=ExecutionIdentity(
            session_id=manifest.session_id,
            request_id=manifest.request_id,
            tenant_id=manifest.tenant_id,
            model_identity=Digest(value=manifest.model.model_hash),
            tokenizer_identity=Digest(value=manifest.model.tokenizer_hash),
            adapter_identity=Digest(value=manifest.model.adapter_hash),
            creation_epoch=0,
            current_owner_epoch=manifest.owner_epoch,
        ),
        token_history=TokenHistoryState(
            component=token_component,
            input_token_ids=manifest.input_token_ids,
            committed_output_token_ids=manifest.committed_output_token_ids,
            uncommitted_speculative_tokens=manifest.uncommitted_speculative_token_ids,
            token_positions=tuple(range(token_count)),
            position_offset=0,
            attention_mask_semantics=manifest.attention_window_semantics,
            tokenizer_fingerprint=Digest(value=manifest.model.tokenizer_hash),
            normalization_contract="identity-token-id-sequence-v1",
        ),
        attention=AttentionState(
            component=attention_component,
            layers=tuple(
                AttentionLayerState(
                    layer_identity=f"hybrid-decoder/layer/{layer}",
                    logical_k_shape=(
                        token_count,
                        manifest.attention_kv_head_count,
                        manifest.attention_head_dimension,
                    ),
                    logical_v_shape=(
                        token_count,
                        manifest.attention_kv_head_count,
                        manifest.attention_head_dimension,
                    ),
                    token_range=TokenRange(start=0, end_exclusive=token_count),
                    head_count=manifest.attention_head_count,
                    kv_head_count=manifest.attention_kv_head_count,
                    head_dimension=manifest.attention_head_dimension,
                    positional_encoding_semantics=manifest.positional_encoding_semantics,
                    attention_window_semantics=manifest.attention_window_semantics,
                    dtype_semantics=DTypeSemantics.INT32,
                )
                for layer in range(manifest.attention_layer_count)
            ),
        ),
        recurrent=(
            RecurrentState(
                component=recurrent_component,
                state_identifier="hybrid-decoder/recurrent-state",
                layer_identity="hybrid-decoder/all-layers",
                logical_shape=(
                    manifest.attention_layer_count,
                    len(manifest.recurrent_state[0]),
                ),
                update_semantics="continuum-recurrent-equation-v1",
                dtype=DTypeSemantics.INT32,
                sequence_position=token_count,
                initialization_contract="seeded-continuum-recurrent-initializer-v1",
            ),
        ),
        sampler=SamplerState(
            component=sampler_component,
            sampling_algorithm="counter-based-guided-sampling",
            seed=manifest.sampler.seed,
            rng_algorithm=manifest.sampler.algorithm,
            rng_counter=manifest.sampler.counter,
            temperature=manifest.sampler.temperature_milli / 1000.0,
            top_k=manifest.sampler.top_k,
            top_p=manifest.sampler.top_p_millionths / 1_000_000.0,
            repetition_penalty=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            deterministic_required=True,
            implementation_independent_state=(
                f"seed={manifest.sampler.seed};counter={manifest.sampler.counter}"
            ),
        ),
        guided_decoding=GuidedDecodingState(
            component=guided_component,
            automaton_identity=Digest(value=manifest.guided_decoding.automaton_hash),
            current_automaton_state=str(manifest.guided_decoding.state),
            tokenizer_contract=Digest(value=manifest.model.tokenizer_hash),
            accepted_prefix=manifest.committed_output_token_ids,
        ),
        client_delivery=ClientDeliveryState(
            component=client_component,
            last_generated_token_index=manifest.client_delivery.last_generated_token_index,
            last_gateway_committed_token_index=(
                manifest.client_delivery.last_gateway_committed_token_index
            ),
            last_client_acknowledged_token_index=(
                manifest.client_delivery.last_client_acknowledged_token_index
            ),
            stream_owner_epoch=manifest.client_delivery.stream_owner_epoch,
            terminal_status=terminal,
        ),
        dependency_graph=StateDependencyGraph(nodes=nodes, edges=dependencies),
        unknown_state_handling=UnknownStateHandling.REJECT,
        exactness_contract=ExactnessClass.EXACT_SEMANTIC,
        extensions=Extensions(
            root={
                "sloforge.runtime/ContinuationHash": manifest.continuation_hash,
                "sloforge.runtime/StateVersion": manifest.state_version,
            }
        ),
    )


def _runtime_identity(captured: CapturedState) -> RuntimeIdentity:
    return RuntimeIdentity(
        runtime_name=captured.runtime.runtime_name,
        runtime_version=captured.runtime.runtime_version,
        adapter_version=captured.runtime.adapter_version,
        build_hash=Digest(value=captured.runtime.build_hash),
        dependency_versions=tuple(
            f"{name}={version}" for name, version in captured.runtime.dependency_versions
        ),
        target_hardware=(captured.runtime.target_hardware,),
    )


def _component_id(kind: RuntimeStateKind) -> str:
    if kind in {
        RuntimeStateKind.ATTENTION_KEY,
        RuntimeStateKind.ATTENTION_VALUE,
        RuntimeStateKind.ATTENTION_PACKED_KV,
    }:
        return "state/attention-kv"
    return {
        RuntimeStateKind.RECURRENT: "state/recurrent",
        RuntimeStateKind.SAMPLER: "state/sampler",
        RuntimeStateKind.GUIDED_DECODING: "state/guided-decoding",
        RuntimeStateKind.TOKEN_HISTORY: "state/token-history",
        RuntimeStateKind.CLIENT_DELIVERY: "state/client-delivery",
    }[kind]


def _strides(shape: tuple[int, ...]) -> tuple[int, ...]:
    strides: list[int] = []
    stride = 1
    for dimension in reversed(shape):
        strides.append(stride)
        stride *= max(1, dimension)
    return tuple(reversed(strides))


def _chunk_id(tenant_id: str, checksum: str) -> str:
    # Tenant scoping prevents equality disclosure through cross-tenant identifiers.
    return sha256(f"{tenant_id}:{checksum}".encode()).hexdigest()


def captured_to_physical_layout(
    captured: CapturedState,
) -> tuple[PhysicalStateLayout, tuple[CapturedChunk, ...], tuple[SegmentManifest, ...]]:
    """Map exact adapter bytes to ABI metadata plus tenant-scoped payload records."""

    captured.verify()
    manifest = captured.logical
    component_ids = (
        "state/token-history",
        "state/attention-kv",
        "state/recurrent",
        "state/sampler",
        "state/guided-decoding",
        "state/client-delivery",
    )
    grouped: dict[str, list[RuntimeSegment]] = {component_id: [] for component_id in component_ids}
    for segment in captured.segments:
        grouped[_component_id(segment.descriptor.state_kind)].append(segment)
    component_sizes = tuple(
        LogicalComponentSize(
            component_id=component_id,
            logical_size_bytes=sum(len(segment.payload) for segment in grouped[component_id]),
        )
        for component_id in component_ids
    )
    layout_id = f"layout/{captured.layout.kind.value}"
    packed_factor = (
        1 if captured.layout.kind is RuntimeLayoutKind.PAGED_TOKEN_MAJOR_SEPARATE_KV else 2
    )
    page_size_bytes = (
        captured.layout.page_size_tokens
        * (manifest.attention_kv_head_count // captured.layout.tensor_parallel_degree)
        * manifest.attention_head_dimension
        * 4
        * packed_factor
    )
    layout_descriptor = LayoutDescriptor(
        layout_id=layout_id,
        kind=LayoutKind.PAGED,
        page_size_bytes=page_size_bytes,
        alignment_bytes=captured.layout.alignment_bytes,
        ordering=(
            Ordering.TOKEN_MAJOR
            if captured.layout.kind is RuntimeLayoutKind.PAGED_TOKEN_MAJOR_SEPARATE_KV
            else Ordering.HEAD_MAJOR
        ),
        k_v_packing=(
            KVPacking.SEPARATE
            if captured.layout.kind is RuntimeLayoutKind.PAGED_TOKEN_MAJOR_SEPARATE_KV
            else KVPacking.PACKED_KV
        ),
    )
    placements = tuple(
        PlacementDescriptor(
            placement_id=f"placement/rank/{rank}",
            location=StorageLocation(
                memory_type="host",
                host_id="continuum-reference-host",
                device_id=device,
                memory_tier="simulated-device-memory",
                fault_domain=f"simulated-rank-{rank}",
            ),
        )
        for rank, device in enumerate(captured.layout.simulated_devices)
    )
    access_ids = {
        component_id: f"access/{component_id.removeprefix('state/')}"
        for component_id in component_ids
    }
    access_patterns = tuple(
        AccessPatternDescriptor(
            access_pattern_id=access_ids[component_id],
            kind=(
                AccessPatternKind.APPEND_ONLY
                if component_id in {"state/token-history", "state/attention-kv"}
                else AccessPatternKind.MUTABLE
            ),
            required_before_resume=True,
            streamable_before_use=False,
            recomputable=component_id in {"state/attention-kv", "state/recurrent"},
        )
        for component_id in component_ids
    )
    offsets = {component_id: 0 for component_id in component_ids}
    physical_segments: list[StateSegment] = []
    shards: list[ShardDescriptor] = []
    page_tables: list[PageTableDescriptor] = []
    chunks: list[CapturedChunk] = []
    manifests: list[SegmentManifest] = []
    for segment in sorted(captured.segments, key=lambda value: value.descriptor.segment_id):
        descriptor = segment.descriptor
        component_id = _component_id(descriptor.state_kind)
        logical_range = ByteRange(offset=offsets[component_id], length=len(segment.payload))
        offsets[component_id] += len(segment.payload)
        shard_id = f"shard/{descriptor.segment_id}"
        chunk_id = _chunk_id(manifest.tenant_id, descriptor.checksum)
        page_ids = (
            (f"{descriptor.segment_id}/page/{descriptor.page_id}",)
            if descriptor.page_id is not None
            else ()
        )
        shards.append(
            ShardDescriptor(
                shard_id=shard_id,
                tensor_parallel_degree=descriptor.shard_count,
                pipeline_stage=0,
                expert_parallel_group=0,
                data_parallel_replica=0,
                rank=descriptor.shard_rank,
                source_logical_slice=logical_range,
                destination_logical_slice=logical_range,
                shard_order=len(shards),
            )
        )
        physical_segments.append(
            StateSegment(
                logical_state_reference=component_id,
                segment_id=descriptor.segment_id,
                logical_byte_range=logical_range,
                physical_byte_range=ByteRange(offset=0, length=len(segment.payload)),
                tensor_shape=descriptor.logical_shape,
                tensor_strides=_strides(descriptor.logical_shape),
                storage_offset=0,
                allocation_id=f"allocation/{descriptor.segment_id}",
                page_ids=page_ids,
                chunk_ids=(chunk_id,),
                current_version=descriptor.version,
                dirty_epoch=descriptor.dirty_epoch,
                checksum=Digest(value=descriptor.checksum),
                compression=CompressionKind.NONE,
                encryption=EncryptionKind.NONE,
                layout_id=layout_id,
                shard_id=shard_id,
                placement_id=f"placement/rank/{descriptor.shard_rank}",
                access_pattern_id=access_ids[component_id],
            )
        )
        if page_ids:
            page_tables.append(
                PageTableDescriptor(
                    segment_id=descriptor.segment_id,
                    entries=(
                        PageTableEntry(
                            logical_token_range=TokenRange(
                                start=descriptor.token_start,
                                end_exclusive=descriptor.token_end,
                            ),
                            physical_page_id=page_ids[0],
                            page_version=descriptor.version,
                            owner_epoch=manifest.owner_epoch,
                            dirty=descriptor.dirty_epoch > 0,
                            copy_on_write_reference_count=1,
                        ),
                    ),
                )
            )
        content_hash = Digest(value=descriptor.checksum)
        chunks.append(
            CapturedChunk(
                segment_id=descriptor.segment_id,
                chunk_id=chunk_id,
                content_hash=content_hash,
                tenant_security_domain=manifest.tenant_id,
                payload=segment.payload,
            )
        )
        manifests.append(
            SegmentManifest(
                segment_id=descriptor.segment_id,
                segment_hash=content_hash,
                chunks=(
                    ExternalChunkReference(
                        chunk_id=chunk_id,
                        content_hash=content_hash,
                        size_bytes=len(segment.payload),
                        tenant_security_domain=manifest.tenant_id,
                        storage_uri=(
                            f"runtime-capture://{captured.handle.snapshot_id}/chunks/{chunk_id}"
                        ),
                        compression=CompressionKind.NONE,
                        encryption=EncryptionKind.NONE,
                    ),
                ),
            )
        )
    physical_plan_hash = _digest_value(
        {
            "kind": captured.layout.kind.value,
            "page_size_tokens": captured.layout.page_size_tokens,
            "simulated_devices": captured.layout.simulated_devices,
            "tensor_parallel_degree": captured.layout.tensor_parallel_degree,
        }
    )
    physical = PhysicalStateLayout(
        layout_id=layout_id,
        runtime=_runtime_identity(captured),
        physical_plan_hash=physical_plan_hash,
        owner_epoch=manifest.owner_epoch,
        logical_component_sizes=component_sizes,
        layout_descriptors=(layout_descriptor,),
        shard_descriptors=tuple(shards),
        placement_descriptors=placements,
        access_patterns=access_patterns,
        segments=tuple(physical_segments),
        page_tables=tuple(page_tables),
        reconstructible_runtime_state=(
            "scheduler queues",
            "process-local handles",
            "allocator internals",
            "runtime worker state",
        ),
        extensions=Extensions(
            root={
                "sloforge.runtime/PageSizeTokens": captured.layout.page_size_tokens,
                "sloforge.runtime/SimulatedDevices": list(captured.layout.simulated_devices),
            }
        ),
    )
    return physical, tuple(chunks), tuple(manifests)


def captured_to_capsule_inputs(captured: CapturedState) -> CapsuleInputs:
    """Return exact typed inputs ready for content-store publication and capsule sealing."""

    logical = captured_to_logical_state(captured)
    physical, chunks, manifests = captured_to_physical_layout(captured)
    return CapsuleInputs(
        logical_state=logical,
        physical_state=physical,
        chunks=chunks,
        segment_manifests=manifests,
    )
