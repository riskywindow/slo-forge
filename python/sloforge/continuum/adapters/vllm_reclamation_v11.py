"""Bounded software state pipeline for BranchFabric Experiment 004 v11.

The frozen v10 converter remains in :mod:`vllm_reclamation`.  This module
reuses its versioned canonical manifest and vLLM 0.23 allocation contract, but
never materializes a complete native or canonical CUDA intermediate.  All
payload ordering is deterministic and all asynchronous work is explicitly
fenced before source release or destination admission.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from sloforge.continuum.adapters.vllm_allocator_epochs import (
    VllmAllocatorEpochError,
    VllmAllocatorEpochProof,
    require_vllm_allocator_epoch_source,
)
from sloforge.continuum.adapters.vllm_reclamation import (
    CanonicalBranchTable,
    CanonicalKvPageDescriptor,
    CanonicalKvTransportManifest,
    CanonicalKvTransportState,
    NativeKvGeometry,
    StagedBranchAllocation,
    StagedGroupImport,
    Vllm0230RestoreStager,
    validate_native_tensors,
    validate_transport_identity,
)
from sloforge.continuum.adapters.vllm_reclamation_v11_accounting import (
    V11PassSpec,
    V11StatePassRecord,
    V11StatePassRecorder,
    validate_authoritative_pass_records,
)
from sloforge.continuum.adapters.vllm_reclamation_v11_sync import (
    V11AdmissionGateWitness,
    V11EngineStepBindingError,
    Vllm0230EngineStepBinding,
    require_production_gate_operation_v11,
    require_production_gate_v11,
)

V11_PIPELINE_VERSION: Literal["1.0.0"] = "1.0.0"
V11_INTEGRITY_DOMAIN = b"SLOForge/BranchFabric/exp004/v11/manifest\x00"
DEFAULT_MAXIMUM_CHUNK_BYTES = 28 * 1024 * 1024
_V11_DESTINATION_PROOF_SEAL = object()
_V11_SCRUB_PROOF_SEAL = object()


class V11IntegrityError(ValueError):
    """The optimized path failed closed on transport or destination bytes."""


class V11RuntimeTeardownRequired(RuntimeError):
    """Destination ownership could not be cleaned up safely after a failed restore."""

    def __init__(
        self,
        message: str,
        *,
        affected_block_ids: Sequence[int],
        teardown_evidence: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.affected_block_ids = tuple(sorted(set(affected_block_ids)))
        self.teardown_evidence = dict(teardown_evidence or {})


@dataclass(frozen=True, slots=True)
class V11PipelineConfig:
    seed: int
    expected_device_type: Literal["cpu", "cuda"]
    expected_cuda_device_index: int | None = None
    expected_gpu_uuid: str | None = None
    maximum_chunk_bytes: int = DEFAULT_MAXIMUM_CHUNK_BYTES
    buffer_count: int = 2
    pin_memory: bool = False
    asynchronous_copy: bool = False

    def __post_init__(self) -> None:
        if self.seed < 0:
            raise ValueError("v11 pipeline seed must be nonnegative")
        if self.maximum_chunk_bytes <= 0:
            raise ValueError("v11 maximum chunk bytes must be positive")
        if not 1 <= self.buffer_count <= 4:
            raise ValueError("v11 buffer count must be between one and four")
        if self.expected_device_type == "cpu" and (self.pin_memory or self.asynchronous_copy):
            raise ValueError("CPU v11 fixtures cannot silently request CUDA pinning or async copy")
        if self.expected_device_type == "cpu" and (
            self.expected_cuda_device_index is not None or self.expected_gpu_uuid is not None
        ):
            raise ValueError("CPU v11 fixtures cannot bind a CUDA device identity")
        if self.expected_device_type == "cuda" and (
            self.expected_cuda_device_index is None or self.expected_cuda_device_index < 0
        ):
            raise ValueError("CUDA v11 requires an exact nonnegative device index")
        if self.expected_device_type == "cuda" and (
            self.expected_gpu_uuid is None or not self.expected_gpu_uuid.startswith("GPU-")
        ):
            raise ValueError("CUDA v11 requires an exact GPU UUID")
        if self.expected_device_type == "cuda" and not self.pin_memory:
            raise ValueError("CUDA v11 capture requires an explicitly pinned final checkpoint")
        if self.expected_device_type == "cuda" and not self.asynchronous_copy:
            raise ValueError("CUDA v11 requires explicit asynchronous copy overlap")
        if self.asynchronous_copy and self.buffer_count < 2:
            raise ValueError("asynchronous v11 overlap requires at least two bounded slots")


@dataclass(frozen=True, slots=True)
class V11ChunkDescriptor:
    chunk_index: int
    logical_page_ids: tuple[str, ...]
    source_block_indices: tuple[int, ...]
    valid_tokens: int
    payload_offset_bytes: int
    payload_bytes: int

    @property
    def page_count(self) -> int:
        return len(self.logical_page_ids)


@dataclass(frozen=True, slots=True)
class V11IntegrityEvidence:
    algorithm: Literal["sha256"]
    payload_sha256: str
    page_sha256: tuple[tuple[str, str], ...]
    manifest_commitment_sha256: str
    payload_bytes_hashed: int
    page_bytes_hashed: int
    source_completion_fenced: bool


@dataclass(frozen=True, slots=True)
class V11VerifiedTransport:
    payload_object_id: int
    payload_data_ptr: int
    payload_version: int
    payload_sha256: str
    manifest_commitment_sha256: str
    page_count: int
    maximum_chunk_bytes: int
    preflight_pass_id_by_page: tuple[tuple[str, str], ...]
    verified_at_monotonic_ns: int


@dataclass(frozen=True, slots=True)
class V11PipelineStats:
    logical_state_bytes: int
    chunk_count: int
    maximum_chunk_bytes: int
    buffer_count: int
    checkpoint_resident_host_bytes: int
    peak_host_temporary_bytes: int
    peak_transform_temporary_bytes: int
    external_movement_bytes: int
    zero_tail_bytes: int
    full_page_zero_bytes_skipped: int


@dataclass(frozen=True, slots=True)
class V11CaptureResult:
    state: CanonicalKvTransportState
    chunks: tuple[V11ChunkDescriptor, ...]
    integrity: V11IntegrityEvidence
    stats: V11PipelineStats
    state_passes: tuple[V11StatePassRecord, ...]


@dataclass(frozen=True, slots=True)
class V11SubsetValidationEvidence:
    logical_page_ids: tuple[str, ...]
    ordered_page_destinations: tuple[tuple[str, int], ...]
    logical_bytes: int
    physical_bytes: int
    destination_bytes_read: int
    tail_bytes_zeroed: int
    transport_manifest_commitment_sha256: str
    destination_target_sha256: str
    destination_mapping_sha256: str
    allocation_epochs: tuple[tuple[int, int], ...]
    allocator_issued: bool
    full_byte_coverage: bool
    destination_exact_match: bool
    completion_fenced: bool
    passed: bool
    _proof_seal: object | None = field(default=None, repr=False, compare=False)


@dataclass(frozen=True, slots=True)
class V11ScrubEvidence:
    logical_page_ids: tuple[str, ...]
    ordered_page_destinations: tuple[tuple[str, int], ...]
    physical_bytes_zeroed: int
    transport_manifest_commitment_sha256: str
    destination_target_sha256: str
    destination_mapping_sha256: str
    allocation_epochs: tuple[tuple[int, int], ...]
    allocator_issued: bool
    destination_raw_zero: bool
    completion_fenced: bool
    passed: bool
    _proof_seal: object | None = field(default=None, repr=False, compare=False)


@dataclass(frozen=True, slots=True)
class V11RestoreResult:
    verified_transport: V11VerifiedTransport
    validation: V11SubsetValidationEvidence
    stats: V11PipelineStats
    state_passes: tuple[V11StatePassRecord, ...]


def _torch() -> Any:
    try:
        import torch  # type: ignore[import-not-found]
    except ImportError as error:  # pragma: no cover - runtime dependency.
        raise RuntimeError("v11 reclamation conversion requires PyTorch") from error
    return torch


def _required_identity(identity: Mapping[str, str]) -> dict[str, str]:
    required = {
        "model_id",
        "model_revision",
        "tokenizer_id",
        "tokenizer_revision",
        "dtype",
        "policy_epoch",
    }
    if set(identity) != required or any(not value for value in identity.values()):
        raise ValueError("transport identity fields are incomplete")
    if identity["dtype"] != "bfloat16":
        raise ValueError("v11 canonical transport supports bfloat16 only")
    return dict(identity)


def _branch_group_id(branch_tables: Sequence[CanonicalBranchTable]) -> str:
    payload = {
        "domain": "SLOForge/BranchFabric/exp004/v11/branch-group/v1",
        "branches": [
            {
                "logical_branch_id": table.logical_branch_id,
                "parent_logical_branch_id": table.parent_logical_branch_id,
                "token_history_sha256": table.token_history_sha256,
            }
            for table in branch_tables
        ],
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return f"branch-group-{digest[:24]}"


def _recorder_device(config: V11PipelineConfig) -> str:
    if config.expected_device_type == "cpu":
        return "cpu"
    assert config.expected_cuda_device_index is not None
    assert config.expected_gpu_uuid is not None
    return f"cuda:{config.expected_cuda_device_index}@{config.expected_gpu_uuid}"


def _require_recorder(
    config: V11PipelineConfig,
    *,
    branch_group: str,
    recorder: V11StatePassRecorder | None,
) -> V11StatePassRecorder:
    if recorder is None:
        return V11StatePassRecorder(
            device=_recorder_device(config),
            branch_group=branch_group,
        )
    if recorder.device != _recorder_device(config) or recorder.branch_group != branch_group:
        raise ValueError("v11 pass recorder differs from the exact device or branch group")
    return recorder


def _pass_spec(
    recorder: V11StatePassRecorder,
    *,
    pass_id: str,
    operation: str,
    source_representation: str,
    destination_representation: str,
    source_memory_tier: str,
    destination_memory_tier: str,
    logical_bytes: int,
    physical_bytes_read: int,
    physical_bytes_written: int,
    temporary_bytes: int,
    chunk_id: str,
    required_avoidable: Literal["REQUIRED", "AVOIDABLE", "MIXED", "DIAGNOSTIC"],
    external_transfer_bytes: int = 0,
    synchronization_dependency: tuple[str, ...] = (),
    required_physical_bytes: int | None = None,
    avoidable_physical_bytes: int | None = None,
    diagnostic_physical_bytes: int | None = None,
    physical_bytes_by_tier: tuple[tuple[str, int], ...] = (),
) -> V11PassSpec:
    return V11PassSpec(
        pass_id=pass_id,
        operation=operation,
        source_representation=source_representation,
        destination_representation=destination_representation,
        source_memory_tier=source_memory_tier,
        destination_memory_tier=destination_memory_tier,
        logical_bytes=logical_bytes,
        physical_bytes_read=physical_bytes_read,
        physical_bytes_written=physical_bytes_written,
        temporary_bytes=temporary_bytes,
        chunk_id=chunk_id,
        branch_group=recorder.branch_group,
        required_avoidable=required_avoidable,
        external_transfer_bytes=external_transfer_bytes,
        synchronization_dependency=synchronization_dependency,
        required_physical_bytes=required_physical_bytes,
        avoidable_physical_bytes=avoidable_physical_bytes,
        diagnostic_physical_bytes=diagnostic_physical_bytes,
        physical_bytes_by_tier=physical_bytes_by_tier,
    )


def _operation_context(
    recorder: V11StatePassRecorder,
    spec: V11PassSpec,
    *,
    stream: Any | None = None,
) -> Any:
    if recorder.device.startswith("cuda"):
        return recorder.cuda_pass(spec, stream=stream)
    return recorder.cpu_pass(spec)


def _chunk_id(chunk: V11ChunkDescriptor) -> str:
    return f"chunk-{chunk.chunk_index:04d}-offset-{chunk.payload_offset_bytes:012d}"


def _require_admission_gate(witness: V11AdmissionGateWitness, scheduler: Any, manager: Any) -> None:
    try:
        require_production_gate_v11(
            witness,
            scheduler,
            manager,
            operation="IMPORT_ADMISSION",
        )
    except V11EngineStepBindingError as error:
        raise V11RuntimeTeardownRequired(
            "v11 engine-step admission gate was not held; runtime teardown required",
            affected_block_ids=(),
        ) from error


def _manifest_commitment(manifest: CanonicalKvTransportManifest) -> str:
    return hashlib.sha256(V11_INTEGRITY_DOMAIN + manifest.canonical_bytes()).hexdigest()


def destination_target_identity_v11(
    tensors: tuple[Any, ...], geometry: NativeKvGeometry, config: V11PipelineConfig
) -> str:
    """Bind evidence to exact backing tensors, geometry, device index, and UUID."""

    normalized = _normalized_block_dimensions(tensors, geometry)
    device_identity = _preflight_device_identity(tensors, config)
    payload = {
        "domain": "SLOForge/BranchFabric/exp004/v11/destination-target/v1",
        "device": device_identity,
        "geometry": geometry.model_dump(mode="json"),
        "normalized_block_dimensions": normalized,
        "tensors": [
            {
                "data_ptr": int(tensor.data_ptr()),
                "dtype": str(tensor.dtype),
                "shape": list(tensor.shape),
                "stride": list(tensor.stride()),
            }
            for tensor in tensors
        ],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _ordered_destination_mapping(
    manifest: CanonicalKvTransportManifest, destinations: Mapping[str, int]
) -> tuple[tuple[str, int], ...]:
    ordered = tuple(
        (page.logical_page_id, destinations[page.logical_page_id])
        for page in manifest.pages
        if page.logical_page_id in destinations
    )
    if len(ordered) != len(destinations) or len({item[0] for item in ordered}) != len(ordered):
        raise ValueError("v11 destination mapping is not an ordered canonical subset")
    return ordered


def _destination_mapping_commitment(
    state: CanonicalKvTransportState,
    verified: V11VerifiedTransport,
    destinations: Mapping[str, int],
    *,
    destination_target_sha256: str,
    allocation_epoch_by_block: Mapping[int, int] | None,
) -> str:
    ordered_allocations = _ordered_destination_allocations(
        state.manifest,
        destinations,
        allocation_epoch_by_block,
    )
    payload = {
        "domain": "SLOForge/BranchFabric/exp004/v11/destination-mapping/v2",
        "manifest_commitment_sha256": _manifest_commitment(state.manifest),
        "payload_sha256": verified.payload_sha256,
        "page_destinations": _ordered_destination_mapping(state.manifest, destinations),
        "destination_target_sha256": destination_target_sha256,
        "allocator_issued_block_epochs": ordered_allocations,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _ordered_destination_allocations(
    manifest: CanonicalKvTransportManifest,
    destinations: Mapping[str, int],
    allocation_epoch_by_block: Mapping[int, int] | None,
) -> tuple[tuple[int, int], ...]:
    """Canonical block/epoch pairs for a destination subset.

    Empty evidence is permitted only for standalone CPU tensor equivalence;
    production stagers and every CUDA write require allocator-issued proof.
    """

    if allocation_epoch_by_block is None:
        return ()
    required = {int(item) for item in destinations.values()}
    available = {int(block_id): int(epoch) for block_id, epoch in allocation_epoch_by_block.items()}
    if not required.issubset(available):
        missing = sorted(required - set(available))
        raise ValueError(f"v11 destination lacks allocator epochs for blocks {missing}")
    if any(available[block_id] <= 0 for block_id in required):
        raise ValueError("v11 allocator-issued epochs must be positive")
    ordered_mapping = _ordered_destination_mapping(manifest, destinations)
    return tuple((block_id, available[block_id]) for _page_id, block_id in ordered_mapping)


def _cpu_bytes_view(tensor: Any) -> memoryview:
    if tensor.device.type != "cpu" or not tensor.is_contiguous():
        raise ValueError("integrity input must be a contiguous CPU tensor")
    return memoryview(tensor.numpy()).cast("B")


def _normalized_block_dimensions(
    tensors: tuple[Any, ...], geometry: NativeKvGeometry
) -> tuple[int, ...]:
    """Preflight every native layer before any destination mutation."""

    validate_native_tensors(tensors, geometry)
    return tuple(
        block_dim % tensor.ndim
        for tensor, block_dim in zip(tensors, geometry.block_dimensions, strict=True)
    )


def _preflight_device_identity(tensors: tuple[Any, ...], config: V11PipelineConfig) -> str:
    torch = _torch()
    device_types = {tensor.device.type for tensor in tensors}
    if device_types != {config.expected_device_type}:
        raise ValueError("v11 tensors differ from the explicit pipeline device type")
    if config.expected_device_type == "cpu":
        return "cpu"
    expected_index = config.expected_cuda_device_index
    assert expected_index is not None
    observed_indices = {
        tensor.device.index if tensor.device.index is not None else int(torch.cuda.current_device())
        for tensor in tensors
    }
    if observed_indices != {expected_index}:
        raise ValueError("v11 tensors differ from the exact configured CUDA device index")
    properties = torch.cuda.get_device_properties(expected_index)
    observed_uuid = getattr(properties, "uuid", None)
    if observed_uuid is None:
        raw_uuid = getattr(torch.cuda, "_raw_device_uuid_nvml", None)
        if callable(raw_uuid):
            uuids = raw_uuid()
            observed_uuid = uuids[expected_index] if uuids is not None else None
    if isinstance(observed_uuid, bytes):
        observed_uuid = observed_uuid.decode("ascii")
    if observed_uuid is None:
        raise RuntimeError("CUDA runtime did not expose a GPU UUID for exact-device validation")
    observed_uuid_text = str(observed_uuid)
    if not observed_uuid_text.startswith("GPU-"):
        observed_uuid_text = f"GPU-{observed_uuid_text}"
    if observed_uuid_text != config.expected_gpu_uuid:
        raise ValueError("v11 CUDA runtime UUID differs from the configured GPU UUID")
    return f"cuda:{expected_index}@{observed_uuid_text}"


def _raw_tensor_equal(left: Any, right: Any) -> bool:
    """Compare exact tensor bits, including NaN payloads and signed zero."""

    torch = _torch()
    if left.shape != right.shape or left.dtype is not right.dtype:
        return False
    if left.element_size() != 2:
        raise ValueError("v11 raw equality currently requires two-byte KV elements")
    return bool(torch.equal(left.view(torch.int16), right.view(torch.int16)))


def _raw_tensor_is_zero(tensor: Any) -> bool:
    torch = _torch()
    if tensor.element_size() != 2:
        raise ValueError("v11 raw zero validation currently requires two-byte KV elements")
    raw = tensor.view(torch.int16)
    return bool(torch.count_nonzero(raw).item() == 0)


def _snapshot_and_verify_host_chunk(
    state: CanonicalKvTransportState,
    chunk: V11ChunkDescriptor,
    *,
    verified_transport: V11VerifiedTransport,
    pin_memory: bool,
    recorder: V11StatePassRecorder,
) -> Any:
    """Take an owned bounded snapshot, then authenticate every byte in it."""

    torch = _torch()
    source = state.payload[
        chunk.payload_offset_bytes : chunk.payload_offset_bytes + chunk.payload_bytes
    ]
    snapshot = torch.empty(
        chunk.payload_bytes,
        dtype=torch.uint8,
        device="cpu",
        pin_memory=pin_memory,
    )
    chunk_id = _chunk_id(chunk)
    preflight_by_page = dict(verified_transport.preflight_pass_id_by_page)
    if len(preflight_by_page) != len(verified_transport.preflight_pass_id_by_page):
        raise V11IntegrityError("v11 preflight provenance contains ambiguous page coverage")
    preflight_dependencies = tuple(
        dict.fromkeys(
            preflight_by_page[logical_page_id]
            for logical_page_id in chunk.logical_page_ids
            if logical_page_id in preflight_by_page
        )
    )
    if len(preflight_dependencies) == 0 or any(
        logical_page_id not in preflight_by_page for logical_page_id in chunk.logical_page_ids
    ):
        raise V11IntegrityError("v11 preflight provenance is missing consumed page coverage")
    snapshot_pass_id = f"restore:{chunk_id}:owned-host-snapshot"
    with recorder.cpu_pass(
        _pass_spec(
            recorder,
            pass_id=snapshot_pass_id,
            operation="owned_snapshot",
            source_representation="CONTINUUM_CANONICAL_TRANSPORT_LAYOUT",
            destination_representation="BOUNDED_AUTHENTICATED_HOST_CHUNK",
            source_memory_tier=(
                "HOST_PINNED" if recorder.device.startswith("cuda") else "HOST_PAGEABLE"
            ),
            destination_memory_tier=("HOST_PINNED" if pin_memory else "HOST_PAGEABLE"),
            logical_bytes=chunk.payload_bytes,
            physical_bytes_read=chunk.payload_bytes,
            physical_bytes_written=chunk.payload_bytes,
            temporary_bytes=chunk.payload_bytes,
            chunk_id=chunk_id,
            required_avoidable="AVOIDABLE",
            synchronization_dependency=preflight_dependencies,
        )
    ):
        snapshot.copy_(source, non_blocking=False)
    view = _cpu_bytes_view(snapshot)
    pages = {page.logical_page_id: page for page in state.manifest.pages}
    with recorder.cpu_pass(
        _pass_spec(
            recorder,
            pass_id=f"restore:{chunk_id}:consumption-sha256",
            operation="sha256_pages",
            source_representation="BOUNDED_AUTHENTICATED_HOST_CHUNK",
            destination_representation="SHA256_CONSUMPTION_AUTHENTICATION",
            source_memory_tier=("HOST_PINNED" if pin_memory else "HOST_PAGEABLE"),
            destination_memory_tier="HOST_PAGEABLE_METADATA",
            logical_bytes=chunk.payload_bytes,
            physical_bytes_read=chunk.payload_bytes,
            physical_bytes_written=0,
            temporary_bytes=0,
            chunk_id=chunk_id,
            required_avoidable="AVOIDABLE",
            synchronization_dependency=(snapshot_pass_id,),
        )
    ):
        relative = 0
        for logical_page_id in chunk.logical_page_ids:
            page = pages[logical_page_id]
            digest = hashlib.sha256(view[relative : relative + page.payload_bytes]).hexdigest()
            if digest != page.content_sha256:
                raise V11IntegrityError(
                    f"v11 consumed canonical page digest mismatch: {logical_page_id}"
                )
            relative += page.payload_bytes
        if relative != chunk.payload_bytes:
            raise V11IntegrityError("v11 consumed chunk coverage does not close exactly")
    return snapshot


def _validate_page_order(
    page_order: Sequence[tuple[str, int, int, tuple[str, ...]]],
    geometry: NativeKvGeometry,
) -> None:
    if not page_order:
        raise ValueError("v11 page order cannot be empty")
    logical_ids = [item[0] for item in page_order]
    source_ids = [item[1] for item in page_order]
    if len(logical_ids) != len(set(logical_ids)):
        raise ValueError("v11 page order contains duplicate logical pages")
    if len(source_ids) != len(set(source_ids)):
        raise ValueError("v11 page order contains duplicate source pages")
    for logical_id, block_index, valid_tokens, owners in page_order:
        if not logical_id or not owners:
            raise ValueError("v11 source page requires identity and ownership")
        if not 0 <= block_index < geometry.num_blocks:
            raise ValueError("v11 source page index is outside the native pool")
        if not 0 < valid_tokens <= geometry.block_size_tokens:
            raise ValueError("v11 page valid-token extent is invalid")


def plan_transport_chunks(
    page_order: Sequence[tuple[str, int, int, tuple[str, ...]]],
    geometry: NativeKvGeometry,
    *,
    maximum_chunk_bytes: int,
) -> tuple[V11ChunkDescriptor, ...]:
    """Split deterministic canonical order at page and valid-extent boundaries."""

    _validate_page_order(page_order, geometry)
    if maximum_chunk_bytes <= 0:
        raise ValueError("maximum chunk bytes must be positive")
    chunks: list[V11ChunkDescriptor] = []
    cursor = 0
    index = 0
    while index < len(page_order):
        valid_tokens = page_order[index][2]
        per_page_bytes = valid_tokens * geometry.logical_token_bytes
        if per_page_bytes > maximum_chunk_bytes:
            raise ValueError("one canonical page exceeds the bounded chunk budget")
        end = index
        chunk_bytes = 0
        while end < len(page_order):
            candidate = page_order[end]
            if candidate[2] != valid_tokens or chunk_bytes + per_page_bytes > maximum_chunk_bytes:
                break
            chunk_bytes += per_page_bytes
            end += 1
        selected = page_order[index:end]
        chunks.append(
            V11ChunkDescriptor(
                chunk_index=len(chunks),
                logical_page_ids=tuple(item[0] for item in selected),
                source_block_indices=tuple(item[1] for item in selected),
                valid_tokens=valid_tokens,
                payload_offset_bytes=cursor,
                payload_bytes=chunk_bytes,
            )
        )
        cursor += chunk_bytes
        index = end
    expected = sum(item[2] * geometry.logical_token_bytes for item in page_order)
    if cursor != expected:
        raise RuntimeError("v11 chunk plan does not conserve canonical payload bytes")
    return tuple(chunks)


def _plan_manifest_chunks(
    manifest: CanonicalKvTransportManifest,
    *,
    selected_ids: set[str],
    maximum_chunk_bytes: int,
) -> tuple[V11ChunkDescriptor, ...]:
    known = {page.logical_page_id for page in manifest.pages}
    if not selected_ids or not selected_ids.issubset(known):
        raise ValueError("v11 restore subset contains no valid canonical page set")
    selected = [page for page in manifest.pages if page.logical_page_id in selected_ids]
    if len(selected) != len(selected_ids):
        raise ValueError("v11 restore subset coverage is not bijective")
    chunks: list[V11ChunkDescriptor] = []
    index = 0
    while index < len(selected):
        first = selected[index]
        if first.payload_bytes > maximum_chunk_bytes:
            raise ValueError("one restore page exceeds the bounded chunk budget")
        end = index + 1
        chunk_bytes = first.payload_bytes
        expected_offset = first.payload_offset_bytes + first.payload_bytes
        while end < len(selected):
            candidate = selected[end]
            if (
                candidate.valid_tokens != first.valid_tokens
                or candidate.payload_offset_bytes != expected_offset
                or chunk_bytes + candidate.payload_bytes > maximum_chunk_bytes
            ):
                break
            chunk_bytes += candidate.payload_bytes
            expected_offset += candidate.payload_bytes
            end += 1
        pages = selected[index:end]
        chunks.append(
            V11ChunkDescriptor(
                chunk_index=len(chunks),
                logical_page_ids=tuple(page.logical_page_id for page in pages),
                source_block_indices=(),
                valid_tokens=first.valid_tokens,
                payload_offset_bytes=first.payload_offset_bytes,
                payload_bytes=chunk_bytes,
            )
        )
        index = end
    return tuple(chunks)


def _pack_source_chunk(
    tensors: tuple[Any, ...],
    geometry: NativeKvGeometry,
    chunk: V11ChunkDescriptor,
    *,
    block_dimensions: tuple[int, ...],
    recorder: V11StatePassRecorder,
) -> tuple[Any, int]:
    torch = _torch()
    device = tensors[0].device
    packed = torch.empty(
        (
            chunk.page_count,
            len(tensors),
            chunk.valid_tokens,
            2,
            geometry.kv_heads,
            geometry.head_size,
        ),
        dtype=torch.bfloat16,
        device=device,
    )
    chunk_id = _chunk_id(chunk)
    index_bytes = len(chunk.source_block_indices) * 8
    index_pass_id = f"capture:{chunk_id}:source-index-upload"
    with _operation_context(
        recorder,
        _pass_spec(
            recorder,
            pass_id=index_pass_id,
            operation="index_upload",
            source_representation="VLLM_NATIVE_BLOCK_IDS",
            destination_representation="CUDA_INDEX_VECTOR",
            source_memory_tier="HOST_PAGEABLE_METADATA",
            destination_memory_tier=(
                "GPU_HBM_TEMPORARY" if recorder.device.startswith("cuda") else "HOST_PAGEABLE"
            ),
            logical_bytes=index_bytes,
            physical_bytes_read=index_bytes,
            physical_bytes_written=index_bytes,
            temporary_bytes=index_bytes,
            chunk_id=chunk_id,
            required_avoidable="AVOIDABLE",
            external_transfer_bytes=index_bytes if recorder.device.startswith("cuda") else 0,
        ),
    ):
        source_indices = torch.tensor(chunk.source_block_indices, dtype=torch.long, device=device)
    target_labels = ("token", "kv", "head", "dim")
    maximum_layer_gather_bytes = 0
    for layer_index, (tensor, block_dim) in enumerate(zip(tensors, block_dimensions, strict=True)):
        pool = tensor.movedim(block_dim, 0)
        layer_physical_bytes = chunk.page_count * geometry.physical_page_bytes // len(tensors)
        gather_pass_id = f"capture:{chunk_id}:layer-{layer_index:02d}:native-gather"
        with _operation_context(
            recorder,
            _pass_spec(
                recorder,
                pass_id=gather_pass_id,
                operation="index_select",
                source_representation="VLLM_NATIVE_PAGED_LAYOUT",
                destination_representation="BOUNDED_NATIVE_LAYER_CHUNK",
                source_memory_tier=(
                    "GPU_HBM_NATIVE_POOL" if recorder.device.startswith("cuda") else "HOST_PAGEABLE"
                ),
                destination_memory_tier=(
                    "GPU_HBM_TEMPORARY" if recorder.device.startswith("cuda") else "HOST_PAGEABLE"
                ),
                logical_bytes=chunk.payload_bytes // len(tensors),
                physical_bytes_read=layer_physical_bytes,
                physical_bytes_written=layer_physical_bytes,
                temporary_bytes=layer_physical_bytes,
                chunk_id=chunk_id,
                required_avoidable="MIXED",
                required_physical_bytes=layer_physical_bytes,
                avoidable_physical_bytes=layer_physical_bytes,
                diagnostic_physical_bytes=0,
                synchronization_dependency=(index_pass_id,),
            ),
        ):
            gathered = pool.index_select(0, source_indices)
        permutation = (
            0,
            *tuple(1 + geometry.source_axis_labels.index(label) for label in target_labels),
        )
        canonical = gathered.permute(*permutation)[:, : chunk.valid_tokens]
        layer_logical_bytes = canonical.numel() * canonical.element_size()
        repack_pass_id = f"capture:{chunk_id}:layer-{layer_index:02d}:bounded-repack"
        with _operation_context(
            recorder,
            _pass_spec(
                recorder,
                pass_id=repack_pass_id,
                operation="copy_repack",
                source_representation="BOUNDED_NATIVE_LAYER_CHUNK",
                destination_representation="CONTINUUM_CANONICAL_TRANSPORT_LAYOUT",
                source_memory_tier=(
                    "GPU_HBM_TEMPORARY" if recorder.device.startswith("cuda") else "HOST_PAGEABLE"
                ),
                destination_memory_tier=(
                    "GPU_HBM_TEMPORARY" if recorder.device.startswith("cuda") else "HOST_PAGEABLE"
                ),
                logical_bytes=layer_logical_bytes,
                physical_bytes_read=layer_logical_bytes,
                physical_bytes_written=layer_logical_bytes,
                temporary_bytes=packed.numel() * packed.element_size(),
                chunk_id=chunk_id,
                required_avoidable="AVOIDABLE",
                synchronization_dependency=(gather_pass_id,),
            ),
        ):
            packed[:, layer_index].copy_(canonical)
        maximum_layer_gather_bytes = max(
            maximum_layer_gather_bytes,
            gathered.numel() * gathered.element_size(),
        )
        del canonical, gathered
    return packed, maximum_layer_gather_bytes


def _page_hashes_from_chunk(
    host_payload: Any,
    chunk: V11ChunkDescriptor,
    *,
    whole: Any,
    page_hashes: dict[str, str],
    recorder: V11StatePassRecorder,
    synchronization_dependency: tuple[str, ...],
) -> None:
    host_chunk = host_payload[
        chunk.payload_offset_bytes : chunk.payload_offset_bytes + chunk.payload_bytes
    ]
    chunk_view = _cpu_bytes_view(host_chunk)
    chunk_id = _chunk_id(chunk)
    with recorder.cpu_pass(
        _pass_spec(
            recorder,
            pass_id=f"capture:{chunk_id}:sha256",
            operation="sha256_whole_and_pages",
            source_representation="CONTINUUM_CANONICAL_TRANSPORT_LAYOUT",
            destination_representation="SHA256_MANIFEST_COMMITMENTS",
            source_memory_tier=(
                "HOST_PINNED" if recorder.device.startswith("cuda") else "HOST_PAGEABLE"
            ),
            destination_memory_tier="HOST_PAGEABLE_METADATA",
            logical_bytes=chunk.payload_bytes,
            physical_bytes_read=chunk.payload_bytes * 2,
            physical_bytes_written=0,
            temporary_bytes=0,
            chunk_id=chunk_id,
            required_avoidable="AVOIDABLE",
            synchronization_dependency=synchronization_dependency,
        )
    ):
        whole.update(chunk_view)
        per_page = chunk.payload_bytes // chunk.page_count
        if per_page * chunk.page_count != chunk.payload_bytes:
            raise RuntimeError("v11 chunk bytes do not divide across pages")
        for page_index, logical_page_id in enumerate(chunk.logical_page_ids):
            start = page_index * per_page
            page_hashes[logical_page_id] = hashlib.sha256(
                chunk_view[start : start + per_page]
            ).hexdigest()


def _build_transport_manifest_v11(
    *,
    geometry: NativeKvGeometry,
    page_order: tuple[tuple[str, int, int, tuple[str, ...]], ...],
    branch_tables: tuple[CanonicalBranchTable, ...],
    identity: Mapping[str, str],
    page_hashes: Mapping[str, str],
    payload_sha256: str,
) -> CanonicalKvTransportManifest:
    pages: list[CanonicalKvPageDescriptor] = []
    cursor = 0
    for logical_id, _block_index, valid_tokens, owners in page_order:
        size = valid_tokens * geometry.logical_token_bytes
        logical_positions = {
            branch.logical_page_ids.index(logical_id)
            for branch in branch_tables
            if logical_id in branch.logical_page_ids
        }
        if len(logical_positions) != 1:
            raise ValueError("v11 logical page occurs at inconsistent branch positions")
        pages.append(
            CanonicalKvPageDescriptor(
                logical_page_id=logical_id,
                logical_token_start=next(iter(logical_positions)) * geometry.block_size_tokens,
                valid_tokens=valid_tokens,
                payload_offset_bytes=cursor,
                payload_bytes=size,
                content_sha256=page_hashes[logical_id],
                branch_ids=tuple(sorted(owners)),
                shared_root=len(owners) > 1,
            )
        )
        cursor += size
    return CanonicalKvTransportManifest(
        model_id=identity["model_id"],
        model_revision=identity["model_revision"],
        tokenizer_id=identity["tokenizer_id"],
        tokenizer_revision=identity["tokenizer_revision"],
        dtype=identity["dtype"],
        policy_epoch=identity["policy_epoch"],
        block_size_tokens=geometry.block_size_tokens,
        layer_names=geometry.layer_names,
        kv_heads=geometry.kv_heads,
        head_size=geometry.head_size,
        element_size_bytes=geometry.element_size_bytes,
        pages=tuple(pages),
        branches=branch_tables,
        logical_state_bytes=cursor,
        physical_source_bytes=len(page_order) * geometry.physical_page_bytes,
        payload_sha256=payload_sha256,
    )


def capture_native_to_transport_v11(
    tensors: tuple[Any, ...],
    geometry: NativeKvGeometry,
    *,
    page_order: tuple[tuple[str, int, int, tuple[str, ...]], ...],
    branch_tables: tuple[CanonicalBranchTable, ...],
    identity: Mapping[str, str],
    config: V11PipelineConfig,
    engine_step_gate: V11AdmissionGateWitness | None = None,
    pass_recorder: V11StatePassRecorder | None = None,
) -> V11CaptureResult:
    """Directly gather bounded native chunks into the final host checkpoint."""

    torch = _torch()
    if config.expected_device_type == "cuda":
        if engine_step_gate is None:
            raise V11EngineStepBindingError(
                "CUDA v11 export requires the production engine-step capture gate"
            )
        require_production_gate_operation_v11(
            engine_step_gate,
            operation="EXPORT_CAPTURE",
        )
    normalized_block_dimensions = _normalized_block_dimensions(tensors, geometry)
    _preflight_device_identity(tensors, config)
    checked_identity = _required_identity(identity)
    branch_group = _branch_group_id(branch_tables)
    recorder = _require_recorder(config, branch_group=branch_group, recorder=pass_recorder)
    if any(tensor.dtype is not torch.bfloat16 for tensor in tensors):
        raise ValueError("v11 source tensors differ from bfloat16 transport identity")
    chunks = plan_transport_chunks(
        page_order,
        geometry,
        maximum_chunk_bytes=config.maximum_chunk_bytes,
    )
    logical_bytes = sum(chunk.payload_bytes for chunk in chunks)
    placeholder_hashes = {item[0]: "0" * 64 for item in page_order}
    _build_transport_manifest_v11(
        geometry=geometry,
        page_order=page_order,
        branch_tables=branch_tables,
        identity=checked_identity,
        page_hashes=placeholder_hashes,
        payload_sha256="0" * 64,
    )
    host_payload = torch.empty(
        logical_bytes,
        dtype=torch.uint8,
        device="cpu",
        pin_memory=config.pin_memory,
    )
    whole = hashlib.sha256()
    page_hashes: dict[str, str] = {}
    peak_transform_bytes = 0

    if config.expected_device_type == "cuda":
        copy_stream = torch.cuda.Stream(device=tensors[0].device)
        compute_stream = torch.cuda.current_stream(device=tensors[0].device)
        pending: deque[tuple[Any, Any, V11ChunkDescriptor]] = deque()

        def finish_oldest() -> None:
            event, packed, completed_chunk = pending.popleft()
            completed_chunk_id = _chunk_id(completed_chunk)
            copied_pass_id = f"capture:{completed_chunk_id}:d2h"
            with recorder.cpu_pass(
                _pass_spec(
                    recorder,
                    pass_id=f"capture:{completed_chunk_id}:completion-fence",
                    operation="cuda_event_synchronize",
                    source_representation="CUDA_EVENT",
                    destination_representation="HOST_COMPLETION_FENCE",
                    source_memory_tier="GPU_CONTROL",
                    destination_memory_tier="HOST_CONTROL",
                    logical_bytes=completed_chunk.payload_bytes,
                    physical_bytes_read=0,
                    physical_bytes_written=0,
                    temporary_bytes=0,
                    chunk_id=completed_chunk_id,
                    required_avoidable="REQUIRED",
                    synchronization_dependency=(copied_pass_id,),
                )
            ):
                event.synchronize()
            _page_hashes_from_chunk(
                host_payload,
                completed_chunk,
                whole=whole,
                page_hashes=page_hashes,
                recorder=recorder,
                synchronization_dependency=(f"capture:{completed_chunk_id}:completion-fence",),
            )
            del packed

        for chunk in chunks:
            while len(pending) >= config.buffer_count:
                finish_oldest()
            packed, gather_bytes = _pack_source_chunk(
                tensors,
                geometry,
                chunk,
                block_dimensions=normalized_block_dimensions,
                recorder=recorder,
            )
            pending_bytes = sum(item[1].numel() * item[1].element_size() for item in pending)
            peak_transform_bytes = max(
                peak_transform_bytes,
                pending_bytes + packed.numel() * packed.element_size() + gather_bytes,
            )
            transform_done = torch.cuda.Event()
            transform_done.record(compute_stream)
            with torch.cuda.stream(copy_stream):
                copy_stream.wait_event(transform_done)
                target = host_payload[
                    chunk.payload_offset_bytes : chunk.payload_offset_bytes + chunk.payload_bytes
                ]
                chunk_id = _chunk_id(chunk)
                dependencies = tuple(
                    f"capture:{chunk_id}:layer-{layer_index:02d}:bounded-repack"
                    for layer_index in range(len(tensors))
                )
                with recorder.cuda_pass(
                    _pass_spec(
                        recorder,
                        pass_id=f"capture:{chunk_id}:d2h",
                        operation="d2h",
                        source_representation="CONTINUUM_CANONICAL_TRANSPORT_LAYOUT",
                        destination_representation="CONTINUUM_CANONICAL_TRANSPORT_LAYOUT",
                        source_memory_tier="GPU_HBM_TEMPORARY",
                        destination_memory_tier="HOST_PINNED",
                        logical_bytes=chunk.payload_bytes,
                        physical_bytes_read=chunk.payload_bytes,
                        physical_bytes_written=chunk.payload_bytes,
                        temporary_bytes=0,
                        chunk_id=chunk_id,
                        required_avoidable="REQUIRED",
                        external_transfer_bytes=chunk.payload_bytes,
                        synchronization_dependency=dependencies,
                    ),
                    stream=copy_stream,
                ):
                    target.copy_(packed.view(torch.uint8).reshape(-1), non_blocking=True)
                packed.record_stream(copy_stream)
                copied = torch.cuda.Event()
                copied.record(copy_stream)
            pending.append((copied, packed, chunk))
        while pending:
            finish_oldest()
    else:
        for chunk in chunks:
            packed, gather_bytes = _pack_source_chunk(
                tensors,
                geometry,
                chunk,
                block_dimensions=normalized_block_dimensions,
                recorder=recorder,
            )
            peak_transform_bytes = max(
                peak_transform_bytes,
                packed.numel() * packed.element_size() + gather_bytes,
            )
            target = host_payload[
                chunk.payload_offset_bytes : chunk.payload_offset_bytes + chunk.payload_bytes
            ]
            chunk_id = _chunk_id(chunk)
            with recorder.cpu_pass(
                _pass_spec(
                    recorder,
                    pass_id=f"capture:{chunk_id}:host-copy",
                    operation="host_copy",
                    source_representation="CONTINUUM_CANONICAL_TRANSPORT_LAYOUT",
                    destination_representation="CONTINUUM_CANONICAL_TRANSPORT_LAYOUT",
                    source_memory_tier="HOST_PAGEABLE",
                    destination_memory_tier="HOST_PAGEABLE",
                    logical_bytes=chunk.payload_bytes,
                    physical_bytes_read=chunk.payload_bytes,
                    physical_bytes_written=chunk.payload_bytes,
                    temporary_bytes=0,
                    chunk_id=chunk_id,
                    required_avoidable="REQUIRED",
                    synchronization_dependency=tuple(
                        f"capture:{chunk_id}:layer-{layer_index:02d}:bounded-repack"
                        for layer_index in range(len(tensors))
                    ),
                )
            ):
                target.copy_(packed.view(torch.uint8).reshape(-1), non_blocking=False)
            _page_hashes_from_chunk(
                host_payload,
                chunk,
                whole=whole,
                page_hashes=page_hashes,
                recorder=recorder,
                synchronization_dependency=(f"capture:{chunk_id}:host-copy",),
            )

    payload_sha256 = whole.hexdigest()
    if config.expected_device_type == "cuda":
        assert engine_step_gate is not None
        require_production_gate_operation_v11(
            engine_step_gate,
            operation="EXPORT_CAPTURE",
        )
    manifest = _build_transport_manifest_v11(
        geometry=geometry,
        page_order=page_order,
        branch_tables=branch_tables,
        identity=checked_identity,
        page_hashes=page_hashes,
        payload_sha256=payload_sha256,
    )
    state = CanonicalKvTransportState(manifest=manifest, payload=host_payload)
    integrity = V11IntegrityEvidence(
        algorithm="sha256",
        payload_sha256=payload_sha256,
        page_sha256=tuple((page.logical_page_id, page.content_sha256) for page in manifest.pages),
        manifest_commitment_sha256=_manifest_commitment(manifest),
        payload_bytes_hashed=logical_bytes,
        page_bytes_hashed=logical_bytes,
        source_completion_fenced=True,
    )
    passes = recorder.resolve()
    validate_authoritative_pass_records(
        passes,
        expected_device=_recorder_device(config),
        expected_branch_group=branch_group,
    )
    return V11CaptureResult(
        state=state,
        chunks=chunks,
        integrity=integrity,
        stats=V11PipelineStats(
            logical_state_bytes=logical_bytes,
            chunk_count=len(chunks),
            maximum_chunk_bytes=max(chunk.payload_bytes for chunk in chunks),
            buffer_count=config.buffer_count,
            checkpoint_resident_host_bytes=logical_bytes,
            peak_host_temporary_bytes=0,
            peak_transform_temporary_bytes=peak_transform_bytes,
            external_movement_bytes=logical_bytes if config.expected_device_type == "cuda" else 0,
            zero_tail_bytes=0,
            full_page_zero_bytes_skipped=0,
        ),
        state_passes=passes,
    )


def verify_transport_streaming_v11(
    state: CanonicalKvTransportState,
    *,
    maximum_chunk_bytes: int,
    pass_recorder: V11StatePassRecorder | None = None,
) -> V11VerifiedTransport:
    """Verify every retained byte without a complete pageable ``bytes`` copy."""

    torch = _torch()
    if not isinstance(state.payload, torch.Tensor):
        raise TypeError("v11 canonical payload must be a torch tensor")
    if (
        state.payload.device.type != "cpu"
        or state.payload.dtype is not torch.uint8
        or not state.payload.is_contiguous()
        or state.payload.numel() != state.manifest.logical_state_bytes
    ):
        raise V11IntegrityError("v11 canonical payload shape or memory domain is invalid")
    chunks = _plan_manifest_chunks(
        state.manifest,
        selected_ids={page.logical_page_id for page in state.manifest.pages},
        maximum_chunk_bytes=maximum_chunk_bytes,
    )
    branch_group = _branch_group_id(state.manifest.branches)
    recorder = pass_recorder or V11StatePassRecorder(
        device="cpu",
        branch_group=branch_group,
    )
    if recorder.branch_group != branch_group:
        raise ValueError("v11 verifier pass recorder differs from the checkpoint branch group")
    whole = hashlib.sha256()
    pages = {page.logical_page_id: page for page in state.manifest.pages}
    seen: list[str] = []
    preflight_pass_id_by_page: list[tuple[str, str]] = []
    for chunk in chunks:
        host_chunk = state.payload[
            chunk.payload_offset_bytes : chunk.payload_offset_bytes + chunk.payload_bytes
        ]
        chunk_view = _cpu_bytes_view(host_chunk)
        chunk_id = _chunk_id(chunk)
        preflight_pass_id = f"restore:preflight:{chunk_id}:sha256"
        with recorder.cpu_pass(
            _pass_spec(
                recorder,
                pass_id=preflight_pass_id,
                operation="sha256_whole_and_pages",
                source_representation="CONTINUUM_CANONICAL_TRANSPORT_LAYOUT",
                destination_representation="SHA256_VERIFIED_TRANSPORT_TOKEN",
                source_memory_tier=(
                    "HOST_PINNED" if recorder.device.startswith("cuda") else "HOST_PAGEABLE"
                ),
                destination_memory_tier="HOST_PAGEABLE_METADATA",
                logical_bytes=chunk.payload_bytes,
                physical_bytes_read=chunk.payload_bytes * 2,
                physical_bytes_written=0,
                temporary_bytes=0,
                chunk_id=chunk_id,
                required_avoidable="AVOIDABLE",
            )
        ):
            whole.update(chunk_view)
            relative = 0
            for logical_page_id in chunk.logical_page_ids:
                page = pages[logical_page_id]
                digest = hashlib.sha256(
                    chunk_view[relative : relative + page.payload_bytes]
                ).hexdigest()
                if digest != page.content_sha256:
                    raise V11IntegrityError(
                        f"v11 canonical page digest mismatch: {logical_page_id}"
                    )
                relative += page.payload_bytes
                seen.append(logical_page_id)
                preflight_pass_id_by_page.append((logical_page_id, preflight_pass_id))
            if relative != chunk.payload_bytes:
                raise V11IntegrityError("v11 chunk page ranges do not close exactly")
    if tuple(seen) != tuple(page.logical_page_id for page in state.manifest.pages):
        raise V11IntegrityError("v11 chunk sequence is missing, duplicated, or reordered")
    digest = whole.hexdigest()
    if digest != state.manifest.payload_sha256:
        raise V11IntegrityError("v11 canonical payload digest mismatch")
    return V11VerifiedTransport(
        payload_object_id=id(state.payload),
        payload_data_ptr=int(state.payload.data_ptr()),
        payload_version=int(state.payload._version),
        payload_sha256=digest,
        manifest_commitment_sha256=_manifest_commitment(state.manifest),
        page_count=len(seen),
        maximum_chunk_bytes=maximum_chunk_bytes,
        preflight_pass_id_by_page=tuple(preflight_pass_id_by_page),
        verified_at_monotonic_ns=time.monotonic_ns(),
    )


def _validate_verified_token(
    state: CanonicalKvTransportState,
    verified: V11VerifiedTransport,
) -> None:
    if verified.maximum_chunk_bytes <= 0:
        raise V11IntegrityError("v11 prevalidated transport token has an invalid chunk bound")
    try:
        expected_chunks = _plan_manifest_chunks(
            state.manifest,
            selected_ids={page.logical_page_id for page in state.manifest.pages},
            maximum_chunk_bytes=verified.maximum_chunk_bytes,
        )
    except (TypeError, ValueError) as error:
        raise V11IntegrityError(
            "v11 prevalidated transport token has invalid preflight provenance"
        ) from error
    expected_preflight_by_page = tuple(
        (page_id, f"restore:preflight:{_chunk_id(chunk)}:sha256")
        for chunk in expected_chunks
        for page_id in chunk.logical_page_ids
    )
    if (
        verified.payload_object_id != id(state.payload)
        or verified.payload_data_ptr != int(state.payload.data_ptr())
        or verified.payload_version != int(state.payload._version)
        or verified.payload_sha256 != state.manifest.payload_sha256
        or verified.manifest_commitment_sha256 != _manifest_commitment(state.manifest)
        or verified.page_count != len(state.manifest.pages)
        or verified.maximum_chunk_bytes <= 0
        or verified.preflight_pass_id_by_page != expected_preflight_by_page
    ):
        raise V11IntegrityError("v11 prevalidated transport token does not bind this state")


def _consecutive_runs(values: Sequence[int]) -> tuple[tuple[int, int, int], ...]:
    if not values:
        return ()
    runs: list[tuple[int, int, int]] = []
    source_start = 0
    destination_start = values[0]
    for index in range(1, len(values) + 1):
        if index == len(values) or values[index] != values[index - 1] + 1:
            runs.append((source_start, index, destination_start))
            if index < len(values):
                source_start = index
                destination_start = values[index]
    return tuple(runs)


def _write_validate_device_chunk(
    device_bytes: Any,
    *,
    chunk: V11ChunkDescriptor,
    manifest: CanonicalKvTransportManifest,
    tensors: tuple[Any, ...],
    geometry: NativeKvGeometry,
    destinations: Mapping[str, int],
    block_dimensions: tuple[int, ...],
    recorder: V11StatePassRecorder,
) -> tuple[int, int]:
    torch = _torch()
    canonical = device_bytes.view(torch.bfloat16).view(
        chunk.page_count,
        len(tensors),
        chunk.valid_tokens,
        2,
        geometry.kv_heads,
        geometry.head_size,
    )
    destination_ids = tuple(destinations[page_id] for page_id in chunk.logical_page_ids)
    if len(destination_ids) != len(set(destination_ids)):
        raise ValueError("v11 restore chunk aliases destination blocks")
    canonical_labels = ("token", "kv", "head", "dim")
    inverse = tuple(canonical_labels.index(label) for label in geometry.source_axis_labels)
    chunk_id = _chunk_id(chunk)
    input_ready_pass_id = (
        f"restore:{chunk_id}:h2d"
        if recorder.device.startswith("cuda")
        else f"restore:{chunk_id}:consumption-sha256"
    )
    tail_bytes_zeroed = 0
    destination_bytes_read = 0
    for layer_index, (tensor, block_dim) in enumerate(zip(tensors, block_dimensions, strict=True)):
        pool = tensor.movedim(block_dim, 0)
        layer_write_pass_ids: list[str] = []
        source_layer = canonical[:, layer_index].permute(
            (0, *tuple(1 + index for index in inverse))
        )
        if chunk.valid_tokens == geometry.block_size_tokens:
            index_bytes = len(destination_ids) * 8
            index_pass_id = f"restore:{chunk_id}:layer-{layer_index:02d}:index-upload"
            with _operation_context(
                recorder,
                _pass_spec(
                    recorder,
                    pass_id=index_pass_id,
                    operation="index_upload",
                    source_representation="VLLM_NATIVE_BLOCK_IDS",
                    destination_representation="CUDA_INDEX_VECTOR",
                    source_memory_tier="HOST_PAGEABLE_METADATA",
                    destination_memory_tier=(
                        "GPU_HBM_TEMPORARY"
                        if recorder.device.startswith("cuda")
                        else "HOST_PAGEABLE"
                    ),
                    logical_bytes=index_bytes,
                    physical_bytes_read=index_bytes,
                    physical_bytes_written=index_bytes,
                    temporary_bytes=index_bytes,
                    chunk_id=chunk_id,
                    required_avoidable="AVOIDABLE",
                    external_transfer_bytes=(
                        index_bytes if recorder.device.startswith("cuda") else 0
                    ),
                    synchronization_dependency=(input_ready_pass_id,),
                ),
            ):
                index = torch.tensor(destination_ids, dtype=torch.long, device=tensor.device)
            source_layer_bytes = source_layer.numel() * source_layer.element_size()
            write_pass_id = f"restore:{chunk_id}:layer-{layer_index:02d}:native-scatter"
            with _operation_context(
                recorder,
                _pass_spec(
                    recorder,
                    pass_id=write_pass_id,
                    operation="index_copy",
                    source_representation="CONTINUUM_CANONICAL_TRANSPORT_LAYOUT",
                    destination_representation="VLLM_NATIVE_PAGED_LAYOUT",
                    source_memory_tier=(
                        "GPU_HBM_TEMPORARY"
                        if recorder.device.startswith("cuda")
                        else "HOST_PAGEABLE"
                    ),
                    destination_memory_tier=(
                        "GPU_HBM_NATIVE_POOL"
                        if recorder.device.startswith("cuda")
                        else "HOST_PAGEABLE"
                    ),
                    logical_bytes=source_layer_bytes,
                    physical_bytes_read=source_layer_bytes,
                    physical_bytes_written=source_layer_bytes,
                    temporary_bytes=0,
                    chunk_id=chunk_id,
                    required_avoidable="REQUIRED",
                    synchronization_dependency=(index_pass_id,),
                ),
            ):
                pool.index_copy_(0, index, source_layer)
            layer_write_pass_ids.append(write_pass_id)
        else:
            token_axis = geometry.source_axis_labels.index("token")
            for page_index, destination in enumerate(destination_ids):
                destination_page = pool[destination]
                page_bytes = source_layer[page_index].numel() * source_layer.element_size()
                write_pass_id = (
                    f"restore:{chunk_id}:layer-{layer_index:02d}:"
                    f"page-{page_index:03d}:native-scatter"
                )
                with _operation_context(
                    recorder,
                    _pass_spec(
                        recorder,
                        pass_id=write_pass_id,
                        operation="partial_page_copy",
                        source_representation="CONTINUUM_CANONICAL_TRANSPORT_LAYOUT",
                        destination_representation="VLLM_NATIVE_PAGED_LAYOUT",
                        source_memory_tier=(
                            "GPU_HBM_TEMPORARY"
                            if recorder.device.startswith("cuda")
                            else "HOST_PAGEABLE"
                        ),
                        destination_memory_tier=(
                            "GPU_HBM_NATIVE_POOL"
                            if recorder.device.startswith("cuda")
                            else "HOST_PAGEABLE"
                        ),
                        logical_bytes=page_bytes,
                        physical_bytes_read=page_bytes,
                        physical_bytes_written=page_bytes,
                        temporary_bytes=0,
                        chunk_id=chunk_id,
                        required_avoidable="REQUIRED",
                        synchronization_dependency=(input_ready_pass_id,),
                    ),
                ):
                    destination_page.narrow(token_axis, 0, chunk.valid_tokens).copy_(
                        source_layer[page_index]
                    )
                layer_write_pass_ids.append(write_pass_id)
                tail_tokens = geometry.block_size_tokens - chunk.valid_tokens
                if tail_tokens:
                    tail = destination_page.narrow(token_axis, chunk.valid_tokens, tail_tokens)
                    tail_bytes = tail.numel() * tail.element_size()
                    zero_pass_id = (
                        f"restore:{chunk_id}:layer-{layer_index:02d}:"
                        f"page-{page_index:03d}:tail-zero"
                    )
                    with _operation_context(
                        recorder,
                        _pass_spec(
                            recorder,
                            pass_id=zero_pass_id,
                            operation="zero_tail",
                            source_representation="ZERO_SCALAR",
                            destination_representation="VLLM_NATIVE_PAGED_LAYOUT",
                            source_memory_tier="NONE",
                            destination_memory_tier=(
                                "GPU_HBM_NATIVE_POOL"
                                if recorder.device.startswith("cuda")
                                else "HOST_PAGEABLE"
                            ),
                            logical_bytes=0,
                            physical_bytes_read=0,
                            physical_bytes_written=tail_bytes,
                            temporary_bytes=0,
                            chunk_id=chunk_id,
                            required_avoidable="REQUIRED",
                            synchronization_dependency=(write_pass_id,),
                        ),
                    ):
                        tail.zero_()
                    tail_bytes_zeroed += tail_bytes
                    tail_validate_pass_id = (
                        f"restore:{chunk_id}:layer-{layer_index:02d}:"
                        f"page-{page_index:03d}:tail-validate"
                    )
                    with _operation_context(
                        recorder,
                        _pass_spec(
                            recorder,
                            pass_id=tail_validate_pass_id,
                            operation="validate_zero_tail",
                            source_representation="VLLM_NATIVE_PAGED_LAYOUT",
                            destination_representation="BOOLEAN_VALIDATION",
                            source_memory_tier=(
                                "GPU_HBM_NATIVE_POOL"
                                if recorder.device.startswith("cuda")
                                else "HOST_PAGEABLE"
                            ),
                            destination_memory_tier="HOST_CONTROL",
                            logical_bytes=0,
                            physical_bytes_read=tail_bytes,
                            physical_bytes_written=0,
                            temporary_bytes=0,
                            chunk_id=chunk_id,
                            required_avoidable="AVOIDABLE",
                            external_transfer_bytes=(
                                1 if recorder.device.startswith("cuda") else 0
                            ),
                            synchronization_dependency=(zero_pass_id,),
                        ),
                    ):
                        tail_is_zero = _raw_tensor_is_zero(tail)
                    if not tail_is_zero:
                        raise V11IntegrityError(
                            f"v11 destination tail zero validation failed: chunk {chunk.chunk_index}"
                        )
        token_axis_with_page = 1 + geometry.source_axis_labels.index("token")
        for source_start, source_end, destination_start in _consecutive_runs(destination_ids):
            destination_view = pool.narrow(0, destination_start, source_end - source_start)
            if chunk.valid_tokens != geometry.block_size_tokens:
                destination_view = destination_view.narrow(
                    token_axis_with_page, 0, chunk.valid_tokens
                )
            expected = source_layer[source_start:source_end]
            expected_bytes = expected.numel() * expected.element_size()
            destination_bytes_read += expected_bytes
            validate_pass_id = (
                f"restore:{chunk_id}:layer-{layer_index:02d}:"
                f"run-{source_start:03d}-{source_end:03d}:raw-bit-validate"
            )
            with _operation_context(
                recorder,
                _pass_spec(
                    recorder,
                    pass_id=validate_pass_id,
                    operation="raw_bit_equal",
                    source_representation="VLLM_NATIVE_PAGED_LAYOUT+EXPECTED_CANONICAL_VIEW",
                    destination_representation="BOOLEAN_VALIDATION",
                    source_memory_tier=(
                        "GPU_HBM_NATIVE_POOL+GPU_HBM_TEMPORARY"
                        if recorder.device.startswith("cuda")
                        else "HOST_PAGEABLE"
                    ),
                    destination_memory_tier="HOST_CONTROL",
                    logical_bytes=expected_bytes,
                    physical_bytes_read=expected_bytes * 2,
                    physical_bytes_written=0,
                    temporary_bytes=0,
                    chunk_id=chunk_id,
                    required_avoidable="AVOIDABLE",
                    external_transfer_bytes=(1 if recorder.device.startswith("cuda") else 0),
                    physical_bytes_by_tier=(
                        (
                            ("GPU_HBM_NATIVE_POOL", expected_bytes),
                            ("GPU_HBM_TEMPORARY", expected_bytes),
                            ("LINK_CONTROL", 1),
                        )
                        if recorder.device.startswith("cuda")
                        else (("HOST_PAGEABLE", expected_bytes * 2),)
                    ),
                    synchronization_dependency=tuple(layer_write_pass_ids),
                ),
            ):
                exact = _raw_tensor_equal(destination_view, expected)
            if not exact:
                failing_page = chunk.logical_page_ids[source_start]
                raise V11IntegrityError(
                    f"v11 destination exact comparison failed: {failing_page} layer {layer_index}"
                )
    expected_logical = sum(
        page.payload_bytes
        for page in manifest.pages
        if page.logical_page_id in set(chunk.logical_page_ids)
    )
    if destination_bytes_read != expected_logical:
        raise RuntimeError("v11 destination validation did not read every logical byte")
    return destination_bytes_read, tail_bytes_zeroed


def scrub_native_pages_v11(
    tensors: tuple[Any, ...],
    geometry: NativeKvGeometry,
    *,
    state: CanonicalKvTransportState,
    verified_transport: V11VerifiedTransport,
    config: V11PipelineConfig,
    allocation_epoch_by_block: Mapping[int, int] | None = None,
    destination_block_indices: Mapping[str, int],
    engine_step_gate: V11AdmissionGateWitness | None = None,
    pass_recorder: V11StatePassRecorder | None = None,
) -> V11ScrubEvidence:
    """Raw-zero every allocated page and fence completion before allocator reuse."""

    torch = _torch()
    if config.expected_device_type == "cuda":
        if engine_step_gate is None:
            raise V11EngineStepBindingError(
                "CUDA v11 scrub requires the production import/admission gate"
            )
        require_production_gate_operation_v11(
            engine_step_gate,
            operation="IMPORT_ADMISSION",
        )
    branch_group = _branch_group_id(state.manifest.branches)
    recorder = _require_recorder(config, branch_group=branch_group, recorder=pass_recorder)
    _validate_verified_token(state, verified_transport)
    block_dimensions = _normalized_block_dimensions(tensors, geometry)
    destination_target_sha256 = destination_target_identity_v11(tensors, geometry, config)
    if not destination_block_indices:
        raise ValueError("v11 failure scrub requires a nonempty destination map")
    destination_ids = tuple(destination_block_indices.values())
    if len(destination_ids) != len(set(destination_ids)) or any(
        not 0 <= item < geometry.num_blocks for item in destination_ids
    ):
        raise ValueError("v11 failure scrub destinations must be unique and in range")
    allocation_epochs = _ordered_destination_allocations(
        state.manifest, destination_block_indices, allocation_epoch_by_block
    )
    if config.expected_device_type == "cuda" and not allocation_epochs:
        raise ValueError("CUDA v11 scrub requires allocator-issued epochs")
    scrub_identity = hashlib.sha256(
        "\x00".join(sorted(destination_block_indices)).encode("utf-8")
    ).hexdigest()[:16]
    chunk_id = f"scrub-{scrub_identity}"
    index_bytes = len(destination_ids) * 8
    index_pass_id = f"restore:{chunk_id}:destination-index-upload"
    with _operation_context(
        recorder,
        _pass_spec(
            recorder,
            pass_id=index_pass_id,
            operation="destination_index_upload",
            source_representation="DESTINATION_BLOCK_IDS",
            destination_representation="CUDA_LONG_INDEX_VECTOR",
            source_memory_tier="HOST_PAGEABLE",
            destination_memory_tier=(
                "GPU_HBM_TEMPORARY" if recorder.device.startswith("cuda") else "HOST_PAGEABLE"
            ),
            logical_bytes=index_bytes,
            physical_bytes_read=index_bytes,
            physical_bytes_written=index_bytes,
            temporary_bytes=index_bytes,
            chunk_id=chunk_id,
            required_avoidable="DIAGNOSTIC",
            external_transfer_bytes=(index_bytes if recorder.device.startswith("cuda") else 0),
        ),
    ):
        index = torch.tensor(destination_ids, dtype=torch.long, device=tensors[0].device)
    zero_pass_ids: list[str] = []
    for layer_index, (tensor, block_dim) in enumerate(zip(tensors, block_dimensions, strict=True)):
        pool = tensor.movedim(block_dim, 0)
        layer_bytes = len(destination_ids) * pool[0].numel() * pool[0].element_size()
        zero_pass_id = f"restore:{chunk_id}:layer-{layer_index:02d}:native-zero"
        with _operation_context(
            recorder,
            _pass_spec(
                recorder,
                pass_id=zero_pass_id,
                operation="native_page_zero",
                source_representation="SCALAR_ZERO+CUDA_LONG_INDEX_VECTOR",
                destination_representation="VLLM_NATIVE_PAGED_LAYOUT",
                source_memory_tier=(
                    "GPU_REGISTER+GPU_HBM_TEMPORARY"
                    if recorder.device.startswith("cuda")
                    else "HOST_CONTROL+HOST_PAGEABLE"
                ),
                destination_memory_tier=(
                    "GPU_HBM_NATIVE_POOL" if recorder.device.startswith("cuda") else "HOST_PAGEABLE"
                ),
                logical_bytes=layer_bytes,
                physical_bytes_read=index_bytes,
                physical_bytes_written=layer_bytes,
                temporary_bytes=0,
                chunk_id=chunk_id,
                required_avoidable="DIAGNOSTIC",
                synchronization_dependency=(index_pass_id,),
                physical_bytes_by_tier=(
                    (
                        ("GPU_HBM_TEMPORARY", index_bytes),
                        ("GPU_HBM_NATIVE_POOL", layer_bytes),
                    )
                    if recorder.device.startswith("cuda")
                    else (("HOST_PAGEABLE", index_bytes + layer_bytes),)
                ),
            ),
        ):
            pool.index_fill_(0, index, 0)
        zero_pass_ids.append(zero_pass_id)
    if tensors[0].device.type == "cuda":
        with recorder.cpu_pass(
            _pass_spec(
                recorder,
                pass_id=f"restore:{chunk_id}:zero-completion-fence",
                operation="cuda_stream_synchronize",
                source_representation="CUDA_STREAM",
                destination_representation="HOST_COMPLETION_FENCE",
                source_memory_tier="GPU_CONTROL",
                destination_memory_tier="HOST_CONTROL",
                logical_bytes=len(destination_ids) * geometry.physical_page_bytes,
                physical_bytes_read=0,
                physical_bytes_written=0,
                temporary_bytes=0,
                chunk_id=chunk_id,
                required_avoidable="DIAGNOSTIC",
                synchronization_dependency=tuple(zero_pass_ids),
            )
        ):
            torch.cuda.current_stream(device=tensors[0].device).synchronize()
    for layer_index, (tensor, block_dim) in enumerate(zip(tensors, block_dimensions, strict=True)):
        pool = tensor.movedim(block_dim, 0)
        layer_bytes = len(destination_ids) * pool[0].numel() * pool[0].element_size()
        with _operation_context(
            recorder,
            _pass_spec(
                recorder,
                pass_id=f"restore:{chunk_id}:layer-{layer_index:02d}:raw-zero-verify",
                operation="index_select_and_raw_zero_verify",
                source_representation="VLLM_NATIVE_PAGED_LAYOUT",
                destination_representation="RAW_ZERO_VALIDATION",
                source_memory_tier=(
                    "GPU_HBM_NATIVE_POOL" if recorder.device.startswith("cuda") else "HOST_PAGEABLE"
                ),
                destination_memory_tier="HOST_CONTROL",
                logical_bytes=layer_bytes,
                physical_bytes_read=layer_bytes * 2,
                physical_bytes_written=layer_bytes,
                temporary_bytes=layer_bytes,
                chunk_id=chunk_id,
                required_avoidable="DIAGNOSTIC",
                external_transfer_bytes=(1 if recorder.device.startswith("cuda") else 0),
                physical_bytes_by_tier=(
                    (
                        ("GPU_HBM_NATIVE_POOL", layer_bytes),
                        ("GPU_HBM_TEMPORARY", layer_bytes * 2),
                        ("LINK_CONTROL", 1),
                    )
                    if recorder.device.startswith("cuda")
                    else (("HOST_PAGEABLE", layer_bytes * 3),)
                ),
                synchronization_dependency=(
                    f"restore:{chunk_id}:layer-{layer_index:02d}:native-zero",
                ),
            ),
        ):
            scrubbed_is_zero = _raw_tensor_is_zero(pool.index_select(0, index))
        if not scrubbed_is_zero:
            raise V11IntegrityError(
                "v11 failure scrub verification failed; destination runtime teardown required"
            )
    return V11ScrubEvidence(
        logical_page_ids=tuple(sorted(destination_block_indices)),
        ordered_page_destinations=_ordered_destination_mapping(
            state.manifest, destination_block_indices
        ),
        physical_bytes_zeroed=len(destination_ids) * geometry.physical_page_bytes,
        transport_manifest_commitment_sha256=_manifest_commitment(state.manifest),
        destination_target_sha256=destination_target_sha256,
        destination_mapping_sha256=_destination_mapping_commitment(
            state,
            verified_transport,
            destination_block_indices,
            destination_target_sha256=destination_target_sha256,
            allocation_epoch_by_block=allocation_epoch_by_block,
        ),
        allocation_epochs=allocation_epochs,
        allocator_issued=bool(allocation_epochs),
        destination_raw_zero=True,
        completion_fenced=True,
        passed=True,
        _proof_seal=_V11_SCRUB_PROOF_SEAL,
    )


def write_and_validate_native_subset_v11(
    state: CanonicalKvTransportState,
    tensors: tuple[Any, ...],
    geometry: NativeKvGeometry,
    *,
    destination_block_indices: Mapping[str, int],
    expected_identity: Mapping[str, str],
    verified_transport: V11VerifiedTransport,
    config: V11PipelineConfig,
    allocation_epoch_by_block: Mapping[int, int] | None = None,
    engine_step_gate: V11AdmissionGateWitness | None = None,
    pass_recorder: V11StatePassRecorder | None = None,
    caller_guarantees_outer_scrub: bool = False,
) -> tuple[V11SubsetValidationEvidence, V11PipelineStats]:
    """H2D, direct-scatter, and exact-validate one unpublished page subset.

    ``caller_guarantees_outer_scrub`` is reserved for a caller such as
    :class:`Vllm0230StreamingRestoreStager` that catches every callback failure
    and performs one typed, completion-fenced scrub over *all* allocations it
    owns. Standalone callers retain the local subset scrub by default.
    """

    torch = _torch()
    if not isinstance(caller_guarantees_outer_scrub, bool):
        raise TypeError("v11 outer-scrub ownership flag must be a boolean")
    if config.expected_device_type == "cuda":
        if engine_step_gate is None:
            raise V11EngineStepBindingError(
                "CUDA v11 native write requires the production import/admission gate"
            )
        require_production_gate_operation_v11(
            engine_step_gate,
            operation="IMPORT_ADMISSION",
        )
    branch_group = _branch_group_id(state.manifest.branches)
    recorder = _require_recorder(config, branch_group=branch_group, recorder=pass_recorder)
    _validate_verified_token(state, verified_transport)
    validate_transport_identity(state.manifest, _required_identity(expected_identity))
    block_dimensions = _normalized_block_dimensions(tensors, geometry)
    _preflight_device_identity(tensors, config)
    destination_target_sha256 = destination_target_identity_v11(tensors, geometry, config)
    if any(tensor.dtype is not torch.bfloat16 for tensor in tensors):
        raise ValueError("v11 destination dtype differs from bfloat16 checkpoint")
    if config.expected_device_type == "cuda" and not state.payload.is_pinned():
        raise ValueError("CUDA v11 restore requires a pinned canonical checkpoint")
    if (
        state.manifest.layer_names != geometry.layer_names
        or state.manifest.block_size_tokens != geometry.block_size_tokens
        or state.manifest.kv_heads != geometry.kv_heads
        or state.manifest.head_size != geometry.head_size
        or state.manifest.element_size_bytes != geometry.element_size_bytes
    ):
        raise ValueError("v11 destination geometry differs from checkpoint")
    selected_ids = set(destination_block_indices)
    destination_ids = tuple(destination_block_indices.values())
    if len(destination_ids) != len(set(destination_ids)) or any(
        not 0 <= item < geometry.num_blocks for item in destination_ids
    ):
        raise ValueError("v11 destination blocks must be unique and in range")
    allocation_epochs = _ordered_destination_allocations(
        state.manifest, destination_block_indices, allocation_epoch_by_block
    )
    if config.expected_device_type == "cuda" and not allocation_epochs:
        raise ValueError("CUDA v11 restore requires allocator-issued epochs")
    chunks = _plan_manifest_chunks(
        state.manifest,
        selected_ids=selected_ids,
        maximum_chunk_bytes=config.maximum_chunk_bytes,
    )
    logical_bytes = sum(chunk.payload_bytes for chunk in chunks)
    destination_bytes_read = 0
    tail_bytes_zeroed = 0
    peak_transform_bytes = 0
    peak_host_temporary_bytes = 0

    try:
        if config.expected_device_type == "cuda":
            copy_stream = torch.cuda.Stream(device=tensors[0].device)
            compute_stream = torch.cuda.current_stream(device=tensors[0].device)
            pending: deque[tuple[Any, Any, Any, V11ChunkDescriptor]] = deque()

            def consume_oldest() -> None:
                nonlocal destination_bytes_read, tail_bytes_zeroed
                event, host_snapshot, device_bytes, completed_chunk = pending.popleft()
                compute_stream.wait_event(event)
                device_bytes.record_stream(compute_stream)
                read_bytes, zeroed = _write_validate_device_chunk(
                    device_bytes,
                    chunk=completed_chunk,
                    manifest=state.manifest,
                    tensors=tensors,
                    geometry=geometry,
                    destinations=destination_block_indices,
                    block_dimensions=block_dimensions,
                    recorder=recorder,
                )
                destination_bytes_read += read_bytes
                tail_bytes_zeroed += zeroed
                del host_snapshot

            for chunk in chunks:
                while len(pending) >= config.buffer_count:
                    consume_oldest()
                host_snapshot = _snapshot_and_verify_host_chunk(
                    state,
                    chunk,
                    verified_transport=verified_transport,
                    pin_memory=True,
                    recorder=recorder,
                )
                with torch.cuda.stream(copy_stream):
                    chunk_id = _chunk_id(chunk)
                    with recorder.cuda_pass(
                        _pass_spec(
                            recorder,
                            pass_id=f"restore:{chunk_id}:h2d",
                            operation="h2d",
                            source_representation="BOUNDED_AUTHENTICATED_HOST_CHUNK",
                            destination_representation="CONTINUUM_CANONICAL_TRANSPORT_LAYOUT",
                            source_memory_tier="HOST_PINNED",
                            destination_memory_tier="GPU_HBM_TEMPORARY",
                            logical_bytes=chunk.payload_bytes,
                            physical_bytes_read=chunk.payload_bytes,
                            physical_bytes_written=chunk.payload_bytes,
                            temporary_bytes=chunk.payload_bytes,
                            chunk_id=chunk_id,
                            required_avoidable="REQUIRED",
                            external_transfer_bytes=chunk.payload_bytes,
                            synchronization_dependency=(f"restore:{chunk_id}:consumption-sha256",),
                        ),
                        stream=copy_stream,
                    ):
                        device_bytes = host_snapshot.to(
                            device=tensors[0].device,
                            non_blocking=True,
                        )
                    copied = torch.cuda.Event()
                    copied.record(copy_stream)
                pending.append((copied, host_snapshot, device_bytes, chunk))
                peak_transform_bytes = max(
                    peak_transform_bytes,
                    sum(item[2].numel() * item[2].element_size() for item in pending),
                )
                peak_host_temporary_bytes = max(
                    peak_host_temporary_bytes,
                    sum(item[1].numel() * item[1].element_size() for item in pending),
                )
            while pending:
                consume_oldest()
            with recorder.cpu_pass(
                _pass_spec(
                    recorder,
                    pass_id=(
                        "restore:subset-"
                        f"{min(chunk.payload_offset_bytes for chunk in chunks):012d}:completion-fence"
                    ),
                    operation="cuda_stream_synchronize",
                    source_representation="CUDA_STREAM",
                    destination_representation="HOST_COMPLETION_FENCE",
                    source_memory_tier="GPU_CONTROL",
                    destination_memory_tier="HOST_CONTROL",
                    logical_bytes=logical_bytes,
                    physical_bytes_read=0,
                    physical_bytes_written=0,
                    temporary_bytes=0,
                    chunk_id=(f"subset-{min(chunk.payload_offset_bytes for chunk in chunks):012d}"),
                    required_avoidable="REQUIRED",
                    synchronization_dependency=tuple(
                        f"restore:{_chunk_id(chunk)}:h2d" for chunk in chunks
                    ),
                )
            ):
                compute_stream.synchronize()
        else:
            for chunk in chunks:
                device_bytes = _snapshot_and_verify_host_chunk(
                    state,
                    chunk,
                    verified_transport=verified_transport,
                    pin_memory=False,
                    recorder=recorder,
                )
                peak_host_temporary_bytes = max(peak_host_temporary_bytes, device_bytes.numel())
                peak_transform_bytes = max(peak_transform_bytes, device_bytes.numel())
                read_bytes, zeroed = _write_validate_device_chunk(
                    device_bytes,
                    chunk=chunk,
                    manifest=state.manifest,
                    tensors=tensors,
                    geometry=geometry,
                    destinations=destination_block_indices,
                    block_dimensions=block_dimensions,
                    recorder=recorder,
                )
                destination_bytes_read += read_bytes
                tail_bytes_zeroed += zeroed
    except BaseException:
        if not caller_guarantees_outer_scrub:
            try:
                scrub_native_pages_v11(
                    tensors,
                    geometry,
                    state=state,
                    verified_transport=verified_transport,
                    config=config,
                    allocation_epoch_by_block=allocation_epoch_by_block,
                    destination_block_indices=destination_block_indices,
                    engine_step_gate=engine_step_gate,
                    pass_recorder=recorder,
                )
            except BaseException as scrub_error:
                raise V11IntegrityError(
                    "v11 write failed and failure scrub could not be proven; "
                    "runtime teardown required"
                ) from scrub_error
        raise

    selected_pages = tuple(
        page for page in state.manifest.pages if page.logical_page_id in selected_ids
    )
    physical_selected_bytes = len(selected_pages) * geometry.physical_page_bytes
    full_page_bytes = sum(
        geometry.physical_page_bytes
        for page in selected_pages
        if page.valid_tokens == geometry.block_size_tokens
    )
    coverage = logical_bytes + tail_bytes_zeroed == physical_selected_bytes
    if destination_bytes_read != logical_bytes or not coverage:
        raise V11IntegrityError("v11 destination full-byte coverage proof failed")
    if config.expected_device_type == "cuda":
        assert engine_step_gate is not None
        require_production_gate_operation_v11(
            engine_step_gate,
            operation="IMPORT_ADMISSION",
        )
    evidence = V11SubsetValidationEvidence(
        logical_page_ids=tuple(page.logical_page_id for page in selected_pages),
        ordered_page_destinations=_ordered_destination_mapping(
            state.manifest, destination_block_indices
        ),
        logical_bytes=logical_bytes,
        physical_bytes=physical_selected_bytes,
        destination_bytes_read=destination_bytes_read,
        tail_bytes_zeroed=tail_bytes_zeroed,
        transport_manifest_commitment_sha256=_manifest_commitment(state.manifest),
        destination_target_sha256=destination_target_sha256,
        destination_mapping_sha256=_destination_mapping_commitment(
            state,
            verified_transport,
            destination_block_indices,
            destination_target_sha256=destination_target_sha256,
            allocation_epoch_by_block=allocation_epoch_by_block,
        ),
        allocation_epochs=allocation_epochs,
        allocator_issued=bool(allocation_epochs),
        full_byte_coverage=True,
        destination_exact_match=True,
        completion_fenced=True,
        passed=True,
        _proof_seal=_V11_DESTINATION_PROOF_SEAL,
    )
    stats = V11PipelineStats(
        logical_state_bytes=logical_bytes,
        chunk_count=len(chunks),
        maximum_chunk_bytes=max(chunk.payload_bytes for chunk in chunks),
        buffer_count=config.buffer_count,
        checkpoint_resident_host_bytes=state.manifest.logical_state_bytes,
        peak_host_temporary_bytes=peak_host_temporary_bytes,
        peak_transform_temporary_bytes=peak_transform_bytes,
        external_movement_bytes=logical_bytes if config.expected_device_type == "cuda" else 0,
        zero_tail_bytes=tail_bytes_zeroed,
        full_page_zero_bytes_skipped=full_page_bytes,
    )
    return evidence, stats


def restore_transport_to_native_v11(
    state: CanonicalKvTransportState,
    tensors: tuple[Any, ...],
    geometry: NativeKvGeometry,
    *,
    destination_block_indices: Mapping[str, int],
    expected_identity: Mapping[str, str],
    config: V11PipelineConfig,
    engine_step_gate: V11AdmissionGateWitness | None = None,
    pass_recorder: V11StatePassRecorder | None = None,
) -> V11RestoreResult:
    branch_group = _branch_group_id(state.manifest.branches)
    recorder = _require_recorder(config, branch_group=branch_group, recorder=pass_recorder)
    verified = verify_transport_streaming_v11(
        state,
        maximum_chunk_bytes=config.maximum_chunk_bytes,
        pass_recorder=recorder,
    )
    expected_ids = {page.logical_page_id for page in state.manifest.pages}
    if set(destination_block_indices) != expected_ids:
        raise ValueError("complete v11 restore must cover every logical page exactly once")
    validation, stats = write_and_validate_native_subset_v11(
        state,
        tensors,
        geometry,
        destination_block_indices=destination_block_indices,
        expected_identity=expected_identity,
        verified_transport=verified,
        config=config,
        engine_step_gate=engine_step_gate,
        pass_recorder=recorder,
    )
    passes = recorder.resolve()
    validate_authoritative_pass_records(
        passes,
        expected_device=_recorder_device(config),
        expected_branch_group=branch_group,
    )
    return V11RestoreResult(
        verified_transport=verified,
        validation=validation,
        stats=stats,
        state_passes=passes,
    )


class Vllm0230StreamingRestoreStager(Vllm0230RestoreStager):
    """v11 allocation/admission controller with one fused bounded callback."""

    def import_group_v11(
        self,
        state: CanonicalKvTransportState,
        *,
        verified_transport: V11VerifiedTransport,
        runtime_request_ids: dict[str, str],
        expected_identity: Mapping[str, str],
        destination_target_sha256: str,
        admission_gate: V11AdmissionGateWitness,
        write_and_validate_pages: Callable[
            [dict[str, int], VllmAllocatorEpochProof], V11SubsetValidationEvidence
        ],
        scrub_pages_on_failure: Callable[
            [dict[str, int], VllmAllocatorEpochProof], V11ScrubEvidence
        ],
        allocation_observer: Callable[[str, int, int], None] | None = None,
    ) -> StagedGroupImport:
        _validate_verified_token(state, verified_transport)
        _require_admission_gate(admission_gate, self.scheduler, self.manager)
        epoch_source = require_vllm_allocator_epoch_source(self.manager)
        validate_transport_identity(state.manifest, _required_identity(expected_identity))
        branches_by_id = {branch.logical_branch_id: branch for branch in state.manifest.branches}
        if set(runtime_request_ids) != set(branches_by_id):
            raise ValueError("runtime incarnation map differs from checkpoint branch set")
        if len(set(runtime_request_ids.values())) != len(runtime_request_ids):
            raise ValueError("restore runtime request IDs must be unique")
        if any(logical == runtime for logical, runtime in runtime_request_ids.items()):
            raise ValueError("restore requires fresh runtime request incarnations")
        staged: list[StagedBranchAllocation] = []
        page_destinations: dict[str, int] = {}
        drained_all: list[int] = []
        detached: list[Any] = []
        current_epoch_proof: VllmAllocatorEpochProof | None = None

        def require_subset_evidence(
            mapping: dict[str, int],
            epoch_proof: VllmAllocatorEpochProof,
            evidence: V11SubsetValidationEvidence,
        ) -> None:
            epoch_source.require_current(
                epoch_proof,
                expected_block_ids=tuple(record.block_id for record in epoch_proof.records),
            )
            allocation_epoch_by_block = epoch_proof.allocation_epoch_by_block
            selected_pages = tuple(
                page for page in state.manifest.pages if page.logical_page_id in mapping
            )
            expected_logical_bytes = sum(page.payload_bytes for page in selected_pages)
            expected_physical_bytes = len(selected_pages) * (
                state.manifest.block_size_tokens
                * len(state.manifest.layer_names)
                * 2
                * state.manifest.kv_heads
                * state.manifest.head_size
                * state.manifest.element_size_bytes
            )
            expected_mapping = _ordered_destination_mapping(state.manifest, mapping)
            expected_commitment = _destination_mapping_commitment(
                state,
                verified_transport,
                mapping,
                destination_target_sha256=destination_target_sha256,
                allocation_epoch_by_block=allocation_epoch_by_block,
            )
            expected_epochs = _ordered_destination_allocations(
                state.manifest,
                mapping,
                allocation_epoch_by_block,
            )
            if not isinstance(evidence, V11SubsetValidationEvidence):
                raise TypeError("v11 fused restore callback returned untyped evidence")
            if (
                set(evidence.logical_page_ids) != set(mapping)
                or evidence.ordered_page_destinations != expected_mapping
                or evidence.logical_bytes != expected_logical_bytes
                or evidence.physical_bytes != expected_physical_bytes
                or evidence.destination_bytes_read != expected_logical_bytes
                or evidence.tail_bytes_zeroed != expected_physical_bytes - expected_logical_bytes
                or evidence.transport_manifest_commitment_sha256
                != _manifest_commitment(state.manifest)
                or evidence.destination_target_sha256 != destination_target_sha256
                or evidence.destination_mapping_sha256 != expected_commitment
                or evidence.allocation_epochs != expected_epochs
                or not evidence.allocator_issued
                or not evidence.full_byte_coverage
                or not evidence.destination_exact_match
                or not evidence.completion_fenced
                or not evidence.passed
                or evidence._proof_seal is not _V11_DESTINATION_PROOF_SEAL
            ):
                raise V11IntegrityError("v11 fused destination evidence failed closed")

        pending_before_restore = tuple(self.manager.take_new_block_ids())
        if pending_before_restore:
            raise V11RuntimeTeardownRequired(
                "v11 restore consumed preexisting zero-queue evidence; runtime teardown required",
                affected_block_ids=pending_before_restore,
            )
        try:
            for index, logical_branch_id in enumerate(sorted(branches_by_id)):
                table = branches_by_id[logical_branch_id]
                request = self.detach_new_request(runtime_request_ids[logical_branch_id])
                detached.append(request)
                try:
                    runtime_tokens = tuple(int(token) for token in request.all_token_ids)
                except (AttributeError, TypeError, ValueError):
                    runtime_tokens = ()
                if runtime_tokens != table.token_ids:
                    raise RuntimeError("restore request token history differs from checkpoint")
                if getattr(request, "num_computed_tokens", None) != 0:
                    raise RuntimeError("restore request was not detached at a fresh token boundary")
                if index == 0:
                    local_blocks, local_tokens = None, 0
                else:
                    local_blocks, local_tokens = self.manager.get_computed_blocks(request)
                    if not 0 <= local_tokens <= table.computed_tokens:
                        raise RuntimeError("vLLM returned an invalid local restore cache hit")
                started = time.monotonic_ns()
                full_table, drained = self._allocate(
                    request,
                    computed_tokens=table.computed_tokens,
                    new_computed_blocks=local_blocks,
                    local_cached_tokens=local_tokens,
                )
                ended = time.monotonic_ns()
                if allocation_observer is not None:
                    allocation_observer(logical_branch_id, started, ended)
                if len(full_table) != len(table.logical_page_ids):
                    raise RuntimeError("destination block table length differs from checkpoint")
                for logical_page_id, destination in zip(
                    table.logical_page_ids, full_table, strict=True
                ):
                    prior = page_destinations.setdefault(logical_page_id, destination)
                    if prior != destination:
                        raise RuntimeError(
                            "shared logical root did not map to one destination page"
                        )
                staged.append(
                    StagedBranchAllocation(
                        logical_branch_id=logical_branch_id,
                        runtime_request_id=request.request_id,
                        computed_tokens=table.computed_tokens,
                        local_cached_tokens=local_tokens,
                        destination_block_indices=full_table,
                        newly_allocated_block_indices=drained,
                    )
                )
                drained_all.extend(drained)
                proof_block_ids = tuple(
                    destination
                    for _page_id, destination in _ordered_destination_mapping(
                        state.manifest, page_destinations
                    )
                )
                current_epoch_proof = epoch_source.proof_for_live_blocks(
                    proof_block_ids,
                    owner_request_ids=tuple(item.request_id for item in detached),
                )
                if index == 0:
                    subset = dict(page_destinations)
                    _require_admission_gate(admission_gate, self.scheduler, self.manager)
                    require_subset_evidence(
                        subset,
                        current_epoch_proof,
                        write_and_validate_pages(subset, current_epoch_proof),
                    )
                    _require_admission_gate(admission_gate, self.scheduler, self.manager)
                    self.manager.cache_blocks(request, table.computed_tokens)
            expected_pages = {page.logical_page_id for page in state.manifest.pages}
            if set(page_destinations) != expected_pages:
                raise RuntimeError("staged destination omits canonical logical pages")
            if len(set(page_destinations.values())) != len(page_destinations):
                raise RuntimeError("distinct logical pages alias one destination KV block")
            first_pages = set(state.manifest.branches[0].logical_page_ids)
            remaining = {
                page_id: destination
                for page_id, destination in page_destinations.items()
                if page_id not in first_pages
            }
            if remaining:
                if current_epoch_proof is None:
                    raise RuntimeError("v11 import lacks allocator epoch proof")
                _require_admission_gate(admission_gate, self.scheduler, self.manager)
                require_subset_evidence(
                    remaining,
                    current_epoch_proof,
                    write_and_validate_pages(remaining, current_epoch_proof),
                )
                _require_admission_gate(admission_gate, self.scheduler, self.manager)
            if self.manager.take_new_block_ids():
                raise RuntimeError("unexpected vLLM page allocation occurred during v11 import")
            for request, allocation in zip(detached[1:], staged[1:], strict=True):
                self.manager.cache_blocks(request, allocation.computed_tokens)
            for request, allocation in zip(detached, staged, strict=True):
                request.num_computed_tokens = allocation.computed_tokens
            if current_epoch_proof is None:
                raise RuntimeError("v11 import lacks allocator epoch proof")
            final_block_ids = tuple(
                destination
                for _page_id, destination in _ordered_destination_mapping(
                    state.manifest, page_destinations
                )
            )
            epoch_source.require_current(
                current_epoch_proof,
                expected_block_ids=final_block_ids,
            )
            _require_admission_gate(admission_gate, self.scheduler, self.manager)
            # Consume immediately before visibility while the engine-step gate
            # is held. A second import using this allocation lifetime fails.
            epoch_source.consume_once(
                current_epoch_proof,
                expected_block_ids=final_block_ids,
            )
            for request in detached:
                _require_admission_gate(admission_gate, self.scheduler, self.manager)
                self.scheduler._enqueue_waiting_request(request)
            _require_admission_gate(admission_gate, self.scheduler, self.manager)
        except BaseException:
            scrub_failure: BaseException | None = None
            cleanup_failures: list[str] = []
            if page_destinations:
                try:
                    if current_epoch_proof is None:
                        raise VllmAllocatorEpochError(
                            "v11 failure scrub lacks allocator epoch proof"
                        )
                    scrub_evidence = scrub_pages_on_failure(
                        dict(page_destinations), current_epoch_proof
                    )
                    scrub_epochs = current_epoch_proof.allocation_epoch_by_block
                    if (
                        not isinstance(scrub_evidence, V11ScrubEvidence)
                        or set(scrub_evidence.logical_page_ids) != set(page_destinations)
                        or scrub_evidence.ordered_page_destinations
                        != _ordered_destination_mapping(state.manifest, page_destinations)
                        or scrub_evidence.physical_bytes_zeroed
                        != len(page_destinations)
                        * (
                            state.manifest.block_size_tokens
                            * len(state.manifest.layer_names)
                            * 2
                            * state.manifest.kv_heads
                            * state.manifest.head_size
                            * state.manifest.element_size_bytes
                        )
                        or scrub_evidence.transport_manifest_commitment_sha256
                        != _manifest_commitment(state.manifest)
                        or scrub_evidence.destination_target_sha256 != destination_target_sha256
                        or scrub_evidence.destination_mapping_sha256
                        != _destination_mapping_commitment(
                            state,
                            verified_transport,
                            page_destinations,
                            destination_target_sha256=destination_target_sha256,
                            allocation_epoch_by_block=scrub_epochs,
                        )
                        or scrub_evidence.allocation_epochs
                        != _ordered_destination_allocations(
                            state.manifest,
                            page_destinations,
                            scrub_epochs,
                        )
                        or not scrub_evidence.allocator_issued
                        or not scrub_evidence.destination_raw_zero
                        or not scrub_evidence.completion_fenced
                        or not scrub_evidence.passed
                        or scrub_evidence._proof_seal is not _V11_SCRUB_PROOF_SEAL
                    ):
                        raise V11IntegrityError("v11 failure scrub evidence failed closed")
                except BaseException as error:
                    scrub_failure = error
                    cleanup_failures.append(f"scrub:{type(error).__name__}")
            group_requests = {request.request_id: request for request in detached}
            for runtime_request_id in runtime_request_ids.values():
                request = getattr(self.scheduler, "requests", {}).get(runtime_request_id)
                if request is not None:
                    group_requests[runtime_request_id] = request
            for request in group_requests.values():
                try:
                    self._remove(self.scheduler.waiting, request)
                except BaseException as error:
                    cleanup_failures.append(f"waiting-remove:{type(error).__name__}")
                try:
                    self._remove(self.scheduler.skipped_waiting, request)
                except BaseException as error:
                    cleanup_failures.append(f"skipped-remove:{type(error).__name__}")
                if scrub_failure is None:
                    try:
                        self.manager.free(request)
                    except BaseException as error:
                        cleanup_failures.append(f"free:{type(error).__name__}")
            reset = getattr(self.manager, "reset_prefix_cache", None)
            if callable(reset):
                try:
                    if reset() is not True:
                        cleanup_failures.append("prefix-cache-reset:false")
                except BaseException as error:
                    cleanup_failures.append(f"prefix-cache-reset:{type(error).__name__}")
            if cleanup_failures:
                raise V11RuntimeTeardownRequired(
                    "v11 cleanup could not prove safe allocator ownership; runtime teardown required: "
                    + ",".join(cleanup_failures),
                    affected_block_ids=tuple(page_destinations.values()),
                ) from scrub_failure
            for request in group_requests.values():
                getattr(self.scheduler, "requests", {}).pop(request.request_id, None)
            raise
        return StagedGroupImport(
            branches=tuple(staged),
            logical_page_destinations=page_destinations,
            zero_queue_drained_block_indices=tuple(drained_all),
        )


__all__ = [
    "DEFAULT_MAXIMUM_CHUNK_BYTES",
    "V11AdmissionGateWitness",
    "V11CaptureResult",
    "V11ChunkDescriptor",
    "V11IntegrityError",
    "V11PipelineConfig",
    "V11RestoreResult",
    "V11RuntimeTeardownRequired",
    "V11ScrubEvidence",
    "V11StatePassRecord",
    "V11SubsetValidationEvidence",
    "V11VerifiedTransport",
    "Vllm0230EngineStepBinding",
    "Vllm0230StreamingRestoreStager",
    "capture_native_to_transport_v11",
    "plan_transport_chunks",
    "restore_transport_to_native_v11",
    "scrub_native_pages_v11",
    "verify_transport_streaming_v11",
    "write_and_validate_native_subset_v11",
]
