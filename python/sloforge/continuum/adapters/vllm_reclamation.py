"""Version-scoped vLLM 0.23 KV payload conversion for Experiment 004.

This module does not broaden the portable Continuum adapter contract.  It is a
bounded experimental bridge for one synchronous, single-GPU, full-attention
vLLM 0.23.0 engine.  Opaque runtime handles stay process-local; the transport
manifest contains logical page identifiers and canonical BF16 bytes only.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from typing import Annotated, Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

VLLM_RECLAMATION_ADAPTER_VERSION: Literal["0.1.0"] = "0.1.0"
VLLM_RECLAMATION_RUNTIME_VERSION: Literal["0.23.0"] = "0.23.0"
TRANSPORT_SCHEMA_VERSION: Literal["sloforge.continuum.vllm-kv-branch-group/v1"] = (
    "sloforge.continuum.vllm-kv-branch-group/v1"
)

NonEmpty = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=512)]
Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, validate_default=True)


class NativeAllocationRef(_StrictModel):
    """Source/destination evidence kept outside the transport manifest."""

    gpu_uuid: NonEmpty
    block_index: int = Field(ge=0)
    allocation_epoch: int = Field(ge=0)


class CanonicalKvPageDescriptor(_StrictModel):
    logical_page_id: NonEmpty
    logical_token_start: int = Field(ge=0)
    valid_tokens: int = Field(gt=0)
    payload_offset_bytes: int = Field(ge=0)
    payload_bytes: int = Field(gt=0)
    content_sha256: Sha256
    branch_ids: tuple[NonEmpty, ...]
    shared_root: bool

    @model_validator(mode="after")
    def valid_page(self) -> Self:
        if not self.branch_ids or len(self.branch_ids) != len(set(self.branch_ids)):
            raise ValueError("canonical page requires unique logical branch owners")
        if self.shared_root != (len(self.branch_ids) > 1):
            raise ValueError("shared-root flag disagrees with logical branch ownership")
        return self


class CanonicalBranchTable(_StrictModel):
    logical_branch_id: NonEmpty
    parent_logical_branch_id: NonEmpty
    token_ids: tuple[int, ...]
    token_history_sha256: Sha256
    computed_tokens: int = Field(gt=0)
    logical_page_ids: tuple[NonEmpty, ...]

    @model_validator(mode="after")
    def valid_table(self) -> Self:
        if any(not 0 <= token < 1 << 64 for token in self.token_ids):
            raise ValueError("branch token IDs must fit unsigned 64-bit integers")
        encoded = b"".join(token.to_bytes(8, "little") for token in self.token_ids)
        if hashlib.sha256(encoded).hexdigest() != self.token_history_sha256:
            raise ValueError("branch token-history digest mismatch")
        if not 0 < self.computed_tokens <= len(self.token_ids):
            raise ValueError("computed token boundary is outside token history")
        if len(self.token_ids) - self.computed_tokens > 1:
            raise ValueError("vLLM 0.23 checkpoint may retain at most one uncomputed token")
        if not self.logical_page_ids or len(self.logical_page_ids) != len(
            set(self.logical_page_ids)
        ):
            raise ValueError("branch table requires unique logical page IDs")
        return self


class CanonicalKvTransportManifest(_StrictModel):
    schema_version: Literal["sloforge.continuum.vllm-kv-branch-group/v1"] = TRANSPORT_SCHEMA_VERSION
    runtime: Literal["vllm"] = "vllm"
    runtime_version: Literal["0.23.0"] = VLLM_RECLAMATION_RUNTIME_VERSION
    adapter_version: Literal["0.1.0"] = VLLM_RECLAMATION_ADAPTER_VERSION
    model_id: NonEmpty
    model_revision: NonEmpty
    tokenizer_id: NonEmpty
    tokenizer_revision: NonEmpty
    dtype: NonEmpty
    policy_epoch: NonEmpty
    block_size_tokens: int = Field(gt=0)
    layer_names: tuple[NonEmpty, ...]
    kv_heads: int = Field(gt=0)
    head_size: int = Field(gt=0)
    element_size_bytes: int = Field(gt=0)
    canonical_layout: Literal["page,layer,token,kv,head,dim"] = "page,layer,token,kv,head,dim"
    pages: tuple[CanonicalKvPageDescriptor, ...]
    branches: tuple[CanonicalBranchTable, ...]
    logical_state_bytes: int = Field(gt=0)
    physical_source_bytes: int = Field(gt=0)
    payload_sha256: Sha256

    @model_validator(mode="after")
    def valid_manifest(self) -> Self:
        if not self.layer_names or len(self.layer_names) != len(set(self.layer_names)):
            raise ValueError("canonical manifest requires unique layers in model order")
        if self.dtype != "bfloat16" or self.element_size_bytes != 2:
            raise ValueError("Experiment 004 transport supports canonical bfloat16 only")
        page_ids = [page.logical_page_id for page in self.pages]
        if not page_ids or len(page_ids) != len(set(page_ids)):
            raise ValueError("canonical manifest requires unique logical page IDs")
        known = set(page_ids)
        if not self.branches:
            raise ValueError("canonical manifest requires at least one logical branch")
        if any(not set(branch.logical_page_ids).issubset(known) for branch in self.branches):
            raise ValueError("branch table references an unknown canonical page")
        if len({branch.logical_branch_id for branch in self.branches}) != len(self.branches):
            raise ValueError("canonical manifest contains duplicate logical branches")
        if tuple(branch.logical_branch_id for branch in self.branches) != tuple(
            sorted(branch.logical_branch_id for branch in self.branches)
        ):
            raise ValueError("canonical branch tables must be sorted by logical branch ID")
        if len({branch.parent_logical_branch_id for branch in self.branches}) != 1:
            raise ValueError("canonical branch group must retain one logical parent")
        branch_ids = {branch.logical_branch_id for branch in self.branches}
        if any(branch.parent_logical_branch_id in branch_ids for branch in self.branches):
            raise ValueError("canonical branch parent cannot be a member of its own group")
        ordered = sorted(self.pages, key=lambda page: page.payload_offset_bytes)
        if list(self.pages) != ordered:
            raise ValueError("canonical pages must be stored in payload order")
        cursor = 0
        references: dict[str, set[str]] = {page_id: set() for page_id in page_ids}
        for page in ordered:
            if page.branch_ids != tuple(sorted(page.branch_ids)):
                raise ValueError("canonical page owners must be sorted")
            if page.payload_offset_bytes != cursor:
                raise ValueError("canonical payload page ranges contain a gap or overlap")
            expected_payload = (
                page.valid_tokens
                * len(self.layer_names)
                * 2
                * self.kv_heads
                * self.head_size
                * self.element_size_bytes
            )
            if page.payload_bytes != expected_payload:
                raise ValueError("canonical page payload size disagrees with valid token count")
            cursor += page.payload_bytes
        if cursor != self.logical_state_bytes:
            raise ValueError("canonical logical byte accounting mismatch")
        physical_page_bytes = (
            self.block_size_tokens
            * len(self.layer_names)
            * 2
            * self.kv_heads
            * self.head_size
            * self.element_size_bytes
        )
        if self.physical_source_bytes != len(self.pages) * physical_page_bytes:
            raise ValueError("physical source byte accounting differs from native page coverage")

        pages_by_id = {page.logical_page_id: page for page in self.pages}
        for branch in self.branches:
            required_pages = (branch.computed_tokens + self.block_size_tokens - 1) // (
                self.block_size_tokens
            )
            if len(branch.logical_page_ids) != required_pages:
                raise ValueError("branch page table does not exactly cover its computed boundary")
            for position, page_id in enumerate(branch.logical_page_ids):
                references[page_id].add(branch.logical_branch_id)
                page = pages_by_id[page_id]
                token_start = position * self.block_size_tokens
                expected_valid = min(self.block_size_tokens, branch.computed_tokens - token_start)
                if page.logical_token_start != token_start or page.valid_tokens != expected_valid:
                    raise ValueError("canonical page token range disagrees with branch position")
        for page in self.pages:
            if references[page.logical_page_id] != set(page.branch_ids):
                raise ValueError("canonical page ownership differs from branch-table references")
            if page.shared_root and set(page.branch_ids) != branch_ids:
                raise ValueError("shared root page must be owned by the complete branch group")

        common_prefix: list[str] = []
        for page_group in zip(*(branch.logical_page_ids for branch in self.branches), strict=False):
            if len(set(page_group)) != 1:
                break
            common_prefix.append(page_group[0])
        shared_ids = {page.logical_page_id for page in self.pages if page.shared_root}
        if shared_ids != set(common_prefix):
            raise ValueError("shared-root descriptors differ from the common branch prefix")
        first = self.branches[0]
        for position, page_id in enumerate(common_prefix):
            page = pages_by_id[page_id]
            start = position * self.block_size_tokens
            stop = start + page.valid_tokens
            expected_tokens = first.token_ids[start:stop]
            if any(branch.token_ids[start:stop] != expected_tokens for branch in self.branches[1:]):
                raise ValueError("shared root page spans divergent token history")
        return self

    def canonical_bytes(self) -> bytes:
        return (
            json.dumps(
                self.model_dump(mode="json"),
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            )
            + "\n"
        ).encode("utf-8")


class NativePageBinding(_StrictModel):
    logical_page_id: NonEmpty
    source: NativeAllocationRef


class NativeCaptureEvidence(_StrictModel):
    bindings: tuple[NativePageBinding, ...]

    @model_validator(mode="after")
    def bijective_source_bindings(self) -> Self:
        logical = [item.logical_page_id for item in self.bindings]
        source_slots = [(item.source.gpu_uuid, item.source.block_index) for item in self.bindings]
        if not self.bindings:
            raise ValueError("native capture evidence requires at least one page binding")
        if len(logical) != len(set(logical)) or len(source_slots) != len(set(source_slots)):
            raise ValueError("native capture bindings must be one-to-one")
        if len({item.source.gpu_uuid for item in self.bindings}) != 1:
            raise ValueError("one native capture must originate on exactly one GPU")
        return self

    def verify_manifest_coverage(self, manifest: CanonicalKvTransportManifest) -> None:
        if {item.logical_page_id for item in self.bindings} != {
            page.logical_page_id for page in manifest.pages
        }:
            raise ValueError("native capture evidence differs from canonical page coverage")


class NativeKvGeometry(_StrictModel):
    layer_names: tuple[NonEmpty, ...]
    num_blocks: int = Field(gt=0)
    block_size_tokens: int = Field(gt=0)
    kv_heads: int = Field(gt=0)
    head_size: int = Field(gt=0)
    element_size_bytes: int = Field(gt=0)
    block_dimensions: tuple[int, ...]
    # Source axes after removing the block dimension, expressed as canonical
    # semantic labels. Every validated Qwen layer must expose exactly these.
    source_axis_labels: tuple[Literal["kv", "token", "head", "dim"], ...]

    @model_validator(mode="after")
    def valid_geometry(self) -> Self:
        if not self.layer_names or len(self.layer_names) != len(set(self.layer_names)):
            raise ValueError("native geometry requires unique layers in model order")
        if len(self.block_dimensions) != len(self.layer_names):
            raise ValueError("one native block dimension is required per layer")
        if set(self.source_axis_labels) != {"kv", "token", "head", "dim"}:
            raise ValueError("native semantic axes must be kv/token/head/dim exactly once")
        return self

    @property
    def logical_token_bytes(self) -> int:
        return len(self.layer_names) * 2 * self.kv_heads * self.head_size * self.element_size_bytes

    @property
    def physical_page_bytes(self) -> int:
        return self.logical_token_bytes * self.block_size_tokens


class RuntimeBranchCaptureInput(_StrictModel):
    """Quiesced request evidence used to derive a canonical capture plan.

    Token history may include a final sampled token whose KV has not been
    computed. ``computed_tokens`` is the authoritative KV boundary, so source
    blocks wholly beyond it are intentionally excluded.
    """

    logical_branch_id: NonEmpty
    parent_logical_branch_id: NonEmpty
    token_ids: tuple[int, ...]
    computed_tokens: int = Field(gt=0)
    source_block_indices: tuple[int, ...]

    @model_validator(mode="after")
    def valid_runtime_capture(self) -> Self:
        if self.computed_tokens > len(self.token_ids):
            raise ValueError("computed KV boundary exceeds branch token history")
        if not self.source_block_indices or len(self.source_block_indices) != len(
            set(self.source_block_indices)
        ):
            raise ValueError("runtime branch requires a unique nonempty physical block table")
        if min(self.source_block_indices) < 0:
            raise ValueError("runtime physical block indices cannot be negative")
        return self


class CanonicalCapturePlan(_StrictModel):
    """Destination-independent tables plus process-local source bindings."""

    branch_tables: tuple[CanonicalBranchTable, ...]
    page_order: tuple[tuple[NonEmpty, int, int, tuple[NonEmpty, ...]], ...]
    capture_evidence: NativeCaptureEvidence
    logical_state_bytes: int = Field(gt=0)
    shared_logical_bytes: int = Field(ge=0)
    private_logical_bytes: int = Field(ge=0)
    physical_source_bytes: int = Field(gt=0)

    @model_validator(mode="after")
    def exact_plan(self) -> Self:
        if not self.branch_tables or not self.page_order:
            raise ValueError("canonical capture plan cannot be empty")
        page_ids = tuple(item[0] for item in self.page_order)
        if len(page_ids) != len(set(page_ids)):
            raise ValueError("canonical capture plan contains duplicate logical pages")
        if {item.logical_page_id for item in self.capture_evidence.bindings} != set(page_ids):
            raise ValueError("source binding evidence differs from canonical page order")
        if any(
            not set(branch.logical_page_ids).issubset(page_ids) for branch in self.branch_tables
        ):
            raise ValueError("branch table references a page outside the capture plan")
        if self.shared_logical_bytes + self.private_logical_bytes != self.logical_state_bytes:
            raise ValueError("shared/private logical bytes do not conserve capture payload")
        return self


def build_canonical_capture_plan(
    *,
    branches: Sequence[RuntimeBranchCaptureInput],
    block_size_tokens: int,
    logical_token_bytes: int,
    physical_page_bytes: int,
    gpu_uuid: str,
    allocation_epoch_by_block: Mapping[int, int],
) -> CanonicalCapturePlan:
    """Derive canonical pages from request histories and native block tables.

    Logical IDs depend on token position and branch ownership, never physical
    block numbers. Source numbers survive only in this process-local plan and
    capture evidence. The function performs no CUDA work and is deterministic.
    """

    if min(block_size_tokens, logical_token_bytes, physical_page_bytes) <= 0:
        raise ValueError("capture geometry values must be positive")
    if physical_page_bytes < block_size_tokens * logical_token_bytes:
        raise ValueError("physical page is smaller than its logical token payload")
    if not gpu_uuid:
        raise ValueError("source GPU UUID cannot be empty")
    ordered_branches = tuple(sorted(branches, key=lambda item: item.logical_branch_id))
    branch_ids = {item.logical_branch_id for item in ordered_branches}
    if not ordered_branches or len(branch_ids) != len(ordered_branches):
        raise ValueError("capture requires unique nonempty logical branches")
    if len({item.parent_logical_branch_id for item in ordered_branches}) != 1:
        raise ValueError("capture branch group must retain one logical parent")
    if any(item.parent_logical_branch_id in branch_ids for item in ordered_branches):
        raise ValueError("capture branch parent cannot be a member of its own group")

    uses_by_block: dict[int, list[tuple[RuntimeBranchCaptureInput, int, int]]] = {}
    selected_blocks: dict[str, tuple[int, ...]] = {}
    for branch in ordered_branches:
        required_pages = (branch.computed_tokens + block_size_tokens - 1) // block_size_tokens
        if len(branch.source_block_indices) < required_pages:
            raise ValueError(
                f"branch {branch.logical_branch_id!r} block table ends before computed KV boundary"
            )
        selected = branch.source_block_indices[:required_pages]
        selected_blocks[branch.logical_branch_id] = selected
        for position, block_index in enumerate(selected):
            valid_tokens = min(
                block_size_tokens,
                branch.computed_tokens - position * block_size_tokens,
            )
            uses_by_block.setdefault(block_index, []).append((branch, position, valid_tokens))

    required_blocks = set(uses_by_block)
    if not required_blocks.issubset(allocation_epoch_by_block):
        missing = sorted(required_blocks - set(allocation_epoch_by_block))
        raise ValueError(f"capture lacks allocation epochs for source blocks {missing}")
    if any(allocation_epoch_by_block[item] < 0 for item in required_blocks):
        raise ValueError("source allocation epochs cannot be negative")

    groups: list[tuple[int, tuple[str, ...], int, int]] = []
    for block_index, uses in uses_by_block.items():
        positions = {item[1] for item in uses}
        valid_extents = {item[2] for item in uses}
        if len(positions) != 1:
            raise ValueError("one physical page appears at different logical token positions")
        if len(valid_extents) != 1:
            raise ValueError("shared physical page has inconsistent valid-token extents")
        position = next(iter(positions))
        valid_tokens = next(iter(valid_extents))
        owners = tuple(sorted(item[0].logical_branch_id for item in uses))
        if len(owners) > 1 and set(owners) != branch_ids:
            raise ValueError("shared physical page is not owned by the complete branch group")
        token_start = position * block_size_tokens
        reference = uses[0][0].token_ids[token_start : token_start + valid_tokens]
        if any(
            item[0].token_ids[token_start : token_start + valid_tokens] != reference
            for item in uses[1:]
        ):
            raise ValueError("shared physical page maps to divergent branch token histories")
        groups.append((position, owners, block_index, valid_tokens))
    groups.sort(key=lambda item: (item[0], item[1]))
    shared_positions = sorted(
        position for position, owners, _block, _valid in groups if len(owners) > 1
    )
    if shared_positions != list(range(len(shared_positions))):
        raise ValueError("shared physical pages do not form one common root prefix")

    page_id_by_block: dict[int, str] = {}
    page_order: list[tuple[str, int, int, tuple[str, ...]]] = []
    bindings: list[NativePageBinding] = []
    shared_logical_bytes = 0
    private_logical_bytes = 0
    for ordinal, (_position, owners, block_index, valid_tokens) in enumerate(groups):
        logical_page_id = f"logical-page-{ordinal:06d}"
        page_id_by_block[block_index] = logical_page_id
        page_order.append((logical_page_id, block_index, valid_tokens, owners))
        bindings.append(
            NativePageBinding(
                logical_page_id=logical_page_id,
                source=NativeAllocationRef(
                    gpu_uuid=gpu_uuid,
                    block_index=block_index,
                    allocation_epoch=allocation_epoch_by_block[block_index],
                ),
            )
        )
        logical_bytes = valid_tokens * logical_token_bytes
        if len(owners) > 1:
            shared_logical_bytes += logical_bytes
        else:
            private_logical_bytes += logical_bytes

    branch_tables = tuple(
        CanonicalBranchTable(
            logical_branch_id=branch.logical_branch_id,
            parent_logical_branch_id=branch.parent_logical_branch_id,
            token_ids=branch.token_ids,
            token_history_sha256=token_history_sha256(branch.token_ids),
            computed_tokens=branch.computed_tokens,
            logical_page_ids=tuple(
                page_id_by_block[item] for item in selected_blocks[branch.logical_branch_id]
            ),
        )
        for branch in ordered_branches
    )
    return CanonicalCapturePlan(
        branch_tables=branch_tables,
        page_order=tuple(page_order),
        capture_evidence=NativeCaptureEvidence(bindings=tuple(bindings)),
        logical_state_bytes=shared_logical_bytes + private_logical_bytes,
        shared_logical_bytes=shared_logical_bytes,
        private_logical_bytes=private_logical_bytes,
        physical_source_bytes=len(groups) * physical_page_bytes,
    )


class StagedBranchAllocation(_StrictModel):
    logical_branch_id: NonEmpty
    runtime_request_id: NonEmpty
    computed_tokens: int = Field(gt=0)
    local_cached_tokens: int = Field(ge=0)
    destination_block_indices: tuple[int, ...]
    newly_allocated_block_indices: tuple[int, ...]

    @model_validator(mode="after")
    def valid_staged_allocation(self) -> Self:
        if not self.destination_block_indices:
            raise ValueError("staged branch requires a destination block table")
        if len(self.destination_block_indices) != len(set(self.destination_block_indices)):
            raise ValueError("staged branch block table contains duplicate pages")
        if not set(self.newly_allocated_block_indices).issubset(self.destination_block_indices):
            raise ValueError("new destination pages are absent from the branch block table")
        if self.local_cached_tokens > self.computed_tokens:
            raise ValueError("local cache hit exceeds the imported token boundary")
        return self


class StagedGroupImport(_StrictModel):
    branches: tuple[StagedBranchAllocation, ...]
    logical_page_destinations: dict[NonEmpty, int]
    zero_queue_drained_block_indices: tuple[int, ...]
    validated: Literal[True] = True
    admitted: Literal[True] = True

    @model_validator(mode="after")
    def valid_group_import(self) -> Self:
        if not self.branches or len({branch.logical_branch_id for branch in self.branches}) != len(
            self.branches
        ):
            raise ValueError("staged group requires unique logical branches")
        newly_allocated = {
            block for branch in self.branches for block in branch.newly_allocated_block_indices
        }
        if len(self.zero_queue_drained_block_indices) != len(
            set(self.zero_queue_drained_block_indices)
        ):
            raise ValueError("zero-queue evidence contains duplicate destination pages")
        if newly_allocated != set(self.zero_queue_drained_block_indices):
            raise ValueError("zero-queue evidence differs from imported destination pages")
        return self


@dataclass(frozen=True, slots=True)
class CanonicalKvTransportState:
    manifest: CanonicalKvTransportManifest
    # torch.Tensor on CPU, dtype uint8, shape [logical_state_bytes].  ``Any``
    # keeps torch optional at module import and out of the JSON boundary.
    payload: Any

    def verify(self) -> None:
        torch = _torch()
        if not isinstance(self.payload, torch.Tensor):
            raise TypeError("canonical payload must be a torch.Tensor")
        if self.payload.device.type != "cpu" or self.payload.dtype is not torch.uint8:
            raise ValueError("canonical payload must be a CPU uint8 tensor")
        if not self.payload.is_contiguous() or self.payload.numel() != (
            self.manifest.logical_state_bytes
        ):
            raise ValueError("canonical payload shape/size mismatch")
        payload_bytes = self.payload.numpy().tobytes(order="C")
        if hashlib.sha256(payload_bytes).hexdigest() != self.manifest.payload_sha256:
            raise ValueError("canonical payload digest mismatch")
        payload_view = memoryview(payload_bytes)
        for page in self.manifest.pages:
            view = payload_view[
                page.payload_offset_bytes : page.payload_offset_bytes + page.payload_bytes
            ]
            if hashlib.sha256(view).hexdigest() != page.content_sha256:
                raise ValueError(f"canonical page digest mismatch: {page.logical_page_id}")


@dataclass(frozen=True, slots=True)
class NaiveNativePageRead:
    """Explicit source materialization with block as the leading axis."""

    layers: tuple[Any, ...]
    page_order: tuple[tuple[str, int, int, tuple[str, ...]], ...]


@dataclass(frozen=True, slots=True)
class NaiveCanonicalDeviceState:
    """Flat canonical bytes resident on the GPU before D2H or after H2D."""

    payload: Any
    page_order: tuple[tuple[str, int, int, tuple[str, ...]], ...]


@dataclass(frozen=True, slots=True)
class NaiveDestinationPageState:
    """Per-layer native-shaped destination pages before final pool writes."""

    layers: tuple[Any, ...]
    destination_ids: tuple[int, ...]


def _torch() -> Any:
    try:
        import torch  # type: ignore[import-not-found]
    except ImportError as error:  # pragma: no cover - torch is a GPU-run dependency.
        raise RuntimeError("vLLM reclamation conversion requires PyTorch") from error
    return torch


def validate_native_tensors(tensors: tuple[Any, ...], geometry: NativeKvGeometry) -> None:
    torch = _torch()
    if len(tensors) != len(geometry.layer_names):
        raise ValueError("native tensor count differs from declared layer count")
    expected_without_block = {
        "kv": 2,
        "token": geometry.block_size_tokens,
        "head": geometry.kv_heads,
        "dim": geometry.head_size,
    }
    for layer_name, tensor, block_dim in zip(
        geometry.layer_names, tensors, geometry.block_dimensions, strict=True
    ):
        if not isinstance(tensor, torch.Tensor) or tensor.ndim != 5:
            raise ValueError(f"{layer_name} KV cache must be a rank-five torch tensor")
        normalized = block_dim % tensor.ndim
        if tensor.shape[normalized] != geometry.num_blocks:
            raise ValueError(f"{layer_name} block dimension disagrees with runtime pool")
        shape = tuple(size for index, size in enumerate(tensor.shape) if index != normalized)
        expected = tuple(expected_without_block[label] for label in geometry.source_axis_labels)
        if shape != expected:
            raise ValueError(
                f"{layer_name} semantic shape {shape!r} differs from declared {expected!r}"
            )
        if tensor.element_size() != geometry.element_size_bytes:
            raise ValueError(f"{layer_name} KV element width differs from runtime identity")


def _canonical_page_tensor(
    tensors: tuple[Any, ...],
    geometry: NativeKvGeometry,
    *,
    block_index: int,
    valid_tokens: int,
) -> Any:
    """Gather one native page into [layer, token, kv, head, dim]."""

    torch = _torch()
    layers: list[Any] = []
    target_labels = ("token", "kv", "head", "dim")
    permutation = tuple(geometry.source_axis_labels.index(label) for label in target_labels)
    for tensor, block_dim in zip(tensors, geometry.block_dimensions, strict=True):
        page = tensor.select(block_dim, block_index)
        canonical = page.permute(*permutation)
        layers.append(canonical[:valid_tokens].contiguous())
    return torch.stack(layers, dim=0).contiguous()


def read_native_pages_to_gpu_intermediate(
    tensors: tuple[Any, ...],
    geometry: NativeKvGeometry,
    *,
    page_order: tuple[tuple[str, int, int, tuple[str, ...]], ...],
) -> NaiveNativePageRead:
    """Materialize source native pages without changing their semantic axes."""

    torch = _torch()
    validate_native_tensors(tensors, geometry)
    if not page_order or len({item[0] for item in page_order}) != len(page_order):
        raise ValueError("page order must contain unique logical page IDs")
    source_ids = tuple(item[1] for item in page_order)
    if len(source_ids) != len(set(source_ids)) or any(
        not 0 <= item < geometry.num_blocks for item in source_ids
    ):
        raise ValueError("source page bindings must be unique and in range")
    for _logical_id, _block, valid_tokens, owners in page_order:
        if not 0 < valid_tokens <= geometry.block_size_tokens or not owners:
            raise ValueError("source page has an invalid logical extent or owner set")
    materialized: list[Any] = []
    for tensor, block_dim in zip(tensors, geometry.block_dimensions, strict=True):
        index = torch.tensor(source_ids, dtype=torch.long, device=tensor.device)
        gathered = torch.index_select(tensor, block_dim, index).movedim(block_dim, 0)
        materialized.append(gathered.contiguous())
    return NaiveNativePageRead(layers=tuple(materialized), page_order=page_order)


def transform_native_read_to_canonical_device(
    state: NaiveNativePageRead, geometry: NativeKvGeometry
) -> NaiveCanonicalDeviceState:
    """Repack explicit native page materializations into canonical flat bytes."""

    torch = _torch()
    if len(state.layers) != len(geometry.layer_names):
        raise ValueError("native page materialization layer count differs from geometry")
    target_labels = ("token", "kv", "head", "dim")
    permutation = tuple(geometry.source_axis_labels.index(label) for label in target_labels)
    pages: list[Any] = []
    for page_index, (_logical, _block, valid_tokens, _owners) in enumerate(state.page_order):
        layers = [
            layer[page_index].permute(*permutation)[:valid_tokens].contiguous()
            for layer in state.layers
        ]
        pages.append(torch.stack(layers, dim=0).contiguous().view(torch.uint8))
    payload = torch.cat([page.reshape(-1) for page in pages], dim=0)
    return NaiveCanonicalDeviceState(payload=payload, page_order=state.page_order)


def copy_canonical_device_to_host(state: NaiveCanonicalDeviceState, *, pin_memory: bool) -> Any:
    """Perform exactly one synchronous D2H copy into an explicit host buffer."""

    torch = _torch()
    if state.payload.device.type != "cuda" or state.payload.dtype is not torch.uint8:
        raise ValueError("canonical D2H source must be a CUDA uint8 tensor")
    host = torch.empty(
        state.payload.numel(), dtype=torch.uint8, device="cpu", pin_memory=pin_memory
    )
    host.copy_(state.payload, non_blocking=False)
    return host


def build_host_transport_state(
    host_payload: Any,
    geometry: NativeKvGeometry,
    *,
    page_order: tuple[tuple[str, int, int, tuple[str, ...]], ...],
    branch_tables: tuple[CanonicalBranchTable, ...],
    identity: dict[str, str],
    verify: bool = True,
) -> CanonicalKvTransportState:
    """Generate integrity metadata and the physical-ID-independent manifest."""

    torch = _torch()
    if (
        not isinstance(host_payload, torch.Tensor)
        or host_payload.device.type != "cpu"
        or host_payload.dtype is not torch.uint8
        or not host_payload.is_contiguous()
    ):
        raise ValueError("transport publication requires a contiguous CPU uint8 payload")
    required_identity = {
        "model_id",
        "model_revision",
        "tokenizer_id",
        "tokenizer_revision",
        "dtype",
        "policy_epoch",
    }
    if set(identity) != required_identity or any(not value for value in identity.values()):
        raise ValueError("transport identity fields are incomplete")
    if identity["dtype"] != "bfloat16":
        raise ValueError("canonical vLLM transport identity must be bfloat16")
    expected_bytes = sum(item[2] * geometry.logical_token_bytes for item in page_order)
    if host_payload.numel() != expected_bytes:
        raise ValueError("host payload bytes differ from canonical page extents")
    payload_bytes = host_payload.numpy().tobytes(order="C")
    payload_view = memoryview(payload_bytes)
    pages: list[CanonicalKvPageDescriptor] = []
    offset = 0
    for logical_id, _block_index, valid_tokens, owners in page_order:
        size = valid_tokens * geometry.logical_token_bytes
        page_bytes = payload_view[offset : offset + size]
        pages.append(
            CanonicalKvPageDescriptor(
                logical_page_id=logical_id,
                logical_token_start=(
                    _logical_page_position(branch_tables, logical_id) * geometry.block_size_tokens
                ),
                valid_tokens=valid_tokens,
                payload_offset_bytes=offset,
                payload_bytes=size,
                content_sha256=hashlib.sha256(page_bytes).hexdigest(),
                branch_ids=tuple(sorted(owners)),
                shared_root=len(owners) > 1,
            )
        )
        offset += size
    manifest = CanonicalKvTransportManifest(
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
        logical_state_bytes=host_payload.numel(),
        physical_source_bytes=len(page_order) * geometry.physical_page_bytes,
        payload_sha256=hashlib.sha256(payload_bytes).hexdigest(),
    )
    result = CanonicalKvTransportState(manifest=manifest, payload=host_payload)
    if verify:
        result.verify()
    return result


def copy_host_transport_to_canonical_device(
    state: CanonicalKvTransportState, *, verify: bool = True
) -> Any:
    """Perform exactly one synchronous H2D copy of canonical bytes.

    ``verify=False`` is reserved for a caller that recorded a separate,
    immediately preceding :meth:`CanonicalKvTransportState.verify` stage.
    """

    if verify:
        state.verify()
    return state.payload.to(device="cuda", non_blocking=False)


def convert_canonical_device_to_native_pages(
    device_payload: Any,
    state: CanonicalKvTransportState,
    geometry: NativeKvGeometry,
    *,
    destination_block_indices: dict[str, int],
    require_complete: bool = True,
) -> NaiveDestinationPageState:
    """Unpack a complete or explicitly bounded subset into native GPU pages."""

    torch = _torch()
    if device_payload.device.type != "cuda" or device_payload.dtype is not torch.uint8:
        raise ValueError("restore conversion requires a CUDA uint8 canonical buffer")
    expected = {page.logical_page_id for page in state.manifest.pages}
    selected_ids = set(destination_block_indices)
    if not selected_ids or not selected_ids.issubset(expected):
        raise ValueError("destination page map contains no valid canonical page subset")
    if require_complete and selected_ids != expected:
        raise ValueError("destination page map must cover each logical page exactly once")
    selected_pages = tuple(
        page for page in state.manifest.pages if page.logical_page_id in selected_ids
    )
    destination_ids = tuple(
        destination_block_indices[page.logical_page_id] for page in selected_pages
    )
    if len(destination_ids) != len(set(destination_ids)) or any(
        not 0 <= block < geometry.num_blocks for block in destination_ids
    ):
        raise ValueError("destination native blocks must be unique and in range")
    canonical_labels = ("token", "kv", "head", "dim")
    inverse = tuple(canonical_labels.index(label) for label in geometry.source_axis_labels)
    layers: list[Any] = []
    for layer_index in range(len(geometry.layer_names)):
        native_pages: list[Any] = []
        for page in selected_pages:
            per_layer_bytes = page.payload_bytes // len(geometry.layer_names)
            start = page.payload_offset_bytes + layer_index * per_layer_bytes
            valid = (
                device_payload[start : start + per_layer_bytes]
                .view(torch.bfloat16)
                .view(page.valid_tokens, 2, geometry.kv_heads, geometry.head_size)
            )
            padded = torch.zeros(
                geometry.block_size_tokens,
                2,
                geometry.kv_heads,
                geometry.head_size,
                dtype=torch.bfloat16,
                device=device_payload.device,
            )
            padded[: page.valid_tokens].copy_(valid)
            native_pages.append(padded.permute(*inverse).contiguous())
        layers.append(torch.stack(native_pages, dim=0))
    return NaiveDestinationPageState(layers=tuple(layers), destination_ids=destination_ids)


def write_native_pages_to_destination(
    state: NaiveDestinationPageState,
    tensors: tuple[Any, ...],
    geometry: NativeKvGeometry,
) -> None:
    """Write native-shaped pages into the validated vLLM pool slots."""

    torch = _torch()
    validate_native_tensors(tensors, geometry)
    if len(state.layers) != len(tensors):
        raise ValueError("destination conversion layer count differs from runtime")
    for tensor, native_pages, block_dim in zip(
        tensors, state.layers, geometry.block_dimensions, strict=True
    ):
        index = torch.tensor(state.destination_ids, dtype=torch.long, device=tensor.device)
        tensor.index_copy_(block_dim, index, native_pages.movedim(0, block_dim))


def native_pages_to_host_transport(
    tensors: tuple[Any, ...],
    geometry: NativeKvGeometry,
    *,
    page_order: tuple[tuple[str, int, int, tuple[str, ...]], ...],
    branch_tables: tuple[CanonicalBranchTable, ...],
    identity: dict[str, str],
    pin_memory: bool,
) -> CanonicalKvTransportState:
    """Naively materialize native pages, repack, and synchronously copy to host.

    ``page_order`` entries are ``(logical_page_id, native_block_index,
    valid_tokens, logical_branch_ids)``. Native block indices are consumed by
    this process-local conversion only and never enter the transport manifest.
    """

    torch = _torch()
    validate_native_tensors(tensors, geometry)
    required_identity = {
        "model_id",
        "model_revision",
        "tokenizer_id",
        "tokenizer_revision",
        "dtype",
        "policy_epoch",
    }
    if set(identity) != required_identity or any(not value for value in identity.values()):
        raise ValueError("transport identity fields are incomplete")
    if identity["dtype"] != "bfloat16" or any(
        tensor.dtype is not torch.bfloat16 for tensor in tensors
    ):
        raise ValueError("source tensors differ from canonical bfloat16 identity")
    if not page_order or len({item[0] for item in page_order}) != len(page_order):
        raise ValueError("page order must contain unique logical page IDs")
    if len({item[1] for item in page_order}) != len(page_order):
        raise ValueError("one source physical page cannot appear twice in canonical order")
    device_payloads: list[Any] = []
    for logical_id, block_index, valid_tokens, _owners in page_order:
        if not logical_id or not 0 <= block_index < geometry.num_blocks:
            raise ValueError("invalid process-local source page binding")
        if not 0 < valid_tokens <= geometry.block_size_tokens:
            raise ValueError("valid token count is outside one native page")
        device_payloads.append(
            _canonical_page_tensor(
                tensors, geometry, block_index=block_index, valid_tokens=valid_tokens
            ).view(torch.uint8)
        )
    flat_device = torch.cat([item.reshape(-1) for item in device_payloads], dim=0)
    host = torch.empty(flat_device.numel(), dtype=torch.uint8, device="cpu", pin_memory=pin_memory)
    host.copy_(flat_device, non_blocking=False)
    payload_bytes = host.numpy().tobytes(order="C")
    pages: list[CanonicalKvPageDescriptor] = []
    offset = 0
    for (logical_id, _block_index, valid_tokens, owners), device_page in zip(
        page_order, device_payloads, strict=True
    ):
        size = device_page.numel()
        page_bytes = payload_bytes[offset : offset + size]
        pages.append(
            CanonicalKvPageDescriptor(
                logical_page_id=logical_id,
                logical_token_start=(
                    _logical_page_position(branch_tables, logical_id) * geometry.block_size_tokens
                ),
                valid_tokens=valid_tokens,
                payload_offset_bytes=offset,
                payload_bytes=size,
                content_sha256=hashlib.sha256(page_bytes).hexdigest(),
                branch_ids=tuple(sorted(owners)),
                shared_root=len(owners) > 1,
            )
        )
        offset += size
    manifest = CanonicalKvTransportManifest(
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
        logical_state_bytes=host.numel(),
        physical_source_bytes=len(page_order) * geometry.physical_page_bytes,
        payload_sha256=hashlib.sha256(payload_bytes).hexdigest(),
    )
    state = CanonicalKvTransportState(manifest=manifest, payload=host)
    state.verify()
    return state


def _logical_page_position(branches: tuple[CanonicalBranchTable, ...], logical_page_id: str) -> int:
    starts: list[int] = []
    for branch in branches:
        if logical_page_id in branch.logical_page_ids:
            starts.append(branch.logical_page_ids.index(logical_page_id))
    if not starts or len(set(starts)) != 1:
        raise ValueError("logical page occurs at inconsistent branch-table positions")
    return starts[0]


def restore_host_transport_to_native_pages(
    state: CanonicalKvTransportState,
    tensors: tuple[Any, ...],
    geometry: NativeKvGeometry,
    *,
    destination_block_indices: dict[str, int],
    expected_identity: dict[str, str],
    require_complete: bool = True,
) -> None:
    """H2D, unpack, and scatter transport pages into fresh native pool slots.

    Restore staging uses ``require_complete=False`` for its disjoint validated
    subsets. Standalone callers default to exact full-manifest coverage.
    """

    torch = _torch()
    state.verify()
    validate_transport_identity(state.manifest, expected_identity)
    validate_native_tensors(tensors, geometry)
    if (
        state.manifest.layer_names != geometry.layer_names
        or state.manifest.block_size_tokens != geometry.block_size_tokens
        or state.manifest.kv_heads != geometry.kv_heads
        or state.manifest.head_size != geometry.head_size
        or state.manifest.element_size_bytes != geometry.element_size_bytes
    ):
        raise ValueError("destination KV geometry differs from checkpoint")
    if any(tensor.dtype is not torch.bfloat16 for tensor in tensors):
        raise ValueError("destination tensor dtype differs from canonical bfloat16 checkpoint")
    expected = {page.logical_page_id for page in state.manifest.pages}
    selected_ids = set(destination_block_indices)
    if not selected_ids or not selected_ids.issubset(expected):
        raise ValueError("destination page map contains no valid canonical page subset")
    if require_complete and selected_ids != expected:
        raise ValueError("destination page map must cover each logical page exactly once")
    selected_pages = tuple(
        page for page in state.manifest.pages if page.logical_page_id in selected_ids
    )
    destination_ids = tuple(
        destination_block_indices[page.logical_page_id] for page in selected_pages
    )
    if len(set(destination_ids)) != len(destination_ids) or any(
        not 0 <= block < geometry.num_blocks for block in destination_ids
    ):
        raise ValueError("destination native blocks must be unique and in range")

    canonical_axis_labels = ("token", "kv", "head", "dim")
    inverse = tuple(canonical_axis_labels.index(label) for label in geometry.source_axis_labels)
    for layer_index, (tensor, block_dim) in enumerate(
        zip(tensors, geometry.block_dimensions, strict=True)
    ):
        native_pages: list[Any] = []
        for page in selected_pages:
            page_start = page.payload_offset_bytes
            per_layer_valid_bytes = page.payload_bytes // len(geometry.layer_names)
            if per_layer_valid_bytes * len(geometry.layer_names) != page.payload_bytes:
                raise ValueError("canonical page does not divide evenly across layers")
            layer_start = page_start + layer_index * per_layer_valid_bytes
            layer_bytes = state.payload[layer_start : layer_start + per_layer_valid_bytes]
            valid = layer_bytes.view(tensor.dtype).view(
                page.valid_tokens, 2, geometry.kv_heads, geometry.head_size
            )
            padded = torch.zeros(
                geometry.block_size_tokens,
                2,
                geometry.kv_heads,
                geometry.head_size,
                dtype=tensor.dtype,
                device=state.payload.device,
            )
            padded[: page.valid_tokens].copy_(valid)
            native_pages.append(padded.permute(*inverse).contiguous())
        source = torch.stack(native_pages, dim=0).to(device=tensor.device, non_blocking=False)
        index = torch.tensor(destination_ids, dtype=torch.long, device=tensor.device)
        # Advanced indexing would write a temporary. index_copy_ mutates the
        # actual vLLM backing tensor for either block-dimension placement.
        tensor.index_copy_(block_dim, index, source.movedim(0, block_dim))


class Vllm0230RestoreStager:
    """Private-API import controller for the exact vLLM 0.23.0 scheduler.

    The caller must hold an exclusive engine-step gate for the entire staging
    call. Requests are first created by the ordinary in-process frontend and
    immediately detached from the waiting queues.  They become schedulable only
    after every page has been imported and group validation succeeds.
    """

    def __init__(self, scheduler: Any, manager: Any) -> None:
        self.scheduler = scheduler
        self.manager = manager
        if getattr(scheduler, "kv_cache_manager", None) is not manager:
            raise ValueError("restore stager scheduler/manager identity mismatch")
        scheduler_config = getattr(scheduler, "scheduler_config", None)
        if bool(getattr(scheduler_config, "async_scheduling", False)):
            raise ValueError("restore staging requires synchronous vLLM scheduling")
        if getattr(scheduler, "running", None):
            raise ValueError("restore staging requires an idle GPU1 scheduler")
        self.needs_kv_cache_zeroing = bool(getattr(scheduler, "needs_kv_cache_zeroing", False))
        for name in (
            "allocate_slots",
            "get_blocks",
            "get_computed_blocks",
            "cache_blocks",
            "take_new_block_ids",
            "free",
        ):
            if not callable(getattr(manager, name, None)):
                raise ValueError(f"vLLM 0.23 restore API {name!r} is unavailable")

    @staticmethod
    def _remove(queue: Any, request: Any) -> None:
        remover = getattr(queue, "remove_requests", None)
        if not callable(remover):
            raise ValueError("vLLM waiting queue lacks bounded bulk removal")
        remover([request])

    def detach_new_request(self, runtime_request_id: str) -> Any:
        request = getattr(self.scheduler, "requests", {}).get(runtime_request_id)
        if request is None:
            raise ValueError(f"restore request {runtime_request_id!r} is absent")
        if request in getattr(self.scheduler, "running", ()):
            raise ValueError("a restore request became runnable before validation")
        self._remove(self.scheduler.waiting, request)
        self._remove(self.scheduler.skipped_waiting, request)
        return request

    def _allocate(
        self,
        request: Any,
        *,
        computed_tokens: int,
        new_computed_blocks: Any | None,
        local_cached_tokens: int,
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        pending_before_allocation = tuple(self.manager.take_new_block_ids())
        if self.needs_kv_cache_zeroing and pending_before_allocation:
            raise RuntimeError("unrelated vLLM pages were pending zero before restore allocation")
        external = computed_tokens - local_cached_tokens
        if external < 0:
            raise ValueError("local cache hit exceeds requested imported tokens")
        allocated = self.manager.allocate_slots(
            request,
            num_new_tokens=0,
            num_new_computed_tokens=local_cached_tokens,
            new_computed_blocks=new_computed_blocks,
            num_external_computed_tokens=external,
            delay_cache_blocks=True,
            full_sequence_must_fit=True,
        )
        if allocated is None:
            raise RuntimeError("vLLM has insufficient free pages for atomic restore")
        drained = tuple(self.manager.take_new_block_ids())
        groups = tuple(tuple(group) for group in self.manager.get_blocks(request.request_id).blocks)
        if len(groups) != 1:
            raise RuntimeError("restore staging only supports one KV cache group")
        full_table = tuple(
            block.block_id for block in groups[0] if not bool(getattr(block, "is_null", False))
        )
        local_block_count = sum(
            1
            for group in getattr(new_computed_blocks, "blocks", ())
            for block in group
            if not bool(getattr(block, "is_null", False))
        )
        expected_new_block_count = len(full_table) - local_block_count
        if (
            not full_table
            or expected_new_block_count < 0
            or len(drained) != len(set(drained))
            or len(drained) != expected_new_block_count
            or not set(drained).issubset(full_table)
        ):
            raise RuntimeError("vLLM restore allocation produced an inconsistent block table")
        return full_table, drained

    def import_group(
        self,
        state: CanonicalKvTransportState,
        *,
        runtime_request_ids: dict[str, str],
        expected_identity: dict[str, str],
        write_pages: Callable[[dict[str, int]], None],
        validate_pages: Callable[[dict[str, int]], bool],
        allocation_observer: Callable[[str, int, int], None] | None = None,
    ) -> StagedGroupImport:
        """Allocate, import, validate, then atomically admit a branch group.

        ``write_pages`` and ``validate_pages`` receive disjoint bounded subsets
        whose union is the complete destination map. The first branch is
        written and validated before its prefix is published; callers must
        therefore support repeated subset calls. The caller-held engine-step
        gate is what makes final serial queue insertion atomic with respect to
        execution.
        """

        state.verify()
        validate_transport_identity(state.manifest, expected_identity)
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
        try:
            for index, logical_branch_id in enumerate(sorted(branches_by_id)):
                table = branches_by_id[logical_branch_id]
                request = self.detach_new_request(runtime_request_ids[logical_branch_id])
                detached.append(request)
                runtime_tokens = getattr(request, "all_token_ids", None)
                runtime_token_ids: tuple[int, ...]
                if runtime_tokens is None:
                    runtime_token_ids = ()
                else:
                    try:
                        runtime_token_ids = tuple(int(token) for token in runtime_tokens)
                    except (TypeError, ValueError):
                        runtime_token_ids = ()
                if runtime_token_ids != table.token_ids:
                    raise RuntimeError("restore request token history differs from checkpoint")
                if getattr(request, "num_computed_tokens", None) != 0:
                    raise RuntimeError("restore request was not detached at a fresh token boundary")
                if index == 0:
                    local_blocks = None
                    local_tokens = 0
                else:
                    local_blocks, local_tokens = self.manager.get_computed_blocks(request)
                    if local_tokens < 0 or local_tokens > table.computed_tokens:
                        raise RuntimeError("vLLM returned an invalid local restore cache hit")
                allocation_started_ns = time.monotonic_ns()
                full_table, drained = self._allocate(
                    request,
                    computed_tokens=table.computed_tokens,
                    new_computed_blocks=local_blocks,
                    local_cached_tokens=local_tokens,
                )
                allocation_ended_ns = time.monotonic_ns()
                if allocation_observer is not None:
                    allocation_observer(
                        logical_branch_id,
                        allocation_started_ns,
                        allocation_ended_ns,
                    )
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
                if index == 0:
                    # Validate the first complete branch before publishing its
                    # prefix hashes. Later detached branches can then obtain
                    # the already-valid common root without exposing any
                    # unvalidated bytes, and the engine-step gate still keeps
                    # the group unschedulable.
                    write_pages(dict(page_destinations))
                    if not validate_pages(dict(page_destinations)):
                        raise ValueError("destination KV validation failed before root publish")
                    self.manager.cache_blocks(request, table.computed_tokens)

            if set(page_destinations) != {page.logical_page_id for page in state.manifest.pages}:
                raise RuntimeError("staged destination omits canonical logical pages")
            if len(set(page_destinations.values())) != len(page_destinations):
                raise RuntimeError("distinct logical pages alias one destination KV block")
            first_branch_pages = set(state.manifest.branches[0].logical_page_ids)
            remaining = {
                page_id: destination
                for page_id, destination in page_destinations.items()
                if page_id not in first_branch_pages
            }
            if remaining:
                write_pages(remaining)
                if not validate_pages(remaining):
                    raise ValueError("destination KV validation failed before group publish")
            if self.manager.take_new_block_ids():
                raise RuntimeError("unexpected vLLM page allocation occurred during H2D import")
            for request, allocation in zip(detached[1:], staged[1:], strict=True):
                self.manager.cache_blocks(request, allocation.computed_tokens)
            for request, allocation in zip(detached, staged, strict=True):
                request.num_computed_tokens = allocation.computed_tokens
            for request in detached:
                self.scheduler._enqueue_waiting_request(request)
        except BaseException:
            # None of the detached requests have executed. Free their private
            # block tables and remove internal prefix hashes before propagating.
            # Remove the entire restore incarnation group, including requests
            # not reached before an early failure, so no sibling can later run
            # without the missing branch.
            group_requests = {request.request_id: request for request in detached}
            for runtime_request_id in runtime_request_ids.values():
                request = getattr(self.scheduler, "requests", {}).get(runtime_request_id)
                if request is not None:
                    group_requests[runtime_request_id] = request
            for request in group_requests.values():
                with suppress(BaseException):
                    self._remove(self.scheduler.waiting, request)
                with suppress(BaseException):
                    self._remove(self.scheduler.skipped_waiting, request)
                with suppress(BaseException):
                    self.manager.free(request)
                getattr(self.scheduler, "requests", {}).pop(request.request_id, None)
            reset = getattr(self.manager, "reset_prefix_cache", None)
            if callable(reset):
                reset()
            raise
        return StagedGroupImport(
            branches=tuple(staged),
            logical_page_destinations=page_destinations,
            zero_queue_drained_block_indices=tuple(drained_all),
        )


def token_history_sha256(token_ids: tuple[int, ...]) -> str:
    if not token_ids or any(not 0 <= token < 1 << 64 for token in token_ids):
        raise ValueError("token history must be non-empty unsigned 64-bit IDs")
    return hashlib.sha256(b"".join(token.to_bytes(8, "little") for token in token_ids)).hexdigest()


def validate_transport_identity(
    manifest: CanonicalKvTransportManifest, expected_identity: dict[str, str]
) -> None:
    expected_fields = {
        "model_id",
        "model_revision",
        "tokenizer_id",
        "tokenizer_revision",
        "dtype",
        "policy_epoch",
    }
    if set(expected_identity) != expected_fields or any(
        not value for value in expected_identity.values()
    ):
        raise ValueError("destination runtime identity fields are incomplete")
    actual = {field: str(getattr(manifest, field)) for field in expected_fields}
    if actual != expected_identity:
        raise ValueError("destination runtime identity differs from checkpoint")


__all__ = [
    "TRANSPORT_SCHEMA_VERSION",
    "VLLM_RECLAMATION_ADAPTER_VERSION",
    "CanonicalBranchTable",
    "CanonicalCapturePlan",
    "CanonicalKvPageDescriptor",
    "CanonicalKvTransportManifest",
    "CanonicalKvTransportState",
    "NaiveCanonicalDeviceState",
    "NaiveDestinationPageState",
    "NaiveNativePageRead",
    "NativeAllocationRef",
    "NativeCaptureEvidence",
    "NativeKvGeometry",
    "NativePageBinding",
    "RuntimeBranchCaptureInput",
    "StagedBranchAllocation",
    "StagedGroupImport",
    "Vllm0230RestoreStager",
    "build_canonical_capture_plan",
    "build_host_transport_state",
    "convert_canonical_device_to_native_pages",
    "copy_canonical_device_to_host",
    "copy_host_transport_to_canonical_device",
    "native_pages_to_host_transport",
    "read_native_pages_to_gpu_intermediate",
    "restore_host_transport_to_native_pages",
    "token_history_sha256",
    "validate_native_tensors",
    "validate_transport_identity",
    "write_native_pages_to_destination",
]
