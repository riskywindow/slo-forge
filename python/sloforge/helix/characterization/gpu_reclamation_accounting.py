"""Exact byte accounting for Experiment 004 GPU state reclamation.

The accounting model deliberately separates three things which are easy to
conflate in profiler output:

* a logical state segment, counted exactly once in the denominator;
* endpoint memory touches (reads and writes); and
* link traffic (D2H or H2D), counted once independently of its endpoints.

Intermediate-byte totals are memory *touches*, not allocation capacity.  A
buffer written once and read once therefore contributes twice.  Temporary
allocation totals and peaks are reported separately and never enter movement
amplification.  Checksum bytes describe integrity work and also stay outside
the amplification numerator; the memory read which fed the checksum remains a
normal endpoint touch.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from enum import StrEnum
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from sloforge.continuum.ir.canonical import canonical_json as _canonical_json

STATE_PASS_SCHEMA_VERSION = "sloforge.branchfabric.state-pass-record/v1"
MOVEMENT_EDGE_SCHEMA_VERSION = "sloforge.branchfabric.state-movement-edge/v1"
MOVEMENT_ACCOUNTING_SCHEMA_VERSION = "sloforge.branchfabric.movement-accounting/v1"
MOVEMENT_REPORT_SCHEMA_VERSION = "sloforge.branchfabric.movement-report/v1"


class MovementAccountingError(ValueError):
    """Raised when byte evidence is ambiguous, duplicated, or inconsistent."""


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, validate_default=True)


class StateSegmentKind(StrEnum):
    """Logical payload classes used by a reclaimed branch group."""

    SHARED_KV = "shared_kv"
    PRIVATE_KV = "private_kv"
    TOKEN_HISTORY = "token_history"
    SAMPLER = "sampler"
    WORKFLOW = "workflow"
    ENVIRONMENT = "environment"
    RUNTIME_METADATA = "runtime_metadata"
    OTHER = "other"


class MemoryDomain(StrEnum):
    """Physical domains with an unambiguous accounting role.

    Source and destination native layouts are distinct even when they occupy
    the same GPU at different points in the transaction.  This makes the
    unavoidable fresh-destination write removable from the requested variant
    without relying on operation-name heuristics.
    """

    NONE = "none"
    SOURCE_GPU_NATIVE_PAGED = "source_gpu_native_paged"
    DESTINATION_GPU_NATIVE_PAGED = "destination_gpu_native_paged"
    GPU_TRANSFORM_BUFFER = "gpu_transform_buffer"
    GPU_TRANSPORT_BUFFER = "gpu_transport_buffer"
    PINNED_HOST_TRANSPORT = "pinned_host_transport"
    PAGEABLE_HOST_BUFFER = "pageable_host_buffer"
    HOST_TRANSPORT_STORE = "host_transport_store"
    HOST_INTEGRITY_METADATA = "host_integrity_metadata"
    HOST_RUNTIME_METADATA = "host_runtime_metadata"


class StatePassOperation(StrEnum):
    """Operations which may make one or more measured physical touches."""

    READ = "read"
    UNPAGE = "unpage"
    REPACK = "repack"
    RESHAPE = "reshape"
    TRANSFORM = "transform"
    CHECKSUM = "checksum"
    D2H = "d2h"
    PUBLISH = "publish"
    ALLOCATE = "allocate"
    H2D = "h2d"
    UNPACK = "unpack"
    REPAGE = "repage"
    WRITE = "write"
    VALIDATE = "validate"
    IMPORT = "import"
    FUSED_NATIVE_TO_HOST = "fused_native_to_host"
    FUSED_HOST_TO_NATIVE = "fused_host_to_native"


class TransferDirection(StrEnum):
    NONE = "none"
    D2H = "d2h"
    H2D = "h2d"


class EdgeKind(StrEnum):
    LOGICAL_PAYLOAD = "logical_payload"
    MEMORY_READ = "memory_read"
    MEMORY_WRITE = "memory_write"
    LINK_TRANSFER = "link_transfer"
    CHECKSUM_WORK = "checksum_work"


class AccountingBucket(StrEnum):
    """Mutually exclusive bucket for one machine-readable byte edge."""

    LOGICAL_STATE = "logical_state"
    PHYSICAL_READ = "physical_bytes_read"
    PHYSICAL_WRITE = "physical_bytes_written"
    D2H = "d2h_bytes"
    H2D = "h2d_bytes"
    HOST_INTERMEDIATE = "host_intermediate_bytes"
    GPU_INTERMEDIATE = "gpu_intermediate_bytes"
    CHECKSUM = "checksum_bytes"


_GPU_DOMAINS = frozenset(
    {
        MemoryDomain.SOURCE_GPU_NATIVE_PAGED,
        MemoryDomain.DESTINATION_GPU_NATIVE_PAGED,
        MemoryDomain.GPU_TRANSFORM_BUFFER,
        MemoryDomain.GPU_TRANSPORT_BUFFER,
    }
)
_GPU_NATIVE_DOMAINS = frozenset(
    {
        MemoryDomain.SOURCE_GPU_NATIVE_PAGED,
        MemoryDomain.DESTINATION_GPU_NATIVE_PAGED,
    }
)
_GPU_INTERMEDIATE_DOMAINS = frozenset(
    {MemoryDomain.GPU_TRANSFORM_BUFFER, MemoryDomain.GPU_TRANSPORT_BUFFER}
)
_HOST_DOMAINS = frozenset(
    {
        MemoryDomain.PINNED_HOST_TRANSPORT,
        MemoryDomain.PAGEABLE_HOST_BUFFER,
        MemoryDomain.HOST_TRANSPORT_STORE,
        MemoryDomain.HOST_INTEGRITY_METADATA,
        MemoryDomain.HOST_RUNTIME_METADATA,
    }
)


class LogicalStateSegment(_StrictModel):
    """One unique, minimum logical payload contribution.

    Shared state appears once here even if all branches refer to it.  Repeated
    processing belongs in :class:`StatePassRecord`, never in this denominator.
    """

    segment_id: str = Field(min_length=1, max_length=256)
    branch_group_id: str = Field(min_length=1, max_length=256)
    kind: StateSegmentKind
    logical_bytes: int = Field(gt=0)
    branch_id: str | None = Field(default=None, min_length=1, max_length=256)

    @model_validator(mode="after")
    def private_state_names_a_branch(self) -> Self:
        if self.kind is StateSegmentKind.PRIVATE_KV and self.branch_id is None:
            raise ValueError("a private KV logical segment must identify its branch")
        return self


class StatePassRecord(_StrictModel):
    """One measured operation over one logical state segment.

    ``*_event_id`` values identify physical byte events across tracers.  They
    are optional for a single-tracer fixture, where ``record_id`` plus the edge
    kind is the identity.  Production collectors should supply them so a
    duplicated CUPTI/NVML/application observation fails validation instead of
    inflating totals.
    """

    schema_version: Literal["sloforge.branchfabric.state-pass-record/v1"] = (
        "sloforge.branchfabric.state-pass-record/v1"
    )
    record_id: str = Field(min_length=1, max_length=256)
    state_segment: str = Field(min_length=1, max_length=256)
    branch_group: str = Field(min_length=1, max_length=256)
    operation: StatePassOperation
    source_memory: MemoryDomain
    destination_memory: MemoryDomain
    bytes_read: int = Field(default=0, ge=0)
    bytes_written: int = Field(default=0, ge=0)
    transfer_direction: TransferDirection = TransferDirection.NONE
    transfer_bytes: int = Field(default=0, ge=0)
    checksum_bytes: int = Field(default=0, ge=0)
    logical_offset_bytes: int = Field(default=0, ge=0)
    logical_bytes: int = Field(gt=0)
    start_ns: int = Field(ge=0)
    end_ns: int = Field(ge=0)
    device: str = Field(min_length=1, max_length=128)
    temporary_allocation_bytes: int = Field(default=0, ge=0)
    temporary_allocation_memory: MemoryDomain | None = None
    temporary_allocation_id: str | None = Field(default=None, min_length=1, max_length=256)
    required_unavoidable: bool
    read_event_id: str | None = Field(default=None, min_length=1, max_length=256)
    write_event_id: str | None = Field(default=None, min_length=1, max_length=256)
    transfer_event_id: str | None = Field(default=None, min_length=1, max_length=256)

    @model_validator(mode="after")
    def byte_event_is_well_formed(self) -> Self:
        if self.end_ns < self.start_ns:
            raise ValueError("state pass end_ns precedes start_ns")
        if self.bytes_read and self.source_memory is MemoryDomain.NONE:
            raise ValueError("a nonzero read requires a physical source memory domain")
        if self.bytes_written and self.destination_memory is MemoryDomain.NONE:
            raise ValueError("a nonzero write requires a physical destination memory domain")
        if self.read_event_id is not None and self.bytes_read == 0:
            raise ValueError("read_event_id is invalid when bytes_read is zero")
        if self.write_event_id is not None and self.bytes_written == 0:
            raise ValueError("write_event_id is invalid when bytes_written is zero")
        if self.transfer_event_id is not None and self.transfer_bytes == 0:
            raise ValueError("transfer_event_id is invalid when transfer_bytes is zero")

        if self.transfer_direction is TransferDirection.NONE:
            if self.transfer_bytes:
                raise ValueError("transfer bytes require an explicit D2H or H2D direction")
            if self.transfer_event_id is not None:
                raise ValueError("a non-transfer pass cannot have a transfer event ID")
        else:
            if self.transfer_bytes == 0:
                raise ValueError("D2H/H2D passes require nonzero transfer bytes")
            if self.bytes_read == 0 or self.bytes_written == 0:
                raise ValueError("link traffic requires measured source and destination touches")
            if self.transfer_direction is TransferDirection.D2H and not (
                self.source_memory in _GPU_DOMAINS and self.destination_memory in _HOST_DOMAINS
            ):
                raise ValueError("D2H traffic must move from a GPU domain to a host domain")
            if self.transfer_direction is TransferDirection.H2D and not (
                self.source_memory in _HOST_DOMAINS and self.destination_memory in _GPU_DOMAINS
            ):
                raise ValueError("H2D traffic must move from a host domain to a GPU domain")

        if self.checksum_bytes and self.bytes_read == 0:
            raise ValueError("checksum work requires a measured memory read")
        if not any(
            (
                self.bytes_read,
                self.bytes_written,
                self.transfer_bytes,
                self.checksum_bytes,
                self.temporary_allocation_bytes,
            )
        ):
            raise ValueError("a state pass must contain a measured byte or allocation event")

        if self.temporary_allocation_bytes:
            if self.temporary_allocation_memory not in (_GPU_INTERMEDIATE_DOMAINS | _HOST_DOMAINS):
                raise ValueError(
                    "temporary allocations must reside in a host/GPU intermediate domain"
                )
            if self.temporary_allocation_id is None:
                raise ValueError("temporary allocations require a stable allocation ID")
            if self.end_ns == self.start_ns:
                raise ValueError("temporary allocation lifetime must have positive duration")
        elif (
            self.temporary_allocation_memory is not None or self.temporary_allocation_id is not None
        ):
            raise ValueError(
                "zero-byte temporary allocations cannot carry memory or allocation IDs"
            )
        return self


class StateMovementEdge(_StrictModel):
    """One graph edge and exactly one byte-accounting contribution."""

    schema_version: Literal["sloforge.branchfabric.state-movement-edge/v1"] = (
        "sloforge.branchfabric.state-movement-edge/v1"
    )
    edge_id: str = Field(min_length=1, max_length=768)
    record_id: str | None = Field(default=None, min_length=1, max_length=256)
    state_segment: str = Field(min_length=1, max_length=256)
    branch_group: str = Field(min_length=1, max_length=256)
    edge_kind: EdgeKind
    accounting_bucket: AccountingBucket
    source: str = Field(min_length=1, max_length=512)
    destination: str = Field(min_length=1, max_length=512)
    bytes: int = Field(gt=0)
    included_in_amplification: bool
    required_unavoidable: bool
    start_ns: int | None = Field(default=None, ge=0)
    end_ns: int | None = Field(default=None, ge=0)


class StateMovementAccounting(_StrictModel):
    """Auditable totals derived only from unique movement edges."""

    schema_version: Literal["sloforge.branchfabric.movement-accounting/v1"] = (
        "sloforge.branchfabric.movement-accounting/v1"
    )
    logical_state_bytes: int = Field(gt=0)
    physical_bytes_read: int = Field(ge=0)
    physical_bytes_written: int = Field(ge=0)
    d2h_bytes: int = Field(ge=0)
    h2d_bytes: int = Field(ge=0)
    host_intermediate_bytes: int = Field(ge=0)
    gpu_intermediate_bytes: int = Field(ge=0)
    checksum_bytes: int = Field(ge=0)
    unavoidable_final_destination_write_bytes: int = Field(ge=0)
    amplification_numerator_bytes: int = Field(ge=0)
    amplification_numerator_excluding_unavoidable_final_writes_bytes: int = Field(ge=0)
    state_movement_amplification: float = Field(ge=0.0, allow_inf_nan=False)
    state_movement_amplification_excluding_unavoidable_final_writes: float = Field(
        ge=0.0, allow_inf_nan=False
    )
    host_temporary_allocation_bytes: int = Field(ge=0)
    gpu_temporary_allocation_bytes: int = Field(ge=0)
    peak_host_temporary_allocation_bytes: int = Field(ge=0)
    peak_gpu_temporary_allocation_bytes: int = Field(ge=0)
    peak_gpu_temporary_allocation_bytes_available: bool = False
    peak_gpu_temporary_allocation_bytes_reason: str | None = Field(
        default="operation-local allocation records do not prove complete GPU tensor lifetimes"
    )
    pass_count: int = Field(ge=1)
    movement_edge_count: int = Field(ge=1)

    @property
    def total_temporary_allocation_bytes(self) -> int:
        return self.host_temporary_allocation_bytes + self.gpu_temporary_allocation_bytes

    @model_validator(mode="after")
    def totals_are_self_consistent(self) -> Self:
        expected = (
            self.physical_bytes_read
            + self.physical_bytes_written
            + self.d2h_bytes
            + self.h2d_bytes
            + self.host_intermediate_bytes
            + self.gpu_intermediate_bytes
        )
        if self.amplification_numerator_bytes != expected:
            raise ValueError("movement amplification numerator does not equal its byte buckets")
        if self.unavoidable_final_destination_write_bytes > self.physical_bytes_written:
            raise ValueError("unavoidable final writes exceed all physical writes")
        expected_excluding = expected - self.unavoidable_final_destination_write_bytes
        if (
            self.amplification_numerator_excluding_unavoidable_final_writes_bytes
            != expected_excluding
        ):
            raise ValueError("final-write-excluding numerator is inconsistent")
        if self.state_movement_amplification != expected / self.logical_state_bytes:
            raise ValueError("state movement amplification is inconsistent")
        if (
            self.state_movement_amplification_excluding_unavoidable_final_writes
            != expected_excluding / self.logical_state_bytes
        ):
            raise ValueError("final-write-excluding amplification is inconsistent")
        if self.peak_host_temporary_allocation_bytes > self.host_temporary_allocation_bytes:
            raise ValueError("peak host allocation exceeds total allocations")
        if self.peak_gpu_temporary_allocation_bytes > self.gpu_temporary_allocation_bytes:
            raise ValueError("peak GPU allocation exceeds total allocations")
        if self.peak_gpu_temporary_allocation_bytes_available != (
            self.peak_gpu_temporary_allocation_bytes_reason is None
        ):
            raise ValueError("GPU temporary-peak availability and reason disagree")
        return self


class StateMovementReport(_StrictModel):
    """Canonical machine-readable source for Experiment 004 movement plots."""

    schema_version: Literal["sloforge.branchfabric.movement-report/v1"] = (
        "sloforge.branchfabric.movement-report/v1"
    )
    logical_segments: tuple[LogicalStateSegment, ...] = Field(min_length=1)
    passes: tuple[StatePassRecord, ...] = Field(min_length=1)
    edges: tuple[StateMovementEdge, ...] = Field(min_length=1)
    accounting: StateMovementAccounting

    @model_validator(mode="after")
    def document_is_canonical_and_consistent(self) -> Self:
        if self.logical_segments != _sort_segments(self.logical_segments):
            raise ValueError("logical segments are not in canonical order")
        if self.passes != _sort_passes(self.passes):
            raise ValueError("state passes are not in canonical order")
        if self.edges != _sort_edges(self.edges):
            raise ValueError("movement edges are not in canonical order")
        _validate_ledger(self.logical_segments, self.passes)
        rebuilt_edges = _build_edges(self.logical_segments, self.passes)
        if self.edges != rebuilt_edges:
            raise ValueError("movement edges do not exactly match state passes")
        rebuilt_accounting = _account(self.logical_segments, self.passes, self.edges)
        if self.accounting != rebuilt_accounting:
            raise ValueError("movement accounting does not exactly match movement edges")
        return self


def _sort_segments(segments: Iterable[LogicalStateSegment]) -> tuple[LogicalStateSegment, ...]:
    return tuple(
        sorted(
            segments,
            key=lambda segment: (
                segment.branch_group_id,
                segment.kind.value,
                segment.branch_id or "",
                segment.segment_id,
            ),
        )
    )


def _sort_passes(passes: Iterable[StatePassRecord]) -> tuple[StatePassRecord, ...]:
    return tuple(sorted(passes, key=lambda item: (item.start_ns, item.end_ns, item.record_id)))


def _sort_edges(edges: Iterable[StateMovementEdge]) -> tuple[StateMovementEdge, ...]:
    return tuple(
        sorted(
            edges,
            key=lambda edge: (
                -1 if edge.start_ns is None else edge.start_ns,
                -1 if edge.end_ns is None else edge.end_ns,
                edge.edge_id,
            ),
        )
    )


def _endpoint_bucket(domain: MemoryDomain, *, write: bool) -> AccountingBucket:
    if domain in _GPU_NATIVE_DOMAINS:
        return AccountingBucket.PHYSICAL_WRITE if write else AccountingBucket.PHYSICAL_READ
    if domain in _GPU_INTERMEDIATE_DOMAINS:
        return AccountingBucket.GPU_INTERMEDIATE
    if domain in _HOST_DOMAINS:
        return AccountingBucket.HOST_INTERMEDIATE
    raise MovementAccountingError(f"memory domain {domain.value!r} cannot carry nonzero bytes")


def _memory_node(domain: MemoryDomain, device: str) -> str:
    return f"memory:{domain.value}@{device}"


def _build_edges(
    segments: Sequence[LogicalStateSegment], passes: Sequence[StatePassRecord]
) -> tuple[StateMovementEdge, ...]:
    edges: list[StateMovementEdge] = []
    for segment in segments:
        source_device = next(
            (
                item.device
                for item in passes
                if item.state_segment == segment.segment_id
                and item.source_memory is MemoryDomain.SOURCE_GPU_NATIVE_PAGED
            ),
            "source-device",
        )
        edges.append(
            StateMovementEdge(
                edge_id=f"logical:{segment.branch_group_id}:{segment.segment_id}",
                record_id=None,
                state_segment=segment.segment_id,
                branch_group=segment.branch_group_id,
                edge_kind=EdgeKind.LOGICAL_PAYLOAD,
                accounting_bucket=AccountingBucket.LOGICAL_STATE,
                source=f"logical:{segment.branch_group_id}:{segment.segment_id}",
                destination=_memory_node(MemoryDomain.SOURCE_GPU_NATIVE_PAGED, source_device),
                bytes=segment.logical_bytes,
                included_in_amplification=False,
                required_unavoidable=True,
            )
        )

    for item in passes:
        operation_node = f"pass:{item.record_id}:{item.operation.value}"
        if item.bytes_read:
            read_identity = item.read_event_id or f"{item.record_id}:read"
            bucket = _endpoint_bucket(item.source_memory, write=False)
            edges.append(
                StateMovementEdge(
                    edge_id=f"touch:{read_identity}",
                    record_id=item.record_id,
                    state_segment=item.state_segment,
                    branch_group=item.branch_group,
                    edge_kind=EdgeKind.MEMORY_READ,
                    accounting_bucket=bucket,
                    source=_memory_node(item.source_memory, item.device),
                    destination=operation_node,
                    bytes=item.bytes_read,
                    included_in_amplification=True,
                    required_unavoidable=item.required_unavoidable,
                    start_ns=item.start_ns,
                    end_ns=item.end_ns,
                )
            )
        if item.bytes_written:
            write_identity = item.write_event_id or f"{item.record_id}:write"
            bucket = _endpoint_bucket(item.destination_memory, write=True)
            edges.append(
                StateMovementEdge(
                    edge_id=f"touch:{write_identity}",
                    record_id=item.record_id,
                    state_segment=item.state_segment,
                    branch_group=item.branch_group,
                    edge_kind=EdgeKind.MEMORY_WRITE,
                    accounting_bucket=bucket,
                    source=operation_node,
                    destination=_memory_node(item.destination_memory, item.device),
                    bytes=item.bytes_written,
                    included_in_amplification=True,
                    required_unavoidable=item.required_unavoidable,
                    start_ns=item.start_ns,
                    end_ns=item.end_ns,
                )
            )
        if item.transfer_bytes:
            transfer_identity = item.transfer_event_id or f"{item.record_id}:transfer"
            transfer_bucket = (
                AccountingBucket.D2H
                if item.transfer_direction is TransferDirection.D2H
                else AccountingBucket.H2D
            )
            edges.append(
                StateMovementEdge(
                    edge_id=f"link:{transfer_identity}",
                    record_id=item.record_id,
                    state_segment=item.state_segment,
                    branch_group=item.branch_group,
                    edge_kind=EdgeKind.LINK_TRANSFER,
                    accounting_bucket=transfer_bucket,
                    source=_memory_node(item.source_memory, item.device),
                    destination=_memory_node(item.destination_memory, item.device),
                    bytes=item.transfer_bytes,
                    included_in_amplification=True,
                    required_unavoidable=item.required_unavoidable,
                    start_ns=item.start_ns,
                    end_ns=item.end_ns,
                )
            )
        if item.checksum_bytes:
            edges.append(
                StateMovementEdge(
                    edge_id=f"checksum:{item.record_id}",
                    record_id=item.record_id,
                    state_segment=item.state_segment,
                    branch_group=item.branch_group,
                    edge_kind=EdgeKind.CHECKSUM_WORK,
                    accounting_bucket=AccountingBucket.CHECKSUM,
                    source=operation_node,
                    destination=f"integrity:{item.branch_group}:{item.state_segment}",
                    bytes=item.checksum_bytes,
                    included_in_amplification=False,
                    required_unavoidable=item.required_unavoidable,
                    start_ns=item.start_ns,
                    end_ns=item.end_ns,
                )
            )
    return _sort_edges(edges)


def _peak_allocation(passes: Sequence[StatePassRecord], domains: frozenset[MemoryDomain]) -> int:
    changes: list[tuple[int, int, int]] = []
    for item in passes:
        if item.temporary_allocation_memory not in domains or not item.temporary_allocation_bytes:
            continue
        # Half-open lifetimes [start_ns, end_ns): releases sort before allocations
        # at the same timestamp and are therefore not spuriously concurrent.
        changes.append((item.start_ns, 1, item.temporary_allocation_bytes))
        changes.append((item.end_ns, 0, -item.temporary_allocation_bytes))
    current = 0
    peak = 0
    for _, _, delta in sorted(changes):
        current += delta
        peak = max(peak, current)
    if current:
        raise MovementAccountingError("temporary allocation lifetimes are unbalanced")
    return peak


def _account(
    segments: Sequence[LogicalStateSegment],
    passes: Sequence[StatePassRecord],
    edges: Sequence[StateMovementEdge],
) -> StateMovementAccounting:
    bucket_totals = {bucket: 0 for bucket in AccountingBucket}
    for edge in edges:
        bucket_totals[edge.accounting_bucket] += edge.bytes

    logical_state_bytes = sum(segment.logical_bytes for segment in segments)
    physical_bytes_read = bucket_totals[AccountingBucket.PHYSICAL_READ]
    physical_bytes_written = bucket_totals[AccountingBucket.PHYSICAL_WRITE]
    d2h_bytes = bucket_totals[AccountingBucket.D2H]
    h2d_bytes = bucket_totals[AccountingBucket.H2D]
    host_intermediate_bytes = bucket_totals[AccountingBucket.HOST_INTERMEDIATE]
    gpu_intermediate_bytes = bucket_totals[AccountingBucket.GPU_INTERMEDIATE]
    checksum_bytes = bucket_totals[AccountingBucket.CHECKSUM]
    numerator = (
        physical_bytes_read
        + physical_bytes_written
        + d2h_bytes
        + h2d_bytes
        + host_intermediate_bytes
        + gpu_intermediate_bytes
    )
    unavoidable_final_writes = sum(
        edge.bytes
        for edge in edges
        if edge.edge_kind is EdgeKind.MEMORY_WRITE
        and edge.destination.startswith(
            f"memory:{MemoryDomain.DESTINATION_GPU_NATIVE_PAGED.value}@"
        )
        and edge.required_unavoidable
    )
    numerator_excluding = numerator - unavoidable_final_writes
    host_allocations = sum(
        item.temporary_allocation_bytes
        for item in passes
        if item.temporary_allocation_memory in _HOST_DOMAINS
    )
    gpu_allocations = sum(
        item.temporary_allocation_bytes
        for item in passes
        if item.temporary_allocation_memory in _GPU_INTERMEDIATE_DOMAINS
    )
    return StateMovementAccounting(
        logical_state_bytes=logical_state_bytes,
        physical_bytes_read=physical_bytes_read,
        physical_bytes_written=physical_bytes_written,
        d2h_bytes=d2h_bytes,
        h2d_bytes=h2d_bytes,
        host_intermediate_bytes=host_intermediate_bytes,
        gpu_intermediate_bytes=gpu_intermediate_bytes,
        checksum_bytes=checksum_bytes,
        unavoidable_final_destination_write_bytes=unavoidable_final_writes,
        amplification_numerator_bytes=numerator,
        amplification_numerator_excluding_unavoidable_final_writes_bytes=numerator_excluding,
        state_movement_amplification=numerator / logical_state_bytes,
        state_movement_amplification_excluding_unavoidable_final_writes=(
            numerator_excluding / logical_state_bytes
        ),
        host_temporary_allocation_bytes=host_allocations,
        gpu_temporary_allocation_bytes=gpu_allocations,
        peak_host_temporary_allocation_bytes=_peak_allocation(passes, _HOST_DOMAINS),
        peak_gpu_temporary_allocation_bytes=_peak_allocation(passes, _GPU_INTERMEDIATE_DOMAINS),
        peak_gpu_temporary_allocation_bytes_available=False,
        peak_gpu_temporary_allocation_bytes_reason=(
            "operation-local allocation records do not prove complete GPU tensor lifetimes; "
            "use sampled/allocator peak evidence"
        ),
        pass_count=len(passes),
        movement_edge_count=len(edges),
    )


def _validate_ledger(
    segments: Sequence[LogicalStateSegment], passes: Sequence[StatePassRecord]
) -> None:
    segment_ids = [segment.segment_id for segment in segments]
    if len(segment_ids) != len(set(segment_ids)):
        raise MovementAccountingError("logical state segment IDs must be unique")
    segment_by_id = {segment.segment_id: segment for segment in segments}

    record_ids = [item.record_id for item in passes]
    if len(record_ids) != len(set(record_ids)):
        raise MovementAccountingError("state pass record IDs must be unique")

    explicit_event_ids: list[str] = []
    allocation_ids: list[str] = []
    fingerprints: set[tuple[Any, ...]] = set()
    for item in passes:
        segment = segment_by_id.get(item.state_segment)
        if segment is None:
            raise MovementAccountingError(
                f"state pass {item.record_id!r} references unknown segment {item.state_segment!r}"
            )
        if segment.branch_group_id != item.branch_group:
            raise MovementAccountingError(
                f"state pass {item.record_id!r} branch group differs from its logical segment"
            )
        if item.logical_offset_bytes + item.logical_bytes > segment.logical_bytes:
            raise MovementAccountingError(
                f"state pass {item.record_id!r} exceeds its logical segment byte range"
            )
        explicit_event_ids.extend(
            event_id
            for event_id in (item.read_event_id, item.write_event_id, item.transfer_event_id)
            if event_id is not None
        )
        if item.temporary_allocation_id is not None:
            allocation_ids.append(item.temporary_allocation_id)

        # An exact duplicate with a renamed record is almost certainly the same
        # application/CUPTI/NVML observation entered twice.  The stable event-ID
        # check below also catches duplicates whose tracer timestamps differ.
        fingerprint = (
            item.state_segment,
            item.branch_group,
            item.operation,
            item.source_memory,
            item.destination_memory,
            item.bytes_read,
            item.bytes_written,
            item.transfer_direction,
            item.transfer_bytes,
            item.checksum_bytes,
            item.logical_offset_bytes,
            item.logical_bytes,
            item.start_ns,
            item.end_ns,
            item.device,
            item.temporary_allocation_bytes,
            item.temporary_allocation_memory,
            item.required_unavoidable,
        )
        if fingerprint in fingerprints:
            raise MovementAccountingError(
                "duplicate state-pass observation detected for a physical byte event"
            )
        fingerprints.add(fingerprint)

    if len(explicit_event_ids) != len(set(explicit_event_ids)):
        raise MovementAccountingError("a physical byte event was reported by more than one pass")
    if len(allocation_ids) != len(set(allocation_ids)):
        raise MovementAccountingError("a temporary allocation was reported more than once")


def build_state_movement_report(
    *,
    logical_segments: Sequence[LogicalStateSegment],
    passes: Sequence[StatePassRecord],
) -> StateMovementReport:
    """Validate a ledger and derive its canonical edges and exact accounting."""

    if not logical_segments:
        raise MovementAccountingError("at least one logical state segment is required")
    if not passes:
        raise MovementAccountingError("at least one state pass is required")
    canonical_segments = _sort_segments(logical_segments)
    canonical_passes = _sort_passes(passes)
    _validate_ledger(canonical_segments, canonical_passes)
    edges = _build_edges(canonical_segments, canonical_passes)
    accounting = _account(canonical_segments, canonical_passes, edges)
    return StateMovementReport(
        logical_segments=canonical_segments,
        passes=canonical_passes,
        edges=edges,
        accounting=accounting,
    )


def calculate_movement_accounting(
    *,
    logical_segments: Sequence[LogicalStateSegment],
    passes: Sequence[StatePassRecord],
) -> StateMovementAccounting:
    """Return byte totals while applying all ledger validation rules."""

    return build_state_movement_report(logical_segments=logical_segments, passes=passes).accounting


def movement_edges(
    *,
    logical_segments: Sequence[LogicalStateSegment],
    passes: Sequence[StatePassRecord],
) -> tuple[StateMovementEdge, ...]:
    """Return canonical graph edges suitable for Sankey or equivalent plots."""

    return build_state_movement_report(logical_segments=logical_segments, passes=passes).edges


def canonical_movement_json(report: StateMovementReport) -> bytes:
    """Serialize a validated movement report deterministically."""

    # Revalidation prevents an untrusted ``model_construct`` document from
    # bypassing cross-record accounting invariants before artifact emission.
    checked = StateMovementReport.model_validate(report.model_dump(mode="python"))
    return _canonical_json(checked)


__all__ = [
    "AccountingBucket",
    "EdgeKind",
    "LogicalStateSegment",
    "MemoryDomain",
    "MovementAccountingError",
    "StateMovementAccounting",
    "StateMovementEdge",
    "StateMovementReport",
    "StatePassOperation",
    "StatePassRecord",
    "StateSegmentKind",
    "TransferDirection",
    "build_state_movement_report",
    "calculate_movement_accounting",
    "canonical_movement_json",
    "movement_edges",
]
