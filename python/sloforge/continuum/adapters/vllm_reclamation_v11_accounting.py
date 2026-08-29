"""Canonical measured pass accounting for the Experiment 004 v11 pipeline.

The v11 adapter emits one record from the operation that actually launches or
executes the memory work.  CUDA events are retained until the owning recorder
is resolved; callers never reconstruct pass traffic from the algorithm after
the transaction.  Byte traffic and timing have one authoritative record, while
profiler and copy-counter observations may only reference its ``pass_id``.
"""

from __future__ import annotations

import importlib
import time
from collections import deque
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

V11_STATE_PASS_SCHEMA = "sloforge.branchfabric.state-pass-record/v11"

# The live-event lane stays small enough to bound CUDA event retention.  The
# total ledger is separately bounded because incremental materialization moves
# resolved events into canonical records without discarding their evidence.
# The diagnostic reserve is deliberately unavailable to normal work: a failed
# import still needs enough accounting capacity to record a 28-layer scrub.
V11_DEFAULT_MAXIMUM_PENDING_CUDA_PASSES = 4096
V11_DEFAULT_MAXIMUM_TOTAL_PASSES = 65536
V11_DEFAULT_DIAGNOSTIC_EMERGENCY_RESERVE = 256

RequiredAvoidable = Literal["REQUIRED", "AVOIDABLE", "MIXED", "DIAGNOSTIC"]


@dataclass(frozen=True, slots=True)
class V11StatePassRecord:
    """Resolved, canonical evidence for one executed state-memory operation."""

    pass_id: str
    operation: str
    source_representation: str
    destination_representation: str
    source_memory_tier: str
    destination_memory_tier: str
    logical_bytes: int
    physical_bytes_read: int
    physical_bytes_written: int
    temporary_bytes: int
    device: str
    cuda_stream: str | None
    start_cuda_event: str | None
    end_cuda_event: str | None
    wall_start_ns: int
    wall_end_ns: int
    synchronization_dependency: tuple[str, ...]
    chunk_id: str
    branch_group: str
    required_avoidable: RequiredAvoidable
    external_transfer_bytes: int = 0
    required_physical_bytes: int | None = None
    avoidable_physical_bytes: int | None = None
    diagnostic_physical_bytes: int | None = None
    physical_bytes_by_tier: tuple[tuple[str, int], ...] = ()
    cuda_elapsed_ns: int | None = None
    synchronization_wait_ns: int = 0
    timing_group_id: str | None = None
    timing_owner: bool = True
    schema_version: str = V11_STATE_PASS_SCHEMA

    def __post_init__(self) -> None:
        text_fields = (
            self.pass_id,
            self.operation,
            self.source_representation,
            self.destination_representation,
            self.source_memory_tier,
            self.destination_memory_tier,
            self.device,
            self.chunk_id,
            self.branch_group,
        )
        if any(not item for item in text_fields):
            raise ValueError("v11 StatePassRecord identity fields must be nonempty")
        if self.schema_version != V11_STATE_PASS_SCHEMA:
            raise ValueError("unsupported v11 StatePassRecord schema")
        if any(
            item < 0
            for item in (
                self.logical_bytes,
                self.physical_bytes_read,
                self.physical_bytes_written,
                self.temporary_bytes,
                self.external_transfer_bytes,
                self.synchronization_wait_ns,
            )
        ):
            raise ValueError("v11 StatePassRecord byte counts and durations must be nonnegative")
        if self.wall_end_ns < self.wall_start_ns:
            raise ValueError("v11 StatePassRecord wall interval is reversed")
        if self.cuda_elapsed_ns is not None and self.cuda_elapsed_ns < 0:
            raise ValueError("v11 StatePassRecord CUDA duration must be nonnegative")
        total = self.physical_touch_bytes
        if not self.physical_bytes_by_tier:
            tier_bytes: dict[str, int] = {}
            if self.physical_bytes_read:
                tier_bytes[self.source_memory_tier] = self.physical_bytes_read
            if self.physical_bytes_written:
                tier_bytes[self.destination_memory_tier] = (
                    tier_bytes.get(self.destination_memory_tier, 0) + self.physical_bytes_written
                )
            if self.external_transfer_bytes:
                link_tier = (
                    "LINK_D2H"
                    if self.operation.lower() == "d2h"
                    else "LINK_H2D"
                    if self.operation.lower() == "h2d"
                    else "LINK_CONTROL"
                )
                tier_bytes[link_tier] = tier_bytes.get(link_tier, 0) + self.external_transfer_bytes
            object.__setattr__(
                self,
                "physical_bytes_by_tier",
                tuple(sorted(tier_bytes.items())),
            )
        tier_names = [name for name, _bytes in self.physical_bytes_by_tier]
        if (
            len(tier_names) != len(set(tier_names))
            or any(not name or byte_count < 0 for name, byte_count in self.physical_bytes_by_tier)
            or sum(byte_count for _name, byte_count in self.physical_bytes_by_tier) != total
        ):
            raise ValueError("v11 StatePassRecord physical tier bytes do not conserve work")
        split = (
            self.required_physical_bytes,
            self.avoidable_physical_bytes,
            self.diagnostic_physical_bytes,
        )
        if all(item is None for item in split):
            resolved = {
                "REQUIRED": (total, 0, 0),
                "AVOIDABLE": (0, total, 0),
                "DIAGNOSTIC": (0, 0, total),
            }.get(self.required_avoidable)
            if resolved is None:
                raise ValueError("MIXED StatePassRecord requires an explicit physical-byte split")
            object.__setattr__(self, "required_physical_bytes", resolved[0])
            object.__setattr__(self, "avoidable_physical_bytes", resolved[1])
            object.__setattr__(self, "diagnostic_physical_bytes", resolved[2])
        elif any(item is None for item in split):
            raise ValueError("v11 StatePassRecord physical-byte split must be complete")
        else:
            concrete = tuple(int(item) for item in split if item is not None)
            if any(item < 0 for item in concrete) or sum(concrete) != total:
                raise ValueError("v11 StatePassRecord physical-byte split does not conserve work")
            expected_classification = (
                "MIXED"
                if concrete[0] and concrete[1]
                else "REQUIRED"
                if concrete[0] and not concrete[1] and not concrete[2]
                else "AVOIDABLE"
                if concrete[1] and not concrete[0] and not concrete[2]
                else "DIAGNOSTIC"
                if concrete[2] and not concrete[0] and not concrete[1]
                else None
            )
            if total != 0 and expected_classification != self.required_avoidable:
                raise ValueError("v11 StatePassRecord classification differs from its byte split")
        cuda_fields = (
            self.cuda_stream,
            self.start_cuda_event,
            self.end_cuda_event,
            self.cuda_elapsed_ns,
        )
        if self.device.startswith("cuda"):
            if any(item is None for item in cuda_fields):
                raise ValueError(
                    "CUDA StatePassRecord requires stream, event, and elapsed evidence"
                )
        elif any(item is not None for item in cuda_fields):
            raise ValueError("non-CUDA StatePassRecord cannot claim CUDA event evidence")
        if len(set(self.synchronization_dependency)) != len(self.synchronization_dependency):
            raise ValueError("v11 StatePassRecord synchronization dependencies must be unique")

    @property
    def bytes_read(self) -> int:
        """Compatibility name used by the offline v11 pass-diff builder."""

        return self.physical_bytes_read

    @property
    def bytes_written(self) -> int:
        """Compatibility name used by the offline v11 pass-diff builder."""

        return self.physical_bytes_written

    @property
    def transfer_bytes(self) -> int:
        """Compatibility name for external D2H/H2D movement."""

        return self.external_transfer_bytes

    @property
    def physical_touch_bytes(self) -> int:
        return self.physical_bytes_read + self.physical_bytes_written + self.external_transfer_bytes

    def as_dict(self) -> dict[str, Any]:
        return asdict(self) | {
            "bytes_read": self.physical_bytes_read,
            "bytes_written": self.physical_bytes_written,
            "transfer_bytes": self.external_transfer_bytes,
            "physical_touch_bytes": self.physical_touch_bytes,
        }


@dataclass(frozen=True, slots=True)
class V11PassSpec:
    """Static identity and byte domains fixed before an operation is launched."""

    pass_id: str
    operation: str
    source_representation: str
    destination_representation: str
    source_memory_tier: str
    destination_memory_tier: str
    logical_bytes: int
    physical_bytes_read: int
    physical_bytes_written: int
    temporary_bytes: int
    chunk_id: str
    branch_group: str
    required_avoidable: RequiredAvoidable
    external_transfer_bytes: int = 0
    required_physical_bytes: int | None = None
    avoidable_physical_bytes: int | None = None
    diagnostic_physical_bytes: int | None = None
    physical_bytes_by_tier: tuple[tuple[str, int], ...] = ()
    synchronization_dependency: tuple[str, ...] = ()
    timing_group_id: str | None = None
    timing_owner: bool = True

    def __post_init__(self) -> None:
        if len(set(self.synchronization_dependency)) != len(self.synchronization_dependency):
            raise ValueError("v11 pass dependencies must be unique")
        for name in (
            "logical_bytes",
            "physical_bytes_read",
            "physical_bytes_written",
            "temporary_bytes",
            "external_transfer_bytes",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"v11 pass {name} must be nonnegative")
        total = (
            self.physical_bytes_read + self.physical_bytes_written + self.external_transfer_bytes
        )
        if self.physical_bytes_by_tier:
            tier_names = [name for name, _bytes in self.physical_bytes_by_tier]
            if (
                len(tier_names) != len(set(tier_names))
                or any(
                    not name or byte_count < 0 for name, byte_count in self.physical_bytes_by_tier
                )
                or sum(byte_count for _name, byte_count in self.physical_bytes_by_tier) != total
            ):
                raise ValueError("v11 pass physical tier bytes do not conserve work")
        split = (
            self.required_physical_bytes,
            self.avoidable_physical_bytes,
            self.diagnostic_physical_bytes,
        )
        if self.required_avoidable == "MIXED":
            if any(item is None for item in split):
                raise ValueError("MIXED v11 pass requires an explicit physical-byte split")
            concrete = tuple(int(item) for item in split if item is not None)
            if concrete[0] <= 0 or concrete[1] <= 0 or concrete[2] != 0:
                raise ValueError("MIXED v11 pass must contain required and avoidable work only")
            if sum(concrete) != total:
                raise ValueError("MIXED v11 pass physical-byte split does not conserve work")
        elif any(item is not None for item in split):
            if any(item is None for item in split):
                raise ValueError("v11 pass physical-byte split must be complete")
            concrete = tuple(int(item) for item in split if item is not None)
            if any(item < 0 for item in concrete) or sum(concrete) != total:
                raise ValueError("v11 pass physical-byte split does not conserve work")

    def resolved_physical_split(self) -> tuple[int, int, int]:
        total = (
            self.physical_bytes_read + self.physical_bytes_written + self.external_transfer_bytes
        )
        if self.required_physical_bytes is not None:
            assert self.avoidable_physical_bytes is not None
            assert self.diagnostic_physical_bytes is not None
            return (
                self.required_physical_bytes,
                self.avoidable_physical_bytes,
                self.diagnostic_physical_bytes,
            )
        if self.required_avoidable == "REQUIRED":
            return total, 0, 0
        if self.required_avoidable == "AVOIDABLE":
            return 0, total, 0
        if self.required_avoidable == "DIAGNOSTIC":
            return 0, 0, total
        raise ValueError("MIXED v11 pass requires an explicit physical-byte split")

    def resolved_physical_tiers(self) -> tuple[tuple[str, int], ...]:
        if self.physical_bytes_by_tier:
            return self.physical_bytes_by_tier
        tiers: dict[str, int] = {}
        if self.physical_bytes_read:
            tiers[self.source_memory_tier] = self.physical_bytes_read
        if self.physical_bytes_written:
            tiers[self.destination_memory_tier] = (
                tiers.get(self.destination_memory_tier, 0) + self.physical_bytes_written
            )
        if self.external_transfer_bytes:
            link_tier = (
                "LINK_D2H"
                if self.operation.lower() == "d2h"
                else "LINK_H2D"
                if self.operation.lower() == "h2d"
                else "LINK_CONTROL"
            )
            tiers[link_tier] = tiers.get(link_tier, 0) + self.external_transfer_bytes
        return tuple(sorted(tiers.items()))


@dataclass(slots=True)
class _PendingCudaPass:
    spec: V11PassSpec
    stream: Any
    stream_id: str
    start_event: Any
    end_event: Any
    start_event_id: str
    end_event_id: str
    wall_start_ns: int
    wall_launch_end_ns: int = 0


@dataclass(slots=True)
class V11StatePassRecorder:
    """Bounded pass recorder with incremental and final CUDA synchronization.

    Normal CUDA work may retain at most the non-reserved portion of the live
    event bound.  Reaching that threshold materializes the pending events into
    canonical records and leaves the recorder appendable.  ``DIAGNOSTIC``
    passes alone may use the emergency lane when that materialization fails,
    allowing failure scrub work to remain measured without letting ordinary
    work borrow cleanup capacity.
    """

    device: str
    branch_group: str
    maximum_pending_cuda_passes: int = V11_DEFAULT_MAXIMUM_PENDING_CUDA_PASSES
    maximum_total_passes: int = V11_DEFAULT_MAXIMUM_TOTAL_PASSES
    diagnostic_emergency_reserve: int = V11_DEFAULT_DIAGNOSTIC_EMERGENCY_RESERVE
    _records: list[V11StatePassRecord] = field(default_factory=list, init=False, repr=False)
    _pending: deque[_PendingCudaPass] = field(default_factory=deque, init=False, repr=False)
    _pass_ids: set[str] = field(default_factory=set, init=False, repr=False)
    _resolved: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.device or not self.branch_group:
            raise ValueError("v11 recorder device and branch group must be explicit")
        if self.maximum_pending_cuda_passes <= 0:
            raise ValueError("v11 recorder pending CUDA-pass bound must be positive")
        if self.maximum_total_passes <= 0:
            raise ValueError("v11 recorder total pass bound must be positive")
        if not 0 < self.diagnostic_emergency_reserve < self.maximum_pending_cuda_passes:
            raise ValueError(
                "v11 recorder diagnostic reserve must be positive and below the pending bound"
            )
        if self.diagnostic_emergency_reserve >= self.maximum_total_passes:
            raise ValueError("v11 recorder diagnostic reserve must be below the total bound")

    @property
    def _normal_pending_limit(self) -> int:
        return self.maximum_pending_cuda_passes - self.diagnostic_emergency_reserve

    @property
    def _normal_total_limit(self) -> int:
        return self.maximum_total_passes - self.diagnostic_emergency_reserve

    def _claim(self, spec: V11PassSpec) -> None:
        if self._resolved:
            raise RuntimeError("v11 pass recorder is already resolved")
        if spec.branch_group != self.branch_group:
            raise ValueError("v11 pass belongs to a different branch group")
        if spec.pass_id in self._pass_ids:
            raise ValueError(f"duplicate v11 pass ID: {spec.pass_id}")
        pass_limit = (
            self.maximum_total_passes
            if spec.required_avoidable == "DIAGNOSTIC"
            else self._normal_total_limit
        )
        if len(self._pass_ids) >= pass_limit:
            if spec.required_avoidable == "DIAGNOSTIC":
                raise RuntimeError("v11 diagnostic emergency total-pass reserve exhausted")
            raise RuntimeError("v11 total pass bound reached; diagnostic reserve is protected")
        self._pass_ids.add(spec.pass_id)

    def _materialize_one(self, pending: _PendingCudaPass) -> V11StatePassRecord:
        """Synchronize and construct one record without mutating pending state."""

        wait_started = time.monotonic_ns()
        pending.end_event.synchronize()
        wait_ended = time.monotonic_ns()
        elapsed_ns = max(
            0,
            round(float(pending.start_event.elapsed_time(pending.end_event)) * 1_000_000),
        )
        spec = pending.spec
        implicit_host_synchronization_ns = (
            pending.wall_launch_end_ns - pending.wall_start_ns
            if spec.operation
            in {
                "raw_bit_equal",
                "validate_zero_tail",
                "index_select_and_raw_zero_verify",
            }
            else 0
        )
        required_bytes, avoidable_bytes, diagnostic_bytes = spec.resolved_physical_split()
        return V11StatePassRecord(
            pass_id=spec.pass_id,
            operation=spec.operation,
            source_representation=spec.source_representation,
            destination_representation=spec.destination_representation,
            source_memory_tier=spec.source_memory_tier,
            destination_memory_tier=spec.destination_memory_tier,
            logical_bytes=spec.logical_bytes,
            physical_bytes_read=spec.physical_bytes_read,
            physical_bytes_written=spec.physical_bytes_written,
            temporary_bytes=spec.temporary_bytes,
            device=self.device,
            cuda_stream=pending.stream_id,
            start_cuda_event=pending.start_event_id,
            end_cuda_event=pending.end_event_id,
            wall_start_ns=pending.wall_start_ns,
            wall_end_ns=pending.wall_launch_end_ns,
            synchronization_dependency=spec.synchronization_dependency,
            chunk_id=spec.chunk_id,
            branch_group=spec.branch_group,
            required_avoidable=spec.required_avoidable,
            external_transfer_bytes=spec.external_transfer_bytes,
            required_physical_bytes=required_bytes,
            avoidable_physical_bytes=avoidable_bytes,
            diagnostic_physical_bytes=diagnostic_bytes,
            physical_bytes_by_tier=spec.resolved_physical_tiers(),
            cuda_elapsed_ns=elapsed_ns,
            synchronization_wait_ns=(wait_ended - wait_started + implicit_host_synchronization_ns),
            timing_group_id=spec.timing_group_id or spec.pass_id,
            timing_owner=spec.timing_owner,
        )

    def _materialize_pending(self) -> None:
        """Materialize pending CUDA records exactly once, retaining failures."""

        while self._pending:
            pending = self._pending[0]
            record = self._materialize_one(pending)
            # Append first.  If synchronization, elapsed-time collection, or
            # record construction failed, the pending item remains retryable
            # and the ledger cannot silently omit it.
            self._records.append(record)
            removed = self._pending.popleft()
            if removed is not pending:  # pragma: no cover - deque invariant.
                raise RuntimeError("v11 recorder pending queue ordering changed")

    def _ensure_pending_capacity(self, spec: V11PassSpec) -> None:
        if len(self._pending) < self._normal_pending_limit:
            return
        try:
            self._materialize_pending()
        except Exception:
            if spec.required_avoidable != "DIAGNOSTIC":
                raise
        if len(self._pending) >= (
            self.maximum_pending_cuda_passes
            if spec.required_avoidable == "DIAGNOSTIC"
            else self._normal_pending_limit
        ):
            if spec.required_avoidable == "DIAGNOSTIC":
                raise RuntimeError("v11 diagnostic emergency CUDA-pass reserve exhausted")
            raise RuntimeError("v11 pending CUDA pass bound reached; diagnostic reserve protected")

    @staticmethod
    def _stream_id(stream: Any) -> str:
        raw = getattr(stream, "cuda_stream", None)
        if raw is None:
            raw = id(stream)
        return f"cuda-stream-{int(raw)}"

    @contextmanager
    def cuda_pass(self, spec: V11PassSpec, *, stream: Any | None = None) -> Iterator[None]:
        """Record events around actual CUDA work without synchronizing at launch time."""

        if not self.device.startswith("cuda"):
            raise ValueError("CUDA pass cannot be recorded for a non-CUDA recorder")
        self._ensure_pending_capacity(spec)
        self._claim(spec)
        try:
            try:
                torch: Any = importlib.import_module("torch")
            except ImportError as error:  # pragma: no cover - CUDA runtime dependency.
                raise RuntimeError("v11 CUDA pass recording requires PyTorch") from error
            active_stream = stream if stream is not None else torch.cuda.current_stream()
            stream_id = self._stream_id(active_stream)
            ordinal = len(self._pending) + len(self._records)
            start_id = f"{spec.pass_id}:cuda-start:{ordinal:06d}"
            end_id = f"{spec.pass_id}:cuda-end:{ordinal:06d}"
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            wall_start = time.monotonic_ns()
            start_event.record(active_stream)
            pending = _PendingCudaPass(
                spec=spec,
                stream=active_stream,
                stream_id=stream_id,
                start_event=start_event,
                end_event=end_event,
                start_event_id=start_id,
                end_event_id=end_id,
                wall_start_ns=wall_start,
            )
            yield
        except BaseException:
            self._pass_ids.remove(spec.pass_id)
            raise
        else:
            try:
                end_event.record(active_stream)
                pending.wall_launch_end_ns = time.monotonic_ns()
                self._pending.append(pending)
            except BaseException:
                self._pass_ids.remove(spec.pass_id)
                raise

    @contextmanager
    def cpu_pass(self, spec: V11PassSpec) -> Iterator[None]:
        """Record a synchronous CPU/host memory operation at its execution site."""

        record_device = "cpu" if self.device.startswith("cuda") else self.device
        self._claim(spec)
        try:
            started = time.monotonic_ns()
            yield
            ended = time.monotonic_ns()
            synchronization_wait_ns = (
                ended - started
                if spec.operation in {"cuda_event_synchronize", "cuda_stream_synchronize"}
                else 0
            )
            self._records.append(
                V11StatePassRecord(
                    pass_id=spec.pass_id,
                    operation=spec.operation,
                    source_representation=spec.source_representation,
                    destination_representation=spec.destination_representation,
                    source_memory_tier=spec.source_memory_tier,
                    destination_memory_tier=spec.destination_memory_tier,
                    logical_bytes=spec.logical_bytes,
                    physical_bytes_read=spec.physical_bytes_read,
                    physical_bytes_written=spec.physical_bytes_written,
                    temporary_bytes=spec.temporary_bytes,
                    device=record_device,
                    cuda_stream=None,
                    start_cuda_event=None,
                    end_cuda_event=None,
                    wall_start_ns=started,
                    wall_end_ns=ended,
                    synchronization_dependency=spec.synchronization_dependency,
                    chunk_id=spec.chunk_id,
                    branch_group=spec.branch_group,
                    required_avoidable=spec.required_avoidable,
                    external_transfer_bytes=spec.external_transfer_bytes,
                    required_physical_bytes=spec.resolved_physical_split()[0],
                    avoidable_physical_bytes=spec.resolved_physical_split()[1],
                    diagnostic_physical_bytes=spec.resolved_physical_split()[2],
                    physical_bytes_by_tier=spec.resolved_physical_tiers(),
                    cuda_elapsed_ns=None,
                    synchronization_wait_ns=synchronization_wait_ns,
                    timing_group_id=spec.timing_group_id or spec.pass_id,
                    timing_owner=spec.timing_owner,
                )
            )
        except BaseException:
            self._pass_ids.remove(spec.pass_id)
            raise

    def resolve(self) -> tuple[V11StatePassRecord, ...]:
        """Synchronize every recorded end event once and freeze canonical records."""

        if self._resolved:
            return tuple(sorted(self._records, key=lambda item: (item.wall_start_ns, item.pass_id)))
        self._materialize_pending()
        self._resolved = True
        return tuple(sorted(self._records, key=lambda item: (item.wall_start_ns, item.pass_id)))

    @property
    def records(self) -> tuple[V11StatePassRecord, ...]:
        if not self._resolved:
            raise RuntimeError("v11 pass records must be resolved before consumption")
        return tuple(sorted(self._records, key=lambda item: (item.wall_start_ns, item.pass_id)))


def validate_authoritative_pass_records(
    records: tuple[V11StatePassRecord, ...],
    *,
    expected_device: str,
    expected_branch_group: str,
) -> None:
    """Fail closed on incomplete, duplicate, or timing-double-counted pass evidence."""

    if not records:
        raise ValueError("v11 authoritative pass ledger cannot be empty")
    pass_ids = [item.pass_id for item in records]
    if len(pass_ids) != len(set(pass_ids)):
        raise ValueError("v11 authoritative pass ledger contains duplicate pass IDs")
    known_pass_ids = set(pass_ids)
    for item in records:
        missing = set(item.synchronization_dependency) - known_pass_ids
        if missing:
            raise ValueError(
                f"v11 pass {item.pass_id} has unknown synchronization dependencies: "
                f"{sorted(missing)}"
            )
        if item.pass_id in item.synchronization_dependency:
            raise ValueError("v11 pass cannot synchronize on itself")
    dependencies = {item.pass_id: item.synchronization_dependency for item in records}
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(pass_id: str) -> None:
        if pass_id in visiting:
            raise ValueError("v11 authoritative pass dependencies contain a cycle")
        if pass_id in visited:
            return
        visiting.add(pass_id)
        for dependency in dependencies[pass_id]:
            visit(dependency)
        visiting.remove(pass_id)
        visited.add(pass_id)

    for pass_id in pass_ids:
        visit(pass_id)
    if any(item.branch_group != expected_branch_group for item in records):
        raise ValueError("v11 authoritative pass ledger mixes branch groups")
    for item in records:
        if item.device not in {expected_device, "cpu"}:
            raise ValueError("v11 authoritative pass ledger mixes devices")
        if item.device.startswith("cuda") and (
            item.start_cuda_event is None
            or item.end_cuda_event is None
            or item.cuda_elapsed_ns is None
        ):
            raise ValueError("v11 CUDA pass lacks measured event evidence")
    by_timing_group: dict[str, list[V11StatePassRecord]] = {}
    for item in records:
        group = item.timing_group_id or item.pass_id
        by_timing_group.setdefault(group, []).append(item)
    for group, members in by_timing_group.items():
        owners = [item for item in members if item.timing_owner]
        if len(owners) != 1:
            raise ValueError(f"v11 timing group {group} must have exactly one timing owner")
        if len(members) > 1:
            owner = owners[0]
            for member in members:
                if (
                    member.wall_start_ns != owner.wall_start_ns
                    or member.wall_end_ns != owner.wall_end_ns
                    or member.start_cuda_event != owner.start_cuda_event
                    or member.end_cuda_event != owner.end_cuda_event
                ):
                    raise ValueError(
                        f"v11 shared timing group {group} does not reference one operation"
                    )


def canonical_pass_totals(records: tuple[V11StatePassRecord, ...]) -> dict[str, int]:
    """Compute totals only from canonical records; no profiler/counter additions."""

    return {
        "logical_bytes": sum(item.logical_bytes for item in records),
        "physical_bytes_read": sum(item.physical_bytes_read for item in records),
        "physical_bytes_written": sum(item.physical_bytes_written for item in records),
        "external_transfer_bytes": sum(item.external_transfer_bytes for item in records),
        "full_physical_bytes": sum(item.physical_touch_bytes for item in records),
        "required_physical_bytes": sum(item.required_physical_bytes or 0 for item in records),
        "avoidable_physical_bytes": sum(item.avoidable_physical_bytes or 0 for item in records),
        "diagnostic_physical_bytes": sum(item.diagnostic_physical_bytes or 0 for item in records),
        "temporary_bytes_sum_not_peak": sum(item.temporary_bytes for item in records),
        "cuda_time_ns_timing_owners": sum(
            item.cuda_elapsed_ns or 0 for item in records if item.timing_owner
        ),
        "synchronization_wait_ns": sum(item.synchronization_wait_ns for item in records),
    }


def canonical_physical_bytes_by_tier(
    records: tuple[V11StatePassRecord, ...],
) -> dict[str, int]:
    """Aggregate tier traffic from the same records used for the total."""

    validate_total = sum(item.physical_touch_bytes for item in records)
    totals: dict[str, int] = {}
    for record in records:
        for tier, byte_count in record.physical_bytes_by_tier:
            totals[tier] = totals.get(tier, 0) + byte_count
    if sum(totals.values()) != validate_total:
        raise ValueError("v11 physical tier accounting does not conserve canonical work")
    return dict(sorted(totals.items()))


__all__ = [
    "V11_DEFAULT_DIAGNOSTIC_EMERGENCY_RESERVE",
    "V11_DEFAULT_MAXIMUM_PENDING_CUDA_PASSES",
    "V11_DEFAULT_MAXIMUM_TOTAL_PASSES",
    "V11_STATE_PASS_SCHEMA",
    "V11PassSpec",
    "V11StatePassRecord",
    "V11StatePassRecorder",
    "canonical_pass_totals",
    "canonical_physical_bytes_by_tier",
    "validate_authoritative_pass_records",
]
