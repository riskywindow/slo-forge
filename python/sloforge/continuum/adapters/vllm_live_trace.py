"""Bounded hardware-backed trace recorder for the live vLLM adapter."""

from __future__ import annotations

import os
import socket
import time
from collections.abc import Callable, Mapping

from sloforge.continuum.adapters.real_runtime import (
    KvStateClass,
    PhysicalKvBlock,
    PhysicalKvLayoutSnapshot,
    PhysicalKvReleaseEvidence,
    RuntimeModelIdentity,
)
from sloforge.helix.characterization.trace import (
    BranchOperationType,
    BranchWorkloadEventV1,
    ClockSource,
    MemoryLocation,
    OperationResult,
    StateOperationEventV1,
    StateOperationType,
    StateSegment,
    TimingMeasurementClass,
    TraceLevel,
    TransportType,
    WorkloadProvenance,
    seal_event,
)

_Attribute = bool | int | float | str | None


class VllmLiveTraceCapacityError(RuntimeError):
    """The explicitly bounded live trace buffer is full."""


class VllmLiveTraceRecorder:
    """Record pointer-free observations from real vLLM GPU KV state.

    Timings describe adapter/runtime calls observed on the host monotonic clock;
    this recorder never fabricates CUDA-kernel durations. Aligned append-only
    sharing is represented by publication, fork/map, append, reclaim, and free
    events. ``STATE_COW`` is rejected because the adapter performs no partial-page
    copy-on-write operation.
    """

    def __init__(
        self,
        *,
        identity: RuntimeModelIdentity,
        trace_id: str,
        level: TraceLevel,
        max_events: int,
        clock_ns: Callable[[], int] = time.monotonic_ns,
        host: str | None = None,
        process_id: int | None = None,
    ) -> None:
        if level is TraceLevel.DISABLED:
            raise ValueError("disabled tracing must not construct a live recorder")
        if not trace_id or len(trace_id) > 512:
            raise ValueError("trace_id must contain 1..512 characters")
        if not 1 <= max_events <= 1_000_000:
            raise ValueError("max_events must be in 1..1000000")
        self._identity = identity
        self._trace_id = trace_id
        self._level = level
        self._max_events = max_events
        self._clock_ns = clock_ns
        self._origin_ns = clock_ns()
        self._host = host or socket.gethostname()
        self._process_id = os.getpid() if process_id is None else process_id
        self._branch_events: list[BranchWorkloadEventV1] = []
        self._state_events: list[StateOperationEventV1] = []
        self._blocks: dict[str, PhysicalKvBlock] = {}
        self._event_record_calls = 0
        self._event_record_wall_time_ns = 0
        self._event_record_cpu_time_ns = 0
        self._layout_observation_calls = 0
        self._layout_diff_wall_time_ns = 0
        self._layout_diff_cpu_time_ns = 0

    @property
    def level(self) -> TraceLevel:
        return self._level

    def _reserve(self) -> None:
        if len(self._branch_events) + len(self._state_events) >= self._max_events:
            raise VllmLiveTraceCapacityError(
                f"live vLLM trace exceeded its {self._max_events}-event bound"
            )

    def record_branch(
        self,
        operation: BranchOperationType,
        *,
        session_id: str,
        branch_group_id: str | None = None,
        branch_id: str | None = None,
        parent_branch_id: str | None = None,
        physical_state_id: str | None = None,
        physical_bytes: int = 0,
        duration_ns: int = 0,
        shared_root: bool = False,
        private_suffix: bool = False,
        fanout: int = 1,
        attributes: Mapping[str, _Attribute] | None = None,
    ) -> None:
        overhead_wall_started = time.perf_counter_ns()
        overhead_cpu_started = time.process_time_ns()
        if operation is BranchOperationType.STATE_COW:
            raise ValueError("aligned append-only vLLM sharing must never emit STATE_COW")
        self._reserve()
        timestamp = self._clock_ns()
        event = BranchWorkloadEventV1(
            collection_level=self._level,
            provenance=WorkloadProvenance.HARDWARE_BACKED_REAL,
            timing_measurement_class=TimingMeasurementClass.HARDWARE_BACKED_REAL,
            trace_id=self._trace_id,
            session_id=session_id,
            branch_group_id=branch_group_id,
            branch_id=branch_id,
            parent_branch_id=parent_branch_id,
            policy_epoch=self._identity.policy_epoch,
            host=self._host,
            process_id=self._process_id,
            rank=0,
            device=self._identity.device,
            monotonic_timestamp_ns=timestamp,
            normalized_timestamp_ns=timestamp - self._origin_ns,
            duration_ns=duration_ns,
            clock_source=ClockSource.MONOTONIC,
            alignment_confidence=1.0,
            operation_type=operation,
            logical_state_id=f"logical:{session_id}",
            physical_state_id=physical_state_id,
            state_segment=StateSegment.KV,
            physical_bytes=physical_bytes,
            location=MemoryLocation.GPU_HBM,
            source_location=MemoryLocation.GPU_HBM,
            destination_location=MemoryLocation.GPU_HBM,
            shared_root=shared_root,
            private_suffix=private_suffix,
            cow_allocation=False,
            execution_latency_ns=duration_ns,
            cpu_time_ns=None,
            gpu_duration_ns=None,
            transport_type=TransportType.NONE,
            fanout=fanout,
            attributes={
                "runtime": self._identity.runtime,
                "runtime_version": self._identity.runtime_version,
                "adapter_version": self._identity.adapter_version,
                **dict(attributes or {}),
            },
        )
        sealed = seal_event(
            event,
            event_sequence=len(self._branch_events),
            collection_level=self._level,
        )
        assert isinstance(sealed, BranchWorkloadEventV1)
        self._branch_events.append(sealed)
        self._event_record_calls += 1
        self._event_record_wall_time_ns += time.perf_counter_ns() - overhead_wall_started
        self._event_record_cpu_time_ns += time.process_time_ns() - overhead_cpu_started

    def _block_attributes(
        self,
        block: PhysicalKvBlock,
        *,
        cause: str,
        prior_refcount: int | None,
    ) -> dict[str, _Attribute]:
        return {
            "runtime": self._identity.runtime,
            "runtime_version": self._identity.runtime_version,
            "adapter_version": self._identity.adapter_version,
            "runtime_block_id": block.runtime_block_id,
            "kv_cache_group": block.kv_cache_group,
            "layer_ids": ",".join(block.layer_ids),
            "device": block.device,
            "dtype": block.dtype,
            "logical_token_start": block.logical_token_start,
            "logical_token_end_exclusive": block.logical_token_end_exclusive,
            "branch_ids": ",".join(block.branch_ids),
            "state_class": block.state_class.value,
            "refcount": block.refcount,
            "prior_refcount": prior_refcount,
            "refcount_evidence": block.refcount_evidence.value,
            "allocation_epoch": block.allocation_epoch,
            "cache_resident": block.cache_resident,
            "cause": cause,
            "cpu_time_measured": False,
            "gpu_time_measured": False,
            "same_runtime_only": True,
            "same_policy_only": True,
        }

    def _record_state(
        self,
        operation: StateOperationType,
        *,
        session_id: str,
        branch_group_id: str | None,
        block: PhysicalKvBlock,
        cause: str,
        prior_refcount: int | None,
        duration_ns: int,
    ) -> None:
        overhead_wall_started = time.perf_counter_ns()
        overhead_cpu_started = time.process_time_ns()
        if operation is StateOperationType.STATE_COW:
            raise ValueError("aligned append-only vLLM sharing must never emit STATE_COW")
        self._reserve()
        timestamp = self._clock_ns()
        representation = f"vllm-0.23.0/kv-page/{block.state_class.value}"
        source_location = MemoryLocation.GPU_HBM
        destination_location = MemoryLocation.GPU_HBM
        source_representation = representation
        destination_representation = representation
        if operation is StateOperationType.STATE_ALLOC:
            source_location = MemoryLocation.UNKNOWN
            source_representation = "vllm-0.23.0/kv-page/unallocated"
        elif operation is StateOperationType.STATE_FREE:
            destination_location = MemoryLocation.UNKNOWN
            destination_representation = "vllm-0.23.0/kv-page/free-pool"
        event = StateOperationEventV1(
            collection_level=self._level,
            provenance=WorkloadProvenance.HARDWARE_BACKED_REAL,
            timing_measurement_class=TimingMeasurementClass.HARDWARE_BACKED_REAL,
            trace_id=self._trace_id,
            session_id=session_id,
            branch_group_id=branch_group_id,
            logical_state_id=f"logical:{session_id}",
            branch_id=block.branch_ids[0] if len(block.branch_ids) == 1 else None,
            tenant_id="sloforge-gpu-validation",
            security_domain="same-process-vllm-adapter",
            host=self._host,
            process_id=self._process_id,
            rank=0,
            device=block.device,
            monotonic_timestamp_ns=timestamp,
            normalized_timestamp_ns=timestamp - self._origin_ns,
            duration_ns=duration_ns,
            clock_source=ClockSource.MONOTONIC,
            alignment_confidence=1.0,
            operation_type=operation,
            state_segment=StateSegment.KV,
            source_physical_representation=source_representation,
            destination_physical_representation=destination_representation,
            bytes=block.bytes,
            alignment_bytes=block.bytes,
            page_size_bytes=block.bytes,
            chunk_size_bytes=block.bytes,
            fanout=max(1, len(block.branch_ids)),
            concurrency=max(1, len(block.branch_ids)),
            operation_latency_ns=duration_ns,
            # The v1 state-operation schema requires integer CPU/GPU fields.
            # Zero is an unavailable sentinel here, not a measured zero; the
            # explicit attributes below prevent analysis from consuming it.
            cpu_time_ns=0,
            gpu_time_ns=0,
            transfer_time_ns=0,
            result=OperationResult.SUCCESS,
            state_epoch=block.allocation_epoch,
            source_location=source_location,
            destination_location=destination_location,
            transport_type=TransportType.NONE,
            attributes=self._block_attributes(block, cause=cause, prior_refcount=prior_refcount),
        )
        sealed = seal_event(
            event,
            event_sequence=len(self._state_events),
            collection_level=self._level,
        )
        assert isinstance(sealed, StateOperationEventV1)
        self._state_events.append(sealed)
        self._event_record_calls += 1
        self._event_record_wall_time_ns += time.perf_counter_ns() - overhead_wall_started
        self._event_record_cpu_time_ns += time.process_time_ns() - overhead_cpu_started

    def observe_layout(
        self,
        snapshot: PhysicalKvLayoutSnapshot,
        *,
        session_id: str,
        branch_group_id: str | None,
        cause: str,
        duration_ns: int = 0,
    ) -> None:
        overhead_wall_started = time.perf_counter_ns()
        overhead_cpu_started = time.process_time_ns()
        event_wall_before = self._event_record_wall_time_ns
        event_cpu_before = self._event_record_cpu_time_ns
        if self._level is TraceLevel.MINIMAL:
            self._layout_observation_calls += 1
            self._layout_diff_wall_time_ns += time.perf_counter_ns() - overhead_wall_started
            self._layout_diff_cpu_time_ns += time.process_time_ns() - overhead_cpu_started
            return
        current = {block.runtime_block_id: block for block in snapshot.blocks}
        newly_observed: set[str] = set()
        for block_id, block in current.items():
            previous = self._blocks.get(block_id)
            if previous is None or previous.allocation_epoch != block.allocation_epoch:
                newly_observed.add(block_id)
                self._record_state(
                    StateOperationType.STATE_ALLOC,
                    session_id=session_id,
                    branch_group_id=branch_group_id,
                    block=block,
                    cause=cause,
                    prior_refcount=None,
                    duration_ns=duration_ns,
                )
                publication = (
                    StateOperationType.STATE_PUBLISH
                    if block.state_class
                    in {KvStateClass.SHARED_PREFIX, KvStateClass.ROOT_REFERENCE}
                    else StateOperationType.STATE_APPEND
                )
                self._record_state(
                    publication,
                    session_id=session_id,
                    branch_group_id=branch_group_id,
                    block=block,
                    cause=cause,
                    prior_refcount=None,
                    duration_ns=0,
                )
                continue
            if block.refcount > previous.refcount:
                operation = StateOperationType.STATE_FORK
            elif block.refcount < previous.refcount:
                operation = StateOperationType.STATE_RECLAIM
            elif (
                block.branch_ids != previous.branch_ids or block.state_class != previous.state_class
            ):
                operation = StateOperationType.STATE_MAP
            else:
                operation = None
            if operation is not None:
                self._record_state(
                    operation,
                    session_id=session_id,
                    branch_group_id=branch_group_id,
                    block=block,
                    cause=cause,
                    prior_refcount=previous.refcount,
                    duration_ns=duration_ns,
                )
        if "suffix_" in cause or cause.endswith("divergence"):
            for block_id, block in current.items():
                if block.state_class is KvStateClass.PRIVATE_SUFFIX:
                    if block_id in newly_observed:
                        continue
                    self._record_state(
                        StateOperationType.STATE_APPEND,
                        session_id=session_id,
                        branch_group_id=branch_group_id,
                        block=block,
                        cause=cause,
                        prior_refcount=block.refcount,
                        duration_ns=0,
                    )
        # A block absent from a session-scoped layout has not necessarily been
        # released by vLLM. Retain it until explicit BlockPool evidence arrives.
        self._blocks.update(current)
        nested_event_wall = self._event_record_wall_time_ns - event_wall_before
        nested_event_cpu = self._event_record_cpu_time_ns - event_cpu_before
        self._layout_observation_calls += 1
        self._layout_diff_wall_time_ns += max(
            0, time.perf_counter_ns() - overhead_wall_started - nested_event_wall
        )
        self._layout_diff_cpu_time_ns += max(
            0, time.process_time_ns() - overhead_cpu_started - nested_event_cpu
        )

    def observe_release_evidence(
        self,
        evidence: PhysicalKvReleaseEvidence,
        *,
        session_id: str,
        branch_group_id: str | None,
        cause: str,
        duration_ns: int = 0,
    ) -> None:
        """Emit reclaim/free only from native exact-ID BlockPool observations."""

        if self._level is TraceLevel.MINIMAL:
            return
        for released in evidence.blocks:
            previous = self._blocks.get(released.runtime_block_id)
            if previous is None:
                continue
            if released.allocation_epoch != previous.allocation_epoch:
                raise ValueError("release evidence allocation epoch changed before trace release")
            if released.native_refcount != 0 or not released.allocator_available:
                raise ValueError("trace release requires native allocator-available evidence")
            observed = previous.model_copy(
                update={
                    "refcount": 0,
                    "branch_ids": (),
                    "cache_resident": released.block_hash_present,
                }
            )
            operation = (
                StateOperationType.STATE_RECLAIM
                if released.block_hash_present
                else StateOperationType.STATE_FREE
            )
            self._record_state(
                operation,
                session_id=session_id,
                branch_group_id=branch_group_id,
                block=observed,
                cause=cause,
                prior_refcount=previous.refcount,
                duration_ns=duration_ns,
            )
            if released.block_hash_present:
                self._blocks[released.runtime_block_id] = observed
            else:
                del self._blocks[released.runtime_block_id]

    def overhead_counters(self) -> dict[str, int | str]:
        """Return pointer-free in-process tracing work observed during the run."""

        return {
            "trace_level": self._level.value,
            "event_record_calls": self._event_record_calls,
            "layout_observation_calls": self._layout_observation_calls,
            "live_event_record_wall_time_ns": self._event_record_wall_time_ns,
            "live_event_record_cpu_time_ns": self._event_record_cpu_time_ns,
            "live_layout_diff_wall_time_ns": self._layout_diff_wall_time_ns,
            "live_layout_diff_cpu_time_ns": self._layout_diff_cpu_time_ns,
        }

    def emit_branch_workload_trace(self) -> tuple[BranchWorkloadEventV1, ...]:
        return tuple(self._branch_events)

    def emit_state_operation_trace(self) -> tuple[StateOperationEventV1, ...]:
        return tuple(self._state_events)


__all__ = ["VllmLiveTraceCapacityError", "VllmLiveTraceRecorder"]
