"""Bounded CPU microcharacterization of Helix and Continuum metadata paths.

The controlled operation stream is synthetic.  Durations, process CPU time, and
Python allocation peaks are measured on the executing host.  Operations use the
real in-memory Continuum content store and SQLite coordinator whenever those
classes expose an isolated public operation.  Page, ancestry, dirty-bit, and
lineage cases are faithful isolated data-structure operations and are labeled as
such.  This distinction is part of every raw sample and derived summary.

The sharded and batched cases are deliberately small software baselines, not a
replacement metadata manager.  They execute the same isolated semantic unit as
the isolated global-lock case, so their comparisons never use the broader real
Continuum operations as the denominator.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import statistics
import threading
import time
import tracemalloc
from collections import deque
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from sloforge.continuum.storage import MemoryContentStore
from sloforge.continuum.transaction import CutoverPhase, DurableCoordinator
from sloforge.helix.characterization.matrix import EvidenceClass
from sloforge.helix.ir import Digest, LineageReference, LineageRelation

MAX_OPERATIONS_PER_SAMPLE = 4096
MAX_REPETITIONS = 100
MAX_WARMUPS = 20
MAX_THREADS = 16
MAX_TOTAL_SAMPLES = 50_000
MAX_OUTPUT_BYTES = 128 * 1024 * 1024
DEFAULT_TIMEOUT_SECONDS = 30.0


class MetadataStudyModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, validate_default=True)


class MetadataOperation(StrEnum):
    BRANCH_CREATE_BOOKKEEPING = "branch_create_bookkeeping"
    ANCESTRY_LOOKUP = "ancestry_lookup"
    PAGE_LOOKUP = "page_lookup"
    REFCOUNT_UPDATE = "refcount_update"
    DIRTY_UPDATE = "dirty_update"
    STATE_HASH_LOOKUP = "state_hash_lookup"
    ALLOCATION = "allocation"
    FREE = "free"
    TRANSACTION_LOOKUP = "transaction_lookup"
    EPOCH_VALIDATION = "epoch_validation"
    PUBLISH = "publish"
    ABORT = "abort"
    RECLAIM = "reclaim"
    LINEAGE_BOOKKEEPING = "lineage_bookkeeping"


class MetadataImplementation(StrEnum):
    CURRENT_SOFTWARE = "current_software"
    ISOLATED_GLOBAL_LOCK = "isolated_global_lock"
    ISOLATED_SHARDED = "isolated_sharded"
    ISOLATED_BATCHED = "isolated_batched"


class MeasurementScope(StrEnum):
    ACTUAL_CONTINUUM_SOFTWARE = "actual_continuum_software"
    EXISTING_HELIX_MODEL = "existing_helix_model"
    FAITHFUL_ISOLATED_OPERATION = "faithful_isolated_operation"
    SOFTWARE_COMPARISON_BASELINE = "software_comparison_baseline"


_ISOLATED_BASELINE_OPERATIONS = frozenset(
    {
        MetadataOperation.BRANCH_CREATE_BOOKKEEPING,
        MetadataOperation.ANCESTRY_LOOKUP,
        MetadataOperation.PAGE_LOOKUP,
        MetadataOperation.REFCOUNT_UPDATE,
        MetadataOperation.DIRTY_UPDATE,
        MetadataOperation.STATE_HASH_LOOKUP,
        MetadataOperation.LINEAGE_BOOKKEEPING,
    }
)


class MetadataStudyConfig(MetadataStudyModel):
    schema_version: Literal["sloforge.branchfabric.metadata-study-config/v1"] = (
        "sloforge.branchfabric.metadata-study-config/v1"
    )
    seed: Annotated[int, Field(ge=0, le=2**64 - 1)]
    operations: tuple[MetadataOperation, ...] = tuple(MetadataOperation)
    operations_per_thread: int = Field(default=32, ge=1, le=MAX_OPERATIONS_PER_SAMPLE)
    warmup_repetitions: int = Field(default=2, ge=0, le=MAX_WARMUPS)
    measurement_repetitions: int = Field(default=7, ge=1, le=MAX_REPETITIONS)
    thread_counts: tuple[int, ...] = (1, 2, 4)
    shard_count: int = Field(default=16, ge=2, le=256)
    working_set_entries: int = Field(default=64, ge=2, le=4096)
    state_payload_bytes: int = Field(default=256, ge=1, le=1024 * 1024)
    include_software_baselines: bool = True
    sample_timeout_seconds: float = Field(
        default=DEFAULT_TIMEOUT_SECONDS,
        gt=0.0,
        le=300.0,
        allow_inf_nan=False,
    )

    @model_validator(mode="after")
    def validate_bounds(self) -> Self:
        if not self.operations or len(set(self.operations)) != len(self.operations):
            raise ValueError("operations must be non-empty and unique")
        if not self.thread_counts or len(set(self.thread_counts)) != len(self.thread_counts):
            raise ValueError("thread_counts must be non-empty and unique")
        if tuple(sorted(self.thread_counts)) != self.thread_counts:
            raise ValueError("thread_counts must be strictly increasing")
        if self.thread_counts[0] != 1 or self.thread_counts[-1] > MAX_THREADS:
            raise ValueError(f"thread_counts must begin at 1 and not exceed {MAX_THREADS}")
        baseline_cases = (
            3 * len(_ISOLATED_BASELINE_OPERATIONS.intersection(self.operations))
            if self.include_software_baselines
            else 0
        )
        case_count = len(self.operations) + baseline_cases
        sample_count = (
            case_count
            * len(self.thread_counts)
            * (self.warmup_repetitions + self.measurement_repetitions)
        )
        if sample_count > MAX_TOTAL_SAMPLES:
            raise ValueError(f"metadata study exceeds {MAX_TOTAL_SAMPLES} samples")
        total_operations = self.operations_per_thread * self.thread_counts[-1]
        if total_operations > MAX_OPERATIONS_PER_SAMPLE:
            raise ValueError(
                "operations_per_thread times maximum threads exceeds the per-sample bound"
            )
        return self


class CounterAvailability(MetadataStudyModel):
    cpu_cycles: Literal["unavailable"] = "unavailable"
    cache_misses: Literal["unavailable"] = "unavailable"
    memory_bandwidth: Literal["unavailable"] = "unavailable"
    lock_wait_time: Literal["unavailable"] = "unavailable"
    syscall_count: Literal["unavailable"] = "unavailable"
    resident_memory_delta: Literal["unavailable"] = "unavailable"
    process_cpu_time: Literal["time.process_time_ns"] = "time.process_time_ns"
    python_allocation_peak: Literal["tracemalloc.get_traced_memory"] = (
        "tracemalloc.get_traced_memory"
    )
    unavailable_reason: str = Field(min_length=1, max_length=1024)


class TrialDescriptor(MetadataStudyModel):
    operation: MetadataOperation
    implementation: MetadataImplementation
    measurement_scope: MeasurementScope
    thread_count: int = Field(ge=1, le=MAX_THREADS)
    repetition: int = Field(ge=0, le=MAX_REPETITIONS)
    warmup: bool
    randomized_sequence: int = Field(ge=0, le=MAX_TOTAL_SAMPLES)
    trial_seed: Annotated[int, Field(ge=0, le=2**64 - 1)]


class MetadataRawSample(MetadataStudyModel):
    schema_version: Literal["sloforge.branchfabric.metadata-raw-sample/v1"] = (
        "sloforge.branchfabric.metadata-raw-sample/v1"
    )
    workload_evidence_class: Literal[EvidenceClass.SYNTHETIC] = EvidenceClass.SYNTHETIC
    timing_evidence_class: Literal[EvidenceClass.HARDWARE_BACKED_REAL] = (
        EvidenceClass.HARDWARE_BACKED_REAL
    )
    descriptor: TrialDescriptor
    operation_count: int = Field(ge=1, le=MAX_OPERATIONS_PER_SAMPLE)
    semantic_updates_per_operation: int = Field(ge=1, le=16)
    monotonic_start_ns: int = Field(ge=0)
    duration_ns: int = Field(gt=0)
    process_cpu_ns: int = Field(ge=0)
    operations_per_second: float = Field(gt=0.0, allow_inf_nan=False)
    cpu_time_per_operation_ns: float = Field(ge=0.0, allow_inf_nan=False)
    traced_python_peak_delta_bytes: int = Field(ge=0)
    result_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_under_test: str = Field(min_length=1, max_length=1024)
    timing_scope: str = Field(min_length=1, max_length=1024)


class MetadataSummary(MetadataStudyModel):
    schema_version: Literal["sloforge.branchfabric.metadata-summary/v1"] = (
        "sloforge.branchfabric.metadata-summary/v1"
    )
    operation: MetadataOperation
    implementation: MetadataImplementation
    measurement_scope: MeasurementScope
    thread_count: int = Field(ge=1, le=MAX_THREADS)
    sample_count: int = Field(ge=1, le=MAX_REPETITIONS)
    operation_count: int = Field(ge=1)
    amortized_wall_ns_per_operation_median: float = Field(gt=0.0, allow_inf_nan=False)
    amortized_wall_ns_per_operation_p95: float = Field(gt=0.0, allow_inf_nan=False)
    amortized_wall_ns_per_operation_p99: float = Field(gt=0.0, allow_inf_nan=False)
    throughput_ops_per_second_median: float = Field(gt=0.0, allow_inf_nan=False)
    throughput_ops_per_second_p95: float = Field(gt=0.0, allow_inf_nan=False)
    process_cpu_per_operation_ns_median: float = Field(ge=0.0, allow_inf_nan=False)
    traced_python_peak_delta_bytes_median: float = Field(ge=0.0, allow_inf_nan=False)
    raw_sample_sequences: tuple[int, ...] = Field(min_length=1, max_length=MAX_REPETITIONS)
    outlier_policy: Literal["none removed"] = "none removed"
    percentile_method: Literal["Hyndman-Fan type 7"] = "Hyndman-Fan type 7"


class ThreadScalingObservation(MetadataStudyModel):
    operation: MetadataOperation
    implementation: MetadataImplementation
    thread_count: int = Field(ge=1, le=MAX_THREADS)
    median_throughput_ops_per_second: float = Field(gt=0.0, allow_inf_nan=False)
    throughput_relative_to_one_thread: float = Field(gt=0.0, allow_inf_nan=False)
    parallel_efficiency: float = Field(gt=0.0, allow_inf_nan=False)
    classification: Literal[
        "one_thread_reference",
        "throughput_increased",
        "thread_scaling_flat",
        "thread_scaling_degraded",
    ]
    interpretation_limit: Literal[
        "Python thread scaling includes GIL and scheduler effects; lock contention is not isolated"
    ] = "Python thread scaling includes GIL and scheduler effects; lock contention is not isolated"


class SoftwareBaselineComparison(MetadataStudyModel):
    operation: MetadataOperation
    candidate: Literal[
        MetadataImplementation.ISOLATED_SHARDED,
        MetadataImplementation.ISOLATED_BATCHED,
    ]
    thread_count: int = Field(ge=1, le=MAX_THREADS)
    current_median_throughput_ops_per_second: float = Field(gt=0.0, allow_inf_nan=False)
    candidate_median_throughput_ops_per_second: float = Field(gt=0.0, allow_inf_nan=False)
    throughput_ratio: float = Field(gt=0.0, allow_inf_nan=False)
    current_median_amortized_wall_ns_per_operation: float = Field(gt=0.0, allow_inf_nan=False)
    candidate_median_amortized_wall_ns_per_operation: float = Field(gt=0.0, allow_inf_nan=False)
    amortized_wall_time_ratio: float = Field(gt=0.0, allow_inf_nan=False)
    matched_semantic_scope: Literal["faithful isolated metadata operation"] = (
        "faithful isolated metadata operation"
    )
    hardware_claim_permitted: Literal[False] = False


class MetadataStudyReport(MetadataStudyModel):
    schema_version: Literal["sloforge.branchfabric.metadata-study/v1"] = (
        "sloforge.branchfabric.metadata-study/v1"
    )
    experiment_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    config: MetadataStudyConfig
    workload_evidence_class: Literal[EvidenceClass.SYNTHETIC] = EvidenceClass.SYNTHETIC
    timing_evidence_class: Literal[EvidenceClass.HARDWARE_BACKED_REAL] = (
        EvidenceClass.HARDWARE_BACKED_REAL
    )
    wall_clock: Literal["time.perf_counter_ns"] = "time.perf_counter_ns"
    cpu_clock: Literal["time.process_time_ns"] = "time.process_time_ns"
    run_order: tuple[TrialDescriptor, ...]
    raw_samples: tuple[MetadataRawSample, ...]
    summaries: tuple[MetadataSummary, ...]
    thread_scaling: tuple[ThreadScalingObservation, ...]
    software_baseline_comparisons: tuple[SoftwareBaselineComparison, ...]
    counters: CounterAvailability
    classifications: tuple[
        Literal["latency_observed", "throughput_observed", "python_thread_scaling_observed"], ...
    ]
    limitations: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def samples_follow_declared_schedule(self) -> Self:
        if len(self.run_order) != len(self.raw_samples):
            raise ValueError("run order and raw sample counts differ")
        if tuple(item.descriptor for item in self.raw_samples) != self.run_order:
            raise ValueError("raw samples do not preserve declared randomized run order")
        return self


@dataclass(slots=True)
class _PreparedCase:
    execute_one: Callable[[int], str]
    execute_many: Callable[[tuple[int, ...]], list[tuple[int, str]]] | None
    cleanup: Callable[[], None]
    source_under_test: str
    timing_scope: str
    semantic_updates_per_operation: int = 1


def _payload(seed: int, index: int, size: int) -> bytes:
    block = hashlib.sha256(f"metadata-payload\0{seed}\0{index}".encode()).digest()
    return (block * math.ceil(size / len(block)))[:size]


def _prepare_store_parents(
    *, seed: int, count: int, payload_bytes: int
) -> tuple[MemoryContentStore, tuple[str, ...]]:
    store = MemoryContentStore()
    parent_ids: list[str] = []
    for index in range(count):
        reference = store.put("tenant-characterization", _payload(seed, index, payload_bytes))
        parent = store.publish(
            tenant_id="tenant-characterization",
            kind="complete",
            chunks=(reference,),
            published_at_ms=index,
        )
        parent_ids.append(parent.manifest_id)
    return store, tuple(parent_ids)


def _prepare_coordinator(
    *, seed: int, count: int, transactions: bool
) -> tuple[DurableCoordinator, tuple[str, ...]]:
    coordinator = DurableCoordinator(":memory:")
    identifiers: list[str] = []
    plan_hash = hashlib.sha256(f"metadata-plan\0{seed}".encode()).hexdigest()
    for index in range(count):
        session_id = f"metadata-session-{seed}-{index}"
        coordinator.create_lease(
            session_id=session_id,
            owner_runtime="reference-cpu",
            expiration_ms=10_000_000,
            initial_token_index=7,
        )
        if transactions:
            transaction = coordinator.begin_transaction(
                session_id=session_id,
                destination_candidate="reference-cpu-destination",
                migration_plan_hash=plan_hash,
                seed=seed + index,
                now_ms=index,
                timeout_ms=1_000_000,
            )
            identifiers.append(transaction.transaction_id)
        else:
            identifiers.append(session_id)
    return coordinator, tuple(identifiers)


def _prepare_current_case(
    operation: MetadataOperation,
    *,
    seed: int,
    count: int,
    working_set: int,
    payload_bytes: int,
) -> _PreparedCase:
    tenant = "tenant-characterization"
    if operation is MetadataOperation.BRANCH_CREATE_BOOKKEEPING:
        store, parents = _prepare_store_parents(seed=seed, count=count, payload_bytes=payload_bytes)

        def fork(index: int) -> str:
            return store.fork(
                tenant_id=tenant,
                parent_manifest_id=parents[index],
                published_at_ms=100_000 + index,
            ).manifest_id

        return _PreparedCase(
            fork,
            None,
            lambda: None,
            "sloforge.continuum.storage.MemoryContentStore.fork",
            "content-addressed fork manifest construction, validation, publication, and refcount",
        )
    if operation is MetadataOperation.ANCESTRY_LOOKUP:
        ancestry = {f"branch-{index}": f"parent-{index // 4}" for index in range(working_set)}
        lock = threading.RLock()

        def ancestry_lookup(index: int) -> str:
            with lock:
                return ancestry[f"branch-{index % working_set}"]

        return _PreparedCase(
            ancestry_lookup,
            None,
            lambda: None,
            "faithful Python RLock + dict matching current in-process metadata patterns",
            "isolated branch-to-parent ancestry lookup",
        )
    if operation is MetadataOperation.PAGE_LOOKUP:
        pages = {f"page-{index}": f"physical-{index}" for index in range(working_set)}
        lock = threading.RLock()

        def page_lookup(index: int) -> str:
            with lock:
                return pages[f"page-{index % working_set}"]

        return _PreparedCase(
            page_lookup,
            None,
            lambda: None,
            "faithful Python RLock + dict matching Continuum reference runtime page maps",
            "isolated page identifier lookup; no device page-table access",
        )
    if operation is MetadataOperation.REFCOUNT_UPDATE:
        store, parents = _prepare_store_parents(seed=seed, count=count, payload_bytes=payload_bytes)

        def refcount_pair(index: int) -> str:
            branch = store.fork(
                tenant_id=tenant,
                parent_manifest_id=parents[index],
                published_at_ms=200_000 + index,
            )
            store.delete_manifest(tenant, branch.manifest_id)
            return branch.manifest_id

        return _PreparedCase(
            refcount_pair,
            None,
            lambda: None,
            "sloforge.continuum.storage.MemoryContentStore.publish/delete_manifest",
            "one increment plus one decrement and manifest bookkeeping per operation",
            semantic_updates_per_operation=2,
        )
    if operation is MetadataOperation.DIRTY_UPDATE:
        histories = {index: deque[int](maxlen=128) for index in range(min(working_set, count))}
        overflow_floor = {index: 0 for index in histories}
        lock = threading.RLock()

        def dirty_update(index: int) -> str:
            key = index % len(histories)
            with lock:
                history = histories[key]
                if len(history) == history.maxlen:
                    discarded = history.popleft()
                    overflow_floor[key] = max(overflow_floor[key], discarded)
                history.append(index + 1)
            return f"{key}:{index + 1}"

        return _PreparedCase(
            dirty_update,
            None,
            lambda: None,
            "faithful clone of DeterministicHybridRuntimeAdapter._record_dirty under its RLock",
            "bounded dirty-history append and overflow-floor update",
        )
    if operation is MetadataOperation.STATE_HASH_LOOKUP:
        store = MemoryContentStore()
        references = tuple(
            store.put(tenant, _payload(seed, index, payload_bytes)) for index in range(working_set)
        )

        def hash_lookup(index: int) -> str:
            reference = references[index % working_set]
            payload = store.read(tenant, reference)
            return hashlib.sha256(payload).hexdigest()

        return _PreparedCase(
            hash_lookup,
            None,
            lambda: None,
            "sloforge.continuum.storage.MemoryContentStore.read",
            "content-addressed lookup plus bounded decode and SHA-256 integrity verification",
        )
    if operation is MetadataOperation.ALLOCATION:
        store = MemoryContentStore()
        payloads = tuple(_payload(seed, index, payload_bytes) for index in range(count))

        def allocate(index: int) -> str:
            return store.put(tenant, payloads[index]).digest

        return _PreparedCase(
            allocate,
            None,
            lambda: None,
            "sloforge.continuum.storage.MemoryContentStore.put",
            "content hash, chunk reference construction, lock, and in-memory chunk allocation",
        )
    if operation is MetadataOperation.FREE:
        store, parents = _prepare_store_parents(seed=seed, count=count, payload_bytes=payload_bytes)

        def free_manifest(index: int) -> str:
            store.delete_manifest(tenant, parents[index])
            return parents[index]

        return _PreparedCase(
            free_manifest,
            None,
            lambda: None,
            "sloforge.continuum.storage.MemoryContentStore.delete_manifest",
            "manifest removal and chunk refcount decrement; chunk reclamation excluded",
        )
    if operation is MetadataOperation.TRANSACTION_LOOKUP:
        coordinator, identifiers = _prepare_coordinator(seed=seed, count=count, transactions=True)

        def transaction_lookup(index: int) -> str:
            return coordinator.transaction(identifiers[index]).transaction_id

        return _PreparedCase(
            transaction_lookup,
            None,
            coordinator.close,
            "sloforge.continuum.transaction.DurableCoordinator.transaction (:memory: SQLite)",
            "locked SQLite lookup plus strict Pydantic JSON deserialization",
        )
    if operation is MetadataOperation.EPOCH_VALIDATION:
        coordinator, sessions = _prepare_coordinator(seed=seed, count=count, transactions=False)

        def epoch_validation(index: int) -> str:
            lease = coordinator.assert_owner(
                session_id=sessions[index],
                owner_runtime="reference-cpu",
                owner_epoch=1,
                fencing_token=1,
                now_ms=index,
            )
            return f"{lease.session_id}:{lease.owner_epoch}"

        return _PreparedCase(
            epoch_validation,
            None,
            coordinator.close,
            "sloforge.continuum.transaction.DurableCoordinator.assert_owner (:memory: SQLite)",
            "lease lookup, strict deserialization, expiration, epoch, runtime, and fence validation",
        )
    if operation is MetadataOperation.PUBLISH:
        store = MemoryContentStore()
        references = tuple(
            store.put(tenant, _payload(seed, index, payload_bytes)) for index in range(count)
        )

        def publish(index: int) -> str:
            return store.publish(
                tenant_id=tenant,
                kind="complete",
                chunks=(references[index],),
                published_at_ms=index,
            ).manifest_id

        return _PreparedCase(
            publish,
            None,
            lambda: None,
            "sloforge.continuum.storage.MemoryContentStore.publish",
            "manifest hash/model validation, existence checks, refcount increment, and publication",
        )
    if operation is MetadataOperation.ABORT:
        coordinator, identifiers = _prepare_coordinator(seed=seed, count=count, transactions=True)
        payload_hash = hashlib.sha256(f"metadata-abort\0{seed}".encode()).hexdigest()

        def abort(index: int) -> str:
            return coordinator.transition(
                identifiers[index],
                expected=CutoverPhase.PROPOSED,
                target=CutoverPhase.REJECTED,
                event_id=f"metadata-abort-{index}",
                at_ms=10_000 + index,
                payload_hash=payload_hash,
                failure_reason="metadata characterization controlled abort",
            ).transaction_id

        return _PreparedCase(
            abort,
            None,
            coordinator.close,
            "sloforge.continuum.transaction.DurableCoordinator.transition (:memory: SQLite)",
            "validated PROPOSED-to-REJECTED journaled transaction transition",
        )
    if operation is MetadataOperation.RECLAIM:
        store = MemoryContentStore()
        for index in range(count):
            store.put(tenant, _payload(seed, index, payload_bytes), expires_at_ms=0)

        def reclaim(index: int) -> str:
            deleted = store.gc(now_ms=1, maximum_deletions=1)
            if len(deleted) != 1:
                raise RuntimeError("reclamation benchmark did not reclaim exactly one chunk")
            return deleted[0]

        return _PreparedCase(
            reclaim,
            None,
            lambda: None,
            "sloforge.continuum.storage.MemoryContentStore.gc",
            "bounded sorted zero-ref expired-chunk scan and one chunk reclamation",
        )
    if operation is MetadataOperation.LINEAGE_BOOKKEEPING:

        def lineage(index: int) -> str:
            digest = hashlib.sha256(f"lineage\0{seed}\0{index}".encode()).hexdigest()
            link = LineageReference(
                artifact_id=f"artifact-{seed}-{index}",
                artifact_kind="metadata-study",
                relation=LineageRelation.PARENT,
                digest=Digest(value=digest),
            )
            return f"{link.artifact_id}:{link.digest.value}"

        return _PreparedCase(
            lineage,
            None,
            lambda: None,
            "sloforge.helix.ir.LineageReference and Digest",
            "strict lineage and digest model construction and validation",
        )
    raise AssertionError(f"unhandled metadata operation {operation}")


class _IsolatedIndex:
    def __init__(self, *, working_set: int, shard_count: int, batched: bool) -> None:
        self.working_set = working_set
        self.shard_count = shard_count
        self.batched = batched
        self.locks = tuple(threading.RLock() for _ in range(shard_count))
        self.ancestry = {f"branch-{index}": f"parent-{index // 4}" for index in range(working_set)}
        self.pages = {f"page-{index}": f"physical-{index}" for index in range(working_set)}
        self.refcounts = {f"page-{index}": 1 for index in range(working_set)}
        self.dirty = {f"page-{index}": 0 for index in range(working_set)}
        self.hashes = {
            hashlib.sha256(f"state-{index}".encode()).hexdigest(): f"state-{index}"
            for index in range(working_set)
        }
        self.hash_keys = tuple(self.hashes)
        self.branches: dict[str, str] = {}
        self.lineage: dict[str, str] = {}

    def _key(self, operation: MetadataOperation, index: int) -> str:
        item = index % self.working_set
        if operation is MetadataOperation.ANCESTRY_LOOKUP:
            return f"branch-{item}"
        if operation in {
            MetadataOperation.PAGE_LOOKUP,
            MetadataOperation.REFCOUNT_UPDATE,
            MetadataOperation.DIRTY_UPDATE,
        }:
            return f"page-{item}"
        if operation is MetadataOperation.STATE_HASH_LOOKUP:
            return self.hash_keys[item]
        if operation is MetadataOperation.BRANCH_CREATE_BOOKKEEPING:
            return f"new-branch-{index}"
        return f"lineage-{index}"

    def _lock(self, index: int) -> threading.RLock:
        # The synthetic key stream already has a deterministic integer identity;
        # using it avoids charging the sharded candidate an unrelated SHA-256.
        return self.locks[index % self.shard_count]

    def _execute_unlocked(self, operation: MetadataOperation, index: int) -> str:
        key = self._key(operation, index)
        if operation is MetadataOperation.ANCESTRY_LOOKUP:
            return self.ancestry[key]
        if operation is MetadataOperation.PAGE_LOOKUP:
            return self.pages[key]
        if operation is MetadataOperation.STATE_HASH_LOOKUP:
            return self.hashes[key]
        if operation is MetadataOperation.REFCOUNT_UPDATE:
            self.refcounts[key] += 1
            return key
        if operation is MetadataOperation.DIRTY_UPDATE:
            self.dirty[key] = index + 1
            return key
        if operation is MetadataOperation.BRANCH_CREATE_BOOKKEEPING:
            parent = f"parent-{index % self.working_set}"
            self.branches[key] = parent
            ref_key = f"page-{index % self.working_set}"
            self.refcounts[ref_key] += 1
            return f"{key}:{parent}"
        if operation is MetadataOperation.LINEAGE_BOOKKEEPING:
            parent = f"artifact-{index % self.working_set}"
            self.lineage[key] = parent
            return f"{key}:{parent}"
        raise AssertionError(f"unsupported isolated operation {operation}")

    def execute(self, operation: MetadataOperation, index: int) -> str:
        with self._lock(index):
            return self._execute_unlocked(operation, index)

    def execute_batch(
        self, operation: MetadataOperation, indices: tuple[int, ...]
    ) -> list[tuple[int, str]]:
        # The batched comparison uses one global critical section for a worker's
        # complete command batch.  This reduces lock operations, but can lengthen
        # occupancy; thread scaling records that tradeoff.
        with self.locks[0]:
            return [(index, self._execute_unlocked(operation, index)) for index in indices]


def _prepare_isolated_case(
    operation: MetadataOperation,
    implementation: MetadataImplementation,
    *,
    working_set: int,
    shard_count: int,
) -> _PreparedCase:
    effective_shards = (
        shard_count if implementation is MetadataImplementation.ISOLATED_SHARDED else 1
    )
    batched = implementation is MetadataImplementation.ISOLATED_BATCHED
    index = _IsolatedIndex(
        working_set=working_set,
        shard_count=effective_shards,
        batched=batched,
    )

    def execute_one(item: int) -> str:
        return index.execute(operation, item)

    def execute_many(items: tuple[int, ...]) -> list[tuple[int, str]]:
        return index.execute_batch(operation, items)

    return _PreparedCase(
        execute_one,
        execute_many if batched else None,
        lambda: None,
        f"faithful isolated {implementation.value} Python metadata index",
        "matched semantic metadata operation; excludes Continuum serialization and integrity work",
    )


def _measurement_scope(
    operation: MetadataOperation, implementation: MetadataImplementation
) -> MeasurementScope:
    if implementation is not MetadataImplementation.CURRENT_SOFTWARE:
        return MeasurementScope.SOFTWARE_COMPARISON_BASELINE
    if operation in {
        MetadataOperation.ANCESTRY_LOOKUP,
        MetadataOperation.PAGE_LOOKUP,
        MetadataOperation.DIRTY_UPDATE,
    }:
        return MeasurementScope.FAITHFUL_ISOLATED_OPERATION
    if operation is MetadataOperation.LINEAGE_BOOKKEEPING:
        return MeasurementScope.EXISTING_HELIX_MODEL
    return MeasurementScope.ACTUAL_CONTINUUM_SOFTWARE


def _case_specs(
    config: MetadataStudyConfig,
) -> list[tuple[MetadataOperation, MetadataImplementation]]:
    cases = [
        (operation, MetadataImplementation.CURRENT_SOFTWARE) for operation in config.operations
    ]
    if config.include_software_baselines:
        for operation in config.operations:
            if operation not in _ISOLATED_BASELINE_OPERATIONS:
                continue
            cases.extend(
                (
                    (operation, MetadataImplementation.ISOLATED_GLOBAL_LOCK),
                    (operation, MetadataImplementation.ISOLATED_SHARDED),
                    (operation, MetadataImplementation.ISOLATED_BATCHED),
                )
            )
    return cases


def _trial_seed(config_seed: int, sequence_material: str) -> int:
    digest = hashlib.sha256(f"metadata-trial\0{config_seed}\0{sequence_material}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def build_metadata_run_order(config: MetadataStudyConfig) -> tuple[TrialDescriptor, ...]:
    """Build a deterministic warmup-first, randomized-within-phase run order."""

    phases = ((True, config.warmup_repetitions), (False, config.measurement_repetitions))
    ordered: list[TrialDescriptor] = []
    sequence = 0
    for warmup, repetitions in phases:
        phase: list[TrialDescriptor] = []
        for operation, implementation in _case_specs(config):
            for thread_count in config.thread_counts:
                for repetition in range(repetitions):
                    material = (
                        f"{operation.value}:{implementation.value}:{thread_count}:"
                        f"{repetition}:{int(warmup)}"
                    )
                    phase.append(
                        TrialDescriptor(
                            operation=operation,
                            implementation=implementation,
                            measurement_scope=_measurement_scope(operation, implementation),
                            thread_count=thread_count,
                            repetition=repetition,
                            warmup=warmup,
                            randomized_sequence=0,
                            trial_seed=_trial_seed(config.seed, material),
                        )
                    )
        random.Random(config.seed ^ (0xA5A5 if warmup else 0x5A5A)).shuffle(phase)
        for item in phase:
            ordered.append(item.model_copy(update={"randomized_sequence": sequence}))
            sequence += 1
    return tuple(ordered)


def _partitions(total: int, threads: int) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(range(worker, total, threads)) for worker in range(threads))


def _execute_prepared(
    prepared: _PreparedCase,
    *,
    total_operations: int,
    thread_count: int,
    timeout_seconds: float,
) -> tuple[int, int, int, int, str]:
    start_gate = threading.Event()
    partitions = _partitions(total_operations, thread_count)

    def worker(indices: tuple[int, ...]) -> list[tuple[int, str]]:
        start_gate.wait(timeout=timeout_seconds)
        if not start_gate.is_set():
            raise TimeoutError("metadata trial start gate timed out")
        if prepared.execute_many is not None:
            return prepared.execute_many(indices)
        return [(index, prepared.execute_one(index)) for index in indices]

    executor = ThreadPoolExecutor(max_workers=thread_count, thread_name_prefix="metadata-study")
    futures = [executor.submit(worker, indices) for indices in partitions]
    tracemalloc.start(1)
    before_current, _before_peak = tracemalloc.get_traced_memory()
    tracemalloc.reset_peak()
    monotonic_start_ns = time.perf_counter_ns()
    cpu_start_ns = time.process_time_ns()
    start_gate.set()
    deadline = time.monotonic() + timeout_seconds
    indexed_results: list[tuple[int, str]] = []
    try:
        for future in futures:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("metadata sample exceeded its timeout")
            try:
                indexed_results.extend(future.result(timeout=remaining))
            except FutureTimeoutError as exc:
                raise TimeoutError("metadata sample exceeded its timeout") from exc
        process_cpu_ns = time.process_time_ns() - cpu_start_ns
        duration_ns = time.perf_counter_ns() - monotonic_start_ns
        _after_current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
        executor.shutdown(wait=True, cancel_futures=True)
        prepared.cleanup()
    if len(indexed_results) != total_operations:
        raise RuntimeError("metadata trial did not return one result per operation")
    indexed_results.sort(key=lambda item: item[0])
    if tuple(index for index, _value in indexed_results) != tuple(range(total_operations)):
        raise RuntimeError("metadata trial returned duplicate or missing operation indexes")
    # Operation scheduling can associate a reclaimed object with a different
    # worker index.  The semantic result is the multiset of outputs, so hashing
    # sorted values preserves determinism without pretending scheduling order is
    # stable.
    canonical = json.dumps(
        sorted(value for _index, value in indexed_results),
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode()
    return (
        monotonic_start_ns,
        max(1, duration_ns),
        process_cpu_ns,
        max(0, peak - before_current),
        hashlib.sha256(canonical).hexdigest(),
    )


def _run_trial(config: MetadataStudyConfig, descriptor: TrialDescriptor) -> MetadataRawSample:
    total_operations = config.operations_per_thread * descriptor.thread_count
    if descriptor.implementation is MetadataImplementation.CURRENT_SOFTWARE:
        prepared = _prepare_current_case(
            descriptor.operation,
            seed=descriptor.trial_seed,
            count=total_operations,
            working_set=min(config.working_set_entries, total_operations),
            payload_bytes=config.state_payload_bytes,
        )
    else:
        prepared = _prepare_isolated_case(
            descriptor.operation,
            descriptor.implementation,
            working_set=min(config.working_set_entries, total_operations),
            shard_count=config.shard_count,
        )
    start_ns, duration_ns, cpu_ns, peak_bytes, result_sha256 = _execute_prepared(
        prepared,
        total_operations=total_operations,
        thread_count=descriptor.thread_count,
        timeout_seconds=config.sample_timeout_seconds,
    )
    return MetadataRawSample(
        descriptor=descriptor,
        operation_count=total_operations,
        semantic_updates_per_operation=prepared.semantic_updates_per_operation,
        monotonic_start_ns=start_ns,
        duration_ns=duration_ns,
        process_cpu_ns=cpu_ns,
        operations_per_second=total_operations * 1_000_000_000.0 / duration_ns,
        cpu_time_per_operation_ns=cpu_ns / total_operations,
        traced_python_peak_delta_bytes=peak_bytes,
        result_sha256=result_sha256,
        source_under_test=prepared.source_under_test,
        timing_scope=prepared.timing_scope,
    )


def _percentile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise ValueError("percentile requires samples")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _summaries(samples: Sequence[MetadataRawSample]) -> tuple[MetadataSummary, ...]:
    groups: dict[
        tuple[MetadataOperation, MetadataImplementation, MeasurementScope, int],
        list[MetadataRawSample],
    ] = {}
    for sample in samples:
        if sample.descriptor.warmup:
            continue
        key = (
            sample.descriptor.operation,
            sample.descriptor.implementation,
            sample.descriptor.measurement_scope,
            sample.descriptor.thread_count,
        )
        groups.setdefault(key, []).append(sample)
    results: list[MetadataSummary] = []
    for (operation, implementation, scope, thread_count), group in sorted(
        groups.items(), key=lambda item: tuple(str(part) for part in item[0])
    ):
        latencies = [item.duration_ns / item.operation_count for item in group]
        throughputs = [item.operations_per_second for item in group]
        results.append(
            MetadataSummary(
                operation=operation,
                implementation=implementation,
                measurement_scope=scope,
                thread_count=thread_count,
                sample_count=len(group),
                operation_count=sum(item.operation_count for item in group),
                amortized_wall_ns_per_operation_median=statistics.median(latencies),
                amortized_wall_ns_per_operation_p95=_percentile(latencies, 0.95),
                amortized_wall_ns_per_operation_p99=_percentile(latencies, 0.99),
                throughput_ops_per_second_median=statistics.median(throughputs),
                throughput_ops_per_second_p95=_percentile(throughputs, 0.95),
                process_cpu_per_operation_ns_median=statistics.median(
                    item.cpu_time_per_operation_ns for item in group
                ),
                traced_python_peak_delta_bytes_median=statistics.median(
                    item.traced_python_peak_delta_bytes for item in group
                ),
                raw_sample_sequences=tuple(
                    item.descriptor.randomized_sequence
                    for item in sorted(group, key=lambda value: value.descriptor.repetition)
                ),
            )
        )
    return tuple(results)


def _thread_scaling(
    summaries: Sequence[MetadataSummary],
) -> tuple[ThreadScalingObservation, ...]:
    references = {
        (summary.operation, summary.implementation): summary.throughput_ops_per_second_median
        for summary in summaries
        if summary.thread_count == 1
    }
    observations: list[ThreadScalingObservation] = []
    for summary in summaries:
        reference = references[(summary.operation, summary.implementation)]
        relative = summary.throughput_ops_per_second_median / reference
        efficiency = relative / summary.thread_count
        classification: Literal[
            "one_thread_reference",
            "throughput_increased",
            "thread_scaling_flat",
            "thread_scaling_degraded",
        ]
        if summary.thread_count == 1:
            classification = "one_thread_reference"
        elif relative < 0.9:
            classification = "thread_scaling_degraded"
        elif relative <= 1.1:
            classification = "thread_scaling_flat"
        else:
            classification = "throughput_increased"
        observations.append(
            ThreadScalingObservation(
                operation=summary.operation,
                implementation=summary.implementation,
                thread_count=summary.thread_count,
                median_throughput_ops_per_second=summary.throughput_ops_per_second_median,
                throughput_relative_to_one_thread=relative,
                parallel_efficiency=efficiency,
                classification=classification,
            )
        )
    return tuple(observations)


def _comparisons(
    summaries: Sequence[MetadataSummary],
) -> tuple[SoftwareBaselineComparison, ...]:
    lookup = {(item.operation, item.implementation, item.thread_count): item for item in summaries}
    comparisons: list[SoftwareBaselineComparison] = []
    for operation in sorted(_ISOLATED_BASELINE_OPERATIONS, key=lambda item: item.value):
        thread_counts = sorted(
            item.thread_count
            for item in summaries
            if item.operation is operation
            and item.implementation is MetadataImplementation.ISOLATED_GLOBAL_LOCK
        )
        for thread_count in thread_counts:
            current = lookup[(operation, MetadataImplementation.ISOLATED_GLOBAL_LOCK, thread_count)]
            for candidate_name in (
                MetadataImplementation.ISOLATED_SHARDED,
                MetadataImplementation.ISOLATED_BATCHED,
            ):
                candidate = lookup[(operation, candidate_name, thread_count)]
                comparisons.append(
                    SoftwareBaselineComparison(
                        operation=operation,
                        candidate=candidate_name,
                        thread_count=thread_count,
                        current_median_throughput_ops_per_second=(
                            current.throughput_ops_per_second_median
                        ),
                        candidate_median_throughput_ops_per_second=(
                            candidate.throughput_ops_per_second_median
                        ),
                        throughput_ratio=(
                            candidate.throughput_ops_per_second_median
                            / current.throughput_ops_per_second_median
                        ),
                        current_median_amortized_wall_ns_per_operation=(
                            current.amortized_wall_ns_per_operation_median
                        ),
                        candidate_median_amortized_wall_ns_per_operation=(
                            candidate.amortized_wall_ns_per_operation_median
                        ),
                        amortized_wall_time_ratio=(
                            candidate.amortized_wall_ns_per_operation_median
                            / current.amortized_wall_ns_per_operation_median
                        ),
                    )
                )
    return tuple(comparisons)


def run_metadata_study(config: MetadataStudyConfig) -> MetadataStudyReport:
    """Run the bounded study and return raw and derived evidence without filtering."""

    schedule = build_metadata_run_order(config)
    samples = tuple(_run_trial(config, descriptor) for descriptor in schedule)
    summaries = _summaries(samples)
    config_payload = config.model_dump_json(exclude_none=False)
    experiment_id = hashlib.sha256(config_payload.encode()).hexdigest()
    return MetadataStudyReport(
        experiment_id=experiment_id,
        config=config,
        run_order=schedule,
        raw_samples=samples,
        summaries=summaries,
        thread_scaling=_thread_scaling(summaries),
        software_baseline_comparisons=_comparisons(summaries),
        counters=CounterAvailability(
            unavailable_reason=(
                "This portable CPU study does not invoke platform perf counters. Values are not "
                "estimated from wall time; use a host-specific profiler in a hardware-backed run."
            )
        ),
        classifications=(
            "latency_observed",
            "throughput_observed",
            "python_thread_scaling_observed",
        ),
        limitations=(
            "Synthetic keys and access streams are controlled sweeps, not production distributions.",
            "perf_counter/process_time include Python, executor dispatch, tracing, and scheduler overhead.",
            "tracemalloc collection is enabled inside every timed interval and can distort absolute latency.",
            "In-memory SQLite excludes filesystem durability and storage latency.",
            "Thread scaling cannot separate the Python GIL, scheduler effects, and lock contention.",
            "No cache-miss, CPU-cycle, memory-bandwidth, lock-wait, or syscall values are inferred.",
            "The optimized baselines cover isolated metadata semantics, not full Continuum transactions.",
        ),
    )


def write_metadata_study(report: MetadataStudyReport, output: Path) -> str:
    """Write one bounded JSON report and return the exact artifact SHA-256."""

    payload = (report.model_dump_json(indent=2) + "\n").encode()
    if len(payload) > MAX_OUTPUT_BYTES:
        raise ValueError(f"metadata report exceeds {MAX_OUTPUT_BYTES} bytes")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(output)
    return hashlib.sha256(payload).hexdigest()


def measurement_samples(
    report: MetadataStudyReport,
    *,
    operation: MetadataOperation | None = None,
    implementation: MetadataImplementation | None = None,
) -> tuple[MetadataRawSample, ...]:
    """Select preserved non-warmup samples without mutating or removing outliers."""

    return tuple(
        sample
        for sample in report.raw_samples
        if not sample.descriptor.warmup
        and (operation is None or sample.descriptor.operation is operation)
        and (implementation is None or sample.descriptor.implementation is implementation)
    )


def measured_operations(report: MetadataStudyReport) -> frozenset[MetadataOperation]:
    return frozenset(sample.descriptor.operation for sample in report.raw_samples)
