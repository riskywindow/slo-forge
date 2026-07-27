"""Causal CPU-reference capacity-reclamation evidence for BranchFabric gates.

This is deliberately a software-only local transaction.  It executes Helix's
capacity decision and Continuum's exact-state, filesystem transfer, ownership,
and integrity paths, but it never represents the reference runtime's simulated
device labels as physical GPU evidence.
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
import queue
import random
import statistics
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, TypeVar

from sloforge.continuum.adapters import (
    ReferenceHeadMajorAdapter,
    ReferenceTokenMajorAdapter,
    StaleOwnerEpochError,
)
from sloforge.continuum.conversion import direct_convert_capture
from sloforge.continuum.operations import (
    checkpoint_full,
    checkpoint_incremental,
    fork_checkpoint,
    restore_reference_capture,
    resume_checkpoint,
)
from sloforge.continuum.storage import ContentStore, FileContentStore, MemoryContentStore
from sloforge.continuum.transaction import (
    DurableCoordinator,
    GatewayCommitLedger,
    SessionLease,
)
from sloforge.continuum.transaction import (
    TokenEvent as GatewayTokenEvent,
)
from sloforge.continuum.transport import LocalFileTransport
from sloforge.helix.scheduler import (
    ClassResourceVectors,
    DecisionKind,
    EffectClass,
    EvidenceRef,
    PreservationMode,
    PreservationOption,
    PrivacyClass,
    ResourcePrices,
    ResourceVector,
    SchedulerConstraints,
    SchedulerPolicy,
    SchedulerRequest,
    ServingDemandSample,
    ServingSLO,
    ValuePrediction,
    WorkClass,
    WorkUnit,
    compile_resource_plan,
)

SEEDS = (41, 73, 113)
TRACE_LEVELS = ("disabled", "minimal", "full")
BASELINES = ("existing_serial", "optimized_bounded_parallel")
BRANCH_COUNT = 8
_STAMP = "2026-08-09T00:00:00Z"
_BASELINE_COMMIT = "46955be24d49af7090429444a0ef68f9a5695283"
_TENANT = "tenant-branchfabric-local"
_T = TypeVar("_T")


@dataclass(frozen=True, slots=True)
class OperationSample:
    operation: str
    branch_id: str
    duration_ns: int
    bytes: int
    concurrency: int
    queue_occupancy: int
    result: str = "success"
    detail: str = ""


@dataclass(slots=True)
class _Recorder:
    trace_level: str
    samples: list[OperationSample] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def measure(
        self,
        operation: str,
        branch_id: str,
        function: Callable[[], _T],
        *,
        bytes_count: int = 0,
        concurrency: int = 1,
        queue_occupancy: int = 0,
        detail: str = "",
    ) -> _T:
        if self.trace_level == "disabled":
            return function()
        started = time.perf_counter_ns()
        try:
            result = function()
        except Exception as error:
            self.record(
                OperationSample(
                    operation=operation,
                    branch_id=branch_id,
                    duration_ns=max(1, time.perf_counter_ns() - started),
                    bytes=bytes_count,
                    concurrency=concurrency,
                    queue_occupancy=queue_occupancy,
                    result="failure",
                    detail=type(error).__name__,
                )
            )
            raise
        self.record(
            OperationSample(
                operation=operation,
                branch_id=branch_id,
                duration_ns=max(1, time.perf_counter_ns() - started),
                bytes=bytes_count,
                concurrency=concurrency,
                queue_occupancy=queue_occupancy,
                detail=detail if self.trace_level == "full" else "",
            )
        )
        return result

    def record(self, sample: OperationSample) -> None:
        if self.trace_level != "disabled":
            with self._lock:
                self.samples.append(sample)


class _RecordingCoordinator(DurableCoordinator):
    def __init__(self, path: Path, recorder: _Recorder, branch_id: str) -> None:
        super().__init__(path)
        self._recorder = recorder
        self._branch_id = branch_id

    def commit_ownership(
        self, identifier: str, *, event_id: str, at_ms: int, state_version: int
    ) -> tuple[Any, SessionLease]:
        return self._recorder.measure(
            "commit",
            self._branch_id,
            lambda: super(_RecordingCoordinator, self).commit_ownership(
                identifier, event_id=event_id, at_ms=at_ms, state_version=state_version
            ),
            detail="durable SQLite ownership CAS",
        )


class _CorruptingReadStore:
    """Read-only fault wrapper; corruption cannot be published or committed."""

    def __init__(self, wrapped: ContentStore) -> None:
        self.wrapped = wrapped

    def read(
        self, tenant_id: str, reference: Any, *, offset: int = 0, length: int | None = None
    ) -> bytes:
        payload = self.wrapped.read(tenant_id, reference, offset=offset, length=length)
        if not payload:
            return b"\x01"
        return bytes((payload[0] ^ 1,)) + payload[1:]

    def put(self, *_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("fault wrapper is read-only")

    def publish(self, *_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("fault wrapper is read-only")


def _resource(cpu: int) -> ResourceVector:
    return ResourceVector(
        cpu_millicores=cpu,
        memory_mib=0,
        gpu_milliunits=0,
        storage_mib=0,
        storage_iops=0,
        network_mbps=0,
    )


def _evidence(label: str) -> EvidenceRef:
    return EvidenceRef(
        artifact_uri=f"raw://branchfabric/reclamation/{label}",
        artifact_sha256="0" * 64,
        sample_ids=(f"sample.{label}",),
    )


def _scheduler(seed: int) -> Any:
    continuum = PreservationOption(
        mode=PreservationMode.CONTINUUM,
        pause_ticks=1,
        checkpoint_interval_ticks=0,
        storage_mib_written=1,
        network_mib_transferred=1,
        cost_microunits=1,
        method_evidence=_evidence("continuum-executed-local"),
    )
    restart = PreservationOption(
        mode=PreservationMode.RESTART,
        pause_ticks=0,
        checkpoint_interval_ticks=0,
        storage_mib_written=0,
        network_mib_transferred=0,
        cost_microunits=0,
    )
    work = tuple(
        WorkUnit(
            work_id=f"rollout.{index}",
            branch_id=f"branch.{index}",
            work_class=WorkClass.ROLLOUT,
            tenant_id="tenant.local",
            privacy=PrivacyClass.TENANT_PRIVATE,
            effect=EffectClass.PURE,
            arrival_tick=0,
            duration_ticks=7,
            policy_age_ticks=0,
            resource_units=1,
            predicted_learning_value=ValuePrediction(
                value=float(BRANCH_COUNT - index),
                model_id="deterministic-local-value",
                model_version="1",
                evidence=_evidence(f"value-{index}"),
            ),
            preservation=(restart, continuum),
        )
        for index in range(BRANCH_COUNT)
    )
    zero = ResourceVector.zero()
    classes = ClassResourceVectors(
        serving=_resource(10),
        rollout=_resource(10),
        environment=zero,
        reward=zero,
        verifier=zero,
        training=zero,
        evaluation=zero,
    )
    forecast = tuple(
        ServingDemandSample(
            tick=tick,
            resource_units=8 if tick in {2, 3} else 2,
            predicted_latency_ms=80.0 if tick in {2, 3} else 20.0,
            predicted_queue_depth=16 if tick in {2, 3} else 2,
            evidence=_evidence(f"serving-{tick}"),
        )
        for tick in range(10)
    )
    request = SchedulerRequest(
        request_id=f"branchfabric.reclamation.{seed}",
        seed=seed,
        policy=SchedulerPolicy.HELIX_VALUE_AWARE,
        horizon_ticks=10,
        capacity=_resource(100),
        resource_vectors=classes,
        serving_slo=ServingSLO(
            reserved_capacity=_resource(80),
            maximum_predicted_latency_ms=100.0,
            maximum_predicted_queue_depth=20,
        ),
        serving_forecast=forecast,
        work=work,
        constraints=SchedulerConstraints(
            max_budget_microunits=10**9,
            prices=ResourcePrices(
                cpu_millicore_tick=1,
                memory_mib_tick=0,
                gpu_milliunit_tick=0,
                storage_mib_tick=0,
                storage_iop_tick=0,
                network_mbps_tick=0,
            ),
            max_policy_staleness_ticks=100,
            allowed_tenant_ids=("tenant.local",),
            maximum_privacy=PrivacyClass.TENANT_PRIVATE,
            allowed_effects=(EffectClass.PURE,),
            max_selected_branches=BRANCH_COUNT,
            max_preemptions_per_work=2,
            max_total_preemptions=BRANCH_COUNT * 2,
            capacity_lending=True,
        ),
        max_audit_records=10_000,
    )
    plan = compile_resource_plan(request)
    if not any(item.kind is DecisionKind.RECLAIM_CAPACITY for item in plan.decisions):
        raise RuntimeError("Helix did not causally reclaim capacity at the serving spike")
    if not all(item.selected_mode is PreservationMode.CONTINUUM for item in plan.preemptions):
        raise RuntimeError("Helix did not select Continuum state preservation")
    return plan


def _lease(coordinator: DurableCoordinator, branch_id: str) -> SessionLease:
    return coordinator.lease(branch_id)


def _payload_bytes(capture: Any) -> int:
    return sum(segment.descriptor.payload_bytes for segment in capture.segments)


def _nearest_rank(values: Sequence[int], fraction: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    return ordered[max(0, math.ceil(fraction * len(ordered)) - 1)]


def _stats(values: Sequence[int]) -> dict[str, int]:
    if not values:
        return {key: 0 for key in ("p50", "p90", "p95", "p99", "max")}
    return {
        "p50": _nearest_rank(values, 0.50),
        "p90": _nearest_rank(values, 0.90),
        "p95": _nearest_rank(values, 0.95),
        "p99": _nearest_rank(values, 0.99),
        "max": max(values),
    }


def _bounded_map(
    tasks: Sequence[Callable[[int, int], _T]], workers: int
) -> tuple[list[_T], int, int]:
    if not 1 <= workers <= BRANCH_COUNT:
        raise ValueError("worker count is outside the bounded branch queue")
    pending: queue.Queue[tuple[int, Callable[[int, int], _T]] | None] = queue.Queue(
        maxsize=BRANCH_COUNT
    )
    results: list[tuple[int, _T]] = []
    errors: list[BaseException] = []
    result_lock = threading.Lock()
    activity_lock = threading.Lock()
    active = 0
    maximum_active = 0
    maximum_occupancy = 0

    def worker() -> None:
        nonlocal active, maximum_active
        while True:
            item = pending.get(timeout=30.0)
            try:
                if item is None:
                    return
                index, function = item
                with activity_lock:
                    active += 1
                    maximum_active = max(maximum_active, active)
                    concurrency = active
                try:
                    value = function(concurrency, pending.qsize())
                except BaseException as error:
                    with result_lock:
                        errors.append(error)
                else:
                    with result_lock:
                        results.append((index, value))
                finally:
                    with activity_lock:
                        active -= 1
            finally:
                pending.task_done()

    threads = [threading.Thread(target=worker, daemon=True) for _ in range(workers)]
    for thread in threads:
        thread.start()
    for index, function in enumerate(tasks):
        pending.put((index, function), timeout=30.0)
        maximum_occupancy = max(maximum_occupancy, pending.qsize())
    for _ in threads:
        pending.put(None, timeout=30.0)
    pending.join()
    for thread in threads:
        thread.join(timeout=30.0)
        if thread.is_alive():
            raise TimeoutError("bounded reclamation worker did not stop")
    if errors:
        raise errors[0]
    return [value for _, value in sorted(results)], maximum_active, maximum_occupancy


def run_trial(
    *, seed: int, trace_level: str, baseline: str, work_root: Path, repetition: int
) -> dict[str, Any]:
    """Execute one causal serving-spike/preserve/reclaim/resume transaction."""

    if seed not in SEEDS:
        raise ValueError("reclamation evidence seed must be one of 41, 73, or 113")
    if trace_level not in TRACE_LEVELS or baseline not in BASELINES:
        raise ValueError("unknown tracing control or software baseline")
    workers = 1 if baseline == "existing_serial" else 4
    recorder = _Recorder(trace_level)
    trial_started = time.perf_counter_ns()
    plan = _scheduler(seed)
    source_store = MemoryContentStore()
    root_runtime = ReferenceTokenMajorAdapter(page_size_tokens=3)
    root_id = f"bf-root-{seed}-{repetition}"
    root_runtime.create_session(
        session_id=root_id,
        request_id="branchfabric-causal-reclamation",
        tenant_id=_TENANT,
        input_token_ids=(2, 3, 5, 7, 11, 13, 17, 19),
        seed=seed,
    )
    for event in root_runtime.stream_tokens(root_id, count=12):
        root_runtime.acknowledge_gateway(
            root_id, token_index=event.token_index, owner_epoch=event.owner_epoch
        )
    root_capture = root_runtime.capture_consistent(root_id)
    root_lease = SessionLease(
        session_id=root_id,
        owner_runtime=root_runtime.identity.runtime_name,
        owner_epoch=1,
        fencing_token=1,
        expiration_ms=120_000,
        coordinator_version=1,
        last_committed_state_version=root_capture.logical.state_version,
        last_committed_token_index=root_capture.logical.client_delivery.last_gateway_committed_token_index,
    )
    root_checkpoint = checkpoint_full(
        root_runtime,
        root_id,
        store=source_store,
        lease=root_lease,
        published_at_ms=1,
        capture_timestamp=_STAMP,
        git_commit=_BASELINE_COMMIT,
        continuum_version="0.1.0",
    )
    branch_ids = tuple(f"bf-branch-{seed}-{repetition}-{index}" for index in range(BRANCH_COUNT))
    branch_leases = tuple(
        SessionLease(
            session_id=branch_id,
            owner_runtime=root_runtime.identity.runtime_name,
            owner_epoch=1,
            fencing_token=1,
            expiration_ms=120_000,
            coordinator_version=1,
            last_committed_state_version=root_capture.logical.state_version,
            last_committed_token_index=root_capture.logical.client_delivery.last_gateway_committed_token_index,
        )
        for branch_id in branch_ids
    )
    fork_started = time.perf_counter_ns()
    forked = fork_checkpoint(
        root_checkpoint,
        store=source_store,
        expected_tenant_id=_TENANT,
        expected_model=root_runtime.config.model,
        branch_leases=branch_leases,
        seed=seed,
        published_at_ms=10,
        capture_timestamp=_STAMP,
        git_commit=_BASELINE_COMMIT,
        continuum_version="0.1.0",
    )
    recorder.record(
        OperationSample(
            "fork",
            root_id,
            max(1, time.perf_counter_ns() - fork_started),
            sum(ref.size_bytes for ref in root_checkpoint.chunk_references),
            1,
            0,
            detail="eight-way content-addressed fork" if trace_level == "full" else "",
        )
    )
    physical_root_bytes = sum(ref.size_bytes for ref in root_checkpoint.chunk_references)
    naive_branch_bytes = sum(
        sum(ref.size_bytes for ref in artifact.chunk_references) for artifact in forked.branches
    )
    unique_refs = {
        ref.digest: ref.size_bytes
        for artifact in forked.branches
        for ref in artifact.chunk_references
    }
    physical_branch_bytes = sum(unique_refs.values())

    coordinators: dict[str, _RecordingCoordinator] = {}
    gateways: dict[str, GatewayCommitLedger] = {}
    runtimes: dict[str, ReferenceTokenMajorAdapter] = {}
    trackers: dict[str, Any] = {}
    ready_started = time.perf_counter_ns()
    for index, (branch_id, artifact) in enumerate(zip(branch_ids, forked.branches, strict=True)):
        coordinator = _RecordingCoordinator(
            work_root / f"coordinator-{index}.sqlite", recorder, branch_id
        )
        coordinator.create_lease(
            session_id=branch_id,
            owner_runtime=root_runtime.identity.runtime_name,
            expiration_ms=120_000,
            initial_token_index=root_capture.logical.client_delivery.last_gateway_committed_token_index,
        )
        gateway = GatewayCommitLedger(work_root / f"gateway-{index}.sqlite")
        gateway.register(session_id=branch_id, owner_epoch=1)
        for token_index, token_id in enumerate(
            artifact.capsule.logical_state.token_history.committed_output_token_ids
        ):
            gateway.accept(
                GatewayTokenEvent(
                    session_id=branch_id,
                    owner_epoch=1,
                    token_index=token_index,
                    token_id=token_id,
                    state_commit_version=(
                        artifact.capsule.transaction.ownership_lease.last_committed_state_version
                    ),
                )
            )
        runtime = ReferenceTokenMajorAdapter(page_size_tokens=3)
        resume_checkpoint(
            artifact,
            store=source_store,
            destination=runtime,
            source=None,
            source_release_confirmed=True,
            expected_tenant_id=_TENANT,
            expected_model=root_runtime.config.model,
            coordinator=coordinator,
            gateway=gateway,
            seed=seed + index,
            now_ms=100,
        )
        coordinators[branch_id] = coordinator
        gateways[branch_id] = gateway
        runtimes[branch_id] = runtime
        trackers[branch_id] = runtime.start_dirty_tracking(branch_id)
    branch_readiness_ns = max(1, time.perf_counter_ns() - ready_started)

    for branch_id in branch_ids:
        runtime = runtimes[branch_id]
        before = runtime.capture_consistent(branch_id)
        started = time.perf_counter_ns()
        for event in runtime.stream_tokens(branch_id, count=4):
            gateways[branch_id].accept(
                GatewayTokenEvent(
                    session_id=event.session_id,
                    owner_epoch=event.owner_epoch,
                    token_index=event.token_index,
                    token_id=event.token_id,
                    state_commit_version=event.state_commit_version,
                    transaction_id=event.transaction_id,
                )
            )
            runtime.acknowledge_gateway(
                branch_id, token_index=event.token_index, owner_epoch=event.owner_epoch
            )
        metadata = runtime.inspect_session(branch_id)
        current_lease = coordinators[branch_id].lease(branch_id)
        coordinators[branch_id].record_committed_progress(
            session_id=branch_id,
            owner_runtime=runtime.identity.runtime_name,
            owner_epoch=metadata.owner_epoch,
            fencing_token=current_lease.fencing_token,
            state_version=metadata.state_version,
            token_index=metadata.committed_output_index,
            now_ms=150,
        )
        after = runtime.capture_consistent(branch_id)
        duration = max(1, time.perf_counter_ns() - started)
        changed = _payload_bytes(after) - _payload_bytes(before)
        recorder.record(OperationSample("cow", branch_id, duration, max(0, changed), 1, 0))
        recorder.record(OperationSample("append", branch_id, duration, max(0, changed), 1, 0))

    spike_tick = min(item.tick for item in plan.ticks if not item.reclaimed_capacity.is_zero())
    admissions_stopped = not any(
        decision.tick == spike_tick and decision.kind is DecisionKind.START
        for decision in plan.decisions
    )
    pause_started = time.perf_counter_ns()

    def checkpoint_task(branch_id: str, artifact: Any) -> Callable[[int, int], tuple[Any, Any]]:
        def execute(concurrency: int, occupancy: int) -> tuple[Any, Any]:
            runtime = runtimes[branch_id]
            runtime.pause_session(branch_id)
            delta = recorder.measure(
                "delta",
                branch_id,
                lambda: runtime.export_final_delta(trackers[branch_id]),
                concurrency=concurrency,
                queue_occupancy=occupancy,
            )
            runtime.stop_dirty_tracking(trackers[branch_id])
            checkpoint = recorder.measure(
                "checkpoint",
                branch_id,
                lambda: checkpoint_incremental(
                    runtime,
                    branch_id,
                    store=source_store,
                    lease=_lease(coordinators[branch_id], branch_id),
                    parent=artifact,
                    published_at_ms=200,
                    capture_timestamp=_STAMP,
                    git_commit=_BASELINE_COMMIT,
                    continuum_version="0.1.0",
                ),
                bytes_count=_payload_bytes(runtime.capture_consistent(branch_id)),
                concurrency=concurrency,
                queue_occupancy=occupancy,
            )
            return delta, checkpoint

        return execute

    checkpoint_results, checkpoint_concurrency, checkpoint_queue = _bounded_map(
        [
            checkpoint_task(branch_id, artifact)
            for branch_id, artifact in zip(branch_ids, forked.branches, strict=True)
        ],
        workers,
    )
    pause_checkpoint_ns = max(1, time.perf_counter_ns() - pause_started)
    checkpoints = [item[1] for item in checkpoint_results]
    delta_bytes = sum(
        segment.descriptor.payload_bytes
        for delta, _checkpoint in checkpoint_results
        for segment in delta.changed_segments
    )

    reclaim_started = time.perf_counter_ns()
    if any(runtimes[key].inspect_session(key).lifecycle.value != "paused" for key in branch_ids):
        raise RuntimeError("capacity reclaimed before every rollout source was paused")
    reclaimed_capacity_units = BRANCH_COUNT
    reclaim_ns = max(1, time.perf_counter_ns() - reclaim_started)
    recorder.record(
        OperationSample(
            "reclaim",
            "branch-group",
            reclaim_ns,
            0,
            1,
            0,
            detail="logical CPU rollout slots; zero physical GPU bytes"
            if trace_level == "full"
            else "",
        )
    )
    slo_restored = plan.ticks[spike_tick].serving_slo_satisfied
    slo_restore_ns = max(1, time.perf_counter_ns() - pause_started)

    destination_store = FileContentStore(work_root / "destination-store")
    transport = LocalFileTransport(work_root / "spool")
    migration_started = time.perf_counter_ns()

    def transfer_task(branch_id: str, checkpoint: Any) -> Callable[[int, int], Any]:
        def execute(concurrency: int, occupancy: int) -> Any:
            capture = restore_reference_capture(
                checkpoint,
                store=source_store,
                expected_tenant_id=_TENANT,
                expected_model=root_runtime.config.model,
            )
            recorder.measure(
                "transform",
                branch_id,
                lambda: direct_convert_capture(
                    capture,
                    destination=ReferenceHeadMajorAdapter(page_size_tokens=5),
                    maximum_temporary_bytes=512,
                ),
                bytes_count=_payload_bytes(capture),
                concurrency=concurrency,
                queue_occupancy=occupancy,
            )
            return recorder.measure(
                "transfer",
                branch_id,
                lambda: transport.transfer(
                    source=source_store,
                    destination=destination_store,
                    tenant_id=_TENANT,
                    references=checkpoint.chunk_references,
                    deadline_us=60_000_000,
                    seed=seed,
                ),
                bytes_count=sum(ref.size_bytes for ref in checkpoint.chunk_references),
                concurrency=concurrency,
                queue_occupancy=occupancy,
                detail="real local filesystem spool with digest validation",
            )

        return execute

    receipts, transfer_concurrency, transfer_queue = _bounded_map(
        [
            transfer_task(branch_id, checkpoint)
            for branch_id, checkpoint in zip(branch_ids, checkpoints, strict=True)
        ],
        workers,
    )
    migration_ns = max(1, time.perf_counter_ns() - migration_started)

    corrupt_rejected = False
    try:
        restore_reference_capture(
            checkpoints[0],
            store=_CorruptingReadStore(destination_store),
            expected_tenant_id=_TENANT,
            expected_model=root_runtime.config.model,
        )
    except Exception:
        corrupt_rejected = True
    if not corrupt_rejected:
        raise RuntimeError("corrupted transferred state was accepted")

    resume_started = time.perf_counter_ns()
    destinations: dict[str, ReferenceHeadMajorAdapter] = {}

    def resume_task(branch_id: str, checkpoint: Any) -> Callable[[int, int], Any]:
        def execute(concurrency: int, occupancy: int) -> Any:
            destination = ReferenceHeadMajorAdapter(page_size_tokens=5)
            result = resume_checkpoint(
                checkpoint,
                store=destination_store,
                destination=destination,
                source=runtimes[branch_id],
                expected_tenant_id=_TENANT,
                expected_model=root_runtime.config.model,
                coordinator=coordinators[branch_id],
                gateway=gateways[branch_id],
                seed=seed + 100,
                now_ms=300,
            )
            with recorder._lock:
                destinations[branch_id] = destination
            return result

        return execute

    resume_results, resume_concurrency, resume_queue = _bounded_map(
        [
            resume_task(branch_id, checkpoint)
            for branch_id, checkpoint in zip(branch_ids, checkpoints, strict=True)
        ],
        workers,
    )
    resume_ns = max(1, time.perf_counter_ns() - resume_started)
    stale_rejected = True
    for branch_id in branch_ids:
        try:
            runtimes[branch_id].generate_token(branch_id)
        except StaleOwnerEpochError:
            pass
        else:
            stale_rejected = False
        for event in destinations[branch_id].stream_tokens(branch_id, count=2):
            gateways[branch_id].accept(
                GatewayTokenEvent(
                    session_id=event.session_id,
                    owner_epoch=event.owner_epoch,
                    token_index=event.token_index,
                    token_id=event.token_id,
                    state_commit_version=event.state_commit_version,
                    transaction_id=event.transaction_id,
                )
            )
            destinations[branch_id].acknowledge_gateway(
                branch_id, token_index=event.token_index, owner_epoch=event.owner_epoch
            )
    if not stale_rejected:
        raise RuntimeError("a stale source emitted after ownership commit")

    total_interruption_ns = max(1, time.perf_counter_ns() - pause_started)
    total_ns = max(1, time.perf_counter_ns() - trial_started)
    state_bytes = sum(receipt.unique_plaintext_bytes for receipt in receipts)
    for coordinator in coordinators.values():
        coordinator.close()
    for gateway in gateways.values():
        gateway.close()
    destination_store.close()

    operations = sorted(
        (asdict(item) for item in recorder.samples),
        key=lambda item: (item["operation"], item["branch_id"], item["duration_ns"]),
    )
    operation_stats = {
        operation: _stats(
            [item["duration_ns"] for item in operations if item["operation"] == operation]
        )
        for operation in (
            "fork",
            "cow",
            "append",
            "checkpoint",
            "delta",
            "transform",
            "transfer",
            "commit",
            "reclaim",
        )
    }
    operation_fractions = {
        operation: sum(
            int(item["duration_ns"]) for item in operations if item["operation"] == operation
        )
        / total_ns
        for operation in operation_stats
    }
    return {
        "schema_version": "sloforge.branchfabric.reclamation-trial/v1",
        "evidence_class": "CPU_REFERENCE_LOCAL_TRANSACTION",
        "baseline_commit": _BASELINE_COMMIT,
        "seed": seed,
        "repetition": repetition,
        "trace_level": trace_level,
        "software_baseline": baseline,
        "worker_bound": workers,
        "branch_count": BRANCH_COUNT,
        "scheduler_plan_id": plan.plan_id,
        "spike_tick": spike_tick,
        "admissions_stopped": admissions_stopped,
        "slo_restored": slo_restored,
        "slo_restore_ns": slo_restore_ns,
        "reclaimed_capacity_units": reclaimed_capacity_units,
        "physical_gpu_capacity_reclaimed": 0,
        "physical_gpu_bytes_reclaimed": 0,
        "work_lost_ticks": sum(item.selected.lost_work_ticks for item in plan.preemptions),
        "work_preserved_ticks": sum(
            item.selected.preserved_work_ticks for item in plan.preemptions
        ),
        "root_state_bytes": physical_root_bytes,
        "naive_branch_state_bytes": naive_branch_bytes,
        "physical_branch_state_bytes": physical_branch_bytes,
        "sharing_saved_bytes": max(0, naive_branch_bytes - physical_branch_bytes),
        "delta_bytes": delta_bytes,
        "transferred_state_bytes": state_bytes,
        "branch_readiness_ns": branch_readiness_ns,
        "pause_checkpoint_ns": pause_checkpoint_ns,
        "migration_ns": migration_ns,
        "resume_ns": resume_ns,
        "total_interruption_ns": total_interruption_ns,
        "total_wall_ns": total_ns,
        "end_to_end_denominator": "total_wall_ns",
        "maximum_queue_occupancy": max(checkpoint_queue, transfer_queue, resume_queue),
        "maximum_concurrency": max(
            checkpoint_concurrency, transfer_concurrency, resume_concurrency
        ),
        "stale_source_rejected": stale_rejected,
        "corrupt_state_rejected": corrupt_rejected,
        "all_transactions_completed": all(
            result.transaction.phase.value == "COMPLETED" for result in resume_results
        ),
        "hidden_fallback": False,
        "requested_engine": "continuum-reference-cpu-local-file",
        "actual_engine": "continuum-reference-cpu-local-file",
        "operation_samples": operations,
        "operation_stats_ns": operation_stats,
        "candidate_operation_fraction_of_total_wall": operation_fractions,
        "candidate_fraction_scope": (
            "summed inclusive host operation time divided by total wall time; parallel and "
            "COW/append observations overlap and fractions are not additive"
        ),
        "limitations": [
            "No physical GPU, GPU memory, PCIe, NVLink, RDMA, NIC, FPGA, or DPU was exercised.",
            "Reference model state is deterministic fixture state, not a production model KV allocation.",
            "Reclaimed capacity is a Helix CPU scheduler unit; physical GPU reclamation is exactly zero.",
            "Wall-clock samples are host observations and are not deterministic protocol outputs.",
        ],
    }


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def run_campaign(
    output: Path, *, repetitions: int = 2, order_seed: int = 20260809
) -> dict[str, Any]:
    """Run a randomized 3-seed x tracing x baseline campaign and preserve raw samples."""

    if not 1 <= repetitions <= 10:
        raise ValueError("repetitions must be within 1..10")
    matrix = [
        (seed, trace, baseline, repetition)
        for seed in SEEDS
        for trace in TRACE_LEVELS
        for baseline in BASELINES
        for repetition in range(repetitions)
    ]
    random.Random(order_seed).shuffle(matrix)
    output.mkdir(parents=True, exist_ok=True)
    raw_root = output / "raw"
    raw_root.mkdir(exist_ok=True)
    trials = []
    with TemporaryDirectory(prefix="sloforge-bf-reclamation-") as temporary:
        temporary_root = Path(temporary)
        for index, (seed, trace, baseline, repetition) in enumerate(matrix):
            trial_root = temporary_root / (f"{index:03d}-{baseline}-{trace}-s{seed}-r{repetition}")
            trial_root.mkdir(parents=True, exist_ok=True)
            trials.append(
                run_trial(
                    seed=seed,
                    trace_level=trace,
                    baseline=baseline,
                    work_root=trial_root,
                    repetition=repetition,
                )
            )
    raw_payload = b"".join(_canonical(trial) + b"\n" for trial in trials)
    raw_path = raw_root / "trials.jsonl"
    raw_path.write_bytes(raw_payload)
    comparisons: dict[str, Any] = {}
    operation_aggregates: dict[str, Any] = {}
    candidate_fractions: dict[str, Any] = {}
    for baseline in BASELINES:
        selected = [trial for trial in trials if trial["software_baseline"] == baseline]
        comparisons[baseline] = {
            metric: _stats([int(trial[metric]) for trial in selected])
            for metric in (
                "branch_readiness_ns",
                "pause_checkpoint_ns",
                "migration_ns",
                "resume_ns",
                "total_interruption_ns",
                "total_wall_ns",
            )
        }
        operation_aggregates[baseline] = {
            operation: _stats(
                [
                    int(sample["duration_ns"])
                    for trial in selected
                    for sample in trial["operation_samples"]
                    if sample["operation"] == operation
                ]
            )
            for operation in (
                "fork",
                "cow",
                "append",
                "checkpoint",
                "delta",
                "transform",
                "transfer",
                "commit",
                "reclaim",
            )
        }
        candidate_fractions[baseline] = {
            operation: statistics.median(
                float(trial["candidate_operation_fraction_of_total_wall"][operation])
                for trial in selected
            )
            for operation in (
                "fork",
                "cow",
                "append",
                "checkpoint",
                "delta",
                "transform",
                "transfer",
                "commit",
                "reclaim",
            )
        }
    serial = comparisons["existing_serial"]["total_interruption_ns"]["p50"]
    optimized = comparisons["optimized_bounded_parallel"]["total_interruption_ns"]["p50"]
    campaign = {
        "schema_version": "sloforge.branchfabric.reclamation-campaign/v1",
        "evidence_class": "CPU_REFERENCE_LOCAL_TRANSACTION",
        "baseline_commit": _BASELINE_COMMIT,
        "seeds": list(SEEDS),
        "repetitions": repetitions,
        "randomized_order_seed": order_seed,
        "trial_count": len(trials),
        "trace_controls": list(TRACE_LEVELS),
        "software_baselines": list(BASELINES),
        "raw_samples": "raw/trials.jsonl",
        "raw_samples_sha256": hashlib.sha256(raw_payload).hexdigest(),
        "comparisons_ns": comparisons,
        "operation_aggregates_ns": operation_aggregates,
        "candidate_operation_fraction_p50_of_total_wall": candidate_fractions,
        "candidate_fraction_denominator": "per-trial total_wall_ns",
        "candidate_fraction_scope": (
            "median of per-trial summed inclusive host operation fractions; parallel and "
            "COW/append observations overlap and fractions are not additive"
        ),
        "maximum_queue_occupancy": max(int(trial["maximum_queue_occupancy"]) for trial in trials),
        "maximum_concurrency": max(int(trial["maximum_concurrency"]) for trial in trials),
        "optimized_interruption_speedup": serial / optimized if optimized else 0.0,
        "all_slo_restored": all(trial["slo_restored"] for trial in trials),
        "all_faults_rejected": all(
            trial["stale_source_rejected"] and trial["corrupt_state_rejected"] for trial in trials
        ),
        "hidden_fallback": False,
        "hardware_claim": False,
        "gpu_measurement": False,
    }
    campaign_path = output / "campaign.json"
    campaign_path.write_bytes(_canonical(campaign) + b"\n")
    environment = {
        "evidence_class": "CPU_REFERENCE_LOCAL_TRANSACTION",
        "platform": platform.platform(),
        "python": platform.python_version(),
        "processor": platform.processor(),
        "physical_gpu_measured": False,
        "network_fabric_measured": False,
        "billable_resources_created": False,
    }
    environment_path = output / "environment.json"
    environment_path.write_bytes(_canonical(environment) + b"\n")
    manifest = {
        "campaign": hashlib.sha256(campaign_path.read_bytes()).hexdigest(),
        "environment": hashlib.sha256(environment_path.read_bytes()).hexdigest(),
        "raw_trials": hashlib.sha256(raw_path.read_bytes()).hexdigest(),
    }
    (output / "MANIFEST.json").write_bytes(_canonical(manifest) + b"\n")
    return campaign


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=2)
    parser.add_argument("--order-seed", type=int, default=20260809)
    arguments = parser.parse_args()
    result = run_campaign(
        arguments.output, repetitions=arguments.repetitions, order_seed=arguments.order_seed
    )
    print(json.dumps(result, sort_keys=True))
