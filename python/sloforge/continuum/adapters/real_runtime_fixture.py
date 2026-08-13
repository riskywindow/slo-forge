"""Deterministic process-local fixture for the real-runtime state boundary.

The fixture models a reference-counted paged KV pool and emits GPU-shaped trace
records, but every record is explicitly marked ``SIMULATED_HARDWARE``.  It is a
semantic test double for adapter lifecycle and accounting tests, not evidence of
GPU execution or of a serving runtime's physical sharing behavior.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from hashlib import sha256

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
    TransportType,
    WorkloadProvenance,
    seal_event,
)

from .real_runtime import (
    MAX_LIVE_BRANCHES,
    ConcurrentDecodeResult,
    GpuMemoryState,
    KvStateClass,
    LiveBranchPoint,
    LiveSessionPhase,
    LogicalRuntimeState,
    PhysicalKvBlock,
    PhysicalKvBlockReleaseEvidence,
    PhysicalKvLayoutSnapshot,
    PhysicalKvReleaseEvidence,
    RealRuntimeCapability,
    RealRuntimeStateAdapter,
    RefcountEvidence,
    RuntimeCapabilityMatrix,
    RuntimeModelIdentity,
    RuntimeScopeContract,
    RuntimeSessionRef,
    SharedRootReference,
    token_history_sha256,
)

FIXTURE_RUNTIME = "sloforge-deterministic-paged-kv-fixture"
FIXTURE_RUNTIME_VERSION = "1.0.0"
FIXTURE_ADAPTER_VERSION = "1.0.0"
MAX_TIMEOUT_S = 3_600.0
MAX_TRACE_EVENTS = 1_000_000


class RuntimeFixtureError(RuntimeError):
    """The deterministic fixture rejected an invalid lifecycle operation."""


@dataclass
class _FixtureBlock:
    runtime_block_id: str
    block_index: int
    kv_cache_group: int
    layer_ids: tuple[str, ...]
    device: str
    dtype: str
    bytes: int
    block_size_tokens: int
    logical_token_start: int
    logical_token_end_exclusive: int
    owner_session_ids: set[str]
    state_class: KvStateClass
    allocation_epoch: int
    cache_resident: bool


@dataclass
class _FixtureSession:
    session_id: str
    adapter_session_id: str
    seed: int
    token_ids: list[int]
    phase: LiveSessionPhase
    branch_id: str | None = None
    parent_session_id: str | None = None
    root_reference_id: str | None = None
    prefix_block_ids: list[str] = field(default_factory=list)
    private_block_ids: list[str] = field(default_factory=list)


@dataclass
class _FixtureRoot:
    root_reference_id: str
    source_session_id: str
    prefix_token_count: int
    block_ids: tuple[str, ...]
    physical_bytes: int
    created_at_monotonic_ns: int
    owner_session_ids: set[str]


class DeterministicRuntimeFixtureAdapter(RealRuntimeStateAdapter):
    """Bounded semantic fixture for shared-prefix and private-suffix invariants."""

    def __init__(
        self,
        *,
        seed: int,
        model_id: str = "fixture/model",
        model_revision: str = "fixture-revision",
        tokenizer_id: str = "fixture/tokenizer",
        tokenizer_revision: str = "fixture-revision",
        dtype: str = "bfloat16",
        device: str = "cuda:0",
        policy_epoch: str = "fixture-policy-1",
        block_size_tokens: int = 16,
        block_bytes: int = 4 * 1024 * 1024,
        kv_pool_block_capacity: int = 4_096,
        layer_count: int = 32,
        max_fanout: int = 32,
        trace_id: str | None = None,
        max_trace_events: int = 200_000,
    ) -> None:
        self._validate_seed(seed)
        if block_size_tokens < 1:
            raise ValueError("block_size_tokens must be positive")
        if block_bytes < 1:
            raise ValueError("block_bytes must be positive")
        if kv_pool_block_capacity < 1:
            raise ValueError("kv_pool_block_capacity must be positive")
        if layer_count < 1:
            raise ValueError("layer_count must be positive")
        if not 1 <= max_fanout <= MAX_LIVE_BRANCHES:
            raise ValueError(f"max_fanout must be within 1..{MAX_LIVE_BRANCHES}")
        if not 1 <= max_trace_events <= MAX_TRACE_EVENTS:
            raise ValueError(f"max_trace_events must be within 1..{MAX_TRACE_EVENTS}")

        self._seed = seed
        self._block_size_tokens = block_size_tokens
        self._block_bytes = block_bytes
        self._pool_capacity = kv_pool_block_capacity
        self._layer_ids = tuple(f"attention.layer.{index}" for index in range(layer_count))
        self._device = self._identifier(device, "device")
        self._dtype = self._identifier(dtype, "dtype")
        self._max_trace_events = max_trace_events
        self._trace_id = trace_id or f"fixture-trace-{seed}"
        self._identifier(self._trace_id, "trace_id")
        self._identity = RuntimeModelIdentity(
            runtime=FIXTURE_RUNTIME,
            runtime_version=FIXTURE_RUNTIME_VERSION,
            adapter_version=FIXTURE_ADAPTER_VERSION,
            model_id=self._identifier(model_id, "model_id"),
            model_revision=self._identifier(model_revision, "model_revision"),
            tokenizer_id=self._identifier(tokenizer_id, "tokenizer_id"),
            tokenizer_revision=self._identifier(tokenizer_revision, "tokenizer_revision"),
            dtype=self._dtype,
            device=self._device,
            policy_epoch=self._identifier(policy_epoch, "policy_epoch"),
        )
        self._scope = RuntimeScopeContract(
            same_process_or_adapter_scope="one DeterministicRuntimeFixtureAdapter instance"
        )
        self._capability_matrix = RuntimeCapabilityMatrix(
            identity=self._identity,
            capabilities=frozenset(
                capability
                for capability in RealRuntimeCapability
                if capability is not RealRuntimeCapability.RUN_CONCURRENT_INDEPENDENT_SESSIONS
            ),
            scope=self._scope,
            version_pin=f"{FIXTURE_RUNTIME}=={FIXTURE_RUNTIME_VERSION}",
            internal_interfaces=("deterministic fixture only; no serving-runtime interface",),
            max_sessions=max_fanout + 1,
            max_fanout=max_fanout,
        )
        self._sessions: dict[str, _FixtureSession] = {}
        self._destroyed_session_ids: set[str] = set()
        self._roots: dict[str, _FixtureRoot] = {}
        self._blocks: dict[str, _FixtureBlock] = {}
        self._released_blocks: dict[str, tuple[int, int]] = {}
        self._free_block_indices = list(range(kv_pool_block_capacity - 1, -1, -1))
        self._allocation_epoch = 0
        self._snapshot_epoch = 0
        self._root_sequence = 0
        self._branchpoint_sequence = 0
        self._clock_ns = 1_000_000 + seed * 1_000
        self._clock_origin_ns = self._clock_ns
        self._branch_trace: list[BranchWorkloadEventV1] = []
        self._state_trace: list[StateOperationEventV1] = []
        self._cleaned = False

    @property
    def capability_matrix(self) -> RuntimeCapabilityMatrix:
        return self._capability_matrix

    def start_session(
        self, session_id: str, *, token_ids: tuple[int, ...], seed: int, timeout_s: float
    ) -> RuntimeSessionRef:
        self._active_call(timeout_s)
        self._validate_seed(seed)
        session_id = self._identifier(session_id, "session_id")
        token_history_sha256(token_ids)
        if not token_ids:
            raise RuntimeFixtureError("fixture sessions require a non-empty prefix token history")
        if session_id in self._sessions or session_id in self._destroyed_session_ids:
            raise RuntimeFixtureError(f"session {session_id!r} already exists or was destroyed")
        if len(self._sessions) >= self._capability_matrix.max_sessions:
            raise RuntimeFixtureError("bounded fixture session capacity is exhausted")
        session = _FixtureSession(
            session_id=session_id,
            adapter_session_id=f"fixture-session:{session_id}",
            seed=seed,
            token_ids=list(token_ids),
            phase=LiveSessionPhase.CREATED,
        )
        self._sessions[session_id] = session
        self._advance(100)
        return self._session_ref(session)

    def prefill_session(self, session_id: str, *, timeout_s: float) -> RuntimeSessionRef:
        self._active_call(timeout_s)
        session = self._session(session_id)
        if session.phase is not LiveSessionPhase.CREATED:
            raise RuntimeFixtureError("prefill requires a newly created session")
        required_blocks = (len(session.token_ids) + self._block_size_tokens - 1) // (
            self._block_size_tokens
        )
        self._require_free_blocks(required_blocks)
        for token_start in range(0, len(session.token_ids), self._block_size_tokens):
            token_end = min(token_start + self._block_size_tokens, len(session.token_ids))
            block = self._allocate_block(
                token_start=token_start,
                token_end=token_end,
                owner_session_id=session.session_id,
                state_class=KvStateClass.RUNTIME_RESERVED,
                cache_resident=False,
            )
            session.prefix_block_ids.append(block.runtime_block_id)
            self._record_state(StateOperationType.STATE_ALLOC, session, block, duration_ns=500)
        session.phase = LiveSessionPhase.PREFILLED
        return self._session_ref(session)

    def pause_at_safe_decode_boundary(
        self, session_id: str, *, timeout_s: float
    ) -> RuntimeSessionRef:
        self._active_call(timeout_s)
        session = self._session(session_id)
        if session.phase not in {LiveSessionPhase.PREFILLED, LiveSessionPhase.READY}:
            raise RuntimeFixtureError("only prefilled or ready sessions can pause")
        session.phase = LiveSessionPhase.PAUSED
        self._advance(100)
        return self._session_ref(session)

    def inspect_logical_state(self, session_id: str, *, timeout_s: float) -> LogicalRuntimeState:
        self._active_call(timeout_s)
        session = self._session(session_id)
        token_ids = tuple(session.token_ids)
        ranges = tuple(
            (block.logical_token_start, block.logical_token_end_exclusive)
            for block in self._session_blocks(session)
        )
        self._advance(50)
        return LogicalRuntimeState(
            session_id=session.session_id,
            model=self._identity,
            token_ids=token_ids,
            token_history_sha256=token_history_sha256(token_ids),
            position_start=0,
            position_end_exclusive=len(token_ids),
            attention_layer_ids=self._layer_ids,
            logical_kv_token_ranges=ranges,
        )

    def inspect_physical_kv_layout(
        self, session_ids: tuple[str, ...], *, timeout_s: float
    ) -> PhysicalKvLayoutSnapshot:
        self._active_call(timeout_s)
        return self._snapshot(session_ids)

    def inspect_gpu_memory_state(self, *, timeout_s: float) -> GpuMemoryState:
        self._active_call(timeout_s)
        assigned = len(self._blocks) * self._block_bytes
        observed = self._advance(50)
        return GpuMemoryState(
            device=self._device,
            observed_at_monotonic_ns=observed,
            nvml_process_bytes=None,
            nvml_device_used_bytes=None,
            torch_allocated_bytes=None,
            torch_reserved_bytes=None,
            kv_pool_reserved_bytes=self._pool_reserved_bytes,
            kv_assigned_bytes=assigned,
            kv_unassigned_bytes=self._pool_reserved_bytes - assigned,
            sample_source="deterministic simulated-GPU fixture; not an NVML or CUDA sample",
        )

    def identify_shared_prefix_blocks(
        self, branch_session_ids: tuple[str, ...], *, timeout_s: float
    ) -> tuple[str, ...]:
        self._active_call(timeout_s)
        sessions = self._branch_sessions(branch_session_ids)
        root_id = self._one_root(sessions)
        root = self._roots[root_id]
        requested = {session.session_id for session in sessions}
        block_ids = tuple(
            block_id
            for block_id in root.block_ids
            if requested.issubset(self._blocks[block_id].owner_session_ids)
            and self._blocks[block_id].state_class is KvStateClass.SHARED_PREFIX
        )
        if block_ids != root.block_ids:
            raise RuntimeFixtureError("shared-prefix ownership no longer matches the live root")
        self._advance(50)
        return block_ids

    def create_shared_root_reference(
        self, session_id: str, *, prefix_token_count: int, timeout_s: float
    ) -> SharedRootReference:
        self._active_call(timeout_s)
        session = self._session(session_id)
        if session.phase not in {LiveSessionPhase.PREFILLED, LiveSessionPhase.PAUSED}:
            raise RuntimeFixtureError("shared-root publication requires a prefilled safe boundary")
        if prefix_token_count != len(session.token_ids):
            raise RuntimeFixtureError("the bounded fixture publishes only the complete prefix")
        if session.root_reference_id is not None:
            root = self._roots[session.root_reference_id]
            if root.prefix_token_count != prefix_token_count:
                raise RuntimeFixtureError("existing root prefix length differs from the request")
            return self._root_reference(root)
        if not session.prefix_block_ids:
            raise RuntimeFixtureError("cannot publish an empty physical prefix")
        root_id = f"fixture-root:{self._root_sequence:08d}"
        self._root_sequence += 1
        created_at = self._advance(100)
        root = _FixtureRoot(
            root_reference_id=root_id,
            source_session_id=session.session_id,
            prefix_token_count=prefix_token_count,
            block_ids=tuple(session.prefix_block_ids),
            physical_bytes=len(session.prefix_block_ids) * self._block_bytes,
            created_at_monotonic_ns=created_at,
            owner_session_ids={session.session_id},
        )
        self._roots[root_id] = root
        session.root_reference_id = root_id
        for block_id in root.block_ids:
            block = self._blocks[block_id]
            block.state_class = KvStateClass.SHARED_PREFIX
            block.cache_resident = True
            self._record_state(StateOperationType.STATE_PUBLISH, session, block, duration_ns=200)
        self._record_branch(
            BranchOperationType.STATE_PUBLISH,
            session,
            physical_state_id=root_id,
            physical_bytes=root.physical_bytes,
            shared_root=True,
            duration_ns=200,
            attributes={"root_reference_count": 1, "prefix_token_count": prefix_token_count},
        )
        return self._root_reference(root)

    def fork_same_policy_session(
        self,
        root_reference_id: str,
        branch_session_id: str,
        *,
        divergent_token_id: int,
        seed: int,
        timeout_s: float,
    ) -> RuntimeSessionRef:
        self._active_call(timeout_s)
        self._validate_seed(seed)
        if divergent_token_id < 0:
            raise ValueError("divergent_token_id must be non-negative")
        branch_session_id = self._identifier(branch_session_id, "branch_session_id")
        if branch_session_id in self._sessions or branch_session_id in self._destroyed_session_ids:
            raise RuntimeFixtureError(
                f"session {branch_session_id!r} already exists or was destroyed"
            )
        if len(self._sessions) >= self._capability_matrix.max_sessions:
            raise RuntimeFixtureError("bounded fixture session capacity is exhausted")
        root = self._root(root_reference_id)
        live_branches = sum(
            session.branch_id is not None and session.root_reference_id == root_reference_id
            for session in self._sessions.values()
        )
        if live_branches >= self._capability_matrix.max_fanout:
            raise RuntimeFixtureError("bounded fixture fanout is exhausted")
        self._require_free_blocks(1)
        source = self._sessions.get(root.source_session_id)
        if source is None:
            source = next(
                (
                    session
                    for session in self._sessions.values()
                    if session.root_reference_id == root_reference_id
                ),
                None,
            )
        if source is None:
            raise RuntimeFixtureError("shared root has no live logical source")
        prefix_tokens = source.token_ids[: root.prefix_token_count]
        branch = _FixtureSession(
            session_id=branch_session_id,
            adapter_session_id=f"fixture-session:{branch_session_id}",
            seed=seed,
            token_ids=list(prefix_tokens),
            phase=LiveSessionPhase.READY,
            branch_id=branch_session_id,
            parent_session_id=root.source_session_id,
            root_reference_id=root_reference_id,
            prefix_block_ids=list(root.block_ids),
        )
        self._sessions[branch_session_id] = branch
        root.owner_session_ids.add(branch_session_id)
        for block_id in root.block_ids:
            block = self._blocks[block_id]
            block.owner_session_ids.add(branch_session_id)
            self._record_state(StateOperationType.STATE_FORK, branch, block, duration_ns=100)
        self._record_branch(
            BranchOperationType.BRANCH_FORK,
            branch,
            physical_state_id=root_reference_id,
            physical_bytes=root.physical_bytes,
            shared_root=True,
            duration_ns=250,
            fanout=live_branches + 1,
            attributes={
                "root_reference_count": len(root.owner_session_ids),
                "prefix_token_count": root.prefix_token_count,
            },
        )
        self._append_token(branch, divergent_token_id, cow=True)
        self._record_branch(
            BranchOperationType.BRANCH_DIVERGENCE,
            branch,
            physical_state_id=branch.private_block_ids[0],
            physical_bytes=self._block_bytes,
            private_suffix=True,
            duration_ns=400,
            fanout=live_branches + 1,
        )
        return self._session_ref(branch)

    def allocate_private_suffix_state(
        self, branch_session_id: str, *, minimum_tokens: int, timeout_s: float
    ) -> PhysicalKvLayoutSnapshot:
        self._active_call(timeout_s)
        if minimum_tokens < 1:
            raise ValueError("minimum_tokens must be positive")
        branch = self._branch_session(branch_session_id)
        root = self._roots[branch.root_reference_id or ""]
        current = len(branch.token_ids) - root.prefix_token_count
        desired_blocks = (minimum_tokens + self._block_size_tokens - 1) // self._block_size_tokens
        self._require_free_blocks(max(0, desired_blocks - len(branch.private_block_ids)))
        for suffix_index in range(current, minimum_tokens):
            token = self._deterministic_token(branch.seed, branch.session_id, suffix_index)
            self._append_token(branch, token, cow=not branch.private_block_ids)
        return self._snapshot((branch.session_id,))

    def resume_branch(self, branch_session_id: str, *, timeout_s: float) -> RuntimeSessionRef:
        self._active_call(timeout_s)
        branch = self._branch_session(branch_session_id)
        if branch.phase not in {LiveSessionPhase.READY, LiveSessionPhase.PAUSED}:
            raise RuntimeFixtureError("resume requires a ready or paused branch")
        branch.phase = LiveSessionPhase.DECODING
        self._advance(100)
        return self._session_ref(branch)

    def run_concurrent_branches(
        self,
        branch_session_ids: tuple[str, ...],
        *,
        maximum_new_tokens: int,
        seed: int,
        timeout_s: float,
    ) -> ConcurrentDecodeResult:
        self._active_call(timeout_s)
        self._validate_seed(seed)
        if maximum_new_tokens < 1:
            raise ValueError("maximum_new_tokens must be positive")
        branches = self._branch_sessions(branch_session_ids)
        root_id = self._one_root(branches)
        if any(branch.phase is not LiveSessionPhase.DECODING for branch in branches):
            raise RuntimeFixtureError("all concurrent branches must be resumed before decode")
        additional_blocks = 0
        for branch in branches:
            root = self._roots[branch.root_reference_id or ""]
            current_suffix = len(branch.token_ids) - root.prefix_token_count
            desired = current_suffix + maximum_new_tokens
            desired_blocks = (desired + self._block_size_tokens - 1) // self._block_size_tokens
            additional_blocks += max(0, desired_blocks - len(branch.private_block_ids))
        self._require_free_blocks(additional_blocks)
        start_ns = self._clock_ns
        outputs: dict[str, list[int]] = {branch.session_id: [] for branch in branches}
        snapshots = [self._snapshot(tuple(branch_session_ids))]
        checkpoints = {1, 16, 64, maximum_new_tokens}
        for output_index in range(maximum_new_tokens):
            for branch in branches:
                token = self._deterministic_token(
                    seed ^ branch.seed,
                    branch.session_id,
                    output_index + len(branch.token_ids),
                )
                outputs[branch.session_id].append(token)
                self._append_token(branch, token, cow=False)
            if output_index + 1 in checkpoints:
                snapshots.append(self._snapshot(tuple(branch_session_ids)))
        for branch in branches:
            branch.phase = LiveSessionPhase.READY
            self._record_branch(
                BranchOperationType.ROLLOUT,
                branch,
                physical_state_id=root_id,
                physical_bytes=sum(
                    self._blocks[block_id].bytes for block_id in branch.private_block_ids
                ),
                private_suffix=True,
                duration_ns=250,
                fanout=len(branches),
                attributes={"generated_tokens": maximum_new_tokens},
            )
        elapsed_ns = max(1, self._clock_ns - start_ns)
        total_tokens = len(branches) * maximum_new_tokens
        decode_active_tokens = len(branches) * max(0, maximum_new_tokens - 1)
        decode_active_elapsed_ns = max(
            1,
            elapsed_ns * max(1, maximum_new_tokens - 1) // maximum_new_tokens,
        )
        ready = {branch.session_id: (index + 1) * 1_000 for index, branch in enumerate(branches)}
        admitted = {branch_id: max(0, latency - 500) for branch_id, latency in ready.items()}
        decode_started = {branch_id: max(0, latency - 250) for branch_id, latency in ready.items()}
        first_token = {branch.session_id: ready[branch.session_id] + 5_000 for branch in branches}
        return ConcurrentDecodeResult(
            branchpoint_id=f"fixture-branchpoint:{root_id}",
            branch_output_token_ids={key: tuple(value) for key, value in outputs.items()},
            request_admitted_latency_ns=admitted,
            first_decode_token_started_latency_ns=decode_started,
            branch_ready_latency_ns=ready,
            first_token_latency_ns=first_token,
            elapsed_ns=elapsed_ns,
            total_output_tokens=total_tokens,
            throughput_tokens_per_second=total_tokens * 1_000_000_000.0 / elapsed_ns,
            decode_active_elapsed_ns=decode_active_elapsed_ns,
            decode_active_boundary_output_counts={branch.session_id: 1 for branch in branches},
            decode_active_tokens=decode_active_tokens,
            completed_branch_ids=tuple(branch.session_id for branch in branches),
            allocation_snapshots=tuple(snapshots),
        )

    def observe_block_allocation(
        self, session_ids: tuple[str, ...], *, timeout_s: float
    ) -> PhysicalKvLayoutSnapshot:
        return self.inspect_physical_kv_layout(session_ids, timeout_s=timeout_s)

    def observe_block_release(
        self, session_ids: tuple[str, ...], *, timeout_s: float
    ) -> PhysicalKvLayoutSnapshot:
        return self.inspect_physical_kv_layout(session_ids, timeout_s=timeout_s)

    def inspect_block_release_evidence(
        self, runtime_block_ids: tuple[str, ...], *, timeout_s: float
    ) -> PhysicalKvReleaseEvidence:
        self._validate_timeout(timeout_s)
        if not runtime_block_ids or len(runtime_block_ids) != len(set(runtime_block_ids)):
            raise ValueError("runtime_block_ids must be non-empty and unique")
        evidence: list[PhysicalKvBlockReleaseEvidence] = []
        for runtime_block_id in runtime_block_ids:
            block = self._blocks.get(runtime_block_id)
            if block is not None:
                evidence.append(
                    PhysicalKvBlockReleaseEvidence(
                        runtime_block_id=runtime_block_id,
                        block_index=block.block_index,
                        allocation_epoch=block.allocation_epoch,
                        native_refcount=len(block.owner_session_ids),
                        block_hash_present=block.cache_resident,
                        allocator_available=False,
                        is_null=False,
                    )
                )
                continue
            try:
                block_index, allocation_epoch = self._released_blocks[runtime_block_id]
            except KeyError as error:
                raise RuntimeFixtureError(
                    f"unknown formerly observed KV block {runtime_block_id!r}"
                ) from error
            evidence.append(
                PhysicalKvBlockReleaseEvidence(
                    runtime_block_id=runtime_block_id,
                    block_index=block_index,
                    allocation_epoch=allocation_epoch,
                    native_refcount=0,
                    block_hash_present=False,
                    allocator_available=True,
                    is_null=False,
                )
            )
        return PhysicalKvReleaseEvidence(
            runtime=FIXTURE_RUNTIME,
            runtime_version=FIXTURE_RUNTIME_VERSION,
            device=self._device,
            observed_at_monotonic_ns=self._advance(50),
            requested_block_ids=runtime_block_ids,
            blocks=tuple(evidence),
            pool_free_block_count=len(self._free_block_indices),
            pool_usable_block_count=self._pool_capacity,
        )

    def report_block_refcounts_or_equivalent(
        self, session_ids: tuple[str, ...], *, timeout_s: float
    ) -> dict[str, int]:
        snapshot = self.inspect_physical_kv_layout(session_ids, timeout_s=timeout_s)
        return {block.runtime_block_id: block.refcount for block in snapshot.blocks}

    def capture_branchpoint(
        self, root_reference_id: str, branch_session_ids: tuple[str, ...], *, timeout_s: float
    ) -> LiveBranchPoint:
        self._active_call(timeout_s)
        root = self._root(root_reference_id)
        branches = self._branch_sessions(branch_session_ids)
        if self._one_root(branches) != root_reference_id:
            raise RuntimeFixtureError("branchpoint branches do not belong to the requested root")
        branchpoint_id = f"fixture-branchpoint:{self._branchpoint_sequence:08d}"
        self._branchpoint_sequence += 1
        captured = self._advance(200)
        representative = branches[0]
        self._record_branch(
            BranchOperationType.BRANCH_POINT,
            representative,
            physical_state_id=root_reference_id,
            physical_bytes=root.physical_bytes,
            shared_root=True,
            duration_ns=200,
            fanout=len(branches),
            attributes={"root_reference_count": len(root.owner_session_ids)},
        )
        return LiveBranchPoint(
            branchpoint_id=branchpoint_id,
            root_reference=self._root_reference(root),
            scope=self._scope,
            model=self._identity,
            parent_session_id=root.source_session_id,
            branch_session_ids=tuple(branch.session_id for branch in branches),
            prefix_token_count=root.prefix_token_count,
            policy_epoch=self._identity.policy_epoch,
            captured_at_monotonic_ns=captured,
        )

    def emit_branch_workload_trace(self) -> tuple[BranchWorkloadEventV1, ...]:
        return tuple(self._branch_trace)

    def emit_state_operation_trace(self) -> tuple[StateOperationEventV1, ...]:
        return tuple(self._state_trace)

    def destroy_branch(self, branch_session_id: str, *, timeout_s: float) -> None:
        self._validate_timeout(timeout_s)
        branch_session_id = self._identifier(branch_session_id, "branch_session_id")
        if branch_session_id in self._destroyed_session_ids:
            return
        branch = self._branch_session(branch_session_id)
        self._destroy_session_state(branch, branch_operation=BranchOperationType.BRANCH_PRUNE)

    def destroy_session(self, session_id: str, *, timeout_s: float) -> None:
        self._validate_timeout(timeout_s)
        session_id = self._identifier(session_id, "session_id")
        if session_id in self._destroyed_session_ids:
            return
        session = self._session(session_id)
        operation = (
            BranchOperationType.BRANCH_PRUNE
            if session.branch_id is not None
            else BranchOperationType.BRANCH_COMPLETE
        )
        self._destroy_session_state(session, branch_operation=operation)

    def cleanup_runtime(self, *, timeout_s: float) -> None:
        self._validate_timeout(timeout_s)
        if self._cleaned:
            return
        for session_id in sorted(
            (session.session_id for session in self._sessions.values() if session.branch_id),
            reverse=True,
        ):
            self._destroy_session_state(
                self._sessions[session_id], branch_operation=BranchOperationType.BRANCH_ABORT
            )
        for session_id in sorted(tuple(self._sessions), reverse=True):
            self._destroy_session_state(
                self._sessions[session_id], branch_operation=BranchOperationType.BRANCH_COMPLETE
            )
        if self._blocks or self._roots or self._sessions:
            raise RuntimeFixtureError("fixture cleanup left live state")
        self._cleaned = True

    @property
    def _pool_reserved_bytes(self) -> int:
        return self._pool_capacity * self._block_bytes

    @staticmethod
    def _identifier(value: str, field_name: str) -> str:
        if not value or value != value.strip() or len(value) > 512:
            raise ValueError(f"{field_name} must be a bounded non-empty canonical string")
        return value

    @staticmethod
    def _validate_seed(seed: int) -> None:
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 1 << 64:
            raise ValueError("seed must be an unsigned 64-bit integer")

    @staticmethod
    def _validate_timeout(timeout_s: float) -> None:
        if isinstance(timeout_s, bool) or not isinstance(timeout_s, (int, float)):
            raise ValueError("timeout_s must be numeric")
        if not math.isfinite(float(timeout_s)) or not 0 < timeout_s <= MAX_TIMEOUT_S:
            raise ValueError(f"timeout_s must be finite and within (0, {MAX_TIMEOUT_S}]")

    def _active_call(self, timeout_s: float) -> None:
        self._validate_timeout(timeout_s)
        if self._cleaned:
            raise RuntimeFixtureError("fixture runtime has already been cleaned up")

    def _session(self, session_id: str) -> _FixtureSession:
        session_id = self._identifier(session_id, "session_id")
        try:
            return self._sessions[session_id]
        except KeyError as error:
            raise RuntimeFixtureError(f"unknown live session {session_id!r}") from error

    def _branch_session(self, session_id: str) -> _FixtureSession:
        session = self._session(session_id)
        if session.branch_id is None or session.root_reference_id is None:
            raise RuntimeFixtureError(f"session {session_id!r} is not a branch")
        return session

    def _branch_sessions(self, session_ids: tuple[str, ...]) -> tuple[_FixtureSession, ...]:
        if not session_ids:
            raise RuntimeFixtureError("at least one branch session is required")
        if len(session_ids) != len(set(session_ids)):
            raise RuntimeFixtureError("branch session identifiers must be unique")
        if len(session_ids) > self._capability_matrix.max_fanout:
            raise RuntimeFixtureError("branch fanout exceeds the fixture capability")
        return tuple(self._branch_session(session_id) for session_id in session_ids)

    def _root(self, root_reference_id: str) -> _FixtureRoot:
        root_reference_id = self._identifier(root_reference_id, "root_reference_id")
        try:
            return self._roots[root_reference_id]
        except KeyError as error:
            raise RuntimeFixtureError(f"unknown live shared root {root_reference_id!r}") from error

    @staticmethod
    def _one_root(sessions: tuple[_FixtureSession, ...]) -> str:
        root_ids = {session.root_reference_id for session in sessions}
        if len(root_ids) != 1 or None in root_ids:
            raise RuntimeFixtureError("all branch sessions must share exactly one live root")
        return next(iter(root_ids))  # type: ignore[return-value]

    def _session_ref(self, session: _FixtureSession) -> RuntimeSessionRef:
        return RuntimeSessionRef(
            session_id=session.session_id,
            adapter_session_id=session.adapter_session_id,
            phase=session.phase,
            seed=session.seed,
            branch_id=session.branch_id,
            parent_session_id=session.parent_session_id,
            root_reference_id=session.root_reference_id,
        )

    def _root_reference(self, root: _FixtureRoot) -> SharedRootReference:
        return SharedRootReference(
            root_reference_id=root.root_reference_id,
            source_session_id=root.source_session_id,
            prefix_token_count=root.prefix_token_count,
            block_ids=root.block_ids,
            physical_bytes=root.physical_bytes,
            adapter_owned_reference_count=len(root.owner_session_ids),
            created_at_monotonic_ns=root.created_at_monotonic_ns,
        )

    def _allocate_block(
        self,
        *,
        token_start: int,
        token_end: int,
        owner_session_id: str,
        state_class: KvStateClass,
        cache_resident: bool,
    ) -> _FixtureBlock:
        if not self._free_block_indices:
            raise RuntimeFixtureError("deterministic KV pool capacity is exhausted")
        block_index = self._free_block_indices.pop()
        self._allocation_epoch += 1
        block_id = f"fixture-kv:{block_index:06d}:epoch-{self._allocation_epoch:08d}"
        block = _FixtureBlock(
            runtime_block_id=block_id,
            block_index=block_index,
            kv_cache_group=0,
            layer_ids=self._layer_ids,
            device=self._device,
            dtype=self._dtype,
            bytes=self._block_bytes,
            block_size_tokens=self._block_size_tokens,
            logical_token_start=token_start,
            logical_token_end_exclusive=token_end,
            owner_session_ids={owner_session_id},
            state_class=state_class,
            allocation_epoch=self._allocation_epoch,
            cache_resident=cache_resident,
        )
        self._blocks[block_id] = block
        return block

    def _require_free_blocks(self, required_blocks: int) -> None:
        if required_blocks > len(self._free_block_indices):
            raise RuntimeFixtureError(
                "deterministic KV pool capacity is insufficient for the complete operation"
            )

    def _release_block(self, block: _FixtureBlock) -> None:
        self._released_blocks[block.runtime_block_id] = (
            block.block_index,
            block.allocation_epoch,
        )
        del self._blocks[block.runtime_block_id]
        self._free_block_indices.append(block.block_index)
        self._free_block_indices.sort(reverse=True)

    def _append_token(self, session: _FixtureSession, token: int, *, cow: bool) -> None:
        if token < 0:
            raise ValueError("token IDs must be non-negative")
        position = len(session.token_ids)
        session.token_ids.append(token)
        block: _FixtureBlock
        allocated = False
        if session.private_block_ids:
            candidate = self._blocks[session.private_block_ids[-1]]
            used = candidate.logical_token_end_exclusive - candidate.logical_token_start
            if used < candidate.block_size_tokens:
                candidate.logical_token_end_exclusive += 1
                block = candidate
            else:
                allocated = True
                block = self._allocate_block(
                    token_start=position,
                    token_end=position + 1,
                    owner_session_id=session.session_id,
                    state_class=KvStateClass.PRIVATE_SUFFIX,
                    cache_resident=False,
                )
                session.private_block_ids.append(block.runtime_block_id)
        else:
            allocated = True
            block = self._allocate_block(
                token_start=position,
                token_end=position + 1,
                owner_session_id=session.session_id,
                state_class=KvStateClass.PRIVATE_SUFFIX,
                cache_resident=False,
            )
            session.private_block_ids.append(block.runtime_block_id)
        if allocated:
            operation = StateOperationType.STATE_COW if cow else StateOperationType.STATE_ALLOC
            self._record_state(operation, session, block, duration_ns=400)
            self._record_branch(
                BranchOperationType(operation.value),
                session,
                physical_state_id=block.runtime_block_id,
                physical_bytes=block.bytes,
                private_suffix=True,
                cow_allocation=cow,
                duration_ns=400,
            )
        self._record_state(
            StateOperationType.STATE_APPEND,
            session,
            block,
            duration_ns=200,
            bytes_override=max(1, self._block_bytes // self._block_size_tokens),
        )

    def _session_blocks(self, session: _FixtureSession) -> tuple[_FixtureBlock, ...]:
        ids = session.prefix_block_ids + session.private_block_ids
        return tuple(self._blocks[block_id] for block_id in ids if block_id in self._blocks)

    def _snapshot(self, session_ids: tuple[str, ...]) -> PhysicalKvLayoutSnapshot:
        if len(session_ids) != len(set(session_ids)):
            raise RuntimeFixtureError("snapshot session identifiers must be unique")
        sessions = (
            tuple(self._session(session_id) for session_id in session_ids)
            if session_ids
            else tuple(self._sessions.values())
        )
        selected_ids = {session.session_id for session in sessions}
        selected_blocks = tuple(
            block
            for block in sorted(
                self._blocks.values(), key=lambda item: (item.block_index, item.allocation_epoch)
            )
            if not selected_ids or selected_ids & block.owner_session_ids
        )
        models = tuple(self._block_model(block) for block in selected_blocks)
        shared_ids = tuple(
            block.runtime_block_id
            for block in selected_blocks
            if block.state_class is KvStateClass.SHARED_PREFIX
        )
        private_ids = tuple(
            block.runtime_block_id
            for block in selected_blocks
            if block.state_class is KvStateClass.PRIVATE_SUFFIX
        )
        relevant_root_ids = {
            session.root_reference_id
            for session in sessions
            if session.root_reference_id is not None
        }
        root_refcount = sum(
            len(self._roots[root_id].owner_session_ids)
            for root_id in relevant_root_ids
            if root_id in self._roots
        )
        assigned = sum(block.bytes for block in selected_blocks)
        self._snapshot_epoch += 1
        return PhysicalKvLayoutSnapshot(
            runtime=FIXTURE_RUNTIME,
            runtime_version=FIXTURE_RUNTIME_VERSION,
            snapshot_epoch=self._snapshot_epoch,
            observed_at_monotonic_ns=self._advance(50),
            session_ids=tuple(session.session_id for session in sessions),
            blocks=models,
            shared_prefix_block_ids=shared_ids,
            private_suffix_block_ids=private_ids,
            root_reference_count=root_refcount,
            physical_assigned_bytes=assigned,
            shared_prefix_bytes=sum(
                block.bytes
                for block in selected_blocks
                if block.state_class is KvStateClass.SHARED_PREFIX
            ),
            private_suffix_bytes=sum(
                block.bytes
                for block in selected_blocks
                if block.state_class is KvStateClass.PRIVATE_SUFFIX
            ),
            kv_pool_reserved_bytes=self._pool_reserved_bytes,
        )

    def _block_model(self, block: _FixtureBlock) -> PhysicalKvBlock:
        return PhysicalKvBlock(
            runtime_block_id=block.runtime_block_id,
            block_index=block.block_index,
            kv_cache_group=block.kv_cache_group,
            layer_ids=block.layer_ids,
            device=block.device,
            dtype=block.dtype,
            bytes=block.bytes,
            block_size_tokens=block.block_size_tokens,
            logical_token_start=block.logical_token_start,
            logical_token_end_exclusive=block.logical_token_end_exclusive,
            branch_ids=tuple(sorted(block.owner_session_ids)),
            refcount=len(block.owner_session_ids),
            refcount_evidence=RefcountEvidence.RUNTIME_NATIVE,
            state_class=block.state_class,
            allocation_epoch=block.allocation_epoch,
            cache_resident=block.cache_resident,
        )

    def _destroy_session_state(
        self, session: _FixtureSession, *, branch_operation: BranchOperationType
    ) -> None:
        self._record_branch(
            branch_operation,
            session,
            physical_state_id=session.root_reference_id,
            physical_bytes=sum(block.bytes for block in self._session_blocks(session)),
            duration_ns=200,
        )
        for block_id in tuple(session.private_block_ids):
            block = self._blocks[block_id]
            block.owner_session_ids.discard(session.session_id)
            self._record_state(StateOperationType.STATE_FREE, session, block, duration_ns=300)
            self._release_block(block)
        root_id = session.root_reference_id
        if root_id is not None and root_id in self._roots:
            root = self._roots[root_id]
            root.owner_session_ids.discard(session.session_id)
            for block_id in root.block_ids:
                block = self._blocks[block_id]
                block.owner_session_ids.discard(session.session_id)
                operation = (
                    StateOperationType.STATE_FREE
                    if not block.owner_session_ids
                    else StateOperationType.STATE_RECLAIM
                )
                self._record_state(operation, session, block, duration_ns=150)
                if not block.owner_session_ids:
                    self._release_block(block)
            if not root.owner_session_ids:
                del self._roots[root_id]
        elif session.prefix_block_ids:
            for block_id in tuple(session.prefix_block_ids):
                if block_id not in self._blocks:
                    continue
                block = self._blocks[block_id]
                block.owner_session_ids.discard(session.session_id)
                self._record_state(StateOperationType.STATE_FREE, session, block, duration_ns=150)
                self._release_block(block)
        session.phase = LiveSessionPhase.DESTROYED
        del self._sessions[session.session_id]
        self._destroyed_session_ids.add(session.session_id)

    def _record_branch(
        self,
        operation: BranchOperationType,
        session: _FixtureSession,
        *,
        physical_state_id: str | None,
        physical_bytes: int,
        duration_ns: int,
        shared_root: bool = False,
        private_suffix: bool = False,
        cow_allocation: bool = False,
        fanout: int = 1,
        attributes: dict[str, bool | int | float | str | None] | None = None,
    ) -> None:
        if len(self._branch_trace) >= self._max_trace_events:
            raise RuntimeFixtureError("bounded branch trace capacity is exhausted")
        timestamp = self._advance(duration_ns)
        trace_attributes = self._trace_attributes(None)
        if attributes:
            trace_attributes.update(attributes)
        event = BranchWorkloadEventV1(
            provenance=WorkloadProvenance.SIMULATED_HARDWARE,
            timing_measurement_class=TimingMeasurementClass.SIMULATED_HARDWARE,
            trace_id=self._trace_id,
            session_id=session.session_id,
            branch_group_id=session.root_reference_id,
            branch_id=session.branch_id,
            parent_branch_id=session.parent_session_id,
            policy_epoch=self._identity.policy_epoch,
            host="deterministic-fixture-host",
            process_id=0,
            rank=0,
            device=self._device,
            monotonic_timestamp_ns=timestamp,
            normalized_timestamp_ns=timestamp - self._clock_origin_ns,
            duration_ns=duration_ns,
            clock_source=ClockSource.SYNTHETIC,
            alignment_confidence=1.0,
            operation_type=operation,
            logical_state_id=f"logical:{session.session_id}",
            physical_state_id=physical_state_id,
            state_segment=StateSegment.KV,
            physical_bytes=physical_bytes,
            location=MemoryLocation.GPU_HBM,
            source_location=MemoryLocation.GPU_HBM,
            destination_location=MemoryLocation.GPU_HBM,
            shared_root=shared_root,
            private_suffix=private_suffix,
            cow_allocation=cow_allocation,
            execution_latency_ns=duration_ns,
            cpu_time_ns=duration_ns // 4,
            gpu_duration_ns=duration_ns,
            fanout=fanout,
            attributes=trace_attributes,
        )
        sealed = seal_event(event, event_sequence=len(self._branch_trace))
        assert isinstance(sealed, BranchWorkloadEventV1)
        self._branch_trace.append(sealed)

    def _record_state(
        self,
        operation: StateOperationType,
        session: _FixtureSession,
        block: _FixtureBlock,
        *,
        duration_ns: int,
        bytes_override: int | None = None,
    ) -> None:
        if len(self._state_trace) >= self._max_trace_events:
            raise RuntimeFixtureError("bounded state trace capacity is exhausted")
        timestamp = self._advance(duration_ns)
        source = MemoryLocation.GPU_HBM
        destination = MemoryLocation.GPU_HBM
        if operation is StateOperationType.STATE_ALLOC:
            source = MemoryLocation.UNKNOWN
        elif operation is StateOperationType.STATE_FREE:
            destination = MemoryLocation.UNKNOWN
        representation = f"{FIXTURE_RUNTIME}/kv-block/{block.state_class.value}"
        event = StateOperationEventV1(
            provenance=WorkloadProvenance.SIMULATED_HARDWARE,
            timing_measurement_class=TimingMeasurementClass.SIMULATED_HARDWARE,
            trace_id=self._trace_id,
            session_id=session.session_id,
            branch_group_id=session.root_reference_id,
            logical_state_id=f"logical:{session.session_id}",
            branch_id=session.branch_id,
            tenant_id="fixture-tenant",
            security_domain="fixture-security-domain",
            host="deterministic-fixture-host",
            process_id=0,
            rank=0,
            device=self._device,
            monotonic_timestamp_ns=timestamp,
            normalized_timestamp_ns=timestamp - self._clock_origin_ns,
            duration_ns=duration_ns,
            clock_source=ClockSource.SYNTHETIC,
            alignment_confidence=1.0,
            operation_type=operation,
            state_segment=StateSegment.KV,
            source_physical_representation=representation,
            destination_physical_representation=representation,
            bytes=block.bytes if bytes_override is None else bytes_override,
            alignment_bytes=self._block_bytes,
            page_size_bytes=self._block_bytes,
            chunk_size_bytes=self._block_bytes,
            fanout=max(1, len(block.owner_session_ids)),
            operation_latency_ns=duration_ns,
            cpu_time_ns=duration_ns // 4,
            gpu_time_ns=duration_ns,
            result=OperationResult.SUCCESS,
            state_epoch=block.allocation_epoch,
            source_location=source,
            destination_location=destination,
            transport_type=TransportType.NONE,
            attributes=self._trace_attributes(block),
        )
        sealed = seal_event(event, event_sequence=len(self._state_trace))
        assert isinstance(sealed, StateOperationEventV1)
        self._state_trace.append(sealed)

    def _trace_attributes(
        self, block: _FixtureBlock | None
    ) -> dict[str, bool | int | float | str | None]:
        values: dict[str, bool | int | float | str | None] = {
            "runtime": FIXTURE_RUNTIME,
            "runtime_version": FIXTURE_RUNTIME_VERSION,
            "adapter_version": FIXTURE_ADAPTER_VERSION,
            "fixture_only": True,
            "simulated_gpu_state": True,
            "real_gpu_measurement": False,
        }
        if block is not None:
            values.update(
                {
                    "runtime_block_id": block.runtime_block_id,
                    "logical_token_start": block.logical_token_start,
                    "logical_token_end_exclusive": block.logical_token_end_exclusive,
                    "state_class": block.state_class.value,
                    "refcount": len(block.owner_session_ids),
                    "refcount_evidence": RefcountEvidence.RUNTIME_NATIVE.value,
                    "allocation_epoch": block.allocation_epoch,
                    "physical_block_bytes": block.bytes,
                }
            )
        return values

    def _advance(self, duration_ns: int) -> int:
        self._clock_ns += duration_ns
        return self._clock_ns

    @staticmethod
    def _deterministic_token(seed: int, branch_id: str, index: int) -> int:
        material = f"{seed}:{branch_id}:{index}".encode()
        return int.from_bytes(sha256(material).digest()[:8], "little") % 32_000


__all__ = [
    "FIXTURE_ADAPTER_VERSION",
    "FIXTURE_RUNTIME",
    "FIXTURE_RUNTIME_VERSION",
    "DeterministicRuntimeFixtureAdapter",
    "RuntimeFixtureError",
]
