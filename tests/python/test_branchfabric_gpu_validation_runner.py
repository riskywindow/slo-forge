from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest
from pydantic import ValidationError

from sloforge.continuum.adapters.real_runtime import (
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
    RefcountEvidence,
    RuntimeCapabilityMatrix,
    RuntimeModelIdentity,
    RuntimeScopeContract,
    RuntimeSessionRef,
    SharedRootReference,
    token_history_sha256,
)
from sloforge.continuum.adapters.sdk import SnapshotConsistencyError
from sloforge.helix.characterization.trace import TraceLevel

_RUNNER_PATH = (
    Path(__file__).parents[2] / "benchmarks" / "branchfabric" / "gpu_validation_runner.py"
)
_RUNNER_SPEC = importlib.util.spec_from_file_location(
    "sloforge_branchfabric_gpu_validation_runner", _RUNNER_PATH
)
assert _RUNNER_SPEC is not None and _RUNNER_SPEC.loader is not None
_RUNNER = importlib.util.module_from_spec(_RUNNER_SPEC)
sys.modules[_RUNNER_SPEC.name] = _RUNNER
_RUNNER_SPEC.loader.exec_module(_RUNNER)

CommandResult = _RUNNER.CommandResult
ExperimentProfile = _RUNNER.ExperimentProfile
ExperimentSpecification = _RUNNER.ExperimentSpecification
GpuSamplingConfiguration = _RUNNER.GpuSamplingConfiguration
NvidiaSmiSampler = _RUNNER.NvidiaSmiSampler
NvidiaSmiProcessMemoryReader = _RUNNER.NvidiaSmiProcessMemoryReader
TrialPath = _RUNNER.TrialPath
VllmEngineConfiguration = _RUNNER.VllmEngineConfiguration
VllmGpuValidationInvocation = _RUNNER.VllmGpuValidationInvocation
build_vllm_adapter_factory = _RUNNER.build_vllm_adapter_factory
run_experiment = _RUNNER.run_experiment

IDENTITY = RuntimeModelIdentity(
    runtime="vllm",
    runtime_version="0.23.0",
    adapter_version="1.0.0",
    model_id="Qwen/Qwen2.5-7B-Instruct",
    model_revision="a09a35458c702b33eeacc393d103063234e8bc28",
    tokenizer_id="Qwen/Qwen2.5-7B-Instruct",
    tokenizer_revision="a09a35458c702b33eeacc393d103063234e8bc28",
    dtype="torch.bfloat16",
    device="cuda:0",
    policy_epoch="policy-1",
)


def _capabilities(shared: bool) -> frozenset[RealRuntimeCapability]:
    values = {
        RealRuntimeCapability.START_SESSION,
        RealRuntimeCapability.PREFILL_SESSION,
        RealRuntimeCapability.PAUSE_AT_SAFE_DECODE_BOUNDARY,
        RealRuntimeCapability.INSPECT_LOGICAL_STATE,
        RealRuntimeCapability.INSPECT_PHYSICAL_KV_LAYOUT,
        RealRuntimeCapability.INSPECT_GPU_MEMORY_STATE,
        RealRuntimeCapability.OBSERVE_BLOCK_ALLOCATION,
        RealRuntimeCapability.OBSERVE_BLOCK_RELEASE,
        RealRuntimeCapability.REPORT_BLOCK_REFCOUNTS,
        RealRuntimeCapability.DESTROY_BRANCH,
        RealRuntimeCapability.DESTROY_SESSION,
        RealRuntimeCapability.CLEANUP_RUNTIME,
    }
    if shared:
        values.update(
            {
                RealRuntimeCapability.IDENTIFY_SHARED_PREFIX_BLOCKS,
                RealRuntimeCapability.FORK_SAME_POLICY_SESSION,
                RealRuntimeCapability.CREATE_SHARED_ROOT_REFERENCE,
                RealRuntimeCapability.ALLOCATE_PRIVATE_SUFFIX_STATE,
                RealRuntimeCapability.RESUME_BRANCH,
                RealRuntimeCapability.RUN_CONCURRENT_BRANCHES,
                RealRuntimeCapability.CAPTURE_BRANCHPOINT,
            }
        )
    else:
        values.add(RealRuntimeCapability.RUN_CONCURRENT_INDEPENDENT_SESSIONS)
    return frozenset(values)


def _matrix(shared: bool, fanout: int) -> RuntimeCapabilityMatrix:
    return RuntimeCapabilityMatrix(
        identity=IDENTITY,
        capabilities=_capabilities(shared),
        scope=RuntimeScopeContract(same_process_or_adapter_scope="deterministic fixture"),
        version_pin="vllm==0.23.0",
        internal_interfaces=("fixture",),
        max_sessions=fanout + 1,
        max_fanout=fanout,
    )


def _block(
    block_id: str,
    index: int,
    owners: tuple[str, ...],
    *,
    shared: bool,
    logical_start: int,
) -> PhysicalKvBlock:
    return PhysicalKvBlock(
        runtime_block_id=block_id,
        block_index=index,
        kv_cache_group=0,
        layer_ids=("model.layers.0.self_attn",),
        device="cuda:0",
        dtype="torch.bfloat16",
        bytes=100,
        block_size_tokens=1024,
        logical_token_start=logical_start,
        logical_token_end_exclusive=logical_start + 1024,
        branch_ids=owners,
        refcount=len(owners),
        refcount_evidence=RefcountEvidence.RUNTIME_NATIVE,
        state_class=KvStateClass.SHARED_PREFIX if shared else KvStateClass.PRIVATE_SUFFIX,
        allocation_epoch=1,
        cache_resident=True,
    )


def _snapshot(
    session_ids: tuple[str, ...],
    *,
    prefix_len: int,
    shared: bool,
    empty: bool = False,
) -> PhysicalKvLayoutSnapshot:
    if empty:
        blocks: tuple[PhysicalKvBlock, ...] = ()
        shared_ids: tuple[str, ...] = ()
        private_ids: tuple[str, ...] = ()
    elif shared:
        root_owners = session_ids
        root = (
            _block("root-0", 0, root_owners, shared=True, logical_start=0),
            _block("root-1", 1, root_owners, shared=True, logical_start=1024),
        )
        private = tuple(
            _block(
                f"private-{index}",
                2 + index,
                (session_id,),
                shared=False,
                logical_start=prefix_len,
            )
            for index, session_id in enumerate(session_ids)
        )
        blocks = root + private
        shared_ids = ("root-0", "root-1")
        private_ids = tuple(block.runtime_block_id for block in private)
    else:
        independent = []
        for owner_index, session_id in enumerate(session_ids):
            independent.extend(
                (
                    _block(
                        f"independent-{owner_index}-root-0",
                        owner_index * 3,
                        (session_id,),
                        shared=False,
                        logical_start=0,
                    ),
                    _block(
                        f"independent-{owner_index}-root-1",
                        owner_index * 3 + 1,
                        (session_id,),
                        shared=False,
                        logical_start=1024,
                    ),
                    _block(
                        f"independent-{owner_index}-suffix",
                        owner_index * 3 + 2,
                        (session_id,),
                        shared=False,
                        logical_start=prefix_len,
                    ),
                )
            )
        blocks = tuple(independent)
        shared_ids = ()
        private_ids = tuple(block.runtime_block_id for block in blocks)
    return PhysicalKvLayoutSnapshot(
        runtime="vllm",
        runtime_version="0.23.0",
        snapshot_epoch=1,
        observed_at_monotonic_ns=1,
        session_ids=session_ids,
        blocks=blocks,
        shared_prefix_block_ids=shared_ids,
        private_suffix_block_ids=private_ids,
        root_reference_count=len(session_ids) if shared and not empty else 0,
        physical_assigned_bytes=len(blocks) * 100,
        shared_prefix_bytes=len(shared_ids) * 100,
        private_suffix_bytes=len(private_ids) * 100,
        kv_pool_reserved_bytes=10_000,
    )


@dataclass
class _Session:
    tokens: tuple[int, ...]
    branch: bool = False


class _FixtureAdapter:
    def __init__(self, *, shared: bool, fanout: int, suffix: int) -> None:
        self.shared = shared
        self.suffix = suffix
        self.capability_matrix = _matrix(shared, fanout)
        self.sessions: dict[str, _Session] = {}
        self.root_id: str | None = None
        self.branch_ids: tuple[str, ...] = ()
        self.destroyed: set[str] = set()
        self.decode_started = False
        self.cleaned = False

    def _root(self) -> str:
        assert self.root_id is not None
        return self.root_id

    def start_session(self, session_id, *, token_ids, seed, timeout_s):
        self.sessions[session_id] = _Session(token_ids)
        return RuntimeSessionRef(
            session_id=session_id,
            adapter_session_id=session_id,
            phase=LiveSessionPhase.CREATED,
            seed=seed,
        )

    def inspect_logical_state(self, session_id, *, timeout_s):
        tokens = self.sessions[session_id].tokens
        return LogicalRuntimeState(
            session_id=session_id,
            model=IDENTITY,
            token_ids=tokens,
            token_history_sha256=token_history_sha256(tokens),
            position_start=0,
            position_end_exclusive=len(tokens),
            attention_layer_ids=("model.layers.0.self_attn",),
            logical_kv_token_ranges=((0, len(tokens)),),
        )

    def prefill_session(self, session_id, *, timeout_s):
        return RuntimeSessionRef(
            session_id=session_id,
            adapter_session_id=session_id,
            phase=LiveSessionPhase.PREFILLED,
            seed=41,
        )

    def pause_at_safe_decode_boundary(self, session_id, *, timeout_s):
        return RuntimeSessionRef(
            session_id=session_id,
            adapter_session_id=session_id,
            phase=LiveSessionPhase.PAUSED,
            seed=41,
        )

    def create_shared_root_reference(self, session_id, *, prefix_token_count, timeout_s):
        self.root_id = session_id
        return SharedRootReference(
            root_reference_id="fixture-root",
            source_session_id=session_id,
            prefix_token_count=prefix_token_count,
            block_ids=("root-0", "root-1"),
            physical_bytes=200,
            adapter_owned_reference_count=1,
            created_at_monotonic_ns=1,
        )

    def fork_same_policy_session(
        self, root_reference_id, branch_session_id, *, divergent_token_id, seed, timeout_s
    ):
        root_session_id = self._root()
        root_tokens = self.sessions[root_session_id].tokens
        self.sessions[branch_session_id] = _Session((*root_tokens, divergent_token_id), branch=True)
        self.branch_ids += (branch_session_id,)
        return RuntimeSessionRef(
            session_id=branch_session_id,
            adapter_session_id=branch_session_id,
            phase=LiveSessionPhase.CREATED,
            seed=seed,
            branch_id=branch_session_id,
            parent_session_id=root_session_id,
            root_reference_id="fixture-root",
        )

    def capture_branchpoint(self, root_reference_id, branch_session_ids, *, timeout_s):
        root = self.create_shared_root_reference(
            self._root(),
            prefix_token_count=len(self.sessions[self._root()].tokens),
            timeout_s=timeout_s,
        )
        return LiveBranchPoint(
            branchpoint_id="fixture-branchpoint",
            root_reference=root,
            scope=self.capability_matrix.scope,
            model=IDENTITY,
            parent_session_id=self._root(),
            branch_session_ids=branch_session_ids,
            prefix_token_count=len(self.sessions[self._root()].tokens),
            policy_epoch=IDENTITY.policy_epoch,
            captured_at_monotonic_ns=1,
        )

    def inspect_physical_kv_layout(self, session_ids, *, timeout_s):
        if self._root() in self.destroyed:
            return _snapshot(session_ids, prefix_len=2048, shared=False, empty=True)
        active = tuple(
            session_id
            for session_id in session_ids
            if session_id in self.sessions
            and self.sessions[session_id].branch
            and session_id not in self.destroyed
        )
        if active:
            if not self.decode_started:
                root_blocks = (
                    _block("root-0", 0, (), shared=True, logical_start=0),
                    _block("root-1", 1, (), shared=True, logical_start=1024),
                )
                return PhysicalKvLayoutSnapshot(
                    runtime="vllm",
                    runtime_version="0.23.0",
                    snapshot_epoch=1,
                    observed_at_monotonic_ns=1,
                    session_ids=active,
                    blocks=root_blocks,
                    shared_prefix_block_ids=("root-0", "root-1"),
                    private_suffix_block_ids=(),
                    root_reference_count=0,
                    physical_assigned_bytes=200,
                    shared_prefix_bytes=200,
                    private_suffix_bytes=0,
                    kv_pool_reserved_bytes=10_000,
                )
            return _snapshot(
                active, prefix_len=len(self.sessions[self._root()].tokens), shared=True
            )
        root_blocks = (
            _block("root-0", 0, (), shared=True, logical_start=0),
            _block("root-1", 1, (), shared=True, logical_start=1024),
        )
        return PhysicalKvLayoutSnapshot(
            runtime="vllm",
            runtime_version="0.23.0",
            snapshot_epoch=1,
            observed_at_monotonic_ns=1,
            session_ids=session_ids,
            blocks=root_blocks,
            shared_prefix_block_ids=("root-0", "root-1"),
            private_suffix_block_ids=(),
            root_reference_count=0,
            physical_assigned_bytes=200,
            shared_prefix_bytes=200,
            private_suffix_bytes=0,
            kv_pool_reserved_bytes=10_000,
        )

    def run_concurrent_branches(self, branch_session_ids, *, maximum_new_tokens, seed, timeout_s):
        self.decode_started = True
        snapshots = tuple(
            _snapshot(
                branch_session_ids,
                prefix_len=len(self.sessions[self._root()].tokens),
                shared=True,
            )
            for _ in (1, 16, 64, 256)
            if _ <= maximum_new_tokens
        )
        return ConcurrentDecodeResult(
            branchpoint_id="fixture-branchpoint",
            branch_output_token_ids={
                session_id: tuple(range(maximum_new_tokens)) for session_id in branch_session_ids
            },
            request_admitted_latency_ns={
                session_id: 50 + index for index, session_id in enumerate(branch_session_ids)
            },
            first_decode_token_started_latency_ns={
                session_id: 75 + index for index, session_id in enumerate(branch_session_ids)
            },
            branch_ready_latency_ns={
                session_id: 100 + index for index, session_id in enumerate(branch_session_ids)
            },
            first_token_latency_ns={
                session_id: 200 + index for index, session_id in enumerate(branch_session_ids)
            },
            elapsed_ns=1_000,
            total_output_tokens=maximum_new_tokens * len(branch_session_ids),
            throughput_tokens_per_second=1_000.0,
            decode_active_elapsed_ns=800,
            decode_active_boundary_output_counts={
                session_id: 1 for session_id in branch_session_ids
            },
            decode_active_tokens=(maximum_new_tokens - 1) * len(branch_session_ids),
            completed_branch_ids=branch_session_ids,
            allocation_snapshots=snapshots,
        )

    def run_concurrent_independent_sessions(
        self, session_ids, *, maximum_new_tokens, seed, timeout_s
    ):
        snapshots = tuple(
            _snapshot(
                session_ids,
                prefix_len=len(self.sessions[session_ids[0]].tokens) - 1,
                shared=False,
            )
            for _ in (1, 16, 64, 256)
            if _ <= maximum_new_tokens
        )
        return ConcurrentDecodeResult(
            branchpoint_id="independent-branchpoint",
            branch_output_token_ids={
                session_id: tuple(range(maximum_new_tokens)) for session_id in session_ids
            },
            request_admitted_latency_ns={
                session_id: 100 + index for index, session_id in enumerate(session_ids)
            },
            first_decode_token_started_latency_ns={
                session_id: 900 + index for index, session_id in enumerate(session_ids)
            },
            branch_ready_latency_ns={
                session_id: 1_000 + index for index, session_id in enumerate(session_ids)
            },
            first_token_latency_ns={
                session_id: 1_100 + index for index, session_id in enumerate(session_ids)
            },
            elapsed_ns=2_000,
            total_output_tokens=maximum_new_tokens * len(session_ids),
            throughput_tokens_per_second=900.0,
            decode_active_elapsed_ns=900,
            decode_active_boundary_output_counts={session_id: 1 for session_id in session_ids},
            decode_active_tokens=(maximum_new_tokens - 1) * len(session_ids),
            completed_branch_ids=session_ids,
            allocation_snapshots=snapshots,
        )

    def observe_block_release(self, session_ids, *, timeout_s):
        return self.inspect_physical_kv_layout(session_ids, timeout_s=timeout_s)

    def inspect_block_release_evidence(self, runtime_block_ids, *, timeout_s):
        root_destroyed = self._root() in self.destroyed
        blocks = []
        for runtime_block_id in runtime_block_ids:
            if runtime_block_id.startswith("private-"):
                index = int(runtime_block_id.removeprefix("private-"))
                owner = self.branch_ids[index]
                released = owner in self.destroyed
                block_index = 2 + index
            elif runtime_block_id.startswith("root-"):
                released = root_destroyed
                block_index = int(runtime_block_id.removeprefix("root-"))
            else:
                raise AssertionError(f"unknown fixture block {runtime_block_id}")
            blocks.append(
                PhysicalKvBlockReleaseEvidence(
                    runtime_block_id=runtime_block_id,
                    block_index=block_index,
                    allocation_epoch=1,
                    native_refcount=0 if released else 1,
                    block_hash_present=not released,
                    allocator_available=released,
                    is_null=False,
                )
            )
        return PhysicalKvReleaseEvidence(
            runtime="vllm",
            runtime_version="0.23.0",
            device="cuda:0",
            observed_at_monotonic_ns=1,
            requested_block_ids=runtime_block_ids,
            blocks=tuple(blocks),
            pool_free_block_count=100 if root_destroyed else len(self.destroyed),
            pool_usable_block_count=100,
        )

    def inspect_gpu_memory_state(self, *, timeout_s):
        return GpuMemoryState(
            device="cuda:0",
            observed_at_monotonic_ns=1,
            nvml_process_bytes=None,
            nvml_device_used_bytes=None,
            torch_allocated_bytes=1_000,
            torch_reserved_bytes=10_000,
            kv_pool_reserved_bytes=10_000,
            kv_assigned_bytes=0,
            kv_unassigned_bytes=10_000,
            sample_source="fixture",
        )

    def destroy_branch(self, branch_session_id, *, timeout_s):
        self.destroyed.add(branch_session_id)

    def destroy_session(self, session_id, *, timeout_s):
        self.destroyed.add(session_id)

    def emit_branch_workload_trace(self):
        return ()

    def emit_state_operation_trace(self):
        return ()

    def cleanup_runtime(self, *, timeout_s):
        self.cleaned = True


def _unavailable_command(argv, timeout_s):
    return CommandResult(returncode=127, stdout="", stderr="not installed")


def _smoke_spec() -> ExperimentSpecification:
    return ExperimentSpecification(
        profile=ExperimentProfile.SMOKE,
        seed=41,
        timeout_s=10.0,
        prefix_token_ids=tuple(range(2048)),
        divergent_token_ids=(3000, 3001),
        fanout=2,
        suffix_tokens=16,
        expected_identity=IDENTITY,
        trace_level=TraceLevel.DISABLED,
        gpu_sampling=GpuSamplingConfiguration(
            interval_s=0.05, command_timeout_s=0.05, maximum_samples=1
        ),
    )


def test_smoke_runner_writes_required_phases_and_cleanup(tmp_path) -> None:
    adapter = _FixtureAdapter(shared=True, fanout=2, suffix=16)
    summary = run_experiment(
        _smoke_spec(),
        lambda path: adapter,
        tmp_path / "smoke",
        command_runner=_unavailable_command,
    )
    assert summary.path is TrialPath.SHARED_ROOT
    assert adapter.cleaned
    assert summary.all_branches_ready_ns >= 101
    assert summary.first_request_admitted_ns < summary.first_branch_ready_ns
    assert summary.first_decode_token_started_ns < summary.first_branch_ready_ns
    assert summary.first_branch_ready_ns < min(summary.first_token_latency_ns.values())
    assert summary.decode_tokens == 30
    assert summary.decode_active_boundary_output_counts == {
        branch_id: 1 for branch_id in summary.branch_ready_latency_ns
    }
    assert summary.tracing_overhead_fraction_available is False
    assert summary.trace_extraction_wall_time_ns == 0
    assert summary.instrumentation_availability.copy_engine_activity_available is False
    assert summary.instrumentation_availability.hbm_bandwidth_available is False
    assert "deleting_branch_a_preserves_branch_b_and_shared_root" in summary.assertions
    assert (
        "branch_a_former_private_ids_native_refcount_zero_and_allocator_available"
        in summary.assertions
    )
    assert "fork_references_exact_published_root_ids_and_bytes" in summary.assertions
    assert "fork_allocates_zero_private_or_additional_assigned_kv_bytes" in summary.assertions
    assert "fork_logical_branches_validated_before_native_scheduler_ownership" in summary.assertions
    release_path = tmp_path / "smoke" / "shared_root" / "raw" / "block-release-evidence.jsonl"
    release_phases = [
        __import__("json").loads(line)["phase"] for line in release_path.read_text().splitlines()
    ]
    assert release_phases == [
        "pre_branch_a_destroy",
        "post_branch_a_destroy",
        "post_root_destroy",
    ]
    snapshot_path = tmp_path / "smoke" / "shared_root" / "raw" / "physical-snapshots.jsonl"
    phases = [
        __import__("json").loads(line)["phase"] for line in snapshot_path.read_text().splitlines()
    ]
    assert phases == [
        "root",
        "fork",
        "divergence",
        "suffix_16",
        "pre_branch_delete",
        "post_branch_delete",
        "post_root_delete",
    ]


def test_smoke_runner_fails_closed_when_private_block_remains_owned(tmp_path) -> None:
    class _LeakyAdapter(_FixtureAdapter):
        def inspect_block_release_evidence(self, runtime_block_ids, *, timeout_s):
            evidence = super().inspect_block_release_evidence(
                runtime_block_ids, timeout_s=timeout_s
            )
            if (
                self.branch_ids
                and self.branch_ids[0] in self.destroyed
                and self._root() not in self.destroyed
            ):
                blocks = tuple(
                    block.model_copy(update={"native_refcount": 1, "allocator_available": False})
                    for block in evidence.blocks
                )
                return evidence.model_copy(update={"blocks": blocks})
            return evidence

    adapter = _LeakyAdapter(shared=True, fanout=2, suffix=16)
    with pytest.raises(SnapshotConsistencyError, match="remains owned"):
        run_experiment(
            _smoke_spec(),
            lambda path: adapter,
            tmp_path / "leaky",
            command_runner=_unavailable_command,
        )
    assert adapter.cleaned


def test_smoke_runner_fails_closed_when_fork_allocates_private_kv(tmp_path) -> None:
    class _AllocatingForkAdapter(_FixtureAdapter):
        def inspect_physical_kv_layout(self, session_ids, *, timeout_s):
            snapshot = super().inspect_physical_kv_layout(session_ids, timeout_s=timeout_s)
            if self.branch_ids and not self.decode_started:
                return _snapshot(
                    self.branch_ids,
                    prefix_len=len(self.sessions[self._root()].tokens),
                    shared=True,
                )
            return snapshot

    adapter = _AllocatingForkAdapter(shared=True, fanout=2, suffix=16)
    with pytest.raises(SnapshotConsistencyError, match="allocated private KV state"):
        run_experiment(
            _smoke_spec(),
            lambda path: adapter,
            tmp_path / "fork-private-allocation",
            command_runner=_unavailable_command,
        )
    assert adapter.cleaned


def test_smoke_runner_fails_closed_when_fork_increases_assigned_bytes(tmp_path) -> None:
    class _AssignedForkAdapter(_FixtureAdapter):
        def inspect_physical_kv_layout(self, session_ids, *, timeout_s):
            snapshot = super().inspect_physical_kv_layout(session_ids, timeout_s=timeout_s)
            if self.branch_ids and not self.decode_started:
                extra = _block(
                    "unclassified-fork-allocation",
                    2,
                    (self.branch_ids[0],),
                    shared=False,
                    logical_start=len(self.sessions[self._root()].tokens),
                )
                return snapshot.model_copy(
                    update={
                        "blocks": (*snapshot.blocks, extra),
                        "physical_assigned_bytes": snapshot.physical_assigned_bytes + extra.bytes,
                    }
                )
            return snapshot

    adapter = _AssignedForkAdapter(shared=True, fanout=2, suffix=16)
    with pytest.raises(SnapshotConsistencyError, match="runtime-assigned KV bytes"):
        run_experiment(
            _smoke_spec(),
            lambda path: adapter,
            tmp_path / "fork-assigned-allocation",
            command_runner=_unavailable_command,
        )
    assert adapter.cleaned


def test_smoke_runner_fails_closed_when_fork_substitutes_root_ids(tmp_path) -> None:
    class _DuplicatingForkAdapter(_FixtureAdapter):
        def inspect_physical_kv_layout(self, session_ids, *, timeout_s):
            snapshot = super().inspect_physical_kv_layout(session_ids, timeout_s=timeout_s)
            if self.branch_ids and not self.decode_started:
                blocks = tuple(
                    block.model_copy(
                        update={"runtime_block_id": f"duplicate-{block.runtime_block_id}"}
                    )
                    for block in snapshot.blocks
                )
                return snapshot.model_copy(
                    update={
                        "blocks": blocks,
                        "shared_prefix_block_ids": tuple(
                            block.runtime_block_id for block in blocks
                        ),
                    }
                )
            return snapshot

    adapter = _DuplicatingForkAdapter(shared=True, fanout=2, suffix=16)
    with pytest.raises(SnapshotConsistencyError, match="exact published root block IDs"):
        run_experiment(
            _smoke_spec(),
            lambda path: adapter,
            tmp_path / "fork-duplicate-root",
            command_runner=_unavailable_command,
        )
    assert adapter.cleaned


def test_independent_trial_fails_closed_when_destroy_leaves_assigned_kv(tmp_path) -> None:
    class _LeakyIndependentAdapter(_FixtureAdapter):
        def inspect_gpu_memory_state(self, *, timeout_s):
            state = super().inspect_gpu_memory_state(timeout_s=timeout_s)
            if self.sessions and set(self.sessions).issubset(self.destroyed):
                return state.model_copy(
                    update={"kv_assigned_bytes": 100, "kv_unassigned_bytes": 9_900}
                )
            return state

    spec = ExperimentSpecification(
        profile=ExperimentProfile.CONTEXT_16K,
        seed=73,
        timeout_s=10.0,
        prefix_token_ids=tuple(range(16_384)),
        divergent_token_ids=(20_000,),
        fanout=1,
        suffix_tokens=256,
        expected_identity=IDENTITY,
        trace_level=TraceLevel.DISABLED,
        gpu_sampling=GpuSamplingConfiguration(
            interval_s=0.05, command_timeout_s=0.05, maximum_samples=1
        ),
    )
    adapter = _LeakyIndependentAdapter(shared=False, fanout=1, suffix=256)
    with pytest.raises(SnapshotConsistencyError, match="runtime-assigned KV bytes"):
        _RUNNER.run_independent_trial(
            adapter,
            spec,
            tmp_path / "independent-leak",
            command_runner=_unavailable_command,
        )
    assert adapter.cleaned


def test_active_decode_accounting_uses_each_exact_common_boundary_count() -> None:
    payload = {
        "branchpoint_id": "boundary-test",
        "branch_output_token_ids": {"fast": (1, 2, 3, 4, 5), "slow": (1, 2, 3)},
        "request_admitted_latency_ns": {"fast": 1, "slow": 2},
        "first_decode_token_started_latency_ns": {"fast": 3, "slow": 4},
        "branch_ready_latency_ns": {"fast": 5, "slow": 6},
        "first_token_latency_ns": {"fast": 7, "slow": 8},
        "elapsed_ns": 100,
        "total_output_tokens": 8,
        "throughput_tokens_per_second": 80_000_000.0,
        "decode_active_elapsed_ns": 50,
        "decode_active_boundary_output_counts": {"fast": 3, "slow": 1},
        "decode_active_tokens": 4,
        "completed_branch_ids": ("fast", "slow"),
        "allocation_snapshots": (),
    }
    result = ConcurrentDecodeResult.model_validate(payload)
    assert result.decode_active_tokens == (5 - 3) + (3 - 1)
    with pytest.raises(ValidationError, match="active decode token accounting mismatch"):
        ConcurrentDecodeResult.model_validate({**payload, "decode_active_tokens": 6})
    with pytest.raises(ValidationError, match="timing milestones are not causally ordered"):
        ConcurrentDecodeResult.model_validate(
            {
                **payload,
                "first_decode_token_started_latency_ns": {"fast": 9, "slow": 4},
            }
        )


def test_trace_drain_cost_cannot_be_reported_as_runtime_overhead(tmp_path) -> None:
    adapter = _FixtureAdapter(shared=True, fanout=2, suffix=16)
    metrics, artifacts = _RUNNER._capture_traces(adapter, TraceLevel.FULL, tmp_path)
    assert metrics.extraction_wall_time_ns >= 0
    assert metrics.extraction_cpu_time_ns >= 0
    assert metrics.live_self_measurement_available is False
    assert metrics.overhead_fraction_available is False
    assert metrics.overhead_fraction_unavailable_reason.startswith("requires paired")
    assert len(artifacts) == 2


def test_16k_pair_runs_independent_before_shared_and_checks_isolation(tmp_path) -> None:
    spec = ExperimentSpecification(
        profile=ExperimentProfile.CONTEXT_16K,
        seed=73,
        timeout_s=10.0,
        prefix_token_ids=tuple(range(16_384)),
        divergent_token_ids=(20_000,),
        fanout=1,
        suffix_tokens=256,
        expected_identity=IDENTITY,
        trace_level=TraceLevel.DISABLED,
        gpu_sampling=GpuSamplingConfiguration(
            interval_s=0.05, command_timeout_s=0.05, maximum_samples=1
        ),
    )
    created = []

    def factory(path):
        if created:
            assert created[-1].cleaned
        adapter = _FixtureAdapter(shared=path is TrialPath.SHARED_ROOT, fanout=1, suffix=256)
        created.append(adapter)
        return adapter

    summary = run_experiment(spec, factory, tmp_path / "full", command_runner=_unavailable_command)
    assert summary.prefill_work_avoided_tokens == 0
    assert summary.independent.path is TrialPath.INDEPENDENT_PREFILL
    assert summary.shared_root.path is TrialPath.SHARED_ROOT
    assert "independent_post_destroy_kv_assigned_bytes_zero" in summary.independent.assertions
    independent_metadata = next(
        phase
        for phase in summary.independent.phase_metrics
        if phase["phase"] == "independent_session_metadata"
    )
    shared_metadata = next(
        phase for phase in summary.shared_root.phase_metrics if phase["phase"] == "fork_metadata"
    )
    assert summary.independent.first_request_admitted_ns == (
        independent_metadata["wall_duration_ns"] + 100
    )
    assert summary.shared_root.first_request_admitted_ns == (
        shared_metadata["wall_duration_ns"] + 50
    )
    assert all(adapter.cleaned for adapter in created)
    rows = (
        (tmp_path / "full" / "independent_prefill" / "raw" / "physical-snapshots.jsonl")
        .read_text()
        .splitlines()
    )
    assert [__import__("json").loads(row)["phase"] for row in rows] == [
        "divergence",
        "suffix_16",
        "suffix_64",
        "suffix_256",
    ]


def test_nvidia_smi_unavailable_row_has_explicit_null_fields() -> None:
    sampler = NvidiaSmiSampler(
        device="cuda:0",
        config=GpuSamplingConfiguration(maximum_samples=1),
        command_runner=_unavailable_command,
        clock_ns=lambda: 123,
    )
    row = sampler.sample_once()
    assert row["available"] is False
    assert row["unavailable_reason"] == "nvidia-smi exited 127"
    for field in (
        "memory_used_bytes",
        "memory_total_bytes",
        "gpu_utilization_percent",
        "memory_utilization_percent",
        "power_watts",
    ):
        assert field in row and row[field] is None
    assert row["raw_stderr"] == "not installed"


def test_nvidia_smi_uses_explicit_physical_selector() -> None:
    sampler = NvidiaSmiSampler(
        device="cuda:0",
        config=GpuSamplingConfiguration(maximum_samples=1, physical_device_selector="GPU-12345678"),
        command_runner=_unavailable_command,
    )
    row = sampler.sample_once()
    assert row["command"][-1] == "GPU-12345678"


def test_phase_memory_reader_reports_current_process_and_device_bytes() -> None:
    commands = []

    def command_runner(argv, timeout_s):
        commands.append((tuple(argv), timeout_s))
        if argv[1] == "--query-gpu=memory.used":
            return CommandResult(returncode=0, stdout="4096\n", stderr="")
        return CommandResult(returncode=0, stdout="42, 3072\n99, 1024\n", stderr="")

    reader = NvidiaSmiProcessMemoryReader(
        physical_device_selector="1",
        command_timeout_s=0.5,
        command_runner=command_runner,
        process_id=42,
    )
    assert reader("cuda:0") == {
        "nvml_process_bytes": 3072 * 1024 * 1024,
        "nvml_device_used_bytes": 4096 * 1024 * 1024,
    }
    assert all(command[-1] == "1" for command, _ in commands)


def test_phase_memory_reader_does_not_treat_pid_namespace_miss_as_zero() -> None:
    def command_runner(argv, timeout_s):
        del timeout_s
        if argv[1] == "--query-gpu=memory.used":
            return CommandResult(returncode=0, stdout="4096\n", stderr="")
        return CommandResult(returncode=0, stdout="99, 1024\n", stderr="")

    reader = NvidiaSmiProcessMemoryReader(
        physical_device_selector="0",
        command_timeout_s=0.5,
        command_runner=command_runner,
        process_id=42,
    )
    assert reader("cuda:0") == {
        "nvml_process_bytes": None,
        "nvml_device_used_bytes": 4096 * 1024 * 1024,
    }


def test_matrix_validation_rejects_invalid_smoke_and_16k_shapes() -> None:
    payload = _smoke_spec().model_dump()
    payload["fanout"] = 1
    payload["divergent_token_ids"] = (3000,)
    with pytest.raises(ValidationError, match="smoke requires fanout 2"):
        ExperimentSpecification.model_validate(payload)

    payload = _smoke_spec().model_dump()
    payload["profile"] = ExperimentProfile.CONTEXT_16K
    with pytest.raises(ValidationError, match="exactly 16384"):
        ExperimentSpecification.model_validate(payload)

    payload = _smoke_spec().model_dump()
    payload["suffix_tokens"] = 8
    with pytest.raises(ValidationError, match="16-token suffix"):
        ExperimentSpecification.model_validate(payload)


def test_identity_gate_still_runs_adapter_cleanup(tmp_path) -> None:
    adapter = _FixtureAdapter(shared=True, fanout=2, suffix=16)
    adapter.capability_matrix = adapter.capability_matrix.model_copy(
        update={"identity": IDENTITY.model_copy(update={"model_revision": "wrong-revision"})}
    )
    with pytest.raises(SnapshotConsistencyError, match="identity does not match"):
        run_experiment(
            _smoke_spec(),
            lambda path: adapter,
            tmp_path / "identity-failure",
            command_runner=_unavailable_command,
        )
    assert adapter.cleaned


def test_launcher_configuration_is_validated_and_manifested(tmp_path) -> None:
    base_spec = _smoke_spec()
    spec = base_spec.model_copy(
        update={
            "gpu_sampling": base_spec.gpu_sampling.model_copy(
                update={"physical_device_selector": "1"}
            )
        }
    )
    invocation = VllmGpuValidationInvocation(
        experiment=spec,
        engine=VllmEngineConfiguration(
            execution_model_path=(
                "/isolated/model/snapshots/a09a35458c702b33eeacc393d103063234e8bc28"
            ),
            max_model_len=4096,
            max_num_seqs=2,
            gpu_memory_utilization=0.8,
            download_dir="/isolated/model",
        ),
    )
    adapter = _FixtureAdapter(shared=True, fanout=2, suffix=16)
    run_experiment(
        spec,
        lambda path: adapter,
        tmp_path / "launcher",
        command_runner=_unavailable_command,
        launcher_configuration=invocation,
    )
    manifest = __import__("json").loads(
        (tmp_path / "launcher" / "artifact-manifest.json").read_text()
    )
    assert [item["relative_path"] for item in manifest["artifacts"]] == [
        "experiment-configuration.json",
        "launcher-configuration.json",
        "shared_root/artifact-manifest.json",
    ]

    full_spec = spec.model_dump(mode="python")
    full_spec.update(
        {
            "profile": ExperimentProfile.CONTEXT_16K,
            "prefix_token_ids": tuple(range(16_384)),
            "divergent_token_ids": (20_000,),
            "fanout": 1,
            "suffix_tokens": 256,
        }
    )
    with pytest.raises(ValidationError, match="max_model_len must cover"):
        VllmGpuValidationInvocation.model_validate(
            {
                "experiment": full_spec,
                "engine": invocation.engine.model_dump(mode="python"),
            }
        )


def test_vllm_factory_pins_local_snapshot_and_isolates_cache_paths(tmp_path, monkeypatch) -> None:
    from sloforge.continuum.adapters.vllm_live import VllmLiveStateAdapter

    base_spec = _smoke_spec()
    spec = base_spec.model_copy(
        update={
            "gpu_sampling": base_spec.gpu_sampling.model_copy(
                update={"physical_device_selector": "1"}
            )
        }
    )
    snapshot = tmp_path / IDENTITY.model_revision
    snapshot.mkdir()
    invocation = VllmGpuValidationInvocation(
        experiment=spec,
        engine=VllmEngineConfiguration(
            execution_model_path=str(snapshot),
            download_dir=str(tmp_path / "downloads"),
            max_model_len=4096,
            max_num_seqs=2,
            gpu_memory_utilization=0.8,
        ),
    )
    calls = []

    def fake_create(**kwargs):
        calls.append(kwargs)
        return _FixtureAdapter(
            shared=kwargs["llm_kwargs"]["enable_prefix_caching"],
            fanout=2,
            suffix=16,
        )

    monkeypatch.setattr(VllmLiveStateAdapter, "create", staticmethod(fake_create))
    factory = build_vllm_adapter_factory(invocation, command_runner=_unavailable_command)
    factory(TrialPath.INDEPENDENT_PREFILL)
    factory(TrialPath.SHARED_ROOT)
    assert [call["llm_kwargs"]["enable_prefix_caching"] for call in calls] == [
        False,
        True,
    ]
    assert all(call["execution_model_path"] == str(snapshot.resolve()) for call in calls)
    assert all(call["trace_level"] is TraceLevel.DISABLED for call in calls)
    assert all(call["llm_kwargs"]["tensor_parallel_size"] == 1 for call in calls)
    assert all(call["llm_kwargs"]["async_scheduling"] is False for call in calls)
    assert all("device" not in call["llm_kwargs"] for call in calls)
    assert all(call["llm_kwargs"]["cpu_offload_gb"] == 0.0 for call in calls)
    assert all(call["llm_kwargs"]["model"] == str(snapshot.resolve()) for call in calls)
