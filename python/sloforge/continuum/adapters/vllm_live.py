"""Fail-closed live GPU KV-state adapter for vLLM 0.23.0.

This adapter is deliberately narrower than the portable Continuum bindings.  It
keeps vLLM objects process-local and exposes pointer-free observations for one
V1, synchronous, in-process, single-GPU, full-attention engine.  Shared-root
branching is supported only at a full KV-block boundary: vLLM reuses immutable
prefix blocks and allocates append-only private suffix blocks.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import math
import os
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
from typing import TYPE_CHECKING, Final, Protocol, cast

if TYPE_CHECKING:
    from sloforge.helix.characterization.trace import BranchWorkloadEventV1, StateOperationEventV1

from sloforge.continuum.adapters.external import UnsupportedRuntimeVersionError
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
    RealRuntimeStateAdapter,
    RefcountEvidence,
    RuntimeCapabilityMatrix,
    RuntimeModelIdentity,
    RuntimeScopeContract,
    RuntimeSessionRef,
    SharedRootReference,
    token_history_sha256,
)
from sloforge.continuum.adapters.sdk import (
    AdapterUnavailableError,
    ResourceLimitError,
    SessionNotFoundError,
    SessionStateError,
    SnapshotConsistencyError,
    UnsupportedCapabilityError,
)
from sloforge.continuum.adapters.vllm_live_trace import VllmLiveTraceRecorder
from sloforge.continuum.adapters.vllm_metadata_0230 import (
    MetadataInstrumentationLevel,
    MetadataOptimization,
    RequestScope0230,
    VllmMetadataRecorder0230,
)
from sloforge.helix.characterization.trace import BranchOperationType, TraceLevel

VLLM_LIVE_ADAPTER_VERSION: Final = "1.0.0"
VLLM_LIVE_RUNTIME_VERSION: Final = "0.23.0"
VLLM_LIVE_INTERNAL_INTERFACES: Final = (
    "vllm.v1.engine.core_client.InprocClient.engine_core",
    "vllm.v1.core.sched.scheduler.Scheduler.requests",
    "vllm.v1.core.sched.scheduler.Scheduler.kv_cache_manager",
    "vllm.v1.core.kv_cache_manager.KVCacheManager.get_blocks",
    "vllm.v1.core.kv_cache_manager.KVCacheManager.allocate_slots",
    "vllm.v1.core.kv_cache_manager.KVCacheManager.free",
    "vllm.v1.core.block_pool.BlockPool.blocks",
    "vllm.v1.core.block_pool.BlockPool.get_num_free_blocks",
    "vllm.v1.core.kv_cache_utils.KVCacheBlock.block_id",
    "vllm.v1.core.kv_cache_utils.KVCacheBlock.ref_cnt",
    "vllm.v1.kv_cache_interface.KVCacheConfig.num_blocks",
    "vllm.v1.kv_cache_interface.KVCacheConfig.kv_cache_groups",
    "vllm.v1.kv_cache_interface.KVCacheConfig.kv_cache_tensors",
    "vllm.v1.kv_cache_interface.KVCacheTensor.size",
    "vllm.v1.kv_cache_interface.KVCacheSpec.page_size_bytes",
    "vllm.v1.worker.gpu_model_runner.GPUModelRunner.kv_caches",
)
_MAX_TIMEOUT_S: Final = 4 * 60 * 60
_MAX_SEED: Final = 1 << 64


class VllmLiveConfigurationError(AdapterUnavailableError):
    """The live engine does not satisfy the adapter's exact runtime contract."""

    code = "vllm_live_configuration_unsupported"


class _BlockLike(Protocol):
    block_id: int
    ref_cnt: int
    block_hash: object | None
    is_null: bool


class _BlockCollectionLike(Protocol):
    blocks: tuple[Sequence[_BlockLike], ...]


class _KvManagerLike(Protocol):
    block_pool: object
    kv_cache_config: object
    allocate_slots: Callable[..., object]
    free: Callable[[object], None]

    def get_blocks(self, request_id: str) -> _BlockCollectionLike: ...

    def reset_prefix_cache(self) -> bool: ...


class _LlmEngineLike(Protocol):
    def add_request(self, request_id: str, prompt: object, params: object) -> str: ...

    def step(self) -> object: ...

    def abort_request(self, request_ids: list[str]) -> None: ...


class _GpuMemoryReader(Protocol):
    def __call__(self, device: str) -> Mapping[str, int | None]: ...


@dataclass(slots=True)
class _LiveSession:
    session_id: str
    token_ids: tuple[int, ...]
    seed: int
    phase: LiveSessionPhase = LiveSessionPhase.CREATED
    runtime_request_id: str | None = None
    parent_session_id: str | None = None
    root_reference_id: str | None = None
    branch_id: str | None = None
    output_token_ids: tuple[int, ...] = ()
    submitted: bool = False
    target_new_tokens: int | None = None


@dataclass(slots=True)
class _ObserverState:
    allocation_epoch: int = 0
    snapshot_epoch: int = 0
    last_request_blocks: dict[str, tuple[tuple[int, ...], ...]] = field(default_factory=dict)
    request_admitted_ns: dict[str, int] = field(default_factory=dict)
    released_request_blocks: dict[str, tuple[tuple[int, ...], ...]] = field(default_factory=dict)
    initial_cached_tokens: dict[str, int] = field(default_factory=dict)
    initial_cached_block_ids: dict[str, tuple[tuple[int, ...], ...]] = field(default_factory=dict)
    block_epochs: dict[int, int] = field(default_factory=dict)


def _require_timeout(timeout_s: float) -> int:
    if not math.isfinite(timeout_s) or not 0 < timeout_s <= _MAX_TIMEOUT_S:
        raise ValueError(f"timeout_s must be finite and in (0, {_MAX_TIMEOUT_S}]")
    return time.monotonic_ns() + int(timeout_s * 1_000_000_000)


def _check_deadline(deadline_ns: int, operation: str) -> None:
    if time.monotonic_ns() > deadline_ns:
        raise TimeoutError(f"vLLM live adapter timed out during {operation}")


def _require_seed(seed: int) -> None:
    if not 0 <= seed < _MAX_SEED:
        raise ValueError("seed must be in [0, 2**64)")


def _runtime_block_id(group_idx: int, block_idx: int) -> str:
    return f"vllm:kv-group:{group_idx}:block:{block_idx}"


def _parse_runtime_block_id(runtime_block_id: str) -> tuple[int, int]:
    parts = runtime_block_id.split(":")
    if len(parts) != 5 or parts[:2] != ["vllm", "kv-group"] or parts[3] != "block":
        raise SnapshotConsistencyError(
            f"malformed vLLM physical block identifier {runtime_block_id!r}",
            operation="inspect_block_release_evidence",
        )
    try:
        group_idx, block_idx = int(parts[2]), int(parts[4])
    except ValueError as error:
        raise SnapshotConsistencyError(
            f"malformed vLLM physical block identifier {runtime_block_id!r}",
            operation="inspect_block_release_evidence",
        ) from error
    if group_idx != 0 or block_idx < 0:
        raise SnapshotConsistencyError(
            "release evidence only supports the validated single KV cache group",
            operation="inspect_block_release_evidence",
        )
    return group_idx, block_idx


def _request_id(request: object) -> str:
    value = getattr(request, "request_id", None)
    if not isinstance(value, str) or not value:
        raise SnapshotConsistencyError(
            "vLLM request lacks a bounded string request_id",
            operation="observe_request",
        )
    return value


class VllmLiveInternalView:
    """Version-scoped structural view over the exact vLLM internals used here."""

    def __init__(self, engine: object, *, runtime_version: str) -> None:
        if runtime_version != VLLM_LIVE_RUNTIME_VERSION:
            raise UnsupportedRuntimeVersionError(
                (
                    f"live vLLM state adapter requires exactly "
                    f"{VLLM_LIVE_RUNTIME_VERSION}, got {runtime_version!r}"
                ),
                operation="live_gpu_state",
                runtime_name="vllm",
                runtime_version=runtime_version,
            )
        self.runtime_version = runtime_version
        self.engine = engine
        self.llm_engine = cast(_LlmEngineLike, self._require_attr(engine, "llm_engine"))
        self.engine_core_client = self._require_attr(self.llm_engine, "engine_core")
        core_module = type(self.engine_core_client).__module__
        core_name = type(self.engine_core_client).__name__
        if core_name != "InprocClient" or not core_module.startswith("vllm.v1."):
            raise self._configuration_error(
                "V1 InprocClient is required; multiprocessing/async clients are unsupported"
            )
        self.engine_core = self._require_attr(self.engine_core_client, "engine_core")
        self.scheduler = self._require_attr(self.engine_core, "scheduler")
        self.manager = cast(_KvManagerLike, self._require_attr(self.scheduler, "kv_cache_manager"))
        self.vllm_config = self._require_attr(self.llm_engine, "vllm_config")
        self._validate_config()
        self.group_spec, self.layer_specs = self._validate_kv_group()
        self.runner = self._resolve_gpu_runner()
        self.device = self._validate_gpu_backing()

    @staticmethod
    def _require_attr(owner: object, name: str) -> object:
        value = getattr(owner, name, None)
        if value is None:
            raise VllmLiveConfigurationError(
                f"required vLLM 0.23.0 internal {name!r} is absent",
                operation="live_gpu_state",
            )
        return value

    @staticmethod
    def _configuration_error(message: str) -> VllmLiveConfigurationError:
        return VllmLiveConfigurationError(message, operation="live_gpu_state")

    def _validate_config(self) -> None:
        parallel = self._require_attr(self.vllm_config, "parallel_config")
        for attr in (
            "tensor_parallel_size",
            "pipeline_parallel_size",
            "data_parallel_size",
        ):
            if getattr(parallel, attr, None) != 1:
                raise self._configuration_error(f"{attr}=1 is required")

        scheduler_config = self._require_attr(self.vllm_config, "scheduler_config")
        if bool(getattr(scheduler_config, "async_scheduling", False)):
            raise self._configuration_error("asynchronous scheduling is unsupported")
        if getattr(scheduler_config, "policy", "fcfs") != "fcfs":
            raise self._configuration_error("Experiment 003 requires FCFS scheduling")
        if getattr(self.vllm_config, "speculative_config", None) is not None:
            raise self._configuration_error("speculative decoding is unsupported")
        if getattr(self.vllm_config, "lora_config", None) is not None:
            raise self._configuration_error("LoRA state is outside the bounded adapter")
        if getattr(self.vllm_config, "kv_transfer_config", None) is not None:
            raise self._configuration_error("KV connectors are outside the bounded adapter")
        if getattr(self.vllm_config, "ec_transfer_config", None) is not None:
            raise self._configuration_error("EC connectors are outside the bounded adapter")

    def _validate_kv_group(self) -> tuple[object, tuple[tuple[str, object], ...]]:
        config = self._require_attr(self.manager, "kv_cache_config")
        groups = getattr(config, "kv_cache_groups", None)
        if not isinstance(groups, list) or len(groups) != 1:
            raise self._configuration_error("exactly one KV cache group is required")
        group = groups[0]
        spec = self._require_attr(group, "kv_cache_spec")
        nested_specs = getattr(spec, "kv_cache_specs", None)
        if isinstance(nested_specs, dict):
            layer_specs = tuple(sorted(nested_specs.items()))
        else:
            layer_names = getattr(group, "layer_names", None)
            if not isinstance(layer_names, list) or not layer_names:
                raise self._configuration_error("KV group layer names are missing")
            layer_specs = tuple((str(layer), spec) for layer in layer_names)
        if not layer_specs:
            raise self._configuration_error("KV group contains no attention layers")
        for _, layer_spec in layer_specs:
            if type(layer_spec).__name__ != "FullAttentionSpec":
                raise self._configuration_error(
                    "only ordinary FullAttentionSpec layers are supported"
                )
            if getattr(layer_spec, "sliding_window", None) is not None:
                raise self._configuration_error("sliding-window attention is unsupported")
            if getattr(layer_spec, "attention_chunk_size", None) is not None:
                raise self._configuration_error("chunked-local attention is unsupported")
        return spec, layer_specs

    def _resolve_gpu_runner(self) -> object:
        executor = self._require_attr(self.engine_core, "model_executor")
        driver_worker = self._require_attr(executor, "driver_worker")
        worker = getattr(driver_worker, "worker", driver_worker)
        return self._require_attr(worker, "model_runner")

    def _validate_gpu_backing(self) -> str:
        device_obj = self._require_attr(self.runner, "device")
        device = str(device_obj)
        device_type = getattr(device_obj, "type", None)
        if not (device.startswith("cuda") or device_type == "cuda"):
            raise self._configuration_error("model runner KV state is not CUDA-backed")
        kv_caches = getattr(self.runner, "kv_caches", None)
        if not isinstance(kv_caches, list) or not kv_caches:
            raise self._configuration_error("model runner exposes no live KV tensors")
        for tensor in kv_caches:
            tensor_device = getattr(tensor, "device", None)
            tensor_type = getattr(tensor_device, "type", None)
            if not (str(tensor_device).startswith("cuda") or tensor_type == "cuda"):
                raise self._configuration_error("a live KV tensor is not CUDA-backed")
        configured = self.kv_pool_reserved_bytes
        actual = sum(self._tensor_nbytes(tensor) for tensor in kv_caches)
        if actual != configured:
            raise self._configuration_error(
                "live CUDA KV tensor bytes do not match KVCacheConfig allocation bytes"
            )
        if configured != self.num_pool_blocks * self.page_size_bytes:
            raise self._configuration_error(
                "KV pool bytes do not equal physical block count times group page bytes"
            )
        return device

    @classmethod
    def _tensor_nbytes(cls, tensor: object) -> int:
        nbytes = getattr(tensor, "nbytes", None)
        if isinstance(nbytes, int) and nbytes > 0:
            return nbytes
        numel = getattr(tensor, "numel", None)
        element_size = getattr(tensor, "element_size", None)
        if callable(numel) and callable(element_size):
            count = numel()
            width = element_size()
            if isinstance(count, int) and isinstance(width, int) and count > 0 and width > 0:
                return count * width
        raise cls._configuration_error("a live KV tensor has no trustworthy byte size")

    @property
    def prefix_caching_enabled(self) -> bool:
        cache_config = self._require_attr(self.vllm_config, "cache_config")
        return bool(getattr(cache_config, "enable_prefix_caching", False))

    @property
    def block_size_tokens(self) -> int:
        value = getattr(self.group_spec, "block_size", None)
        if not isinstance(value, int) or value <= 0:
            raise self._configuration_error("invalid KV block size")
        return value

    @property
    def page_size_bytes(self) -> int:
        # A scheduler block ID selects the same slot in every physical KV
        # tensor represented by the one validated cache group.  The group's
        # FullAttentionSpec page size is only one layer's page size for common
        # uniform models (including Qwen2.5), so charging it directly would
        # undercount physical state by the number of attention layers.
        config = self._require_attr(self.manager, "kv_cache_config")
        configured_num_blocks = getattr(config, "num_blocks", None)
        if configured_num_blocks != self.num_pool_blocks:
            raise self._configuration_error(
                "KVCacheConfig block count does not match the scheduler block pool"
            )
        tensors = getattr(config, "kv_cache_tensors", None)
        if not isinstance(tensors, list) or not tensors:
            raise self._configuration_error("KV cache tensor allocation metadata is absent")
        tensor_sizes = [getattr(tensor, "size", None) for tensor in tensors]
        if not all(isinstance(size, int) and size > 0 for size in tensor_sizes):
            raise self._configuration_error("KV cache tensor sizes are invalid")
        num_blocks = self.num_pool_blocks
        if any(cast(int, size) % num_blocks for size in tensor_sizes):
            raise self._configuration_error(
                "KV cache tensor size is not an integral number of physical blocks"
            )
        physical_page_bytes = sum(cast(int, size) // num_blocks for size in tensor_sizes)
        layer_page_bytes = [getattr(spec, "page_size_bytes", None) for _, spec in self.layer_specs]
        if not all(isinstance(size, int) and size > 0 for size in layer_page_bytes):
            raise self._configuration_error("invalid per-layer KV page size")
        if physical_page_bytes != sum(cast(int, size) for size in layer_page_bytes):
            raise self._configuration_error(
                "one-group KV tensor geometry does not match its per-layer page sizes"
            )
        return physical_page_bytes

    @property
    def layer_ids(self) -> tuple[str, ...]:
        return tuple(name for name, _ in self.layer_specs)

    @property
    def dtype(self) -> str:
        dtypes = {str(getattr(spec, "dtype", "unknown")) for _, spec in self.layer_specs}
        if len(dtypes) != 1 or "unknown" in dtypes:
            raise self._configuration_error("KV layer dtypes are absent or non-uniform")
        return dtypes.pop()

    @property
    def kv_pool_reserved_bytes(self) -> int:
        config = self._require_attr(self.manager, "kv_cache_config")
        tensors = getattr(config, "kv_cache_tensors", None)
        if not isinstance(tensors, list) or not tensors:
            raise self._configuration_error("KV cache tensor allocation metadata is absent")
        sizes = [getattr(tensor, "size", None) for tensor in tensors]
        if not all(isinstance(size, int) and size > 0 for size in sizes):
            raise self._configuration_error("KV cache tensor sizes are invalid")
        return sum(cast(int, size) for size in sizes)

    @property
    def num_pool_blocks(self) -> int:
        pool = self._require_attr(self.manager, "block_pool")
        blocks = getattr(pool, "blocks", None)
        if not isinstance(blocks, list) or not blocks:
            raise self._configuration_error("vLLM block pool is absent")
        return len(blocks)

    def pool_block(self, block_id: int) -> _BlockLike:
        pool = self._require_attr(self.manager, "block_pool")
        blocks = getattr(pool, "blocks", None)
        if not isinstance(blocks, list) or not 0 <= block_id < len(blocks):
            raise SnapshotConsistencyError(
                f"vLLM returned invalid physical block id {block_id}",
                operation="inspect_physical_kv_layout",
            )
        return cast(_BlockLike, blocks[block_id])

    @property
    def pool_free_block_count(self) -> int:
        pool = self._require_attr(self.manager, "block_pool")
        reader = getattr(pool, "get_num_free_blocks", None)
        if not callable(reader):
            raise self._configuration_error("BlockPool.get_num_free_blocks is unavailable")
        value = reader()
        if not isinstance(value, int) or not 0 <= value <= self.num_pool_blocks - 1:
            raise self._configuration_error("BlockPool returned an invalid free-block count")
        return value

    @property
    def pool_usable_block_count(self) -> int:
        # vLLM reserves exactly one null block that never backs request KV state.
        return self.num_pool_blocks - 1

    def request_blocks(self, request_id: str) -> tuple[tuple[_BlockLike, ...], ...]:
        result = self.manager.get_blocks(request_id)
        groups = tuple(tuple(group) for group in result.blocks)
        if len(groups) != 1:
            raise SnapshotConsistencyError(
                "vLLM request block table no longer matches the one-group contract",
                operation="inspect_physical_kv_layout",
                session_id=request_id,
            )
        return groups

    def request_object(self, request_id: str) -> object | None:
        requests = getattr(self.scheduler, "requests", None)
        if not isinstance(requests, dict):
            raise self._configuration_error("scheduler request table is unavailable")
        return requests.get(request_id)


class VllmLiveStateAdapter(RealRuntimeStateAdapter):
    """Same-process shared-root adapter for one validated vLLM 0.23.0 engine."""

    def __init__(
        self,
        engine: object,
        *,
        identity: RuntimeModelIdentity,
        runtime_version: str | None = None,
        max_fanout: int = 32,
        default_maximum_new_tokens: int = 256,
        gpu_memory_reader: _GpuMemoryReader | None = None,
        trace_level: TraceLevel = TraceLevel.DISABLED,
        trace_id: str | None = None,
        max_trace_events: int = 262_144,
        metadata_instrumentation_level: MetadataInstrumentationLevel = (
            MetadataInstrumentationLevel.DISABLED
        ),
        metadata_optimization: MetadataOptimization = MetadataOptimization.BASELINE,
        clock_ns: Callable[[], int] = time.monotonic_ns,
    ) -> None:
        if identity.runtime != "vllm" or identity.runtime_version != VLLM_LIVE_RUNTIME_VERSION:
            raise ValueError("identity must name the exact vLLM 0.23.0 runtime")
        if identity.adapter_version != VLLM_LIVE_ADAPTER_VERSION:
            raise ValueError("identity adapter_version does not match this adapter")
        if not 1 <= max_fanout <= 64:
            raise ValueError("max_fanout must be in 1..64")
        if not 1 <= default_maximum_new_tokens <= 65_535:
            raise ValueError("default maximum continuation must be in 1..65535")
        discovered = runtime_version or importlib.metadata.version("vllm")
        self._view = VllmLiveInternalView(engine, runtime_version=discovered)
        if identity.device != self._view.device:
            raise VllmLiveConfigurationError(
                "runtime identity device does not match the live CUDA KV tensors",
                operation="initialize",
            )
        live_dtype = self._view.dtype.removeprefix("torch.")
        if identity.dtype.removeprefix("torch.") != live_dtype:
            raise VllmLiveConfigurationError(
                "runtime identity dtype does not match the live KV tensors",
                operation="initialize",
            )
        self._engine = engine
        self._identity = identity
        self._max_fanout = max_fanout
        self._default_maximum_new_tokens = default_maximum_new_tokens
        self._gpu_memory_reader = gpu_memory_reader
        self._clock_ns = clock_ns
        self._trace_recorder = (
            None
            if trace_level is TraceLevel.DISABLED
            else VllmLiveTraceRecorder(
                identity=identity,
                trace_id=trace_id
                or "vllm-live:"
                + sha256(identity.model_dump_json().encode("utf-8")).hexdigest()[:24],
                level=trace_level,
                max_events=max_trace_events,
                clock_ns=clock_ns,
            )
        )
        self._sessions: dict[str, _LiveSession] = {}
        self._runtime_to_session: dict[str, str] = {}
        self._metadata_pending_session_id: str | None = None
        self._roots: dict[str, SharedRootReference] = {}
        self._released_experiment_block_ids: set[str] = set()
        self._observer = _ObserverState()
        self._closed = False
        self._original_allocate_slots = self._view.manager.allocate_slots
        self._original_free = self._view.manager.free
        self._install_observers()
        self._metadata_recorder = VllmMetadataRecorder0230(
            self._view,
            level=metadata_instrumentation_level,
            optimization=metadata_optimization,
            request_scope=self._metadata_request_scope,
            maximum_spans=max_trace_events,
            clock_ns=clock_ns,
        )

    @classmethod
    def create(
        cls,
        *,
        identity: RuntimeModelIdentity,
        execution_model_path: str,
        seed: int,
        timeout_s: float,
        llm_kwargs: Mapping[str, object],
        max_fanout: int = 32,
        default_maximum_new_tokens: int = 256,
        gpu_memory_reader: _GpuMemoryReader | None = None,
        trace_level: TraceLevel = TraceLevel.DISABLED,
        trace_id: str | None = None,
        max_trace_events: int = 262_144,
        metadata_instrumentation_level: MetadataInstrumentationLevel = (
            MetadataInstrumentationLevel.DISABLED
        ),
        metadata_optimization: MetadataOptimization = MetadataOptimization.BASELINE,
    ) -> VllmLiveStateAdapter:
        """Create an isolated vLLM engine without importing vLLM at module import."""

        _require_seed(seed)
        deadline = _require_timeout(timeout_s)
        if identity.device != "cuda:0":
            raise ValueError("the bounded TP=1 live adapter requires identity.device='cuda:0'")
        requested_path = Path(execution_model_path)
        if not requested_path.is_absolute():
            raise ValueError("execution_model_path must be absolute")
        try:
            model_path = requested_path.resolve(strict=True)
        except OSError as error:
            raise VllmLiveConfigurationError(
                "execution_model_path does not resolve to a local snapshot",
                operation="create",
            ) from error
        if not model_path.is_dir() or model_path.name != identity.model_revision:
            raise VllmLiveConfigurationError(
                "execution_model_path must be an immutable snapshot directory named by model_revision",
                operation="create",
            )
        required_snapshot_files = ("config.json", "tokenizer_config.json")
        if any(not (model_path / name).is_file() for name in required_snapshot_files) or not any(
            model_path.glob("*.safetensors")
        ):
            raise VllmLiveConfigurationError(
                "local snapshot lacks config, tokenizer metadata, or safetensor weights",
                operation="create",
            )
        required = {
            "model",
            "revision",
            "tokenizer",
            "tokenizer_revision",
            "dtype",
            "max_model_len",
            "enable_prefix_caching",
            "block_size",
            "max_num_seqs",
            "seed",
            "tensor_parallel_size",
            "async_scheduling",
            "cpu_offload_gb",
        }
        missing = sorted(required - llm_kwargs.keys())
        if missing:
            raise ValueError(f"llm_kwargs must explicitly set: {', '.join(missing)}")
        expected = {
            "model": str(model_path),
            "revision": identity.model_revision,
            "tokenizer": str(model_path),
            "tokenizer_revision": identity.tokenizer_revision,
            "dtype": identity.dtype,
            "seed": seed,
            "tensor_parallel_size": 1,
            "async_scheduling": False,
            "cpu_offload_gb": 0.0,
        }
        mismatches = [key for key, value in expected.items() if llm_kwargs.get(key) != value]
        if mismatches:
            raise ValueError(
                "llm_kwargs disagree with the pinned identity/run: " + ", ".join(sorted(mismatches))
            )
        if not isinstance(llm_kwargs["max_model_len"], int) or llm_kwargs["max_model_len"] <= 0:
            raise ValueError("max_model_len must be an explicitly bounded positive integer")
        if not isinstance(llm_kwargs["block_size"], int) or llm_kwargs["block_size"] <= 0:
            raise ValueError("block_size must be an explicitly bounded positive integer")
        if not isinstance(llm_kwargs["max_num_seqs"], int) or not (
            max_fanout <= llm_kwargs["max_num_seqs"] <= 64
        ):
            raise ValueError("max_num_seqs must cover max_fanout and be at most 64")
        try:
            discovered_version = importlib.metadata.version("vllm")
        except importlib.metadata.PackageNotFoundError as error:
            raise AdapterUnavailableError(
                "vllm distribution is not installed", operation="create"
            ) from error
        if discovered_version != VLLM_LIVE_RUNTIME_VERSION:
            raise UnsupportedRuntimeVersionError(
                (
                    "live vLLM state adapter requires exactly "
                    f"{VLLM_LIVE_RUNTIME_VERSION}, got {discovered_version!r}"
                ),
                operation="create",
                runtime_name="vllm",
                runtime_version=discovered_version,
            )
        if "vllm" in sys.modules and os.environ.get("VLLM_ENABLE_V1_MULTIPROCESSING") != "0":
            raise VllmLiveConfigurationError(
                "vLLM was imported before in-process V1 was selected",
                operation="create",
            )
        configured_multiprocessing = os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
        if configured_multiprocessing != "0":
            raise VllmLiveConfigurationError(
                "VLLM_ENABLE_V1_MULTIPROCESSING must equal '0'",
                operation="create",
            )
        bounded_kwargs = dict(llm_kwargs)
        module = importlib.import_module("vllm")
        llm_factory = getattr(module, "LLM", None)
        if not callable(llm_factory):
            raise AdapterUnavailableError(
                "installed vLLM exposes no LLM factory", operation="create"
            )
        engine = llm_factory(**bounded_kwargs)
        if time.monotonic_ns() > deadline:
            core = getattr(getattr(engine, "llm_engine", None), "engine_core", None)
            shutdown = getattr(core, "shutdown", None)
            if callable(shutdown):
                shutdown()
            raise TimeoutError("vLLM engine initialization exceeded timeout_s")
        return cls(
            engine,
            identity=identity,
            runtime_version=discovered_version,
            max_fanout=max_fanout,
            default_maximum_new_tokens=default_maximum_new_tokens,
            gpu_memory_reader=gpu_memory_reader,
            trace_level=trace_level,
            trace_id=trace_id or f"vllm-live:{seed}",
            max_trace_events=max_trace_events,
            metadata_instrumentation_level=metadata_instrumentation_level,
            metadata_optimization=metadata_optimization,
        )

    @property
    def capability_matrix(self) -> RuntimeCapabilityMatrix:
        capabilities = {
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
        if self._view.prefix_caching_enabled:
            capabilities.update(
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
            capabilities.add(RealRuntimeCapability.RUN_CONCURRENT_INDEPENDENT_SESSIONS)
        if self._trace_recorder is not None:
            capabilities.update(
                {
                    RealRuntimeCapability.EMIT_BRANCH_WORKLOAD_TRACE,
                    RealRuntimeCapability.EMIT_STATE_OPERATION_TRACE,
                }
            )
        return RuntimeCapabilityMatrix(
            identity=self._identity,
            capabilities=frozenset(capabilities),
            scope=RuntimeScopeContract(
                same_process_or_adapter_scope=(
                    "one vLLM 0.23.0 InprocClient engine and one adapter lifetime"
                )
            ),
            version_pin=f"vllm=={VLLM_LIVE_RUNTIME_VERSION}",
            internal_interfaces=VLLM_LIVE_INTERNAL_INTERFACES,
            max_sessions=self._max_fanout + 1,
            max_fanout=self._max_fanout,
        )

    def _install_observers(self) -> None:
        def observed_allocate(request: object, *args: object, **kwargs: object) -> object:
            request_id = _request_id(request)
            previous_ids = {
                block_id
                for group in self._observer.last_request_blocks.get(request_id, ())
                for block_id in group
            }
            cached_tokens_value = kwargs.get(
                "num_new_computed_tokens", args[1] if len(args) > 1 else 0
            )
            if not isinstance(cached_tokens_value, int) or cached_tokens_value < 0:
                raise SnapshotConsistencyError(
                    "vLLM allocate_slots exposed invalid prefix-cache hit evidence",
                    operation="observe_block_allocation",
                    session_id=request_id,
                )
            cached_blocks_value = kwargs.get(
                "new_computed_blocks", args[2] if len(args) > 2 else None
            )
            cached_groups = getattr(cached_blocks_value, "blocks", ())
            cached_ids = tuple(tuple(block.block_id for block in group) for group in cached_groups)
            # KVCacheManager.empty_kv_cache_blocks preserves one empty tuple per
            # cache group.  That structural shape is not a cache hit: normalize
            # it to the same empty evidence used when no block container exists.
            if not any(cached_ids):
                cached_ids = ()
            self._observer.initial_cached_tokens.setdefault(request_id, cached_tokens_value)
            self._observer.initial_cached_block_ids.setdefault(request_id, cached_ids)
            result = self._original_allocate_slots(request, *args, **kwargs)
            if result is not None:
                groups = self._view.request_blocks(request_id)
                newly_allocated_groups = getattr(result, "blocks", ())
                observed_new_ids: list[int] = []
                for group in newly_allocated_groups:
                    for block in group:
                        if block.is_null:
                            continue
                        if block.block_id not in observed_new_ids:
                            observed_new_ids.append(block.block_id)
                cached_id_set = {block_id for group in cached_ids for block_id in group}
                # vLLM 0.23's external-computed-token restore path appends the
                # allocated pages to the request table and the zeroing queue,
                # but intentionally returns no blocks from allocate_slots():
                # there are no tokens left for the scheduler to compute.  The
                # full table delta is therefore the authoritative allocation
                # observation for those pages. Prefix-cache hits and pages
                # already owned by the request are not new allocations.
                for group in groups:
                    for block in group:
                        if (
                            block.block_id not in previous_ids
                            and block.block_id not in cached_id_set
                            and block.block_id not in observed_new_ids
                        ):
                            observed_new_ids.append(block.block_id)
                for block_id in observed_new_ids:
                    self._observer.allocation_epoch += 1
                    self._observer.block_epochs[block_id] = self._observer.allocation_epoch
                self._observer.last_request_blocks[request_id] = tuple(
                    tuple(block.block_id for block in group) for group in groups
                )
                self._observer.request_admitted_ns.setdefault(request_id, self._clock_ns())
            return result

        def observed_free(request: object) -> None:
            request_id = _request_id(request)
            try:
                groups = self._view.request_blocks(request_id)
            except (KeyError, SnapshotConsistencyError):
                groups = ()
            self._observer.released_request_blocks[request_id] = tuple(
                tuple(block.block_id for block in group) for group in groups
            )
            self._original_free(request)

        self._view.manager.allocate_slots = observed_allocate
        self._view.manager.free = observed_free

    def _metadata_request_scope(self, request_id: str) -> RequestScope0230 | None:
        session = self._sessions.get(request_id)
        if session is None:
            mapped = self._runtime_to_session.get(request_id)
            session = self._sessions.get(mapped) if mapped is not None else None
        if session is None and self._metadata_pending_session_id is not None:
            session = self._sessions.get(self._metadata_pending_session_id)
        if session is None:
            return None
        root = self._roots.get(session.root_reference_id or "")
        prefix_tokens = root.prefix_token_count if root is not None else None
        return RequestScope0230(
            request_id=request_id,
            root_reference_id=session.root_reference_id,
            source_session_id=None if root is None else root.source_session_id,
            prefix_token_count=prefix_tokens,
            eligible_for_root_hash_template=(
                root is not None
                and session.branch_id is not None
                and len(session.token_ids) == root.prefix_token_count + 1
            ),
        )

    def begin_post_root_metadata_observation(
        self, *, branch_count: int, prefix_block_count: int
    ) -> None:
        """Begin one continuous, version-scoped POST_ROOT_READY parent span."""

        self._metadata_recorder.begin_post_root_ready(
            branch_count=branch_count,
            prefix_block_count=prefix_block_count,
        )

    def end_post_root_metadata_observation(self) -> None:
        self._metadata_recorder.end_post_root_ready()

    def metadata_span(
        self, name: str, category: str, attributes: Mapping[str, object] | None = None
    ) -> AbstractContextManager[None]:
        """Return a context manager for an explicit SLOForge-side child span."""

        return self._metadata_recorder.span(name, category, attributes)

    def inspect_metadata_observation(self) -> dict[str, object]:
        return self._metadata_recorder.snapshot()

    def _require_open(self, operation: str) -> None:
        if self._closed:
            raise SessionStateError(
                "vLLM live adapter has already been cleaned up", operation=operation
            )

    def _session(self, session_id: str, operation: str) -> _LiveSession:
        self._require_open(operation)
        try:
            return self._sessions[session_id]
        except KeyError as error:
            raise SessionNotFoundError(
                f"unknown live vLLM session {session_id!r}",
                operation=operation,
                session_id=session_id,
            ) from error

    def _session_ref(self, session: _LiveSession) -> RuntimeSessionRef:
        return RuntimeSessionRef(
            session_id=session.session_id,
            adapter_session_id=session.runtime_request_id or session.session_id,
            phase=session.phase,
            seed=session.seed,
            branch_id=session.branch_id,
            parent_session_id=session.parent_session_id,
            root_reference_id=session.root_reference_id,
        )

    def start_session(
        self,
        session_id: str,
        *,
        token_ids: tuple[int, ...],
        seed: int,
        timeout_s: float,
    ) -> RuntimeSessionRef:
        deadline = _require_timeout(timeout_s)
        self._require_open("start_session")
        _require_seed(seed)
        if not session_id or len(session_id) > 512:
            raise ValueError("session_id must contain 1..512 characters")
        if session_id in self._sessions:
            raise SessionStateError(
                "session already exists", operation="start_session", session_id=session_id
            )
        if len(self._sessions) >= self._max_fanout + 1:
            raise ResourceLimitError(
                "live session limit reached", operation="start_session", session_id=session_id
            )
        if not token_ids or len(token_ids) > 1_048_576 or any(token < 0 for token in token_ids):
            raise ValueError("token_ids must be a non-empty bounded non-negative history")
        session = _LiveSession(session_id=session_id, token_ids=token_ids, seed=seed)
        self._sessions[session_id] = session
        _check_deadline(deadline, "start_session")
        return self._session_ref(session)

    def _sampling_params(self, maximum_new_tokens: int, seed: int) -> object:
        module = importlib.import_module("vllm")
        sampling_factory = getattr(module, "SamplingParams", None)
        if not callable(sampling_factory):
            raise AdapterUnavailableError(
                "installed vLLM exposes no SamplingParams", operation="sampling"
            )
        return sampling_factory(
            max_tokens=maximum_new_tokens,
            min_tokens=maximum_new_tokens,
            ignore_eos=True,
            temperature=0.0,
            seed=seed,
        )

    def _effective_seed(self, run_seed: int, session: _LiveSession) -> int:
        payload = f"{run_seed}\0{session.seed}\0{session.session_id}".encode()
        return int.from_bytes(sha256(payload).digest()[:8], "little") % (2**63 - 1)

    def _submit(self, session: _LiveSession, maximum_new_tokens: int, run_seed: int) -> None:
        if session.submitted:
            if session.target_new_tokens != maximum_new_tokens:
                raise SessionStateError(
                    "a submitted vLLM request cannot change its continuation target",
                    operation="submit",
                    session_id=session.session_id,
                )
            return
        if self._metadata_recorder.level is MetadataInstrumentationLevel.DISABLED:
            actual = self._submit_request(session, maximum_new_tokens, run_seed)
        else:
            with self._metadata_recorder.span(
                "python_runtime_request_construction",
                "REQUEST_BUILD",
                {"session_id": session.session_id, "prompt_tokens": len(session.token_ids)},
            ):
                actual = self._submit_request(session, maximum_new_tokens, run_seed)
        if not isinstance(actual, str) or not actual:
            raise SnapshotConsistencyError(
                "vLLM returned an invalid internal request ID",
                operation="submit",
                session_id=session.session_id,
            )
        session.runtime_request_id = actual
        session.submitted = True
        session.target_new_tokens = maximum_new_tokens
        self._runtime_to_session[actual] = session.session_id
        self._metadata_recorder.mark_scheduler_wait_start()

    def _submit_request(
        self, session: _LiveSession, maximum_new_tokens: int, run_seed: int
    ) -> object:
        params = self._sampling_params(maximum_new_tokens, self._effective_seed(run_seed, session))
        prompt_token_ids = list(session.token_ids)
        self._metadata_recorder.record_prompt_token_writes(len(prompt_token_ids))
        prompt = {"prompt_token_ids": prompt_token_ids}
        self._metadata_pending_session_id = session.session_id
        try:
            return self._view.llm_engine.add_request(session.session_id, prompt, params)
        finally:
            self._metadata_pending_session_id = None

    def _process_outputs(self, outputs: object, first_token_ns: dict[str, int]) -> None:
        if self._metadata_recorder.level is MetadataInstrumentationLevel.DISABLED:
            self._process_outputs_untraced(outputs, first_token_ns)
            return
        with self._metadata_recorder.span(
            "frontend_output_return_and_commit",
            "OUTPUT_TOKEN_COMMIT",
        ):
            self._process_outputs_untraced(outputs, first_token_ns)

    def _process_outputs_untraced(self, outputs: object, first_token_ns: dict[str, int]) -> None:
        if not isinstance(outputs, list):
            return
        for output in outputs:
            output_request_id = getattr(output, "request_id", None)
            session = self._sessions.get(str(output_request_id))
            if session is None:
                mapped = self._runtime_to_session.get(str(output_request_id))
                session = self._sessions.get(mapped) if mapped else None
            if session is None:
                continue
            completions = getattr(output, "outputs", None)
            if not isinstance(completions, list) or not completions:
                continue
            token_ids = getattr(completions[0], "token_ids", ())
            if isinstance(token_ids, (list, tuple)):
                session.output_token_ids = tuple(int(token) for token in token_ids)
            if session.output_token_ids:
                first_token_ns.setdefault(session.session_id, self._clock_ns())
            if bool(getattr(output, "finished", False)):
                session.phase = LiveSessionPhase.PAUSED

    def prefill_session(self, session_id: str, *, timeout_s: float) -> RuntimeSessionRef:
        deadline = _require_timeout(timeout_s)
        session = self._session(session_id, "prefill_session")
        if session.parent_session_id is not None:
            raise SessionStateError(
                "prefill_session is reserved for a shared-root source",
                operation="prefill_session",
                session_id=session_id,
            )
        if session.phase is not LiveSessionPhase.CREATED:
            raise SessionStateError(
                "session is not awaiting prefill",
                operation="prefill_session",
                session_id=session_id,
            )
        self._submit(session, 1, session.seed)
        assert session.runtime_request_id is not None
        while session.runtime_request_id not in self._observer.released_request_blocks:
            _check_deadline(deadline, "prefill_session")
            outputs = self._view.llm_engine.step()
            self._process_outputs(outputs, {})
        block_ids = self._observer.last_request_blocks.get(session.runtime_request_id)
        if not block_ids or len(block_ids) != 1:
            raise SnapshotConsistencyError(
                "root prefill did not allocate one observable GPU KV block table",
                operation="prefill_session",
                session_id=session_id,
            )
        session.output_token_ids = ()  # sampled probe token is outside the branch root
        session.phase = LiveSessionPhase.PREFILLED
        _check_deadline(deadline, "prefill_session")
        return self._session_ref(session)

    def pause_at_safe_decode_boundary(
        self, session_id: str, *, timeout_s: float
    ) -> RuntimeSessionRef:
        deadline = _require_timeout(timeout_s)
        session = self._session(session_id, "pause_at_safe_decode_boundary")
        scheduler_config = getattr(self._view.vllm_config, "scheduler_config", None)
        if bool(getattr(scheduler_config, "async_scheduling", False)):
            raise SessionStateError(
                "async scheduling cannot establish the required boundary",
                operation="pause_at_safe_decode_boundary",
                session_id=session_id,
            )
        if session.phase is LiveSessionPhase.DESTROYED:
            raise SessionStateError(
                "destroyed session cannot be paused",
                operation="pause_at_safe_decode_boundary",
                session_id=session_id,
            )
        session.phase = LiveSessionPhase.PAUSED
        _check_deadline(deadline, "pause_at_safe_decode_boundary")
        return self._session_ref(session)

    def inspect_logical_state(self, session_id: str, *, timeout_s: float) -> LogicalRuntimeState:
        deadline = _require_timeout(timeout_s)
        session = self._session(session_id, "inspect_logical_state")
        token_ids = session.token_ids + session.output_token_ids
        if session.runtime_request_id:
            request = self._view.request_object(session.runtime_request_id)
            runtime_tokens = getattr(request, "all_token_ids", None)
            if isinstance(runtime_tokens, list):
                token_ids = tuple(int(token) for token in runtime_tokens)
        ranges = ((0, len(token_ids)),) if token_ids else ()
        result = LogicalRuntimeState(
            session_id=session_id,
            model=self._identity,
            token_ids=token_ids,
            token_history_sha256=token_history_sha256(token_ids),
            position_start=0,
            position_end_exclusive=len(token_ids),
            attention_layer_ids=self._view.layer_ids,
            logical_kv_token_ranges=ranges,
        )
        _check_deadline(deadline, "inspect_logical_state")
        return result

    def _active_block_tables(
        self, session_ids: tuple[str, ...]
    ) -> dict[str, tuple[tuple[_BlockLike, ...], ...]]:
        tables: dict[str, tuple[tuple[_BlockLike, ...], ...]] = {}
        for session_id in session_ids:
            session = self._session(session_id, "inspect_physical_kv_layout")
            runtime_id = session.runtime_request_id
            if runtime_id and self._view.request_object(runtime_id) is not None:
                tables[session_id] = self._view.request_blocks(runtime_id)
        return tables

    def _root_ids_for_sessions(self, session_ids: tuple[str, ...]) -> set[str]:
        root_ids: set[str] = set()
        for session_id in session_ids:
            session = self._sessions[session_id]
            if session.root_reference_id and session.root_reference_id in self._roots:
                root_ids.update(self._roots[session.root_reference_id].block_ids)
            for root in self._roots.values():
                if root.source_session_id == session_id:
                    root_ids.update(root.block_ids)
        return root_ids

    def _root_positions_for_sessions(self, session_ids: tuple[str, ...]) -> dict[str, int]:
        positions: dict[str, int] = {}
        roots = self._root_ids_for_sessions(session_ids)
        for root in self._roots.values():
            for position, runtime_id in enumerate(root.block_ids):
                if runtime_id in roots:
                    prior = positions.setdefault(runtime_id, position)
                    if prior != position:
                        raise SnapshotConsistencyError(
                            "a physical root block has inconsistent logical positions",
                            operation="inspect_physical_kv_layout",
                        )
        return positions

    def inspect_physical_kv_layout(
        self, session_ids: tuple[str, ...], *, timeout_s: float
    ) -> PhysicalKvLayoutSnapshot:
        deadline = _require_timeout(timeout_s)
        if not session_ids or len(session_ids) != len(set(session_ids)):
            raise ValueError("session_ids must be non-empty and unique")
        tables = self._active_block_tables(session_ids)
        root_ids = self._root_ids_for_sessions(session_ids)
        owners: dict[str, set[str]] = {}
        positions = self._root_positions_for_sessions(session_ids)
        for session_id, groups in tables.items():
            for group_idx, group in enumerate(groups):
                for position, block in enumerate(group):
                    if block.is_null:
                        continue
                    runtime_id = _runtime_block_id(group_idx, block.block_id)
                    owners.setdefault(runtime_id, set()).add(session_id)
                    prior = positions.setdefault(runtime_id, position)
                    if prior != position:
                        raise SnapshotConsistencyError(
                            "one physical KV block appeared at inconsistent logical positions",
                            operation="inspect_physical_kv_layout",
                        )

        observed_ids = set(owners) | root_ids
        blocks: list[PhysicalKvBlock] = []
        shared_ids: list[str] = []
        private_ids: list[str] = []
        for runtime_id in sorted(observed_ids):
            parts = runtime_id.split(":")
            if len(parts) != 5 or parts[:2] != ["vllm", "kv-group"] or parts[3] != "block":
                raise SnapshotConsistencyError(
                    "adapter produced a malformed vLLM block identifier",
                    operation="inspect_physical_kv_layout",
                )
            group_idx = int(parts[2])
            block_idx = int(parts[4])
            block = self._view.pool_block(block_idx)
            branch_ids = tuple(sorted(owners.get(runtime_id, ())))
            position = positions.get(runtime_id, len(blocks))
            logical_start = position * self._view.block_size_tokens
            logical_end = logical_start + self._view.block_size_tokens
            if runtime_id in root_ids:
                state_class = (
                    KvStateClass.SHARED_PREFIX if branch_ids else KvStateClass.ROOT_REFERENCE
                )
                shared_ids.append(runtime_id)
            else:
                state_class = KvStateClass.PRIVATE_SUFFIX
                private_ids.append(runtime_id)
            blocks.append(
                PhysicalKvBlock(
                    runtime_block_id=runtime_id,
                    block_index=block_idx,
                    kv_cache_group=group_idx,
                    layer_ids=self._view.layer_ids,
                    device=self._view.device,
                    dtype=self._view.dtype,
                    bytes=self._view.page_size_bytes,
                    block_size_tokens=self._view.block_size_tokens,
                    logical_token_start=logical_start,
                    logical_token_end_exclusive=logical_end,
                    branch_ids=branch_ids,
                    refcount=block.ref_cnt,
                    refcount_evidence=RefcountEvidence.RUNTIME_NATIVE,
                    state_class=state_class,
                    allocation_epoch=self._observer.block_epochs.get(block_idx, 0),
                    cache_resident=block.block_hash is not None,
                )
            )
        self._observer.snapshot_epoch += 1
        root_branch_ids = {
            branch
            for runtime_id, branches in owners.items()
            if runtime_id in root_ids
            for branch in branches
        }
        assigned = sum(block.bytes for block in blocks)
        shared_bytes = len(shared_ids) * self._view.page_size_bytes
        private_bytes = len(private_ids) * self._view.page_size_bytes
        result = PhysicalKvLayoutSnapshot(
            runtime="vllm",
            runtime_version=VLLM_LIVE_RUNTIME_VERSION,
            snapshot_epoch=self._observer.snapshot_epoch,
            observed_at_monotonic_ns=self._clock_ns(),
            session_ids=session_ids,
            blocks=tuple(blocks),
            shared_prefix_block_ids=tuple(shared_ids),
            private_suffix_block_ids=tuple(private_ids),
            root_reference_count=len(root_branch_ids),
            physical_assigned_bytes=assigned,
            shared_prefix_bytes=shared_bytes,
            private_suffix_bytes=private_bytes,
            kv_pool_reserved_bytes=self._view.kv_pool_reserved_bytes,
        )
        _check_deadline(deadline, "inspect_physical_kv_layout")
        return result

    def inspect_gpu_memory_state(self, *, timeout_s: float) -> GpuMemoryState:
        deadline = _require_timeout(timeout_s)
        self._require_open("inspect_gpu_memory_state")
        pool = self._view.manager.block_pool
        blocks = getattr(pool, "blocks", None)
        if not isinstance(blocks, list):
            raise SnapshotConsistencyError(
                "vLLM block pool disappeared", operation="inspect_gpu_memory_state"
            )
        assigned_blocks = sum(
            1
            for block in cast(list[_BlockLike], blocks)
            if (block.ref_cnt > 0 or block.block_hash is not None) and not block.is_null
        )
        pool_bytes = self._view.kv_pool_reserved_bytes
        assigned_bytes = assigned_blocks * self._view.page_size_bytes
        if assigned_bytes > pool_bytes:
            raise SnapshotConsistencyError(
                "vLLM active block accounting exceeds the GPU KV pool",
                operation="inspect_gpu_memory_state",
            )
        samples: dict[str, int | None] = {}
        source = "vllm_runtime_native_kv_pool"
        if self._gpu_memory_reader is not None:
            samples.update(self._gpu_memory_reader(self._view.device))
            source += "+injected_gpu_memory_reader"
        try:
            torch = importlib.import_module("torch")
            index = getattr(getattr(self._view.runner, "device", None), "index", None)
            samples["torch_allocated_bytes"] = int(torch.cuda.memory_allocated(index))
            samples["torch_reserved_bytes"] = int(torch.cuda.memory_reserved(index))
            source += "+torch_cuda"
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError):
            pass
        result = GpuMemoryState(
            device=self._view.device,
            observed_at_monotonic_ns=self._clock_ns(),
            nvml_process_bytes=samples.get("nvml_process_bytes"),
            nvml_device_used_bytes=samples.get("nvml_device_used_bytes"),
            torch_allocated_bytes=samples.get("torch_allocated_bytes"),
            torch_reserved_bytes=samples.get("torch_reserved_bytes"),
            kv_pool_reserved_bytes=pool_bytes,
            kv_assigned_bytes=assigned_bytes,
            kv_unassigned_bytes=pool_bytes - assigned_bytes,
            sample_source=source,
        )
        _check_deadline(deadline, "inspect_gpu_memory_state")
        return result

    def identify_shared_prefix_blocks(
        self, branch_session_ids: tuple[str, ...], *, timeout_s: float
    ) -> tuple[str, ...]:
        deadline = _require_timeout(timeout_s)
        if not self._view.prefix_caching_enabled:
            raise UnsupportedCapabilityError(
                "prefix caching is disabled for this engine",
                operation="identify_shared_prefix_blocks",
            )
        tables = self._active_block_tables(branch_session_ids)
        if len(tables) != len(branch_session_ids):
            raise SessionStateError(
                "all branches must have live runtime block tables",
                operation="identify_shared_prefix_blocks",
            )
        rows = [groups[0] for groups in tables.values()]
        shared: list[str] = []
        for blocks_at_position in zip(*rows, strict=False):
            ids = {block.block_id for block in blocks_at_position}
            if len(ids) != 1:
                break
            block_id = ids.pop()
            shared.append(_runtime_block_id(0, block_id))
        _check_deadline(deadline, "identify_shared_prefix_blocks")
        return tuple(shared)

    def create_shared_root_reference(
        self, session_id: str, *, prefix_token_count: int, timeout_s: float
    ) -> SharedRootReference:
        deadline = _require_timeout(timeout_s)
        if not self._view.prefix_caching_enabled:
            raise UnsupportedCapabilityError(
                "shared-root creation requires vLLM prefix caching",
                operation="create_shared_root_reference",
                session_id=session_id,
            )
        session = self._session(session_id, "create_shared_root_reference")
        if any(root.source_session_id == session_id for root in self._roots.values()):
            raise SessionStateError(
                "source session already owns a shared-root lifetime reference",
                operation="create_shared_root_reference",
                session_id=session_id,
            )
        if session.phase not in {LiveSessionPhase.PREFILLED, LiveSessionPhase.PAUSED}:
            raise SessionStateError(
                "shared root must be captured after completed prefill",
                operation="create_shared_root_reference",
                session_id=session_id,
            )
        if prefix_token_count != len(session.token_ids):
            raise ValueError("shared root must cover the complete source prompt")
        if prefix_token_count % self._view.block_size_tokens:
            raise ValueError("shared root token count must be KV-block aligned")
        runtime_id = session.runtime_request_id
        raw_groups = self._observer.last_request_blocks.get(runtime_id or "")
        expected = prefix_token_count // self._view.block_size_tokens
        if not raw_groups or len(raw_groups) != 1 or len(raw_groups[0]) != expected:
            raise SnapshotConsistencyError(
                "prefill block table does not exactly cover the aligned shared root",
                operation="create_shared_root_reference",
                session_id=session_id,
            )
        block_ids = tuple(_runtime_block_id(0, block) for block in raw_groups[0])
        for block_idx in raw_groups[0]:
            block = self._view.pool_block(block_idx)
            if block.ref_cnt != 0 or block.block_hash is None:
                raise SnapshotConsistencyError(
                    "completed root is not resident and free in the vLLM prefix cache",
                    operation="create_shared_root_reference",
                    session_id=session_id,
                )
        digest = sha256((session_id + "\0" + "\0".join(block_ids)).encode("utf-8")).hexdigest()[:24]
        root_id = f"vllm-root:{session_id}:{digest}"
        root = SharedRootReference(
            root_reference_id=root_id,
            source_session_id=session_id,
            prefix_token_count=prefix_token_count,
            block_ids=block_ids,
            physical_bytes=len(block_ids) * self._view.page_size_bytes,
            adapter_owned_reference_count=1,
            created_at_monotonic_ns=self._clock_ns(),
        )
        self._roots[root_id] = root
        if session.runtime_request_id is None:
            raise SnapshotConsistencyError(
                "shared-root source lacks its completed runtime request identity",
                operation="create_shared_root_reference",
                session_id=session_id,
            )
        self._metadata_recorder.publish_root_hash_template(
            root_reference_id=root_id,
            source_request_id=session.runtime_request_id,
            prefix_block_count=len(block_ids),
        )
        if self._trace_recorder is not None:
            snapshot = self.inspect_physical_kv_layout(
                (session_id,),
                timeout_s=max((deadline - time.monotonic_ns()) / 1_000_000_000, 1e-9),
            )
            self._trace_recorder.observe_layout(
                snapshot,
                session_id=session_id,
                branch_group_id=root_id,
                cause="root_publication",
            )
            self._trace_recorder.record_branch(
                BranchOperationType.STATE_PUBLISH,
                session_id=session_id,
                branch_group_id=root_id,
                physical_state_id=root_id,
                physical_bytes=root.physical_bytes,
                shared_root=True,
                attributes={"prefix_token_count": prefix_token_count},
            )
        _check_deadline(deadline, "create_shared_root_reference")
        return root

    def fork_same_policy_session(
        self,
        root_reference_id: str,
        branch_session_id: str,
        *,
        divergent_token_id: int,
        seed: int,
        timeout_s: float,
    ) -> RuntimeSessionRef:
        deadline = _require_timeout(timeout_s)
        self._require_open("fork_same_policy_session")
        _require_seed(seed)
        if divergent_token_id < 0:
            raise ValueError("divergent_token_id must be non-negative")
        try:
            root = self._roots[root_reference_id]
        except KeyError as error:
            raise SessionNotFoundError(
                "unknown shared-root reference",
                operation="fork_same_policy_session",
                session_id=branch_session_id,
            ) from error
        if branch_session_id in self._sessions:
            raise SessionStateError(
                "branch session already exists",
                operation="fork_same_policy_session",
                session_id=branch_session_id,
            )
        active_branches = sum(session.branch_id is not None for session in self._sessions.values())
        if active_branches >= self._max_fanout:
            raise ResourceLimitError(
                "branch fanout limit reached",
                operation="fork_same_policy_session",
                session_id=branch_session_id,
            )
        parent = self._sessions[root.source_session_id]
        tokens = (*parent.token_ids[: root.prefix_token_count], divergent_token_id)
        branch = _LiveSession(
            session_id=branch_session_id,
            token_ids=tokens,
            seed=seed,
            parent_session_id=root.source_session_id,
            root_reference_id=root_reference_id,
            branch_id=branch_session_id,
        )
        self._sessions[branch_session_id] = branch
        self._metadata_recorder.record_branch_session_allocation()
        if self._trace_recorder is not None:
            self._trace_recorder.record_branch(
                BranchOperationType.BRANCH_FORK,
                session_id=branch_session_id,
                branch_group_id=root_reference_id,
                branch_id=branch_session_id,
                parent_branch_id=root.source_session_id,
                physical_state_id=root_reference_id,
                physical_bytes=root.physical_bytes,
                shared_root=True,
                fanout=max(1, active_branches + 1),
                attributes={
                    "divergent_token_id": divergent_token_id,
                    "metadata_only_before_runtime_admission": True,
                },
            )
        _check_deadline(deadline, "fork_same_policy_session")
        return self._session_ref(branch)

    def fork_same_policy_sessions_0230(
        self,
        root_reference_id: str,
        branch_specs: tuple[tuple[str, int, int], ...],
        *,
        timeout_s: float,
    ) -> tuple[RuntimeSessionRef, ...]:
        """Bulk-create logical branches for the scoped Experiment 003 fast path.

        ``branch_specs`` rows are ``(session_id, divergent_token_id, seed)``.
        This is deliberately not added to generic Continuum IR: it is an
        internal vLLM-0.23.0/SLOForge optimization that validates immutable root
        metadata and capacity once, then creates the same logical sessions as
        repeated :meth:`fork_same_policy_session` calls.
        """

        deadline = _require_timeout(timeout_s)
        self._require_open("fork_same_policy_sessions_0230")
        if not branch_specs or len(branch_specs) > self._max_fanout:
            raise ValueError("branch_specs must contain 1..max_fanout rows")
        session_ids = tuple(row[0] for row in branch_specs)
        if any(not session_id or len(session_id) > 512 for session_id in session_ids):
            raise ValueError("branch session IDs must contain 1..512 characters")
        if len(session_ids) != len(set(session_ids)):
            raise ValueError("branch session IDs must be unique")
        if any(session_id in self._sessions for session_id in session_ids):
            raise SessionStateError(
                "a bulk branch session already exists",
                operation="fork_same_policy_sessions_0230",
            )
        for _, divergent_token_id, seed in branch_specs:
            if divergent_token_id < 0:
                raise ValueError("divergent token IDs must be non-negative")
            _require_seed(seed)
        try:
            root = self._roots[root_reference_id]
        except KeyError as error:
            raise SessionNotFoundError(
                "unknown shared-root reference",
                operation="fork_same_policy_sessions_0230",
            ) from error
        active_branches = sum(session.branch_id is not None for session in self._sessions.values())
        if active_branches + len(branch_specs) > self._max_fanout:
            raise ResourceLimitError(
                "bulk branch fanout limit reached",
                operation="fork_same_policy_sessions_0230",
            )
        parent = self._sessions[root.source_session_id]
        prefix = (
            parent.token_ids
            if root.prefix_token_count == len(parent.token_ids)
            else parent.token_ids[: root.prefix_token_count]
        )
        refs: list[RuntimeSessionRef] = []
        with self._metadata_recorder.span(
            "bulk_branch_session_construction",
            "HELIX_ORCHESTRATION",
            {
                "branch_count": len(branch_specs),
                "prefix_block_count": len(root.block_ids),
            },
        ):
            for offset, (session_id, divergent_token_id, seed) in enumerate(branch_specs, 1):
                branch = _LiveSession(
                    session_id=session_id,
                    token_ids=(*prefix, divergent_token_id),
                    seed=seed,
                    parent_session_id=root.source_session_id,
                    root_reference_id=root_reference_id,
                    branch_id=session_id,
                )
                self._sessions[session_id] = branch
                self._metadata_recorder.record_branch_session_allocation()
                if self._trace_recorder is not None:
                    self._trace_recorder.record_branch(
                        BranchOperationType.BRANCH_FORK,
                        session_id=session_id,
                        branch_group_id=root_reference_id,
                        branch_id=session_id,
                        parent_branch_id=root.source_session_id,
                        physical_state_id=root_reference_id,
                        physical_bytes=root.physical_bytes,
                        shared_root=True,
                        fanout=max(1, active_branches + offset),
                        attributes={
                            "divergent_token_id": divergent_token_id,
                            "metadata_only_before_runtime_admission": True,
                            "bulk_branch_template": True,
                        },
                    )
                refs.append(self._session_ref(branch))
        _check_deadline(deadline, "fork_same_policy_sessions_0230")
        return tuple(refs)

    def allocate_private_suffix_state(
        self, branch_session_id: str, *, minimum_tokens: int, timeout_s: float
    ) -> PhysicalKvLayoutSnapshot:
        deadline = _require_timeout(timeout_s)
        if minimum_tokens <= 0:
            raise ValueError("minimum_tokens must be positive")
        branch = self._session(branch_session_id, "allocate_private_suffix_state")
        if branch.root_reference_id is None:
            raise SessionStateError(
                "session is not a shared-root branch",
                operation="allocate_private_suffix_state",
                session_id=branch_session_id,
            )
        target = max(minimum_tokens, self._default_maximum_new_tokens) + 1
        self._submit(branch, target, branch.seed)
        first_token_ns: dict[str, int] = {}
        while True:
            _check_deadline(deadline, "allocate_private_suffix_state")
            self._process_outputs(self._view.llm_engine.step(), first_token_ns)
            snapshot = self.inspect_physical_kv_layout(
                (branch_session_id,),
                timeout_s=max((deadline - time.monotonic_ns()) / 1_000_000_000, 1e-9),
            )
            if snapshot.private_suffix_block_ids:
                branch.phase = LiveSessionPhase.READY
                if self._trace_recorder is not None:
                    self._trace_recorder.observe_layout(
                        snapshot,
                        session_id=branch_session_id,
                        branch_group_id=branch.root_reference_id,
                        cause="divergence",
                    )
                return snapshot

    def resume_branch(self, branch_session_id: str, *, timeout_s: float) -> RuntimeSessionRef:
        deadline = _require_timeout(timeout_s)
        branch = self._session(branch_session_id, "resume_branch")
        if branch.root_reference_id is None or branch.phase is LiveSessionPhase.DESTROYED:
            raise SessionStateError(
                "session is not a resumable live branch",
                operation="resume_branch",
                session_id=branch_session_id,
            )
        branch.phase = LiveSessionPhase.READY
        _check_deadline(deadline, "resume_branch")
        return self._session_ref(branch)

    def run_concurrent_branches(
        self,
        branch_session_ids: tuple[str, ...],
        *,
        maximum_new_tokens: int,
        seed: int,
        timeout_s: float,
        on_all_branches_ready: Callable[[], None] | None = None,
    ) -> ConcurrentDecodeResult:
        deadline = _require_timeout(timeout_s)
        _require_seed(seed)
        if not 1 <= len(branch_session_ids) <= self._max_fanout:
            raise ValueError("branch_session_ids fanout is outside adapter bounds")
        if len(branch_session_ids) != len(set(branch_session_ids)):
            raise ValueError("branch_session_ids must be unique")
        if not 1 <= maximum_new_tokens <= 65_535:
            raise ValueError("maximum_new_tokens must be in 1..65535")
        branches = [
            self._session(session_id, "run_concurrent_branches")
            for session_id in branch_session_ids
        ]
        root_ids = {branch.root_reference_id for branch in branches}
        if None in root_ids or len(root_ids) != 1:
            raise SessionStateError(
                "all concurrent branches must descend from one shared root",
                operation="run_concurrent_branches",
            )
        root_id = cast(str, next(iter(root_ids)))
        root = self._roots[root_id]
        started_ns = self._clock_ns()
        for branch in branches:
            # Leave one token of headroom so the target snapshot is captured
            # while every request still owns its physical block table.
            self._submit(branch, maximum_new_tokens + 1, seed)
            branch.phase = LiveSessionPhase.DECODING
        first_token_ns: dict[str, int] = {}
        first_decode_token_started_ns: dict[str, int] = {}
        branch_ready_ns: dict[str, int] = {}
        decode_active_started_ns: int | None = None
        decode_active_boundary_output_counts: dict[str, int] | None = None
        snapshots: list[PhysicalKvLayoutSnapshot] = []
        captured_thresholds: set[int] = set()
        thresholds = {1, 16, 64, maximum_new_tokens}
        while any(len(branch.output_token_ids) < maximum_new_tokens for branch in branches):
            _check_deadline(deadline, "run_concurrent_branches")
            step_started_ns = self._clock_ns()
            outputs = self._view.llm_engine.step()
            step_returned_ns = self._clock_ns()
            previously_active = set(first_token_ns)
            self._process_outputs(outputs, first_token_ns)
            for branch_id in set(first_token_ns) - previously_active:
                first_decode_token_started_ns[branch_id] = step_started_ns
                branch_ready_ns[branch_id] = step_returned_ns
            if decode_active_started_ns is None and len(first_token_ns) == len(branches):
                # Establish one observable boundary after the step that made
                # every branch active. A faster branch may have emitted more
                # than one token by this point, so retain each exact count.
                decode_active_started_ns = self._clock_ns()
                decode_active_boundary_output_counts = {
                    branch.session_id: len(branch.output_token_ids) for branch in branches
                }
                if on_all_branches_ready is not None:
                    on_all_branches_ready()
                    on_all_branches_ready = None
            self._validate_shared_branch_tables(root, branches)
            progress = min(len(branch.output_token_ids) for branch in branches)
            for threshold in sorted(thresholds):
                if progress >= threshold and threshold not in captured_thresholds:
                    snapshot = self.inspect_physical_kv_layout(
                        branch_session_ids,
                        timeout_s=max(
                            (deadline - time.monotonic_ns()) / 1_000_000_000,
                            1e-9,
                        ),
                    )
                    snapshots.append(snapshot)
                    if self._trace_recorder is not None:
                        cause = "divergence" if threshold == 1 else f"suffix_{threshold}"
                        self._trace_recorder.observe_layout(
                            snapshot,
                            session_id=branches[0].session_id,
                            branch_group_id=root_id,
                            cause=cause,
                        )
                        if threshold == 1:
                            for branch in branches:
                                self._trace_recorder.record_branch(
                                    BranchOperationType.BRANCH_DIVERGENCE,
                                    session_id=branch.session_id,
                                    branch_group_id=root_id,
                                    branch_id=branch.branch_id,
                                    parent_branch_id=branch.parent_session_id,
                                    private_suffix=True,
                                    fanout=len(branches),
                                )
                    captured_thresholds.add(threshold)
        ended_ns = self._clock_ns()
        elapsed_ns = ended_ns - started_ns
        if elapsed_ns <= 0:
            raise SnapshotConsistencyError(
                "non-positive concurrent decode duration",
                operation="run_concurrent_branches",
            )
        admitted_latency: dict[str, int] = {}
        decode_started_latency: dict[str, int] = {}
        ready_latency: dict[str, int] = {}
        first_latency: dict[str, int] = {}
        for branch in branches:
            assert branch.runtime_request_id is not None
            first_at = first_token_ns.get(branch.session_id)
            if first_at is None:
                raise SnapshotConsistencyError(
                    "vLLM did not expose full-prefill first-token readiness evidence",
                    operation="run_concurrent_branches",
                    session_id=branch.session_id,
                )
            admitted_at = self._observer.request_admitted_ns.get(branch.runtime_request_id)
            decode_started_at = first_decode_token_started_ns.get(branch.session_id)
            ready_at = branch_ready_ns.get(branch.session_id)
            if admitted_at is None or decode_started_at is None or ready_at is None:
                raise SnapshotConsistencyError(
                    "vLLM did not expose complete admission/readiness/decode-start evidence",
                    operation="run_concurrent_branches",
                    session_id=branch.session_id,
                )
            admitted_latency[branch.session_id] = admitted_at - started_ns
            decode_started_latency[branch.session_id] = (
                max(decode_started_at, admitted_at) - started_ns
            )
            ready_latency[branch.session_id] = ready_at - started_ns
            first_latency[branch.session_id] = first_at - started_ns
            branch.phase = LiveSessionPhase.PAUSED
            if self._trace_recorder is not None:
                self._trace_recorder.record_branch(
                    BranchOperationType.BRANCH_READY,
                    session_id=branch.session_id,
                    branch_group_id=root_id,
                    branch_id=branch.branch_id,
                    parent_branch_id=branch.parent_session_id,
                    duration_ns=ready_latency[branch.session_id],
                    shared_root=True,
                    private_suffix=True,
                    fanout=len(branches),
                    attributes={
                        "request_admitted_latency_ns": admitted_latency[branch.session_id],
                        "first_decode_token_started_latency_ns": decode_started_latency[
                            branch.session_id
                        ],
                        "first_token_latency_ns": first_latency[branch.session_id],
                    },
                )
        truncated_outputs = {
            branch.session_id: branch.output_token_ids[:maximum_new_tokens] for branch in branches
        }
        total = sum(len(tokens) for tokens in truncated_outputs.values())
        if decode_active_started_ns is None or decode_active_boundary_output_counts is None:
            raise SnapshotConsistencyError(
                "concurrent decode never reached the all-branches-active boundary",
                operation="run_concurrent_branches",
            )
        decode_active_elapsed_ns = ended_ns - decode_active_started_ns
        decode_active_tokens = sum(
            len(tokens) - decode_active_boundary_output_counts[branch_id]
            for branch_id, tokens in truncated_outputs.items()
        )
        return ConcurrentDecodeResult(
            branchpoint_id=self._branchpoint_id(root_id, branch_session_ids),
            branch_output_token_ids=truncated_outputs,
            request_admitted_latency_ns=admitted_latency,
            first_decode_token_started_latency_ns=decode_started_latency,
            branch_ready_latency_ns=ready_latency,
            first_token_latency_ns=first_latency,
            elapsed_ns=elapsed_ns,
            decode_active_elapsed_ns=decode_active_elapsed_ns,
            decode_active_boundary_output_counts=decode_active_boundary_output_counts,
            decode_active_tokens=decode_active_tokens,
            total_output_tokens=total,
            throughput_tokens_per_second=total / (elapsed_ns / 1_000_000_000),
            completed_branch_ids=tuple(
                branch.session_id
                for branch in branches
                if len(branch.output_token_ids) >= maximum_new_tokens
            ),
            allocation_snapshots=tuple(snapshots),
        )

    def _validate_shared_branch_tables(
        self, root: SharedRootReference, branches: Sequence[_LiveSession]
    ) -> None:
        expected = tuple(int(runtime_id.rsplit(":", 1)[1]) for runtime_id in root.block_ids)
        private_sets: list[set[int]] = []
        for branch in branches:
            assert branch.runtime_request_id is not None
            request_id = branch.runtime_request_id
            groups = self._observer.last_request_blocks.get(request_id)
            if groups is None:
                continue
            if groups[0][: len(expected)] != expected:
                raise SnapshotConsistencyError(
                    "branch does not reference the published physical root in order",
                    operation="run_concurrent_branches",
                    session_id=branch.session_id,
                )
            if self._observer.initial_cached_tokens.get(request_id) != root.prefix_token_count:
                raise SnapshotConsistencyError(
                    "optimized branch did not reuse exactly the complete root prefix",
                    operation="run_concurrent_branches",
                    session_id=branch.session_id,
                )
            cached_ids = self._observer.initial_cached_block_ids.get(request_id, ())
            if not cached_ids or cached_ids[0] != expected:
                raise SnapshotConsistencyError(
                    "runtime cache-hit blocks differ from the published root",
                    operation="run_concurrent_branches",
                    session_id=branch.session_id,
                )
            private_sets.append(set(groups[0][len(expected) :]))
        if any(
            left & right
            for index, left in enumerate(private_sets)
            for right in private_sets[index + 1 :]
        ):
            raise SnapshotConsistencyError(
                "two branches share a mutable private suffix block",
                operation="run_concurrent_branches",
            )
        if len(private_sets) != len(branches):
            return
        expected_refcount = len(branches)
        if any(
            self._view.pool_block(block_idx).ref_cnt != expected_refcount for block_idx in expected
        ):
            raise SnapshotConsistencyError(
                "native root refs disagree with request-table owners",
                operation="run_concurrent_branches",
            )

    def run_concurrent_independent_sessions(
        self,
        session_ids: tuple[str, ...],
        *,
        maximum_new_tokens: int,
        seed: int,
        timeout_s: float,
        on_all_branches_ready: Callable[[], None] | None = None,
    ) -> ConcurrentDecodeResult:
        """Run a cache-disabled, pairwise-disjoint independent-prefill baseline."""

        deadline = _require_timeout(timeout_s)
        _require_seed(seed)
        if self._view.prefix_caching_enabled:
            raise UnsupportedCapabilityError(
                "independent baseline requires a cache-disabled engine",
                operation="run_concurrent_independent_sessions",
            )
        if not 1 <= len(session_ids) <= self._max_fanout or len(session_ids) != len(
            set(session_ids)
        ):
            raise ValueError("session_ids must be unique and within the fanout bound")
        if not 1 <= maximum_new_tokens <= 65_535:
            raise ValueError("maximum_new_tokens must be in 1..65535")
        sessions = [
            self._session(session_id, "run_concurrent_independent_sessions")
            for session_id in session_ids
        ]
        if any(session.phase is not LiveSessionPhase.CREATED for session in sessions):
            raise SessionStateError(
                "independent baseline requires fresh CREATED sessions",
                operation="run_concurrent_independent_sessions",
            )
        started_ns = self._clock_ns()
        for session in sessions:
            self._submit(session, maximum_new_tokens + 1, seed)
            session.phase = LiveSessionPhase.DECODING
        first_token_ns: dict[str, int] = {}
        first_decode_token_started_ns: dict[str, int] = {}
        branch_ready_ns: dict[str, int] = {}
        decode_active_started_ns: int | None = None
        decode_active_boundary_output_counts: dict[str, int] | None = None
        snapshots: list[PhysicalKvLayoutSnapshot] = []
        captured_thresholds: set[int] = set()
        thresholds = {1, 16, 64, maximum_new_tokens}
        while min(len(session.output_token_ids) for session in sessions) < maximum_new_tokens:
            _check_deadline(deadline, "run_concurrent_independent_sessions")
            step_started_ns = self._clock_ns()
            runtime_outputs = self._view.llm_engine.step()
            step_returned_ns = self._clock_ns()
            previously_active = set(first_token_ns)
            self._process_outputs(runtime_outputs, first_token_ns)
            for session_id in set(first_token_ns) - previously_active:
                first_decode_token_started_ns[session_id] = step_started_ns
                branch_ready_ns[session_id] = step_returned_ns
            if decode_active_started_ns is None and len(first_token_ns) == len(sessions):
                decode_active_started_ns = self._clock_ns()
                decode_active_boundary_output_counts = {
                    session.session_id: len(session.output_token_ids) for session in sessions
                }
                if on_all_branches_ready is not None:
                    on_all_branches_ready()
                    on_all_branches_ready = None
            progress = min(len(session.output_token_ids) for session in sessions)
            for threshold in sorted(thresholds):
                if progress >= threshold and threshold not in captured_thresholds:
                    snapshot = self.inspect_physical_kv_layout(
                        session_ids,
                        timeout_s=max(
                            (deadline - time.monotonic_ns()) / 1_000_000_000,
                            1e-9,
                        ),
                    )
                    snapshots.append(snapshot)
                    if self._trace_recorder is not None:
                        cause = "divergence" if threshold == 1 else f"suffix_{threshold}"
                        self._trace_recorder.observe_layout(
                            snapshot,
                            session_id=sessions[0].session_id,
                            branch_group_id=None,
                            cause=f"independent_{cause}",
                        )
                    captured_thresholds.add(threshold)
        prefix_sets: list[set[int]] = []
        for session in sessions:
            assert session.runtime_request_id is not None
            request_id = session.runtime_request_id
            if self._observer.initial_cached_tokens.get(
                request_id
            ) != 0 or self._observer.initial_cached_block_ids.get(request_id, ()):
                raise SnapshotConsistencyError(
                    "independent baseline observed a hidden prefix-cache hit",
                    operation="run_concurrent_independent_sessions",
                    session_id=session.session_id,
                )
            groups = self._observer.last_request_blocks.get(request_id)
            if not groups:
                raise SnapshotConsistencyError(
                    "independent request has no live block table at target boundary",
                    operation="run_concurrent_independent_sessions",
                    session_id=session.session_id,
                )
            prefix_count = math.ceil(len(session.token_ids) / self._view.block_size_tokens)
            prefix_sets.append(set(groups[0][:prefix_count]))
        if any(
            left & right
            for index, left in enumerate(prefix_sets)
            for right in prefix_sets[index + 1 :]
        ):
            raise SnapshotConsistencyError(
                "independent requests share physical prefix blocks",
                operation="run_concurrent_independent_sessions",
            )
        ended_ns = self._clock_ns()
        elapsed_ns = ended_ns - started_ns
        outputs = {
            session.session_id: session.output_token_ids[:maximum_new_tokens]
            for session in sessions
        }
        admitted: dict[str, int] = {}
        decode_started: dict[str, int] = {}
        ready: dict[str, int] = {}
        first = {
            session.session_id: first_token_ns[session.session_id] - started_ns
            for session in sessions
        }
        for session in sessions:
            assert session.runtime_request_id is not None
            admitted_at = self._observer.request_admitted_ns.get(session.runtime_request_id)
            decode_started_at = first_decode_token_started_ns.get(session.session_id)
            ready_at = branch_ready_ns.get(session.session_id)
            if admitted_at is None or decode_started_at is None or ready_at is None:
                raise SnapshotConsistencyError(
                    "vLLM did not expose complete admission/readiness/decode-start evidence",
                    operation="run_concurrent_independent_sessions",
                    session_id=session.session_id,
                )
            admitted[session.session_id] = admitted_at - started_ns
            decode_started[session.session_id] = max(decode_started_at, admitted_at) - started_ns
            ready[session.session_id] = ready_at - started_ns
        if self._trace_recorder is not None:
            for session in sessions:
                self._trace_recorder.record_branch(
                    BranchOperationType.BRANCH_READY,
                    session_id=session.session_id,
                    branch_id=session.session_id,
                    duration_ns=ready[session.session_id],
                    private_suffix=True,
                    fanout=len(sessions),
                    attributes={
                        "independent_prefill": True,
                        "request_admitted_latency_ns": admitted[session.session_id],
                        "first_decode_token_started_latency_ns": decode_started[session.session_id],
                        "first_token_latency_ns": first[session.session_id],
                    },
                )
        total = sum(len(tokens) for tokens in outputs.values())
        if decode_active_started_ns is None or decode_active_boundary_output_counts is None:
            raise SnapshotConsistencyError(
                "independent decode never reached the all-branches-active boundary",
                operation="run_concurrent_independent_sessions",
            )
        decode_active_elapsed_ns = ended_ns - decode_active_started_ns
        decode_active_tokens = sum(
            len(tokens) - decode_active_boundary_output_counts[session_id]
            for session_id, tokens in outputs.items()
        )
        return ConcurrentDecodeResult(
            branchpoint_id=self._branchpoint_id("independent", session_ids),
            branch_output_token_ids=outputs,
            request_admitted_latency_ns=admitted,
            first_decode_token_started_latency_ns=decode_started,
            branch_ready_latency_ns=ready,
            first_token_latency_ns=first,
            elapsed_ns=elapsed_ns,
            decode_active_elapsed_ns=decode_active_elapsed_ns,
            decode_active_boundary_output_counts=decode_active_boundary_output_counts,
            decode_active_tokens=decode_active_tokens,
            total_output_tokens=total,
            throughput_tokens_per_second=total / (elapsed_ns / 1_000_000_000),
            completed_branch_ids=session_ids,
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
        deadline = _require_timeout(timeout_s)
        self._require_open("inspect_block_release_evidence")
        if not runtime_block_ids or len(runtime_block_ids) != len(set(runtime_block_ids)):
            raise ValueError("runtime_block_ids must be non-empty and unique")
        evidence: list[PhysicalKvBlockReleaseEvidence] = []
        for runtime_block_id in runtime_block_ids:
            _, block_idx = _parse_runtime_block_id(runtime_block_id)
            block = self._view.pool_block(block_idx)
            evidence.append(
                PhysicalKvBlockReleaseEvidence(
                    runtime_block_id=runtime_block_id,
                    block_index=block_idx,
                    allocation_epoch=self._observer.block_epochs.get(block_idx, 0),
                    native_refcount=block.ref_cnt,
                    block_hash_present=block.block_hash is not None,
                    # In vLLM 0.23.0 every non-null ref_cnt==0 block is in
                    # BlockPool's free queue (possibly as a cached eviction candidate).
                    allocator_available=block.ref_cnt == 0 and not block.is_null,
                    is_null=block.is_null,
                )
            )
        result = PhysicalKvReleaseEvidence(
            runtime="vllm",
            runtime_version=VLLM_LIVE_RUNTIME_VERSION,
            device=self._view.device,
            observed_at_monotonic_ns=self._clock_ns(),
            requested_block_ids=runtime_block_ids,
            blocks=tuple(evidence),
            pool_free_block_count=self._view.pool_free_block_count,
            pool_usable_block_count=self._view.pool_usable_block_count,
        )
        _check_deadline(deadline, "inspect_block_release_evidence")
        return result

    def report_block_refcounts_or_equivalent(
        self, session_ids: tuple[str, ...], *, timeout_s: float
    ) -> dict[str, int]:
        snapshot = self.inspect_physical_kv_layout(session_ids, timeout_s=timeout_s)
        return {block.runtime_block_id: block.refcount for block in snapshot.blocks}

    @staticmethod
    def _branchpoint_id(root_id: str, branch_ids: tuple[str, ...]) -> str:
        digest = sha256((root_id + "\0" + "\0".join(branch_ids)).encode()).hexdigest()[:24]
        return f"vllm-branchpoint:{digest}"

    def capture_branchpoint(
        self,
        root_reference_id: str,
        branch_session_ids: tuple[str, ...],
        *,
        timeout_s: float,
    ) -> LiveBranchPoint:
        deadline = _require_timeout(timeout_s)
        try:
            root = self._roots[root_reference_id]
        except KeyError as error:
            raise SessionNotFoundError(
                "unknown shared-root reference", operation="capture_branchpoint"
            ) from error
        if not branch_session_ids or len(branch_session_ids) != len(set(branch_session_ids)):
            raise ValueError("branch_session_ids must be non-empty and unique")
        for branch_id in branch_session_ids:
            branch = self._session(branch_id, "capture_branchpoint")
            if branch.root_reference_id != root_reference_id:
                raise SessionStateError(
                    "branch ancestry does not match the requested root",
                    operation="capture_branchpoint",
                    session_id=branch_id,
                )
        result = LiveBranchPoint(
            branchpoint_id=self._branchpoint_id(root_reference_id, branch_session_ids),
            root_reference=root,
            scope=self.capability_matrix.scope,
            model=self._identity,
            parent_session_id=root.source_session_id,
            branch_session_ids=branch_session_ids,
            prefix_token_count=root.prefix_token_count,
            policy_epoch=self._identity.policy_epoch,
            captured_at_monotonic_ns=self._clock_ns(),
        )
        _check_deadline(deadline, "capture_branchpoint")
        return result

    def emit_branch_workload_trace(self) -> tuple[BranchWorkloadEventV1, ...]:
        if self._trace_recorder is None:
            raise UnsupportedCapabilityError(
                "live GPU trace recording is disabled for this adapter",
                operation="emit_BranchWorkloadTrace",
            )
        return self._trace_recorder.emit_branch_workload_trace()

    def emit_state_operation_trace(self) -> tuple[StateOperationEventV1, ...]:
        if self._trace_recorder is None:
            raise UnsupportedCapabilityError(
                "live GPU trace recording is disabled for this adapter",
                operation="emit_StateOperationTrace",
            )
        return self._trace_recorder.emit_state_operation_trace()

    def inspect_trace_self_overhead(self) -> dict[str, int | str]:
        """Expose bounded, pointer-free recorder work, not end-to-end perturbation."""

        if self._trace_recorder is None:
            raise UnsupportedCapabilityError(
                "live GPU trace recording is disabled for this adapter",
                operation="inspect_trace_self_overhead",
            )
        return self._trace_recorder.overhead_counters()

    def _trace_lifecycle_layout(self, *, session_id: str, cause: str, deadline_ns: int) -> None:
        if self._trace_recorder is None:
            return
        active = tuple(
            session.session_id
            for session in self._sessions.values()
            if session.phase is not LiveSessionPhase.DESTROYED
        )
        observed = active or (session_id,)
        snapshot = self.inspect_physical_kv_layout(
            observed,
            timeout_s=max((deadline_ns - time.monotonic_ns()) / 1_000_000_000, 1e-9),
        )
        session = self._sessions[session_id]
        self._trace_recorder.observe_layout(
            snapshot,
            session_id=session_id,
            branch_group_id=session.root_reference_id,
            cause=cause,
        )

    def _active_root_ids_for_sibling(self, session: _LiveSession) -> tuple[int, ...]:
        if (
            not session.runtime_request_id
            or self._view.request_object(session.runtime_request_id) is None
        ):
            return ()
        root = self._roots.get(session.root_reference_id or "")
        if root is None:
            return ()
        count = len(root.block_ids)
        return tuple(
            block.block_id
            for block in self._view.request_blocks(session.runtime_request_id)[0][:count]
        )

    def _recorded_request_block_ids(self, session: _LiveSession) -> tuple[str, ...]:
        if not session.runtime_request_id:
            return ()
        groups = self._observer.last_request_blocks.get(session.runtime_request_id, ())
        return tuple(
            _runtime_block_id(group_idx, block_idx)
            for group_idx, group in enumerate(groups)
            for block_idx in group
        )

    @staticmethod
    def _require_native_release(
        evidence: PhysicalKvReleaseEvidence,
        *,
        operation: str,
        session_id: str,
        require_hash_cleared: bool,
    ) -> None:
        unreleased = [
            block.runtime_block_id
            for block in evidence.blocks
            if block.native_refcount != 0
            or not block.allocator_available
            or (require_hash_cleared and block.block_hash_present)
        ]
        if unreleased:
            raise SnapshotConsistencyError(
                "native vLLM BlockPool did not release exact KV blocks: " + ", ".join(unreleased),
                operation=operation,
                session_id=session_id,
            )

    def destroy_branch(self, branch_session_id: str, *, timeout_s: float) -> None:
        deadline = _require_timeout(timeout_s)
        branch = self._session(branch_session_id, "destroy_branch")
        if branch.branch_id is None:
            raise SessionStateError(
                "session is not a branch",
                operation="destroy_branch",
                session_id=branch_session_id,
            )
        if branch.phase is LiveSessionPhase.DESTROYED:
            return
        siblings = [
            session
            for session in self._sessions.values()
            if session.session_id != branch_session_id
            and session.root_reference_id == branch.root_reference_id
            and session.phase is not LiveSessionPhase.DESTROYED
        ]
        before = {
            sibling.session_id: self._active_root_ids_for_sibling(sibling) for sibling in siblings
        }
        root = self._roots.get(branch.root_reference_id or "")
        root_ids = set(root.block_ids if root is not None else ())
        private_block_ids = tuple(
            block_id
            for block_id in self._recorded_request_block_ids(branch)
            if block_id not in root_ids
        )
        if (
            branch.submitted
            and branch.runtime_request_id
            and self._view.request_object(branch.runtime_request_id) is not None
        ):
            self._view.llm_engine.abort_request([branch.session_id])
        branch.phase = LiveSessionPhase.DESTROYED
        after = {
            sibling.session_id: self._active_root_ids_for_sibling(sibling) for sibling in siblings
        }
        if before != after:
            raise SnapshotConsistencyError(
                "destroying one vLLM branch changed a sibling shared-root table",
                operation="destroy_branch",
                session_id=branch_session_id,
            )
        release_evidence = None
        if private_block_ids:
            release_evidence = self.inspect_block_release_evidence(
                private_block_ids,
                timeout_s=max((deadline - time.monotonic_ns()) / 1_000_000_000, 1e-9),
            )
            self._require_native_release(
                release_evidence,
                operation="destroy_branch",
                session_id=branch_session_id,
                require_hash_cleared=False,
            )
            self._released_experiment_block_ids.update(private_block_ids)
        if self._trace_recorder is not None:
            self._trace_recorder.record_branch(
                BranchOperationType.BRANCH_PRUNE,
                session_id=branch_session_id,
                branch_group_id=branch.root_reference_id,
                branch_id=branch.branch_id,
                parent_branch_id=branch.parent_session_id,
                shared_root=True,
                private_suffix=True,
                attributes={"runtime_request_released": True},
            )
            if release_evidence is not None:
                self._trace_recorder.observe_release_evidence(
                    release_evidence,
                    session_id=branch_session_id,
                    branch_group_id=branch.root_reference_id,
                    cause="branch_release_native_block_pool",
                )
            self._trace_lifecycle_layout(
                session_id=branch_session_id,
                cause="branch_release",
                deadline_ns=deadline,
            )
        _check_deadline(deadline, "destroy_branch")

    def destroy_session(self, session_id: str, *, timeout_s: float) -> None:
        deadline = _require_timeout(timeout_s)
        session = self._session(session_id, "destroy_session")
        if session.branch_id is not None:
            self.destroy_branch(session_id, timeout_s=timeout_s)
            return
        if session.phase is LiveSessionPhase.DESTROYED:
            return
        live_children = [
            child
            for child in self._sessions.values()
            if child.parent_session_id == session_id
            and child.phase is not LiveSessionPhase.DESTROYED
        ]
        if live_children:
            raise SessionStateError(
                "shared root cannot be destroyed while branches remain live",
                operation="destroy_session",
                session_id=session_id,
            )
        if (
            session.submitted
            and session.runtime_request_id
            and self._view.request_object(session.runtime_request_id) is not None
        ):
            self._view.llm_engine.abort_request([session.session_id])
        roots = [
            root_id for root_id, root in self._roots.items() if root.source_session_id == session_id
        ]
        root_block_ids = tuple(
            block_id for root_id in roots for block_id in self._roots[root_id].block_ids
        )
        if roots and not self._view.manager.reset_prefix_cache():
            raise SnapshotConsistencyError(
                "vLLM refused final shared-root cache release",
                operation="destroy_session",
                session_id=session_id,
            )
        release_evidence = None
        release_ids = tuple(sorted(set(root_block_ids) | self._released_experiment_block_ids))
        if roots and release_ids:
            release_evidence = self.inspect_block_release_evidence(
                release_ids,
                timeout_s=max((deadline - time.monotonic_ns()) / 1_000_000_000, 1e-9),
            )
            self._require_native_release(
                release_evidence,
                operation="destroy_session",
                session_id=session_id,
                require_hash_cleared=True,
            )
            if release_evidence.pool_free_block_count != release_evidence.pool_usable_block_count:
                raise SnapshotConsistencyError(
                    "final prefix-cache reset did not recover the complete usable KV block pool",
                    operation="destroy_session",
                    session_id=session_id,
                )
        for root_id in roots:
            del self._roots[root_id]
        session.phase = LiveSessionPhase.DESTROYED
        if self._trace_recorder is not None:
            if release_evidence is not None:
                self._trace_recorder.observe_release_evidence(
                    release_evidence,
                    session_id=session_id,
                    branch_group_id=roots[0] if roots else None,
                    cause="root_release_native_block_pool_reset",
                )
            self._trace_lifecycle_layout(
                session_id=session_id,
                cause="root_release" if roots else "session_release",
                deadline_ns=deadline,
            )
            self._trace_recorder.record_branch(
                BranchOperationType.STATE_FREE,
                session_id=session_id,
                physical_state_id=roots[0] if roots else None,
                shared_root=bool(roots),
                attributes={"prefix_cache_reset": bool(roots)},
            )
        _check_deadline(deadline, "destroy_session")

    def cleanup_runtime(self, *, timeout_s: float) -> None:
        if self._closed:
            return
        deadline = _require_timeout(timeout_s)
        failures: list[BaseException] = []
        for session in tuple(self._sessions.values()):
            if session.phase is LiveSessionPhase.DESTROYED or not session.submitted:
                session.phase = LiveSessionPhase.DESTROYED
                continue
            try:
                self._view.llm_engine.abort_request([session.session_id])
            except BaseException as error:
                failures.append(error)
            session.phase = LiveSessionPhase.DESTROYED
        # Experiment 004 stages fresh runtime incarnations directly through
        # the exact vLLM scheduler API. A failure after atomic admission can
        # therefore leave requests that are deliberately absent from the
        # adapter's logical-session archive. No engine step is permitted once
        # final cleanup begins, so detach and free any such request before the
        # prefix-cache reset and engine shutdown.
        scheduler = self._view.scheduler
        remaining_requests = tuple(getattr(scheduler, "requests", {}).values())
        for request in remaining_requests:
            try:
                for queue_name in ("waiting", "skipped_waiting"):
                    queue = getattr(scheduler, queue_name, None)
                    remover = getattr(queue, "remove_requests", None)
                    if callable(remover):
                        remover([request])
                running = getattr(scheduler, "running", None)
                if running is not None and request in running:
                    running.remove(request)
                self._view.manager.free(request)
                getattr(scheduler, "requests", {}).pop(_request_id(request), None)
            except BaseException as error:
                failures.append(error)
        try:
            if not self._view.manager.reset_prefix_cache():
                failures.append(RuntimeError("vLLM refused prefix-cache cleanup"))
        except BaseException as error:
            failures.append(error)
        try:
            self._metadata_recorder.close()
        except BaseException as error:
            failures.append(error)
        self._view.manager.allocate_slots = self._original_allocate_slots
        self._view.manager.free = self._original_free
        shutdown = getattr(self._view.engine_core_client, "shutdown", None)
        if callable(shutdown):
            try:
                shutdown()
            except BaseException as error:
                failures.append(error)
        self._roots.clear()
        self._sessions.clear()
        self._released_experiment_block_ids.clear()
        self._runtime_to_session.clear()
        self._closed = True
        # Bound methods retain their manager and the internal view retains the
        # CUDA model runner/KV tensors.  Drop every runtime-ephemeral reference
        # so a subsequent comparison engine cannot inherit HBM from this one.
        del self._original_allocate_slots
        del self._original_free
        del self._engine
        del self._view
        _check_deadline(deadline, "cleanup_runtime")
        if failures:
            raise RuntimeError(
                "vLLM cleanup encountered failures: "
                + ", ".join(type(error).__name__ for error in failures)
            )


__all__ = [
    "VLLM_LIVE_ADAPTER_VERSION",
    "VLLM_LIVE_INTERNAL_INTERFACES",
    "VLLM_LIVE_RUNTIME_VERSION",
    "VllmLiveConfigurationError",
    "VllmLiveInternalView",
    "VllmLiveStateAdapter",
]
