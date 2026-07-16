"""Two version-scoped adapters for the Continuum deterministic CPU runtime."""

from __future__ import annotations

from hashlib import sha256

from sloforge.continuum.adapters.sdk import (
    LayoutKind,
    ModelContract,
    RuntimeIdentity,
    RuntimeLayout,
)
from sloforge.continuum.reference.models import HybridDecoderConfig
from sloforge.continuum.reference.runtime import DeterministicHybridRuntimeAdapter


def _hash(label: str) -> str:
    return sha256(label.encode("utf-8")).hexdigest()


class ReferenceTokenMajorAdapter(DeterministicHybridRuntimeAdapter):
    """Adapter A: paged token-major, TP=4, separate K and V segments."""

    def __init__(
        self,
        *,
        page_size_tokens: int = 3,
        model: ModelContract | None = None,
        max_sessions: int = 32,
        max_dirty_events: int = 128,
    ) -> None:
        layout = RuntimeLayout(
            kind=LayoutKind.PAGED_TOKEN_MAJOR_SEPARATE_KV,
            page_size_tokens=page_size_tokens,
            tensor_parallel_degree=4,
            ordering="token-major",
            kv_packing="separate-k-v",
            alignment_bytes=64,
            simulated_devices=tuple(f"sim-source-gpu-{rank}" for rank in range(4)),
        )
        defaults = HybridDecoderConfig(layout=layout, max_dirty_events=max_dirty_events)
        config = HybridDecoderConfig(
            layout=layout,
            layers=defaults.layers,
            kv_heads=defaults.kv_heads,
            head_dimension=defaults.head_dimension,
            recurrent_width=defaults.recurrent_width,
            max_context_tokens=defaults.max_context_tokens,
            model=model or defaults.model,
            automaton_id=defaults.automaton_id,
            max_dirty_events=max_dirty_events,
        )
        super().__init__(
            identity=RuntimeIdentity(
                runtime_name="continuum-reference-token-major",
                runtime_version="1.0.0",
                adapter_version="continuum-adapter-a/1.0.0",
                build_hash=_hash("continuum-reference-token-major-build-v1"),
                dependency_versions=(("python", "3.11+"),),
                target_hardware="deterministic-cpu-with-4-simulated-gpus",
            ),
            config=config,
            max_sessions=max_sessions,
        )


class ReferenceHeadMajorAdapter(DeterministicHybridRuntimeAdapter):
    """Adapter B: paged head-major, TP=2, packed K/V segments."""

    def __init__(
        self,
        *,
        page_size_tokens: int = 5,
        model: ModelContract | None = None,
        max_sessions: int = 32,
        max_dirty_events: int = 128,
    ) -> None:
        layout = RuntimeLayout(
            kind=LayoutKind.PAGED_HEAD_MAJOR_PACKED_KV,
            page_size_tokens=page_size_tokens,
            tensor_parallel_degree=2,
            ordering="head-major",
            kv_packing="packed-k-v",
            alignment_bytes=128,
            simulated_devices=tuple(f"sim-destination-gpu-{rank}" for rank in range(2)),
        )
        defaults = HybridDecoderConfig(layout=layout, max_dirty_events=max_dirty_events)
        config = HybridDecoderConfig(
            layout=layout,
            layers=defaults.layers,
            kv_heads=defaults.kv_heads,
            head_dimension=defaults.head_dimension,
            recurrent_width=defaults.recurrent_width,
            max_context_tokens=defaults.max_context_tokens,
            model=model or defaults.model,
            automaton_id=defaults.automaton_id,
            max_dirty_events=max_dirty_events,
        )
        super().__init__(
            identity=RuntimeIdentity(
                runtime_name="continuum-reference-head-major",
                runtime_version="1.0.0",
                adapter_version="continuum-adapter-b/1.0.0",
                build_hash=_hash("continuum-reference-head-major-build-v1"),
                dependency_versions=(("python", "3.11+"),),
                target_hardware="deterministic-cpu-with-2-simulated-gpus",
            ),
            config=config,
            max_sessions=max_sessions,
        )
