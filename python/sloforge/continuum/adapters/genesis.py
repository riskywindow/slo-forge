"""Continuum binding for a validated Genesis-generated runtime bundle.

The current Genesis runtime publishes deterministic bounded streaming and
cancellation, but not a public live-state export.  This binding validates and
loads real generated bundles through Genesis's public loader while failing closed
for active-state migration until a generated runtime emits that contract.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Final, cast

from sloforge.continuum.adapters.external import (
    ADAPTER_VERSION,
    AdapterProbe,
    CapabilityName,
    IntegrationStatus,
)
from sloforge.continuum.adapters.sdk import ResourceLimitError, UnsupportedCapabilityError
from sloforge.genesis.runtime import BaselineStreamingRuntime, EventKind, load_generated_runtime

GENESIS_RUNTIME_SCHEMA_VERSION: Final = "1.0.0"
GENESIS_EVIDENCE: Final = (
    "python/sloforge/genesis/runtime/generator.py:GENERATED_RUNTIME_SCHEMA_VERSION",
    "python/sloforge/genesis/runtime/generator.py:load_generated_runtime",
    "python/sloforge/genesis/runtime/core.py:BaselineStreamingRuntime",
)
MAX_CONFIG_BYTES: Final = 1024 * 1024
MAX_SMOKE_TOKENS: Final = 128


@dataclass(frozen=True, slots=True)
class GenesisRuntimeDescriptor:
    config_path: Path
    schema_version: str
    runtime_id: str
    package_hash: str
    generation_seed: int
    state_allocator_layout: str
    page_bytes: int

    def __post_init__(self) -> None:
        if self.schema_version != GENESIS_RUNTIME_SCHEMA_VERSION:
            raise ValueError("unsupported Genesis generated-runtime schema version")
        for name, digest in (("runtime_id", self.runtime_id), ("package_hash", self.package_hash)):
            if len(digest) != 64 or any(
                character not in "0123456789abcdef" for character in digest
            ):
                raise ValueError(f"Genesis {name} must be a lowercase sha256 digest")
        if not 0 <= self.generation_seed < 1 << 64:
            raise ValueError("Genesis generation seed must be an unsigned 64-bit integer")
        if self.state_allocator_layout not in {"contiguous", "paged"}:
            raise ValueError("Genesis state allocator layout is unsupported")
        if self.page_bytes <= 0 or self.page_bytes & (self.page_bytes - 1):
            raise ValueError("Genesis state allocator page size must be a power of two")


@dataclass(frozen=True, slots=True)
class GenesisSmokeResult:
    runtime_id: str
    token_ids: tuple[int, ...]
    terminal_kind: str
    health_before: str
    health_after: str


def probe_genesis() -> AdapterProbe:
    return AdapterProbe(
        runtime_name="genesis",
        runtime_version=GENESIS_RUNTIME_SCHEMA_VERSION,
        adapter_version=ADAPTER_VERSION,
        status=IntegrationStatus.READY,
        capabilities=frozenset(
            {
                CapabilityName.RUNTIME_INSPECTION,
                CapabilityName.BOUNDED_STREAMING,
                CapabilityName.CANCELLATION,
                CapabilityName.GENERATED_RUNTIME_LOADING,
            }
        ),
        missing_requirements=(),
        evidence=GENESIS_EVIDENCE,
        exercised=True,
    )


def inspect_genesis_runtime(config_path: Path) -> GenesisRuntimeDescriptor:
    if config_path.is_symlink() or not config_path.is_file():
        raise ValueError("Genesis runtime configuration must be a regular non-symlink file")
    if config_path.stat().st_size > MAX_CONFIG_BYTES:
        raise ResourceLimitError(
            "Genesis runtime configuration exceeds the bounded document size",
            operation="inspect_genesis_runtime",
        )
    value = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError("Genesis runtime configuration must be a JSON object")
    config = cast(dict[str, object], value)
    required = {
        "schema_version",
        "runtime_id",
        "reference_package_root",
        "package_hash",
        "inspection_hash",
        "genome_hash",
        "generation_seed",
        "limits",
    }
    optional = {"policy_bytecode_path", "policy_bytecode_sha256", "state_allocator"}
    if not required.issubset(config) or not set(config).issubset(required | optional):
        raise ValueError("Genesis runtime configuration contains missing or unknown fields")
    allocator = config.get("state_allocator")
    if allocator is None:
        allocator = {"layout": "contiguous", "page_bytes": 64}
    if not isinstance(allocator, dict):
        raise TypeError("Genesis state allocator must be a JSON object")
    layout = allocator.get("layout")
    page_bytes = allocator.get("page_bytes")
    if (
        not isinstance(layout, str)
        or not isinstance(page_bytes, int)
        or isinstance(page_bytes, bool)
    ):
        raise TypeError("Genesis state allocator layout and page size are invalid")
    schema_version = config["schema_version"]
    runtime_id = config["runtime_id"]
    package_hash = config["package_hash"]
    generation_seed = config["generation_seed"]
    if (
        not isinstance(schema_version, str)
        or not isinstance(runtime_id, str)
        or not isinstance(package_hash, str)
        or not isinstance(generation_seed, int)
        or isinstance(generation_seed, bool)
    ):
        raise TypeError("Genesis runtime configuration identity fields are invalid")
    return GenesisRuntimeDescriptor(
        config_path=config_path.resolve(strict=True),
        schema_version=schema_version,
        runtime_id=runtime_id,
        package_hash=package_hash,
        generation_seed=generation_seed,
        state_allocator_layout=layout,
        page_bytes=page_bytes,
    )


class GenesisRuntimeBinding:
    def __init__(self, descriptor: GenesisRuntimeDescriptor) -> None:
        self.descriptor = descriptor
        self.probe = probe_genesis()

    @classmethod
    def from_config(cls, config_path: Path) -> GenesisRuntimeBinding:
        return cls(inspect_genesis_runtime(config_path))

    def load(self) -> BaselineStreamingRuntime:
        self.probe.require_capability(CapabilityName.GENERATED_RUNTIME_LOADING)
        return load_generated_runtime(
            self.descriptor.config_path,
            seed=self.descriptor.generation_seed,
            allow_untrusted_in_process=True,
        )

    def run_cpu_smoke(
        self,
        *,
        request_id: str,
        text: str,
        maximum_new_tokens: int,
        seed: int,
        timeout_seconds: float,
    ) -> GenesisSmokeResult:
        self.probe.require_capability(CapabilityName.BOUNDED_STREAMING)
        if not 1 <= maximum_new_tokens <= MAX_SMOKE_TOKENS:
            raise ValueError(f"Genesis smoke token count must be in 1..{MAX_SMOKE_TOKENS}")
        runtime = self.load()
        health_after = "unknown"
        token_ids: list[int] = []
        terminal = "missing"
        runtime.start()
        try:
            health_before = str(runtime.health()["status"])
            handle = runtime.submit_text(
                request_id=request_id,
                text=text,
                maximum_new_tokens=maximum_new_tokens,
                seed=seed,
                timeout_seconds=timeout_seconds,
            )
            for event in handle.events(timeout_seconds):
                if event.kind is EventKind.TOKEN:
                    if event.token_id is None:
                        raise RuntimeError("Genesis published a token event without a token ID")
                    token_ids.append(event.token_id)
                else:
                    terminal = event.kind.value
        finally:
            runtime.shutdown(timeout_seconds)
            health_after = str(runtime.health()["status"])
        if terminal == "missing":
            raise RuntimeError("Genesis stream ended without an explicit terminal event")
        return GenesisSmokeResult(
            runtime_id=self.descriptor.runtime_id,
            token_ids=tuple(token_ids),
            terminal_kind=terminal,
            health_before=health_before,
            health_after=health_after,
        )

    def require_portable_execution_state_export(self) -> None:
        self.probe.require_ready(operation="portable_execution_state_export")
        raise UnsupportedCapabilityError(
            (
                "Genesis generated-runtime schema 1.0.0 provides bounded streaming but "
                "does not publish an active execution-state export contract"
            ),
            operation="portable_execution_state_export",
        )


__all__ = [
    "GENESIS_EVIDENCE",
    "GENESIS_RUNTIME_SCHEMA_VERSION",
    "GenesisRuntimeBinding",
    "GenesisRuntimeDescriptor",
    "GenesisSmokeResult",
    "inspect_genesis_runtime",
    "probe_genesis",
]
