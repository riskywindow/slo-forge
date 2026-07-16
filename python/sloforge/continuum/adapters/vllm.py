"""Version-gated vLLM KV-connector integration.

The upstream connector moves vLLM-owned KV/hidden-state buffers.  It does not by
itself define token ownership, logical state compatibility, or a complete portable
execution-state export.  Continuum therefore uses it only as a byte-movement hook
after independent compatibility and conversion planning.
"""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final, Literal

from sloforge.continuum.adapters.external import (
    AdapterProbe,
    CapabilityName,
    RuntimePackageView,
    SemanticVersion,
    VersionPolicy,
    discover_installed_package,
    evaluate_package,
)
from sloforge.continuum.adapters.sdk import UnsupportedCapabilityError

VLLM_VERSION_POLICY: Final = VersionPolicy(
    minimum_inclusive=SemanticVersion(0, 9, 0),
    maximum_exclusive=SemanticVersion(0, 24, 0),
)
VLLM_REQUIREMENTS: Final = (
    "vllm.config.kv_transfer:KVTransferConfig",
    "vllm.distributed.kv_transfer.kv_connector.v1.base:KVConnectorBase_V1",
)
VLLM_EVIDENCE: Final = (
    "https://docs.vllm.ai/en/latest/api/vllm/config/kv_transfer.html",
    "https://docs.vllm.ai/en/latest/api/vllm/distributed/kv_transfer/kv_connector/v1/base.html",
    "https://github.com/vllm-project/vllm/blob/v0.23.0/vllm/config/kv_transfer.py",
    "https://github.com/vllm-project/vllm/blob/v0.23.0/vllm/distributed/kv_transfer/kv_connector/v1/base.py",
)


@dataclass(frozen=True, slots=True)
class VllmKvTransferRequest:
    connector: str
    role: Literal["kv_producer", "kv_consumer", "kv_both"]
    engine_id: str
    buffer_device: Literal["cpu", "cuda", "xpu"]
    extra_config: Mapping[str, object]

    def __post_init__(self) -> None:
        if not self.connector or len(self.connector) > 256:
            raise ValueError("vLLM connector name must contain 1..256 characters")
        if not self.engine_id or len(self.engine_id) > 256:
            raise ValueError("vLLM engine ID must contain 1..256 characters")
        if len(self.extra_config) > 128 or not all(
            isinstance(key, str) and 0 < len(key) <= 256 for key in self.extra_config
        ):
            raise ValueError("vLLM connector extra configuration is not bounded")


def probe_vllm(view: RuntimePackageView | None = None) -> AdapterProbe:
    discovered = view
    if discovered is None:
        discovered = discover_installed_package(
            distribution_name="vllm",
            import_name="vllm",
            required_symbols=VLLM_REQUIREMENTS,
        )
    return evaluate_package(
        runtime_name="vllm",
        view=discovered,
        policy=VLLM_VERSION_POLICY,
        requirements=VLLM_REQUIREMENTS,
        capabilities=frozenset(
            {
                CapabilityName.RUNTIME_INSPECTION,
                CapabilityName.KV_TRANSFER_CONFIGURATION,
                CapabilityName.KV_CONNECTOR_V1,
            }
        ),
        evidence=VLLM_EVIDENCE,
    )


class VllmRuntimeBinding:
    """Isolate the explicitly unstable vLLM connector surface from core IR."""

    def __init__(self, probe: AdapterProbe | None = None) -> None:
        self.probe = probe or probe_vllm()

    def build_kv_transfer_config(self, request: VllmKvTransferRequest) -> object:
        self.probe.require_capability(CapabilityName.KV_TRANSFER_CONFIGURATION)
        module = importlib.import_module("vllm.config.kv_transfer")
        config_type = module.KVTransferConfig
        config = config_type(
            kv_connector=request.connector,
            engine_id=request.engine_id,
            kv_buffer_device=request.buffer_device,
            kv_role=request.role,
            kv_connector_extra_config=dict(request.extra_config),
        )
        if not bool(getattr(config, "is_kv_transfer_instance", False)):
            raise ValueError("vLLM rejected the requested KV transfer role or connector")
        return config

    def validate_connector_class(self, connector_type: object) -> None:
        self.probe.require_capability(CapabilityName.KV_CONNECTOR_V1)
        module = importlib.import_module("vllm.distributed.kv_transfer.kv_connector.v1.base")
        base = module.KVConnectorBase_V1
        if not isinstance(connector_type, type) or not issubclass(connector_type, base):
            raise TypeError("connector must derive from vLLM KVConnectorBase_V1")

    def require_portable_execution_state_export(self) -> None:
        self.probe.require_ready(operation="portable_execution_state_export")
        raise UnsupportedCapabilityError(
            (
                "vLLM KVConnectorBase_V1 exposes runtime-native cache movement, not a "
                "complete Continuum logical-state and ownership export"
            ),
            operation="portable_execution_state_export",
        )


__all__ = [
    "VLLM_EVIDENCE",
    "VLLM_REQUIREMENTS",
    "VLLM_VERSION_POLICY",
    "VllmKvTransferRequest",
    "VllmRuntimeBinding",
    "probe_vllm",
]
