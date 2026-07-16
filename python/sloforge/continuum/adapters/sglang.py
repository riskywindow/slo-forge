"""Version-gated SGLang prefill/decode disaggregation integration.

SGLang's public server arguments expose disaggregation roles and transport choice.
They do not expose a runtime-independent state capsule, so this module creates only
a validated launch configuration and never claims active-state import/export.
"""

from __future__ import annotations

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

SGLANG_VERSION_POLICY: Final = VersionPolicy(
    minimum_inclusive=SemanticVersion(0, 4, 6),
    maximum_exclusive=SemanticVersion(0, 6, 0),
)
SGLANG_REQUIREMENTS: Final = (
    "sglang.srt.server_args:ServerArgs",
    "sglang.srt.server_args:ServerArgs.disaggregation_mode",
    "sglang.srt.server_args:ServerArgs.disaggregation_transfer_backend",
    "sglang.srt.server_args:ServerArgs.page_size",
    "sglang.srt.server_args:ServerArgs.tp_size",
    "sglang.srt.server_args:ServerArgs.pp_size",
)
SGLANG_EVIDENCE: Final = (
    "https://docs.sglang.ai/backend/pd_disaggregation.html",
    "https://github.com/sgl-project/sglang/blob/v0.5.12/python/sglang/srt/server_args.py",
    "https://github.com/sgl-project/sglang/tree/v0.5.12/python/sglang/srt/disaggregation",
)


@dataclass(frozen=True, slots=True)
class SglangPdConfiguration:
    role: Literal["prefill", "decode"]
    transfer_backend: Literal["nixl", "mooncake"]
    page_size_tokens: int
    tensor_parallel_degree: int
    pipeline_parallel_degree: int
    random_seed: int

    def __post_init__(self) -> None:
        if not 1 <= self.page_size_tokens <= 1_048_576:
            raise ValueError("SGLang page size must be in 1..1048576 tokens")
        if not 1 <= self.tensor_parallel_degree <= 1024:
            raise ValueError("SGLang tensor parallel degree must be in 1..1024")
        if not 1 <= self.pipeline_parallel_degree <= 1024:
            raise ValueError("SGLang pipeline parallel degree must be in 1..1024")
        if not 0 <= self.random_seed < 1 << 64:
            raise ValueError("SGLang seed must be an unsigned 64-bit integer")

    def to_launch_arguments(self) -> tuple[str, ...]:
        return (
            "--disaggregation-mode",
            self.role,
            "--disaggregation-transfer-backend",
            self.transfer_backend,
            "--page-size",
            str(self.page_size_tokens),
            "--tp-size",
            str(self.tensor_parallel_degree),
            "--pp-size",
            str(self.pipeline_parallel_degree),
            "--random-seed",
            str(self.random_seed),
        )


def probe_sglang(view: RuntimePackageView | None = None) -> AdapterProbe:
    discovered = view
    if discovered is None:
        discovered = discover_installed_package(
            distribution_name="sglang",
            import_name="sglang",
            required_symbols=SGLANG_REQUIREMENTS,
        )
    return evaluate_package(
        runtime_name="sglang",
        view=discovered,
        policy=SGLANG_VERSION_POLICY,
        requirements=SGLANG_REQUIREMENTS,
        capabilities=frozenset(
            {
                CapabilityName.RUNTIME_INSPECTION,
                CapabilityName.PD_DISAGGREGATION_CONFIGURATION,
                CapabilityName.NIXL_TRANSPORT_CONFIGURATION,
                CapabilityName.MOONCAKE_TRANSPORT_CONFIGURATION,
            }
        ),
        evidence=SGLANG_EVIDENCE,
    )


class SglangRuntimeBinding:
    def __init__(self, probe: AdapterProbe | None = None) -> None:
        self.probe = probe or probe_sglang()

    def build_pd_launch_arguments(self, config: SglangPdConfiguration) -> tuple[str, ...]:
        self.probe.require_capability(CapabilityName.PD_DISAGGREGATION_CONFIGURATION)
        transport_capability = (
            CapabilityName.NIXL_TRANSPORT_CONFIGURATION
            if config.transfer_backend == "nixl"
            else CapabilityName.MOONCAKE_TRANSPORT_CONFIGURATION
        )
        self.probe.require_capability(transport_capability)
        return config.to_launch_arguments()

    def require_portable_execution_state_export(self) -> None:
        self.probe.require_ready(operation="portable_execution_state_export")
        raise UnsupportedCapabilityError(
            (
                "SGLang PD disaggregation exposes runtime-native transfer configuration, "
                "not a complete Continuum logical-state and ownership export"
            ),
            operation="portable_execution_state_export",
        )


__all__ = [
    "SGLANG_EVIDENCE",
    "SGLANG_REQUIREMENTS",
    "SGLANG_VERSION_POLICY",
    "SglangPdConfiguration",
    "SglangRuntimeBinding",
    "probe_sglang",
]
