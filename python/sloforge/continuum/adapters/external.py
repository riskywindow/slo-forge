"""Version-scoped discovery for optional Continuum runtime integrations.

This module never starts an engine, allocates a device, or interprets active state.
It validates a deliberately small set of public API symbols before an integration
may advertise a capability.  Static package views are accepted only for fixture
conformance and are always reported as unexercised.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import re
from dataclasses import dataclass
from enum import StrEnum
from hashlib import sha256
from typing import Final

from sloforge.continuum.adapters.sdk import (
    AdapterUnavailableError,
    UnsupportedCapabilityError,
)

ADAPTER_VERSION: Final = "1.0.0"
_VERSION_PATTERN = re.compile(
    r"^(\d+)\.(\d+)\.(\d+)(?:\.post\d+)?(?:\+[0-9A-Za-z][0-9A-Za-z._-]*)?$"
)


class IntegrationStatus(StrEnum):
    READY = "ready"
    PACKAGE_NOT_INSTALLED = "package_not_installed"
    VERSION_UNSUPPORTED = "version_unsupported"
    API_INCOMPATIBLE = "api_incompatible"
    CONFIGURATION_REQUIRED = "configuration_required"


class CapabilityName(StrEnum):
    RUNTIME_INSPECTION = "runtime_inspection"
    CPU_TENSOR_STATE = "cpu_tensor_state"
    CANONICAL_MODEL_STATE_DICT = "canonical_model_state_dict"
    RNG_STATE = "rng_state"
    BOUNDED_STREAMING = "bounded_streaming"
    CANCELLATION = "cancellation"
    GENERATED_RUNTIME_LOADING = "generated_runtime_loading"
    KV_TRANSFER_CONFIGURATION = "kv_transfer_configuration"
    KV_CONNECTOR_V1 = "kv_connector_v1"
    PD_DISAGGREGATION_CONFIGURATION = "pd_disaggregation_configuration"
    NIXL_TRANSPORT_CONFIGURATION = "nixl_transport_configuration"
    MOONCAKE_TRANSPORT_CONFIGURATION = "mooncake_transport_configuration"


@dataclass(frozen=True, slots=True, order=True)
class SemanticVersion:
    major: int
    minor: int
    patch: int

    @classmethod
    def parse(cls, value: str) -> SemanticVersion:
        match = _VERSION_PATTERN.fullmatch(value)
        if match is None:
            raise ValueError(f"version {value!r} is not a stable semantic release or local build")
        return cls(int(match.group(1)), int(match.group(2)), int(match.group(3) or 0))


@dataclass(frozen=True, slots=True)
class VersionPolicy:
    minimum_inclusive: SemanticVersion
    maximum_exclusive: SemanticVersion

    def accepts(self, value: str) -> bool:
        try:
            parsed = SemanticVersion.parse(value)
        except ValueError:
            return False
        return self.minimum_inclusive <= parsed < self.maximum_exclusive

    @property
    def description(self) -> str:
        lower = self.minimum_inclusive
        upper = self.maximum_exclusive
        return (
            f">={lower.major}.{lower.minor}.{lower.patch},"
            f"<{upper.major}.{upper.minor}.{upper.patch}"
        )


@dataclass(frozen=True, slots=True)
class RuntimePackageView:
    distribution_name: str
    import_name: str
    version: str
    available_symbols: frozenset[str]
    source: str
    discovery_errors: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.source not in {"installed", "static_fixture", "repository"}:
            raise ValueError("package-view source is not recognized")
        if not self.distribution_name or not self.import_name or not self.version:
            raise ValueError("package view identity fields cannot be empty")

    @property
    def exercised(self) -> bool:
        return self.source in {"installed", "repository"}


@dataclass(frozen=True, slots=True)
class AdapterProbe:
    runtime_name: str
    runtime_version: str | None
    adapter_version: str
    status: IntegrationStatus
    capabilities: frozenset[CapabilityName]
    missing_requirements: tuple[str, ...]
    evidence: tuple[str, ...]
    exercised: bool
    public_api_only: bool = True

    @property
    def build_hash(self) -> str:
        payload = "\0".join(
            (
                self.runtime_name,
                self.runtime_version or "unavailable",
                self.adapter_version,
                self.status.value,
                *(capability.value for capability in sorted(self.capabilities)),
                *self.missing_requirements,
                *self.evidence,
            )
        )
        return sha256(payload.encode("utf-8")).hexdigest()

    def require_ready(self, *, operation: str = "inspect") -> None:
        if self.status is IntegrationStatus.PACKAGE_NOT_INSTALLED:
            raise OptionalRuntimeNotInstalledError(
                f"optional runtime {self.runtime_name} is not installed",
                operation=operation,
                runtime_name=self.runtime_name,
            )
        if self.status is IntegrationStatus.VERSION_UNSUPPORTED:
            raise UnsupportedRuntimeVersionError(
                (
                    f"runtime {self.runtime_name} version {self.runtime_version!r} "
                    f"is outside the adapter contract"
                ),
                operation=operation,
                runtime_name=self.runtime_name,
                runtime_version=self.runtime_version,
            )
        if self.status is IntegrationStatus.API_INCOMPATIBLE:
            raise ExternalRuntimeApiError(
                f"runtime {self.runtime_name} is missing required public API symbols",
                operation=operation,
                runtime_name=self.runtime_name,
                missing_requirements=self.missing_requirements,
            )
        if self.status is IntegrationStatus.CONFIGURATION_REQUIRED:
            raise ExternalRuntimeConfigurationError(
                f"runtime {self.runtime_name} requires a validated runtime configuration",
                operation=operation,
                runtime_name=self.runtime_name,
            )

    def require_capability(self, capability: CapabilityName) -> None:
        self.require_ready(operation=capability.value)
        if capability not in self.capabilities:
            raise UnsupportedCapabilityError(
                (
                    f"runtime {self.runtime_name} public APIs do not expose "
                    f"{capability.value} through this adapter version"
                ),
                operation=capability.value,
            )


class OptionalRuntimeNotInstalledError(AdapterUnavailableError):
    code = "optional_runtime_not_installed"

    def __init__(self, message: str, *, operation: str, runtime_name: str) -> None:
        super().__init__(message, operation=operation)
        self.runtime_name = runtime_name


class UnsupportedRuntimeVersionError(AdapterUnavailableError):
    code = "unsupported_runtime_version"

    def __init__(
        self,
        message: str,
        *,
        operation: str,
        runtime_name: str,
        runtime_version: str | None,
    ) -> None:
        super().__init__(message, operation=operation)
        self.runtime_name = runtime_name
        self.runtime_version = runtime_version


class ExternalRuntimeApiError(AdapterUnavailableError):
    code = "external_runtime_api_incompatible"

    def __init__(
        self,
        message: str,
        *,
        operation: str,
        runtime_name: str,
        missing_requirements: tuple[str, ...],
    ) -> None:
        super().__init__(message, operation=operation)
        self.runtime_name = runtime_name
        self.missing_requirements = missing_requirements


class ExternalRuntimeConfigurationError(AdapterUnavailableError):
    code = "external_runtime_configuration_required"

    def __init__(self, message: str, *, operation: str, runtime_name: str) -> None:
        super().__init__(message, operation=operation)
        self.runtime_name = runtime_name


def discover_installed_package(
    *,
    distribution_name: str,
    import_name: str,
    required_symbols: tuple[str, ...],
) -> RuntimePackageView | None:
    """Inspect a package's metadata and named public symbols without starting an engine."""

    try:
        version = importlib.metadata.version(distribution_name)
    except importlib.metadata.PackageNotFoundError:
        return None
    available: set[str] = set()
    errors: list[str] = []
    for symbol in required_symbols:
        module_name, separator, attribute_path = symbol.partition(":")
        if not separator or not module_name or not attribute_path:
            raise ValueError(f"invalid required symbol declaration: {symbol!r}")
        try:
            value: object = importlib.import_module(module_name)
            for attribute in attribute_path.split("."):
                value = getattr(value, attribute)
        except (AttributeError, ImportError, OSError, RuntimeError) as error:
            errors.append(f"{symbol}:{type(error).__name__}")
        else:
            available.add(symbol)
    return RuntimePackageView(
        distribution_name=distribution_name,
        import_name=import_name,
        version=version,
        available_symbols=frozenset(available),
        source="installed",
        discovery_errors=tuple(errors),
    )


def evaluate_package(
    *,
    runtime_name: str,
    view: RuntimePackageView | None,
    policy: VersionPolicy,
    requirements: tuple[str, ...],
    capabilities: frozenset[CapabilityName],
    evidence: tuple[str, ...],
) -> AdapterProbe:
    if view is None:
        return AdapterProbe(
            runtime_name=runtime_name,
            runtime_version=None,
            adapter_version=ADAPTER_VERSION,
            status=IntegrationStatus.PACKAGE_NOT_INSTALLED,
            capabilities=frozenset(),
            missing_requirements=(f"install {runtime_name} within {policy.description}",),
            evidence=evidence,
            exercised=False,
        )
    if not policy.accepts(view.version):
        return AdapterProbe(
            runtime_name=runtime_name,
            runtime_version=view.version,
            adapter_version=ADAPTER_VERSION,
            status=IntegrationStatus.VERSION_UNSUPPORTED,
            capabilities=frozenset(),
            missing_requirements=(f"supported version range is {policy.description}",),
            evidence=evidence,
            exercised=view.exercised,
        )
    missing = tuple(sorted(set(requirements) - set(view.available_symbols)))
    if missing:
        return AdapterProbe(
            runtime_name=runtime_name,
            runtime_version=view.version,
            adapter_version=ADAPTER_VERSION,
            status=IntegrationStatus.API_INCOMPATIBLE,
            capabilities=frozenset(),
            missing_requirements=missing + view.discovery_errors,
            evidence=evidence,
            exercised=view.exercised,
        )
    return AdapterProbe(
        runtime_name=runtime_name,
        runtime_version=view.version,
        adapter_version=ADAPTER_VERSION,
        status=IntegrationStatus.READY,
        capabilities=capabilities,
        missing_requirements=(),
        evidence=evidence,
        exercised=view.exercised,
    )
