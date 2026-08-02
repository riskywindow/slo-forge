"""Fail-closed capability adapter for optional Triton experiments."""

from __future__ import annotations

import importlib.util
import os
from importlib import metadata

from .models import AdapterStatus, TritonAdapterReport


def triton_adapter_status() -> TritonAdapterReport:
    """Report availability without importing Triton or claiming GPU execution."""

    specification = importlib.util.find_spec("triton")
    if specification is None:
        return TritonAdapterReport(
            status=AdapterStatus.UNAVAILABLE,
            installed_version=None,
            reason="Triton is not installed; no generated GPU code was compiled or benchmarked",
        )
    try:
        version = metadata.version("triton")
    except metadata.PackageNotFoundError:
        version = "unknown"
    opted_in = os.environ.get("SLOFORGE_GENESIS_ALLOW_GPU", "").lower() in {"1", "true", "yes"}
    reason = (
        "GPU opt-in is present, but this CPU run has no hardware harness; Triton remains unexercised"
        if opted_in
        else "GPU execution requires SLOFORGE_GENESIS_ALLOW_GPU=1; Triton remains unexercised"
    )
    return TritonAdapterReport(
        status=AdapterStatus.UNEXERCISED,
        installed_version=version,
        reason=reason,
    )
