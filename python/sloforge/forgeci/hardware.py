"""Explicit ForgeCI host requirement validation without device fallback."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
from pathlib import Path

from sloforge.forgeci.models import EnvironmentSpec, HardwareObservation, HardwareRequirement


class RequirementMismatch(RuntimeError):
    """The current host cannot faithfully run the requested matrix case."""


def _memory_gib() -> float | None:
    if not hasattr(os, "sysconf"):
        return None
    try:
        pages = int(os.sysconf("SC_PHYS_PAGES"))
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
    except (OSError, ValueError):
        return None
    return pages * page_size / 1024**3


def _gpus() -> tuple[tuple[str, float], ...]:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return ()
    completed = subprocess.run(
        [
            executable,
            "--query-gpu=name,memory.total",
            "--format=csv,noheader,nounits",
        ],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=5.0,
        check=False,
    )
    if completed.returncode != 0:
        raise RequirementMismatch(f"nvidia-smi probe failed: {completed.stderr.strip()}")
    devices: list[tuple[str, float]] = []
    for line in completed.stdout.splitlines():
        fields = [field.strip() for field in line.rsplit(",", maxsplit=1)]
        if len(fields) != 2:
            raise RequirementMismatch(f"unrecognized nvidia-smi device record: {line!r}")
        devices.append((fields[0], float(fields[1]) / 1024.0))
    return tuple(devices)


def observe_hardware() -> HardwareObservation:
    """Capture only capabilities observed on this host."""

    architecture = platform.machine() or "unknown"
    cpu_cores = os.cpu_count() or 1
    memory = _memory_gib()
    devices = _gpus()
    rdma = Path("/sys/class/infiniband").is_dir() and any(Path("/sys/class/infiniband").iterdir())
    topology = os.environ.get("SLOFORGE_TOPOLOGY_FINGERPRINT")
    facts = {
        "architecture": architecture,
        "cpu_cores": cpu_cores,
        "memory_gib": memory,
        "gpu_models": [name for name, _ in devices],
        "gpu_memory_gib": [memory_gib for _, memory_gib in devices],
        "rdma_available": rdma,
        "topology_fingerprint": topology,
    }
    fingerprint = hashlib.sha256(
        json.dumps(facts, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return HardwareObservation(
        architecture=architecture,
        cpu_cores=cpu_cores,
        memory_gib=memory,
        gpu_count=len(devices),
        gpu_models=tuple(name for name, _ in devices),
        gpu_memory_gib=tuple(memory_gib for _, memory_gib in devices),
        rdma_available=rdma,
        topology_fingerprint=topology,
        fingerprint=fingerprint,
    )


def validate_requirements(
    hardware: HardwareRequirement, environment: EnvironmentSpec
) -> HardwareObservation:
    """Reject unavailable hardware or versions rather than using another execution path."""

    observed = observe_hardware()
    if hardware.architecture not in {"cpu", "any", observed.architecture}:
        raise RequirementMismatch(
            f"required architecture {hardware.architecture}, observed {observed.architecture}"
        )
    if observed.cpu_cores < hardware.minimum_cpu_cores:
        raise RequirementMismatch(
            f"required {hardware.minimum_cpu_cores} CPU cores, observed {observed.cpu_cores}"
        )
    if observed.memory_gib is None and hardware.minimum_memory_gib > 0.0:
        raise RequirementMismatch("system memory could not be observed")
    if observed.memory_gib is not None and observed.memory_gib < hardware.minimum_memory_gib:
        raise RequirementMismatch(
            f"required {hardware.minimum_memory_gib:g} GiB memory, "
            f"observed {observed.memory_gib:g} GiB"
        )
    if observed.gpu_count < hardware.gpu_count:
        raise RequirementMismatch(
            f"required {hardware.gpu_count} GPUs, observed {observed.gpu_count}; CPU fallback disabled"
        )
    if hardware.gpu_model is not None and not all(
        hardware.gpu_model.casefold() in model.casefold()
        for model in observed.gpu_models[: hardware.gpu_count]
    ):
        raise RequirementMismatch(f"required GPU model {hardware.gpu_model!r} was not observed")
    if hardware.minimum_gpu_memory_gib is not None and any(
        value < hardware.minimum_gpu_memory_gib
        for value in observed.gpu_memory_gib[: hardware.gpu_count]
    ):
        raise RequirementMismatch("one or more selected GPUs have insufficient memory")
    if hardware.requires_rdma and not observed.rdma_available:
        raise RequirementMismatch("RDMA was required but no InfiniBand device was observed")
    if (
        hardware.topology_fingerprint is not None
        and observed.topology_fingerprint != hardware.topology_fingerprint
    ):
        raise RequirementMismatch("observed topology fingerprint does not match the matrix")

    if environment.python_version is not None and not platform.python_version().startswith(
        environment.python_version
    ):
        raise RequirementMismatch(
            f"required Python {environment.python_version}, observed {platform.python_version()}"
        )
    if environment.pytorch_version is not None:
        try:
            installed_torch = importlib.metadata.version("torch")
        except importlib.metadata.PackageNotFoundError as error:
            raise RequirementMismatch("PyTorch is required but not installed") from error
        if installed_torch != environment.pytorch_version:
            raise RequirementMismatch(
                f"required PyTorch {environment.pytorch_version}, observed {installed_torch}"
            )
    explicit_versions = (
        ("CUDA", environment.cuda_version, "SLOFORGE_CUDA_VERSION"),
        (
            "communication library",
            environment.communication_library_version,
            "SLOFORGE_COMMUNICATION_LIBRARY_VERSION",
        ),
        ("container image", environment.container_image, "SLOFORGE_CONTAINER_IMAGE"),
    )
    for label, required, variable in explicit_versions:
        if required is not None and os.environ.get(variable) != required:
            observed_value = os.environ.get(variable, "<unreported>")
            raise RequirementMismatch(f"required {label} {required}, observed {observed_value}")
    return observed
