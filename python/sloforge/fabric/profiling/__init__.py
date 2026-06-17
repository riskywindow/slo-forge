"""Fabric benchmarking public API."""

from sloforge.fabric.ir import FabricProfile
from sloforge.fabric.profiling.adapters import (
    AdapterAvailability,
    AdapterCommand,
    adapter_inventory,
    build_ibverbs_command,
    build_nccl_tests_command,
    build_nvidia_smi_command,
)
from sloforge.fabric.profiling.benchmark import (
    benchmark_host_memory,
    benchmark_synthetic_fabric,
    load_profile,
    save_profile,
)
from sloforge.fabric.profiling.conversion import to_canonical_profile
from sloforge.fabric.profiling.models import (
    BenchmarkCase,
    BenchmarkResult,
    BenchmarkStatus,
    Direction,
    MeasurementMode,
    Primitive,
    RawSample,
    RobustSummary,
)
from sloforge.fabric.profiling.models import (
    FabricProfile as RawFabricProfile,
)
from sloforge.fabric.profiling.runner import (
    AdapterExecutionError,
    CommandCapture,
    NcclTestsRow,
    NvidiaInventoryRecord,
    execute_bounded,
    parse_nccl_tests_output,
    read_nvidia_inventory,
    run_nccl_tests_profile,
)

__all__ = [
    "AdapterAvailability",
    "AdapterCommand",
    "AdapterExecutionError",
    "BenchmarkCase",
    "BenchmarkResult",
    "BenchmarkStatus",
    "CommandCapture",
    "Direction",
    "FabricProfile",
    "MeasurementMode",
    "NcclTestsRow",
    "NvidiaInventoryRecord",
    "Primitive",
    "RawFabricProfile",
    "RawSample",
    "RobustSummary",
    "adapter_inventory",
    "benchmark_host_memory",
    "benchmark_synthetic_fabric",
    "build_ibverbs_command",
    "build_nccl_tests_command",
    "build_nvidia_smi_command",
    "execute_bounded",
    "load_profile",
    "parse_nccl_tests_output",
    "read_nvidia_inventory",
    "run_nccl_tests_profile",
    "save_profile",
    "to_canonical_profile",
]
