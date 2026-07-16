"""Optional, explicitly selected PyTorch conversion backend.

This module never falls back between CPU and CUDA. A result is returned only
after independent comparison with the trusted NumPy canonical converter.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any

import numpy as np

from sloforge.continuum.adapters.external import AdapterProbe
from sloforge.continuum.adapters.pytorch import probe_pytorch
from sloforge.continuum.adapters.sdk import UnsupportedCapabilityError
from sloforge.continuum.compatibility import ExactnessClass

from .compiler import ConversionCompilationError, canonical_convert, compile_conversion
from .layouts import KVLayout, KVLayoutKind, PhysicalKVState, allocate_state, decode_logical


@dataclass(frozen=True, slots=True)
class PyTorchConversionEvidence:
    backend: str
    requested_device: str
    exercised_device: str
    torch_version: str
    source_hash: str
    canonical_hash: str
    converted_hash: str
    canonical_match: bool
    maximum_controlled_tensor_bytes: int
    maximum_temporary_bytes: int
    declared_exactness: ExactnessClass
    source_to_destination_maximum_absolute_error: float
    numeric_tolerance: float | None
    numeric_contract_satisfied: bool
    gpu_exercised: bool
    verification_scope: str = "complete destination physical bytes versus canonical_cpu_v1"


def _torch_dtype(torch: Any, dtype: np.dtype[Any]) -> Any:
    value = getattr(torch, dtype.name, None)
    if value is None:
        raise UnsupportedCapabilityError(
            f"PyTorch does not expose destination dtype {dtype.name!r}",
            operation="pytorch_state_conversion",
        )
    return value


def _tensor(torch: Any, array: np.ndarray[Any, Any], *, device: str, dtype: Any) -> Any:
    if any(stride < 0 for stride in array.strides):
        raise ConversionCompilationError(
            "PyTorch backend rejects negative-stride source views instead of copying implicitly"
        )
    return torch.from_numpy(array).to(device=device, dtype=dtype)


def _schedule_bound(
    source: KVLayout,
    destination: KVLayout,
    maximum_temporary_bytes: int,
) -> int:
    scalar_count = source.layer_count * source.kv_head_count * source.head_dim * 2
    controlled_per_token = scalar_count * (
        np.dtype(source.dtype).itemsize + np.dtype(destination.dtype).itemsize
    )
    if maximum_temporary_bytes < controlled_per_token:
        raise ConversionCompilationError(
            "PyTorch memory bound cannot hold one source and destination token slice"
        )
    compiler_per_token = scalar_count * max(
        np.dtype(source.dtype).itemsize,
        np.dtype(destination.dtype).itemsize,
    )
    chunk_tokens = maximum_temporary_bytes // controlled_per_token
    return max(compiler_per_token, chunk_tokens * compiler_per_token)


def pytorch_convert(
    source: PhysicalKVState,
    destination: KVLayout,
    *,
    maximum_temporary_bytes: int,
    device: str,
    probe: AdapterProbe | None = None,
    numeric_tolerance: float | None = None,
) -> tuple[PhysicalKVState, PyTorchConversionEvidence]:
    """Convert with PyTorch on exactly ``cpu`` or ``cuda`` and verify independently."""

    if device not in {"cpu", "cuda"}:
        raise ValueError("PyTorch conversion device must be exactly 'cpu' or 'cuda'")
    selected_probe = probe if probe is not None else probe_pytorch()
    selected_probe.require_ready(operation="pytorch_state_conversion")
    try:
        torch = importlib.import_module("torch")
    except (ImportError, OSError) as error:
        raise UnsupportedCapabilityError(
            "the version-gated PyTorch module could not be loaded",
            operation="pytorch_state_conversion",
        ) from error
    if device == "cuda" and not bool(torch.cuda.is_available()):
        raise UnsupportedCapabilityError(
            "CUDA was explicitly requested but is unavailable; CPU fallback is prohibited",
            operation="pytorch_state_conversion",
        )

    source.verify_integrity()
    schedule_bound = _schedule_bound(source.layout, destination, maximum_temporary_bytes)
    program = compile_conversion(
        source.layout,
        destination,
        maximum_temporary_bytes=schedule_bound,
    )
    output = allocate_state(destination)
    output_shards = {shard.rank: shard for shard in output.shards}
    destination_dtype = np.dtype(destination.dtype)
    torch_dtype = _torch_dtype(torch, destination_dtype)
    maximum_controlled = 0

    for assignment in program.chunk_schedule.chunks:
        token_count = assignment.token_end - assignment.token_start
        local_heads = assignment.head_end - assignment.head_start
        if destination.kind is KVLayoutKind.TOKEN_MAJOR_SEPARATE:
            separate_shape = (
                destination.layer_count,
                token_count,
                local_heads,
                destination.head_dim,
            )
            key_output = torch.empty(separate_shape, device=device, dtype=torch_dtype)
            value_output = torch.empty(separate_shape, device=device, dtype=torch_dtype)
            controlled = key_output.numel() * key_output.element_size() * 2
        else:
            packed_shape = (
                destination.layer_count,
                local_heads,
                token_count,
                2,
                destination.head_dim,
            )
            packed_output = torch.empty(packed_shape, device=device, dtype=torch_dtype)
            controlled = packed_output.numel() * packed_output.element_size()

        covered = np.zeros(local_heads, dtype=np.bool_)
        for source_shard in source.shards:
            overlap_start = max(assignment.head_start, source_shard.head_start)
            overlap_end = min(assignment.head_end, source_shard.head_end)
            if overlap_start >= overlap_end:
                continue
            source_start = overlap_start - source_shard.head_start
            source_end = overlap_end - source_shard.head_start
            output_start = overlap_start - assignment.head_start
            output_end = overlap_end - assignment.head_start
            token_slice = slice(assignment.token_start, assignment.token_end)
            if source.layout.kind is KVLayoutKind.TOKEN_MAJOR_SEPARATE:
                if source_shard.key is None or source_shard.value is None:
                    raise ConversionCompilationError("source shard omitted separate K/V tensors")
                key_input = _tensor(
                    torch,
                    source_shard.key[:, token_slice, source_start:source_end, :],
                    device=device,
                    dtype=torch_dtype,
                )
                value_input = _tensor(
                    torch,
                    source_shard.value[:, token_slice, source_start:source_end, :],
                    device=device,
                    dtype=torch_dtype,
                )
                controlled_here = controlled + (
                    key_input.numel() * key_input.element_size()
                    + value_input.numel() * value_input.element_size()
                )
                if destination.kind is KVLayoutKind.TOKEN_MAJOR_SEPARATE:
                    key_output[:, :, output_start:output_end, :] = key_input
                    value_output[:, :, output_start:output_end, :] = value_input
                else:
                    packed_output[:, output_start:output_end, :, 0, :] = key_input.permute(
                        0, 2, 1, 3
                    )
                    packed_output[:, output_start:output_end, :, 1, :] = value_input.permute(
                        0, 2, 1, 3
                    )
            else:
                if source_shard.packed is None:
                    raise ConversionCompilationError("source shard omitted packed K/V tensor")
                packed_input = _tensor(
                    torch,
                    source_shard.packed[:, source_start:source_end, token_slice, :, :],
                    device=device,
                    dtype=torch_dtype,
                )
                controlled_here = controlled + packed_input.numel() * packed_input.element_size()
                if destination.kind is KVLayoutKind.TOKEN_MAJOR_SEPARATE:
                    key_output[:, :, output_start:output_end, :] = packed_input[
                        :, :, :, 0, :
                    ].permute(0, 2, 1, 3)
                    value_output[:, :, output_start:output_end, :] = packed_input[
                        :, :, :, 1, :
                    ].permute(0, 2, 1, 3)
                else:
                    packed_output[:, output_start:output_end, :, :, :] = packed_input
            maximum_controlled = max(maximum_controlled, int(controlled_here))
            covered[output_start:output_end] = True
        if not np.all(covered):
            raise ConversionCompilationError("source shards do not cover destination head range")
        if maximum_controlled > maximum_temporary_bytes:
            raise ConversionCompilationError(
                "PyTorch converter exceeded its controlled tensor bound"
            )

        output_shard = output_shards[assignment.destination_rank]
        token_slice = slice(assignment.token_start, assignment.token_end)
        if destination.kind is KVLayoutKind.TOKEN_MAJOR_SEPARATE:
            if output_shard.key is None or output_shard.value is None:
                raise ConversionCompilationError("destination omitted separate K/V tensors")
            output_shard.key[:, token_slice, :, :] = key_output.detach().cpu().numpy()
            output_shard.value[:, token_slice, :, :] = value_output.detach().cpu().numpy()
        else:
            if output_shard.packed is None:
                raise ConversionCompilationError("destination omitted packed K/V tensor")
            output_shard.packed[:, :, token_slice, :, :] = packed_output.detach().cpu().numpy()

    converted = PhysicalKVState(layout=destination, shards=output.shards)
    canonical = canonical_convert(source, destination)
    converted.verify_integrity()
    canonical.verify_integrity()
    canonical_match = converted.content_hash == canonical.content_hash
    if not canonical_match:
        raise ConversionCompilationError(
            "PyTorch conversion differs from the independently computed canonical output"
        )
    source_key, source_value = decode_logical(source)
    destination_key, destination_value = decode_logical(canonical)
    if source_key.size + source_value.size:
        key_error = float(
            np.max(np.abs(source_key.astype(np.float64) - destination_key.astype(np.float64)))
        )
        value_error = float(
            np.max(np.abs(source_value.astype(np.float64) - destination_value.astype(np.float64)))
        )
        source_to_destination_error = max(key_error, value_error)
    else:
        source_to_destination_error = 0.0
    numeric_contract_satisfied = program.exactness is ExactnessClass.EXACT_SEMANTIC or (
        numeric_tolerance is not None and source_to_destination_error <= numeric_tolerance
    )
    if not numeric_contract_satisfied:
        raise ConversionCompilationError(
            "PyTorch numerical conversion has no satisfied source-to-destination tolerance"
        )
    return converted, PyTorchConversionEvidence(
        backend="pytorch_explicit_device_v1",
        requested_device=device,
        exercised_device=device,
        torch_version=str(torch.__version__),
        source_hash=source.content_hash,
        canonical_hash=canonical.content_hash,
        converted_hash=converted.content_hash,
        canonical_match=True,
        maximum_controlled_tensor_bytes=maximum_controlled,
        maximum_temporary_bytes=maximum_temporary_bytes,
        declared_exactness=program.exactness,
        source_to_destination_maximum_absolute_error=source_to_destination_error,
        numeric_tolerance=numeric_tolerance,
        numeric_contract_satisfied=True,
        gpu_exercised=device == "cuda",
    )


__all__ = ["PyTorchConversionEvidence", "pytorch_convert"]
