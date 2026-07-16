"""PyTorch public-API bridge for explicitly supplied CPU execution tensors.

PyTorch does not publish a generic live-inference-session checkpoint ABI.  This
bridge therefore handles only tensors and RNG state that a model-specific runtime
adapter explicitly supplies.  It never discovers KV cache by inspecting allocator
internals and never moves a GPU tensor to CPU implicitly.
"""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
from typing import Final, Protocol, cast

from sloforge.continuum.adapters.external import (
    AdapterProbe,
    CapabilityName,
    RuntimePackageView,
    SemanticVersion,
    VersionPolicy,
    discover_installed_package,
    evaluate_package,
)
from sloforge.continuum.adapters.sdk import ResourceLimitError, UnsupportedCapabilityError

PYTORCH_VERSION_POLICY: Final = VersionPolicy(
    minimum_inclusive=SemanticVersion(2, 5, 0),
    maximum_exclusive=SemanticVersion(2, 14, 0),
)
PYTORCH_REQUIREMENTS: Final = (
    "torch:Tensor",
    "torch:get_rng_state",
    "torch:set_rng_state",
    "torch:frombuffer",
    "torch.distributed.checkpoint.state_dict:get_model_state_dict",
)
PYTORCH_EVIDENCE: Final = (
    "https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.state_dict",
    "https://docs.pytorch.org/docs/main/distributed.checkpoint.html",
    "https://docs.pytorch.org/docs/stable/distributed.tensor.html",
)
MAX_TENSOR_BYTES: Final = 64 * 1024 * 1024
MAX_RNG_STATE_BYTES: Final = 1024 * 1024


class _NumpyBytes(Protocol):
    def tobytes(self) -> bytes: ...


class _TorchDevice(Protocol):
    type: str


class _TorchTensor(Protocol):
    device: _TorchDevice
    dtype: object
    shape: tuple[int, ...]

    def stride(self) -> tuple[int, ...]: ...

    def detach(self) -> _TorchTensor: ...

    def contiguous(self) -> _TorchTensor: ...

    def view(self, *shape_or_dtype: object) -> _TorchTensor: ...

    def flatten(self) -> _TorchTensor: ...

    def numpy(self) -> _NumpyBytes: ...

    def clone(self) -> _TorchTensor: ...

    def reshape(self, shape: tuple[int, ...]) -> _TorchTensor: ...


class _TorchModule(Protocol):
    Tensor: type[object]
    dtype: type[object]
    uint8: object

    def frombuffer(self, payload: bytearray, *, dtype: object) -> _TorchTensor: ...

    def get_rng_state(self) -> _TorchTensor: ...

    def set_rng_state(self, state: _TorchTensor) -> None: ...


class _CheckpointStateDictModule(Protocol):
    def get_model_state_dict(self, model: object) -> Mapping[str, object]: ...


@dataclass(frozen=True, slots=True)
class PyTorchTensorState:
    shape: tuple[int, ...]
    dtype: str
    source_strides: tuple[int, ...]
    payload: bytes
    checksum: str
    device: str = "cpu"
    encoding: str = "contiguous-native-endian-v1"

    def __post_init__(self) -> None:
        if any(dimension <= 0 for dimension in self.shape):
            raise ValueError("tensor state dimensions must be positive")
        if len(self.shape) != len(self.source_strides):
            raise ValueError("tensor shape and stride ranks differ")
        if self.device != "cpu":
            raise ValueError("portable PyTorch tensor fixture must be CPU-resident")
        if not 0 < len(self.payload) <= MAX_TENSOR_BYTES:
            raise ValueError("tensor payload is empty or exceeds the adapter bound")
        if sha256(self.payload).hexdigest() != self.checksum:
            raise ValueError("PyTorch tensor state checksum mismatch")


def probe_pytorch(view: RuntimePackageView | None = None) -> AdapterProbe:
    discovered = view
    if discovered is None:
        discovered = discover_installed_package(
            distribution_name="torch",
            import_name="torch",
            required_symbols=PYTORCH_REQUIREMENTS,
        )
    return evaluate_package(
        runtime_name="pytorch",
        view=discovered,
        policy=PYTORCH_VERSION_POLICY,
        requirements=PYTORCH_REQUIREMENTS,
        capabilities=frozenset(
            {
                CapabilityName.RUNTIME_INSPECTION,
                CapabilityName.CPU_TENSOR_STATE,
                CapabilityName.CANONICAL_MODEL_STATE_DICT,
                CapabilityName.RNG_STATE,
            }
        ),
        evidence=PYTORCH_EVIDENCE,
    )


class PyTorchRuntimeBinding:
    """Bounded bridge for state already exposed by a PyTorch runtime adapter."""

    def __init__(self, probe: AdapterProbe | None = None) -> None:
        self.probe = probe or probe_pytorch()

    def _torch(self, capability: CapabilityName) -> _TorchModule:
        self.probe.require_capability(capability)
        return cast(_TorchModule, importlib.import_module("torch"))

    def capture_cpu_tensor(self, tensor: object) -> PyTorchTensorState:
        torch = self._torch(CapabilityName.CPU_TENSOR_STATE)
        tensor_type = torch.Tensor
        if not isinstance(tensor, tensor_type):
            raise TypeError("capture_cpu_tensor requires a torch.Tensor")
        typed_tensor = cast(_TorchTensor, tensor)
        device = str(typed_tensor.device.type)
        if device != "cpu":
            raise UnsupportedCapabilityError(
                "GPU tensor capture requires an explicit device-aware Continuum lowering",
                operation="capture_cpu_tensor",
            )
        contiguous = typed_tensor.detach().contiguous()
        payload = contiguous.view(torch.uint8).flatten().numpy().tobytes()
        if len(payload) > MAX_TENSOR_BYTES:
            raise ResourceLimitError(
                "PyTorch tensor exceeds the bounded adapter segment size",
                operation="capture_cpu_tensor",
            )
        return PyTorchTensorState(
            shape=tuple(int(item) for item in typed_tensor.shape),
            dtype=str(typed_tensor.dtype).removeprefix("torch."),
            source_strides=tuple(int(item) for item in typed_tensor.stride()),
            payload=payload,
            checksum=sha256(payload).hexdigest(),
        )

    def import_cpu_tensor(self, state: PyTorchTensorState) -> object:
        torch = self._torch(CapabilityName.CPU_TENSOR_STATE)
        if sha256(state.payload).hexdigest() != state.checksum:
            raise ValueError("PyTorch tensor state checksum mismatch")
        dtype = getattr(torch, state.dtype, None)
        if dtype is None or not isinstance(dtype, torch.dtype):
            raise UnsupportedCapabilityError(
                f"PyTorch dtype {state.dtype!r} is not available in this runtime version",
                operation="import_cpu_tensor",
            )
        raw = torch.frombuffer(bytearray(state.payload), dtype=torch.uint8).clone()
        try:
            restored = raw.view(dtype).reshape(state.shape)
        except RuntimeError as error:
            raise ValueError(
                "tensor payload size does not match declared shape and dtype"
            ) from error
        return restored

    def capture_cpu_rng_state(self) -> bytes:
        torch = self._torch(CapabilityName.RNG_STATE)
        payload = torch.get_rng_state().contiguous().numpy().tobytes()
        if not 0 < len(payload) <= MAX_RNG_STATE_BYTES:
            raise ResourceLimitError(
                "PyTorch CPU RNG state exceeds the adapter bound",
                operation="capture_cpu_rng_state",
            )
        return payload

    def restore_cpu_rng_state(self, payload: bytes) -> None:
        torch = self._torch(CapabilityName.RNG_STATE)
        if not 0 < len(payload) <= MAX_RNG_STATE_BYTES:
            raise ResourceLimitError(
                "PyTorch CPU RNG state is empty or exceeds the adapter bound",
                operation="restore_cpu_rng_state",
            )
        state = torch.frombuffer(bytearray(payload), dtype=torch.uint8).clone()
        torch.set_rng_state(state)

    def canonical_model_state_dict(self, model: object) -> Mapping[str, object]:
        self._torch(CapabilityName.CANONICAL_MODEL_STATE_DICT)
        module = cast(
            _CheckpointStateDictModule,
            importlib.import_module("torch.distributed.checkpoint.state_dict"),
        )
        state = module.get_model_state_dict(model)
        if not isinstance(state, Mapping) or not all(isinstance(key, str) and key for key in state):
            raise TypeError("PyTorch returned a non-canonical model state dictionary")
        return state


__all__ = [
    "PYTORCH_EVIDENCE",
    "PYTORCH_REQUIREMENTS",
    "PYTORCH_VERSION_POLICY",
    "PyTorchRuntimeBinding",
    "PyTorchTensorState",
    "probe_pytorch",
]
