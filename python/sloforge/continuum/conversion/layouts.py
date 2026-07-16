"""Trusted CPU representation of two portable KV physical layouts."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from enum import StrEnum

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.number]


class KVLayoutKind(StrEnum):
    TOKEN_MAJOR_SEPARATE = "paged_token_major_separate"
    HEAD_MAJOR_PACKED = "paged_head_major_packed"


class LayoutValidationError(ValueError):
    """Raised when physical state does not satisfy its declared layout."""


class StateIntegrityError(ValueError):
    """Raised when state bytes no longer match the authenticated content hash."""


@dataclass(frozen=True)
class KVLayout:
    kind: KVLayoutKind
    tensor_parallel_degree: int
    page_size_tokens: int
    layer_count: int
    token_count: int
    kv_head_count: int
    head_dim: int
    dtype: str = "float32"

    def __post_init__(self) -> None:
        values = (
            self.tensor_parallel_degree,
            self.page_size_tokens,
            self.layer_count,
            self.kv_head_count,
            self.head_dim,
        )
        if any(value <= 0 for value in values) or self.token_count < 0:
            raise LayoutValidationError(
                "layout dimensions must be positive; token_count may be zero"
            )
        if self.kv_head_count % self.tensor_parallel_degree != 0:
            raise LayoutValidationError("kv_head_count must be divisible by tensor_parallel_degree")
        dtype = np.dtype(self.dtype)
        if dtype.kind not in {"f", "i", "u"}:
            raise LayoutValidationError("trusted KV conversion requires a numeric tensor dtype")

    @property
    def padded_token_count(self) -> int:
        if self.token_count == 0:
            return 0
        return math.ceil(self.token_count / self.page_size_tokens) * self.page_size_tokens

    @property
    def heads_per_shard(self) -> int:
        return self.kv_head_count // self.tensor_parallel_degree

    @property
    def logical_shape(self) -> tuple[int, int, int, int]:
        return (self.layer_count, self.token_count, self.kv_head_count, self.head_dim)

    @property
    def logical_nbytes(self) -> int:
        return int(np.prod(self.logical_shape, dtype=np.int64)) * np.dtype(self.dtype).itemsize * 2

    @property
    def physical_nbytes(self) -> int:
        elements = (
            self.layer_count * self.padded_token_count * self.kv_head_count * self.head_dim * 2
        )
        return elements * np.dtype(self.dtype).itemsize

    def fingerprint(self) -> str:
        payload = json.dumps(
            {
                "dtype": self.dtype,
                "head_dim": self.head_dim,
                "kind": self.kind.value,
                "kv_head_count": self.kv_head_count,
                "layer_count": self.layer_count,
                "page_size_tokens": self.page_size_tokens,
                "tensor_parallel_degree": self.tensor_parallel_degree,
                "token_count": self.token_count,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class KVShard:
    rank: int
    head_start: int
    head_end: int
    key: FloatArray | None = None
    value: FloatArray | None = None
    packed: FloatArray | None = None


def _state_checksum(layout: KVLayout, shards: tuple[KVShard, ...]) -> str:
    digest = hashlib.sha256()
    digest.update(layout.fingerprint().encode("ascii"))
    for shard in sorted(shards, key=lambda candidate: candidate.rank):
        digest.update(f"{shard.rank}:{shard.head_start}:{shard.head_end}".encode("ascii"))
        arrays = (shard.key, shard.value, shard.packed)
        for array in arrays:
            if array is None:
                digest.update(b"none")
                continue
            canonical = np.ascontiguousarray(array)
            digest.update(canonical.dtype.str.encode("ascii"))
            digest.update(str(canonical.shape).encode("ascii"))
            digest.update(canonical.tobytes(order="C"))
    return digest.hexdigest()


@dataclass(frozen=True)
class PhysicalKVState:
    layout: KVLayout
    shards: tuple[KVShard, ...]
    content_hash: str = ""

    def __post_init__(self) -> None:
        _validate_shards(self.layout, self.shards)
        if not self.content_hash:
            object.__setattr__(self, "content_hash", _state_checksum(self.layout, self.shards))

    def verify_integrity(self) -> None:
        observed = _state_checksum(self.layout, self.shards)
        if observed != self.content_hash:
            raise StateIntegrityError(
                f"physical state checksum mismatch: expected {self.content_hash}, observed {observed}"
            )


def _validate_shards(layout: KVLayout, shards: tuple[KVShard, ...]) -> None:
    if len(shards) != layout.tensor_parallel_degree:
        raise LayoutValidationError("shard count does not match tensor-parallel degree")
    expected_dtype = np.dtype(layout.dtype)
    padded = layout.padded_token_count
    heads = layout.heads_per_shard
    for rank, shard in enumerate(sorted(shards, key=lambda candidate: candidate.rank)):
        if shard.rank != rank:
            raise LayoutValidationError("shard ranks must be contiguous and unique")
        if (shard.head_start, shard.head_end) != (rank * heads, (rank + 1) * heads):
            raise LayoutValidationError("shards must cover a disjoint contiguous head partition")
        if layout.kind is KVLayoutKind.TOKEN_MAJOR_SEPARATE:
            separate_shape = (layout.layer_count, padded, heads, layout.head_dim)
            if shard.key is None or shard.value is None or shard.packed is not None:
                raise LayoutValidationError("token-major layout requires separate K and V arrays")
            if shard.key.shape != separate_shape or shard.value.shape != separate_shape:
                raise LayoutValidationError("token-major shard shape does not match layout")
            if shard.key.dtype != expected_dtype or shard.value.dtype != expected_dtype:
                raise LayoutValidationError("token-major shard dtype does not match layout")
        else:
            packed_shape = (layout.layer_count, heads, padded, 2, layout.head_dim)
            if shard.packed is None or shard.key is not None or shard.value is not None:
                raise LayoutValidationError("head-major layout requires one packed K/V array")
            if shard.packed.shape != packed_shape:
                raise LayoutValidationError("head-major packed shard shape does not match layout")
            if shard.packed.dtype != expected_dtype:
                raise LayoutValidationError("head-major packed shard dtype does not match layout")


def allocate_state(layout: KVLayout) -> PhysicalKVState:
    dtype = np.dtype(layout.dtype)
    shards: list[KVShard] = []
    for rank in range(layout.tensor_parallel_degree):
        head_start = rank * layout.heads_per_shard
        head_end = head_start + layout.heads_per_shard
        if layout.kind is KVLayoutKind.TOKEN_MAJOR_SEPARATE:
            separate_shape = (
                layout.layer_count,
                layout.padded_token_count,
                layout.heads_per_shard,
                layout.head_dim,
            )
            shards.append(
                KVShard(
                    rank=rank,
                    head_start=head_start,
                    head_end=head_end,
                    key=np.zeros(separate_shape, dtype=dtype),
                    value=np.zeros(separate_shape, dtype=dtype),
                )
            )
        else:
            packed_shape = (
                layout.layer_count,
                layout.heads_per_shard,
                layout.padded_token_count,
                2,
                layout.head_dim,
            )
            shards.append(
                KVShard(
                    rank=rank,
                    head_start=head_start,
                    head_end=head_end,
                    packed=np.zeros(packed_shape, dtype=dtype),
                )
            )
    return PhysicalKVState(layout=layout, shards=tuple(shards))


def encode_logical(key: FloatArray, value: FloatArray, layout: KVLayout) -> PhysicalKVState:
    """Trusted encoding from canonical [layer, token, head, dim] tensors."""

    expected = layout.logical_shape
    if key.shape != expected or value.shape != expected:
        raise LayoutValidationError(f"canonical tensor shape must be {expected}")
    destination = allocate_state(layout)
    dtype = np.dtype(layout.dtype)
    for shard in destination.shards:
        head_slice = slice(shard.head_start, shard.head_end)
        key_slice = key[:, :, head_slice, :].astype(dtype, copy=False)
        value_slice = value[:, :, head_slice, :].astype(dtype, copy=False)
        if layout.kind is KVLayoutKind.TOKEN_MAJOR_SEPARATE:
            assert shard.key is not None and shard.value is not None
            shard.key[:, : layout.token_count, :, :] = key_slice
            shard.value[:, : layout.token_count, :, :] = value_slice
        else:
            assert shard.packed is not None
            shard.packed[:, :, : layout.token_count, 0, :] = key_slice.transpose(0, 2, 1, 3)
            shard.packed[:, :, : layout.token_count, 1, :] = value_slice.transpose(0, 2, 1, 3)
    return PhysicalKVState(layout=layout, shards=destination.shards)


def decode_logical(state: PhysicalKVState) -> tuple[FloatArray, FloatArray]:
    """Trusted canonical materialization used as the correctness fallback."""

    state.verify_integrity()
    layout = state.layout
    dtype = np.dtype(layout.dtype)
    key = np.empty(layout.logical_shape, dtype=dtype)
    value = np.empty(layout.logical_shape, dtype=dtype)
    for shard in state.shards:
        head_slice = slice(shard.head_start, shard.head_end)
        if layout.kind is KVLayoutKind.TOKEN_MAJOR_SEPARATE:
            assert shard.key is not None and shard.value is not None
            key[:, :, head_slice, :] = shard.key[:, : layout.token_count, :, :]
            value[:, :, head_slice, :] = shard.value[:, : layout.token_count, :, :]
        else:
            assert shard.packed is not None
            key[:, :, head_slice, :] = shard.packed[:, :, : layout.token_count, 0, :].transpose(
                0, 2, 1, 3
            )
            value[:, :, head_slice, :] = shard.packed[:, :, : layout.token_count, 1, :].transpose(
                0, 2, 1, 3
            )
    return key, value


def make_random_state(layout: KVLayout, *, seed: int) -> PhysicalKVState:
    """Create deterministic, non-fabricated state for tests and measured selection."""

    rng = np.random.default_rng(seed)
    key = rng.standard_normal(layout.logical_shape).astype(layout.dtype)
    value = rng.standard_normal(layout.logical_shape).astype(layout.dtype)
    return encode_logical(key, value, layout)
