"""Standalone exact saturating recurrent-state update.

The implementation has no SLOForge dependency and intentionally targets a
small, reviewable Python CPU domain. Python's ``round`` supplies ties-to-even
rounding.
"""

from __future__ import annotations

import math
from collections.abc import MutableSequence, Sequence

MINIMUM_STATE = -127
MAXIMUM_STATE = 127
MAXIMUM_COUNT = 128
ALLOWED_STRIDES = (1, 2, 3)


def reference_quantized_state_update(previous: int, activation: float) -> int:
    """Return one exact symmetric-int8 state update.

    ``previous`` excludes -128 so that the domain is symmetric. Non-finite
    activations are rejected. Finite values whose scaled intermediate
    overflows binary64 saturate according to its sign.
    """

    if type(previous) is not int or not MINIMUM_STATE <= previous <= MAXIMUM_STATE:
        raise ValueError("previous state must be an integer in [-127, 127]")
    if type(activation) not in {int, float} or type(activation) is bool:
        raise ValueError("activation must be a real scalar")
    activation_value = float(activation)
    if not math.isfinite(activation_value):
        raise ValueError("activation must be finite")
    combined = previous * 0.625 + activation_value * 31.0
    if math.isinf(combined):
        return MAXIMUM_STATE if combined > 0 else MINIMUM_STATE
    return max(MINIMUM_STATE, min(MAXIMUM_STATE, round(combined)))


def _validate_view(
    storage: Sequence[object],
    *,
    count: int,
    offset: int,
    stride: int,
    name: str,
) -> None:
    if type(count) is not int or not 1 <= count <= MAXIMUM_COUNT:
        raise ValueError(f"count must be an integer in [1, {MAXIMUM_COUNT}]")
    if type(offset) is not int or offset < 0:
        raise ValueError(f"{name} offset must be a non-negative integer")
    if type(stride) is not int or stride not in ALLOWED_STRIDES:
        raise ValueError(f"{name} stride must be one of {ALLOWED_STRIDES}")
    final_index = offset + (count - 1) * stride
    if final_index >= len(storage):
        raise ValueError(f"{name} storage is too small for the declared view")


def quantized_recurrent_state_update(
    previous_storage: MutableSequence[int],
    activation_storage: Sequence[float],
    count: int,
    previous_offset: int = 0,
    previous_stride: int = 1,
    activation_offset: int = 0,
    activation_stride: int = 1,
    output_alias_previous: bool = False,
) -> list[int]:
    """Update a bounded strided state view and return logical-order results.

    When ``output_alias_previous`` is true, validation and calculation finish
    before any state is committed. A rejected input therefore cannot expose a
    partially updated state through this function.
    """

    if type(output_alias_previous) is not bool:
        raise ValueError("output_alias_previous must be boolean")
    _validate_view(
        previous_storage,
        count=count,
        offset=previous_offset,
        stride=previous_stride,
        name="previous",
    )
    _validate_view(
        activation_storage,
        count=count,
        offset=activation_offset,
        stride=activation_stride,
        name="activation",
    )
    if output_alias_previous and not isinstance(previous_storage, MutableSequence):
        raise ValueError("aliased output requires mutable previous storage")

    previous_values: list[int] = []
    activation_values: list[float] = []
    for logical_index in range(count):
        previous = previous_storage[previous_offset + logical_index * previous_stride]
        activation = activation_storage[activation_offset + logical_index * activation_stride]
        if type(previous) is not int or not MINIMUM_STATE <= previous <= MAXIMUM_STATE:
            raise ValueError("previous state must be an integer in [-127, 127]")
        if type(activation) not in {int, float} or type(activation) is bool:
            raise ValueError("activation must be a real scalar")
        activation_value = float(activation)
        if not math.isfinite(activation_value):
            raise ValueError("activation must be finite")
        previous_values.append(previous)
        activation_values.append(activation_value)

    result = [
        reference_quantized_state_update(previous, activation)
        for previous, activation in zip(previous_values, activation_values, strict=True)
    ]
    if output_alias_previous:
        for logical_index, updated in enumerate(result):
            previous_storage[previous_offset + logical_index * previous_stride] = updated
    return result


__all__ = [
    "ALLOWED_STRIDES",
    "MAXIMUM_COUNT",
    "MAXIMUM_STATE",
    "MINIMUM_STATE",
    "quantized_recurrent_state_update",
    "reference_quantized_state_update",
]
