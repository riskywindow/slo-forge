"""Deterministic schema-aware cases for the quantized recurrent-state update."""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from typing import Literal

from .models import CaseCategory, KernelCase


def reference_quantized_state_update(previous: int, activation: float) -> int:
    if not -127 <= previous <= 127:
        raise ValueError("previous state outside symmetric int8 domain")
    if not math.isfinite(activation):
        raise ValueError("activation must be finite")
    return max(-127, min(127, round(previous * 0.625 + activation * 31.0)))


def _storage(
    values: Sequence[int | float], *, offset: int, stride: int, fill: int | float
) -> list[int | float]:
    storage = [fill] * (offset + (len(values) - 1) * stride + 1)
    for index, value in enumerate(values):
        storage[offset + index * stride] = value
    return storage


def _case(
    *,
    case_id: str,
    previous: list[int],
    activations: list[float],
    previous_offset: int,
    previous_stride: int,
    activation_offset: int,
    activation_stride: int,
    alias: bool,
    category: CaseCategory,
) -> KernelCase:
    previous_storage = _storage(previous, offset=previous_offset, stride=previous_stride, fill=17)
    activation_storage = _storage(
        activations, offset=activation_offset, stride=activation_stride, fill=0.125
    )
    expected_error: Literal["ValueError"] | None = None
    try:
        expected = tuple(
            reference_quantized_state_update(previous_value, activation)
            for previous_value, activation in zip(previous, activations, strict=True)
        )
    except ValueError:
        expected = None
        expected_error = "ValueError"
    return KernelCase(
        case_id=case_id,
        previous_storage=tuple(int(value) for value in previous_storage),
        activation_storage=tuple(repr(float(value)) for value in activation_storage),
        count=len(previous),
        previous_offset=previous_offset,
        previous_stride=previous_stride,
        activation_offset=activation_offset,
        activation_stride=activation_stride,
        output_alias_previous=alias,
        expected=expected,
        expected_error=expected_error,
        category=category,
    )


def generate_correctness_cases(*, seed: int, randomized_cases: int = 24) -> tuple[KernelCase, ...]:
    if seed < 0 or not 1 <= randomized_cases <= 1_000:
        raise ValueError("case seed or randomized-case bound is invalid")
    cases = [
        _case(
            case_id="edge-scalar-tie",
            previous=[0],
            activations=[0.5 / 31.0],
            previous_offset=0,
            previous_stride=1,
            activation_offset=0,
            activation_stride=1,
            alias=False,
            category="edge",
        ),
        _case(
            case_id="edge-saturation",
            previous=[-127, 127, 0, 1],
            activations=[-100.0, 100.0, 0.0, 0.5 / 31.0],
            previous_offset=0,
            previous_stride=1,
            activation_offset=0,
            activation_stride=1,
            alias=False,
            category="edge",
        ),
        _case(
            case_id="stride-noncontiguous",
            previous=[-41, 0, 41, 126],
            activations=[-1.5, -0.5, 0.5, 1.5],
            previous_offset=1,
            previous_stride=2,
            activation_offset=2,
            activation_stride=3,
            alias=False,
            category="noncontiguous",
        ),
        _case(
            case_id="alias-exact-view",
            previous=[-5, 6, 17],
            activations=[0.2, -0.4, 0.6],
            previous_offset=2,
            previous_stride=2,
            activation_offset=1,
            activation_stride=2,
            alias=True,
            category="alias",
        ),
        _case(
            case_id="edge-batch-32",
            previous=[(index % 31) - 15 for index in range(32)],
            activations=[(index - 16) / 8.0 for index in range(32)],
            previous_offset=0,
            previous_stride=1,
            activation_offset=0,
            activation_stride=1,
            alias=False,
            category="edge",
        ),
        _case(
            case_id="edge-int8-domain-low",
            previous=list(range(-127, 1)),
            activations=[0.0] * 128,
            previous_offset=0,
            previous_stride=1,
            activation_offset=0,
            activation_stride=1,
            alias=False,
            category="edge",
        ),
        _case(
            case_id="edge-int8-domain-high",
            previous=list(range(1, 128)),
            activations=[0.0] * 127,
            previous_offset=0,
            previous_stride=1,
            activation_offset=0,
            activation_stride=1,
            alias=True,
            category="edge",
        ),
        _case(
            case_id="nonfinite-nan",
            previous=[0],
            activations=[math.nan],
            previous_offset=0,
            previous_stride=1,
            activation_offset=0,
            activation_stride=1,
            alias=False,
            category="nonfinite",
        ),
        _case(
            case_id="nonfinite-positive-infinity",
            previous=[0],
            activations=[math.inf],
            previous_offset=0,
            previous_stride=1,
            activation_offset=0,
            activation_stride=1,
            alias=False,
            category="nonfinite",
        ),
        _case(
            case_id="nonfinite-negative-infinity",
            previous=[0],
            activations=[-math.inf],
            previous_offset=0,
            previous_stride=1,
            activation_offset=0,
            activation_stride=1,
            alias=False,
            category="nonfinite",
        ),
    ]
    generator = random.Random(seed)
    for index in range(randomized_cases):
        count = generator.randint(1, 128)
        previous = [generator.randint(-127, 127) for _ in range(count)]
        activations = [generator.uniform(-8.0, 8.0) for _ in range(count)]
        previous_stride = generator.choice((1, 2, 3))
        activation_stride = generator.choice((1, 2, 3))
        category: CaseCategory = "stride" if previous_stride == activation_stride == 1 else "random"
        cases.append(
            _case(
                case_id=f"random-{index:03d}",
                previous=previous,
                activations=activations,
                previous_offset=generator.randint(0, 2),
                previous_stride=previous_stride,
                activation_offset=generator.randint(0, 2),
                activation_stride=activation_stride,
                alias=generator.choice((False, True)),
                category=category,
            )
        )
    return tuple(cases)
