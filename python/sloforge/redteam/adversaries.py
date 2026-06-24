"""Deterministic bounded adversarial input and schedule generators."""

from __future__ import annotations

import hashlib
import random
import struct
from typing import Literal, cast

from sloforge.genesis.ir import (
    Precision,
    RequestEventCase,
    ResourceCase,
    TensorInputCase,
    TopologyCase,
)

from .models import (
    ResourceAdversarialCase,
    ResourceAdversaryConfiguration,
    ScheduleAdversarialCase,
    ScheduleAdversaryConfiguration,
    TensorAdversarialCase,
    TensorAdversaryConfiguration,
    TopologyAdversarialCase,
    TopologyAdversaryConfiguration,
)


def _identifier(prefix: str, seed: int, index: int, description: str) -> str:
    digest = hashlib.sha256(f"{seed}:{index}:{description}".encode()).hexdigest()[:20]
    return f"{prefix}-{digest}"


def _contiguous_strides(shape: tuple[int, ...]) -> tuple[int, ...]:
    stride = 1
    result: list[int] = []
    for dimension in reversed(shape):
        result.append(stride)
        stride *= dimension
    return tuple(reversed(result))


def _values_hex(dtype: Precision, count: int, pattern: int) -> str:
    bounded = max(1, min(count, 32))
    if dtype is Precision.FLOAT64:
        words = (
            0x0000000000000000,
            0x3FF0000000000000,
            0x7FF0000000000000,
            0x7FF8000000000000,
            0xFFF0000000000000,
            1,
        )
        return b"".join(
            struct.pack("<Q", words[(pattern + i) % len(words)]) for i in range(bounded)
        ).hex()
    if dtype is Precision.FLOAT16:
        words = (0x0000, 0x3C00, 0x7C00, 0x7E00, 0xFC00, 0x0001)
        return b"".join(
            struct.pack("<H", words[(pattern + i) % len(words)]) for i in range(bounded)
        ).hex()
    if dtype is Precision.BFLOAT16:
        words = (0x0000, 0x3F80, 0x7F80, 0x7FC0, 0xFF80, 0x0001)
        return b"".join(
            struct.pack("<H", words[(pattern + i) % len(words)]) for i in range(bounded)
        ).hex()
    if dtype is Precision.FLOAT32:
        words = (0x00000000, 0x3F800000, 0x7F800000, 0x7FC00000, 0xFF800000, 1)
        return b"".join(
            struct.pack("<I", words[(pattern + i) % len(words)]) for i in range(bounded)
        ).hex()
    integer_formats: dict[Precision, tuple[str, tuple[int, ...]]] = {
        Precision.INT64: ("<Q", (0, 1, 0x7FFFFFFFFFFFFFFF, 0x8000000000000000, 0xFFFFFFFFFFFFFFFF)),
        Precision.INT32: ("<I", (0, 1, 0x7FFFFFFF, 0x80000000, 0xFFFFFFFF)),
        Precision.INT16: ("<H", (0, 1, 0x7FFF, 0x8000, 0xFFFF)),
    }
    if dtype in integer_formats:
        format_string, integer_words = integer_formats[dtype]
        return b"".join(
            struct.pack(
                format_string,
                integer_words[(pattern + i) % len(integer_words)],
            )
            for i in range(bounded)
        ).hex()
    if dtype is Precision.INT4:
        nibbles = tuple((pattern + i) & 0xF for i in range(bounded))
        return bytes(
            nibbles[index] | ((nibbles[index + 1] if index + 1 < len(nibbles) else 0) << 4)
            for index in range(0, len(nibbles), 2)
        ).hex()
    byte_patterns = (0x00, 0x01, 0x7F, 0x80, 0xFE, 0xFF)
    return bytes(byte_patterns[(pattern + i) % len(byte_patterns)] for i in range(bounded)).hex()


def generate_tensor_cases(
    configuration: TensorAdversaryConfiguration,
) -> tuple[TensorAdversarialCase, ...]:
    randomizer = random.Random(configuration.seed)
    dtypes = (
        Precision.FLOAT64,
        Precision.FLOAT32,
        Precision.FLOAT16,
        Precision.BFLOAT16,
        Precision.FP8,
        Precision.INT64,
        Precision.INT32,
        Precision.INT16,
        Precision.INT8,
        Precision.UINT8,
        Precision.INT4,
        Precision.BOOL,
    )
    dimensions = tuple(
        item
        for item in (1, 2, 3, 7, 8, 15, 16, 17, 31, 32, 33, 127, 128, 129, 257)
        if item <= configuration.maximum_dimension
    ) or (1,)
    cases: list[TensorAdversarialCase] = []
    first_shape = (
        min(2, configuration.maximum_dimension),
        min(3, configuration.maximum_dimension),
    )
    first = TensorInputCase(
        shape=first_shape,
        strides=(1, first_shape[0]),
        dtype=Precision.FLOAT32,
        values_hex=_values_hex(Precision.FLOAT32, first_shape[0] * first_shape[1], 3),
        non_contiguous=True,
    )
    cases.append(
        TensorAdversarialCase(
            case_id=_identifier("tensor", configuration.seed, 0, "non-contiguous-nan"),
            input=first,
        )
    )
    for index in range(1, configuration.maximum_cases):
        rank = randomizer.randint(1, configuration.maximum_rank)
        shape = tuple(randomizer.choice(dimensions) for _ in range(rank))
        contiguous = _contiguous_strides(shape)
        make_non_contiguous = index % 2 == 1
        strides = (
            tuple(
                value * (2 if position == rank - 1 else 1)
                for position, value in enumerate(contiguous)
            )
            if make_non_contiguous
            else contiguous
        )
        dtype = dtypes[(index - 1) % len(dtypes)]
        element_count = 1
        for dimension in shape:
            element_count *= dimension
        case = TensorInputCase(
            shape=shape,
            strides=strides,
            dtype=dtype,
            values_hex=_values_hex(dtype, element_count, index),
            non_contiguous=make_non_contiguous,
        )
        cases.append(
            TensorAdversarialCase(
                case_id=_identifier("tensor", configuration.seed, index, repr(case)),
                input=case,
            )
        )
    return tuple(cases)


def _renumber(events: tuple[RequestEventCase, ...]) -> tuple[RequestEventCase, ...]:
    return tuple(
        RequestEventCase(
            at_step=index,
            request_id=event.request_id,
            action=event.action,
            worker_id=event.worker_id,
        )
        for index, event in enumerate(events)
    )


def generate_schedule_cases(
    configuration: ScheduleAdversaryConfiguration,
) -> tuple[ScheduleAdversarialCase, ...]:
    randomizer = random.Random(configuration.seed)
    unsafe = _renumber(
        (
            RequestEventCase(at_step=0, request_id="r0", action="admit"),
            RequestEventCase(
                at_step=1,
                request_id="r1" if configuration.request_count > 1 else "r0",
                action="admit",
            ),
            RequestEventCase(at_step=2, request_id="r0", action="schedule", worker_id="w0"),
            RequestEventCase(
                at_step=3,
                request_id="r1" if configuration.request_count > 1 else "r0",
                action="decode",
                worker_id="w1" if configuration.worker_count > 1 else "w0",
            ),
            RequestEventCase(at_step=4, request_id="r0", action="emit", worker_id="w0"),
            RequestEventCase(at_step=5, request_id="r0", action="cancel", worker_id="w0"),
            RequestEventCase(
                at_step=6,
                request_id="r1" if configuration.request_count > 1 else "r0",
                action="fail",
                worker_id="w1" if configuration.worker_count > 1 else "w0",
            ),
            RequestEventCase(
                at_step=7,
                request_id="r0",
                action="retry",
                worker_id="w1" if configuration.worker_count > 1 else "w0",
            ),
        )[: configuration.maximum_events]
    )
    cases = [
        ScheduleAdversarialCase(
            case_id=_identifier("schedule", configuration.seed, 0, "cancel-retry-after-emit"),
            events=unsafe,
        )
    ]
    actions = (
        "admit",
        "schedule",
        "prefill",
        "decode",
        "emit",
        "cancel",
        "disconnect",
        "fail",
        "retry",
    )
    for index in range(1, configuration.maximum_cases):
        length = randomizer.randint(4, configuration.maximum_events)
        events = tuple(
            RequestEventCase(
                at_step=step,
                request_id=f"r{randomizer.randrange(configuration.request_count)}",
                action=cast(
                    Literal[
                        "admit",
                        "schedule",
                        "prefill",
                        "decode",
                        "emit",
                        "cancel",
                        "disconnect",
                        "fail",
                        "retry",
                    ],
                    actions[randomizer.randrange(len(actions))],
                ),
                worker_id=(
                    f"w{randomizer.randrange(configuration.worker_count)}" if step % 2 else None
                ),
            )
            for step in range(length)
        )
        cases.append(
            ScheduleAdversarialCase(
                case_id=_identifier("schedule", configuration.seed, index, repr(events)),
                events=events,
            )
        )
    return tuple(cases)


def generate_topology_cases(
    configuration: TopologyAdversaryConfiguration,
) -> tuple[TopologyAdversarialCase, ...]:
    randomizer = random.Random(configuration.seed)
    cases: list[TopologyAdversarialCase] = []
    for index in range(configuration.maximum_cases):
        hosts = 1 if index == 0 else randomizer.randint(1, configuration.maximum_hosts)
        devices = (
            min(2, configuration.maximum_devices_per_host)
            if index == 0
            else randomizer.randint(1, configuration.maximum_devices_per_host)
        )
        failed_link = "nvlink-0-1" if devices > 1 else "pcie-0"
        failed = (failed_link,) if index % 3 == 0 else ()
        degraded = (f"pcie-{index % max(1, devices)}",) if index % 2 else ()
        topology = TopologyCase(
            hosts=hosts,
            devices_per_host=devices,
            failed_links=failed,
            degraded_links=degraded,
        )
        cases.append(
            TopologyAdversarialCase(
                case_id=_identifier("topology", configuration.seed, index, repr(topology)),
                topology=topology,
            )
        )
    return tuple(cases)


def generate_resource_cases(
    configuration: ResourceAdversaryConfiguration,
) -> tuple[ResourceAdversarialCase, ...]:
    randomizer = random.Random(configuration.seed)
    queue_boundaries = tuple(
        item
        for item in (0, 1, 7, 8, 9, 15, 16, 17, 31, 32, 33)
        if item <= configuration.maximum_queue_depth
    ) or (0,)
    cases: list[ResourceAdversarialCase] = []
    for index in range(configuration.maximum_cases):
        resource = ResourceCase(
            device_bytes=(
                min(1024, configuration.maximum_device_bytes)
                if index == 0
                else randomizer.randrange(configuration.maximum_device_bytes + 1)
            ),
            host_bytes=(
                min(2048, configuration.maximum_host_bytes)
                if index == 0
                else randomizer.randrange(configuration.maximum_host_bytes + 1)
            ),
            queue_depth=(
                min(8, configuration.maximum_queue_depth)
                if index == 0
                else queue_boundaries[index % len(queue_boundaries)]
            ),
            process_count=(
                min(1, configuration.maximum_process_count)
                if index == 0
                else randomizer.randrange(configuration.maximum_process_count + 1)
            ),
        )
        cases.append(
            ResourceAdversarialCase(
                case_id=_identifier("resource", configuration.seed, index, repr(resource)),
                resource=resource,
            )
        )
    return tuple(cases)


__all__ = [
    "generate_resource_cases",
    "generate_schedule_cases",
    "generate_tensor_cases",
    "generate_topology_cases",
]
