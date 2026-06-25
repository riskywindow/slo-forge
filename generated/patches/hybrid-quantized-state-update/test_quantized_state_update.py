from __future__ import annotations

import math
import random
import unittest

from quantized_state_update import (
    quantized_recurrent_state_update,
    reference_quantized_state_update,
)


def _storage(values: list[int | float], offset: int, stride: int) -> list[int | float]:
    result: list[int | float] = [99] * (offset + (len(values) - 1) * stride + 1)
    for index, value in enumerate(values):
        result[offset + index * stride] = value
    return result


class QuantizedStateUpdateTests(unittest.TestCase):
    def test_complete_previous_state_domain_at_zero_activation(self) -> None:
        for previous in (list(range(-127, 1)), list(range(1, 128))):
            with self.subTest(first=previous[0], last=previous[-1]):
                observed = quantized_recurrent_state_update(
                    previous, [0.0] * len(previous), len(previous)
                )
                expected = [reference_quantized_state_update(value, 0.0) for value in previous]
                self.assertEqual(observed, expected)

    def test_ties_to_even_and_saturation(self) -> None:
        activations = [0.5 / 31.0, 1.5 / 31.0, -0.5 / 31.0, -1.5 / 31.0, 100.0]
        self.assertEqual(
            quantized_recurrent_state_update([0] * 5, activations, 5),
            [0, 2, 0, -2, 127],
        )
        self.assertEqual(reference_quantized_state_update(0, -100.0), -127)

    def test_finite_overflow_saturates(self) -> None:
        self.assertEqual(reference_quantized_state_update(0, 1.0e308), 127)
        self.assertEqual(reference_quantized_state_update(0, -1.0e308), -127)

    def test_noncontiguous_views(self) -> None:
        previous_values = [-41, 0, 41, 126]
        activation_values = [-1.5, -0.5, 0.5, 1.5]
        previous = _storage(previous_values, 1, 2)
        activation = _storage(activation_values, 2, 3)
        observed = quantized_recurrent_state_update(
            previous,  # type: ignore[arg-type]
            activation,  # type: ignore[arg-type]
            4,
            previous_offset=1,
            previous_stride=2,
            activation_offset=2,
            activation_stride=3,
        )
        expected = [
            reference_quantized_state_update(left, right)
            for left, right in zip(previous_values, activation_values, strict=True)
        ]
        self.assertEqual(observed, expected)

    def test_alias_commit_uses_exact_view(self) -> None:
        previous = [77, -5, 77, 6, 77, 17]
        activations = [0.2, -0.4, 0.6]
        expected = [
            reference_quantized_state_update(value, activation)
            for value, activation in zip((-5, 6, 17), activations, strict=True)
        ]
        observed = quantized_recurrent_state_update(
            previous,
            activations,
            3,
            previous_offset=1,
            previous_stride=2,
            output_alias_previous=True,
        )
        self.assertEqual(observed, expected)
        self.assertEqual(previous, [77, expected[0], 77, expected[1], 77, expected[2]])

    def test_alias_failure_is_atomic(self) -> None:
        previous = [1, 2, 3]
        original = previous.copy()
        with self.assertRaisesRegex(ValueError, "finite"):
            quantized_recurrent_state_update(
                previous,
                [0.0, math.nan, 0.0],
                3,
                output_alias_previous=True,
            )
        self.assertEqual(previous, original)

    def test_invalid_domains_fail_closed(self) -> None:
        invalid_calls = (
            lambda: quantized_recurrent_state_update([0], [0.0], 0),
            lambda: quantized_recurrent_state_update([0], [0.0], 1, previous_stride=4),
            lambda: quantized_recurrent_state_update([], [0.0], 1),
            lambda: quantized_recurrent_state_update([128], [0.0], 1),
            lambda: quantized_recurrent_state_update([0], [True], 1),
            lambda: quantized_recurrent_state_update([0], [math.inf], 1),
        )
        for call in invalid_calls:
            with self.subTest(call=call), self.assertRaises(ValueError):
                call()

    def test_seeded_random_differential(self) -> None:
        generator = random.Random(73129)
        for case_index in range(500):
            count = generator.randint(1, 128)
            previous_values = [generator.randint(-127, 127) for _ in range(count)]
            activation_values = [generator.uniform(-8.0, 8.0) for _ in range(count)]
            previous_offset = generator.randint(0, 2)
            activation_offset = generator.randint(0, 2)
            previous_stride = generator.choice((1, 2, 3))
            activation_stride = generator.choice((1, 2, 3))
            previous = _storage(previous_values, previous_offset, previous_stride)
            activations = _storage(activation_values, activation_offset, activation_stride)
            observed = quantized_recurrent_state_update(
                previous,  # type: ignore[arg-type]
                activations,  # type: ignore[arg-type]
                count,
                previous_offset,
                previous_stride,
                activation_offset,
                activation_stride,
                output_alias_previous=bool(case_index % 2),
            )
            expected = [
                reference_quantized_state_update(left, right)
                for left, right in zip(previous_values, activation_values, strict=True)
            ]
            self.assertEqual(observed, expected)


if __name__ == "__main__":
    unittest.main()
